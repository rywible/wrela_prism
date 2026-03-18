// Quarter-resolution volumetric cloud raymarch compute shader.

const PI: f32 = 3.14159265359;
const PLANET_RADIUS: f32 = 6360.0;
const MAX_MARCH: i32 = 96;
const MAX_LIGHT: i32 = 12;
// Reversed-Z: near=1.0, far=0.0. Sky pixels have depth near 0.
const SKY_DEPTH_THRESHOLD: f32 = 0.0005;

struct CloudUniforms {
    inv_view_proj: mat4x4<f32>,
    prev_view_proj: mat4x4<f32>,
    camera_position: vec4<f32>,
    sun_direction: vec4<f32>,
    sun_color: vec4<f32>,
    sky_ambient: vec4<f32>,
    cloud_params: vec4<f32>,    // (coverage, first_frame_flag, time, frame_index)
    screen_params: vec4<f32>,   // (quarter_w, quarter_h, 1/quarter_w, 1/quarter_h)
    cloud_profile: vec4<f32>,   // (density_scale, cloud_base, cloud_top, detail_erosion)
    cloud_profile2: vec4<f32>,  // (wind_speed, march_steps_f32, light_steps_f32, temporal_blend)
    prev_time: vec4<f32>,       // (prev_elapsed, 0, 0, 0)
};

@group(0) @binding(0) var<uniform> u: CloudUniforms;
@group(0) @binding(1) var shape_noise: texture_3d<f32>;
@group(0) @binding(2) var detail_noise: texture_3d<f32>;
@group(0) @binding(3) var weather_map: texture_2d<f32>;
@group(0) @binding(4) var noise_sampler: sampler;
@group(0) @binding(5) var history_tex: texture_2d<f32>;
@group(0) @binding(6) var output_tex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(7) var depth_tex: texture_depth_2d;
@group(0) @binding(8) var cloud_depth_tex: texture_storage_2d<r32float, write>;
@group(0) @binding(9) var sky_view_lut: texture_2d<f32>;
@group(0) @binding(10) var lut_sampler: sampler;

// ──────────────────────── Helpers ────────────────────────

fn ray_sphere(origin: vec3<f32>, dir: vec3<f32>, radius: f32) -> vec2<f32> {
    let b = dot(origin, dir);
    let c = dot(origin, origin) - radius * radius;
    let disc = b * b - c;
    if disc < 0.0 { return vec2<f32>(-1.0, -1.0); }
    let s = sqrt(disc);
    return vec2<f32>(-b - s, -b + s);
}

fn phase_cloud(cos_theta: f32) -> f32 {
    let g1 = 0.8; let g2 = -0.3;
    let fwd = (1.0 - g1*g1) / (4.0*PI * pow(1.0 + g1*g1 - 2.0*g1*cos_theta, 1.5));
    let back = (1.0 - g2*g2) / (4.0*PI * pow(1.0 + g2*g2 - 2.0*g2*cos_theta, 1.5));
    return mix(fwd, back, 0.25);
}

fn temporal_hash(pixel: vec2<u32>, frame: u32) -> f32 {
    let p = vec2<f32>(pixel) + 0.5;
    return fract(p.x * 0.7548776662 + p.y * 0.5698402909 + f32(frame) * 1.61803398875);
}

fn remap(value: f32, lo: f32, hi: f32, new_lo: f32, new_hi: f32) -> f32 {
    return new_lo + (saturate((value - lo) / (hi - lo))) * (new_hi - new_lo);
}

// ──────────────────────── Sky LUT ────────────────────────

fn dir_to_sky_view_uv(dir: vec3<f32>) -> vec2<f32> {
    let az = atan2(dir.x, dir.z);
    let u_val = select(az / (2.0*PI) + 1.0, az / (2.0*PI), az >= 0.0);
    let el = asin(clamp(dir.y, -1.0, 1.0));
    var v: f32;
    if el < 0.0 { v = 0.5 - sqrt(-el / (PI*0.5)) * 0.5; }
    else { v = 0.5 + sqrt(el / (PI*0.5)) * 0.5; }
    return vec2<f32>(u_val, v);
}

fn sample_sky_ambient(hf: f32, sun_col: vec3<f32>) -> vec3<f32> {
    let sky_up = textureSampleLevel(sky_view_lut, lut_sampler,
        dir_to_sky_view_uv(vec3<f32>(0.0, 1.0, 0.0)), 0.0).rgb;
    let sky_horiz = textureSampleLevel(sky_view_lut, lut_sampler,
        dir_to_sky_view_uv(vec3<f32>(1.0, 0.05, 0.0)), 0.0).rgb;
    // Cloud bottoms bathe in warm horizon light, tops see cooler zenith
    let sky = mix(sky_horiz * 1.1, sky_up * 0.55 + sky_horiz * 0.45, hf);
    // Warm ground-bounce on bases — amber tint from terrain
    let warm_base = sun_col * vec3<f32>(1.15, 0.85, 0.55) * 0.22 * pow(1.0 - hf, 1.5);
    // Sunlit warmth on tops
    let warm_top = sun_col * vec3<f32>(1.06, 0.97, 0.86) * 0.08 * hf;
    return sky + warm_base + warm_top;
}

// ──────────────────────── Cloud Density ────────────────────────

fn sample_cloud_density(pos_km: vec3<f32>, time: f32, ray_dist: f32) -> f32 {
    let cloud_base = u.cloud_profile.y;
    let cloud_top = u.cloud_profile.z;
    let erosion_str = u.cloud_profile.w;
    let wind_speed = u.cloud_profile2.x;
    let coverage = u.cloud_params.x;

    let altitude = pos_km.y - PLANET_RADIUS;
    let hf = clamp((altitude - cloud_base) / (cloud_top - cloud_base), 0.0, 1.0);
    let lod = saturate((ray_dist - 2.0) / 18.0);

    // ── Height profile ──
    let base_ramp = smoothstep(0.0, 0.14, hf);
    let top_round = 1.0 - smoothstep(0.60, 1.0, hf);
    let height_profile = base_ramp * top_round;

    // ── Weather map: spatially-varying coverage ──
    let wd = vec2<f32>(time * 0.0024, time * 0.0010) * wind_speed;
    let wm_broad  = textureSampleLevel(weather_map, noise_sampler, fract(pos_km.xz * 0.0012 + wd * 0.03), 0.0);
    let wm_detail = textureSampleLevel(weather_map, noise_sampler, fract(pos_km.xz * 0.004  + wd * 0.07), 0.0);

    // Weather coverage: broad masses + detail breakup
    let weather_raw = wm_broad.r * 0.60 + wm_detail.r * 0.40;

    // Soft weather gating: coverage controls how broadly clouds fill the sky
    // Low coverage = only high-weather areas, high coverage = nearly everywhere
    let gate_lo = max(0.50 - coverage * 0.65, 0.0);
    let gate_hi = gate_lo + 0.40;
    let local_cov = coverage * smoothstep(gate_lo, gate_hi, weather_raw);

    // ── Shape noise ──
    var flat_pos = vec3<f32>(pos_km.x, (altitude - cloud_base) * 0.7, pos_km.z);

    // Curl displacement for organic motion
    let curl_uv = fract(pos_km.xz * 0.004 + wd * 0.07);
    let curl_eps = 0.002;
    let cx = textureSampleLevel(weather_map, noise_sampler, curl_uv + vec2<f32>(0.0, curl_eps), 0.0).r;
    let cz = textureSampleLevel(weather_map, noise_sampler, curl_uv + vec2<f32>(curl_eps, 0.0), 0.0).r;
    let curl_str = mix(0.06, 0.22, hf) * 0.005 * wind_speed;
    flat_pos.x += (cx - wm_detail.r) / curl_eps * time * curl_str;
    flat_pos.z -= (cz - wm_detail.r) / curl_eps * time * curl_str;

    let bwind = vec3<f32>(time * 0.0024, 0.0, time * 0.0010) * wind_speed;
    let hwind = vec3<f32>(time * 0.0036, 0.0, time * 0.0015) * wind_speed;

    // Low-frequency billows (non-harmonic to break tiling)
    let shape = textureSampleLevel(shape_noise, noise_sampler,
        fract(flat_pos * 0.11 + bwind) * 0.98 + 0.01, 0.0);
    let mid = textureSampleLevel(shape_noise, noise_sampler,
        fract(flat_pos * 0.29 + hwind * 0.85) * 0.98 + 0.01, 0.0);

    // Shape noise blend
    let SN = shape.r * 0.55 + shape.g * 0.25 + mid.r * 0.20 * (1.0 - lod * 0.5);

    // Cloud signal: weather places clouds, noise adds billowy detail
    // local_cov sets the density ceiling, noise carves shape out of it
    let cloud_signal = local_cov - (1.0 - SN) * 0.40;

    // Smooth threshold produces soft cloud boundaries
    let shaped = smoothstep(-0.05, 0.18, cloud_signal);

    // Cloud-type from weather green channel modulates height
    let cloud_type = wm_broad.g;
    let anvil = saturate(hf - 0.6) * 0.5 * cloud_type;
    let density_raw = shaped * height_profile * (1.0 + anvil);

    // ── Detail erosion: cauliflower edges ──
    let det = textureSampleLevel(detail_noise, noise_sampler,
        fract(flat_pos * 2.6 + hwind * 1.6) * 0.98 + 0.01, 0.0);
    let erosion_noise = det.r * 0.40 + det.g * 0.32 + det.b * 0.28;
    // Strong at edges (low density_raw), weaker in dense core
    let edge_mask = pow(1.0 - saturate(density_raw * 1.8), 0.7);
    let height_boost = 1.0 + smoothstep(0.50, 1.0, hf) * 0.6;
    let erosion = erosion_noise * 0.48 * erosion_str * edge_mask * height_boost * (1.0 - lod * 0.70);

    return max(density_raw - erosion, 0.0);
}

// ──────────────────────── Cloud Raymarch ────────────────────────

@compute @workgroup_size(8, 8)
fn cloud_march(@builtin(global_invocation_id) gid: vec3<u32>) {
    let qs = vec2<u32>(u32(u.screen_params.x), u32(u.screen_params.y));
    if gid.x >= qs.x || gid.y >= qs.y { return; }

    let frame_idx = u32(u.cloud_params.w);
    let march_steps = i32(u.cloud_profile2.y);
    let light_steps = i32(u.cloud_profile2.z);
    let density_scale = u.cloud_profile.x;
    let cloud_base = u.cloud_profile.y;
    let cloud_top = u.cloud_profile.z;

    let res_div = u32(u.sky_ambient.w);
    let depth_raw = textureLoad(depth_tex, vec2<i32>(gid.xy * res_div), 0);
    if depth_raw > SKY_DEPTH_THRESHOLD {
        textureStore(output_tex, gid.xy, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        textureStore(cloud_depth_tex, gid.xy, vec4<f32>(0.0));
        return;
    }

    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(qs);
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0);
    // Reversed-Z: z=1 near, z=0 far. Use z=0.01 to avoid w=0 at infinity.
    let near_pt = u.inv_view_proj * vec4<f32>(ndc, 1.0, 1.0);
    let far_pt  = u.inv_view_proj * vec4<f32>(ndc, 0.01, 1.0);
    let ray_dir = normalize(far_pt.xyz / far_pt.w - near_pt.xyz / near_pt.w);

    let sun_dir = normalize(u.sun_direction.xyz);
    let sun_color = u.sun_color.rgb * u.sun_color.w;
    let time = u.cloud_params.z;

    var inscattered = vec3<f32>(0.0);
    var transmittance = 1.0;
    var cloud_dist = 0.0;

    if ray_dir.y > -0.22 {
        let cam_km = u.camera_position.xyz * 0.001;
        let origin = vec3<f32>(cam_km.x, PLANET_RADIUS + max(cam_km.y, 0.001), cam_km.z);
        let hit_lo = ray_sphere(origin, ray_dir, PLANET_RADIUS + cloud_base);
        let hit_hi = ray_sphere(origin, ray_dir, PLANET_RADIUS + cloud_top);

        if hit_hi.y >= 0.0 {
            let t0 = max(hit_lo.y, 0.0);
            let t1 = hit_hi.y;

            if t0 < t1 {
                let ds = (t1 - t0) / f32(march_steps);
                let cos_theta = dot(ray_dir, sun_dir);
                let phase = min(phase_cloud(cos_theta), 1.4);
                let sun_facing = clamp(cos_theta, 0.0, 1.0);
                let jitter = temporal_hash(gid.xy, frame_idx) * ds;

                for (var i = 0; i < MAX_MARCH; i++) {
                    if i >= march_steps || transmittance < 0.01 { break; }

                    let t = t0 + (f32(i) + 0.3) * ds + jitter;
                    let pos = origin + ray_dir * t;
                    let density = sample_cloud_density(pos, time, t);

                    if density > 0.001 {
                        let ext = density * density_scale;
                        let s_trans = exp(-ext * ds);

                        // Light march
                        var l_dens = 0.0;
                        let l_step = (cloud_top - cloud_base) / f32(light_steps);
                        for (var j = 0; j < MAX_LIGHT; j++) {
                            if j >= light_steps { break; }
                            l_dens += sample_cloud_density(
                                pos + sun_dir * (f32(j) + 0.5) * l_step, time, t);
                        }
                        let l_opt = l_dens * l_step * density_scale;
                        let l_trans = exp(-l_opt);

                        let hf = clamp((length(pos) - PLANET_RADIUS - cloud_base) / (cloud_top - cloud_base), 0.0, 1.0);

                        // ── Multi-scatter approximation (Wrenninge 2013) ──
                        // As light penetrates deeper, scattering becomes more isotropic
                        // and the medium glows from within. Three octaves of scatter.
                        let ms_amount = 0.28;
                        let ms_atten = 0.30;  // each octave attenuated
                        let ms_phase_atten = 0.50;
                        var ms_contrib = vec3<f32>(0.0);
                        var ms_a = ms_amount;
                        var ms_p = 1.0;
                        for (var o = 0; o < 3; o++) {
                            let oct_trans = exp(-l_opt * ms_p);
                            let oct_phase = mix(1.0 / (4.0 * PI), phase, ms_p);
                            ms_contrib += sun_color * ms_a * (1.0 - oct_trans * 0.4) * oct_phase;
                            ms_a *= ms_atten;
                            ms_p *= ms_phase_atten;
                        }

                        // Direct: attenuated sun + multi-scatter glow
                        let direct = sun_color * l_trans * phase + ms_contrib;

                        // ── Ambient: sky LUT with warm bias ──
                        let sky_amb = sample_sky_ambient(hf, sun_color);
                        let amb_w = mix(0.50, 0.85, hf) * (1.0 - l_trans * 0.20);
                        let ambient = sky_amb * amb_w;

                        // ── Silver lining: golden rim on backlit edges ──
                        let silver_str = pow(sun_facing, 2.0) * 0.65 * (1.0 - sun_facing * 0.35);
                        let rim_color = sun_color * vec3<f32>(1.08, 0.98, 0.85); // warm golden
                        let silver = rim_color * silver_str * exp(-density * 2.0) * (0.6 + hf * 0.5);

                        // ── Powder: darker deep interiors ──
                        let powder = mix(1.0, 1.0 - exp(-l_opt * 2.5), 0.50);

                        // ── Base darkening: stronger shadow on bottoms ──
                        let base_dark = mix(0.32, 1.0, smoothstep(0.0, 0.35, hf));

                        let lum = direct * powder * base_dark + ambient + silver;
                        let integrated = lum * (1.0 - s_trans);
                        inscattered += transmittance * integrated;
                        cloud_dist += transmittance * (1.0 - s_trans) * t;
                        transmittance *= s_trans;
                    }
                }
            }
        }
    }

    let opacity = 1.0 - transmittance;
    let avg_dist = select(0.0, cloud_dist / max(opacity, 0.001), opacity > 0.001);
    textureStore(output_tex, gid.xy, vec4<f32>(inscattered, transmittance));
    textureStore(cloud_depth_tex, gid.xy, vec4<f32>(avg_dist, 0.0, 0.0, 0.0));
}
