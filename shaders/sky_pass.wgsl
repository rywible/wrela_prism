// LUT-sampled atmospheric sky pass.
//
// Replaces per-pixel atmosphere raymarching with a single texture fetch from
// the precomputed sky-view LUT. Sun disk with limb darkening and multi-band
// corona. Horizon aerial perspective enrichment.

const PI: f32 = 3.14159265359;

const PLANET_RADIUS: f32 = 6360.0;
const ATMO_RADIUS: f32 = 6420.0;

struct FrameUniforms {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    camera_position: vec4<f32>,
    sun_direction: vec4<f32>,
    sun_color: vec4<f32>,
    fog_color: vec4<f32>,
    sky_zenith: vec4<f32>,
    sky_horizon: vec4<f32>,
    fog_params: vec4<f32>,
    lighting_params: vec4<f32>,
    light_vp: mat4x4<f32>,
    ambient_up: vec4<f32>,
    ambient_down: vec4<f32>,
    ambient_right: vec4<f32>,
    ambient_left: vec4<f32>,
    ambient_front: vec4<f32>,
    ambient_back: vec4<f32>,
    atmosphere_params: vec4<f32>,   // (sun_angular_radius, rayleigh_strength, mie_strength, mie_anisotropy)
    shaft_params: vec4<f32>,        // (horizon_haze, shaft_intensity, shaft_decay, 0)
    time_params: vec4<f32>,
    screen_size: vec4<f32>,
    wind_params: vec4<f32>,
    light_vp_1: mat4x4<f32>,
    light_vp_2: mat4x4<f32>,
    light_vp_3: mat4x4<f32>,
    cascade_splits: vec4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: FrameUniforms;
@group(0) @binding(1) var vis_texture: texture_2d<u32>;
@group(0) @binding(2) var sky_view_lut: texture_2d<f32>;
@group(0) @binding(3) var transmittance_lut: texture_2d<f32>;
@group(0) @binding(4) var lut_sampler: sampler;

struct FullscreenOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_sky(@builtin(vertex_index) vid: u32) -> FullscreenOutput {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    let p = pos[vid];
    var out: FullscreenOutput;
    out.position = vec4<f32>(p, 0.0, 1.0);
    out.uv = p * 0.5 + 0.5;
    out.uv.y = 1.0 - out.uv.y;
    return out;
}

// ──────────────────────── Sky-View LUT Sampling ────────────────────────

// Convert ray direction to sky-view LUT UV coordinates.
// Must match the parameterization in sky_lut.wgsl compute_sky_view.
fn dir_to_sky_view_uv(dir: vec3<f32>) -> vec2<f32> {
    let azimuth = atan2(dir.x, dir.z);
    let u = azimuth / (2.0 * PI);
    // Wrap to [0,1]
    let u_wrapped = select(u + 1.0, u, u >= 0.0);

    let elevation = asin(clamp(dir.y, -1.0, 1.0));

    // Non-linear mapping: more texels near horizon (must match sky_lut.wgsl)
    var v: f32;
    if elevation < 0.0 {
        let t = sqrt(-elevation / (PI * 0.5));
        v = 0.5 - t * 0.5;
    } else {
        let t = sqrt(elevation / (PI * 0.5));
        v = 0.5 + t * 0.5;
    }

    return vec2<f32>(u_wrapped, v);
}

// Sample transmittance LUT for sun extinction at horizon
fn sample_transmittance_at(cos_zenith: f32, height: f32) -> vec3<f32> {
    let u = (cos_zenith + 1.0) * 0.5;
    let v = height / (ATMO_RADIUS - PLANET_RADIUS);
    let uv = clamp(vec2<f32>(u, v), vec2<f32>(0.001), vec2<f32>(0.999));
    return textureSampleLevel(transmittance_lut, lut_sampler, uv, 0.0).rgb;
}

fn soft_sky_rolloff(color: vec3<f32>) -> vec3<f32> {
    return color / (vec3<f32>(1.0) + color * vec3<f32>(0.22, 0.18, 0.14));
}

fn horizon_alignment(view_dir: vec3<f32>, sun_dir: vec3<f32>) -> f32 {
    let view_xz = vec2<f32>(view_dir.x, view_dir.z);
    let sun_xz = vec2<f32>(sun_dir.x, sun_dir.z);
    let view_len = length(view_xz);
    let sun_len = length(sun_xz);
    if view_len < 1e-4 || sun_len < 1e-4 {
        return 0.5;
    }
    return clamp(dot(view_xz / view_len, sun_xz / sun_len), 0.0, 1.0);
}

fn lower_atmosphere_haze(
    ray_dir: vec3<f32>,
    sun_dir: vec3<f32>,
    sun_transmittance: vec3<f32>,
    cloud_cover: f32,
) -> vec3<f32> {
    let horizon = exp(-abs(ray_dir.y) * 13.0);
    let lower_mix = smoothstep(-0.36, 0.14, ray_dir.y);
    let sun_side = pow(horizon_alignment(ray_dir, sun_dir), 1.35);
    let dust = uniforms.fog_color.rgb * vec3<f32>(0.86, 0.82, 0.70) * (0.28 + 0.20 * lower_mix);
    let warm = uniforms.sun_color.rgb * sun_transmittance * vec3<f32>(0.28, 0.19, 0.09)
        * (0.30 + 0.40 * sun_side)
        * (1.0 - cloud_cover * 0.18);
    let lift = vec3<f32>(0.04, 0.032, 0.020);
    return (dust + warm + lift) * horizon;
}

// ──────────────────────── Main Fragment ────────────────────────

@fragment
fn fs_sky(input: FullscreenOutput) -> @location(0) vec4<f32> {
    let pixel = vec2<u32>(input.position.xy);
    let vis_id = textureLoad(vis_texture, pixel, 0).r;

    if vis_id != 0u {
        discard;
    }

    let ndc = vec2<f32>(input.uv.x * 2.0 - 1.0, (1.0 - input.uv.y) * 2.0 - 1.0);
    let near = uniforms.inv_view_proj * vec4<f32>(ndc, 0.0, 1.0);
    let far = uniforms.inv_view_proj * vec4<f32>(ndc, 1.0, 1.0);
    let ray_dir = normalize(far.xyz / far.w - near.xyz / near.w);

    let sun_dir = normalize(uniforms.sun_direction.xyz);
    let sun_angular_radius = uniforms.atmosphere_params.x;
    let exposure = uniforms.lighting_params.x;
    let sun_color = uniforms.sun_color.rgb * uniforms.sun_color.w;
    let sky_strength = max(uniforms.sky_zenith.w, 0.001);
    let cloud_cover = clamp(uniforms.shaft_params.w, 0.0, 1.0);
    let sun_height = max(sun_dir.y, 0.0);
    let clear_sun = clamp(1.0 - cloud_cover * (0.16 + 0.08 * sun_height), 0.68, 1.0);
    let diffuse_sun = mix(1.0, 0.93, cloud_cover);
    let cos_theta = dot(ray_dir, sun_dir);
    let angular_dist = acos(clamp(cos_theta, -1.0, 1.0));
    let sun_transmittance = sample_transmittance_at(sun_dir.y, 0.001);

    // ── Sky-view LUT: physical Rayleigh + Mie scattering ──
    let sky_uv = dir_to_sky_view_uv(ray_dir);
    let lut_color = textureSampleLevel(sky_view_lut, lut_sampler, sky_uv, 0.0).rgb;
    let scattering = lut_color * sun_color * diffuse_sun * sky_strength;
    let lower_haze = lower_atmosphere_haze(ray_dir, sun_dir, sun_transmittance, cloud_cover);

    // ── Build sky color in layers ──

    let elev = clamp(ray_dir.y * 0.75 + 0.32, 0.0, 1.0);
    let zenith_weight = pow(elev, 0.68);

    // Gradient colors are now a gentle grade over the LUT, not the main sky body.
    let horizon_blend = smoothstep(-0.08, 0.18, ray_dir.y);
    let lower_grade = mix(
        uniforms.fog_color.rgb * vec3<f32>(0.28, 0.28, 0.22) + uniforms.sun_color.rgb * vec3<f32>(0.08, 0.06, 0.03),
        uniforms.sky_horizon.rgb * 0.16 + uniforms.sun_color.rgb * vec3<f32>(0.05, 0.04, 0.03),
        horizon_blend
    );
    let upper_grade = mix(
        uniforms.sky_horizon.rgb * 0.07,
        uniforms.sky_zenith.rgb * 0.18,
        zenith_weight
    );
    let base_grade = mix(lower_grade, upper_grade, smoothstep(-0.02, 0.62, ray_dir.y));

    let scatter_boost = mix(1.46, 1.84, smoothstep(-0.10, 0.40, ray_dir.y));
    let scatter_contrib = soft_sky_rolloff(scattering * scatter_boost);
    var sky = scatter_contrib + base_grade + lower_haze;

    // Layer 3: Sun-side warmth — golden, not red (avoids purple mixing with blue)
    let sun_facing = max(cos_theta, 0.0);
    let sun_warmth = pow(sun_facing, 1.7) * 0.065 * clear_sun;
    sky += vec3<f32>(0.55, 0.46, 0.16) * sun_warmth * sun_transmittance;

    // Layer 4: Warm horizon band — concentrated at the horizon line
    let horizon_band = exp(-abs(ray_dir.y) * 6.8);
    let haze_strength = uniforms.shaft_params.x * mix(1.0, 1.30, cloud_cover);
    let sun_horiz_dot = max(dot(
        normalize(vec3<f32>(ray_dir.x, 0.0, ray_dir.z)),
        normalize(vec3<f32>(sun_dir.x, 0.0, sun_dir.z))), 0.0);
    let warm_horiz = horizon_band * pow(sun_horiz_dot, 1.5) * haze_strength;
    sky += vec3<f32>(0.12, 0.075, 0.030) * warm_horiz * sun_transmittance;

    // Layer 5: Mie forward-scatter — warm glow around the sun
    let mie_glow = exp(-angular_dist * 4.4);
    sky += vec3<f32>(1.0, 0.87, 0.58) * sun_color * sun_transmittance * mie_glow * 0.08 * clear_sun;

    // ── Sun disk — small, intense, limb-darkened ──
    let cos_sun = cos(sun_angular_radius);
    if cos_theta > cos_sun {
        let limb = pow(1.0 - angular_dist / sun_angular_radius, 0.4);
        sky += sun_color * limb * sun_transmittance * 42.0 * clear_sun;
    }

    // ── Corona — tight warm ring right around disk ──
    let corona = exp(-angular_dist * 40.0);
    sky += vec3<f32>(1.0, 0.92, 0.70) * sun_color * sun_transmittance * corona * 0.30 * clear_sun;

    let sky_luma = dot(sky, vec3<f32>(0.2126, 0.7152, 0.0722));
    sky = mix(sky, vec3<f32>(sky_luma), cloud_cover * 0.04);

    // Exposure (must match material pass)
    sky *= exposure;

    return vec4<f32>(sky, 1.0);
}
