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
    let clear_sun = clamp(1.0 - cloud_cover * (0.24 + 0.12 * sun_height), 0.55, 1.0);
    let diffuse_sun = mix(1.0, 0.88, cloud_cover);
    let cos_theta = dot(ray_dir, sun_dir);
    let angular_dist = acos(clamp(cos_theta, -1.0, 1.0));
    let sun_transmittance = sample_transmittance_at(sun_dir.y, 0.001);

    // ── Sky-view LUT: physical Rayleigh + Mie scattering ──
    let sky_uv = dir_to_sky_view_uv(ray_dir);
    let lut_color = textureSampleLevel(sky_view_lut, lut_sampler, sky_uv, 0.0).rgb;
    let scattering = lut_color * sun_color * diffuse_sun * sky_strength;

    // ── Build sky color in layers ──

    // Elevation factor: 1.0 at zenith, 0.0 at horizon
    let elev = max(ray_dir.y, 0.0);
    let zenith_weight = pow(elev, 0.5);

    // Gradient colors are now a gentle grade over the LUT, not the main sky body.
    let horizon_blend = smoothstep(0.0, 0.10, elev);
    let lower_grade = mix(
        uniforms.fog_color.rgb * (0.42 + cloud_cover * 0.18),
        uniforms.sky_horizon.rgb * (0.28 + clear_sun * 0.10),
        horizon_blend
    );
    let upper_grade = mix(
        uniforms.sky_horizon.rgb * 0.12,
        uniforms.sky_zenith.rgb * 0.24,
        zenith_weight
    );
    let base_grade = mix(lower_grade, upper_grade, smoothstep(0.03, 0.58, elev));

    let scatter_boost = mix(1.18, 1.72, smoothstep(0.0, 0.32, elev));
    let scatter_contrib = soft_sky_rolloff(scattering * scatter_boost);
    var sky = scatter_contrib + base_grade;

    // Layer 3: Sun-side warmth — golden, not red (avoids purple mixing with blue)
    let sun_facing = max(cos_theta, 0.0);
    let sun_warmth = sun_facing * sun_facing * 0.035 * clear_sun;
    sky += vec3<f32>(0.55, 0.46, 0.16) * sun_warmth * sun_transmittance;

    // Layer 4: Warm horizon band — concentrated at the horizon line
    let horizon_band = exp(-abs(ray_dir.y) * 8.0);
    let haze_strength = uniforms.shaft_params.x * mix(1.0, 1.30, cloud_cover);
    let sun_horiz_dot = max(dot(
        normalize(vec3<f32>(ray_dir.x, 0.0, ray_dir.z)),
        normalize(vec3<f32>(sun_dir.x, 0.0, sun_dir.z))), 0.0);
    let warm_horiz = horizon_band * pow(sun_horiz_dot, 2.0) * haze_strength;
    sky += vec3<f32>(0.18, 0.10, 0.035) * warm_horiz;

    // Layer 5: Mie forward-scatter — warm glow around the sun
    let mie_glow = exp(-angular_dist * 3.5);
    sky += vec3<f32>(1.0, 0.87, 0.58) * sun_color * sun_transmittance * mie_glow * 0.05 * clear_sun;

    // ── Sun disk — small, intense, limb-darkened ──
    let cos_sun = cos(sun_angular_radius);
    if cos_theta > cos_sun {
        let limb = pow(1.0 - angular_dist / sun_angular_radius, 0.4);
        sky += sun_color * limb * sun_transmittance * 34.0 * clear_sun;
    }

    // ── Corona — tight warm ring right around disk ──
    let corona = exp(-angular_dist * 40.0);
    sky += vec3<f32>(1.0, 0.92, 0.70) * sun_color * sun_transmittance * corona * 0.22 * clear_sun;

    let sky_luma = dot(sky, vec3<f32>(0.2126, 0.7152, 0.0722));
    sky = mix(sky, vec3<f32>(sky_luma), cloud_cover * 0.08);

    // Exposure (must match material pass)
    sky *= exposure;

    return vec4<f32>(sky, 1.0);
}
