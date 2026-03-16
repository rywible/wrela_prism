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
    let cos_theta = dot(ray_dir, sun_dir);
    let angular_dist = acos(clamp(cos_theta, -1.0, 1.0));
    let sun_transmittance = sample_transmittance_at(sun_dir.y, 0.001);

    // ── Sky-view LUT: physical Rayleigh + Mie scattering ──
    let sky_uv = dir_to_sky_view_uv(ray_dir);
    let lut_color = textureSampleLevel(sky_view_lut, lut_sampler, sky_uv, 0.0).rgb;
    let scattering = lut_color * sun_color;

    // ── Build sky color in layers ──

    // Elevation factor: 1.0 at zenith, 0.0 at horizon
    let elev = max(ray_dir.y, 0.0);
    let zenith_weight = pow(elev, 0.5);

    // Layer 1: 3-zone base gradient
    //   Horizon (elev=0): fog_color — matches material-pass aerial perspective
    //   Low sky (elev~5°): sky_horizon — the bright pale sky color
    //   Zenith (elev=90°): sky_zenith — deep saturated blue
    let horizon_blend = smoothstep(0.0, 0.08, elev);  // fog→horizon in first ~5°
    let lower_sky = mix(uniforms.fog_color.rgb, uniforms.sky_horizon.rgb * 0.6, horizon_blend);
    let base_sky = mix(lower_sky, uniforms.sky_zenith.rgb, zenith_weight);

    // Layer 2: Physical scattering from LUT
    // Color-selective clamp: allow blue Rayleigh through, limit R/G from Mie
    // to prevent warm scattering from turning the blue sky lavender.
    let scatter_boost = mix(0.2, 2.0, smoothstep(0.0, 0.25, elev));
    let scatter_raw = scattering * scatter_boost;
    let scatter_contrib = min(scatter_raw, vec3<f32>(0.05, 0.07, 0.18));
    var sky = base_sky + scatter_contrib;

    // Layer 3: Sun-side warmth — golden, not red (avoids purple mixing with blue)
    let sun_facing = max(cos_theta, 0.0);
    let sun_warmth = sun_facing * sun_facing * 0.03;
    sky += vec3<f32>(0.5, 0.5, 0.1) * sun_warmth * sun_transmittance;

    // Layer 4: Warm horizon band — concentrated at the horizon line
    let horizon_band = exp(-abs(ray_dir.y) * 8.0);
    let haze_strength = uniforms.shaft_params.x;
    let sun_horiz_dot = max(dot(
        normalize(vec3<f32>(ray_dir.x, 0.0, ray_dir.z)),
        normalize(vec3<f32>(sun_dir.x, 0.0, sun_dir.z))), 0.0);
    let warm_horiz = horizon_band * pow(sun_horiz_dot, 2.0) * haze_strength;
    sky += vec3<f32>(0.15, 0.08, 0.02) * warm_horiz;

    // Layer 5: Mie forward-scatter — warm glow around the sun
    let mie_glow = exp(-angular_dist * 3.5);
    sky += vec3<f32>(1.0, 0.85, 0.55) * sun_color * sun_transmittance * mie_glow * 0.04;

    // ── Sun disk — small, intense, limb-darkened ──
    let cos_sun = cos(sun_angular_radius);
    if cos_theta > cos_sun {
        let limb = pow(1.0 - angular_dist / sun_angular_radius, 0.4);
        sky += sun_color * limb * sun_transmittance * 50.0;
    }

    // ── Corona — tight warm ring right around disk ──
    let corona = exp(-angular_dist * 40.0);
    sky += vec3<f32>(1.0, 0.92, 0.70) * sun_color * sun_transmittance * corona * 0.3;

    // Exposure (must match material pass)
    sky *= exposure;

    return vec4<f32>(sky, 1.0);
}
