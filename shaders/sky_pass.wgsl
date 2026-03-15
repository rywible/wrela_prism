// Physically-based atmospheric sky pass — fills empty visibility buffer pixels.
//
// Implements single-scattering Rayleigh + Mie atmosphere with optical depth
// integration, limb-darkened sun disk, and multi-scatter approximation.

const PI: f32 = 3.14159265359;

// Planet / atmosphere geometry (km)
const PLANET_RADIUS: f32 = 6360.0;
const ATMO_RADIUS: f32 = 6420.0;

// Scale heights (km)
const H_RAYLEIGH: f32 = 8.5;
const H_MIE: f32 = 1.2;

// Sea-level scattering coefficients (per km)
const BETA_R: vec3<f32> = vec3<f32>(5.8e-3, 13.5e-3, 33.1e-3);
const BETA_M: f32 = 2.1e-2;

const NUM_SAMPLES: i32 = 12;
const NUM_LIGHT_SAMPLES: i32 = 6;

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
};

@group(0) @binding(0) var<uniform> uniforms: FrameUniforms;
@group(0) @binding(1) var vis_texture: texture_2d<u32>;

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

// Ray-sphere intersection. Returns (t_near, t_far) or (-1, -1) on miss.
fn ray_sphere(origin: vec3<f32>, dir: vec3<f32>, radius: f32) -> vec2<f32> {
    let b = dot(origin, dir);
    let c = dot(origin, origin) - radius * radius;
    let disc = b * b - c;
    if disc < 0.0 {
        return vec2<f32>(-1.0, -1.0);
    }
    let s = sqrt(disc);
    return vec2<f32>(-b - s, -b + s);
}

// Rayleigh phase function
fn phase_rayleigh(cos_theta: f32) -> f32 {
    return 3.0 / (16.0 * PI) * (1.0 + cos_theta * cos_theta);
}

// Henyey-Greenstein phase function for Mie scattering
fn phase_mie(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = 1.0 + g2 - 2.0 * g * cos_theta;
    return (1.0 - g2) / (4.0 * PI * pow(denom, 1.5));
}

// Compute optical depth along a ray segment through the atmosphere
fn optical_depth(origin: vec3<f32>, dir: vec3<f32>, length: f32) -> vec2<f32> {
    let step_size = length / f32(NUM_LIGHT_SAMPLES);
    var depth_r = 0.0;
    var depth_m = 0.0;
    for (var i = 0; i < NUM_LIGHT_SAMPLES; i++) {
        let pos = origin + dir * (f32(i) + 0.5) * step_size;
        let altitude = max(glength(pos) - PLANET_RADIUS, 0.0);
        depth_r += exp(-altitude / H_RAYLEIGH);
        depth_m += exp(-altitude / H_MIE);
    }
    return vec2<f32>(depth_r, depth_m) * step_size;
}

// Length helper (avoiding name clash with built-in)
fn glength(v: vec3<f32>) -> f32 {
    return length(v);
}

@fragment
fn fs_sky(input: FullscreenOutput) -> @location(0) vec4<f32> {
    let pixel = vec2<u32>(input.position.xy);
    let vis_id = textureLoad(vis_texture, pixel, 0).r;

    // Only fill empty pixels
    if vis_id != 0u {
        discard;
    }

    // Reconstruct world ray direction
    let ndc = vec2<f32>(input.uv.x * 2.0 - 1.0, (1.0 - input.uv.y) * 2.0 - 1.0);
    let near = uniforms.inv_view_proj * vec4<f32>(ndc, 0.0, 1.0);
    let far = uniforms.inv_view_proj * vec4<f32>(ndc, 1.0, 1.0);
    let ray_dir = normalize(far.xyz / far.w - near.xyz / near.w);

    let sun_dir = normalize(uniforms.sun_direction.xyz);

    // Artist multipliers
    let rayleigh_mult = uniforms.atmosphere_params.y;
    let mie_mult = uniforms.atmosphere_params.z;
    let g = uniforms.atmosphere_params.w;
    let sun_angular_radius = uniforms.atmosphere_params.x;
    let sky_strength = uniforms.sky_zenith.w;

    // Place camera at ground level on the planet surface
    let ray_origin = vec3<f32>(0.0, PLANET_RADIUS + 0.001, 0.0);

    // Intersect ray with atmosphere
    let atmo_hit = ray_sphere(ray_origin, ray_dir, ATMO_RADIUS);
    if atmo_hit.y < 0.0 {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }

    let t_start = max(atmo_hit.x, 0.0);
    let t_end = atmo_hit.y;
    let step_size = (t_end - t_start) / f32(NUM_SAMPLES);

    let cos_theta = dot(ray_dir, sun_dir);
    let phase_r = phase_rayleigh(cos_theta);
    let phase_m = phase_mie(cos_theta, g);

    var total_r = vec3<f32>(0.0);
    var total_m = vec3<f32>(0.0);
    var optical_r = 0.0;
    var optical_m = 0.0;

    for (var i = 0; i < NUM_SAMPLES; i++) {
        let t = t_start + (f32(i) + 0.5) * step_size;
        let pos = ray_origin + ray_dir * t;
        let altitude = max(glength(pos) - PLANET_RADIUS, 0.0);

        // Local density
        let density_r = exp(-altitude / H_RAYLEIGH) * step_size;
        let density_m = exp(-altitude / H_MIE) * step_size;

        // Accumulate optical depth along view ray
        optical_r += density_r;
        optical_m += density_m;

        // Light ray: from sample point toward sun
        let sun_hit = ray_sphere(pos, sun_dir, ATMO_RADIUS);
        if sun_hit.y > 0.0 {
            let light_depth = optical_depth(pos, sun_dir, sun_hit.y);

            // Total transmittance: view ray segment + light ray
            let tau = BETA_R * (optical_r + light_depth.x) * rayleigh_mult
                    + BETA_M * (optical_m + light_depth.y) * mie_mult;
            let transmittance = exp(-tau);

            total_r += density_r * transmittance;
            total_m += density_m * transmittance;
        }
    }

    let sun_color = uniforms.sun_color.rgb * uniforms.sun_color.w;
    var sky = sun_color * (total_r * BETA_R * phase_r * rayleigh_mult
                         + total_m * BETA_M * phase_m * mie_mult);

    // Multiple scattering approximation: constant ambient scatter
    let sky_zenith_color = uniforms.sky_zenith.rgb;
    sky += 0.03 * sky_zenith_color * sky_strength;

    // Sun disk with limb darkening
    let cos_sun = cos(sun_angular_radius);
    if cos_theta > cos_sun {
        let angular_dist = acos(clamp(cos_theta, -1.0, 1.0));
        let limb = pow(1.0 - angular_dist / sun_angular_radius, 0.4);
        // Transmittance to sun through atmosphere
        let sun_tau = BETA_R * optical_r * rayleigh_mult + BETA_M * optical_m * mie_mult;
        let sun_transmittance = exp(-sun_tau);
        sky += sun_color * limb * sun_transmittance * 8.0;
    }

    // Horizon haze (artist-controlled)
    let haze_strength = uniforms.shaft_params.x;
    let horizon_glow = exp(-abs(ray_dir.y) * 8.0) * haze_strength;
    let haze_color = mix(uniforms.sky_horizon.rgb, uniforms.fog_color.rgb, 0.5) * horizon_glow;
    sky += haze_color;

    // Apply sky strength
    sky *= sky_strength;

    // Apply exposure
    sky *= uniforms.lighting_params.x;

    return vec4<f32>(sky, 1.0);
}
