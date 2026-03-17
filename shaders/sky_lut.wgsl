// Atmosphere LUT generation — three compute entry points.
//
// Based on Hillaire 2020 "A Scalable and Production Ready Sky and Atmosphere Rendering Technique."
//
// Entry points:
//   compute_transmittance  — 256×64 Rgba16Float
//   compute_multiscatter   — 32×32 Rgba16Float
//   compute_sky_view       — 192×108 Rgba16Float

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

// Absorption (ozone)
const BETA_OZ: vec3<f32> = vec3<f32>(0.65e-3, 1.88e-3, 0.085e-3);

struct AtmosphereParams {
    sun_direction: vec4<f32>,        // xyz = direction, w = angular_radius
    rayleigh_mie: vec4<f32>,         // x = rayleigh_strength, y = mie_strength, z = mie_anisotropy, w = unused
    screen_size: vec4<f32>,          // x = width, y = height
};

@group(0) @binding(0) var<uniform> params: AtmosphereParams;
@group(0) @binding(1) var transmittance_lut: texture_2d<f32>;
@group(0) @binding(2) var multiscatter_lut: texture_2d<f32>;
@group(0) @binding(3) var output_tex: texture_storage_2d<rgba16float, write>;

// ──────────────────────── Helpers ────────────────────────

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

fn phase_rayleigh(cos_theta: f32) -> f32 {
    return 3.0 / (16.0 * PI) * (1.0 + cos_theta * cos_theta);
}

fn phase_mie(cos_theta: f32, g: f32) -> f32 {
    // Dual-lobe: forward peak (g) + back-scatter (g2=-0.3)
    let g2_fwd = g * g;
    let fwd = (1.0 - g2_fwd) / (4.0 * PI * pow(1.0 + g2_fwd - 2.0 * g * cos_theta, 1.5));
    let g_back = -0.3;
    let g2_back = g_back * g_back;
    let back = (1.0 - g2_back) / (4.0 * PI * pow(1.0 + g2_back - 2.0 * g_back * cos_theta, 1.5));
    return mix(fwd, back, 0.25);
}

fn density_at_height(altitude: f32) -> vec2<f32> {
    return vec2<f32>(
        exp(-altitude / H_RAYLEIGH),
        exp(-altitude / H_MIE),
    );
}

fn ozone_density(altitude: f32) -> f32 {
    return max(1.0 - abs(altitude - 25.0) / 15.0, 0.0);
}

// ──────────────────────── Transmittance LUT ────────────────────────
//
// Parameterization: x = cos(zenith_angle) mapped [0,1], y = height mapped [0,1]
// Size: 256 × 64
// Integration: 40 steps along ray from height to atmosphere top.

fn transmittance_uv_to_params(uv: vec2<f32>) -> vec2<f32> {
    // y → height above surface [0, atmo_top]
    let h = uv.y * (ATMO_RADIUS - PLANET_RADIUS);
    // x → cos(zenith), remapped with sqrt for more precision near horizon
    let cos_zen = 2.0 * uv.x - 1.0;
    return vec2<f32>(cos_zen, h);
}

fn compute_transmittance_impl(cos_zenith: f32, height: f32) -> vec3<f32> {
    let origin = vec3<f32>(0.0, PLANET_RADIUS + height, 0.0);
    let dir = vec3<f32>(sqrt(max(1.0 - cos_zenith * cos_zenith, 0.0)), cos_zenith, 0.0);

    let hit = ray_sphere(origin, dir, ATMO_RADIUS);
    if hit.y < 0.0 {
        return vec3<f32>(1.0);
    }
    // Note: no ground occlusion here. The transmittance LUT measures pure
    // atmospheric optical depth. Ground occlusion is handled in the sky-view
    // LUT via ray_length clamping at ground_hit.x (see compute_sky_view).

    let ray_length = hit.y;

    let steps = 40;
    let step_size = ray_length / f32(steps);
    var optical_r = 0.0;
    var optical_m = 0.0;
    var optical_oz = 0.0;

    for (var i = 0; i < steps; i++) {
        let t = (f32(i) + 0.5) * step_size;
        let pos = origin + dir * t;
        let alt = max(length(pos) - PLANET_RADIUS, 0.0);
        let d = density_at_height(alt);
        optical_r += d.x * step_size;
        optical_m += d.y * step_size;
        optical_oz += ozone_density(alt) * step_size;
    }

    let tau = BETA_R * optical_r + BETA_M * optical_m + BETA_OZ * optical_oz;
    return exp(-tau);
}

@compute @workgroup_size(8, 8)
fn compute_transmittance(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = vec2<u32>(256u, 64u);
    if gid.x >= size.x || gid.y >= size.y {
        return;
    }
    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(size);
    let p = transmittance_uv_to_params(uv);
    let trans = compute_transmittance_impl(p.x, p.y);
    textureStore(output_tex, gid.xy, vec4<f32>(trans, 1.0));
}

// ──────────────────────── Transmittance LUT Lookup ────────────────────────

fn sample_transmittance(cos_zenith: f32, height: f32) -> vec3<f32> {
    let u = (cos_zenith + 1.0) * 0.5;
    let v = height / (ATMO_RADIUS - PLANET_RADIUS);
    let uv = clamp(vec2<f32>(u, v), vec2<f32>(0.0), vec2<f32>(1.0));
    let texel = clamp(vec2<i32>(uv * vec2<f32>(256.0, 64.0)), vec2<i32>(0), vec2<i32>(255, 63));
    return textureLoad(transmittance_lut, texel, 0).rgb;
}

// ──────────────────────── Multi-Scatter LUT ────────────────────────
//
// Parameterization: x = cos(sun_zenith), y = height
// Size: 32 × 32
// Integration: 64 hemisphere directions × 20 steps each.

@compute @workgroup_size(8, 8)
fn compute_multiscatter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = vec2<u32>(32u, 32u);
    if gid.x >= size.x || gid.y >= size.y {
        return;
    }
    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(size);
    let cos_sun_zenith = uv.x * 2.0 - 1.0;
    let height = uv.y * (ATMO_RADIUS - PLANET_RADIUS);

    let sun_dir = vec3<f32>(sqrt(max(1.0 - cos_sun_zenith * cos_sun_zenith, 0.0)), cos_sun_zenith, 0.0);
    let origin = vec3<f32>(0.0, PLANET_RADIUS + height, 0.0);

    let rayleigh_mult = params.rayleigh_mie.x;
    let mie_mult = params.rayleigh_mie.y;

    // Integrate over hemisphere
    var total_luminance = vec3<f32>(0.0);
    var total_fms = vec3<f32>(0.0); // multi-scatter factor
    let num_dirs = 64;
    let num_steps = 20;

    for (var i = 0; i < num_dirs; i++) {
        // Uniform sphere sampling with Fibonacci spiral
        let fi = f32(i);
        let golden = (1.0 + sqrt(5.0)) * 0.5;
        let theta = acos(1.0 - 2.0 * (fi + 0.5) / f32(num_dirs));
        let phi = 2.0 * PI * fi / golden;
        let dir = vec3<f32>(sin(theta) * cos(phi), cos(theta), sin(theta) * sin(phi));

        let hit_atmo = ray_sphere(origin, dir, ATMO_RADIUS);
        let hit_ground = ray_sphere(origin, dir, PLANET_RADIUS);

        var ray_len = hit_atmo.y;
        if hit_ground.x > 0.0 {
            ray_len = min(ray_len, hit_ground.x);
        }
        if ray_len <= 0.0 { continue; }

        let step_size = ray_len / f32(num_steps);
        var optical_r = 0.0;
        var optical_m = 0.0;
        var optical_oz = 0.0;
        var luminance = vec3<f32>(0.0);
        var fms = vec3<f32>(0.0);

        for (var j = 0; j < num_steps; j++) {
            let t = (f32(j) + 0.5) * step_size;
            let pos = origin + dir * t;
            let alt = max(length(pos) - PLANET_RADIUS, 0.0);
            let d = density_at_height(alt);

            optical_r += d.x * step_size;
            optical_m += d.y * step_size;
            optical_oz += ozone_density(alt) * step_size;

            let local_tau = BETA_R * optical_r * rayleigh_mult
                + BETA_M * optical_m * mie_mult
                + BETA_OZ * optical_oz;
            let local_trans = exp(-local_tau);

            let cos_zen_sun = normalize(pos).y * sun_dir.y + normalize(pos).x * sun_dir.x;
            let sun_trans = sample_transmittance(cos_zen_sun, alt);

            let scatter_r = d.x * BETA_R * rayleigh_mult;
            let scatter_m = d.y * BETA_M * mie_mult;
            let scatter_total = scatter_r + scatter_m;

            // Isotropic phase for multi-scatter (1 / 4π)
            let phase = 1.0 / (4.0 * PI);

            luminance += sun_trans * scatter_total * phase * local_trans * step_size;
            fms += scatter_total * phase * local_trans * step_size;
        }

        total_luminance += luminance;
        total_fms += fms;
    }

    // Average over directions (solid angle = 4π, each sample covers 4π/num_dirs)
    let inv_samples = 4.0 * PI / f32(num_dirs);
    let l = total_luminance * inv_samples;
    let f = total_fms * inv_samples;

    // Multi-scatter contribution: L_ms = L / (1 - F)
    let ms = l / max(vec3<f32>(1.0) - f, vec3<f32>(0.001));

    textureStore(output_tex, gid.xy, vec4<f32>(ms, 1.0));
}

// ──────────────────────── Sky-View LUT ────────────────────────
//
// Parameterization: x = azimuth [0,2π], y = elevation with non-linear mapping
// Size: 192 × 108
// Reads transmittance + multi-scatter LUTs.

fn sky_view_uv_to_dir(uv: vec2<f32>) -> vec3<f32> {
    let azimuth = uv.x * 2.0 * PI;

    // Non-linear elevation mapping: more texels near horizon
    let v = uv.y;
    var elevation: f32;
    if v < 0.5 {
        let t = 1.0 - 2.0 * v;
        elevation = -PI * 0.5 * t * t;
    } else {
        let t = 2.0 * v - 1.0;
        elevation = PI * 0.5 * t * t;
    }

    return vec3<f32>(
        cos(elevation) * sin(azimuth),
        sin(elevation),
        cos(elevation) * cos(azimuth),
    );
}

fn sample_multiscatter(cos_sun_zenith: f32, height: f32) -> vec3<f32> {
    let u = (cos_sun_zenith + 1.0) * 0.5;
    let v = height / (ATMO_RADIUS - PLANET_RADIUS);
    let uv = clamp(vec2<f32>(u, v), vec2<f32>(0.0), vec2<f32>(1.0));
    let texel = clamp(vec2<i32>(uv * vec2<f32>(32.0, 32.0)), vec2<i32>(0), vec2<i32>(31, 31));
    return textureLoad(multiscatter_lut, texel, 0).rgb;
}

@compute @workgroup_size(8, 8)
fn compute_sky_view(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = vec2<u32>(192u, 108u);
    if gid.x >= size.x || gid.y >= size.y {
        return;
    }
    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(size);
    let ray_dir = sky_view_uv_to_dir(uv);
    let sun_dir = normalize(params.sun_direction.xyz);

    let rayleigh_mult = params.rayleigh_mie.x;
    let mie_mult = params.rayleigh_mie.y;
    let g = params.rayleigh_mie.z;

    let height = 0.001; // Camera at ground level
    let origin = vec3<f32>(0.0, PLANET_RADIUS + height, 0.0);

    let hit = ray_sphere(origin, ray_dir, ATMO_RADIUS);
    if hit.y < 0.0 {
        textureStore(output_tex, gid.xy, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        return;
    }

    // Check ground intersection
    let ground_hit = ray_sphere(origin, ray_dir, PLANET_RADIUS);
    var ray_length = hit.y;
    var hit_ground = false;
    if ground_hit.x > 0.0 {
        ray_length = ground_hit.x;
        hit_ground = true;
    }

    let steps = 32;
    let step_size = ray_length / f32(steps);

    let cos_theta = dot(ray_dir, sun_dir);
    let phase_r = phase_rayleigh(cos_theta);
    let phase_m = phase_mie(cos_theta, g);

    var total = vec3<f32>(0.0);
    var optical_r = 0.0;
    var optical_m = 0.0;
    var optical_oz = 0.0;

    for (var i = 0; i < steps; i++) {
        let t = (f32(i) + 0.5) * step_size;
        let pos = origin + ray_dir * t;
        let alt = max(length(pos) - PLANET_RADIUS, 0.0);
        let d = density_at_height(alt);

        let dr = d.x * step_size;
        let dm = d.y * step_size;
        let doz = ozone_density(alt) * step_size;
        optical_r += dr;
        optical_m += dm;
        optical_oz += doz;

        // View transmittance
        let view_tau = BETA_R * optical_r * rayleigh_mult
            + BETA_M * optical_m * mie_mult
            + BETA_OZ * optical_oz;
        let view_trans = exp(-view_tau);

        // Sun transmittance from LUT
        let pos_norm = normalize(pos);
        let cos_sun_zen = dot(pos_norm, sun_dir);
        let sun_trans = sample_transmittance(cos_sun_zen, alt);

        // Single scatter
        let scatter_r = dr * BETA_R * phase_r * rayleigh_mult;
        let scatter_m = dm * BETA_M * phase_m * mie_mult;
        let single = sun_trans * (scatter_r + scatter_m);

        // Multi-scatter
        let ms = sample_multiscatter(cos_sun_zen, alt);
        let scatter_total_iso = (d.x * BETA_R * rayleigh_mult + d.y * BETA_M * mie_mult) * step_size;
        let multi = ms * scatter_total_iso;

        total += view_trans * (single + multi);
    }

    textureStore(output_tex, gid.xy, vec4<f32>(total, 1.0));
}
