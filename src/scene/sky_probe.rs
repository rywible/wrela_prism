//! CPU-side atmospheric scattering for ambient probe computation.
//!
//! Ports the Rayleigh+Mie scattering math from `sky_pass.wgsl` to Rust,
//! evaluating 6 hemisphere directions to produce a physically-grounded
//! ambient probe that tracks sun position automatically.

use glam::Vec3;

use super::SceneSettings;

// Planet / atmosphere geometry (km) — same constants as sky_pass.wgsl
const PLANET_RADIUS: f32 = 6360.0;
const ATMO_RADIUS: f32 = 6420.0;
const H_RAYLEIGH: f32 = 8.5;
const H_MIE: f32 = 1.2;
const BETA_R: Vec3 = Vec3::new(5.8e-3, 13.5e-3, 33.1e-3);
const BETA_M: f32 = 2.1e-2;
const BETA_OZ: Vec3 = Vec3::new(0.65e-3, 1.88e-3, 0.085e-3);
const PI: f32 = std::f32::consts::PI;

const NUM_SAMPLES: usize = 12;
const NUM_LIGHT_SAMPLES: usize = 6;

/// 6-face ambient probe computed from atmospheric scattering.
pub struct SkyProbe {
    pub up: Vec3,
    pub down: Vec3,
    pub right: Vec3,
    pub left: Vec3,
    pub front: Vec3,
    pub back: Vec3,
}

fn ozone_density(altitude: f32) -> f32 {
    (1.0 - ((altitude - 25.0).abs() / 15.0)).max(0.0)
}

fn cloud_direct_visibility(settings: &SceneSettings, sun_height: f32) -> f32 {
    let coverage = settings.cloud_coverage.clamp(0.0, 1.0);
    let thickness = ((settings.cloud_profile.cloud_top_km - settings.cloud_profile.cloud_base_km)
        / 2.0)
        .clamp(0.6, 1.8);
    let density = (settings.cloud_profile.density_scale / 24.0).clamp(0.4, 1.2);
    let occlusion = coverage * (0.14 + 0.08 * sun_height + 0.05 * thickness + 0.04 * density);
    (1.0 - occlusion).clamp(0.68, 1.0)
}

fn cloud_diffuse_boost(settings: &SceneSettings, sun_height: f32) -> f32 {
    let coverage = settings.cloud_coverage.clamp(0.0, 1.0);
    let overcast = coverage * (1.0 - 0.25 * sun_height);
    1.0 + overcast * 0.10
}

/// Ray-sphere intersection. Returns (t_near, t_far) or (-1, -1) on miss.
fn ray_sphere(origin: Vec3, dir: Vec3, radius: f32) -> (f32, f32) {
    let b = origin.dot(dir);
    let c = origin.dot(origin) - radius * radius;
    let disc = b * b - c;
    if disc < 0.0 {
        return (-1.0, -1.0);
    }
    let s = disc.sqrt();
    (-b - s, -b + s)
}

/// Rayleigh phase function.
fn phase_rayleigh(cos_theta: f32) -> f32 {
    3.0 / (16.0 * PI) * (1.0 + cos_theta * cos_theta)
}

/// Dual-lobe Henyey-Greenstein phase function for Mie scattering.
/// Must match sky_lut.wgsl phase_mie: forward peak (g) + back-scatter (g=-0.3), 75/25 mix.
fn phase_mie(cos_theta: f32, g: f32) -> f32 {
    let g2_fwd = g * g;
    let fwd = (1.0 - g2_fwd) / (4.0 * PI * (1.0 + g2_fwd - 2.0 * g * cos_theta).powf(1.5));
    let g_back = -0.3_f32;
    let g2_back = g_back * g_back;
    let back = (1.0 - g2_back) / (4.0 * PI * (1.0 + g2_back - 2.0 * g_back * cos_theta).powf(1.5));
    fwd * 0.75 + back * 0.25
}

/// Compute optical depth along a ray segment through the atmosphere.
fn optical_depth(origin: Vec3, dir: Vec3, length: f32) -> (f32, f32, f32) {
    let step_size = length / NUM_LIGHT_SAMPLES as f32;
    let mut depth_r = 0.0_f32;
    let mut depth_m = 0.0_f32;
    let mut depth_oz = 0.0_f32;
    for i in 0..NUM_LIGHT_SAMPLES {
        let pos = origin + dir * (i as f32 + 0.5) * step_size;
        let altitude = (pos.length() - PLANET_RADIUS).max(0.0);
        depth_r += (-altitude / H_RAYLEIGH).exp();
        depth_m += (-altitude / H_MIE).exp();
        depth_oz += ozone_density(altitude);
    }
    (
        depth_r * step_size,
        depth_m * step_size,
        depth_oz * step_size,
    )
}

/// Evaluate single-scattering sky radiance along a given direction.
fn sky_radiance(ray_dir: Vec3, sun_dir: Vec3, settings: &SceneSettings) -> Vec3 {
    let rayleigh_mult = settings.rayleigh_strength;
    let mie_mult = settings.mie_strength;
    let g = settings.mie_anisotropy;
    let sun_height = sun_dir.y.max(0.0);
    let sun_visibility = cloud_direct_visibility(settings, sun_height);
    let diffuse_boost = cloud_diffuse_boost(settings, sun_height);

    let ray_origin = Vec3::new(0.0, PLANET_RADIUS + 0.001, 0.0);
    let atmo_hit = ray_sphere(ray_origin, ray_dir, ATMO_RADIUS);
    if atmo_hit.1 < 0.0 {
        return Vec3::ZERO;
    }

    let t_start = atmo_hit.0.max(0.0);
    let t_end = atmo_hit.1;
    let step_size = (t_end - t_start) / NUM_SAMPLES as f32;

    let cos_theta = ray_dir.dot(sun_dir);
    let phase_r = phase_rayleigh(cos_theta);
    let phase_m = phase_mie(cos_theta, g);

    let mut total_r = Vec3::ZERO;
    let mut total_m = Vec3::ZERO;
    let mut optical_r = 0.0_f32;
    let mut optical_m = 0.0_f32;
    let mut optical_oz = 0.0_f32;

    for i in 0..NUM_SAMPLES {
        let t = t_start + (i as f32 + 0.5) * step_size;
        let pos = ray_origin + ray_dir * t;
        let altitude = (pos.length() - PLANET_RADIUS).max(0.0);

        let density_r = (-altitude / H_RAYLEIGH).exp() * step_size;
        let density_m = (-altitude / H_MIE).exp() * step_size;

        optical_r += density_r;
        optical_m += density_m;
        optical_oz += ozone_density(altitude) * step_size;

        let sun_hit = ray_sphere(pos, sun_dir, ATMO_RADIUS);
        if sun_hit.1 > 0.0 {
            let (light_depth_r, light_depth_m, light_depth_oz) =
                optical_depth(pos, sun_dir, sun_hit.1);

            let tau = BETA_R * (optical_r + light_depth_r) * rayleigh_mult
                + Vec3::splat(BETA_M * (optical_m + light_depth_m) * mie_mult)
                + BETA_OZ * (optical_oz + light_depth_oz);
            let transmittance = Vec3::new((-tau.x).exp(), (-tau.y).exp(), (-tau.z).exp());

            total_r += transmittance * density_r;
            total_m += transmittance * density_m;
        }
    }

    let sun_color = settings.sun_color * settings.sun_strength * sun_visibility;
    sun_color
        * (total_r * BETA_R * phase_r * rayleigh_mult + total_m * BETA_M * phase_m * mie_mult)
        * diffuse_boost
}

/// Compute a 6-face ambient probe from atmospheric scattering.
///
/// Uses 26 hemisphere samples per face (plus cross-face bleed) for a smooth
/// probe that tracks the sky-view LUT parameters. Falls back to per-direction
/// analytical scattering for the base, then scales by `ambient_intensity`.
/// CPU cost is negligible (~100us for 6 × 26 evaluations).
pub fn compute_sky_probe(settings: &SceneSettings, ambient_intensity: f32) -> SkyProbe {
    let sun_dir = settings.sun_direction.normalize();
    let sun_height = sun_dir.y.max(0.0);
    let sun_visibility = cloud_direct_visibility(settings, sun_height);
    let diffuse_boost = cloud_diffuse_boost(settings, sun_height);
    // 6 cardinal directions with hemisphere sample offsets
    let directions = [
        Vec3::Y,     // up
        Vec3::NEG_Y, // down
        Vec3::X,     // right
        Vec3::NEG_X, // left
        Vec3::Z,     // front
        Vec3::NEG_Z, // back
    ];

    // For each face, integrate over a hemisphere with cosine weighting.
    // Use 26 Fibonacci-spiral samples for smooth coverage.
    let num_samples = 26;
    let golden_ratio = (1.0 + 5.0_f32.sqrt()) * 0.5;

    let mut radiances = [Vec3::ZERO; 6];

    for (face, &face_dir) in directions.iter().enumerate() {
        // Build tangent frame for this face
        let up = if face_dir.y.abs() > 0.9 {
            Vec3::Z
        } else {
            Vec3::Y
        };
        let tangent = face_dir.cross(up).normalize();
        let bitangent = tangent.cross(face_dir).normalize();

        let mut total = Vec3::ZERO;
        let mut weight_sum = 0.0_f32;

        for i in 0..num_samples {
            let fi = i as f32;
            // Cosine-weighted hemisphere sampling
            let cos_theta = ((fi + 0.5) / num_samples as f32).sqrt();
            let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
            let phi = 2.0 * PI * fi / golden_ratio;

            let local = Vec3::new(sin_theta * phi.cos(), sin_theta * phi.sin(), cos_theta);
            let world_dir =
                (tangent * local.x + bitangent * local.y + face_dir * local.z).normalize();

            let radiance = sky_radiance(world_dir, sun_dir, settings);
            let w = cos_theta; // cosine weight
            total += radiance * w;
            weight_sum += w;
        }

        if weight_sum > 0.0 {
            total /= weight_sum;
        }

        // Scale by sun color and multi-scatter approximation
        radiances[face] = total * 4.0 * ambient_intensity;
    }

    // Multi-scatter isotropic term
    let sky_zenith = settings.sky_zenith * settings.sky_strength;
    let multi_scatter = sky_zenith
        * (0.10 + settings.cloud_coverage.clamp(0.0, 1.0) * 0.03)
        * ambient_intensity
        * diffuse_boost;
    for r in &mut radiances {
        *r += multi_scatter;
    }

    // Directional ground-bounce tint: warm light scattered upward from earth
    let ground_tint =
        Vec3::new(0.06, 0.04, 0.025) * settings.sun_strength * sun_visibility * ambient_intensity;
    radiances[1] += ground_tint; // down face gets ground bounce

    // Floor to prevent pitch-black shadows
    let floor = Vec3::splat(0.08) * ambient_intensity;
    for r in &mut radiances {
        *r = r.max(floor);
    }

    SkyProbe {
        up: radiances[0],
        down: radiances[1],
        right: radiances[2],
        left: radiances[3],
        front: radiances[4],
        back: radiances[5],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::soundstage::redwood_stage;

    #[test]
    fn cloud_cover_reduces_direct_sun_visibility() {
        let mut clear = redwood_stage::hero().scene_settings;
        clear.cloud_coverage = 0.05;

        let mut overcast = redwood_stage::hero().scene_settings;
        overcast.cloud_coverage = 0.95;

        let clear_vis = cloud_direct_visibility(&clear, clear.sun_direction.normalize().y.max(0.0));
        let overcast_vis =
            cloud_direct_visibility(&overcast, overcast.sun_direction.normalize().y.max(0.0));

        assert!(clear_vis > overcast_vis);
        assert!(overcast_vis >= 0.35);
    }
}
