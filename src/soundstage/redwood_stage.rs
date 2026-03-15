use glam::Vec3;

use super::{SoundstageConfig, SoundstageLayout};
use crate::camera::{CameraBookmark, CameraNavigationConfig};
use crate::scene::SceneSettings;

fn stage_layout() -> SoundstageLayout {
    SoundstageLayout {
        ground_radius: 320.0,
        ground_thickness: 7.5,
        shadow_center: Vec3::new(0.0, 11.0, 0.0),
        shadow_focus_radius: 200.0,
        shadow_depth: 420.0,
    }
}

fn camera_navigation() -> CameraNavigationConfig {
    CameraNavigationConfig {
        min_pitch: -0.65,
        max_pitch: 1.15,
        min_height: 2.5,
        max_height: 82.0,
        roam_center: Vec3::new(0.0, 11.0, 0.0),
        roam_radius: 118.0,
        base_move_speed: 24.0,
        sprint_multiplier: 2.4,
        mouse_sensitivity: 0.010,
        wheel_dolly_speed: 7.0,
    }
}

pub fn hero() -> SoundstageConfig {
    SoundstageConfig {
        scene_settings: SceneSettings {
            sun_direction: Vec3::new(-0.52, 0.58, -0.38).normalize(),
            sun_color: Vec3::new(1.00, 0.94, 0.84),
            sun_strength: 1.34,
            sun_angular_radius: 0.0105,
            fog_density: 0.0010,
            fog_height_falloff: 0.17,
            fog_start: 120.0,
            fog_end: 420.0,
            fog_color: Vec3::new(0.56, 0.67, 0.82),
            fog_sky_mix: 0.01,
            sky_zenith: Vec3::new(0.20, 0.36, 0.70),
            sky_horizon: Vec3::new(0.72, 0.82, 0.94),
            sky_strength: 0.22,
            rayleigh_strength: 1.05,
            mie_strength: 0.18,
            mie_anisotropy: 0.76,
            horizon_haze: 0.28,
            shaft_intensity: 0.18,
            shaft_decay: 0.95,
            ambient_up: Vec3::new(0.30, 0.34, 0.42),
            ambient_down: Vec3::new(0.15, 0.12, 0.09),
            ambient_right: Vec3::new(0.17, 0.18, 0.21),
            ambient_left: Vec3::new(0.15, 0.17, 0.20),
            ambient_front: Vec3::new(0.16, 0.18, 0.21),
            ambient_back: Vec3::new(0.15, 0.16, 0.19),
            exposure: 1.06,
            tonemap_strength: 1.0,
            wind: crate::scene::WindSettings::default(),
            contact_shadow_strength: 0.18,
        },
        layout: stage_layout(),
        camera_bookmark: CameraBookmark {
            position: Vec3::new(74.250, 24.600, 48.500),
            yaw: 0.57 + std::f32::consts::PI,
            pitch: 0.16,
            fov_y_degrees: 23.5,
        },
        camera_navigation: camera_navigation(),
        clear_color: [0.18, 0.32, 0.58, 1.0],
    }
}

pub fn low_angle() -> SoundstageConfig {
    SoundstageConfig {
        scene_settings: SceneSettings {
            sun_direction: Vec3::new(-0.60, 0.56, 0.30).normalize(),
            sun_color: Vec3::new(0.99, 0.95, 0.88),
            sun_strength: 1.18,
            sun_angular_radius: 0.0100,
            fog_density: 0.0012,
            fog_height_falloff: 0.17,
            fog_start: 110.0,
            fog_end: 400.0,
            fog_color: Vec3::new(0.53, 0.64, 0.79),
            fog_sky_mix: 0.01,
            sky_zenith: Vec3::new(0.18, 0.33, 0.64),
            sky_horizon: Vec3::new(0.68, 0.79, 0.92),
            sky_strength: 0.20,
            rayleigh_strength: 0.98,
            mie_strength: 0.18,
            mie_anisotropy: 0.78,
            horizon_haze: 0.30,
            shaft_intensity: 0.16,
            shaft_decay: 0.95,
            ambient_up: Vec3::new(0.24, 0.28, 0.38),
            ambient_down: Vec3::new(0.15, 0.12, 0.09),
            ambient_right: Vec3::new(0.14, 0.16, 0.18),
            ambient_left: Vec3::new(0.14, 0.16, 0.18),
            ambient_front: Vec3::new(0.14, 0.16, 0.18),
            ambient_back: Vec3::new(0.14, 0.16, 0.18),
            exposure: 1.03,
            tonemap_strength: 1.0,
            wind: crate::scene::WindSettings::default(),
            contact_shadow_strength: 0.16,
        },
        layout: stage_layout(),
        camera_bookmark: CameraBookmark {
            position: Vec3::new(72.400, 14.200, -41.300),
            yaw: -0.50 + std::f32::consts::PI,
            pitch: 0.05,
            fov_y_degrees: 21.5,
        },
        camera_navigation: camera_navigation(),
        clear_color: [0.16, 0.30, 0.54, 1.0],
    }
}

pub fn silhouette() -> SoundstageConfig {
    SoundstageConfig {
        scene_settings: SceneSettings {
            sun_direction: Vec3::new(-0.58, 0.55, -0.30).normalize(),
            sun_color: Vec3::new(0.97, 0.94, 0.89),
            sun_strength: 1.26,
            sun_angular_radius: 0.0100,
            fog_density: 0.0013,
            fog_height_falloff: 0.15,
            fog_start: 100.0,
            fog_end: 390.0,
            fog_color: Vec3::new(0.48, 0.58, 0.72),
            fog_sky_mix: 0.00,
            sky_zenith: Vec3::new(0.15, 0.28, 0.52),
            sky_horizon: Vec3::new(0.64, 0.74, 0.88),
            sky_strength: 0.21,
            rayleigh_strength: 0.94,
            mie_strength: 0.22,
            mie_anisotropy: 0.79,
            horizon_haze: 0.34,
            shaft_intensity: 0.14,
            shaft_decay: 0.94,
            ambient_up: Vec3::new(0.19, 0.22, 0.30),
            ambient_down: Vec3::new(0.12, 0.10, 0.07),
            ambient_right: Vec3::new(0.12, 0.13, 0.16),
            ambient_left: Vec3::new(0.12, 0.13, 0.16),
            ambient_front: Vec3::new(0.12, 0.13, 0.16),
            ambient_back: Vec3::new(0.12, 0.13, 0.16),
            exposure: 1.04,
            tonemap_strength: 1.0,
            wind: crate::scene::WindSettings::default(),
            contact_shadow_strength: 0.15,
        },
        layout: stage_layout(),
        camera_bookmark: CameraBookmark {
            position: Vec3::new(29.300, 25.613, 54.827),
            yaw: 1.08 + std::f32::consts::PI,
            pitch: 0.24,
            fov_y_degrees: 24.0,
        },
        camera_navigation: camera_navigation(),
        clear_color: [0.14, 0.26, 0.48, 1.0],
    }
}

pub fn neutral_debug() -> SoundstageConfig {
    SoundstageConfig {
        scene_settings: SceneSettings {
            sun_direction: Vec3::new(-0.42, 0.60, -0.50).normalize(),
            sun_color: Vec3::splat(1.0),
            sun_strength: 1.10,
            sun_angular_radius: 0.0098,
            fog_density: 0.0009,
            fog_height_falloff: 0.10,
            fog_start: 130.0,
            fog_end: 420.0,
            fog_color: Vec3::new(0.57, 0.67, 0.78),
            fog_sky_mix: 0.02,
            sky_zenith: Vec3::new(0.30, 0.42, 0.60),
            sky_horizon: Vec3::new(0.74, 0.82, 0.92),
            sky_strength: 0.21,
            rayleigh_strength: 0.86,
            mie_strength: 0.14,
            mie_anisotropy: 0.70,
            horizon_haze: 0.22,
            shaft_intensity: 0.08,
            shaft_decay: 0.96,
            ambient_up: Vec3::new(0.34, 0.38, 0.45),
            ambient_down: Vec3::new(0.20, 0.16, 0.14),
            ambient_right: Vec3::new(0.30, 0.31, 0.31),
            ambient_left: Vec3::new(0.30, 0.31, 0.31),
            ambient_front: Vec3::new(0.30, 0.31, 0.31),
            ambient_back: Vec3::new(0.30, 0.31, 0.31),
            exposure: 1.10,
            tonemap_strength: 0.88,
            wind: crate::scene::WindSettings::default(),
            contact_shadow_strength: 0.08,
        },
        layout: stage_layout(),
        camera_bookmark: CameraBookmark {
            position: Vec3::new(37.109, 22.311, 39.771),
            yaw: 0.82 + std::f32::consts::PI,
            pitch: 0.24,
            fov_y_degrees: 32.0,
        },
        camera_navigation: camera_navigation(),
        clear_color: [0.34, 0.42, 0.55, 1.0],
    }
}
