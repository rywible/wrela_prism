pub mod redwood_stage;

use crate::camera::{CameraBookmark, CameraNavigationConfig, CameraState, ThirdPersonConfig};
use crate::scene::SceneSettings;
use glam::Vec3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LookDevPreset {
    Hero,
    LowAngle,
    Silhouette,
    NeutralDebug,
}

impl LookDevPreset {
    pub fn hero_default() -> Self {
        Self::Hero
    }

    pub fn from_name(value: &str) -> Option<Self> {
        match value {
            "hero" => Some(Self::Hero),
            "low_angle" => Some(Self::LowAngle),
            "silhouette" => Some(Self::Silhouette),
            "neutral_debug" => Some(Self::NeutralDebug),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SoundstageLayout {
    pub ground_radius: f32,
    pub ground_thickness: f32,
    pub shadow_center: Vec3,
    pub shadow_focus_radius: f32,
    pub shadow_depth: f32,
}

pub struct SoundstageConfig {
    pub scene_settings: SceneSettings,
    pub layout: SoundstageLayout,
    pub camera_bookmark: CameraBookmark,
    pub camera_navigation: CameraNavigationConfig,
    pub third_person: ThirdPersonConfig,
    pub clear_color: [f64; 4],
}

impl SoundstageConfig {
    pub fn camera(&self, aspect: f32) -> CameraState {
        CameraState::from_bookmark(aspect, self.camera_bookmark)
            .with_navigation(self.camera_navigation)
    }
}

pub fn soundstage_for_preset(preset: LookDevPreset) -> SoundstageConfig {
    match preset {
        LookDevPreset::Hero => redwood_stage::hero(),
        LookDevPreset::LowAngle => redwood_stage::low_angle(),
        LookDevPreset::Silhouette => redwood_stage::silhouette(),
        LookDevPreset::NeutralDebug => redwood_stage::neutral_debug(),
    }
}
