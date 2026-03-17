use glam::{Mat4, Vec2, Vec3};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CameraMode {
    FreeCam,
    ThirdPerson,
}

#[derive(Clone, Copy, Debug)]
pub struct ThirdPersonConfig {
    pub arm_length: f32,
    pub arm_height: f32,
    pub shoulder_offset: f32,
    pub min_pitch: f32,
    pub max_pitch: f32,
    pub orbit_sensitivity: f32,
}

impl Default for ThirdPersonConfig {
    fn default() -> Self {
        Self {
            arm_length: 4.0,
            arm_height: 2.5,
            shoulder_offset: 0.8,
            min_pitch: -0.3,
            max_pitch: 1.4,
            orbit_sensitivity: 0.008,
        }
    }
}

/// Third-person orbit state: yaw/pitch around the character.
#[derive(Clone, Copy, Debug)]
pub struct ThirdPersonState {
    pub orbit_yaw: f32,
    pub orbit_pitch: f32,
    pub config: ThirdPersonConfig,
}

impl ThirdPersonState {
    pub fn new(config: ThirdPersonConfig) -> Self {
        Self {
            orbit_yaw: 0.0,
            orbit_pitch: 0.2,
            config,
        }
    }

    pub fn apply_look_delta(&mut self, delta: Vec2) {
        self.orbit_yaw -= delta.x * self.config.orbit_sensitivity;
        self.orbit_pitch = (self.orbit_pitch + delta.y * self.config.orbit_sensitivity)
            .clamp(self.config.min_pitch, self.config.max_pitch);
    }

    /// Compute camera position given the character's feet position.
    pub fn compute_camera(&self, character_pos: Vec3) -> (Vec3, f32, f32) {
        let (yaw_sin, yaw_cos) = self.orbit_yaw.sin_cos();
        let (pitch_sin, pitch_cos) = self.orbit_pitch.sin_cos();

        // Camera sits behind and above the character.
        let arm_back = Vec3::new(
            -yaw_cos * pitch_cos,
            pitch_sin,
            -yaw_sin * pitch_cos,
        ) * self.config.arm_length;

        // Shoulder offset (rightward from camera's perspective).
        let right = Vec3::new(-yaw_sin, 0.0, yaw_cos);
        let shoulder = right * self.config.shoulder_offset;

        let eye_target = character_pos + Vec3::Y * self.config.arm_height;
        let cam_pos = eye_target + arm_back + shoulder;

        (cam_pos, self.orbit_yaw, self.orbit_pitch)
    }

    /// Move the character on the Y=0 plane relative to orbit yaw.
    pub fn move_character(
        &self,
        character_pos: Vec3,
        planar_input: Vec2,
        dt: f32,
        sprint: bool,
        nav: &CameraNavigationConfig,
    ) -> Vec3 {
        let planar = planar_input.clamp_length_max(1.0);
        if planar.length_squared() < 1e-6 {
            return character_pos;
        }
        let speed = nav.base_move_speed
            * 0.5
            * if sprint {
                nav.sprint_multiplier.max(1.0)
            } else {
                1.0
            };
        let flat_forward = Vec3::new(self.orbit_yaw.cos(), 0.0, self.orbit_yaw.sin());
        let flat_right = Vec3::new(-self.orbit_yaw.sin(), 0.0, self.orbit_yaw.cos());
        let displacement = (flat_forward * planar.y + flat_right * planar.x) * speed * dt.max(0.0);
        character_pos + displacement
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CameraBookmark {
    pub position: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub fov_y_degrees: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct CameraNavigationConfig {
    pub min_pitch: f32,
    pub max_pitch: f32,
    pub min_height: f32,
    pub max_height: f32,
    pub roam_center: Vec3,
    pub roam_radius: f32,
    pub base_move_speed: f32,
    pub sprint_multiplier: f32,
    pub mouse_sensitivity: f32,
    pub wheel_dolly_speed: f32,
}

impl Default for CameraNavigationConfig {
    fn default() -> Self {
        Self {
            min_pitch: -1.2,
            max_pitch: 1.2,
            min_height: -1000.0,
            max_height: 3000.0,
            roam_center: Vec3::ZERO,
            roam_radius: 1500.0,
            base_move_speed: 24.0,
            sprint_multiplier: 2.5,
            mouse_sensitivity: 0.010,
            wheel_dolly_speed: 7.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CameraState {
    pub position: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub aspect: f32,
    pub fov_y_radians: f32,
    pub near_plane: f32,
    pub far_plane: f32,
    pub navigation: CameraNavigationConfig,
}

impl CameraState {
    pub fn new(aspect: f32) -> Self {
        Self::from_bookmark(
            aspect,
            CameraBookmark {
                position: Vec3::new(56.0, 23.0, 35.0),
                yaw: 0.56 + std::f32::consts::PI,
                pitch: 0.20,
                fov_y_degrees: 27.0,
            },
        )
    }

    pub fn from_bookmark(aspect: f32, bookmark: CameraBookmark) -> Self {
        let mut camera = Self {
            position: bookmark.position,
            yaw: bookmark.yaw,
            pitch: bookmark.pitch,
            aspect: aspect.max(0.001),
            fov_y_radians: bookmark.fov_y_degrees.to_radians(),
            near_plane: 0.1,
            far_plane: 3000.0,
            navigation: CameraNavigationConfig::default(),
        };
        camera.enforce_constraints();
        camera
    }

    pub fn with_navigation(mut self, navigation: CameraNavigationConfig) -> Self {
        self.navigation = navigation;
        self.enforce_constraints();
        self
    }

    pub fn position(&self) -> Vec3 {
        self.position
    }

    pub fn forward(&self) -> Vec3 {
        let (yaw_sin, yaw_cos) = self.yaw.sin_cos();
        let (pitch_sin, pitch_cos) = self.pitch.sin_cos();
        Vec3::new(yaw_cos * pitch_cos, -pitch_sin, yaw_sin * pitch_cos).normalize()
    }

    pub fn right(&self) -> Vec3 {
        self.forward().cross(Vec3::Y).normalize_or_zero()
    }

    pub fn up(&self) -> Vec3 {
        self.right().cross(self.forward()).normalize_or_zero()
    }

    pub fn apply_look_delta(&mut self, delta: Vec2) {
        self.yaw -= delta.x * self.navigation.mouse_sensitivity;
        self.pitch += delta.y * self.navigation.mouse_sensitivity;
        self.enforce_constraints();
    }

    pub fn move_on_plane(
        &mut self,
        planar_input: Vec2,
        vertical_input: f32,
        dt: f32,
        sprint: bool,
    ) {
        let planar = planar_input.clamp_length_max(1.0);
        let speed = self.navigation.base_move_speed
            * if sprint {
                self.navigation.sprint_multiplier.max(1.0)
            } else {
                1.0
            };
        let flat_forward = Vec3::new(self.forward().x, 0.0, self.forward().z).normalize_or_zero();
        let flat_right = Vec3::new(self.right().x, 0.0, self.right().z).normalize_or_zero();
        let displacement =
            (flat_forward * planar.y + flat_right * planar.x + Vec3::Y * vertical_input)
                * speed
                * dt.max(0.0);
        self.position += displacement;
        self.enforce_constraints();
    }

    pub fn dolly(&mut self, amount: f32) {
        self.position += self.forward() * (amount * self.navigation.wheel_dolly_speed);
        self.enforce_constraints();
    }

    pub fn set_override(
        &mut self,
        position: Option<Vec3>,
        yaw_radians: Option<f32>,
        pitch_radians: Option<f32>,
    ) {
        if let Some(position) = position {
            self.position = position;
        }
        if let Some(yaw_radians) = yaw_radians {
            self.yaw = yaw_radians;
        }
        if let Some(pitch_radians) = pitch_radians {
            self.pitch = pitch_radians;
        }
        self.enforce_constraints();
    }

    pub fn set_aspect(&mut self, width: u32, height: u32) {
        self.aspect = width as f32 / height.max(1) as f32;
    }

    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_to_rh(self.position, self.forward(), Vec3::Y)
    }

    pub fn projection_matrix(&self) -> Mat4 {
        Mat4::perspective_rh(
            self.fov_y_radians,
            self.aspect.max(0.001),
            self.near_plane,
            self.far_plane,
        )
    }

    pub fn view_projection_matrix(&self) -> Mat4 {
        self.projection_matrix() * self.view_matrix()
    }

    fn enforce_constraints(&mut self) {
        self.pitch = self
            .pitch
            .clamp(self.navigation.min_pitch, self.navigation.max_pitch);
        self.position.y = self
            .position
            .y
            .clamp(self.navigation.min_height, self.navigation.max_height);
        let delta = self.position - self.navigation.roam_center;
        let horizontal = Vec2::new(delta.x, delta.z);
        if horizontal.length() > self.navigation.roam_radius.max(0.001) {
            let clamped = horizontal.normalize() * self.navigation.roam_radius.max(0.001);
            self.position.x = self.navigation.roam_center.x + clamped.x;
            self.position.z = self.navigation.roam_center.z + clamped.y;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{CameraBookmark, CameraNavigationConfig, CameraState};
    use glam::{Vec2, Vec3};

    #[test]
    fn camera_matrices_remain_finite_after_look_move_and_dolly() {
        let mut camera = CameraState::from_bookmark(
            16.0 / 9.0,
            CameraBookmark {
                position: Vec3::new(50.0, 18.0, 32.0),
                yaw: 3.7,
                pitch: -0.2,
                fov_y_degrees: 27.0,
            },
        );
        for _ in 0..32 {
            camera.apply_look_delta(Vec2::new(4.0, -2.5));
            camera.move_on_plane(Vec2::new(0.65, 1.0), 0.2, 1.0 / 60.0, true);
            camera.dolly(0.5);
        }

        let view_proj = camera.view_projection_matrix().to_cols_array();
        assert!(view_proj.iter().all(|value| value.is_finite()));
        assert!(camera.position().is_finite());
    }

    #[test]
    fn camera_constraints_keep_freecam_inside_stage_bounds() {
        let mut camera = CameraState::from_bookmark(
            16.0 / 9.0,
            CameraBookmark {
                position: Vec3::new(90.0, 10.0, 90.0),
                yaw: 3.2,
                pitch: 0.0,
                fov_y_degrees: 30.0,
            },
        )
        .with_navigation(CameraNavigationConfig {
            min_pitch: -0.45,
            max_pitch: 1.1,
            min_height: 4.0,
            max_height: 40.0,
            roam_center: Vec3::new(0.0, 10.0, 0.0),
            roam_radius: 64.0,
            base_move_speed: 24.0,
            sprint_multiplier: 2.0,
            mouse_sensitivity: 0.01,
            wheel_dolly_speed: 8.0,
        });

        for _ in 0..48 {
            camera.apply_look_delta(Vec2::new(0.0, 80.0));
            camera.move_on_plane(Vec2::new(1.0, 1.0), -1.0, 1.0 / 30.0, true);
            camera.dolly(3.0);
        }

        let horizontal = Vec2::new(camera.position.x, camera.position.z);
        assert!(camera.position.y >= 4.0 - 0.001);
        assert!(horizontal.length() <= 64.001);
        assert!(camera.pitch >= -0.45 - 0.001);
    }
}
