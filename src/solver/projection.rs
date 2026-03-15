use glam::{Mat4, Vec3};

use crate::scene::bounds::BoundingSphere;

/// Compute the screen-space diameter (in pixels) of a bounding sphere.
///
/// Uses analytical angular projection (camera-orientation invariant).
/// Returns 0.0 if the sphere is behind the camera or fully off-screen.
pub fn screen_diameter(
    sphere: &BoundingSphere,
    view_proj: Mat4,
    camera_position: Vec3,
    fov_y_radians: f32,
    screen_width: f32,
    screen_height: f32,
) -> f32 {
    // Project center to clip space for behind-camera and frustum checks
    let center_clip = view_proj * sphere.center.extend(1.0);

    // Behind camera
    if center_clip.w <= 0.0 {
        return 0.0;
    }

    let ndc = center_clip.truncate() / center_clip.w;

    // Analytical angular diameter — independent of camera orientation
    let dist_sq = camera_position.distance_squared(sphere.center);
    let rad_sq = sphere.radius * sphere.radius;
    if dist_sq <= rad_sq {
        // Camera inside the sphere — fills the screen
        return screen_width.max(screen_height);
    }

    // Exact perspective projection of the sphere's silhouette cone:
    // half-angle = asin(r/d), tan(half-angle) = r / sqrt(d² - r²)
    // NDC half-extent = tan(half-angle) / tan(fov_y/2)
    // pixel_diameter = 2 × NDC_half × screen_height/2
    let half_fov_tan = (fov_y_radians * 0.5).tan();
    let pixel_diameter = sphere.radius * screen_height / ((dist_sq - rad_sq).sqrt() * half_fov_tan);

    // Frustum culling using per-axis NDC radius
    let ndc_radius_x = pixel_diameter / screen_width;
    let ndc_radius_y = pixel_diameter / screen_height;
    if ndc.x - ndc_radius_x > 1.0
        || ndc.x + ndc_radius_x < -1.0
        || ndc.y - ndc_radius_y > 1.0
        || ndc.y + ndc_radius_y < -1.0
    {
        return 0.0;
    }

    pixel_diameter.max(0.0)
}

/// Estimate pixel count covered by a bounding sphere (circle approximation).
pub fn estimated_pixel_count(screen_diam: f32) -> f32 {
    let r = screen_diam * 0.5;
    std::f32::consts::PI * r * r
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Mat4, Vec3};

    const CAM_POS: Vec3 = Vec3::new(0.0, 0.0, 50.0);
    const FOV_Y: f32 = std::f32::consts::FRAC_PI_4; // 45 degrees

    fn test_vp() -> Mat4 {
        let view = Mat4::look_at_rh(CAM_POS, Vec3::ZERO, Vec3::Y);
        let proj = Mat4::perspective_rh(FOV_Y, 16.0 / 9.0, 0.1, 250.0);
        proj * view
    }

    #[test]
    fn sphere_at_origin_has_nonzero_diameter() {
        let sphere = BoundingSphere {
            center: Vec3::ZERO,
            radius: 5.0,
        };
        let d = screen_diameter(&sphere, test_vp(), CAM_POS, FOV_Y, 1920.0, 1080.0);
        assert!(d > 50.0, "expected large screen diameter, got {d}");
    }

    #[test]
    fn sphere_behind_camera_returns_zero() {
        let sphere = BoundingSphere {
            center: Vec3::new(0.0, 0.0, 100.0),
            radius: 5.0,
        };
        let d = screen_diameter(&sphere, test_vp(), CAM_POS, FOV_Y, 1920.0, 1080.0);
        assert!(d < 1.0, "sphere behind camera should be ~0, got {d}");
    }

    #[test]
    fn closer_sphere_is_larger() {
        let vp = test_vp();
        let near = BoundingSphere {
            center: Vec3::new(0.0, 0.0, 10.0),
            radius: 5.0,
        };
        let far = BoundingSphere {
            center: Vec3::new(0.0, 0.0, -30.0),
            radius: 5.0,
        };
        let dn = screen_diameter(&near, vp, CAM_POS, FOV_Y, 1920.0, 1080.0);
        let df = screen_diameter(&far, vp, CAM_POS, FOV_Y, 1920.0, 1080.0);
        assert!(
            dn > df,
            "near sphere ({dn}) should be larger than far sphere ({df})"
        );
    }

    #[test]
    fn top_down_view_still_works() {
        // Camera looking straight down — Vec3::Y offset would give 0 in old code
        let cam = Vec3::new(0.0, 50.0, 0.0);
        let view = Mat4::look_at_rh(cam, Vec3::ZERO, Vec3::Z);
        let proj = Mat4::perspective_rh(FOV_Y, 16.0 / 9.0, 0.1, 250.0);
        let vp = proj * view;
        let sphere = BoundingSphere {
            center: Vec3::ZERO,
            radius: 5.0,
        };
        let d = screen_diameter(&sphere, vp, cam, FOV_Y, 1920.0, 1080.0);
        assert!(
            d > 50.0,
            "top-down view should still produce valid diameter, got {d}"
        );
    }

    #[test]
    fn wide_aspect_ratio_uses_correct_scaling() {
        let sphere = BoundingSphere {
            center: Vec3::ZERO,
            radius: 5.0,
        };
        // Same sphere at same distance, different aspect ratios — vertical FOV is fixed
        // so pixel diameter (based on screen_height) should be the same
        let d_normal = screen_diameter(&sphere, test_vp(), CAM_POS, FOV_Y, 1920.0, 1080.0);

        let cam = CAM_POS;
        let view = Mat4::look_at_rh(cam, Vec3::ZERO, Vec3::Y);
        let proj_wide = Mat4::perspective_rh(FOV_Y, 32.0 / 9.0, 0.1, 250.0);
        let vp_wide = proj_wide * view;
        let d_wide = screen_diameter(&sphere, vp_wide, cam, FOV_Y, 3840.0, 1080.0);

        // Same screen_height and fov_y → same pixel diameter
        assert!(
            (d_normal - d_wide).abs() < 1.0,
            "same vertical FOV should give same diameter: normal={d_normal}, wide={d_wide}"
        );
    }
}
