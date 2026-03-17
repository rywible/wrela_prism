//! Two-bone IK solver for foot placement.

use glam::{Mat4, Quat, Vec3};

use super::Pose;
use crate::subjects::humanoid::*;

/// Analytical two-bone IK solver using law of cosines.
pub fn two_bone_ik(
    hip: Vec3,
    knee: Vec3,
    ankle: Vec3,
    target: Vec3,
    pole_hint: Vec3,
) -> (Quat, Quat) {
    let upper_len = (knee - hip).length();
    let lower_len = (ankle - knee).length();
    let to_target = target - hip;
    let target_dist = to_target.length().min(upper_len + lower_len - 0.001);

    if target_dist < 0.001 {
        return (Quat::IDENTITY, Quat::IDENTITY);
    }

    let cos_knee = ((upper_len * upper_len + lower_len * lower_len - target_dist * target_dist)
        / (2.0 * upper_len * lower_len))
        .clamp(-1.0, 1.0);
    let _knee_angle = std::f32::consts::PI - cos_knee.acos();

    let cos_hip = ((upper_len * upper_len + target_dist * target_dist - lower_len * lower_len)
        / (2.0 * upper_len * target_dist))
        .clamp(-1.0, 1.0);
    let hip_angle = cos_hip.acos();

    let target_dir = to_target.normalize_or_zero();
    let pole_dir = (pole_hint - hip).normalize_or_zero();
    let ik_plane_normal = target_dir.cross(pole_dir).normalize_or_zero();

    if ik_plane_normal.length_squared() < 0.01 {
        return (Quat::IDENTITY, Quat::IDENTITY);
    }

    let original_upper_dir = (knee - hip).normalize_or_zero();
    let desired_upper_dir = Quat::from_axis_angle(ik_plane_normal, -hip_angle) * target_dir;
    let upper_rot = Quat::from_rotation_arc(original_upper_dir, desired_upper_dir);

    let original_lower_dir = (ankle - knee).normalize_or_zero();
    let upper_rotated_lower = upper_rot * original_lower_dir;
    let new_knee = hip + desired_upper_dir * upper_len;
    let desired_lower_dir = (target - new_knee).normalize_or_zero();
    let lower_rot = Quat::from_rotation_arc(upper_rotated_lower, desired_lower_dir);

    (upper_rot, lower_rot)
}

/// Correct foot positions to plant on the ground plane.
pub fn correct_feet(
    pose: &mut Pose,
    skeleton: &HumanoidSkeleton,
    bone_matrices: &[Mat4],
    ground_height: f32,
) {
    correct_single_foot(
        pose,
        bone_matrices,
        ground_height,
        UPPER_LEG_L,
        LOWER_LEG_L,
        FOOT_L,
    );
    correct_single_foot(
        pose,
        bone_matrices,
        ground_height,
        UPPER_LEG_R,
        LOWER_LEG_R,
        FOOT_R,
    );
    let _ = skeleton;
}

fn correct_single_foot(
    pose: &mut Pose,
    bone_matrices: &[Mat4],
    ground_height: f32,
    upper_id: usize,
    lower_id: usize,
    foot_id: usize,
) {
    let hip_world = bone_matrices[upper_id].col(3).truncate();
    let knee_world = bone_matrices[lower_id].col(3).truncate();
    let foot_world = bone_matrices[foot_id].col(3).truncate();

    let foot_error = foot_world.y - ground_height;
    if foot_error.abs() < 0.15 {
        let mut target = foot_world;
        target.y = ground_height;

        let pole = knee_world + Vec3::new(0.0, 0.0, 0.3);

        let (upper_correction, lower_correction) =
            two_bone_ik(hip_world, knee_world, foot_world, target, pole);

        pose.transforms[upper_id].rotation = upper_correction * pose.transforms[upper_id].rotation;
        pose.transforms[lower_id].rotation = lower_correction * pose.transforms[lower_id].rotation;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::animation::Pose;
    use crate::subjects::humanoid::{build_skeleton, HumanoidParams, BONE_COUNT};

    #[test]
    fn feet_near_ground_after_correction() {
        let params = HumanoidParams::default();
        let skeleton = build_skeleton(&params);
        let mut pose = Pose::identity(BONE_COUNT);
        let matrices = pose.evaluate(&skeleton);
        correct_feet(&mut pose, &skeleton, &matrices, 0.0);
        let corrected_matrices = pose.evaluate(&skeleton);

        let foot_l_y = corrected_matrices[FOOT_L].col(3).y;
        let foot_r_y = corrected_matrices[FOOT_R].col(3).y;

        assert!(
            foot_l_y.abs() < 0.2,
            "left foot y={foot_l_y} should be near 0"
        );
        assert!(
            foot_r_y.abs() < 0.2,
            "right foot y={foot_r_y} should be near 0"
        );
    }
}
