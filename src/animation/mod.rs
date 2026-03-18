//! Animation runtime: pose representation and forward kinematics.

pub mod ik;

use glam::{Mat4, Quat, Vec3};

use crate::subjects::humanoid::HumanoidSkeleton;

/// Per-bone local transform.
#[derive(Clone, Copy, Debug)]
pub struct BoneTransform {
    pub rotation: Quat,
    pub translation: Vec3,
}

impl Default for BoneTransform {
    fn default() -> Self {
        Self {
            rotation: Quat::IDENTITY,
            translation: Vec3::ZERO,
        }
    }
}

/// A pose is a set of per-bone local transforms applied on top of the rest pose.
#[derive(Clone, Debug)]
pub struct Pose {
    pub transforms: Vec<BoneTransform>,
}

impl Pose {
    /// Identity pose (no modification to rest pose).
    pub fn identity(bone_count: usize) -> Self {
        Self {
            transforms: vec![BoneTransform::default(); bone_count],
        }
    }

    /// Evaluate forward kinematics: walk the hierarchy, accumulate world matrices.
    ///
    /// Returns one `Mat4` per bone representing the world-space transform.
    pub fn evaluate(&self, skeleton: &HumanoidSkeleton) -> Vec<Mat4> {
        let bone_count = skeleton.bones.len();
        let mut world_matrices = vec![Mat4::IDENTITY; bone_count];

        for i in 0..bone_count {
            let bone = &skeleton.bones[i];
            let bt = &self.transforms[i];

            let local_translation = bone.rest_position + bt.translation;

            world_matrices[i] = match bone.parent {
                Some(parent_idx) => {
                    let offset = bone.rest_position - skeleton.bones[parent_idx].rest_position;
                    let relative_mat = Mat4::from_rotation_translation(bt.rotation, offset);
                    world_matrices[parent_idx] * relative_mat
                }
                None => Mat4::from_rotation_translation(bt.rotation, local_translation),
            };
        }

        world_matrices
    }

    /// Compute skinning matrices: `world_mat[i] * inverse_bind_mat[i]` for each bone.
    ///
    /// If `inverse_bind_matrices` is provided, uses the cached values.
    /// Otherwise computes them from scratch (allocates).
    pub fn evaluate_skinning_matrices(
        &self,
        skeleton: &HumanoidSkeleton,
        inverse_bind_matrices: Option<&[Mat4]>,
    ) -> Vec<Mat4> {
        let world_matrices = self.evaluate(skeleton);

        if let Some(inv_bind) = inverse_bind_matrices {
            world_matrices
                .iter()
                .zip(inv_bind.iter())
                .map(|(world, inv)| *world * *inv)
                .collect()
        } else {
            let rest_pose = Pose::identity(skeleton.bones.len());
            let bind_matrices = rest_pose.evaluate(skeleton);

            world_matrices
                .iter()
                .zip(bind_matrices.iter())
                .map(|(world, bind)| *world * bind.inverse())
                .collect()
        }
    }

    /// Compute the inverse bind matrices for a skeleton (constant, can be cached).
    pub fn compute_inverse_bind_matrices(skeleton: &HumanoidSkeleton) -> Vec<Mat4> {
        let rest_pose = Pose::identity(skeleton.bones.len());
        let bind_matrices = rest_pose.evaluate(skeleton);
        bind_matrices.iter().map(|m| m.inverse()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::subjects::humanoid::{build_skeleton, HumanoidParams, BONE_COUNT, PELVIS};

    #[test]
    fn identity_pose_produces_correct_count() {
        let skeleton = build_skeleton(&HumanoidParams::default());
        let pose = Pose::identity(BONE_COUNT);
        let matrices = pose.evaluate(&skeleton);
        assert_eq!(matrices.len(), BONE_COUNT);
    }

    #[test]
    fn identity_pose_root_at_pelvis() {
        let params = HumanoidParams::default();
        let skeleton = build_skeleton(&params);
        let pose = Pose::identity(BONE_COUNT);
        let matrices = pose.evaluate(&skeleton);
        let pelvis_pos = matrices[PELVIS].col(3).truncate();
        let expected_y = params.height * params.leg_length_ratio;
        assert!(
            (pelvis_pos.y - expected_y).abs() < 0.01,
            "pelvis at y={} expected ~{}",
            pelvis_pos.y,
            expected_y
        );
    }

    #[test]
    fn identity_skinning_matrices_are_identity() {
        let skeleton = build_skeleton(&HumanoidParams::default());
        let pose = Pose::identity(BONE_COUNT);
        let skinning = pose.evaluate_skinning_matrices(&skeleton, None);
        for (i, mat) in skinning.iter().enumerate() {
            let diff = (*mat - Mat4::IDENTITY).col(3).truncate().length();
            assert!(
                diff < 0.01,
                "bone {i} skinning matrix should be ~identity, translation diff={diff}"
            );
        }
    }
}
