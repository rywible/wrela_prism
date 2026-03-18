//! Per-frame humanoid animation: evaluates pose, IK, skinning, and re-uploads vertices.

use glam::Mat4;

use crate::animation::Pose;
use crate::runtime_scene::{AnimatedRegion, RuntimeSceneGpu};
use crate::scene::Vertex;
use crate::subjects::humanoid::{
    build_body_mesh, build_skeleton, HumanoidParams, HumanoidSkeleton, BONE_COUNT,
};
use crate::subjects::humanoid_skin::{compute_skin_weights, skin_mesh, SkinWeights};

pub struct HumanoidAnimator {
    skeleton: HumanoidSkeleton,
    rest_vertices: Vec<Vertex>,
    skin_weights: Vec<SkinWeights>,
    inverse_bind_matrices: Vec<Mat4>,
    pose: Pose,
    meshlet_region: AnimatedRegion,
    shadow_mesh_idx: Option<usize>,
}

impl HumanoidAnimator {
    pub fn new(
        params: &HumanoidParams,
        meshlet_region: AnimatedRegion,
        shadow_mesh_idx: Option<usize>,
    ) -> Self {
        let skeleton = build_skeleton(params);
        let (rest_vertices, _indices) = build_body_mesh(&skeleton, params);
        let skin_weights = compute_skin_weights(&skeleton, &rest_vertices);
        let inverse_bind_matrices = Pose::compute_inverse_bind_matrices(&skeleton);

        Self {
            skeleton,
            rest_vertices,
            skin_weights,
            inverse_bind_matrices,
            pose: Pose::identity(BONE_COUNT),
            meshlet_region,
            shadow_mesh_idx,
        }
    }

    /// Advance animation by `dt` seconds and re-upload deformed vertices.
    pub fn tick(&mut self, _dt: f32, queue: &wgpu::Queue, scene_gpu: &RuntimeSceneGpu) {
        let skinning_matrices = self
            .pose
            .evaluate_skinning_matrices(&self.skeleton, Some(&self.inverse_bind_matrices));
        let deformed = skin_mesh(&self.rest_vertices, &self.skin_weights, &skinning_matrices);

        scene_gpu.update_animated_vertices(queue, &self.meshlet_region, &deformed);

        if let Some(shadow_idx) = self.shadow_mesh_idx {
            scene_gpu.update_shadow_vertices(queue, shadow_idx, &deformed);
        }
    }
}
