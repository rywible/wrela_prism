use glam::Vec3;

use crate::material::MaterialId;
use crate::scene::{GeometryDef, SceneGraph, SceneHandle, SdfTree};
use crate::subjects::redwood_growth::{
    build_trunk_sdf_grown, generate_foliage_anchors_grown, trunk_capsule_data_grown, RedwoodParams,
};

pub const MATERIAL_BARK: MaterialId = MaterialId(0);

/// Build the redwood trunk as a smooth union of tapered capsules.
pub fn build_redwood_trunk(graph: &mut SceneGraph) -> SceneHandle {
    let tree = build_trunk_sdf();
    graph.add(GeometryDef::Sdf(tree), MATERIAL_BARK)
}

/// Build the SDF tree for the redwood trunk.
pub fn build_trunk_sdf() -> SdfTree {
    build_trunk_sdf_grown(&RedwoodParams::default())
}

/// Get all trunk capsule segments for GPU upload.
pub fn trunk_capsule_data() -> Vec<(Vec3, Vec3, f32, f32)> {
    trunk_capsule_data_grown(&RedwoodParams::default())
}

/// Generate foliage anchor positions from branch tips and along the canopy.
/// Returns (anchor_position, cluster_radius) pairs.
pub fn generate_foliage_anchors() -> Vec<(Vec3, f32)> {
    generate_foliage_anchors_grown(&RedwoodParams::default())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scene_graph_with_redwood() {
        let mut graph = SceneGraph::new();
        let handle = build_redwood_trunk(&mut graph);
        let node = graph.node(handle).unwrap();
        // Bounds should encompass the trunk
        assert!(node.world_bounds.min.y <= 0.0);
        assert!(node.world_bounds.max.y >= 20.0);
        // Lipschitz should be 1.0
        assert!((node.lipschitz_constant - 1.0).abs() < 1e-6);
    }
}
