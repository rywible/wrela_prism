//! Converts a GrowthTree into the existing TreeSkeleton format
//! consumed by tube_mesh and needle_sprays.

use glam::Vec3;

use super::tree::GrowthTree;
use super::GrowthParams;
use crate::subjects::redwood_growth::{BranchKind, RedwoodParams, SkeletonNode, TreeSkeleton};

/// Convert a grown biological tree into the rendering skeleton format.
pub(crate) fn growth_to_skeleton(tree: &GrowthTree) -> TreeSkeleton {
    let mut skeleton = TreeSkeleton::new();

    for seg in &tree.segments {
        let kind = if !seg.is_alive {
            BranchKind::DeadStub
        } else {
            match seg.branch_order {
                0 => BranchKind::Trunk,
                1 => BranchKind::LiveScaffold,
                2 => BranchKind::LiveSecondary,
                _ => BranchKind::LiveFine,
            }
        };

        // Quantize azimuth into 10 sectors for lobe_id.
        let lobe_id = if seg.branch_order > 0 {
            Some(((seg.azimuth / std::f32::consts::TAU * 10.0).floor() as u8) % 10)
        } else {
            None
        };

        skeleton.push(SkeletonNode {
            position: seg.position,
            radius: seg.radius,
            parent: seg.parent,
            depth: seg.branch_order,
            kind,
            lobe_id,
        });
    }

    skeleton
}

/// Generate foliage anchors from the growth tree.
/// Selects the top segments by leaf_area, capped at 300 to keep foliage mesh
/// within GPU buffer limits while maintaining crown coverage.
pub fn growth_foliage_anchors(tree: &GrowthTree) -> Vec<(Vec3, f32)> {
    const MAX_ANCHORS: usize = 300;

    let mut candidates: Vec<(Vec3, f32)> = tree
        .segments
        .iter()
        .filter(|seg| seg.is_alive && seg.leaf_area > 0.1)
        .map(|seg| (seg.position, seg.leaf_area))
        .collect();

    // Sort by leaf_area descending, take the densest clusters.
    candidates.sort_by(|a, b| b.1.total_cmp(&a.1));
    candidates.truncate(MAX_ANCHORS);

    // Convert leaf_area to radius with minimum size for visual coverage.
    candidates
        .into_iter()
        .map(|(pos, area)| (pos, (area.sqrt() * 2.5).max(3.0)))
        .collect()
}

/// Provide the RedwoodParams subset needed by tube_mesh.
/// tube_mesh reads: seed, trunk_height, flute_count, flute_depth.
/// Derives trunk_height from actual grown tree geometry when available.
pub fn growth_mesh_params_from_tree(params: &GrowthParams, tree: &GrowthTree) -> RedwoodParams {
    let trunk_height = tree
        .segments
        .iter()
        .filter(|s| s.branch_order == 0 && s.is_alive)
        .map(|s| s.position.y)
        .fold(0.0f32, |a, b| a.max(b))
        .max(1.0);
    growth_mesh_params_with_height(params, trunk_height)
}

/// Fallback when no tree is available.
pub fn growth_mesh_params(params: &GrowthParams) -> RedwoodParams {
    let trunk_height = params.species.max_height * (params.target_age as f32 / 500.0).min(1.0);
    growth_mesh_params_with_height(params, trunk_height)
}

fn growth_mesh_params_with_height(params: &GrowthParams, trunk_height: f32) -> RedwoodParams {
    RedwoodParams {
        seed: params.seed,
        trunk_height,
        flute_count: params.species.flute_count,
        flute_depth: params.species.flute_depth,
        // Fill remaining fields with defaults (unused by tube_mesh).
        base_radius: params.species.base_diameter / 2.0,
        tip_radius: 0.55,
        trunk_segments: 85,
        trunk_wobble: 0.07,
        crown_start_frac: 0.38,
        branch_count: 95,
        max_branch_depth: 5,
        smooth_k_trunk: 0.30,
        smooth_k_branch: 0.06,
        smooth_k_fine: 0.03,
        columnar_fraction: 0.70,
    }
}

// ──────────────────────── Tests ────────────────────────

#[cfg(test)]
mod tests {
    use super::super::tree::GrowthSegment;
    use super::super::GrowthParams;
    use super::*;

    fn make_grown_tree() -> GrowthTree {
        let mut tree = GrowthTree::new();

        // Build a minimal tree with trunk + branches.
        let mut prev = tree.push(GrowthSegment::new(Vec3::ZERO, None, 0));
        tree.segments[prev].radius = 4.0;
        tree.segments[prev].is_meristem = false;

        for i in 1..10 {
            let pos = Vec3::new(0.0, i as f32 * 7.0, 0.0);
            let mut seg = GrowthSegment::new(pos, Some(prev), 0);
            seg.radius = 4.0 - i as f32 * 0.3;
            seg.is_meristem = i == 9;
            seg.leaf_area = if i > 3 { 2.0 } else { 0.0 };
            prev = tree.push(seg);
        }

        // Add scaffold branch.
        let branch_parent = 5;
        let mut bseg = GrowthSegment::new(Vec3::new(5.0, 35.0, 0.0), Some(branch_parent), 1);
        bseg.radius = 0.8;
        bseg.leaf_area = 3.0;
        bseg.azimuth = 1.2;
        let bidx = tree.push(bseg);

        // Add fine branch.
        let mut fseg = GrowthSegment::new(Vec3::new(7.0, 36.0, 1.0), Some(bidx), 3);
        fseg.radius = 0.2;
        fseg.leaf_area = 1.5;
        fseg.azimuth = 2.0;
        tree.push(fseg);

        // Add dead stub.
        let mut dseg = GrowthSegment::new(Vec3::new(-2.0, 15.0, 0.0), Some(2), 1);
        dseg.is_alive = false;
        dseg.radius = 0.3;
        tree.push(dseg);

        tree
    }

    #[test]
    fn skeleton_has_correct_size() {
        let tree = make_grown_tree();
        let skeleton = growth_to_skeleton(&tree);
        assert_eq!(skeleton.nodes.len(), tree.segments.len());
    }

    #[test]
    fn skeleton_maps_branch_kinds_correctly() {
        let tree = make_grown_tree();
        let skeleton = growth_to_skeleton(&tree);

        // Trunk segments (order 0).
        assert_eq!(skeleton.nodes[0].kind, BranchKind::Trunk);
        assert_eq!(skeleton.nodes[5].kind, BranchKind::Trunk);

        // Scaffold branch (order 1).
        assert_eq!(skeleton.nodes[10].kind, BranchKind::LiveScaffold);

        // Fine branch (order 3).
        assert_eq!(skeleton.nodes[11].kind, BranchKind::LiveFine);

        // Dead stub.
        assert_eq!(skeleton.nodes[12].kind, BranchKind::DeadStub);
    }

    #[test]
    fn trunk_nodes_have_monotonic_y() {
        let tree = make_grown_tree();
        let skeleton = growth_to_skeleton(&tree);

        let trunk_y: Vec<f32> = skeleton
            .nodes
            .iter()
            .filter(|n| n.kind == BranchKind::Trunk)
            .map(|n| n.position.y)
            .collect();

        for window in trunk_y.windows(2) {
            assert!(
                window[1] >= window[0],
                "trunk Y should be monotonically increasing: {trunk_y:?}"
            );
        }
    }

    #[test]
    fn contains_all_branch_kind_variants() {
        let tree = make_grown_tree();
        let skeleton = growth_to_skeleton(&tree);

        let has_trunk = skeleton.nodes.iter().any(|n| n.kind == BranchKind::Trunk);
        let has_scaffold = skeleton
            .nodes
            .iter()
            .any(|n| n.kind == BranchKind::LiveScaffold);
        let has_fine = skeleton
            .nodes
            .iter()
            .any(|n| n.kind == BranchKind::LiveFine);
        let has_dead = skeleton
            .nodes
            .iter()
            .any(|n| n.kind == BranchKind::DeadStub);

        assert!(has_trunk, "should have Trunk nodes");
        assert!(has_scaffold, "should have LiveScaffold nodes");
        assert!(has_fine, "should have LiveFine nodes");
        assert!(has_dead, "should have DeadStub nodes");
    }

    #[test]
    fn determinism_same_seed_same_tree() {
        let params = GrowthParams {
            target_age: 10,
            ..GrowthParams::default()
        };

        let tree_a = super::super::grow_tree_cpu(&params);
        let tree_b = super::super::grow_tree_cpu(&params);

        let skel_a = growth_to_skeleton(&tree_a);
        let skel_b = growth_to_skeleton(&tree_b);

        assert_eq!(skel_a.nodes.len(), skel_b.nodes.len());
        for (a, b) in skel_a.nodes.iter().zip(skel_b.nodes.iter()) {
            assert_eq!(a.position, b.position);
            assert_eq!(a.radius, b.radius);
            assert_eq!(a.kind, b.kind);
        }
    }

    #[test]
    fn foliage_anchors_positive_radius() {
        let tree = make_grown_tree();
        let anchors = growth_foliage_anchors(&tree);

        assert!(!anchors.is_empty(), "should produce foliage anchors");
        for (pos, r) in &anchors {
            assert!(*r > 0.0, "anchor radius must be positive");
            assert!(pos.is_finite(), "anchor position must be finite");
        }
    }

    #[test]
    fn growth_mesh_params_has_required_fields() {
        let params = GrowthParams::default();
        let mesh_params = growth_mesh_params(&params);
        assert!(mesh_params.trunk_height > 0.0);
        assert!(mesh_params.flute_count > 0);
        assert!(mesh_params.flute_depth > 0.0);
    }
}
