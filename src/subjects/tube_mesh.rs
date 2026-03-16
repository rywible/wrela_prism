//! Tube extrusion mesh generation for tree branches.
//!
//! Replaces SDF + marching cubes with direct polygon tube extrusion:
//! polygon rings along each skeleton path, stitched into quads.

use glam::Vec3;

use crate::art_direction::pack_semantic_channels;
use crate::scene::{Vertex, MATERIAL_TRUNK};
use crate::subjects::redwood_growth::{BranchKind, RedwoodParams, SkeletonNode, TreeSkeleton};
use crate::util::hash01;

// ──────────────────────── Helpers ────────────────────────

/// Integer hash → [-1, 1).
fn signed_hash(n: u32) -> f32 {
    crate::util::hash01(n) * 2.0 - 1.0
}

/// Number of polygon sides for a tube ring, based on max radius.
fn segment_sides(max_radius: f32) -> usize {
    if max_radius > 2.0 {
        16
    } else if max_radius > 0.8 {
        12
    } else if max_radius > 0.3 {
        8
    } else {
        6
    }
}

// ──────────────────────── Path Collection ────────────────────────

/// Build a children index: children[i] = list of node indices whose parent is i.
pub(crate) fn build_children_index(skeleton: &TreeSkeleton) -> Vec<Vec<usize>> {
    let mut children = vec![Vec::new(); skeleton.nodes.len()];
    for (idx, node) in skeleton.nodes.iter().enumerate() {
        if let Some(parent_idx) = node.parent {
            children[parent_idx].push(idx);
        }
    }
    children
}

/// Collect trunk path: node indices where kind == Trunk, ordered root → tip.
pub(crate) fn collect_trunk_path(skeleton: &TreeSkeleton) -> Vec<usize> {
    let children = build_children_index(skeleton);
    let mut path = Vec::new();
    // Root is always node 0, should be Trunk.
    if skeleton.nodes.is_empty() {
        return path;
    }
    let mut cur = 0usize;
    loop {
        if skeleton.nodes[cur].kind != BranchKind::Trunk {
            break;
        }
        path.push(cur);
        // Follow the trunk child (there should be exactly one trunk child per trunk node).
        let trunk_child = children[cur]
            .iter()
            .copied()
            .find(|&c| skeleton.nodes[c].kind == BranchKind::Trunk);
        match trunk_child {
            Some(next) => cur = next,
            None => break,
        }
    }
    path
}

/// Collect branch paths (non-trunk chains) from fork/trunk attachment to tip/next fork.
///
/// Each path is a `Vec<usize>` starting from a node that has a trunk or fork parent
/// and extending until the node has no children or reaches a fork (multiple children).
pub(crate) fn collect_branch_paths(
    skeleton: &TreeSkeleton,
    children: &[Vec<usize>],
) -> Vec<Vec<usize>> {
    let mut paths = Vec::new();
    let mut visited = vec![false; skeleton.nodes.len()];

    // Mark all trunk nodes as visited so we don't walk them.
    for (idx, node) in skeleton.nodes.iter().enumerate() {
        if node.kind == BranchKind::Trunk {
            visited[idx] = true;
        }
    }

    // Find branch roots: non-trunk nodes whose parent is a trunk node or already visited.
    for (idx, node) in skeleton.nodes.iter().enumerate() {
        if visited[idx] {
            continue;
        }
        if let Some(parent_idx) = node.parent {
            if !visited[parent_idx] && skeleton.nodes[parent_idx].kind != BranchKind::Trunk {
                // This node's parent is a non-trunk, non-visited node. It will be collected
                // as part of another chain starting earlier.
                continue;
            }
        }
        // Start a chain from this node.
        let mut chain = Vec::new();
        let mut cur = idx;
        loop {
            if visited[cur] {
                break;
            }
            visited[cur] = true;
            chain.push(cur);
            // Continue along the chain if there's exactly one non-trunk child.
            let non_trunk_children: Vec<usize> = children[cur]
                .iter()
                .copied()
                .filter(|&c| skeleton.nodes[c].kind != BranchKind::Trunk && !visited[c])
                .collect();
            if non_trunk_children.len() == 1 {
                cur = non_trunk_children[0];
            } else {
                // Fork or tip — end this chain.
                // If fork, the remaining children will be picked up as new chain roots
                // on subsequent iterations.
                break;
            }
        }
        if !chain.is_empty() {
            paths.push(chain);
        }
    }

    paths
}

// ──────────────────────── Tube Frames ────────────────────────

/// A reference frame at a point along a tube path.
struct TubeFrame {
    center: Vec3,
    right: Vec3,
    up: Vec3,
    radius: f32,
    height_ratio: f32,
}

/// Build twist-free reference frames along a path of skeleton nodes.
fn build_tube_frames(nodes: &[SkeletonNode], path: &[usize], trunk_height: f32) -> Vec<TubeFrame> {
    if path.is_empty() {
        return Vec::new();
    }
    let mut frames = Vec::with_capacity(path.len());
    let mut prev_right = Vec3::X;

    for (i, &ni) in path.iter().enumerate() {
        let node = &nodes[ni];
        let center = node.position;

        // Forward-differencing tangent.
        let tangent = if i + 1 < path.len() {
            let next = &nodes[path[i + 1]];
            (next.position - center).normalize_or_zero()
        } else if i > 0 {
            let prev = &nodes[path[i - 1]];
            (center - prev.position).normalize_or_zero()
        } else {
            Vec3::Y
        };

        // Project previous right onto the plane perpendicular to tangent (twist-free).
        let projected = (prev_right - tangent * prev_right.dot(tangent)).normalize_or_zero();
        let right = if projected.length_squared() > 0.01 {
            projected
        } else {
            // Degenerate case: pick an arbitrary perpendicular.
            let arbitrary = if tangent.y.abs() > 0.9 {
                Vec3::X
            } else {
                Vec3::Y
            };
            tangent.cross(arbitrary).normalize()
        };
        let up = tangent.cross(right).normalize();

        let height_ratio = if trunk_height > 0.0 {
            (center.y / trunk_height).clamp(0.0, 1.0)
        } else {
            0.0
        };

        prev_right = right;

        frames.push(TubeFrame {
            center,
            right,
            up,
            radius: node.radius,
            height_ratio,
        });
    }

    frames
}

// ──────────────────────── Bark Profile ────────────────────────

/// Bark wobble scale for a point around a ring. Returns a multiplier in [0.88, 1.12].
fn bark_profile(seed: u32, frame: &TubeFrame, around_t: f32, is_main_trunk: bool) -> f32 {
    let theta = around_t * std::f32::consts::TAU;
    let phase_a = hash01(seed.wrapping_add(7)) * std::f32::consts::TAU;
    let phase_b = hash01(seed.wrapping_add(13)) * std::f32::consts::TAU;
    let phase_c = hash01(seed.wrapping_add(29)) * std::f32::consts::TAU;

    // Macro shape: fewer broader undulations around the circumference.
    let macro_freq = if is_main_trunk { 3.0 } else { 2.0 };
    let macro_amp = if is_main_trunk { 0.040 } else { 0.020 };
    let macro_wave = (theta * macro_freq + phase_b).sin() * macro_amp;

    // Keep the mesh bark contribution height-stable; the actual trunk growth curve
    // already provides the flare, and extra lower-trunk-only shaping creates bands.
    let furrow_phase = frame.height_ratio * if is_main_trunk { 3.0 } else { 5.0 };
    let primary_furrow_amp = if is_main_trunk { 0.020 } else { 0.013 };
    let primary_furrow = (theta * (if is_main_trunk { 8.0 } else { 6.0 }) + furrow_phase + phase_c)
        .sin()
        * primary_furrow_amp;
    let secondary_furrow = (theta * (if is_main_trunk { 13.0 } else { 9.0 }) - furrow_phase * 1.4
        + phase_a * 0.7)
        .sin()
        * if is_main_trunk { 0.007 } else { 0.006 };

    // High-frequency noise.
    let noise_seed = seed
        .wrapping_mul(31)
        .wrapping_add((around_t * 100.0) as u32);
    let noise = signed_hash(noise_seed) * if is_main_trunk { 0.008 } else { 0.010 };

    let scale = 1.0 + macro_wave + primary_furrow + secondary_furrow + noise;
    scale.clamp(0.84, 1.20)
}

// ──────────────────────── Tube Extrusion ────────────────────────

/// Extrude polygon rings along a path and stitch them into a triangle mesh.
fn append_tube_along_path(
    verts: &mut Vec<Vertex>,
    indices: &mut Vec<u32>,
    nodes: &[SkeletonNode],
    path: &[usize],
    seed: u32,
    is_main_trunk: bool,
) {
    if path.len() < 2 {
        return;
    }

    let trunk_height = nodes
        .iter()
        .map(|n| n.position.y)
        .fold(0.0f32, |a, b| a.max(b));

    let max_radius = path
        .iter()
        .map(|&ni| nodes[ni].radius)
        .fold(0.0f32, |a, b| a.max(b));
    let sides = segment_sides(max_radius);
    let ring_columns = sides + 1;

    let frames = build_tube_frames(nodes, path, trunk_height);

    let base_vert = verts.len() as u32;

    // Emit rings.
    for (fi, frame) in frames.iter().enumerate() {
        // Compute curvature from radius change between adjacent frames
        let curvature = if frames.len() >= 2 {
            let prev_r = if fi > 0 {
                frames[fi - 1].radius
            } else {
                frame.radius
            };
            let next_r = if fi + 1 < frames.len() {
                frames[fi + 1].radius
            } else {
                frame.radius
            };
            ((prev_r - next_r).abs() / max_radius.max(0.01)).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let edge_sharpness = if fi == 0 || fi == frames.len() - 1 {
            0.9 // High at junctions/tips
        } else {
            0.3 + curvature * 0.4
        };
        let surface_noise = 0.5; // Moderate — bark is textured
        let importance = (frame.radius / max_radius.max(0.01)).clamp(0.0, 1.0);
        let sc = pack_semantic_channels(curvature, edge_sharpness, surface_noise, importance);

        for s in 0..=sides {
            let around_t = s as f32 / sides as f32;
            let theta = around_t * std::f32::consts::TAU;

            let bark_scale = bark_profile(seed, frame, around_t, is_main_trunk);

            let r = frame.radius * bark_scale;
            let offset = frame.right * (theta.cos() * r) + frame.up * (theta.sin() * r);
            let position = frame.center + offset;
            let normal = offset.normalize_or_zero();

            let u = around_t;
            let v = frame.height_ratio;
            let ao = 1.0;

            verts.push(Vertex {
                position: position.into(),
                normal: [normal.x, normal.y, normal.z],
                material: MATERIAL_TRUNK,
                feature_id: 0,
                uv: [u, v],
                ao,
                semantic_channels: sc,
            });
        }
    }

    // Stitch adjacent rings into quads.
    let ring_count = frames.len();
    for ring_i in 0..(ring_count - 1) {
        let row0 = base_vert + (ring_i * ring_columns) as u32;
        let row1 = base_vert + ((ring_i + 1) * ring_columns) as u32;
        for s in 0..sides {
            let a = row0 + s as u32;
            let b = row0 + s as u32 + 1;
            let c = row1 + s as u32;
            let d = row1 + s as u32 + 1;
            // Two triangles per quad.
            indices.extend_from_slice(&[a, c, b, b, c, d]);
        }
    }
}

/// Append a 2-ring collar at a branch junction (slightly flared).
fn append_branch_collar(
    verts: &mut Vec<Vertex>,
    indices: &mut Vec<u32>,
    root: Vec3,
    branch_first: Vec3,
    parent_radius: f32,
    branch_radius: f32,
    seed: u32,
    _salt: u32,
    trunk_height: f32,
) {
    let dir = (branch_first - root).normalize_or_zero();
    if dir.length_squared() < 0.01 {
        return;
    }

    let arbitrary = if dir.y.abs() > 0.9 { Vec3::X } else { Vec3::Y };
    let right = dir.cross(arbitrary).normalize();
    let up = right.cross(dir).normalize();

    let collar_len = (branch_first - root).length().min(parent_radius * 0.5);
    let flare_radius = branch_radius * 1.25;

    let sides = segment_sides(parent_radius);
    let ring_columns = sides + 1;
    let base_vert = verts.len() as u32;

    let height_ratio_root = if trunk_height > 0.0 {
        (root.y / trunk_height).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let height_ratio_end = if trunk_height > 0.0 {
        ((root + dir * collar_len).y / trunk_height).clamp(0.0, 1.0)
    } else {
        0.0
    };

    // Collar semantic channels: high edge_sharpness at branch junction
    let collar_importance = (branch_radius / parent_radius.max(0.01)).clamp(0.0, 1.0);
    let collar_sc = pack_semantic_channels(0.6, 0.9, 0.5, collar_importance);

    // Ring 0: at root, flared radius.
    for s in 0..=sides {
        let around_t = s as f32 / sides as f32;
        let theta = around_t * std::f32::consts::TAU;
        let bark_scale = 1.0 + signed_hash(seed.wrapping_add(s as u32 * 7)) * 0.02;
        let r = flare_radius * bark_scale;
        let offset = right * (theta.cos() * r) + up * (theta.sin() * r);
        let position = root + offset;
        let normal = offset.normalize_or_zero();
        let ao = 1.0;
        verts.push(Vertex {
            position: position.into(),
            normal: [normal.x, normal.y, normal.z],
            material: MATERIAL_TRUNK,
            feature_id: 0,
            uv: [around_t, height_ratio_root],
            ao,
            semantic_channels: collar_sc,
        });
    }

    // Ring 1: slightly along branch, normal radius.
    for s in 0..=sides {
        let around_t = s as f32 / sides as f32;
        let theta = around_t * std::f32::consts::TAU;
        let bark_scale = 1.0 + signed_hash(seed.wrapping_add(s as u32 * 11 + 200)) * 0.02;
        let r = branch_radius * bark_scale;
        let offset = right * (theta.cos() * r) + up * (theta.sin() * r);
        let position = root + dir * collar_len + offset;
        let normal = offset.normalize_or_zero();
        let ao = 1.0;
        verts.push(Vertex {
            position: position.into(),
            normal: [normal.x, normal.y, normal.z],
            material: MATERIAL_TRUNK,
            feature_id: 0,
            uv: [around_t, height_ratio_end],
            ao,
            semantic_channels: collar_sc,
        });
    }

    // Stitch the two rings.
    let row0 = base_vert;
    let row1 = base_vert + ring_columns as u32;
    for s in 0..sides {
        let a = row0 + s as u32;
        let b = row0 + s as u32 + 1;
        let c = row1 + s as u32;
        let d = row1 + s as u32 + 1;
        indices.extend_from_slice(&[a, c, b, b, c, d]);
    }
}

/// Append a fan-triangulated disk cap at a branch tip.
fn append_cap_disk(
    verts: &mut Vec<Vertex>,
    indices: &mut Vec<u32>,
    center: Vec3,
    axis: Vec3,
    radius: f32,
    trunk_height: f32,
) {
    let axis = axis.normalize_or_zero();
    if axis.length_squared() < 0.01 {
        return;
    }

    let arbitrary = if axis.y.abs() > 0.9 { Vec3::X } else { Vec3::Y };
    let right = axis.cross(arbitrary).normalize();
    let up = right.cross(axis).normalize();

    let sides = segment_sides(radius);
    let height_ratio = if trunk_height > 0.0 {
        (center.y / trunk_height).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let ao = 1.0;
    // Cap disk: low importance, high edge at tip
    let cap_sc = pack_semantic_channels(0.1, 0.7, 0.3, 0.2);

    let center_idx = verts.len() as u32;
    verts.push(Vertex {
        position: center.into(),
        normal: [axis.x, axis.y, axis.z],
        material: MATERIAL_TRUNK,
        feature_id: 0,
        uv: [0.5, height_ratio],
        ao,
        semantic_channels: cap_sc,
    });

    for s in 0..sides {
        let theta = (s as f32 / sides as f32) * std::f32::consts::TAU;
        let offset = right * (theta.cos() * radius) + up * (theta.sin() * radius);
        let position = center + offset;
        verts.push(Vertex {
            position: position.into(),
            normal: [axis.x, axis.y, axis.z],
            material: MATERIAL_TRUNK,
            feature_id: 0,
            uv: [0.5 + theta.cos() * 0.5, height_ratio],
            ao,
            semantic_channels: cap_sc,
        });
    }

    for s in 0..sides {
        let s_next = (s + 1) % sides;
        indices.extend_from_slice(&[
            center_idx,
            center_idx + 1 + s as u32,
            center_idx + 1 + s_next as u32,
        ]);
    }
}

// ──────────────────────── Public Entry Point ────────────────────────

/// Build a complete trunk mesh from skeleton parameters using tube extrusion.
///
/// Returns (vertices, indices) ready for GPU upload.
pub fn build_trunk_mesh(params: &RedwoodParams) -> (Vec<Vertex>, Vec<u32>) {
    let skeleton = crate::subjects::redwood_growth::build_skeleton(params);
    build_trunk_mesh_from_skeleton(params, &skeleton)
}

/// Build trunk mesh from a pre-built skeleton (avoids rebuilding when skeleton is shared).
pub(crate) fn build_trunk_mesh_from_skeleton(
    params: &RedwoodParams,
    skeleton: &TreeSkeleton,
) -> (Vec<Vertex>, Vec<u32>) {
    let children = build_children_index(skeleton);

    let mut verts = Vec::new();
    let mut indices = Vec::new();

    // 1. Extrude trunk tube.
    let trunk_path = collect_trunk_path(&skeleton);
    if !trunk_path.is_empty() {
        append_tube_along_path(
            &mut verts,
            &mut indices,
            &skeleton.nodes,
            &trunk_path,
            params.seed as u32,
            true,
        );

        // 2. Trunk tip cap.
        let tip_idx = *trunk_path.last().unwrap();
        let tip_pos = skeleton.nodes[tip_idx].position;
        let tip_radius = skeleton.nodes[tip_idx].radius;
        let tip_tangent = if trunk_path.len() >= 2 {
            let prev_idx = trunk_path[trunk_path.len() - 2];
            (tip_pos - skeleton.nodes[prev_idx].position).normalize_or_zero()
        } else {
            Vec3::Y
        };
        append_cap_disk(
            &mut verts,
            &mut indices,
            tip_pos,
            tip_tangent,
            tip_radius,
            params.trunk_height,
        );
    }

    // 4. Branch paths.
    let branch_paths = collect_branch_paths(&skeleton, &children);
    for (path_i, path) in branch_paths.iter().enumerate() {
        if path.len() < 2 {
            continue;
        }

        if matches!(skeleton.nodes[path[0]].kind, BranchKind::Buttress) {
            continue;
        }

        let branch_seed = params.seed as u32 + 1000 + path_i as u32;
        let branch_root_kind = skeleton.nodes[path[0]].kind;
        let use_structural_bark = branch_root_kind.is_structural_trunk();

        // Extrude the branch tube.
        append_tube_along_path(
            &mut verts,
            &mut indices,
            &skeleton.nodes,
            path,
            branch_seed,
            use_structural_bark,
        );

        // Branch collars help aerial limbs, but buttresses should blend into the
        // trunk base instead of creating a visible ring.
        let first_node = &skeleton.nodes[path[0]];
        if !matches!(first_node.kind, BranchKind::Buttress) {
            if let Some(parent_idx) = first_node.parent {
                let parent_pos = skeleton.nodes[parent_idx].position;
                let parent_radius = skeleton.nodes[parent_idx].radius;
                let branch_radius = first_node.radius;
                append_branch_collar(
                    &mut verts,
                    &mut indices,
                    parent_pos,
                    first_node.position,
                    parent_radius,
                    branch_radius,
                    branch_seed,
                    path_i as u32,
                    params.trunk_height,
                );
            }
        }

        // Branch tip cap.
        let tip_idx = *path.last().unwrap();
        let tip_pos = skeleton.nodes[tip_idx].position;
        let tip_radius = skeleton.nodes[tip_idx].radius;
        let tip_tangent = if path.len() >= 2 {
            let prev_idx = path[path.len() - 2];
            (tip_pos - skeleton.nodes[prev_idx].position).normalize_or_zero()
        } else {
            Vec3::Y
        };
        append_cap_disk(
            &mut verts,
            &mut indices,
            tip_pos,
            tip_tangent,
            tip_radius,
            params.trunk_height,
        );
    }

    (verts, indices)
}

// ──────────────────────── Tests ────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::subjects::redwood_growth::{build_skeleton, BranchKind, RedwoodParams};

    #[test]
    fn children_index_matches_parent_links() {
        let params = RedwoodParams::default();
        let skeleton = build_skeleton(&params);
        let children = build_children_index(&skeleton);

        // Every node's parent should list that node as a child.
        for (idx, node) in skeleton.nodes.iter().enumerate() {
            if let Some(parent_idx) = node.parent {
                assert!(
                    children[parent_idx].contains(&idx),
                    "node {} has parent {} but is not in children[{}]",
                    idx,
                    parent_idx,
                    parent_idx
                );
            }
        }

        // Every entry in children[p] should have parent == p.
        for (parent_idx, child_list) in children.iter().enumerate() {
            for &child_idx in child_list {
                assert_eq!(
                    skeleton.nodes[child_idx].parent,
                    Some(parent_idx),
                    "children[{}] contains {} but that node's parent is {:?}",
                    parent_idx,
                    child_idx,
                    skeleton.nodes[child_idx].parent
                );
            }
        }
    }

    #[test]
    fn branch_paths_cover_all_non_trunk_nodes() {
        let params = RedwoodParams::default();
        let skeleton = build_skeleton(&params);
        let children = build_children_index(&skeleton);
        let paths = collect_branch_paths(&skeleton, &children);

        let mut covered: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for path in &paths {
            for &idx in path {
                assert!(
                    covered.insert(idx),
                    "node {} appears in multiple branch paths",
                    idx
                );
            }
        }

        // Every non-trunk node should appear in exactly one path.
        for (idx, node) in skeleton.nodes.iter().enumerate() {
            if node.kind != BranchKind::Trunk {
                assert!(
                    covered.contains(&idx),
                    "non-trunk node {} (kind={:?}) not covered by any branch path",
                    idx,
                    node.kind
                );
            }
        }
    }

    #[test]
    fn extrude_two_node_path_produces_valid_mesh() {
        let nodes = vec![
            SkeletonNode {
                position: Vec3::ZERO,
                radius: 1.0,
                parent: None,
                depth: 0,
                kind: BranchKind::Trunk,
                lobe_id: None,
            },
            SkeletonNode {
                position: Vec3::new(0.0, 5.0, 0.0),
                radius: 0.5,
                parent: Some(0),
                depth: 0,
                kind: BranchKind::Trunk,
                lobe_id: None,
            },
        ];
        let path = vec![0, 1];
        let mut verts = Vec::new();
        let mut indices = Vec::new();

        append_tube_along_path(&mut verts, &mut indices, &nodes, &path, 42, true);

        // 2 rings * (sides + 1) columns (extra column for UV seam wrap)
        let sides = segment_sides(1.0);
        let ring_columns = sides + 1;
        assert_eq!(
            verts.len(),
            2 * ring_columns,
            "expected 2 rings of {} vertices",
            ring_columns
        );

        // (ring_count - 1) * sides * 6 indices (2 triangles per quad, 3 indices per tri)
        let expected_indices = 1 * sides * 6;
        assert_eq!(
            indices.len(),
            expected_indices,
            "expected {} indices",
            expected_indices
        );

        // All indices in bounds.
        for &idx in &indices {
            assert!(
                (idx as usize) < verts.len(),
                "index {} out of bounds (verts.len() = {})",
                idx,
                verts.len()
            );
        }

        // All materials are MATERIAL_TRUNK.
        for v in &verts {
            assert_eq!(
                v.material, MATERIAL_TRUNK,
                "all vertices should be trunk material"
            );
        }

        // All normals are finite and roughly unit length.
        for v in &verts {
            let n = Vec3::from(v.normal);
            assert!(n.is_finite(), "normal should be finite");
            assert!(
                (n.length() - 1.0).abs() < 0.1,
                "normal should be roughly unit length, got {}",
                n.length()
            );
        }
    }

    #[test]
    fn build_trunk_mesh_produces_renderable_geometry() {
        let params = RedwoodParams::default();
        let (verts, indices) = build_trunk_mesh(&params);

        // Should have nontrivial geometry.
        assert!(
            verts.len() > 100,
            "expected substantial vertex count, got {}",
            verts.len()
        );
        assert!(
            indices.len() > 100,
            "expected substantial index count, got {}",
            indices.len()
        );

        // All indices in bounds.
        for &idx in &indices {
            assert!(
                (idx as usize) < verts.len(),
                "index {} out of bounds (verts.len() = {})",
                idx,
                verts.len()
            );
        }

        // All materials == MATERIAL_TRUNK.
        for v in &verts {
            assert_eq!(
                v.material, MATERIAL_TRUNK,
                "all vertices should have trunk material"
            );
        }

        // Index count is a multiple of 3 (triangles).
        assert_eq!(
            indices.len() % 3,
            0,
            "index count should be a multiple of 3"
        );
    }
}
