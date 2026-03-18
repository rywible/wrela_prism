use std::collections::BTreeMap;

use meshopt::{simplify as meshopt_simplify, SimplifyOptions, VertexDataAdapter};
use tracing::info;

use super::bounds::MeshletBounds;
use super::partition::{build_meshlets_from_mesh, MeshletInfo, PartitionResult};
use crate::scene::Vertex;

/// A group of meshlets at the same DAG level. Represents a node in the LOD DAG.
#[derive(Clone, Debug)]
pub struct MeshletGroup {
    /// Range of meshlet indices at this group's level.
    pub meshlet_start: u32,
    pub meshlet_count: u32,
    /// Range of child group indices (finer level). Empty = leaf.
    pub child_start: u32,
    pub child_count: u32,
    /// Max screen-space error (in world units) if this group is rendered instead of children.
    pub error: f32,
    /// Bounding sphere encompassing all meshlets in this group.
    pub bounds: MeshletBounds,
    /// LOD level (0 = finest, increasing = coarser).
    pub level: u32,
}

/// The complete meshlet DAG hierarchy.
#[derive(Clone, Debug)]
pub struct MeshletDag {
    /// All meshlets across all LOD levels.
    pub meshlets: Vec<MeshletInfo>,
    /// Global vertex index buffer (meshlet local vertex → original vertex).
    pub meshlet_vertices: Vec<u32>,
    /// Local triangle indices (3 bytes per tri, packed with padding).
    pub meshlet_triangles: Vec<u8>,
    /// DAG group nodes.
    pub groups: Vec<MeshletGroup>,
    /// The original vertices (all levels share a single vertex buffer).
    pub vertices: Vec<Vertex>,
    /// Start index of each level's meshlets in `meshlets`.
    pub level_offsets: Vec<u32>,
}

impl MeshletDag {
    /// Total number of meshlets across all LOD levels.
    pub fn total_meshlet_count(&self) -> usize {
        self.meshlets.len()
    }
}

/// Target group size when partitioning meshlets for simplification.
const GROUP_TARGET_SIZE: usize = 4;
/// Target simplification ratio per level.
const SIMPLIFY_RATIO: f32 = 0.25;
/// Maximum DAG levels to prevent runaway recursion.
const MAX_LEVELS: u32 = 12;
/// Minimum triangle count to stop building further LOD levels.
const MIN_TRIANGLES_TO_SIMPLIFY: usize = 32;

/// Build a complete meshlet DAG from a triangle mesh.
///
/// The finest level (0) contains meshlets built directly from the input.
/// Each coarser level groups meshlets, merges+simplifies geometry, and
/// re-meshletizes the simplified result.
pub fn build_meshlet_dag(vertices: Vec<Vertex>, indices: &[u32]) -> MeshletDag {
    let mut dag = MeshletDag {
        meshlets: Vec::new(),
        meshlet_vertices: Vec::new(),
        meshlet_triangles: Vec::new(),
        groups: Vec::new(),
        vertices,
        level_offsets: Vec::new(),
    };

    // Level 0: build meshlets from full-detail mesh
    let level0 = build_meshlets_from_mesh(&dag.vertices, indices, 0);
    let level0_meshlet_count = level0.meshlets.len();
    dag.level_offsets.push(0);
    append_partition_result(&mut dag, level0);

    info!(
        "meshlet DAG level 0: {} meshlets, {} vertices",
        level0_meshlet_count,
        dag.vertices.len()
    );

    // Build leaf groups for level 0
    let leaf_groups = build_leaf_groups(&dag, 0, level0_meshlet_count as u32);
    let leaf_group_start = dag.groups.len() as u32;
    dag.groups.extend(leaf_groups);

    // Build coarser levels
    let mut prev_level_group_start = leaf_group_start;
    let mut prev_level_group_count = dag.groups.len() as u32 - leaf_group_start;

    for level in 1..=MAX_LEVELS {
        let total_tris: u32 = (prev_level_group_start
            ..prev_level_group_start + prev_level_group_count)
            .map(|gi| {
                let g = &dag.groups[gi as usize];
                (g.meshlet_start..g.meshlet_start + g.meshlet_count)
                    .map(|mi| dag.meshlets[mi as usize].triangle_count)
                    .sum::<u32>()
            })
            .sum();

        if (total_tris as usize) < MIN_TRIANGLES_TO_SIMPLIFY || prev_level_group_count <= 1 {
            break;
        }

        let result = build_coarser_level(
            &mut dag,
            prev_level_group_start,
            prev_level_group_count,
            level,
        );

        match result {
            Some((new_group_start, new_group_count)) => {
                info!(
                    "meshlet DAG level {}: {} groups, {} meshlets total",
                    level,
                    new_group_count,
                    dag.meshlets.len()
                );
                prev_level_group_start = new_group_start;
                prev_level_group_count = new_group_count;
            }
            None => break,
        }
    }

    info!(
        "meshlet DAG complete: {} levels, {} meshlets, {} groups",
        dag.level_offsets.len(),
        dag.meshlets.len(),
        dag.groups.len()
    );

    dag
}

/// Build leaf groups for level 0 meshlets (each meshlet is its own group).
fn build_leaf_groups(dag: &MeshletDag, _level: u32, meshlet_count: u32) -> Vec<MeshletGroup> {
    (0..meshlet_count)
        .map(|i| MeshletGroup {
            meshlet_start: i,
            meshlet_count: 1,
            child_start: 0,
            child_count: 0,
            error: 0.0,
            bounds: dag.meshlets[i as usize].bounds,
            level: 0,
        })
        .collect()
}

/// Build one coarser level of the DAG.
///
/// Groups child groups spatially, merges their geometry, simplifies,
/// re-meshletizes, and creates parent groups.
fn build_coarser_level(
    dag: &mut MeshletDag,
    child_group_start: u32,
    child_group_count: u32,
    level: u32,
) -> Option<(u32, u32)> {
    // Partition child groups into spatial clusters using meshopt
    let partition_assignments =
        partition_groups_spatially(dag, child_group_start, child_group_count);

    let num_partitions = partition_assignments
        .iter()
        .copied()
        .max()
        .map_or(0, |m| m + 1);
    if num_partitions == 0 {
        return None;
    }

    // Parent groups encode children as a contiguous [start, count) range. meshopt's
    // partition assignments are per-child labels only, so we need to physically reorder
    // this level's groups to make each partition contiguous before wiring parent links.
    // This is safe because no coarser level refers to this slice yet.
    let mut partition_sizes = vec![0u32; num_partitions as usize];
    for &partition_id in &partition_assignments {
        partition_sizes[partition_id as usize] += 1;
    }
    reorder_child_groups_by_partition(dag, child_group_start, &partition_assignments);

    let meshlet_start_before = dag.meshlets.len() as u32;
    dag.level_offsets.push(meshlet_start_before);
    let new_group_start = dag.groups.len() as u32;
    let mut new_group_count = 0u32;
    let mut partition_child_offset = 0u32;

    for partition_id in 0..num_partitions {
        let child_count = partition_sizes[partition_id as usize];
        let child_start = child_group_start + partition_child_offset;
        partition_child_offset += child_count;

        let child_indices: Vec<u32> = (0..child_count).map(|i| child_start + i).collect();

        if child_indices.is_empty() {
            continue;
        }

        // Merge geometry from all meshlets in these child groups
        let (merged_verts, merged_indices) = merge_group_geometry(dag, &child_indices);

        if merged_indices.is_empty() {
            continue;
        }

        // Simplify the merged geometry
        let target_count =
            ((merged_indices.len() / 3) as f32 * SIMPLIFY_RATIO).max(4.0) as usize * 3;
        let (simplified_verts, simplified_indices, simplify_error) =
            simplify_mesh(&merged_verts, &merged_indices, target_count);

        if simplified_indices.is_empty() {
            continue;
        }

        // Append simplified vertices to the global vertex buffer
        let vertex_base = dag.vertices.len() as u32;
        dag.vertices.extend_from_slice(&simplified_verts);

        // Remap indices to global vertex space
        let global_indices: Vec<u32> = simplified_indices
            .iter()
            .map(|&i| i + vertex_base)
            .collect();

        // Build meshlets from simplified geometry
        let partition_result = build_meshlets_from_mesh(&dag.vertices, &global_indices, level);
        let meshlet_start = dag.meshlets.len() as u32;
        let meshlet_count = partition_result.meshlets.len() as u32;
        append_partition_result(dag, partition_result);

        // Compute merged bounds for this parent group
        let child_bounds: Vec<MeshletBounds> = child_indices
            .iter()
            .map(|&ci| dag.groups[ci as usize].bounds)
            .collect();
        let merged_bounds = MeshletBounds::merge(&child_bounds);

        // Cumulative error: this level's simplification error plus the worst
        // child error. This ensures error is monotonically non-decreasing up
        // the DAG, which is required for the adaptive screen-space cut to work.
        let max_child_error = child_indices
            .iter()
            .map(|&ci| dag.groups[ci as usize].error)
            .fold(0.0f32, f32::max);
        let cumulative_error = simplify_error + max_child_error;

        // Floor error by the group's spatial extent. Without this, small-radius
        // partitions (thin branches, twigs) report near-zero error even when
        // entire features vanish during simplification. The radius floor ensures
        // the LOD cut refines toward leaves when the camera is close enough to
        // see a group's spatial extent.
        let radius_floor = merged_bounds.radius * 0.15;
        let error = cumulative_error.max(radius_floor);

        dag.groups.push(MeshletGroup {
            meshlet_start,
            meshlet_count,
            child_start,
            child_count,
            error,
            bounds: merged_bounds,
            level,
        });
        new_group_count += 1;
    }

    if new_group_count == 0 {
        // Remove the level offset we added
        dag.level_offsets.pop();
        return None;
    }

    Some((new_group_start, new_group_count))
}

fn reorder_child_groups_by_partition(
    dag: &mut MeshletDag,
    child_group_start: u32,
    partition_assignments: &[u32],
) {
    let child_start = child_group_start as usize;
    let child_end = child_start + partition_assignments.len();
    let child_slice = &mut dag.groups[child_start..child_end];

    let mut order: Vec<usize> = (0..partition_assignments.len()).collect();
    order.sort_by_key(|&idx| partition_assignments[idx]);

    let reordered: Vec<MeshletGroup> = order
        .into_iter()
        .map(|idx| child_slice[idx].clone())
        .collect();
    child_slice.clone_from_slice(&reordered);
}

/// Partition groups into spatial clusters.
fn partition_groups_spatially(dag: &MeshletDag, group_start: u32, group_count: u32) -> Vec<u32> {
    if group_count <= GROUP_TARGET_SIZE as u32 {
        return vec![0; group_count as usize];
    }

    let mut assignments = vec![0u32; group_count as usize];
    let mut groups_by_material = BTreeMap::<u32, Vec<u32>>::new();
    for i in 0..group_count {
        let group = &dag.groups[(group_start + i) as usize];
        groups_by_material
            .entry(group_material_bucket(dag, group))
            .or_default()
            .push(i);
    }

    let vertex_data = bytemuck::cast_slice::<Vertex, u8>(&dag.vertices);
    let stride = std::mem::size_of::<Vertex>();
    let adapter = VertexDataAdapter::new(vertex_data, stride, 0).ok();
    let vertex_count = dag.vertices.len();
    let mut next_partition_id = 0u32;

    for local_group_indices in groups_by_material.values() {
        let mut bucket_assignments = vec![0u32; local_group_indices.len()];

        if local_group_indices.len() > GROUP_TARGET_SIZE {
            let mut cluster_indices = Vec::new();
            let mut cluster_index_counts = Vec::new();
            for &local_idx in local_group_indices {
                let group = &dag.groups[(group_start + local_idx) as usize];
                let mut indices_for_group = Vec::new();
                for mi in group.meshlet_start..group.meshlet_start + group.meshlet_count {
                    let m = &dag.meshlets[mi as usize];
                    let vert_start = m.vertex_offset as usize;
                    let vert_end = vert_start + m.vertex_count as usize;
                    for vi in vert_start..vert_end {
                        indices_for_group.push(dag.meshlet_vertices[vi]);
                    }
                }
                cluster_index_counts.push(indices_for_group.len() as u32);
                cluster_indices.extend(indices_for_group);
            }

            if let Some(adapter) = &adapter {
                meshopt::clusterize::partition_clusters_with_positions(
                    &mut bucket_assignments,
                    &cluster_indices,
                    &cluster_index_counts,
                    adapter,
                    GROUP_TARGET_SIZE,
                );
            } else {
                meshopt::clusterize::partition_clusters(
                    &mut bucket_assignments,
                    &cluster_indices,
                    &cluster_index_counts,
                    vertex_count,
                    GROUP_TARGET_SIZE,
                );
            }
        }

        let bucket_partition_count = bucket_assignments
            .iter()
            .copied()
            .max()
            .map_or(1, |m| m + 1);
        for (bucket_idx, &local_idx) in local_group_indices.iter().enumerate() {
            assignments[local_idx as usize] = next_partition_id + bucket_assignments[bucket_idx];
        }
        next_partition_id += bucket_partition_count;
    }

    assignments
}

fn group_material_bucket(dag: &MeshletDag, group: &MeshletGroup) -> u32 {
    for meshlet_idx in group.meshlet_start..group.meshlet_start + group.meshlet_count {
        let meshlet = &dag.meshlets[meshlet_idx as usize];
        if meshlet.vertex_count == 0 {
            continue;
        }
        let global_vertex_idx = dag.meshlet_vertices[meshlet.vertex_offset as usize] as usize;
        return dag.vertices[global_vertex_idx].material;
    }
    crate::scene::MATERIAL_TRUNK
}

/// Merge all geometry from the given groups into a single vertex/index buffer.
fn merge_group_geometry(dag: &MeshletDag, group_indices: &[u32]) -> (Vec<Vertex>, Vec<u32>) {
    let mut verts = Vec::new();
    let mut indices = Vec::new();
    let mut remap = vec![u32::MAX; dag.vertices.len()];

    for &gi in group_indices {
        let group = &dag.groups[gi as usize];
        for mi in group.meshlet_start..group.meshlet_start + group.meshlet_count {
            let m = &dag.meshlets[mi as usize];
            let tri_start = m.triangle_offset as usize;
            let tri_end = tri_start + m.triangle_count as usize * 3;

            for t in (tri_start..tri_end).step_by(3) {
                for k in 0..3 {
                    let local_vert_idx = dag.meshlet_triangles[t + k] as u32;
                    let global_vert_idx =
                        dag.meshlet_vertices[m.vertex_offset as usize + local_vert_idx as usize];

                    let new_idx = if remap[global_vert_idx as usize] != u32::MAX {
                        remap[global_vert_idx as usize]
                    } else {
                        let idx = verts.len() as u32;
                        verts.push(dag.vertices[global_vert_idx as usize]);
                        remap[global_vert_idx as usize] = idx;
                        idx
                    };
                    indices.push(new_idx);
                }
            }
        }
    }

    (verts, indices)
}

/// Compute the total surface area of a triangle mesh.
fn compute_triangle_area(vertices: &[Vertex], indices: &[u32]) -> f32 {
    let mut area = 0.0f32;
    for tri in indices.chunks(3) {
        if tri.len() < 3 {
            break;
        }
        let a = glam::Vec3::from_array(vertices[tri[0] as usize].position);
        let b = glam::Vec3::from_array(vertices[tri[1] as usize].position);
        let c = glam::Vec3::from_array(vertices[tri[2] as usize].position);
        let e1 = b - a;
        let e2 = c - a;
        area += 0.5 * e1.cross(e2).length();
    }
    area
}

/// Simplify a mesh, returning new vertices, indices, and the error metric.
/// Applies area preservation: if simplification shrinks foliage area significantly,
/// surviving vertices are scaled outward from centroid to compensate.
fn simplify_mesh(
    vertices: &[Vertex],
    indices: &[u32],
    target_index_count: usize,
) -> (Vec<Vertex>, Vec<u32>, f32) {
    let vertex_data = bytemuck::cast_slice::<Vertex, u8>(vertices);
    let stride = std::mem::size_of::<Vertex>();
    let adapter = match VertexDataAdapter::new(vertex_data, stride, 0) {
        Ok(a) => a,
        Err(_) => return (vertices.to_vec(), indices.to_vec(), 0.0),
    };

    // Compute bounding box diagonal to convert meshopt's relative error to world space.
    let (mut bb_min, mut bb_max) = (glam::Vec3::splat(f32::MAX), glam::Vec3::splat(f32::MIN));
    for v in vertices {
        let p = glam::Vec3::from_array(v.position);
        bb_min = bb_min.min(p);
        bb_max = bb_max.max(p);
    }
    let extent = (bb_max - bb_min).length();

    let mut result_error = 0.0f32;
    let simplified_indices = meshopt_simplify(
        indices,
        &adapter,
        target_index_count,
        1e-2, // target_error (relative to mesh extents)
        SimplifyOptions::None | SimplifyOptions::Sparse,
        Some(&mut result_error),
    );

    // meshopt result_error is relative to mesh extents — scale to world units.
    let world_error = result_error * extent;

    if simplified_indices.is_empty() {
        return (vertices.to_vec(), indices.to_vec(), 0.0);
    }

    // Compact vertex buffer to only referenced vertices
    let mut used = vec![false; vertices.len()];
    for &i in &simplified_indices {
        used[i as usize] = true;
    }

    let mut remap = vec![0u32; vertices.len()];
    let mut compact_verts = Vec::new();
    for (i, &is_used) in used.iter().enumerate() {
        if is_used {
            remap[i] = compact_verts.len() as u32;
            compact_verts.push(vertices[i]);
        }
    }

    let compact_indices: Vec<u32> = simplified_indices
        .iter()
        .map(|&i| remap[i as usize])
        .collect();

    // Area preservation: scale surviving vertices outward if area shrank too much
    let input_area = compute_triangle_area(vertices, indices);
    if input_area > 1e-6 {
        let output_area = compute_triangle_area(&compact_verts, &compact_indices);
        let ratio = output_area / input_area;
        if ratio < 0.85 {
            let scale = (input_area / output_area).sqrt().min(2.0);
            // Compute centroid
            let mut centroid = glam::Vec3::ZERO;
            for v in &compact_verts {
                centroid += glam::Vec3::from_array(v.position);
            }
            centroid /= compact_verts.len().max(1) as f32;
            // Scale outward
            for v in &mut compact_verts {
                let pos = glam::Vec3::from_array(v.position);
                let scaled = centroid + (pos - centroid) * scale;
                v.position = scaled.to_array();
            }
        }
    }

    (compact_verts, compact_indices, world_error)
}

/// Append a PartitionResult into the global DAG buffers.
fn append_partition_result(dag: &mut MeshletDag, result: PartitionResult) {
    let vert_offset = dag.meshlet_vertices.len() as u32;
    let tri_offset = dag.meshlet_triangles.len() as u32;

    for mut m in result.meshlets {
        m.vertex_offset += vert_offset;
        m.triangle_offset += tri_offset;
        dag.meshlets.push(m);
    }

    dag.meshlet_vertices.extend(result.meshlet_vertices);
    dag.meshlet_triangles.extend(result.meshlet_triangles);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::meshlet::make_grid;

    #[test]
    fn dag_has_multiple_levels() {
        let (verts, indices) = make_grid(30);
        let dag = build_meshlet_dag(verts, &indices);
        assert!(
            dag.level_offsets.len() >= 2,
            "expected at least 2 LOD levels, got {}",
            dag.level_offsets.len()
        );
        assert!(!dag.groups.is_empty());
    }

    #[test]
    fn dag_level0_accounts_for_all_triangles() {
        let (verts, indices) = make_grid(20);
        let expected_tris = (indices.len() / 3) as u32;
        let dag = build_meshlet_dag(verts, &indices);

        let level0_end = if dag.level_offsets.len() > 1 {
            dag.level_offsets[1]
        } else {
            dag.meshlets.len() as u32
        };

        let level0_tris: u32 = dag.meshlets[..level0_end as usize]
            .iter()
            .map(|m| m.triangle_count)
            .sum();

        assert_eq!(level0_tris, expected_tris);
    }

    #[test]
    fn coarser_levels_have_fewer_triangles() {
        let (verts, indices) = make_grid(30);
        let dag = build_meshlet_dag(verts, &indices);

        if dag.level_offsets.len() < 2 {
            return; // Can't test if only one level
        }

        let level0_end = dag.level_offsets[1] as usize;
        let level0_tris: u32 = dag.meshlets[..level0_end]
            .iter()
            .map(|m| m.triangle_count)
            .sum();

        let level1_end = if dag.level_offsets.len() > 2 {
            dag.level_offsets[2] as usize
        } else {
            dag.meshlets.len()
        };
        let level1_tris: u32 = dag.meshlets[level0_end..level1_end]
            .iter()
            .map(|m| m.triangle_count)
            .sum();

        assert!(
            level1_tris < level0_tris,
            "level 1 ({} tris) should have fewer triangles than level 0 ({} tris)",
            level1_tris,
            level0_tris
        );
    }

    #[test]
    fn dag_groups_have_valid_references() {
        let (verts, indices) = make_grid(20);
        let dag = build_meshlet_dag(verts, &indices);

        for (i, group) in dag.groups.iter().enumerate() {
            let meshlet_end = group.meshlet_start + group.meshlet_count;
            assert!(
                (meshlet_end as usize) <= dag.meshlets.len(),
                "group {} meshlet range {}..{} exceeds meshlets len {}",
                i,
                group.meshlet_start,
                meshlet_end,
                dag.meshlets.len()
            );
            assert!(group.bounds.radius > 0.0, "group {} has zero radius", i);
        }
    }

    #[test]
    fn every_non_root_group_has_exactly_one_parent() {
        let (verts, indices) = make_grid(30);
        let dag = build_meshlet_dag(verts, &indices);

        let mut parent_counts = vec![0u32; dag.groups.len()];
        for group in &dag.groups {
            for child_idx in group.child_start..group.child_start + group.child_count {
                parent_counts[child_idx as usize] += 1;
            }
        }

        let root_count = parent_counts.iter().filter(|&&count| count == 0).count();
        assert!(root_count >= 1, "expected at least one root group");

        for (idx, &parent_count) in parent_counts.iter().enumerate() {
            if parent_count == 0 {
                continue;
            }
            assert_eq!(
                parent_count, 1,
                "group {} should have exactly one parent, found {}",
                idx, parent_count
            );
        }
    }

    #[test]
    fn dag_from_trunk_mesh() {
        use crate::subjects::redwood_growth::RedwoodParams;
        use crate::subjects::tube_mesh::build_trunk_mesh;

        let params = RedwoodParams::default();
        let (verts, indices) = build_trunk_mesh(&params);
        let dag = build_meshlet_dag(verts, &indices);

        assert!(!dag.meshlets.is_empty());
        assert!(!dag.groups.is_empty());
        assert!(
            dag.level_offsets.len() >= 2,
            "trunk should produce multi-level DAG"
        );
    }
}
