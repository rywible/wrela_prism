use meshopt::{simplify as meshopt_simplify, SimplifyOptions, VertexDataAdapter};
use tracing::info;

use crate::scene::Vertex;
use super::bounds::MeshletBounds;
use super::partition::{build_meshlets_from_mesh, MeshletInfo, PartitionResult};

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
        let total_tris: u32 = (prev_level_group_start..prev_level_group_start + prev_level_group_count)
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
                    level, new_group_count, dag.meshlets.len()
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
    let partition_assignments = partition_groups_spatially(
        dag,
        child_group_start,
        child_group_count,
    );

    let num_partitions = partition_assignments.iter().copied().max().map_or(0, |m| m + 1);
    if num_partitions == 0 {
        return None;
    }

    let meshlet_start_before = dag.meshlets.len() as u32;
    dag.level_offsets.push(meshlet_start_before);
    let new_group_start = dag.groups.len() as u32;
    let mut new_group_count = 0u32;

    for partition_id in 0..num_partitions {
        // Collect child groups in this partition
        let child_indices: Vec<u32> = (0..child_group_count)
            .filter(|&i| partition_assignments[i as usize] == partition_id)
            .map(|i| child_group_start + i)
            .collect();

        if child_indices.is_empty() {
            continue;
        }

        // Merge geometry from all meshlets in these child groups
        let (merged_verts, merged_indices) = merge_group_geometry(dag, &child_indices);

        if merged_indices.is_empty() {
            continue;
        }

        // Simplify the merged geometry
        let target_count = ((merged_indices.len() / 3) as f32 * SIMPLIFY_RATIO).max(4.0) as usize * 3;
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

        dag.groups.push(MeshletGroup {
            meshlet_start,
            meshlet_count,
            child_start: child_indices[0],
            child_count: child_indices.len() as u32,
            error: simplify_error,
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

/// Partition groups into spatial clusters.
fn partition_groups_spatially(
    dag: &MeshletDag,
    group_start: u32,
    group_count: u32,
) -> Vec<u32> {
    if group_count <= GROUP_TARGET_SIZE as u32 {
        return vec![0; group_count as usize];
    }

    // Build cluster index data for meshopt's partition_clusters
    let mut cluster_indices = Vec::new();
    let mut cluster_index_counts = Vec::new();

    for i in 0..group_count {
        let group = &dag.groups[(group_start + i) as usize];
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

    let vertex_count = dag.vertices.len();
    let mut assignments = vec![0u32; group_count as usize];

    // Use meshopt's cluster partitioner with vertex positions
    let vertex_data = bytemuck::cast_slice::<Vertex, u8>(&dag.vertices);
    let stride = std::mem::size_of::<Vertex>();
    if let Ok(adapter) = VertexDataAdapter::new(vertex_data, stride, 0) {
        meshopt::clusterize::partition_clusters_with_positions(
            &mut assignments,
            &cluster_indices,
            &cluster_index_counts,
            &adapter,
            GROUP_TARGET_SIZE,
        );
    } else {
        meshopt::clusterize::partition_clusters(
            &mut assignments,
            &cluster_indices,
            &cluster_index_counts,
            vertex_count,
            GROUP_TARGET_SIZE,
        );
    }

    assignments
}

/// Merge all geometry from the given groups into a single vertex/index buffer.
fn merge_group_geometry(
    dag: &MeshletDag,
    group_indices: &[u32],
) -> (Vec<Vertex>, Vec<u32>) {
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
                    let global_vert_idx = dag.meshlet_vertices
                        [m.vertex_offset as usize + local_vert_idx as usize];

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

/// Simplify a mesh, returning new vertices, indices, and the error metric.
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

    let mut result_error = 0.0f32;
    let simplified_indices = meshopt_simplify(
        indices,
        &adapter,
        target_index_count,
        1e-2, // target_error (relative)
        SimplifyOptions::None | SimplifyOptions::Sparse,
        Some(&mut result_error),
    );

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

    (compact_verts, compact_indices, result_error)
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
            level1_tris, level0_tris
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
                i, group.meshlet_start, meshlet_end, dag.meshlets.len()
            );
            assert!(group.bounds.radius > 0.0, "group {} has zero radius", i);
        }
    }

    #[test]
    fn dag_from_trunk_mesh() {
        use crate::subjects::tube_mesh::build_trunk_mesh;
        use crate::subjects::redwood_growth::RedwoodParams;

        let params = RedwoodParams::default();
        let (verts, indices) = build_trunk_mesh(&params);
        let dag = build_meshlet_dag(verts, &indices);

        assert!(!dag.meshlets.is_empty());
        assert!(!dag.groups.is_empty());
        assert!(dag.level_offsets.len() >= 2, "trunk should produce multi-level DAG");
    }
}
