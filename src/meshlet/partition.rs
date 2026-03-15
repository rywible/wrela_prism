use meshopt::{clusterize, VertexDataAdapter};

use crate::scene::Vertex;
use super::bounds::MeshletBounds;

/// Maximum vertices per meshlet (meshopt limit: 256).
pub const MAX_MESHLET_VERTICES: usize = 64;
/// Maximum triangles per meshlet (meshopt limit: 512, must be divisible by 4).
pub const MAX_MESHLET_TRIANGLES: usize = 124;

/// A single meshlet at a specific LOD level.
#[derive(Clone, Debug)]
pub struct MeshletInfo {
    /// Offset into the global vertex index buffer.
    pub vertex_offset: u32,
    /// Number of local vertex indices.
    pub vertex_count: u32,
    /// Offset into the global triangle index buffer.
    pub triangle_offset: u32,
    /// Number of triangles.
    pub triangle_count: u32,
    /// Bounding sphere + normal cone.
    pub bounds: MeshletBounds,
    /// LOD level (0 = finest).
    pub lod_level: u32,
}

/// Result of meshlet partitioning.
pub struct PartitionResult {
    pub meshlets: Vec<MeshletInfo>,
    /// Global vertex indices (each meshlet references a range of these).
    pub meshlet_vertices: Vec<u32>,
    /// Local triangle indices (3 u8s per triangle, packed).
    pub meshlet_triangles: Vec<u8>,
}

/// Build meshlets from a triangle mesh.
pub fn build_meshlets_from_mesh(
    vertices: &[Vertex],
    indices: &[u32],
    lod_level: u32,
) -> PartitionResult {
    let vertex_data = bytemuck::cast_slice::<Vertex, u8>(vertices);
    let vertex_stride = std::mem::size_of::<Vertex>();
    // Position is the first 3 floats (offset 0) in Vertex
    let adapter = VertexDataAdapter::new(vertex_data, vertex_stride, 0)
        .expect("invalid vertex data for meshopt");

    let raw = clusterize::build_meshlets(
        indices,
        &adapter,
        MAX_MESHLET_VERTICES,
        MAX_MESHLET_TRIANGLES,
        0.5, // cone_weight: balance topology and culling
    );

    let mut meshlets = Vec::with_capacity(raw.len());
    for i in 0..raw.len() {
        let m = &raw.meshlets[i];
        let meshlet_ref = raw.get(i);
        let bounds = clusterize::compute_meshlet_bounds(meshlet_ref, &adapter);
        meshlets.push(MeshletInfo {
            vertex_offset: m.vertex_offset,
            vertex_count: m.vertex_count,
            triangle_offset: m.triangle_offset,
            triangle_count: m.triangle_count,
            bounds: MeshletBounds::from_meshopt(&bounds),
            lod_level,
        });
    }

    PartitionResult {
        meshlets,
        meshlet_vertices: raw.vertices,
        meshlet_triangles: raw.triangles,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::{Vertex, MATERIAL_TRUNK};
    use crate::meshlet::make_grid;

    fn make_triangle() -> (Vec<Vertex>, Vec<u32>) {
        let verts = vec![
            Vertex { position: [0.0, 0.0, 0.0], normal: [0.0, 1.0, 0.0], material: MATERIAL_TRUNK, uv: [0.0, 0.0], ao: 1.0 },
            Vertex { position: [1.0, 0.0, 0.0], normal: [0.0, 1.0, 0.0], material: MATERIAL_TRUNK, uv: [1.0, 0.0], ao: 1.0 },
            Vertex { position: [0.0, 0.0, 1.0], normal: [0.0, 1.0, 0.0], material: MATERIAL_TRUNK, uv: [0.0, 1.0], ao: 1.0 },
        ];
        let indices = vec![0, 1, 2];
        (verts, indices)
    }

    #[test]
    fn single_triangle_meshlet() {
        let (verts, indices) = make_triangle();
        let result = build_meshlets_from_mesh(&verts, &indices, 0);
        assert_eq!(result.meshlets.len(), 1);
        assert_eq!(result.meshlets[0].triangle_count, 1);
        assert_eq!(result.meshlets[0].vertex_count, 3);
        assert!(result.meshlets[0].bounds.radius > 0.0);
    }

    #[test]
    fn grid_creates_multiple_meshlets() {
        let (verts, indices) = make_grid(20);
        let result = build_meshlets_from_mesh(&verts, &indices, 0);
        // 20x20 grid = 800 triangles, at 124 tris/meshlet ≈ 7+ meshlets
        assert!(result.meshlets.len() >= 4, "expected multiple meshlets, got {}", result.meshlets.len());

        // All triangles accounted for
        let total_tris: u32 = result.meshlets.iter().map(|m| m.triangle_count).sum();
        assert_eq!(total_tris, 800);
    }

    #[test]
    fn meshlet_bounds_are_valid() {
        let (verts, indices) = make_grid(10);
        let result = build_meshlets_from_mesh(&verts, &indices, 0);
        for m in &result.meshlets {
            assert!(m.bounds.radius > 0.0);
            assert!(m.bounds.radius < 20.0); // reasonable for a 10x10 grid
        }
    }
}
