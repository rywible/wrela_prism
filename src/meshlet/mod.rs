pub mod bounds;
pub mod gpu_buffers;
pub mod partition;
pub mod simplify;

pub use bounds::MeshletBounds;
pub use gpu_buffers::GpuMeshletBuffers;
pub use partition::MeshletInfo;
pub use simplify::{build_meshlet_dag, MeshletDag, MeshletGroup};

#[cfg(test)]
pub(crate) fn make_grid(n: usize) -> (Vec<crate::scene::Vertex>, Vec<u32>) {
    use crate::scene::{Vertex, MATERIAL_TRUNK};

    let mut verts = Vec::new();
    for z in 0..=n {
        for x in 0..=n {
            verts.push(Vertex {
                position: [x as f32, 0.0, z as f32],
                normal: [0.0, 1.0, 0.0],
                material: MATERIAL_TRUNK,
                feature_id: 0,
                uv: [x as f32 / n as f32, z as f32 / n as f32],
                ao: 1.0,
                semantic_channels: 0,
            });
        }
    }
    let mut indices = Vec::new();
    let stride = (n + 1) as u32;
    for z in 0..n as u32 {
        for x in 0..n as u32 {
            let a = z * stride + x;
            let b = a + 1;
            let c = a + stride;
            let d = c + 1;
            indices.extend_from_slice(&[a, c, b, b, c, d]);
        }
    }
    (verts, indices)
}
