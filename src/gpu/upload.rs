use wgpu::util::DeviceExt;

use crate::scene::Vertex;

pub struct GpuMesh {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub index_count: u32,
    /// Active index count for LOD control. Defaults to `index_count`.
    /// Draw calls should use this instead of `index_count`.
    pub active_index_count: u32,
    pub indirect_buffer: Option<wgpu::Buffer>,
}

pub fn upload_mesh(
    device: &wgpu::Device,
    vertices: &[Vertex],
    indices: &[u32],
    label: &str,
) -> GpuMesh {
    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(&format!("prism-{label}-vertices")),
        contents: bytemuck::cast_slice(vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(&format!("prism-{label}-indices")),
        contents: bytemuck::cast_slice(indices),
        usage: wgpu::BufferUsages::INDEX,
    });
    let index_count = indices.len() as u32;
    GpuMesh {
        vertex_buffer,
        index_buffer,
        index_count,
        active_index_count: index_count,
        indirect_buffer: None,
    }
}
