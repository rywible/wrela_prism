use wgpu::util::DeviceExt;

use crate::scene::Vertex;

pub struct GpuMesh {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub index_count: u32,
}

pub fn upload_mesh(
    device: &wgpu::Device,
    vertices: &[Vertex],
    indices: &[u32],
    label: &str,
) -> GpuMesh {
    upload_mesh_inner(device, vertices, indices, label, false)
}

/// Like `upload_mesh` but the vertex buffer uses `VERTEX | COPY_DST` so it can be
/// re-uploaded each frame for animated geometry.
pub fn upload_mesh_animated(
    device: &wgpu::Device,
    vertices: &[Vertex],
    indices: &[u32],
    label: &str,
) -> GpuMesh {
    upload_mesh_inner(device, vertices, indices, label, true)
}

fn upload_mesh_inner(
    device: &wgpu::Device,
    vertices: &[Vertex],
    indices: &[u32],
    label: &str,
    animated: bool,
) -> GpuMesh {
    let mut usage = wgpu::BufferUsages::VERTEX;
    if animated {
        usage |= wgpu::BufferUsages::COPY_DST;
    }
    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(&format!("prism-{label}-vertices")),
        contents: bytemuck::cast_slice(vertices),
        usage,
    });
    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(&format!("prism-{label}-indices")),
        contents: bytemuck::cast_slice(indices),
        usage: wgpu::BufferUsages::INDEX,
    });
    GpuMesh {
        vertex_buffer,
        index_buffer,
        index_count: indices.len() as u32,
    }
}
