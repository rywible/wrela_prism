use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use super::bounds::MeshletBounds;
use super::simplify::MeshletDag;

/// GPU-side meshlet descriptor, stored in a storage buffer.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuMeshletDescriptor {
    /// Offset into the global vertex index buffer.
    pub vertex_offset: u32,
    /// Number of vertex indices.
    pub vertex_count: u32,
    /// Offset into the global triangle buffer (byte offset).
    pub triangle_offset: u32,
    /// Number of triangles.
    pub triangle_count: u32,
}

/// GPU-side meshlet group descriptor for the LOD DAG.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuMeshletGroup {
    /// Start index in the meshlet descriptor array.
    pub meshlet_start: u32,
    /// Number of meshlets in this group.
    pub meshlet_count: u32,
    /// Start index of child groups (finer level).
    pub child_start: u32,
    /// Number of child groups.
    pub child_count: u32,
    /// Simplification error (world-space units).
    pub error: f32,
    /// LOD level.
    pub level: u32,
    pub _pad: [u32; 2],
}

/// All GPU buffers needed to render the meshlet DAG.
pub struct GpuMeshletBuffers {
    /// Storage buffer of all vertices (all LOD levels).
    pub vertex_buffer: wgpu::Buffer,
    pub vertex_count: u32,
    /// Storage buffer of meshlet vertex indices (u32 per entry).
    pub meshlet_vertex_buffer: wgpu::Buffer,
    /// Storage buffer of meshlet triangle indices (u8s, read as u32s).
    pub meshlet_triangle_buffer: wgpu::Buffer,
    /// Storage buffer of GpuMeshletDescriptor.
    pub meshlet_desc_buffer: wgpu::Buffer,
    pub meshlet_count: u32,
    /// Storage buffer of MeshletBounds (per meshlet).
    pub meshlet_bounds_buffer: wgpu::Buffer,
    /// Storage buffer of GpuMeshletGroup (DAG nodes).
    pub group_buffer: wgpu::Buffer,
    pub group_count: u32,
}

impl GpuMeshletBuffers {
    pub fn from_dag(device: &wgpu::Device, dag: &MeshletDag) -> Self {
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("meshlet-vertices"),
            contents: bytemuck::cast_slice(&dag.vertices),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let meshlet_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("meshlet-vertex-indices"),
            contents: bytemuck::cast_slice(&dag.meshlet_vertices),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Pad triangle buffer to 4-byte alignment for u32 reads in shaders
        let mut tri_data = dag.meshlet_triangles.clone();
        while tri_data.len() % 4 != 0 {
            tri_data.push(0);
        }
        let meshlet_triangle_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("meshlet-triangle-indices"),
                contents: &tri_data,
                usage: wgpu::BufferUsages::STORAGE,
            });

        // Build meshlet descriptors
        let descs: Vec<GpuMeshletDescriptor> = dag
            .meshlets
            .iter()
            .map(|m| GpuMeshletDescriptor {
                vertex_offset: m.vertex_offset,
                vertex_count: m.vertex_count,
                triangle_offset: m.triangle_offset,
                triangle_count: m.triangle_count,
            })
            .collect();

        let meshlet_desc_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("meshlet-descriptors"),
            contents: bytemuck::cast_slice(&descs),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Meshlet bounds
        let bounds: Vec<MeshletBounds> = dag.meshlets.iter().map(|m| m.bounds).collect();
        let meshlet_bounds_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("meshlet-bounds"),
            contents: bytemuck::cast_slice(&bounds),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Group descriptors
        let groups: Vec<GpuMeshletGroup> = dag
            .groups
            .iter()
            .map(|g| GpuMeshletGroup {
                meshlet_start: g.meshlet_start,
                meshlet_count: g.meshlet_count,
                child_start: g.child_start,
                child_count: g.child_count,
                error: g.error,
                level: g.level,
                _pad: [0; 2],
            })
            .collect();

        let group_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("meshlet-groups"),
            contents: bytemuck::cast_slice(&groups),
            usage: wgpu::BufferUsages::STORAGE,
        });

        Self {
            vertex_count: dag.vertices.len() as u32,
            vertex_buffer,
            meshlet_vertex_buffer,
            meshlet_triangle_buffer,
            meshlet_desc_buffer,
            meshlet_count: dag.meshlets.len() as u32,
            meshlet_bounds_buffer,
            group_buffer,
            group_count: dag.groups.len() as u32,
        }
    }
}
