// Mesh shader visibility buffer fill — hardware rasterization path.
// (Future: requires wgpu mesh shader support via `enable wgpu_mesh_shader;`)
//
// For now this file is not loaded; the fallback vertex-pulling pipeline
// in meshlet_hw_vis_fallback.wgsl is used instead.

enable wgpu_mesh_shader;

struct FrameUniforms {
    view_proj: mat4x4<f32>,
    camera_position: vec4<f32>,
    screen_size: vec4<f32>,
    error_threshold: vec4<f32>,
};
@group(0) @binding(0) var<uniform> frame: FrameUniforms;

struct Vertex {
    pos_x: f32, pos_y: f32, pos_z: f32,
    nor_x: f32, nor_y: f32, nor_z: f32,
    material: u32,
    feature_id: u32,
    uv_x: f32, uv_y: f32,
    ao: f32,
    semantic_channels: u32,
};

struct MeshletDescriptor {
    vertex_offset: u32,
    vertex_count: u32,
    triangle_offset: u32,
    triangle_count: u32,
};

@group(1) @binding(0) var<storage, read> vertices: array<Vertex>;
@group(1) @binding(1) var<storage, read> meshlet_vertex_indices: array<u32>;
@group(1) @binding(2) var<storage, read> meshlet_triangle_indices: array<u32>;
@group(1) @binding(3) var<storage, read> meshlet_descs: array<MeshletDescriptor>;

@group(2) @binding(0) var<storage, read> hw_dispatch_list: array<u32>;
@group(2) @binding(1) var<storage, read> hw_dispatch_count: atomic<u32>;

fn load_tri_idx(desc: MeshletDescriptor, tri: u32, vert: u32) -> u32 {
    let byte_idx = desc.triangle_offset + tri * 3u + vert;
    let word_idx = byte_idx / 4u;
    let byte_off = byte_idx % 4u;
    return (meshlet_triangle_indices[word_idx] >> (byte_off * 8u)) & 0xFFu;
}

// Task + mesh shader entry points will be defined when wgpu mesh shader
// WGSL syntax stabilizes. The fallback pipeline covers all functionality.
