// Vertex-pulling visibility buffer fill (fallback for when mesh shaders are unavailable).
//
// Each draw instance = one meshlet from the HW dispatch list.
// vertex_index encodes which triangle and vertex within the meshlet:
//   tri_idx = vertex_index / 3
//   vert_in_tri = vertex_index % 3
// Triangles beyond the meshlet's count output degenerate (w=0) positions.

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
    uv_x: f32, uv_y: f32,
    ao: f32,
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

fn load_tri_idx(desc: MeshletDescriptor, tri: u32, vert: u32) -> u32 {
    let byte_idx = desc.triangle_offset + tri * 3u + vert;
    let word_idx = byte_idx / 4u;
    let byte_off = byte_idx % 4u;
    return (meshlet_triangle_indices[word_idx] >> (byte_off * 8u)) & 0xFFu;
}

struct VsOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) visibility_id: u32,
};

@vertex
fn vs_main(
    @builtin(vertex_index) vid: u32,
    @builtin(instance_index) iid: u32,
) -> VsOutput {
    let meshlet_idx = hw_dispatch_list[iid];
    let desc = meshlet_descs[meshlet_idx];

    let tri_idx = vid / 3u;
    let vert_in_tri = vid % 3u;

    // Triangles beyond the meshlet's count → degenerate (clipped)
    if tri_idx >= desc.triangle_count {
        var out: VsOutput;
        out.position = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        out.visibility_id = 0u;
        return out;
    }

    let local_vi = load_tri_idx(desc, tri_idx, vert_in_tri);
    let global_vi = meshlet_vertex_indices[desc.vertex_offset + local_vi];
    let v = vertices[global_vi];

    let world_pos = vec4<f32>(v.pos_x, v.pos_y, v.pos_z, 1.0);
    let clip_pos = frame.view_proj * world_pos;

    var out: VsOutput;
    out.position = clip_pos;
    // Bias by 1 so 0 remains the "empty pixel" sentinel in the vis buffer.
    out.visibility_id = ((meshlet_idx << 8u) | tri_idx) + 1u;
    return out;
}

@fragment
fn fs_main(input: VsOutput) -> @location(0) u32 {
    return input.visibility_id;
}
