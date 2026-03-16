// Compute shader software rasterizer for sub-pixel triangles.
// One workgroup per meshlet (128 threads = up to 128 triangles).
// Each thread rasterizes one triangle using edge functions + atomicMax.

struct FrameUniforms {
    view_proj: mat4x4<f32>,
    camera_position: vec4<f32>,
    screen_size: vec4<f32>,
    error_threshold: vec4<f32>,
};
@group(0) @binding(0) var<uniform> frame: FrameUniforms;

// Vertex with scalar fields to match Rust's 48-byte #[repr(C)] layout.
// WGSL vec3<f32> has 16-byte alignment in storage buffers, which would
// add padding and mismatch the Rust struct. Scalar fields avoid this.
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

// SW dispatch list
@group(2) @binding(0) var<storage, read> sw_dispatch_list: array<u32>;

// Visibility buffer (R32Uint, written via atomicMax)
// Packed as: depth_u16 << 16 | visibility_id_u16
@group(2) @binding(1) var<storage, read_write> visbuf: array<atomic<u32>>;

fn project_vertex(v: Vertex) -> vec3<f32> {
    let pos = vec3<f32>(v.pos_x, v.pos_y, v.pos_z);
    let clip = frame.view_proj * vec4<f32>(pos, 1.0);
    if clip.w <= 0.0 {
        return vec3<f32>(-1.0, -1.0, -1.0);
    }
    let ndc = clip.xyz / clip.w;
    let screen_x = (ndc.x * 0.5 + 0.5) * frame.screen_size.x;
    let screen_y = (1.0 - (ndc.y * 0.5 + 0.5)) * frame.screen_size.y;
    return vec3<f32>(screen_x, screen_y, ndc.z);
}

fn edge_function(a: vec2<f32>, b: vec2<f32>, c: vec2<f32>) -> f32 {
    return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x);
}

fn load_tri_idx(desc: MeshletDescriptor, tri: u32, vert: u32) -> u32 {
    let byte_idx = desc.triangle_offset + tri * 3u + vert;
    let word_idx = byte_idx / 4u;
    let byte_off = byte_idx % 4u;
    return (meshlet_triangle_indices[word_idx] >> (byte_off * 8u)) & 0xFFu;
}

@compute @workgroup_size(128)
fn sw_rasterize(
    @builtin(local_invocation_index) lid: u32,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    let meshlet_idx = sw_dispatch_list[wgid.x];
    let desc = meshlet_descs[meshlet_idx];

    if lid >= desc.triangle_count {
        return;
    }

    let i0 = load_tri_idx(desc, lid, 0u);
    let i1 = load_tri_idx(desc, lid, 1u);
    let i2 = load_tri_idx(desc, lid, 2u);

    let gi0 = meshlet_vertex_indices[desc.vertex_offset + i0];
    let gi1 = meshlet_vertex_indices[desc.vertex_offset + i1];
    let gi2 = meshlet_vertex_indices[desc.vertex_offset + i2];

    let s0 = project_vertex(vertices[gi0]);
    let s1 = project_vertex(vertices[gi1]);
    let s2 = project_vertex(vertices[gi2]);

    if s0.z < 0.0 || s1.z < 0.0 || s2.z < 0.0 { return; }

    let bb_min = vec2<i32>(
        i32(floor(min(min(s0.x, s1.x), s2.x))),
        i32(floor(min(min(s0.y, s1.y), s2.y))),
    );
    let bb_max = vec2<i32>(
        i32(ceil(max(max(s0.x, s1.x), s2.x))),
        i32(ceil(max(max(s0.y, s1.y), s2.y))),
    );

    let w = i32(frame.screen_size.x);
    let h = i32(frame.screen_size.y);
    let clamped_min = max(bb_min, vec2<i32>(0, 0));
    let clamped_max = min(bb_max, vec2<i32>(w - 1, h - 1));

    let area = edge_function(s0.xy, s1.xy, s2.xy);
    if abs(area) < 0.001 { return; }
    let inv_area = 1.0 / area;

    // Bias by 1 so 0 remains the "empty pixel" sentinel in the vis buffer.
    let visibility_id = ((meshlet_idx << 8u) | lid) + 1u;

    for (var y = clamped_min.y; y <= clamped_max.y; y = y + 1) {
        for (var x = clamped_min.x; x <= clamped_max.x; x = x + 1) {
            let p = vec2<f32>(f32(x) + 0.5, f32(y) + 0.5);

            let w0 = edge_function(s1.xy, s2.xy, p) * inv_area;
            let w1 = edge_function(s2.xy, s0.xy, p) * inv_area;
            let w2 = 1.0 - w0 - w1;

            let inside = select(
                w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0,
                w0 <= 0.0 && w1 <= 0.0 && w2 <= 0.0,
                area < 0.0
            );

            if inside {
                let depth = w0 * s0.z + w1 * s1.z + w2 * s2.z;
                let depth_u16 = u32(clamp((1.0 - depth) * 65535.0, 0.0, 65535.0));
                let packed = (depth_u16 << 16u) | (visibility_id & 0xFFFFu);
                let pixel_idx = u32(y) * u32(w) + u32(x);
                atomicMax(&visbuf[pixel_idx], packed);
            }
        }
    }
}
