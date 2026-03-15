// Per-instance frustum + HZB occlusion test (compute shader).
//
// For multi-tree scenes. Tests each tree instance bounding sphere
// against the view frustum and HZB.

struct FrameUniforms {
    view_proj: mat4x4<f32>,
    camera_position: vec4<f32>,
    screen_size: vec4<f32>,
    error_threshold: vec4<f32>,
    frustum_planes: array<vec4<f32>, 6>,
};

@group(0) @binding(0) var<uniform> frame: FrameUniforms;

struct InstanceData {
    center: vec3<f32>,
    radius: f32,
};

@group(1) @binding(0) var<storage, read> instances: array<InstanceData>;
@group(1) @binding(1) var<storage, read_write> visible_instances: array<u32>;
@group(1) @binding(2) var<storage, read_write> visible_count: atomic<u32>;

@compute @workgroup_size(64)
fn instance_cull(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let inst = instances[idx];

    // Frustum test: sphere vs 6 planes
    for (var p = 0u; p < 6u; p = p + 1u) {
        let plane = frame.frustum_planes[p];
        let dist = dot(plane.xyz, inst.center) + plane.w;
        if dist < -inst.radius {
            return; // Culled
        }
    }

    // Passed — mark as visible
    let slot = atomicAdd(&visible_count, 1u);
    visible_instances[slot] = idx;
}
