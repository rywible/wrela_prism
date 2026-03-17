// Per-meshlet frustum culling (compute shader).
//
// CPU selects groups via adaptive DAG cut; this shader iterates each group's
// meshlets, applies frustum culling, and writes survivors to the HW dispatch list.

struct FrameUniforms {
    view_proj: mat4x4<f32>,
    camera_position: vec4<f32>,
    screen_size: vec4<f32>,
    error_threshold: vec4<f32>,   // (error_threshold_px, fov_factor, 0, 0)
    frustum_planes: array<vec4<f32>, 6>,
};
@group(0) @binding(0) var<uniform> frame: FrameUniforms;

struct MeshletBounds {
    center: vec3<f32>,
    radius: f32,
    cone_axis: vec3<f32>,
    cone_cutoff: f32,
};

struct MeshletGroup {
    meshlet_start: u32,
    meshlet_count: u32,
    child_start: u32,
    child_count: u32,
    error: f32,
    level: u32,
    _pad0: u32,
    _pad1: u32,
};

struct MeshletDescriptor {
    vertex_offset: u32,
    vertex_count: u32,
    triangle_offset: u32,
    triangle_count: u32,
};

@group(1) @binding(0) var<storage, read> meshlet_bounds: array<MeshletBounds>;
@group(1) @binding(1) var<storage, read> meshlet_groups: array<MeshletGroup>;
@group(1) @binding(2) var<storage, read> meshlet_descs: array<MeshletDescriptor>;

// Output dispatch lists
@group(2) @binding(0) var<storage, read_write> hw_dispatch_list: array<u32>;
@group(2) @binding(1) var<storage, read_write> hw_dispatch_count: atomic<u32>;
@group(2) @binding(2) var<storage, read_write> sw_dispatch_list: array<u32>;
@group(2) @binding(3) var<storage, read_write> sw_dispatch_count: atomic<u32>;

// Groups to process (written by CPU or previous pass)
@group(2) @binding(4) var<storage, read> group_queue: array<u32>;
@group(2) @binding(5) var<storage, read> group_queue_count: u32;

fn project_sphere_diameter(center: vec3<f32>, radius: f32) -> f32 {
    let dist = distance(frame.camera_position.xyz, center);
    if dist <= radius {
        return max(frame.screen_size.x, frame.screen_size.y);
    }
    let fov_factor = frame.error_threshold.y;
    return radius * frame.screen_size.y * (2.0 * fov_factor)
        / sqrt(dist * dist - radius * radius);
}

fn project_error(error: f32, center: vec3<f32>) -> f32 {
    let dist = distance(frame.camera_position.xyz, center);
    if dist < 0.001 {
        return 1000.0;
    }
    let fov_factor = frame.error_threshold.y;
    return error * frame.screen_size.y * fov_factor / dist;
}

fn frustum_cull_sphere(center: vec3<f32>, radius: f32) -> bool {
    let world_center = vec4<f32>(center, 1.0);
    for (var i = 0u; i < 6u; i = i + 1u) {
        let plane = frame.frustum_planes[i];
        if dot(plane.xyz, world_center.xyz) + plane.w < -radius {
            return true;
        }
    }
    let clip = frame.view_proj * world_center;
    if clip.w <= 0.0 {
        return true;
    }
    return false;
}

@compute @workgroup_size(64)
fn meshlet_cull(@builtin(global_invocation_id) gid: vec3<u32>) {
    let queue_idx = gid.x;
    if queue_idx >= group_queue_count {
        return;
    }

    let group_idx = group_queue[queue_idx];
    let group = meshlet_groups[group_idx];

    // DAG cut decision already made by CPU — this shader only does per-meshlet culling.

    // Process each meshlet in this group
    for (var mi = 0u; mi < group.meshlet_count; mi = mi + 1u) {
        let meshlet_idx = group.meshlet_start + mi;
        let bounds = meshlet_bounds[meshlet_idx];

        // Frustum cull
        if frustum_cull_sphere(bounds.center, bounds.radius) {
            continue;
        }

        // The current frame pipeline resolves only the HW visibility buffer path.
        // Route all surviving meshlets there until the SW path is fully integrated.
        _ = project_sphere_diameter(bounds.center, bounds.radius);
        let slot = atomicAdd(&hw_dispatch_count, 1u);
        hw_dispatch_list[slot] = meshlet_idx;
    }
}
