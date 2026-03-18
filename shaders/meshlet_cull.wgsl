// Per-meshlet frustum + HZB occlusion culling (compute shader).
//
// Two-phase occlusion culling to eliminate 1-frame popping:
//   Phase 1 (meshlet_cull): Cull with prev-frame HZB. Survivors → hw_dispatch_list.
//          HZB-rejected meshlets → phase2_reject_list for re-testing.
//   Phase 2 (meshlet_cull_phase2): Re-test rejects against fresh mid-frame HZB.
//          Survivors → hw_dispatch_list for a second raster pass.

struct FrameUniforms {
    view_proj: mat4x4<f32>,
    camera_position: vec4<f32>,
    screen_size: vec4<f32>,
    error_threshold: vec4<f32>,   // (error_threshold_px, fov_factor, 0, 0)
    frustum_planes: array<vec4<f32>, 6>,
    hzb_params: vec4<f32>,        // (enable, mip_count, hzb_width, hzb_height)
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

// HZB texture (previous frame for phase 1, fresh mid-frame for phase 2)
@group(2) @binding(6) var hzb_tex: texture_2d<f32>;

// Phase 2 reject list — meshlets that failed HZB in phase 1, to be re-tested
@group(2) @binding(7) var<storage, read_write> phase2_reject_list: array<u32>;
@group(2) @binding(8) var<storage, read_write> phase2_reject_count: atomic<u32>;

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

fn hzb_occlusion_cull(center: vec3<f32>, radius: f32) -> bool {
    if frame.hzb_params.x < 0.5 { return false; } // disabled

    let expanded = radius * 1.1; // 10% conservative for 1-frame lag
    let clip = frame.view_proj * vec4(center, 1.0);
    if clip.w <= 0.0 { return false; } // behind camera
    let ndc = clip.xyz / clip.w;

    // Sphere's nearest depth (reversed-Z: closer = larger value)
    let to_cam = frame.camera_position.xyz - center;
    let to_cam_len = length(to_cam);
    let nearest_pt = select(
        center + (to_cam / to_cam_len) * expanded,
        center,
        to_cam_len < 0.001,
    );
    let nearest_clip = frame.view_proj * vec4(nearest_pt, 1.0);
    if nearest_clip.w <= 0.0 { return false; }
    let sphere_z = nearest_clip.z / nearest_clip.w;

    // Project to screen rect, select mip
    let dist = max(to_cam_len - expanded, 0.01);
    let proj_radius = expanded * frame.error_threshold.y * 2.0 / dist;
    let rect_px = proj_radius * frame.hzb_params.zw * 0.5;
    let mip = clamp(
        u32(ceil(log2(max(rect_px.x, max(rect_px.y, 1.0))))),
        0u,
        u32(frame.hzb_params.y) - 1u,
    );

    // Sample HZB (min depth = farthest in reversed-Z)
    let uv = vec2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    let mip_size = vec2<u32>(
        max(u32(frame.hzb_params.z) >> mip, 1u),
        max(u32(frame.hzb_params.w) >> mip, 1u),
    );
    let texel = clamp(
        vec2<i32>(uv * vec2<f32>(mip_size)),
        vec2(0),
        vec2<i32>(mip_size) - 1,
    );
    let hzb_depth = textureLoad(hzb_tex, texel, i32(mip)).r;

    // Occluded if sphere nearest is farther than HZB farthest
    return sphere_z < hzb_depth;
}

// Phase 1: Cull with previous frame's HZB.
// Survivors go to hw_dispatch_list, HZB-occluded meshlets go to phase2_reject_list.
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

        // Frustum cull — definitely invisible, discard permanently
        if frustum_cull_sphere(bounds.center, bounds.radius) {
            continue;
        }

        // HZB occlusion cull (uses previous frame's HZB)
        if hzb_occlusion_cull(bounds.center, bounds.radius) {
            // Not definitely invisible — queue for phase 2 re-test with fresh HZB.
            // Bounds-check before writing: the reject list holds at most 65536 entries.
            // If the list is full we silently drop the meshlet (it will be absent for
            // one frame rather than causing an out-of-bounds GPU write).
            let reject_slot = atomicAdd(&phase2_reject_count, 1u);
            if reject_slot < 65536u {
                phase2_reject_list[reject_slot] = meshlet_idx;
            }
            continue;
        }

        // Route all surviving meshlets to the HW visibility buffer path.
        _ = project_sphere_diameter(bounds.center, bounds.radius);
        let slot = atomicAdd(&hw_dispatch_count, 1u);
        hw_dispatch_list[slot] = meshlet_idx;
    }
}

// Phase 2: Re-test HZB-rejected meshlets against the fresh mid-frame HZB.
// Survivors go to hw_dispatch_list for a second raster pass.
// Note: uses the SAME bind group layout — phase2_reject_list is now the input
// (read via group_queue bindings are unused), and hzb_tex points to the fresh HZB.
// We dispatch one thread per rejected meshlet.
@compute @workgroup_size(64)
fn meshlet_cull_phase2(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    // group_queue_count is reused to hold the phase2 reject count for this dispatch
    if idx >= group_queue_count {
        return;
    }

    // Read meshlet index from the reject list (bound at group_queue binding)
    let meshlet_idx = group_queue[idx];
    let bounds = meshlet_bounds[meshlet_idx];

    // Re-test against fresh HZB (built after phase 1 raster)
    if hzb_occlusion_cull(bounds.center, bounds.radius) {
        return; // Still occluded — discard
    }

    // Survivor — add to HW dispatch list for phase 2 raster
    let slot = atomicAdd(&hw_dispatch_count, 1u);
    hw_dispatch_list[slot] = meshlet_idx;
}
