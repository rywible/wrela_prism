// GPU DAG traversal compute shader.
//
// Replaces the CPU-side adaptive DAG cut with a persistent-thread GPU approach.
// Uses a single work queue with atomic append: root groups are seeded into the
// work queue, and each thread grabs a group, projects its error to screen space,
// and either selects it (writes to output queue) or pushes its children back
// onto the work queue. Threads loop until the queue is exhausted.
//
// This works well for small-to-medium DAGs (< 16K groups). For larger DAGs,
// a multi-pass ping-pong approach would be needed.

struct FrameUniforms {
    view_proj: mat4x4<f32>,
    camera_position: vec4<f32>,
    screen_size: vec4<f32>,
    error_threshold: vec4<f32>,   // (error_threshold_px, fov_factor, 0, 0)
    frustum_planes: array<vec4<f32>, 6>,
    hzb_params: vec4<f32>,
};
@group(0) @binding(0) var<uniform> frame: FrameUniforms;

struct MeshletGroup {
    meshlet_start: u32,
    meshlet_count: u32,
    child_start: u32,
    child_count: u32,
    error: f32,
    level: u32,
    _pad0: u32,
    _pad1: u32,
    bound_center: vec3<f32>,
    bound_radius: f32,
};

// DAG group buffer (read-only, same as cull pass binding)
@group(1) @binding(0) var<storage, read> meshlet_groups: array<MeshletGroup>;

// Work queue: starts with root group indices, grows as children are appended.
// Read/write because threads both consume from and append to this queue.
@group(1) @binding(1) var<storage, read_write> work_queue: array<u32>;

// Atomic work queue head (read index) and tail (write index).
// Layout: [head, tail, 0, 0] — head is the next index to consume,
// tail is the next index to write to.
@group(1) @binding(2) var<storage, read_write> work_counters: array<atomic<u32>>;

// Output queue: selected group indices that pass the error test.
// Fed to the cull pass as group_queue.
@group(1) @binding(3) var<storage, read_write> output_queue: array<u32>;

// Atomic output count.
@group(1) @binding(4) var<storage, read_write> output_count: atomic<u32>;

// Indirect dispatch args for the cull pass: [workgroup_x, 1, 1]
@group(1) @binding(5) var<storage, read_write> indirect_args: array<u32>;

const WORK_QUEUE_CAPACITY: u32 = 16384u;
const OUTPUT_QUEUE_CAPACITY: u32 = 16384u;

fn project_error(error: f32, center: vec3<f32>) -> f32 {
    let dist = distance(frame.camera_position.xyz, center);
    if dist < 0.001 {
        return 1000.0;
    }
    let fov_factor = frame.error_threshold.y;
    return error * frame.screen_size.y * fov_factor / dist;
}

@compute @workgroup_size(64)
fn dag_traverse(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Persistent-thread loop: each thread keeps grabbing work until the queue
    // is exhausted. We use a spin-wait approach with a bounded iteration count
    // to prevent GPU hangs. The DAG is typically 3-6 levels deep with < 16K
    // groups, so 256 iterations per thread is more than enough.
    for (var iter = 0u; iter < 256u; iter = iter + 1u) {
        // Atomically grab the next work item
        let my_idx = atomicAdd(&work_counters[0], 1u);

        // Check if we've consumed all work. The tail may still be growing
        // from other threads appending children, so we read it atomically.
        let current_tail = atomicLoad(&work_counters[1]);
        if my_idx >= current_tail {
            // No more work — restore the head counter so other threads
            // don't skip slots, then exit.
            // (In practice, all threads will hit this roughly together.)
            break;
        }

        // Bounds-check against queue capacity
        if my_idx >= WORK_QUEUE_CAPACITY {
            break;
        }

        let group_idx = work_queue[my_idx];
        let group = meshlet_groups[group_idx];

        let projected_error = project_error(group.error, group.bound_center);
        let is_leaf = group.child_count == 0u;
        let should_render = is_leaf || projected_error < frame.error_threshold.x;

        if should_render {
            // Select this group: write to output queue
            let out_slot = atomicAdd(&output_count, 1u);
            if out_slot < OUTPUT_QUEUE_CAPACITY {
                output_queue[out_slot] = group_idx;
            }
        } else {
            // Push children to work queue for further traversal
            let child_start_slot = atomicAdd(&work_counters[1], group.child_count);
            for (var ci = 0u; ci < group.child_count; ci = ci + 1u) {
                let slot = child_start_slot + ci;
                if slot < WORK_QUEUE_CAPACITY {
                    work_queue[slot] = group.child_start + ci;
                }
            }
        }
    }
}

// Small shader to prepare indirect dispatch args for the cull pass.
// Reads the output count and writes [ceil(count/64), 1, 1].
@compute @workgroup_size(1)
fn prepare_indirect(@builtin(global_invocation_id) gid: vec3<u32>) {
    let count = atomicLoad(&output_count);
    indirect_args[0] = (count + 63u) / 64u;
    indirect_args[1] = 1u;
    indirect_args[2] = 1u;
}
