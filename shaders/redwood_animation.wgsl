struct RedwoodAnimationUniforms {
    level_start: u32,
    level_count: u32,
    node_count: u32,
    vertex_count: u32,
    time: f32,
    dt: f32,
    _pad0: vec2<f32>,
    wind_direction: vec4<f32>,
    wind_profile: vec4<f32>,
    style_profile: vec4<f32>,
};

struct RigNode {
    rest_position: vec4<f32>,
    rest_parent_position: vec4<f32>,
    base_dir_len: vec4<f32>,
    sim_params: vec4<f32>,
    parent_index: i32,
    _pad0: vec3<u32>,
};

struct RigState {
    current_position: vec4<f32>,
    velocity: vec4<f32>,
};

struct Vertex {
    pos_x: f32, pos_y: f32, pos_z: f32,
    nor_x: f32, nor_y: f32, nor_z: f32,
    material: u32,
    feature_id: u32,
    uv_x: f32, uv_y: f32,
    ao: f32,
    semantic_channels: u32,
};

struct AnimatedRestVertex {
    base: Vertex,
    local_position: vec4<f32>,
    local_normal: vec4<f32>,
    parent_index: u32,
    node_index: u32,
    flags: u32,
    _pad0: u32,
};

@group(0) @binding(0) var<uniform> uniforms: RedwoodAnimationUniforms;
@group(0) @binding(1) var<storage, read> rig_nodes: array<RigNode>;
@group(0) @binding(2) var<storage, read_write> rig_state: array<RigState>;

@group(1) @binding(0) var<storage, read> rest_vertices: array<AnimatedRestVertex>;
@group(2) @binding(0) var<storage, read_write> out_vertices: array<Vertex>;

fn hash11(x: f32) -> f32 {
    return fract(sin(x * 91.3458 + 12.345) * 43758.5453);
}

fn sample_wind(rest_pos: vec3<f32>) -> vec3<f32> {
    let dir = normalize(vec3<f32>(uniforms.wind_direction.xy, 0.0) + vec3<f32>(0.0001, 0.0, 0.0001));
    let mean_speed = uniforms.wind_profile.x;
    let gust_strength = uniforms.wind_profile.y;
    let gust_frequency = uniforms.wind_profile.z;
    let turbulence = uniforms.wind_profile.w;
    let style_amp = uniforms.style_profile.x;
    let style_response = uniforms.style_profile.y;
    let gust_sharpness = uniforms.style_profile.z;
    let t = uniforms.time;
    let macro_wave = sin(t * gust_frequency + rest_pos.x * 0.03 + rest_pos.z * 0.05) * 0.5 + 0.5;
    let gust = pow(macro_wave, gust_sharpness) * gust_strength;
    let swirl = vec3<f32>(
        sin(t * 1.9 + rest_pos.y * 0.18 + rest_pos.z * 0.07),
        sin(t * 2.6 + rest_pos.x * 0.05 + rest_pos.z * 0.04),
        cos(t * 1.7 + rest_pos.x * 0.06 - rest_pos.y * 0.11)
    ) * turbulence;
    let flutter = vec3<f32>(
        sin(t * 7.0 + rest_pos.x * 0.22),
        cos(t * 6.1 + rest_pos.z * 0.19),
        sin(t * 8.4 + rest_pos.y * 0.35)
    ) * 0.08;
    return (dir * (mean_speed + gust) + swirl * 0.22 + flutter) * style_amp * style_response;
}

@compute @workgroup_size(64)
fn simulate_nodes(@builtin(global_invocation_id) gid: vec3<u32>) {
    let local_idx = gid.x;
    if local_idx >= uniforms.level_count {
        return;
    }
    let idx = uniforms.level_start + local_idx;
    if idx >= uniforms.node_count {
        return;
    }

    let node = rig_nodes[idx];
    let rest_pos = node.rest_position.xyz;
    let rest_parent = node.rest_parent_position.xyz;
    let base_dir = node.base_dir_len.xyz;
    let base_len = max(node.base_dir_len.w, 0.001);
    let stiffness = node.sim_params.x;
    let damping = node.sim_params.y;
    let drag = node.sim_params.z;
    let max_disp = node.sim_params.w;

    var parent_pos = rest_parent;
    if node.parent_index >= 0 {
        parent_pos = rig_state[u32(node.parent_index)].current_position.xyz;
    }

    let wind_force = select(sample_wind(rest_pos), vec3<f32>(0.0), uniforms.wind_direction.z > 0.5);
    let base_tip = parent_pos + base_dir * base_len;
    let current = rig_state[idx].current_position.xyz;
    let velocity = rig_state[idx].velocity.xyz;
    let target_pos = base_tip + wind_force * drag + vec3<f32>(0.0, -drag * 0.08, 0.0);
    let accel = (target_pos - current) * stiffness - velocity * damping;
    let new_velocity = velocity + accel * uniforms.dt;
    var proposed = current + new_velocity * uniforms.dt;
    let offset = proposed - base_tip;
    let offset_len = length(offset);
    if offset_len > max_disp {
        proposed = base_tip + offset / offset_len * max_disp;
    }
    let constrained_dir = normalize((proposed - parent_pos) + base_dir * 0.0001);
    let final_pos = parent_pos + constrained_dir * base_len;

    rig_state[idx].current_position = vec4<f32>(final_pos, 1.0);
    rig_state[idx].velocity = vec4<f32>((final_pos - current) / max(uniforms.dt, 1e-4), 0.0);
}

@compute @workgroup_size(64)
fn deform_vertices(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= uniforms.vertex_count {
        return;
    }
    let rest = rest_vertices[idx];
    var outv = rest.base;

    if rest.flags == 0u || rest.node_index == 0xFFFFFFFFu {
        out_vertices[idx] = outv;
        return;
    }

    let node_pos = rig_state[rest.node_index].current_position.xyz;
    var parent_pos = rig_nodes[rest.node_index].rest_parent_position.xyz;
    if rest.parent_index != 0xFFFFFFFFu {
        parent_pos = rig_state[rest.parent_index].current_position.xyz;
    }

    let dir = normalize((node_pos - parent_pos) + vec3<f32>(0.0001, 0.0, 0.0001));
    let arbitrary = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(dir.y) > 0.92);
    var right = normalize(cross(dir, arbitrary));
    if length(right) < 0.001 {
        right = vec3<f32>(1.0, 0.0, 0.0);
    }
    let up = normalize(cross(right, dir));

    let local_pos = rest.local_position.xyz;
    let world_pos = parent_pos + right * local_pos.x + up * local_pos.y + dir * local_pos.z;
    let local_normal = rest.local_normal.xyz;
    let world_normal = normalize(right * local_normal.x + up * local_normal.y + dir * local_normal.z);

    outv.pos_x = world_pos.x;
    outv.pos_y = world_pos.y;
    outv.pos_z = world_pos.z;
    outv.nor_x = world_normal.x;
    outv.nor_y = world_normal.y;
    outv.nor_z = world_normal.z;
    out_vertices[idx] = outv;
}
