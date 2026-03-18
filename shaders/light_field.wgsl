// Light field compute shaders for biological growth simulation.
//
// Three entry points:
//   voxelize_density — sum foliage density from capsule segments into 3D volume
//   propagate_light  — Beer-Lambert top-down light propagation per XZ column
//   query_light      — sample light at arbitrary world positions

struct LightFieldConfig {
    volume_min: vec3<f32>,
    extinction: f32,
    volume_max: vec3<f32>,
    resolution: u32,
    voxel_size: vec3<f32>,
    capsule_count: u32,
    query_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

struct Capsule {
    a: vec3<f32>,
    radius_a: f32,
    b: vec3<f32>,
    radius_b: f32,
    leaf_density: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

// ──────────────────────── Voxelize Density ────────────────────────

@group(0) @binding(0) var<uniform> density_config: LightFieldConfig;
@group(0) @binding(1) var<storage, read> capsules: array<Capsule>;
@group(0) @binding(2) var density_tex: texture_storage_3d<r32float, write>;

/// Distance from point p to the closest point on capsule segment ab.
fn capsule_distance(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, ra: f32, rb: f32) -> f32 {
    let ab = b - a;
    let len_sq = dot(ab, ab);
    if len_sq < 0.0001 {
        return length(p - a) - ra;
    }
    let t = clamp(dot(p - a, ab) / len_sq, 0.0, 1.0);
    let closest = a + ab * t;
    let r = mix(ra, rb, t);
    return length(p - closest) - r;
}

@compute @workgroup_size(4, 4, 4)
fn voxelize_density(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = density_config.resolution;
    if gid.x >= res || gid.y >= res || gid.z >= res {
        return;
    }

    // Voxel center in world space.
    let uvw = (vec3<f32>(gid) + 0.5) / f32(res);
    let world_pos = density_config.volume_min + uvw * (density_config.volume_max - density_config.volume_min);

    var total_density: f32 = 0.0;

    for (var i: u32 = 0u; i < density_config.capsule_count; i++) {
        let cap = capsules[i];
        let d = capsule_distance(world_pos, cap.a, cap.b, cap.radius_a, cap.radius_b);
        // Smooth falloff: density contribution within and near the capsule.
        if d < cap.radius_a * 2.0 {
            let contrib = max(0.0, 1.0 - d / (cap.radius_a * 2.0));
            total_density += cap.leaf_density * contrib * contrib;
        }
    }

    textureStore(density_tex, gid, vec4<f32>(total_density, 0.0, 0.0, 0.0));
}

// ──────────────────────── Propagate Light ────────────────────────

@group(0) @binding(0) var<uniform> prop_config: LightFieldConfig;
@group(0) @binding(1) var density_read: texture_3d<f32>;
@group(0) @binding(2) var light_tex: texture_storage_3d<r32float, write>;

@compute @workgroup_size(8, 8)
fn propagate_light(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = prop_config.resolution;
    if gid.x >= res || gid.y >= res {
        return;
    }

    let voxel_height = prop_config.voxel_size.y;

    // March top to bottom along Y for this XZ column.
    var light: f32 = 1.0;
    for (var y: i32 = i32(res) - 1; y >= 0; y--) {
        let coord = vec3<u32>(gid.x, u32(y), gid.y); // Note: gid.y maps to Z
        let density = textureLoad(density_read, coord, 0).r;

        // Beer-Lambert: light attenuates through density.
        light = light * exp(-density * prop_config.extinction * voxel_height);

        textureStore(light_tex, coord, vec4<f32>(light, 0.0, 0.0, 0.0));
    }
}

// ──────────────────────── Query Light ────────────────────────

@group(0) @binding(0) var<uniform> query_config: LightFieldConfig;
@group(0) @binding(1) var light_read: texture_3d<f32>;
@group(0) @binding(2) var<storage, read> query_points: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> query_results: array<f32>;

@compute @workgroup_size(64)
fn query_light(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= query_config.query_count {
        return;
    }

    let world_pos = query_points[idx].xyz;

    // Convert to volume UV coordinates.
    let vol_range = query_config.volume_max - query_config.volume_min;
    let uvw = (world_pos - query_config.volume_min) / vol_range;

    // Check bounds.
    if any(uvw < vec3<f32>(0.0)) || any(uvw > vec3<f32>(1.0)) {
        query_results[idx] = 1.0; // Outside volume = full light.
        return;
    }

    // Nearest-voxel lookup (sufficient at 128^3 resolution).
    let res = f32(query_config.resolution);
    let voxel_f = uvw * res - 0.5;
    let v0 = vec3<u32>(clamp(vec3<i32>(voxel_f), vec3<i32>(0), vec3<i32>(i32(query_config.resolution) - 1)));

    let light = textureLoad(light_read, v0, 0).r;
    query_results[idx] = clamp(light, 0.0, 1.0);
}
