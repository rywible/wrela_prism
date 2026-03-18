// Froxel volumetric fog — injection pass (compute).
//
// For each froxel voxel in a 160x88x128 3D grid:
// 1. Compute world position via inverse exponential depth
// 2. Evaluate density with height falloff
// 3. Sample shadow map for sun visibility
// 4. Compute in-scattering (Henyey-Greenstein phase + ambient)
// 5. Write to froxel volume: rgb = scattering, a = extinction

struct FogUniforms {
    inv_view_proj: mat4x4<f32>,
    prev_inv_view_proj: mat4x4<f32>,
    camera_position: vec4<f32>,
    sun_direction: vec4<f32>,
    sun_color: vec4<f32>,          // rgb = color, a = strength
    fog_params: vec4<f32>,         // density, height_falloff, anisotropy, temporal_weight
    fog_albedo: vec4<f32>,         // rgb = albedo, a = near_plane
    grid_params: vec4<f32>,        // grid_w, grid_h, grid_d, far_plane
    ambient_color: vec4<f32>,      // rgb = ambient, a = frame_index
    light_vp: mat4x4<f32>,        // cascade 0 light VP
    light_vp_1: mat4x4<f32>,      // cascade 1
    light_vp_2: mat4x4<f32>,      // cascade 2
    light_vp_3: mat4x4<f32>,      // cascade 3
    cascade_splits: vec4<f32>,     // view-space split distances
    view_proj: mat4x4<f32>,       // current frame view-projection
};

@group(0) @binding(0) var<uniform> u: FogUniforms;
@group(0) @binding(1) var scatter_volume: texture_storage_3d<rgba16float, write>;
@group(0) @binding(2) var shadow_map: texture_depth_2d_array;
@group(0) @binding(3) var shadow_sampler: sampler_comparison;
@group(0) @binding(4) var prev_scatter_volume: texture_3d<f32>;
@group(0) @binding(5) var prev_sampler: sampler;

const PI: f32 = 3.14159265359;

fn slice_to_depth(slice: f32) -> f32 {
    let near = u.fog_albedo.w;
    let far = u.grid_params.w;
    let num_slices = u.grid_params.z;
    return near * pow(far / near, slice / num_slices);
}

fn depth_to_slice(depth: f32) -> f32 {
    let near = u.fog_albedo.w;
    let far = u.grid_params.w;
    let num_slices = u.grid_params.z;
    return num_slices * log(depth / near) / log(far / near);
}

fn froxel_to_world(coord: vec3<f32>) -> vec3<f32> {
    let grid_w = u.grid_params.x;
    let grid_h = u.grid_params.y;

    // Screen UV from froxel XY
    let uv = (coord.xy + 0.5) / vec2<f32>(grid_w, grid_h);
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0);

    // Exponential depth from slice
    let depth = slice_to_depth(coord.z + 0.5);

    // Reconstruct view ray via inverse VP.
    // Use two points with non-degenerate w (z=1 near, z=0.5 mid) to avoid
    // division by zero at z=0 (infinity in reversed-Z).
    let near_clip = u.inv_view_proj * vec4<f32>(ndc, 1.0, 1.0);
    let mid_clip = u.inv_view_proj * vec4<f32>(ndc, 0.5, 1.0);
    let near_world = near_clip.xyz / near_clip.w;
    let mid_world = mid_clip.xyz / mid_clip.w;
    let ray_dir = normalize(mid_world - near_world);
    let cam_pos = u.camera_position.xyz;

    // Place at the given depth along the ray
    return cam_pos + ray_dir * depth;
}

fn henyey_greenstein(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = 1.0 + g2 - 2.0 * g * cos_theta;
    return (1.0 - g2) / (4.0 * PI * pow(denom, 1.5));
}

fn sample_shadow(world_pos: vec3<f32>) -> f32 {
    // Determine cascade by view-space depth
    let view_pos = u.view_proj * vec4<f32>(world_pos, 1.0);
    let view_depth = abs(view_pos.w); // linear depth

    var cascade = 0u;
    if view_depth > u.cascade_splits.x {
        cascade = 1u;
    }
    if view_depth > u.cascade_splits.y {
        cascade = 2u;
    }
    if view_depth > u.cascade_splits.z {
        cascade = 3u;
    }

    var light_vp: mat4x4<f32>;
    switch cascade {
        case 0u: { light_vp = u.light_vp; }
        case 1u: { light_vp = u.light_vp_1; }
        case 2u: { light_vp = u.light_vp_2; }
        default: { light_vp = u.light_vp_3; }
    }

    let light_clip = light_vp * vec4<f32>(world_pos, 1.0);
    let light_ndc = light_clip.xyz / light_clip.w;
    let shadow_uv = light_ndc.xy * 0.5 + 0.5;
    let shadow_depth = light_ndc.z;

    // Out-of-bounds: assume lit
    if shadow_uv.x < 0.0 || shadow_uv.x > 1.0 || shadow_uv.y < 0.0 || shadow_uv.y > 1.0 {
        return 1.0;
    }

    // Flip Y for shadow map sampling
    let sample_uv = vec2<f32>(shadow_uv.x, 1.0 - shadow_uv.y);
    return textureSampleCompareLevel(shadow_map, shadow_sampler, sample_uv, cascade, shadow_depth);
}

@compute @workgroup_size(8, 8, 1)
fn fog_inject(@builtin(global_invocation_id) gid: vec3<u32>) {
    let grid_w = u32(u.grid_params.x);
    let grid_h = u32(u.grid_params.y);
    let grid_d = u32(u.grid_params.z);

    if gid.x >= grid_w || gid.y >= grid_h || gid.z >= grid_d {
        return;
    }

    let coord = vec3<f32>(f32(gid.x), f32(gid.y), f32(gid.z));
    let world_pos = froxel_to_world(coord);

    // Density: exponential height falloff
    let base_density = u.fog_params.x;
    let height_falloff = u.fog_params.y;
    let density = base_density * exp(-max(world_pos.y, 0.0) * height_falloff);

    // Extinction coefficient
    let extinction = density;

    // In-scattering
    let albedo = u.fog_albedo.rgb;
    let anisotropy = u.fog_params.z;
    let sun_dir = normalize(u.sun_direction.xyz);
    let to_frag = world_pos - u.camera_position.xyz;
    let frag_dist = length(to_frag);
    let view_dir = select(vec3<f32>(0.0, 0.0, 1.0), to_frag / frag_dist, frag_dist > 0.001);
    let cos_theta = dot(view_dir, sun_dir);

    // Phase function
    let phase = henyey_greenstein(cos_theta, anisotropy);

    // Shadow map visibility at froxel center
    let shadow = sample_shadow(world_pos);

    // Sun in-scattering
    let sun_color = u.sun_color.rgb * u.sun_color.a;
    let sun_scatter = phase * sun_color * shadow * density * albedo;

    // Ambient in-scattering (omnidirectional)
    let ambient = u.ambient_color.rgb * density * albedo;

    let total_scatter = sun_scatter + ambient;

    // Temporal reprojection: reproject to previous frame's froxel space
    let temporal_weight = u.fog_params.w;
    let frame_index = u32(u.ambient_color.w);

    var result = vec4<f32>(total_scatter, extinction);

    if temporal_weight > 0.001 && frame_index > 0u {
        // Reproject world position to previous frame's froxel coords
        let prev_clip = u.prev_inv_view_proj * vec4<f32>(0.0, 0.0, 0.0, 1.0);
        // Actually reproject: compute previous frame screen UV
        let prev_vp_clip = u.view_proj * vec4<f32>(world_pos, 1.0);
        // For temporal stability, we blend with the previous frame's injection
        // using the same froxel coordinates (assuming camera didn't move much)
        let prev_uv = (coord + 0.5) / vec3<f32>(f32(grid_w), f32(grid_h), f32(grid_d));
        let prev_sample = textureSampleLevel(prev_scatter_volume, prev_sampler, prev_uv, 0.0);

        // Blend current with previous
        result = mix(result, prev_sample, temporal_weight);
    }

    textureStore(scatter_volume, gid, result);
}
