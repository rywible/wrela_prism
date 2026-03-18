// TAA motion vector generation (compute shader).
//
// Reconstructs world position from depth + inv_view_proj (unjittered),
// projects with prev_view_proj, outputs 2D screen-space velocity in UV space.
//
// Sky pixels use a far-plane direction reprojection instead of zero motion,
// so the sky correctly reprojects during camera rotation.
//
// Reversed-Z: depth values [1,0] — real geometry has depth > 0, sky ≈ 0.

struct TaaUniforms {
    inv_view_proj: mat4x4<f32>,
    prev_view_proj: mat4x4<f32>,
    screen_size: vec4<f32>,    // w, h, 1/w, 1/h
    jitter: vec4<f32>,         // jx, jy, prev_jx, prev_jy (pixels)
    params: vec4<f32>,         // blend_weight, first_frame_flag, 0, 0
};

@group(0) @binding(0) var<uniform> taa: TaaUniforms;
@group(0) @binding(1) var depth_tex: texture_depth_2d;
@group(0) @binding(2) var motion_out: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn taa_motion(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = vec2<u32>(u32(taa.screen_size.x), u32(taa.screen_size.y));
    if gid.x >= size.x || gid.y >= size.y { return; }

    let pixel = vec2<i32>(gid.xy);
    let depth = textureLoad(depth_tex, pixel, 0);

    // Current pixel UV (center of pixel)
    let uv = (vec2<f32>(gid.xy) + 0.5) * taa.screen_size.zw;
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);

    // Sky pixels: reproject using a far-plane direction (handles camera rotation)
    if depth <= 0.0001 {
        // Reconstruct view direction at this pixel using a small synthetic depth
        let sky_clip = vec4<f32>(ndc, 0.001, 1.0);
        let sky_world_h = taa.inv_view_proj * sky_clip;
        let sky_dir = normalize(sky_world_h.xyz / sky_world_h.w);
        // Project this direction far away into the previous frame
        let sky_prev_clip = taa.prev_view_proj * vec4<f32>(sky_dir * 10000.0, 1.0);
        if sky_prev_clip.w <= 0.0 {
            textureStore(motion_out, pixel, vec4<f32>(0.0, 0.0, 0.0, 0.0));
            return;
        }
        let sky_prev_ndc = sky_prev_clip.xy / sky_prev_clip.w;
        let sky_prev_uv = vec2<f32>(sky_prev_ndc.x * 0.5 + 0.5, 0.5 - sky_prev_ndc.y * 0.5);
        let sky_motion = uv - sky_prev_uv;
        textureStore(motion_out, pixel, vec4<f32>(sky_motion, 0.0, 0.0));
        return;
    }

    // Reconstruct world position from unjittered inv_view_proj
    let clip = vec4<f32>(ndc, depth, 1.0);
    let world_h = taa.inv_view_proj * clip;
    let world = world_h.xyz / world_h.w;

    // Project to previous frame using unjittered prev_view_proj
    let prev_clip = taa.prev_view_proj * vec4<f32>(world, 1.0);
    if prev_clip.w <= 0.0 {
        textureStore(motion_out, pixel, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }
    let prev_ndc = prev_clip.xy / prev_clip.w;
    let prev_uv = vec2<f32>(prev_ndc.x * 0.5 + 0.5, 0.5 - prev_ndc.y * 0.5);

    // Compute motion in UV space, subtract jitter difference to prevent flicker
    var motion = uv - prev_uv;
    let jitter_diff = vec2<f32>(
        (taa.jitter.x - taa.jitter.z) * taa.screen_size.z,
        -(taa.jitter.y - taa.jitter.w) * taa.screen_size.w,
    );
    motion -= jitter_diff;

    textureStore(motion_out, pixel, vec4<f32>(motion, 0.0, 0.0));
}
