// Cloud temporal reprojection with 3x3 neighborhood clamping.
//
// Key changes from previous version:
// - Always reproject from cloud hit point (no opacity-weighted mix with far plane)
// - Profile-driven temporal blend weight from cloud_profile2.w
// - Simpler variance rejection via single color_range check
// - Bilinear history sampling via textureSampleLevel

struct CloudUniforms {
    inv_view_proj: mat4x4<f32>,
    prev_view_proj: mat4x4<f32>,
    camera_position: vec4<f32>,
    sun_direction: vec4<f32>,
    sun_color: vec4<f32>,
    sky_ambient: vec4<f32>,
    cloud_params: vec4<f32>,    // (coverage, first_frame_flag, time, frame_index)
    screen_params: vec4<f32>,   // (quarter_w, quarter_h, 1/quarter_w, 1/quarter_h)
    cloud_profile: vec4<f32>,   // (density_scale, cloud_base, cloud_top, detail_erosion)
    cloud_profile2: vec4<f32>,  // (wind_speed, march_steps, light_steps, temporal_blend)
};

@group(0) @binding(0) var<uniform> u: CloudUniforms;
@group(0) @binding(1) var raw_tex: texture_2d<f32>;       // current frame raw march
@group(0) @binding(2) var history_tex: texture_2d<f32>;    // previous frame blended
@group(0) @binding(3) var output_tex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var cloud_depth_tex: texture_2d<f32>; // cloud hit distance
@group(0) @binding(5) var history_sampler: sampler;         // bilinear filtering

@compute @workgroup_size(8, 8)
fn cloud_temporal(@builtin(global_invocation_id) gid: vec3<u32>) {
    let quarter_size = vec2<u32>(u32(u.screen_params.x), u32(u.screen_params.y));
    if gid.x >= quarter_size.x || gid.y >= quarter_size.y {
        return;
    }

    let current = textureLoad(raw_tex, vec2<i32>(gid.xy), 0);

    // Profile-driven temporal blend weight
    let temporal_blend = u.cloud_profile2.w;

    // Very transparent pixels: write current directly, no history needed
    let opacity = 1.0 - current.a;
    if opacity < 0.002 {
        textureStore(output_tex, gid.xy, current);
        return;
    }

    // 3x3 neighborhood min/max from current frame
    var min_val = current;
    var max_val = current;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let coord = clamp(
                vec2<i32>(gid.xy) + vec2<i32>(dx, dy),
                vec2<i32>(0),
                vec2<i32>(quarter_size) - vec2<i32>(1)
            );
            let s = textureLoad(raw_tex, coord, 0);
            min_val = min(min_val, s);
            max_val = max(max_val, s);
        }
    }

    // Reproject from cloud hit point — always use cloud distance, not far plane
    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(quarter_size);
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0);
    let near_pt = u.inv_view_proj * vec4<f32>(ndc, 0.0, 1.0);
    let far_pt = u.inv_view_proj * vec4<f32>(ndc, 1.0, 1.0);
    let ray_origin = near_pt.xyz / near_pt.w;
    let ray_dir = normalize(far_pt.xyz / far_pt.w - ray_origin);

    let cloud_dist = textureLoad(cloud_depth_tex, vec2<i32>(gid.xy), 0).r;
    // Cloud march stores distance in atmosphere-space kilometers; convert to scene meters
    let cloud_point = ray_origin + ray_dir * max(cloud_dist * 1000.0, 1.0);

    let prev_clip = u.prev_view_proj * vec4<f32>(cloud_point, 1.0);
    let prev_ndc = prev_clip.xy / prev_clip.w;
    let prev_uv = vec2<f32>(prev_ndc.x * 0.5 + 0.5, 0.5 - prev_ndc.y * 0.5);

    var result = current;

    let skip_temporal = u.cloud_params.y > 0.5;

    if !skip_temporal && all(prev_uv >= vec2<f32>(0.0)) && all(prev_uv <= vec2<f32>(1.0)) {
        // Bilinear history sampling for smoother reprojection
        let history = textureSampleLevel(history_tex, history_sampler, prev_uv, 0.0);

        // Clamp history to dynamic neighborhood bounds
        let clamped = clamp(history, min_val, max_val);

        // Simple variance rejection: reduce blend when neighborhood is changing fast
        let color_range = max(
            max(max_val.r - min_val.r, max_val.g - min_val.g),
            max(max_val.b - min_val.b, abs(max_val.a - min_val.a)),
        );
        var history_weight = temporal_blend;
        history_weight *= 1.0 - smoothstep(0.15, 0.55, color_range);

        result = mix(current, clamped, clamp(history_weight, 0.0, 0.95));
    }

    textureStore(output_tex, gid.xy, result);
}
