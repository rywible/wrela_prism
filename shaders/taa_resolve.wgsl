// TAA temporal resolve (compute shader).
//
// Reads current HDR color (post-bloom), motion vectors, and history.
// Applies variance clipping in YCoCg space (mean ± gamma * stddev).
// Uses Reinhard luminance weighting to prevent HDR fireflies.
// Includes inline sharpening (3x3 unsharp mask) to counteract temporal blur.
// Disocclusion: motion > 48px or out-of-bounds → current only.
// First frame: output current only (flag in params.y).

struct TaaUniforms {
    inv_view_proj: mat4x4<f32>,
    prev_view_proj: mat4x4<f32>,
    screen_size: vec4<f32>,    // w, h, 1/w, 1/h
    jitter: vec4<f32>,         // jx, jy, prev_jx, prev_jy (pixels)
    params: vec4<f32>,         // blend_weight, first_frame_flag, 0, 0
};

@group(0) @binding(0) var<uniform> taa: TaaUniforms;
@group(0) @binding(1) var scene_color: texture_2d<f32>;
@group(0) @binding(2) var motion_tex: texture_2d<f32>;
@group(0) @binding(3) var history_tex: texture_2d<f32>;
@group(0) @binding(4) var history_samp: sampler;
@group(0) @binding(5) var output_tex: texture_storage_2d<rgba16float, write>;

fn rgb_to_ycocg(rgb: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
         0.25 * rgb.r + 0.5 * rgb.g + 0.25 * rgb.b,
         0.5 * rgb.r - 0.5 * rgb.b,
        -0.25 * rgb.r + 0.5 * rgb.g - 0.25 * rgb.b,
    );
}

fn ycocg_to_rgb(ycocg: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        ycocg.x + ycocg.y - ycocg.z,
        ycocg.x + ycocg.z,
        ycocg.x - ycocg.y - ycocg.z,
    );
}

fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// Reinhard tonemapping weight — suppresses HDR fireflies in the blend
fn hdr_weight(c: vec3<f32>) -> f32 {
    return 1.0 / (1.0 + luminance(c));
}

@compute @workgroup_size(8, 8)
fn taa_resolve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = vec2<u32>(u32(taa.screen_size.x), u32(taa.screen_size.y));
    if gid.x >= size.x || gid.y >= size.y { return; }

    let pixel = vec2<i32>(gid.xy);
    let current = textureLoad(scene_color, pixel, 0).rgb;

    // First frame: output current only, no history blending
    if taa.params.y > 0.5 {
        textureStore(output_tex, pixel, vec4<f32>(current, 1.0));
        return;
    }

    // Read motion vector and reproject
    let motion = textureLoad(motion_tex, pixel, 0).rg;
    let uv = (vec2<f32>(gid.xy) + 0.5) * taa.screen_size.zw;
    let prev_uv = uv - motion;

    // Disocclusion: out-of-bounds or very large motion (>48 pixels)
    let motion_px = length(motion * taa.screen_size.xy);
    let oob = prev_uv.x < 0.0 || prev_uv.x > 1.0 || prev_uv.y < 0.0 || prev_uv.y > 1.0;
    if oob || motion_px > 48.0 {
        textureStore(output_tex, pixel, vec4<f32>(current, 1.0));
        return;
    }

    // Sample history with bilinear filtering at reprojected UV
    let history = textureSampleLevel(history_tex, history_samp, prev_uv, 0.0).rgb;

    // 3×3 neighborhood variance clipping in YCoCg space (tighter than min/max AABB)
    var moment1 = vec3<f32>(0.0);
    var moment2 = vec3<f32>(0.0);
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let neighbor = clamp(
                pixel + vec2<i32>(dx, dy),
                vec2<i32>(0),
                vec2<i32>(size) - 1,
            );
            let s = rgb_to_ycocg(textureLoad(scene_color, neighbor, 0).rgb);
            moment1 += s;
            moment2 += s * s;
        }
    }
    let mean = moment1 / 9.0;
    let variance = max(moment2 / 9.0 - mean * mean, vec3<f32>(0.0));
    let stddev = sqrt(variance);
    let gamma = 1.25; // tightness: lower = less ghosting, higher = more stability
    let clip_min = mean - stddev * gamma;
    let clip_max = mean + stddev * gamma;

    // Clip history to variance-based bounds
    let clamped_history = ycocg_to_rgb(clamp(rgb_to_ycocg(history), clip_min, clip_max));

    // Reinhard-weighted blend to prevent HDR firefly flicker
    let w_current = hdr_weight(current);
    let w_history = hdr_weight(clamped_history);
    let blend = taa.params.x; // 0.1 = 10% current, 90% history
    let result = (current * w_current * blend + clamped_history * w_history * (1.0 - blend))
              / (w_current * blend + w_history * (1.0 - blend));

    // Inline sharpening — uses the variance-clipped current-frame center and its
    // immediate neighbors (all from same temporal blend pass) for consistency.
    // We approximate the blur kernel from the already-computed neighborhood mean.
    let sharp = result + (result - ycocg_to_rgb(mean)) * 0.15;

    textureStore(output_tex, pixel, vec4<f32>(max(sharp, vec3<f32>(0.0)), 1.0));
}
