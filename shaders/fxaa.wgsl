// FXAA 3.11 — Fast Approximate Anti-Aliasing (fullscreen fragment shader).
//
// Operates on LDR output (after tonemap). Uses luminance-based edge detection
// and sub-pixel shift along detected edge direction.

@group(0) @binding(0) var ldr_tex: texture_2d<f32>;
@group(0) @binding(1) var ldr_sampler: sampler;

struct FxaaUniforms {
    screen_size: vec4<f32>,   // width, height, 1/width, 1/height
};
@group(0) @binding(2) var<uniform> fu: FxaaUniforms;

struct FullscreenOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) vi: u32) -> FullscreenOut {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    var out: FullscreenOut;
    let pos = positions[vi];
    out.position = vec4<f32>(pos, 0.0, 1.0);
    out.uv = vec2<f32>(pos.x * 0.5 + 0.5, 0.5 - pos.y * 0.5);
    return out;
}

fn fxaa_luma(c: vec3<f32>) -> f32 {
    // Green-weighted luminance (cheaper, good approximation)
    return dot(c, vec3<f32>(0.299, 0.587, 0.114));
}

// FXAA quality preset: 12 steps, medium quality
const FXAA_EDGE_THRESHOLD: f32 = 0.0625;      // 1/16
const FXAA_EDGE_THRESHOLD_MIN: f32 = 0.0312;   // 1/32
const FXAA_SUBPIX_QUALITY: f32 = 0.75;
const FXAA_SEARCH_STEPS: i32 = 10;
const FXAA_SEARCH_ACCELERATION: f32 = 1.5;

@fragment
fn fs_fxaa(input: FullscreenOut) -> @location(0) vec4<f32> {
    let uv = input.uv;
    let texel = vec2<f32>(fu.screen_size.z, fu.screen_size.w);

    // Sample center and 4 neighbors
    let rgb_m = textureSampleLevel(ldr_tex, ldr_sampler, uv, 0.0).rgb;
    let rgb_n = textureSampleLevel(ldr_tex, ldr_sampler, uv + vec2<f32>(0.0, -texel.y), 0.0).rgb;
    let rgb_s = textureSampleLevel(ldr_tex, ldr_sampler, uv + vec2<f32>(0.0,  texel.y), 0.0).rgb;
    let rgb_e = textureSampleLevel(ldr_tex, ldr_sampler, uv + vec2<f32>( texel.x, 0.0), 0.0).rgb;
    let rgb_w = textureSampleLevel(ldr_tex, ldr_sampler, uv + vec2<f32>(-texel.x, 0.0), 0.0).rgb;

    let luma_m = fxaa_luma(rgb_m);
    let luma_n = fxaa_luma(rgb_n);
    let luma_s = fxaa_luma(rgb_s);
    let luma_e = fxaa_luma(rgb_e);
    let luma_w = fxaa_luma(rgb_w);

    let luma_min = min(luma_m, min(min(luma_n, luma_s), min(luma_e, luma_w)));
    let luma_max = max(luma_m, max(max(luma_n, luma_s), max(luma_e, luma_w)));
    let luma_range = luma_max - luma_min;

    // Early exit for low-contrast regions
    if luma_range < max(FXAA_EDGE_THRESHOLD_MIN, luma_max * FXAA_EDGE_THRESHOLD) {
        return vec4<f32>(rgb_m, 1.0);
    }

    // Sample diagonal neighbors
    let rgb_nw = textureSampleLevel(ldr_tex, ldr_sampler, uv + vec2<f32>(-texel.x, -texel.y), 0.0).rgb;
    let rgb_ne = textureSampleLevel(ldr_tex, ldr_sampler, uv + vec2<f32>( texel.x, -texel.y), 0.0).rgb;
    let rgb_sw = textureSampleLevel(ldr_tex, ldr_sampler, uv + vec2<f32>(-texel.x,  texel.y), 0.0).rgb;
    let rgb_se = textureSampleLevel(ldr_tex, ldr_sampler, uv + vec2<f32>( texel.x,  texel.y), 0.0).rgb;

    let luma_nw = fxaa_luma(rgb_nw);
    let luma_ne = fxaa_luma(rgb_ne);
    let luma_sw = fxaa_luma(rgb_sw);
    let luma_se = fxaa_luma(rgb_se);

    // Sub-pixel aliasing detection
    let luma_ns = luma_n + luma_s;
    let luma_ew = luma_e + luma_w;
    let luma_corners_top = luma_nw + luma_ne;
    let luma_corners_bot = luma_sw + luma_se;
    let luma_corners_left = luma_nw + luma_sw;
    let luma_corners_right = luma_ne + luma_se;

    let edge_h = abs(luma_corners_top - 2.0 * luma_n) + abs(luma_ew - 2.0 * luma_m) * 2.0 + abs(luma_corners_bot - 2.0 * luma_s);
    let edge_v = abs(luma_corners_left - 2.0 * luma_w) + abs(luma_ns - 2.0 * luma_m) * 2.0 + abs(luma_corners_right - 2.0 * luma_e);

    let is_horizontal = edge_h >= edge_v;

    // Choose edge normal direction
    var luma_neg: f32;
    var luma_pos: f32;
    var step_length: f32;
    if is_horizontal {
        luma_neg = luma_n;
        luma_pos = luma_s;
        step_length = texel.y;
    } else {
        luma_neg = luma_w;
        luma_pos = luma_e;
        step_length = texel.x;
    }

    let gradient_neg = abs(luma_neg - luma_m);
    let gradient_pos = abs(luma_pos - luma_m);
    let is_neg = gradient_neg >= gradient_pos;

    var luma_local_avg: f32;
    var gradient_scaled: f32;
    if is_neg {
        step_length = -step_length;
        luma_local_avg = 0.5 * (luma_neg + luma_m);
        gradient_scaled = gradient_neg;
    } else {
        luma_local_avg = 0.5 * (luma_pos + luma_m);
        gradient_scaled = gradient_pos;
    }

    // Move UV to edge
    var uv_edge = uv;
    if is_horizontal {
        uv_edge.y += step_length * 0.5;
    } else {
        uv_edge.x += step_length * 0.5;
    }

    // Search along edge in both directions
    var edge_step: vec2<f32>;
    if is_horizontal {
        edge_step = vec2<f32>(texel.x, 0.0);
    } else {
        edge_step = vec2<f32>(0.0, texel.y);
    }

    var uv_neg = uv_edge - edge_step;
    var uv_pos = uv_edge + edge_step;
    var luma_end_neg = fxaa_luma(textureSampleLevel(ldr_tex, ldr_sampler, uv_neg, 0.0).rgb) - luma_local_avg;
    var luma_end_pos = fxaa_luma(textureSampleLevel(ldr_tex, ldr_sampler, uv_pos, 0.0).rgb) - luma_local_avg;

    var reached_neg = abs(luma_end_neg) >= gradient_scaled;
    var reached_pos = abs(luma_end_pos) >= gradient_scaled;

    // Search outward from edge
    for (var i = 1; i < FXAA_SEARCH_STEPS; i++) {
        let step_scale = 1.0 + f32(i) * (FXAA_SEARCH_ACCELERATION - 1.0);
        if !reached_neg {
            uv_neg -= edge_step * step_scale;
            luma_end_neg = fxaa_luma(textureSampleLevel(ldr_tex, ldr_sampler, uv_neg, 0.0).rgb) - luma_local_avg;
            reached_neg = abs(luma_end_neg) >= gradient_scaled;
        }
        if !reached_pos {
            uv_pos += edge_step * step_scale;
            luma_end_pos = fxaa_luma(textureSampleLevel(ldr_tex, ldr_sampler, uv_pos, 0.0).rgb) - luma_local_avg;
            reached_pos = abs(luma_end_pos) >= gradient_scaled;
        }
        if reached_neg && reached_pos { break; }
    }

    // Compute edge blend factor
    var dist_neg: f32;
    var dist_pos: f32;
    if is_horizontal {
        dist_neg = uv.x - uv_neg.x;
        dist_pos = uv_pos.x - uv.x;
    } else {
        dist_neg = uv.y - uv_neg.y;
        dist_pos = uv_pos.y - uv.y;
    }

    let is_closer_neg = dist_neg < dist_pos;
    let dist_final = min(dist_neg, dist_pos);
    let edge_length = dist_neg + dist_pos;
    let pixel_offset = 0.5 - dist_final / edge_length;

    // Verify the closer end has a luma step in the correct direction
    let is_luma_correct = select(
        (luma_end_pos < 0.0) != (luma_m - luma_local_avg < 0.0),
        (luma_end_neg < 0.0) != (luma_m - luma_local_avg < 0.0),
        is_closer_neg,
    );
    let edge_blend = select(0.0, pixel_offset, is_luma_correct);

    // Sub-pixel anti-aliasing
    let luma_avg = (2.0 * luma_ns + 2.0 * luma_ew + luma_corners_top + luma_corners_bot) / 12.0;
    let sub_pixel_offset = clamp(abs(luma_avg - luma_m) / luma_range, 0.0, 1.0);
    let sub_pixel_blend = (-2.0 * sub_pixel_offset + 3.0) * sub_pixel_offset * sub_pixel_offset;
    let sub_pixel_factor = sub_pixel_blend * sub_pixel_blend * FXAA_SUBPIX_QUALITY;

    let final_offset = max(edge_blend, sub_pixel_factor);

    var final_uv = uv;
    if is_horizontal {
        final_uv.y += final_offset * step_length;
    } else {
        final_uv.x += final_offset * step_length;
    }

    let result = textureSampleLevel(ldr_tex, ldr_sampler, final_uv, 0.0).rgb;
    return vec4<f32>(result, 1.0);
}
