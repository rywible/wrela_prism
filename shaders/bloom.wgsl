// Bloom compute shaders: threshold extraction + separable Gaussian blur.

struct BloomUniforms {
    screen_size: vec4<f32>,       // full_width, full_height, half_width, half_height
    params: vec4<f32>,            // threshold, intensity, 0, 0
};

// --- Threshold pass ---
@group(0) @binding(0)
var<uniform> u: BloomUniforms;

@group(0) @binding(1)
var scene_color: texture_2d<f32>;

@group(0) @binding(2)
var bloom_out: texture_storage_2d<rgba16float, write>;

fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

@compute @workgroup_size(8, 8)
fn bloom_threshold(@builtin(global_invocation_id) gid: vec3<u32>) {
    let half_w = u32(u.screen_size.z);
    let half_h = u32(u.screen_size.w);
    if gid.x >= half_w || gid.y >= half_h {
        return;
    }

    // Sample 2x2 block from full-res and average (box downsample)
    let full_coord = vec2<i32>(gid.xy) * 2;
    let c00 = textureLoad(scene_color, full_coord, 0).rgb;
    let c10 = textureLoad(scene_color, full_coord + vec2(1, 0), 0).rgb;
    let c01 = textureLoad(scene_color, full_coord + vec2(0, 1), 0).rgb;
    let c11 = textureLoad(scene_color, full_coord + vec2(1, 1), 0).rgb;
    let avg = (c00 + c10 + c01 + c11) * 0.25;

    let threshold = u.params.x;
    let lum = luminance(avg);
    let bright = avg * max(lum - threshold, 0.0) / max(lum, 0.001);

    textureStore(bloom_out, vec2<i32>(gid.xy), vec4(bright, 1.0));
}

// --- Horizontal blur pass ---
@group(0) @binding(0)
var<uniform> blur_h_u: BloomUniforms;

@group(0) @binding(1)
var blur_h_input: texture_2d<f32>;

@group(0) @binding(2)
var blur_h_output: texture_storage_2d<rgba16float, write>;

const BLUR_WEIGHTS: array<f32, 4> = array<f32, 4>(0.3829, 0.2417, 0.0606, 0.0060);

@compute @workgroup_size(8, 8)
fn bloom_blur_h(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = u32(blur_h_u.screen_size.z);
    let h = u32(blur_h_u.screen_size.w);
    if gid.x >= w || gid.y >= h {
        return;
    }

    let coord = vec2<i32>(gid.xy);
    var result = textureLoad(blur_h_input, coord, 0).rgb * BLUR_WEIGHTS[0];
    for (var i = 1; i < 4; i++) {
        let offset = vec2<i32>(i, 0);
        let left = max(coord - offset, vec2(0));
        let right = min(coord + offset, vec2(i32(w) - 1, i32(h) - 1));
        result += textureLoad(blur_h_input, left, 0).rgb * BLUR_WEIGHTS[i];
        result += textureLoad(blur_h_input, right, 0).rgb * BLUR_WEIGHTS[i];
    }

    textureStore(blur_h_output, coord, vec4(result, 1.0));
}

// --- Vertical blur pass ---
@group(0) @binding(0)
var<uniform> blur_v_u: BloomUniforms;

@group(0) @binding(1)
var blur_v_input: texture_2d<f32>;

@group(0) @binding(2)
var blur_v_output: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn bloom_blur_v(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = u32(blur_v_u.screen_size.z);
    let h = u32(blur_v_u.screen_size.w);
    if gid.x >= w || gid.y >= h {
        return;
    }

    let coord = vec2<i32>(gid.xy);
    var result = textureLoad(blur_v_input, coord, 0).rgb * BLUR_WEIGHTS[0];
    for (var i = 1; i < 4; i++) {
        let offset = vec2<i32>(0, i);
        let top = max(coord - offset, vec2(0));
        let bot = min(coord + offset, vec2(i32(w) - 1, i32(h) - 1));
        result += textureLoad(blur_v_input, top, 0).rgb * BLUR_WEIGHTS[i];
        result += textureLoad(blur_v_input, bot, 0).rgb * BLUR_WEIGHTS[i];
    }

    textureStore(blur_v_output, coord, vec4(result, 1.0));
}
