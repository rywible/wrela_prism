// Bloom compute shaders: threshold + progressive downsample + tent-filter upsample.

struct BloomUniforms {
    screen_size: vec4<f32>,       // full_width, full_height, level_width, level_height
    params: vec4<f32>,            // threshold, intensity, bloom_softness, 0
    bloom_tint: vec4<f32>,        // rgb = tint color (default 1,1,1), w = unused
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
    let softness = u.params.z;
    let lum = luminance(avg);
    let knee = max(lum - threshold + softness, 0.0);
    let bright = avg * min(knee, lum) / max(lum, 0.001);

    textureStore(bloom_out, vec2<i32>(gid.xy), vec4(bright, 1.0));
}

// --- Downsample pass (2x2 box filter) ---
@group(0) @binding(0)
var<uniform> ds_u: BloomUniforms;

@group(0) @binding(1)
var ds_input: texture_2d<f32>;

@group(0) @binding(2)
var ds_output: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn bloom_downsample(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_w = u32(ds_u.screen_size.z);
    let out_h = u32(ds_u.screen_size.w);
    if gid.x >= out_w || gid.y >= out_h {
        return;
    }

    // 2x2 box downsample from the larger input
    let src_coord = vec2<i32>(gid.xy) * 2;
    let c00 = textureLoad(ds_input, src_coord, 0).rgb;
    let c10 = textureLoad(ds_input, src_coord + vec2(1, 0), 0).rgb;
    let c01 = textureLoad(ds_input, src_coord + vec2(0, 1), 0).rgb;
    let c11 = textureLoad(ds_input, src_coord + vec2(1, 1), 0).rgb;
    let avg = (c00 + c10 + c01 + c11) * 0.25;

    textureStore(ds_output, vec2<i32>(gid.xy), vec4(avg, 1.0));
}

// --- Upsample pass (tent filter + additive blend) ---
// binding 0: uniforms
// binding 1: smaller level texture (to upsample from)
// binding 2: current level texture (to read and blend with)
// binding 3: output storage texture (to write blended result)
@group(0) @binding(0)
var<uniform> us_u: BloomUniforms;

@group(0) @binding(1)
var us_smaller: texture_2d<f32>;

@group(0) @binding(2)
var us_current: texture_2d<f32>;

@group(0) @binding(3)
var us_output: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn bloom_upsample(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_w = u32(us_u.screen_size.z);
    let out_h = u32(us_u.screen_size.w);
    if gid.x >= out_w || gid.y >= out_h {
        return;
    }

    // 3x3 tent filter upsample from the smaller mip
    let src_dims = textureDimensions(us_smaller);
    let src_coord = vec2<i32>(gid.xy) / 2;
    let sc = clamp(src_coord, vec2(1), vec2(i32(src_dims.x) - 2, i32(src_dims.y) - 2));

    var sum = vec3<f32>(0.0);
    // Center (weight 4/16)
    sum += textureLoad(us_smaller, sc, 0).rgb * 4.0;
    // Cardinal neighbors (weight 2/16 each)
    sum += textureLoad(us_smaller, sc + vec2(-1, 0), 0).rgb * 2.0;
    sum += textureLoad(us_smaller, sc + vec2(1, 0), 0).rgb * 2.0;
    sum += textureLoad(us_smaller, sc + vec2(0, -1), 0).rgb * 2.0;
    sum += textureLoad(us_smaller, sc + vec2(0, 1), 0).rgb * 2.0;
    // Diagonal neighbors (weight 1/16 each)
    sum += textureLoad(us_smaller, sc + vec2(-1, -1), 0).rgb;
    sum += textureLoad(us_smaller, sc + vec2(1, -1), 0).rgb;
    sum += textureLoad(us_smaller, sc + vec2(-1, 1), 0).rgb;
    sum += textureLoad(us_smaller, sc + vec2(1, 1), 0).rgb;
    sum /= 16.0;

    // Blend with current level
    let current = textureLoad(us_current, vec2<i32>(gid.xy), 0).rgb;
    let blended = current + sum;

    textureStore(us_output, vec2<i32>(gid.xy), vec4(blended, 1.0));
}
