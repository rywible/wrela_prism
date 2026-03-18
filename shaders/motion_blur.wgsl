// Motion Blur — tile-max velocity + per-pixel directional blur.
//
// Two compute sub-passes:
// 1. Tile-max: downsample motion vectors to 20x20 tiles, take max velocity.
// 2. Directional blur: blur along tile velocity direction (12 samples).

struct MotionBlurUniforms {
    screen_size: vec4<f32>,   // width, height, 1/width, 1/height
    params: vec4<f32>,        // x=max_blur_px, y=tile_size, z=unused, w=unused
};

// ── Tile-max pass ──

@group(0) @binding(0) var<uniform> tile_uniforms: MotionBlurUniforms;
@group(0) @binding(1) var motion_tex: texture_2d<f32>;
@group(0) @binding(2) var tile_max_out: texture_storage_2d<rg32float, write>;

@compute @workgroup_size(8, 8)
fn tile_max(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tile_coord = vec2<i32>(gid.xy);
    let tile_size = i32(tile_uniforms.params.y);
    let screen_size = vec2<i32>(tile_uniforms.screen_size.xy);

    let tiles_x = (screen_size.x + tile_size - 1) / tile_size;
    let tiles_y = (screen_size.y + tile_size - 1) / tile_size;
    if tile_coord.x >= tiles_x || tile_coord.y >= tiles_y { return; }

    let base = tile_coord * tile_size;
    var max_vel = vec2<f32>(0.0);
    var max_len2 = 0.0;

    for (var dy = 0; dy < tile_size; dy++) {
        for (var dx = 0; dx < tile_size; dx++) {
            let px = base + vec2<i32>(dx, dy);
            if px.x >= screen_size.x || px.y >= screen_size.y { continue; }
            let vel = textureLoad(motion_tex, px, 0).rg;
            let len2 = dot(vel, vel);
            if len2 > max_len2 {
                max_vel = vel;
                max_len2 = len2;
            }
        }
    }

    textureStore(tile_max_out, tile_coord, vec4<f32>(max_vel, 0.0, 0.0));
}

// ── Directional blur pass ──

@group(0) @binding(0) var<uniform> blur_uniforms: MotionBlurUniforms;
@group(0) @binding(1) var tile_max_tex: texture_2d<f32>;
@group(0) @binding(2) var hdr_tex: texture_2d<f32>;
@group(0) @binding(3) var blur_out: texture_storage_2d<rgba16float, write>;

const NUM_SAMPLES: i32 = 12;

@compute @workgroup_size(8, 8)
fn motion_blur(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = vec2<i32>(gid.xy);
    let size = vec2<i32>(blur_uniforms.screen_size.xy);
    if px.x >= size.x || px.y >= size.y { return; }

    let tile_size = i32(blur_uniforms.params.y);
    let tile_coord = px / tile_size;

    // Read tile-max velocity (in UV space from TAA motion vectors)
    let tile_vel = textureLoad(tile_max_tex, tile_coord, 0).rg;

    // Convert UV velocity to pixel velocity
    let vel_px = tile_vel * vec2<f32>(size);
    let vel_len = length(vel_px);

    // Skip blur for near-static pixels
    if vel_len < 0.5 {
        let passthrough = textureLoad(hdr_tex, px, 0);
        textureStore(blur_out, px, passthrough);
        return;
    }

    // Clamp max blur length
    let max_blur = blur_uniforms.params.x;
    let clamped_len = min(vel_len, max_blur);
    let dir = normalize(vel_px) * clamped_len;

    var color_sum = vec3<f32>(0.0);
    for (var i = 0; i < NUM_SAMPLES; i++) {
        let t = (f32(i) / f32(NUM_SAMPLES - 1)) - 0.5;  // -0.5 to 0.5
        let sample_px = px + vec2<i32>(dir * t);
        let clamped = clamp(sample_px, vec2<i32>(0), size - 1);
        color_sum += textureLoad(hdr_tex, clamped, 0).rgb;
    }

    let result = color_sum / f32(NUM_SAMPLES);
    textureStore(blur_out, px, vec4<f32>(result, 1.0));
}
