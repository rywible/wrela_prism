// Depth of Field — CoC computation + disk bokeh blur (near/far separation).
//
// Two compute sub-passes:
// 1. CoC pass: compute circle of confusion per pixel from depth + focus params.
// 2. Blur pass: weighted disk blur using CoC radius (near/far combined).

struct DofUniforms {
    screen_size: vec4<f32>,      // width, height, 1/width, 1/height
    focus_params: vec4<f32>,     // x=focus_distance, y=aperture, z=max_coc_px, w=unused
};

@group(0) @binding(0) var<uniform> uniforms: DofUniforms;
@group(0) @binding(1) var depth_tex: texture_depth_2d;
@group(0) @binding(2) var hdr_tex: texture_2d<f32>;
@group(0) @binding(3) var coc_tex: texture_storage_2d<r16float, write>;

// Linearize reversed-Z depth (near=1, far~=0) to view-space distance.
fn linearize_depth(d: f32) -> f32 {
    // Reversed-Z infinite far: z_ndc = near / dist => dist = near / z_ndc
    let near = 0.1;  // must match camera near plane
    if d <= 0.0 { return 10000.0; }
    return near / d;
}

@compute @workgroup_size(8, 8)
fn dof_coc(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = vec2<i32>(gid.xy);
    let size = vec2<i32>(uniforms.screen_size.xy);
    if px.x >= size.x || px.y >= size.y { return; }

    let raw_depth = textureLoad(depth_tex, px, 0);
    let dist = linearize_depth(raw_depth);

    let focus_dist = uniforms.focus_params.x;
    let aperture = uniforms.focus_params.y;
    let max_coc = uniforms.focus_params.z;

    // Simplified CoC: aperture * |1 - focus_distance / depth|
    // Positive = far field, negative = near field
    var coc = aperture * (1.0 - focus_dist / max(dist, 0.001));
    coc = clamp(coc, -max_coc, max_coc);

    // Normalize to pixel units (scale by screen height for consistent look)
    let coc_px = coc * uniforms.screen_size.y;

    textureStore(coc_tex, px, vec4<f32>(coc_px, 0.0, 0.0, 0.0));
}

// Blur pass bindings
@group(0) @binding(0) var<uniform> blur_uniforms: DofUniforms;
@group(0) @binding(1) var coc_in: texture_2d<f32>;
@group(0) @binding(2) var hdr_in: texture_2d<f32>;
@group(0) @binding(3) var blur_out: texture_storage_2d<rgba16float, write>;

// 16-tap disk kernel (Poisson disk approximation)
const DISK_OFFSETS: array<vec2<f32>, 16> = array<vec2<f32>, 16>(
    vec2<f32>( 0.0000,  0.0000),
    vec2<f32>( 0.5412,  0.1845),
    vec2<f32>(-0.1693,  0.5859),
    vec2<f32>(-0.5712, -0.0415),
    vec2<f32>( 0.0168, -0.5960),
    vec2<f32>( 0.7707,  0.5063),
    vec2<f32>(-0.6638,  0.5810),
    vec2<f32>(-0.7564, -0.4619),
    vec2<f32>( 0.2538, -0.8538),
    vec2<f32>( 0.9281, -0.1937),
    vec2<f32>(-0.2117,  0.9575),
    vec2<f32>(-0.9849,  0.1232),
    vec2<f32>(-0.3080, -0.9315),
    vec2<f32>( 0.5931, -0.7524),
    vec2<f32>( 0.8415,  0.8231),
    vec2<f32>(-0.8836, -0.8100),
);

@compute @workgroup_size(8, 8)
fn dof_blur(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = vec2<i32>(gid.xy);
    let size = vec2<i32>(blur_uniforms.screen_size.xy);
    if px.x >= size.x || px.y >= size.y { return; }

    let center_coc = textureLoad(coc_in, px, 0).r;
    let abs_coc = abs(center_coc);

    // Skip blur for pixels with negligible CoC
    if abs_coc < 0.5 {
        let passthrough = textureLoad(hdr_in, px, 0);
        textureStore(blur_out, px, passthrough);
        return;
    }

    // Clamp blur radius to something reasonable
    let blur_radius = min(abs_coc, blur_uniforms.focus_params.z);

    var color_sum = vec3<f32>(0.0);
    var weight_sum = 0.0;

    for (var i = 0; i < 16; i++) {
        let offset = DISK_OFFSETS[i] * blur_radius;
        let sample_px = px + vec2<i32>(offset);
        let clamped = clamp(sample_px, vec2<i32>(0), size - 1);

        let sample_coc = textureLoad(coc_in, clamped, 0).r;
        let sample_color = textureLoad(hdr_in, clamped, 0).rgb;

        // Weight: favor samples whose CoC agrees with the blur direction
        // Near field (coc < 0) bleeds over everything; far field only over far
        var w = 1.0;
        if center_coc > 0.0 {
            // Far field: only accept samples that are also far or at center
            w = select(0.1, 1.0, sample_coc >= -0.5);
        }
        // Distance falloff within the disk
        let dist = length(offset) / max(blur_radius, 1.0);
        w *= smoothstep(1.2, 0.0, dist);
        w = max(w, 0.001);

        color_sum += sample_color * w;
        weight_sum += w;
    }

    let result = color_sum / max(weight_sum, 0.001);
    textureStore(blur_out, px, vec4<f32>(result, 1.0));
}
