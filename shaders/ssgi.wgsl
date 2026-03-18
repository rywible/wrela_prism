// Screen-Space Global Illumination — sample + blur compute shaders.
//
// Sample pass: 8 hemisphere samples per pixel in screen space,
// reads scene color at hit points, weighted by cosine / distance².
// Blur pass: bilateral blur with depth + normal edge-stopping.

struct SsgiUniforms {
    view: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    projection: mat4x4<f32>,
    inv_projection: mat4x4<f32>,
    screen_size: vec4<f32>,   // (width, height, 1/width, 1/height)
    params: vec4<f32>,        // (gi_intensity, radius, frame_index, 0)
};

@group(0) @binding(0) var<uniform> uniforms: SsgiUniforms;
@group(0) @binding(1) var normal_tex: texture_2d<f32>;
@group(0) @binding(2) var depth_tex: texture_depth_2d;
@group(0) @binding(3) var color_tex: texture_2d<f32>;
@group(0) @binding(4) var output_tex: texture_storage_2d<rgba16float, write>;

const PI: f32 = 3.14159265359;
const NUM_SAMPLES: u32 = 8u;

// Per-pixel hash for sample rotation
fn hash_pixel(pixel: vec2<u32>) -> f32 {
    let p = vec2<f32>(f32(pixel.x), f32(pixel.y));
    return fract(sin(dot(p, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}

// Reconstruct view-space position from depth and screen UV
fn view_pos_from_depth(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0);
    let clip = vec4<f32>(ndc, depth, 1.0);
    let view = uniforms.inv_projection * clip;
    return view.xyz / view.w;
}

// Cosine-weighted hemisphere directions (precomputed, low-discrepancy)
const HEMISPHERE_DIRS: array<vec3<f32>, 8> = array<vec3<f32>, 8>(
    vec3<f32>( 0.5774,  0.5774,  0.5774),
    vec3<f32>(-0.5774,  0.5774,  0.5774),
    vec3<f32>( 0.5774,  0.5774, -0.5774),
    vec3<f32>(-0.5774,  0.5774, -0.5774),
    vec3<f32>( 0.0,     1.0,     0.0),
    vec3<f32>( 0.7071,  0.7071,  0.0),
    vec3<f32>( 0.0,     0.7071,  0.7071),
    vec3<f32>(-0.7071,  0.7071,  0.0),
);

@compute @workgroup_size(8, 8)
fn ssgi_sample(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pixel = gid.xy;
    let screen = vec2<u32>(u32(uniforms.screen_size.x), u32(uniforms.screen_size.y));
    if pixel.x >= screen.x || pixel.y >= screen.y {
        return;
    }

    let uv = (vec2<f32>(pixel) + 0.5) * uniforms.screen_size.zw;

    // Read depth and normal
    let depth = textureLoad(depth_tex, pixel, 0);
    if depth <= 0.0001 {
        textureStore(output_tex, pixel, vec4<f32>(0.0));
        return;
    }

    let normal_encoded = textureLoad(normal_tex, pixel, 0).rgb;
    let world_normal = normalize(normal_encoded * 2.0 - 1.0);
    // Transform normal to view space
    let view_normal = normalize((uniforms.view * vec4<f32>(world_normal, 0.0)).xyz);

    let view_pos = view_pos_from_depth(uv, depth);

    let gi_radius = uniforms.params.y;

    // Per-pixel rotation angle + per-frame golden angle offset for TAA accumulation
    let golden_angle = 2.3999632; // π(3−√5) radians — maximally irrational rotation
    let rotation = hash_pixel(pixel) * PI * 2.0 + uniforms.params.z * golden_angle;
    let cos_r = cos(rotation);
    let sin_r = sin(rotation);

    var indirect = vec3<f32>(0.0);
    var total_weight = 0.0;

    for (var i = 0u; i < NUM_SAMPLES; i++) {
        // Get hemisphere sample direction, rotate per-pixel
        var sample_dir = HEMISPHERE_DIRS[i];
        let rotated_x = sample_dir.x * cos_r - sample_dir.z * sin_r;
        let rotated_z = sample_dir.x * sin_r + sample_dir.z * cos_r;
        sample_dir = vec3<f32>(rotated_x, sample_dir.y, rotated_z);

        // Orient to normal hemisphere
        if dot(sample_dir, view_normal) < 0.0 {
            sample_dir = -sample_dir;
        }

        // Scale and offset in view space
        let sample_scale = gi_radius * (f32(i) + 1.0) / f32(NUM_SAMPLES);
        let sample_pos = view_pos + sample_dir * sample_scale;

        // Project to screen
        let clip = uniforms.projection * vec4<f32>(sample_pos, 1.0);
        let ndc = clip.xyz / clip.w;
        let sample_uv = vec2<f32>(ndc.x * 0.5 + 0.5, 1.0 - (ndc.y * 0.5 + 0.5));
        let sample_pixel = vec2<u32>(vec2<i32>(sample_uv * vec2<f32>(screen)));

        // Bounds check
        if sample_pixel.x >= screen.x || sample_pixel.y >= screen.y {
            continue;
        }

        // Read scene depth at sample position
        let sample_depth = textureLoad(depth_tex, sample_pixel, 0);
        let sample_view_pos = view_pos_from_depth(sample_uv, sample_depth);

        // Check if sample is close enough (screen-space hit)
        let diff = sample_view_pos - view_pos;
        let dist = length(diff);
        if dist > gi_radius * 2.0 || dist < 0.01 {
            continue;
        }

        // Read color at hit point
        let hit_color = textureLoad(color_tex, sample_pixel, 0).rgb;

        // Weight by cosine of angle between normal and sample direction
        let cos_angle = max(dot(view_normal, normalize(diff)), 0.0);
        // Distance attenuation (inverse square)
        let atten = 1.0 / (1.0 + dist * dist);
        let weight = cos_angle * atten;

        indirect += hit_color * weight;
        total_weight += weight;
    }

    if total_weight > 0.001 {
        indirect /= total_weight;
    }

    indirect *= uniforms.params.x; // gi_intensity

    textureStore(output_tex, pixel, vec4<f32>(indirect, 1.0));
}

// -- Bilateral blur pass --

@group(0) @binding(0) var<uniform> blur_uniforms: SsgiUniforms;
@group(0) @binding(1) var blur_normal_tex: texture_2d<f32>;
@group(0) @binding(2) var blur_depth_tex: texture_depth_2d;
@group(0) @binding(3) var blur_input_tex: texture_2d<f32>;
@group(0) @binding(4) var blur_output_tex: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn ssgi_blur(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pixel = gid.xy;
    let screen = vec2<u32>(u32(blur_uniforms.screen_size.x), u32(blur_uniforms.screen_size.y));
    if pixel.x >= screen.x || pixel.y >= screen.y {
        return;
    }

    let center_depth = textureLoad(blur_depth_tex, pixel, 0);
    let center_normal = textureLoad(blur_normal_tex, pixel, 0).rgb * 2.0 - 1.0;
    let center_color = textureLoad(blur_input_tex, pixel, 0).rgb;

    if center_depth <= 0.0001 {
        textureStore(blur_output_tex, pixel, vec4<f32>(0.0));
        return;
    }

    var result = center_color;
    var total_weight = 1.0;

    // 3x3 bilateral kernel
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            if dx == 0 && dy == 0 { continue; }

            let neighbor = vec2<i32>(vec2<i32>(pixel) + vec2<i32>(dx, dy));
            if neighbor.x < 0 || neighbor.y < 0 ||
               neighbor.x >= i32(screen.x) || neighbor.y >= i32(screen.y) {
                continue;
            }
            let np = vec2<u32>(neighbor);

            let n_depth = textureLoad(blur_depth_tex, np, 0);
            let n_normal = textureLoad(blur_normal_tex, np, 0).rgb * 2.0 - 1.0;
            let n_color = textureLoad(blur_input_tex, np, 0).rgb;

            // Depth edge-stopping
            let depth_diff = abs(center_depth - n_depth);
            let depth_weight = exp(-depth_diff * 1000.0);

            // Normal edge-stopping
            let normal_dot = max(dot(center_normal, n_normal), 0.0);
            let normal_weight = pow(normal_dot, 8.0);

            let w = depth_weight * normal_weight;
            result += n_color * w;
            total_weight += w;
        }
    }

    result /= total_weight;
    textureStore(blur_output_tex, pixel, vec4<f32>(result, 1.0));
}
