// Screen-space ambient occlusion — sample + bilateral blur compute shaders.

struct SsaoUniforms {
    projection: mat4x4<f32>,
    view: mat4x4<f32>,
    inv_projection: mat4x4<f32>,
    screen_size: vec4<f32>,     // width, height, 1/width, 1/height
    params: vec4<f32>,          // radius, bias, intensity, frame_index
};

@group(0) @binding(0)
var<uniform> u: SsaoUniforms;

@group(0) @binding(1)
var gbuffer1_tex: texture_2d<f32>;

@group(0) @binding(2)
var depth_tex: texture_depth_2d;

@group(0) @binding(3)
var ao_out: texture_storage_2d<r32float, write>;

// Precomputed hemisphere kernel (32 samples: 8 fine + 8 medium + 8 tight + 8 wide)
// Hammersley distribution with cosine weighting for even hemisphere coverage.
const KERNEL_SIZE: u32 = 32u;
const KERNEL: array<vec3<f32>, 32> = array<vec3<f32>, 32>(
    // 8 fine-radius samples (0.01-0.04) — captures micro-occlusion
    vec3(0.012, 0.028, 0.010),
    vec3(-0.018, 0.032, 0.014),
    vec3(0.025, 0.020, -0.015),
    vec3(-0.010, 0.035, -0.012),
    vec3(0.022, 0.018, 0.020),
    vec3(-0.015, 0.030, -0.018),
    vec3(0.008, 0.038, 0.012),
    vec3(-0.020, 0.025, 0.016),
    // 8 medium-radius samples (0.10-0.20) — mid-range occlusion
    vec3(0.10, 0.14, 0.08),
    vec3(-0.12, 0.16, 0.10),
    vec3(0.15, 0.10, -0.12),
    vec3(-0.08, 0.18, -0.10),
    vec3(0.11, 0.13, 0.14),
    vec3(-0.14, 0.12, -0.11),
    vec3(0.06, 0.19, 0.08),
    vec3(-0.10, 0.15, 0.12),
    // 8 tight-radius samples (original)
    vec3(0.04, 0.08, 0.04),
    vec3(-0.06, 0.05, 0.08),
    vec3(0.08, 0.03, -0.06),
    vec3(-0.03, 0.10, -0.04),
    vec3(0.05, 0.07, 0.07),
    vec3(-0.07, 0.06, -0.05),
    vec3(0.02, 0.12, 0.03),
    vec3(-0.04, 0.09, 0.06),
    // 8 wide-radius samples (original)
    vec3(0.20, 0.35, 0.15),
    vec3(-0.25, 0.30, 0.22),
    vec3(0.30, 0.20, -0.18),
    vec3(-0.15, 0.40, -0.20),
    vec3(0.22, 0.28, 0.25),
    vec3(-0.28, 0.25, -0.22),
    vec3(0.12, 0.45, 0.10),
    vec3(-0.18, 0.35, 0.28),
);

fn decode_world_normal(encoded: vec3<f32>) -> vec3<f32> {
    return normalize(encoded * 2.0 - 1.0);
}

fn reconstruct_view_pos(pixel: vec2<i32>, depth: f32) -> vec3<f32> {
    let uv = (vec2<f32>(pixel) + 0.5) * u.screen_size.zw;
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, depth, 1.0);
    let view = u.inv_projection * ndc;
    return view.xyz / view.w;
}

// Hash for pseudo-random rotation per pixel
fn hash_pixel(p: vec2<u32>) -> f32 {
    let h = p.x * 73856093u ^ p.y * 19349663u;
    return f32(h % 1024u) / 1024.0;
}

@compute @workgroup_size(8, 8)
fn ssao_sample(@builtin(global_invocation_id) gid: vec3<u32>) {
    let screen_w = u32(u.screen_size.x);
    let screen_h = u32(u.screen_size.y);
    if gid.x >= screen_w || gid.y >= screen_h {
        return;
    }

    let coord = vec2<i32>(gid.xy);
    let depth = textureLoad(depth_tex, coord, 0);

    // Skip sky pixels (reversed-Z: sky is near 0)
    if depth <= 0.0001 {
        textureStore(ao_out, coord, vec4(1.0));
        return;
    }

    let view_pos = reconstruct_view_pos(coord, depth);
    let g1 = textureLoad(gbuffer1_tex, coord, 0);
    let normal_world = decode_world_normal(g1.xyz);
    let normal = normalize((u.view * vec4(normal_world, 0.0)).xyz);

    let radius = u.params.x;
    let bias = u.params.y;
    let intensity = u.params.z;

    // Per-pixel random rotation angle + per-frame golden angle offset for TAA accumulation
    let golden_angle = 2.3999632; // π(3−√5) radians — maximally irrational rotation
    let angle = hash_pixel(gid.xy) * 6.2831853 + u.params.w * golden_angle;
    let cos_a = cos(angle);
    let sin_a = sin(angle);

    var occlusion = 0.0;
    for (var i = 0u; i < KERNEL_SIZE; i++) {
        var sample_vec = KERNEL[i];
        // Random rotation around Z
        let rotated = vec3(
            sample_vec.x * cos_a - sample_vec.z * sin_a,
            sample_vec.y,
            sample_vec.x * sin_a + sample_vec.z * cos_a,
        );
        // Orient to normal hemisphere
        let flip = select(1.0, -1.0, dot(rotated, normal) < 0.0);
        let sample_pos = view_pos + rotated * flip * radius;

        // Project sample to screen
        let clip = u.projection * vec4(sample_pos, 1.0);
        let ndc_xy = clip.xy / clip.w;
        let sample_uv = vec2(ndc_xy.x * 0.5 + 0.5, 0.5 - ndc_xy.y * 0.5);
        let sample_pixel = vec2<i32>(sample_uv * u.screen_size.xy);

        // Bounds check
        if sample_pixel.x < 0 || sample_pixel.x >= i32(screen_w) ||
           sample_pixel.y < 0 || sample_pixel.y >= i32(screen_h) {
            continue;
        }

        let sample_depth = textureLoad(depth_tex, sample_pixel, 0);
        let sample_view = reconstruct_view_pos(sample_pixel, sample_depth);

        // Range check and occlusion
        let range_check = smoothstep(0.0, 1.0, radius / max(abs(view_pos.z - sample_view.z), 0.001));
        let is_occluded = select(0.0, 1.0, sample_view.z > sample_pos.z + bias);
        occlusion += is_occluded * range_check;
    }

    let ao = 1.0 - (occlusion / f32(KERNEL_SIZE)) * intensity;
    textureStore(ao_out, coord, vec4(clamp(ao, 0.0, 1.0)));
}

// --- Bilateral blur pass ---

@group(0) @binding(0)
var<uniform> blur_u: SsaoUniforms;

@group(0) @binding(1)
var gbuffer1_blur: texture_2d<f32>;

@group(0) @binding(2)
var depth_blur: texture_depth_2d;

@group(0) @binding(3)
var raw_ao_tex: texture_2d<f32>;

@group(0) @binding(4)
var blurred_ao_out: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn ssao_blur(@builtin(global_invocation_id) gid: vec3<u32>) {
    let screen_w = u32(blur_u.screen_size.x);
    let screen_h = u32(blur_u.screen_size.y);
    if gid.x >= screen_w || gid.y >= screen_h {
        return;
    }

    let coord = vec2<i32>(gid.xy);
    let center_depth = textureLoad(depth_blur, coord, 0);
    let center_g1 = textureLoad(gbuffer1_blur, coord, 0);
    let center_normal = decode_world_normal(center_g1.xyz);

    var total_ao = 0.0;
    var total_weight = 0.0;

    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let sample_coord = coord + vec2<i32>(dx, dy);
            if sample_coord.x < 0 || sample_coord.x >= i32(screen_w) ||
               sample_coord.y < 0 || sample_coord.y >= i32(screen_h) {
                continue;
            }

            let sample_ao = textureLoad(raw_ao_tex, sample_coord, 0).r;
            let sample_depth = textureLoad(depth_blur, sample_coord, 0);
            let sample_g1 = textureLoad(gbuffer1_blur, sample_coord, 0);
            let sample_normal = decode_world_normal(sample_g1.xyz);

            // Depth similarity weight
            let depth_diff = abs(center_depth - sample_depth);
            let depth_weight = exp(-depth_diff * 1000.0);

            // Normal similarity weight
            let normal_weight = max(dot(center_normal, sample_normal), 0.0);

            let w = depth_weight * normal_weight;
            total_ao += sample_ao * w;
            total_weight += w;
        }
    }

    let final_ao = total_ao / max(total_weight, 0.0001);
    textureStore(blurred_ao_out, coord, vec4(final_ao));
}
