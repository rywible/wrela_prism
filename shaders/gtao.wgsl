// Ground Truth Ambient Occlusion (GTAO) — horizon-based AO with bent normal output.
//
// For each pixel, traces the depth buffer in N azimuthal directions, finds the
// maximum horizon angle per slice, and integrates the cosine-weighted unoccluded
// solid angle. Outputs R16Float AO + Rgba16Float bent normal.
//
// Reference: Jimenez et al., "Practical Real-Time Strategies for Accurate Indirect Occlusion"

struct GtaoUniforms {
    projection: mat4x4<f32>,
    view: mat4x4<f32>,
    inv_projection: mat4x4<f32>,
    screen_size: vec4<f32>,     // width, height, 1/width, 1/height
    params: vec4<f32>,          // radius, bias, intensity, frame_index
};

// =========================================================================
// GTAO sample pass
// =========================================================================

@group(0) @binding(0)
var<uniform> u: GtaoUniforms;

@group(0) @binding(1)
var gbuffer1_tex: texture_2d<f32>;

@group(0) @binding(2)
var depth_tex: texture_depth_2d;

@group(0) @binding(3)
var ao_out: texture_storage_2d<r32float, write>;

@group(0) @binding(4)
var bent_normal_out: texture_storage_2d<rgba16float, write>;

const PI: f32 = 3.14159265;
const TWO_PI: f32 = 6.28318530;

const NUM_DIRECTIONS: u32 = 6u;
const NUM_STEPS: u32 = 10u;

fn decode_world_normal(encoded: vec3<f32>) -> vec3<f32> {
    return normalize(encoded * 2.0 - 1.0);
}

fn reconstruct_view_pos(pixel: vec2<i32>, depth: f32) -> vec3<f32> {
    let uv = (vec2<f32>(pixel) + 0.5) * u.screen_size.zw;
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, depth, 1.0);
    let view = u.inv_projection * ndc;
    return view.xyz / view.w;
}

// Spatial hash for per-pixel jitter (interleaved gradient noise variant)
fn hash_pixel(p: vec2<u32>) -> f32 {
    let h = p.x * 73856093u ^ p.y * 19349663u;
    return f32(h % 1024u) / 1024.0;
}

// Second hash channel for step offset jitter
fn hash_pixel2(p: vec2<u32>) -> f32 {
    let h = p.x * 83492791u ^ p.y * 37139213u;
    return f32(h % 1024u) / 1024.0;
}

// Integrate the cosine-weighted AO for a single horizon slice.
// Given horizon angles h1, h2 (in [-pi/2, pi/2] relative to the tangent plane)
// and the projected normal angle n, returns the fraction of unoccluded solid angle.
fn integrate_arc(h1: f32, h2: f32, n: f32) -> f32 {
    let cos_n = cos(n);
    let sin_n = sin(n);
    // Cosine-weighted integral of the visible arc:
    // 0.25 * (-cos(2*h1 - n) + cos_n + 2*h1*sin_n)
    // + 0.25 * (-cos(2*h2 - n) + cos_n + 2*h2*sin_n)
    let a1 = -cos(2.0 * h1 - n) + cos_n + 2.0 * h1 * sin_n;
    let a2 = -cos(2.0 * h2 - n) + cos_n + 2.0 * h2 * sin_n;
    return 0.25 * (a1 + a2);
}

@compute @workgroup_size(8, 8)
fn gtao_sample(@builtin(global_invocation_id) gid: vec3<u32>) {
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
        textureStore(bent_normal_out, coord, vec4(0.0, 1.0, 0.0, 0.0));
        return;
    }

    let view_pos = reconstruct_view_pos(coord, depth);
    let g1 = textureLoad(gbuffer1_tex, coord, 0);
    let normal_world = decode_world_normal(g1.xyz);
    let normal_view = normalize((u.view * vec4(normal_world, 0.0)).xyz);

    let radius = u.params.x;
    let bias = u.params.y;
    let intensity = u.params.z;
    let frame_index = u.params.w;

    // Per-pixel jitter: spatial noise + temporal golden angle rotation
    let golden_angle = 2.3999632; // pi * (3 - sqrt(5))
    let spatial_noise = hash_pixel(gid.xy);
    let step_noise = hash_pixel2(gid.xy);
    let angle_offset = spatial_noise * PI + frame_index * golden_angle;

    // Screen-space radius: project world-space radius to pixels
    // Use the z-depth of the pixel to determine how many pixels the radius covers
    let ss_radius = max(radius * abs(u.projection[0][0]) / max(-view_pos.z, 0.01), 2.0);
    let max_ss_radius = min(ss_radius, min(f32(screen_w), f32(screen_h)) * 0.25);
    let step_size = max_ss_radius / f32(NUM_STEPS);

    var total_ao = 0.0;
    var bent_normal = vec3<f32>(0.0);

    for (var dir = 0u; dir < NUM_DIRECTIONS; dir++) {
        let angle = (f32(dir) + 0.5) / f32(NUM_DIRECTIONS) * PI + angle_offset;
        let dir2d = vec2<f32>(cos(angle), sin(angle));

        // Find horizon angle for positive and negative directions along this slice
        var max_horizon_pos = -PI * 0.5 + bias;
        var max_horizon_neg = -PI * 0.5 + bias;

        // Project view-space normal onto this slice direction
        // The tangent vector in view space along this screen direction
        let tangent_ss = vec3<f32>(dir2d.x * u.screen_size.z, dir2d.y * u.screen_size.w, 0.0);

        for (var step = 1u; step <= NUM_STEPS; step++) {
            let offset_len = (f32(step) + step_noise) * step_size;
            let sample_offset = dir2d * offset_len;

            // Positive direction
            {
                let sample_pixel = vec2<i32>(vec2<f32>(coord) + sample_offset + 0.5);
                if sample_pixel.x >= 0 && sample_pixel.x < i32(screen_w) &&
                   sample_pixel.y >= 0 && sample_pixel.y < i32(screen_h) {
                    let sample_depth = textureLoad(depth_tex, sample_pixel, 0);
                    if sample_depth > 0.0001 {
                        let sample_view_pos = reconstruct_view_pos(sample_pixel, sample_depth);
                        let delta = sample_view_pos - view_pos;
                        let dist = length(delta);
                        if dist > 0.001 && dist < radius * 2.0 {
                            let horizon = atan2(-delta.z, length(delta.xy));
                            // Distance-based attenuation
                            let falloff = saturate(1.0 - dist / (radius * 2.0));
                            let weighted_horizon = mix(-PI * 0.5, horizon, falloff);
                            max_horizon_pos = max(max_horizon_pos, weighted_horizon);
                        }
                    }
                }
            }

            // Negative direction
            {
                let sample_pixel = vec2<i32>(vec2<f32>(coord) - sample_offset + 0.5);
                if sample_pixel.x >= 0 && sample_pixel.x < i32(screen_w) &&
                   sample_pixel.y >= 0 && sample_pixel.y < i32(screen_h) {
                    let sample_depth = textureLoad(depth_tex, sample_pixel, 0);
                    if sample_depth > 0.0001 {
                        let sample_view_pos = reconstruct_view_pos(sample_pixel, sample_depth);
                        let delta = sample_view_pos - view_pos;
                        let dist = length(delta);
                        if dist > 0.001 && dist < radius * 2.0 {
                            let horizon = atan2(-delta.z, length(delta.xy));
                            let falloff = saturate(1.0 - dist / (radius * 2.0));
                            let weighted_horizon = mix(-PI * 0.5, horizon, falloff);
                            max_horizon_neg = max(max_horizon_neg, weighted_horizon);
                        }
                    }
                }
            }
        }

        // Project normal onto the slice plane to get the normal angle
        // Slice is defined by dir2d in screen space; we need it in view space
        let slice_dir_view = normalize(vec3<f32>(dir2d.x, dir2d.y, 0.0));
        let proj_normal = dot(normal_view, slice_dir_view);
        let proj_normal_z = normal_view.z;
        let n_angle = atan2(proj_normal_z, abs(proj_normal) + 0.0001);

        // Integrate the visible arc
        let vis_pos = integrate_arc(max_horizon_pos, PI * 0.5, n_angle);
        let vis_neg = integrate_arc(max_horizon_neg, PI * 0.5, -n_angle);
        total_ao += (vis_pos + vis_neg) * 0.5;

        // Accumulate bent normal: average direction of unoccluded sky
        // The mid-angle between the two horizon limits, projected back to 3D
        let avg_horizon_pos = (max_horizon_pos + PI * 0.5) * 0.5;
        let avg_horizon_neg = (max_horizon_neg + PI * 0.5) * 0.5;
        let bent_dir_pos = vec3<f32>(dir2d * cos(avg_horizon_pos), sin(avg_horizon_pos));
        let bent_dir_neg = vec3<f32>(-dir2d * cos(avg_horizon_neg), sin(avg_horizon_neg));
        bent_normal += bent_dir_pos + bent_dir_neg;
    }

    total_ao /= f32(NUM_DIRECTIONS);
    let ao = clamp(1.0 - (1.0 - total_ao) * intensity, 0.0, 1.0);

    // Normalize bent normal (falls back to surface normal if fully unoccluded)
    var bn = bent_normal;
    let bn_len = length(bn);
    if bn_len > 0.001 {
        bn = bn / bn_len;
    } else {
        bn = normal_view;
    }

    textureStore(ao_out, coord, vec4(ao));
    textureStore(bent_normal_out, coord, vec4(bn, 0.0));
}

// =========================================================================
// Bilateral blur pass (edge-preserving, same approach as SSAO)
// =========================================================================

@group(0) @binding(0)
var<uniform> blur_u: GtaoUniforms;

@group(0) @binding(1)
var gbuffer1_blur: texture_2d<f32>;

@group(0) @binding(2)
var depth_blur: texture_depth_2d;

@group(0) @binding(3)
var raw_ao_tex: texture_2d<f32>;

@group(0) @binding(4)
var raw_bent_normal_tex: texture_2d<f32>;

@group(0) @binding(5)
var blurred_ao_out: texture_storage_2d<r32float, write>;

@group(0) @binding(6)
var blurred_bent_normal_out: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn gtao_blur(@builtin(global_invocation_id) gid: vec3<u32>) {
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
    var total_bn = vec3<f32>(0.0);
    var total_weight = 0.0;

    for (var dy = -2; dy <= 2; dy++) {
        for (var dx = -2; dx <= 2; dx++) {
            let sample_coord = coord + vec2<i32>(dx, dy);
            if sample_coord.x < 0 || sample_coord.x >= i32(screen_w) ||
               sample_coord.y < 0 || sample_coord.y >= i32(screen_h) {
                continue;
            }

            let sample_ao = textureLoad(raw_ao_tex, sample_coord, 0).r;
            let sample_bn = textureLoad(raw_bent_normal_tex, sample_coord, 0).xyz;
            let sample_depth = textureLoad(depth_blur, sample_coord, 0);
            let sample_g1 = textureLoad(gbuffer1_blur, sample_coord, 0);
            let sample_normal = decode_world_normal(sample_g1.xyz);

            // Depth similarity weight
            let depth_diff = abs(center_depth - sample_depth);
            let depth_weight = exp(-depth_diff * 1000.0);

            // Normal similarity weight
            let normal_weight = max(dot(center_normal, sample_normal), 0.0);

            // Spatial Gaussian weight (sigma ~ 1.5)
            let spatial_dist = f32(dx * dx + dy * dy);
            let spatial_weight = exp(-spatial_dist / 4.5);

            let w = depth_weight * normal_weight * spatial_weight;
            total_ao += sample_ao * w;
            total_bn += sample_bn * w;
            total_weight += w;
        }
    }

    let final_ao = total_ao / max(total_weight, 0.0001);
    var final_bn = total_bn / max(total_weight, 0.0001);
    let bn_len = length(final_bn);
    if bn_len > 0.001 {
        final_bn = final_bn / bn_len;
    }

    textureStore(blurred_ao_out, coord, vec4(final_ao));
    textureStore(blurred_bent_normal_out, coord, vec4(final_bn, 0.0));
}
