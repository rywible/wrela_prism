// Cloud composite — fullscreen fragment shader.
// Blends quarter-resolution cloud texture onto scene color.
// Depth-aware bilateral upsampling for clean cloud edges.
// scene_color = scene_color * cloud.a + cloud.rgb

@group(0) @binding(0) var cloud_tex: texture_2d<f32>;
@group(0) @binding(1) var cloud_sampler: sampler;
@group(0) @binding(2) var depth_tex: texture_2d<f32>;

// Cloud depth texture for bilateral weighting (R32Float)
@group(0) @binding(4) var cloud_depth_tex: texture_2d<f32>;

// Must match cloud_march.wgsl and sun_shafts_occlusion.wgsl.
const SKY_DEPTH_THRESHOLD: f32 = 0.9995;

// Must match full CloudUniforms layout so screen_params reads at the correct offset
struct CompositeUniforms {
    inv_view_proj: mat4x4<f32>,
    prev_view_proj: mat4x4<f32>,
    camera_position: vec4<f32>,
    sun_direction: vec4<f32>,
    sun_color: vec4<f32>,
    sky_ambient: vec4<f32>,
    cloud_params: vec4<f32>,
    screen_params: vec4<f32>,
    cloud_profile: vec4<f32>,
    cloud_profile2: vec4<f32>,
};
@group(0) @binding(3) var<uniform> cu: CompositeUniforms;

struct FullscreenOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

fn sample_cloud_tent(uv: vec2<f32>, texel_size: vec2<f32>) -> vec4<f32> {
    let center = clamp(uv, texel_size * 0.5, vec2<f32>(1.0) - texel_size * 0.5);
    let offsets = array<vec2<f32>, 9>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(0.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0, 0.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0),
    );
    let weights = array<f32, 9>(
        1.0, 2.0, 1.0,
        2.0, 4.0, 2.0,
        1.0, 2.0, 1.0,
    );

    var sum = vec4<f32>(0.0);
    var total = 0.0;
    for (var i = 0; i < 9; i++) {
        let sample_uv = clamp(center + offsets[i] * texel_size, texel_size * 0.5, vec2<f32>(1.0) - texel_size * 0.5);
        let sample_cloud = textureSampleLevel(cloud_tex, cloud_sampler, sample_uv, 0.0);
        sum += sample_cloud * weights[i];
        total += weights[i];
    }
    return sum / total;
}

@vertex
fn vs_composite(@builtin(vertex_index) vid: u32) -> FullscreenOutput {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    let p = pos[vid];
    var out: FullscreenOutput;
    out.position = vec4<f32>(p, 0.0, 1.0);
    out.uv = p * 0.5 + 0.5;
    out.uv.y = 1.0 - out.uv.y;
    return out;
}

fn ray_dir_from_uv(uv: vec2<f32>) -> vec3<f32> {
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0);
    let near_pt = cu.inv_view_proj * vec4<f32>(ndc, 0.0, 1.0);
    let far_pt = cu.inv_view_proj * vec4<f32>(ndc, 1.0, 1.0);
    return normalize(far_pt.xyz / far_pt.w - near_pt.xyz / near_pt.w);
}

@fragment
fn fs_composite(input: FullscreenOutput) -> @location(0) vec4<f32> {
    let quarter_size = vec2<f32>(cu.screen_params.x, cu.screen_params.y);
    let texel_size = vec2<f32>(cu.screen_params.z, cu.screen_params.w);

    // Full-res scene depth at this pixel
    let full_depth = textureLoad(depth_tex, vec2<i32>(input.position.xy), 0).r;

    // Quarter-res texel coordinate (continuous)
    let qcoord = input.uv * quarter_size - 0.5;
    let base = floor(qcoord);
    let frac_coord = qcoord - base;

    // 2x2 bilinear offsets
    let offsets = array<vec2<f32>, 4>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0),
    );

    // Bilinear weights
    let bw = array<f32, 4>(
        (1.0 - frac_coord.x) * (1.0 - frac_coord.y),
        frac_coord.x * (1.0 - frac_coord.y),
        (1.0 - frac_coord.x) * frac_coord.y,
        frac_coord.x * frac_coord.y,
    );

    // Sky pixels: tent filter for smooth interpolation
    // Geometry-adjacent pixels: bilateral upsampling to prevent cloud bleeding onto surfaces
    let is_sky = full_depth > SKY_DEPTH_THRESHOLD;

    var cloud: vec4<f32>;
    if is_sky {
        cloud = sample_cloud_tent(input.uv, texel_size);
    } else {
        // Bilateral upsampling near geometry
        let scene_depth_sigma_sq = 0.001;
        var total_color = vec4<f32>(0.0);
        var total_weight = 0.0;

        for (var i = 0; i < 4; i++) {
            let sample_coord = vec2<i32>(base + offsets[i]);
            let clamped = clamp(sample_coord, vec2<i32>(0), vec2<i32>(quarter_size) - vec2<i32>(1));
            let cloud_sample = textureLoad(cloud_tex, clamped, 0);

            let res_div = cu.sky_ambient.w; // resolution divisor (2 or 4)
            let full_pixel = clamp(
                vec2<i32>((vec2<f32>(clamped) + 0.5) * res_div),
                vec2<i32>(0),
                vec2<i32>(textureDimensions(depth_tex)) - vec2<i32>(1)
            );
            let sample_depth = textureLoad(depth_tex, full_pixel, 0).r;
            let sample_cloud_d = textureLoad(cloud_depth_tex, clamped, 0).r;
            var depth_weight = 1.0;
            if sample_cloud_d > 0.1 {
                let depth_diff = full_depth - sample_depth;
                depth_weight = exp(-(depth_diff * depth_diff) / scene_depth_sigma_sq);
            }

            let w = bw[i] * depth_weight;
            total_color += cloud_sample * w;
            total_weight += w;
        }

        if total_weight > 0.001 {
            cloud = total_color / total_weight;
        } else {
            cloud = textureSample(cloud_tex, cloud_sampler, input.uv);
        }
    }

    let ray_dir = ray_dir_from_uv(input.uv);
    let horizon_fade = smoothstep(0.01, 0.085, ray_dir.y);
    cloud = vec4<f32>(
        cloud.rgb * horizon_fade,
        mix(1.0, cloud.a, horizon_fade),
    );

    // cloud.rgb = inscattered light, cloud.a = transmittance
    return vec4<f32>(cloud.rgb, cloud.a);
}
