// Fullscreen ACES tonemap pass — reads HDR scene_color, outputs LDR.

@group(0) @binding(0)
var hdr_tex: texture_2d<f32>;
@group(0) @binding(1)
var ao_tex: texture_2d<f32>;
@group(0) @binding(2)
var nearest_sampler: sampler;

struct TonemapUniforms {
    screen_size: vec4<f32>,  // width, height, 0, 0
    time_params: vec4<f32>,  // elapsed_secs, 0, 0, 0
    color_grade: vec4<f32>,  // xyz = tint, w = strength (0 = no grading)
};
@group(0) @binding(3)
var<uniform> tu: TonemapUniforms;

struct FullscreenOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) vi: u32) -> FullscreenOut {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    var out: FullscreenOut;
    let pos = positions[vi];
    out.position = vec4<f32>(pos, 0.0, 1.0);
    out.uv = vec2<f32>(pos.x * 0.5 + 0.5, 0.5 - pos.y * 0.5);
    return out;
}

fn aces_tonemap(color: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((color * (a * color + b)) / (color * (c * color + d) + e), vec3(0.0), vec3(1.0));
}

fn luminance(color: vec3<f32>) -> f32 {
    return dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn hable_curve(x: vec3<f32>) -> vec3<f32> {
    let a = 0.15;
    let b = 0.50;
    let c = 0.10;
    let d = 0.20;
    let e = 0.02;
    let f = 0.30;
    return ((x * (a * x + c * b) + d * e) / (x * (a * x + b) + d * f)) - e / f;
}

fn filmic_tonemap(color: vec3<f32>) -> vec3<f32> {
    let white_scale = 1.0 / hable_curve(vec3<f32>(11.2));
    let mapped = hable_curve(max(color - vec3<f32>(0.004), vec3<f32>(0.0))) * white_scale;
    return clamp(mapped, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn sample_auto_exposure() -> f32 {
    let taps = array<vec2<f32>, 9>(
        vec2<f32>(0.18, 0.20),
        vec2<f32>(0.50, 0.18),
        vec2<f32>(0.82, 0.20),
        vec2<f32>(0.22, 0.44),
        vec2<f32>(0.50, 0.45),
        vec2<f32>(0.78, 0.44),
        vec2<f32>(0.20, 0.72),
        vec2<f32>(0.50, 0.74),
        vec2<f32>(0.80, 0.72),
    );
    let max_coord = vec2<i32>(
        max(i32(tu.screen_size.x) - 1, 0),
        max(i32(tu.screen_size.y) - 1, 0),
    );

    var weighted_log_lum = 0.0;
    var weight_sum = 0.0;

    for (var i = 0; i < 9; i++) {
        let uv = taps[i];
        let centered = vec2<f32>((uv.x - 0.5) * (tu.screen_size.x / max(tu.screen_size.y, 1.0)), uv.y - 0.45);
        let weight = exp(-dot(centered, centered) * 4.0);
        let px = clamp(vec2<i32>(uv * vec2<f32>(max_coord)), vec2<i32>(0), max_coord);
        let sample_hdr = max(textureLoad(hdr_tex, px, 0).rgb, vec3<f32>(0.0));
        let sample_lum = min(luminance(sample_hdr), 3.0);
        weighted_log_lum += log2(max(sample_lum, 1e-4)) * weight;
        weight_sum += weight;
    }

    let avg_lum = exp2(weighted_log_lum / max(weight_sum, 1e-4));
    let exposure_target = clamp(0.48 / max(avg_lum, 1e-4), 0.84, 1.60);
    return mix(1.0, exposure_target, 0.75);
}

@fragment
fn fs_tonemap(input: FullscreenOut) -> @location(0) vec4<f32> {
    let pixel = vec2<i32>(input.position.xy);
    let ao = textureLoad(ao_tex, pixel, 0).r;
    let auto_exposure = sample_auto_exposure();
    let hdr = textureLoad(hdr_tex, pixel, 0).rgb * ao * auto_exposure;
    var ldr = filmic_tonemap(hdr);
    let ldr_luma = luminance(ldr);
    let highlight_desat = smoothstep(0.55, 0.98, ldr_luma);
    ldr = mix(ldr, vec3<f32>(ldr_luma), highlight_desat * 0.08);

    // Art direction color grading (applied after tonemap, before vignette)
    let grade_strength = tu.color_grade.w;
    if grade_strength > 0.001 {
        ldr = mix(ldr, ldr * tu.color_grade.xyz, grade_strength);
    }

    // Vignette
    let uv = input.position.xy / tu.screen_size.xy;
    let vignette = 1.0 - 0.3 * dot(uv - 0.5, uv - 0.5);
    ldr *= vignette;

    // Triangular-PDF dither via interleaved gradient noise — breaks 8-bit banding
    let dither_pos = input.position.xy;
    let d1 = fract(52.9829189 * fract(dot(dither_pos + tu.time_params.x * 1.3, vec2(0.06711056, 0.00583715))));
    let d2 = fract(52.9829189 * fract(dot(dither_pos + tu.time_params.x * 2.7 + 0.5, vec2(0.06711056, 0.00583715))));
    ldr += (d1 + d2 - 1.0) * (1.0 / 255.0);

    return vec4<f32>(ldr, 1.0);
}
