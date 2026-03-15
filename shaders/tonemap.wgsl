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

@fragment
fn fs_tonemap(input: FullscreenOut) -> @location(0) vec4<f32> {
    let pixel = vec2<i32>(input.position.xy);
    let ao = textureLoad(ao_tex, pixel, 0).r;
    let hdr = textureLoad(hdr_tex, pixel, 0).rgb * ao;
    var ldr = aces_tonemap(hdr);

    // Vignette
    let uv = input.position.xy / tu.screen_size.xy;
    let vignette = 1.0 - 0.3 * dot(uv - 0.5, uv - 0.5);
    ldr *= vignette;

    // Film grain
    let grain = fract(sin(dot(uv * tu.time_params.x, vec2(12.9898, 78.233))) * 43758.5453);
    ldr += (grain - 0.5) * 0.015;

    return vec4<f32>(ldr, 1.0);
}
