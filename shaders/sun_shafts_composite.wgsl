struct SunShaftUniforms {
    sun_screen: vec4<f32>,
    sun_color: vec4<f32>,
    shaft_params: vec4<f32>,
    screen_size: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> u: SunShaftUniforms;
@group(0) @binding(1)
var shaft_tex: texture_2d<f32>;
@group(0) @binding(2)
var linear_sampler: sampler;

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

@fragment
fn fs_composite(input: FullscreenOut) -> @location(0) vec4<f32> {
    let shaft = textureSample(shaft_tex, linear_sampler, input.uv).rgb;
    let haze = 0.4 + u.sun_screen.w * 0.8;
    let shafts = shaft * u.sun_color.rgb * u.sun_color.w * (0.14 + haze * 0.12);
    return vec4<f32>(shafts, 1.0);
}
