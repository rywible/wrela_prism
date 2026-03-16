// SSGI additive composite pass — blends indirect illumination onto scene color.

@group(0) @binding(0) var gi_texture: texture_2d<f32>;

struct FullscreenOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

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

@fragment
fn fs_composite(input: FullscreenOutput) -> @location(0) vec4<f32> {
    let pixel = vec2<u32>(input.position.xy);
    let gi_color = textureLoad(gi_texture, pixel, 0).rgb;
    return vec4<f32>(gi_color, 1.0);
}
