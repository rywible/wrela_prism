@group(0) @binding(0) var ssr_tex: texture_2d<f32>;

struct FullscreenOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_composite(@builtin(vertex_index) vi: u32) -> FullscreenOut {
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
    let pixel = vec2<i32>(input.position.xy);
    let reflection = textureLoad(ssr_tex, pixel, 0);
    return vec4<f32>(reflection.rgb, 1.0);
}
