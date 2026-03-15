struct SunShaftUniforms {
    sun_screen: vec4<f32>,
    sun_color: vec4<f32>,
    shaft_params: vec4<f32>,
    screen_size: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> u: SunShaftUniforms;
@group(0) @binding(1)
var depth_tex: texture_2d<f32>;

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

fn occlusion_at(uv: vec2<f32>) -> f32 {
    if any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0)) {
        return 0.0;
    }
    let dims = vec2<f32>(textureDimensions(depth_tex));
    let pixel = vec2<i32>(clamp(uv * dims, vec2<f32>(0.0), dims - vec2<f32>(1.0)));
    let depth = textureLoad(depth_tex, pixel, 0).r;
    let sky = select(0.0, 1.0, depth >= 0.9995);
    let sun_distance = length(uv - u.sun_screen.xy);
    let falloff = smoothstep(0.95, 0.05, sun_distance);
    return sky * falloff * u.sun_screen.z;
}

@fragment
fn fs_occlusion(input: FullscreenOut) -> @location(0) vec4<f32> {
    let intensity = u.shaft_params.x;
    let decay = clamp(u.shaft_params.y, 0.0, 0.999);
    let density = u.shaft_params.z;
    let delta = (u.sun_screen.xy - input.uv) * (density / 16.0);
    var coord = input.uv;
    var illumination_decay = 1.0;
    var scattered = 0.0;
    for (var i = 0u; i < 16u; i++) {
        coord += delta;
        let sample = occlusion_at(coord);
        scattered += sample * illumination_decay;
        illumination_decay *= decay;
    }

    let near_sun = smoothstep(1.15, 0.0, length(input.uv - u.sun_screen.xy));
    let shaft = scattered * intensity * near_sun;
    return vec4<f32>(vec3<f32>(shaft), 1.0);
}
