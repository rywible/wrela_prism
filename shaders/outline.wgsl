// Screen-space outline pass.
//
// Sobel edge detection on depth + normals, composited onto the scene
// with alpha blending. Early-out when outline_strength == 0.

struct ArtDirectionUniforms {
    silhouette_inflate: f32,
    normal_smoothing: f32,
    detail_suppression: f32,
    color_discipline: f32,

    shading_bands: f32,
    band_softness: f32,
    specular_flattening: f32,
    rim_light_strength: f32,

    shadow_color_shift: vec4<f32>,
    rim_light_color: vec4<f32>,

    palette_saturation: f32,
    palette_hue_shift: f32,
    palette_value_contrast: f32,
    foliage_transmission_boost: f32,

    outline_strength: f32,
    outline_thickness: f32,
    outline_depth_sensitivity: f32,
    outline_normal_sensitivity: f32,

    outline_color: vec4<f32>,
    color_grade_tint: vec4<f32>,
    bloom_tint: vec4<f32>,

    detail_fade_distance: f32,
    skip_bark_texture: f32,
    skip_normal_perturbation: f32,
    foliage_shadow_band_softness: f32,

    wind_amplitude_scale: f32,
    wind_response_scale: f32,
    wind_gust_sharpness: f32,
    _pad4d: f32,
};

@group(0) @binding(0) var<uniform> art: ArtDirectionUniforms;

@group(1) @binding(0) var depth_tex: texture_2d<f32>;
@group(1) @binding(1) var normal_tex: texture_2d<f32>;
@group(1) @binding(2) var outline_sampler: sampler;

// -- Fullscreen triangle vertex shader --

struct FullscreenOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> FullscreenOutput {
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

// -- Fragment shader --

@fragment
fn fs_outline(input: FullscreenOutput) -> @location(0) vec4<f32> {
    let dims = vec2<f32>(textureDimensions(depth_tex));
    let texel = 1.0 / dims;
    let thickness = max(art.outline_thickness, 1.0);
    let step = texel * thickness;

    let pixel = vec2<i32>(input.position.xy);

    // Sample depth in a 3x3 cross pattern
    let d_c  = textureLoad(depth_tex, pixel, 0).r;
    let d_l  = textureLoad(depth_tex, pixel + vec2<i32>(-1, 0) * i32(thickness), 0).r;
    let d_r  = textureLoad(depth_tex, pixel + vec2<i32>( 1, 0) * i32(thickness), 0).r;
    let d_u  = textureLoad(depth_tex, pixel + vec2<i32>( 0,-1) * i32(thickness), 0).r;
    let d_d  = textureLoad(depth_tex, pixel + vec2<i32>( 0, 1) * i32(thickness), 0).r;

    // Sobel-like depth discontinuity
    let depth_edge = abs(d_l - d_r) + abs(d_u - d_d);
    // Scale by inverse depth for perspective-correct edges
    let depth_scale = 1.0 / max(d_c, 0.001);
    let depth_magnitude = depth_edge * depth_scale * art.outline_depth_sensitivity * 50.0;

    // Sample normals in a 3x3 cross pattern
    let n_c = textureLoad(normal_tex, pixel, 0).rgb * 2.0 - 1.0;
    let n_l = textureLoad(normal_tex, pixel + vec2<i32>(-1, 0) * i32(thickness), 0).rgb * 2.0 - 1.0;
    let n_r = textureLoad(normal_tex, pixel + vec2<i32>( 1, 0) * i32(thickness), 0).rgb * 2.0 - 1.0;
    let n_u = textureLoad(normal_tex, pixel + vec2<i32>( 0,-1) * i32(thickness), 0).rgb * 2.0 - 1.0;
    let n_d = textureLoad(normal_tex, pixel + vec2<i32>( 0, 1) * i32(thickness), 0).rgb * 2.0 - 1.0;

    // Normal discontinuity — threshold to ignore fine texture, only catch major creases
    let normal_edge_h = length(n_l - n_r);
    let normal_edge_v = length(n_u - n_d);
    let raw_normal_edge = (normal_edge_h + normal_edge_v);
    // Threshold: only normals with large angular change (>0.5) contribute
    let normal_magnitude = smoothstep(0.5, 1.2, raw_normal_edge) * art.outline_normal_sensitivity;

    // Combine edges — depth dominates for silhouettes, normals add crease lines
    let edge = clamp(max(depth_magnitude, normal_magnitude), 0.0, 1.0);

    // Soften outer edge for ink-like falloff
    let ink_edge = smoothstep(0.1, 0.6, edge);

    // Output with alpha for blending
    return vec4<f32>(art.outline_color.rgb, ink_edge * art.outline_strength);
}
