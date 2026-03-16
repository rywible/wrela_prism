struct ShadowUniforms {
    light_vp: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> shadow_uniforms: ShadowUniforms;

struct ShadowVertexIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) material: u32,
    @location(3) feature_id: u32,
    @location(4) uv: vec2<f32>,
    @location(5) ao: f32,
    @location(6) semantic_channels: u32,
};

struct ShadowVertexOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) world_position: vec3<f32>,
    @location(2) @interpolate(flat) material: u32,
};

@vertex
fn vs_shadow(input: ShadowVertexIn) -> ShadowVertexOut {
    var output: ShadowVertexOut;
    output.clip_position = shadow_uniforms.light_vp * vec4<f32>(input.position, 1.0);
    output.uv = input.uv;
    output.world_position = input.position;
    output.material = input.material;
    return output;
}

fn foliage_alpha_mask_shadow(uv: vec2<f32>, position: vec3<f32>) -> f32 {
    let centered_x = abs(uv.x * 2.0 - 1.0);
    let stem_profile = smoothstep(0.01, 0.08, uv.y) * (1.0 - smoothstep(0.93, 1.0, uv.y));
    let taper = mix(0.44, 0.09, smoothstep(0.04, 0.98, uv.y));
    let width_noise =
        0.055 * sin(uv.y * 10.0 + position.x * 5.0)
        + 0.030 * sin(uv.y * 21.0 + position.z * 8.0)
        + 0.018 * sin(uv.y * 37.0 + position.y * 4.0);
    let outer_width = max(taper + width_noise, 0.06);
    let outer = 1.0 - smoothstep(outer_width, outer_width + 0.34, centered_x);
    let lobe_shift =
        0.10 * sin(uv.y * 7.0 + position.x * 4.0)
        + 0.04 * sin(uv.y * 15.0 + position.z * 6.0);
    let inner = 1.0
        - smoothstep(
            outer_width * 0.56,
            outer_width * 0.56 + 0.24,
            abs((uv.x * 2.0 - 1.0) * 0.82 - lobe_shift),
        );
    let tip_cluster = 1.0 - smoothstep(0.76, 1.0, uv.y) * 0.28;
    let alpha = max(outer * 0.74, inner) * stem_profile * tip_cluster;
    return smoothstep(0.06, 0.90, alpha);
}

@fragment
fn fs_shadow(input: ShadowVertexOut) {
    if input.material == 1u {
        let alpha = foliage_alpha_mask_shadow(input.uv, input.world_position);
        if alpha < 0.3 {
            discard;
        }
    }
}
