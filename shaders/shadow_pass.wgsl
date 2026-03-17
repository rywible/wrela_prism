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
};

@vertex
fn vs_shadow(input: ShadowVertexIn) -> ShadowVertexOut {
    var output: ShadowVertexOut;
    output.clip_position = shadow_uniforms.light_vp * vec4<f32>(input.position, 1.0);
    return output;
}

@fragment
fn fs_shadow() {
    // All fragments write depth — no alpha discard needed for opaque needle sprays.
}
