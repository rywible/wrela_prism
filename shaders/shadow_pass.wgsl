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

const MATERIAL_FOLIAGE: u32 = 1u;

struct ShadowVertexOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) alpha: f32,
};

@vertex
fn vs_shadow(input: ShadowVertexIn) -> ShadowVertexOut {
    // Unpack alpha from semantic_channels for foliage only (bits 0-7: 0-255 → 0.0-1.0).
    // Other materials use byte 0 for art-direction data, so default to fully opaque.
    var alpha = 1.0;
    if input.material == MATERIAL_FOLIAGE {
        let raw_alpha = input.semantic_channels & 0xFFu;
        alpha = select(f32(raw_alpha) / 255.0, 1.0, input.semantic_channels == 0u);
    }

    var output: ShadowVertexOut;
    output.clip_position = shadow_uniforms.light_vp * vec4<f32>(input.position, 1.0);
    output.alpha = alpha;
    return output;
}

@fragment
fn fs_shadow(input: ShadowVertexOut) {
    // Alpha test: discard transparent fragments for correct foliage shadow casting
    if input.alpha < 0.5 {
        discard;
    }
}
