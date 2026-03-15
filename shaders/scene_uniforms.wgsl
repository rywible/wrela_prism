// Uniform struct for the G-buffer raster pass.

struct GBufferUniforms {
    view_proj: mat4x4<f32>,
    time_params: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> scene: GBufferUniforms;
