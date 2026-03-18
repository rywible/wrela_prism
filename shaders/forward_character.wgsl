// Forward-rendered character capsule with Lambert shading + height fog.
//
// Bind groups:
//   group(0) binding(0) — FrameUniforms (scene-wide, same layout as material_resolve)
//   group(1) binding(0) — ModelUniforms (per-character world transform)

// ---------------------------------------------------------------------------
// Scene uniforms — mirrors the FrameUniforms in material_resolve.wgsl
// ---------------------------------------------------------------------------

struct FrameUniforms {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    camera_position: vec4<f32>,
    sun_direction: vec4<f32>,
    sun_color: vec4<f32>,
    fog_color: vec4<f32>,
    sky_zenith: vec4<f32>,
    sky_horizon: vec4<f32>,
    fog_params: vec4<f32>,          // (density, height_falloff, start, end)
    lighting_params: vec4<f32>,     // (exposure, tonemap_strength, contact_shadow, gi_intensity)
    light_vp: mat4x4<f32>,
    ambient_up: vec4<f32>,
    ambient_down: vec4<f32>,
    ambient_right: vec4<f32>,
    ambient_left: vec4<f32>,
    ambient_front: vec4<f32>,
    ambient_back: vec4<f32>,
    atmosphere_params: vec4<f32>,
    shaft_params: vec4<f32>,
    time_params: vec4<f32>,
    screen_size: vec4<f32>,
    wind_params: vec4<f32>,
    light_vp_1: mat4x4<f32>,
    light_vp_2: mat4x4<f32>,
    light_vp_3: mat4x4<f32>,
    cascade_splits: vec4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: FrameUniforms;

// ---------------------------------------------------------------------------
// Per-character model transform
// ---------------------------------------------------------------------------

struct ModelUniforms {
    model: mat4x4<f32>,
};

@group(1) @binding(0) var<uniform> model: ModelUniforms;

// ---------------------------------------------------------------------------
// Vertex stage
// ---------------------------------------------------------------------------

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    let world_pos = (model.model * vec4<f32>(in.position, 1.0)).xyz;

    // Normal transform: upper-left 3x3 of the model matrix.
    // For a uniform-scale capsule this is sufficient (no need for inverse-transpose).
    let world_normal = normalize((model.model * vec4<f32>(in.normal, 0.0)).xyz);

    var out: VertexOutput;
    out.clip_position = uniforms.view_proj * vec4<f32>(world_pos, 1.0);
    out.world_position = world_pos;
    out.world_normal = world_normal;
    return out;
}

// ---------------------------------------------------------------------------
// Fragment stage — Lambert + simple hemisphere ambient + height fog
// ---------------------------------------------------------------------------

// Neutral clay albedo for the placeholder capsule.
const BASE_ALBEDO: vec3<f32> = vec3<f32>(0.5, 0.45, 0.4);

fn height_fog(world_pos: vec3<f32>, dist: f32) -> f32 {
    let density  = uniforms.fog_params.x;
    let falloff  = uniforms.fog_params.y;
    let fog_start = uniforms.fog_params.z;
    let fog_end   = uniforms.fog_params.w;

    let distance_factor = clamp((dist - fog_start) / max(fog_end - fog_start, 0.001), 0.0, 1.0);
    let distance_fog = 1.0 - exp(-pow(distance_factor, 2.2) * density * 1500.0);
    let ground_mist = exp(-max(world_pos.y, 0.0) * falloff);
    let fog = clamp(distance_fog * (0.24 + ground_mist * 0.76), 0.0, 1.0);

    return fog;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let N = normalize(in.world_normal);
    let L = normalize(uniforms.sun_direction.xyz);

    // Lambert diffuse
    let NdotL = max(dot(N, L), 0.0);
    let diffuse = BASE_ALBEDO * uniforms.sun_color.rgb * NdotL;

    // Simple hemisphere ambient: blend between ambient_down and ambient_up by
    // how much the normal points upward, giving soft sky/ground fill light.
    let hemi_t = N.y * 0.5 + 0.5;
    let ambient = BASE_ALBEDO * mix(uniforms.ambient_down.rgb, uniforms.ambient_up.rgb, hemi_t);

    var color = diffuse + ambient;

    // Height fog — same curve as the main resolve pass (without per-material tweaks).
    let camera_pos = uniforms.camera_position.xyz;
    let cam_dist = length(in.world_position - camera_pos);
    let fog_factor = height_fog(in.world_position, cam_dist);
    color = mix(color, uniforms.fog_color.rgb * uniforms.sun_color.rgb, fog_factor);

    // Exposure is applied once in the tonemap pass — output raw HDR here.

    return vec4<f32>(color, 1.0);
}
