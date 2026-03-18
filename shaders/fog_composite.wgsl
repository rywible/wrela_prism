// Froxel volumetric fog — composite pass (fullscreen fragment).
//
// Reads scene depth to determine froxel slice, samples integrated fog volume,
// and applies: output = scene_color * transmittance + inscatter

struct FogCompositeUniforms {
    grid_params: vec4<f32>,   // grid_w, grid_h, grid_d, far_plane
    near_plane: vec4<f32>,    // x = near, y/z/w = unused
};

@group(0) @binding(0) var<uniform> u: FogCompositeUniforms;
@group(0) @binding(1) var integrated_fog: texture_3d<f32>;
@group(0) @binding(2) var fog_sampler: sampler;
@group(0) @binding(3) var depth_tex: texture_depth_2d;

struct FullscreenOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_fog_composite(@builtin(vertex_index) vid: u32) -> FullscreenOutput {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    let p = positions[vid];
    var out: FullscreenOutput;
    out.position = vec4<f32>(p, 0.0, 1.0);
    out.uv = vec2<f32>(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
    return out;
}

fn depth_to_linear(d: f32) -> f32 {
    let near = u.near_plane.x;
    // Reversed-Z infinite projection: z = near / depth_buffer_value
    // When depth_buffer=1.0 → linear=near, depth_buffer→0 → linear→infinity
    if d < 0.000001 {
        return u.grid_params.w; // far plane for sky pixels
    }
    return near / d;
}

fn depth_to_slice(linear_depth: f32) -> f32 {
    let near = u.near_plane.x;
    let far = u.grid_params.w;
    let num_slices = u.grid_params.z;
    let clamped = clamp(linear_depth, near, far);
    return num_slices * log(clamped / near) / log(far / near);
}

@fragment
fn fs_fog_composite(input: FullscreenOutput) -> @location(0) vec4<f32> {
    let pixel = vec2<i32>(input.position.xy);

    // Read scene depth (reversed-Z)
    let raw_depth = textureLoad(depth_tex, pixel, 0);
    let linear_depth = depth_to_linear(raw_depth);

    // Convert to froxel slice coordinate
    let slice = depth_to_slice(linear_depth);
    let num_slices = u.grid_params.z;

    // 3D UVW for the integrated fog volume
    let grid_w = u.grid_params.x;
    let grid_h = u.grid_params.y;
    let uvw = vec3<f32>(
        input.uv.x,
        input.uv.y,
        clamp(slice / num_slices, 0.0, 1.0 - 0.5 / num_slices),
    );

    let fog = textureSampleLevel(integrated_fog, fog_sampler, uvw, 0.0);
    let inscatter = fog.rgb;
    let transmittance = fog.a;

    // output.rgb = inscatter (pre-multiplied), output.a = transmittance
    // The blend state will compute: scene_color * transmittance + inscatter
    return vec4<f32>(inscatter, transmittance);
}
