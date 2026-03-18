// Froxel volumetric fog — integration pass (compute).
//
// Marches front-to-back through froxel slices for each pixel column,
// accumulating in-scattering and transmittance.
// Output: Rgba16Float 3D texture (rgb = inscatter, a = transmittance)

struct FogIntegrateUniforms {
    grid_params: vec4<f32>,   // grid_w, grid_h, grid_d, far_plane
    near_plane: vec4<f32>,    // x = near, y/z/w = unused
};

@group(0) @binding(0) var<uniform> u: FogIntegrateUniforms;
@group(0) @binding(1) var scatter_volume: texture_3d<f32>;
@group(0) @binding(2) var integrated_volume: texture_storage_3d<rgba16float, write>;

fn slice_to_depth(slice: f32) -> f32 {
    let near = u.near_plane.x;
    let far = u.grid_params.w;
    let num_slices = u.grid_params.z;
    return near * pow(far / near, slice / num_slices);
}

@compute @workgroup_size(8, 8, 1)
fn fog_integrate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let grid_w = u32(u.grid_params.x);
    let grid_h = u32(u.grid_params.y);
    let grid_d = u32(u.grid_params.z);

    // Each thread processes one pixel column (x, y) through all depth slices
    if gid.x >= grid_w || gid.y >= grid_h {
        return;
    }

    var transmittance = 1.0;
    var inscatter = vec3<f32>(0.0);

    for (var z = 0u; z < grid_d; z++) {
        let texel = vec3<u32>(gid.x, gid.y, z);
        let froxel = textureLoad(scatter_volume, texel, 0);

        let scattering = froxel.rgb;
        let extinction = froxel.a;

        // Compute slice thickness (difference between consecutive slice depths)
        let depth_near = slice_to_depth(f32(z));
        let depth_far = slice_to_depth(f32(z) + 1.0);
        let slice_thickness = depth_far - depth_near;

        // Beer's law: transmittance through this slice
        let slice_transmittance = exp(-extinction * slice_thickness);

        // Integrate in-scattering with energy-conserving formula
        // Using the exact integral: S * (1 - e^(-sigma * dt)) / sigma
        // This is more accurate than the simple Euler step for thick slices
        let scatter_integral = select(
            scattering * slice_thickness,
            scattering * (1.0 - slice_transmittance) / max(extinction, 0.00001),
            extinction > 0.00001
        );

        inscatter += transmittance * scatter_integral;
        transmittance *= slice_transmittance;

        // Store accumulated result at this slice
        textureStore(integrated_volume, texel, vec4<f32>(inscatter, transmittance));
    }
}
