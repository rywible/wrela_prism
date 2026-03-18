// Hierarchical Z-buffer generation (compute shader).
//
// Reversed-Z: near=1.0, far=0.0. Conservative (farthest) depth is the
// minimum value. Reads previous mip and writes min depth to current mip.

@group(0) @binding(0) var src_depth: texture_2d<f32>;
@group(0) @binding(1) var dst_depth: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn hzb_build(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dst_size = textureDimensions(dst_depth);
    if gid.x >= dst_size.x || gid.y >= dst_size.y {
        return;
    }

    let src_coord = gid.xy * 2u;

    // Sample 2x2 block from source and take the minimum (farthest depth in reversed-Z)
    let d00 = textureLoad(src_depth, src_coord + vec2<u32>(0u, 0u), 0).r;
    let d10 = textureLoad(src_depth, src_coord + vec2<u32>(1u, 0u), 0).r;
    let d01 = textureLoad(src_depth, src_coord + vec2<u32>(0u, 1u), 0).r;
    let d11 = textureLoad(src_depth, src_coord + vec2<u32>(1u, 1u), 0).r;

    let min_depth = min(min(d00, d10), min(d01, d11));
    textureStore(dst_depth, gid.xy, vec4<f32>(min_depth, 0.0, 0.0, 0.0));
}
