// Copy Depth32Float → R32Float for HZB mip-0.
// Required because texture_depth_2d cannot be bound as texture_2d<f32>.

@group(0) @binding(0) var depth_src: texture_depth_2d;
@group(0) @binding(1) var depth_dst: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn hzb_depth_copy(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(depth_dst);
    if gid.x >= size.x || gid.y >= size.y { return; }

    let depth = textureLoad(depth_src, vec2<i32>(gid.xy), 0);
    textureStore(depth_dst, gid.xy, vec4<f32>(depth, 0.0, 0.0, 0.0));
}
