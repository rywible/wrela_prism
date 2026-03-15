// Copy Depth32Float → R32Float via a fullscreen compute shader.
// Direct copy_texture_to_texture doesn't work between these formats.

@group(0) @binding(0) var depth_src: texture_depth_2d;
@group(0) @binding(1) var depth_dst: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn depth_copy(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(depth_dst);
    if gid.x >= dims.x || gid.y >= dims.y {
        return;
    }
    let depth = textureLoad(depth_src, gid.xy, 0);
    textureStore(depth_dst, gid.xy, vec4<f32>(depth, 0.0, 0.0, 0.0));
}
