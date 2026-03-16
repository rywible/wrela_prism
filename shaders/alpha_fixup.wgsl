// Clears vis_buffer to 0 for pixels where scene_color alpha is 0 (failed alpha test).
// This allows the sky pass to fill those pixels with sky color.

@group(0) @binding(0) var scene_color: texture_2d<f32>;
@group(0) @binding(1) var vis_buffer: texture_storage_2d<r32uint, write>;

@compute @workgroup_size(8, 8)
fn alpha_fixup(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(scene_color);
    if gid.x >= dims.x || gid.y >= dims.y {
        return;
    }
    let alpha = textureLoad(scene_color, vec2<i32>(gid.xy), 0).a;
    if alpha < 0.01 {
        textureStore(vis_buffer, gid.xy, vec4<u32>(0u));
    }
}
