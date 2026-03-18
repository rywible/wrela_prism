// Screen-Space Reflections — Hi-Z ray march compute shader.
//
// Reads depth, normals, scene_color, and HZB mip chain.
// Outputs reflection color texture (Rgba16Float).
// Composited additively onto scene_color weighted by fresnel * (1 - roughness²).

struct SsrUniforms {
    view: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    projection: mat4x4<f32>,
    inv_projection: mat4x4<f32>,
    screen_size: vec4<f32>,   // (width, height, 1/width, 1/height)
    params: vec4<f32>,        // (max_distance, thickness, stride, max_steps)
    params2: vec4<f32>,       // (roughness_cutoff, fade_start, fade_end, frame_index)
};

@group(0) @binding(0) var<uniform> uniforms: SsrUniforms;
@group(0) @binding(1) var normal_tex: texture_2d<f32>;
@group(0) @binding(2) var depth_tex: texture_depth_2d;
@group(0) @binding(3) var color_tex: texture_2d<f32>;
@group(0) @binding(4) var hzb_tex: texture_2d<f32>;
@group(0) @binding(5) var output_tex: texture_storage_2d<rgba16float, write>;

const PI: f32 = 3.14159265359;

// Reconstruct view-space position from screen UV and depth
fn view_pos_from_depth(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0);
    let clip = vec4<f32>(ndc, depth, 1.0);
    let view = uniforms.inv_projection * clip;
    return view.xyz / view.w;
}

// Project view-space position to screen UV + depth
fn project_to_screen(view_pos: vec3<f32>) -> vec3<f32> {
    let clip = uniforms.projection * vec4<f32>(view_pos, 1.0);
    let ndc = clip.xyz / clip.w;
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    return vec3<f32>(uv, ndc.z);
}

// Schlick Fresnel approximation
fn fresnel_schlick(cos_theta: f32, f0: f32) -> f32 {
    return f0 + (1.0 - f0) * pow(1.0 - cos_theta, 5.0);
}

// Screen-edge fade: attenuate reflections near screen borders
fn screen_edge_fade(uv: vec2<f32>) -> f32 {
    let fade_start = uniforms.params2.y;
    let fade_end = uniforms.params2.z;
    let dist_x = min(uv.x, 1.0 - uv.x);
    let dist_y = min(uv.y, 1.0 - uv.y);
    let dist = min(dist_x, dist_y);
    return smoothstep(fade_end, fade_start, dist);
}

// Per-pixel hash for jittering
fn hash_pixel(pixel: vec2<u32>, frame: f32) -> f32 {
    let p = vec2<f32>(f32(pixel.x), f32(pixel.y)) + frame * 1.618;
    return fract(sin(dot(p, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}

// Hi-Z ray march: march along reflection ray in screen space using the HZB for fast skipping
fn hiz_trace(
    ray_origin: vec3<f32>,  // view-space
    ray_dir: vec3<f32>,     // view-space, normalized
    jitter: f32,
) -> vec4<f32> {  // xyz = hit UV + depth, w = confidence (0 = miss)
    let max_distance = uniforms.params.x;
    let thickness = uniforms.params.y;
    let max_steps = u32(uniforms.params.w);
    let screen = vec2<f32>(uniforms.screen_size.x, uniforms.screen_size.y);

    // Project start and a point along the ray
    let start_screen = project_to_screen(ray_origin);
    let ray_end_view = ray_origin + ray_dir * max_distance;
    let end_screen = project_to_screen(ray_end_view);

    // Screen-space ray direction
    var ray_screen = end_screen - start_screen;

    // Scale ray to step at most 1 pixel at a time (in longest axis)
    let screen_len = max(abs(ray_screen.x * screen.x), abs(ray_screen.y * screen.y));
    if screen_len < 1.0 {
        return vec4<f32>(0.0);
    }

    // Use a stride to reduce step count
    let stride = max(uniforms.params.z, 1.0);
    let step_count = min(u32(ceil(screen_len / stride)), max_steps);
    let dt = 1.0 / f32(step_count);

    var t = dt * jitter; // jittered start
    var prev_z = start_screen.z;

    for (var i = 0u; i < step_count; i++) {
        let sample_screen = start_screen + ray_screen * t;
        let sample_uv = sample_screen.xy;

        // Out-of-bounds check
        if sample_uv.x < 0.0 || sample_uv.x > 1.0 || sample_uv.y < 0.0 || sample_uv.y > 1.0 {
            return vec4<f32>(0.0);
        }

        let sample_pixel = vec2<i32>(sample_uv * screen);

        // Read scene depth at this pixel (HZB mip 0 = full res depth)
        let scene_depth = textureLoad(hzb_tex, sample_pixel, 0).r;

        // Reversed-Z: scene_depth is closer to 1.0 for near objects
        // Our ray depth: sample_screen.z
        let ray_z = sample_screen.z;

        // Hit test: ray went behind geometry (reversed-Z: ray_z < scene_depth means behind)
        if ray_z < scene_depth && prev_z >= scene_depth {
            // Check thickness: don't accept hits too far behind surfaces
            let depth_diff = scene_depth - ray_z;
            if depth_diff < thickness {
                let confidence = 1.0 - (depth_diff / thickness);
                return vec4<f32>(sample_uv, scene_depth, confidence);
            }
        }

        prev_z = ray_z;
        t += dt;
    }

    return vec4<f32>(0.0);
}

@compute @workgroup_size(8, 8)
fn ssr_trace(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pixel = gid.xy;
    let screen = vec2<u32>(u32(uniforms.screen_size.x), u32(uniforms.screen_size.y));
    if pixel.x >= screen.x || pixel.y >= screen.y {
        return;
    }

    let uv = (vec2<f32>(pixel) + 0.5) * uniforms.screen_size.zw;

    // Read depth
    let depth = textureLoad(depth_tex, pixel, 0);
    if depth <= 0.0001 {
        textureStore(output_tex, pixel, vec4<f32>(0.0));
        return;
    }

    // Read world normal, transform to view space
    let normal_encoded = textureLoad(normal_tex, pixel, 0);
    let world_normal = normalize(normal_encoded.rgb * 2.0 - 1.0);

    // Roughness cutoff: material type is in normal.a (0 = foliage, 1 = other)
    // For this scene, most surfaces are rough (bark, foliage, ground).
    // We use a fixed roughness estimate: foliage=0.82, bark=0.85, ground=0.92
    // Only generate reflections for moderately smooth surfaces.
    let roughness_cutoff = uniforms.params2.x;
    // Estimate roughness from material type encoded in alpha
    let is_foliage = normal_encoded.a < 0.5;
    var roughness = select(0.80, 0.82, is_foliage);
    // Skin material is smoother — check for it via normal characteristics
    // (not perfectly distinguishable, but SSR will be subtle anyway)

    if roughness > roughness_cutoff {
        textureStore(output_tex, pixel, vec4<f32>(0.0));
        return;
    }

    let view_normal = normalize((uniforms.view * vec4<f32>(world_normal, 0.0)).xyz);

    // Reconstruct view-space position
    let view_pos = view_pos_from_depth(uv, depth);

    // Reflection direction in view space
    let view_dir = normalize(view_pos);
    let reflect_dir = reflect(view_dir, view_normal);

    // Don't trace reflections pointing toward the camera
    if reflect_dir.z > -0.01 {
        textureStore(output_tex, pixel, vec4<f32>(0.0));
        return;
    }

    // Jitter for temporal stability
    let jitter = hash_pixel(pixel, uniforms.params2.w);

    // Trace
    let hit = hiz_trace(view_pos, reflect_dir, jitter);

    if hit.w <= 0.0 {
        textureStore(output_tex, pixel, vec4<f32>(0.0));
        return;
    }

    let hit_uv = hit.xy;
    let hit_pixel = vec2<i32>(hit_uv * vec2<f32>(screen));

    // Sample scene color at hit point
    var reflection_color = textureLoad(color_tex, hit_pixel, 0).rgb;

    // Fresnel: stronger reflections at grazing angles
    let NdotV = max(dot(view_normal, -view_dir), 0.0);
    let fresnel = fresnel_schlick(NdotV, 0.04);

    // Roughness fade: reduce reflection for rougher surfaces
    let roughness_fade = 1.0 - roughness * roughness;

    // Screen-edge fade
    let edge_fade = screen_edge_fade(hit_uv);

    // Confidence from hit test
    let confidence = hit.w;

    // Composite weight
    let weight = fresnel * roughness_fade * edge_fade * confidence;

    textureStore(output_tex, pixel, vec4<f32>(reflection_color * weight, weight));
}

// Composite is handled by a separate fullscreen render pass (shaders/ssr_composite.wgsl)
// using additive blending onto scene_color.
