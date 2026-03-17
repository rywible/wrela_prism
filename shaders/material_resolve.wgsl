// Fullscreen material resolve pass.
//
// Reads the visibility buffer, reconstructs attributes via barycentric interpolation,
// applies Cook-Torrance GGX PBR lighting with procedural materials,
// and outputs HDR color + world normals.

const PI: f32 = 3.14159265359;
const TWO_PI: f32 = 6.28318530718;

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
    light_vp: mat4x4<f32>,         // cascade 0
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
    light_vp_1: mat4x4<f32>,       // cascade 1
    light_vp_2: mat4x4<f32>,       // cascade 2
    light_vp_3: mat4x4<f32>,       // cascade 3
    cascade_splits: vec4<f32>,      // view-space split distances
};

@group(0) @binding(0) var<uniform> uniforms: FrameUniforms;

struct BarkParams {
    trunk_radius_base: f32, trunk_radius_tip: f32,
    bark_thickness_base: f32, bark_thickness_tip: f32,
    trunk_height: f32, stiffness_ratio: f32,
    fiber_density: f32, seed: u32,
};
@group(0) @binding(1) var<uniform> bark: BarkParams;

struct Vertex {
    pos_x: f32, pos_y: f32, pos_z: f32,
    nor_x: f32, nor_y: f32, nor_z: f32,
    material: u32,
    feature_id: u32,
    uv_x: f32, uv_y: f32,
    ao: f32,
    semantic_channels: u32,
};

struct MeshletDescriptor {
    vertex_offset: u32,
    vertex_count: u32,
    triangle_offset: u32,
    triangle_count: u32,
};

@group(1) @binding(0) var<storage, read> all_vertices: array<Vertex>;
@group(1) @binding(1) var<storage, read> meshlet_vertex_indices: array<u32>;
@group(1) @binding(2) var<storage, read> meshlet_triangle_indices: array<u32>;
@group(1) @binding(3) var<storage, read> meshlet_descs: array<MeshletDescriptor>;

// Visibility buffer (R32Uint texture)
@group(2) @binding(0) var vis_texture: texture_2d<u32>;

// Shadow map (2D array for CSM cascades)
@group(2) @binding(1) var shadow_map: texture_depth_2d_array;
@group(2) @binding(2) var shadow_sampler: sampler_comparison;

// Baked bark textures
@group(2) @binding(3) var bark_albedo_rough: texture_2d<f32>;
@group(2) @binding(4) var bark_normal_ao: texture_2d<f32>;
@group(2) @binding(5) var bark_sampler: sampler;
@group(2) @binding(6) var bark_height: texture_2d<f32>;

// Depth buffer (R32Float, for contact shadows)
@group(2) @binding(7) var depth_float: texture_2d<f32>;

// Sky-view LUT for aerial perspective (must match sky_pass.wgsl parameterization)
@group(2) @binding(8) var sky_view_lut: texture_2d<f32>;
@group(2) @binding(9) var sky_lut_sampler: sampler;

// -- Art Direction Uniforms (bind group 3) --

struct ArtDirectionUniforms {
    // Tier 1: Geometry Deformation
    silhouette_inflate: f32,
    normal_smoothing: f32,
    detail_suppression: f32,
    color_discipline: f32,

    // Tier 2: Shading Transforms
    shading_bands: f32,
    band_softness: f32,
    specular_flattening: f32,
    rim_light_strength: f32,

    shadow_color_shift: vec4<f32>,
    rim_light_color: vec4<f32>,

    palette_saturation: f32,
    palette_hue_shift: f32,
    palette_value_contrast: f32,
    transmission_boost: f32,

    // Tier 3: Post-Processing
    outline_strength: f32,
    outline_thickness: f32,
    outline_depth_sensitivity: f32,
    outline_normal_sensitivity: f32,

    outline_color: vec4<f32>,
    color_grade_tint: vec4<f32>,
    bloom_tint: vec4<f32>,

    detail_fade_distance: f32,
    skip_bark_texture: f32,
    skip_normal_perturbation: f32,
    _pad3c: f32,
};

struct StylePalette {
    shadow: vec4<f32>,
    dark: vec4<f32>,
    mid: vec4<f32>,
    light: vec4<f32>,
    accent: vec4<f32>,
    bark: vec4<f32>,
    foliage: vec4<f32>,
    earth: vec4<f32>,
    sky_tint: vec4<f32>,
};

@group(3) @binding(0) var<uniform> art: ArtDirectionUniforms;
@group(3) @binding(1) var<uniform> palette: StylePalette;

// -- Art Direction Helper Functions --

// Quantize lighting into discrete bands via luminance remapping.
// Preserves hue — only the light/dark relationship is banded.
// Clamps luminance to [0,1] for banding, then rescales, so HDR values
// don't produce blown-out jumps.
// No-op when bands < 0.01.
fn apply_light_banding(color: vec3<f32>, bands: f32, softness: f32) -> vec3<f32> {
    if bands < 0.01 {
        return color;
    }
    let lum = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
    if lum < 0.0001 {
        return color;
    }
    // Clamp to [0,1] for banding, preserve HDR headroom
    let clamped_lum = min(lum, 1.0);
    let scaled = clamped_lum * bands;
    let band_idx = floor(scaled);
    let frac = scaled - band_idx;
    // Soft transition between bands (softness controls width of gradient)
    let t = smoothstep(0.5 - softness, 0.5 + softness, frac);
    let banded = (band_idx + t) / bands;
    // Blend between original and banded luminance
    let final_lum = mix(lum, banded + max(lum - 1.0, 0.0), 0.75);
    return color * (max(final_lum, 0.0) / lum);
}

// Shift shadow regions toward a target color.
// `shift` is the target shadow hue (e.g. cool blue-violet for ufotable).
// Shadow areas blend toward shift; lit areas untouched.
// No-op when shift == vec3(0).
fn apply_shadow_color_shift(color: vec3<f32>, NdotL: f32, shadow: f32, shift: vec3<f32>) -> vec3<f32> {
    if dot(shift, shift) < 0.0001 {
        return color;
    }
    let shadow_factor = 1.0 - NdotL * shadow;
    // Only shift when solidly in shadow — the squared factor keeps lit areas clean
    let shift_amount = shadow_factor * shadow_factor * shadow_factor;
    // Preserve luminance: shift the chrominance, not the brightness
    let lum = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
    let shift_lum = max(dot(shift, vec3<f32>(0.2126, 0.7152, 0.0722)), 0.001);
    let relit_shift = shift * max(lum / shift_lum, 0.0);
    return mix(color, relit_shift, shift_amount * 0.55);
}

// Fresnel-based edge glow. No-op when strength < 0.01.
// Wide, bright anime-style rim light.
fn compute_rim_light(normal: vec3<f32>, view_dir: vec3<f32>, strength: f32, rim_color: vec3<f32>) -> vec3<f32> {
    if strength < 0.01 {
        return vec3<f32>(0.0);
    }
    let NdotV = max(dot(normal, view_dir), 0.0);
    // Wide rim (power 1.5) that's visible across more of the surface
    let fresnel = pow(1.0 - NdotV, 1.5);
    // Boost to make it clearly visible
    return rim_color * fresnel * strength * 0.08;
}

// Palette-based hue shift. Preserves the original value (light/dark) and
// shifts the chrominance toward the palette's material hint + tonal slot.
// This gives ufotable's "curated color family" look without crushing detail.
// No-op when discipline < 0.01.
fn apply_palette_remap(base_color: vec3<f32>, material: u32, discipline: f32) -> vec3<f32> {
    if discipline < 0.01 {
        return base_color;
    }
    let lum = dot(base_color, vec3<f32>(0.2126, 0.7152, 0.0722));
    if lum < 0.001 {
        return base_color;
    }

    // Select tonal slot based on luminance (for hue reference only)
    var tonal: vec3<f32>;
    if lum < 0.12 {
        tonal = palette.shadow.rgb;
    } else if lum < 0.3 {
        tonal = mix(palette.dark.rgb, palette.mid.rgb, (lum - 0.12) / 0.18);
    } else if lum < 0.6 {
        tonal = mix(palette.mid.rgb, palette.light.rgb, (lum - 0.3) / 0.3);
    } else {
        tonal = palette.light.rgb;
    }

    // Select material hint — this preserves material identity
    var hint: vec3<f32>;
    if material == MATERIAL_TRUNK {
        hint = palette.bark.rgb;
    } else if material == MATERIAL_FOLIAGE {
        hint = palette.foliage.rgb;
    } else {
        hint = palette.earth.rgb;
    }

    // Compute target chrominance: blend tonal hue with material hint
    // Ground uses more material hint to stay earthy, trunk/foliage use more tonal
    let hint_weight = select(0.45, 0.70, material == MATERIAL_GROUND);
    let target_chroma = mix(tonal, hint, hint_weight);
    let target_lum = max(dot(target_chroma, vec3<f32>(0.2126, 0.7152, 0.0722)), 0.001);

    // Normalize target to unit luminance, then reapply original luminance
    // This shifts HUE toward palette while preserving VALUE
    let relit = target_chroma * (lum / target_lum);

    // Discipline controls how far we shift toward palette
    return mix(base_color, relit, discipline * 0.85);
}

// Flatten specular contribution. No-op when flattening == 0.
fn apply_specular_flatten(specular: vec3<f32>, flattening: f32) -> vec3<f32> {
    return specular * (1.0 - flattening);
}

// -- Vertex output --

struct FullscreenOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> FullscreenOutput {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    let p = pos[vid];
    var out: FullscreenOutput;
    out.position = vec4<f32>(p, 0.0, 1.0);
    out.uv = p * 0.5 + 0.5;
    out.uv.y = 1.0 - out.uv.y;
    return out;
}

// -- Material Constants --

const MATERIAL_TRUNK: u32 = 0u;
const MATERIAL_FOLIAGE: u32 = 1u;
const MATERIAL_GROUND: u32 = 2u;
const MATERIAL_SKIN: u32 = 3u;

// -- Helpers --

fn load_triangle_index(desc: MeshletDescriptor, tri_idx: u32, vert_in_tri: u32) -> u32 {
    let byte_idx = desc.triangle_offset + tri_idx * 3u + vert_in_tri;
    let word_idx = byte_idx / 4u;
    let byte_offset = byte_idx % 4u;
    return (meshlet_triangle_indices[word_idx] >> (byte_offset * 8u)) & 0xFFu;
}

fn compute_barycentrics(
    screen_pos: vec2<f32>,
    v0: vec4<f32>, v1: vec4<f32>, v2: vec4<f32>,
) -> vec3<f32> {
    let s0 = v0.xy / v0.w;
    let s1 = v1.xy / v1.w;
    let s2 = v2.xy / v2.w;

    let ndc = screen_pos * 2.0 - 1.0;
    let ndc_y_flip = vec2<f32>(ndc.x, -ndc.y);

    let e1 = s1 - s0;
    let e2 = s2 - s0;
    let ep = ndc_y_flip - s0;

    let d11 = dot(e1, e1);
    let d12 = dot(e1, e2);
    let d22 = dot(e2, e2);
    let dp1 = dot(ep, e1);
    let dp2 = dot(ep, e2);

    let inv_denom = 1.0 / (d11 * d22 - d12 * d12);
    let u = (d22 * dp1 - d12 * dp2) * inv_denom;
    let v = (d11 * dp2 - d12 * dp1) * inv_denom;

    let lambda = vec3<f32>(1.0 - u - v, u, v);
    let w_inv = vec3<f32>(1.0 / v0.w, 1.0 / v1.w, 1.0 / v2.w);
    let corrected = lambda * w_inv;
    let sum = corrected.x + corrected.y + corrected.z;

    return corrected / sum;
}

fn six_face_ambient(normal: vec3<f32>) -> vec3<f32> {
    let nx = normal.x;
    let ny = normal.y;
    let nz = normal.z;
    return uniforms.ambient_right.rgb * max(nx, 0.0)
         + uniforms.ambient_left.rgb * max(-nx, 0.0)
         + uniforms.ambient_up.rgb * max(ny, 0.0)
         + uniforms.ambient_down.rgb * max(-ny, 0.0)
         + uniforms.ambient_front.rgb * max(nz, 0.0)
         + uniforms.ambient_back.rgb * max(-nz, 0.0);
}

// Get cascade light VP matrix by index
fn get_cascade_vp(idx: i32) -> mat4x4<f32> {
    if idx == 0 { return uniforms.light_vp; }
    if idx == 1 { return uniforms.light_vp_1; }
    if idx == 2 { return uniforms.light_vp_2; }
    return uniforms.light_vp_3;
}

// 4-tap rotated Poisson disk offsets (normalized, scale by texel size)
const POISSON_DISK: array<vec2<f32>, 4> = array<vec2<f32>, 4>(
    vec2<f32>(-0.94201624, -0.39906216),
    vec2<f32>( 0.94558609, -0.76890725),
    vec2<f32>(-0.09418410, -0.92938870),
    vec2<f32>( 0.34495938,  0.29387760),
);

fn sample_shadow_cascade(world_pos: vec3<f32>, cascade_idx: i32) -> f32 {
    let vp = get_cascade_vp(cascade_idx);
    let light_clip = vp * vec4<f32>(world_pos, 1.0);
    let light_ndc = light_clip.xyz / light_clip.w;
    let shadow_uv = light_ndc.xy * 0.5 + 0.5;
    let shadow_uv_flipped = vec2<f32>(shadow_uv.x, 1.0 - shadow_uv.y);

    if shadow_uv_flipped.x < 0.0 || shadow_uv_flipped.x > 1.0 ||
       shadow_uv_flipped.y < 0.0 || shadow_uv_flipped.y > 1.0 {
        return 1.0;
    }

    let depth = light_ndc.z;
    let texel_size = 1.0 / 2048.0;

    // 4-tap rotated Poisson PCF
    var shadow = 0.0;
    for (var i = 0; i < 4; i++) {
        let offset = POISSON_DISK[i] * texel_size * 1.5;
        shadow += textureSampleCompareLevel(
            shadow_map, shadow_sampler,
            shadow_uv_flipped + offset,
            cascade_idx,
            depth
        );
    }
    return shadow * 0.25;
}

fn sample_shadow(world_pos: vec3<f32>) -> f32 {
    // Compute view-space depth for cascade selection
    let view_pos = uniforms.view_proj * vec4<f32>(world_pos, 1.0);
    let view_depth = view_pos.w; // w = -view_z for perspective RH

    // Select cascade based on view-space depth
    var cascade_idx = 3;
    if view_depth < uniforms.cascade_splits.x {
        cascade_idx = 0;
    } else if view_depth < uniforms.cascade_splits.y {
        cascade_idx = 1;
    } else if view_depth < uniforms.cascade_splits.z {
        cascade_idx = 2;
    }

    let shadow = sample_shadow_cascade(world_pos, cascade_idx);

    // Blend at cascade boundaries to avoid hard seams
    let blend_range = 0.1; // 10% of cascade range for blending
    var blend_factor = 0.0;
    if cascade_idx < 3 {
        var split_dist: f32;
        if cascade_idx == 0 { split_dist = uniforms.cascade_splits.x; }
        else if cascade_idx == 1 { split_dist = uniforms.cascade_splits.y; }
        else { split_dist = uniforms.cascade_splits.z; }

        let blend_start = split_dist * (1.0 - blend_range);
        if view_depth > blend_start {
            blend_factor = (view_depth - blend_start) / (split_dist - blend_start);
            blend_factor = clamp(blend_factor, 0.0, 1.0);
        }
    }

    if blend_factor > 0.01 && cascade_idx < 3 {
        let next_shadow = sample_shadow_cascade(world_pos, cascade_idx + 1);
        return mix(shadow, next_shadow, blend_factor);
    }

    return shadow;
}

fn horizon_alignment(view_dir: vec3<f32>, sun_dir: vec3<f32>) -> f32 {
    let view_xz = vec2<f32>(view_dir.x, view_dir.z);
    let sun_xz = vec2<f32>(sun_dir.x, sun_dir.z);
    let view_len = length(view_xz);
    let sun_len = length(sun_xz);
    if view_len < 1e-4 || sun_len < 1e-4 {
        return 0.5;
    }
    return clamp(dot(view_xz / view_len, sun_xz / sun_len), 0.0, 1.0);
}

fn lower_atmosphere_haze(dir: vec3<f32>, sun_dir: vec3<f32>) -> vec3<f32> {
    let cloud_cover = clamp(uniforms.shaft_params.w, 0.0, 1.0);
    let horizon = exp(-abs(dir.y) * 12.5);
    let lower_mix = smoothstep(-0.36, 0.16, dir.y);
    let sun_side = pow(horizon_alignment(dir, sun_dir), 1.35);
    let dust = uniforms.fog_color.rgb * vec3<f32>(0.86, 0.82, 0.70) * (0.28 + 0.20 * lower_mix);
    let warm = uniforms.sun_color.rgb * vec3<f32>(0.26, 0.18, 0.09)
        * (0.28 + 0.40 * sun_side)
        * (1.0 - cloud_cover * 0.18);
    let lift = vec3<f32>(0.04, 0.032, 0.020);
    return (dust + warm + lift) * horizon;
}

// Screen-space contact shadows: 16-step ray march along sun direction.
// Returns 0.0 (shadowed) to 1.0 (lit).
fn contact_shadow(world_pos: vec3<f32>, sun_dir: vec3<f32>) -> f32 {
    let screen_size = uniforms.screen_size.xy;

    // Project start position to screen space
    let clip_start = uniforms.view_proj * vec4<f32>(world_pos, 1.0);
    let ndc_start = clip_start.xyz / clip_start.w;
    let screen_start = vec2<f32>(
        (ndc_start.x * 0.5 + 0.5) * screen_size.x,
        (1.0 - (ndc_start.y * 0.5 + 0.5)) * screen_size.y
    );
    let depth_start = ndc_start.z;

    // Project a point offset along sun direction
    let ray_length = 0.5; // world-space ray length
    let end_pos = world_pos + sun_dir * ray_length;
    let clip_end = uniforms.view_proj * vec4<f32>(end_pos, 1.0);
    let ndc_end = clip_end.xyz / clip_end.w;
    let screen_end = vec2<f32>(
        (ndc_end.x * 0.5 + 0.5) * screen_size.x,
        (1.0 - (ndc_end.y * 0.5 + 0.5)) * screen_size.y
    );
    let depth_end = ndc_end.z;

    // Clamp max screen-space ray length
    let screen_delta = screen_end - screen_start;
    let screen_len = length(screen_delta);
    let max_screen_len = 0.15 * max(screen_size.x, screen_size.y);
    if screen_len < 1.0 {
        return 1.0;
    }
    let scale = min(1.0, max_screen_len / screen_len);

    let step_screen = screen_delta * scale / 16.0;
    let step_depth = (depth_end - depth_start) * scale / 16.0;

    var ray_screen = screen_start;
    var ray_depth = depth_start;

    // Adaptive thickness based on depth (thicker when further away)
    let base_thickness = 0.0005;

    for (var i = 1; i <= 16; i++) {
        ray_screen += step_screen;
        ray_depth += step_depth;

        let pixel = vec2<i32>(ray_screen);
        if pixel.x < 0 || pixel.y < 0 ||
           pixel.x >= i32(screen_size.x) || pixel.y >= i32(screen_size.y) {
            break;
        }

        let scene_depth = textureLoad(depth_float, vec2<u32>(vec2<i32>(pixel)), 0).r;
        let thickness = base_thickness * (1.0 + ray_depth * 5.0);

        if ray_depth > scene_depth && (ray_depth - scene_depth) < thickness {
            // Fade out shadow for samples near the end of the ray
            let fade = 1.0 - f32(i) / 16.0;
            return 1.0 - fade;
        }
    }

    return 1.0;
}

// Sample sky-view LUT for physically consistent aerial perspective.
// CANONICAL: UV mapping must match sky_lut.wgsl compute_sky_view AND sky_pass.wgsl dir_to_sky_view_uv.
fn sample_sky_lut(dir: vec3<f32>) -> vec3<f32> {
    let elevation = asin(clamp(dir.y, -1.0, 1.0));
    var v: f32;
    if elevation < 0.0 {
        let t = sqrt(-elevation / (PI * 0.5));
        v = 0.5 - t * 0.5;
    } else {
        let t = sqrt(elevation / (PI * 0.5));
        v = 0.5 + t * 0.5;
    }
    let azimuth = atan2(dir.x, dir.z);
    var u = azimuth / (2.0 * PI);
    u = select(u + 1.0, u, u >= 0.0);
    let sun_color = uniforms.sun_color.rgb * uniforms.sun_color.w;
    let cloud_cover = clamp(uniforms.shaft_params.w, 0.0, 1.0);
    let sky_scale = mix(1.02, 1.10, cloud_cover);
    return textureSampleLevel(sky_view_lut, sky_lut_sampler, vec2<f32>(u, v), 0.0).rgb
        * sun_color
        * sky_scale;
}

fn direct_sun_visibility() -> f32 {
    let cloud_cover = clamp(uniforms.shaft_params.w, 0.0, 1.0);
    let sun_height = max(uniforms.sun_direction.y, 0.0);
    return clamp(1.0 - cloud_cover * (0.16 + 0.08 * sun_height), 0.68, 1.0);
}

fn sky_fill_visibility() -> f32 {
    let cloud_cover = clamp(uniforms.shaft_params.w, 0.0, 1.0);
    return mix(1.0, 1.22, cloud_cover);
}

fn sample_aerial_sky(dir: vec3<f32>) -> vec3<f32> {
    let sun_dir = normalize(uniforms.sun_direction.xyz);
    let cloud_cover = clamp(uniforms.shaft_params.w, 0.0, 1.0);
    let zenith_t = pow(smoothstep(-0.08, 0.58, dir.y), 0.55);
    let horizon_t = smoothstep(-0.10, 0.18, dir.y);
    let scattering = sample_sky_lut(dir) * mix(1.10, 0.98, cloud_cover);
    let lower_haze = lower_atmosphere_haze(dir, sun_dir);
    let lower_grade = mix(
        uniforms.fog_color.rgb * vec3<f32>(0.28, 0.28, 0.22) + uniforms.sun_color.rgb * vec3<f32>(0.08, 0.06, 0.03),
        uniforms.sky_horizon.rgb * 0.14 + uniforms.sun_color.rgb * vec3<f32>(0.05, 0.04, 0.03),
        horizon_t
    );
    let upper_grade = mix(uniforms.sky_horizon.rgb * 0.06, uniforms.sky_zenith.rgb * 0.14, zenith_t);
    let grade = mix(lower_grade, upper_grade, smoothstep(-0.02, 0.60, dir.y));
    return scattering + grade + lower_haze;
}

// -- PBR Functions --

// GGX Normal Distribution Function
fn D_GGX(NdotH: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let d = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

// Smith's geometry function (height-correlated)
fn G_SmithGGX(NdotV: f32, NdotL: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let ggxV = NdotL * sqrt(NdotV * NdotV * (1.0 - a2) + a2);
    let ggxL = NdotV * sqrt(NdotL * NdotL * (1.0 - a2) + a2);
    return 0.5 / max(ggxV + ggxL, 1e-7);
}

// Schlick Fresnel approximation
fn F_Schlick(VdotH: f32, F0: vec3<f32>) -> vec3<f32> {
    let t = 1.0 - VdotH;
    let t2 = t * t;
    let t5 = t2 * t2 * t;
    return F0 + (1.0 - F0) * t5;
}

// Combined Cook-Torrance specular BRDF
fn specular_brdf(N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, roughness: f32, F0: vec3<f32>) -> vec3<f32> {
    let H = normalize(V + L);
    let NdotH = max(dot(N, H), 0.0);
    let NdotV = max(dot(N, V), 1e-4);
    let NdotL = max(dot(N, L), 0.0);
    let VdotH = max(dot(V, H), 0.0);

    let D = D_GGX(NdotH, roughness);
    let G = G_SmithGGX(NdotV, NdotL, roughness);
    let F = F_Schlick(VdotH, F0);

    return D * G * F;
}

// -- Parallax Occlusion Mapping --

fn parallax_occlusion_map(
    uv: vec2<f32>,
    view_ts: vec3<f32>,
    height_scale: f32,
) -> vec2<f32> {
    // Clamp view_ts.z to prevent extreme stepping at grazing angles
    let safe_z = max(view_ts.z, 0.15);
    // Adaptive layer count: more at grazing angles
    let num_layers = mix(32.0, 8.0, safe_z);
    let layer_depth = 1.0 / num_layers;
    let delta_uv = view_ts.xy * height_scale / (safe_z * num_layers);

    var current_uv = uv;
    var current_depth = 0.0;
    var current_height = textureSample(bark_height, bark_sampler, current_uv).r;

    // Linear march
    for (var i = 0u; i < 32u; i++) {
        if current_depth >= current_height { break; }
        current_uv -= delta_uv;
        current_height = textureSample(bark_height, bark_sampler, current_uv).r;
        current_depth += layer_depth;
    }

    // Binary refinement (2 steps)
    var prev_uv = current_uv + delta_uv;
    for (var j = 0u; j < 2u; j++) {
        let mid_uv = (current_uv + prev_uv) * 0.5;
        let mid_height = textureSample(bark_height, bark_sampler, mid_uv).r;
        let mid_depth = current_depth - layer_depth * 0.5;
        if mid_depth < mid_height { prev_uv = mid_uv; }
        else { current_uv = mid_uv; current_depth = mid_depth; }
    }
    return current_uv;
}

// -- Fiber Sheen (Charlie distribution) --

fn D_Charlie(NdotH: f32, roughness: f32) -> f32 {
    let alpha = roughness * roughness;
    let inv_alpha = 1.0 / alpha;
    let sin2h = 1.0 - NdotH * NdotH;
    return (2.0 + inv_alpha) * pow(max(sin2h, 0.0), inv_alpha * 0.5) / TWO_PI;
}

fn sheen_brdf(N: vec3<f32>, V: vec3<f32>, L: vec3<f32>,
              fiber_dir: vec3<f32>, sheen_roughness: f32) -> f32 {
    let H = normalize(V + L);
    let NdotH = max(dot(N, H), 0.0);
    let NdotL = max(dot(N, L), 0.0);
    let NdotV = max(dot(N, V), 1e-4);
    // Peak perpendicular to fiber direction
    let FdotH = abs(dot(fiber_dir, H));
    let fiber_factor = 1.0 - FdotH * FdotH;
    let D = D_Charlie(NdotH, sheen_roughness);
    let V_sheen = 1.0 / (4.0 * (NdotL + NdotV - NdotL * NdotV));
    return D * V_sheen * fiber_factor;
}

// -- Runtime Procedural Micro-Detail --

fn hash_2d(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453);
}

fn value_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(hash_2d(i), hash_2d(i + vec2(1.0, 0.0)), u.x),
        mix(hash_2d(i + vec2(0.0, 1.0)), hash_2d(i + vec2(1.0, 1.0)), u.x),
        u.y
    );
}

fn micro_fiber_noise(uv: vec2<f32>, detail_fade: f32) -> vec2<f32> {
    // Stretch 8:1 along V for fine vertical fiber striations
    var perturbation = vec2<f32>(0.0);
    // Three octaves at fine scales — subtle normal perturbation only
    let freqs = array<f32, 3>(180.0, 420.0, 900.0);
    let amps = array<f32, 3>(0.006, 0.003, 0.0015);
    for (var i = 0u; i < 3u; i++) {
        let sample_uv = vec2<f32>(uv.x * freqs[i], uv.y * freqs[i] * 0.125);
        let n = value_noise(sample_uv) * 2.0 - 1.0;
        perturbation.x += n * amps[i];  // perpendicular to fiber
    }
    return perturbation * detail_fade;
}

// -- Procedural Materials (ported from tree_lab) --

fn material_color(material: u32, position: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    // MATERIAL_TRUNK is handled by the procedural bark path in fs_resolve().
    if material == MATERIAL_SKIN {
        // Warm skin tone — muted, slightly stylized
        return vec3<f32>(0.55, 0.38, 0.30);
    }
    if material == MATERIAL_FOLIAGE {
        let canopy_noise =
            0.5 + 0.5 * sin(position.x * 4.8 + position.z * 5.9 + position.y * 3.1)
            + 0.25 * sin(position.y * 10.0 + position.x * 8.0);
        let tuft_shadow = 0.5 + 0.5 * sin(position.y * 15.0 + position.x * 6.0 - position.z * 7.0);
        let sun_tip_noise = 0.5 + 0.5 * sin(position.x * 2.1 - position.z * 2.7 + position.y * 1.4);
        let deep = vec3<f32>(0.018, 0.042, 0.022);
        let mid = vec3<f32>(0.034, 0.074, 0.036);
        let lit = vec3<f32>(0.080, 0.118, 0.048);
        let warm_tip = vec3<f32>(0.140, 0.150, 0.055);
        let shadow_mix = clamp(tuft_shadow * 0.62 + canopy_noise * 0.16, 0.0, 1.0);
        let lit_mix = clamp(canopy_noise * 0.18 + normal.y * 0.16, 0.0, 1.0);
        let tip_mix = clamp(sun_tip_noise * 0.28 + normal.y * 0.24, 0.0, 1.0);
        return mix(mix(mix(deep, mid, shadow_mix), lit, lit_mix), warm_tip, tip_mix * 0.45);
    }

    // Ground
    let radial = length(position.xz);
    let macro_a = value_noise(position.xz * 0.018 + vec2<f32>(7.4, 2.1));
    let macro_b = value_noise(position.xz * 0.043 + vec2<f32>(19.3, 11.7));
    let litter_noise = value_noise(position.xz * 0.24 + vec2<f32>(3.4, 8.8));
    let detail_noise = value_noise(position.xz * 0.62 + vec2<f32>(23.0, 2.0));
    let root_zone = 1.0 - smoothstep(3.5, 18.0, radial);
    let near_tree = 1.0 - smoothstep(14.0, 64.0, radial);
    let far_plain = smoothstep(70.0, 260.0, radial);

    let duff_bed = clamp(macro_a * 0.55 + macro_b * 0.25 + near_tree * 0.30, 0.0, 1.0);
    let moss_mask = smoothstep(0.48, 0.80, macro_b) * smoothstep(0.16, 0.84, macro_a) * (1.0 - root_zone * 0.30);
    let clearing_mask = smoothstep(0.54, 0.86, macro_a - macro_b * 0.18);
    let litter_mask = smoothstep(0.46, 0.88, litter_noise + macro_a * 0.22) * (0.55 + 0.45 * near_tree);
    let mineral_mask = smoothstep(0.42, 0.76, detail_noise + macro_b * 0.12) * far_plain;

    let damp_soil = vec3<f32>(0.18, 0.13, 0.09);
    let warm_duff = vec3<f32>(0.40, 0.26, 0.14);
    let dry_soil = vec3<f32>(0.30, 0.21, 0.12);
    let root_dark = vec3<f32>(0.13, 0.09, 0.06);
    let muted_moss = vec3<f32>(0.11, 0.14, 0.09);
    let pale_litter = vec3<f32>(0.46, 0.31, 0.16);
    let mineral_dust = vec3<f32>(0.34, 0.26, 0.18);

    var ground = mix(damp_soil, warm_duff, duff_bed * 0.64 + 0.16);
    ground = mix(ground, root_dark, root_zone * 0.32);
    ground = mix(ground, muted_moss, moss_mask * 0.34);
    ground = mix(ground, dry_soil, clearing_mask * 0.26);
    ground = mix(ground, pale_litter, litter_mask * 0.18);
    ground = mix(ground, mineral_dust, mineral_mask * 0.16);

    let tonal_breakup = 0.94 + 0.10 * (macro_b * 0.7 + detail_noise * 0.3);
    ground *= tonal_breakup;
    ground = mix(ground, vec3<f32>(0.36, 0.28, 0.18), far_plain * 0.24);

    return ground;
}

// -- Height fog with per-material attenuation --

fn height_fog(world_pos: vec3<f32>, dist: f32, material: u32) -> f32 {
    let density = uniforms.fog_params.x;
    let falloff = uniforms.fog_params.y;
    let fog_start = uniforms.fog_params.z;
    let fog_end = uniforms.fog_params.w;

    let distance_factor = clamp((dist - fog_start) / max(fog_end - fog_start, 0.001), 0.0, 1.0);
    let distance_fog = 1.0 - exp(-pow(distance_factor, 2.2) * density * 1500.0);
    let ground_mist = exp(-max(world_pos.y, 0.0) * falloff);
    var fog = clamp(distance_fog * (0.24 + ground_mist * 0.76), 0.0, 1.0);

    // Per-material fog attenuation
    if material == MATERIAL_FOLIAGE {
        fog *= 0.84;
    } else if material == MATERIAL_TRUNK {
        fog *= 0.90;
    }
    // Ground: fade into atmospheric haze earlier so the slab edge doesn't read as a horizon stripe.
    if material == MATERIAL_GROUND {
        let edge_fade = smoothstep(fog_start * 0.70, fog_end * 1.05, dist);
        fog = max(fog, edge_fade * 0.88);
    }

    return fog;
}

// -- Fragment shader --

struct MaterialOutput {
    @location(0) color: vec4<f32>,
    @location(1) normal: vec4<f32>,
};

@fragment
fn fs_resolve(input: FullscreenOutput) -> MaterialOutput {
    let pixel = vec2<u32>(input.position.xy);
    let vis_id = textureLoad(vis_texture, pixel, 0).r;

    // Empty pixel — sky pass will fill this
    if vis_id == 0u {
        var out: MaterialOutput;
        out.color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        out.normal = vec4<f32>(0.0, 1.0, 0.0, 0.0);
        return out;
    }

    let packed_vis_id = vis_id - 1u;
    let meshlet_idx = packed_vis_id >> 8u;
    let tri_idx = packed_vis_id & 0xFFu;
    let desc = meshlet_descs[meshlet_idx];

    // Load triangle vertex indices
    let li0 = load_triangle_index(desc, tri_idx, 0u);
    let li1 = load_triangle_index(desc, tri_idx, 1u);
    let li2 = load_triangle_index(desc, tri_idx, 2u);

    let gi0 = meshlet_vertex_indices[desc.vertex_offset + li0];
    let gi1 = meshlet_vertex_indices[desc.vertex_offset + li1];
    let gi2 = meshlet_vertex_indices[desc.vertex_offset + li2];

    let v0 = all_vertices[gi0];
    let v1 = all_vertices[gi1];
    let v2 = all_vertices[gi2];

    // Project vertices to clip space for barycentric computation
    let clip0 = uniforms.view_proj * vec4<f32>(v0.pos_x, v0.pos_y, v0.pos_z, 1.0);
    let clip1 = uniforms.view_proj * vec4<f32>(v1.pos_x, v1.pos_y, v1.pos_z, 1.0);
    let clip2 = uniforms.view_proj * vec4<f32>(v2.pos_x, v2.pos_y, v2.pos_z, 1.0);
    let bary = compute_barycentrics(input.uv, clip0, clip1, clip2);

    // Interpolate all attributes using barycentrics
    let world_pos = vec3<f32>(v0.pos_x, v0.pos_y, v0.pos_z) * bary.x
                  + vec3<f32>(v1.pos_x, v1.pos_y, v1.pos_z) * bary.y
                  + vec3<f32>(v2.pos_x, v2.pos_y, v2.pos_z) * bary.z;
    let normal = normalize(
        vec3<f32>(v0.nor_x, v0.nor_y, v0.nor_z) * bary.x
      + vec3<f32>(v1.nor_x, v1.nor_y, v1.nor_z) * bary.y
      + vec3<f32>(v2.nor_x, v2.nor_y, v2.nor_z) * bary.z
    );
    let uv = vec2<f32>(v0.uv_x, v0.uv_y) * bary.x
           + vec2<f32>(v1.uv_x, v1.uv_y) * bary.y
           + vec2<f32>(v2.uv_x, v2.uv_y) * bary.z;
    let ao = v0.ao * bary.x + v1.ao * bary.y + v2.ao * bary.z;
    let material = v0.material;

    // -- Lighting Setup --

    let sun_dir = normalize(uniforms.sun_direction.xyz);
    let sun_color = uniforms.sun_color.rgb;
    let sun_strength = uniforms.sun_color.w;
    let sun_visibility = direct_sun_visibility();
    let sky_fill_scale = sky_fill_visibility();
    let sun_energy = sun_strength * sun_visibility;
    let camera_pos = uniforms.camera_position.xyz;
    let view_dir = normalize(camera_pos - world_pos);
    let cam_dist = distance(world_pos, camera_pos);

    var NdotL = max(dot(normal, sun_dir), 0.0);
    let NdotV = max(dot(normal, view_dir), 1e-4);
    var shadow = sample_shadow(world_pos);

    // Art direction: band the NdotL * shadow term for cel-shading
    // This creates the discrete light-to-shadow transition ufotable is known for
    if art.shading_bands > 0.01 {
        let light_factor = NdotL * shadow;
        let bands = art.shading_bands;
        let softness = art.band_softness;
        let scaled = light_factor * bands;
        let band_idx = floor(scaled);
        let frac = scaled - band_idx;
        let t = smoothstep(0.5 - softness, 0.5 + softness, frac);
        let banded = (band_idx + t) / bands;
        // Split banded result back into NdotL and shadow
        // Keep shadow as-is for shadow color shift, adjust NdotL
        if NdotL > 0.001 {
            NdotL = banded / max(shadow, 0.001);
            NdotL = clamp(NdotL, 0.0, 1.0);
        }
    }

    // Screen-space contact shadows
    let contact_strength = uniforms.lighting_params.z;
    if contact_strength > 0.001 {
        let cs = contact_shadow(world_pos, sun_dir);
        shadow *= mix(1.0, cs, contact_strength);
    }
    shadow = mix(1.0, shadow, sun_visibility);

    var final_normal = normal;
    var color: vec3<f32>;

    if material == MATERIAL_TRUNK {
        // === Texture-based bark (baked from procedural) ===

        // Tangent basis (trunk ≈ world Y)
        let trunk_up = vec3<f32>(0.0, 1.0, 0.0);
        var T = normalize(cross(normal, trunk_up));
        if length(T) < 0.01 { T = vec3<f32>(1.0, 0.0, 0.0); }
        let B = normalize(cross(T, normal));

        // Quality budget: skip bark texture sampling when stylized
        var bark_a: vec4<f32>;
        var bark_b: vec4<f32>;
        var bark_h: f32;
        var pom_uv = uv;

        if art.skip_bark_texture > 0.5 {
            // Use palette bark color directly — no texture reads
            bark_a = vec4<f32>(palette.bark.rgb * 0.8 + 0.1, 0.85);
            bark_b = vec4<f32>(0.5, 0.5, 1.0, 0.0); // neutral normal, no AO baked
            bark_h = 0.5; // flat height
        } else {
            // POM: offset UVs based on view direction through height field
            let view_world = normalize(camera_pos - world_pos);
            let view_ts = vec3<f32>(dot(view_world, T), dot(view_world, B), dot(view_world, normal));
            let pom_dist_fade = 1.0 - smoothstep(8.0, 25.0, cam_dist);
            // Fade out POM at grazing angles to prevent stepping artifacts
            let pom_angle_fade = smoothstep(0.2, 0.5, view_ts.z);
            let pom_fade = pom_dist_fade * pom_angle_fade;
            if pom_fade > 0.01 {
                pom_uv = parallax_occlusion_map(uv, view_ts, 0.018 * pom_fade);
            }

            bark_a = textureSample(bark_albedo_rough, bark_sampler, pom_uv);
            bark_b = textureSample(bark_normal_ao, bark_sampler, pom_uv);
            bark_h = textureSample(bark_height, bark_sampler, pom_uv).r;
        }

        // Height-driven furrow darkening: deep furrows nearly black, ridges warm
        // detail_suppression flattens this contrast for stylized looks
        let ds = art.detail_suppression;
        let furrow_darken = mix(
            mix(0.16, 1.0, smoothstep(0.10, 0.42, bark_h)),
            mix(0.65, 1.0, smoothstep(0.30, 0.55, bark_h)),
            ds
        );
        // Blend textured bark toward flat painted color for stylized looks
        let textured_bark = bark_a.rgb * furrow_darken;
        let flat_bark = palette.bark.rgb * 0.85 + 0.12;
        let bark_color = mix(textured_bark, flat_bark, smoothstep(0.15, 0.6, ds));
        // Furrows are rougher (more matte), ridges slightly smoother
        let bark_roughness = bark_a.a + (1.0 - bark_h) * 0.08 * (1.0 - ds);
        // Height-driven AO: furrows trap light aggressively (flatten when stylized)
        let height_ao = mix(
            mix(0.15, 1.0, smoothstep(0.06, 0.30, bark_h)),
            mix(0.5, 1.0, smoothstep(0.06, 0.30, bark_h)),
            ds
        );
        let bark_ao = bark_b.b * ao * height_ao;

        // Unpack tangent-space normal from baked texture
        var perturbed_normal = normal;
        if art.skip_normal_perturbation < 0.5 {
            let tn = bark_b.rg * 2.0 - 1.0;
            let tn_z = sqrt(max(1.0 - tn.x * tn.x - tn.y * tn.y, 0.0));
            perturbed_normal = normalize(T * tn.x + B * tn.y + normal * tn_z);

            // Runtime micro-detail: fiber noise below texture resolution
            // Tight distance fade (only visible <8m) + angle fade to prevent banding
            let detail_dist_fade = smoothstep(10.0, 3.0, cam_dist);
            let detail_angle_fade = smoothstep(0.15, 0.4, max(dot(normal, view_dir), 0.0));
            let detail_fade = detail_dist_fade * detail_angle_fade;
            if detail_fade > 0.01 {
                let micro = micro_fiber_noise(pom_uv, detail_fade);
                perturbed_normal = normalize(perturbed_normal + T * micro.x + B * micro.y);
            }
        }

        // Normal map influence — aggressively reduced for cel-shading
        // At full suppression, normals are nearly geometric (smooth bands)
        let base_normal_blend = select(0.78, 0.0, art.skip_normal_perturbation > 0.5);
        let normal_blend = base_normal_blend * (1.0 - ds) * (1.0 - ds);
        let lighting_normal = normalize(mix(normal, perturbed_normal, normal_blend));

        // Diffuse with wrap lighting for porous bark
        let base_NdotL = max(dot(lighting_normal, sun_dir), 0.0);
        let diffuse_term = sun_color * sun_energy * base_NdotL * shadow;
        let wrap_dot = dot(lighting_normal, sun_dir) * 0.5 + 0.5;
        // Suppress wrap fill in deep furrows — they should stay dark
        let furrow_fill_mask = smoothstep(0.15, 0.50, bark_h);
        let wrap_fill = sun_color * sun_energy * 0.16 * max(wrap_dot, 0.0) * (1.0 - shadow * 0.5) * furrow_fill_mask;

        // Micro-cavity self-shadowing: normals facing away from light get extra darkening
        let cavity_shadow = smoothstep(-0.1, 0.3, dot(perturbed_normal, sun_dir));

        // Anisotropic specular: fiber-aligned Kajiya-Kay approximation
        // Tangent direction is approximately trunk_up (vertical fibers)
        let fiber_tangent = normalize(trunk_up - normal * dot(trunk_up, normal));
        let TdotH = dot(fiber_tangent, normalize(view_dir + sun_dir));
        let aniso_spec = pow(sqrt(max(1.0 - TdotH * TdotH, 0.0)), 1.0 / max(bark_roughness * 0.7, 0.05));
        let aniso_term = aniso_spec * 0.015 * sun_color * sun_energy * base_NdotL * shadow;

        // Isotropic GGX specular (reduced, complementing aniso)
        let bark_f0 = vec3<f32>(0.04);
        let spec = specular_brdf(lighting_normal, view_dir, sun_dir, bark_roughness, bark_f0);
        let specular_term = spec * base_NdotL * sun_color * sun_energy * shadow * 0.6 + aniso_term;

        // Fiber sheen: very subtle warm glow perpendicular to fiber direction
        let sheen_rough = bark_roughness * 0.85 + 0.15;
        let sheen_tint = bark_color * 0.3 + vec3<f32>(0.18, 0.08, 0.03);
        let sheen = sheen_brdf(lighting_normal, view_dir, sun_dir, fiber_tangent, sheen_rough);
        let sheen_term = sheen_tint * sheen * base_NdotL * sun_color * sun_energy * shadow * 0.03;

        // Subsurface scattering — bark is porous, light penetrates ridges
        let shedding_mask = bark_b.a;
        let back_scatter = pow(max(dot(view_dir, -sun_dir), 0.0), 3.0);
        let sss_color = vec3<f32>(0.9, 0.25, 0.04);
        // SSS suppressed in deep furrows (light can't penetrate there)
        let sss_height_mask = smoothstep(0.15, 0.45, bark_h);
        let sss_base = sss_color * 0.04 * (0.3 + 0.7 * back_scatter) * sun_energy * sss_height_mask;
        let sss_shed = sss_color * shedding_mask * 0.16 * (0.2 + 0.8 * back_scatter) * sun_energy * sss_height_mask;
        let edge_sss = sss_base + sss_shed;

        // Ambient with strong AO — deep darks in crevices
        let bark_ambient = six_face_ambient(perturbed_normal) * bark_ao * bark_ao * 1.18;

        color = bark_color * (bark_ambient + (diffuse_term + wrap_fill) * cavity_shadow)
            + apply_specular_flatten(specular_term + sheen_term, art.specular_flattening) + edge_sss;
        final_normal = perturbed_normal;
    } else {
        // === Standard PBR pipeline (foliage + ground) ===
        var roughness: f32;
        var metallic: f32;
        var f0: vec3<f32>;
        var sss_strength: f32;

        if material == MATERIAL_FOLIAGE {
            roughness = 0.74;
            metallic = 0.0;
            f0 = vec3<f32>(0.025);
            sss_strength = 0.35;
        } else if material == MATERIAL_SKIN {
            roughness = 0.65;
            metallic = 0.0;
            f0 = vec3<f32>(0.04);
            sss_strength = 0.15;
        } else {
            // Ground
            roughness = 0.92;
            metallic = 0.0;
            f0 = vec3<f32>(0.03);
            sss_strength = 0.0;
        }

        let base_color = material_color(material, world_pos, normal);

        // Diffuse: Lambertian with energy conservation
        let kD = (1.0 - metallic) * base_color / PI;
        let diffuse = kD * NdotL * sun_color * sun_energy * shadow;

        // Specular: Cook-Torrance GGX (art direction can flatten)
        let spec = specular_brdf(normal, view_dir, sun_dir, roughness, f0);
        let specular = apply_specular_flatten(
            spec * NdotL * sun_color * sun_energy * shadow,
            art.specular_flattening
        );

        // Ambient: 6-face directional probe x AO
        let ambient = six_face_ambient(normal) * ao * base_color;

        // Sky fill: hemisphere-weighted sky tint (LUT-based for consistency)
        let sky = sample_sky_lut(reflect(-view_dir, normal));
        let sky_fill = sky * 0.05 * ao * sky_fill_scale;

        // Two-sided foliage transmission
        var sss = vec3<f32>(0.0);
        var diffuse_term = diffuse;
        var ambient_term = ambient;
        var sky_fill_term = sky_fill;
        if material == MATERIAL_FOLIAGE {
            // Flip normal for back-facing fragments
            var foliage_normal = normal;
            if dot(normal, view_dir) < 0.0 {
                foliage_normal = -normal;
            }

            // View-dependent transmission (NOT gated by shadow map)
            let subsurface_color = vec3<f32>(0.46, 0.60, 0.16);
            let transmission = subsurface_color
                * pow(max(dot(view_dir, -sun_dir), 0.0), 4.0)
                * sun_color * sun_energy * 0.75;

            // Diffuse with wrap lighting
            let wrap_NdotL = dot(foliage_normal, sun_dir) * 0.3 + 0.7;
            let foliage_diffuse = kD * max(wrap_NdotL, 0.0) * sun_color * sun_energy * shadow;

            diffuse_term = mix(diffuse, foliage_diffuse, 0.92);
            ambient_term = ambient * (1.10 + max(normal.y, 0.0) * 0.38);
            sky_fill_term = sky_fill * 2.20;
            sss = transmission * sss_strength;
        } else {
            let sun_warmth = smoothstep(0.0, 0.85, NdotL);
            diffuse_term *= mix(vec3<f32>(1.0), vec3<f32>(1.14, 1.03, 0.84), sun_warmth) * 1.12;
            ambient_term = ambient * 1.26;
            sky_fill_term = sky_fill * 1.75;
        }

        // Ground contact shadow & root zone
        var ground_darken = 1.0;
        if material == MATERIAL_GROUND {
            let radial = length(world_pos.xz);
            let contact = 1.0 - smoothstep(1.6, 6.8, radial);
            let root_zone = 1.0 - smoothstep(4.2, 13.5, radial);
            ground_darken = (1.0 - contact * uniforms.lighting_params.z) * (1.0 - root_zone * 0.08);
        }

        // Foliage rim
        var extra = vec3<f32>(0.0);
        if material == MATERIAL_FOLIAGE {
            let rim = pow(1.0 - max(dot(normal, view_dir), 0.0), 2.0);
            extra = rim * vec3<f32>(0.010, 0.014, 0.006);
        }

        // Ground bounce: sunlit earth scatters warm light upward
        let ground_bounce_color = vec3<f32>(0.10, 0.075, 0.040);
        let ground_facing = max(-normal.y, 0.0);
        let ground_bounce = ground_bounce_color * ground_facing * sun_energy;

        // Canopy scatter: filtered green light under tree
        let canopy_top = bark.trunk_height * 0.9;
        let under_canopy = smoothstep(canopy_top, canopy_top - 12.0, world_pos.y);
        let canopy_scatter = vec3<f32>(0.012, 0.020, 0.008) * under_canopy * sun_energy;

        // Compose lighting
        color = (ambient_term + diffuse_term + specular + sky_fill_term + sss + extra) * ground_darken
              + (ground_bounce + canopy_scatter) * ao;
    }

    // -- Art Direction Style Transforms --
    // Applied after PBR lighting, before fog. All no-ops at identity (default values).
    // (Light banding is applied earlier, to NdotL, for correct cel-shading)

    // 1. Shadow color shift — only on trunk/foliage (ground keeps warm shadow)
    if material != MATERIAL_GROUND {
        color = apply_shadow_color_shift(color, NdotL, shadow, art.shadow_color_shift.rgb);
    }

    // 2. Rim lighting
    color += compute_rim_light(final_normal, view_dir, art.rim_light_strength, art.rim_light_color.rgb);

    // 4. Palette remap
    color = apply_palette_remap(color, material, art.color_discipline);

    if material == MATERIAL_GROUND {
        let distance_lift = smoothstep(uniforms.fog_params.z * 0.55, uniforms.fog_params.w * 0.92, cam_dist);
        color = mix(color, color + vec3<f32>(0.12, 0.09, 0.05), distance_lift * 0.22);
    }

    // -- Aerial perspective from shared atmosphere model --
    // Height fog provides the distance-based opacity curve.
    // The inscatter color comes from the same sky LUT as the sky pass.
    // Lift sub-horizon directions gradually so distant ground converges toward
    // lower-sky haze instead of a hard horizon-color band.
    let fog_factor = height_fog(world_pos, cam_dist, material);
    let fog_dir = normalize(world_pos - camera_pos);
    var aerial = sample_aerial_sky(fog_dir);
    if material == MATERIAL_GROUND {
        let ground_haze = lower_atmosphere_haze(fog_dir, sun_dir) + vec3<f32>(0.18, 0.15, 0.10);
        let sky_lift = smoothstep(-0.18, 0.16, fog_dir.y);
        aerial = mix(ground_haze, aerial * 1.08, sky_lift);
    }
    color = mix(color, aerial, fog_factor);

    // -- Apply exposure --
    color *= uniforms.lighting_params.x;

    var out: MaterialOutput;
    out.color = vec4<f32>(color, 1.0);
    out.normal = vec4<f32>(final_normal * 0.5 + 0.5, 1.0);
    return out;
}
