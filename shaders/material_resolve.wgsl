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
    lighting_params: vec4<f32>,     // (exposure, tonemap_strength, contact_shadow, 0)
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
    uv_x: f32, uv_y: f32,
    ao: f32,
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

// Shadow map
@group(2) @binding(1) var shadow_map: texture_depth_2d;
@group(2) @binding(2) var shadow_sampler: sampler_comparison;

// Alpha mask for foliage
@group(2) @binding(3) var alpha_mask: texture_2d<f32>;
@group(2) @binding(4) var alpha_sampler: sampler;

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

fn sample_shadow(world_pos: vec3<f32>) -> f32 {
    let light_clip = uniforms.light_vp * vec4<f32>(world_pos, 1.0);
    let light_ndc = light_clip.xyz / light_clip.w;
    let shadow_uv = light_ndc.xy * 0.5 + 0.5;
    let shadow_uv_flipped = vec2<f32>(shadow_uv.x, 1.0 - shadow_uv.y);

    if shadow_uv_flipped.x < 0.0 || shadow_uv_flipped.x > 1.0 ||
       shadow_uv_flipped.y < 0.0 || shadow_uv_flipped.y > 1.0 {
        return 1.0;
    }

    let depth = light_ndc.z;
    return textureSampleCompare(shadow_map, shadow_sampler, shadow_uv_flipped, depth);
}

fn sky_color_dir(dir: vec3<f32>) -> vec3<f32> {
    let horizon_mix = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
    return mix(uniforms.sky_horizon.rgb, uniforms.sky_zenith.rgb, horizon_mix);
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

// -- Procedural Materials (ported from tree_lab) --

fn material_color(material: u32, position: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    // MATERIAL_TRUNK is handled by the procedural bark path in fs_resolve().
    if material == MATERIAL_FOLIAGE {
        let canopy_noise =
            0.5 + 0.5 * sin(position.x * 4.8 + position.z * 5.9 + position.y * 3.1)
            + 0.25 * sin(position.y * 10.0 + position.x * 8.0);
        let tuft_shadow = 0.5 + 0.5 * sin(position.y * 15.0 + position.x * 6.0 - position.z * 7.0);
        let deep = vec3<f32>(0.012, 0.040, 0.020);
        let mid = vec3<f32>(0.030, 0.078, 0.036);
        let lit = vec3<f32>(0.055, 0.112, 0.050);
        let shadow_mix = clamp(tuft_shadow * 0.68 + canopy_noise * 0.18, 0.0, 1.0);
        let lit_mix = clamp(canopy_noise * 0.24 + normal.y * 0.10, 0.0, 1.0);
        return mix(mix(deep, mid, shadow_mix), lit, lit_mix);
    }

    // Ground
    let radial = length(position.xz);
    let soil_band = 0.5 + 0.5 * sin(radial * 0.85 + position.x * 0.18);
    let moss_patch = 0.5 + 0.5 * sin(position.x * 0.42 + position.z * 0.58 + soil_band * 2.0);
    let moss_patch2 = 0.5 + 0.5 * sin(position.x * 0.71 - position.z * 0.34 + soil_band * 1.4);
    let cool_soil = vec3<f32>(0.14, 0.12, 0.11);
    let dark_soil = vec3<f32>(0.08, 0.07, 0.065);
    let muted_moss = vec3<f32>(0.060, 0.075, 0.060);
    let soil = mix(dark_soil, cool_soil, soil_band * 0.38 + 0.16);

    // Needle litter
    let needle_noise = sin(position.x * 18.0 + position.z * 22.0) * sin(position.z * 14.0 - position.x * 9.0);
    let needle_litter = vec3<f32>(0.12, 0.08, 0.05);
    var ground = mix(soil, needle_litter, clamp(needle_noise * 0.5 + 0.5, 0.0, 1.0) * 0.22);

    // Moss patches
    ground = mix(ground, muted_moss, moss_patch * 0.28 + moss_patch2 * 0.14);

    // Root zone shadow
    let root_shadow = 1.0 - (1.0 - smoothstep(0.5, 3.0, radial)) * 0.15;
    ground *= root_shadow;

    return ground;
}

fn foliage_alpha_mask(uv: vec2<f32>, position: vec3<f32>) -> f32 {
    let centered_x = abs(uv.x * 2.0 - 1.0);
    let stem_profile = smoothstep(0.01, 0.08, uv.y) * (1.0 - smoothstep(0.93, 1.0, uv.y));
    let taper = mix(0.44, 0.09, smoothstep(0.04, 0.98, uv.y));
    let width_noise =
        0.055 * sin(uv.y * 10.0 + position.x * 5.0)
        + 0.030 * sin(uv.y * 21.0 + position.z * 8.0)
        + 0.018 * sin(uv.y * 37.0 + position.y * 4.0);
    let lobe_shift =
        0.10 * sin(uv.y * 7.0 + position.x * 4.0)
        + 0.04 * sin(uv.y * 15.0 + position.z * 6.0);
    let outer_width = max(taper + width_noise, 0.06);
    let outer = 1.0 - smoothstep(outer_width, outer_width + 0.34, centered_x);
    let inner = 1.0
        - smoothstep(
            outer_width * 0.56,
            outer_width * 0.56 + 0.24,
            abs((uv.x * 2.0 - 1.0) * 0.82 - lobe_shift),
        );
    let tip_cluster = 1.0 - smoothstep(0.76, 1.0, uv.y) * 0.28;
    let alpha = max(outer * 0.74, inner) * stem_profile * tip_cluster;
    return smoothstep(0.06, 0.90, alpha);
}

// -- Height fog with per-material attenuation --

fn height_fog(world_pos: vec3<f32>, dist: f32, material: u32) -> f32 {
    let density = uniforms.fog_params.x;
    let falloff = uniforms.fog_params.y;
    let fog_start = uniforms.fog_params.z;
    let fog_end = uniforms.fog_params.w;

    let distance_factor = clamp((dist - fog_start) / max(fog_end - fog_start, 0.001), 0.0, 1.0);
    let distance_fog = 1.0 - exp(-distance_factor * density * 8.0);
    let ground_mist = exp(-max(world_pos.y, 0.0) * falloff);
    var fog = clamp(distance_fog * (0.18 + ground_mist * 0.82), 0.0, 1.0);

    // Per-material fog attenuation
    if material == MATERIAL_FOLIAGE {
        fog *= 0.84;
    } else if material == MATERIAL_TRUNK {
        fog *= 0.90;
    }
    // Ground: 100% (no attenuation)

    return fog;
}

// -- Noise Primitives --

fn hash11(seed_val: u32, n: u32) -> f32 {
    var h = seed_val ^ (n * 747796405u + 2891336453u);
    h = ((h >> 16u) ^ h) * 2654435769u;
    h = ((h >> 16u) ^ h);
    return f32(h) / 2147483648.0 - 1.0;
}

fn mod_positive(x: f32, m: f32) -> f32 {
    return ((x % m) + m) % m;
}

fn hash_smooth(seed_val: u32, x: f32) -> f32 {
    let xi = floor(x);
    let xf = fract(x);
    let t = xf * xf * (3.0 - 2.0 * xf);
    let ia = u32(bitcast<i32>(i32(xi)) & 0x7FFFFFFF);
    let ib = u32(bitcast<i32>(i32(xi) + 1) & 0x7FFFFFFF);
    return mix(hash11(seed_val, ia), hash11(seed_val, ib), t);
}

fn fbm(seed_val: u32, x: f32) -> f32 {
    var value = 0.0;
    var amp = 0.5;
    var freq = 1.0;
    for (var i = 0u; i < 3u; i++) {
        value += amp * hash_smooth(seed_val + i * 17u, x * freq);
        amp *= 0.5;
        freq *= 2.0;
    }
    return value;
}

fn hash21(seed_val: u32, p: vec2<f32>) -> f32 {
    let px = u32(bitcast<i32>(i32(floor(p.x))) & 0x7FFFFFFF);
    let py = u32(bitcast<i32>(i32(floor(p.y))) & 0x7FFFFFFF);
    var h = seed_val ^ (px * 747796405u + py * 2891336453u);
    h = ((h >> 16u) ^ h) * 2654435769u;
    h = ((h >> 16u) ^ h);
    return f32(h) / 2147483648.0 - 1.0;
}

struct CrackResult {
    furrow_depth: f32,
    ridge_u: f32,
    nearest_crack_theta: f32,
    crack_spacing: f32,
};

fn bark_cracks(theta: f32, h: f32, height_t: f32, R_h: f32, t_h: f32,
               stiffness_ratio: f32, seed_val: u32) -> CrackResult {
    let s_h = PI * t_h * sqrt(stiffness_ratio);
    let widened_spacing = s_h * mix(1.35, 0.94, sqrt(height_t));
    let n = max(floor(TWO_PI * R_h / widened_spacing), 3.0);
    let crack_spacing = TWO_PI / n;

    let i_center = round(theta / crack_spacing);
    var min_dist = 1e9;
    var nearest_crack = theta;
    var second_dist = 1e9;

    for (var di = -1i; di <= 1i; di++) {
        let i = i_center + f32(di);
        let crack_seed = seed_val + u32(abs(i32(i)));
        // Keep the channels tall and continuous, with only modest wandering.
        let jitter_scale = mix(0.18, 0.08, height_t);
        let jitter_slow = fbm(crack_seed, h / 5.5) * jitter_scale * crack_spacing;
        let jitter_kink = hash_smooth(crack_seed + 31u, h * 0.45) * jitter_scale * 0.38 * crack_spacing;
        let jitter_fast = hash_smooth(crack_seed + 67u, h * 1.4) * jitter_scale * 0.12 * crack_spacing;
        let theta_crack = i * crack_spacing + jitter_slow + jitter_kink + jitter_fast;
        let d = abs(mod_positive(theta - theta_crack + PI, TWO_PI) - PI);
        if d < min_dist {
            second_dist = min_dist;
            min_dist = d;
            nearest_crack = theta_crack;
        } else if d < second_dist {
            second_dist = d;
        }
    }

    // Wide fissured furrows — real redwood has deep wide channels
    let crack_id = nearest_crack * 2.3 + h * 0.15;
    let width_vary = 0.72 + 0.56 * (hash_smooth(seed_val + 90u, crack_id) * 0.5 + 0.5);
    let depth_vary = 0.55 + 0.45 * (hash_smooth(seed_val + 95u, crack_id * 1.7) * 0.5 + 0.5);
    // Wider at base (height_t near 0), narrower at crown
    let height_width = mix(0.36, 0.15, pow(height_t, 0.58));
    let furrow_width = crack_spacing * height_width * width_vary;
    let furrow_depth = smoothstep(furrow_width, furrow_width * 0.26, min_dist)
        * depth_vary
        * mix(1.18, 0.72, pow(height_t, 0.75));
    let ridge_span = min_dist + second_dist;
    let ridge_u = clamp(min_dist / max(ridge_span * 0.5, 0.001), 0.0, 1.0);

    var result: CrackResult;
    result.furrow_depth = furrow_depth;
    result.ridge_u = ridge_u;
    result.nearest_crack_theta = nearest_crack;
    result.crack_spacing = crack_spacing;
    return result;
}

struct RidgeResult {
    profile: f32,
    normal_perturbation: f32,
};

fn bark_ridge_profile(ridge_u: f32, depth_h: f32) -> RidgeResult {
    let w = 0.12; // Steeper edge
    let c = 0.08;
    let edge = smoothstep(0.0, w, ridge_u) * smoothstep(0.0, w, 1.0 - ridge_u);
    // Plateau crown instead of continuous sine
    let crown = 1.0 + c * (smoothstep(0.1, 0.9, ridge_u) * smoothstep(0.1, 0.9, 1.0 - ridge_u));
    let profile = edge * crown * depth_h;

    let d_edge_left = (1.0 / w) * smoothstep(0.0, w, 1.0 - ridge_u);
    let d_edge_right = -(1.0 / w) * smoothstep(0.0, w, ridge_u);
    let d_edge = select(select(0.0, d_edge_right, ridge_u > 1.0 - w), d_edge_left, ridge_u < w);
    let d_crown = 0.0; // Negligible derivative on the plateau
    let d_profile = (d_edge * crown + edge * d_crown) * depth_h;

    var result: RidgeResult;
    result.profile = profile;
    result.normal_perturbation = d_profile;
    return result;
}

fn perp2d(v: vec2<f32>) -> vec2<f32> { return vec2<f32>(-v.y, v.x); }

fn bark_fiber_flow(theta: f32, h: f32, nearest_crack: f32,
                   crack_spacing: f32, seed_val: u32) -> vec2<f32> {
    let dist = abs(mod_positive(theta - nearest_crack + PI, TWO_PI) - PI);
    let fan_width = 0.3 * crack_spacing;
    let proximity = 1.0 - smoothstep(0.0, fan_width, dist);
    let fan_angle = atan2(theta - nearest_crack, 2.0);
    let fan_dir = normalize(vec2<f32>(sin(fan_angle), cos(fan_angle)));
    let base_flow = normalize(mix(vec2<f32>(0.0, 1.0), fan_dir, proximity * 0.5));

    // Burl swirls (Idea 3)
    let burl_uv = vec2<f32>(theta * 6.0, h * 0.15);
    let burl_center = floor(burl_uv) + 0.5;
    let burl_local = fract(burl_uv) - 0.5;
    let burl_dist = length(burl_local);
    let burl_hash = hash21(seed_val + 300u, burl_center);
    let is_burl = step(0.85, burl_hash + 0.5); // 15% chance
    let swirl_dir = vec2<f32>(-burl_local.y, burl_local.x);
    let swirl_amount = is_burl * smoothstep(0.5, 0.1, burl_dist) * 2.5;

    return normalize(base_flow + swirl_dir * swirl_amount);
}

struct FiberBundleResult { bundle: f32, normal_perturbation: f32, };

fn bark_fiber_bundles(theta: f32, h: f32, radius: f32, fiber_dir: vec2<f32>,
                      density: f32, seed_val: u32) -> FiberBundleResult {
    // Keep the bundle count periodic around the circumference so the wrap seam disappears.
    let radius_scale = sqrt(max(radius, 0.25));
    let bundle_count = max(6.0, floor(density * (1.35 + radius_scale)));
    let modulation_count = max(1.0, floor(bundle_count * 0.25));
    let along = h * (0.70 + 0.30 * abs(fiber_dir.y));
    let cross_shear = fiber_dir.x * 0.22;
    let bundle_phase = theta * bundle_count + along * cross_shear;
    let bundle = sin(bundle_phase) * 0.5 + 0.5;
    let modulation_phase =
        theta * modulation_count + along * 0.06 + hash11(seed_val + 7u, 19u) * PI;
    let modulation = 0.72 + 0.28 * (0.5 + 0.5 * sin(modulation_phase));
    let bundle_mod = bundle * modulation;
    let d_bundle = cos(bundle_phase) * bundle_count;
    var r: FiberBundleResult;
    r.bundle = bundle_mod; r.normal_perturbation = d_bundle;
    return r;
}

// -- Bark shedding layer --

struct SheddingResult {
    exposed: f32,       // 0 = intact outer bark, 1 = fully exposed inner bark
    edge_shadow: f32,   // dark undercut at peel boundary
    edge_lift: f32,     // bright catch-light on curled lip
};

fn bark_shedding(theta: f32, h: f32, nearest_crack: f32,
                 crack_spacing: f32, height_t: f32, seed_val: u32) -> SheddingResult {
    // Elongated strip noise — stretched 6:1 along height (bark sheds in long strips)
    let strip_theta = theta * 3.0;
    let strip_h = h * 0.5;
    let strip_noise = fbm(seed_val + 200u, strip_theta * 2.7 + strip_h * 0.3)
                    + 0.5 * fbm(seed_val + 213u, strip_theta * 1.1 - strip_h * 0.7);

    // Height-band modulator — shedding concentrates mid-trunk
    let band = smoothstep(0.08, 0.25, height_t) * smoothstep(0.95, 0.7, height_t);

    // Crack-aligned tearing — strips tend to detach along existing furrows
    let crack_dist = abs(mod_positive(theta - nearest_crack + PI, TWO_PI) - PI);
    let crack_proximity = 1.0 - smoothstep(0.0, crack_spacing * 0.35, crack_dist);

    // Combine: higher values = more likely to have shed
    let shed_probability = strip_noise * 0.5 + 0.5;
    let shed_threshold = 0.57 - band * 0.10 - crack_proximity * 0.06;
    let raw_exposed = smoothstep(shed_threshold, shed_threshold + 0.14, shed_probability)
        * band
        * 0.65;

    // Edge detection: visible peel boundary
    let edge_band = smoothstep(shed_threshold - 0.04, shed_threshold, shed_probability)
                  * (1.0 - smoothstep(shed_threshold, shed_threshold + 0.04, shed_probability));

    var result: SheddingResult;
    result.exposed = raw_exposed;
    result.edge_shadow = edge_band * 0.65;
    result.edge_lift = edge_band * raw_exposed * 0.18;
    return result;
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

    // -- Foliage alpha test with sky fallback --
    if material == MATERIAL_FOLIAGE {
        let alpha = foliage_alpha_mask(uv, world_pos);
        if alpha < 0.001 {
            // Output transparent black — sky pass will fill
            var out: MaterialOutput;
            out.color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            out.normal = vec4<f32>(0.0, 1.0, 0.0, 0.0);
            return out;
        }
    }

    // -- Lighting Setup --

    let sun_dir = normalize(uniforms.sun_direction.xyz);
    let sun_color = uniforms.sun_color.rgb;
    let sun_strength = uniforms.sun_color.w;
    let camera_pos = uniforms.camera_position.xyz;
    let view_dir = normalize(camera_pos - world_pos);
    let cam_dist = distance(world_pos, camera_pos);

    let NdotL = max(dot(normal, sun_dir), 0.0);
    let NdotV = max(dot(normal, view_dir), 1e-4);
    let shadow = sample_shadow(world_pos);

    var final_normal = normal;
    var color: vec3<f32>;

    if material == MATERIAL_TRUNK {
        // === Procedural bark evaluation ===
        let theta = uv.x * TWO_PI;
        let h = uv.y * bark.trunk_height;
        let height_t = clamp(uv.y, 0.0, 1.0);
        let R_h = mix(bark.trunk_radius_base, bark.trunk_radius_tip, height_t);
        let t_h = mix(bark.bark_thickness_base, bark.bark_thickness_tip, height_t);

        let cracks = bark_cracks(theta, h, height_t, R_h, t_h, bark.stiffness_ratio, bark.seed);

        // Distance LOD (swap to DAG-level when GPU cull is activated)
        // (cam_dist computed above, shared with fog)
        // Wider transition bands prevent visible LOD pops
        let lod_weight_fiber = 1.0 - smoothstep(3.0, 12.0, cam_dist);
        let lod_weight_flow  = 1.0 - smoothstep(8.0, 25.0, cam_dist);
        let lod_weight_ridge = 1.0 - smoothstep(20.0, 55.0, cam_dist);
        // Shared noise (used by both distant and close bark paths)
        let age = 1.0 - height_t;
        let patch_noise = fbm(bark.seed + 77u, theta * 1.8 + h * 0.4) * 0.5 + 0.5;
        let micro_a = hash_smooth(bark.seed + 100u, theta * 18.0 + h * 3.2);

        // Distant bark baseline; we blend this with the close bark path to avoid
        // a visible LOD seam across the trunk.
        var distant_color = mix(vec3<f32>(0.42, 0.18, 0.07), vec3<f32>(0.30, 0.13, 0.05), age * 0.46 + patch_noise * 0.22);
        distant_color = mix(distant_color, vec3<f32>(0.11, 0.06, 0.04), age * age * 0.48);
        distant_color = mix(distant_color, vec3<f32>(0.12, 0.08, 0.07), cracks.furrow_depth * 0.16);
        distant_color *= (1.0 - cracks.furrow_depth * 0.62) * (1.0 + micro_a * 0.10);
        let distant_base_ndotl = max(dot(normal, sun_dir), 0.0);
        let distant_wrap_dot = dot(normal, sun_dir) * 0.4 + 0.6;
        let distant_lit = distant_color * (six_face_ambient(normal) * ao * 1.3
            + sun_color * sun_strength * distant_base_ndotl * shadow
            + sun_color * sun_strength * 0.12 * max(distant_wrap_dot, 0.0));

        // Full bark evaluation (Layers 2-5)

        // Ridge profile
        let depth_h = mix(1.0, 0.3, height_t);
        let ridge = bark_ridge_profile(cracks.ridge_u, depth_h);

        // Tangent basis (trunk ≈ world Y)
        let trunk_up = vec3<f32>(0.0, 1.0, 0.0);
        var tangent_theta = cross(normal, trunk_up);
        if length(tangent_theta) < 0.01 { tangent_theta = vec3<f32>(1.0, 0.0, 0.0); }
        tangent_theta = normalize(tangent_theta);
        let tangent_h = normalize(cross(tangent_theta, normal));

        let perturbed = normalize(normal
            + tangent_theta * ridge.normal_perturbation * 0.50 * lod_weight_ridge);

        // Fiber flow and bundles (LOD-gated)
        let fiber_dir_full = bark_fiber_flow(theta, h, cracks.nearest_crack_theta,
                                              cracks.crack_spacing, bark.seed);
        let fiber_dir = mix(vec2<f32>(0.0, 1.0), fiber_dir_full, lod_weight_flow);
        let bundles = bark_fiber_bundles(theta, h, R_h, fiber_dir, bark.fiber_density, bark.seed);

        // Fiber micro-normal (LOD-gated)
        let fiber_perp = normalize(tangent_theta * perp2d(fiber_dir).x
                                 + tangent_h * perp2d(fiber_dir).y);
        var accumulated_normal = normalize(perturbed
            + fiber_perp * bundles.normal_perturbation * 0.04 * lod_weight_fiber);

        // Surface roughness normals — grit/weathering at high frequency
        // Two overlapping scales of hash noise perturb the normal everywhere
        let grit_u1 = hash_smooth(bark.seed + 160u, theta * 45.0 + h * 12.0);
        let grit_v1 = hash_smooth(bark.seed + 170u, theta * 38.0 - h * 15.0);
        let grit_u2 = hash_smooth(bark.seed + 180u, theta * 90.0 + h * 28.0);
        let grit_v2 = hash_smooth(bark.seed + 190u, theta * 75.0 - h * 35.0);
        // Grit fades slower than fibers — stays visible at mid range
        let lod_weight_grit = 1.0 - smoothstep(5.0, 35.0, cam_dist);
        let grit_strength = 0.18 * lod_weight_grit;
        accumulated_normal = normalize(accumulated_normal
            + tangent_theta * (grit_u1 * 0.6 + grit_u2 * 0.4) * grit_strength
            + tangent_h * (grit_v1 * 0.6 + grit_v2 * 0.4) * grit_strength);

        // Shedding layer (LOD-gated with fiber flow)
        let shedding = bark_shedding(theta, h, cracks.nearest_crack_theta,
                                      cracks.crack_spacing, height_t, bark.seed);
        let shed_amount = shedding.exposed * lod_weight_flow * 0.28;

            // --- Multi-scale noise for organic variation (micro_a and patch_noise hoisted above) ---
            let micro_b = hash_smooth(bark.seed + 117u, theta * 7.3 - h * 5.1);
            let micro_c = hash_smooth(bark.seed + 131u, theta * 42.0 + h * 11.0);
            let micro = micro_a * 0.10 + micro_b * 0.07 + micro_c * 0.04;

            // --- Color palette: warm cinnamon bark with cool, shadowed fissures ---
            let SUNLIT_OCHRE    = vec3<f32>(0.56, 0.32, 0.14);
            let REDWOOD_HEART   = vec3<f32>(0.41, 0.18, 0.07);
            let WARM_BROWN      = vec3<f32>(0.30, 0.13, 0.05);
            let DEEP_FISSURE    = vec3<f32>(0.10, 0.06, 0.04);
            let COOL_SOOT       = vec3<f32>(0.14, 0.10, 0.09);
            let INNER_FIBER     = vec3<f32>(0.47, 0.22, 0.10);
            let INNER_RUST      = vec3<f32>(0.37, 0.15, 0.07);

            let age_factor = 1.0 - height_t;

            // Outer bark — redwood orange in light, deeper brown low on the trunk
            var outer_color = mix(REDWOOD_HEART, WARM_BROWN, age_factor * 0.36 + patch_noise * 0.18);
            outer_color = mix(outer_color, DEEP_FISSURE, age_factor * age_factor * 0.34);
            outer_color = mix(outer_color, SUNLIT_OCHRE, ridge.profile * (0.20 + (1.0 - height_t) * 0.16));
            outer_color = mix(outer_color, COOL_SOOT, patch_noise * 0.08 * age_factor);
            outer_color *= (1.0 + micro);

            // Inner bark (freshly exposed, warmer/redder)
            let inner_vary = hash_smooth(bark.seed + 50u, theta * 4.0 + h * 0.8) * 0.5 + 0.5;
            var inner_color = mix(INNER_FIBER, INNER_RUST, inner_vary);
            inner_color *= (1.0 + micro * 0.5);

            var bark_color = mix(outer_color, inner_color, shed_amount * 0.65);

            // Furrow color — darken and cool the deep channels.
            bark_color = mix(bark_color, DEEP_FISSURE, cracks.furrow_depth * 0.70);
            bark_color = mix(bark_color, COOL_SOOT, cracks.furrow_depth * 0.22);

            // Keep peeled areas subtle so the bark stays fibrous instead of papery.
            bark_color *= (1.0 - shedding.edge_shadow * lod_weight_flow * 0.35);

            // Subsurface Scattering / Translucency on Peeling Edges (Idea 6)
            let back_scatter = pow(max(dot(view_dir, -sun_dir), 0.0), 4.0);
            let sss_color = vec3<f32>(1.0, 0.4, 0.05); // Luminous fiery orange
            let edge_sss = sss_color * shedding.edge_lift * (0.25 + 0.75 * back_scatter) * lod_weight_flow;
            bark_color += edge_sss;

            // Fiber color modulation — visible everywhere via color, not just specular
            let effective_bundle = mix(0.5, bundles.bundle, lod_weight_fiber);
            bark_color *= (0.96 + 0.09 * effective_bundle);

            // --- Roughness: narrow range, mostly rough ---
            let outer_roughness = mix(0.78, 0.96, cracks.furrow_depth);
            let bark_roughness = mix(outer_roughness, 0.70, shed_amount * 0.18);

            // --- Procedural AO — strong in narrow crevices ---
            let ao_noise = hash_smooth(bark.seed + 140u, theta * 5.0 + h * 1.3) * 0.1;
            let furrow_ao = 1.0 - cracks.furrow_depth * (0.70 + ao_noise);
            let ridge_cavity = 1.0 - (1.0 - cracks.ridge_u) * 0.08;
            let bundle_ao_raw = bundles.bundle * 0.2 + 0.8;
            let bundle_ao = mix(1.0, bundle_ao_raw, lod_weight_fiber);
            let bark_ao = ao * furrow_ao * ridge_cavity * bundle_ao;

            // --- Lighting ---
            let lighting_normal = normalize(mix(normal, accumulated_normal, 0.34));
            let base_NdotL = max(dot(lighting_normal, sun_dir), 0.0);
            let diffuse_term = sun_color * sun_strength * base_NdotL * shadow;
            // Wrap lighting: fills shadow side with warm scattered light
            let wrap_dot = dot(lighting_normal, sun_dir) * 0.4 + 0.6;
            let wrap_fill = sun_color * sun_strength * 0.15 * max(wrap_dot, 0.0);

            // Specular: Anisotropic Kajiya-Kay highlights along fibers (Idea 4)
            let spec_power = mix(36.0, 10.0, bark_roughness);
            let spec_intensity = mix(0.032, 0.007, bark_roughness) * 2.0;
            let half_dir = normalize(sun_dir + view_dir);
            let fiber_tangent = normalize(tangent_theta * fiber_dir.x + tangent_h * fiber_dir.y);
            let dot_TH = dot(fiber_tangent, half_dir);
            let sin_TH = sqrt(max(1.0 - dot_TH * dot_TH, 0.0));
            let spec = pow(max(sin_TH, 0.0), spec_power) * spec_intensity;

            // Ambient: perturbed normal for subtle directional variation
        let bark_ambient = six_face_ambient(accumulated_normal) * bark_ao * 1.3;

        let full_lit = bark_color * (bark_ambient + diffuse_term + wrap_fill)
            + sun_color * spec * shadow;

        let bark_lod_blend = clamp(lod_weight_ridge, 0.0, 1.0);
        color = mix(distant_lit, full_lit, bark_lod_blend);
        final_normal = normalize(mix(normal, accumulated_normal, bark_lod_blend));
    } else {
        // === Standard PBR pipeline (foliage + ground) ===
        var roughness: f32;
        var metallic: f32;
        var f0: vec3<f32>;
        var sss_strength: f32;

        if material == MATERIAL_FOLIAGE {
            roughness = 0.55;
            metallic = 0.0;
            f0 = vec3<f32>(0.04);
            sss_strength = 0.35;
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
        let diffuse = kD * NdotL * sun_color * sun_strength * shadow;

        // Specular: Cook-Torrance GGX
        let spec = specular_brdf(normal, view_dir, sun_dir, roughness, f0);
        let specular = spec * NdotL * sun_color * sun_strength * shadow;

        // Ambient: 6-face directional probe x AO
        let ambient = six_face_ambient(normal) * ao * base_color;

        // Sky fill: hemisphere-weighted sky tint
        let sky = sky_color_dir(reflect(-view_dir, normal));
        let sky_fill = sky * 0.04 * ao;

        // Subsurface scattering for foliage
        var sss = vec3<f32>(0.0);
        if sss_strength > 0.0 {
            let sss_dot = max(dot(-normal, sun_dir), 0.0);
            let sss_term = pow(sss_dot, 3.0) * sss_strength;
            sss = sss_term * vec3<f32>(0.15, 0.25, 0.05) * sun_color * sun_strength * (1.0 - shadow * 0.7);
        }

        // Ground contact shadow & root zone
        var ground_darken = 1.0;
        if material == MATERIAL_GROUND {
            let radial = length(world_pos.xz);
            let contact = 1.0 - smoothstep(2.0, 7.8, radial);
            let root_zone = 1.0 - smoothstep(4.2, 12.5, radial);
            ground_darken = (1.0 - contact * uniforms.lighting_params.z) * (1.0 - root_zone * 0.12);
        }

        // Foliage rim
        var extra = vec3<f32>(0.0);
        if material == MATERIAL_FOLIAGE {
            let rim = pow(1.0 - max(dot(normal, view_dir), 0.0), 2.0);
            extra = rim * vec3<f32>(0.001, 0.002, 0.0015);
        }

        // Compose lighting
        color = (ambient + diffuse + specular + sky_fill + sss + extra) * ground_darken;
    }

    // -- Fog with aerial perspective --
    let fog_factor = height_fog(world_pos, cam_dist, material);
    let aerial = mix(uniforms.fog_color.rgb, sky_color_dir(normalize(world_pos - camera_pos)), 0.3);
    color = mix(color, aerial, fog_factor);

    // -- Apply exposure --
    color *= uniforms.lighting_params.x;

    var out: MaterialOutput;
    out.color = vec4<f32>(color, 1.0);
    out.normal = vec4<f32>(final_normal * 0.5 + 0.5, 1.0);
    return out;
}
