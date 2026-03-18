// Environment cubemap prefiltering with GGX importance sampling.
//
// Two entry points:
//   sky_to_cubemap  — render sky into cubemap face (mip 0)
//   prefilter_mip   — convolve cubemap for a given roughness level
//
// Uses the sky-view LUT + lower atmosphere haze for physically-based sky radiance.

const PI: f32 = 3.14159265359;

struct PrefilterParams {
    face: u32,             // cubemap face index (0-5)
    mip_level: u32,        // target mip level
    face_size: u32,        // output face resolution for this mip
    max_mip: u32,          // total mip count - 1
    sun_direction: vec4<f32>,
    sun_color: vec4<f32>,  // rgb + strength in w
    fog_color: vec4<f32>,
    sky_zenith: vec4<f32>,
    sky_horizon: vec4<f32>,
    atmosphere_params: vec4<f32>, // rayleigh_strength, mie_strength, mie_anisotropy, cloud_cover
};

@group(0) @binding(0) var<uniform> params: PrefilterParams;
@group(0) @binding(1) var sky_view_lut: texture_2d<f32>;
@group(0) @binding(2) var lut_sampler: sampler;
@group(0) @binding(3) var env_cubemap_src: texture_cube<f32>;
@group(0) @binding(4) var cubemap_sampler: sampler;
@group(0) @binding(5) var output_face: texture_storage_2d<rgba16float, write>;

// ──────────────────────── Direction helpers ────────────────────────

// Convert cubemap face + UV to world direction.
fn face_uv_to_dir(face: u32, uv: vec2<f32>) -> vec3<f32> {
    // Map UV from [0,1] to [-1,1]
    let s = uv * 2.0 - 1.0;

    switch face {
        case 0u { return normalize(vec3<f32>( 1.0,  -s.y, -s.x)); } // +X
        case 1u { return normalize(vec3<f32>(-1.0,  -s.y,  s.x)); } // -X
        case 2u { return normalize(vec3<f32>( s.x,   1.0,  s.y)); } // +Y
        case 3u { return normalize(vec3<f32>( s.x,  -1.0, -s.y)); } // -Y
        case 4u { return normalize(vec3<f32>( s.x,  -s.y,  1.0)); } // +Z
        default { return normalize(vec3<f32>(-s.x,  -s.y, -1.0)); } // -Z
    }
}

// ──────────────────────── Sky sampling ────────────────────────

fn sample_sky_lut_dir(dir: vec3<f32>) -> vec3<f32> {
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
    let sun_color = params.sun_color.rgb * params.sun_color.w;
    let cloud_cover = clamp(params.atmosphere_params.w, 0.0, 1.0);
    let sky_scale = mix(1.02, 1.10, cloud_cover);
    return textureSampleLevel(sky_view_lut, lut_sampler, vec2<f32>(u, v), 0.0).rgb
        * sun_color
        * sky_scale;
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
    let cloud_cover = clamp(params.atmosphere_params.w, 0.0, 1.0);
    let horizon = exp(-abs(dir.y) * 12.5);
    let lower_mix = smoothstep(-0.36, 0.16, dir.y);
    let sun_side = pow(horizon_alignment(dir, sun_dir), 1.35);
    let dust = params.fog_color.rgb * vec3<f32>(0.86, 0.82, 0.70) * (0.28 + 0.20 * lower_mix);
    let warm = params.sun_color.rgb * vec3<f32>(0.26, 0.18, 0.09)
        * (0.28 + 0.40 * sun_side)
        * (1.0 - cloud_cover * 0.18);
    let lift = vec3<f32>(0.04, 0.032, 0.020);
    return (dust + warm + lift) * horizon;
}

fn sample_sky_radiance(dir: vec3<f32>) -> vec3<f32> {
    let sun_dir = normalize(params.sun_direction.xyz);
    let cloud_cover = clamp(params.atmosphere_params.w, 0.0, 1.0);
    let zenith_t = pow(smoothstep(-0.08, 0.58, dir.y), 0.55);
    let horizon_t = smoothstep(-0.10, 0.18, dir.y);
    let scattering = sample_sky_lut_dir(dir) * mix(1.10, 0.98, cloud_cover);
    let lower_haze = lower_atmosphere_haze(dir, sun_dir);
    let lower_grade = mix(
        params.fog_color.rgb * vec3<f32>(0.28, 0.28, 0.22) + params.sun_color.rgb * vec3<f32>(0.08, 0.06, 0.03),
        params.sky_horizon.rgb * 0.14 + params.sun_color.rgb * vec3<f32>(0.05, 0.04, 0.03),
        horizon_t
    );
    let upper_grade = mix(params.sky_horizon.rgb * 0.06, params.sky_zenith.rgb * 0.14, zenith_t);
    let grade = mix(lower_grade, upper_grade, smoothstep(-0.02, 0.60, dir.y));
    return scattering + grade + lower_haze;
}

// ──────────────────────── Sky → Cubemap ────────────────────────

@compute @workgroup_size(8, 8, 1)
fn sky_to_cubemap(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.face_size || gid.y >= params.face_size {
        return;
    }

    let uv = (vec2<f32>(gid.xy) + 0.5) / f32(params.face_size);
    let dir = face_uv_to_dir(params.face, uv);
    let radiance = sample_sky_radiance(dir);

    textureStore(output_face, gid.xy, vec4<f32>(radiance, 1.0));
}

// ──────────────────────── GGX prefilter ────────────────────────

const PREFILTER_SAMPLES: u32 = 512u;

fn radical_inverse_vdc(bits_in: u32) -> f32 {
    var bits = bits_in;
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return f32(bits) * 2.3283064365386963e-10;
}

fn hammersley(i: u32, n: u32) -> vec2<f32> {
    return vec2<f32>(f32(i) / f32(n), radical_inverse_vdc(i));
}

fn importance_sample_ggx(xi: vec2<f32>, roughness: f32, N: vec3<f32>) -> vec3<f32> {
    let a = roughness * roughness;
    let phi = 2.0 * PI * xi.x;
    let cos_theta = sqrt((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y));
    let sin_theta = sqrt(1.0 - cos_theta * cos_theta);

    // Tangent space
    let H_tangent = vec3<f32>(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);

    // Build TBN from N
    var up: vec3<f32>;
    if abs(N.y) < 0.999 {
        up = vec3<f32>(0.0, 1.0, 0.0);
    } else {
        up = vec3<f32>(1.0, 0.0, 0.0);
    }
    let T = normalize(cross(up, N));
    let B = cross(N, T);

    return normalize(T * H_tangent.x + B * H_tangent.y + N * H_tangent.z);
}

@compute @workgroup_size(8, 8, 1)
fn prefilter_mip(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.face_size || gid.y >= params.face_size {
        return;
    }

    let uv = (vec2<f32>(gid.xy) + 0.5) / f32(params.face_size);
    let N = face_uv_to_dir(params.face, uv);
    let V = N; // Approximation: V = N for prefiltering

    let roughness = max(f32(params.mip_level) / f32(params.max_mip), 1e-4);

    var color = vec3<f32>(0.0);
    var total_weight = 0.0;

    for (var i = 0u; i < PREFILTER_SAMPLES; i++) {
        let xi = hammersley(i, PREFILTER_SAMPLES);
        let H = importance_sample_ggx(xi, roughness, N);
        let L = 2.0 * dot(V, H) * H - V;

        let NdotL = dot(N, L);
        if NdotL > 0.0 {
            // Mip bias based on PDF to reduce fireflies
            let NdotH = max(dot(N, H), 0.0);
            let a2 = roughness * roughness * roughness * roughness;
            let d = NdotH * NdotH * (a2 - 1.0) + 1.0;
            let D = a2 / (PI * d * d);
            let pdf = D * NdotH / (4.0 * max(dot(V, H), 0.001)) + 0.0001;
            let sa_texel = 4.0 * PI / (6.0 * f32(params.face_size) * f32(params.face_size));
            let sa_sample = 1.0 / (f32(PREFILTER_SAMPLES) * pdf + 0.0001);
            let mip_bias = max(0.5 * log2(sa_sample / sa_texel), 0.0);

            let sample_color = textureSampleLevel(env_cubemap_src, cubemap_sampler, L, mip_bias).rgb;
            color += sample_color * NdotL;
            total_weight += NdotL;
        }
    }

    if total_weight > 0.0 {
        color /= total_weight;
    }

    textureStore(output_face, gid.xy, vec4<f32>(color, 1.0));
}
