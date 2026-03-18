// Split-sum BRDF integration LUT (compute shader).
//
// Generates a 256x256 Rg16Float texture parameterized by (NdotV, roughness).
// Each texel integrates the GGX BRDF over the hemisphere using importance sampling,
// outputting scale (r) and bias (g) for the Fresnel-weighted specular term:
//   specular = prefiltered * (F0 * scale + bias)

const PI: f32 = 3.14159265359;
const SAMPLE_COUNT: u32 = 1024u;

@group(0) @binding(0) var output_lut: texture_storage_2d<rg16float, write>;

// Radical inverse (Van der Corput sequence) for Hammersley sampling.
fn radical_inverse_vdc(bits_in: u32) -> f32 {
    var bits = bits_in;
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return f32(bits) * 2.3283064365386963e-10; // / 0x100000000
}

fn hammersley(i: u32, n: u32) -> vec2<f32> {
    return vec2<f32>(f32(i) / f32(n), radical_inverse_vdc(i));
}

// GGX importance sampling: generate a half-vector H on the hemisphere.
fn importance_sample_ggx(xi: vec2<f32>, roughness: f32) -> vec3<f32> {
    let a = roughness * roughness;
    let phi = 2.0 * PI * xi.x;
    let cos_theta = sqrt((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y));
    let sin_theta = sqrt(1.0 - cos_theta * cos_theta);
    return vec3<f32>(
        cos(phi) * sin_theta,
        sin(phi) * sin_theta,
        cos_theta,
    );
}

// Smith's geometry function (GGX, height-correlated) for IBL.
// Uses k = roughness^2 / 2 (IBL remapping).
fn geometry_smith_ibl(NdotV: f32, NdotL: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let k = a / 2.0;
    let g1v = NdotV / (NdotV * (1.0 - k) + k);
    let g1l = NdotL / (NdotL * (1.0 - k) + k);
    return g1v * g1l;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(output_lut);
    if gid.x >= dims.x || gid.y >= dims.y {
        return;
    }

    let NdotV = max(f32(gid.x) / f32(dims.x - 1u), 1e-4);
    let roughness = max(f32(gid.y) / f32(dims.y - 1u), 1e-4);

    // Reference frame: N = (0, 0, 1)
    let V = vec3<f32>(sqrt(1.0 - NdotV * NdotV), 0.0, NdotV);

    var scale = 0.0;
    var bias = 0.0;

    for (var i = 0u; i < SAMPLE_COUNT; i++) {
        let xi = hammersley(i, SAMPLE_COUNT);
        let H = importance_sample_ggx(xi, roughness);
        let L = 2.0 * dot(V, H) * H - V;

        let NdotL = max(L.z, 0.0);
        let NdotH = max(H.z, 0.0);
        let VdotH = max(dot(V, H), 0.0);

        if NdotL > 0.0 {
            let G = geometry_smith_ibl(NdotV, NdotL, roughness);
            // G_Vis = G * VdotH / (NdotH * NdotV)
            let G_Vis = (G * VdotH) / max(NdotH * NdotV, 1e-7);
            let Fc = pow(1.0 - VdotH, 5.0);

            scale += G_Vis * (1.0 - Fc);
            bias += G_Vis * Fc;
        }
    }

    scale /= f32(SAMPLE_COUNT);
    bias /= f32(SAMPLE_COUNT);

    textureStore(output_lut, gid.xy, vec4<f32>(scale, bias, 0.0, 0.0));
}
