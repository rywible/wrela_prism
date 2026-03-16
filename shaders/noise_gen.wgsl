// 3D noise texture generation compute shader.
//
// Two entry points:
//   compute_shape_noise  — 128³ Rgba8Unorm (Perlin-Worley shape + Worley coverage)
//   compute_detail_noise — 32³ Rgba8Unorm (3 octaves Worley)

const PI: f32 = 3.14159265359;

// ──────────────────────── Hash Functions ────────────────────────

fn hash33(p: vec3<f32>) -> vec3<f32> {
    var q = vec3<f32>(
        dot(p, vec3<f32>(127.1, 311.7, 74.7)),
        dot(p, vec3<f32>(269.5, 183.3, 246.1)),
        dot(p, vec3<f32>(113.5, 271.9, 124.6)),
    );
    return fract(sin(q) * 43758.5453123);
}

fn hash31(p: vec3<f32>) -> f32 {
    var q = fract(p * 0.1031);
    q += dot(q, q.yzx + 33.33);
    return fract((q.x + q.y) * q.z);
}

// ──────────────────────── Worley Noise ────────────────────────

fn worley(p: vec3<f32>) -> f32 {
    let cell = floor(p);
    let frac = fract(p);
    var min_dist = 1.0;

    for (var z = -1; z <= 1; z++) {
        for (var y = -1; y <= 1; y++) {
            for (var x = -1; x <= 1; x++) {
                let offset = vec3<f32>(f32(x), f32(y), f32(z));
                let neighbor = cell + offset;
                let point = hash33(neighbor) + offset - frac;
                let dist = dot(point, point);
                min_dist = min(min_dist, dist);
            }
        }
    }
    return sqrt(min_dist);
}

// Inverted Worley — 1.0 = at cell center, 0.0 = at boundary
fn worley_inv(p: vec3<f32>) -> f32 {
    return 1.0 - worley(p);
}

// ──────────────────────── Perlin Noise ────────────────────────

fn grad(hash: f32, p: vec3<f32>) -> f32 {
    let h = u32(hash * 16.0) & 15u;
    let u_val = select(p.y, p.x, h < 8u);
    let v_val = select(select(p.z, p.x, h == 12u || h == 14u), p.y, h < 4u);
    return select(-u_val, u_val, (h & 1u) == 0u) + select(-v_val, v_val, (h & 2u) == 0u);
}

fn perlin(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    // Quintic interpolation
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);

    let n000 = grad(hash31(i + vec3<f32>(0.0, 0.0, 0.0)), f - vec3<f32>(0.0, 0.0, 0.0));
    let n100 = grad(hash31(i + vec3<f32>(1.0, 0.0, 0.0)), f - vec3<f32>(1.0, 0.0, 0.0));
    let n010 = grad(hash31(i + vec3<f32>(0.0, 1.0, 0.0)), f - vec3<f32>(0.0, 1.0, 0.0));
    let n110 = grad(hash31(i + vec3<f32>(1.0, 1.0, 0.0)), f - vec3<f32>(1.0, 1.0, 0.0));
    let n001 = grad(hash31(i + vec3<f32>(0.0, 0.0, 1.0)), f - vec3<f32>(0.0, 0.0, 1.0));
    let n101 = grad(hash31(i + vec3<f32>(1.0, 0.0, 1.0)), f - vec3<f32>(1.0, 0.0, 1.0));
    let n011 = grad(hash31(i + vec3<f32>(0.0, 1.0, 1.0)), f - vec3<f32>(0.0, 1.0, 1.0));
    let n111 = grad(hash31(i + vec3<f32>(1.0, 1.0, 1.0)), f - vec3<f32>(1.0, 1.0, 1.0));

    let x0 = mix(n000, n100, u.x);
    let x1 = mix(n010, n110, u.x);
    let x2 = mix(n001, n101, u.x);
    let x3 = mix(n011, n111, u.x);
    let y0 = mix(x0, x1, u.y);
    let y1 = mix(x2, x3, u.y);
    return mix(y0, y1, u.z);
}

// Perlin remapped to [0,1]
fn perlin01(p: vec3<f32>) -> f32 {
    return perlin(p) * 0.5 + 0.5;
}

// ──────────────────────── Perlin-Worley Hybrid ────────────────────────

// Remap utility: maps value from [low,high] to [0,1]
fn remap(value: f32, low: f32, high: f32) -> f32 {
    return clamp((value - low) / max(high - low, 0.001), 0.0, 1.0);
}

// Perlin-Worley: remap Perlin using Worley for cloud-like shapes
fn perlin_worley(p: vec3<f32>, freq_perlin: f32, freq_worley: f32) -> f32 {
    let pn = perlin01(p * freq_perlin);
    let wn = worley_inv(p * freq_worley);
    // Remap: Perlin for large-scale variation, Worley defines hard cellular edges
    return remap(pn, wn * 0.5, 1.0);
}

// ──────────────────────── FBM Helpers ────────────────────────

fn worley_fbm3(p: vec3<f32>, freq: f32) -> f32 {
    return worley_inv(p * freq) * 0.625
         + worley_inv(p * freq * 2.0) * 0.25
         + worley_inv(p * freq * 4.0) * 0.125;
}

// ──────────────────────── Shape Noise (128³) ────────────────────────
//
// R: Low-freq Perlin-Worley  (cloud shape, ~3km scale)
// G: Mid-freq Perlin-Worley  (structure, ~1km scale)
// B: Mid-freq Worley FBM     (cauliflower edge detail)
// A: Low-freq Worley          (coverage variation)

@compute @workgroup_size(4, 4, 4)
fn compute_shape_noise(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = 128u;
    if any(gid >= vec3<u32>(size)) {
        return;
    }

    let uvw = vec3<f32>(gid) / f32(size);

    let r = perlin_worley(uvw, 4.0, 4.0);
    let g = perlin_worley(uvw, 8.0, 8.0);
    let b = worley_fbm3(uvw, 10.0);
    let a = worley_fbm3(uvw, 3.0);

    textureStore(output_tex, gid, vec4<f32>(r, g, b, a));
}

// ──────────────────────── Detail Noise (32³) ────────────────────────
//
// RGB: 3 octaves of Worley at increasing frequency
// A: 1.0

@compute @workgroup_size(4, 4, 4)
fn compute_detail_noise(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = 32u;
    if any(gid >= vec3<u32>(size)) {
        return;
    }

    let uvw = vec3<f32>(gid) / f32(size);

    let r = worley_inv(uvw * 4.0);
    let g = worley_inv(uvw * 8.0);
    let b = worley_inv(uvw * 16.0);

    textureStore(output_tex, gid, vec4<f32>(r, g, b, 1.0));
}

@group(0) @binding(0) var output_tex: texture_storage_3d<rgba8unorm, write>;
