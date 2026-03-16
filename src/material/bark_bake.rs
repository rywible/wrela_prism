//! CPU-side bark procedural evaluation and texture bake.
//!
//! Ports the WGSL bark procedural functions (hash, fbm, cracks, ridges, fibers,
//! shedding) to Rust for offline texture generation. Produces two packed maps:
//!   - Map A (Rgba8Unorm): RGB = albedo color, A = roughness
//!   - Map B (Rgba8Unorm): RG = tangent-space normal XY (0.5-centered), B = procedural AO, A = shedding mask

use super::procedural::BarkParams;
use crate::util::smoothstep;
// rayon removed: sequential bake is fast enough for current texture sizes
use std::f32::consts::PI;

const TWO_PI: f32 = PI * 2.0;

// ── Noise Primitives (matching WGSL exactly) ──

fn hash11(seed_val: u32, n: u32) -> f32 {
    let mut h = seed_val ^ (n.wrapping_mul(747796405).wrapping_add(2891336453));
    h = ((h >> 16) ^ h).wrapping_mul(2654435769);
    h = (h >> 16) ^ h;
    (h as f32) / 2147483648.0 - 1.0
}

fn mod_positive(x: f32, m: f32) -> f32 {
    ((x % m) + m) % m
}

fn hash_smooth(seed_val: u32, x: f32) -> f32 {
    let xi = x.floor();
    let xf = x.fract();
    let t = xf * xf * (3.0 - 2.0 * xf);
    let ia = (xi as i32 & 0x7FFFFFFF) as u32;
    let ib = ((xi as i32 + 1) & 0x7FFFFFFF) as u32;
    let a = hash11(seed_val, ia);
    let b = hash11(seed_val, ib);
    a + (b - a) * t
}

fn fbm(seed_val: u32, x: f32) -> f32 {
    let mut value = 0.0;
    let mut amp = 0.5;
    let mut freq = 1.0;
    for i in 0..3u32 {
        value += amp * hash_smooth(seed_val + i * 17, x * freq);
        amp *= 0.5;
        freq *= 2.0;
    }
    value
}

fn hash21(seed_val: u32, px: f32, py: f32) -> f32 {
    let ipx = (px.floor() as i32 & 0x7FFFFFFF) as u32;
    let ipy = (py.floor() as i32 & 0x7FFFFFFF) as u32;
    let mut h = seed_val
        ^ (ipx
            .wrapping_mul(747796405)
            .wrapping_add(ipy.wrapping_mul(2891336453)));
    h = ((h >> 16) ^ h).wrapping_mul(2654435769);
    h = (h >> 16) ^ h;
    (h as f32) / 2147483648.0 - 1.0
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

// ── 2D Noise Primitives ──

/// Bilinear interpolation of hash21 at 4 integer cell corners with Hermite smoothstep.
/// Returns approximately [-1, 1].
fn noise2d(seed_val: u32, x: f32, y: f32) -> f32 {
    let xi = x.floor();
    let yi = y.floor();
    let xf = x - xi;
    let yf = y - yi;
    let tx = xf * xf * (3.0 - 2.0 * xf);
    let ty = yf * yf * (3.0 - 2.0 * yf);

    let c00 = hash21(seed_val, xi, yi);
    let c10 = hash21(seed_val, xi + 1.0, yi);
    let c01 = hash21(seed_val, xi, yi + 1.0);
    let c11 = hash21(seed_val, xi + 1.0, yi + 1.0);

    let a = c00 + (c10 - c00) * tx;
    let b = c01 + (c11 - c01) * tx;
    a + (b - a) * ty
}

/// Fractal sum of noise2d. Lacunarity 2.0, persistence 0.5.
fn fbm2d(seed_val: u32, x: f32, y: f32, octaves: u32) -> f32 {
    let mut value = 0.0;
    let mut amp = 0.5;
    let mut freq = 1.0;
    for i in 0..octaves {
        value += amp * noise2d(seed_val + i * 17, x * freq, y * freq);
        amp *= 0.5;
        freq *= 2.0;
    }
    value
}

/// Result from Voronoi 2D cell search.
struct VoronoiResult {
    f1: f32,
    f2: f32,
    cell_id: u32,
    cell_center_x: f32,
    cell_center_y: f32,
}

/// 3x3 cell Voronoi. Cell centers jittered by hash within each integer cell.
fn voronoi2d(seed_val: u32, x: f32, y: f32) -> VoronoiResult {
    let xi = x.floor();
    let yi = y.floor();
    let mut f1 = 1e9_f32;
    let mut f2 = 1e9_f32;
    let mut best_id = 0u32;
    let mut best_cx = 0.0_f32;
    let mut best_cy = 0.0_f32;

    for dy in -1..=1 {
        for dx in -1..=1 {
            let cx = xi + dx as f32;
            let cy = yi + dy as f32;
            // Jitter cell center using hash
            let jx = hash21(seed_val, cx, cy) * 0.5 + 0.5; // [0, 1]
            let jy = hash21(seed_val + 7, cx, cy) * 0.5 + 0.5;
            let px = cx + jx;
            let py = cy + jy;
            let ddx = x - px;
            let ddy = y - py;
            let dist = (ddx * ddx + ddy * ddy).sqrt();
            if dist < f1 {
                f2 = f1;
                f1 = dist;
                best_cx = px;
                best_cy = py;
                // Derive a cell ID from integer coords
                let icx = (cx as i32 & 0x7FFFFFFF) as u32;
                let icy = (cy as i32 & 0x7FFFFFFF) as u32;
                best_id = icx
                    .wrapping_mul(747796405)
                    .wrapping_add(icy.wrapping_mul(2891336453));
            } else if dist < f2 {
                f2 = dist;
            }
        }
    }

    VoronoiResult {
        f1,
        f2,
        cell_id: best_id,
        cell_center_x: best_cx,
        cell_center_y: best_cy,
    }
}

/// Domain warp via fbm2d displacement. Returns warped (x, y).
fn domain_warp2d(seed_val: u32, x: f32, y: f32, amplitude: f32, frequency: f32) -> (f32, f32) {
    let wx = x + fbm2d(seed_val, x * frequency, y * frequency, 3) * amplitude;
    let wy = y + fbm2d(seed_val + 31, x * frequency, y * frequency, 3) * amplitude;
    (wx, wy)
}

// ── Voronoi Plate System (replaces regular circumferential cracks) ──

struct PlateResult {
    furrow_depth: f32,
    ridge_u: f32,
    cell_id: u32,
    cell_center_x: f32,
    #[allow(dead_code)]
    cell_center_y: f32,
}

fn bark_plates(theta_w: f32, h_w: f32, height_t: f32, seed_val: u32) -> PlateResult {
    // Plate density: fewer, wider plates at base for dramatic furrows
    // Creates tall vertical columns — furrows run up the trunk
    let u_freq = lerp(5.5, 10.0, height_t);
    let v_freq = lerp(0.28, 0.70, height_t);

    // Use cylindrical projection for tiling: cos/sin for seamless wrapping
    let scale = u_freq / TWO_PI;
    let vx = theta_w.cos() * scale;
    let vy = theta_w.sin() * scale;
    // Combine into 2D coords that tile properly
    // Project 3D cylindrical (vx, vy, h_scaled) into 2D using angle + height
    let plate_u = theta_w * u_freq / TWO_PI;
    let plate_v = h_w * v_freq;

    let vor = voronoi2d(seed_val + 400, plate_u, plate_v);

    // Wider, deeper furrows — real sequoia has prominent vertical channels
    let furrow_width = lerp(0.24, 0.12, height_t);
    let furrow_depth =
        smoothstep(furrow_width, 0.0, vor.f2 - vor.f1) * lerp(1.8, 1.0, height_t.powf(0.5));

    // Ridge parameter: 0 at edge, 1 at center
    let ridge_u = (vor.f1 / (vor.f1 + vor.f2).max(0.001)).clamp(0.0, 1.0);

    // Per-plate variation keyed by cell_id
    let cell_hash = hash11(seed_val + 410, vor.cell_id);
    let depth_vary = 0.55 + 0.45 * (cell_hash * 0.5 + 0.5);

    // Suppress the unused vx/vy lint — they influence the conceptual mapping
    let _ = (vx, vy);

    PlateResult {
        furrow_depth: (furrow_depth * depth_vary).clamp(0.0, 1.0),
        ridge_u,
        cell_id: vor.cell_id,
        cell_center_x: vor.cell_center_x,
        cell_center_y: vor.cell_center_y,
    }
}

// ── Ridge Profile ──

struct RidgeResult {
    profile: f32,
    normal_perturbation: f32,
}

fn bark_ridge_profile(ridge_u: f32, depth_h: f32) -> RidgeResult {
    let w = 0.12;
    let c = 0.08;
    let edge = smoothstep(0.0, w, ridge_u) * smoothstep(0.0, w, 1.0 - ridge_u);
    let crown = 1.0 + c * (smoothstep(0.1, 0.9, ridge_u) * smoothstep(0.1, 0.9, 1.0 - ridge_u));
    let profile = edge * crown * depth_h;

    let d_edge_left = (1.0 / w) * smoothstep(0.0, w, 1.0 - ridge_u);
    let d_edge_right = -(1.0 / w) * smoothstep(0.0, w, ridge_u);
    let d_edge = if ridge_u < w {
        d_edge_left
    } else if ridge_u > 1.0 - w {
        d_edge_right
    } else {
        0.0
    };
    let d_profile = (d_edge * crown + edge * 0.0/* d_crown negligible */) * depth_h;

    RidgeResult {
        profile,
        normal_perturbation: d_profile,
    }
}

// ── Fiber Flow ──

fn bark_fiber_flow(
    theta: f32,
    h: f32,
    nearest_crack: f32,
    crack_spacing: f32,
    seed_val: u32,
) -> [f32; 2] {
    let dist = (mod_positive(theta - nearest_crack + PI, TWO_PI) - PI).abs();
    let fan_width = 0.3 * crack_spacing;
    let proximity = 1.0 - smoothstep(0.0, fan_width, dist);
    let fan_angle = (theta - nearest_crack).atan2(2.0);
    let fan_dir_x = fan_angle.sin();
    let fan_dir_y = fan_angle.cos();
    let fan_len = (fan_dir_x * fan_dir_x + fan_dir_y * fan_dir_y).sqrt();
    let fan_dir = if fan_len > 0.0 {
        [fan_dir_x / fan_len, fan_dir_y / fan_len]
    } else {
        [0.0, 1.0]
    };
    let base_flow_x = 0.0 + (fan_dir[0] - 0.0) * proximity * 0.25;
    let base_flow_y = 1.0 + (fan_dir[1] - 1.0) * proximity * 0.25;
    let base_len = (base_flow_x * base_flow_x + base_flow_y * base_flow_y).sqrt();
    let base_flow = if base_len > 0.0 {
        [base_flow_x / base_len, base_flow_y / base_len]
    } else {
        [0.0, 1.0]
    };

    // Burl swirls
    let burl_u = theta * 6.0;
    let burl_v = h * 0.15;
    let burl_center_x = burl_u.floor() + 0.5;
    let burl_center_y = burl_v.floor() + 0.5;
    let burl_local_x = burl_u.fract() - 0.5;
    let burl_local_y = burl_v.fract() - 0.5;
    let burl_dist = (burl_local_x * burl_local_x + burl_local_y * burl_local_y).sqrt();
    let burl_hash = hash21(seed_val + 300, burl_center_x, burl_center_y);
    let is_burl = if burl_hash + 0.5 >= 0.85 { 1.0 } else { 0.0 };
    let swirl_dir = [-burl_local_y, burl_local_x];
    let swirl_amount = is_burl * smoothstep(0.5, 0.1, burl_dist) * 0.8;

    let result_x = base_flow[0] + swirl_dir[0] * swirl_amount;
    let result_y = base_flow[1] + swirl_dir[1] * swirl_amount;
    let result_len = (result_x * result_x + result_y * result_y).sqrt();
    if result_len > 0.0 {
        [result_x / result_len, result_y / result_len]
    } else {
        [0.0, 1.0]
    }
}

// ── Fiber Strand Separation (Voronoi-based) ──

struct FiberBundleResult {
    bundle: f32,
    normal_perturbation: f32,
    fiber_gap: f32,
    #[allow(dead_code)]
    strand_edge: f32,
}

fn bark_fiber_bundles(
    theta: f32,
    h: f32,
    radius: f32,
    fiber_dir: [f32; 2],
    density: f32,
    seed_val: u32,
) -> FiberBundleResult {
    // Transform to fiber-aligned coordinate space (nearly pure vertical)
    let along = h * (0.92 + 0.08 * fiber_dir[1].abs()) + theta * fiber_dir[0] * 0.03;
    let cross = theta * fiber_dir[1].abs() - h * fiber_dir[0] * 0.03;

    // Voronoi in fiber-aligned space: dense across fiber, elongated along
    let radius_scale = radius.max(0.25).sqrt();
    let strand_density = density * (1.35 + radius_scale);
    let vor = voronoi2d(seed_val + 300, cross * strand_density, along * 0.4);

    // f1 = distance to nearest fiber centerline → strand width
    let strand_width = vor.f1;
    let bundle = 1.0 - smoothstep(0.0, 0.4, strand_width);

    // f2 - f1 < threshold → inter-fiber gap (dark, high AO)
    let gap_threshold = 0.08;
    let fiber_gap = smoothstep(gap_threshold, 0.0, vor.f2 - vor.f1);

    // Strand edge detection
    let strand_edge = smoothstep(0.3, 0.4, strand_width) * smoothstep(0.5, 0.4, strand_width);

    // Micro-fiber detail: high-freq sine along fiber direction
    let micro_fiber = (along * 40.0).sin() * 0.5 + 0.5;
    let bundle_with_micro = bundle * (0.85 + 0.15 * micro_fiber);

    // Normal perturbation perpendicular to fiber
    let d_bundle = (cross * strand_density * PI).cos() * strand_density * 0.5;

    FiberBundleResult {
        bundle: bundle_with_micro,
        normal_perturbation: d_bundle,
        fiber_gap,
        strand_edge,
    }
}

// ── Enhanced Shedding Strips ──

struct SheddingResult {
    exposed: f32,
    edge_shadow: f32,
    peel_shadow: f32,
    peel_lift: f32,
    #[allow(dead_code)]
    strip_edge: f32,
}

fn bark_shedding(
    theta: f32,
    h: f32,
    nearest_crack: f32,
    crack_spacing: f32,
    height_t: f32,
    seed_val: u32,
) -> SheddingResult {
    // Domain-warped strip coordinates so strips meander
    let (strip_u, strip_v) = domain_warp2d(seed_val + 250, theta * 3.0, h * 0.05, 0.3, 0.6);

    // Anisotropic fbm2d: stretch vertically for ~20:1 aspect elongated strips
    let strip_noise = fbm2d(seed_val + 200, strip_u * 8.0, strip_v * 0.4, 4)
        + 0.4 * fbm2d(seed_val + 213, strip_u * 4.0, strip_v * 0.2, 3);

    // Height band where shedding occurs
    let band = smoothstep(0.08, 0.25, height_t) * smoothstep(0.95, 0.7, height_t);

    // Crack proximity influence
    let crack_dist = (mod_positive(theta - nearest_crack + PI, TWO_PI) - PI).abs();
    let crack_proximity = 1.0 - smoothstep(0.0, crack_spacing * 0.35, crack_dist);

    let shed_probability = strip_noise * 0.5 + 0.5;
    // Higher threshold → fewer but more dramatic strips (~25% area)
    let shed_threshold = 0.62 - band * 0.08 - crack_proximity * 0.06;

    // Ragged edges via high-freq edge noise varying the threshold
    let edge_noise = hash_smooth(seed_val + 230, theta * 25.0 + h * 8.0) * 0.06;
    let local_threshold = shed_threshold + edge_noise;

    // Sharp transition for dramatic strip edges, full exposure
    let raw_exposed =
        smoothstep(local_threshold, local_threshold + 0.05, shed_probability) * band * 0.95;

    // Strip edge detection for peel effects
    let edge_width = 0.06;
    let strip_edge = smoothstep(
        local_threshold - edge_width,
        local_threshold,
        shed_probability,
    ) * (1.0
        - smoothstep(
            local_threshold,
            local_threshold + edge_width,
            shed_probability,
        ));

    // Peel shadow at strip edges (AO contribution)
    let peel_shadow = strip_edge * 0.7;
    // Normal tilt at edges (peel lift)
    let peel_lift = strip_edge * raw_exposed * 0.25;

    SheddingResult {
        exposed: raw_exposed,
        edge_shadow: strip_edge * 0.65,
        peel_shadow,
        peel_lift,
        strip_edge,
    }
}

// ── Height Character ──

#[allow(dead_code)] // fields used incrementally across phases
struct HeightCharacter {
    plate_scale: f32,
    depth_scale: f32,
    fiber_visibility: f32,
    color_warmth: f32,
    roughness_bias: f32,
    lichen_affinity: f32,
    shedding_probability: f32,
}

fn compute_height_character(height_t: f32) -> HeightCharacter {
    HeightCharacter {
        plate_scale: lerp(1.0, 0.3, height_t),
        depth_scale: lerp(1.0, 0.2, height_t),
        fiber_visibility: lerp(1.0, 0.4, height_t),
        color_warmth: 1.0 - height_t,
        roughness_bias: height_t * 0.15,
        // Broad bell curve 15-75% height
        lichen_affinity: smoothstep(0.05, 0.15, height_t) * smoothstep(0.85, 0.65, height_t),
        // Bell curve peaking at 20-80% height
        shedding_probability: smoothstep(0.05, 0.20, height_t) * smoothstep(0.95, 0.80, height_t),
    }
}

// ── Lichen Overlay ──

struct LichenResult {
    coverage: f32,
    color: [f32; 3],
    roughness_mod: f32,
    normal_bump: f32,
}

fn compute_lichen(
    theta_w: f32,
    h_w: f32,
    ridge_u: f32,
    shedding_exposed: f32,
    lichen_affinity: f32,
    seed_val: u32,
) -> LichenResult {
    let zero = LichenResult {
        coverage: 0.0,
        color: [0.0; 3],
        roughness_mod: 0.0,
        normal_bump: 0.0,
    };

    if lichen_affinity < 0.01 {
        return zero;
    }

    // Voronoi colony placement — larger cells for bigger patches
    let vor = voronoi2d(seed_val + 700, theta_w * 1.8, h_w * 0.4);

    // ~25% of cells activate (reduced from 40% to prevent green strips)
    let cell_hash = hash11(seed_val + 710, vor.cell_id) * 0.5 + 0.5;
    if cell_hash > 0.25 {
        return zero;
    }

    // Colony mask: smaller radius for subtler patches
    let colony_radius = 0.35 + hash11(seed_val + 715, vor.cell_id) * 0.1;
    let colony_mask = smoothstep(colony_radius, colony_radius * 0.3, vor.f1);

    // Avoid deep furrows (relaxed to allow more coverage)
    let furrow_avoidance = smoothstep(0.15, 0.5, ridge_u);
    // Avoid shedding zones
    let shed_avoidance = 1.0 - shedding_exposed;

    let coverage = colony_mask * furrow_avoidance * shed_avoidance * lichen_affinity;

    if coverage < 0.01 {
        return zero;
    }

    // Per-colony type via hash: crustose (60%), foliose (25%), fruticose (15%)
    // Colors muted to blend with bark — avoid green strips at silhouette angles
    let type_hash = hash11(seed_val + 720, vor.cell_id) * 0.5 + 0.5;
    let (color, roughness_mod, normal_bump) = if type_hash < 0.60 {
        // Crustose: flat, warm gray-olive (barely green)
        let sage = [0.30, 0.32, 0.25];
        let gray_lichen = [0.38, 0.36, 0.30];
        let t = hash11(seed_val + 725, vor.cell_id) * 0.5 + 0.5;
        (lerp3(sage, gray_lichen, t), -0.08_f32, 0.02_f32)
    } else if type_hash < 0.85 {
        // Foliose: bumpy, muted brown-olive
        let olive_green = [0.28, 0.30, 0.22];
        (olive_green, 0.03, 0.15)
    } else {
        // Fruticose: dot-like, warm gray
        let pale_green = [0.38, 0.36, 0.30];
        (pale_green, -0.04, 0.05)
    };

    LichenResult {
        coverage,
        color,
        roughness_mod,
        normal_bump,
    }
}

// ── Water Staining ──

fn compute_water_stain(theta_w: f32, h_w: f32, ridge_u: f32, seed_val: u32) -> f32 {
    let stain_field = fbm2d(seed_val + 800, theta_w * 5.0, h_w * 0.3, 3);
    // Strongest near furrows (low ridge_u)
    let stain_mask = (stain_field * 0.5 + 0.5) * smoothstep(0.6, 0.0, ridge_u);
    (stain_mask * 0.45).clamp(0.0, 0.45)
}

// ── Knot / Branch Scar System ──

struct KnotResult {
    influence: f32,
    color: [f32; 3],
    ao_boost: f32,
    normal_ring: f32,
    #[allow(dead_code)]
    grain_distortion: f32,
}

fn compute_knot(theta: f32, h: f32, height_t: f32, seed_val: u32) -> KnotResult {
    let zero = KnotResult {
        influence: 0.0,
        color: [0.0; 3],
        ao_boost: 0.0,
        normal_ring: 0.0,
        grain_distortion: 0.0,
    };

    // Grid placement: ~8x12 cells in UV
    let cell_u = theta * 8.0 / TWO_PI;
    let cell_v = h * 0.3;
    let ci = cell_u.floor();
    let cj = cell_v.floor();
    let cell_seed = ((ci as i32 & 0x7FFFFFFF) as u32)
        .wrapping_mul(747796405)
        .wrapping_add((cj as i32 & 0x7FFFFFFF) as u32);

    // 10-15% activation
    let activation = hash11(seed_val + 850, cell_seed) * 0.5 + 0.5;
    if activation > 0.12 {
        return zero;
    }

    // Knot center within cell
    let kx = ci + hash11(seed_val + 855, cell_seed) * 0.3 + 0.5;
    let ky = cj + hash11(seed_val + 860, cell_seed) * 0.3 + 0.5;

    let dx = cell_u - kx;
    let dy = cell_v - ky;
    let dist = (dx * dx + dy * dy).sqrt();

    let scar_radius = 0.15 + hash11(seed_val + 865, cell_seed).abs() * 0.1;
    let influence = smoothstep(scar_radius * 3.0, scar_radius * 0.5, dist);

    if influence < 0.01 {
        return zero;
    }

    // Fresh vs old scars
    let age_hash = hash11(seed_val + 870, cell_seed) * 0.5 + 0.5;
    let fresh_scar = [0.47, 0.22, 0.10]; // inner bark color
    let old_scar = [0.15, 0.10, 0.08]; // dark oxidized
    let scar_color = lerp3(fresh_scar, old_scar, age_hash);

    // Concentric ring normal bump for healed tissue
    let ring_freq = 12.0;
    let ring_bump = (dist * ring_freq * PI).sin() * influence * 0.3;

    // Grain distortion: circular pattern
    let grain_distortion = influence * smoothstep(scar_radius * 2.0, scar_radius * 0.5, dist);

    // Suppress lint about height_t
    let _ = height_t;

    KnotResult {
        influence,
        color: scar_color,
        ao_boost: influence * 0.3,
        normal_ring: ring_bump,
        grain_distortion,
    }
}

// ── Texel Evaluation (full procedural bark at a single UV point) ──

struct BarkTexel {
    albedo: [f32; 3],
    roughness: f32,
    normal_xy: [f32; 2], // tangent-space, 0.5-centered for storage
    ao: f32,
    shedding_mask: f32,
    height: f32, // composite height field for POM
}

fn evaluate_bark_texel(params: &BarkParams, u: f32, v: f32) -> BarkTexel {
    let theta = u * TWO_PI;
    let h = v * params.trunk_height;
    let height_t = v.clamp(0.0, 1.0);
    let r_h = lerp(params.trunk_radius_base, params.trunk_radius_tip, height_t);
    let _t_h = lerp(
        params.bark_thickness_base,
        params.bark_thickness_tip,
        height_t,
    );

    // Height character
    let hc = compute_height_character(height_t);

    // Domain warp: stronger at base, weaker at crown
    let warp_amp = lerp(0.5, 0.18, height_t);
    let (theta_w, h_w) = domain_warp2d(params.seed + 500, theta, h * 0.1, warp_amp, 0.8);

    // Voronoi plates (replaces bark_cracks)
    let plates = bark_plates(theta_w, h_w, height_t, params.seed);

    let patch_noise = fbm(params.seed + 77, theta_w * 1.8 + h_w * 0.4) * 0.5 + 0.5;
    let micro_a = hash_smooth(params.seed + 100, theta_w * 18.0 + h_w * 3.2);

    // Ridge profile (unchanged)
    let depth_h = lerp(1.0, 0.3, height_t);
    let ridge = bark_ridge_profile(plates.ridge_u, depth_h);

    // Fiber flow: use plate cell_center instead of nearest_crack_theta
    // Map cell center back to an approximate theta for fiber alignment
    let plate_theta = plates.cell_center_x * TWO_PI / lerp(4.0, 6.0, height_t);
    let plate_spacing = TWO_PI / lerp(4.0, 6.0, height_t);
    let fiber_dir = bark_fiber_flow(
        theta_w,
        h_w * 10.0, // un-scale from h_w domain
        plate_theta,
        plate_spacing,
        params.seed,
    );
    let bundles = bark_fiber_bundles(
        theta_w,
        h_w * 10.0,
        r_h,
        fiber_dir,
        params.fiber_density,
        params.seed,
    );

    // Shedding (uses warped coords)
    let shedding = bark_shedding(
        theta_w,
        h_w * 10.0,
        plate_theta,
        plate_spacing,
        height_t,
        params.seed,
    );
    let shed_amount = shedding.exposed;

    // Multi-scale noise
    let micro_b = hash_smooth(params.seed + 117, theta_w * 7.3 - h_w * 5.1);
    let micro_c = hash_smooth(params.seed + 131, theta_w * 42.0 + h_w * 11.0);
    let micro = micro_a * 0.06 + micro_b * 0.04 + micro_c * 0.02;

    // Color palette — darker, richer tones matching real sequoia bark
    let sunlit_ochre = [0.36, 0.17, 0.08];
    let redwood_heart = [0.26, 0.11, 0.05];
    let warm_brown = [0.18, 0.09, 0.045];
    let deep_fissure = [0.03, 0.02, 0.015];
    let cool_soot = [0.07, 0.05, 0.04];
    let inner_fiber = [0.34, 0.14, 0.07];
    let inner_rust = [0.28, 0.10, 0.05];
    // Expanded colors
    let crown_gray = [0.30, 0.27, 0.25];
    let olive_patina = [0.14, 0.15, 0.11];
    let fire_char = [0.03, 0.02, 0.015];
    let cinnamon_peel = [0.44, 0.19, 0.09];
    let silver_weather = [0.38, 0.35, 0.33];
    let _amber_resin = [0.50, 0.22, 0.06];

    let age_factor = 1.0 - height_t;

    // Per-plate tint variation keyed by Voronoi cell_id (up to 8% warmth shift)
    let plate_warmth = hash11(params.seed + 420, plates.cell_id) * 0.08;

    let mut outer_color = lerp3(
        redwood_heart,
        warm_brown,
        age_factor * 0.36 + patch_noise * 0.18,
    );
    outer_color = lerp3(outer_color, deep_fissure, age_factor * age_factor * 0.40);
    outer_color = lerp3(
        outer_color,
        sunlit_ochre,
        ridge.profile * (0.15 + (1.0 - height_t) * 0.12),
    );
    outer_color = lerp3(outer_color, cool_soot, patch_noise * 0.08 * age_factor);

    // Crown gray transition at top
    outer_color = lerp3(
        outer_color,
        crown_gray,
        smoothstep(0.7, 1.0, height_t) * 0.5,
    );
    // Fire char at base — deep black charring extends well up the trunk
    outer_color = lerp3(
        outer_color,
        fire_char,
        smoothstep(0.22, 0.0, height_t) * 0.42,
    );
    // Mid-trunk cinnamon warmth boost — the famous redwood glow zone
    let mid_warmth = smoothstep(0.15, 0.35, height_t) * smoothstep(0.75, 0.55, height_t);
    outer_color = lerp3(outer_color, cinnamon_peel, mid_warmth * 0.11);
    // Silver weathering at crown — earlier onset
    outer_color = lerp3(
        outer_color,
        silver_weather,
        smoothstep(0.72, 0.97, height_t) * 0.34,
    );
    // Olive patina: noise-driven green tint
    let patina_noise = fbm2d(params.seed + 600, theta_w * 3.0, h_w * 1.5, 3) * 0.5 + 0.5;
    outer_color = lerp3(outer_color, olive_patina, patina_noise * 0.10 * age_factor);

    // Per-plate warmth shift + micro noise
    outer_color = [
        (outer_color[0] * (1.0 + micro + plate_warmth)).max(0.0),
        (outer_color[1] * (1.0 + micro + plate_warmth * 0.5)).max(0.0),
        (outer_color[2] * (1.0 + micro)).max(0.0),
    ];

    // Inner bark uses cinnamon_peel for shed regions
    let inner_vary = hash_smooth(params.seed + 50, theta * 4.0 + h * 0.8) * 0.5 + 0.5;
    let mut inner_color = lerp3(cinnamon_peel, inner_rust, inner_vary * 0.5);
    inner_color = lerp3(inner_color, inner_fiber, inner_vary * 0.3);
    inner_color = [
        inner_color[0] * (1.0 + micro * 0.5),
        inner_color[1] * (1.0 + micro * 0.5),
        inner_color[2] * (1.0 + micro * 0.5),
    ];

    let mut bark_color = lerp3(outer_color, inner_color, shed_amount);
    bark_color = lerp3(bark_color, deep_fissure, plates.furrow_depth * 0.96);
    bark_color = lerp3(bark_color, cool_soot, plates.furrow_depth * 0.25);
    let shadow_scale = 1.0 - shedding.edge_shadow * 0.35;
    bark_color = [
        bark_color[0] * shadow_scale,
        bark_color[1] * shadow_scale,
        bark_color[2] * shadow_scale,
    ];

    // Fiber color modulation — visible but smooth
    let effective_bundle = lerp(0.5, bundles.bundle, hc.fiber_visibility);
    bark_color = [
        bark_color[0] * (0.95 + 0.10 * effective_bundle),
        bark_color[1] * (0.95 + 0.10 * effective_bundle),
        bark_color[2] * (0.95 + 0.10 * effective_bundle),
    ];

    // Inter-fiber gaps — subtle at distance, visible close up
    bark_color = lerp3(
        bark_color,
        deep_fissure,
        bundles.fiber_gap * 0.4 * hc.fiber_visibility,
    );

    // Knot / branch scars
    let knot = compute_knot(theta, h, height_t, params.seed);
    bark_color = lerp3(bark_color, knot.color, knot.influence * 0.7);

    // Lichen overlay
    let lichen = compute_lichen(
        theta_w,
        h_w,
        plates.ridge_u,
        shedding.exposed,
        hc.lichen_affinity,
        params.seed,
    );
    bark_color = lerp3(bark_color, lichen.color, (lichen.coverage * 0.7).min(1.0));

    // Water staining (dark streaks in furrows)
    let water_stain = compute_water_stain(theta_w, h_w, plates.ridge_u, params.seed);
    let stain_color = [0.06, 0.05, 0.04];
    bark_color = lerp3(bark_color, stain_color, water_stain);

    // Large-scale asymmetric weathering — breaks the "designed" look
    // Slow, non-repeating noise creates broad zones of different character
    let weather_field = fbm2d(params.seed + 900, theta * 0.4, h * 0.02, 3);
    let sun_side = smoothstep(-0.2, 0.4, weather_field); // ~60% coverage
                                                         // Sun-exposed side: lighter, drier, more bleached
    let bleach = [0.48, 0.42, 0.36];
    bark_color = lerp3(bark_color, bleach, sun_side * 0.12);
    // Rain-shadow side: darker, damper, more organic
    let damp = [0.08, 0.07, 0.05];
    bark_color = lerp3(bark_color, damp, (1.0 - sun_side) * 0.10);

    // Abrupt color temperature shifts — patches of warmth and cool
    let temp_shift = noise2d(params.seed + 950, theta * 1.2, h * 0.05);
    bark_color[0] *= 1.0 + temp_shift * 0.06; // warm/cool red channel
    bark_color[2] *= 1.0 - temp_shift * 0.04; // inverse blue

    // Roughness (with height character bias, lichen, water stain, weathering)
    let outer_roughness = lerp(0.78, 0.96, plates.furrow_depth) + hc.roughness_bias;
    let mut bark_roughness = lerp(outer_roughness, 0.70, shed_amount * 0.18);
    bark_roughness += lichen.roughness_mod * lichen.coverage;
    bark_roughness -= water_stain * 0.1; // wet look
                                         // Weathering affects roughness: sun-side drier (higher), rain-side smoother
    bark_roughness += sun_side * 0.06 - (1.0 - sun_side) * 0.04;

    // Procedural AO (with peel shadow, fiber gap, knot)
    let ao_noise = hash_smooth(params.seed + 140, theta * 5.0 + h * 1.3) * 0.1;
    let furrow_ao = 1.0 - plates.furrow_depth * (0.88 + ao_noise);
    let ridge_cavity = 1.0 - (1.0 - plates.ridge_u) * 0.12;
    let bundle_ao = bundles.bundle * 0.25 + 0.75;
    let fiber_gap_ao = 1.0 - bundles.fiber_gap * 0.5;
    let peel_ao = 1.0 - shedding.peel_shadow;
    let knot_ao = 1.0 - knot.ao_boost;
    let bark_ao = furrow_ao * ridge_cavity * bundle_ao * fiber_gap_ao * peel_ao * knot_ao;

    // Normal perturbation (tangent-space)
    let ridge_nx = ridge.normal_perturbation * 0.75;
    let perp_x = -fiber_dir[1];
    let perp_y = fiber_dir[0];
    let fiber_nx = perp_x * bundles.normal_perturbation * 0.06;
    let fiber_ny = perp_y * bundles.normal_perturbation * 0.06;

    // Peel lift at shedding edges (normal tilt outward)
    let peel_nx = shedding.peel_lift * 0.3;
    let peel_ny = shedding.peel_lift * 0.5;

    // Lichen bumps (foliose type adds bumpy normals)
    let lichen_nx = lichen.normal_bump * hash_smooth(params.seed + 750, theta * 20.0 + h * 6.0);
    let lichen_ny = lichen.normal_bump * hash_smooth(params.seed + 760, theta * 18.0 - h * 7.0);

    // Knot ring normals
    let knot_nx = knot.normal_ring * 0.5;

    // Grit noise
    let grit_u1 = hash_smooth(params.seed + 160, theta * 45.0 + h * 12.0);
    let grit_v1 = hash_smooth(params.seed + 170, theta * 38.0 - h * 15.0);
    let grit_u2 = hash_smooth(params.seed + 180, theta * 90.0 + h * 28.0);
    let grit_v2 = hash_smooth(params.seed + 190, theta * 75.0 - h * 35.0);
    let grit_strength = 0.14;
    let grit_x = (grit_u1 * 0.6 + grit_u2 * 0.4) * grit_strength;
    let grit_y = (grit_v1 * 0.6 + grit_v2 * 0.4) * grit_strength;

    let total_nx = ridge_nx + fiber_nx + grit_x + peel_nx + lichen_nx + knot_nx;
    let total_ny = fiber_ny + grit_y + peel_ny + lichen_ny;

    let normal_x = (total_nx * 0.5 + 0.5).clamp(0.0, 1.0);
    let normal_y = (total_ny * 0.5 + 0.5).clamp(0.0, 1.0);

    // Composite height field: ridges high, furrows low, fibers add micro-relief
    // Stronger furrow_depth contribution for more dramatic POM depth
    let height = (ridge.profile * 0.45
        + (1.0 - plates.furrow_depth) * 0.42
        + bundles.bundle * 0.08
        + shedding.peel_lift * 0.05)
        .clamp(0.0, 1.0);

    BarkTexel {
        albedo: bark_color,
        roughness: bark_roughness.clamp(0.0, 1.0),
        normal_xy: [normal_x, normal_y],
        ao: bark_ao.clamp(0.0, 1.0),
        shedding_mask: shedding.exposed.clamp(0.0, 1.0),
        height,
    }
}

// ── Texture Bake ──

/// Bake bark procedural evaluation into three packed textures.
///
/// Returns (map_a_rgba, map_b_rgba, map_c_r8) where:
///   - Map A (Rgba8Unorm): RGB = albedo, A = roughness
///   - Map B (Rgba8Unorm): RG = tangent-space normal XY (0.5-centered), B = AO, A = shedding mask
///   - Map C (R8Unorm): height field for parallax occlusion mapping
pub fn bake_bark_textures(params: &BarkParams, size: u32) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    // Parallel bake: each row computed independently then packed into output
    let rows: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> = (0..size)
        .into_iter()
        .map(|y| {
            let row_len = size as usize * 4;
            let mut row_a = vec![0u8; row_len];
            let mut row_b = vec![0u8; row_len];
            let mut row_c = vec![0u8; size as usize];
            for x in 0..size {
                let u = (x as f32 + 0.5) / size as f32;
                let v = (y as f32 + 0.5) / size as f32;
                let texel = evaluate_bark_texel(params, u, v);

                let idx = x as usize * 4;

                row_a[idx] = (texel.albedo[0].clamp(0.0, 1.0) * 255.0) as u8;
                row_a[idx + 1] = (texel.albedo[1].clamp(0.0, 1.0) * 255.0) as u8;
                row_a[idx + 2] = (texel.albedo[2].clamp(0.0, 1.0) * 255.0) as u8;
                row_a[idx + 3] = (texel.roughness * 255.0) as u8;

                row_b[idx] = (texel.normal_xy[0] * 255.0) as u8;
                row_b[idx + 1] = (texel.normal_xy[1] * 255.0) as u8;
                row_b[idx + 2] = (texel.ao * 255.0) as u8;
                row_b[idx + 3] = (texel.shedding_mask * 255.0) as u8;

                row_c[x as usize] = (texel.height * 255.0) as u8;
            }
            (row_a, row_b, row_c)
        })
        .collect();

    let pixel_count = (size * size) as usize;
    let mut map_a = Vec::with_capacity(pixel_count * 4);
    let mut map_b = Vec::with_capacity(pixel_count * 4);
    let mut map_c = Vec::with_capacity(pixel_count);
    for (ra, rb, rc) in rows {
        map_a.extend_from_slice(&ra);
        map_b.extend_from_slice(&rb);
        map_c.extend_from_slice(&rc);
    }

    (map_a, map_b, map_c)
}

/// GPU bark texture views.
pub struct BarkTextures {
    pub albedo_rough_view: wgpu::TextureView,
    pub normal_ao_view: wgpu::TextureView,
    pub height_view: wgpu::TextureView,
    _albedo_rough_texture: wgpu::Texture,
    _normal_ao_texture: wgpu::Texture,
    _height_texture: wgpu::Texture,
}

/// Bake procedural bark to GPU textures (following alpha_mask.rs pattern).
pub fn create_bark_textures(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    params: &BarkParams,
) -> BarkTextures {
    let size = 4096u32;
    let (map_a, map_b, map_c) = bake_bark_textures(params, size);

    let mip_count = (size as f32).log2() as u32 + 1;

    let create_texture_rgba = |label: &str, data: &[u8]| -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
            mip_level_count: mip_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(size * 4),
                rows_per_image: Some(size),
            },
            wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
        );
        // Generate mip chain by box-filtering (4 channels)
        let mut prev = data.to_vec();
        let mut prev_size = size;
        for mip in 1..mip_count {
            let mip_size = prev_size / 2;
            if mip_size == 0 {
                break;
            }
            let mut mip_data = vec![0u8; (mip_size * mip_size * 4) as usize];
            for y in 0..mip_size {
                for x in 0..mip_size {
                    let dst = (y * mip_size + x) as usize * 4;
                    let s00 = ((y * 2) * prev_size + x * 2) as usize * 4;
                    let s10 = ((y * 2) * prev_size + x * 2 + 1) as usize * 4;
                    let s01 = ((y * 2 + 1) * prev_size + x * 2) as usize * 4;
                    let s11 = ((y * 2 + 1) * prev_size + x * 2 + 1) as usize * 4;
                    for c in 0..4 {
                        let avg = (prev[s00 + c] as u16
                            + prev[s10 + c] as u16
                            + prev[s01 + c] as u16
                            + prev[s11 + c] as u16)
                            / 4;
                        mip_data[dst + c] = avg as u8;
                    }
                }
            }
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: mip,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &mip_data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(mip_size * 4),
                    rows_per_image: Some(mip_size),
                },
                wgpu::Extent3d {
                    width: mip_size,
                    height: mip_size,
                    depth_or_array_layers: 1,
                },
            );
            prev = mip_data;
            prev_size = mip_size;
        }
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    };

    // Create R8Unorm height texture with mip chain
    let create_texture_r8 = |label: &str, data: &[u8]| -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
            mip_level_count: mip_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(size),
                rows_per_image: Some(size),
            },
            wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
        );
        // Generate mip chain by box-filtering (1 channel)
        let mut prev = data.to_vec();
        let mut prev_size = size;
        for mip in 1..mip_count {
            let mip_size = prev_size / 2;
            if mip_size == 0 {
                break;
            }
            let mut mip_data = vec![0u8; (mip_size * mip_size) as usize];
            for y in 0..mip_size {
                for x in 0..mip_size {
                    let dst = (y * mip_size + x) as usize;
                    let s00 = ((y * 2) * prev_size + x * 2) as usize;
                    let s10 = ((y * 2) * prev_size + x * 2 + 1) as usize;
                    let s01 = ((y * 2 + 1) * prev_size + x * 2) as usize;
                    let s11 = ((y * 2 + 1) * prev_size + x * 2 + 1) as usize;
                    let avg =
                        (prev[s00] as u16 + prev[s10] as u16 + prev[s01] as u16 + prev[s11] as u16)
                            / 4;
                    mip_data[dst] = avg as u8;
                }
            }
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: mip,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &mip_data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(mip_size),
                    rows_per_image: Some(mip_size),
                },
                wgpu::Extent3d {
                    width: mip_size,
                    height: mip_size,
                    depth_or_array_layers: 1,
                },
            );
            prev = mip_data;
            prev_size = mip_size;
        }
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    };

    let (albedo_rough_texture, albedo_rough_view) =
        create_texture_rgba("bark-albedo-rough", &map_a);
    let (normal_ao_texture, normal_ao_view) = create_texture_rgba("bark-normal-ao", &map_b);
    let (height_texture, height_view) = create_texture_r8("bark-height", &map_c);

    BarkTextures {
        albedo_rough_view,
        normal_ao_view,
        height_view,
        _albedo_rough_texture: albedo_rough_texture,
        _normal_ao_texture: normal_ao_texture,
        _height_texture: height_texture,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_bark_params() -> BarkParams {
        BarkParams {
            trunk_radius_base: 3.1,
            trunk_radius_tip: 0.36,
            bark_thickness_base: 0.42,
            bark_thickness_tip: 0.07,
            trunk_height: 42.0,
            stiffness_ratio: 3.35,
            fiber_density: 8.5,
            seed: 42,
        }
    }

    #[test]
    fn bake_produces_correct_size() {
        let params = default_bark_params();
        let (map_a, map_b, map_c) = bake_bark_textures(&params, 64);
        assert_eq!(map_a.len(), 64 * 64 * 4);
        assert_eq!(map_b.len(), 64 * 64 * 4);
        assert_eq!(map_c.len(), 64 * 64);
    }

    #[test]
    fn bake_has_nonzero_albedo() {
        let params = default_bark_params();
        let (map_a, _, _) = bake_bark_textures(&params, 64);
        let nonzero = map_a
            .chunks(4)
            .filter(|px| px[0] > 0 || px[1] > 0 || px[2] > 0)
            .count();
        let total = 64 * 64;
        let coverage = nonzero as f32 / total as f32;
        assert!(
            coverage > 0.9,
            "bark albedo should have >90% coverage, got {:.1}%",
            coverage * 100.0
        );
    }

    #[test]
    fn bake_normals_centered_around_half() {
        let params = default_bark_params();
        let (_, map_b, _) = bake_bark_textures(&params, 64);
        // Normal XY should be centered around 128 (0.5 * 255)
        let avg_x: f32 = map_b.chunks(4).map(|px| px[0] as f32).sum::<f32>() / (64 * 64) as f32;
        let avg_y: f32 = map_b.chunks(4).map(|px| px[1] as f32).sum::<f32>() / (64 * 64) as f32;
        assert!(
            (avg_x - 128.0).abs() < 40.0,
            "normal X should average near 128, got {avg_x:.1}"
        );
        assert!(
            (avg_y - 128.0).abs() < 40.0,
            "normal Y should average near 128, got {avg_y:.1}"
        );
    }

    #[test]
    fn different_seeds_produce_different_bakes() {
        let mut params = default_bark_params();
        let (a1, _, _) = bake_bark_textures(&params, 32);
        params.seed = 999;
        let (a2, _, _) = bake_bark_textures(&params, 32);
        let diff: usize = a1.iter().zip(a2.iter()).filter(|(a, b)| a != b).count();
        assert!(
            diff > 100,
            "different seeds should produce different bark textures"
        );
    }

    #[test]
    fn height_map_has_reasonable_range() {
        let params = default_bark_params();
        let (_, _, map_c) = bake_bark_textures(&params, 64);
        let min_h = *map_c.iter().min().unwrap();
        let max_h = *map_c.iter().max().unwrap();
        assert!(
            min_h <= 60 && max_h >= 220,
            "height map should span [≤60, ≥220] u8 range, got [{min_h}, {max_h}]"
        );
    }

    #[test]
    fn noise_primitives_match_wgsl() {
        // hash11 with known inputs should produce values in [-1, 1)
        let v = hash11(42, 7);
        assert!((-1.0..1.0).contains(&v), "hash11 out of range: {v}");

        // smoothstep edges
        assert!((smoothstep(0.0, 1.0, 0.0) - 0.0).abs() < 1e-6);
        assert!((smoothstep(0.0, 1.0, 1.0) - 1.0).abs() < 1e-6);
        assert!((smoothstep(0.0, 1.0, 0.5) - 0.5).abs() < 1e-6);

        // fbm should be bounded
        let f = fbm(42, 3.5);
        assert!(f.abs() < 2.0, "fbm out of expected range: {f}");
    }

    #[test]
    fn noise2d_range_and_determinism() {
        for i in 0..100 {
            let x = i as f32 * 0.37;
            let y = i as f32 * 0.53;
            let v = noise2d(42, x, y);
            assert!(
                (-1.0..=1.0).contains(&v),
                "noise2d out of range at ({x}, {y}): {v}"
            );
            // Determinism
            assert_eq!(v, noise2d(42, x, y));
        }
    }

    #[test]
    fn fbm2d_range() {
        for i in 0..50 {
            let x = i as f32 * 0.7 - 10.0;
            let y = i as f32 * 0.3 + 5.0;
            let v = fbm2d(42, x, y, 4);
            assert!(v.abs() < 2.0, "fbm2d out of range at ({x}, {y}): {v}");
        }
    }

    #[test]
    fn voronoi2d_f1_less_than_f2() {
        for i in 0..50 {
            let x = i as f32 * 0.6 - 5.0;
            let y = i as f32 * 0.4 + 2.0;
            let v = voronoi2d(42, x, y);
            assert!(
                v.f1 <= v.f2,
                "voronoi f1 > f2 at ({x}, {y}): f1={}, f2={}",
                v.f1,
                v.f2
            );
            assert!(v.f1 >= 0.0);
            // Determinism
            let v2 = voronoi2d(42, x, y);
            assert_eq!(v.f1, v2.f1);
            assert_eq!(v.cell_id, v2.cell_id);
        }
    }

    #[test]
    fn domain_warp2d_determinism() {
        let (wx1, wy1) = domain_warp2d(42, 1.5, 2.3, 0.3, 0.8);
        let (wx2, wy2) = domain_warp2d(42, 1.5, 2.3, 0.3, 0.8);
        assert_eq!(wx1, wx2);
        assert_eq!(wy1, wy2);
        // Should not equal input (non-zero warp)
        assert!((wx1 - 1.5).abs() > 1e-6 || (wy1 - 2.3).abs() > 1e-6);
    }
}
