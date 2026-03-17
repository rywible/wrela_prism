use glam::{Vec2, Vec3};

use crate::scene::{SdfPrimitive, SdfTree};
use crate::util::smoothstep;

// ──────────────────────── Inline PRNG ────────────────────────

/// Splitmix64 PRNG — deterministic, no external dependency.
pub(crate) struct TreeRng {
    state: u64,
}

impl TreeRng {
    pub(crate) fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    pub(crate) fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    pub(crate) fn next_f32_range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.next_f32() * (hi - lo)
    }
}

// ──────────────────────── Parameters ────────────────────────

/// Growth parameters for procedural redwood generation.
#[derive(Clone, Debug)]
pub struct RedwoodParams {
    pub seed: u64,
    pub trunk_height: f32,
    pub base_radius: f32,
    pub tip_radius: f32,
    pub trunk_segments: usize,
    pub trunk_wobble: f32,
    pub crown_start_frac: f32,
    pub branch_count: usize,
    pub max_branch_depth: u8,
    pub smooth_k_trunk: f32,
    pub smooth_k_branch: f32,
    pub smooth_k_fine: f32,
    pub columnar_fraction: f32,
    pub flute_count: u32,
    pub flute_depth: f32,
}

impl Default for RedwoodParams {
    fn default() -> Self {
        Self {
            seed: 42,
            trunk_height: 65.0,
            base_radius: 4.5,
            tip_radius: 0.55,
            trunk_segments: 85,
            trunk_wobble: 0.07,
            crown_start_frac: 0.38,
            branch_count: 95,
            max_branch_depth: 5,
            smooth_k_trunk: 0.30,
            smooth_k_branch: 0.06,
            smooth_k_fine: 0.03,
            columnar_fraction: 0.70,
            flute_count: 8,
            flute_depth: 0.20,
        }
    }
}

// ──────────────────────── Skeleton ────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BranchKind {
    Trunk,
    LiveScaffold,
    LiveSecondary,
    LiveFine,
    DeadStub,
}

impl BranchKind {
    pub(crate) fn is_live_foliage(self) -> bool {
        matches!(
            self,
            Self::LiveScaffold | Self::LiveSecondary | Self::LiveFine
        )
    }

    pub(crate) fn is_structural_trunk(self) -> bool {
        matches!(self, Self::Trunk)
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct SkeletonNode {
    pub(crate) position: Vec3,
    pub(crate) radius: f32,
    pub(crate) parent: Option<usize>,
    pub(crate) depth: u8,
    pub(crate) kind: BranchKind,
    pub(crate) lobe_id: Option<u8>,
}

pub(crate) struct TreeSkeleton {
    pub(crate) nodes: Vec<SkeletonNode>,
}

impl TreeSkeleton {
    pub(crate) fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    pub(crate) fn push(&mut self, node: SkeletonNode) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(node);
        idx
    }
}

#[derive(Clone, Copy, Debug)]
enum PrimaryStyle {
    LowerScaffold,
    UpperSecondary,
    FillerLive,
    DeadStub,
}

impl PrimaryStyle {
    fn is_live(self) -> bool {
        !matches!(self, Self::DeadStub)
    }
}

#[derive(Clone, Copy, Debug)]
struct PrimaryBranchSpec {
    style: PrimaryStyle,
    trunk_frac: f32,
    azimuth: f32,
    pitch_deg: f32,
    length: f32,
    lobe_id: Option<u8>,
}

#[derive(Clone, Copy, Debug)]
struct CanopyRegionSeed {
    anchor: Vec3,
    radius: f32,
    axis: Vec3,
    _interior_weight: f32,
    lobe_id: u8,
    score: f32,
}

// ──────────────────────── Phase 1: Trunk ────────────────────────

pub(crate) fn build_skeleton(params: &RedwoodParams) -> TreeSkeleton {
    let mut rng = TreeRng::new(params.seed);
    let mut skeleton = TreeSkeleton::new();
    grow_trunk(&mut skeleton, params, &mut rng);
    grow_branches(&mut skeleton, params, &mut rng);
    skeleton
}

fn grow_trunk(skeleton: &mut TreeSkeleton, params: &RedwoodParams, rng: &mut TreeRng) {
    let lean_angle = rng.next_f32() * std::f32::consts::TAU;
    let lean_mag = rng.next_f32_range(0.02, 0.22);
    let lean_dir = Vec3::new(lean_angle.cos(), 0.0, lean_angle.sin());

    let mut wobble_x = 0.0;
    let mut wobble_z = 0.0;
    const SMOOTH_ALPHA: f32 = 0.24;

    skeleton.push(SkeletonNode {
        position: Vec3::ZERO,
        radius: params.base_radius,
        parent: None,
        depth: 0,
        kind: BranchKind::Trunk,
        lobe_id: None,
    });

    for i in 1..=params.trunk_segments {
        let t = i as f32 / params.trunk_segments as f32;
        let y = t * params.trunk_height;

        // Two-phase columnar taper: lower portion stays nearly constant,
        // upper portion tapers more steeply to the tip.
        let cf = params.columnar_fraction;
        let radius = if t <= cf {
            // Lower columnar section: gentle taper from 1.0 to 0.85 of base
            let s = smoothstep(0.0, cf, t);
            let factor = 1.0 - 0.15 * s; // lerp(1.0, 0.85, s)
            params.base_radius * factor
        } else {
            // Upper section: quadratic taper from 0.85*base to tip
            let t_upper = (t - cf) / (1.0 - cf);
            let r_at_crown = params.base_radius * 0.85;
            r_at_crown + (params.tip_radius - r_at_crown) * t_upper.powf(2.0)
        };

        let raw_x = rng.next_f32_range(-params.trunk_wobble, params.trunk_wobble);
        let raw_z = rng.next_f32_range(-params.trunk_wobble, params.trunk_wobble);
        wobble_x = wobble_x * (1.0 - SMOOTH_ALPHA) + raw_x * SMOOTH_ALPHA;
        wobble_z = wobble_z * (1.0 - SMOOTH_ALPHA) + raw_z * SMOOTH_ALPHA;

        let lean_offset = lean_dir * lean_mag * t * t;
        let pos = Vec3::new(wobble_x + lean_offset.x, y, wobble_z + lean_offset.z);

        skeleton.push(SkeletonNode {
            position: pos,
            radius,
            parent: Some(i - 1),
            depth: 0,
            kind: BranchKind::Trunk,
            lobe_id: None,
        });
    }
}

// ──────────────────────── Phase 2: Branches ────────────────────────

fn grow_branches(skeleton: &mut TreeSkeleton, params: &RedwoodParams, rng: &mut TreeRng) {
    let trunk_node_count = skeleton
        .nodes
        .iter()
        .filter(|node| matches!(node.kind, BranchKind::Trunk))
        .count();

    let lobe_angles = build_lobe_angles(rng);
    let specs = build_primary_branch_specs(params, &lobe_angles, rng);
    for spec in specs {
        grow_primary_branch(skeleton, params, rng, trunk_node_count, spec);
    }

    // Apex crown: 14-22 branches filling the top of the crown
    let apex_count = 14 + (rng.next_u64() % 9) as usize;
    for i in 0..apex_count {
        let lobe_id = (i % lobe_angles.len().max(1)) as u8;
        let apex_t = i as f32 / apex_count.max(1) as f32;
        let apex_spec = PrimaryBranchSpec {
            style: PrimaryStyle::FillerLive,
            // Distribute from 0.85 to 0.99 along trunk
            trunk_frac: 0.85 + apex_t * 0.14,
            azimuth: wrap_angle(lobe_angles[lobe_id as usize] + rng.next_f32_range(-0.50, 0.50)),
            // Higher branches point more steeply upward
            pitch_deg: rng.next_f32_range(25.0, 50.0) + apex_t * 20.0,
            length: params.trunk_height * rng.next_f32_range(0.03, 0.08) * (1.0 - apex_t * 0.4),
            lobe_id: Some(lobe_id),
        };
        grow_primary_branch(skeleton, params, rng, trunk_node_count, apex_spec);
    }
}

fn build_lobe_angles(rng: &mut TreeRng) -> Vec<f32> {
    // More lobes for denser, more uniform coverage
    let lobe_count = 10 + (rng.next_u64() % 3) as usize;
    let gap_count = 1 + usize::from(rng.next_f32() < 0.40);
    let gap_size = rng.next_f32_range(0.18, 0.35);
    let base_angle = rng.next_f32() * std::f32::consts::TAU;
    let mut gap_indices = Vec::with_capacity(gap_count);
    while gap_indices.len() < gap_count {
        let candidate = (rng.next_u64() % lobe_count as u64) as usize;
        if !gap_indices.contains(&candidate) {
            gap_indices.push(candidate);
        }
    }
    gap_indices.sort_unstable();

    let base_step = (std::f32::consts::TAU - gap_size * gap_count as f32) / lobe_count as f32;
    let mut lobe_angles = Vec::with_capacity(lobe_count);
    let mut angle = base_angle;
    for i in 0..lobe_count {
        lobe_angles.push(wrap_angle(angle + rng.next_f32_range(-0.10, 0.10)));
        angle += base_step;
        if gap_indices.binary_search(&i).is_ok() {
            angle += gap_size;
        }
    }
    lobe_angles
}

fn build_primary_branch_specs(
    params: &RedwoodParams,
    lobe_angles: &[f32],
    rng: &mut TreeRng,
) -> Vec<PrimaryBranchSpec> {
    let lobe_count = lobe_angles.len().max(1);
    let desired_lower = 16 + (rng.next_u64() % 4) as usize;
    let desired_upper = 35 + (rng.next_u64() % 6) as usize;
    let desired_stub = 8 + (rng.next_u64() % 3) as usize;

    let lower_count = desired_lower.min(params.branch_count.max(1));
    let remaining_after_lower = params.branch_count.saturating_sub(lower_count);
    let upper_count = desired_upper
        .min(remaining_after_lower.saturating_sub(desired_stub.min(remaining_after_lower)));
    let remaining_after_upper = params
        .branch_count
        .saturating_sub(lower_count + upper_count);
    let stub_count = desired_stub.min(remaining_after_upper);
    let filler_count = params
        .branch_count
        .saturating_sub(lower_count + upper_count + stub_count);

    let mut specs = Vec::with_capacity(params.branch_count);

    for i in 0..lower_count {
        let lower_t = if lower_count > 1 {
            i as f32 / (lower_count - 1) as f32
        } else {
            0.0
        };
        let lobe_id = (i % lobe_count) as u8;
        let height_frac = params.crown_start_frac + lower_t * 0.25;
        specs.push(PrimaryBranchSpec {
            style: PrimaryStyle::LowerScaffold,
            trunk_frac: (height_frac + rng.next_f32_range(-0.015, 0.015))
                .clamp(params.crown_start_frac, params.crown_start_frac + 0.28),
            azimuth: wrap_angle(lobe_angles[lobe_id as usize] + rng.next_f32_range(-0.35, 0.35)),
            pitch_deg: rng.next_f32_range(-5.0, 12.0),
            length: params.trunk_height
                * crown_envelope(height_frac, params.crown_start_frac)
                * 0.15
                * rng.next_f32_range(0.85, 1.15),
            lobe_id: Some(lobe_id),
        });
    }

    for i in 0..upper_count {
        let upper_t = (i as f32 + 0.5) / upper_count.max(1) as f32;
        let lobe_id = ((i + lower_count) % lobe_count) as u8;
        let upper_start = params.crown_start_frac + 0.25;
        let height_frac = upper_start + upper_t * (0.92 - upper_start);
        specs.push(PrimaryBranchSpec {
            style: PrimaryStyle::UpperSecondary,
            trunk_frac: (height_frac + rng.next_f32_range(-0.015, 0.015)).clamp(upper_start, 0.92),
            azimuth: wrap_angle(lobe_angles[lobe_id as usize] + rng.next_f32_range(-0.28, 0.28)),
            pitch_deg: rng.next_f32_range(10.0, 35.0) + upper_t * 15.0,
            length: params.trunk_height
                * crown_envelope(height_frac, params.crown_start_frac)
                * 0.13
                * rng.next_f32_range(0.85, 1.15),
            lobe_id: Some(lobe_id),
        });
    }

    for i in 0..filler_count {
        let filler_t = (i as f32 + 0.5) / filler_count.max(1) as f32;
        let lobe_id = ((i * 2 + 1) % lobe_count) as u8;
        let height_frac = 0.88 + filler_t * 0.10;
        specs.push(PrimaryBranchSpec {
            style: PrimaryStyle::FillerLive,
            trunk_frac: (height_frac + rng.next_f32_range(-0.016, 0.016)).clamp(0.86, 0.98),
            azimuth: wrap_angle(lobe_angles[lobe_id as usize] + rng.next_f32_range(-0.30, 0.30)),
            pitch_deg: rng.next_f32_range(35.0, 65.0),
            length: params.trunk_height
                * crown_envelope(height_frac, params.crown_start_frac)
                * 0.09
                * rng.next_f32_range(0.85, 1.15),
            lobe_id: Some(lobe_id),
        });
    }

    for _ in 0..stub_count {
        specs.push(PrimaryBranchSpec {
            style: PrimaryStyle::DeadStub,
            trunk_frac: rng.next_f32_range(0.15, 0.58),
            azimuth: rng.next_f32() * std::f32::consts::TAU,
            pitch_deg: rng.next_f32_range(-18.0, -5.0),
            length: params.trunk_height * rng.next_f32_range(0.04, 0.08),
            lobe_id: None,
        });
    }

    specs.sort_by(|a, b| a.trunk_frac.total_cmp(&b.trunk_frac));
    specs
}

/// Conical crown envelope: widest near crown base, tapering toward top.
/// Returns a scale factor [0, 1] for branch length at a given trunk fraction.
/// Uses a softer taper to produce a fuller, more natural crown silhouette.
fn crown_envelope(height_frac: f32, crown_start_frac: f32) -> f32 {
    if height_frac < crown_start_frac {
        return 0.0;
    }
    let t = (height_frac - crown_start_frac) / (1.0 - crown_start_frac);
    // Softer taper: stays wider longer, then narrows near top
    (1.0 - t * t).max(0.0)
}

fn grow_primary_branch(
    skeleton: &mut TreeSkeleton,
    params: &RedwoodParams,
    rng: &mut TreeRng,
    trunk_node_count: usize,
    spec: PrimaryBranchSpec,
) {
    let trunk_idx = find_trunk_node_at_frac(skeleton, spec.trunk_frac, params, trunk_node_count);
    let trunk_pos = skeleton.nodes[trunk_idx].position;
    let trunk_r = skeleton.nodes[trunk_idx].radius;

    let (
        seg_count,
        base_radius_factor,
        radius_curve,
        lift_per_seg,
        droop_per_seg,
        jitter_scale,
        branch_kind,
        vigor,
    ) = match spec.style {
        PrimaryStyle::LowerScaffold => (
            8usize,
            0.18f32,
            1.50f32,
            0.012f32,
            0.018f32,
            0.05f32,
            BranchKind::LiveScaffold,
            1.0f32,
        ),
        PrimaryStyle::UpperSecondary => (
            6usize,
            0.10f32,
            1.50f32,
            0.035f32,
            0.008f32,
            0.04f32,
            BranchKind::LiveSecondary,
            0.90f32,
        ),
        PrimaryStyle::FillerLive => (
            4usize,
            0.07f32,
            1.40f32,
            0.030f32,
            0.006f32,
            0.04f32,
            BranchKind::LiveSecondary,
            0.78f32,
        ),
        PrimaryStyle::DeadStub => (
            3usize,
            0.05f32,
            1.20f32,
            -0.015f32,
            0.040f32,
            0.04f32,
            BranchKind::DeadStub,
            0.0f32,
        ),
    };

    let pitch = spec.pitch_deg.to_radians();
    let mut dir = Vec3::new(
        spec.azimuth.cos() * pitch.cos(),
        pitch.sin(),
        spec.azimuth.sin() * pitch.cos(),
    )
    .normalize_or_zero();

    let mut prev_idx = trunk_idx;
    let mut prev_pos = trunk_pos;
    let seg_len = spec.length / seg_count as f32;
    let base_radius = trunk_r * base_radius_factor;

    for seg_i in 0..seg_count {
        let seg_t = (seg_i + 1) as f32 / seg_count as f32;
        let radial = Vec3::new(dir.x, 0.0, dir.z).normalize_or_zero();
        let swirl = Vec3::new(-radial.z, 0.0, radial.x)
            * rng.next_f32_range(-jitter_scale, jitter_scale)
            * 0.5;
        let lift = lift_per_seg - droop_per_seg * seg_t + rng.next_f32_range(-0.018, 0.022);
        let lateral_jitter = Vec3::new(
            rng.next_f32_range(-jitter_scale, jitter_scale),
            0.0,
            rng.next_f32_range(-jitter_scale, jitter_scale),
        ) * 0.35;
        dir = (dir + Vec3::Y * lift + swirl + lateral_jitter).normalize_or_zero();

        let pos = prev_pos + dir * seg_len;
        let radius = base_radius * (1.0 - seg_t).powf(radius_curve) + params.tip_radius * 0.04;
        let node_idx = skeleton.push(SkeletonNode {
            position: pos,
            radius,
            parent: Some(prev_idx),
            depth: 1,
            kind: branch_kind,
            lobe_id: spec.lobe_id,
        });

        if spec.style.is_live() && params.max_branch_depth > 1 {
            let spawn_here = match spec.style {
                PrimaryStyle::LowerScaffold => seg_i >= 1,
                PrimaryStyle::UpperSecondary => {
                    seg_i + 1 == seg_count || (seg_i >= 1 && seg_count > 2)
                }
                PrimaryStyle::FillerLive => {
                    seg_i + 1 == seg_count || (seg_i == 0 && seg_count > 1 && rng.next_f32() < 0.35)
                }
                PrimaryStyle::DeadStub => false,
            };
            if spawn_here {
                let child_length = spec.length * (0.58 - seg_t * 0.09).max(0.24);
                grow_live_sub_branches(
                    skeleton,
                    node_idx,
                    2,
                    child_length,
                    params,
                    rng,
                    spec.lobe_id.unwrap_or(0),
                    vigor,
                );
            }
        }

        prev_idx = node_idx;
        prev_pos = pos;
    }
}

fn grow_live_sub_branches(
    skeleton: &mut TreeSkeleton,
    parent_idx: usize,
    depth: u8,
    parent_length: f32,
    params: &RedwoodParams,
    rng: &mut TreeRng,
    lobe_id: u8,
    vigor: f32,
) {
    if depth > params.max_branch_depth || parent_length < 0.6 {
        return;
    }

    let child_count = match depth {
        2 => {
            if vigor > 0.85 {
                2 + usize::from(rng.next_f32() < 0.50)
            } else if rng.next_f32() < 0.80 {
                1 + usize::from(rng.next_f32() < 0.35 * vigor.max(0.5))
            } else {
                1
            }
        }
        3 => {
            if rng.next_f32() < 0.72 * vigor.max(0.48) {
                1 + usize::from(rng.next_f32() < 0.25)
            } else {
                0
            }
        }
        _ => {
            if rng.next_f32() < 0.55 * vigor.max(0.45) {
                1
            } else {
                0
            }
        }
    };
    if child_count == 0 {
        return;
    }

    let parent_pos = skeleton.nodes[parent_idx].position;
    let parent_r = skeleton.nodes[parent_idx].radius;
    let parent_dir = segment_direction(skeleton, parent_idx).unwrap_or(Vec3::Y);
    let basis_up = if parent_dir.y.abs() > 0.85 {
        Vec3::X
    } else {
        Vec3::Y
    };
    let right = parent_dir.cross(basis_up).normalize_or_zero();
    let up = right.cross(parent_dir).normalize_or_zero();

    for child_i in 0..child_count {
        let spread = if depth == 2 { 0.72 } else { 0.92 };
        let lateral =
            right * rng.next_f32_range(-spread, spread) + up * rng.next_f32_range(0.05, 0.28);
        let mut dir = (parent_dir * rng.next_f32_range(0.75, 1.05)
            + lateral
            + Vec3::Y * rng.next_f32_range(-0.04, 0.14))
        .normalize_or_zero();
        if depth == 2 {
            dir.y += rng.next_f32_range(0.01, 0.08);
        } else {
            dir.y -= rng.next_f32_range(0.0, 0.06);
        }

        let seg_count = match depth {
            2 => 4,
            3 => 3,
            4..=5 => 2,
            _ => 2,
        };
        let base_radius = match depth {
            2 => parent_r * 0.65,
            3 => parent_r * 0.50,
            4..=5 => parent_r * 0.35,
            _ => parent_r * 0.35,
        };
        let child_len = parent_length
            * rng.next_f32_range(
                if depth == 2 { 0.68 } else { 0.46 },
                if depth == 2 { 0.92 } else { 0.68 },
            )
            * vigor.max(0.55);
        let seg_len = child_len / seg_count as f32;

        let mut prev_idx = parent_idx;
        let mut prev_pos = parent_pos;
        let mut branch_dir = dir;

        for seg_i in 0..seg_count {
            let seg_t = (seg_i + 1) as f32 / seg_count as f32;
            branch_dir = (branch_dir
                + right * rng.next_f32_range(-0.10, 0.10) * 0.4
                + Vec3::Y * rng.next_f32_range(-0.05, 0.08))
            .normalize_or_zero();
            let pos = prev_pos + branch_dir * seg_len;
            let radius = base_radius * (1.0 - seg_t).powf(1.40) + params.tip_radius * 0.02;
            let node_idx = skeleton.push(SkeletonNode {
                position: pos,
                radius,
                parent: Some(prev_idx),
                depth,
                kind: BranchKind::LiveFine,
                lobe_id: Some(lobe_id),
            });
            prev_idx = node_idx;
            prev_pos = pos;
        }

        if depth < params.max_branch_depth {
            let recurse_strength = vigor * if child_i == 0 { 0.75 } else { 0.55 };
            if recurse_strength > 0.28 && rng.next_f32() < 0.86 {
                grow_live_sub_branches(
                    skeleton,
                    prev_idx,
                    depth + 1,
                    child_len * 0.58,
                    params,
                    rng,
                    lobe_id,
                    recurse_strength,
                );
            }
        }
    }
}

fn segment_direction(skeleton: &TreeSkeleton, node_idx: usize) -> Option<Vec3> {
    let node = skeleton.nodes.get(node_idx)?;
    let parent_idx = node.parent?;
    Some((node.position - skeleton.nodes[parent_idx].position).normalize_or_zero())
}

fn find_trunk_node_at_frac(
    skeleton: &TreeSkeleton,
    t: f32,
    params: &RedwoodParams,
    trunk_count: usize,
) -> usize {
    let target_y = t * params.trunk_height;
    let mut best_idx = 0;
    let mut best_dist = f32::MAX;
    for i in 0..trunk_count {
        let d = (skeleton.nodes[i].position.y - target_y).abs();
        if d < best_dist {
            best_dist = d;
            best_idx = i;
        }
    }
    best_idx
}

fn wrap_angle(angle: f32) -> f32 {
    angle.rem_euclid(std::f32::consts::TAU)
}

// ──────────────────────── Phase 3: SDF Assembly ────────────────────────

/// Convert skeleton to flat capsule list: `(a, b, radius_a, radius_b)`.
fn skeleton_to_capsules(skeleton: &TreeSkeleton) -> Vec<(Vec3, Vec3, f32, f32)> {
    let mut capsules = Vec::new();
    for node in &skeleton.nodes {
        if let Some(parent_idx) = node.parent {
            let parent = &skeleton.nodes[parent_idx];
            capsules.push((parent.position, node.position, parent.radius, node.radius));
        }
    }
    capsules
}

/// Convert skeleton to SdfTree with per-depth smooth_k blending.
fn skeleton_to_sdf_tree(skeleton: &TreeSkeleton, params: &RedwoodParams) -> SdfTree {
    let mut trunk_capsules: Vec<SdfTree> = Vec::new();
    let mut branch_capsules: Vec<SdfTree> = Vec::new();
    let mut fine_capsules: Vec<SdfTree> = Vec::new();

    for node in &skeleton.nodes {
        if let Some(parent_idx) = node.parent {
            let parent = &skeleton.nodes[parent_idx];
            let prim = SdfTree::Primitive(SdfPrimitive::TaperedCapsule {
                a: parent.position,
                b: node.position,
                radius_a: parent.radius,
                radius_b: node.radius,
            });

            if node.kind.is_structural_trunk() {
                trunk_capsules.push(prim);
            } else if node.depth <= 1 {
                branch_capsules.push(prim);
            } else {
                fine_capsules.push(prim);
            }
        }
    }

    let trunk_tree = merge_balanced(&mut trunk_capsules, params.smooth_k_trunk);
    let branch_tree = merge_balanced(&mut branch_capsules, params.smooth_k_branch);
    let fine_tree = merge_balanced(&mut fine_capsules, params.smooth_k_fine);

    let mut result = trunk_tree;
    if let Some(bt) = branch_tree {
        result = Some(match result {
            Some(r) => SdfTree::SmoothUnion {
                a: Box::new(r),
                b: Box::new(bt),
                radius: params.smooth_k_branch,
            },
            None => bt,
        });
    }
    if let Some(ft) = fine_tree {
        result = Some(match result {
            Some(r) => SdfTree::SmoothUnion {
                a: Box::new(r),
                b: Box::new(ft),
                radius: params.smooth_k_fine,
            },
            None => ft,
        });
    }

    result.unwrap()
}

fn merge_balanced(nodes: &mut Vec<SdfTree>, smooth_k: f32) -> Option<SdfTree> {
    if nodes.is_empty() {
        return None;
    }
    while nodes.len() > 1 {
        let mut merged = Vec::with_capacity(nodes.len().div_ceil(2));
        while nodes.len() >= 2 {
            let b = nodes.pop().unwrap();
            let a = nodes.pop().unwrap();
            merged.push(SdfTree::SmoothUnion {
                a: Box::new(a),
                b: Box::new(b),
                radius: smooth_k,
            });
        }
        if let Some(remaining) = nodes.pop() {
            merged.push(remaining);
        }
        *nodes = merged;
    }
    nodes.pop()
}

// ──────────────────────── Phase 4: Canopy Regions ────────────────────────

fn skeleton_to_canopy_regions(
    skeleton: &TreeSkeleton,
    params: &RedwoodParams,
    _rng: &mut TreeRng,
) -> Vec<CanopyRegionSeed> {
    let min_y = params.trunk_height * (params.crown_start_frac - 0.05).max(0.0);
    let dominant_min_y = params.trunk_height * (params.crown_start_frac - 0.03).max(0.0);
    let children = crate::subjects::tube_mesh::build_children_index(skeleton);
    let mut candidates = Vec::new();

    for (idx, node) in skeleton.nodes.iter().enumerate() {
        if !node.kind.is_live_foliage() {
            continue;
        }
        let Some(parent_idx) = node.parent else {
            continue;
        };
        let parent_pos = skeleton.nodes[parent_idx].position;
        let axis = (node.position - parent_pos).normalize_or_zero();
        if axis.length_squared() < 1e-5 {
            continue;
        }

        let live_child_count = children[idx]
            .iter()
            .filter(|&&child_idx| skeleton.nodes[child_idx].kind.is_live_foliage())
            .count();
        let terminal = live_child_count == 0;
        let dominant = matches!(node.kind, BranchKind::LiveScaffold);

        if node.position.y < min_y && !(dominant && node.position.y >= dominant_min_y) {
            continue;
        }

        let seg_len = (node.position - parent_pos).length();
        let height_norm =
            ((node.position.y - min_y) / (params.trunk_height * 0.32)).clamp(0.0, 1.0);
        let radial_dist = Vec2::new(node.position.x, node.position.z).length();
        let radial_norm = (radial_dist / (params.trunk_height * 0.22)).clamp(0.0, 1.2);
        let outward = Vec3::new(node.position.x, 0.0, node.position.z).normalize_or_zero();
        let lobe_id = node.lobe_id.unwrap_or(0);
        let height_band = (1.0 - (height_norm - 0.62).abs() * 0.95).clamp(0.25, 1.0);
        let base_score = height_band * 0.72
            + radial_norm * 1.40
            + if dominant { 0.55 } else { 0.0 }
            + if terminal { 0.28 } else { 0.0 }
            + node.depth as f32 * 0.12;

        // Place canopy clusters directly on and around branch segments.
        // Clusters are oversized to overlap heavily, creating a continuous canopy mass
        // that hides the branch structure.
        let mid_anchor = parent_pos.lerp(node.position, 0.5);

        // Large cluster centered on branch, wrapping the whole segment
        candidates.push(CanopyRegionSeed {
            anchor: mid_anchor + outward * 0.3 + Vec3::Y * 0.3,
            radius: (seg_len * 2.5 + 2.0).clamp(4.0, 9.0),
            axis: (axis + Vec3::Y * 0.18).normalize_or_zero(),
            _interior_weight: 0.50,
            lobe_id,
            score: base_score + 0.35,
        });

        // Tip cluster wrapping branch end
        candidates.push(CanopyRegionSeed {
            anchor: node.position + outward * 0.25 + Vec3::NEG_Y * 0.1,
            radius: (seg_len * 2.0 + 1.5).clamp(3.5, 7.0),
            axis: (axis + outward * 0.12 + Vec3::NEG_Y * 0.08).normalize_or_zero(),
            _interior_weight: if terminal { 0.18 } else { 0.35 },
            lobe_id,
            score: base_score + 0.25,
        });

        // Under-branch drape: fills the gap below horizontal branches
        if dominant || terminal {
            candidates.push(CanopyRegionSeed {
                anchor: mid_anchor + Vec3::NEG_Y * 0.5 + outward * 0.4,
                radius: (seg_len * 1.8 + 1.2).clamp(3.0, 6.5),
                axis: (axis + Vec3::NEG_Y * 0.25).normalize_or_zero(),
                _interior_weight: 0.60,
                lobe_id,
                score: base_score + 0.12,
            });
        }

        // Inner fill: clusters centered on/near trunk at branch junction height
        // This is critical for hiding the trunk through the crown interior
        {
            // Large cluster right at branch base, barely offset from trunk
            candidates.push(CanopyRegionSeed {
                anchor: parent_pos + outward * 0.4 + Vec3::Y * 0.2,
                radius: (seg_len * 2.2 + 2.5).clamp(4.0, 9.0),
                axis: (outward * 0.2 + Vec3::Y * 0.5).normalize_or_zero(),
                _interior_weight: 0.75,
                lobe_id,
                score: base_score + 0.28,
            });

            // Opposite-side fill to wrap around trunk
            candidates.push(CanopyRegionSeed {
                anchor: parent_pos - outward * 0.3 + Vec3::Y * 0.15,
                radius: (seg_len * 1.5 + 2.0).clamp(3.5, 7.0),
                axis: (-outward * 0.15 + Vec3::Y * 0.4).normalize_or_zero(),
                _interior_weight: 0.80,
                lobe_id,
                score: base_score + 0.15,
            });
        }
    }

    // Crown-tip clusters: wrap the trunk leader with foliage at the top
    let max_lobe = candidates
        .iter()
        .map(|c| c.lobe_id as usize)
        .max()
        .unwrap_or(0)
        .max(1);
    for tip_i in 0..12 {
        let angle = (tip_i as f32 / 12.0) * std::f32::consts::TAU;
        let outward = Vec3::new(angle.cos(), 0.0, angle.sin());
        let tip_y = params.trunk_height * (0.90 + (tip_i as f32 / 12.0) * 0.08);
        candidates.push(CanopyRegionSeed {
            anchor: Vec3::new(outward.x * 0.6, tip_y, outward.z * 0.6),
            radius: 4.5,
            axis: (outward * 0.2 + Vec3::Y * 0.8).normalize_or_zero(),
            _interior_weight: 0.40,
            lobe_id: (tip_i % max_lobe) as u8,
            score: 4.5,
        });
    }

    // Interior trunk-hugging clusters: fill the gap where trunk shows through crown
    let crown_base_y = params.trunk_height * params.crown_start_frac;
    let crown_top_y = params.trunk_height * 0.92;
    for fill_i in 0..24 {
        let fill_t = fill_i as f32 / 24.0;
        let y = crown_base_y + fill_t * (crown_top_y - crown_base_y);
        let angle = (fill_i as f32 / 24.0) * std::f32::consts::TAU * 3.7; // spiral
        let outward = Vec3::new(angle.cos(), 0.0, angle.sin());
        candidates.push(CanopyRegionSeed {
            anchor: Vec3::new(outward.x * 0.8, y, outward.z * 0.8),
            radius: 6.0,
            axis: (outward * 0.15 + Vec3::Y * 0.40).normalize_or_zero(),
            _interior_weight: 0.80,
            lobe_id: (fill_i % max_lobe) as u8,
            score: 3.8,
        });
    }

    if candidates.is_empty() {
        return Vec::new();
    }

    let target_count = 260usize;
    let bucket_count = candidates
        .iter()
        .map(|candidate| candidate.lobe_id as usize)
        .max()
        .unwrap_or(0)
        + 1;
    let mut buckets = vec![Vec::new(); bucket_count.max(1)];
    for candidate in candidates {
        buckets[candidate.lobe_id as usize].push(candidate);
    }
    for bucket in &mut buckets {
        bucket.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| b.anchor.y.total_cmp(&a.anchor.y))
        });
        if !bucket.is_empty() {
            let sample_count = bucket.len().min(4);
            let mut centroid = Vec3::ZERO;
            let mut axis_sum = Vec3::ZERO;
            let mut max_radius: f32 = 0.0;
            let mut max_score: f32 = bucket[0].score;
            for candidate in bucket.iter().take(sample_count) {
                centroid += candidate.anchor;
                axis_sum += candidate.axis;
                max_radius = max_radius.max(candidate.radius);
                max_score = max_score.max(candidate.score);
            }
            centroid /= sample_count as f32;
            let outward = Vec3::new(centroid.x, 0.0, centroid.z).normalize_or_zero();
            let axis = (axis_sum / sample_count as f32).normalize_or_zero();
            let lobe_id = bucket[0].lobe_id;

            bucket.push(CanopyRegionSeed {
                anchor: centroid + outward * 0.62 + axis * 0.28 + Vec3::NEG_Y * 0.30,
                radius: (max_radius * 1.12 + 0.35).clamp(2.0, 4.2),
                axis: (axis + outward * 0.22 + Vec3::NEG_Y * 0.08).normalize_or_zero(),
                _interior_weight: 0.18,
                lobe_id,
                score: max_score + 0.36,
            });
            bucket.push(CanopyRegionSeed {
                anchor: centroid - outward * 0.12 + Vec3::NEG_Y * 0.08,
                radius: (max_radius * 0.96 + 0.18).clamp(1.8, 3.6),
                axis,
                _interior_weight: 0.50,
                lobe_id,
                score: max_score + 0.18,
            });
            if sample_count >= 3 {
                bucket.push(CanopyRegionSeed {
                    anchor: centroid + axis * 0.42 + outward * 0.32 + Vec3::Y * 0.22,
                    radius: (max_radius * 0.84 + 0.10).clamp(1.5, 3.1),
                    axis: (axis + outward * 0.16 + Vec3::Y * 0.12).normalize_or_zero(),
                    _interior_weight: 0.28,
                    lobe_id,
                    score: max_score + 0.08,
                });
            }
            bucket.push(CanopyRegionSeed {
                anchor: Vec3::new(
                    outward.x * 1.25,
                    centroid
                        .y
                        .clamp(params.trunk_height * 0.82, params.trunk_height * 0.95),
                    outward.z * 1.25,
                ) + axis * 0.12,
                radius: (max_radius * 0.72 + 0.14).clamp(1.35, 2.5),
                axis: (axis + outward * 0.18 + Vec3::Y * 0.20).normalize_or_zero(),
                _interior_weight: 0.40,
                lobe_id,
                score: max_score + 0.14,
            });
        }
        bucket.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| b.anchor.y.total_cmp(&a.anchor.y))
        });
    }

    let mut cursors = vec![0usize; buckets.len()];
    let mut selected = Vec::new();
    loop {
        let mut progressed = false;
        for bucket_idx in 0..buckets.len() {
            if cursors[bucket_idx] < buckets[bucket_idx].len() {
                selected.push(buckets[bucket_idx][cursors[bucket_idx]]);
                cursors[bucket_idx] += 1;
                progressed = true;
                if selected.len() >= target_count {
                    break;
                }
            }
        }
        if selected.len() >= target_count || !progressed {
            break;
        }
    }

    if selected.len() < 40 {
        let mut spill = Vec::new();
        for (bucket_idx, bucket) in buckets.iter().enumerate() {
            spill.extend(bucket[cursors[bucket_idx]..].iter().copied());
        }
        spill.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| b.anchor.y.total_cmp(&a.anchor.y))
        });
        let needed = 40usize.saturating_sub(selected.len());
        selected.extend(spill.into_iter().take(needed));
    }

    selected.truncate(target_count.min(selected.len()));
    selected
}

// ──────────────────────── Public API ────────────────────────

/// Build the SDF tree for a procedurally grown redwood.
pub fn build_trunk_sdf_grown(params: &RedwoodParams) -> SdfTree {
    let skeleton = build_skeleton(params);
    skeleton_to_sdf_tree(&skeleton, params)
}

/// Get all capsule segments for GPU upload from a procedurally grown redwood.
pub fn trunk_capsule_data_grown(params: &RedwoodParams) -> Vec<(Vec3, Vec3, f32, f32)> {
    let skeleton = build_skeleton(params);
    skeleton_to_capsules(&skeleton)
}

/// Generate foliage anchor positions from a procedurally grown redwood.
pub fn generate_foliage_anchors_grown(params: &RedwoodParams) -> Vec<(Vec3, f32)> {
    let skeleton = build_skeleton(params);
    generate_foliage_anchors_from_skeleton(params, &skeleton)
}

/// Generate foliage anchors from a pre-built skeleton (avoids rebuilding when skeleton is shared).
pub(crate) fn generate_foliage_anchors_from_skeleton(
    params: &RedwoodParams,
    skeleton: &TreeSkeleton,
) -> Vec<(Vec3, f32)> {
    let mut foliage_rng = TreeRng::new(params.seed.wrapping_add(0xdeadbeef));
    skeleton_to_canopy_regions(skeleton, params, &mut foliage_rng)
        .into_iter()
        .map(|region| (region.anchor, region.radius))
        .collect()
}

// ──────────────────────── Tests ────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rng_determinism() {
        let mut a = TreeRng::new(12345);
        let mut b = TreeRng::new(12345);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn rng_f32_in_range() {
        let mut rng = TreeRng::new(42);
        for _ in 0..1000 {
            let v = rng.next_f32();
            assert!((0.0..1.0).contains(&v), "next_f32 out of range: {v}");
        }
    }

    #[test]
    fn deterministic_capsules() {
        let params = RedwoodParams::default();
        let a = trunk_capsule_data_grown(&params);
        let b = trunk_capsule_data_grown(&params);
        assert_eq!(a.len(), b.len());
        for (ca, cb) in a.iter().zip(b.iter()) {
            assert_eq!(ca.0, cb.0);
            assert_eq!(ca.1, cb.1);
            assert_eq!(ca.2, cb.2);
            assert_eq!(ca.3, cb.3);
        }
    }

    #[test]
    fn different_seeds_produce_variation() {
        let a = trunk_capsule_data_grown(&RedwoodParams {
            seed: 1,
            ..Default::default()
        });
        let b = trunk_capsule_data_grown(&RedwoodParams {
            seed: 2,
            ..Default::default()
        });
        let differs = a.len() != b.len() || a.iter().zip(b.iter()).any(|(ca, cb)| ca.0 != cb.0);
        assert!(differs, "different seeds should produce different trees");
    }

    #[test]
    fn capsule_budget_in_range() {
        let params = RedwoodParams::default();
        let capsules = trunk_capsule_data_grown(&params);
        assert!(
            capsules.len() >= 200,
            "too few capsules: {}",
            capsules.len()
        );
        assert!(
            capsules.len() <= 20000,
            "too many capsules: {}",
            capsules.len()
        );
    }

    #[test]
    fn all_nodes_connected() {
        let params = RedwoodParams::default();
        let skeleton = build_skeleton(&params);

        for (i, node) in skeleton.nodes.iter().enumerate() {
            if i == 0 {
                assert!(node.parent.is_none(), "root should have no parent");
            } else {
                let pidx = node.parent.expect("non-root node must have parent");
                assert!(pidx < i, "parent index must be less than child index");
            }
        }
    }

    #[test]
    fn live_branch_angles_reasonable() {
        let params = RedwoodParams::default();
        let skeleton = build_skeleton(&params);

        for node in &skeleton.nodes {
            if !node.kind.is_live_foliage() || node.parent.is_none() {
                continue;
            }
            let parent = &skeleton.nodes[node.parent.unwrap()];
            let dir = (node.position - parent.position).normalize_or_zero();
            let cos_angle = dir.dot(Vec3::Y);
            assert!(
                cos_angle > -0.7,
                "branch too far from trunk: cos_angle={cos_angle}, dir={dir:?}"
            );
        }
    }

    #[test]
    fn branch_merge_uses_branch_smoothing_for_primary_branches() {
        let params = RedwoodParams::default();
        let tree = build_trunk_sdf_grown(&params);
        let branch_smooth_count = count_smooth_union_radius(&tree, params.smooth_k_branch);
        assert!(
            branch_smooth_count > 1,
            "expected depth-1 branches to use smooth_k_branch, got {branch_smooth_count}"
        );
    }

    #[test]
    fn trunk_maintains_a_long_gradual_taper() {
        let params = RedwoodParams::default();
        let skeleton = build_skeleton(&params);
        let trunk_nodes: Vec<_> = skeleton
            .nodes
            .iter()
            .filter(|node| matches!(node.kind, BranchKind::Trunk))
            .collect();
        let radius_at = |frac: f32| {
            let target_y = frac * params.trunk_height;
            trunk_nodes
                .iter()
                .min_by(|a, b| {
                    (a.position.y - target_y)
                        .abs()
                        .total_cmp(&(b.position.y - target_y).abs())
                })
                .map(|node| node.radius)
                .unwrap_or(0.0)
        };

        let r50 = radius_at(0.50);
        let r80 = radius_at(0.80);
        let r92 = radius_at(0.92);

        assert!(
            r50 > r80 && r80 > r92,
            "trunk radius should keep narrowing through the upper bole"
        );
        assert!(
            r80 > params.base_radius * 0.30,
            "upper trunk should still carry meaningful thickness at 80% height"
        );
        assert!(
            r92 > params.tip_radius * 1.3,
            "leader should keep some body instead of pinching to the tip too early"
        );
        assert!(
            r92 / r80 > 0.45,
            "upper taper should stay gradual instead of collapsing into a cone"
        );
    }

    #[test]
    fn foliage_anchors_in_crown() {
        let params = RedwoodParams::default();
        let anchors = generate_foliage_anchors_grown(&params);
        assert!(!anchors.is_empty(), "should produce foliage anchors");

        let min_y = params.trunk_height * (params.crown_start_frac - 0.10).max(0.0);
        for (pos, r) in &anchors {
            assert!(
                pos.y > min_y,
                "foliage anchor below crown: y={}, min={min_y}",
                pos.y
            );
            assert!(*r > 0.0, "foliage radius must be positive");
        }
    }

    fn count_smooth_union_radius(tree: &SdfTree, radius: f32) -> usize {
        match tree {
            SdfTree::Primitive(_) => 0,
            SdfTree::Union(a, b) | SdfTree::Intersection(a, b) | SdfTree::Difference(a, b) => {
                count_smooth_union_radius(a, radius) + count_smooth_union_radius(b, radius)
            }
            SdfTree::SmoothUnion { a, b, radius: r } => {
                usize::from((*r - radius).abs() < 1e-4)
                    + count_smooth_union_radius(a, radius)
                    + count_smooth_union_radius(b, radius)
            }
        }
    }
}
