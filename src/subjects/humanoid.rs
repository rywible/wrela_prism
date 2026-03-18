//! Procedural humanoid skeleton and body mesh generation.
//!
//! Produces a stylized humanoid figure with 22 bones and an extruded tube mesh
//! suitable for CPU skinning and rendering through the visibility buffer pipeline.

use std::f32::consts::{PI, TAU};

use glam::{Quat, Vec2, Vec3};

use crate::scene::Vertex;
use crate::util::Rng;

// ──────────────────────── Skeleton ────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JointType {
    Ball,
    Hinge,
}

#[derive(Clone, Debug)]
pub struct Bone {
    pub parent: Option<usize>,
    pub rest_position: Vec3,
    pub rest_rotation: Quat,
    pub joint_type: JointType,
    pub rotation_limits: [Vec2; 3],
    pub length: f32,
}

#[derive(Clone, Debug)]
pub struct HumanoidSkeleton {
    pub bones: Vec<Bone>,
    pub bone_names: Vec<&'static str>,
}

// Bone index constants (22 bones).
pub const PELVIS: usize = 0;
pub const SPINE_LOWER: usize = 1;
pub const SPINE_UPPER: usize = 2;
pub const NECK: usize = 3;
pub const HEAD: usize = 4;
pub const SHOULDER_L: usize = 5;
pub const UPPER_ARM_L: usize = 6;
pub const LOWER_ARM_L: usize = 7;
pub const HAND_L: usize = 8;
pub const SHOULDER_R: usize = 9;
pub const UPPER_ARM_R: usize = 10;
pub const LOWER_ARM_R: usize = 11;
pub const HAND_R: usize = 12;
pub const HIP_L: usize = 13;
pub const UPPER_LEG_L: usize = 14;
pub const LOWER_LEG_L: usize = 15;
pub const FOOT_L: usize = 16;
pub const TOE_L: usize = 17;
pub const HIP_R: usize = 18;
pub const UPPER_LEG_R: usize = 19;
pub const LOWER_LEG_R: usize = 20;
pub const FOOT_R: usize = 21;

pub const BONE_COUNT: usize = 22;

pub const BONE_NAMES: [&str; BONE_COUNT] = [
    "Pelvis",
    "SpineLower",
    "SpineUpper",
    "Neck",
    "Head",
    "ShoulderL",
    "UpperArmL",
    "LowerArmL",
    "HandL",
    "ShoulderR",
    "UpperArmR",
    "LowerArmR",
    "HandR",
    "HipL",
    "UpperLegL",
    "LowerLegL",
    "FootL",
    "ToeL",
    "HipR",
    "UpperLegR",
    "LowerLegR",
    "FootR",
];

// ──────────────────────── Parameters ────────────────────────

#[derive(Clone, Debug)]
pub struct HumanoidParams {
    pub height: f32,
    pub head_scale: f32,
    pub shoulder_width: f32,
    pub torso_length: f32,
    pub leg_length_ratio: f32,
    pub arm_length_ratio: f32,
    pub muscle_bulk: f32,
    pub stylization: f32,
    pub seed: u64,
}

impl Default for HumanoidParams {
    fn default() -> Self {
        Self {
            height: 1.8,
            head_scale: 1.0,
            shoulder_width: 0.42,
            torso_length: 0.50,
            leg_length_ratio: 0.53,
            arm_length_ratio: 0.44,
            muscle_bulk: 0.5,
            stylization: 0.5,
            seed: 42,
        }
    }
}

// ──────────────────────── Build Skeleton ────────────────────────

/// Constructs the 22-bone hierarchy with rest-pose positions derived from proportions.
pub fn build_skeleton(params: &HumanoidParams) -> HumanoidSkeleton {
    let mut rng = Rng::new(params.seed);
    let h = params.height;
    let leg_len = h * params.leg_length_ratio;
    let upper_leg = leg_len * 0.54;
    let lower_leg = leg_len * 0.46;
    let foot_len = h * 0.04;
    let toe_len = h * 0.03;
    let torso = params.torso_length * h;
    let spine_lower_len = torso * 0.45;
    let spine_upper_len = torso * 0.55;
    let neck_len = h * 0.04;
    let head_len = h * 0.12 * params.head_scale;
    let arm_len = h * params.arm_length_ratio;
    let upper_arm = arm_len * 0.52;
    let lower_arm = arm_len * 0.36;
    let hand_len = arm_len * 0.12;
    let sw = params.shoulder_width;
    let hip_width = sw * 0.55;

    let asym = rng.next_f32_range(-0.005, 0.005);

    let ball = JointType::Ball;
    let hinge = JointType::Hinge;
    let no_limit = [Vec2::new(-PI, PI); 3];
    let hinge_limit = [Vec2::new(-PI, PI), Vec2::ZERO, Vec2::ZERO];

    let pelvis_y = leg_len;

    let mut bones = Vec::with_capacity(BONE_COUNT);

    macro_rules! bone {
        ($parent:expr, $pos:expr, $jt:expr, $len:expr) => {
            bones.push(Bone {
                parent: $parent,
                rest_position: $pos,
                rest_rotation: Quat::IDENTITY,
                joint_type: $jt,
                rotation_limits: if $jt == JointType::Hinge {
                    hinge_limit
                } else {
                    no_limit
                },
                length: $len,
            });
        };
    }

    // 0: Pelvis (root)
    bone!(None, Vec3::new(0.0, pelvis_y, 0.0), ball, 0.0);
    // 1: SpineLower
    bone!(
        Some(PELVIS),
        Vec3::new(0.0, pelvis_y, 0.0),
        ball,
        spine_lower_len
    );
    // 2: SpineUpper
    bone!(
        Some(SPINE_LOWER),
        Vec3::new(0.0, pelvis_y + spine_lower_len, 0.0),
        ball,
        spine_upper_len
    );
    // 3: Neck
    bone!(
        Some(SPINE_UPPER),
        Vec3::new(0.0, pelvis_y + torso, 0.0),
        ball,
        neck_len
    );
    // 4: Head
    bone!(
        Some(NECK),
        Vec3::new(0.0, pelvis_y + torso + neck_len, 0.0),
        ball,
        head_len
    );

    // Left arm chain
    bone!(
        Some(SPINE_UPPER),
        Vec3::new(-sw / 2.0, pelvis_y + torso, 0.0),
        ball,
        0.0
    );
    bone!(
        Some(SHOULDER_L),
        Vec3::new(-sw / 2.0, pelvis_y + torso, 0.0),
        ball,
        upper_arm
    );
    bone!(
        Some(UPPER_ARM_L),
        Vec3::new(-sw / 2.0, pelvis_y + torso - upper_arm, 0.0),
        hinge,
        lower_arm
    );
    bone!(
        Some(LOWER_ARM_L),
        Vec3::new(-sw / 2.0, pelvis_y + torso - upper_arm - lower_arm, 0.0),
        ball,
        hand_len
    );

    // Right arm chain
    bone!(
        Some(SPINE_UPPER),
        Vec3::new(sw / 2.0 + asym, pelvis_y + torso, 0.0),
        ball,
        0.0
    );
    bone!(
        Some(SHOULDER_R),
        Vec3::new(sw / 2.0 + asym, pelvis_y + torso, 0.0),
        ball,
        upper_arm
    );
    bone!(
        Some(UPPER_ARM_R),
        Vec3::new(sw / 2.0 + asym, pelvis_y + torso - upper_arm, 0.0),
        hinge,
        lower_arm
    );
    bone!(
        Some(LOWER_ARM_R),
        Vec3::new(
            sw / 2.0 + asym,
            pelvis_y + torso - upper_arm - lower_arm,
            0.0
        ),
        ball,
        hand_len
    );

    // Left leg chain
    bone!(
        Some(PELVIS),
        Vec3::new(-hip_width / 2.0, pelvis_y, 0.0),
        ball,
        0.0
    );
    bone!(
        Some(HIP_L),
        Vec3::new(-hip_width / 2.0, pelvis_y, 0.0),
        ball,
        upper_leg
    );
    bone!(
        Some(UPPER_LEG_L),
        Vec3::new(-hip_width / 2.0, pelvis_y - upper_leg, 0.0),
        hinge,
        lower_leg
    );
    bone!(
        Some(LOWER_LEG_L),
        Vec3::new(-hip_width / 2.0, pelvis_y - leg_len, 0.0),
        ball,
        foot_len
    );
    bone!(
        Some(FOOT_L),
        Vec3::new(-hip_width / 2.0, pelvis_y - leg_len, foot_len),
        ball,
        toe_len
    );

    // Right leg chain
    bone!(
        Some(PELVIS),
        Vec3::new(hip_width / 2.0, pelvis_y, 0.0),
        ball,
        0.0
    );
    bone!(
        Some(HIP_R),
        Vec3::new(hip_width / 2.0, pelvis_y, 0.0),
        ball,
        upper_leg
    );
    bone!(
        Some(UPPER_LEG_R),
        Vec3::new(hip_width / 2.0, pelvis_y - upper_leg, 0.0),
        hinge,
        lower_leg
    );
    bone!(
        Some(LOWER_LEG_R),
        Vec3::new(hip_width / 2.0, pelvis_y - leg_len, 0.0),
        ball,
        foot_len
    );

    assert_eq!(bones.len(), BONE_COUNT);

    HumanoidSkeleton {
        bones,
        bone_names: BONE_NAMES.to_vec(),
    }
}

// ──────────────────────── Body Mesh Generation ────────────────────────

fn ring_sides(radius: f32) -> usize {
    if radius > 0.15 {
        12
    } else if radius > 0.06 {
        8
    } else {
        6
    }
}

struct LimbFrame {
    center: Vec3,
    right: Vec3,
    up: Vec3,
    radius: f32,
    v_coord: f32,
}

fn build_limb_frames(positions: &[Vec3], radii: &[f32], total_length: f32) -> Vec<LimbFrame> {
    if positions.is_empty() {
        return Vec::new();
    }

    let mut dedup_positions = Vec::with_capacity(positions.len());
    let mut dedup_radii = Vec::with_capacity(radii.len());
    for i in 0..positions.len() {
        if i == 0 || (positions[i] - positions[i - 1]).length() > 1e-5 {
            dedup_positions.push(positions[i]);
            dedup_radii.push(radii[i]);
        }
    }

    if dedup_positions.len() < 2 {
        return vec![LimbFrame {
            center: dedup_positions[0],
            right: Vec3::X,
            up: Vec3::Z,
            radius: dedup_radii[0],
            v_coord: 0.0,
        }];
    }

    let mut frames = Vec::with_capacity(dedup_positions.len());
    let mut prev_right = Vec3::X;
    let mut accum_dist = 0.0_f32;

    for i in 0..dedup_positions.len() {
        let center = dedup_positions[i];

        let tangent = if i + 1 < dedup_positions.len() {
            (dedup_positions[i + 1] - center).normalize_or_zero()
        } else if i > 0 {
            (center - dedup_positions[i - 1]).normalize_or_zero()
        } else {
            Vec3::Y
        };

        let tangent = if tangent.length_squared() < 0.01 {
            Vec3::Y
        } else {
            tangent
        };

        let projected = (prev_right - tangent * prev_right.dot(tangent)).normalize_or_zero();
        let right = if projected.length_squared() > 0.01 {
            projected
        } else {
            let arbitrary = if tangent.y.abs() > 0.9 {
                Vec3::X
            } else {
                Vec3::Y
            };
            tangent.cross(arbitrary).normalize()
        };
        let up = tangent.cross(right).normalize();
        prev_right = right;

        if i > 0 {
            accum_dist += (center - dedup_positions[i - 1]).length();
        }
        let v_coord = if total_length > 0.0 {
            accum_dist / total_length
        } else {
            0.0
        };

        frames.push(LimbFrame {
            center,
            right,
            up,
            radius: dedup_radii[i],
            v_coord,
        });
    }
    frames
}

fn emit_ring(vertices: &mut Vec<Vertex>, frame: &LimbFrame, sides: usize, material: u32) -> u32 {
    let base = vertices.len() as u32;
    for i in 0..sides {
        let angle = (i as f32 / sides as f32) * TAU;
        let (sin_a, cos_a) = angle.sin_cos();
        let offset = frame.right * cos_a * frame.radius + frame.up * sin_a * frame.radius;
        let pos = frame.center + offset;
        let normal = offset.normalize_or_zero();
        let u = i as f32 / sides as f32;
        vertices.push(Vertex {
            position: pos.into(),
            normal: normal.into(),
            material,
            feature_id: 0,
            uv: [u, frame.v_coord],
            ao: 1.0,
            semantic_channels: 0,
        });
    }
    base
}

fn stitch_rings(indices: &mut Vec<u32>, ring_a: u32, ring_b: u32, sides: usize) {
    for i in 0..sides {
        let i0 = i as u32;
        let i1 = ((i + 1) % sides) as u32;
        indices.push(ring_a + i0);
        indices.push(ring_b + i0);
        indices.push(ring_b + i1);
        indices.push(ring_a + i0);
        indices.push(ring_b + i1);
        indices.push(ring_a + i1);
    }
}

fn limb_radius(t: f32, r0: f32, r1: f32) -> f32 {
    r0 + (r1 - r0) * t
}

fn build_chain_tube(
    vertices: &mut Vec<Vertex>,
    indices: &mut Vec<u32>,
    positions: &[Vec3],
    radii: &[f32],
    material: u32,
    subdivisions: usize,
) {
    if positions.len() < 2 {
        return;
    }

    let mut subdiv_positions = Vec::new();
    let mut subdiv_radii = Vec::new();

    for seg in 0..positions.len() - 1 {
        let steps = subdivisions.max(1);
        for s in 0..steps {
            let t = s as f32 / steps as f32;
            subdiv_positions.push(positions[seg].lerp(positions[seg + 1], t));
            subdiv_radii.push(limb_radius(t, radii[seg], radii[seg + 1]));
        }
    }
    subdiv_positions.push(*positions.last().unwrap());
    subdiv_radii.push(*radii.last().unwrap());

    let total_length: f32 = positions.windows(2).map(|w| (w[1] - w[0]).length()).sum();
    let frames = build_limb_frames(&subdiv_positions, &subdiv_radii, total_length);

    let max_radius = subdiv_radii.iter().copied().fold(0.0_f32, f32::max);
    let sides = ring_sides(max_radius);

    let mut prev_ring = None;
    for frame in &frames {
        let ring = emit_ring(vertices, frame, sides, material);
        if let Some(prev) = prev_ring {
            stitch_rings(indices, prev, ring, sides);
        }
        prev_ring = Some(ring);
    }
}

fn build_head_sphere(
    vertices: &mut Vec<Vertex>,
    indices: &mut Vec<u32>,
    center: Vec3,
    radius: f32,
    material: u32,
) {
    let lat_rings = 8;
    let lon_sides = 12;

    let mut ring_starts = Vec::new();

    let bottom_idx = vertices.len() as u32;
    vertices.push(Vertex {
        position: (center + Vec3::new(0.0, -radius, 0.0)).into(),
        normal: [0.0, -1.0, 0.0],
        material,
        feature_id: 0,
        uv: [0.5, 0.0],
        ao: 1.0,
        semantic_channels: 0,
    });

    for lat in 1..lat_rings {
        let theta = PI * lat as f32 / lat_rings as f32;
        let y = -radius * theta.cos();
        let r = radius * theta.sin();
        let v = lat as f32 / lat_rings as f32;
        let start = vertices.len() as u32;
        for lon in 0..lon_sides {
            let phi = TAU * lon as f32 / lon_sides as f32;
            let x = r * phi.cos();
            let z = r * phi.sin();
            let pos = center + Vec3::new(x, y, z);
            let normal = Vec3::new(x, y, z).normalize();
            let u = lon as f32 / lon_sides as f32;
            vertices.push(Vertex {
                position: pos.into(),
                normal: normal.into(),
                material,
                feature_id: 0,
                uv: [u, v],
                ao: 1.0,
                semantic_channels: 0,
            });
        }
        ring_starts.push(start);
    }

    let top_idx = vertices.len() as u32;
    vertices.push(Vertex {
        position: (center + Vec3::new(0.0, radius, 0.0)).into(),
        normal: [0.0, 1.0, 0.0],
        material,
        feature_id: 0,
        uv: [0.5, 1.0],
        ao: 1.0,
        semantic_channels: 0,
    });

    if let Some(&first_ring) = ring_starts.first() {
        for i in 0..lon_sides {
            let i0 = first_ring + i as u32;
            let i1 = first_ring + ((i + 1) % lon_sides) as u32;
            indices.push(bottom_idx);
            indices.push(i0);
            indices.push(i1);
        }
    }

    for r in 0..ring_starts.len().saturating_sub(1) {
        stitch_rings(indices, ring_starts[r], ring_starts[r + 1], lon_sides);
    }

    if let Some(&last_ring) = ring_starts.last() {
        for i in 0..lon_sides {
            let i0 = last_ring + i as u32;
            let i1 = last_ring + ((i + 1) % lon_sides) as u32;
            indices.push(top_idx);
            indices.push(i1);
            indices.push(i0);
        }
    }
}

/// Build the body mesh from a humanoid skeleton.
pub fn build_body_mesh(
    skeleton: &HumanoidSkeleton,
    params: &HumanoidParams,
) -> (Vec<Vertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    let material = crate::scene::MATERIAL_SKIN;
    let bulk = params.muscle_bulk;

    let pos = |bone: usize| skeleton.bones[bone].rest_position;

    let base_r = |r: f32| r * (0.8 + bulk * 0.4);

    // Torso chain
    {
        let positions = vec![pos(PELVIS), pos(SPINE_LOWER), pos(SPINE_UPPER), pos(NECK)];
        let sw = params.shoulder_width;
        let radii = vec![
            base_r(sw * 0.40),
            base_r(sw * 0.38),
            base_r(sw * 0.50),
            base_r(sw * 0.22),
        ];
        build_chain_tube(&mut vertices, &mut indices, &positions, &radii, material, 3);
    }

    // Left arm chain
    {
        let positions = vec![
            pos(SHOULDER_L),
            pos(UPPER_ARM_L),
            pos(LOWER_ARM_L),
            pos(HAND_L),
        ];
        let radii = vec![base_r(0.05), base_r(0.045), base_r(0.035), base_r(0.025)];
        build_chain_tube(&mut vertices, &mut indices, &positions, &radii, material, 2);
    }

    // Right arm chain
    {
        let positions = vec![
            pos(SHOULDER_R),
            pos(UPPER_ARM_R),
            pos(LOWER_ARM_R),
            pos(HAND_R),
        ];
        let radii = vec![base_r(0.05), base_r(0.045), base_r(0.035), base_r(0.025)];
        build_chain_tube(&mut vertices, &mut indices, &positions, &radii, material, 2);
    }

    // Left leg chain
    {
        let positions = vec![pos(HIP_L), pos(UPPER_LEG_L), pos(LOWER_LEG_L), pos(FOOT_L)];
        let radii = vec![base_r(0.07), base_r(0.065), base_r(0.045), base_r(0.035)];
        build_chain_tube(&mut vertices, &mut indices, &positions, &radii, material, 2);
    }

    // Right leg chain
    {
        let positions = vec![pos(HIP_R), pos(UPPER_LEG_R), pos(LOWER_LEG_R), pos(FOOT_R)];
        let radii = vec![base_r(0.07), base_r(0.065), base_r(0.045), base_r(0.035)];
        build_chain_tube(&mut vertices, &mut indices, &positions, &radii, material, 2);
    }

    // Head sphere
    {
        let neck_pos = pos(NECK);
        let head_pos = pos(HEAD);
        let head_center = (neck_pos + head_pos) * 0.5 + Vec3::new(0.0, 0.02, 0.0);
        let head_radius = params.height * 0.07 * params.head_scale;
        build_head_sphere(
            &mut vertices,
            &mut indices,
            head_center,
            head_radius,
            material,
        );
    }

    (vertices, indices)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skeleton_has_22_bones() {
        let skeleton = build_skeleton(&HumanoidParams::default());
        assert_eq!(skeleton.bones.len(), BONE_COUNT);
        assert_eq!(skeleton.bone_names.len(), BONE_COUNT);
    }

    #[test]
    fn all_parent_indices_valid() {
        let skeleton = build_skeleton(&HumanoidParams::default());
        for (i, bone) in skeleton.bones.iter().enumerate() {
            if let Some(parent) = bone.parent {
                assert!(parent < i, "bone {i} parent {parent} must be < {i}");
            }
        }
    }

    #[test]
    fn pelvis_height_matches_leg_ratio() {
        let params = HumanoidParams::default();
        let skeleton = build_skeleton(&params);
        let pelvis_y = skeleton.bones[PELVIS].rest_position.y;
        let expected = params.height * params.leg_length_ratio;
        assert!(
            (pelvis_y - expected).abs() < 0.01,
            "pelvis_y={pelvis_y} expected~{expected}"
        );
    }

    #[test]
    fn body_mesh_nonempty_and_valid() {
        let params = HumanoidParams::default();
        let skeleton = build_skeleton(&params);
        let (vertices, indices_out) = build_body_mesh(&skeleton, &params);

        assert!(!vertices.is_empty(), "vertices should not be empty");
        assert!(!indices_out.is_empty(), "indices should not be empty");
        assert_eq!(
            indices_out.len() % 3,
            0,
            "index count must be multiple of 3"
        );

        let vert_count = vertices.len() as u32;
        for &idx in &indices_out {
            assert!(
                idx < vert_count,
                "index {idx} out of bounds (vertex count {vert_count})"
            );
        }

        assert!(
            vertices.len() >= 100,
            "expected at least 100 vertices, got {}",
            vertices.len()
        );
        assert!(
            vertices.len() <= 5000,
            "expected at most 5000 vertices, got {}",
            vertices.len()
        );

        for v in &vertices {
            let n = Vec3::from(v.normal);
            let len = n.length();
            assert!(
                (len - 1.0).abs() < 0.1,
                "normal length {len} should be ~1.0"
            );
        }
    }
}
