//! Skinning weights computation and CPU linear blend skinning for humanoid meshes.

use glam::{Mat4, Vec3, Vec4};

use crate::scene::Vertex;
use crate::subjects::humanoid::HumanoidSkeleton;

// ──────────────────────── Skin Weights ────────────────────────

#[derive(Clone, Copy, Debug)]
pub struct SkinInfluence {
    pub bone: u16,
    pub weight: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct SkinWeights {
    pub influences: [SkinInfluence; 4],
}

impl SkinWeights {
    pub fn normalize(&mut self) {
        let sum: f32 = self.influences.iter().map(|i| i.weight).sum();
        if sum > 0.0 {
            for inf in &mut self.influences {
                inf.weight /= sum;
            }
        }
    }
}

fn point_to_segment_distance(p: Vec3, a: Vec3, b: Vec3) -> f32 {
    let ab = b - a;
    let len_sq = ab.length_squared();
    if len_sq < 1e-10 {
        return (p - a).length();
    }
    let t = ((p - a).dot(ab) / len_sq).clamp(0.0, 1.0);
    let closest = a + ab * t;
    (p - closest).length()
}

/// Compute per-vertex skin weights by finding the 4 closest bones.
pub fn compute_skin_weights(skeleton: &HumanoidSkeleton, vertices: &[Vertex]) -> Vec<SkinWeights> {
    vertices
        .iter()
        .map(|vertex| {
            let vpos = Vec3::from(vertex.position);

            let mut distances: Vec<(u16, f32)> = Vec::with_capacity(skeleton.bones.len());
            for (i, bone) in skeleton.bones.iter().enumerate() {
                let bone_pos = bone.rest_position;
                let parent_pos = match bone.parent {
                    Some(pi) => skeleton.bones[pi].rest_position,
                    None => bone_pos,
                };
                let dist = point_to_segment_distance(vpos, parent_pos, bone_pos);
                distances.push((i as u16, dist));
            }

            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            let mut influences = [SkinInfluence {
                bone: 0,
                weight: 0.0,
            }; 4];
            for (slot, &(bone_idx, dist)) in distances.iter().take(4).enumerate() {
                let inv_dist_sq = 1.0 / (dist * dist + 1e-6);
                influences[slot] = SkinInfluence {
                    bone: bone_idx,
                    weight: inv_dist_sq,
                };
            }

            let mut sw = SkinWeights { influences };
            sw.normalize();
            sw
        })
        .collect()
}

// ──────────────────────── CPU Skinning ────────────────────────

/// Apply linear blend skinning to the rest-pose mesh using per-bone skinning matrices.
pub fn skin_mesh(
    rest_vertices: &[Vertex],
    weights: &[SkinWeights],
    bone_matrices: &[Mat4],
) -> Vec<Vertex> {
    rest_vertices
        .iter()
        .zip(weights.iter())
        .map(|(vertex, sw)| {
            let rest_pos = Vec3::from(vertex.position);
            let rest_norm = Vec3::from(vertex.normal);

            let mut blended_pos = Vec3::ZERO;
            let mut blended_norm = Vec3::ZERO;

            for inf in &sw.influences {
                if inf.weight <= 0.0 {
                    continue;
                }
                let mat = bone_matrices[inf.bone as usize];
                blended_pos += (mat * Vec4::new(rest_pos.x, rest_pos.y, rest_pos.z, 1.0))
                    .truncate()
                    * inf.weight;
                blended_norm += (mat * Vec4::new(rest_norm.x, rest_norm.y, rest_norm.z, 0.0))
                    .truncate()
                    * inf.weight;
            }

            let final_norm = blended_norm.normalize_or_zero();

            Vertex {
                position: blended_pos.into(),
                normal: final_norm.into(),
                material: vertex.material,
                feature_id: vertex.feature_id,
                uv: vertex.uv,
                ao: vertex.ao,
                semantic_channels: vertex.semantic_channels,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::subjects::humanoid::{build_body_mesh, build_skeleton, HumanoidParams};

    #[test]
    fn weights_parallel_to_vertices() {
        let params = HumanoidParams::default();
        let skeleton = build_skeleton(&params);
        let (vertices, _) = build_body_mesh(&skeleton, &params);
        let weights = compute_skin_weights(&skeleton, &vertices);
        assert_eq!(weights.len(), vertices.len());
    }

    #[test]
    fn weights_sum_to_one() {
        let params = HumanoidParams::default();
        let skeleton = build_skeleton(&params);
        let (vertices, _) = build_body_mesh(&skeleton, &params);
        let weights = compute_skin_weights(&skeleton, &vertices);
        for (i, sw) in weights.iter().enumerate() {
            let sum: f32 = sw.influences.iter().map(|inf| inf.weight).sum();
            assert!(
                (sum - 1.0).abs() < 0.01,
                "vertex {i} weight sum {sum} should be ~1.0"
            );
        }
    }

    #[test]
    fn no_negative_weights() {
        let params = HumanoidParams::default();
        let skeleton = build_skeleton(&params);
        let (vertices, _) = build_body_mesh(&skeleton, &params);
        let weights = compute_skin_weights(&skeleton, &vertices);
        for sw in &weights {
            for inf in &sw.influences {
                assert!(inf.weight >= 0.0, "negative weight {}", inf.weight);
            }
        }
    }

    #[test]
    fn identity_skinning_preserves_mesh() {
        let params = HumanoidParams::default();
        let skeleton = build_skeleton(&params);
        let (vertices, _) = build_body_mesh(&skeleton, &params);
        let weights = compute_skin_weights(&skeleton, &vertices);

        let identity_matrices = vec![Mat4::IDENTITY; skeleton.bones.len()];
        let result = skin_mesh(&vertices, &weights, &identity_matrices);

        assert_eq!(result.len(), vertices.len());
        for (i, (a, b)) in vertices.iter().zip(result.iter()).enumerate() {
            let da = Vec3::from(a.position) - Vec3::from(b.position);
            assert!(
                da.length() < 0.001,
                "vertex {i} moved by {} under identity skinning",
                da.length()
            );
        }
    }
}
