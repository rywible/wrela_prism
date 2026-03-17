/// Opaque needle spray geometry for foliage — replaces billboard cards.
///
/// Each foliage anchor generates a cluster of thin elongated quads arranged
/// as radial fans, producing dense opaque needle masses that feed naturally
/// into the meshlet DAG builder.
use glam::Vec3;

use crate::scene::{Vertex, MATERIAL_FOLIAGE};
use crate::subjects::redwood_growth::{RedwoodParams, TreeRng};

/// Build needle spray geometry from anchor points.
///
/// Each anchor generates `sprays_per_cluster` sprays, where each spray is
/// 6 thin elongated quads in a radial fan (12 triangles per spray).
/// Returns vertices and indices ready to merge into the meshlet DAG builder.
pub fn build_needle_sprays(
    params: &RedwoodParams,
    anchors: &[(Vec3, f32)],
    sprays_per_cluster: u32,
) -> (Vec<Vertex>, Vec<u32>) {
    let quads_per_spray = 6u32;
    let total_sprays = anchors.len() * sprays_per_cluster as usize;
    let total_quads = total_sprays * quads_per_spray as usize;
    let mut verts = Vec::with_capacity(total_quads * 4);
    let mut indices = Vec::with_capacity(total_quads * 6);
    let mut rng = TreeRng::new(params.seed.wrapping_add(0xf011a9e));

    for &(center, cluster_radius) in anchors {
        for _ in 0..sprays_per_cluster {
            // Random position within the cluster sphere
            let offset =
                random_sphere_point(&mut rng) * cluster_radius * rng.next_f32_range(0.3, 1.0);
            let pos = center + offset;

            // Random twig axis with upward bias
            let mut twig_axis = random_sphere_point(&mut rng);
            twig_axis.y = twig_axis.y.abs() * 0.5 + 0.3; // upward bias
            twig_axis = twig_axis.normalize_or_zero();

            // AO based on height
            let ao = (pos.y / params.trunk_height).clamp(0.3, 1.0);

            // Build 6 quads in a radial fan around the twig axis
            for fan_i in 0..quads_per_spray {
                let fan_angle = (fan_i as f32 / quads_per_spray as f32) * std::f32::consts::TAU
                    + rng.next_f32_range(-0.15, 0.15);

                // Tilt the needle ±30° from the twig axis
                let tilt = rng.next_f32_range(-30.0_f32, 30.0).to_radians();

                // Build a local frame for this needle quad
                let arbitrary = if twig_axis.y.abs() > 0.9 {
                    Vec3::X
                } else {
                    Vec3::Y
                };
                let right = twig_axis.cross(arbitrary).normalize_or_zero();
                let up = right.cross(twig_axis).normalize_or_zero();

                // Rotate the needle around the twig axis
                let rotated_right = right * fan_angle.cos() + up * fan_angle.sin();
                let rotated_up = -right * fan_angle.sin() + up * fan_angle.cos();

                // Apply tilt
                let needle_dir =
                    (twig_axis * tilt.cos() + rotated_up * tilt.sin()).normalize_or_zero();
                let needle_right = rotated_right;

                // Needle dimensions: elongated (length >> width)
                let half_len = cluster_radius * rng.next_f32_range(0.18, 0.38);
                let half_width = half_len * rng.next_f32_range(0.10, 0.22);

                let base_idx = verts.len() as u32;

                // Compute face normal
                let face_normal = needle_dir.cross(needle_right).normalize_or_zero();

                // 4 corners of the elongated quad
                let corners = [
                    pos - needle_dir * half_len - needle_right * half_width,
                    pos + needle_dir * half_len - needle_right * half_width,
                    pos + needle_dir * half_len + needle_right * half_width,
                    pos - needle_dir * half_len + needle_right * half_width,
                ];

                for corner in &corners {
                    verts.push(Vertex {
                        position: (*corner).into(),
                        normal: [face_normal.x, face_normal.y, face_normal.z],
                        material: MATERIAL_FOLIAGE,
                        feature_id: 0,
                        uv: [0.0, 0.0],
                        ao,
                        semantic_channels: 0,
                    });
                }

                // Two triangles per quad
                indices.extend_from_slice(&[
                    base_idx,
                    base_idx + 1,
                    base_idx + 2,
                    base_idx,
                    base_idx + 2,
                    base_idx + 3,
                ]);
            }
        }
    }

    (verts, indices)
}

/// Density tiers for needle spray LOD.
pub fn sprays_per_cluster_for_tier(tier: u32) -> u32 {
    match tier {
        0 => 10,
        1 => 18,
        2 => 28,
        _ => 18,
    }
}

fn random_sphere_point(rng: &mut TreeRng) -> Vec3 {
    loop {
        let x = rng.next_f32_range(-1.0, 1.0);
        let y = rng.next_f32_range(-1.0, 1.0);
        let z = rng.next_f32_range(-1.0, 1.0);
        let v = Vec3::new(x, y, z);
        let len_sq = v.length_squared();
        if len_sq > 0.01 && len_sq <= 1.0 {
            return v.normalize();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_sprays_from_anchors() {
        let params = RedwoodParams::default();
        let anchors = vec![
            (Vec3::new(0.0, 40.0, 0.0), 3.0),
            (Vec3::new(5.0, 45.0, 2.0), 2.5),
        ];
        let (verts, indices) = build_needle_sprays(&params, &anchors, 20);

        // 2 anchors * 20 sprays * 6 quads * 4 verts = 960 verts
        assert_eq!(verts.len(), 960);
        // 2 anchors * 20 sprays * 6 quads * 6 indices = 1440 indices
        assert_eq!(indices.len(), 1440);

        // All materials should be foliage
        for v in &verts {
            assert_eq!(v.material, MATERIAL_FOLIAGE);
        }

        // All indices in bounds
        for &i in &indices {
            assert!((i as usize) < verts.len());
        }
    }

    #[test]
    fn density_tiers_increase() {
        assert!(sprays_per_cluster_for_tier(0) < sprays_per_cluster_for_tier(1));
        assert!(sprays_per_cluster_for_tier(1) < sprays_per_cluster_for_tier(2));
    }

    #[test]
    fn needle_sprays_merge_into_dag() {
        use crate::meshlet::build_meshlet_dag;
        use crate::subjects::tube_mesh::build_trunk_mesh;

        let params = RedwoodParams::default();
        let (mut trunk_verts, mut trunk_indices) = build_trunk_mesh(&params);

        let anchors = vec![
            (Vec3::new(0.0, 45.0, 0.0), 3.0),
            (Vec3::new(2.0, 50.0, 1.0), 2.5),
        ];
        let (foliage_verts, foliage_indices) = build_needle_sprays(&params, &anchors, 20);

        let vert_offset = trunk_verts.len() as u32;
        trunk_verts.extend(foliage_verts);
        trunk_indices.extend(foliage_indices.iter().map(|&i| i + vert_offset));

        let dag = build_meshlet_dag(trunk_verts, &trunk_indices);
        assert!(!dag.meshlets.is_empty());
        assert!(dag.level_offsets.len() >= 2);
    }
}
