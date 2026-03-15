/// Oriented billboard quads for foliage, fed into the meshlet DAG builder.
///
/// Each foliage anchor point generates a cluster of oriented quads
/// (camera-facing billboards are baked at build time into a few orientations).
/// These quads carry MATERIAL_FOLIAGE and UV coordinates into the alpha mask.

use glam::Vec3;

use crate::scene::{Vertex, MATERIAL_FOLIAGE};
use crate::subjects::redwood_growth::{RedwoodParams, TreeRng};

/// Build foliage billboard geometry from anchor points.
///
/// Each anchor generates `cards_per_cluster` oriented quads.
/// Returns vertices and indices ready to merge into the meshlet DAG builder.
pub fn build_foliage_billboards(
    params: &RedwoodParams,
    anchors: &[(Vec3, f32)],
    cards_per_cluster: u32,
) -> (Vec<Vertex>, Vec<u32>) {
    let total_cards = anchors.len() * cards_per_cluster as usize;
    let mut verts = Vec::with_capacity(total_cards * 4);
    let mut indices = Vec::with_capacity(total_cards * 6);
    let mut rng = TreeRng::new(params.seed.wrapping_add(0xf011a9e));

    for &(center, cluster_radius) in anchors {
        let card_size = cluster_radius * 0.6;

        for _ in 0..cards_per_cluster {
            // Random position within the cluster sphere
            let offset = random_sphere_point(&mut rng) * cluster_radius * rng.next_f32_range(0.3, 1.0);
            let pos = center + offset;

            // Random orientation (3 basis axes, not camera-facing — baked orientations)
            let normal = random_hemisphere_up(&mut rng);
            let right = normal.cross(Vec3::Y).normalize_or_zero();
            let right = if right.length_squared() > 0.01 {
                right
            } else {
                normal.cross(Vec3::X).normalize()
            };
            let up = right.cross(normal).normalize();

            let half = card_size * rng.next_f32_range(0.4, 0.7);
            let base_idx = verts.len() as u32;

            // 4 vertices for the quad
            let corners = [
                pos - right * half - up * half,
                pos + right * half - up * half,
                pos + right * half + up * half,
                pos - right * half + up * half,
            ];
            let uvs = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];

            // AO based on height (lower = more occluded)
            let ao = (pos.y / 40.0).clamp(0.3, 1.0);

            for (corner, uv) in corners.iter().zip(uvs.iter()) {
                verts.push(Vertex {
                    position: (*corner).into(),
                    normal: [normal.x, normal.y, normal.z],
                    material: MATERIAL_FOLIAGE,
                    uv: *uv,
                    ao,
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

    (verts, indices)
}

/// Density tiers for foliage LOD.
pub fn cards_per_cluster_for_tier(tier: u32) -> u32 {
    match tier {
        0 => 8,  // low
        1 => 16, // medium
        2 => 28, // high
        _ => 16,
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

fn random_hemisphere_up(rng: &mut TreeRng) -> Vec3 {
    let mut v = random_sphere_point(rng);
    if v.y < 0.0 {
        v.y = -v.y;
    }
    // Bias upward for tree foliage
    v.y = v.y * 0.5 + 0.5;
    v.normalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_quads_from_anchors() {
        let params = RedwoodParams::default();
        let anchors = vec![
            (Vec3::new(0.0, 20.0, 0.0), 3.0),
            (Vec3::new(5.0, 25.0, 2.0), 2.5),
        ];
        let (verts, indices) = build_foliage_billboards(&params, &anchors, 8);

        // 2 anchors * 8 cards * 4 verts = 64 verts
        assert_eq!(verts.len(), 64);
        // 2 anchors * 8 cards * 6 indices = 96 indices
        assert_eq!(indices.len(), 96);

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
        assert!(cards_per_cluster_for_tier(0) < cards_per_cluster_for_tier(1));
        assert!(cards_per_cluster_for_tier(1) < cards_per_cluster_for_tier(2));
    }

    #[test]
    fn foliage_merges_into_dag() {
        use crate::meshlet::build_meshlet_dag;
        use crate::subjects::tube_mesh::build_trunk_mesh;

        let params = RedwoodParams::default();
        let (mut trunk_verts, mut trunk_indices) = build_trunk_mesh(&params);

        // Generate foliage
        let anchors = vec![
            (Vec3::new(0.0, 25.0, 0.0), 3.0),
            (Vec3::new(2.0, 28.0, 1.0), 2.5),
        ];
        let (foliage_verts, foliage_indices) =
            build_foliage_billboards(&params, &anchors, 8);

        // Merge foliage into trunk mesh
        let vert_offset = trunk_verts.len() as u32;
        trunk_verts.extend(foliage_verts);
        trunk_indices.extend(foliage_indices.iter().map(|&i| i + vert_offset));

        let dag = build_meshlet_dag(trunk_verts, &trunk_indices);
        assert!(!dag.meshlets.is_empty());
        assert!(dag.level_offsets.len() >= 2);
    }
}
