use glam::{Affine3A, Vec3};

use crate::scene::{Vertex, MATERIAL_FOLIAGE};

/// Generate a single quad (2 triangles) for a leaf card.
/// The card is a billboard-like quad oriented by the given transform.
pub fn leaf_card_vertices(
    transform: &Affine3A,
    width: f32,
    height: f32,
) -> ([Vertex; 4], [u32; 6]) {
    let hw = width * 0.5;
    let hh = height * 0.5;

    // Local quad corners (in XY plane)
    let local_corners = [
        Vec3::new(-hw, -hh, 0.0),
        Vec3::new(hw, -hh, 0.0),
        Vec3::new(hw, hh, 0.0),
        Vec3::new(-hw, hh, 0.0),
    ];

    let local_normal = Vec3::Z;
    let world_normal = transform.transform_vector3(local_normal).normalize();

    let uvs = [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]];

    let vertices = std::array::from_fn(|i| {
        let world_pos = transform.transform_point3(local_corners[i]);
        Vertex {
            position: world_pos.to_array(),
            normal: world_normal.to_array(),
            material: MATERIAL_FOLIAGE,
            feature_id: 0,
            uv: uvs[i],
            ao: 0.85,
            semantic_channels: 0,
        }
    });

    let indices = [0, 1, 2, 0, 2, 3];
    (vertices, indices)
}

/// Generate a cluster of leaf cards around an anchor point.
/// Returns (vertices, indices) for all cards in the cluster.
pub fn leaf_cluster(
    anchor: Vec3,
    cluster_radius: f32,
    card_count: u32,
    card_width: f32,
    card_height: f32,
) -> (Vec<Vertex>, Vec<u32>) {
    let mut all_verts = Vec::with_capacity(card_count as usize * 4);
    let mut all_indices = Vec::with_capacity(card_count as usize * 6);

    for i in 0..card_count {
        let t = i as f32 / card_count as f32;
        let golden_angle = t * std::f32::consts::TAU * 0.618034; // golden ratio spiral

        // Distribute cards on a hemisphere around the anchor
        let y_offset = t * cluster_radius * 0.8 - cluster_radius * 0.2;
        let r = cluster_radius * (1.0 - t * 0.3);
        let offset = Vec3::new(golden_angle.cos() * r, y_offset, golden_angle.sin() * r);

        let card_pos = anchor + offset;

        // Orient card to face outward from anchor, with some randomization
        let outward = offset.normalize_or_zero();
        let up = Vec3::new((t * 13.7).sin() * 0.3, 1.0, (t * 7.3).cos() * 0.3).normalize();
        let right = up.cross(outward).normalize_or_zero();
        let actual_up = outward.cross(right).normalize_or_zero();

        let transform = Affine3A::from_cols(
            right.into(),
            actual_up.into(),
            outward.into(),
            card_pos.into(),
        );

        let base_idx = all_verts.len() as u32;
        let (verts, indices) = leaf_card_vertices(&transform, card_width, card_height);
        all_verts.extend_from_slice(&verts);
        for idx in indices {
            all_indices.push(base_idx + idx);
        }
    }

    (all_verts, all_indices)
}
