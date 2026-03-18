use crate::scene::{Vertex, MATERIAL_GROUND};
use crate::util::smoothstep;

const GROUND_RING_FRACTIONS: &[f32] = &[
    0.004, 0.009, 0.018, 0.032, 0.055, 0.09, 0.16, 0.28, 0.46, 0.66, 0.82, 0.93, 1.0,
];

fn ground_height_sample(x: f32, z: f32, radius: f32) -> f32 {
    let dist = (x * x + z * z).sqrt();
    let inner_relief = 1.0 - smoothstep(18.0, 110.0, dist);
    let root_zone = 1.0 - smoothstep(2.5, 16.0, dist);
    let outer_plain = smoothstep(radius * 0.42, radius * 0.88, dist);
    let edge_sink = smoothstep(radius * 0.82, radius, dist);

    let broad_roll = (x * 0.018 + z * 0.011).sin() * 0.22;
    let broad_cross = (x * 0.010 - z * 0.016 + 1.7).sin() * 0.16;
    let shallow_swale = (dist * 0.020 + 0.8).sin() * 0.18;
    let mid_ripple = ((x * 0.085).sin() * (z * 0.074 - 0.6).cos()) * 0.08;
    let detail = ((x * 0.24 + 0.4).sin() * (z * 0.18 - 0.7).cos()) * 0.022;
    let radial_relief = (dist * 0.42).sin() * 0.12 * root_zone;
    let buttress_echo = (x.atan2(z) * 4.0).sin() * 0.10 * root_zone;

    let broad = (broad_roll + broad_cross + shallow_swale) * (1.0 - outer_plain * 0.45);
    let micro = mid_ripple * (1.0 - outer_plain * 0.82) + detail * inner_relief;

    broad + micro + radial_relief + buttress_echo - edge_sink * 0.8
}

fn ground_normal_sample(x: f32, z: f32, radius: f32) -> [f32; 3] {
    let eps = 0.75;
    let dx = ground_height_sample(x + eps, z, radius) - ground_height_sample(x - eps, z, radius);
    let dz = ground_height_sample(x, z + eps, radius) - ground_height_sample(x, z - eps, radius);
    glam::Vec3::new(-dx / (2.0 * eps), 1.0, -dz / (2.0 * eps))
        .normalize_or_zero()
        .to_array()
}

pub fn build_ground_slab(radius: f32, thickness: f32, segments: u32) -> (Vec<Vertex>, Vec<u32>) {
    let ring_fractions = GROUND_RING_FRACTIONS;
    let num_rings = ring_fractions.len();
    let surface_vert_count = 1 + num_rings * segments as usize;
    let vert_count = surface_vert_count * 2 + segments as usize * 2;
    let tri_count =
        (segments as usize + (num_rings - 1) * segments as usize * 2) * 2 + segments as usize * 2;
    let mut vertices = Vec::with_capacity(vert_count);
    let mut indices = Vec::with_capacity(tri_count * 3);

    vertices.push(Vertex {
        position: [0.0, ground_height_sample(0.0, 0.0, radius), 0.0],
        normal: ground_normal_sample(0.0, 0.0, radius),
        material: MATERIAL_GROUND,
        feature_id: 0,
        uv: [0.5, 0.5],
        ao: 0.58,
        semantic_channels: 0,
    });

    for &frac in ring_fractions {
        let r = frac * radius;
        for i in 0..segments {
            let angle = i as f32 / segments as f32 * std::f32::consts::TAU;
            let x = angle.cos() * r;
            let z = angle.sin() * r;
            let y = ground_height_sample(x, z, radius);
            let dist = r;
            let ao = 0.58 + 0.42 * smoothstep(3.0, 18.0, dist);

            vertices.push(Vertex {
                position: [x, y, z],
                normal: ground_normal_sample(x, z, radius),
                material: MATERIAL_GROUND,
                feature_id: 0,
                uv: [
                    0.5 + angle.cos() * 0.5 * frac,
                    0.5 + angle.sin() * 0.5 * frac,
                ],
                ao,
                semantic_channels: 0,
            });
        }
    }

    for i in 0..segments {
        let next = if i + 1 < segments { i + 1 } else { 0 };
        indices.push(0);
        indices.push(1 + next);
        indices.push(1 + i);
    }

    for ring_idx in 0..num_rings - 1 {
        let ring_start = 1 + ring_idx as u32 * segments;
        let next_ring_start = 1 + (ring_idx as u32 + 1) * segments;
        for i in 0..segments {
            let next = if i + 1 < segments { i + 1 } else { 0 };
            indices.push(ring_start + i);
            indices.push(next_ring_start + next);
            indices.push(next_ring_start + i);

            indices.push(ring_start + i);
            indices.push(ring_start + next);
            indices.push(next_ring_start + next);
        }
    }

    let bottom_center_idx = vertices.len() as u32;
    vertices.push(Vertex {
        position: [0.0, -thickness, 0.0],
        normal: [0.0, -1.0, 0.0],
        material: MATERIAL_GROUND,
        feature_id: 0,
        uv: [0.5, 0.5],
        ao: 0.95,
        semantic_channels: 0,
    });

    for &frac in ring_fractions {
        let r = frac * radius;
        for i in 0..segments {
            let angle = i as f32 / segments as f32 * std::f32::consts::TAU;
            let x = angle.cos() * r;
            let z = angle.sin() * r;
            vertices.push(Vertex {
                position: [x, -thickness, z],
                normal: [0.0, -1.0, 0.0],
                material: MATERIAL_GROUND,
                feature_id: 0,
                uv: [
                    0.5 + angle.cos() * 0.5 * frac,
                    0.5 + angle.sin() * 0.5 * frac,
                ],
                ao: 0.95,
                semantic_channels: 0,
            });
        }
    }

    let bottom_ring_offset = bottom_center_idx + 1;
    for i in 0..segments {
        let next = if i + 1 < segments { i + 1 } else { 0 };
        indices.push(bottom_center_idx);
        indices.push(bottom_ring_offset + i);
        indices.push(bottom_ring_offset + next);
    }

    for ring_idx in 0..num_rings - 1 {
        let ring_start = bottom_ring_offset + ring_idx as u32 * segments;
        let next_ring_start = bottom_ring_offset + (ring_idx as u32 + 1) * segments;
        for i in 0..segments {
            let next = if i + 1 < segments { i + 1 } else { 0 };
            indices.push(ring_start + i);
            indices.push(next_ring_start + i);
            indices.push(next_ring_start + next);

            indices.push(ring_start + i);
            indices.push(next_ring_start + next);
            indices.push(ring_start + next);
        }
    }

    let side_vert_offset = vertices.len() as u32;
    let skirt_bottom_radius = radius * 1.10;
    for i in 0..segments {
        let angle = i as f32 / segments as f32 * std::f32::consts::TAU;
        let x_top = angle.cos() * radius;
        let z_top = angle.sin() * radius;
        let x_bottom = angle.cos() * skirt_bottom_radius;
        let z_bottom = angle.sin() * skirt_bottom_radius;
        let top_y = ground_height_sample(x_top, z_top, radius);
        let normal = glam::Vec3::new(angle.cos(), 0.72, angle.sin())
            .normalize_or_zero()
            .to_array();
        let u = i as f32 / segments as f32;

        vertices.push(Vertex {
            position: [x_top, top_y, z_top],
            normal,
            material: MATERIAL_GROUND,
            feature_id: 0,
            uv: [u, 0.0],
            ao: 0.92,
            semantic_channels: 0,
        });
        vertices.push(Vertex {
            position: [x_bottom, -thickness, z_bottom],
            normal,
            material: MATERIAL_GROUND,
            feature_id: 0,
            uv: [u, 1.0],
            ao: 0.95,
            semantic_channels: 0,
        });
    }

    for i in 0..segments {
        let next = if i + 1 < segments { i + 1 } else { 0 };
        let top_i = side_vert_offset + i * 2;
        let bottom_i = top_i + 1;
        let top_next = side_vert_offset + next * 2;
        let bottom_next = top_next + 1;

        indices.push(top_i);
        indices.push(top_next);
        indices.push(bottom_next);

        indices.push(top_i);
        indices.push(bottom_next);
        indices.push(bottom_i);
    }

    (vertices, indices)
}

#[cfg(test)]
mod tests {
    use super::build_ground_slab;

    #[test]
    fn ground_slab_has_bottom_faces_and_normalized_normals() {
        let (vertices, indices) = build_ground_slab(120.0, 6.0, 64);
        assert!(!indices.is_empty());
        assert!(vertices.iter().any(|v| v.position[1] < -0.1));
        assert!(vertices.iter().any(|v| v.normal[1] < -0.9));
        assert!(vertices.iter().any(|v| v.normal[1] > 0.99));
        assert!(vertices.iter().all(|v| {
            let n = glam::Vec3::from_array(v.normal);
            (n.length() - 1.0).abs() < 0.01
        }));
    }
}
