use crate::scene::{Vertex, MATERIAL_GROUND};
use crate::util::smoothstep;

const GROUND_RING_FRACTIONS: &[f32] = &[0.005, 0.01, 0.02, 0.033, 0.08, 0.17, 0.5, 1.0];

fn ground_height_sample(x: f32, z: f32) -> f32 {
    let dist = (x * x + z * z).sqrt();
    let inner_relief = 1.0 - smoothstep(10.0, 24.0, dist);
    let root_zone = 1.0 - smoothstep(2.0, 11.0, dist);
    let broad_ripple = (x * 0.36 + z * 0.18).sin() * 0.028;
    let cross_ripple = (x * 0.22 - z * 0.31 + 1.3).sin() * 0.015;
    let detail = ((x * 1.7 + 0.4).sin() * (z * 1.3 - 0.7).cos()) * 0.008;
    let radial_relief = (dist * 0.62).sin() * 0.034 * root_zone;
    let buttress_echo = (x.atan2(z) * 4.0).sin() * 0.016 * root_zone;
    (broad_ripple + cross_ripple + detail) * inner_relief + radial_relief + buttress_echo
}

fn ground_normal_sample(x: f32, z: f32) -> [f32; 3] {
    let eps = 0.2;
    let dx = ground_height_sample(x + eps, z) - ground_height_sample(x - eps, z);
    let dz = ground_height_sample(x, z + eps) - ground_height_sample(x, z - eps);
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
        position: [0.0, ground_height_sample(0.0, 0.0), 0.0],
        normal: ground_normal_sample(0.0, 0.0),
        material: MATERIAL_GROUND,
        uv: [0.5, 0.5],
        ao: 0.55,
    });

    for &frac in ring_fractions {
        let r = frac * radius;
        for i in 0..segments {
            let angle = i as f32 / segments as f32 * std::f32::consts::TAU;
            let x = angle.cos() * r;
            let z = angle.sin() * r;
            let y = ground_height_sample(x, z);
            let dist = r;
            let ao = 0.55 + 0.45 * smoothstep(2.0, 10.0, dist);

            vertices.push(Vertex {
                position: [x, y, z],
                normal: ground_normal_sample(x, z),
                material: MATERIAL_GROUND,
                uv: [
                    0.5 + angle.cos() * 0.5 * frac,
                    0.5 + angle.sin() * 0.5 * frac,
                ],
                ao,
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
        uv: [0.5, 0.5],
        ao: 0.95,
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
                uv: [
                    0.5 + angle.cos() * 0.5 * frac,
                    0.5 + angle.sin() * 0.5 * frac,
                ],
                ao: 0.95,
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
    let skirt_bottom_radius = radius * 1.24;
    for i in 0..segments {
        let angle = i as f32 / segments as f32 * std::f32::consts::TAU;
        let x_top = angle.cos() * radius;
        let z_top = angle.sin() * radius;
        let x_bottom = angle.cos() * skirt_bottom_radius;
        let z_bottom = angle.sin() * skirt_bottom_radius;
        let top_y = ground_height_sample(x_top, z_top);
        let normal = glam::Vec3::new(angle.cos(), 0.48, angle.sin())
            .normalize_or_zero()
            .to_array();
        let u = i as f32 / segments as f32;

        vertices.push(Vertex {
            position: [x_top, top_y, z_top],
            normal,
            material: MATERIAL_GROUND,
            uv: [u, 0.0],
            ao: 0.92,
        });
        vertices.push(Vertex {
            position: [x_bottom, -thickness, z_bottom],
            normal,
            material: MATERIAL_GROUND,
            uv: [u, 1.0],
            ao: 0.95,
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
