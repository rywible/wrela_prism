use std::f32::consts::{PI, TAU};

use crate::scene::Vertex;

/// Longitude slices around the capsule axis.
const SLICES: u32 = 16;
/// Latitude rings per hemisphere cap (excluding the pole vertex).
const HEMI_RINGS: u32 = 5;
/// Rings along the cylinder body (excluding the two rings shared with the caps).
const CYL_RINGS: u32 = 4;

const TOTAL_HEIGHT: f32 = 1.8;
const RADIUS: f32 = 0.3;

/// Build a capsule mesh: cylinder body with hemisphere caps.
///
/// The capsule spans from y = 0 to y = 1.8 with radius 0.3.
/// Produces approximately 400 triangles with outward normals, simple UV mapping,
/// and AO = 1.0 throughout.
///
/// Returns `(vertices, indices)` ready for GPU upload.
pub fn build_capsule_mesh() -> (Vec<Vertex>, Vec<u32>) {
    // Derived dimensions.
    let cyl_bottom = RADIUS; // y = 0.3
    let cyl_top = TOTAL_HEIGHT - RADIUS; // y = 1.5
    let cyl_height = cyl_top - cyl_bottom;

    // Estimate vertex / index counts.
    //   Bottom pole + top pole = 2
    //   Hemisphere rings: 2 * HEMI_RINGS * SLICES
    //   Cylinder rings (interior): CYL_RINGS * SLICES
    //   The equator rings of each hemisphere serve as the cylinder endpoints,
    //   so no extra verts needed there.
    let ring_count = 2 * HEMI_RINGS + CYL_RINGS;
    let vert_count = 2 + ring_count as usize * SLICES as usize;
    // Triangles:
    //   Pole fans: 2 * SLICES
    //   Band quads: (ring_count - 1) * SLICES * 2
    let tri_count = 2 * SLICES as usize + (ring_count as usize - 1) * SLICES as usize * 2;
    let mut vertices = Vec::with_capacity(vert_count);
    let mut indices = Vec::with_capacity(tri_count * 3);

    // ---- Helper to push a ring of vertices at a given latitude (for caps)
    //      or at a given y (for cylinder). Returns the starting vertex index.

    // We build the mesh as a sequence of rings from bottom pole to top pole.
    // Ring order:
    //   0: bottom pole (single vertex)
    //   1..HEMI_RINGS: bottom hemisphere rings (latitude increasing from pole to equator)
    //   HEMI_RINGS+1 .. HEMI_RINGS+CYL_RINGS: cylinder interior rings
    //   HEMI_RINGS+CYL_RINGS+1 .. 2*HEMI_RINGS+CYL_RINGS: top hemisphere rings
    //   last: top pole (single vertex)

    // ---- Bottom pole ----
    vertices.push(Vertex {
        position: [0.0, 0.0, 0.0],
        normal: [0.0, -1.0, 0.0],
        material: 0,
        feature_id: 0,
        uv: [0.5, 0.0],
        ao: 1.0,
        semantic_channels: 0,
    });

    // ---- Bottom hemisphere rings (latitude from near-pole toward equator) ----
    for ring in 1..=HEMI_RINGS {
        // Latitude angle: 0 at pole (-Y), PI/2 at equator.
        let lat = (ring as f32 / HEMI_RINGS as f32) * (PI / 2.0);
        let cos_lat = lat.cos(); // distance below center along axis
        let sin_lat = lat.sin(); // radial distance from axis

        let y = cyl_bottom - cos_lat * RADIUS; // hemisphere center is at cyl_bottom
        let r = sin_lat * RADIUS;

        let ny = -cos_lat;
        let nr = sin_lat;

        let v = (RADIUS - cos_lat * RADIUS) / TOTAL_HEIGHT; // 0 at pole, RADIUS/H at equator

        for slice in 0..SLICES {
            let theta = slice as f32 / SLICES as f32 * TAU;
            let (sin_t, cos_t) = theta.sin_cos();

            vertices.push(Vertex {
                position: [cos_t * r, y, sin_t * r],
                normal: norm3(cos_t * nr, ny, sin_t * nr),
                material: 0,
                feature_id: 0,
                uv: [slice as f32 / SLICES as f32, v],
                ao: 1.0,
                semantic_channels: 0,
            });
        }
    }

    // ---- Cylinder interior rings ----
    for ring in 1..=CYL_RINGS {
        let t = ring as f32 / (CYL_RINGS + 1) as f32;
        let y = cyl_bottom + t * cyl_height;
        let v = (y) / TOTAL_HEIGHT;

        for slice in 0..SLICES {
            let theta = slice as f32 / SLICES as f32 * TAU;
            let (sin_t, cos_t) = theta.sin_cos();

            vertices.push(Vertex {
                position: [cos_t * RADIUS, y, sin_t * RADIUS],
                normal: norm3(cos_t, 0.0, sin_t),
                material: 0,
                feature_id: 0,
                uv: [slice as f32 / SLICES as f32, v],
                ao: 1.0,
                semantic_channels: 0,
            });
        }
    }

    // ---- Top hemisphere rings (latitude from equator toward pole) ----
    for ring in 1..=HEMI_RINGS {
        // ring 1 is near equator, HEMI_RINGS is near pole.
        // We go from lat = PI/2 (equator) down toward 0 (pole at +Y).
        let lat = (1.0 - ring as f32 / HEMI_RINGS as f32) * (PI / 2.0);
        let cos_lat = lat.cos();
        let sin_lat = lat.sin();

        let y = cyl_top + cos_lat * RADIUS; // hemisphere center at cyl_top
        let r = sin_lat * RADIUS;

        let ny = cos_lat;
        let nr = sin_lat;

        let v = y / TOTAL_HEIGHT;

        for slice in 0..SLICES {
            let theta = slice as f32 / SLICES as f32 * TAU;
            let (sin_t, cos_t) = theta.sin_cos();

            vertices.push(Vertex {
                position: [cos_t * r, y, sin_t * r],
                normal: norm3(cos_t * nr, ny, sin_t * nr),
                material: 0,
                feature_id: 0,
                uv: [slice as f32 / SLICES as f32, v],
                ao: 1.0,
                semantic_channels: 0,
            });
        }
    }

    // ---- Top pole ----
    let top_pole = vertices.len() as u32;
    vertices.push(Vertex {
        position: [0.0, TOTAL_HEIGHT, 0.0],
        normal: [0.0, 1.0, 0.0],
        material: 0,
        feature_id: 0,
        uv: [0.5, 1.0],
        ao: 1.0,
        semantic_channels: 0,
    });

    // ---- Indexing ----

    // Bottom pole fan: vertex 0 connects to ring 1 (vertices 1..1+SLICES).
    let first_ring_start: u32 = 1;
    for i in 0..SLICES {
        let next = (i + 1) % SLICES;
        indices.push(0); // bottom pole
        indices.push(first_ring_start + i);
        indices.push(first_ring_start + next);
    }

    // Band quads between consecutive rings.
    let total_rings = ring_count; // number of full rings (each has SLICES verts)
    for ring_idx in 0..total_rings - 1 {
        let cur_start = 1 + ring_idx * SLICES;
        let next_start = 1 + (ring_idx + 1) * SLICES;

        for i in 0..SLICES {
            let next = (i + 1) % SLICES;

            // Two triangles per quad.
            indices.push(cur_start + i);
            indices.push(next_start + next);
            indices.push(next_start + i);

            indices.push(cur_start + i);
            indices.push(cur_start + next);
            indices.push(next_start + next);
        }
    }

    // Top pole fan: top_pole connects to last ring.
    let last_ring_start = 1 + (total_rings - 1) * SLICES;
    for i in 0..SLICES {
        let next = (i + 1) % SLICES;
        indices.push(top_pole);
        indices.push(last_ring_start + next);
        indices.push(last_ring_start + i);
    }

    (vertices, indices)
}

fn norm3(x: f32, y: f32, z: f32) -> [f32; 3] {
    let v = glam::Vec3::new(x, y, z);
    let len = v.length();
    if len < 1e-8 {
        [0.0, 1.0, 0.0]
    } else {
        (v / len).to_array()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capsule_mesh_vertex_and_index_counts() {
        let (verts, idxs) = build_capsule_mesh();
        // Should have reasonable counts.
        assert!(!verts.is_empty());
        assert!(!idxs.is_empty());
        assert_eq!(idxs.len() % 3, 0, "index count must be a multiple of 3");
        let tri_count = idxs.len() / 3;
        assert!(
            (350..=450).contains(&tri_count),
            "expected ~400 triangles, got {tri_count}"
        );
    }

    #[test]
    fn capsule_mesh_bounds() {
        let (verts, _) = build_capsule_mesh();

        let min_y = verts.iter().map(|v| v.position[1]).fold(f32::MAX, f32::min);
        let max_y = verts.iter().map(|v| v.position[1]).fold(f32::MIN, f32::max);
        assert!(min_y.abs() < 1e-5, "bottom should be at y=0, got {min_y}");
        assert!(
            (max_y - TOTAL_HEIGHT).abs() < 1e-5,
            "top should be at y={TOTAL_HEIGHT}, got {max_y}"
        );

        // Radial extent should not exceed RADIUS (+ epsilon).
        for v in &verts {
            let r = (v.position[0] * v.position[0] + v.position[2] * v.position[2]).sqrt();
            assert!(
                r <= RADIUS + 1e-5,
                "vertex radial extent {r} exceeds radius {RADIUS}"
            );
        }
    }

    #[test]
    fn capsule_mesh_normals_are_unit_length() {
        let (verts, _) = build_capsule_mesh();
        for v in &verts {
            let n = glam::Vec3::from_array(v.normal);
            assert!(
                (n.length() - 1.0).abs() < 0.01,
                "normal not unit length: {:?} (len={})",
                v.normal,
                n.length()
            );
        }
    }

    #[test]
    fn capsule_mesh_indices_in_range() {
        let (verts, idxs) = build_capsule_mesh();
        let max_idx = verts.len() as u32;
        for &idx in &idxs {
            assert!(idx < max_idx, "index {idx} out of range (max {max_idx})");
        }
    }
}
