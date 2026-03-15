use bytemuck::{Pod, Zeroable};
use glam::Vec3;

/// Bounding sphere for a meshlet or meshlet group.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct MeshletBounds {
    pub center: [f32; 3],
    pub radius: f32,
    /// Normal cone axis (for backface culling).
    pub cone_axis: [f32; 3],
    /// Normal cone cutoff (dot threshold).
    pub cone_cutoff: f32,
}

impl MeshletBounds {
    pub fn from_meshopt(b: &meshopt::clusterize::Bounds) -> Self {
        Self {
            center: b.center,
            radius: b.radius,
            cone_axis: [b.cone_axis_s8[0] as f32 / 127.0, b.cone_axis_s8[1] as f32 / 127.0, b.cone_axis_s8[2] as f32 / 127.0],
            cone_cutoff: b.cone_cutoff_s8 as f32 / 127.0,
        }
    }

    /// Merge multiple bounds into one encompassing bounding sphere.
    pub fn merge(bounds_list: &[MeshletBounds]) -> Self {
        if bounds_list.is_empty() {
            return Self::zeroed();
        }
        if bounds_list.len() == 1 {
            return bounds_list[0];
        }

        // Ritter's bounding sphere algorithm
        let mut center = Vec3::from(bounds_list[0].center);
        let mut radius = bounds_list[0].radius;

        for b in &bounds_list[1..] {
            let other_center = Vec3::from(b.center);
            let other_radius = b.radius;
            let d = center.distance(other_center);

            if d + other_radius <= radius {
                // Already contained
                continue;
            }
            if d + radius <= other_radius {
                // Other contains us
                center = other_center;
                radius = other_radius;
                continue;
            }

            let new_radius = (d + radius + other_radius) * 0.5;
            let t = (new_radius - radius) / d;
            center = center + (other_center - center) * t;
            radius = new_radius;
        }

        Self {
            center: center.into(),
            radius,
            // Merged cone is invalid — disable cone culling for groups
            cone_axis: [0.0; 3],
            cone_cutoff: 1.0,
        }
    }
}
