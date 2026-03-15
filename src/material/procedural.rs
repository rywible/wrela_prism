use bytemuck::{Pod, Zeroable};
use crate::subjects::redwood_growth::RedwoodParams;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct BarkParams {
    pub trunk_radius_base: f32,
    pub trunk_radius_tip: f32,
    pub bark_thickness_base: f32,
    pub bark_thickness_tip: f32,
    pub trunk_height: f32,
    pub stiffness_ratio: f32,
    pub fiber_density: f32,
    pub seed: u32,
}

impl BarkParams {
    pub fn from_redwood(params: &RedwoodParams) -> Self {
        Self {
            trunk_radius_base: params.base_radius,
            trunk_radius_tip: params.tip_radius,
            bark_thickness_base: 0.42,
            bark_thickness_tip: 0.07,
            trunk_height: params.trunk_height,
            stiffness_ratio: 3.35,
            fiber_density: 8.5,
            seed: params.seed as u32,
        }
    }
}
