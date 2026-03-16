pub mod bark_bake;
pub mod procedural;

/// Material identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MaterialId(pub u32);

/// PBR material parameters.
#[derive(Clone, Copy, Debug)]
pub struct PbrParams {
    pub base_color: [f32; 3],
    pub metallic: f32,
    pub roughness: f32,
    pub ao: f32,
}

impl Default for PbrParams {
    fn default() -> Self {
        Self {
            base_color: [0.5, 0.5, 0.5],
            metallic: 0.0,
            roughness: 0.8,
            ao: 1.0,
        }
    }
}
