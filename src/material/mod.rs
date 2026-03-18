pub mod bark_bake;
pub mod procedural;

/// Material identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MaterialId(pub u32);
