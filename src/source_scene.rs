use glam::Affine3A;

use crate::growth::GrowthParams;
use crate::material::MaterialId;
use crate::scene::{bounds::Aabb, ParametricDef, SdfTree, Vertex};
use crate::soundstage::SoundstageLayout;
use crate::subjects::humanoid::HumanoidParams;
use crate::subjects::redwood_growth::RedwoodParams;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SourceNodeId(pub u32);

#[derive(Clone, Copy, Debug)]
pub struct SourceTransform {
    pub affine: Affine3A,
}

impl SourceTransform {
    pub const IDENTITY: Self = Self {
        affine: Affine3A::IDENTITY,
    };

    pub fn from_translation(translation: glam::Vec3) -> Self {
        Self {
            affine: Affine3A::from_translation(translation),
        }
    }
}

impl Default for SourceTransform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

#[derive(Clone, Copy, Debug)]
pub enum SourceBounds {
    Auto,
    Explicit(Aabb),
}

impl Default for SourceBounds {
    fn default() -> Self {
        Self::Auto
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SourceMaterialRef {
    pub override_material: Option<MaterialId>,
}

#[derive(Clone, Copy, Debug, Default)]
pub enum SourceLodPolicy {
    #[default]
    Auto,
    ForceMeshlets,
}

#[derive(Clone, Copy, Debug)]
pub struct ChunkingConfig {
    pub chunk_size: f32,
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self { chunk_size: 512.0 }
    }
}

#[derive(Clone, Debug)]
pub struct SourceTriangleMesh {
    pub label: String,
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

#[derive(Clone, Debug)]
pub enum ProceduralSubject {
    RedwoodTree {
        params: RedwoodParams,
        foliage_tier: u32,
    },
    GrowthTree {
        params: GrowthParams,
        foliage_tier: u32,
    },
    GroundSlab {
        radius: f32,
        thickness: f32,
        segments: u32,
    },
    Humanoid {
        params: HumanoidParams,
    },
}

#[derive(Clone, Debug)]
pub enum SourceGeometry {
    Sdf(SdfTree),
    Parametric(ParametricDef),
    ProceduralSubject(ProceduralSubject),
    TriangleMesh(SourceTriangleMesh),
    Instanced {
        base: Box<SourceGeometry>,
        transforms: Vec<SourceTransform>,
    },
}

#[derive(Clone, Debug)]
pub struct SourceNode {
    pub id: SourceNodeId,
    pub name: String,
    pub transform: SourceTransform,
    pub geometry: SourceGeometry,
    pub material: SourceMaterialRef,
    pub lod_policy: SourceLodPolicy,
    pub bounds: SourceBounds,
    pub casts_shadows: bool,
}

#[derive(Clone, Debug)]
pub struct SourceScene {
    pub label: String,
    pub nodes: Vec<SourceNode>,
    pub chunking: ChunkingConfig,
}

impl SourceScene {
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            nodes: Vec::new(),
            chunking: ChunkingConfig::default(),
        }
    }
}

pub struct SourceSceneBuilder {
    scene: SourceScene,
    next_id: u32,
}

impl SourceSceneBuilder {
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            scene: SourceScene::new(label),
            next_id: 0,
        }
    }

    pub fn with_chunk_size(mut self, chunk_size: f32) -> Self {
        self.scene.chunking.chunk_size = chunk_size.max(1.0);
        self
    }

    pub fn push_node(
        &mut self,
        name: impl Into<String>,
        geometry: SourceGeometry,
        transform: SourceTransform,
    ) -> SourceNodeId {
        let id = SourceNodeId(self.next_id);
        self.next_id += 1;
        self.scene.nodes.push(SourceNode {
            id,
            name: name.into(),
            transform,
            geometry,
            material: SourceMaterialRef::default(),
            lod_policy: SourceLodPolicy::Auto,
            bounds: SourceBounds::Auto,
            casts_shadows: true,
        });
        id
    }

    pub fn build(self) -> SourceScene {
        self.scene
    }

    pub fn redwood_soundstage(layout: &SoundstageLayout, seed: Option<u64>) -> SourceScene {
        let mut builder =
            Self::new("redwood_soundstage").with_chunk_size(layout.ground_radius * 4.0);
        let mut params = RedwoodParams::default();
        if let Some(seed) = seed {
            params.seed = seed;
        }
        builder.push_node(
            "hero_redwood",
            SourceGeometry::ProceduralSubject(ProceduralSubject::RedwoodTree {
                params,
                foliage_tier: 2,
            }),
            SourceTransform::IDENTITY,
        );
        builder.push_node(
            "ground_slab",
            SourceGeometry::ProceduralSubject(ProceduralSubject::GroundSlab {
                radius: layout.ground_radius,
                thickness: layout.ground_thickness,
                segments: 128,
            }),
            SourceTransform::IDENTITY,
        );
        builder.push_node(
            "humanoid",
            SourceGeometry::ProceduralSubject(ProceduralSubject::Humanoid {
                params: HumanoidParams::default(),
            }),
            SourceTransform::from_translation(glam::Vec3::new(15.0, 0.0, 10.0)),
        );
        builder.build()
    }

    pub fn redwood_soundstage_growth(layout: &SoundstageLayout, seed: Option<u64>) -> SourceScene {
        let mut builder =
            Self::new("redwood_soundstage_growth").with_chunk_size(layout.ground_radius * 4.0);
        let mut params = GrowthParams::default();
        if let Some(seed) = seed {
            params.seed = seed;
        }
        builder.push_node(
            "hero_redwood_growth",
            SourceGeometry::ProceduralSubject(ProceduralSubject::GrowthTree {
                params,
                foliage_tier: 1,
            }),
            SourceTransform::IDENTITY,
        );
        builder.push_node(
            "ground_slab",
            SourceGeometry::ProceduralSubject(ProceduralSubject::GroundSlab {
                radius: layout.ground_radius,
                thickness: layout.ground_thickness,
                segments: 128,
            }),
            SourceTransform::IDENTITY,
        );
        builder.push_node(
            "humanoid",
            SourceGeometry::ProceduralSubject(ProceduralSubject::Humanoid {
                params: HumanoidParams::default(),
            }),
            SourceTransform::from_translation(glam::Vec3::new(15.0, 0.0, 10.0)),
        );
        builder.build()
    }
}

#[cfg(test)]
mod tests {
    use super::{ProceduralSubject, SourceGeometry, SourceSceneBuilder, SourceTransform};

    #[test]
    fn redwood_soundstage_builder_produces_expected_nodes() {
        let scene = SourceSceneBuilder::redwood_soundstage(
            &crate::soundstage::redwood_stage::hero().layout,
            Some(7),
        );
        assert_eq!(scene.nodes.len(), 3);
        assert!(matches!(
            scene.nodes[0].geometry,
            SourceGeometry::ProceduralSubject(ProceduralSubject::RedwoodTree { .. })
        ));
    }

    #[test]
    fn instanced_geometry_can_be_authored() {
        let mut builder = SourceSceneBuilder::new("instanced");
        builder.push_node(
            "instanced_mesh",
            SourceGeometry::Instanced {
                base: Box::new(SourceGeometry::TriangleMesh(super::SourceTriangleMesh {
                    label: "tri".into(),
                    vertices: vec![],
                    indices: vec![],
                })),
                transforms: vec![SourceTransform::IDENTITY, SourceTransform::IDENTITY],
            },
            SourceTransform::IDENTITY,
        );
        let scene = builder.build();
        assert_eq!(scene.nodes.len(), 1);
    }
}
