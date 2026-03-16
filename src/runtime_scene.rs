use std::collections::BTreeMap;

use glam::Affine3A;

use crate::gpu::upload::{upload_mesh, GpuMesh};
use crate::gpu::GpuContext;
use crate::material::MaterialId;
use crate::meshlet::{GpuMeshletBuffers, MeshletDag};
use crate::scene::bounds::{Aabb, BoundingSphere};
use crate::scene::Vertex;
use crate::scene_data::DEFAULT_FOLIAGE_ALPHA_SEED;
use crate::solver::projection::screen_diameter;
use crate::source_scene::{SourceNodeId, SourceTransform};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ChunkId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RuntimePrototypeId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RuntimeInstanceId(pub u32);

#[derive(Clone, Debug)]
pub struct PrototypeSurface {
    pub label: String,
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub material_id: MaterialId,
    pub casts_shadows: bool,
    pub alpha_tested: bool,
    pub local_bounds: Aabb,
}

#[derive(Clone, Debug)]
pub struct RuntimePrototype {
    pub id: RuntimePrototypeId,
    pub label: String,
    pub surfaces: Vec<PrototypeSurface>,
}

#[derive(Clone, Debug)]
pub struct RuntimeInstance {
    pub id: RuntimeInstanceId,
    pub source_node_id: SourceNodeId,
    pub prototype_id: RuntimePrototypeId,
    pub chunk_id: ChunkId,
    pub label: String,
    pub transform: SourceTransform,
    pub world_bounds: Aabb,
}

#[derive(Clone, Debug)]
pub struct RuntimeChunk {
    pub id: ChunkId,
    pub bounds: Aabb,
    pub instance_ids: Vec<RuntimeInstanceId>,
    pub prototype_ids: Vec<RuntimePrototypeId>,
}

#[derive(Clone, Debug)]
pub struct ChunkCompileInput {
    pub id: ChunkId,
    pub node_ids: Vec<SourceNodeId>,
    pub bounds: Aabb,
}

#[derive(Clone, Copy, Debug)]
pub struct MaterialEntry {
    pub id: MaterialId,
    pub alpha_tested: bool,
    pub casts_shadows: bool,
}

#[derive(Clone, Debug, Default)]
pub struct MaterialTable {
    pub materials: Vec<MaterialEntry>,
}

impl MaterialTable {
    pub fn register(&mut self, entry: MaterialEntry) {
        if self
            .materials
            .iter()
            .any(|existing| existing.id == entry.id)
        {
            return;
        }
        self.materials.push(entry);
    }
}

#[derive(Clone, Debug, Default)]
pub struct SceneSpatialIndex {
    pub chunk_bounds: Vec<(ChunkId, BoundingSphere)>,
}

impl SceneSpatialIndex {
    pub fn visible_chunks(
        &self,
        camera: &crate::camera::CameraState,
        width: u32,
        height: u32,
    ) -> Vec<ChunkId> {
        let view_proj = camera.view_projection_matrix();
        let camera_position = camera.position();
        self.chunk_bounds
            .iter()
            .filter_map(|(chunk_id, sphere)| {
                let diameter = screen_diameter(
                    sphere,
                    view_proj,
                    camera_position,
                    camera.fov_y_radians,
                    width as f32,
                    height as f32,
                );
                (diameter > 0.0).then_some(*chunk_id)
            })
            .collect()
    }
}

#[derive(Clone, Debug)]
pub struct MeshletSet {
    pub dag: MeshletDag,
}

#[derive(Clone, Debug)]
pub struct RuntimeScene {
    pub label: String,
    pub chunks: Vec<RuntimeChunk>,
    pub instances: Vec<RuntimeInstance>,
    pub prototypes: Vec<RuntimePrototype>,
    pub material_table: MaterialTable,
    pub spatial_index: SceneSpatialIndex,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChunkResidentState {
    Unloaded,
    LoadedCpu,
}

pub struct SceneResidencyManager {
    chunk_states: BTreeMap<ChunkId, ChunkResidentState>,
}

impl SceneResidencyManager {
    pub fn new(scene: &RuntimeScene) -> Self {
        let chunk_states = scene
            .chunks
            .iter()
            .map(|chunk| (chunk.id, ChunkResidentState::LoadedCpu))
            .collect();
        Self { chunk_states }
    }

    pub fn evict_chunk(&mut self, chunk_id: ChunkId) {
        self.chunk_states
            .insert(chunk_id, ChunkResidentState::Unloaded);
    }

    pub fn request_chunk(&mut self, chunk_id: ChunkId) {
        self.chunk_states
            .insert(chunk_id, ChunkResidentState::LoadedCpu);
    }

    pub fn loaded_chunks(&self) -> Vec<ChunkId> {
        self.chunk_states
            .iter()
            .filter_map(|(chunk_id, state)| {
                (*state == ChunkResidentState::LoadedCpu).then_some(*chunk_id)
            })
            .collect()
    }
}

pub struct RuntimeSceneGpu {
    pub runtime_scene: RuntimeScene,
    pub dag: MeshletDag,
    pub meshlet_buffers: GpuMeshletBuffers,
    pub shadow_meshes: Vec<GpuMesh>,
    pub shadow_opaque_list: Vec<usize>,
    pub shadow_transparent_list: Vec<usize>,
    pub alpha_mask_texture: wgpu::Texture,
    pub alpha_mask_view: wgpu::TextureView,
    pub bark_textures: crate::material::bark_bake::BarkTextures,
    pub resident_chunks: Vec<ChunkId>,
}

impl RuntimeSceneGpu {
    pub fn upload(
        gpu: &GpuContext,
        compiled: &crate::compiler::CompiledScene,
        residency: &SceneResidencyManager,
        bark_params: &crate::material::procedural::BarkParams,
    ) -> Self {
        let mut resident_chunks = residency.loaded_chunks();
        if resident_chunks.is_empty() {
            resident_chunks = compiled
                .runtime_scene
                .chunks
                .iter()
                .map(|chunk| chunk.id)
                .collect();
        }

        let mut dags = Vec::new();
        let mut shadow_meshes = Vec::new();
        let mut shadow_opaque_list = Vec::new();
        let mut shadow_transparent_list = Vec::new();

        for chunk in compiled
            .compiled_chunks
            .iter()
            .filter(|chunk| resident_chunks.contains(&chunk.id))
        {
            dags.push(chunk.meshlet_set.dag.clone());
        }

        for instance in compiled
            .runtime_scene
            .instances
            .iter()
            .filter(|instance| resident_chunks.contains(&instance.chunk_id))
        {
            let prototype = &compiled.runtime_scene.prototypes[instance.prototype_id.0 as usize];
            for surface in &prototype.surfaces {
                let (vertices, indices) =
                    expand_surface_with_transform(surface, instance.transform.affine);
                let mesh = upload_mesh(
                    &gpu.device,
                    &vertices,
                    &indices,
                    &format!("{}-{}", instance.label, surface.label),
                );
                let mesh_idx = shadow_meshes.len();
                shadow_meshes.push(mesh);
                if surface.casts_shadows {
                    if surface.alpha_tested {
                        shadow_transparent_list.push(mesh_idx);
                    } else {
                        shadow_opaque_list.push(mesh_idx);
                    }
                }
            }
        }

        let dag = merge_meshlet_dags(&dags);
        let meshlet_buffers = GpuMeshletBuffers::from_dag(&gpu.device, &dag);
        let (alpha_mask_texture, alpha_mask_view) =
            crate::subjects::alpha_mask::create_alpha_mask_texture(
                &gpu.device,
                &gpu.queue,
                DEFAULT_FOLIAGE_ALPHA_SEED,
            );

        let bark_textures =
            crate::material::bark_bake::create_bark_textures(&gpu.device, &gpu.queue, bark_params);

        Self {
            runtime_scene: compiled.runtime_scene.clone(),
            dag,
            meshlet_buffers,
            shadow_meshes,
            shadow_opaque_list,
            shadow_transparent_list,
            alpha_mask_texture,
            alpha_mask_view,
            bark_textures,
            resident_chunks,
        }
    }
}

pub(crate) fn compute_mesh_bounds(vertices: &[Vertex]) -> Aabb {
    let mut bounds = Aabb::empty();
    for vertex in vertices {
        bounds.expand_point(glam::Vec3::from_array(vertex.position));
    }
    bounds
}

pub(crate) fn transform_aabb(bounds: &Aabb, transform: Affine3A) -> Aabb {
    let corners = [
        bounds.min,
        glam::Vec3::new(bounds.max.x, bounds.min.y, bounds.min.z),
        glam::Vec3::new(bounds.min.x, bounds.max.y, bounds.min.z),
        glam::Vec3::new(bounds.min.x, bounds.min.y, bounds.max.z),
        glam::Vec3::new(bounds.max.x, bounds.max.y, bounds.min.z),
        glam::Vec3::new(bounds.max.x, bounds.min.y, bounds.max.z),
        glam::Vec3::new(bounds.min.x, bounds.max.y, bounds.max.z),
        bounds.max,
    ];

    let mut transformed = Aabb::empty();
    for corner in corners {
        transformed.expand_point(transform.transform_point3(corner));
    }
    transformed
}

pub(crate) fn expand_surface_with_transform(
    surface: &PrototypeSurface,
    transform: Affine3A,
) -> (Vec<Vertex>, Vec<u32>) {
    let vertices = surface
        .vertices
        .iter()
        .map(|vertex| crate::scene::transform_vertex(vertex, transform))
        .collect();
    (vertices, surface.indices.clone())
}

fn merge_meshlet_dags(dags: &[MeshletDag]) -> MeshletDag {
    if dags.is_empty() {
        return MeshletDag {
            meshlets: Vec::new(),
            meshlet_vertices: Vec::new(),
            meshlet_triangles: Vec::new(),
            groups: Vec::new(),
            vertices: Vec::new(),
            level_offsets: vec![0],
        };
    }
    if dags.len() == 1 {
        return dags[0].clone();
    }

    let mut merged = MeshletDag {
        meshlets: Vec::new(),
        meshlet_vertices: Vec::new(),
        meshlet_triangles: Vec::new(),
        groups: Vec::new(),
        vertices: Vec::new(),
        level_offsets: vec![0],
    };

    for dag in dags {
        let vertex_base = merged.vertices.len() as u32;
        let meshlet_vertex_base = merged.meshlet_vertices.len() as u32;
        let tri_base = merged.meshlet_triangles.len() as u32;
        let meshlet_base = merged.meshlets.len() as u32;
        let group_base = merged.groups.len() as u32;

        merged.vertices.extend_from_slice(&dag.vertices);
        merged
            .meshlet_vertices
            .extend(dag.meshlet_vertices.iter().map(|index| index + vertex_base));
        merged
            .meshlet_triangles
            .extend_from_slice(&dag.meshlet_triangles);

        merged
            .meshlets
            .extend(dag.meshlets.iter().cloned().map(|mut meshlet| {
                meshlet.vertex_offset += meshlet_vertex_base;
                meshlet.triangle_offset += tri_base;
                meshlet
            }));

        merged
            .groups
            .extend(dag.groups.iter().cloned().map(|mut group| {
                group.meshlet_start += meshlet_base;
                if group.child_count > 0 {
                    group.child_start += group_base;
                }
                group
            }));
    }

    merged
}

#[cfg(test)]
mod tests {
    use super::{
        compute_mesh_bounds, expand_surface_with_transform, transform_aabb, ChunkId, MaterialTable,
        PrototypeSurface, RuntimeChunk, RuntimeInstanceId, RuntimePrototype, RuntimePrototypeId,
        RuntimeScene, SceneResidencyManager, SceneSpatialIndex,
    };
    use crate::scene::{Vertex, MATERIAL_TRUNK};
    #[test]
    fn transform_aabb_moves_bounds() {
        let bounds = compute_mesh_bounds(&[Vertex {
            position: [0.0, 0.0, 0.0],
            normal: [0.0, 1.0, 0.0],
            material: MATERIAL_TRUNK,
            feature_id: 0,
            uv: [0.0, 0.0],
            ao: 1.0,
            semantic_channels: 0,
        }]);
        let moved = transform_aabb(
            &bounds,
            glam::Affine3A::from_translation(glam::Vec3::X * 4.0),
        );
        assert!((moved.center().x - 4.0).abs() < 0.01);
    }

    #[test]
    fn residency_manager_tracks_loaded_chunks() {
        let scene = RuntimeScene {
            label: "test".into(),
            chunks: vec![RuntimeChunk {
                id: ChunkId(0),
                bounds: crate::scene::bounds::Aabb::from_center_half_extents(
                    glam::Vec3::ZERO,
                    glam::Vec3::ONE,
                ),
                instance_ids: vec![RuntimeInstanceId(0)],
                prototype_ids: vec![RuntimePrototypeId(0)],
            }],
            instances: vec![],
            prototypes: vec![RuntimePrototype {
                id: RuntimePrototypeId(0),
                label: "proto".into(),
                surfaces: vec![PrototypeSurface {
                    label: "surface".into(),
                    vertices: vec![],
                    indices: vec![],
                    material_id: crate::material::MaterialId(MATERIAL_TRUNK),
                    casts_shadows: true,
                    alpha_tested: false,
                    local_bounds: crate::scene::bounds::Aabb::from_center_half_extents(
                        glam::Vec3::ZERO,
                        glam::Vec3::ONE,
                    ),
                }],
            }],
            material_table: MaterialTable::default(),
            spatial_index: SceneSpatialIndex::default(),
        };

        let mut residency = SceneResidencyManager::new(&scene);
        assert_eq!(residency.loaded_chunks(), vec![ChunkId(0)]);
        residency.evict_chunk(ChunkId(0));
        assert!(residency.loaded_chunks().is_empty());
    }

    #[test]
    fn expand_surface_applies_transform() {
        let surface = PrototypeSurface {
            label: "quad".into(),
            vertices: vec![Vertex {
                position: [1.0, 2.0, 3.0],
                normal: [0.0, 1.0, 0.0],
                material: MATERIAL_TRUNK,
                feature_id: 0,
                uv: [0.0, 0.0],
                ao: 1.0,
                semantic_channels: 0,
            }],
            indices: vec![0],
            material_id: crate::material::MaterialId(MATERIAL_TRUNK),
            casts_shadows: true,
            alpha_tested: false,
            local_bounds: crate::scene::bounds::Aabb::from_center_half_extents(
                glam::Vec3::new(1.0, 2.0, 3.0),
                glam::Vec3::ZERO,
            ),
        };
        let (vertices, _) = expand_surface_with_transform(
            &surface,
            glam::Affine3A::from_translation(glam::Vec3::new(4.0, 0.0, 0.0)),
        );
        assert_eq!(vertices[0].position, [5.0, 2.0, 3.0]);
    }
}
