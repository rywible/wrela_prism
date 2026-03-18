use std::collections::{BTreeMap, BTreeSet};

use crate::growth::GpuAccess;
use crate::material::MaterialId;
use crate::meshlet::build_meshlet_dag;
use crate::runtime_scene::{
    compute_mesh_bounds, transform_aabb, ChunkCompileInput, ChunkId, MaterialEntry, MaterialTable,
    MeshletSet, PrototypeSurface, RuntimeChunk, RuntimeInstance, RuntimeInstanceId,
    RuntimePrototype, RuntimePrototypeId, RuntimeScene, SceneSpatialIndex,
};
use crate::scene::bounds::{Aabb, BoundingSphere};
use crate::scene::Vertex;
use crate::source_scene::{
    ProceduralSubject, SourceGeometry, SourceMaterialRef, SourceNode, SourceScene, SourceTransform,
    SourceTriangleMesh,
};

#[derive(Clone, Debug)]
pub struct CompiledChunk {
    pub id: ChunkId,
    pub bounds: Aabb,
    pub instance_ids: Vec<RuntimeInstanceId>,
    pub prototype_ids: Vec<RuntimePrototypeId>,
    pub meshlet_set: MeshletSet,
    /// Vertex ranges within the chunk's DAG for animated surfaces.
    /// Each entry: `(label, start_vertex_index, vertex_count)`.
    pub animated_vertex_ranges: Vec<(String, u32, u32)>,
}

#[derive(Clone, Debug)]
pub struct CompiledScene {
    pub runtime_scene: RuntimeScene,
    pub chunk_compile_inputs: Vec<ChunkCompileInput>,
    pub compiled_chunks: Vec<CompiledChunk>,
    pub diagnostics: Vec<String>,
}

pub struct SceneCompiler;

impl SceneCompiler {
    pub fn new() -> Self {
        Self
    }

    pub fn compile(&self, source: &SourceScene) -> CompiledScene {
        self.compile_with_gpu(source, None)
    }

    pub fn compile_with_gpu(
        &self,
        source: &SourceScene,
        gpu: Option<GpuAccess<'_>>,
    ) -> CompiledScene {
        let mut diagnostics = Vec::new();
        let mut material_table = MaterialTable::default();
        let mut prototypes = Vec::new();
        let mut instances = Vec::new();
        let mut chunk_key_to_id = BTreeMap::<(i32, i32), ChunkId>::new();
        let mut chunk_instance_ids = BTreeMap::<ChunkId, Vec<RuntimeInstanceId>>::new();
        let mut chunk_prototype_ids = BTreeMap::<ChunkId, BTreeSet<RuntimePrototypeId>>::new();
        let mut next_prototype_id = 0u32;
        let mut next_instance_id = 0u32;

        for node in &source.nodes {
            self.compile_node(
                source,
                node,
                gpu.as_ref(),
                &mut next_prototype_id,
                &mut next_instance_id,
                &mut prototypes,
                &mut instances,
                &mut material_table,
                &mut chunk_key_to_id,
                &mut chunk_instance_ids,
                &mut chunk_prototype_ids,
                &mut diagnostics,
            );
        }

        let chunks = build_runtime_chunks(&instances, &chunk_instance_ids, &chunk_prototype_ids);
        let chunk_compile_inputs = chunks
            .iter()
            .map(|chunk| ChunkCompileInput {
                id: chunk.id,
                node_ids: chunk
                    .instance_ids
                    .iter()
                    .map(|instance_id| instances[instance_id.0 as usize].source_node_id)
                    .collect(),
                bounds: chunk.bounds,
            })
            .collect::<Vec<_>>();

        let spatial_index = SceneSpatialIndex {
            chunk_bounds: chunks
                .iter()
                .map(|chunk| (chunk.id, BoundingSphere::from_aabb(&chunk.bounds)))
                .collect(),
        };

        let runtime_scene = RuntimeScene {
            label: source.label.clone(),
            chunks: chunks.clone(),
            instances,
            prototypes,
            material_table,
            spatial_index,
        };

        let compiled_chunks = chunks
            .iter()
            .map(|chunk| compile_chunk(chunk, &runtime_scene))
            .collect();

        CompiledScene {
            runtime_scene,
            chunk_compile_inputs,
            compiled_chunks,
            diagnostics,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn compile_node(
        &self,
        source: &SourceScene,
        node: &SourceNode,
        gpu: Option<&GpuAccess<'_>>,
        next_prototype_id: &mut u32,
        next_instance_id: &mut u32,
        prototypes: &mut Vec<RuntimePrototype>,
        instances: &mut Vec<RuntimeInstance>,
        material_table: &mut MaterialTable,
        chunk_key_to_id: &mut BTreeMap<(i32, i32), ChunkId>,
        chunk_instance_ids: &mut BTreeMap<ChunkId, Vec<RuntimeInstanceId>>,
        chunk_prototype_ids: &mut BTreeMap<ChunkId, BTreeSet<RuntimePrototypeId>>,
        diagnostics: &mut Vec<String>,
    ) {
        match &node.geometry {
            SourceGeometry::Instanced { base, transforms } => {
                let Some(prototype) = realize_geometry_to_prototype(
                    RuntimePrototypeId(*next_prototype_id),
                    &node.name,
                    base,
                    node.material,
                    node.casts_shadows,
                    gpu,
                    diagnostics,
                ) else {
                    return;
                };
                for surface in &prototype.surfaces {
                    material_table.register(MaterialEntry {
                        id: surface.material_id,
                        casts_shadows: surface.casts_shadows,
                    });
                }

                let prototype_id = prototype.id;
                *next_prototype_id += 1;
                prototypes.push(prototype.clone());

                for instanced_transform in transforms {
                    let combined = SourceTransform {
                        affine: node.transform.affine * instanced_transform.affine,
                    };
                    let world_bounds = prototype_world_bounds(&prototype, combined);
                    let chunk_id = assign_chunk(source, chunk_key_to_id, &world_bounds);
                    let instance_id = RuntimeInstanceId(*next_instance_id);
                    *next_instance_id += 1;
                    instances.push(RuntimeInstance {
                        id: instance_id,
                        source_node_id: node.id,
                        prototype_id,
                        chunk_id,
                        label: node.name.clone(),
                        transform: combined,
                        world_bounds,
                    });
                    chunk_instance_ids
                        .entry(chunk_id)
                        .or_default()
                        .push(instance_id);
                    chunk_prototype_ids
                        .entry(chunk_id)
                        .or_default()
                        .insert(prototype_id);
                }
            }
            geometry => {
                let Some(prototype) = realize_geometry_to_prototype(
                    RuntimePrototypeId(*next_prototype_id),
                    &node.name,
                    geometry,
                    node.material,
                    node.casts_shadows,
                    gpu,
                    diagnostics,
                ) else {
                    return;
                };
                for surface in &prototype.surfaces {
                    material_table.register(MaterialEntry {
                        id: surface.material_id,
                        casts_shadows: surface.casts_shadows,
                    });
                }

                let prototype_id = prototype.id;
                *next_prototype_id += 1;
                let world_bounds = prototype_world_bounds(&prototype, node.transform);
                let chunk_id = assign_chunk(source, chunk_key_to_id, &world_bounds);
                let instance_id = RuntimeInstanceId(*next_instance_id);
                *next_instance_id += 1;
                prototypes.push(prototype);
                instances.push(RuntimeInstance {
                    id: instance_id,
                    source_node_id: node.id,
                    prototype_id,
                    chunk_id,
                    label: node.name.clone(),
                    transform: node.transform,
                    world_bounds,
                });
                chunk_instance_ids
                    .entry(chunk_id)
                    .or_default()
                    .push(instance_id);
                chunk_prototype_ids
                    .entry(chunk_id)
                    .or_default()
                    .insert(prototype_id);
            }
        }
    }
}

fn build_runtime_chunks(
    instances: &[RuntimeInstance],
    chunk_instance_ids: &BTreeMap<ChunkId, Vec<RuntimeInstanceId>>,
    chunk_prototype_ids: &BTreeMap<ChunkId, BTreeSet<RuntimePrototypeId>>,
) -> Vec<RuntimeChunk> {
    chunk_instance_ids
        .iter()
        .map(|(chunk_id, instance_ids)| {
            let mut bounds = Aabb::empty();
            for instance_id in instance_ids {
                bounds = bounds.union(&instances[instance_id.0 as usize].world_bounds);
            }
            RuntimeChunk {
                id: *chunk_id,
                bounds,
                instance_ids: instance_ids.clone(),
                prototype_ids: chunk_prototype_ids
                    .get(chunk_id)
                    .map(|ids| ids.iter().copied().collect())
                    .unwrap_or_default(),
            }
        })
        .collect()
}

fn compile_chunk(chunk: &RuntimeChunk, runtime_scene: &RuntimeScene) -> CompiledChunk {
    use crate::runtime_scene::merge_meshlet_dags;

    let mut dags = Vec::new();
    // Track which DAG index corresponds to an animated surface: (dag_index, vertex_count, label).
    let mut animated_dag_info: Vec<(usize, u32, String)> = Vec::new();

    for instance_id in &chunk.instance_ids {
        let instance = &runtime_scene.instances[instance_id.0 as usize];
        let prototype = &runtime_scene.prototypes[instance.prototype_id.0 as usize];
        for surface in &prototype.surfaces {
            let vertices = surface
                .vertices
                .iter()
                .map(|vertex| crate::scene::transform_vertex(vertex, instance.transform.affine))
                .collect::<Vec<_>>();
            let vert_count = vertices.len() as u32;
            let mut dag = build_meshlet_dag(vertices, &surface.indices);
            if surface.animated {
                // Inflate meshlet bounds for animated geometry so it isn't culled during motion
                for meshlet in &mut dag.meshlets {
                    meshlet.bounds.radius += 0.54;
                }
                animated_dag_info.push((dags.len(), vert_count, surface.label.clone()));
            }
            dags.push(dag);
        }
    }

    let (merged_dag, vertex_bases) = merge_meshlet_dags(&dags);

    // Compute animated vertex ranges for ALL animated surfaces
    let animated_vertex_ranges: Vec<(String, u32, u32)> = animated_dag_info
        .iter()
        .map(|(dag_idx, count, label)| (label.clone(), vertex_bases[*dag_idx], *count))
        .collect();

    let meshlet_set = MeshletSet { dag: merged_dag };

    CompiledChunk {
        id: chunk.id,
        bounds: chunk.bounds,
        instance_ids: chunk.instance_ids.clone(),
        prototype_ids: chunk.prototype_ids.clone(),
        meshlet_set,
        animated_vertex_ranges,
    }
}

fn realize_geometry_to_prototype(
    prototype_id: RuntimePrototypeId,
    label: &str,
    geometry: &SourceGeometry,
    material: SourceMaterialRef,
    casts_shadows: bool,
    gpu: Option<&GpuAccess<'_>>,
    diagnostics: &mut Vec<String>,
) -> Option<RuntimePrototype> {
    let surfaces = match geometry {
        SourceGeometry::ProceduralSubject(subject) => {
            realize_procedural_subject(label, subject, material, casts_shadows, gpu)
        }
        SourceGeometry::TriangleMesh(mesh) => {
            vec![build_surface_from_mesh(mesh, material, casts_shadows)]
        }
        SourceGeometry::Sdf(_) => {
            diagnostics.push(format!(
                "prototype '{label}' uses SDF geometry, which is not yet realized"
            ));
            return None;
        }
        SourceGeometry::Parametric(_) => {
            diagnostics.push(format!(
                "prototype '{label}' uses parametric geometry, which is not yet realized"
            ));
            return None;
        }
        SourceGeometry::Instanced { .. } => {
            diagnostics.push(format!(
                "prototype '{label}' nested instancing is not supported"
            ));
            return None;
        }
    };

    Some(RuntimePrototype {
        id: prototype_id,
        label: label.to_string(),
        surfaces,
    })
}

fn realize_procedural_subject(
    label: &str,
    subject: &ProceduralSubject,
    material: SourceMaterialRef,
    casts_shadows: bool,
    gpu: Option<&GpuAccess<'_>>,
) -> Vec<PrototypeSurface> {
    match subject {
        ProceduralSubject::RedwoodTree {
            params,
            foliage_tier,
        } => {
            let skeleton = crate::subjects::redwood_growth::build_skeleton(params);
            let (trunk_vertices, trunk_indices) =
                crate::subjects::tube_mesh::build_trunk_mesh_from_skeleton(params, &skeleton);
            let anchors = crate::subjects::redwood_growth::generate_foliage_anchors_from_skeleton(
                params, &skeleton,
            );
            let (foliage_vertices, foliage_indices) =
                crate::subjects::needle_sprays::build_needle_sprays(
                    params,
                    &anchors,
                    crate::subjects::needle_sprays::sprays_per_cluster_for_tier(*foliage_tier),
                );
            vec![
                build_surface(
                    format!("{label}/trunk"),
                    trunk_vertices,
                    trunk_indices,
                    material
                        .override_material
                        .unwrap_or(MaterialId(crate::scene::MATERIAL_TRUNK)),
                    casts_shadows,
                ),
                build_surface(
                    format!("{label}/foliage"),
                    foliage_vertices,
                    foliage_indices,
                    material
                        .override_material
                        .unwrap_or(MaterialId(crate::scene::MATERIAL_FOLIAGE)),
                    casts_shadows,
                ),
            ]
        }
        ProceduralSubject::GrowthTree {
            params,
            foliage_tier,
        } => {
            let gpu_access = gpu.map(|g| crate::growth::GpuAccess {
                device: g.device,
                queue: g.queue,
            });
            let growth_tree = crate::growth::grow_tree(params, gpu_access);
            let skeleton = crate::growth::skeleton::growth_to_skeleton(&growth_tree);
            let mesh_params =
                crate::growth::skeleton::growth_mesh_params_from_tree(params, &growth_tree);
            let (trunk_vertices, trunk_indices) =
                crate::subjects::tube_mesh::build_trunk_mesh_from_skeleton(&mesh_params, &skeleton);
            let anchors = crate::growth::skeleton::growth_foliage_anchors(&growth_tree);
            let (foliage_vertices, foliage_indices) =
                crate::subjects::needle_sprays::build_needle_sprays(
                    &mesh_params,
                    &anchors,
                    crate::subjects::needle_sprays::sprays_per_cluster_for_tier(*foliage_tier),
                );
            vec![
                build_surface(
                    format!("{label}/trunk"),
                    trunk_vertices,
                    trunk_indices,
                    material
                        .override_material
                        .unwrap_or(MaterialId(crate::scene::MATERIAL_TRUNK)),
                    casts_shadows,
                ),
                build_surface(
                    format!("{label}/foliage"),
                    foliage_vertices,
                    foliage_indices,
                    material
                        .override_material
                        .unwrap_or(MaterialId(crate::scene::MATERIAL_FOLIAGE)),
                    casts_shadows,
                ),
            ]
        }
        ProceduralSubject::GroundSlab {
            radius,
            thickness,
            segments,
        } => {
            let (vertices, indices) =
                crate::subjects::ground_slab::build_ground_slab(*radius, *thickness, *segments);
            vec![build_surface(
                format!("{label}/ground"),
                vertices,
                indices,
                material
                    .override_material
                    .unwrap_or(MaterialId(crate::scene::MATERIAL_GROUND)),
                casts_shadows,
            )]
        }
        ProceduralSubject::Humanoid { params } => {
            let skeleton = crate::subjects::humanoid::build_skeleton(params);
            let (vertices, indices) = crate::subjects::humanoid::build_body_mesh(&skeleton, params);
            vec![build_animated_surface(
                format!("{label}/body"),
                vertices,
                indices,
                material
                    .override_material
                    .unwrap_or(MaterialId(crate::scene::MATERIAL_SKIN)),
                casts_shadows,
            )]
        }
    }
}

fn build_surface_from_mesh(
    mesh: &SourceTriangleMesh,
    material: SourceMaterialRef,
    casts_shadows: bool,
) -> PrototypeSurface {
    let override_material = material.override_material.map(|id| id.0);
    let vertices = mesh
        .vertices
        .iter()
        .map(|vertex| {
            let mut v = *vertex;
            if let Some(override_material) = override_material {
                v.material = override_material;
            }
            v
        })
        .collect::<Vec<_>>();

    build_surface(
        mesh.label.clone(),
        vertices,
        mesh.indices.clone(),
        material
            .override_material
            .unwrap_or(MaterialId(mesh.vertices.first().map_or(0, |v| v.material))),
        casts_shadows,
    )
}

fn build_surface(
    label: String,
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
    material_id: MaterialId,
    casts_shadows: bool,
) -> PrototypeSurface {
    let mut vertices = vertices;
    for vertex in &mut vertices {
        vertex.material = material_id.0;
    }
    let local_bounds = compute_mesh_bounds(&vertices);
    PrototypeSurface {
        label,
        vertices,
        indices,
        material_id,
        casts_shadows,
        animated: false,
        local_bounds,
    }
}

fn build_animated_surface(
    label: String,
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
    material_id: MaterialId,
    casts_shadows: bool,
) -> PrototypeSurface {
    let mut surface = build_surface(label, vertices, indices, material_id, casts_shadows);
    surface.animated = true;
    surface
}

fn prototype_world_bounds(prototype: &RuntimePrototype, transform: SourceTransform) -> Aabb {
    prototype
        .surfaces
        .iter()
        .fold(Aabb::empty(), |bounds, surface| {
            bounds.union(&transform_aabb(&surface.local_bounds, transform.affine))
        })
}

fn assign_chunk(
    source: &SourceScene,
    chunk_key_to_id: &mut BTreeMap<(i32, i32), ChunkId>,
    world_bounds: &Aabb,
) -> ChunkId {
    let chunk_size = source.chunking.chunk_size.max(1.0);
    let center = world_bounds.center();
    let key = (
        (center.x / chunk_size).floor() as i32,
        (center.z / chunk_size).floor() as i32,
    );
    if let Some(chunk_id) = chunk_key_to_id.get(&key) {
        *chunk_id
    } else {
        let chunk_id = ChunkId(chunk_key_to_id.len() as u32);
        chunk_key_to_id.insert(key, chunk_id);
        chunk_id
    }
}

#[cfg(test)]
mod tests {
    use crate::compiler::SceneCompiler;
    use crate::meshlet::make_grid;
    use crate::scene::{Vertex, MATERIAL_TRUNK};
    use crate::source_scene::{
        SourceGeometry, SourceSceneBuilder, SourceTransform, SourceTriangleMesh,
    };

    fn compile_signature(scene: &crate::source_scene::SourceScene) -> (usize, usize, usize, usize) {
        let compiled = SceneCompiler::new().compile(scene);
        (
            compiled.runtime_scene.prototypes.len(),
            compiled.runtime_scene.instances.len(),
            compiled.runtime_scene.chunks.len(),
            compiled
                .compiled_chunks
                .iter()
                .map(|chunk| chunk.meshlet_set.dag.meshlets.len())
                .sum(),
        )
    }

    #[test]
    fn source_scene_compiles_deterministically() {
        let scene = SourceSceneBuilder::redwood_soundstage(
            &crate::soundstage::redwood_stage::hero().layout,
            Some(13),
        );
        assert_eq!(compile_signature(&scene), compile_signature(&scene));
    }

    #[test]
    fn instanced_content_creates_shared_prototype_and_distinct_instances() {
        let triangle = SourceTriangleMesh {
            label: "triangle".into(),
            vertices: vec![
                Vertex {
                    position: [0.0, 0.0, 0.0],
                    normal: [0.0, 1.0, 0.0],
                    material: MATERIAL_TRUNK,
                    feature_id: 0,
                    uv: [0.0, 0.0],
                    ao: 1.0,
                    semantic_channels: 0,
                },
                Vertex {
                    position: [1.0, 0.0, 0.0],
                    normal: [0.0, 1.0, 0.0],
                    material: MATERIAL_TRUNK,
                    feature_id: 0,
                    uv: [1.0, 0.0],
                    ao: 1.0,
                    semantic_channels: 0,
                },
                Vertex {
                    position: [0.0, 0.0, 1.0],
                    normal: [0.0, 1.0, 0.0],
                    material: MATERIAL_TRUNK,
                    feature_id: 0,
                    uv: [0.0, 1.0],
                    ao: 1.0,
                    semantic_channels: 0,
                },
            ],
            indices: vec![0, 1, 2],
        };
        let mut builder = SourceSceneBuilder::new("instanced").with_chunk_size(4.0);
        builder.push_node(
            "forest_patch",
            SourceGeometry::Instanced {
                base: Box::new(SourceGeometry::TriangleMesh(triangle)),
                transforms: vec![
                    SourceTransform::from_translation(glam::Vec3::ZERO),
                    SourceTransform::from_translation(glam::Vec3::new(8.0, 0.0, 0.0)),
                ],
            },
            SourceTransform::IDENTITY,
        );
        let compiled = SceneCompiler::new().compile(&builder.build());
        assert_eq!(compiled.runtime_scene.prototypes.len(), 1);
        assert_eq!(compiled.runtime_scene.instances.len(), 2);
        assert_eq!(compiled.runtime_scene.chunks.len(), 2);
    }

    fn group_material(
        dag: &crate::meshlet::MeshletDag,
        group: &crate::meshlet::MeshletGroup,
    ) -> u32 {
        let meshlet = &dag.meshlets[group.meshlet_start as usize];
        let global_vertex = dag.meshlet_vertices[meshlet.vertex_offset as usize] as usize;
        dag.vertices[global_vertex].material
    }

    #[test]
    fn compiled_chunk_dag_keeps_parent_groups_within_surface_material_boundaries() {
        let (trunk_vertices, trunk_indices) = make_grid(10);
        let (mut ground_vertices, ground_indices) = make_grid(10);
        for vertex in &mut ground_vertices {
            vertex.material = crate::scene::MATERIAL_GROUND;
        }

        let mut builder = SourceSceneBuilder::new("mixed_chunk").with_chunk_size(200.0);
        builder.push_node(
            "trunk_patch",
            SourceGeometry::TriangleMesh(SourceTriangleMesh {
                label: "trunk".into(),
                vertices: trunk_vertices,
                indices: trunk_indices,
            }),
            SourceTransform::IDENTITY,
        );
        builder.push_node(
            "ground_patch",
            SourceGeometry::TriangleMesh(SourceTriangleMesh {
                label: "ground".into(),
                vertices: ground_vertices,
                indices: ground_indices,
            }),
            SourceTransform::from_translation(glam::Vec3::new(40.0, 0.0, 0.0)),
        );

        let compiled = SceneCompiler::new().compile(&builder.build());
        let dag = &compiled.compiled_chunks[0].meshlet_set.dag;

        for group in dag.groups.iter().filter(|group| group.child_count > 0) {
            let expected = group_material(dag, &dag.groups[group.child_start as usize]);
            for child_idx in group.child_start..group.child_start + group.child_count {
                let child_material = group_material(dag, &dag.groups[child_idx as usize]);
                assert_eq!(
                    child_material, expected,
                    "parent group should stay within a single surface/material bucket"
                );
            }
        }
    }
}
