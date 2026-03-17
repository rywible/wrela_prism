//! Forward render pass for the placeholder character capsule.
//!
//! Draws a single capsule mesh with Lambert shading + fog, writing to both
//! the HDR color target and the shared depth buffer so the character
//! participates in SSAO, sun shafts, bloom, and tonemapping.

use wgpu::util::DeviceExt;

use crate::subjects::capsule_mesh;

/// Per-frame model transform uploaded as a uniform.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ModelUniforms {
    model: [[f32; 4]; 4],
}

pub struct ForwardCharacterPass {
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    model_buffer: wgpu::Buffer,
    _model_bgl: wgpu::BindGroupLayout,
    model_bg: wgpu::BindGroup,
    /// Bind group layout for scene/lighting uniforms (group 0).
    scene_bgl: wgpu::BindGroupLayout,
}

impl ForwardCharacterPass {
    pub fn new(device: &wgpu::Device, lighting_buffer: &wgpu::Buffer) -> Self {
        let (vertices, indices) = capsule_mesh::build_capsule_mesh();

        // Flatten vertices to position + normal pairs for the shader.
        let vertex_data: Vec<[f32; 6]> = vertices
            .iter()
            .map(|v| {
                [
                    v.position[0],
                    v.position[1],
                    v.position[2],
                    v.normal[0],
                    v.normal[1],
                    v.normal[2],
                ]
            })
            .collect();

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("character-vertex-buffer"),
            contents: bytemuck::cast_slice(&vertex_data),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("character-index-buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let model_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("character-model-uniform"),
            size: std::mem::size_of::<ModelUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Scene/lighting uniform bind group layout (group 0).
        let scene_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("forward-char-scene-bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        // Model uniform bind group layout (group 1).
        let model_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("forward-char-model-bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let model_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("forward-char-model-bg"),
            layout: &model_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: model_buffer.as_entire_binding(),
            }],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("forward-character-shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/forward_character.wgsl").into(),
            ),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("forward-character-pipeline-layout"),
            bind_group_layouts: &[&scene_bgl, &model_bgl],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("forward-character-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<[f32; 6]>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        // position
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 0,
                            shader_location: 0,
                        },
                        // normal
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 12,
                            shader_location: 1,
                        },
                    ],
                }],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: super::HDR_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // We don't create the scene bind group here — it's created on demand
        // since the lighting buffer is owned by VisbufPipeline.
        let _ = lighting_buffer;

        Self {
            pipeline,
            vertex_buffer,
            index_buffer,
            index_count: indices.len() as u32,
            model_buffer,
            _model_bgl: model_bgl,
            model_bg,
            scene_bgl,
        }
    }

    /// Create the scene bind group (group 0) referencing the lighting uniform buffer.
    pub fn create_scene_bind_group(
        &self,
        device: &wgpu::Device,
        lighting_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("forward-char-scene-bg"),
            layout: &self.scene_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: lighting_buffer.as_entire_binding(),
            }],
        })
    }

    /// Upload the character's world transform for this frame.
    pub fn write_model(&self, queue: &wgpu::Queue, model_matrix: glam::Mat4) {
        let uniforms = ModelUniforms {
            model: model_matrix.to_cols_array_2d(),
        };
        queue.write_buffer(&self.model_buffer, 0, bytemuck::bytes_of(&uniforms));
    }

    /// Encode the forward character draw into the command encoder.
    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        scene_bg: &wgpu::BindGroup,
        color_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
    ) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("forward-character-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            multiview_mask: None,
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, scene_bg, &[]);
        pass.set_bind_group(1, &self.model_bg, &[]);
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..self.index_count, 0, 0..1);
    }
}
