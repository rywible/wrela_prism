use glam::Mat4;

use crate::gpu::upload::GpuMesh;
use crate::gpu::GpuContext;
use crate::scene::shadow::ShadowMap;
use crate::scene::Vertex;
use crate::scene::NUM_CASCADES;

/// Single cascade light VP — one per dynamic offset slot.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShadowCascadeUniforms {
    pub light_vp: [[f32; 4]; 4],
}

pub struct ShadowPass {
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    pipeline: wgpu::RenderPipeline,
    cascade_views: [wgpu::TextureView; NUM_CASCADES],
    /// Aligned offset between cascade uniform slots.
    dyn_alignment: u32,
}

impl ShadowPass {
    pub fn new(gpu: &GpuContext, shadow_map: &ShadowMap) -> Self {
        let device = &gpu.device;

        // Dynamic uniform offset alignment (typically 256 on most GPUs)
        let min_align = device.limits().min_uniform_buffer_offset_alignment as u64;
        let slot_size = std::mem::size_of::<ShadowCascadeUniforms>() as u64;
        let dyn_alignment = (slot_size.div_ceil(min_align) * min_align) as u32;

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("prism-shadow-uniforms"),
            size: dyn_alignment as u64 * NUM_CASCADES as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("prism-shadow-bind-group-layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: Some(
                        std::num::NonZeroU64::new(
                            std::mem::size_of::<ShadowCascadeUniforms>() as u64
                        )
                        .unwrap(),
                    ),
                },
                count: None,
            }],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("prism-shadow-bind-group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &uniform_buffer,
                    offset: 0,
                    size: Some(
                        std::num::NonZeroU64::new(
                            std::mem::size_of::<ShadowCascadeUniforms>() as u64
                        )
                        .unwrap(),
                    ),
                }),
            }],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("prism-shadow-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/shadow_pass.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("prism-shadow-pipeline-layout"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        let depth_stencil = wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState {
                constant: 2,
                slope_scale: 2.0,
                clamp: 0.0,
            },
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("prism-shadow-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_shadow"),
                buffers: &[Vertex::layout()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_shadow"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(depth_stencil.clone()),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let cascade_views = std::array::from_fn(|i| {
            shadow_map
                .texture
                .create_view(&wgpu::TextureViewDescriptor {
                    label: Some(&format!("prism-shadow-pass-cascade-{i}")),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: i as u32,
                    array_layer_count: Some(1),
                    ..Default::default()
                })
        });

        Self {
            uniform_buffer,
            bind_group,
            pipeline,
            cascade_views,
            dyn_alignment,
        }
    }

    pub fn write_uniforms(&self, queue: &wgpu::Queue, light_vps: &[Mat4; NUM_CASCADES]) {
        for (i, vp) in light_vps.iter().enumerate() {
            let uniforms = ShadowCascadeUniforms {
                light_vp: vp.to_cols_array_2d(),
            };
            queue.write_buffer(
                &self.uniform_buffer,
                self.dyn_alignment as u64 * i as u64,
                bytemuck::bytes_of(&uniforms),
            );
        }
    }

    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        meshes: &[GpuMesh],
        opaque_list: &[usize],
    ) {
        for cascade in 0..NUM_CASCADES {
            let dyn_offset = self.dyn_alignment * cascade as u32;

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(&format!("prism-shadow-pass-cascade-{cascade}")),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.cascade_views[cascade],
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });

            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[dyn_offset]);

            for &idx in opaque_list {
                let mesh = &meshes[idx];
                pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..mesh.index_count, 0, 0..1);
            }
        }
    }
}
