use bytemuck::{Pod, Zeroable};

use crate::camera::CameraState;
use crate::gpu::GpuContext;
use crate::scene::SceneSettings;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct SunShaftUniforms {
    pub sun_screen: [f32; 4],
    pub sun_color: [f32; 4],
    pub shaft_params: [f32; 4],
    pub screen_size: [f32; 4],
}

impl SunShaftUniforms {
    pub fn from_camera(
        camera: &CameraState,
        settings: &SceneSettings,
        width: u32,
        height: u32,
    ) -> Self {
        let sun_point = camera.position() + settings.sun_direction.normalize() * 4096.0;
        let clip = camera.view_projection_matrix() * sun_point.extend(1.0);
        let ndc = if clip.w.abs() > f32::EPSILON {
            clip.truncate() / clip.w
        } else {
            glam::Vec3::ZERO
        };
        let sun_screen = glam::Vec2::new(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
        let facing = camera.forward().dot(settings.sun_direction.normalize()).max(0.0);
        let on_screen = if clip.w > 0.0
            && ndc.x >= -1.3
            && ndc.x <= 1.3
            && ndc.y >= -1.3
            && ndc.y <= 1.3
        {
            1.0
        } else {
            0.0
        };

        Self {
            sun_screen: [sun_screen.x, sun_screen.y, facing * on_screen, settings.horizon_haze],
            sun_color: [
                settings.sun_color.x,
                settings.sun_color.y,
                settings.sun_color.z,
                settings.sun_strength,
            ],
            shaft_params: [
                settings.shaft_intensity,
                settings.shaft_decay,
                0.92,
                settings.horizon_haze,
            ],
            screen_size: [width as f32, height as f32, 0.0, 0.0],
        }
    }
}

pub struct SunShaftPass {
    pub uniform_buffer: wgpu::Buffer,
    shaft_texture: wgpu::Texture,
    shaft_view: wgpu::TextureView,
    shaft_sampler: wgpu::Sampler,
    occlusion_layout: wgpu::BindGroupLayout,
    composite_layout: wgpu::BindGroupLayout,
    occlusion_pipeline: wgpu::RenderPipeline,
    composite_pipeline: wgpu::RenderPipeline,
    shaft_width: u32,
    shaft_height: u32,
}

impl SunShaftPass {
    pub fn new(gpu: &GpuContext) -> Self {
        Self::new_for_size(&gpu.device, gpu.width(), gpu.height())
    }

    pub fn new_for_size(
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> Self {
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("prism-sun-shaft-uniforms"),
            size: std::mem::size_of::<SunShaftUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let shaft_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("prism-sun-shaft-linear-sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let (shaft_texture, shaft_view, shaft_width, shaft_height) =
            create_shaft_target(device, width, height, "prism-sun-shaft-target");
        let occlusion_layout = create_occlusion_layout(device);
        let composite_layout = create_composite_layout(device);

        let occlusion_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("prism-sun-shaft-occlusion-shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/sun_shafts_occlusion.wgsl").into(),
            ),
        });
        let composite_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("prism-sun-shaft-composite-shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/sun_shafts_composite.wgsl").into(),
            ),
        });

        let occlusion_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("prism-sun-shaft-occlusion-layout"),
                bind_group_layouts: &[&occlusion_layout],
                immediate_size: 0,
            });
        let occlusion_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("prism-sun-shaft-occlusion-pipeline"),
            layout: Some(&occlusion_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &occlusion_shader,
                entry_point: Some("vs_fullscreen"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &occlusion_shader,
                entry_point: Some("fs_occlusion"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let composite_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("prism-sun-shaft-composite-layout"),
                bind_group_layouts: &[&composite_layout],
                immediate_size: 0,
            });
        let composite_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("prism-sun-shaft-composite-pipeline"),
            layout: Some(&composite_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &composite_shader,
                entry_point: Some("vs_fullscreen"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &composite_shader,
                entry_point: Some("fs_composite"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: crate::pipeline::HDR_FORMAT,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent::OVER,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            uniform_buffer,
            shaft_texture,
            shaft_view,
            shaft_sampler,
            occlusion_layout,
            composite_layout,
            occlusion_pipeline,
            composite_pipeline,
            shaft_width,
            shaft_height,
        }
    }

    pub fn resize(
        &mut self,
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) {
        let (shaft_texture, shaft_view, shaft_width, shaft_height) =
            create_shaft_target(device, width, height, "prism-sun-shaft-target");
        self.shaft_texture = shaft_texture;
        self.shaft_view = shaft_view;
        self.shaft_width = shaft_width;
        self.shaft_height = shaft_height;
    }

    pub fn write_uniforms(
        &self,
        queue: &wgpu::Queue,
        camera: &CameraState,
        settings: &SceneSettings,
        width: u32,
        height: u32,
    ) {
        let uniforms = SunShaftUniforms::from_camera(camera, settings, width, height);
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
    }

    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        depth_view: &wgpu::TextureView,
        scene_color_view: &wgpu::TextureView,
    ) {
        let occlusion_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("prism-sun-shaft-occlusion-bind-group"),
            layout: &self.occlusion_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
            ],
        });

        let composite_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("prism-sun-shaft-composite-bind-group"),
            layout: &self.composite_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.shaft_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.shaft_sampler),
                },
            ],
        });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("prism-sun-shaft-occlusion-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.shaft_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.occlusion_pipeline);
            pass.set_bind_group(0, &occlusion_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("prism-sun-shaft-composite-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: scene_color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.composite_pipeline);
            pass.set_bind_group(0, &composite_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
    }
}

fn create_shaft_target(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    label: &str,
) -> (wgpu::Texture, wgpu::TextureView, u32, u32) {
    let shaft_width = (width.max(1) / 2).max(1);
    let shaft_height = (height.max(1) / 2).max(1);
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: shaft_width,
            height: shaft_height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view, shaft_width, shaft_height)
}

fn create_occlusion_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("prism-sun-shaft-occlusion-bind-group-layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: Some(
                        std::num::NonZeroU64::new(std::mem::size_of::<SunShaftUniforms>() as u64)
                            .unwrap(),
                    ),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
        ],
    })
}

fn create_composite_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("prism-sun-shaft-composite-bind-group-layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: Some(
                        std::num::NonZeroU64::new(std::mem::size_of::<SunShaftUniforms>() as u64)
                            .unwrap(),
                    ),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    })
}
