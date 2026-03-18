use glam::Mat4;

use super::HDR_FORMAT;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SsgiUniforms {
    view: [[f32; 4]; 4],
    inv_view: [[f32; 4]; 4],
    projection: [[f32; 4]; 4],
    inv_projection: [[f32; 4]; 4],
    screen_size: [f32; 4],
    params: [f32; 4],
}

pub struct SsgiPass {
    sample_pipeline: wgpu::ComputePipeline,
    blur_pipeline: wgpu::ComputePipeline,
    composite_pipeline: wgpu::RenderPipeline,
    sample_bgl: wgpu::BindGroupLayout,
    blur_bgl: wgpu::BindGroupLayout,
    composite_bgl: wgpu::BindGroupLayout,
    pub raw_gi_texture: wgpu::Texture,
    pub raw_gi_view: wgpu::TextureView,
    pub blurred_gi_texture: wgpu::Texture,
    pub blurred_gi_view: wgpu::TextureView,
    uniform_buffer: wgpu::Buffer,
    width: u32,
    height: u32,
}

impl SsgiPass {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let ssgi_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("prism-ssgi-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/ssgi.wgsl").into()),
        });
        let composite_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("prism-ssgi-composite-shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/ssgi_composite.wgsl").into(),
            ),
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("prism-ssgi-uniforms"),
            size: std::mem::size_of::<SsgiUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let raw_gi_texture = create_gi_texture(device, width, height, "prism-ssgi-raw");
        let raw_gi_view = raw_gi_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let blurred_gi_texture = create_gi_texture(device, width, height, "prism-ssgi-blurred");
        let blurred_gi_view =
            blurred_gi_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Sample pass bind group layout:
        // 0: uniforms, 1: normals, 2: depth, 3: scene color, 4: output (storage)
        let sample_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("prism-ssgi-sample-bgl"),
            entries: &[
                uniform_entry(0),
                texture_entry(1), // normals
                depth_texture_entry(2), // depth
                texture_entry(3), // scene color
                storage_texture_entry(4, wgpu::TextureFormat::Rgba16Float),
            ],
        });

        // Blur pass: same layout but input 3 is raw GI instead of scene color
        let blur_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("prism-ssgi-blur-bgl"),
            entries: &[
                uniform_entry(0),
                texture_entry(1), // normals
                depth_texture_entry(2), // depth
                texture_entry(3), // raw GI input
                storage_texture_entry(4, wgpu::TextureFormat::Rgba16Float),
            ],
        });

        // Composite bind group layout: just the GI texture
        let composite_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("prism-ssgi-composite-bgl"),
            entries: &[texture_entry(0)],
        });

        // Create compute pipelines
        let sample_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("prism-ssgi-sample-pl"),
            bind_group_layouts: &[&sample_bgl],
            immediate_size: 0,
        });
        let sample_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("prism-ssgi-sample-pipeline"),
            layout: Some(&sample_pl),
            module: &ssgi_shader,
            entry_point: Some("ssgi_sample"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let blur_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("prism-ssgi-blur-pl"),
            bind_group_layouts: &[&blur_bgl],
            immediate_size: 0,
        });
        let blur_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("prism-ssgi-blur-pipeline"),
            layout: Some(&blur_pl),
            module: &ssgi_shader,
            entry_point: Some("ssgi_blur"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // Create composite render pipeline (additive blend)
        let composite_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("prism-ssgi-composite-pl"),
            bind_group_layouts: &[&composite_bgl],
            immediate_size: 0,
        });
        let composite_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("prism-ssgi-composite-pipeline"),
            layout: Some(&composite_pl),
            vertex: wgpu::VertexState {
                module: &composite_shader,
                entry_point: Some("vs_composite"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &composite_shader,
                entry_point: Some("fs_composite"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: HDR_FORMAT,
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
            sample_pipeline,
            blur_pipeline,
            composite_pipeline,
            sample_bgl,
            blur_bgl,
            composite_bgl,
            raw_gi_texture,
            raw_gi_view,
            blurred_gi_texture,
            blurred_gi_view,
            uniform_buffer,
            width,
            height,
        }
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.raw_gi_texture = create_gi_texture(device, width, height, "prism-ssgi-raw");
        self.raw_gi_view = self
            .raw_gi_texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.blurred_gi_texture = create_gi_texture(device, width, height, "prism-ssgi-blurred");
        self.blurred_gi_view = self
            .blurred_gi_texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.width = width;
        self.height = height;
    }

    pub fn write_uniforms(
        &self,
        queue: &wgpu::Queue,
        view: Mat4,
        projection: Mat4,
        width: u32,
        height: u32,
        gi_intensity: f32,
        frame_index: u32,
    ) {
        let uniforms = SsgiUniforms {
            view: view.to_cols_array_2d(),
            inv_view: view.inverse().to_cols_array_2d(),
            projection: projection.to_cols_array_2d(),
            inv_projection: projection.inverse().to_cols_array_2d(),
            screen_size: [
                width as f32,
                height as f32,
                1.0 / width as f32,
                1.0 / height as f32,
            ],
            params: [gi_intensity, 2.0, frame_index as f32, 0.0], // gi_intensity, radius, frame_index
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
    }

    /// Execute the SSGI sample → blur → composite passes.
    pub fn execute(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        normal_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        color_view: &wgpu::TextureView,
    ) {
        // -- Sample pass --
        let sample_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("prism-ssgi-sample-bg"),
            layout: &self.sample_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(normal_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&self.raw_gi_view),
                },
            ],
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("prism-ssgi-sample-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.sample_pipeline);
            pass.set_bind_group(0, &sample_bg, &[]);
            pass.dispatch_workgroups(self.width.div_ceil(8), self.height.div_ceil(8), 1);
        }

        // -- Blur pass --
        let blur_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("prism-ssgi-blur-bg"),
            layout: &self.blur_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(normal_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&self.raw_gi_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&self.blurred_gi_view),
                },
            ],
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("prism-ssgi-blur-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.blur_pipeline);
            pass.set_bind_group(0, &blur_bg, &[]);
            pass.dispatch_workgroups(self.width.div_ceil(8), self.height.div_ceil(8), 1);
        }

        // -- Composite pass (additive blend onto scene color) --
        let composite_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("prism-ssgi-composite-bg"),
            layout: &self.composite_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&self.blurred_gi_view),
            }],
        });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("prism-ssgi-composite-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_view,
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
            pass.set_bind_group(0, &composite_bg, &[]);
            pass.draw(0..3, 0..1);
        }
    }
}

fn create_gi_texture(device: &wgpu::Device, width: u32, height: u32, label: &str) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    })
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: Some(
                std::num::NonZeroU64::new(std::mem::size_of::<SsgiUniforms>() as u64).unwrap(),
            ),
        },
        count: None,
    }
}

fn texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn depth_texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Depth,
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn storage_texture_entry(binding: u32, format: wgpu::TextureFormat) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    }
}
