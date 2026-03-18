use glam::Mat4;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GtaoUniforms {
    projection: [[f32; 4]; 4],
    view: [[f32; 4]; 4],
    inv_projection: [[f32; 4]; 4],
    screen_size: [f32; 4],
    params: [f32; 4],
}

pub struct GtaoPass {
    sample_pipeline: wgpu::ComputePipeline,
    blur_pipeline: wgpu::ComputePipeline,
    sample_bind_group_layout: wgpu::BindGroupLayout,
    blur_bind_group_layout: wgpu::BindGroupLayout,
    raw_ao_texture: wgpu::Texture,
    raw_ao_view: wgpu::TextureView,
    blurred_ao_texture: wgpu::Texture,
    blurred_ao_view: wgpu::TextureView,
    raw_bent_normal_texture: wgpu::Texture,
    raw_bent_normal_view: wgpu::TextureView,
    blurred_bent_normal_texture: wgpu::Texture,
    blurred_bent_normal_view: wgpu::TextureView,
    uniform_buffer: wgpu::Buffer,
    width: u32,
    height: u32,
}

impl GtaoPass {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("prism-gtao-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/gtao.wgsl").into()),
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("prism-gtao-uniforms"),
            size: std::mem::size_of::<GtaoUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let raw_ao_texture = create_ao_texture(device, width, height, "prism-gtao-raw");
        let raw_ao_view = raw_ao_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let blurred_ao_texture = create_ao_texture(device, width, height, "prism-gtao-blurred");
        let blurred_ao_view =
            blurred_ao_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let raw_bent_normal_texture =
            create_bent_normal_texture(device, width, height, "prism-gtao-bent-raw");
        let raw_bent_normal_view =
            raw_bent_normal_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let blurred_bent_normal_texture =
            create_bent_normal_texture(device, width, height, "prism-gtao-bent-blurred");
        let blurred_bent_normal_view =
            blurred_bent_normal_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Sample pass bind group layout
        let sample_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("prism-gtao-sample-bgl"),
                entries: &[
                    // 0: uniforms
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: Some(
                                std::num::NonZeroU64::new(
                                    std::mem::size_of::<GtaoUniforms>() as u64
                                )
                                .unwrap(),
                            ),
                        },
                        count: None,
                    },
                    // 1: normals texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // 2: depth
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // 3: ao output (storage)
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::R32Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    // 4: bent normal output (storage)
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba16Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            });

        // Blur pass bind group layout
        let blur_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("prism-gtao-blur-bgl"),
                entries: &[
                    // 0: uniforms
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: Some(
                                std::num::NonZeroU64::new(
                                    std::mem::size_of::<GtaoUniforms>() as u64
                                )
                                .unwrap(),
                            ),
                        },
                        count: None,
                    },
                    // 1: normals texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // 2: depth
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // 3: raw ao input
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // 4: raw bent normal input
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // 5: blurred ao output (storage)
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::R32Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    // 6: blurred bent normal output (storage)
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba16Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            });

        let sample_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("prism-gtao-sample-pl"),
            bind_group_layouts: &[&sample_bind_group_layout],
            immediate_size: 0,
        });
        let sample_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("prism-gtao-sample-pipeline"),
            layout: Some(&sample_pl),
            module: &shader,
            entry_point: Some("gtao_sample"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let blur_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("prism-gtao-blur-pl"),
            bind_group_layouts: &[&blur_bind_group_layout],
            immediate_size: 0,
        });
        let blur_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("prism-gtao-blur-pipeline"),
            layout: Some(&blur_pl),
            module: &shader,
            entry_point: Some("gtao_blur"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            sample_pipeline,
            blur_pipeline,
            sample_bind_group_layout,
            blur_bind_group_layout,
            raw_ao_texture,
            raw_ao_view,
            blurred_ao_texture,
            blurred_ao_view,
            raw_bent_normal_texture,
            raw_bent_normal_view,
            blurred_bent_normal_texture,
            blurred_bent_normal_view,
            uniform_buffer,
            width,
            height,
        }
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.raw_ao_texture = create_ao_texture(device, width, height, "prism-gtao-raw");
        self.raw_ao_view = self
            .raw_ao_texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.blurred_ao_texture = create_ao_texture(device, width, height, "prism-gtao-blurred");
        self.blurred_ao_view = self
            .blurred_ao_texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        self.raw_bent_normal_texture =
            create_bent_normal_texture(device, width, height, "prism-gtao-bent-raw");
        self.raw_bent_normal_view = self
            .raw_bent_normal_texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.blurred_bent_normal_texture =
            create_bent_normal_texture(device, width, height, "prism-gtao-bent-blurred");
        self.blurred_bent_normal_view = self
            .blurred_bent_normal_texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        self.width = width;
        self.height = height;
    }

    pub fn write_uniforms(
        &self,
        queue: &wgpu::Queue,
        projection: Mat4,
        view: Mat4,
        width: u32,
        height: u32,
        frame_index: u32,
    ) {
        let inv_proj = projection.inverse();
        let uniforms = GtaoUniforms {
            projection: projection.to_cols_array_2d(),
            view: view.to_cols_array_2d(),
            inv_projection: inv_proj.to_cols_array_2d(),
            screen_size: super::pack_screen_size(width, height),
            params: [0.5, 0.025, 1.5, frame_index as f32], // radius, bias, intensity, frame_index
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
    }

    /// Execute the GTAO sample + blur passes.
    ///
    /// `normal_view` is the texture view containing world-space normals.
    /// `depth_view` is the texture view containing the depth buffer.
    pub fn execute(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        normal_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
    ) {
        // Sample pass
        let sample_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("prism-gtao-sample-bg"),
            layout: &self.sample_bind_group_layout,
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
                    resource: wgpu::BindingResource::TextureView(&self.raw_ao_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&self.raw_bent_normal_view),
                },
            ],
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("prism-gtao-sample-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.sample_pipeline);
            pass.set_bind_group(0, &sample_bg, &[]);
            pass.dispatch_workgroups(self.width.div_ceil(8), self.height.div_ceil(8), 1);
        }

        // Blur pass
        let blur_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("prism-gtao-blur-bg"),
            layout: &self.blur_bind_group_layout,
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
                    resource: wgpu::BindingResource::TextureView(&self.raw_ao_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&self.raw_bent_normal_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&self.blurred_ao_view),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(&self.blurred_bent_normal_view),
                },
            ],
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("prism-gtao-blur-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.blur_pipeline);
            pass.set_bind_group(0, &blur_bg, &[]);
            pass.dispatch_workgroups(self.width.div_ceil(8), self.height.div_ceil(8), 1);
        }
    }

    pub fn ao_view(&self) -> &wgpu::TextureView {
        &self.blurred_ao_view
    }

    pub fn bent_normal_view(&self) -> &wgpu::TextureView {
        &self.blurred_bent_normal_view
    }
}

fn create_ao_texture(device: &wgpu::Device, width: u32, height: u32, label: &str) -> wgpu::Texture {
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
        format: wgpu::TextureFormat::R32Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    })
}

fn create_bent_normal_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    label: &str,
) -> wgpu::Texture {
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
