use glam::Mat4;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SsaoUniforms {
    projection: [[f32; 4]; 4],
    view: [[f32; 4]; 4],
    inv_projection: [[f32; 4]; 4],
    screen_size: [f32; 4],
    params: [f32; 4],
}

pub struct SsaoPass {
    sample_pipeline: wgpu::ComputePipeline,
    blur_pipeline: wgpu::ComputePipeline,
    sample_bind_group_layout: wgpu::BindGroupLayout,
    blur_bind_group_layout: wgpu::BindGroupLayout,
    pub raw_ao_texture: wgpu::Texture,
    pub raw_ao_view: wgpu::TextureView,
    pub blurred_ao_texture: wgpu::Texture,
    pub blurred_ao_view: wgpu::TextureView,
    uniform_buffer: wgpu::Buffer,
    width: u32,
    height: u32,
}

impl SsaoPass {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("prism-ssao-shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/ssao.wgsl").into(),
            ),
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("prism-ssao-uniforms"),
            size: std::mem::size_of::<SsaoUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let raw_ao_texture = create_ao_texture(device, width, height, "prism-ssao-raw");
        let raw_ao_view = raw_ao_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let blurred_ao_texture = create_ao_texture(device, width, height, "prism-ssao-blurred");
        let blurred_ao_view = blurred_ao_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Sample pass bind group layout
        let sample_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("prism-ssao-sample-bgl"),
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
                                    std::mem::size_of::<SsaoUniforms>() as u64,
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
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
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
                ],
            });

        // Blur pass bind group layout
        let blur_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("prism-ssao-blur-bgl"),
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
                                    std::mem::size_of::<SsaoUniforms>() as u64,
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
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
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
                    // 4: blurred ao output (storage)
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::R32Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            });

        let sample_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("prism-ssao-sample-pl"),
            bind_group_layouts: &[&sample_bind_group_layout],
            immediate_size: 0,
        });
        let sample_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("prism-ssao-sample-pipeline"),
            layout: Some(&sample_pl),
            module: &shader,
            entry_point: Some("ssao_sample"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let blur_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("prism-ssao-blur-pl"),
            bind_group_layouts: &[&blur_bind_group_layout],
            immediate_size: 0,
        });
        let blur_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("prism-ssao-blur-pipeline"),
            layout: Some(&blur_pl),
            module: &shader,
            entry_point: Some("ssao_blur"),
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
            uniform_buffer,
            width,
            height,
        }
    }

    pub fn write_uniforms(
        &self,
        queue: &wgpu::Queue,
        projection: Mat4,
        view: Mat4,
        width: u32,
        height: u32,
    ) {
        let inv_proj = projection.inverse();
        let uniforms = SsaoUniforms {
            projection: projection.to_cols_array_2d(),
            view: view.to_cols_array_2d(),
            inv_projection: inv_proj.to_cols_array_2d(),
            screen_size: [
                width as f32,
                height as f32,
                1.0 / width as f32,
                1.0 / height as f32,
            ],
            params: [0.5, 0.025, 1.5, 0.0], // radius, bias, intensity
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
    }

    /// Execute the SSAO sample + blur passes.
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
            label: Some("prism-ssao-sample-bg"),
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
            ],
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("prism-ssao-sample-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.sample_pipeline);
            pass.set_bind_group(0, &sample_bg, &[]);
            pass.dispatch_workgroups(self.width.div_ceil(8), self.height.div_ceil(8), 1);
        }

        // Blur pass
        let blur_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("prism-ssao-blur-bg"),
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
                    resource: wgpu::BindingResource::TextureView(&self.blurred_ao_view),
                },
            ],
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("prism-ssao-blur-pass"),
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
}

fn create_ao_texture(
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
        format: wgpu::TextureFormat::R32Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    })
}
