/// Hierarchical Z-buffer generation (compute shader).
///
/// Builds a depth mip-chain from the visibility buffer's depth.
/// Used for occlusion culling in subsequent frames.
pub struct HzbPass {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    /// HZB mip chain texture.
    pub hzb_texture: Option<wgpu::Texture>,
    pub hzb_views: Vec<wgpu::TextureView>,
    pub mip_count: u32,
}

impl HzbPass {
    pub fn new(device: &wgpu::Device) -> Self {
        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("hzb-bgl"),
                entries: &[
                    // Source depth (read)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // Destination depth (write)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
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

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hzb-build-shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/hzb_build.wgsl").into(),
            ),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("hzb-layout"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("hzb-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("hzb_build"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            pipeline,
            bind_group_layout,
            hzb_texture: None,
            hzb_views: Vec::new(),
            mip_count: 0,
        }
    }

    /// Create the HZB mip chain for the given resolution.
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let mip_count = ((width.max(height) as f32).log2().floor() as u32).max(1);
        self.mip_count = mip_count;

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("hzb-texture"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: mip_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        self.hzb_views = (0..mip_count)
            .map(|mip| {
                texture.create_view(&wgpu::TextureViewDescriptor {
                    base_mip_level: mip,
                    mip_level_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();

        self.hzb_texture = Some(texture);
    }

    /// Build the HZB mip chain from a depth source.
    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        depth_source_view: &wgpu::TextureView,
        width: u32,
        height: u32,
    ) {
        if self.mip_count < 2 || self.hzb_views.is_empty() {
            return;
        }

        // First mip: source is the depth buffer
        let first_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hzb-mip0-bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(depth_source_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.hzb_views[0]),
                },
            ],
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("hzb-mip0"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &first_bg, &[]);
            let w = (width + 15) / 16;
            let h = (height + 15) / 16;
            pass.dispatch_workgroups(w, h, 1);
        }

        // Subsequent mips: source is previous mip
        for mip in 1..self.mip_count.min(self.hzb_views.len() as u32) {
            let src_view = &self.hzb_views[mip as usize - 1];
            let dst_view = &self.hzb_views[mip as usize];

            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(src_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(dst_view),
                    },
                ],
            });

            let mip_w = (width >> mip).max(1);
            let mip_h = (height >> mip).max(1);

            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups((mip_w + 7) / 8, (mip_h + 7) / 8, 1);
        }
    }
}
