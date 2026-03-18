//! Motion Blur pass — tile-max velocity + per-pixel directional blur.
//!
//! Two compute sub-passes:
//! 1. Tile-max: downsample motion vectors to tiles, take max velocity.
//! 2. Directional blur: blur along tile velocity direction.
//!
//! Operates on HDR data (before tonemap). Requires TAA motion vectors.

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MotionBlurUniforms {
    screen_size: [f32; 4],
    params: [f32; 4],
}

const TILE_SIZE: u32 = 20;

pub struct MotionBlurPass {
    tile_max_pipeline: wgpu::ComputePipeline,
    tile_max_bgl: wgpu::BindGroupLayout,
    blur_pipeline: wgpu::ComputePipeline,
    blur_bgl: wgpu::BindGroupLayout,

    tile_max_texture: wgpu::Texture,
    tile_max_view: wgpu::TextureView,
    blur_texture: wgpu::Texture,
    blur_view: wgpu::TextureView,

    uniform_buffer: wgpu::Buffer,
    width: u32,
    height: u32,
}

impl MotionBlurPass {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("motion-blur-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/motion_blur.wgsl").into()),
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("motion-blur-uniforms"),
            size: std::mem::size_of::<MotionBlurUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Tile-max bind group layout
        let tile_max_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("motion-blur-tile-max-bgl"),
            entries: &[
                // 0: uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 1: motion vectors
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
                // 2: tile-max output
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rg16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        // Blur bind group layout
        let blur_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("motion-blur-bgl"),
            entries: &[
                // 0: uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 1: tile-max texture
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
                // 2: HDR input
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
                // 3: blurred HDR output
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
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

        let tile_max_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("motion-blur-tile-max-pl"),
            bind_group_layouts: &[&tile_max_bgl],
            immediate_size: 0,
        });
        let tile_max_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("motion-blur-tile-max-pipeline"),
            layout: Some(&tile_max_pl),
            module: &shader,
            entry_point: Some("tile_max"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let blur_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("motion-blur-pl"),
            bind_group_layouts: &[&blur_bgl],
            immediate_size: 0,
        });
        let blur_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("motion-blur-pipeline"),
            layout: Some(&blur_pl),
            module: &shader,
            entry_point: Some("motion_blur"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let tiles_x = width.div_ceil(TILE_SIZE);
        let tiles_y = height.div_ceil(TILE_SIZE);
        let (tile_max_texture, tile_max_view) =
            create_tile_texture(device, tiles_x, tiles_y, "motion-blur-tile-max");
        let (blur_texture, blur_view) = create_hdr_texture(device, width, height, "motion-blur");

        Self {
            tile_max_pipeline,
            tile_max_bgl,
            blur_pipeline,
            blur_bgl,
            tile_max_texture,
            tile_max_view,
            blur_texture,
            blur_view,
            uniform_buffer,
            width,
            height,
        }
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        let tiles_x = width.div_ceil(TILE_SIZE);
        let tiles_y = height.div_ceil(TILE_SIZE);
        let (tt, tv) = create_tile_texture(device, tiles_x, tiles_y, "motion-blur-tile-max");
        self.tile_max_texture = tt;
        self.tile_max_view = tv;
        let (bt, bv) = create_hdr_texture(device, width, height, "motion-blur");
        self.blur_texture = bt;
        self.blur_view = bv;
    }

    /// Encode tile-max + directional blur passes.
    ///
    /// `motion_view` is the TAA motion vector texture (Rgba16Float, UV-space velocity in RG).
    /// `hdr_view` is the current HDR scene color.
    pub fn encode(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        motion_view: &wgpu::TextureView,
        hdr_view: &wgpu::TextureView,
    ) {
        let max_blur_px = 20.0_f32;
        let uniforms = MotionBlurUniforms {
            screen_size: super::pack_screen_size(self.width, self.height),
            params: [max_blur_px, TILE_SIZE as f32, 0.0, 0.0],
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        // Tile-max pass
        let tile_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("motion-blur-tile-max-bg"),
            layout: &self.tile_max_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(motion_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&self.tile_max_view),
                },
            ],
        });

        let tiles_x = self.width.div_ceil(TILE_SIZE);
        let tiles_y = self.height.div_ceil(TILE_SIZE);

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("motion-blur-tile-max-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.tile_max_pipeline);
            pass.set_bind_group(0, &tile_bg, &[]);
            pass.dispatch_workgroups(tiles_x.div_ceil(8), tiles_y.div_ceil(8), 1);
        }

        // Directional blur pass
        let blur_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("motion-blur-bg"),
            layout: &self.blur_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.tile_max_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(hdr_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&self.blur_view),
                },
            ],
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("motion-blur-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.blur_pipeline);
            pass.set_bind_group(0, &blur_bg, &[]);
            pass.dispatch_workgroups(self.width.div_ceil(8), self.height.div_ceil(8), 1);
        }
    }

    /// Returns the blurred HDR output view (for tonemap input).
    pub fn output_view(&self) -> &wgpu::TextureView {
        &self.blur_view
    }
}

fn create_tile_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    label: &str,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rg16Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

fn create_hdr_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    label: &str,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
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
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}
