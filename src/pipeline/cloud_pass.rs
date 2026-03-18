/// Quarter-resolution volumetric cloud pass with temporal reprojection.
///
/// Two sub-passes:
/// 1. Compute: Raymarch clouds at 1/4 resolution using precomputed 3D noise
/// 2. Render: Composite clouds onto scene at full resolution
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};

use crate::gpu::GpuTimingContext;

use super::noise_textures::NoiseTextures;
use super::HDR_FORMAT;

/// Optional timing indices for the three cloud sub-passes.
pub struct CloudTimingSlots {
    pub march: (u32, u32),
    pub temporal: (u32, u32),
    pub composite: (u32, u32),
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct CloudUniforms {
    inv_view_proj: [[f32; 4]; 4],
    prev_view_proj: [[f32; 4]; 4],
    camera_position: [f32; 4],
    sun_direction: [f32; 4],
    sun_color: [f32; 4],
    sky_ambient: [f32; 4],
    cloud_params: [f32; 4],   // coverage, first_frame_flag, time, frame_index
    screen_params: [f32; 4],  // quarter_w, quarter_h, 1/qw, 1/qh
    cloud_profile: [f32; 4],  // density_scale, cloud_base, cloud_top, detail_erosion
    cloud_profile2: [f32; 4], // wind_speed, march_steps, light_steps, temporal_blend
    prev_time: [f32; 4],      // prev_elapsed, 0, 0, 0
}

pub struct CloudPass {
    march_pipeline: wgpu::ComputePipeline,
    march_bgl: wgpu::BindGroupLayout,
    temporal_pipeline: wgpu::ComputePipeline,
    temporal_bgl: wgpu::BindGroupLayout,
    composite_pipeline: wgpu::RenderPipeline,
    composite_bgl: wgpu::BindGroupLayout,

    cloud_raw_texture: wgpu::Texture,
    cloud_raw_view: wgpu::TextureView,
    cloud_texture: wgpu::Texture,
    cloud_view: wgpu::TextureView,
    cloud_depth_texture: wgpu::Texture,
    cloud_depth_view: wgpu::TextureView,
    history_texture: wgpu::Texture,
    history_view: wgpu::TextureView,
    composite_sampler: wgpu::Sampler,
    temporal_sampler: wgpu::Sampler,

    uniform_buffer: wgpu::Buffer,
    quarter_w: u32,
    quarter_h: u32,
    pub resolution_divisor: u32,
    frame_index: u32,
    frame_count: u32,
    prev_view_proj: Mat4,
    prev_elapsed: f32,
}

impl CloudPass {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let resolution_divisor = 2u32; // default half-res; updated via set_resolution
        let quarter_w = (width / resolution_divisor).max(1);
        let quarter_h = (height / resolution_divisor).max(1);

        let (cloud_raw_texture, cloud_raw_view) =
            create_cloud_texture(device, quarter_w, quarter_h, "cloud-raw");
        let (cloud_texture, cloud_view) =
            create_cloud_texture(device, quarter_w, quarter_h, "cloud-current");
        let (cloud_depth_texture, cloud_depth_view) =
            create_cloud_depth_texture(device, quarter_w, quarter_h);
        let (history_texture, history_view) =
            create_cloud_texture(device, quarter_w, quarter_h, "cloud-history");

        let composite_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("cloud-composite-sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let temporal_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("cloud-temporal-sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cloud-uniforms"),
            size: std::mem::size_of::<CloudUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // March compute bind group layout
        let march_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cloud-march-bgl"),
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
                // 1: shape noise 3D
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                // 2: detail noise 3D
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                // 3: weather map 2D
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // 4: noise sampler (repeat)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // 5: history texture (previous frame)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // 6: output texture (storage, write)
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
                // 7: scene depth texture
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // 8: cloud depth output (storage, write)
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // 9: sky-view LUT (read) — for unified cloud/sky ambient
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // 10: LUT sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let march_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cloud-march-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/cloud_march.wgsl").into()),
        });

        let march_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cloud-march-layout"),
            bind_group_layouts: &[&march_bgl],
            immediate_size: 0,
        });

        let march_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cloud-march-pipeline"),
            layout: Some(&march_pl),
            module: &march_shader,
            entry_point: Some("cloud_march"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // Temporal reprojection compute pipeline (3×3 neighborhood clamping)
        let temporal_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cloud-temporal-bgl"),
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
                // 1: raw current frame (read)
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
                // 2: history (read, filterable for bilinear sampling)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // 3: output (storage, write)
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
                // 4: cloud depth (read)
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
                // 5: history sampler (bilinear filtering)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let temporal_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cloud-temporal-shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/cloud_temporal.wgsl").into(),
            ),
        });

        let temporal_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cloud-temporal-layout"),
            bind_group_layouts: &[&temporal_bgl],
            immediate_size: 0,
        });

        let temporal_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cloud-temporal-pipeline"),
            layout: Some(&temporal_pl),
            module: &temporal_shader,
            entry_point: Some("cloud_temporal"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // Composite render pipeline
        let composite_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cloud-composite-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // Depth texture for bilateral upsampling
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Composite uniforms (screen_params)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Cloud depth texture for bilateral weighting
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        let composite_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cloud-composite-shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/cloud_composite.wgsl").into(),
            ),
        });

        let composite_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cloud-composite-layout"),
            bind_group_layouts: &[&composite_bgl],
            immediate_size: 0,
        });

        // Blend state: scene_color = scene_color * cloud.a + cloud.rgb
        // src = cloud output, dst = scene
        // color: src_factor = One, dst_factor = SrcAlpha (cloud.a = transmittance)
        let composite_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("cloud-composite-pipeline"),
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
                            dst_factor: wgpu::BlendFactor::SrcAlpha,
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
            march_pipeline,
            march_bgl,
            temporal_pipeline,
            temporal_bgl,
            composite_pipeline,
            composite_bgl,
            cloud_raw_texture,
            cloud_raw_view,
            cloud_texture,
            cloud_view,
            cloud_depth_texture,
            cloud_depth_view,
            history_texture,
            history_view,
            composite_sampler,
            temporal_sampler,
            uniform_buffer,
            quarter_w,
            quarter_h,
            resolution_divisor,
            frame_index: 0,
            frame_count: 0,
            prev_view_proj: Mat4::IDENTITY,
            prev_elapsed: 0.0,
        }
    }

    /// Update cloud resolution divisor from profile.
    pub fn set_resolution(&mut self, resolution: crate::scene::CloudResolution) {
        use crate::scene::CloudResolution;
        self.resolution_divisor = match resolution {
            CloudResolution::Quarter => 4,
            CloudResolution::Half => 2,
        };
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.quarter_w = (width / self.resolution_divisor).max(1);
        self.quarter_h = (height / self.resolution_divisor).max(1);
        let (rt, rv) = create_cloud_texture(device, self.quarter_w, self.quarter_h, "cloud-raw");
        let (ct, cv) =
            create_cloud_texture(device, self.quarter_w, self.quarter_h, "cloud-current");
        let (dt, dv) = create_cloud_depth_texture(device, self.quarter_w, self.quarter_h);
        let (ht, hv) =
            create_cloud_texture(device, self.quarter_w, self.quarter_h, "cloud-history");
        self.cloud_raw_texture = rt;
        self.cloud_raw_view = rv;
        self.cloud_texture = ct;
        self.cloud_view = cv;
        self.cloud_depth_texture = dt;
        self.cloud_depth_view = dv;
        self.history_texture = ht;
        self.history_view = hv;
        // Reset temporal state so first frame after resize skips history blend
        self.frame_count = 0;
    }

    pub fn write_uniforms(
        &mut self,
        queue: &wgpu::Queue,
        inv_view_proj: Mat4,
        camera_position: Vec3,
        sun_direction: Vec3,
        sun_color: Vec3,
        sun_strength: f32,
        sky_ambient: Vec3,
        cloud_coverage: f32,
        elapsed_secs: f32,
        profile: &crate::scene::CloudProfile,
    ) {
        let first_frame_flag = if self.frame_count < 2 { 1.0 } else { 0.0 };
        let uniforms = CloudUniforms {
            inv_view_proj: inv_view_proj.to_cols_array_2d(),
            prev_view_proj: self.prev_view_proj.to_cols_array_2d(),
            camera_position: [camera_position.x, camera_position.y, camera_position.z, 1.0],
            sun_direction: [sun_direction.x, sun_direction.y, sun_direction.z, 0.0],
            sun_color: [sun_color.x, sun_color.y, sun_color.z, sun_strength],
            sky_ambient: [
                sky_ambient.x,
                sky_ambient.y,
                sky_ambient.z,
                self.resolution_divisor as f32,
            ],
            cloud_params: [
                cloud_coverage,
                first_frame_flag,
                elapsed_secs,
                self.frame_index as f32,
            ],
            screen_params: [
                self.quarter_w as f32,
                self.quarter_h as f32,
                1.0 / self.quarter_w as f32,
                1.0 / self.quarter_h as f32,
            ],
            cloud_profile: [
                profile.density_scale,
                profile.cloud_base_km,
                profile.cloud_top_km,
                profile.detail_erosion,
            ],
            cloud_profile2: [
                profile.wind_speed,
                profile.march_steps as f32,
                profile.light_steps as f32,
                profile.temporal_blend,
            ],
            prev_time: [self.prev_elapsed, 0.0, 0.0, 0.0],
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
        self.prev_elapsed = elapsed_secs;
    }

    /// Encode the cloud march compute + temporal + composite render passes.
    pub fn encode(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        noise: &NoiseTextures,
        scene_color_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        view_proj: Mat4,
        timing: Option<(&GpuTimingContext, &CloudTimingSlots)>,
        sky_lut_view: &wgpu::TextureView,
        sky_lut_sampler: &wgpu::Sampler,
    ) {
        // Swap cloud ↔ history (ping-pong)
        std::mem::swap(&mut self.cloud_texture, &mut self.history_texture);
        std::mem::swap(&mut self.cloud_view, &mut self.history_view);

        // -- Pass 1: Cloud march at quarter res → raw texture --
        let march_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cloud-march-bg"),
            layout: &self.march_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&noise.shape_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&noise.detail_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&noise.weather_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&noise.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&self.history_view),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(&self.cloud_raw_view),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::TextureView(&self.cloud_depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: wgpu::BindingResource::TextureView(sky_lut_view),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: wgpu::BindingResource::Sampler(sky_lut_sampler),
                },
            ],
        });

        {
            let tw = timing.map(|(t, s)| t.compute_timestamp_writes(s.march.0, s.march.1));
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cloud-march-pass"),
                timestamp_writes: tw,
            });
            pass.set_pipeline(&self.march_pipeline);
            pass.set_bind_group(0, &march_bg, &[]);
            pass.dispatch_workgroups(self.quarter_w.div_ceil(8), self.quarter_h.div_ceil(8), 1);
        }

        // -- Pass 2: Temporal reprojection (raw + history → cloud) --
        let temporal_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cloud-temporal-bg"),
            layout: &self.temporal_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.cloud_raw_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&self.history_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&self.cloud_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&self.cloud_depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&self.temporal_sampler),
                },
            ],
        });

        {
            let tw = timing.map(|(t, s)| t.compute_timestamp_writes(s.temporal.0, s.temporal.1));
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cloud-temporal-pass"),
                timestamp_writes: tw,
            });
            pass.set_pipeline(&self.temporal_pipeline);
            pass.set_bind_group(0, &temporal_bg, &[]);
            pass.dispatch_workgroups(self.quarter_w.div_ceil(8), self.quarter_h.div_ceil(8), 1);
        }

        // -- Render: Composite clouds onto scene --
        let composite_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cloud-composite-bg"),
            layout: &self.composite_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.cloud_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.composite_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&self.cloud_depth_view),
                },
            ],
        });

        {
            let tw = timing.map(|(t, s)| t.render_timestamp_writes(s.composite.0, s.composite.1));
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("cloud-composite-pass"),
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
                timestamp_writes: tw,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.composite_pipeline);
            pass.set_bind_group(0, &composite_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // Update state for next frame
        self.prev_view_proj = view_proj;
        self.frame_index = self.frame_index.wrapping_add(1);
        self.frame_count = self.frame_count.saturating_add(1);
    }
}

fn create_cloud_texture(
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
        usage: wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

fn create_cloud_depth_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("cloud-depth"),
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
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}
