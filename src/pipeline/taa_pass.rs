/// Temporal Anti-Aliasing pass with motion vectors and neighborhood clamping.
///
/// Two compute sub-passes:
/// 1. Motion vector generation — reconstructs world position from depth,
///    reprojects with prev_view_proj, outputs UV-space velocity.
/// 2. Temporal resolve — 3×3 YCoCg AABB clamping, 90/10 blend with history.
///
/// Ping-pong history textures avoid copy passes.
use bytemuck::{Pod, Zeroable};
use glam::Mat4;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct TaaUniforms {
    inv_view_proj: [[f32; 4]; 4],
    prev_view_proj: [[f32; 4]; 4],
    screen_size: [f32; 4],
    jitter: [f32; 4],
    params: [f32; 4],
}

pub struct TaaPass {
    motion_pipeline: wgpu::ComputePipeline,
    motion_bgl: wgpu::BindGroupLayout,
    resolve_pipeline: wgpu::ComputePipeline,
    resolve_bgl: wgpu::BindGroupLayout,

    motion_texture: wgpu::Texture,
    motion_view: wgpu::TextureView,

    history_a: wgpu::Texture,
    history_a_view: wgpu::TextureView,
    history_b: wgpu::Texture,
    history_b_view: wgpu::TextureView,

    history_sampler: wgpu::Sampler,
    uniform_buffer: wgpu::Buffer,

    prev_view_proj: Mat4,
    frame_count: u32,
    write_to_a: bool,
    prev_jitter: [f32; 2],
    width: u32,
    height: u32,
}

impl TaaPass {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        // -- Motion vector pipeline --
        let motion_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("taa-motion-bgl"),
            entries: &[
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
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
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

        let motion_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("taa-motion-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/taa_motion.wgsl").into()),
        });

        let motion_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("taa-motion-layout"),
            bind_group_layouts: &[&motion_bgl],
            immediate_size: 0,
        });

        let motion_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("taa-motion-pipeline"),
            layout: Some(&motion_pl),
            module: &motion_shader,
            entry_point: Some("taa_motion"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // -- Resolve pipeline --
        let resolve_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("taa-resolve-bgl"),
            entries: &[
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
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
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

        let resolve_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("taa-resolve-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/taa_resolve.wgsl").into()),
        });

        let resolve_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("taa-resolve-layout"),
            bind_group_layouts: &[&resolve_bgl],
            immediate_size: 0,
        });

        let resolve_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("taa-resolve-pipeline"),
            layout: Some(&resolve_pl),
            module: &resolve_shader,
            entry_point: Some("taa_resolve"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // -- Textures --
        let (motion_texture, motion_view) = create_taa_texture(device, width, height, "taa-motion");
        let (history_a, history_a_view) =
            create_taa_texture(device, width, height, "taa-history-a");
        let (history_b, history_b_view) =
            create_taa_texture(device, width, height, "taa-history-b");

        let history_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("taa-history-sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("taa-uniforms"),
            size: std::mem::size_of::<TaaUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            motion_pipeline,
            motion_bgl,
            resolve_pipeline,
            resolve_bgl,
            motion_texture,
            motion_view,
            history_a,
            history_a_view,
            history_b,
            history_b_view,
            history_sampler,
            uniform_buffer,
            prev_view_proj: Mat4::IDENTITY,
            frame_count: 0,
            write_to_a: true,
            prev_jitter: [0.0; 2],
            width,
            height,
        }
    }

    /// Current frame's sub-pixel jitter offset in pixels (Halton 2,3 sequence, ±0.5px).
    pub fn jitter(&self) -> (f32, f32) {
        let idx = (self.frame_count % 16) + 1;
        (halton(idx, 2) - 0.5, halton(idx, 3) - 0.5)
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        let (mt, mv) = create_taa_texture(device, width, height, "taa-motion");
        self.motion_texture = mt;
        self.motion_view = mv;
        let (ha, hav) = create_taa_texture(device, width, height, "taa-history-a");
        self.history_a = ha;
        self.history_a_view = hav;
        let (hb, hbv) = create_taa_texture(device, width, height, "taa-history-b");
        self.history_b = hb;
        self.history_b_view = hbv;
        self.frame_count = 0;
    }

    /// Encode motion vector generation + temporal resolve.
    ///
    /// After this call, `output_view()` returns the resolved HDR texture for tonemap.
    pub fn encode(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        depth_view: &wgpu::TextureView,
        scene_color_view: &wgpu::TextureView,
        inv_view_proj: Mat4,
        view_proj: Mat4,
        jitter: [f32; 2],
    ) {
        // Write uniforms
        let first_frame = if self.frame_count < 2 { 1.0 } else { 0.0 };
        let uniforms = TaaUniforms {
            inv_view_proj: inv_view_proj.to_cols_array_2d(),
            prev_view_proj: self.prev_view_proj.to_cols_array_2d(),
            screen_size: [
                self.width as f32,
                self.height as f32,
                1.0 / self.width as f32,
                1.0 / self.height as f32,
            ],
            jitter: [
                jitter[0],
                jitter[1],
                self.prev_jitter[0],
                self.prev_jitter[1],
            ],
            params: [0.1, first_frame, 0.0, 0.0],
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        // Ping-pong: read from one history, write to the other
        let (read_view, write_view) = if self.write_to_a {
            (&self.history_b_view, &self.history_a_view)
        } else {
            (&self.history_a_view, &self.history_b_view)
        };

        // -- Motion vector pass --
        let motion_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("taa-motion-bg"),
            layout: &self.motion_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&self.motion_view),
                },
            ],
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("taa-motion-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.motion_pipeline);
            pass.set_bind_group(0, &motion_bg, &[]);
            pass.dispatch_workgroups(self.width.div_ceil(8), self.height.div_ceil(8), 1);
        }

        // -- Resolve pass --
        let resolve_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("taa-resolve-bg"),
            layout: &self.resolve_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(scene_color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&self.motion_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(read_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&self.history_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(write_view),
                },
            ],
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("taa-resolve-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.resolve_pipeline);
            pass.set_bind_group(0, &resolve_bg, &[]);
            pass.dispatch_workgroups(self.width.div_ceil(8), self.height.div_ceil(8), 1);
        }

        // Update state for next frame
        self.prev_view_proj = view_proj;
        self.prev_jitter = jitter;
        self.frame_count = self.frame_count.saturating_add(1);
        self.write_to_a = !self.write_to_a;
    }

    /// Returns the most recently written history texture (TAA output for tonemap).
    pub fn output_view(&self) -> &wgpu::TextureView {
        // After encode(), write_to_a has been flipped, so the just-written
        // texture is the opposite of the current write_to_a flag.
        if self.write_to_a {
            &self.history_b_view
        } else {
            &self.history_a_view
        }
    }
}

fn halton(index: u32, base: u32) -> f32 {
    let mut result = 0.0_f32;
    let mut f = 1.0 / base as f32;
    let mut i = index;
    while i > 0 {
        result += f * (i % base) as f32;
        i /= base;
        f /= base as f32;
    }
    result
}

fn create_taa_texture(
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
