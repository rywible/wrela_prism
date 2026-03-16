/// Atmosphere LUT precomputation pass.
///
/// Three LUTs generated via compute shaders:
/// - Transmittance LUT: 256×64 Rgba16Float
/// - Multi-scatter LUT: 32×32 Rgba16Float (depends on transmittance)
/// - Sky-view LUT: 192×108 Rgba16Float (depends on transmittance + multi-scatter)
///
/// Dirty tracking: only regenerates when sun_direction or atmosphere params change.
use bytemuck::{Pod, Zeroable};
use glam::Vec3;

const TRANSMITTANCE_W: u32 = 256;
const TRANSMITTANCE_H: u32 = 64;
const MULTISCATTER_W: u32 = 32;
const MULTISCATTER_H: u32 = 32;
const SKY_VIEW_W: u32 = 192;
const SKY_VIEW_H: u32 = 108;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct AtmosphereParams {
    sun_direction: [f32; 4],
    rayleigh_mie: [f32; 4],
    screen_size: [f32; 4],
}

pub struct SkyLutPass {
    _transmittance_texture: wgpu::Texture,
    pub transmittance_view: wgpu::TextureView,
    _multiscatter_texture: wgpu::Texture,
    pub multiscatter_view: wgpu::TextureView,
    _sky_view_texture: wgpu::Texture,
    pub sky_view_view: wgpu::TextureView,
    pub lut_sampler: wgpu::Sampler,

    uniform_buffer: wgpu::Buffer,
    bgl: wgpu::BindGroupLayout,
    transmittance_pipeline: wgpu::ComputePipeline,
    multiscatter_pipeline: wgpu::ComputePipeline,
    sky_view_pipeline: wgpu::ComputePipeline,

    // Dirty tracking
    cached_sun_dir: [f32; 3],
    cached_rayleigh: f32,
    cached_mie: f32,
    cached_anisotropy: f32,
    dirty: bool,
}

impl SkyLutPass {
    pub fn new(device: &wgpu::Device) -> Self {
        let transmittance_texture = create_lut_texture(
            device,
            TRANSMITTANCE_W,
            TRANSMITTANCE_H,
            "sky-lut-transmittance",
        );
        let transmittance_view =
            transmittance_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let multiscatter_texture = create_lut_texture(
            device,
            MULTISCATTER_W,
            MULTISCATTER_H,
            "sky-lut-multiscatter",
        );
        let multiscatter_view =
            multiscatter_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let sky_view_texture =
            create_lut_texture(device, SKY_VIEW_W, SKY_VIEW_H, "sky-lut-sky-view");
        let sky_view_view = sky_view_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let lut_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("sky-lut-sampler"),
            address_mode_u: wgpu::AddressMode::Repeat, // azimuth wraps around
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sky-lut-params"),
            size: std::mem::size_of::<AtmosphereParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Bind group layout:
        // 0: uniform buffer
        // 1: transmittance LUT (read)
        // 2: multi-scatter LUT (read)
        // 3: output storage texture (write)
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sky-lut-bgl"),
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
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sky-lut-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/sky_lut.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sky-lut-layout"),
            bind_group_layouts: &[&bgl],
            immediate_size: 0,
        });

        let transmittance_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("sky-lut-transmittance-pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("compute_transmittance"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let multiscatter_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("sky-lut-multiscatter-pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("compute_multiscatter"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let sky_view_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("sky-lut-sky-view-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("compute_sky_view"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            _transmittance_texture: transmittance_texture,
            transmittance_view,
            _multiscatter_texture: multiscatter_texture,
            multiscatter_view,
            _sky_view_texture: sky_view_texture,
            sky_view_view,
            lut_sampler,
            uniform_buffer,
            bgl,
            transmittance_pipeline,
            multiscatter_pipeline,
            sky_view_pipeline,
            cached_sun_dir: [0.0; 3],
            cached_rayleigh: 0.0,
            cached_mie: 0.0,
            cached_anisotropy: 0.0,
            dirty: true,
        }
    }

    /// Update parameters and check if LUTs need regeneration.
    pub fn update(
        &mut self,
        sun_direction: Vec3,
        rayleigh_strength: f32,
        mie_strength: f32,
        mie_anisotropy: f32,
    ) {
        let sun = sun_direction.normalize();
        let sd = [sun.x, sun.y, sun.z];
        if sd != self.cached_sun_dir
            || rayleigh_strength != self.cached_rayleigh
            || mie_strength != self.cached_mie
            || mie_anisotropy != self.cached_anisotropy
        {
            self.cached_sun_dir = sd;
            self.cached_rayleigh = rayleigh_strength;
            self.cached_mie = mie_strength;
            self.cached_anisotropy = mie_anisotropy;
            self.dirty = true;
        }
    }

    /// Generate LUTs if dirty. Returns true if work was dispatched.
    pub fn generate_if_dirty(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) -> bool {
        if !self.dirty {
            return false;
        }
        self.dirty = false;

        // Write uniform buffer
        let uniforms = AtmosphereParams {
            sun_direction: [
                self.cached_sun_dir[0],
                self.cached_sun_dir[1],
                self.cached_sun_dir[2],
                0.0,
            ],
            rayleigh_mie: [
                self.cached_rayleigh,
                self.cached_mie,
                self.cached_anisotropy,
                0.0,
            ],
            screen_size: [0.0; 4],
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        // Create a dummy 1×1 texture for unused bindings
        let dummy = create_lut_texture(device, 1, 1, "sky-lut-dummy");
        let dummy_view = dummy.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sky-lut-gen-encoder"),
        });

        // Pass 1: Transmittance (no dependencies)
        {
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("sky-lut-transmittance-bg"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&dummy_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&dummy_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&self.transmittance_view),
                    },
                ],
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sky-lut-transmittance-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.transmittance_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(TRANSMITTANCE_W.div_ceil(8), TRANSMITTANCE_H.div_ceil(8), 1);
        }

        // Pass 2: Multi-scatter (reads transmittance)
        {
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("sky-lut-multiscatter-bg"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.transmittance_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&dummy_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&self.multiscatter_view),
                    },
                ],
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sky-lut-multiscatter-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.multiscatter_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(MULTISCATTER_W.div_ceil(8), MULTISCATTER_H.div_ceil(8), 1);
        }

        // Pass 3: Sky-view (reads transmittance + multi-scatter)
        {
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("sky-lut-sky-view-bg"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.transmittance_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&self.multiscatter_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&self.sky_view_view),
                    },
                ],
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sky-lut-sky-view-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.sky_view_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(SKY_VIEW_W.div_ceil(8), SKY_VIEW_H.div_ceil(8), 1);
        }

        queue.submit([encoder.finish()]);
        true
    }
}

fn create_lut_texture(
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
