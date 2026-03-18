//! Image-Based Lighting (IBL) pass — specular environment map pipeline.
//!
//! Three stages:
//! 1. **BRDF LUT** (one-time): 256x256 Rg16Float split-sum integration table.
//! 2. **Sky cubemap** (per-dirty): Render atmospheric sky into 6-face cubemap from sky-view LUT.
//! 3. **Prefiltered environment map**: GGX importance-sample blur per mip level.
//!
//! The material resolve shader uses the prefiltered cubemap + BRDF LUT for
//! physically correct image-based specular lighting.

use bytemuck::{Pod, Zeroable};

use crate::scene::SceneSettings;

/// Cubemap face resolution (per face, mip 0).
const CUBEMAP_SIZE: u32 = 128;
/// BRDF LUT dimensions.
const BRDF_LUT_SIZE: u32 = 256;
/// Number of mip levels for the prefiltered environment map.
const MIP_LEVELS: u32 = 7;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct PrefilterParams {
    face: u32,
    mip_level: u32,
    face_size: u32,
    max_mip: u32,
    sun_direction: [f32; 4],
    sun_color: [f32; 4],
    fog_color: [f32; 4],
    sky_zenith: [f32; 4],
    sky_horizon: [f32; 4],
    atmosphere_params: [f32; 4],
}

pub struct IblPass {
    // Textures
    _brdf_lut_texture: wgpu::Texture,
    brdf_lut_view: wgpu::TextureView,
    env_cubemap: wgpu::Texture,
    env_cubemap_view: wgpu::TextureView,
    cubemap_sampler: wgpu::Sampler,
    brdf_lut_sampler: wgpu::Sampler,

    // Pipelines
    brdf_lut_pipeline: wgpu::ComputePipeline,
    sky_to_cubemap_pipeline: wgpu::ComputePipeline,
    prefilter_pipeline: wgpu::ComputePipeline,

    // Bind group layouts
    brdf_bgl: wgpu::BindGroupLayout,
    prefilter_bgl: wgpu::BindGroupLayout,

    // Uniform buffer for prefilter params
    prefilter_uniform_buffer: wgpu::Buffer,

    // State
    brdf_generated: bool,
    last_sun_dir: [f32; 3],
    last_sun_strength: f32,
    last_rayleigh: f32,
    last_mie: f32,
    last_mie_anisotropy: f32,
}

impl IblPass {
    pub fn new(device: &wgpu::Device) -> Self {
        // -- BRDF LUT texture (Rg16Float, 256x256) --
        let brdf_lut_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ibl-brdf-lut"),
            size: wgpu::Extent3d {
                width: BRDF_LUT_SIZE,
                height: BRDF_LUT_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rg16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let brdf_lut_view = brdf_lut_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // -- Environment cubemap (Rgba16Float, 128x128 per face, 7 mip levels) --
        let env_cubemap = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ibl-env-cubemap"),
            size: wgpu::Extent3d {
                width: CUBEMAP_SIZE,
                height: CUBEMAP_SIZE,
                depth_or_array_layers: 6,
            },
            mip_level_count: MIP_LEVELS,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let env_cubemap_view = env_cubemap.create_view(&wgpu::TextureViewDescriptor {
            label: Some("ibl-env-cubemap-view"),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..Default::default()
        });

        // -- Samplers --
        let cubemap_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ibl-cubemap-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        let brdf_lut_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ibl-brdf-lut-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // -- BRDF LUT pipeline --
        let brdf_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ibl-brdf-bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rg16Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            }],
        });

        let brdf_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ibl-brdf-lut-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/brdf_lut.wgsl").into()),
        });

        let brdf_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ibl-brdf-layout"),
            bind_group_layouts: &[&brdf_bgl],
            immediate_size: 0,
        });

        let brdf_lut_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ibl-brdf-lut-pipeline"),
            layout: Some(&brdf_layout),
            module: &brdf_shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // -- Prefilter pipeline (shared layout for sky_to_cubemap and prefilter_mip) --
        let prefilter_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ibl-prefilter-bgl"),
            entries: &[
                // 0: PrefilterParams uniform
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
                // 1: Sky-view LUT
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // 2: LUT sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // 3: Source cubemap (for prefilter reads)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::Cube,
                        multisampled: false,
                    },
                    count: None,
                },
                // 4: Cubemap sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // 5: Output face (storage texture, one face at a time)
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

        let prefilter_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ibl-prefilter-shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/prefilter_env.wgsl").into(),
            ),
        });

        let prefilter_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ibl-prefilter-layout"),
            bind_group_layouts: &[&prefilter_bgl],
            immediate_size: 0,
        });

        let sky_to_cubemap_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("ibl-sky-to-cubemap-pipeline"),
                layout: Some(&prefilter_layout),
                module: &prefilter_shader,
                entry_point: Some("sky_to_cubemap"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let prefilter_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ibl-prefilter-pipeline"),
            layout: Some(&prefilter_layout),
            module: &prefilter_shader,
            entry_point: Some("prefilter_mip"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // -- Uniform buffer --
        let prefilter_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ibl-prefilter-params"),
            size: std::mem::size_of::<PrefilterParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            _brdf_lut_texture: brdf_lut_texture,
            brdf_lut_view,
            env_cubemap,
            env_cubemap_view,
            cubemap_sampler,
            brdf_lut_sampler,
            brdf_lut_pipeline,
            sky_to_cubemap_pipeline,
            prefilter_pipeline,
            brdf_bgl,
            prefilter_bgl,
            prefilter_uniform_buffer,
            brdf_generated: false,
            last_sun_dir: [0.0; 3],
            last_sun_strength: 0.0,
            last_rayleigh: 0.0,
            last_mie: 0.0,
            last_mie_anisotropy: 0.0,
        }
    }

    /// Generate the BRDF LUT (one-time).
    pub fn ensure_brdf_lut(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if self.brdf_generated {
            return;
        }
        self.brdf_generated = true;

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ibl-brdf-bg"),
            layout: &self.brdf_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&self.brdf_lut_view),
            }],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ibl-brdf-encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ibl-brdf-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.brdf_lut_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(BRDF_LUT_SIZE.div_ceil(8), BRDF_LUT_SIZE.div_ceil(8), 1);
        }
        queue.submit([encoder.finish()]);
    }

    /// Check if the cubemap needs regeneration based on sky parameters.
    fn is_dirty(&self, settings: &SceneSettings) -> bool {
        let sun = settings.sun_direction.normalize();
        let sd = [sun.x, sun.y, sun.z];
        sd != self.last_sun_dir
            || settings.sun_strength != self.last_sun_strength
            || settings.rayleigh_strength != self.last_rayleigh
            || settings.mie_strength != self.last_mie
            || settings.mie_anisotropy != self.last_mie_anisotropy
    }

    /// Update the cubemap from the sky-view LUT when parameters change.
    pub fn update_cubemap(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        sky_view_view: &wgpu::TextureView,
        sky_lut_sampler: &wgpu::Sampler,
        settings: &SceneSettings,
    ) {
        if !self.is_dirty(settings) {
            return;
        }

        let sun = settings.sun_direction.normalize();
        self.last_sun_dir = [sun.x, sun.y, sun.z];
        self.last_sun_strength = settings.sun_strength;
        self.last_rayleigh = settings.rayleigh_strength;
        self.last_mie = settings.mie_strength;
        self.last_mie_anisotropy = settings.mie_anisotropy;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ibl-cubemap-encoder"),
        });

        // Stage 1: Render sky into cubemap mip 0 (6 faces)
        for face in 0..6u32 {
            let face_view = self.env_cubemap.create_view(&wgpu::TextureViewDescriptor {
                label: Some("ibl-cubemap-face-mip0"),
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: face,
                array_layer_count: Some(1),
                base_mip_level: 0,
                mip_level_count: Some(1),
                ..Default::default()
            });

            let params = PrefilterParams {
                face,
                mip_level: 0,
                face_size: CUBEMAP_SIZE,
                max_mip: MIP_LEVELS - 1,
                sun_direction: [sun.x, sun.y, sun.z, 0.0],
                sun_color: [
                    settings.sun_color.x,
                    settings.sun_color.y,
                    settings.sun_color.z,
                    settings.sun_strength,
                ],
                fog_color: [
                    settings.fog_color.x,
                    settings.fog_color.y,
                    settings.fog_color.z,
                    1.0,
                ],
                sky_zenith: [
                    settings.sky_zenith.x,
                    settings.sky_zenith.y,
                    settings.sky_zenith.z,
                    1.0,
                ],
                sky_horizon: [
                    settings.sky_horizon.x,
                    settings.sky_horizon.y,
                    settings.sky_horizon.z,
                    1.0,
                ],
                atmosphere_params: [
                    settings.rayleigh_strength,
                    settings.mie_strength,
                    settings.mie_anisotropy,
                    settings.cloud_coverage,
                ],
            };
            queue.write_buffer(
                &self.prefilter_uniform_buffer,
                0,
                bytemuck::bytes_of(&params),
            );

            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ibl-sky-to-cubemap-bg"),
                layout: &self.prefilter_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.prefilter_uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(sky_view_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(sky_lut_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&self.env_cubemap_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::Sampler(&self.cubemap_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::TextureView(&face_view),
                    },
                ],
            });

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("ibl-sky-to-cubemap-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.sky_to_cubemap_pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(CUBEMAP_SIZE.div_ceil(8), CUBEMAP_SIZE.div_ceil(8), 1);
            }
        }

        // Submit the sky-to-cubemap pass so mip 0 is available for reading
        queue.submit([encoder.finish()]);

        // Stage 2: Prefilter mips 1..MIP_LEVELS
        // Each mip reads from the full cubemap (all mips available via trilinear)
        // and writes to a single face of the target mip.
        for mip in 1..MIP_LEVELS {
            let mip_size = (CUBEMAP_SIZE >> mip).max(1);

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("ibl-prefilter-encoder"),
            });

            for face in 0..6u32 {
                let face_view = self.env_cubemap.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("ibl-cubemap-face-mip"),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: face,
                    array_layer_count: Some(1),
                    base_mip_level: mip,
                    mip_level_count: Some(1),
                    ..Default::default()
                });

                let params = PrefilterParams {
                    face,
                    mip_level: mip,
                    face_size: mip_size,
                    max_mip: MIP_LEVELS - 1,
                    sun_direction: [sun.x, sun.y, sun.z, 0.0],
                    sun_color: [
                        settings.sun_color.x,
                        settings.sun_color.y,
                        settings.sun_color.z,
                        settings.sun_strength,
                    ],
                    fog_color: [
                        settings.fog_color.x,
                        settings.fog_color.y,
                        settings.fog_color.z,
                        1.0,
                    ],
                    sky_zenith: [
                        settings.sky_zenith.x,
                        settings.sky_zenith.y,
                        settings.sky_zenith.z,
                        1.0,
                    ],
                    sky_horizon: [
                        settings.sky_horizon.x,
                        settings.sky_horizon.y,
                        settings.sky_horizon.z,
                        1.0,
                    ],
                    atmosphere_params: [
                        settings.rayleigh_strength,
                        settings.mie_strength,
                        settings.mie_anisotropy,
                        settings.cloud_coverage,
                    ],
                };
                queue.write_buffer(
                    &self.prefilter_uniform_buffer,
                    0,
                    bytemuck::bytes_of(&params),
                );

                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("ibl-prefilter-bg"),
                    layout: &self.prefilter_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.prefilter_uniform_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(sky_view_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(sky_lut_sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(&self.env_cubemap_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::Sampler(&self.cubemap_sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: wgpu::BindingResource::TextureView(&face_view),
                        },
                    ],
                });

                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("ibl-prefilter-pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.prefilter_pipeline);
                    pass.set_bind_group(0, &bg, &[]);
                    pass.dispatch_workgroups(mip_size.div_ceil(8), mip_size.div_ceil(8), 1);
                }
            }

            queue.submit([encoder.finish()]);
        }
    }

    /// Prefiltered environment cubemap view (all mips, all faces, Cube dimension).
    pub fn env_cubemap_view(&self) -> &wgpu::TextureView {
        &self.env_cubemap_view
    }

    /// BRDF LUT view (2D, Rg16Float).
    pub fn brdf_lut_view(&self) -> &wgpu::TextureView {
        &self.brdf_lut_view
    }

    /// Cubemap sampler (trilinear, ClampToEdge).
    pub fn cubemap_sampler(&self) -> &wgpu::Sampler {
        &self.cubemap_sampler
    }

    /// BRDF LUT sampler (bilinear, ClampToEdge).
    pub fn brdf_lut_sampler(&self) -> &wgpu::Sampler {
        &self.brdf_lut_sampler
    }
}
