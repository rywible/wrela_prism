//! GPU-accelerated 3D light field for growth simulation.
//!
//! Uses a 128³ voxel volume to compute light availability via Beer-Lambert
//! extinction. Three compute passes: voxelize density, propagate light
//! top-down, and query light at meristem positions.

use std::sync::mpsc;

use glam::Vec3;

const VOLUME_RES: u32 = 128;

/// Configuration uniform for the light field shaders.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LightFieldConfig {
    volume_min: [f32; 3],
    extinction: f32,
    volume_max: [f32; 3],
    resolution: u32,
    voxel_size: [f32; 3],
    capsule_count: u32,
    query_count: u32,
    _pad: [u32; 3],
}

/// A capsule segment for the GPU density voxelization.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuCapsule {
    a: [f32; 3],
    radius_a: f32,
    b: [f32; 3],
    radius_b: f32,
    leaf_density: f32,
    _pad: [f32; 3],
}

/// GPU-side 3D light field for forest-scale light competition.
pub struct LightFieldGpu {
    #[allow(dead_code)]
    density_texture: wgpu::Texture,
    density_view: wgpu::TextureView,
    #[allow(dead_code)]
    light_texture: wgpu::Texture,
    light_view: wgpu::TextureView,

    density_pipeline: wgpu::ComputePipeline,
    propagate_pipeline: wgpu::ComputePipeline,
    query_pipeline: wgpu::ComputePipeline,

    density_bgl: wgpu::BindGroupLayout,
    propagate_bgl: wgpu::BindGroupLayout,
    query_bgl: wgpu::BindGroupLayout,

    config_buffer: wgpu::Buffer,
    capsule_buffer: wgpu::Buffer,
    query_buffer: wgpu::Buffer,
    result_buffer: wgpu::Buffer,
    readback_buffer: wgpu::Buffer,

    max_capsules: u32,
    max_queries: u32,
}

impl LightFieldGpu {
    pub fn new(device: &wgpu::Device) -> Self {
        let max_capsules = 16384u32;
        let max_queries = 8192u32;

        // 3D textures: density and light (R32Float, 128³).
        let tex_desc = |label| wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: VOLUME_RES,
                height: VOLUME_RES,
                depth_or_array_layers: VOLUME_RES,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };

        let density_texture = device.create_texture(&tex_desc("light-field-density"));
        let density_view = density_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let light_texture = device.create_texture(&tex_desc("light-field-light"));
        let light_view = light_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Buffers.
        let config_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("light-field-config"),
            size: std::mem::size_of::<LightFieldConfig>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let capsule_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("light-field-capsules"),
            size: (max_capsules as usize * std::mem::size_of::<GpuCapsule>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let query_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("light-field-queries"),
            size: (max_queries as usize * 16) as u64, // vec4<f32> per query
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let result_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("light-field-results"),
            size: (max_queries as usize * 4) as u64, // f32 per result
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("light-field-readback"),
            size: (max_queries as usize * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Shader module.
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("light-field-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/light_field.wgsl").into()),
        });

        // Bind group layouts.
        let density_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("light-field-density-bgl"),
            entries: &[
                // binding 0: config uniform
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
                // binding 1: capsules storage
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 2: density texture (write)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D3,
                    },
                    count: None,
                },
            ],
        });

        let propagate_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("light-field-propagate-bgl"),
            entries: &[
                // binding 0: config
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
                // binding 1: density texture (read via sampler)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                // binding 2: light texture (write)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D3,
                    },
                    count: None,
                },
            ],
        });

        let query_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("light-field-query-bgl"),
            entries: &[
                // binding 0: config
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
                // binding 1: light texture (read)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                // binding 2: query points
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 3: results
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Pipelines.
        let density_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("light-field-density-layout"),
            bind_group_layouts: &[&density_bgl],
            immediate_size: 0,
        });
        let density_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("light-field-density-pipeline"),
            layout: Some(&density_layout),
            module: &shader,
            entry_point: Some("voxelize_density"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let propagate_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("light-field-propagate-layout"),
            bind_group_layouts: &[&propagate_bgl],
            immediate_size: 0,
        });
        let propagate_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("light-field-propagate-pipeline"),
            layout: Some(&propagate_layout),
            module: &shader,
            entry_point: Some("propagate_light"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let query_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("light-field-query-layout"),
            bind_group_layouts: &[&query_bgl],
            immediate_size: 0,
        });
        let query_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("light-field-query-pipeline"),
            layout: Some(&query_layout),
            module: &shader,
            entry_point: Some("query_light"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            density_texture,
            density_view,
            light_texture,
            light_view,
            density_pipeline,
            propagate_pipeline,
            query_pipeline,
            density_bgl,
            propagate_bgl,
            query_bgl,
            config_buffer,
            capsule_buffer,
            query_buffer,
            result_buffer,
            readback_buffer,
            max_capsules,
            max_queries,
        }
    }

    /// Query light values at meristem positions using the GPU light field.
    ///
    /// `capsules`: (a, b, radius_a, radius_b, leaf_density) for each foliage-bearing segment.
    /// `positions`: world-space query points (one per meristem).
    /// Returns light value [0..1] for each query point.
    pub fn query_meristem_light(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        capsules: &[(Vec3, Vec3, f32, f32, f32)],
        positions: &[Vec3],
    ) -> Vec<f32> {
        if positions.is_empty() {
            return Vec::new();
        }

        // Compute volume bounds from capsules + padding.
        let mut vol_min = Vec3::splat(f32::MAX);
        let mut vol_max = Vec3::splat(f32::MIN);
        for &(a, b, ra, rb, _) in capsules {
            let max_r = ra.max(rb);
            vol_min = vol_min
                .min(a - Vec3::splat(max_r))
                .min(b - Vec3::splat(max_r));
            vol_max = vol_max
                .max(a + Vec3::splat(max_r))
                .max(b + Vec3::splat(max_r));
        }
        // Add padding.
        let padding = (vol_max - vol_min).length() * 0.05;
        vol_min -= Vec3::splat(padding);
        vol_max += Vec3::splat(padding);

        let voxel_size = (vol_max - vol_min) / VOLUME_RES as f32;

        let capsule_count = (capsules.len() as u32).min(self.max_capsules);
        let query_count = (positions.len() as u32).min(self.max_queries);

        // Upload config.
        let config = LightFieldConfig {
            volume_min: vol_min.to_array(),
            extinction: 0.8,
            volume_max: vol_max.to_array(),
            resolution: VOLUME_RES,
            voxel_size: voxel_size.to_array(),
            capsule_count,
            query_count,
            _pad: [0; 3],
        };
        queue.write_buffer(&self.config_buffer, 0, bytemuck::bytes_of(&config));

        // Upload capsules.
        let gpu_capsules: Vec<GpuCapsule> = capsules
            .iter()
            .take(capsule_count as usize)
            .map(|&(a, b, ra, rb, ld)| GpuCapsule {
                a: a.to_array(),
                radius_a: ra,
                b: b.to_array(),
                radius_b: rb,
                leaf_density: ld,
                _pad: [0.0; 3],
            })
            .collect();
        queue.write_buffer(&self.capsule_buffer, 0, bytemuck::cast_slice(&gpu_capsules));

        // Upload query points as vec4 (xyz + padding).
        let query_data: Vec<[f32; 4]> = positions
            .iter()
            .take(query_count as usize)
            .map(|p| [p.x, p.y, p.z, 0.0])
            .collect();
        queue.write_buffer(&self.query_buffer, 0, bytemuck::cast_slice(&query_data));

        // Create bind groups.
        let density_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("light-field-density-bg"),
            layout: &self.density_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.config_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.capsule_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&self.density_view),
                },
            ],
        });

        let propagate_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("light-field-propagate-bg"),
            layout: &self.propagate_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.config_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.density_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&self.light_view),
                },
            ],
        });

        let query_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("light-field-query-bg"),
            layout: &self.query_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.config_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.light_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.query_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.result_buffer.as_entire_binding(),
                },
            ],
        });

        // Encode and dispatch.
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("light-field-encoder"),
        });

        // Pass 1: Voxelize density.
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("voxelize-density"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.density_pipeline);
            pass.set_bind_group(0, &density_bg, &[]);
            let groups = VOLUME_RES.div_ceil(4);
            pass.dispatch_workgroups(groups, groups, groups);
        }

        // Pass 2: Propagate light top-down.
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("propagate-light"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.propagate_pipeline);
            pass.set_bind_group(0, &propagate_bg, &[]);
            let groups_xz = VOLUME_RES.div_ceil(8);
            pass.dispatch_workgroups(groups_xz, groups_xz, 1);
        }

        // Pass 3: Query light at positions.
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("query-light"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.query_pipeline);
            pass.set_bind_group(0, &query_bg, &[]);
            pass.dispatch_workgroups(query_count.div_ceil(64), 1, 1);
        }

        // Copy results to readback buffer.
        encoder.copy_buffer_to_buffer(
            &self.result_buffer,
            0,
            &self.readback_buffer,
            0,
            (query_count as u64) * 4,
        );

        queue.submit([encoder.finish()]);

        // Map and read back.
        let slice = self.readback_buffer.slice(..((query_count as u64) * 4));
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        let _ = device.poll(wgpu::PollType::wait_indefinitely());

        if receiver.recv().ok().and_then(|r| r.ok()).is_none() {
            // Fallback: return neutral light values.
            return vec![0.5; positions.len()];
        }

        let mapped = slice.get_mapped_range();
        let results: &[f32] = bytemuck::cast_slice(&mapped);
        let out = results[..query_count as usize].to_vec();
        drop(mapped);
        self.readback_buffer.unmap();

        out
    }
}
