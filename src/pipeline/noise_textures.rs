/// 3D noise texture precomputation for volumetric clouds.
///
/// Generates once at init:
/// - Shape noise: 128³ Rgba8Unorm (Perlin-Worley + Worley)
/// - Detail noise: 32³ Rgba8Unorm (multi-octave Worley)
/// - Weather map: 512² Rgba8Unorm (CPU-generated Perlin FBM)

const SHAPE_SIZE: u32 = 128;
const DETAIL_SIZE: u32 = 32;
const WEATHER_SIZE: u32 = 512;

pub struct NoiseTextures {
    pub shape_texture: wgpu::Texture,
    pub shape_view: wgpu::TextureView,
    pub detail_texture: wgpu::Texture,
    pub detail_view: wgpu::TextureView,
    pub weather_texture: wgpu::Texture,
    pub weather_view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,

    shape_pipeline: wgpu::ComputePipeline,
    detail_pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    generated: bool,
}

impl NoiseTextures {
    pub fn new(device: &wgpu::Device) -> Self {
        let shape_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("noise-shape-3d"),
            size: wgpu::Extent3d {
                width: SHAPE_SIZE,
                height: SHAPE_SIZE,
                depth_or_array_layers: SHAPE_SIZE,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let shape_view = shape_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let detail_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("noise-detail-3d"),
            size: wgpu::Extent3d {
                width: DETAIL_SIZE,
                height: DETAIL_SIZE,
                depth_or_array_layers: DETAIL_SIZE,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let detail_view = detail_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let weather_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("noise-weather-2d"),
            size: wgpu::Extent3d {
                width: WEATHER_SIZE,
                height: WEATHER_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let weather_view = weather_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("noise-repeat-sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        // Bind group layout: single storage texture (3D, write)
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("noise-gen-bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    view_dimension: wgpu::TextureViewDimension::D3,
                },
                count: None,
            }],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("noise-gen-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/noise_gen.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("noise-gen-layout"),
            bind_group_layouts: &[&bgl],
            immediate_size: 0,
        });

        let shape_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("noise-shape-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("compute_shape_noise"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let detail_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("noise-detail-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("compute_detail_noise"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            shape_texture,
            shape_view,
            detail_texture,
            detail_view,
            weather_texture,
            weather_view,
            sampler,
            shape_pipeline,
            detail_pipeline,
            bgl,
            generated: false,
        }
    }

    /// Generate all noise textures. Call once on first frame.
    /// Returns true if work was dispatched (first call only).
    pub fn ensure_generated(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) -> bool {
        if self.generated {
            return false;
        }
        self.generated = true;

        // GPU: shape noise
        let shape_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("noise-shape-bg"),
            layout: &self.bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&self.shape_view),
            }],
        });

        // GPU: detail noise
        let detail_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("noise-detail-bg"),
            layout: &self.bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&self.detail_view),
            }],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("noise-gen-encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("noise-shape-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.shape_pipeline);
            pass.set_bind_group(0, &shape_bg, &[]);
            let groups = SHAPE_SIZE.div_ceil(4);
            pass.dispatch_workgroups(groups, groups, groups);
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("noise-detail-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.detail_pipeline);
            pass.set_bind_group(0, &detail_bg, &[]);
            let groups = DETAIL_SIZE.div_ceil(4);
            pass.dispatch_workgroups(groups, groups, groups);
        }

        queue.submit([encoder.finish()]);

        // CPU: weather map
        self.generate_weather_map(queue);

        tracing::info!(
            "Noise textures generated (shape {}³, detail {}³, weather {}²)",
            SHAPE_SIZE,
            DETAIL_SIZE,
            WEATHER_SIZE
        );
        true
    }

    fn generate_weather_map(&self, queue: &wgpu::Queue) {
        let size = WEATHER_SIZE as usize;
        let mut data = vec![0u8; size * size * 4];

        for y in 0..size {
            for x in 0..size {
                let u = x as f32 / size as f32;
                let v = y as f32 / size as f32;

                // Tileable FBM via cross-fade with contrast restoration
                // The cross-fade smooths variation, so we remap to restore dynamic range
                // Boost persistence for first octaves to increase dynamic range
                let raw_cov = tileable_fbm(u, v, 4.0, 0.0, 6, 0.55);
                // Map [-1,1] FBM output to [0,1] with contrast expansion
                let coverage = (raw_cov * 0.8 + 0.5).clamp(0.0, 1.0);
                let cloud_type = tileable_fbm(u, v, 2.0, 100.0, 3, 0.5) * 0.5 + 0.5;
                let max_alt = tileable_fbm(u, v, 3.0, 200.0, 3, 0.5) * 0.5 + 0.5;
                let absorption: f32 = 0.5 + tileable_fbm(u, v, 2.5, 300.0, 2, 0.5) * 0.3;

                let idx = (y * size + x) * 4;
                data[idx] = (coverage.clamp(0.0, 1.0) * 255.0) as u8;
                data[idx + 1] = (cloud_type.clamp(0.0, 1.0) * 255.0) as u8;
                data[idx + 2] = (max_alt.clamp(0.0, 1.0) * 255.0) as u8;
                data[idx + 3] = (absorption.clamp(0.0, 1.0) * 255.0) as u8;
            }
        }

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.weather_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(WEATHER_SIZE * 4),
                rows_per_image: Some(WEATHER_SIZE),
            },
            wgpu::Extent3d {
                width: WEATHER_SIZE,
                height: WEATHER_SIZE,
                depth_or_array_layers: 1,
            },
        );
    }
}

// ──────────────────────── CPU Perlin FBM ────────────────────────

fn cpu_hash(x: f32, y: f32) -> f32 {
    let p = glam::Vec2::new(x, y);
    let p3 = glam::Vec3::new(p.x, p.y, p.x) * 0.1031;
    let p3 = glam::Vec3::new(p3.x.fract(), p3.y.fract(), p3.z.fract());
    let d = p3.dot(glam::Vec3::new(p3.y + 33.33, p3.z + 33.33, p3.x + 33.33));
    let p3 = glam::Vec3::new(p3.x + d, p3.y + d, p3.z + d);
    ((p3.x + p3.y) * p3.z).fract()
}

fn cpu_value_noise(x: f32, y: f32) -> f32 {
    let ix = x.floor();
    let iy = y.floor();
    let fx = x - ix;
    let fy = y - iy;
    // Quintic interpolation
    let ux = fx * fx * fx * (fx * (fx * 6.0 - 15.0) + 10.0);
    let uy = fy * fy * fy * (fy * (fy * 6.0 - 15.0) + 10.0);

    let n00 = cpu_hash(ix, iy);
    let n10 = cpu_hash(ix + 1.0, iy);
    let n01 = cpu_hash(ix, iy + 1.0);
    let n11 = cpu_hash(ix + 1.0, iy + 1.0);

    let nx0 = n00 + (n10 - n00) * ux;
    let nx1 = n01 + (n11 - n01) * ux;
    nx0 + (nx1 - nx0) * uy
}

/// Tileable FBM: first 2 octaves use periodic noise for seamless tiling,
/// remaining octaves use regular noise (small-scale detail won't show seams).
fn tileable_fbm(u: f32, v: f32, scale: f32, offset: f32, octaves: u32, persistence: f32) -> f32 {
    let x = u * scale + offset;
    let y = v * scale + offset;
    let period = scale as i32;
    let mut val = 0.0_f32;
    let mut amp = 0.5_f32;
    let mut freq = 1.0_f32;
    let mut p = period;
    for i in 0..octaves {
        let px = x * freq;
        let py = y * freq;
        if i < 2 {
            // Low-frequency octaves: periodic for seamless tiling
            val += amp * (cpu_value_noise_periodic(px, py, p) * 2.0 - 1.0);
            p *= 2;
        } else {
            // High-frequency octaves: regular noise with rotation for detail
            let rx = px * 0.866 + py * 0.5;
            let ry = py * 0.866 - px * 0.5;
            val += amp * (cpu_value_noise(rx, ry) * 2.0 - 1.0);
        }
        freq *= 2.0;
        amp *= persistence;
    }
    val
}

/// Value noise with periodic wrapping at `period` integer cells.
fn cpu_value_noise_periodic(x: f32, y: f32, period: i32) -> f32 {
    let ix = x.floor() as i32;
    let iy = y.floor() as i32;
    let fx = x - x.floor();
    let fy = y - y.floor();
    let ux = fx * fx * fx * (fx * (fx * 6.0 - 15.0) + 10.0);
    let uy = fy * fy * fy * (fy * (fy * 6.0 - 15.0) + 10.0);

    // Wrap integer coordinates for periodicity
    let wrap = |v: i32| -> f32 { ((v % period + period) % period) as f32 };
    let n00 = cpu_hash(wrap(ix), wrap(iy));
    let n10 = cpu_hash(wrap(ix + 1), wrap(iy));
    let n01 = cpu_hash(wrap(ix), wrap(iy + 1));
    let n11 = cpu_hash(wrap(ix + 1), wrap(iy + 1));

    let nx0 = n00 + (n10 - n00) * ux;
    let nx1 = n01 + (n11 - n01) * ux;
    nx0 + (nx1 - nx0) * uy
}
