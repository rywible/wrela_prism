use crate::scene::AreaLight;

/// Maximum number of area lights that can be uploaded to the GPU.
pub const MAX_AREA_LIGHTS: usize = 16;

/// GPU-packed area light data (32 bytes per light).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuAreaLight {
    /// xyz = position (sphere) or start (tube), w = radius
    pub pos_radius: [f32; 4],
    /// xyz = color * intensity, w = light_type (0 = sphere, 1 = tube)
    pub color_type: [f32; 4],
    /// xyz = end position (tube only), w = unused
    pub end_unused: [f32; 4],
    /// Reserved for future use
    pub _pad: [f32; 4],
}

/// GPU uniform holding the array of area lights + count.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AreaLightUniforms {
    /// x = light_count, yzw = reserved
    pub params: [f32; 4],
    pub lights: [GpuAreaLight; MAX_AREA_LIGHTS],
}

/// Manages GPU resources for area lights (light buffer + LTC LUT textures).
pub struct AreaLightPass {
    pub ltc_lut_texture: wgpu::Texture,
    pub ltc_lut_view: wgpu::TextureView,
    pub ltc_amp_texture: wgpu::Texture,
    pub ltc_amp_view: wgpu::TextureView,
    pub light_buffer: wgpu::Buffer,
    pub light_count: u32,
}

impl AreaLightPass {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        // Create and upload LTC matrix LUT (64x64 Rgba32Float)
        let ltc_lut_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("prism-ltc-lut"),
            size: wgpu::Extent3d {
                width: LTC_LUT_SIZE,
                height: LTC_LUT_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let ltc_lut_view = ltc_lut_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Upload precomputed LTC matrix data
        let lut_data = generate_ltc_lut();
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &ltc_lut_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&lut_data),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(LTC_LUT_SIZE * 16), // 4 f32 * 4 bytes = 16 bytes per texel
                rows_per_image: Some(LTC_LUT_SIZE),
            },
            wgpu::Extent3d {
                width: LTC_LUT_SIZE,
                height: LTC_LUT_SIZE,
                depth_or_array_layers: 1,
            },
        );

        // Create amplitude LUT (64x64 Rgba32Float)
        let ltc_amp_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("prism-ltc-amp"),
            size: wgpu::Extent3d {
                width: LTC_LUT_SIZE,
                height: LTC_LUT_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let ltc_amp_view = ltc_amp_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let amp_data = generate_ltc_amp();
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &ltc_amp_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&amp_data),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(LTC_LUT_SIZE * 16),
                rows_per_image: Some(LTC_LUT_SIZE),
            },
            wgpu::Extent3d {
                width: LTC_LUT_SIZE,
                height: LTC_LUT_SIZE,
                depth_or_array_layers: 1,
            },
        );

        // Create area light uniform buffer
        let light_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("prism-area-lights"),
            size: std::mem::size_of::<AreaLightUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            ltc_lut_texture,
            ltc_lut_view,
            ltc_amp_texture,
            ltc_amp_view,
            light_buffer,
            light_count: 0,
        }
    }

    /// Upload area light data to the GPU.
    pub fn update_lights(&mut self, queue: &wgpu::Queue, lights: &[AreaLight]) {
        let count = lights.len().min(MAX_AREA_LIGHTS);
        self.light_count = count as u32;

        let mut gpu_lights = [GpuAreaLight {
            pos_radius: [0.0; 4],
            color_type: [0.0; 4],
            end_unused: [0.0; 4],
            _pad: [0.0; 4],
        }; MAX_AREA_LIGHTS];

        for (i, light) in lights.iter().take(count).enumerate() {
            match light {
                AreaLight::Sphere {
                    position,
                    radius,
                    color,
                    intensity,
                } => {
                    gpu_lights[i] = GpuAreaLight {
                        pos_radius: [position.x, position.y, position.z, *radius],
                        color_type: [
                            color.x * intensity,
                            color.y * intensity,
                            color.z * intensity,
                            0.0, // type = sphere
                        ],
                        end_unused: [0.0; 4],
                        _pad: [0.0; 4],
                    };
                }
                AreaLight::Tube {
                    start,
                    end,
                    radius,
                    color,
                    intensity,
                } => {
                    gpu_lights[i] = GpuAreaLight {
                        pos_radius: [start.x, start.y, start.z, *radius],
                        color_type: [
                            color.x * intensity,
                            color.y * intensity,
                            color.z * intensity,
                            1.0, // type = tube
                        ],
                        end_unused: [end.x, end.y, end.z, 0.0],
                        _pad: [0.0; 4],
                    };
                }
            }
        }

        let uniforms = AreaLightUniforms {
            params: [count as f32, 0.0, 0.0, 0.0],
            lights: gpu_lights,
        };

        queue.write_buffer(&self.light_buffer, 0, bytemuck::bytes_of(&uniforms));
    }
}

// ──────────────────────── LTC LUT Generation ────────────────────────

const LTC_LUT_SIZE: u32 = 64;

/// Generate the LTC inverse matrix LUT.
///
/// Maps (roughness, NdotV) to 4 coefficients of the 3x3 inverse LTC matrix.
/// The matrix M^-1 transforms the distribution lobe to a clamped cosine.
///
/// For a full implementation, these would come from Heitz et al. 2016 precomputed tables.
/// Here we use an analytical approximation that produces reasonable results for
/// GGX distributions, following the simplified fitting from the LTC paper.
fn generate_ltc_lut() -> Vec<[f32; 4]> {
    let size = LTC_LUT_SIZE as usize;
    let mut data = vec![[0.0f32; 4]; size * size];

    for y in 0..size {
        for x in 0..size {
            let roughness = (y as f32 + 0.5) / size as f32;
            let ndotv = (x as f32 + 0.5) / size as f32;

            // Approximate LTC matrix coefficients based on GGX roughness + view angle.
            // M^-1 = [[a, 0, b], [0, c, 0], [d, 0, e]] compressed to 4 values:
            // store (a, b, d, e) — c is derived as 1.0 for isotropic lobes.
            let alpha = roughness * roughness; // GGX alpha
            let alpha2 = alpha * alpha;

            // Fitting from Heitz 2016, simplified:
            // At roughness=0 (mirror): M^-1 approaches reflection-aligned delta
            // At roughness=1 (diffuse): M^-1 approaches identity (cosine lobe)
            let cos_theta = ndotv;
            let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();

            // Scale factor for lobe width
            let a = 1.0 / (1.0 + alpha2 * (1.0 - cos_theta));
            // Off-diagonal: skew toward reflection direction
            let b = -sin_theta * alpha2 / (1.0 + alpha2);
            // Lower row: normalization
            let d = sin_theta * alpha2 / (1.0 + alpha2);
            let e = cos_theta / (1.0 + alpha2 * (1.0 - cos_theta));

            data[y * size + x] = [a, b, d, e];
        }
    }

    data
}

/// Generate the LTC amplitude/magnitude LUT.
///
/// Maps (roughness, NdotV) to the Fresnel-weighted magnitude of the lobe.
/// Used to scale the area light contribution.
fn generate_ltc_amp() -> Vec<[f32; 4]> {
    let size = LTC_LUT_SIZE as usize;
    let mut data = vec![[0.0f32; 4]; size * size];

    for y in 0..size {
        for x in 0..size {
            let roughness = (y as f32 + 0.5) / size as f32;
            let ndotv = (x as f32 + 0.5) / size as f32;

            let alpha = roughness * roughness;

            // Fresnel term (Schlick with F0=0.04 for dielectrics)
            let f0 = 0.04_f32;
            let fresnel = f0 + (1.0 - f0) * (1.0 - ndotv).max(0.0).powf(5.0);

            // Amplitude: energy of the transformed lobe
            // Decreases with roughness (diffuse surfaces reflect less of the area light)
            let magnitude = (1.0 - alpha * 0.5) * fresnel + alpha * 0.1;

            // Store magnitude in R, Fresnel in G, reserved in BA
            data[y * size + x] = [magnitude, fresnel, 0.0, 1.0];
        }
    }

    data
}
