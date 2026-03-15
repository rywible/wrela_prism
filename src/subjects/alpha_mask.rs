/// Pre-baked procedural needle/leaf alpha mask texture (256x256, R8 grayscale).
///
/// Used by foliage billboards for alpha-tested rendering in the visibility buffer.

use crate::util::{hash01, smoothstep};

const MASK_SIZE: u32 = 256;

/// Generate a procedural needle cluster alpha mask.
///
/// Creates a texture where white regions represent leaf/needle clusters
/// and black regions are transparent. The pattern includes:
/// - Radiating needle shapes from center
/// - Noise for natural variation
/// - Soft edges for anti-aliasing at the boundary
pub fn generate_needle_mask(seed: u32) -> Vec<u8> {
    let mut data = vec![0u8; (MASK_SIZE * MASK_SIZE) as usize];

    for y in 0..MASK_SIZE {
        for x in 0..MASK_SIZE {
            let u = (x as f32 / MASK_SIZE as f32) * 2.0 - 1.0;
            let v = (y as f32 / MASK_SIZE as f32) * 2.0 - 1.0;
            let r = (u * u + v * v).sqrt();

            // Base circular falloff
            let circle = 1.0 - smoothstep(0.6, 0.95, r);
            if circle <= 0.0 {
                continue;
            }

            // Radiating needle pattern
            let angle = v.atan2(u);
            let needle_count = 12.0 + hash01(seed.wrapping_add(1)) * 6.0;
            let needle = ((angle * needle_count).sin() * 0.5 + 0.5).powf(0.3);

            // Radial density variation
            let radial_density = smoothstep(0.05, 0.3, r) * (1.0 - smoothstep(0.7, 0.9, r));

            // Noise for natural variation
            let noise = hash01(
                seed.wrapping_add(x)
                    .wrapping_mul(31)
                    .wrapping_add(y.wrapping_mul(97)),
            );
            let noise_threshold = 0.25 + radial_density * 0.3;

            let alpha = if needle * circle > 0.3 && noise > noise_threshold {
                (circle * needle * 1.5).clamp(0.0, 1.0)
            } else {
                0.0
            };

            data[(y * MASK_SIZE + x) as usize] = (alpha * 255.0) as u8;
        }
    }

    data
}

/// Upload the alpha mask as a GPU texture.
pub fn create_alpha_mask_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    seed: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let data = generate_needle_mask(seed);

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("foliage-alpha-mask"),
        size: wgpu::Extent3d {
            width: MASK_SIZE,
            height: MASK_SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &data,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(MASK_SIZE),
            rows_per_image: Some(MASK_SIZE),
        },
        wgpu::Extent3d {
            width: MASK_SIZE,
            height: MASK_SIZE,
            depth_or_array_layers: 1,
        },
    );

    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mask_has_correct_size() {
        let data = generate_needle_mask(42);
        assert_eq!(data.len(), 256 * 256);
    }

    #[test]
    fn mask_has_nonzero_coverage() {
        let data = generate_needle_mask(42);
        let nonzero = data.iter().filter(|&&v| v > 0).count();
        let total = data.len();
        let coverage = nonzero as f32 / total as f32;
        assert!(
            coverage > 0.05 && coverage < 0.95,
            "mask coverage {:.1}% should be between 5% and 95%",
            coverage * 100.0
        );
    }

    #[test]
    fn different_seeds_produce_different_masks() {
        let a = generate_needle_mask(1);
        let b = generate_needle_mask(2);
        let diff: usize = a.iter().zip(b.iter()).filter(|(a, b)| a != b).count();
        assert!(diff > 100, "different seeds should produce visibly different masks");
    }
}
