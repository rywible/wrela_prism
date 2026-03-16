use crate::subjects::redwood_growth::FoliageSprayVariant;
use crate::util::smoothstep;

pub const FOLIAGE_VARIANT_COUNT: u32 = 5;
const ATLAS_SIZE: u32 = 256;

fn shape_alpha(variant: FoliageSprayVariant, px: f32, py: f32) -> f32 {
    let radial = (px * px + py * py).sqrt();
    let tall_radial = ((px * 0.9) * (px * 0.9) + (py * 1.25) * (py * 1.25)).sqrt();
    match variant {
        FoliageSprayVariant::SideSpray => {
            let fan = 1.0 - smoothstep(0.10, 0.68, (px * 0.85).abs() + (py + 0.20).abs() * 0.72);
            let plume = 1.0 - smoothstep(0.18, 0.96, tall_radial);
            (fan * 0.55 + plume * 0.85).clamp(0.0, 1.0)
        }
        FoliageSprayVariant::TerminalTuft => {
            let bulb = 1.0 - smoothstep(0.22, 0.94, radial);
            let tip = 1.0
                - smoothstep(
                    0.08,
                    0.78,
                    ((px * 1.1) * (px * 1.1) + (py - 0.18).powi(2)).sqrt(),
                );
            (bulb * 0.7 + tip * 0.6).clamp(0.0, 1.0)
        }
        FoliageSprayVariant::DroopingSecondary => {
            let drape = 1.0
                - smoothstep(
                    0.12,
                    0.84,
                    ((px * 0.8) * (px * 0.8) + (py + 0.28).powi(2) * 0.42).sqrt(),
                );
            let tail = 1.0
                - smoothstep(
                    0.08,
                    0.62,
                    ((px * 1.2) * (px * 1.2) + (py - 0.24).powi(2) * 1.8).sqrt(),
                );
            (drape * 0.75 + tail * 0.45).clamp(0.0, 1.0)
        }
        FoliageSprayVariant::SparseSilhouette => {
            let shell = 1.0 - smoothstep(0.42, 0.98, radial);
            let notch = smoothstep(0.02, 0.28, (px + 0.12).abs() + (py - 0.1).abs());
            (shell * notch).clamp(0.0, 1.0)
        }
        FoliageSprayVariant::BrokenSparse => {
            let sparse = 1.0 - smoothstep(0.28, 0.92, radial);
            let cut = smoothstep(0.0, 0.24, px * 0.55 + py * 0.72 + 0.08);
            (sparse * cut).clamp(0.0, 1.0)
        }
    }
}

pub fn generate_surface_atlas() -> Vec<u8> {
    let mut data = vec![0u8; (ATLAS_SIZE * ATLAS_SIZE * FOLIAGE_VARIANT_COUNT * 4) as usize];

    for layer in 0..FOLIAGE_VARIANT_COUNT {
        let variant = match layer {
            0 => FoliageSprayVariant::SideSpray,
            1 => FoliageSprayVariant::TerminalTuft,
            2 => FoliageSprayVariant::DroopingSecondary,
            3 => FoliageSprayVariant::SparseSilhouette,
            _ => FoliageSprayVariant::BrokenSparse,
        };

        for y in 0..ATLAS_SIZE {
            for x in 0..ATLAS_SIZE {
                let u = (x as f32 + 0.5) / ATLAS_SIZE as f32;
                let v = (y as f32 + 0.5) / ATLAS_SIZE as f32;
                let px = (u - 0.5) * 2.0;
                let py = 1.0 - v * 2.0;

                let alpha = shape_alpha(variant, px, py);
                let ax = shape_alpha(variant, px + 1.0 / ATLAS_SIZE as f32, py)
                    - shape_alpha(variant, px - 1.0 / ATLAS_SIZE as f32, py);
                let ay = shape_alpha(variant, px, py + 1.0 / ATLAS_SIZE as f32)
                    - shape_alpha(variant, px, py - 1.0 / ATLAS_SIZE as f32);
                let nx = (-ax * 2.4).clamp(-1.0, 1.0) * 0.5 + 0.5;
                let ny = (-ay * 2.4).clamp(-1.0, 1.0) * 0.5 + 0.5;
                let thickness = (alpha
                    * (0.72 + (1.0 - px.abs()) * 0.18 + (1.0 - py.abs()) * 0.10))
                    .clamp(0.0, 1.0);

                let base = (((layer * ATLAS_SIZE + y) * ATLAS_SIZE + x) * 4) as usize;
                data[base] = (alpha * 255.0) as u8;
                data[base + 1] = (nx * 255.0) as u8;
                data[base + 2] = (ny * 255.0) as u8;
                data[base + 3] = (thickness * 255.0) as u8;
            }
        }
    }

    data
}

pub fn create_surface_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> (wgpu::Texture, wgpu::TextureView) {
    let data = generate_surface_atlas();
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("foliage-surface-atlas"),
        size: wgpu::Extent3d {
            width: ATLAS_SIZE,
            height: ATLAS_SIZE,
            depth_or_array_layers: FOLIAGE_VARIANT_COUNT,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
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
            bytes_per_row: Some(ATLAS_SIZE * 4),
            rows_per_image: Some(ATLAS_SIZE),
        },
        wgpu::Extent3d {
            width: ATLAS_SIZE,
            height: ATLAS_SIZE,
            depth_or_array_layers: FOLIAGE_VARIANT_COUNT,
        },
    );

    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("foliage-surface-atlas-view"),
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    });
    (texture, view)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atlas_has_expected_size() {
        assert_eq!(
            generate_surface_atlas().len(),
            (ATLAS_SIZE * ATLAS_SIZE * FOLIAGE_VARIANT_COUNT * 4) as usize
        );
    }
}
