pub mod bloom_pass;
pub mod cloud_pass;
pub mod cull_pass;
pub mod hw_raster_pass;
pub mod hzb_pass;
pub mod material_pass;
pub mod noise_textures;
pub mod outline_pass;
pub mod shadow_pass;
pub mod sky_lut_pass;
pub mod sky_pass;
pub mod ssao_pass;
pub mod ssgi_pass;
pub mod sun_shaft_pass;
pub mod sw_raster_pass;
pub mod tonemap_pass;
pub mod visbuf_pipeline;

pub const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

use std::sync::mpsc;

/// Debug overlay mode toggled by keyboard.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DebugOverlay {
    #[default]
    None,
    /// F1: Structure only (skip foliage).
    StructureOnly,
    /// F2: Canopy only (skip opaque).
    CanopyOnly,
    /// F3: Wind displacement magnitude (colorized).
    WindMagnitude,
    /// F4: LOD heatmap (clusters colored by density tier).
    LodHeatmap,
}

use crate::gpu::GpuContext;
use anyhow::{Context, Result};

pub(crate) fn storage_entry(
    binding: u32,
    read_only: bool,
    visibility: wgpu::ShaderStages,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

pub struct CapturedFrame {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

pub(crate) fn readback_texture(
    gpu: &GpuContext,
    texture: &wgpu::Texture,
    width: u32,
    height: u32,
) -> Result<CapturedFrame> {
    let unpadded_bytes_per_row = width * 4;
    let padded_bytes_per_row =
        unpadded_bytes_per_row.next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let buffer_size = padded_bytes_per_row as u64 * height as u64;

    let output_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("prism-capture-buffer"),
        size: buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("prism-capture-copy-encoder"),
        });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &output_buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    gpu.queue.submit([encoder.finish()]);

    let slice = output_buffer.slice(..);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    receiver
        .recv()
        .context("capture map_async channel closed")?
        .context("failed to map capture buffer")?;

    let mapped = slice.get_mapped_range();
    let mut rgba = vec![0_u8; (unpadded_bytes_per_row * height) as usize];
    for row in 0..height as usize {
        let src_offset = row * padded_bytes_per_row as usize;
        let dst_offset = row * unpadded_bytes_per_row as usize;
        rgba[dst_offset..dst_offset + unpadded_bytes_per_row as usize]
            .copy_from_slice(&mapped[src_offset..src_offset + unpadded_bytes_per_row as usize]);
    }
    if matches!(
        texture.format(),
        wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
    ) {
        for pixel in rgba.chunks_exact_mut(4) {
            pixel.swap(0, 2);
        }
    }
    drop(mapped);
    output_buffer.unmap();

    Ok(CapturedFrame {
        width,
        height,
        rgba,
    })
}
