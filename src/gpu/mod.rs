pub mod upload;

use std::sync::Arc;

use anyhow::{Context, Result};
use tracing::debug;
use winit::window::Window;

// ──────────────────────── GPU Timing ────────────────────────

/// Maximum number of pass timing slots (begin + end = 2 queries per pass).
const MAX_TIMING_PASSES: usize = 16;
const MAX_TIMING_QUERIES: u32 = (MAX_TIMING_PASSES * 2) as u32;

pub struct GpuTimingContext {
    query_set: wgpu::QuerySet,
    resolve_buffer: wgpu::Buffer,
    read_buffer: wgpu::Buffer,
    timestamp_period: f32,
    pass_names: Vec<&'static str>,
    /// Results from the previous frame (one-frame delay avoids GPU stalls).
    pending_read: bool,
}

impl GpuTimingContext {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("gpu-timing-query-set"),
            ty: wgpu::QueryType::Timestamp,
            count: MAX_TIMING_QUERIES,
        });

        let buf_size = (MAX_TIMING_QUERIES as u64) * 8; // u64 per query
        let resolve_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu-timing-resolve"),
            size: buf_size,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let read_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu-timing-read"),
            size: buf_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let timestamp_period = queue.get_timestamp_period();

        Self {
            query_set,
            resolve_buffer,
            read_buffer,
            timestamp_period,
            pass_names: Vec::new(),
            pending_read: false,
        }
    }

    /// Register a pass and return `(begin_index, end_index)` for timestamp_writes.
    pub fn register_pass(&mut self, name: &'static str) -> Option<(u32, u32)> {
        let idx = self.pass_names.len();
        if idx >= MAX_TIMING_PASSES {
            return None;
        }
        self.pass_names.push(name);
        let begin = (idx * 2) as u32;
        let end = begin + 1;
        Some((begin, end))
    }

    /// Build `wgpu::ComputePassTimestampWrites` for a registered pass.
    pub fn compute_timestamp_writes(
        &self,
        begin: u32,
        end: u32,
    ) -> wgpu::ComputePassTimestampWrites<'_> {
        wgpu::ComputePassTimestampWrites {
            query_set: &self.query_set,
            beginning_of_pass_write_index: Some(begin),
            end_of_pass_write_index: Some(end),
        }
    }

    /// Build `wgpu::RenderPassTimestampWrites` for a registered pass.
    pub fn render_timestamp_writes(
        &self,
        begin: u32,
        end: u32,
    ) -> wgpu::RenderPassTimestampWrites<'_> {
        wgpu::RenderPassTimestampWrites {
            query_set: &self.query_set,
            beginning_of_pass_write_index: Some(begin),
            end_of_pass_write_index: Some(end),
        }
    }

    /// Resolve queries and copy to read buffer. Call after encoding all passes.
    pub fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        let count = (self.pass_names.len() * 2) as u32;
        if count == 0 {
            return;
        }
        encoder.resolve_query_set(&self.query_set, 0..count, &self.resolve_buffer, 0);
        encoder.copy_buffer_to_buffer(
            &self.resolve_buffer,
            0,
            &self.read_buffer,
            0,
            (count as u64) * 8,
        );
    }

    /// Begin reading the previous frame's results (non-blocking map).
    pub fn begin_readback(&mut self) {
        let slice = self.read_buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.pending_read = true;
    }

    /// Poll and log results if ready. Call at start of next frame.
    pub fn poll_and_log(&mut self, device: &wgpu::Device) {
        if !self.pending_read {
            return;
        }
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        {
            let data = self.read_buffer.slice(..).get_mapped_range();
            let timestamps: &[u64] = bytemuck::cast_slice(&data[..self.pass_names.len() * 2 * 8]);
            for (i, name) in self.pass_names.iter().enumerate() {
                let begin = timestamps[i * 2];
                let end = timestamps[i * 2 + 1];
                if end > begin {
                    let ns = (end - begin) as f64 * self.timestamp_period as f64;
                    debug!(target: "wrela_prism::timing", "{}: {:.2}ms", name, ns / 1_000_000.0);
                }
            }
        }
        self.read_buffer.unmap();
        self.pending_read = false;
        self.pass_names.clear();
    }
}

// ──────────────────────── GPU Context ────────────────────────

pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface: wgpu::Surface<'static>,
    pub config: wgpu::SurfaceConfiguration,
    pub surface_format: wgpu::TextureFormat,
    pub timestamp_supported: bool,
}

impl GpuContext {
    pub async fn new(window: Arc<Window>) -> Result<Self> {
        let size = window.inner_size();
        let instance = wgpu::Instance::default();
        let surface = instance
            .create_surface(window)
            .context("failed to create wgpu surface")?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .context("failed to find a suitable graphics adapter")?;

        // Try with timestamp query first, fall back to empty features
        let timestamp_supported = adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        let required_features = if timestamp_supported {
            wgpu::Features::TIMESTAMP_QUERY
        } else {
            wgpu::Features::empty()
        };

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("wrela-prism-device"),
                required_features,
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                experimental_features: wgpu::ExperimentalFeatures::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .context("failed to create wgpu device")?;

        let capabilities = surface.get_capabilities(&adapter);
        let surface_format = capabilities
            .formats
            .iter()
            .copied()
            .find(wgpu::TextureFormat::is_srgb)
            .or_else(|| capabilities.formats.first().copied())
            .context("surface reported no texture formats")?;
        let present_mode = capabilities
            .present_modes
            .iter()
            .copied()
            .find(|mode| *mode == wgpu::PresentMode::Fifo)
            .unwrap_or(wgpu::PresentMode::AutoVsync);
        let alpha_mode = capabilities
            .alpha_modes
            .first()
            .copied()
            .unwrap_or(wgpu::CompositeAlphaMode::Auto);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_DST,
            format: surface_format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode,
            desired_maximum_frame_latency: 2,
            alpha_mode,
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        Ok(Self {
            device,
            queue,
            surface,
            config,
            surface_format,
            timestamp_supported,
        })
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
    }

    pub fn width(&self) -> u32 {
        self.config.width
    }

    pub fn height(&self) -> u32 {
        self.config.height
    }
}
