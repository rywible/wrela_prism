use crate::gpu::GpuContext;
use crate::material::procedural::BarkParams;
use crate::pipeline::visbuf_pipeline::VisbufPipeline;
use crate::runtime_scene::RuntimeSceneGpu;

use crate::camera::CameraState;
use crate::pipeline::DebugOverlay;
use crate::scene::SceneSettings;

#[derive(Clone, Copy)]
pub struct FrameInputs<'a> {
    pub gpu: &'a GpuContext,
    pub camera: &'a CameraState,
    pub settings: &'a SceneSettings,
    pub elapsed_secs: f32,
    pub debug_overlay: DebugOverlay,
}

pub struct Renderer {
    pipeline: VisbufPipeline,
}

impl Renderer {
    pub fn new(gpu: &GpuContext) -> Self {
        Self {
            pipeline: VisbufPipeline::new(gpu),
        }
    }

    pub fn resize(&mut self, gpu: &GpuContext, scene: &RuntimeSceneGpu) {
        self.pipeline.resize(gpu);
        self.pipeline
            .sync_scene_resources(&gpu.device, &gpu.queue, scene);
    }

    pub fn configure_shadow(
        &mut self,
        shadow_center: glam::Vec3,
        shadow_base_radius: f32,
        shadow_depth: f32,
    ) {
        self.pipeline.shadow_center = shadow_center;
        self.pipeline.shadow_base_radius = shadow_base_radius;
        self.pipeline.shadow_depth = shadow_depth;
    }

    pub fn write_bark_params(&self, queue: &wgpu::Queue, bark_params: &BarkParams) {
        queue.write_buffer(
            &self.pipeline.bark_uniform_buffer,
            0,
            bytemuck::bytes_of(bark_params),
        );
    }

    pub fn render(
        &mut self,
        scene: &RuntimeSceneGpu,
        frame: &FrameInputs<'_>,
    ) -> std::result::Result<(), wgpu::SurfaceError> {
        self.pipeline.render(
            frame.gpu,
            scene,
            frame.camera,
            frame.settings,
            frame.elapsed_secs,
        )
    }

    pub fn capture_frame(
        &self,
        gpu: &GpuContext,
        elapsed_secs: f32,
    ) -> anyhow::Result<crate::pipeline::CapturedFrame> {
        self.pipeline.capture_frame(gpu, elapsed_secs)
    }

    /// Return a reference to the art direction uniform buffer for CPU-side upload.
    pub fn art_direction_buffer(&self) -> &wgpu::Buffer {
        &self.pipeline.art_direction_buffer
    }

    /// Return a reference to the art direction palette buffer for CPU-side upload.
    pub fn art_direction_palette_buffer(&self) -> &wgpu::Buffer {
        &self.pipeline.art_direction_palette_buffer
    }

    /// Set whether the outline pass should be skipped (when outline_strength is ~0).
    pub fn set_outline_skip(&mut self, skip: bool) {
        self.pipeline.art_direction_outline_skip = skip;
    }

    /// Set LOD bias from art direction quality budget.
    pub fn set_lod_bias(&mut self, bias: f32) {
        self.pipeline.art_direction_lod_bias = bias;
    }

    /// Set art direction post-processing parameters for bloom and tonemap.
    pub fn set_art_direction_post(
        &mut self,
        bloom_tint: [f32; 3],
        bloom_softness: f32,
        color_grade: [f32; 4],
    ) {
        self.pipeline.art_direction_bloom_tint = bloom_tint;
        self.pipeline.art_direction_bloom_softness = bloom_softness;
        self.pipeline.art_direction_color_grade = color_grade;
    }
}
