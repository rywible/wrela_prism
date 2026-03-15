use crate::camera::CameraState;
use crate::gpu::GpuContext;
use crate::material::procedural::BarkParams;
use crate::pipeline::{visbuf_pipeline::VisbufPipeline, DebugOverlay};
use crate::runtime_scene::RuntimeSceneGpu;
use crate::scene::SceneSettings;

#[derive(Clone, Copy)]
pub struct FrameInputs<'a> {
    pub gpu: &'a GpuContext,
    pub camera: &'a CameraState,
    pub settings: &'a SceneSettings,
    pub elapsed_secs: f32,
    pub debug_overlay: DebugOverlay,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct FrameGraphContext {
    pub surface_size: (u32, u32),
    pub debug_overlay: DebugOverlay,
}

#[derive(Clone, Debug, Default)]
pub struct FrameVisibilityState {
    pub visible_chunks: Vec<crate::runtime_scene::ChunkId>,
    pub visible_instance_count: usize,
}

pub struct Renderer {
    pipeline: VisbufPipeline,
    frame_graph: FrameGraphContext,
    visibility_state: FrameVisibilityState,
}

impl Renderer {
    pub fn new(gpu: &GpuContext) -> Self {
        Self {
            pipeline: VisbufPipeline::new(gpu),
            frame_graph: FrameGraphContext {
                surface_size: (gpu.width(), gpu.height()),
                debug_overlay: DebugOverlay::None,
            },
            visibility_state: FrameVisibilityState::default(),
        }
    }

    pub fn resize(&mut self, gpu: &GpuContext, scene: &RuntimeSceneGpu) {
        self.frame_graph.surface_size = (gpu.width(), gpu.height());
        self.pipeline.resize(gpu);
        self.pipeline.sync_scene_resources(&gpu.device, scene);
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
        self.frame_graph.surface_size = (frame.gpu.width(), frame.gpu.height());
        self.frame_graph.debug_overlay = frame.debug_overlay;
        self.visibility_state.visible_chunks = scene.runtime_scene.spatial_index.visible_chunks(
            frame.camera,
            frame.gpu.width(),
            frame.gpu.height(),
        );
        self.visibility_state.visible_instance_count = scene
            .runtime_scene
            .instances
            .iter()
            .filter(|instance| self.visibility_state.visible_chunks.contains(&instance.chunk_id))
            .count();

        self.pipeline.sync_scene_resources(&frame.gpu.device, scene);
        self.pipeline.render(
            frame.gpu,
            scene,
            frame.camera,
            frame.settings,
            frame.elapsed_secs,
        )
    }

    pub fn visibility_state(&self) -> &FrameVisibilityState {
        &self.visibility_state
    }

    pub fn capture_frame(
        &self,
        gpu: &GpuContext,
        elapsed_secs: f32,
    ) -> anyhow::Result<crate::pipeline::CapturedFrame> {
        self.pipeline.capture_frame(gpu, elapsed_secs)
    }
}
