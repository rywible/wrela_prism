use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use tracing::{error, info};
use tracing_subscriber::EnvFilter;
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, ElementState, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window, WindowAttributes, WindowId},
};

use crate::camera::CameraState;
use crate::compiler::{CompiledScene, SceneCompiler};
use crate::gpu::GpuContext;
use crate::renderer::{FrameInputs, Renderer};
use crate::runtime_scene::{RuntimeSceneGpu, SceneResidencyManager};
use crate::scene::SceneSettings;
use crate::soundstage::{soundstage_for_preset, LookDevPreset};
use crate::source_scene::SourceScene;

pub struct App {
    runtime: Option<RuntimeState>,
    options: AppOptions,
}

#[derive(Debug)]
struct AppOptions {
    capture_on_launch: Option<PathBuf>,
    capture_size: Option<(u32, u32)>,
    preset: LookDevPreset,
    camera_overrides: CameraOverrides,
    seed: Option<u64>,
}

#[derive(Clone, Copy, Debug, Default)]
struct CameraOverrides {
    position: Option<glam::Vec3>,
    yaw_radians: Option<f32>,
    pitch_radians: Option<f32>,
}

struct RuntimeState {
    window: Arc<Window>,
    gpu: GpuContext,
    renderer: Renderer,
    _source_scene: SourceScene,
    _compiled_scene: CompiledScene,
    _residency: SceneResidencyManager,
    runtime_scene_gpu: RuntimeSceneGpu,
    camera: CameraState,
    scene_settings: SceneSettings,
    cursor_captured: bool,
    pressed_keys: HashSet<KeyCode>,
    start_time: Instant,
    last_frame_instant: Instant,
    pending_capture_path: Option<PathBuf>,
    _pending_capture_size: Option<(u32, u32)>,
    exit_after_capture: bool,
    debug_overlay: crate::pipeline::DebugOverlay,
}

impl RuntimeState {
    fn new(event_loop: &ActiveEventLoop, options: &AppOptions) -> Result<Self> {
        let window = Arc::new(
            event_loop.create_window(
                WindowAttributes::default()
                    .with_title("Wrela Prism")
                    .with_inner_size(winit::dpi::PhysicalSize::new(1440, 960)),
            )?,
        );

        let size = window.inner_size();
        let stage_config = soundstage_for_preset(options.preset);
        let mut camera = stage_config.camera(size.width as f32 / size.height.max(1) as f32);
        camera.set_override(
            options.camera_overrides.position,
            options.camera_overrides.yaw_radians,
            options.camera_overrides.pitch_radians,
        );

        let gpu = pollster::block_on(GpuContext::new(window.clone()))?;
        let mut renderer = Renderer::new(&gpu);
        renderer.configure_shadow(
            stage_config.layout.shadow_center,
            stage_config.layout.shadow_focus_radius,
            stage_config.layout.shadow_depth,
        );

        // Build redwood tree parameters
        let redwood_params = {
            let mut p = crate::subjects::redwood_growth::RedwoodParams::default();
            if let Some(seed) = options.seed {
                p.seed = seed;
            }
            p
        };

        let bark_params = crate::material::procedural::BarkParams::from_redwood(&redwood_params);
        renderer.write_bark_params(&gpu.queue, &bark_params);

        let source_scene =
            crate::source_scene::SourceSceneBuilder::redwood_soundstage(&stage_config.layout, options.seed);
        let compiled_scene = SceneCompiler::new().compile(&source_scene);
        let residency = SceneResidencyManager::new(&compiled_scene.runtime_scene);
        let runtime_scene_gpu = RuntimeSceneGpu::upload(&gpu, &compiled_scene, &residency);
        renderer.resize(&gpu, &runtime_scene_gpu);

        info!(
            "compiled scene '{}' => {} prototypes, {} instances, {} chunks, {} meshlets",
            compiled_scene.runtime_scene.label,
            compiled_scene.runtime_scene.prototypes.len(),
            compiled_scene.runtime_scene.instances.len(),
            compiled_scene.runtime_scene.chunks.len(),
            runtime_scene_gpu.dag.meshlets.len(),
        );

        let now = Instant::now();
        info!("wrela_prism initialized — {}x{}", size.width, size.height);

        Ok(Self {
            window,
            gpu,
            renderer,
            _source_scene: source_scene,
            _compiled_scene: compiled_scene,
            _residency: residency,
            runtime_scene_gpu,
            camera,
            scene_settings: stage_config.scene_settings,
            cursor_captured: false,
            pressed_keys: HashSet::new(),
            start_time: now,
            last_frame_instant: now,
            pending_capture_path: options.capture_on_launch.clone(),
            _pending_capture_size: options.capture_size,
            exit_after_capture: options.capture_on_launch.is_some(),
            debug_overlay: crate::pipeline::DebugOverlay::default(),
        })
    }

    fn resize(&mut self, size: winit::dpi::PhysicalSize<u32>) {
        self.gpu.resize(size.width, size.height);
        self.camera.set_aspect(size.width, size.height);
        self.renderer.resize(&self.gpu, &self.runtime_scene_gpu);
    }

    fn elapsed_secs(&self) -> f32 {
        self.start_time.elapsed().as_secs_f32()
    }

    fn capture_cursor(&mut self) {
        if self.cursor_captured {
            return;
        }
        if let Err(err) = self
            .window
            .set_cursor_grab(CursorGrabMode::Locked)
            .or_else(|_| self.window.set_cursor_grab(CursorGrabMode::Confined))
        {
            error!("failed to grab cursor: {err}");
            return;
        }
        self.window.set_cursor_visible(false);
        self.cursor_captured = true;
    }

    fn release_cursor(&mut self) {
        if !self.cursor_captured {
            return;
        }
        if let Err(err) = self.window.set_cursor_grab(CursorGrabMode::None) {
            error!("failed to release cursor: {err}");
        }
        self.window.set_cursor_visible(true);
        self.cursor_captured = false;
    }

    fn update_pressed_key(&mut self, key: KeyCode, state: ElementState) {
        match state {
            ElementState::Pressed => {
                self.pressed_keys.insert(key);
            }
            ElementState::Released => {
                self.pressed_keys.remove(&key);
            }
        }
    }

    fn frame_delta_secs(&mut self) -> f32 {
        let now = Instant::now();
        let dt = (now - self.last_frame_instant).as_secs_f32();
        self.last_frame_instant = now;
        dt.clamp(1.0 / 240.0, 0.25)
    }

    fn apply_camera_input(&mut self, dt: f32) {
        let axis = |positive: KeyCode, negative: KeyCode, keys: &HashSet<KeyCode>| -> f32 {
            let pos = keys.contains(&positive) as i32;
            let neg = keys.contains(&negative) as i32;
            (pos - neg) as f32
        };

        let strafe = axis(KeyCode::KeyD, KeyCode::KeyA, &self.pressed_keys);
        let forward = axis(KeyCode::KeyW, KeyCode::KeyS, &self.pressed_keys);
        let vertical = axis(KeyCode::KeyE, KeyCode::KeyQ, &self.pressed_keys);
        let sprint = self.pressed_keys.contains(&KeyCode::ShiftLeft)
            || self.pressed_keys.contains(&KeyCode::ShiftRight);

        if strafe != 0.0 || forward != 0.0 || vertical != 0.0 {
            self.camera
                .move_on_plane(glam::Vec2::new(strafe, forward), vertical, dt, sprint);
        }
    }

    fn redraw(&mut self, event_loop: &ActiveEventLoop) {
        let dt = self.frame_delta_secs();
        self.apply_camera_input(dt);

        let elapsed = self.elapsed_secs();
        let frame_inputs = FrameInputs {
            gpu: &self.gpu,
            camera: &self.camera,
            settings: &self.scene_settings,
            elapsed_secs: elapsed,
            debug_overlay: self.debug_overlay,
        };
        match self.renderer.render(&self.runtime_scene_gpu, &frame_inputs) {
            Ok(()) => {
                if let Some(path) = self.pending_capture_path.take() {
                    match self.renderer.capture_frame(&self.gpu, elapsed) {
                        Ok(captured) => {
                            if let Err(e) = write_png(&path, &captured) {
                                error!("capture failed: {e:#}");
                            } else {
                                info!("captured {}x{} → {}", captured.width, captured.height, path.display());
                            }
                        }
                        Err(e) => error!("capture failed: {e:#}"),
                    }
                    if self.exit_after_capture {
                        event_loop.exit();
                    }
                }
            }
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                self.resize(self.window.inner_size());
            }
            Err(wgpu::SurfaceError::OutOfMemory) => {
                error!("wgpu out of memory");
                event_loop.exit();
            }
            Err(wgpu::SurfaceError::Timeout) => {}
            Err(wgpu::SurfaceError::Other) => {}
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.runtime.is_none() {
            match RuntimeState::new(event_loop, &self.options) {
                Ok(runtime) => self.runtime = Some(runtime),
                Err(e) => {
                    error!("failed to initialize: {e:#}");
                    event_loop.exit();
                }
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(runtime) = self.runtime.as_mut() else {
            return;
        };
        if runtime.window.id() != window_id {
            return;
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => runtime.resize(size),
            WindowEvent::ScaleFactorChanged { .. } => runtime.resize(runtime.window.inner_size()),
            WindowEvent::RedrawRequested => runtime.redraw(event_loop),
            WindowEvent::Focused(false) => {
                runtime.release_cursor();
                runtime.pressed_keys.clear();
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    runtime.update_pressed_key(code, event.state);
                    if event.state == ElementState::Pressed {
                        match code {
                            KeyCode::Escape => runtime.release_cursor(),
                            KeyCode::F1 => {
                                runtime.debug_overlay = if runtime.debug_overlay
                                    == crate::pipeline::DebugOverlay::StructureOnly
                                {
                                    crate::pipeline::DebugOverlay::None
                                } else {
                                    crate::pipeline::DebugOverlay::StructureOnly
                                };
                                info!("debug overlay: {:?}", runtime.debug_overlay);
                            }
                            KeyCode::F2 => {
                                runtime.debug_overlay = if runtime.debug_overlay
                                    == crate::pipeline::DebugOverlay::CanopyOnly
                                {
                                    crate::pipeline::DebugOverlay::None
                                } else {
                                    crate::pipeline::DebugOverlay::CanopyOnly
                                };
                                info!("debug overlay: {:?}", runtime.debug_overlay);
                            }
                            KeyCode::F3 => {
                                runtime.debug_overlay = if runtime.debug_overlay
                                    == crate::pipeline::DebugOverlay::WindMagnitude
                                {
                                    crate::pipeline::DebugOverlay::None
                                } else {
                                    crate::pipeline::DebugOverlay::WindMagnitude
                                };
                                info!("debug overlay: {:?}", runtime.debug_overlay);
                            }
                            KeyCode::F4 => {
                                runtime.debug_overlay = if runtime.debug_overlay
                                    == crate::pipeline::DebugOverlay::LodHeatmap
                                {
                                    crate::pipeline::DebugOverlay::None
                                } else {
                                    crate::pipeline::DebugOverlay::LodHeatmap
                                };
                                info!("debug overlay: {:?}", runtime.debug_overlay);
                            }
                            _ => {}
                        }
                    }
                }
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => {
                if state == ElementState::Pressed {
                    runtime.capture_cursor();
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(position) => position.y as f32 / 32.0,
                };
                runtime.camera.dolly(scroll);
            }
            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
        let Some(runtime) = self.runtime.as_mut() else {
            return;
        };
        if !runtime.cursor_captured {
            return;
        }

        if let DeviceEvent::MouseMotion { delta } = event {
            runtime
                .camera
                .apply_look_delta(glam::Vec2::new(delta.0 as f32, delta.1 as f32));
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(runtime) = self.runtime.as_ref() {
            runtime.window.request_redraw();
        }
    }
}

pub fn run() -> Result<()> {
    let options = AppOptions::from_args()?;
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("wrela_prism=info,wgpu=warn")),
        )
        .init();

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App {
        runtime: None,
        options,
    };
    event_loop.run_app(&mut app)?;
    Ok(())
}

impl AppOptions {
    fn from_args() -> Result<Self> {
        Self::from_iter(std::env::args().skip(1))
    }

    fn from_iter<I>(args: I) -> Result<Self>
    where
        I: IntoIterator,
        I::Item: Into<String>,
    {
        let mut args = args.into_iter().map(Into::into);
        let mut options = Self::default();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--capture" => {
                    let Some(path) = args.next() else {
                        bail!("--capture requires a file path");
                    };
                    options.capture_on_launch = Some(PathBuf::from(path));
                }
                "--preset" => {
                    let Some(name) = args.next() else {
                        bail!(
                            "--preset requires one of: hero, low_angle, silhouette, neutral_debug"
                        );
                    };
                    options.preset = LookDevPreset::from_name(&name).ok_or_else(|| {
                        anyhow::anyhow!(
                            "invalid preset '{name}', expected one of: hero, low_angle, silhouette, neutral_debug"
                        )
                    })?;
                }
                "--seed" => {
                    let Some(value) = args.next() else {
                        bail!("--seed requires a u64 value");
                    };
                    options.seed = Some(value.parse()?);
                }
                "--capture-size" => {
                    let Some(size) = args.next() else {
                        bail!("--capture-size requires WIDTHxHEIGHT");
                    };
                    let Some((w, h)) = size.split_once('x') else {
                        bail!("invalid capture size '{size}', expected WIDTHxHEIGHT");
                    };
                    let w: u32 = w.parse()?;
                    let h: u32 = h.parse()?;
                    options.capture_size = Some((w, h));
                }
                "--camera-yaw" => {
                    let Some(value) = args.next() else {
                        bail!("--camera-yaw requires degrees");
                    };
                    let degrees: f32 = value.parse()?;
                    options.camera_overrides.yaw_radians = Some(degrees.to_radians());
                }
                "--camera-pitch" => {
                    let Some(value) = args.next() else {
                        bail!("--camera-pitch requires degrees");
                    };
                    let degrees: f32 = value.parse()?;
                    options.camera_overrides.pitch_radians = Some(degrees.to_radians());
                }
                "--camera-position" => {
                    let Some(value) = args.next() else {
                        bail!("--camera-position requires x,y,z");
                    };
                    options.camera_overrides.position = Some(parse_camera_position(&value)?);
                }
                other => bail!("unrecognized argument: {other}"),
            }
        }
        Ok(options)
    }
}

fn parse_camera_position(value: &str) -> Result<glam::Vec3> {
    let mut parts = value.split(',');
    let Some(x) = parts.next() else {
        bail!("invalid camera position '{value}', expected x,y,z");
    };
    let Some(y) = parts.next() else {
        bail!("invalid camera position '{value}', expected x,y,z");
    };
    let Some(z) = parts.next() else {
        bail!("invalid camera position '{value}', expected x,y,z");
    };
    if parts.next().is_some() {
        bail!("invalid camera position '{value}', expected x,y,z");
    }
    Ok(glam::Vec3::new(x.parse()?, y.parse()?, z.parse()?))
}

impl Default for AppOptions {
    fn default() -> Self {
        Self {
            capture_on_launch: None,
            capture_size: None,
            preset: LookDevPreset::hero_default(),
            camera_overrides: CameraOverrides::default(),
            seed: None,
        }
    }
}

fn write_png(path: &Path, captured: &crate::pipeline::CapturedFrame) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let image = image::ImageBuffer::<image::Rgba<u8>, Vec<u8>>::from_raw(
        captured.width,
        captured.height,
        captured.rgba.clone(),
    )
    .context("capture buffer dimensions mismatch")?;
    image
        .save(path)
        .with_context(|| format!("failed to save {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn camera_override_parsing_uses_world_position_and_angles() {
        let options = AppOptions::from_iter([
            "--camera-position",
            "12.5,8.0,-4.5",
            "--camera-yaw",
            "90",
            "--camera-pitch",
            "-15",
        ])
        .unwrap();
        assert_eq!(
            options.camera_overrides.position,
            Some(glam::Vec3::new(12.5, 8.0, -4.5))
        );
        assert!(
            (options.camera_overrides.yaw_radians.unwrap() - std::f32::consts::FRAC_PI_2).abs()
                < 0.0001
        );
        assert!(
            (options.camera_overrides.pitch_radians.unwrap() - (-15.0_f32).to_radians()).abs()
                < 0.0001
        );
    }

}
