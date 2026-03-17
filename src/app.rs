use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

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

use crate::art_direction::ArtDirection;
use crate::camera::{CameraMode, CameraState, ThirdPersonConfig, ThirdPersonState};

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
    warmup_frames: Option<u32>,
    preset: LookDevPreset,
    camera_overrides: CameraOverrides,
    seed: Option<u64>,
    style: Option<String>,
    style_blend: Option<f32>,
    growth_sim: bool,
    third_person: bool,
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
    exit_after_capture: bool,
    warmup_remaining: u32,
    debug_overlay: crate::pipeline::DebugOverlay,
    art_direction: ArtDirection,
    humanoid_animator: Option<crate::humanoid_animator::HumanoidAnimator>,
    camera_mode: CameraMode,
    character_pos: glam::Vec3,
    third_person: ThirdPersonState,
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

        // Build scene based on mode.
        let (bark_params, source_scene) = if options.growth_sim {
            let mut growth_params = crate::growth::GrowthParams::default();
            if let Some(seed) = options.seed {
                growth_params.seed = seed;
            }
            let bp = crate::material::procedural::BarkParams::from_growth(&growth_params);
            let scene = crate::source_scene::SourceSceneBuilder::redwood_soundstage_growth(
                &stage_config.layout,
                options.seed,
            );
            (bp, scene)
        } else {
            let mut p = crate::subjects::redwood_growth::RedwoodParams::default();
            if let Some(seed) = options.seed {
                p.seed = seed;
            }
            let bp = crate::material::procedural::BarkParams::from_redwood(&p);
            let scene = crate::source_scene::SourceSceneBuilder::redwood_soundstage(
                &stage_config.layout,
                options.seed,
            );
            (bp, scene)
        };

        renderer.write_bark_params(&gpu.queue, &bark_params);

        let gpu_access = crate::growth::GpuAccess {
            device: &gpu.device,
            queue: &gpu.queue,
        };
        let compiled_scene = SceneCompiler::new().compile_with_gpu(&source_scene, Some(gpu_access));
        let residency = SceneResidencyManager::new(&compiled_scene.runtime_scene);
        let runtime_scene_gpu =
            RuntimeSceneGpu::upload(&gpu, &compiled_scene, &residency, &bark_params);
        renderer.resize(&gpu, &runtime_scene_gpu);

        info!(
            "compiled scene '{}' => {} prototypes, {} instances, {} chunks, {} meshlets",
            compiled_scene.runtime_scene.label,
            compiled_scene.runtime_scene.prototypes.len(),
            compiled_scene.runtime_scene.instances.len(),
            compiled_scene.runtime_scene.chunks.len(),
            runtime_scene_gpu.dag.meshlets.len(),
        );

        // Initialize art direction from CLI --style flag
        let art_direction = if let Some(ref style_name) = options.style {
            if let Some(snapshot) = ArtDirection::find_style(style_name) {
                let blend = options.style_blend.unwrap_or(1.0);
                let blended = crate::art_direction::blend_snapshots(
                    &crate::art_direction::IDENTITY,
                    snapshot,
                    blend,
                );
                info!("art direction: style '{}' (blend={:.2})", style_name, blend);
                ArtDirection::from_snapshot(&blended)
            } else {
                error!("unknown style '{}', using identity", style_name);
                ArtDirection::new()
            }
        } else {
            ArtDirection::new()
        };

        // Initialize humanoid animator if animated regions exist
        info!(
            "animated regions: {}, animated shadow indices: {}",
            runtime_scene_gpu.animated_regions.len(),
            runtime_scene_gpu.animated_shadow_indices.len(),
        );
        for (label, region) in &runtime_scene_gpu.animated_regions {
            info!(
                "  animated region '{}': byte_offset={}, vertex_count={}",
                label, region.vertex_byte_offset, region.vertex_count
            );
        }
        let humanoid_animator =
            if let Some((_, region)) = runtime_scene_gpu.animated_regions.first() {
                let shadow_idx = runtime_scene_gpu
                    .animated_shadow_indices
                    .first()
                    .map(|(_, idx)| *idx);
                let params = crate::subjects::humanoid::HumanoidParams::default();
                info!(
                    "humanoid animator: region byte_offset={}, vertex_count={}, shadow_idx={:?}",
                    region.vertex_byte_offset, region.vertex_count, shadow_idx
                );
                Some(crate::humanoid_animator::HumanoidAnimator::new(
                    &params,
                    crate::runtime_scene::AnimatedRegion {
                        vertex_byte_offset: region.vertex_byte_offset,
                        vertex_count: region.vertex_count,
                    },
                    shadow_idx,
                ))
            } else {
                info!("no animated regions found — humanoid animator disabled");
                None
            };

        let now = Instant::now();
        info!("wrela_prism initialized — {}x{}", size.width, size.height);

        // Warmup frames: 48 default for captures gives ~99.5% cloud temporal convergence
        // (0.9^48 ≈ 0.005 remaining from initial frame). Override with --warmup-frames.
        let warmup_remaining =
            options
                .warmup_frames
                .unwrap_or(if options.capture_on_launch.is_some() {
                    48
                } else {
                    0
                });

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
            exit_after_capture: options.capture_on_launch.is_some(),
            warmup_remaining,
            debug_overlay: crate::pipeline::DebugOverlay::default(),
            art_direction,
            humanoid_animator,
            camera_mode: if options.third_person {
                CameraMode::ThirdPerson
            } else {
                CameraMode::FreeCam
            },
            character_pos: glam::Vec3::new(8.0, 0.0, 0.0),
            third_person: ThirdPersonState::new(ThirdPersonConfig::default()),
        })
    }

    fn toggle_debug_overlay(&mut self, target: crate::pipeline::DebugOverlay) {
        self.debug_overlay = if self.debug_overlay == target {
            crate::pipeline::DebugOverlay::None
        } else {
            target
        };
        info!("debug overlay: {:?}", self.debug_overlay);
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

        match self.camera_mode {
            CameraMode::FreeCam => {
                if strafe != 0.0 || forward != 0.0 || vertical != 0.0 {
                    self.camera.move_on_plane(
                        glam::Vec2::new(strafe, forward),
                        vertical,
                        dt,
                        sprint,
                    );
                }
            }
            CameraMode::ThirdPerson => {
                // WASD moves character on Y=0 plane
                let planar = glam::Vec2::new(strafe, forward);
                self.character_pos = self.third_person.move_character(
                    self.character_pos,
                    planar,
                    dt,
                    sprint,
                    &self.camera.navigation,
                );

                // Compute camera position from orbit
                let (cam_pos, yaw, pitch) =
                    self.third_person.compute_camera(self.character_pos);
                self.camera.position = cam_pos;
                self.camera.yaw = yaw;
                self.camera.pitch = pitch;
            }
        }
    }

    fn redraw(&mut self, event_loop: &ActiveEventLoop) {
        let dt = self.frame_delta_secs();
        self.apply_camera_input(dt);

        // Update character visibility and model matrix for the forward pass
        let is_third_person = self.camera_mode == CameraMode::ThirdPerson;
        self.renderer.set_character_visible(is_third_person);
        if is_third_person {
            let model = glam::Mat4::from_translation(self.character_pos);
            self.renderer.set_character_model(model);
        }

        // Tick humanoid animation before render
        if let Some(ref mut animator) = self.humanoid_animator {
            animator.tick(dt, &self.gpu.queue, &self.runtime_scene_gpu);
        }

        // Update art direction (advance transitions, upload uniforms to GPU)
        self.art_direction.update(
            Duration::from_secs_f32(dt),
            &self.gpu.queue,
            self.renderer.art_direction_buffer(),
            self.renderer.art_direction_palette_buffer(),
        );
        self.renderer
            .set_outline_skip(self.art_direction.should_skip_outline());
        let post = self.art_direction.post_params();
        self.renderer.set_art_direction_post(
            post.bloom_tint,
            post.bloom_softness,
            post.color_grade,
        );
        self.renderer
            .set_lod_bias(self.art_direction.quality_budget.lod_bias);

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
                // Warmup frames: render normally but defer capture
                if self.warmup_remaining > 0 {
                    self.warmup_remaining -= 1;
                    if self.warmup_remaining > 0 {
                        return; // keep rendering warmup frames
                    }
                }
                if let Some(path) = self.pending_capture_path.take() {
                    match self.renderer.capture_frame(&self.gpu, elapsed) {
                        Ok(captured) => {
                            if let Err(e) = write_png(&path, &captured) {
                                error!("capture failed: {e:#}");
                            } else {
                                info!(
                                    "captured {}x{} → {}",
                                    captured.width,
                                    captured.height,
                                    path.display()
                                );
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
                            KeyCode::Tab => {
                                runtime.camera_mode = match runtime.camera_mode {
                                    CameraMode::FreeCam => {
                                        info!("camera mode: third-person");
                                        CameraMode::ThirdPerson
                                    }
                                    CameraMode::ThirdPerson => {
                                        info!("camera mode: free-cam");
                                        CameraMode::FreeCam
                                    }
                                };
                            }
                            KeyCode::F1 => runtime
                                .toggle_debug_overlay(crate::pipeline::DebugOverlay::StructureOnly),
                            KeyCode::F2 => runtime
                                .toggle_debug_overlay(crate::pipeline::DebugOverlay::CanopyOnly),
                            KeyCode::F3 => runtime
                                .toggle_debug_overlay(crate::pipeline::DebugOverlay::WindMagnitude),
                            KeyCode::F4 => runtime
                                .toggle_debug_overlay(crate::pipeline::DebugOverlay::LodHeatmap),
                            KeyCode::F5 => {
                                let styles = ArtDirection::named_styles();
                                runtime.art_direction.named_style_index =
                                    (runtime.art_direction.named_style_index + 1) % styles.len();
                                let (name, snapshot) =
                                    &styles[runtime.art_direction.named_style_index];
                                runtime.art_direction.transition_to(
                                    snapshot,
                                    Duration::from_millis(500),
                                    crate::art_direction::EaseCurve::EaseInOut,
                                );
                                info!("art direction: transitioning to '{}'", name);
                            }
                            KeyCode::F6 => {
                                let axes = runtime.art_direction.current_axes();
                                info!(
                                    "art direction axes: soft={:.2} exag={:.2} shadow={:.2} warm={:.2} atmo={:.2} detail={:.2} outline={:.2} palette={:.2}",
                                    axes.softness, axes.exaggeration, axes.shadow_graphicness,
                                    axes.palette_warmth, axes.atmospheric_depth, axes.surface_detail,
                                    axes.outline_presence, axes.color_discipline
                                );
                            }
                            KeyCode::BracketLeft | KeyCode::BracketRight => {
                                let styles = ArtDirection::named_styles();
                                let idx = runtime.art_direction.named_style_index;
                                if idx > 0 {
                                    let (name, target) = &styles[idx];
                                    let delta = if code == KeyCode::BracketRight {
                                        0.1
                                    } else {
                                        -0.1
                                    };
                                    let axes = runtime.art_direction.current_axes();
                                    // Use outline_presence as a proxy for overall blend level
                                    let current_blend = axes.outline_presence
                                        / target.axes.outline_presence.max(0.01);
                                    let new_blend = (current_blend + delta).clamp(0.0, 1.0);
                                    let blended = crate::art_direction::blend_snapshots(
                                        &crate::art_direction::IDENTITY,
                                        target,
                                        new_blend,
                                    );
                                    runtime.art_direction.transition_to(
                                        &blended,
                                        Duration::from_millis(100),
                                        crate::art_direction::EaseCurve::Linear,
                                    );
                                    info!("art direction: blend '{}' = {:.1}", name, new_blend);
                                }
                            }
                            KeyCode::Digit0 => {
                                runtime.art_direction.transition_to(
                                    &crate::art_direction::IDENTITY,
                                    Duration::from_millis(300),
                                    crate::art_direction::EaseCurve::EaseOut,
                                );
                                runtime.art_direction.named_style_index = 0;
                                info!("art direction: resetting to identity");
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
            let delta_vec = glam::Vec2::new(delta.0 as f32, delta.1 as f32);
            match runtime.camera_mode {
                CameraMode::FreeCam => {
                    runtime.camera.apply_look_delta(delta_vec);
                }
                CameraMode::ThirdPerson => {
                    runtime.third_person.apply_look_delta(delta_vec);
                }
            }
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
                "--style" => {
                    let Some(name) = args.next() else {
                        bail!("--style requires a style name (identity, demon_slayer)");
                    };
                    options.style = Some(name);
                }
                "--warmup-frames" => {
                    let Some(value) = args.next() else {
                        bail!("--warmup-frames requires a u32 value");
                    };
                    options.warmup_frames = Some(value.parse()?);
                }
                "--style-blend" => {
                    let Some(value) = args.next() else {
                        bail!("--style-blend requires a float (0.0-1.0)");
                    };
                    options.style_blend = Some(value.parse()?);
                }
                "--growth-sim" => {
                    options.growth_sim = true;
                }
                "--third-person" => {
                    options.third_person = true;
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
            warmup_frames: None,
            preset: LookDevPreset::hero_default(),
            camera_overrides: CameraOverrides::default(),
            seed: None,
            style: None,
            style_blend: None,
            growth_sim: false,
            third_person: false,
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
