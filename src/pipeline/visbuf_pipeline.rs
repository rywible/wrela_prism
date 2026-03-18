use crate::art_direction::{ArtDirectionUniforms, StylePalette};
use crate::camera::CameraState;
use crate::gpu::{GpuContext, GpuTimingContext};
use crate::runtime_scene::RuntimeSceneGpu;
use crate::scene::shadow::{compute_all_cascade_vps, ShadowMap};
use crate::scene::{LightingUniforms, SceneSettings};

use super::area_light_pass::AreaLightPass;
use super::bloom_pass::BloomPass;
use super::cloud_pass::CloudPass;
use super::cull_pass::CullPass;
use super::dag_traverse_pass::DagTraversePass;
use super::dof_pass::DofPass;
use super::forward_character::ForwardCharacterPass;
use super::fxaa_pass::FxaaPass;
use super::gtao_pass::GtaoPass;
use super::hw_raster_pass::{
    extract_frustum_planes, DispatchLists, HwRasterPass, VisbufFrameUniforms, VisibilityBuffer,
};
use super::hzb_pass::HzbPass;
use super::ibl_pass::IblPass;
use super::material_pass::MaterialPass;
use super::motion_blur_pass::MotionBlurPass;
use super::noise_textures::NoiseTextures;
use super::outline_pass::OutlinePass;
use super::shadow_pass::ShadowPass;
use super::sky_lut_pass::SkyLutPass;
use super::sky_pass::SkyPass;
use super::ssgi_pass::SsgiPass;
use super::ssr_pass::SsrPass;
use super::sun_shaft_pass::SunShaftPass;
use super::sw_raster_pass::SwRasterPass;
use super::taa_pass::TaaPass;
use super::tonemap_pass::TonemapPass;
use super::volumetric_fog_pass::VolumetricFogPass;
use super::HDR_FORMAT;

#[derive(Clone, Copy, Default)]
enum HdrStage {
    #[default]
    Raw,
    AfterDof,
    AfterMotionBlur,
}

struct SceneColorTarget {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
}

/// 8-pass Nanite-style visibility buffer pipeline.
pub struct VisbufPipeline {
    // Core passes
    pub hw_raster: HwRasterPass,
    pub cull_pass: CullPass,
    pub dag_traverse: DagTraversePass,
    pub sw_raster: SwRasterPass,
    pub material_pass: MaterialPass,
    pub hzb_pass: HzbPass,

    // Existing passes (kept)
    pub shadow_pass: ShadowPass,
    pub ibl_pass: IblPass,
    pub sky_lut_pass: SkyLutPass,
    pub sky_pass: SkyPass,
    pub cloud_pass: CloudPass,
    pub ssgi_pass: Option<SsgiPass>,
    pub ssr_pass: Option<SsrPass>,
    pub area_light_pass: AreaLightPass,
    pub gtao_pass: Option<GtaoPass>,
    pub sun_shaft_pass: SunShaftPass,
    pub outline_pass: OutlinePass,
    pub bloom_pass: Option<BloomPass>,
    pub tonemap_pass: TonemapPass,

    // GPU resources
    pub shadow_map: ShadowMap,
    pub vis_buffer: VisibilityBuffer,
    pub dispatch_lists: DispatchLists,
    scene_color: SceneColorTarget,

    // Bind groups (cached)
    frame_bg: wgpu::BindGroup,
    mesh_bg: Option<wgpu::BindGroup>,
    cull_bg: Option<wgpu::BindGroup>,
    dag_bg: Option<wgpu::BindGroup>,
    dispatch_bg: Option<wgpu::BindGroup>,
    phase2_dispatch_bg: Option<wgpu::BindGroup>,
    vis_bg: Option<wgpu::BindGroup>,
    sw_dispatch_bg: Option<wgpu::BindGroup>,
    hw_dispatch_bg: Option<wgpu::BindGroup>,
    material_frame_bg: Option<wgpu::BindGroup>,
    sky_bg: Option<wgpu::BindGroup>,

    // Shadow parameters
    pub shadow_depth: f32,

    // Lighting uniform buffer (shared between material pass and post-processing)
    pub lighting_uniform_buffer: wgpu::Buffer,

    // Bark params uniform buffer (procedural bark material)
    pub bark_uniform_buffer: wgpu::Buffer,

    // Art direction uniform buffers + bind group
    pub art_direction_buffer: wgpu::Buffer,
    pub art_direction_palette_buffer: wgpu::Buffer,
    pub art_direction_bgl: wgpu::BindGroupLayout,
    pub art_direction_bg: wgpu::BindGroup,
    pub art_direction_outline_skip: bool,
    pub art_direction_bloom_tint: [f32; 3],
    pub art_direction_bloom_softness: f32,
    pub art_direction_color_grade: [f32; 4],
    pub art_direction_lod_bias: f32,

    // Temporal anti-aliasing
    pub taa_pass: Option<TaaPass>,

    // Cinematic post-processing (Phase 7)
    pub dof_pass: Option<DofPass>,
    pub motion_blur_pass: Option<MotionBlurPass>,
    pub fxaa_pass: Option<FxaaPass>,

    // Forward-rendered character capsule
    pub forward_character: ForwardCharacterPass,
    forward_char_scene_bg: Option<wgpu::BindGroup>,
    /// Character model matrix set per-frame from app.
    pub character_model: glam::Mat4,
    /// Whether to draw the character (third-person mode).
    pub character_visible: bool,

    // Cached per-frame values (set during render, used by capture_frame)
    last_manual_exposure: f32,
    last_dt: f32,
    last_elapsed: f32,
    /// Which HDR source was actually fed to tonemap last frame.
    last_hdr_source_stage: HdrStage,

    // Sky probe cache — avoids recomputing 1,872 optical depth evaluations when params unchanged
    sky_probe_cache: crate::scene::sky_probe::SkyProbeCache,

    // Froxel volumetric fog
    pub volumetric_fog_pass: VolumetricFogPass,

    // 3D noise textures for volumetric clouds
    pub noise_textures: NoiseTextures,

    // HZB occlusion culling state
    hzb_ready: bool,

    // Frame counter for temporal effects (SSAO/SSGI rotation, etc.)
    frame_index: u32,

    // GPU timing (None if TIMESTAMP_QUERY unsupported)
    pub timing: Option<GpuTimingContext>,
}

impl VisbufPipeline {
    pub fn new(gpu: &GpuContext) -> Self {
        let shadow_map = ShadowMap::new(&gpu.device);
        let shadow_pass = ShadowPass::new(gpu, &shadow_map);
        let vis_buffer = VisibilityBuffer::new(&gpu.device, gpu.width(), gpu.height());
        let dispatch_lists = DispatchLists::new(&gpu.device, 65536);

        let hw_raster = HwRasterPass::new(&gpu.device);
        let frame_bg = hw_raster.create_frame_bind_group(&gpu.device);

        let cull_pass = CullPass::new(&gpu.device, hw_raster.frame_bind_group_layout());
        let dag_traverse = DagTraversePass::new(&gpu.device, hw_raster.frame_bind_group_layout());
        let sw_raster = SwRasterPass::new(
            &gpu.device,
            hw_raster.frame_bind_group_layout(),
            hw_raster.mesh_bind_group_layout(),
        );
        let mut hzb_pass = HzbPass::new(&gpu.device);
        hzb_pass.resize(&gpu.device, gpu.width(), gpu.height());

        let sky_pass = SkyPass::new(&gpu.device);
        let sky_lut_pass = SkyLutPass::new(&gpu.device);
        let ibl_pass = IblPass::new(&gpu.device);
        let noise_textures = NoiseTextures::new(&gpu.device);
        let volumetric_fog_pass = VolumetricFogPass::new(&gpu.device);
        let cloud_pass = CloudPass::new(&gpu.device, gpu.width(), gpu.height());
        let ssgi_pass = Some(SsgiPass::new(&gpu.device, gpu.width(), gpu.height()));
        let ssr_pass = Some(SsrPass::new(&gpu.device, gpu.width(), gpu.height()));
        let area_light_pass = AreaLightPass::new(&gpu.device, &gpu.queue);
        let gtao_pass = Some(GtaoPass::new(&gpu.device, gpu.width(), gpu.height()));
        let sun_shaft_pass = SunShaftPass::new(gpu);
        let bloom_pass = Some(BloomPass::new(&gpu.device, gpu.width(), gpu.height()));
        let tonemap_pass = TonemapPass::new(gpu);

        let scene_color = create_scene_color_target(
            &gpu.device,
            gpu.width(),
            gpu.height(),
            HDR_FORMAT,
            "visbuf-scene-color",
        );

        let lighting_uniform_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("visbuf-lighting-uniforms"),
            size: std::mem::size_of::<LightingUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bark_uniform_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("visbuf-bark-params"),
            size: std::mem::size_of::<crate::material::procedural::BarkParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Art direction uniform buffers
        let art_direction_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("visbuf-art-direction-uniforms"),
            size: std::mem::size_of::<ArtDirectionUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let art_direction_palette_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("visbuf-art-direction-palette"),
            size: std::mem::size_of::<StylePalette>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let art_direction_bgl =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("art-direction-bgl"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });
        let art_direction_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("art-direction-bg"),
            layout: &art_direction_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: art_direction_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: art_direction_palette_buffer.as_entire_binding(),
                },
            ],
        });

        let material_pass = MaterialPass::new(
            &gpu.device,
            hw_raster.mesh_bind_group_layout(),
            gpu.width(),
            gpu.height(),
            &art_direction_bgl,
        );
        let outline_pass = OutlinePass::new(&gpu.device, &art_direction_bgl);

        let material_frame_bg = Some(material_pass.create_frame_bind_group(
            &gpu.device,
            &lighting_uniform_buffer,
            &bark_uniform_buffer,
            &area_light_pass,
        ));

        let sky_bg = Some(sky_pass.create_bind_group(
            &gpu.device,
            &lighting_uniform_buffer,
            &vis_buffer,
            &sky_lut_pass,
        ));

        let taa_pass = Some(TaaPass::new(&gpu.device, gpu.width(), gpu.height()));

        let dof_pass = Some(DofPass::new(&gpu.device, gpu.width(), gpu.height()));
        let motion_blur_pass = Some(MotionBlurPass::new(&gpu.device, gpu.width(), gpu.height()));
        let fxaa_pass = Some(FxaaPass::new(gpu));

        let forward_character = ForwardCharacterPass::new(&gpu.device, &lighting_uniform_buffer);
        let forward_char_scene_bg =
            Some(forward_character.create_scene_bind_group(&gpu.device, &lighting_uniform_buffer));

        Self {
            hw_raster,
            cull_pass,
            dag_traverse,
            sw_raster,
            material_pass,
            hzb_pass,
            shadow_pass,
            ibl_pass,
            sky_lut_pass,
            sky_pass,
            cloud_pass,
            ssgi_pass,
            ssr_pass,
            area_light_pass,
            gtao_pass,
            sun_shaft_pass,
            outline_pass,
            bloom_pass,
            tonemap_pass,
            taa_pass,
            dof_pass,
            motion_blur_pass,
            fxaa_pass,
            shadow_map,
            vis_buffer,
            dispatch_lists,
            scene_color,
            frame_bg,
            mesh_bg: None,
            cull_bg: None,
            dag_bg: None,
            dispatch_bg: None,
            phase2_dispatch_bg: None,
            vis_bg: None,
            sw_dispatch_bg: None,
            hw_dispatch_bg: None,
            material_frame_bg,
            sky_bg,
            shadow_depth: 140.0,
            lighting_uniform_buffer,
            bark_uniform_buffer,
            art_direction_buffer,
            art_direction_palette_buffer,
            art_direction_bgl,
            art_direction_bg,
            art_direction_outline_skip: true,
            art_direction_bloom_tint: [1.0, 1.0, 1.0],
            art_direction_bloom_softness: 0.0,
            art_direction_color_grade: [1.0, 1.0, 1.0, 0.0],
            art_direction_lod_bias: 0.0,
            forward_character,
            forward_char_scene_bg,
            character_model: glam::Mat4::IDENTITY,
            character_visible: false,
            last_manual_exposure: 1.0,
            last_dt: 0.0,
            last_elapsed: 0.0,
            last_hdr_source_stage: HdrStage::Raw,
            sky_probe_cache: crate::scene::sky_probe::SkyProbeCache::new(),
            hzb_ready: false,
            frame_index: 0,
            volumetric_fog_pass,
            noise_textures,
            timing: if gpu.timestamp_supported {
                Some(GpuTimingContext::new(&gpu.device, &gpu.queue))
            } else {
                None
            },
        }
    }

    pub fn sync_scene_resources(
        &mut self,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        scene: &RuntimeSceneGpu,
    ) {
        self.mesh_bg = Some(
            self.hw_raster
                .create_mesh_bind_group(device, &scene.meshlet_buffers),
        );
        self.cull_bg = Some(
            self.cull_pass
                .create_cull_bind_group(device, &scene.meshlet_buffers),
        );

        // GPU DAG traversal: compute root group indices and create bind group
        self.dag_traverse.sync_roots(&scene.dag);
        self.dag_bg = Some(
            self.dag_traverse
                .create_bind_group(device, &scene.meshlet_buffers),
        );

        let hzb_view = self
            .hzb_pass
            .hzb_full_view
            .as_ref()
            .expect("HZB not initialized");
        // CPU DAG cut mode: cull pass reads from its own group_queue_buffer
        self.dispatch_bg = Some(self.cull_pass.create_dispatch_bind_group(
            device,
            &self.dispatch_lists,
            hzb_view,
        ));
        self.phase2_dispatch_bg = Some(self.cull_pass.create_phase2_dispatch_bind_group(
            device,
            &self.dispatch_lists,
            hzb_view,
        ));
        self.sw_dispatch_bg = Some(self.sw_raster.create_dispatch_bind_group(
            device,
            &self.dispatch_lists,
            &self.vis_buffer,
        ));
        self.hw_dispatch_bg = Some(
            self.hw_raster
                .create_dispatch_bind_group(device, &self.dispatch_lists),
        );
        self.rebuild_vis_bind_group(
            device,
            &scene.bark_textures.albedo_rough_view,
            &scene.bark_textures.normal_ao_view,
            &scene.bark_textures.height_view,
        );
    }

    /// Rebuild visibility-dependent bind groups (after resize).
    pub fn rebuild_vis_bind_group(
        &mut self,
        device: &wgpu::Device,
        bark_albedo_rough_view: &wgpu::TextureView,
        bark_normal_ao_view: &wgpu::TextureView,
        bark_height_view: &wgpu::TextureView,
    ) {
        self.vis_bg = Some(self.material_pass.create_vis_bind_group(
            device,
            &self.vis_buffer,
            &self.shadow_map,
            bark_albedo_rough_view,
            bark_normal_ao_view,
            bark_height_view,
            &self.vis_buffer.depth_view,
            &self.sky_lut_pass,
            &self.ibl_pass,
        ));
        // Sky bind group depends on vis_buffer + LUTs
        self.sky_bg = Some(self.sky_pass.create_bind_group(
            device,
            &self.lighting_uniform_buffer,
            &self.vis_buffer,
            &self.sky_lut_pass,
        ));
    }

    /// Resize resolution-dependent resources.
    /// Note: `material_frame_bg` is NOT rebuilt here because it only references
    /// uniform buffers (lighting + bark), which are resolution-independent.
    pub fn resize(&mut self, gpu: &GpuContext) {
        self.vis_buffer = VisibilityBuffer::new(&gpu.device, gpu.width(), gpu.height());
        self.material_pass
            .resize(&gpu.device, gpu.width(), gpu.height());
        self.hzb_pass.resize(&gpu.device, gpu.width(), gpu.height());

        self.scene_color = create_scene_color_target(
            &gpu.device,
            gpu.width(),
            gpu.height(),
            HDR_FORMAT,
            "visbuf-scene-color",
        );

        self.cloud_pass
            .resize(&gpu.device, gpu.width(), gpu.height());
        if let Some(ssgi) = &mut self.ssgi_pass {
            ssgi.resize(&gpu.device, gpu.width(), gpu.height());
        }
        if let Some(ssr) = &mut self.ssr_pass {
            ssr.resize(&gpu.device, gpu.width(), gpu.height());
        }
        if let Some(gtao) = &mut self.gtao_pass {
            gtao.resize(&gpu.device, gpu.width(), gpu.height());
        }
        if self.bloom_pass.is_some() {
            self.bloom_pass = Some(BloomPass::new(&gpu.device, gpu.width(), gpu.height()));
        }
        self.sun_shaft_pass
            .resize(&gpu.device, gpu.width(), gpu.height());
        self.tonemap_pass = TonemapPass::new(gpu);
        if let Some(taa) = &mut self.taa_pass {
            taa.resize(&gpu.device, gpu.width(), gpu.height());
        }
        if let Some(dof) = &mut self.dof_pass {
            dof.resize(&gpu.device, gpu.width(), gpu.height());
        }
        if let Some(mb) = &mut self.motion_blur_pass {
            mb.resize(&gpu.device, gpu.width(), gpu.height());
        }
        if let Some(fxaa) = &mut self.fxaa_pass {
            fxaa.resize(gpu);
        }
        self.hzb_ready = false;

        if self.mesh_bg.is_some() {
            // SW dispatch bind group depends on vis_buffer
            self.sw_dispatch_bg = Some(self.sw_raster.create_dispatch_bind_group(
                &gpu.device,
                &self.dispatch_lists,
                &self.vis_buffer,
            ));
            // Rebuild HW dispatch bind group
            self.hw_dispatch_bg = Some(
                self.hw_raster
                    .create_dispatch_bind_group(&gpu.device, &self.dispatch_lists),
            );
            // Rebuild cull dispatch bind groups (depend on HZB view which changed)
            if self.dispatch_bg.is_some() {
                let hzb_view = self
                    .hzb_pass
                    .hzb_full_view
                    .as_ref()
                    .expect("HZB not initialized");
                self.dispatch_bg = Some(self.cull_pass.create_dispatch_bind_group(
                    &gpu.device,
                    &self.dispatch_lists,
                    hzb_view,
                ));
                self.phase2_dispatch_bg = Some(self.cull_pass.create_phase2_dispatch_bind_group(
                    &gpu.device,
                    &self.dispatch_lists,
                    hzb_view,
                ));
            }
        }
    }

    /// Render a complete frame using the 8-pass Nanite pipeline.
    pub fn render(
        &mut self,
        gpu: &GpuContext,
        scene: &RuntimeSceneGpu,
        camera: &CameraState,
        settings: &SceneSettings,
        elapsed_secs: f32,
    ) -> std::result::Result<(), wgpu::SurfaceError> {
        // Poll previous frame's timing results (one-frame delay)
        if let Some(timing) = &mut self.timing {
            timing.poll_and_log(&gpu.device);
        }

        // Generate noise textures on first frame
        self.noise_textures
            .ensure_generated(&gpu.device, &gpu.queue);

        // Update atmosphere LUTs (dirty-tracked — only regenerates when params change)
        self.sky_lut_pass.update(
            settings.sun_direction,
            settings.rayleigh_strength,
            settings.mie_strength,
            settings.mie_anisotropy,
        );
        self.sky_lut_pass.generate_if_dirty(&gpu.device, &gpu.queue);

        // Generate BRDF LUT (one-time) and update IBL cubemap (dirty-tracked)
        self.ibl_pass.ensure_brdf_lut(&gpu.device, &gpu.queue);
        self.ibl_pass.update_cubemap(
            &gpu.device,
            &gpu.queue,
            &self.sky_lut_pass.sky_view_view,
            &self.sky_lut_pass.lut_sampler,
            settings,
        );

        let frame = gpu.surface.get_current_texture()?;

        let view_proj = camera.view_projection_matrix();
        let cam_pos = camera.position();
        let fov_y = camera.fov_y_radians;

        // Compute TAA jitter (Halton 2,3 sequence, ±0.5 pixel)
        let (jx, jy) = self
            .taa_pass
            .as_ref()
            .map(|t| t.jitter())
            .unwrap_or((0.0, 0.0));
        let jittered_vp = if self.taa_pass.is_some() {
            apply_jitter(view_proj, jx, jy, gpu.width(), gpu.height())
        } else {
            view_proj
        };

        // Update frame uniforms (jittered VP for rasterization, unjittered for culling)
        let visbuf_uniforms = VisbufFrameUniforms {
            view_proj: jittered_vp.to_cols_array_2d(),
            camera_position: [cam_pos.x, cam_pos.y, cam_pos.z, 1.0],
            screen_size: [
                gpu.width() as f32,
                gpu.height() as f32,
                1.0 / gpu.width() as f32,
                1.0 / gpu.height() as f32,
            ],
            error_threshold: [
                1.0 + self.art_direction_lod_bias, // error_threshold_px (quality budget adds bias)
                1.0 / (2.0 * (fov_y * 0.5).tan()), // fov_factor
                0.0,
                0.0,
            ],
            frustum_planes: extract_frustum_planes(view_proj),
            hzb_params: [
                if self.hzb_ready { 1.0 } else { 0.0 },
                self.hzb_pass.mip_count as f32,
                gpu.width() as f32,
                gpu.height() as f32,
            ],
        };
        self.hw_raster.write_uniforms(&gpu.queue, &visbuf_uniforms);

        // Compute cascade shadow maps
        let (cascade_vps, cascade_splits) =
            compute_all_cascade_vps(camera, settings.sun_direction, self.shadow_depth);

        // Update lighting uniforms
        let lighting = LightingUniforms::from_camera(
            camera,
            settings,
            &cascade_vps,
            &cascade_splits,
            elapsed_secs,
            gpu.width(),
            gpu.height(),
            &mut self.sky_probe_cache,
        );
        gpu.queue.write_buffer(
            &self.lighting_uniform_buffer,
            0,
            bytemuck::bytes_of(&lighting),
        );

        // Update area lights
        self.area_light_pass
            .update_lights(&gpu.queue, &settings.area_lights);

        // Shadow uniforms
        self.shadow_pass.write_uniforms(&gpu.queue, &cascade_vps);
        self.sun_shaft_pass
            .write_uniforms(&gpu.queue, camera, settings, gpu.width(), gpu.height());

        // Register timing passes for this frame
        let _t_shadow = self.timing.as_mut().and_then(|t| t.register_pass("shadow"));
        let _t_material = self
            .timing
            .as_mut()
            .and_then(|t| t.register_pass("material"));
        let _t_sky = self.timing.as_mut().and_then(|t| t.register_pass("sky"));
        let t_cloud_march = self
            .timing
            .as_mut()
            .and_then(|t| t.register_pass("cloud_march"));
        let t_cloud_temporal = self
            .timing
            .as_mut()
            .and_then(|t| t.register_pass("cloud_temporal"));
        let t_cloud_composite = self
            .timing
            .as_mut()
            .and_then(|t| t.register_pass("cloud_composite"));
        let _t_bloom = self.timing.as_mut().and_then(|t| t.register_pass("bloom"));
        let _t_tonemap = self
            .timing
            .as_mut()
            .and_then(|t| t.register_pass("tonemap"));

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("visbuf-frame-encoder"),
            });

        // -- Pass 1: Shadow pass --
        self.shadow_pass.encode(
            &mut encoder,
            &scene.shadow_meshes,
            &scene.shadow_opaque_list,
        );

        // =====================================================================
        // Two-Phase Occlusion Culling
        // =====================================================================
        //
        // Phase 1: Cull with prev-frame HZB → raster survivors → mid-frame HZB
        // Phase 2: Re-test HZB-rejected meshlets against fresh HZB → raster
        // Final HZB build (for next frame's phase 1)

        // -- Phase 1: CPU DAG cut + GPU per-meshlet cull --
        self.vis_buffer.clear(&mut encoder);
        self.dispatch_lists.clear(&mut encoder);
        self.cull_pass.clear_phase2(&mut encoder);

        // CPU DAG cut: traverse DAG on CPU, upload selected groups to GPU.
        // TODO: GPU DAG traversal (dag_traverse_pass) produces correct sky/ground
        // but misses tree geometry — needs debugging of error projection in
        // dag_traverse.wgsl before it can replace the CPU path.
        let fov_factor = visbuf_uniforms.error_threshold[1];
        let error_threshold_px = visbuf_uniforms.error_threshold[0];
        let group_count = self.cull_pass.cpu_dag_cut_adaptive(
            &gpu.queue,
            &scene.dag,
            cam_pos,
            gpu.height() as f32,
            fov_factor,
            error_threshold_px,
        );

        if let (Some(cull_bg), Some(dispatch_bg)) = (&self.cull_bg, &self.dispatch_bg) {
            self.cull_pass.encode(
                &mut encoder,
                &self.frame_bg,
                cull_bg,
                dispatch_bg,
                group_count,
            );
        }

        self.dispatch_lists.prepare_indirect(&mut encoder);

        // -- Phase 1: HW rasterize (clears vis buffer + depth) --
        if let (Some(mesh_bg), Some(hw_dispatch_bg)) = (&self.mesh_bg, &self.hw_dispatch_bg) {
            self.hw_raster.encode(
                &mut encoder,
                &self.frame_bg,
                mesh_bg,
                hw_dispatch_bg,
                &self.vis_buffer,
                &self.dispatch_lists,
            );
        }

        // -- Mid-frame HZB build (from phase 1 raster depth) --
        self.hzb_pass.encode(
            &gpu.device,
            &mut encoder,
            &self.vis_buffer.depth_view,
            gpu.width(),
            gpu.height(),
        );
        self.hzb_ready = true;

        // -- Phase 2: Re-test HZB-rejected meshlets against fresh HZB --
        // WebGPU does not support reading phase2_reject_count back to the CPU for an
        // indirect compute dispatch, so we over-dispatch using total DAG meshlet count
        // as a conservative upper bound. Over-dispatched threads exit early via the
        // `idx >= group_queue_count` guard in the shader.
        self.dispatch_lists.clear_hw_count(&mut encoder);

        let phase2_max_rejects = scene.dag.total_dag_meshlet_count().min(65536) as u32;

        if let (Some(cull_bg), Some(phase2_dispatch_bg)) = (&self.cull_bg, &self.phase2_dispatch_bg)
        {
            self.cull_pass.encode_phase2(
                &mut encoder,
                &self.frame_bg,
                cull_bg,
                phase2_dispatch_bg,
                phase2_max_rejects,
            );
        }

        self.dispatch_lists.prepare_indirect(&mut encoder);

        // -- Phase 2: HW rasterize (loads existing vis buffer + depth, no clear) --
        if let (Some(mesh_bg), Some(hw_dispatch_bg)) = (&self.mesh_bg, &self.hw_dispatch_bg) {
            self.hw_raster.encode_phase2(
                &mut encoder,
                &self.frame_bg,
                mesh_bg,
                hw_dispatch_bg,
                &self.vis_buffer,
                &self.dispatch_lists,
            );
        }

        // -- SW rasterize --
        // TODO: re-enable when GPU cull pass routes meshlets to SW path.

        // -- Material resolve --
        if let (Some(mesh_bg), Some(vis_bg), Some(material_frame_bg)) =
            (&self.mesh_bg, &self.vis_bg, &self.material_frame_bg)
        {
            self.material_pass.encode(
                &mut encoder,
                material_frame_bg,
                mesh_bg,
                vis_bg,
                &self.art_direction_bg,
                &self.scene_color.view,
            );
        }

        // -- Forward character (capsule) --
        if self.character_visible {
            if let Some(scene_bg) = &self.forward_char_scene_bg {
                self.forward_character
                    .write_model(&gpu.queue, self.character_model);
                self.forward_character.encode(
                    &mut encoder,
                    scene_bg,
                    &self.scene_color.view,
                    &self.vis_buffer.depth_view,
                );
            }
        }

        // -- Final HZB build (after phase 2 raster + character, for next frame) --
        self.hzb_pass.encode(
            &gpu.device,
            &mut encoder,
            &self.vis_buffer.depth_view,
            gpu.width(),
            gpu.height(),
        );

        // -- Pass 5.5: Sky pass (fills empty vis pixels with atmospheric sky) --
        if let Some(sky_bg) = &self.sky_bg {
            self.sky_pass
                .encode(&mut encoder, sky_bg, &self.scene_color.view);
        }

        // Unjittered inverse view-projection (shared by cloud pass + TAA)
        let inv_vp = view_proj.inverse();

        // -- Pass 5.55: Cloud pass (volumetric clouds at profile resolution) --
        {
            // Apply resolution tier from profile (may trigger resize)
            let new_div = match settings.cloud_profile.resolution {
                crate::scene::CloudResolution::Quarter => 4u32,
                crate::scene::CloudResolution::Half => 2u32,
            };
            if self.cloud_pass.resolution_divisor != new_div {
                self.cloud_pass
                    .set_resolution(settings.cloud_profile.resolution);
                self.cloud_pass
                    .resize(&gpu.device, gpu.width(), gpu.height());
            }
            // Sky ambient is now sampled from sky-view LUT in the shader.
            // Pass a placeholder — the shader overrides with LUT values.
            let sky_amb = glam::Vec3::splat(0.15);
            self.cloud_pass.write_uniforms(
                &gpu.queue,
                inv_vp,
                cam_pos,
                settings.sun_direction,
                settings.sun_color,
                settings.sun_strength,
                sky_amb,
                settings.cloud_coverage,
                elapsed_secs,
                &settings.cloud_profile,
            );
            let cloud_timing_slots = match (t_cloud_march, t_cloud_temporal, t_cloud_composite) {
                (Some(march), Some(temporal), Some(composite)) => {
                    Some(super::cloud_pass::CloudTimingSlots {
                        march,
                        temporal,
                        composite,
                    })
                }
                _ => None,
            };
            let timing_ref = self.timing.as_ref().zip(cloud_timing_slots.as_ref());
            self.cloud_pass.encode(
                &gpu.device,
                &mut encoder,
                &self.noise_textures,
                &self.scene_color.view,
                &self.vis_buffer.depth_view,
                view_proj,
                timing_ref,
                &self.sky_lut_pass.sky_view_view,
                &self.sky_lut_pass.lut_sampler,
            );
        }

        // -- Pass 5.57: Froxel volumetric fog (inject → integrate → composite) --
        if settings.fog_volume_density > 0.0001 {
            // Use sky probe ambient for fog ambient scattering
            let fog_ambient = glam::Vec3::new(
                lighting.ambient_up[0],
                lighting.ambient_up[1],
                lighting.ambient_up[2],
            ) * settings.ambient_intensity
                * 0.3;

            self.volumetric_fog_pass.encode(
                &gpu.device,
                &gpu.queue,
                &mut encoder,
                &self.shadow_map,
                &self.vis_buffer.depth_view,
                &self.scene_color.view,
                inv_vp,
                view_proj,
                cam_pos,
                settings.sun_direction,
                settings.sun_color,
                settings.sun_strength,
                fog_ambient,
                settings.fog_volume_density,
                settings.fog_height_falloff,
                settings.fog_volume_albedo,
                settings.fog_volume_anisotropy,
                &cascade_vps,
                &cascade_splits,
                camera.near_plane,
                camera.far_plane,
            );
        }

        // -- Pass 5.6: SSGI (screen-space global illumination) --
        if settings.gi_intensity > 0.001 {
            if let Some(ssgi) = &self.ssgi_pass {
                let view = camera.view_matrix();
                let projection = camera.projection_matrix();
                let gi_intensity = settings.gi_intensity;
                ssgi.write_uniforms(
                    &gpu.queue,
                    view,
                    projection,
                    gpu.width(),
                    gpu.height(),
                    gi_intensity,
                    self.frame_index,
                );
                ssgi.execute(
                    &gpu.device,
                    &mut encoder,
                    &self.material_pass.normal_view,
                    &self.vis_buffer.depth_view,
                    &self.scene_color.view,
                );
            }
        }

        // -- Pass 5.65: SSR (screen-space reflections) --
        if let Some(ssr) = &self.ssr_pass {
            let view = camera.view_matrix();
            let projection = camera.projection_matrix();
            ssr.write_uniforms(
                &gpu.queue,
                view,
                projection,
                gpu.width(),
                gpu.height(),
                self.frame_index,
            );
            if let Some(hzb_view) = &self.hzb_pass.hzb_views.first() {
                ssr.execute(
                    &gpu.device,
                    &mut encoder,
                    &self.material_pass.normal_view,
                    &self.vis_buffer.depth_view,
                    &self.scene_color.view,
                    hzb_view,
                );
            }
        }

        // -- Pass 6: GTAO (horizon-based AO with bent normals) --
        if let Some(gtao) = &self.gtao_pass {
            let projection = camera.projection_matrix();
            let view = camera.view_matrix();
            gtao.write_uniforms(
                &gpu.queue,
                projection,
                view,
                gpu.width(),
                gpu.height(),
                self.frame_index,
            );
            gtao.execute(
                &gpu.device,
                &mut encoder,
                &self.material_pass.normal_view,
                &self.vis_buffer.depth_view,
            );
        }

        // -- Pass 7: Sun shafts --
        self.sun_shaft_pass.encode(
            &gpu.device,
            &mut encoder,
            &self.vis_buffer.depth_view,
            &self.scene_color.view,
        );

        // -- Pass 7.5: Outline --
        // Skip entirely when outline_strength is effectively zero.
        if !self.art_direction_outline_skip {
            self.outline_pass.encode(
                &gpu.device,
                &mut encoder,
                &self.vis_buffer.depth_view,
                &self.material_pass.normal_view,
                &self.scene_color.view,
                &self.art_direction_bg,
            );
        }

        // -- Pass 8: Bloom + Tonemap --
        if let Some(bloom) = &self.bloom_pass {
            bloom.execute(
                &gpu.device,
                &mut encoder,
                &gpu.queue,
                &self.scene_color.view,
                &self.scene_color.view,
                gpu.width(),
                gpu.height(),
                self.art_direction_bloom_tint,
                self.art_direction_bloom_softness,
            );
        }

        // -- Pass 8.5: TAA (temporal resolve after bloom, before tonemap) --
        if let Some(taa) = &mut self.taa_pass {
            taa.encode(
                &gpu.device,
                &gpu.queue,
                &mut encoder,
                &self.vis_buffer.depth_view,
                &self.scene_color.view,
                inv_vp,
                view_proj,
                [jx, jy],
            );
        }

        // Determine which HDR post-processing stages are active this frame.
        let dof_active = settings.dof_enabled && self.dof_pass.is_some();
        let mb_active = settings.motion_blur_enabled
            && self.motion_blur_pass.is_some()
            && self.taa_pass.is_some();
        let fxaa_active = settings.fxaa_enabled && self.taa_pass.is_none();

        // -- Pass 8.6: DoF (after TAA, reads HDR, outputs HDR blurred) --
        if dof_active {
            let dof = self.dof_pass.as_ref().unwrap();
            let hdr_in = self.tonemap_source();
            dof.encode(
                &gpu.device,
                &gpu.queue,
                &mut encoder,
                &self.vis_buffer.depth_view,
                hdr_in,
                settings.focus_distance,
                settings.aperture,
            );
        }

        // -- Pass 8.7: Motion blur (after DoF, reads HDR, outputs HDR blurred) --
        if mb_active {
            let mb = self.motion_blur_pass.as_ref().unwrap();
            let taa = self.taa_pass.as_ref().unwrap();
            let hdr_in = if dof_active {
                self.dof_pass.as_ref().unwrap().output_view()
            } else {
                self.tonemap_source()
            };
            mb.encode(
                &gpu.device,
                &gpu.queue,
                &mut encoder,
                taa.motion_view(),
                hdr_in,
            );
        }

        // Resolve final HDR source for tonemap
        let hdr_stage = if mb_active {
            HdrStage::AfterMotionBlur
        } else if dof_active {
            HdrStage::AfterDof
        } else {
            HdrStage::Raw
        };

        let frame_view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Determine tonemap target: if FXAA is active (and TAA is off), write to
        // intermediate LDR texture; otherwise write directly to the swapchain.
        let tonemap_target = if fxaa_active {
            self.fxaa_pass
                .as_ref()
                .map(|f| f.ldr_input_view())
                .unwrap_or(&frame_view)
        } else {
            &frame_view
        };

        let dt = (elapsed_secs - self.last_elapsed).max(0.0).min(0.25);

        // Scope to consume `final_hdr_source` borrow before mutating self fields
        {
            let final_hdr_source = match hdr_stage {
                HdrStage::AfterMotionBlur => self.motion_blur_pass.as_ref().unwrap().output_view(),
                HdrStage::AfterDof => self.dof_pass.as_ref().unwrap().output_view(),
                HdrStage::Raw => self.tonemap_source(),
            };

            self.tonemap_pass.encode(
                &gpu.device,
                &mut encoder,
                &gpu.queue,
                final_hdr_source,
                self.gtao_pass.as_ref().map(|gtao| gtao.ao_view()),
                tonemap_target,
                gpu.width(),
                gpu.height(),
                elapsed_secs,
                settings.exposure,
                dt,
                self.art_direction_color_grade,
                settings.ca_strength,
                settings.film_grain_strength,
            );
        }

        self.last_hdr_source_stage = hdr_stage;
        self.last_manual_exposure = settings.exposure;
        self.last_dt = dt;
        self.last_elapsed = elapsed_secs;

        // -- Pass 9: FXAA (after tonemap, only when TAA is off, reads LDR, outputs LDR) --
        if fxaa_active {
            if let Some(fxaa) = &self.fxaa_pass {
                fxaa.encode(
                    &gpu.device,
                    &mut encoder,
                    &gpu.queue,
                    &frame_view,
                    gpu.width(),
                    gpu.height(),
                );
            }
        }

        // Resolve GPU timing queries
        if let Some(timing) = &self.timing {
            timing.resolve(&mut encoder);
        }

        gpu.queue.submit([encoder.finish()]);

        // Begin async readback of timing data (read next frame)
        if let Some(timing) = &mut self.timing {
            timing.begin_readback();
        }

        frame.present();
        self.frame_index = self.frame_index.wrapping_add(1);
        Ok(())
    }

    /// Capture the current frame to a CPU-readable RGBA buffer.
    ///
    /// Must be called after `render()` — re-runs only the tonemap pass to a
    /// temporary COPY_SRC texture, then reads it back. Zero overhead on
    /// non-capture frames.
    pub fn capture_frame(
        &self,
        gpu: &GpuContext,
        elapsed_secs: f32,
    ) -> anyhow::Result<super::CapturedFrame> {
        let width = gpu.width();
        let height = gpu.height();

        let capture_texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("visbuf-capture-target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: gpu.surface_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let capture_view = capture_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("visbuf-capture-encoder"),
            });
        self.tonemap_pass.encode(
            &gpu.device,
            &mut encoder,
            &gpu.queue,
            self.last_tonemap_hdr_source(),
            self.gtao_pass.as_ref().map(|gtao| gtao.ao_view()),
            &capture_view,
            width,
            height,
            elapsed_secs,
            self.last_manual_exposure,
            self.last_dt,
            self.art_direction_color_grade,
            0.0, // no CA for captures
            0.0, // no film grain for captures
        );
        gpu.queue.submit([encoder.finish()]);

        super::readback_texture(gpu, &capture_texture, width, height)
    }

    /// HDR source for tonemap: TAA output when active, otherwise raw scene color.
    fn tonemap_source(&self) -> &wgpu::TextureView {
        self.taa_pass
            .as_ref()
            .map(|t| t.output_view())
            .unwrap_or(&self.scene_color.view)
    }

    /// The HDR source that was actually used for tonemap last frame (accounts for
    /// DoF and motion blur being active).
    fn last_tonemap_hdr_source(&self) -> &wgpu::TextureView {
        match self.last_hdr_source_stage {
            HdrStage::AfterMotionBlur => self
                .motion_blur_pass
                .as_ref()
                .map(|mb| mb.output_view())
                .unwrap_or_else(|| self.tonemap_source()),
            HdrStage::AfterDof => self
                .dof_pass
                .as_ref()
                .map(|dof| dof.output_view())
                .unwrap_or_else(|| self.tonemap_source()),
            HdrStage::Raw => self.tonemap_source(),
        }
    }
}

fn apply_jitter(vp: glam::Mat4, jx: f32, jy: f32, w: u32, h: u32) -> glam::Mat4 {
    // Add sub-pixel offset to the projection. Modifying z_axis (column 2) adds a
    // term that scales with clip.w, producing a constant NDC shift after perspective
    // division — independent of depth.
    let mut j = vp;
    j.z_axis.x += jx * 2.0 / w as f32;
    j.z_axis.y += jy * 2.0 / h as f32;
    j
}

fn create_scene_color_target(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    label: &str,
) -> SceneColorTarget {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    SceneColorTarget {
        _texture: texture,
        view,
    }
}
