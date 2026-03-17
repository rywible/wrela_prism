use crate::art_direction::{ArtDirectionUniforms, StylePalette};
use crate::camera::CameraState;
use crate::gpu::{GpuContext, GpuTimingContext};
use crate::runtime_scene::RuntimeSceneGpu;
use crate::scene::shadow::{compute_all_cascade_vps, ShadowMap};
use crate::scene::{LightingUniforms, SceneSettings};

use super::bloom_pass::BloomPass;
use super::cloud_pass::CloudPass;
use super::cull_pass::CullPass;
use super::forward_character::ForwardCharacterPass;
use super::hw_raster_pass::{
    extract_frustum_planes, DispatchLists, HwRasterPass, VisbufFrameUniforms, VisibilityBuffer,
};
use super::hzb_pass::HzbPass;
use super::material_pass::MaterialPass;
use super::noise_textures::NoiseTextures;
use super::outline_pass::OutlinePass;
use super::shadow_pass::ShadowPass;
use super::sky_lut_pass::SkyLutPass;
use super::sky_pass::SkyPass;
use super::ssao_pass::SsaoPass;
use super::ssgi_pass::SsgiPass;
use super::sun_shaft_pass::SunShaftPass;
use super::sw_raster_pass::SwRasterPass;
use super::tonemap_pass::TonemapPass;
use super::HDR_FORMAT;
struct SceneColorTarget {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
}

/// 8-pass Nanite-style visibility buffer pipeline.
pub struct VisbufPipeline {
    // Core passes
    pub hw_raster: HwRasterPass,
    pub cull_pass: CullPass,
    pub sw_raster: SwRasterPass,
    pub material_pass: MaterialPass,
    pub hzb_pass: HzbPass,

    // Existing passes (kept)
    pub shadow_pass: ShadowPass,
    pub sky_lut_pass: SkyLutPass,
    pub sky_pass: SkyPass,
    pub cloud_pass: CloudPass,
    pub ssgi_pass: Option<SsgiPass>,
    pub ssao_pass: Option<SsaoPass>,
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
    dispatch_bg: Option<wgpu::BindGroup>,
    vis_bg: Option<wgpu::BindGroup>,
    sw_dispatch_bg: Option<wgpu::BindGroup>,
    hw_dispatch_bg: Option<wgpu::BindGroup>,
    material_frame_bg: Option<wgpu::BindGroup>,
    sky_bg: Option<wgpu::BindGroup>,

    // Shadow parameters
    pub shadow_center: glam::Vec3,
    pub shadow_base_radius: f32,
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

    // Forward-rendered character capsule
    pub forward_character: ForwardCharacterPass,
    forward_char_scene_bg: Option<wgpu::BindGroup>,
    /// Character model matrix set per-frame from app.
    pub character_model: glam::Mat4,
    /// Whether to draw the character (third-person mode).
    pub character_visible: bool,

    // Depth copy (Depth32Float → R32Float via compute)
    depth_copy_pipeline: wgpu::ComputePipeline,
    depth_copy_bgl: wgpu::BindGroupLayout,
    depth_copy_bg: Option<wgpu::BindGroup>,

    // 3D noise textures for volumetric clouds
    pub noise_textures: NoiseTextures,

    // GPU timing (None if TIMESTAMP_QUERY unsupported)
    pub timing: Option<GpuTimingContext>,
}

impl VisbufPipeline {
    pub fn new(gpu: &GpuContext) -> Self {
        let shadow_map = ShadowMap::new(&gpu.device);
        let shadow_pass = ShadowPass::new(gpu, &shadow_map);
        let vis_buffer = VisibilityBuffer::new(&gpu.device, gpu.width(), gpu.height());
        let dispatch_lists = DispatchLists::new(&gpu.device, 8192);

        let hw_raster = HwRasterPass::new(&gpu.device);
        let frame_bg = hw_raster.create_frame_bind_group(&gpu.device);

        let cull_pass = CullPass::new(&gpu.device, hw_raster.frame_bind_group_layout());
        let sw_raster = SwRasterPass::new(
            &gpu.device,
            hw_raster.frame_bind_group_layout(),
            hw_raster.mesh_bind_group_layout(),
        );
        let mut hzb_pass = HzbPass::new(&gpu.device);
        hzb_pass.resize(&gpu.device, gpu.width(), gpu.height());

        let sky_pass = SkyPass::new(&gpu.device);
        let sky_lut_pass = SkyLutPass::new(&gpu.device);
        let noise_textures = NoiseTextures::new(&gpu.device);
        let cloud_pass = CloudPass::new(&gpu.device, gpu.width(), gpu.height());
        let ssgi_pass = Some(SsgiPass::new(&gpu.device, gpu.width(), gpu.height()));
        let ssao_pass = Some(SsaoPass::new(&gpu.device, gpu.width(), gpu.height()));
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
        ));

        let sky_bg = Some(sky_pass.create_bind_group(
            &gpu.device,
            &lighting_uniform_buffer,
            &vis_buffer,
            &sky_lut_pass,
        ));

        // Depth copy compute pipeline (Depth32Float → R32Float)
        let depth_copy_bgl =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("depth-copy-bgl"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Depth,
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::StorageTexture {
                                access: wgpu::StorageTextureAccess::WriteOnly,
                                format: wgpu::TextureFormat::R32Float,
                                view_dimension: wgpu::TextureViewDimension::D2,
                            },
                            count: None,
                        },
                    ],
                });
        let depth_copy_shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("depth-copy-shader"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../../shaders/depth_to_float.wgsl").into(),
                ),
            });
        let depth_copy_layout =
            gpu.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("depth-copy-layout"),
                    bind_group_layouts: &[&depth_copy_bgl],
                    immediate_size: 0,
                });
        let depth_copy_pipeline =
            gpu.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("depth-copy-pipeline"),
                    layout: Some(&depth_copy_layout),
                    module: &depth_copy_shader,
                    entry_point: Some("depth_copy"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                });

        let depth_copy_bg = Some(create_depth_copy_bind_group(
            &gpu.device,
            &depth_copy_bgl,
            &vis_buffer,
        ));

        let forward_character = ForwardCharacterPass::new(&gpu.device, &lighting_uniform_buffer);
        let forward_char_scene_bg =
            Some(forward_character.create_scene_bind_group(&gpu.device, &lighting_uniform_buffer));

        Self {
            hw_raster,
            cull_pass,
            sw_raster,
            material_pass,
            hzb_pass,
            shadow_pass,
            sky_lut_pass,
            sky_pass,
            cloud_pass,
            ssgi_pass,
            ssao_pass,
            sun_shaft_pass,
            outline_pass,
            bloom_pass,
            tonemap_pass,
            shadow_map,
            vis_buffer,
            dispatch_lists,
            scene_color,
            frame_bg,
            mesh_bg: None,
            cull_bg: None,
            dispatch_bg: None,
            vis_bg: None,
            sw_dispatch_bg: None,
            hw_dispatch_bg: None,
            material_frame_bg,
            sky_bg,
            shadow_center: glam::Vec3::new(0.0, 10.0, 0.0),
            shadow_base_radius: 30.0,
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
            depth_copy_pipeline,
            depth_copy_bgl,
            depth_copy_bg,
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
        self.dispatch_bg = Some(
            self.cull_pass
                .create_dispatch_bind_group(device, &self.dispatch_lists),
        );
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

        // Adaptive DAG cut runs per-frame in render().
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
            &self.vis_buffer.depth_float_view,
            &self.sky_lut_pass,
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
        if self.ssao_pass.is_some() {
            self.ssao_pass = Some(SsaoPass::new(&gpu.device, gpu.width(), gpu.height()));
        }
        if self.bloom_pass.is_some() {
            self.bloom_pass = Some(BloomPass::new(&gpu.device, gpu.width(), gpu.height()));
        }
        self.sun_shaft_pass
            .resize(&gpu.device, gpu.width(), gpu.height());
        self.tonemap_pass = TonemapPass::new(gpu);

        // Rebuild depth copy bind group (depends on vis_buffer)
        self.depth_copy_bg = Some(create_depth_copy_bind_group(
            &gpu.device,
            &self.depth_copy_bgl,
            &self.vis_buffer,
        ));

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

        let frame = gpu.surface.get_current_texture()?;

        let view_proj = camera.view_projection_matrix();
        let cam_pos = camera.position();
        let fov_y = camera.fov_y_radians;

        // Update frame uniforms
        let visbuf_uniforms = VisbufFrameUniforms {
            view_proj: view_proj.to_cols_array_2d(),
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
        );
        gpu.queue.write_buffer(
            &self.lighting_uniform_buffer,
            0,
            bytemuck::bytes_of(&lighting),
        );

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

        // -- Pass 2: Adaptive DAG cut (CPU group selection) + GPU per-meshlet cull --
        self.vis_buffer.clear(&mut encoder);
        self.dispatch_lists.clear(&mut encoder);

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

        // -- Pass 3: HW rasterize --
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

        // -- Pass 4: SW rasterize --
        // TODO: re-enable when GPU cull pass routes meshlets to SW path.

        // Copy Depth32Float → R32Float via compute (formats not copy-compatible)
        if let Some(depth_copy_bg) = &self.depth_copy_bg {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("depth-copy-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.depth_copy_pipeline);
            pass.set_bind_group(0, depth_copy_bg, &[]);
            let w = (self.vis_buffer.width + 7) / 8;
            let h = (self.vis_buffer.height + 7) / 8;
            pass.dispatch_workgroups(w, h, 1);
        }

        // -- Pass 5: Material resolve --
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

        // -- Pass 5.1: Forward character (capsule) --
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

            // Re-copy depth after character was drawn (character writes to depth)
            if let Some(depth_copy_bg) = &self.depth_copy_bg {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("depth-copy-pass-post-char"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.depth_copy_pipeline);
                pass.set_bind_group(0, depth_copy_bg, &[]);
                let w = (self.vis_buffer.width + 7) / 8;
                let h = (self.vis_buffer.height + 7) / 8;
                pass.dispatch_workgroups(w, h, 1);
            }
        }

        // -- Pass 5.5: Sky pass (fills empty vis pixels with atmospheric sky) --
        if let Some(sky_bg) = &self.sky_bg {
            self.sky_pass
                .encode(&mut encoder, sky_bg, &self.scene_color.view);
        }

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
            let inv_vp = view_proj.inverse();
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
                &self.vis_buffer.depth_float_view,
                view_proj,
                timing_ref,
                &self.sky_lut_pass.sky_view_view,
                &self.sky_lut_pass.lut_sampler,
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
                );
                ssgi.execute(
                    &gpu.device,
                    &mut encoder,
                    &self.material_pass.normal_view,
                    &self.vis_buffer.depth_float_view,
                    &self.scene_color.view,
                );
            }
        }

        // -- Pass 6: SSAO --
        if let Some(ssao) = &self.ssao_pass {
            let projection = camera.projection_matrix();
            let view = camera.view_matrix();
            ssao.write_uniforms(&gpu.queue, projection, view, gpu.width(), gpu.height());
            ssao.execute(
                &gpu.device,
                &mut encoder,
                &self.material_pass.normal_view,
                &self.vis_buffer.depth_float_view,
            );
        }

        // -- Pass 7: Sun shafts --
        self.sun_shaft_pass.encode(
            &gpu.device,
            &mut encoder,
            &self.vis_buffer.depth_float_view,
            &self.scene_color.view,
        );

        // -- Pass 7.5: Outline --
        // Skip entirely when outline_strength is effectively zero.
        if !self.art_direction_outline_skip {
            self.outline_pass.encode(
                &gpu.device,
                &mut encoder,
                &self.vis_buffer.depth_float_view,
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

        let frame_view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.tonemap_pass.encode(
            &gpu.device,
            &mut encoder,
            &gpu.queue,
            &self.scene_color.view,
            self.ssao_pass.as_ref().map(|ssao| ssao.ao_view()),
            &frame_view,
            gpu.width(),
            gpu.height(),
            elapsed_secs,
            self.art_direction_color_grade,
        );

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
            &self.scene_color.view,
            self.ssao_pass.as_ref().map(|ssao| ssao.ao_view()),
            &capture_view,
            width,
            height,
            elapsed_secs,
            self.art_direction_color_grade,
        );
        gpu.queue.submit([encoder.finish()]);

        super::readback_texture(gpu, &capture_texture, width, height)
    }
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

fn create_depth_copy_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    vis_buffer: &VisibilityBuffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("depth-copy-bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&vis_buffer.depth_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&vis_buffer.depth_float_view),
            },
        ],
    })
}
