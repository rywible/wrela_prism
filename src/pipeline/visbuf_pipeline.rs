use crate::camera::CameraState;
use crate::gpu::GpuContext;
use crate::runtime_scene::RuntimeSceneGpu;
use crate::scene::shadow::{compute_light_vp, ShadowMap};
use crate::scene::{LightingUniforms, SceneSettings};

use super::bloom_pass::BloomPass;
use super::cull_pass::CullPass;
use super::hw_raster_pass::{DispatchLists, HwRasterPass, VisbufFrameUniforms, VisibilityBuffer};
use super::hzb_pass::HzbPass;
use super::material_pass::MaterialPass;
use super::shadow_pass::ShadowPass;
use super::sky_pass::SkyPass;
use super::ssao_pass::SsaoPass;
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
    pub sky_pass: SkyPass,
    pub ssao_pass: Option<SsaoPass>,
    pub sun_shaft_pass: SunShaftPass,
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

    // Depth copy (Depth32Float → R32Float via compute)
    depth_copy_pipeline: wgpu::ComputePipeline,
    depth_copy_bgl: wgpu::BindGroupLayout,
    depth_copy_bg: Option<wgpu::BindGroup>,
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
        let material_pass = MaterialPass::new(
            &gpu.device,
            hw_raster.mesh_bind_group_layout(),
            gpu.width(),
            gpu.height(),
        );
        let mut hzb_pass = HzbPass::new(&gpu.device);
        hzb_pass.resize(&gpu.device, gpu.width(), gpu.height());

        let sky_pass = SkyPass::new(&gpu.device);
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

        let material_frame_bg = Some(material_pass.create_frame_bind_group(
            &gpu.device,
            &lighting_uniform_buffer,
            &bark_uniform_buffer,
        ));

        let sky_bg = Some(sky_pass.create_bind_group(
            &gpu.device,
            &lighting_uniform_buffer,
            &vis_buffer,
        ));

        // Depth copy compute pipeline (Depth32Float → R32Float)
        let depth_copy_bgl = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
        let depth_copy_shader = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("depth-copy-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/depth_to_float.wgsl").into()),
        });
        let depth_copy_layout = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("depth-copy-layout"),
            bind_group_layouts: &[&depth_copy_bgl],
            immediate_size: 0,
        });
        let depth_copy_pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
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

        Self {
            hw_raster,
            cull_pass,
            sw_raster,
            material_pass,
            hzb_pass,
            shadow_pass,
            sky_pass,
            ssao_pass,
            sun_shaft_pass,
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
            depth_copy_pipeline,
            depth_copy_bgl,
            depth_copy_bg,
        }
    }

    pub fn sync_scene_resources(
        &mut self,
        device: &wgpu::Device,
        scene: &RuntimeSceneGpu,
    ) {
        self.mesh_bg = Some(self.hw_raster.create_mesh_bind_group(device, &scene.meshlet_buffers));
        self.cull_bg = Some(self.cull_pass.create_cull_bind_group(device, &scene.meshlet_buffers));
        self.dispatch_bg = Some(
            self.cull_pass.create_dispatch_bind_group(device, &self.dispatch_lists),
        );
        self.sw_dispatch_bg = Some(
            self.sw_raster.create_dispatch_bind_group(device, &self.dispatch_lists, &self.vis_buffer),
        );
        self.hw_dispatch_bg = Some(
            self.hw_raster.create_dispatch_bind_group(device, &self.dispatch_lists),
        );
        self.rebuild_vis_bind_group(device, &scene.alpha_mask_view);
    }

    /// Rebuild visibility-dependent bind groups (after resize or alpha mask change).
    pub fn rebuild_vis_bind_group(
        &mut self,
        device: &wgpu::Device,
        alpha_view: &wgpu::TextureView,
    ) {
        self.vis_bg = Some(self.material_pass.create_vis_bind_group(
            device,
            &self.vis_buffer,
            &self.shadow_map,
            alpha_view,
        ));
        // Sky bind group depends on vis_buffer
        self.sky_bg = Some(self.sky_pass.create_bind_group(
            device,
            &self.lighting_uniform_buffer,
            &self.vis_buffer,
        ));
    }

    /// Resize resolution-dependent resources.
    /// Note: `material_frame_bg` is NOT rebuilt here because it only references
    /// uniform buffers (lighting + bark), which are resolution-independent.
    pub fn resize(&mut self, gpu: &GpuContext) {
        self.vis_buffer = VisibilityBuffer::new(&gpu.device, gpu.width(), gpu.height());
        self.material_pass.resize(&gpu.device, gpu.width(), gpu.height());
        self.hzb_pass.resize(&gpu.device, gpu.width(), gpu.height());

        self.scene_color = create_scene_color_target(
            &gpu.device,
            gpu.width(),
            gpu.height(),
            HDR_FORMAT,
            "visbuf-scene-color",
        );

        if self.ssao_pass.is_some() {
            self.ssao_pass = Some(SsaoPass::new(&gpu.device, gpu.width(), gpu.height()));
        }
        if self.bloom_pass.is_some() {
            self.bloom_pass = Some(BloomPass::new(&gpu.device, gpu.width(), gpu.height()));
        }
        self.sun_shaft_pass.resize(&gpu.device, gpu.width(), gpu.height());
        self.tonemap_pass = TonemapPass::new(gpu);

        // Rebuild depth copy bind group (depends on vis_buffer)
        self.depth_copy_bg = Some(create_depth_copy_bind_group(
            &gpu.device,
            &self.depth_copy_bgl,
            &self.vis_buffer,
        ));

        if self.mesh_bg.is_some() {
            // SW dispatch bind group depends on vis_buffer
            self.sw_dispatch_bg = Some(
                self.sw_raster.create_dispatch_bind_group(
                    &gpu.device,
                    &self.dispatch_lists,
                    &self.vis_buffer,
                ),
            );
            // Rebuild HW dispatch bind group
            self.hw_dispatch_bg = Some(
                self.hw_raster.create_dispatch_bind_group(
                    &gpu.device,
                    &self.dispatch_lists,
                ),
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
                1.0, // error_threshold_px
                1.0 / (2.0 * (fov_y * 0.5).tan()), // fov_factor
                0.0,
                0.0,
            ],
        };
        self.hw_raster.write_uniforms(&gpu.queue, &visbuf_uniforms);

        // Update lighting uniforms
        let shadow_focus_radius = self.shadow_base_radius;
        let light_vp = compute_light_vp(
            settings.sun_direction,
            self.shadow_center,
            shadow_focus_radius,
            self.shadow_depth,
        );
        let lighting = LightingUniforms::from_camera(
            camera,
            settings,
            light_vp,
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
        self.shadow_pass.write_uniforms(&gpu.queue, light_vp);
        self.sun_shaft_pass.write_uniforms(
            &gpu.queue,
            camera,
            settings,
            gpu.width(),
            gpu.height(),
        );

        // CPU-side DAG cut — writes meshlet dispatch list directly
        let _meshlet_count = self.cull_pass.cpu_dag_cut(
            &gpu.queue,
            &scene.dag,
            &self.dispatch_lists,
        );

        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("visbuf-frame-encoder"),
        });

        // -- Pass 1: Shadow pass --
        self.shadow_pass.encode(
            &mut encoder,
            &scene.shadow_meshes,
            &scene.shadow_opaque_list,
            &scene.shadow_transparent_list,
        );

        // -- Pass 2: Prepare indirect draw from CPU-written dispatch count --
        self.vis_buffer.clear(&mut encoder);
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
        // Currently cpu_dag_cut writes all meshlets to HW dispatch only.

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
                &self.scene_color.view,
            );
        }

        // -- Pass 5.5: Sky pass (fills empty vis pixels with atmospheric sky) --
        if let Some(sky_bg) = &self.sky_bg {
            self.sky_pass.encode(
                &mut encoder,
                sky_bg,
                &self.scene_color.view,
            );
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
        );

        gpu.queue.submit([encoder.finish()]);
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
