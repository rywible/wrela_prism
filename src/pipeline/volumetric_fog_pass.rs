/// Froxel volumetric fog pass — 3D scattering + transmittance with temporal stability.
///
/// Three sub-passes:
/// 1. Inject (compute): Fill froxel grid with density + in-scattering per voxel
/// 2. Integrate (compute): Front-to-back march through froxel slices
/// 3. Composite (render): Apply fog to scene color via blend
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};

use crate::scene::shadow::ShadowMap;

use super::HDR_FORMAT;

/// Froxel grid dimensions.
const FROXEL_W: u32 = 160;
const FROXEL_H: u32 = 88;
const FROXEL_D: u32 = 128;

// ─── Uniform layouts ───

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct FogInjectUniforms {
    inv_view_proj: [[f32; 4]; 4],
    prev_inv_view_proj: [[f32; 4]; 4],
    camera_position: [f32; 4],
    sun_direction: [f32; 4],
    sun_color: [f32; 4],
    fog_params: [f32; 4], // density, height_falloff, anisotropy, temporal_weight
    fog_albedo: [f32; 4], // rgb = albedo, a = near_plane
    grid_params: [f32; 4], // grid_w, grid_h, grid_d, far_plane
    ambient_color: [f32; 4], // rgb = ambient, a = frame_index
    light_vp: [[f32; 4]; 4],
    light_vp_1: [[f32; 4]; 4],
    light_vp_2: [[f32; 4]; 4],
    light_vp_3: [[f32; 4]; 4],
    cascade_splits: [f32; 4],
    view_proj: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct FogIntegrateUniforms {
    grid_params: [f32; 4], // grid_w, grid_h, grid_d, far_plane
    near_plane: [f32; 4],  // x = near, y/z/w = unused
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct FogCompositeUniforms {
    grid_params: [f32; 4], // grid_w, grid_h, grid_d, far_plane
    near_plane: [f32; 4],  // x = near, y/z/w = unused
}

// ─── Pass struct ───

pub struct VolumetricFogPass {
    // Pipelines
    inject_pipeline: wgpu::ComputePipeline,
    inject_bgl: wgpu::BindGroupLayout,
    integrate_pipeline: wgpu::ComputePipeline,
    integrate_bgl: wgpu::BindGroupLayout,
    composite_pipeline: wgpu::RenderPipeline,
    composite_bgl: wgpu::BindGroupLayout,

    // 3D textures (ping-pong for temporal)
    _scatter_volume_a: wgpu::Texture,
    scatter_volume_a_view: wgpu::TextureView,
    _scatter_volume_b: wgpu::Texture,
    scatter_volume_b_view: wgpu::TextureView,
    _integrated_volume: wgpu::Texture,
    integrated_volume_view: wgpu::TextureView,

    // Samplers
    fog_sampler: wgpu::Sampler,
    prev_sampler: wgpu::Sampler,

    // Uniform buffers
    inject_uniform_buffer: wgpu::Buffer,
    integrate_uniform_buffer: wgpu::Buffer,
    composite_uniform_buffer: wgpu::Buffer,

    // State
    frame_index: u32,
    prev_inv_view_proj: Mat4,
}

impl VolumetricFogPass {
    pub fn new(device: &wgpu::Device) -> Self {
        // Create 3D textures
        let (scatter_volume_a, scatter_volume_a_view) =
            create_froxel_volume(device, "fog-scatter-a");
        let (scatter_volume_b, scatter_volume_b_view) =
            create_froxel_volume(device, "fog-scatter-b");
        let (integrated_volume, integrated_volume_view) =
            create_froxel_volume(device, "fog-integrated");

        let fog_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("fog-composite-sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        let prev_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("fog-prev-sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        // Uniform buffers
        let inject_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fog-inject-uniforms"),
            size: std::mem::size_of::<FogInjectUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let integrate_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fog-integrate-uniforms"),
            size: std::mem::size_of::<FogIntegrateUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let composite_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fog-composite-uniforms"),
            size: std::mem::size_of::<FogCompositeUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ─── Inject pipeline ───

        let inject_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fog-inject-bgl"),
            entries: &[
                // 0: uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 1: scatter volume (write)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D3,
                    },
                    count: None,
                },
                // 2: shadow map
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                // 3: shadow sampler (comparison)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
                // 4: previous scatter volume (read, for temporal blending)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                // 5: previous volume sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let inject_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fog-inject-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/fog_inject.wgsl").into()),
        });

        let inject_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fog-inject-layout"),
            bind_group_layouts: &[&inject_bgl],
            immediate_size: 0,
        });

        let inject_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("fog-inject-pipeline"),
            layout: Some(&inject_pl),
            module: &inject_shader,
            entry_point: Some("fog_inject"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // ─── Integrate pipeline ───

        let integrate_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fog-integrate-bgl"),
            entries: &[
                // 0: uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 1: scatter volume (read)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                // 2: integrated volume (write)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D3,
                    },
                    count: None,
                },
            ],
        });

        let integrate_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fog-integrate-shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/fog_integrate.wgsl").into(),
            ),
        });

        let integrate_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fog-integrate-layout"),
            bind_group_layouts: &[&integrate_bgl],
            immediate_size: 0,
        });

        let integrate_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("fog-integrate-pipeline"),
            layout: Some(&integrate_pl),
            module: &integrate_shader,
            entry_point: Some("fog_integrate"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // ─── Composite pipeline ───

        let composite_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fog-composite-bgl"),
            entries: &[
                // 0: composite uniforms
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
                // 1: integrated fog volume
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                // 2: fog sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // 3: depth texture
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        let composite_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fog-composite-shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/fog_composite.wgsl").into(),
            ),
        });

        let composite_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fog-composite-layout"),
            bind_group_layouts: &[&composite_bgl],
            immediate_size: 0,
        });

        // Blend: scene_color = scene_color * fog.a + fog.rgb
        // fog.rgb = inscatter (pre-multiplied), fog.a = transmittance
        // src = fog output, dst = scene
        // color: src_factor = One, dst_factor = SrcAlpha
        let composite_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("fog-composite-pipeline"),
            layout: Some(&composite_pl),
            vertex: wgpu::VertexState {
                module: &composite_shader,
                entry_point: Some("vs_fog_composite"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &composite_shader,
                entry_point: Some("fs_fog_composite"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: HDR_FORMAT,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::SrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent::OVER,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            inject_pipeline,
            inject_bgl,
            integrate_pipeline,
            integrate_bgl,
            composite_pipeline,
            composite_bgl,
            _scatter_volume_a: scatter_volume_a,
            scatter_volume_a_view,
            _scatter_volume_b: scatter_volume_b,
            scatter_volume_b_view,
            _integrated_volume: integrated_volume,
            integrated_volume_view,
            fog_sampler,
            prev_sampler,
            inject_uniform_buffer,
            integrate_uniform_buffer,
            composite_uniform_buffer,
            frame_index: 0,
            prev_inv_view_proj: Mat4::IDENTITY,
        }
    }

    /// Encode the full volumetric fog pipeline: inject → integrate → composite.
    #[allow(clippy::too_many_arguments)]
    pub fn encode(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        shadow_map: &ShadowMap,
        depth_view: &wgpu::TextureView,
        scene_color_view: &wgpu::TextureView,
        inv_view_proj: Mat4,
        view_proj: Mat4,
        camera_position: Vec3,
        sun_direction: Vec3,
        sun_color: Vec3,
        sun_strength: f32,
        ambient_color: Vec3,
        fog_volume_density: f32,
        fog_height_falloff: f32,
        fog_volume_albedo: Vec3,
        fog_volume_anisotropy: f32,
        cascade_vps: &[Mat4; 4],
        cascade_splits: &[f32; 4],
        near_plane: f32,
        far_plane: f32,
    ) {
        // Ping-pong: current writes to A, previous is B
        // After encoding, we swap them
        let (current_view, prev_view) = if self.frame_index.is_multiple_of(2) {
            (&self.scatter_volume_a_view, &self.scatter_volume_b_view)
        } else {
            (&self.scatter_volume_b_view, &self.scatter_volume_a_view)
        };

        let temporal_weight = if self.frame_index > 0 { 0.85_f32 } else { 0.0 };

        // Write inject uniforms
        let inject_uniforms = FogInjectUniforms {
            inv_view_proj: inv_view_proj.to_cols_array_2d(),
            prev_inv_view_proj: self.prev_inv_view_proj.to_cols_array_2d(),
            camera_position: [camera_position.x, camera_position.y, camera_position.z, 1.0],
            sun_direction: [sun_direction.x, sun_direction.y, sun_direction.z, 0.0],
            sun_color: [sun_color.x, sun_color.y, sun_color.z, sun_strength],
            fog_params: [
                fog_volume_density,
                fog_height_falloff,
                fog_volume_anisotropy,
                temporal_weight,
            ],
            fog_albedo: [
                fog_volume_albedo.x,
                fog_volume_albedo.y,
                fog_volume_albedo.z,
                near_plane,
            ],
            grid_params: [FROXEL_W as f32, FROXEL_H as f32, FROXEL_D as f32, far_plane],
            ambient_color: [
                ambient_color.x,
                ambient_color.y,
                ambient_color.z,
                self.frame_index as f32,
            ],
            light_vp: cascade_vps[0].to_cols_array_2d(),
            light_vp_1: cascade_vps[1].to_cols_array_2d(),
            light_vp_2: cascade_vps[2].to_cols_array_2d(),
            light_vp_3: cascade_vps[3].to_cols_array_2d(),
            cascade_splits: *cascade_splits,
            view_proj: view_proj.to_cols_array_2d(),
        };
        queue.write_buffer(
            &self.inject_uniform_buffer,
            0,
            bytemuck::bytes_of(&inject_uniforms),
        );

        // Write integrate uniforms
        let integrate_uniforms = FogIntegrateUniforms {
            grid_params: [FROXEL_W as f32, FROXEL_H as f32, FROXEL_D as f32, far_plane],
            near_plane: [near_plane, 0.0, 0.0, 0.0],
        };
        queue.write_buffer(
            &self.integrate_uniform_buffer,
            0,
            bytemuck::bytes_of(&integrate_uniforms),
        );

        // Write composite uniforms
        let composite_uniforms = FogCompositeUniforms {
            grid_params: [FROXEL_W as f32, FROXEL_H as f32, FROXEL_D as f32, far_plane],
            near_plane: [near_plane, 0.0, 0.0, 0.0],
        };
        queue.write_buffer(
            &self.composite_uniform_buffer,
            0,
            bytemuck::bytes_of(&composite_uniforms),
        );

        // ─── Pass 1: Inject ───
        let inject_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fog-inject-bg"),
            layout: &self.inject_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.inject_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(current_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&shadow_map.view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&shadow_map.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(prev_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&self.prev_sampler),
                },
            ],
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("fog-inject-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.inject_pipeline);
            pass.set_bind_group(0, &inject_bg, &[]);
            pass.dispatch_workgroups(FROXEL_W.div_ceil(8), FROXEL_H.div_ceil(8), FROXEL_D);
        }

        // ─── Pass 2: Integrate ───
        let integrate_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fog-integrate-bg"),
            layout: &self.integrate_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.integrate_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(current_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&self.integrated_volume_view),
                },
            ],
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("fog-integrate-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.integrate_pipeline);
            pass.set_bind_group(0, &integrate_bg, &[]);
            // Each thread handles one pixel column, iterating through all depth slices
            pass.dispatch_workgroups(FROXEL_W.div_ceil(8), FROXEL_H.div_ceil(8), 1);
        }

        // ─── Pass 3: Composite ───
        let composite_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fog-composite-bg"),
            layout: &self.composite_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.composite_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.integrated_volume_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.fog_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
            ],
        });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("fog-composite-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: scene_color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.composite_pipeline);
            pass.set_bind_group(0, &composite_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // Update state for next frame
        self.prev_inv_view_proj = inv_view_proj;
        self.frame_index = self.frame_index.wrapping_add(1);
    }
}

fn create_froxel_volume(device: &wgpu::Device, label: &str) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: FROXEL_W,
            height: FROXEL_H,
            depth_or_array_layers: FROXEL_D,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D3),
        ..Default::default()
    });
    (texture, view)
}
