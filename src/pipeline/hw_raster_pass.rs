use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::meshlet::GpuMeshletBuffers;

/// Frame uniforms for the visibility buffer passes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct VisbufFrameUniforms {
    pub view_proj: [[f32; 4]; 4],
    pub camera_position: [f32; 4],
    pub screen_size: [f32; 4],
    pub error_threshold: [f32; 4],
    pub frustum_planes: [[f32; 4]; 6],
}

/// Max vertices per draw call = MAX_MESHLET_TRIANGLES * 3
const MAX_VERTS_PER_MESHLET_DRAW: u32 = 124 * 3;

/// Dispatch lists for HW and SW rasterization, written by the cull pass.
pub struct DispatchLists {
    pub hw_dispatch_buffer: wgpu::Buffer,
    pub hw_count_buffer: wgpu::Buffer,
    pub sw_dispatch_buffer: wgpu::Buffer,
    pub sw_count_buffer: wgpu::Buffer,
    /// Indirect draw args: [vertex_count, instance_count, first_vertex, first_instance]
    pub hw_indirect_buffer: wgpu::Buffer,
    /// Maximum meshlet count (buffer capacity).
    pub max_meshlets: u32,
}

impl DispatchLists {
    pub fn new(device: &wgpu::Device, max_meshlets: u32) -> Self {
        let hw_dispatch_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hw-dispatch-list"),
            size: (max_meshlets as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let hw_count_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hw-dispatch-count"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let sw_dispatch_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sw-dispatch-list"),
            size: (max_meshlets as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sw_count_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sw-dispatch-count"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Indirect draw args: [vertex_count, instance_count, first_vertex, first_instance]
        // instance_count starts at 0 and is populated by copying hw_count_buffer after cull.
        let hw_indirect_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("hw-indirect-draw"),
            contents: bytemuck::cast_slice(&[MAX_VERTS_PER_MESHLET_DRAW, 0u32, 0u32, 0u32]),
            usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
        });

        Self {
            hw_dispatch_buffer,
            hw_count_buffer,
            sw_dispatch_buffer,
            sw_count_buffer,
            hw_indirect_buffer,
            max_meshlets,
        }
    }

    /// Clear dispatch counts to zero and reset indirect buffer instance_count.
    pub fn clear(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.clear_buffer(&self.hw_count_buffer, 0, None);
        encoder.clear_buffer(&self.sw_count_buffer, 0, None);
        // Reset instance_count (offset 4) to 0 in indirect buffer.
        // We'll copy from hw_count_buffer after cull pass.
    }

    /// After cull pass: copy hw_count into indirect buffer's instance_count field.
    pub fn prepare_indirect(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.copy_buffer_to_buffer(
            &self.hw_count_buffer,
            0,
            &self.hw_indirect_buffer,
            4, // offset of instance_count in DrawIndirectArgs
            4,
        );
    }
}

/// Visibility buffer textures.
pub struct VisibilityBuffer {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub depth_texture: wgpu::Texture,
    pub depth_view: wgpu::TextureView,
    /// R32Float copy of depth for SSAO and sun shaft sampling.
    pub depth_float_texture: wgpu::Texture,
    pub depth_float_view: wgpu::TextureView,
    /// Storage buffer for SW rasterizer atomic writes.
    pub storage_buffer: wgpu::Buffer,
    pub width: u32,
    pub height: u32,
}

impl VisibilityBuffer {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("visibility-buffer"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Uint,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("visibility-depth"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // R32Float copy for post-processing that needs float-sampled depth
        let depth_float_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("visibility-depth-float"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        let depth_float_view =
            depth_float_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let storage_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("visibility-storage"),
            size: (width as u64) * (height as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            texture,
            view,
            depth_texture,
            depth_view,
            depth_float_texture,
            depth_float_view,
            storage_buffer,
            width,
            height,
        }
    }

    pub fn clear(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.clear_buffer(&self.storage_buffer, 0, None);
    }
}

/// HW rasterization pass — vertex-pulling fallback pipeline.
///
/// Uses a standard vertex/fragment pipeline that reads meshlet data from
/// storage buffers. Each instance is one meshlet; vertex_index encodes
/// triangle and vertex within the meshlet.
pub struct HwRasterPass {
    pub uniform_buffer: wgpu::Buffer,
    bind_group_layout: wgpu::BindGroupLayout,
    mesh_bind_group_layout: wgpu::BindGroupLayout,
    dispatch_bind_group_layout: wgpu::BindGroupLayout,
    render_pipeline: wgpu::RenderPipeline,
}

impl HwRasterPass {
    pub fn new(device: &wgpu::Device) -> Self {
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("visbuf-frame-uniforms"),
            size: std::mem::size_of::<VisbufFrameUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("visbuf-frame-bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT | wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let mesh_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("visbuf-mesh-bgl"),
                entries: &[
                    storage_entry(
                        0,
                        true,
                        wgpu::ShaderStages::VERTEX_FRAGMENT | wgpu::ShaderStages::COMPUTE,
                    ),
                    storage_entry(
                        1,
                        true,
                        wgpu::ShaderStages::VERTEX_FRAGMENT | wgpu::ShaderStages::COMPUTE,
                    ),
                    storage_entry(
                        2,
                        true,
                        wgpu::ShaderStages::VERTEX_FRAGMENT | wgpu::ShaderStages::COMPUTE,
                    ),
                    storage_entry(
                        3,
                        true,
                        wgpu::ShaderStages::VERTEX_FRAGMENT | wgpu::ShaderStages::COMPUTE,
                    ),
                ],
            });

        let dispatch_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("hw-dispatch-bgl"),
                entries: &[storage_entry(0, true, wgpu::ShaderStages::VERTEX)],
            });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hw-vis-fallback-shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/meshlet_hw_vis_fallback.wgsl").into(),
            ),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("hw-vis-layout"),
            bind_group_layouts: &[
                &bind_group_layout,
                &mesh_bind_group_layout,
                &dispatch_bind_group_layout,
            ],
            immediate_size: 0,
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("hw-vis-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::R32Uint,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None, // let culling happen in the cull pass
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            uniform_buffer,
            bind_group_layout,
            mesh_bind_group_layout,
            dispatch_bind_group_layout,
            render_pipeline,
        }
    }

    pub fn write_uniforms(&self, queue: &wgpu::Queue, uniforms: &VisbufFrameUniforms) {
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(uniforms));
    }

    pub fn frame_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }

    pub fn mesh_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.mesh_bind_group_layout
    }

    pub fn create_frame_bind_group(&self, device: &wgpu::Device) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("visbuf-frame-bg"),
            layout: &self.bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: self.uniform_buffer.as_entire_binding(),
            }],
        })
    }

    pub fn create_mesh_bind_group(
        &self,
        device: &wgpu::Device,
        buffers: &GpuMeshletBuffers,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("visbuf-mesh-bg"),
            layout: &self.mesh_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.vertex_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.meshlet_vertex_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.meshlet_triangle_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffers.meshlet_desc_buffer.as_entire_binding(),
                },
            ],
        })
    }

    pub fn create_dispatch_bind_group(
        &self,
        device: &wgpu::Device,
        dispatch: &DispatchLists,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hw-dispatch-bg"),
            layout: &self.dispatch_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: dispatch.hw_dispatch_buffer.as_entire_binding(),
            }],
        })
    }

    /// Encode the HW visibility buffer fill pass.
    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        frame_bg: &wgpu::BindGroup,
        mesh_bg: &wgpu::BindGroup,
        dispatch_bg: &wgpu::BindGroup,
        vis_buffer: &VisibilityBuffer,
        dispatch: &DispatchLists,
    ) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("hw-vis-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &vis_buffer.view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
                        a: 0.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &vis_buffer.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            occlusion_query_set: None,
            timestamp_writes: None,
            multiview_mask: None,
        });

        pass.set_pipeline(&self.render_pipeline);
        pass.set_bind_group(0, frame_bg, &[]);
        pass.set_bind_group(1, mesh_bg, &[]);
        pass.set_bind_group(2, dispatch_bg, &[]);
        pass.draw_indirect(&dispatch.hw_indirect_buffer, 0);
    }
}

fn normalize_plane(plane: glam::Vec4) -> [f32; 4] {
    let normal = plane.truncate();
    let len = normal.length();
    if len <= 1e-6 {
        return plane.to_array();
    }
    (plane / len).to_array()
}

pub fn extract_frustum_planes(view_proj: glam::Mat4) -> [[f32; 4]; 6] {
    let row0 = glam::Vec4::new(
        view_proj.x_axis.x,
        view_proj.y_axis.x,
        view_proj.z_axis.x,
        view_proj.w_axis.x,
    );
    let row1 = glam::Vec4::new(
        view_proj.x_axis.y,
        view_proj.y_axis.y,
        view_proj.z_axis.y,
        view_proj.w_axis.y,
    );
    let row2 = glam::Vec4::new(
        view_proj.x_axis.z,
        view_proj.y_axis.z,
        view_proj.z_axis.z,
        view_proj.w_axis.z,
    );
    let row3 = glam::Vec4::new(
        view_proj.x_axis.w,
        view_proj.y_axis.w,
        view_proj.z_axis.w,
        view_proj.w_axis.w,
    );

    [
        normalize_plane(row3 + row0),
        normalize_plane(row3 - row0),
        normalize_plane(row3 + row1),
        normalize_plane(row3 - row1),
        normalize_plane(row2),
        normalize_plane(row3 - row2),
    ]
}

use super::storage_entry;

#[cfg(test)]
mod tests {
    use super::extract_frustum_planes;
    use crate::camera::CameraState;
    use glam::{Vec3, Vec4};

    fn sphere_inside_all_planes(planes: &[[f32; 4]; 6], center: Vec3, radius: f32) -> bool {
        planes.iter().all(|plane| {
            let plane = Vec4::from_array(*plane);
            plane.truncate().dot(center) + plane.w >= -radius
        })
    }

    #[test]
    fn extracted_frustum_planes_accept_visible_points_and_reject_points_behind_camera() {
        let camera = CameraState::new(16.0 / 9.0);
        let planes = extract_frustum_planes(camera.view_projection_matrix());

        let visible = camera.position() + camera.forward() * 20.0;
        let behind = camera.position() - camera.forward() * 20.0;

        assert!(sphere_inside_all_planes(&planes, visible, 0.5));
        assert!(!sphere_inside_all_planes(&planes, behind, 0.5));
    }
}
