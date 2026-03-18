use glam::Mat4;

use super::HDR_FORMAT;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SsrUniforms {
    view: [[f32; 4]; 4],
    inv_view: [[f32; 4]; 4],
    projection: [[f32; 4]; 4],
    inv_projection: [[f32; 4]; 4],
    screen_size: [f32; 4],
    params: [f32; 4],  // max_distance, thickness, stride, max_steps
    params2: [f32; 4], // roughness_cutoff, fade_start, fade_end, frame_index
}

pub struct SsrPass {
    trace_pipeline: wgpu::ComputePipeline,
    composite_pipeline: wgpu::RenderPipeline,
    trace_bgl: wgpu::BindGroupLayout,
    composite_bgl: wgpu::BindGroupLayout,
    pub reflection_texture: wgpu::Texture,
    pub reflection_view: wgpu::TextureView,
    uniform_buffer: wgpu::Buffer,
    width: u32,
    height: u32,
}

impl SsrPass {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let ssr_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("prism-ssr-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/ssr.wgsl").into()),
        });

        let composite_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("prism-ssr-composite-shader"),
            source: wgpu::ShaderSource::Wgsl(SSR_COMPOSITE_WGSL.into()),
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("prism-ssr-uniforms"),
            size: std::mem::size_of::<SsrUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let reflection_texture = create_reflection_texture(device, width, height);
        let reflection_view =
            reflection_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Trace bind group layout:
        // 0: uniforms, 1: normals, 2: depth, 3: scene color, 4: HZB, 5: output (storage)
        let trace_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("prism-ssr-trace-bgl"),
            entries: &[
                uniform_entry(0),
                texture_entry(1),       // normals
                depth_texture_entry(2), // depth
                texture_entry(3),       // scene color
                texture_entry(4),       // HZB mip 0
                storage_texture_entry(5, wgpu::TextureFormat::Rgba16Float),
            ],
        });

        // Composite bind group layout: just the reflection texture
        let composite_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("prism-ssr-composite-bgl"),
            entries: &[texture_entry_fragment(0)],
        });

        // Trace compute pipeline
        let trace_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("prism-ssr-trace-pl"),
            bind_group_layouts: &[&trace_bgl],
            immediate_size: 0,
        });
        let trace_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("prism-ssr-trace-pipeline"),
            layout: Some(&trace_pl),
            module: &ssr_shader,
            entry_point: Some("ssr_trace"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // Composite render pipeline (additive blend)
        let composite_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("prism-ssr-composite-pl"),
            bind_group_layouts: &[&composite_bgl],
            immediate_size: 0,
        });
        let composite_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("prism-ssr-composite-pipeline"),
            layout: Some(&composite_pl),
            vertex: wgpu::VertexState {
                module: &composite_shader,
                entry_point: Some("vs_composite"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &composite_shader,
                entry_point: Some("fs_composite"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: HDR_FORMAT,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
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
            trace_pipeline,
            composite_pipeline,
            trace_bgl,
            composite_bgl,
            reflection_texture,
            reflection_view,
            uniform_buffer,
            width,
            height,
        }
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.reflection_texture = create_reflection_texture(device, width, height);
        self.reflection_view = self
            .reflection_texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.width = width;
        self.height = height;
    }

    pub fn write_uniforms(
        &self,
        queue: &wgpu::Queue,
        view: Mat4,
        projection: Mat4,
        width: u32,
        height: u32,
        frame_index: u32,
    ) {
        let uniforms = SsrUniforms {
            view: view.to_cols_array_2d(),
            inv_view: view.inverse().to_cols_array_2d(),
            projection: projection.to_cols_array_2d(),
            inv_projection: projection.inverse().to_cols_array_2d(),
            screen_size: [
                width as f32,
                height as f32,
                1.0 / width as f32,
                1.0 / height as f32,
            ],
            params: [
                50.0,  // max_distance (meters in view space)
                0.15,  // thickness (depth threshold for hit acceptance)
                2.0,   // stride (pixels per step)
                128.0, // max_steps
            ],
            params2: [
                0.95, // roughness_cutoff (skip very rough surfaces)
                0.1,  // fade_start (screen edge fade starts at 10% from edge)
                0.02, // fade_end (fully faded at 2% from edge)
                frame_index as f32,
            ],
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
    }

    /// Execute the SSR trace and composite passes.
    pub fn execute(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        normal_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        color_view: &wgpu::TextureView,
        hzb_view: &wgpu::TextureView,
    ) {
        // -- Trace pass (compute) --
        let trace_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("prism-ssr-trace-bg"),
            layout: &self.trace_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(normal_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(hzb_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&self.reflection_view),
                },
            ],
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("prism-ssr-trace-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.trace_pipeline);
            pass.set_bind_group(0, &trace_bg, &[]);
            pass.dispatch_workgroups(self.width.div_ceil(8), self.height.div_ceil(8), 1);
        }

        // -- Composite pass (additive blend onto scene color) --
        let composite_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("prism-ssr-composite-bg"),
            layout: &self.composite_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&self.reflection_view),
            }],
        });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("prism-ssr-composite-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_view,
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
    }
}

fn create_reflection_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("prism-ssr-reflection"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    })
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: Some(
                std::num::NonZeroU64::new(std::mem::size_of::<SsrUniforms>() as u64).unwrap(),
            ),
        },
        count: None,
    }
}

fn texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn texture_entry_fragment(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn depth_texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Depth,
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn storage_texture_entry(binding: u32, format: wgpu::TextureFormat) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    }
}

/// Inline WGSL for the SSR composite fullscreen pass (additive blend).
const SSR_COMPOSITE_WGSL: &str = r#"
@group(0) @binding(0) var ssr_tex: texture_2d<f32>;

struct FullscreenOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_composite(@builtin(vertex_index) vi: u32) -> FullscreenOut {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    var out: FullscreenOut;
    let pos = positions[vi];
    out.position = vec4<f32>(pos, 0.0, 1.0);
    out.uv = vec2<f32>(pos.x * 0.5 + 0.5, 0.5 - pos.y * 0.5);
    return out;
}

@fragment
fn fs_composite(input: FullscreenOut) -> @location(0) vec4<f32> {
    let pixel = vec2<i32>(input.position.xy);
    let reflection = textureLoad(ssr_tex, pixel, 0);
    return vec4<f32>(reflection.rgb, 1.0);
}
"#;
