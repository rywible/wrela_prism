use super::HDR_FORMAT;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BloomUniforms {
    screen_size: [f32; 4], // full_width, full_height, half_width, half_height
    params: [f32; 4],      // threshold, intensity, 0, 0
}

pub struct BloomPass {
    threshold_pipeline: wgpu::ComputePipeline,
    blur_h_pipeline: wgpu::ComputePipeline,
    blur_v_pipeline: wgpu::ComputePipeline,
    composite_pipeline: wgpu::RenderPipeline,
    threshold_bgl: wgpu::BindGroupLayout,
    blur_bgl: wgpu::BindGroupLayout,
    composite_bgl: wgpu::BindGroupLayout,
    _bloom_a: wgpu::Texture,
    bloom_a_view: wgpu::TextureView,
    _bloom_b: wgpu::Texture,
    bloom_b_view: wgpu::TextureView,
    uniform_buffer: wgpu::Buffer,
    sampler: wgpu::Sampler,
    half_width: u32,
    half_height: u32,
}

impl BloomPass {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let half_width = (width / 2).max(1);
        let half_height = (height / 2).max(1);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("prism-bloom-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/bloom.wgsl").into()),
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("prism-bloom-uniforms"),
            size: std::mem::size_of::<BloomUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bloom_a = create_bloom_texture(device, half_width, half_height, "prism-bloom-a");
        let bloom_a_view = bloom_a.create_view(&wgpu::TextureViewDescriptor::default());
        let bloom_b = create_bloom_texture(device, half_width, half_height, "prism-bloom-b");
        let bloom_b_view = bloom_b.create_view(&wgpu::TextureViewDescriptor::default());

        // Threshold bind group layout: uniforms + scene_color + bloom_out
        let threshold_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("prism-bloom-threshold-bgl"),
            entries: &[
                uniform_entry(0),
                texture_entry(1),
                storage_entry(2, HDR_FORMAT),
            ],
        });

        // Blur bind group layout: uniforms + input + output
        let blur_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("prism-bloom-blur-bgl"),
            entries: &[
                uniform_entry(0),
                texture_entry(1),
                storage_entry(2, HDR_FORMAT),
            ],
        });

        let threshold_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("prism-bloom-threshold-pl"),
            bind_group_layouts: &[&threshold_bgl],
            immediate_size: 0,
        });
        let blur_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("prism-bloom-blur-pl"),
            bind_group_layouts: &[&blur_bgl],
            immediate_size: 0,
        });

        let threshold_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("prism-bloom-threshold-pipeline"),
                layout: Some(&threshold_pl),
                module: &shader,
                entry_point: Some("bloom_threshold"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let blur_h_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("prism-bloom-blur-h-pipeline"),
            layout: Some(&blur_pl),
            module: &shader,
            entry_point: Some("bloom_blur_h"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let blur_v_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("prism-bloom-blur-v-pipeline"),
            layout: Some(&blur_pl),
            module: &shader,
            entry_point: Some("bloom_blur_v"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // Composite render pipeline — additive blend fullscreen triangle
        let composite_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("prism-bloom-composite-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let composite_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("prism-bloom-composite-pl"),
            bind_group_layouts: &[&composite_bgl],
            immediate_size: 0,
        });

        let composite_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("prism-bloom-composite-shader"),
            source: wgpu::ShaderSource::Wgsl(BLOOM_COMPOSITE_SHADER.into()),
        });

        let composite_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("prism-bloom-composite-pipeline"),
                layout: Some(&composite_pl_layout),
                vertex: wgpu::VertexState {
                    module: &composite_shader,
                    entry_point: Some("vs_fullscreen"),
                    buffers: &[],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &composite_shader,
                    entry_point: Some("fs_bloom_composite"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: HDR_FORMAT,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::One,
                                dst_factor: wgpu::BlendFactor::One,
                                operation: wgpu::BlendOperation::Add,
                            },
                            alpha: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::Zero,
                                dst_factor: wgpu::BlendFactor::One,
                                operation: wgpu::BlendOperation::Add,
                            },
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

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("prism-bloom-sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        Self {
            threshold_pipeline,
            blur_h_pipeline,
            blur_v_pipeline,
            composite_pipeline,
            threshold_bgl,
            blur_bgl,
            composite_bgl,
            _bloom_a: bloom_a,
            bloom_a_view,
            _bloom_b: bloom_b,
            bloom_b_view,
            uniform_buffer,
            sampler,
            half_width,
            half_height,
        }
    }

    pub fn execute(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        scene_color_view: &wgpu::TextureView,
        scene_color_target: &wgpu::TextureView,
        full_width: u32,
        full_height: u32,
    ) {
        let uniforms = BloomUniforms {
            screen_size: [
                full_width as f32,
                full_height as f32,
                self.half_width as f32,
                self.half_height as f32,
            ],
            params: [1.0, 0.3, 0.0, 0.0], // threshold=1.0, intensity=0.3
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        let wg_x = self.half_width.div_ceil(8);
        let wg_y = self.half_height.div_ceil(8);

        // 1) Threshold: scene_color → bloom_a
        let threshold_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("prism-bloom-threshold-bg"),
            layout: &self.threshold_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(scene_color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&self.bloom_a_view),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("prism-bloom-threshold"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.threshold_pipeline);
            pass.set_bind_group(0, &threshold_bg, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        // 2) Horizontal blur: bloom_a → bloom_b
        let blur_h_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("prism-bloom-blur-h-bg"),
            layout: &self.blur_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.bloom_a_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&self.bloom_b_view),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("prism-bloom-blur-h"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.blur_h_pipeline);
            pass.set_bind_group(0, &blur_h_bg, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        // 3) Vertical blur: bloom_b → bloom_a
        let blur_v_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("prism-bloom-blur-v-bg"),
            layout: &self.blur_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.bloom_b_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&self.bloom_a_view),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("prism-bloom-blur-v"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.blur_v_pipeline);
            pass.set_bind_group(0, &blur_v_bg, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        // 4) Composite: additive blend bloom_a onto scene_color
        let composite_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("prism-bloom-composite-bg"),
            layout: &self.composite_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.bloom_a_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("prism-bloom-composite"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: scene_color_target,
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

fn create_bloom_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    label: &str,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: HDR_FORMAT,
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
                std::num::NonZeroU64::new(std::mem::size_of::<BloomUniforms>() as u64).unwrap(),
            ),
        },
        count: None,
    }
}

fn texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn storage_entry(binding: u32, format: wgpu::TextureFormat) -> wgpu::BindGroupLayoutEntry {
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

const BLOOM_COMPOSITE_SHADER: &str = r#"
@group(0) @binding(0)
var bloom_tex: texture_2d<f32>;
@group(0) @binding(1)
var bloom_sampler: sampler;

struct FullscreenOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) vi: u32) -> FullscreenOut {
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
fn fs_bloom_composite(input: FullscreenOut) -> @location(0) vec4<f32> {
    let bloom = textureSample(bloom_tex, bloom_sampler, input.uv).rgb;
    return vec4<f32>(bloom * 0.3, 0.0);
}
"#;
