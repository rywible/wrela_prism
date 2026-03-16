use super::HDR_FORMAT;

const BLOOM_LEVELS: usize = 5;
// Uniform buffer offset alignment (WebGPU max for min_uniform_buffer_offset_alignment)
const UNIFORM_ALIGN: u64 = 256;
// 1 threshold + 4 downsample + 4 upsample = 9 slots
const BLOOM_UNIFORM_SLOTS: u64 = 9;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BloomUniforms {
    screen_size: [f32; 4], // full_width, full_height, level_width, level_height
    params: [f32; 4],      // threshold, intensity, bloom_softness, 0
    bloom_tint: [f32; 4],  // rgb = tint color (default 1,1,1), w = unused
}

pub struct BloomPass {
    threshold_pipeline: wgpu::ComputePipeline,
    downsample_pipeline: wgpu::ComputePipeline,
    upsample_pipeline: wgpu::ComputePipeline,
    composite_pipeline: wgpu::RenderPipeline,
    threshold_bgl: wgpu::BindGroupLayout,
    downsample_bgl: wgpu::BindGroupLayout,
    upsample_bgl: wgpu::BindGroupLayout,
    composite_bgl: wgpu::BindGroupLayout,
    // Two textures per level for ping-pong during upsample
    level_textures: Vec<(wgpu::Texture, wgpu::TextureView)>,
    level_alt_textures: Vec<(wgpu::Texture, wgpu::TextureView)>,
    level_sizes: Vec<(u32, u32)>,
    uniform_buffer: wgpu::Buffer,
    sampler: wgpu::Sampler,
}

impl BloomPass {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("prism-bloom-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/bloom.wgsl").into()),
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("prism-bloom-uniforms"),
            size: UNIFORM_ALIGN * BLOOM_UNIFORM_SLOTS,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Compute level sizes: half, quarter, eighth, sixteenth, thirty-second
        let level_sizes = compute_level_sizes(width, height);
        let level_textures = create_level_textures(device, &level_sizes, "bloom");
        let level_alt_textures = create_level_textures(device, &level_sizes, "bloom-alt");

        // Threshold bind group layout: uniforms + scene_color + bloom_out
        let threshold_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("prism-bloom-threshold-bgl"),
            entries: &[
                uniform_entry(0),
                texture_entry(1),
                storage_entry(2, HDR_FORMAT),
            ],
        });

        // Downsample bind group layout: uniforms + input + output
        let downsample_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("prism-bloom-downsample-bgl"),
            entries: &[
                uniform_entry(0),
                texture_entry(1),
                storage_entry(2, HDR_FORMAT),
            ],
        });

        // Upsample bind group layout: uniforms + input (smaller) + dest (read) + output (write)
        let upsample_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("prism-bloom-upsample-bgl"),
            entries: &[
                uniform_entry(0),
                texture_entry(1),
                texture_entry(2),
                storage_entry(3, HDR_FORMAT),
            ],
        });

        let threshold_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("prism-bloom-threshold-pl"),
            bind_group_layouts: &[&threshold_bgl],
            immediate_size: 0,
        });
        let downsample_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("prism-bloom-downsample-pl"),
            bind_group_layouts: &[&downsample_bgl],
            immediate_size: 0,
        });
        let upsample_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("prism-bloom-upsample-pl"),
            bind_group_layouts: &[&upsample_bgl],
            immediate_size: 0,
        });

        let threshold_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("prism-bloom-threshold-pipeline"),
            layout: Some(&threshold_pl),
            module: &shader,
            entry_point: Some("bloom_threshold"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let downsample_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("prism-bloom-downsample-pipeline"),
                layout: Some(&downsample_pl),
                module: &shader,
                entry_point: Some("bloom_downsample"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let upsample_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("prism-bloom-upsample-pipeline"),
            layout: Some(&upsample_pl),
            module: &shader,
            entry_point: Some("bloom_upsample"),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
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

        let composite_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("prism-bloom-composite-pl"),
            bind_group_layouts: &[&composite_bgl],
            immediate_size: 0,
        });

        let composite_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("prism-bloom-composite-shader"),
            source: wgpu::ShaderSource::Wgsl(BLOOM_COMPOSITE_SHADER.into()),
        });

        let composite_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
            downsample_pipeline,
            upsample_pipeline,
            composite_pipeline,
            threshold_bgl,
            downsample_bgl,
            upsample_bgl,
            composite_bgl,
            level_textures,
            level_alt_textures,
            level_sizes,
            uniform_buffer,
            sampler,
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
        bloom_tint: [f32; 3],
        bloom_softness: f32,
    ) {
        let uniform_size = std::mem::size_of::<BloomUniforms>();
        let tint = [bloom_tint[0], bloom_tint[1], bloom_tint[2], 0.0];
        let params = [0.5f32, 0.35, bloom_softness, 0.0];

        // Pre-write all per-pass uniform data at 256-byte aligned offsets.
        // queue.write_buffer at the SAME offset races (last wins before submit),
        // so each pass must use a DIFFERENT offset.
        // Slot 0: threshold, Slots 1-4: downsample, Slots 5-8: upsample

        // Slot 0: threshold
        let (lw, lh) = self.level_sizes[0];
        let threshold_u = BloomUniforms {
            screen_size: [full_width as f32, full_height as f32, lw as f32, lh as f32],
            params,
            bloom_tint: tint,
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&threshold_u));

        // Slots 1-4: downsample levels
        for i in 0..BLOOM_LEVELS - 1 {
            let (out_w, out_h) = self.level_sizes[i + 1];
            let u = BloomUniforms {
                screen_size: [0.0, 0.0, out_w as f32, out_h as f32],
                params,
                bloom_tint: tint,
            };
            queue.write_buffer(
                &self.uniform_buffer,
                UNIFORM_ALIGN * (1 + i as u64),
                bytemuck::bytes_of(&u),
            );
        }

        // Slots 5-8: upsample levels (i = 3, 2, 1, 0)
        for (slot, i) in (0..BLOOM_LEVELS - 1).rev().enumerate() {
            let (out_w, out_h) = self.level_sizes[i];
            let u = BloomUniforms {
                screen_size: [0.0, 0.0, out_w as f32, out_h as f32],
                params,
                bloom_tint: tint,
            };
            queue.write_buffer(
                &self.uniform_buffer,
                UNIFORM_ALIGN * (5 + slot as u64),
                bytemuck::bytes_of(&u),
            );
        }

        let binding_size = std::num::NonZeroU64::new(uniform_size as u64).unwrap();

        // Helper: create a BufferBinding at a given slot
        let buf_at = |slot: u64| -> wgpu::BindingResource {
            wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &self.uniform_buffer,
                offset: UNIFORM_ALIGN * slot,
                size: Some(binding_size),
            })
        };

        // 1) Threshold: scene_color → level 0 (slot 0)
        {
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("prism-bloom-threshold-bg"),
                layout: &self.threshold_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buf_at(0),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(scene_color_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&self.level_textures[0].1),
                    },
                ],
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("prism-bloom-threshold"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.threshold_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(lw.div_ceil(8), lh.div_ceil(8), 1);
        }

        // 2) Downsample: level 0 → 1 → 2 → 3 → 4 (slots 1-4)
        for i in 0..BLOOM_LEVELS - 1 {
            let (out_w, out_h) = self.level_sizes[i + 1];
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("prism-bloom-downsample-bg"),
                layout: &self.downsample_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buf_at(1 + i as u64),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.level_textures[i].1),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&self.level_textures[i + 1].1),
                    },
                ],
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("prism-bloom-downsample"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.downsample_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(out_w.div_ceil(8), out_h.div_ceil(8), 1);
        }

        // 3) Upsample: level 4 → 3 → 2 → 1 → 0 (slots 5-8)
        for (slot, i) in (0..BLOOM_LEVELS - 1).rev().enumerate() {
            let (out_w, out_h) = self.level_sizes[i];

            let src_view = if i == BLOOM_LEVELS - 2 {
                &self.level_textures[i + 1].1
            } else {
                &self.level_alt_textures[i + 1].1
            };

            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("prism-bloom-upsample-bg"),
                layout: &self.upsample_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buf_at(5 + slot as u64),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(src_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&self.level_textures[i].1),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&self.level_alt_textures[i].1),
                    },
                ],
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("prism-bloom-upsample"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.upsample_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(out_w.div_ceil(8), out_h.div_ceil(8), 1);
        }

        // 4) Composite: additive blend level_alt_textures[0] onto scene_color
        // Reuse slot 0 (has correct bloom_tint)
        let composite_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("prism-bloom-composite-bg"),
            layout: &self.composite_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.level_alt_textures[0].1),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_at(0),
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

fn compute_level_sizes(width: u32, height: u32) -> Vec<(u32, u32)> {
    let mut sizes = Vec::with_capacity(BLOOM_LEVELS);
    let mut w = width;
    let mut h = height;
    for _ in 0..BLOOM_LEVELS {
        w = (w / 2).max(1);
        h = (h / 2).max(1);
        sizes.push((w, h));
    }
    sizes
}

fn create_level_textures(
    device: &wgpu::Device,
    sizes: &[(u32, u32)],
    prefix: &str,
) -> Vec<(wgpu::Texture, wgpu::TextureView)> {
    sizes
        .iter()
        .enumerate()
        .map(|(i, &(w, h))| {
            let texture = create_bloom_texture(device, w, h, &format!("{prefix}-L{i}"));
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            (texture, view)
        })
        .collect()
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
@group(0) @binding(2)
var<uniform> bloom_u: BloomCompUniforms;

struct BloomCompUniforms {
    screen_size: vec4<f32>,
    params: vec4<f32>,
    bloom_tint: vec4<f32>,
};

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
    let tinted = bloom * bloom_u.bloom_tint.rgb;
    return vec4<f32>(tinted * 0.28, 0.0);
}
"#;
