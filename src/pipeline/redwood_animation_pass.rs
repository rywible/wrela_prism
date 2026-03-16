use bytemuck::{Pod, Zeroable};

use crate::art_direction::ArtDirectionUniforms;
use crate::runtime_scene::{AnimatedRedwoodGpu, AnimatedRedwoodLevelRange};
use crate::scene::WindSettings;

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct RedwoodAnimationUniforms {
    pub level_start: u32,
    pub level_count: u32,
    pub node_count: u32,
    pub vertex_count: u32,
    pub time: f32,
    pub dt: f32,
    pub _pad0: [f32; 2],
    pub wind_direction: [f32; 4],
    pub wind_profile: [f32; 4],
    pub style_profile: [f32; 4],
}

pub struct RedwoodAnimationPass {
    uniform_buffer: wgpu::Buffer,
    common_bgl: wgpu::BindGroupLayout,
    rest_bgl: wgpu::BindGroupLayout,
    output_bgl: wgpu::BindGroupLayout,
    sim_pipeline: wgpu::ComputePipeline,
    deform_pipeline: wgpu::ComputePipeline,
}

impl RedwoodAnimationPass {
    pub fn new(device: &wgpu::Device) -> Self {
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("redwood-animation-uniforms"),
            size: std::mem::size_of::<RedwoodAnimationUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let common_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("redwood-animation-common-bgl"),
            entries: &[
                uniform_entry(0),
                super::storage_entry(1, true, wgpu::ShaderStages::COMPUTE),
                super::storage_entry(2, false, wgpu::ShaderStages::COMPUTE),
            ],
        });
        let rest_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("redwood-animation-rest-bgl"),
            entries: &[super::storage_entry(0, true, wgpu::ShaderStages::COMPUTE)],
        });
        let output_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("redwood-animation-output-bgl"),
            entries: &[super::storage_entry(0, false, wgpu::ShaderStages::COMPUTE)],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("redwood-animation-shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/redwood_animation.wgsl").into(),
            ),
        });

        let sim_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("redwood-animation-sim-layout"),
            bind_group_layouts: &[&common_bgl],
            immediate_size: 0,
        });
        let deform_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("redwood-animation-deform-layout"),
            bind_group_layouts: &[&common_bgl, &rest_bgl, &output_bgl],
            immediate_size: 0,
        });

        let sim_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("redwood-animation-sim-pipeline"),
            layout: Some(&sim_layout),
            module: &shader,
            entry_point: Some("simulate_nodes"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let deform_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("redwood-animation-deform-pipeline"),
            layout: Some(&deform_layout),
            module: &shader,
            entry_point: Some("deform_vertices"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            uniform_buffer,
            common_bgl,
            rest_bgl,
            output_bgl,
            sim_pipeline,
            deform_pipeline,
        }
    }

    fn write_uniforms(
        &self,
        queue: &wgpu::Queue,
        level_range: AnimatedRedwoodLevelRange,
        vertex_count: u32,
        time: f32,
        dt: f32,
        wind: &WindSettings,
        art: &ArtDirectionUniforms,
        node_count: u32,
    ) {
        let dir = wind.direction.normalize_or_zero();
        let uniforms = RedwoodAnimationUniforms {
            level_start: level_range.start,
            level_count: level_range.count,
            node_count,
            vertex_count,
            time,
            dt,
            _pad0: [0.0; 2],
            wind_direction: [dir.x, dir.y, if wind.frozen { 1.0 } else { 0.0 }, 0.0],
            wind_profile: [
                wind.mean_speed,
                wind.gust_strength,
                wind.gust_frequency,
                wind.turbulence,
            ],
            style_profile: [
                art.wind_amplitude_scale,
                art.wind_response_scale,
                art.wind_gust_sharpness,
                art.foliage_transmission_boost,
            ],
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
    }

    fn create_common_bind_group(
        &self,
        device: &wgpu::Device,
        animated: &AnimatedRedwoodGpu,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("redwood-animation-common-bg"),
            layout: &self.common_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: animated.rig_node_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: animated.rig_state_buffer.as_entire_binding(),
                },
            ],
        })
    }

    fn create_rest_bind_group(
        &self,
        device: &wgpu::Device,
        buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("redwood-animation-rest-bg"),
            layout: &self.rest_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        })
    }

    fn create_output_bind_group(
        &self,
        device: &wgpu::Device,
        buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("redwood-animation-output-bg"),
            layout: &self.output_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        })
    }

    pub fn encode(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        animated: &AnimatedRedwoodGpu,
        dag_vertex_buffer: &wgpu::Buffer,
        time: f32,
        dt: f32,
        wind: &WindSettings,
        art: &ArtDirectionUniforms,
    ) {
        if dt <= 0.0 || animated.level_ranges.is_empty() {
            return;
        }

        let node_count = animated.level_ranges.iter().map(|range| range.count).sum();
        let common_bg = self.create_common_bind_group(device, animated);
        let dag_rest_bg = self.create_rest_bind_group(device, &animated.rest_vertex_buffer);
        let dag_out_bg = self.create_output_bind_group(device, dag_vertex_buffer);
        let shadow_rest_bg =
            self.create_rest_bind_group(device, &animated.shadow_rest_vertex_buffer);
        let shadow_out_bg = self.create_output_bind_group(device, &animated.shadow_vertex_buffer);

        for level_range in &animated.level_ranges {
            self.write_uniforms(
                queue,
                level_range.clone(),
                animated.rest_vertex_count,
                time,
                dt,
                wind,
                art,
                node_count,
            );
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("redwood-animation-sim-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.sim_pipeline);
            pass.set_bind_group(0, &common_bg, &[]);
            let groups = (level_range.count + 63) / 64;
            pass.dispatch_workgroups(groups.max(1), 1, 1);
        }

        self.write_uniforms(
            queue,
            AnimatedRedwoodLevelRange {
                start: 0,
                count: node_count,
            },
            animated.rest_vertex_count,
            time,
            dt,
            wind,
            art,
            node_count,
        );

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("redwood-animation-dag-deform-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.deform_pipeline);
            pass.set_bind_group(0, &common_bg, &[]);
            pass.set_bind_group(1, &dag_rest_bg, &[]);
            pass.set_bind_group(2, &dag_out_bg, &[]);
            let groups = (animated.rest_vertex_count + 63) / 64;
            pass.dispatch_workgroups(groups.max(1), 1, 1);
        }

        self.write_uniforms(
            queue,
            AnimatedRedwoodLevelRange {
                start: 0,
                count: node_count,
            },
            animated.shadow_vertex_count,
            time,
            dt,
            wind,
            art,
            node_count,
        );

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("redwood-animation-shadow-deform-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.deform_pipeline);
            pass.set_bind_group(0, &common_bg, &[]);
            pass.set_bind_group(1, &shadow_rest_bg, &[]);
            pass.set_bind_group(2, &shadow_out_bg, &[]);
            let groups = (animated.shadow_vertex_count + 63) / 64;
            pass.dispatch_workgroups(groups.max(1), 1, 1);
        }
    }
}
