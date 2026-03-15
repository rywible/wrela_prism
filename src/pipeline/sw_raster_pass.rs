use super::hw_raster_pass::{DispatchLists, VisibilityBuffer};

/// Software rasterizer pass (compute shader) for sub-pixel triangles.
///
/// Meshlets routed here by the cull pass have screen-space bounding sphere
/// diameter < 32 pixels. The compute shader uses edge functions + atomicMax
/// to write the visibility buffer.
pub struct SwRasterPass {
    pipeline: wgpu::ComputePipeline,
    dispatch_bind_group_layout: wgpu::BindGroupLayout,
}

impl SwRasterPass {
    pub fn new(
        device: &wgpu::Device,
        frame_bgl: &wgpu::BindGroupLayout,
        mesh_bgl: &wgpu::BindGroupLayout,
    ) -> Self {
        let dispatch_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("sw-raster-dispatch-bgl"),
                entries: &[
                    // sw_dispatch_list
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // visbuf (atomic storage)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sw-raster-shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/meshlet_sw_vis.wgsl").into(),
            ),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sw-raster-layout"),
            bind_group_layouts: &[frame_bgl, mesh_bgl, &dispatch_bind_group_layout],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("sw-raster-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("sw_rasterize"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            pipeline,
            dispatch_bind_group_layout,
        }
    }

    pub fn create_dispatch_bind_group(
        &self,
        device: &wgpu::Device,
        dispatch: &DispatchLists,
        vis_buffer: &VisibilityBuffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sw-raster-dispatch-bg"),
            layout: &self.dispatch_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: dispatch.sw_dispatch_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: vis_buffer.storage_buffer.as_entire_binding(),
                },
            ],
        })
    }

    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        frame_bg: &wgpu::BindGroup,
        mesh_bg: &wgpu::BindGroup,
        dispatch_bg: &wgpu::BindGroup,
        meshlet_count: u32,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("sw-raster-pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, frame_bg, &[]);
        pass.set_bind_group(1, mesh_bg, &[]);
        pass.set_bind_group(2, dispatch_bg, &[]);

        // Dispatch one workgroup per meshlet in the SW list
        // In production, use indirect dispatch from sw_count_buffer
        pass.dispatch_workgroups(meshlet_count.max(1), 1, 1);
    }
}
