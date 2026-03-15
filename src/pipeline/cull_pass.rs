use crate::meshlet::{GpuMeshletBuffers, MeshletDag};
use super::hw_raster_pass::DispatchLists;

/// CPU-side DAG traversal + GPU meshlet culling.
///
/// The DAG cut is computed on the CPU (simple iteration over groups).
/// Per-meshlet frustum + cone culling is done on the GPU via compute shader.
pub struct CullPass {
    /// Group queue buffer (CPU → GPU).
    group_queue_buffer: wgpu::Buffer,
    group_queue_count_buffer: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    cull_bind_group_layout: wgpu::BindGroupLayout,
    dispatch_bind_group_layout: wgpu::BindGroupLayout,
}

impl CullPass {
    pub fn new(
        device: &wgpu::Device,
        frame_bgl: &wgpu::BindGroupLayout,
    ) -> Self {
        let cull_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("cull-data-bgl"),
                entries: &[
                    super::storage_entry(0, true, wgpu::ShaderStages::COMPUTE), // meshlet_bounds
                    super::storage_entry(1, true, wgpu::ShaderStages::COMPUTE), // meshlet_groups
                    super::storage_entry(2, true, wgpu::ShaderStages::COMPUTE), // meshlet_descs
                ],
            });

        let dispatch_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("cull-dispatch-bgl"),
                entries: &[
                    super::storage_entry(0, false, wgpu::ShaderStages::COMPUTE), // hw_dispatch_list
                    super::storage_entry(1, false, wgpu::ShaderStages::COMPUTE), // hw_dispatch_count
                    super::storage_entry(2, false, wgpu::ShaderStages::COMPUTE), // sw_dispatch_list
                    super::storage_entry(3, false, wgpu::ShaderStages::COMPUTE), // sw_dispatch_count
                    super::storage_entry(4, true, wgpu::ShaderStages::COMPUTE),  // group_queue
                    super::storage_entry(5, true, wgpu::ShaderStages::COMPUTE),  // group_queue_count (uniform-like)
                ],
            });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("meshlet-cull-shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/meshlet_cull.wgsl").into(),
            ),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cull-pipeline-layout"),
            bind_group_layouts: &[frame_bgl, &cull_bind_group_layout, &dispatch_bind_group_layout],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("meshlet-cull-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("meshlet_cull"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let group_queue_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("group-queue"),
            size: 4096 * 4, // up to 4096 groups
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let group_queue_count_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("group-queue-count"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            group_queue_buffer,
            group_queue_count_buffer,
            pipeline,
            cull_bind_group_layout,
            dispatch_bind_group_layout,
        }
    }

    /// Perform CPU-side DAG cut and upload the group queue.
    ///
    /// Returns the number of groups to process on the GPU.
    /// Perform CPU-side DAG cut and write meshlet dispatch list directly.
    ///
    /// Bypasses the GPU cull shader — writes the HW dispatch list and count
    /// directly from the CPU for maximum reliability.
    pub fn cpu_dag_cut(
        &self,
        queue: &wgpu::Queue,
        dag: &MeshletDag,
        dispatch: &super::hw_raster_pass::DispatchLists,
    ) -> u32 {
        // Collect all meshlet indices from leaf groups (level 0)
        let mut meshlet_indices = Vec::new();

        for group in &dag.groups {
            if group.child_count == 0 {
                // Leaf group — emit all its meshlets
                for mi in group.meshlet_start..group.meshlet_start + group.meshlet_count {
                    meshlet_indices.push(mi);
                }
            }
        }

        // Deduplicate (groups may overlap)
        meshlet_indices.sort_unstable();
        meshlet_indices.dedup();

        let count = meshlet_indices.len().min(dispatch.max_meshlets as usize) as u32;

        // Write meshlet indices directly to HW dispatch list
        if count > 0 {
            queue.write_buffer(
                &dispatch.hw_dispatch_buffer,
                0,
                bytemuck::cast_slice(&meshlet_indices[..count as usize]),
            );
        }

        // Write the count directly to HW count buffer
        queue.write_buffer(
            &dispatch.hw_count_buffer,
            0,
            bytemuck::bytes_of(&count),
        );

        count
    }

    pub fn create_cull_bind_group(
        &self,
        device: &wgpu::Device,
        buffers: &GpuMeshletBuffers,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cull-data-bg"),
            layout: &self.cull_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.meshlet_bounds_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.group_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
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
            label: Some("cull-dispatch-bg"),
            layout: &self.dispatch_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: dispatch.hw_dispatch_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dispatch.hw_count_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: dispatch.sw_dispatch_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dispatch.sw_count_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.group_queue_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.group_queue_count_buffer.as_entire_binding(),
                },
            ],
        })
    }

    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        frame_bg: &wgpu::BindGroup,
        cull_bg: &wgpu::BindGroup,
        dispatch_bg: &wgpu::BindGroup,
        group_count: u32,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("meshlet-cull-pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, frame_bg, &[]);
        pass.set_bind_group(1, cull_bg, &[]);
        pass.set_bind_group(2, dispatch_bg, &[]);

        let workgroups = (group_count + 63) / 64;
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
}

