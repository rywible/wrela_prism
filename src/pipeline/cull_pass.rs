use super::hw_raster_pass::DispatchLists;
use crate::meshlet::GpuMeshletBuffers;
#[cfg(feature = "cpu_dag_cut")]
use crate::meshlet::MeshletDag;

/// CPU-side adaptive DAG traversal + GPU per-meshlet culling.
///
/// Each frame the CPU traverses the meshlet DAG from root groups, projecting
/// each group's simplification error to screen-space pixels. Groups whose error
/// is below the threshold are selected; otherwise their children are visited.
/// Selected group indices are uploaded to the GPU, where a compute shader does
/// per-meshlet frustum culling, writing survivors to the HW dispatch list.
///
/// Two-phase occlusion culling eliminates 1-frame popping:
///   Phase 1: Cull with prev-frame HZB → raster → rebuild HZB (mid-frame)
///   Phase 2: Re-test HZB-rejected meshlets against fresh HZB → raster survivors
pub struct CullPass {
    /// Group queue buffer (CPU → GPU).
    group_queue_buffer: wgpu::Buffer,
    group_queue_count_buffer: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    phase2_pipeline: wgpu::ComputePipeline,
    cull_bind_group_layout: wgpu::BindGroupLayout,
    dispatch_bind_group_layout: wgpu::BindGroupLayout,
    /// Phase 2 reject list: meshlet indices that failed HZB in phase 1.
    phase2_queue_buffer: wgpu::Buffer,
    /// Atomic counter for the reject list.
    phase2_count_buffer: wgpu::Buffer,
    /// Scratch buffer for binding slot 7 in the phase2 dispatch bind group.
    /// Prevents aliasing phase2_queue_buffer as both read-only (binding 4) and
    /// read_write (binding 7) in the same bind group, which WebGPU forbids.
    phase2_scratch_list_buffer: wgpu::Buffer,
    /// Scratch buffer for binding slot 8 in the phase2 dispatch bind group.
    phase2_scratch_count_buffer: wgpu::Buffer,
}

impl CullPass {
    pub fn new(device: &wgpu::Device, frame_bgl: &wgpu::BindGroupLayout) -> Self {
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
                    super::storage_entry(5, true, wgpu::ShaderStages::COMPUTE), // group_queue_count (uniform-like)
                    // HZB texture (previous frame for P1, fresh mid-frame for P2)
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // Phase 2 reject list (written in P1, read in P2)
                    super::storage_entry(7, false, wgpu::ShaderStages::COMPUTE), // phase2_reject_list
                    super::storage_entry(8, false, wgpu::ShaderStages::COMPUTE), // phase2_reject_count
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
            bind_group_layouts: &[
                frame_bgl,
                &cull_bind_group_layout,
                &dispatch_bind_group_layout,
            ],
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

        let phase2_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("meshlet-cull-phase2-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("meshlet_cull_phase2"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let group_queue_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("group-queue"),
            size: 16384 * 4, // up to 16384 groups
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let group_queue_count_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("group-queue-count"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Phase 2 reject list: meshlets HZB-rejected in phase 1, up to 65536 entries
        let phase2_queue_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("phase2-reject-list"),
            size: 65536 * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let phase2_count_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("phase2-reject-count"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Small scratch buffers used only to satisfy the phase2 dispatch bind group
        // layout at slots 7 and 8 (the phase2_reject_list / phase2_reject_count
        // write targets). The phase2 shader never writes to these slots, so a
        // single-element buffer is sufficient. This avoids the WebGPU validation
        // hazard that would arise from binding phase2_queue_buffer as both
        // read-only (binding 4) and read_write (binding 7) in the same bind group.
        let phase2_scratch_list_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("phase2-scratch-list"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let phase2_scratch_count_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("phase2-scratch-count"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        Self {
            group_queue_buffer,
            group_queue_count_buffer,
            pipeline,
            phase2_pipeline,
            cull_bind_group_layout,
            dispatch_bind_group_layout,
            phase2_queue_buffer,
            phase2_count_buffer,
            phase2_scratch_list_buffer,
            phase2_scratch_count_buffer,
        }
    }

    /// Perform adaptive CPU-side DAG cut and upload the selected group queue.
    ///
    /// Traverses the DAG top-down from root groups. For each group, projects its
    /// simplification error to screen-space pixels. If the error is below the
    /// threshold (or the group is a leaf), the group is selected for rendering.
    /// Otherwise its children are visited recursively.
    ///
    /// Returns the number of selected groups (to dispatch the GPU cull shader).
    ///
    /// Retained for debugging/validation. The default code path now uses
    /// `DagTraversePass` for GPU-driven DAG traversal.
    #[cfg(feature = "cpu_dag_cut")]
    pub fn cpu_dag_cut_adaptive(
        &self,
        queue: &wgpu::Queue,
        dag: &MeshletDag,
        camera_position: glam::Vec3,
        screen_height: f32,
        fov_factor: f32,
        error_threshold_px: f32,
    ) -> u32 {
        if dag.groups.is_empty() {
            queue.write_buffer(&self.group_queue_count_buffer, 0, bytemuck::bytes_of(&0u32));
            return 0;
        }

        let mut selected_groups: Vec<u32> = Vec::new();
        let mut stack = dag.root_group_indices();

        while let Some(group_idx) = stack.pop() {
            let group = &dag.groups[group_idx as usize];
            let center = glam::Vec3::from_array(group.bounds.center);
            let dist = camera_position.distance(center);

            let projected_error = if dist < 0.001 {
                1000.0
            } else {
                group.error * screen_height * fov_factor / dist
            };

            let is_leaf = group.child_count == 0;
            let should_render = is_leaf || projected_error < error_threshold_px;

            if should_render {
                selected_groups.push(group_idx);
            } else {
                // Recurse to finer-detail children
                for ci in 0..group.child_count {
                    stack.push(group.child_start + ci);
                }
            }
        }

        let count = selected_groups
            .len()
            .min(self.group_queue_buffer.size() as usize / 4) as u32;

        if count > 0 {
            queue.write_buffer(
                &self.group_queue_buffer,
                0,
                bytemuck::cast_slice(&selected_groups[..count as usize]),
            );
        }
        queue.write_buffer(
            &self.group_queue_count_buffer,
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

    /// Create the dispatch bind group for phase 1 culling.
    /// Binds hw/sw dispatch lists, group queue, HZB, and phase2 reject buffers.
    #[cfg(feature = "cpu_dag_cut")]
    pub fn create_dispatch_bind_group(
        &self,
        device: &wgpu::Device,
        dispatch: &DispatchLists,
        hzb_view: &wgpu::TextureView,
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
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(hzb_view),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: self.phase2_queue_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: self.phase2_count_buffer.as_entire_binding(),
                },
            ],
        })
    }

    /// Create the dispatch bind group for GPU-driven DAG traversal mode.
    ///
    /// Binds the DAG traverse pass's output buffers as the group queue input
    /// instead of the CPU-uploaded buffers. All other bindings are the same.
    pub fn create_dispatch_bind_group_gpu_driven(
        &self,
        device: &wgpu::Device,
        dispatch: &DispatchLists,
        hzb_view: &wgpu::TextureView,
        gpu_group_queue: &wgpu::Buffer,
        gpu_group_count: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cull-dispatch-bg-gpu-driven"),
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
                // GPU-produced group queue from DAG traverse pass
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: gpu_group_queue.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: gpu_group_count.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(hzb_view),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: self.phase2_queue_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: self.phase2_count_buffer.as_entire_binding(),
                },
            ],
        })
    }

    /// Encode phase 1 cull pass using indirect dispatch (GPU-driven DAG traversal mode).
    ///
    /// Instead of a CPU-provided group count, reads the workgroup count from
    /// the indirect args buffer produced by the DAG traverse pass.
    pub fn encode_indirect(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        frame_bg: &wgpu::BindGroup,
        cull_bg: &wgpu::BindGroup,
        dispatch_bg: &wgpu::BindGroup,
        indirect_buffer: &wgpu::Buffer,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("meshlet-cull-phase1-indirect"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, frame_bg, &[]);
        pass.set_bind_group(1, cull_bg, &[]);
        pass.set_bind_group(2, dispatch_bg, &[]);

        pass.dispatch_workgroups_indirect(indirect_buffer, 0);
    }

    /// Create the dispatch bind group for phase 2 culling.
    ///
    /// Re-uses the same layout but swaps bindings so the phase2 reject list
    /// is bound as the "group_queue" input (binding 4) and the phase2 reject
    /// count as "group_queue_count" (binding 5). The HZB view points to the
    /// fresh mid-frame HZB. hw_dispatch_list/count are the output targets.
    pub fn create_phase2_dispatch_bind_group(
        &self,
        device: &wgpu::Device,
        dispatch: &DispatchLists,
        hzb_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cull-phase2-dispatch-bg"),
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
                // Phase 2 reject list as input (re-bound to group_queue slot)
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.phase2_queue_buffer.as_entire_binding(),
                },
                // Phase 2 reject count as input (re-bound to group_queue_count slot)
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.phase2_count_buffer.as_entire_binding(),
                },
                // Fresh mid-frame HZB
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(hzb_view),
                },
                // Scratch buffers for slots 7/8: the phase2 shader never writes to
                // these, but they must be bound to satisfy the layout. Using dedicated
                // scratch buffers avoids aliasing phase2_queue_buffer / phase2_count_buffer
                // as both read-only (bindings 4/5) and read_write (bindings 7/8) within
                // the same bind group, which WebGPU validation forbids.
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: self.phase2_scratch_list_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: self.phase2_scratch_count_buffer.as_entire_binding(),
                },
            ],
        })
    }

    /// Clear the phase 2 reject count to zero.
    pub fn clear_phase2(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.clear_buffer(&self.phase2_count_buffer, 0, None);
    }

    /// Encode phase 1 cull pass (uses prev-frame HZB).
    #[cfg(feature = "cpu_dag_cut")]
    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        frame_bg: &wgpu::BindGroup,
        cull_bg: &wgpu::BindGroup,
        dispatch_bg: &wgpu::BindGroup,
        group_count: u32,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("meshlet-cull-phase1"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, frame_bg, &[]);
        pass.set_bind_group(1, cull_bg, &[]);
        pass.set_bind_group(2, dispatch_bg, &[]);

        let workgroups = group_count.div_ceil(64);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }

    /// Encode phase 2 cull pass (uses fresh mid-frame HZB).
    ///
    /// `reject_count` is the number of meshlets in the phase2 reject list.
    /// The phase2 dispatch bind group must have been created with
    /// `create_phase2_dispatch_bind_group()`.
    pub fn encode_phase2(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        frame_bg: &wgpu::BindGroup,
        cull_bg: &wgpu::BindGroup,
        phase2_dispatch_bg: &wgpu::BindGroup,
        reject_count: u32,
    ) {
        if reject_count == 0 {
            return;
        }

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("meshlet-cull-phase2"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.phase2_pipeline);
        pass.set_bind_group(0, frame_bg, &[]);
        pass.set_bind_group(1, cull_bg, &[]);
        pass.set_bind_group(2, phase2_dispatch_bg, &[]);

        let workgroups = reject_count.div_ceil(64);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
}

#[cfg(test)]
mod tests {
    use crate::meshlet::bounds::MeshletBounds;
    use crate::meshlet::{MeshletDag, MeshletGroup};

    fn bounds() -> MeshletBounds {
        MeshletBounds {
            center: [0.0, 0.0, 0.0],
            radius: 1.0,
            cone_axis: [0.0, 1.0, 0.0],
            cone_cutoff: 1.0,
        }
    }

    #[test]
    fn root_group_detection_handles_merged_dags_with_different_heights() {
        let dag = MeshletDag {
            meshlets: Vec::new(),
            meshlet_vertices: Vec::new(),
            meshlet_triangles: Vec::new(),
            groups: vec![
                MeshletGroup {
                    meshlet_start: 0,
                    meshlet_count: 1,
                    child_start: 0,
                    child_count: 0,
                    error: 0.0,
                    bounds: bounds(),
                    level: 0,
                },
                MeshletGroup {
                    meshlet_start: 1,
                    meshlet_count: 1,
                    child_start: 0,
                    child_count: 0,
                    error: 0.0,
                    bounds: bounds(),
                    level: 0,
                },
                MeshletGroup {
                    meshlet_start: 2,
                    meshlet_count: 1,
                    child_start: 0,
                    child_count: 2,
                    error: 1.0,
                    bounds: bounds(),
                    level: 1,
                },
                MeshletGroup {
                    meshlet_start: 3,
                    meshlet_count: 1,
                    child_start: 0,
                    child_count: 0,
                    error: 0.0,
                    bounds: bounds(),
                    level: 0,
                },
            ],
            vertices: Vec::new(),
            level_offsets: vec![0],
        };

        assert_eq!(dag.root_group_indices(), vec![2, 3]);
    }
}
