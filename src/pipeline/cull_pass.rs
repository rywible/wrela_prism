use super::hw_raster_pass::DispatchLists;
use crate::meshlet::{GpuMeshletBuffers, MeshletDag};

/// CPU-side adaptive DAG traversal + GPU per-meshlet culling.
///
/// Each frame the CPU traverses the meshlet DAG from root groups, projecting
/// each group's simplification error to screen-space pixels. Groups whose error
/// is below the threshold are selected; otherwise their children are visited.
/// Selected group indices are uploaded to the GPU, where a compute shader does
/// per-meshlet frustum culling, writing survivors to the HW dispatch list.
pub struct CullPass {
    /// Group queue buffer (CPU → GPU).
    group_queue_buffer: wgpu::Buffer,
    group_queue_count_buffer: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    cull_bind_group_layout: wgpu::BindGroupLayout,
    dispatch_bind_group_layout: wgpu::BindGroupLayout,
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

    /// Perform adaptive CPU-side DAG cut and upload the selected group queue.
    ///
    /// Traverses the DAG top-down from root groups. For each group, projects its
    /// simplification error to screen-space pixels. If the error is below the
    /// threshold (or the group is a leaf), the group is selected for rendering.
    /// Otherwise its children are visited recursively.
    ///
    /// Returns the number of selected groups (to dispatch the GPU cull shader).
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
        let mut stack = root_group_indices(dag);

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

fn root_group_indices(dag: &MeshletDag) -> Vec<u32> {
    if dag.groups.is_empty() {
        return Vec::new();
    }

    let mut has_parent = vec![false; dag.groups.len()];
    for group in &dag.groups {
        for child_idx in group.child_start..group.child_start + group.child_count {
            if let Some(slot) = has_parent.get_mut(child_idx as usize) {
                *slot = true;
            }
        }
    }

    let mut roots: Vec<u32> = has_parent
        .iter()
        .enumerate()
        .filter_map(|(idx, has_parent)| (!*has_parent).then_some(idx as u32))
        .collect();

    if roots.is_empty() {
        let max_level = dag.groups.iter().map(|g| g.level).max().unwrap_or(0);
        roots = dag
            .groups
            .iter()
            .enumerate()
            .filter_map(|(idx, group)| (group.level == max_level).then_some(idx as u32))
            .collect();
    }

    roots
}

#[cfg(test)]
mod tests {
    use super::root_group_indices;
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

        assert_eq!(root_group_indices(&dag), vec![2, 3]);
    }
}
