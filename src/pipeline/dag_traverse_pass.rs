use crate::meshlet::{GpuMeshletBuffers, MeshletDag};

/// GPU-driven DAG traversal pass.
///
/// Replaces the CPU-side `cpu_dag_cut_adaptive()` with a persistent-thread
/// compute shader that traverses the meshlet LOD DAG entirely on the GPU.
///
/// The pass uses a single atomic work queue seeded with root group indices.
/// Each GPU thread grabs a group, projects its simplification error to screen
/// space, and either selects it (writes to output queue) or appends its
/// children back to the work queue. After traversal completes, a tiny
/// `prepare_indirect` shader writes the indirect dispatch args for the
/// downstream cull pass.
///
/// Output: `output_queue_buffer` + `output_count_buffer` contain the selected
/// group indices, ready to be consumed by `CullPass` via indirect dispatch.
pub struct DagTraversePass {
    traverse_pipeline: wgpu::ComputePipeline,
    prepare_indirect_pipeline: wgpu::ComputePipeline,
    dag_bind_group_layout: wgpu::BindGroupLayout,

    // Work queue (input/output for traversal)
    work_queue_buffer: wgpu::Buffer,
    /// Atomic counters: [head, tail, 0, 0]
    work_counters_buffer: wgpu::Buffer,

    // Output queue (selected groups -> feed to cull pass)
    output_queue_buffer: wgpu::Buffer,
    output_count_buffer: wgpu::Buffer,

    // Indirect dispatch args for cull pass: [workgroup_x, 1, 1]
    indirect_args_buffer: wgpu::Buffer,

    // Root group indices (uploaded once per scene)
    root_buffer_staging: Vec<u32>,
    root_count: u32,

    // Dispatch size for the traversal pass (enough threads to drain the queue)
    traverse_workgroups: u32,
}

const WORK_QUEUE_CAPACITY: u32 = 16384;
const OUTPUT_QUEUE_CAPACITY: u32 = 16384;

impl DagTraversePass {
    pub fn new(device: &wgpu::Device, frame_bgl: &wgpu::BindGroupLayout) -> Self {
        let dag_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("dag-traverse-bgl"),
                entries: &[
                    // binding 0: meshlet_groups (read-only)
                    super::storage_entry(0, true, wgpu::ShaderStages::COMPUTE),
                    // binding 1: work_queue (read_write)
                    super::storage_entry(1, false, wgpu::ShaderStages::COMPUTE),
                    // binding 2: work_counters (read_write, atomics)
                    super::storage_entry(2, false, wgpu::ShaderStages::COMPUTE),
                    // binding 3: output_queue (read_write)
                    super::storage_entry(3, false, wgpu::ShaderStages::COMPUTE),
                    // binding 4: output_count (read_write, atomic)
                    super::storage_entry(4, false, wgpu::ShaderStages::COMPUTE),
                    // binding 5: indirect_args (read_write)
                    super::storage_entry(5, false, wgpu::ShaderStages::COMPUTE),
                ],
            });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("dag-traverse-shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/dag_traverse.wgsl").into(),
            ),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("dag-traverse-pipeline-layout"),
            bind_group_layouts: &[frame_bgl, &dag_bind_group_layout],
            immediate_size: 0,
        });

        let traverse_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("dag-traverse-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("dag_traverse"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let prepare_indirect_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("dag-prepare-indirect-pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("prepare_indirect"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let work_queue_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dag-work-queue"),
            size: WORK_QUEUE_CAPACITY as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let work_counters_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dag-work-counters"),
            size: 16, // [head, tail, 0, 0] = 4 x u32
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let output_queue_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dag-output-queue"),
            size: OUTPUT_QUEUE_CAPACITY as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let output_count_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dag-output-count"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let indirect_args_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dag-indirect-args"),
            size: 12, // [workgroup_x, workgroup_y, workgroup_z]
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            traverse_pipeline,
            prepare_indirect_pipeline,
            dag_bind_group_layout,
            work_queue_buffer,
            work_counters_buffer,
            output_queue_buffer,
            output_count_buffer,
            indirect_args_buffer,
            root_buffer_staging: Vec::new(),
            root_count: 0,
            traverse_workgroups: 1,
        }
    }

    /// Compute and cache root group indices from the DAG. Call once per scene load.
    pub fn sync_roots(&mut self, dag: &MeshletDag) {
        self.root_buffer_staging = dag.root_group_indices();
        self.root_count = self.root_buffer_staging.len() as u32;
        // Dispatch enough workgroups to have at least as many threads as there
        // could be work items. The DAG has at most `dag.groups.len()` nodes, but
        // the persistent-thread loop in the shader handles more work than threads.
        // We launch enough threads to cover the maximum queue depth.
        let max_possible_work = dag.groups.len().min(WORK_QUEUE_CAPACITY as usize) as u32;
        self.traverse_workgroups = max_possible_work.div_ceil(64).max(1);
    }

    /// Create the bind group for the DAG traversal pass.
    pub fn create_bind_group(
        &self,
        device: &wgpu::Device,
        buffers: &GpuMeshletBuffers,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dag-traverse-bg"),
            layout: &self.dag_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.group_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.work_queue_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.work_counters_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.output_queue_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.output_count_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.indirect_args_buffer.as_entire_binding(),
                },
            ],
        })
    }

    /// The output queue buffer containing selected group indices.
    /// The cull pass binds this as its `group_queue` input.
    pub fn output_queue_buffer(&self) -> &wgpu::Buffer {
        &self.output_queue_buffer
    }

    /// The output count buffer (atomic u32).
    /// The cull pass binds this as its `group_queue_count` input.
    pub fn output_count_buffer(&self) -> &wgpu::Buffer {
        &self.output_count_buffer
    }

    /// The indirect dispatch args buffer: `[workgroup_x, 1, 1]`.
    /// The cull pass uses this for `dispatch_workgroups_indirect`.
    pub fn indirect_args_buffer(&self) -> &wgpu::Buffer {
        &self.indirect_args_buffer
    }

    /// Seed the work queue with root group indices and clear output counters.
    /// Must be called each frame before `encode()`.
    pub fn seed_work_queue(&self, queue: &wgpu::Queue) {
        if self.root_count == 0 {
            // Nothing to traverse — ensure output is zero
            queue.write_buffer(&self.output_count_buffer, 0, bytemuck::bytes_of(&0u32));
            queue.write_buffer(
                &self.indirect_args_buffer,
                0,
                bytemuck::cast_slice(&[0u32, 1u32, 1u32]),
            );
            return;
        }

        // Upload root group indices into the work queue
        let count = self.root_count.min(WORK_QUEUE_CAPACITY);
        queue.write_buffer(
            &self.work_queue_buffer,
            0,
            bytemuck::cast_slice(&self.root_buffer_staging[..count as usize]),
        );

        // Initialize counters: head = 0, tail = root_count
        let counters = [0u32, count, 0u32, 0u32];
        queue.write_buffer(
            &self.work_counters_buffer,
            0,
            bytemuck::cast_slice(&counters),
        );

        // Clear output count
        queue.write_buffer(&self.output_count_buffer, 0, bytemuck::bytes_of(&0u32));
    }

    /// Encode the GPU DAG traversal + indirect dispatch preparation.
    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        frame_bg: &wgpu::BindGroup,
        dag_bg: &wgpu::BindGroup,
    ) {
        if self.root_count == 0 {
            return;
        }

        // Pass 1: DAG traversal (persistent threads)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("dag-traverse"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.traverse_pipeline);
            pass.set_bind_group(0, frame_bg, &[]);
            pass.set_bind_group(1, dag_bg, &[]);
            pass.dispatch_workgroups(self.traverse_workgroups, 1, 1);
        }

        // Pass 2: Prepare indirect dispatch args for cull pass
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("dag-prepare-indirect"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.prepare_indirect_pipeline);
            pass.set_bind_group(0, frame_bg, &[]);
            pass.set_bind_group(1, dag_bg, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
    }
}

