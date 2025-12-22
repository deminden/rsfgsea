use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuResult {
    pub es: f32,
    pub peak_idx: u32,
}

pub struct GpuEngine {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
}

impl GpuEngine {
    pub async fn new() -> Result<Self> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .ok_or_else(|| anyhow::anyhow!("No GPU adapter found"))?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("rsfgsea-gpu-device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    ..Default::default()
                },
                None,
            )
            .await?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gsea_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gsea_pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(Self {
            device,
            queue,
            pipeline,
        })
    }

    pub fn compute_es_batch(
        &self,
        abs_scores: &[f32],
        subsets_indices: &[u32], // Flattened: [perm0_idx0, ..., perm0_idxK, perm1_idx0, ...]
        k: u32,
        n_total: u32,
        batch_size: u32,
    ) -> Result<Vec<GpuResult>> {
        let _n_miss = (n_total - k) as f32;

        // Setup buffers
        let scores_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("scores_buffer"),
                contents: bytemuck::cast_slice(abs_scores),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let subsets_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("subsets_buffer"),
                contents: bytemuck::cast_slice(subsets_indices),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let results_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("results_buffer"),
            size: (batch_size as usize * std::mem::size_of::<GpuResult>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = [k as f32, n_total as f32, batch_size as f32, 0.0]; // Padding
        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("params_buffer"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bind_group"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: scores_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: subsets_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: results_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(batch_size.div_ceil(64), 1, 1);
        }

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_buffer"),
            size: (batch_size as usize * std::mem::size_of::<GpuResult>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(
            &results_buffer,
            0,
            &staging_buffer,
            0,
            staging_buffer.size(),
        );
        self.queue.submit(Some(encoder.finish()));

        // Read back
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel::<
            Option<Result<(), wgpu::BufferAsyncError>>,
        >();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(Some(v)).unwrap());
        self.device.poll(wgpu::Maintain::Wait);

        pollster::block_on(receiver.receive()).unwrap().unwrap()?;
        let data = buffer_slice.get_mapped_range();
        let results: Vec<GpuResult> = bytemuck::cast_slice(&data).to_vec();

        Ok(results)
    }
}
