use crate::gpu::{GpuEngine, GpuResult};
use anyhow::Result;
use bytemuck::cast_slice;
use futures_intrusive::channel::shared::oneshot_channel;
use wgpu::util::DeviceExt;

impl GpuEngine {
    fn max_storage_binding_size(&self) -> usize {
        self.device.limits().max_storage_buffer_binding_size as usize
    }

    pub(crate) fn capped_batch_size(&self, k: usize, requested_batch: usize) -> Result<usize> {
        if k == 0 {
            return Err(anyhow::anyhow!("Pathway size (k) must be > 0"));
        }
        let max_binding = self.max_storage_binding_size();
        let bytes_per_subset = k
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(|| anyhow::anyhow!("Subset size overflow for k={k}"))?;
        let max_by_subsets = max_binding / bytes_per_subset;
        let max_by_results = max_binding / std::mem::size_of::<GpuResult>();
        let cap = max_by_subsets.min(max_by_results).max(1);
        Ok(requested_batch.min(cap))
    }

    pub fn compute_es_batch(
        &self,
        abs_scores: &[f32],
        subsets_indices: &[u32],
        k: u32,
        n_total: u32,
        batch_size: u32,
        score_type: u32,
    ) -> Result<Vec<GpuResult>> {
        let scores_buffer = self.upload_scores(abs_scores);
        self.compute_es_batch_with_buffer(
            &scores_buffer,
            subsets_indices,
            k,
            n_total,
            batch_size,
            score_type,
        )
    }

    pub fn upload_scores(&self, abs_scores: &[f32]) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("scores_buffer"),
                contents: cast_slice(abs_scores),
                usage: wgpu::BufferUsages::STORAGE,
            })
    }

    pub fn compute_es_batch_with_buffer(
        &self,
        scores_buffer: &wgpu::Buffer,
        subsets_indices: &[u32],
        k: u32,
        n_total: u32,
        batch_size: u32,
        score_type: u32,
    ) -> Result<Vec<GpuResult>> {
        let max_binding = self.max_storage_binding_size();
        let subsets_bytes = subsets_indices
            .len()
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(|| anyhow::anyhow!("subsets buffer size overflow"))?;
        let results_bytes = batch_size as usize * std::mem::size_of::<GpuResult>();
        if subsets_bytes > max_binding || results_bytes > max_binding {
            return Err(anyhow::anyhow!(
                "GPU batch exceeds storage binding limit: subsets={} bytes, results={} bytes, limit={} bytes. Reduce batch size.",
                subsets_bytes,
                results_bytes,
                max_binding
            ));
        }

        let subsets_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("subsets_buffer"),
                contents: cast_slice(subsets_indices),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let results_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("results_buffer"),
            size: (batch_size as usize * std::mem::size_of::<GpuResult>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = [
            k as f32,
            n_total as f32,
            batch_size as f32,
            score_type as f32,
        ];
        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("params_buffer"),
                contents: cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

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
            results_buffer.size(),
        );
        self.queue.submit(Some(encoder.finish()));

        let (sender, receiver) = oneshot_channel::<Option<Result<(), wgpu::BufferAsyncError>>>();
        let slice = staging_buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, move |v| sender.send(Some(v)).unwrap());
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| anyhow::anyhow!("GPU poll failed: {e}"))?;

        pollster::block_on(receiver.receive()).unwrap().unwrap()?;
        let data = slice.get_mapped_range();
        let results: Vec<GpuResult> = cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        Ok(results)
    }
}
