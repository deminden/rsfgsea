use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use rayon::prelude::*;
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

        // List all adapters to pick the best one (prefer NVIDIA)
        let adapters = instance.enumerate_adapters(wgpu::Backends::all());
        let mut selected_adapter = None;

        println!("Available GPUs:");
        for adapter in adapters {
            let info = adapter.get_info();
            println!(
                "  - Name: {:?}, Type: {:?}, Backend: {:?}, Vendor: 0x{:04x}",
                info.name, info.device_type, info.backend, info.vendor
            );
            if info.name.to_lowercase().contains("nvidia") || info.vendor == 0x10de {
                selected_adapter = Some(adapter);
            }
        }

        let adapter = if let Some(a) = selected_adapter {
            println!("Selecting NVIDIA GPU.");
            a
        } else {
            instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    ..Default::default()
                })
                .await
                .ok_or_else(|| anyhow::anyhow!("No GPU adapter found"))?
        };

        println!("Selected GPU: {:?}", adapter.get_info().name);

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
        subsets_indices: &[u32],
        k: u32,
        n_total: u32,
        batch_size: u32,
        score_type: u32, // 0: Std, 1: Pos, 2: Neg
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
                contents: bytemuck::cast_slice(abs_scores),
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
                contents: bytemuck::cast_slice(&params),
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

        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel::<
            Option<Result<(), wgpu::BufferAsyncError>>,
        >();
        let slice = staging_buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, move |v| sender.send(Some(v)).unwrap());
        self.device.poll(wgpu::Maintain::Wait);

        pollster::block_on(receiver.receive()).unwrap().unwrap()?;
        let data = slice.get_mapped_range();
        let results: Vec<GpuResult> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        Ok(results)
    }

    pub fn fgsea_simple_pathway(
        &self,
        pathway_indices: &[usize],
        abs_scores: &[f32],
        n_perm: usize,
        seed: u64,
        score_type: u32,
    ) -> Result<FgseaSimpleResult> {
        let scores_buffer = self.upload_scores(abs_scores);
        self.fgsea_simple_pathway_with_buffer(
            &scores_buffer,
            pathway_indices,
            abs_scores,
            n_perm,
            seed,
            score_type,
        )
    }

    pub fn fgsea_simple_pathway_with_buffer(
        &self,
        scores_buffer: &wgpu::Buffer,
        pathway_indices: &[usize],
        abs_scores: &[f32],
        n_perm: usize,
        seed: u64,
        score_type: u32,
    ) -> Result<FgseaSimpleResult> {
        use rand::prelude::*;
        use rand::rngs::StdRng;

        let n_total = abs_scores.len();
        let k = pathway_indices.len();

        if k == 0 || k >= n_total {
            return Err(anyhow::anyhow!("Invalid pathway size"));
        }

        let mut sorted_pathway = pathway_indices.to_vec();
        sorted_pathway.sort_unstable();
        let obs_es = self.calculate_es_cpu(&sorted_pathway, abs_scores, score_type)?;

        let _rng = StdRng::seed_from_u64(seed);
        let batch_size = 200000; // Increased for better throughput on high-end GPUs
        let num_batches = n_perm.div_ceil(batch_size);

        let mut n_le_es = 0u64;
        let mut n_ge_es = 0u64;
        let mut n_le_zero = 0u64;
        let mut n_ge_zero = 0u64;
        let mut le_zero_sum = 0.0f64;
        let mut ge_zero_sum = 0.0f64;

        let mut total_perm_gen_time = std::time::Duration::from_secs(0);
        let mut total_gpu_comp_time = std::time::Duration::from_secs(0);

        let pool: Vec<usize> = (0..n_total).collect();

        for batch_idx in 0..num_batches {
            let current_batch_size = if batch_idx == num_batches - 1 {
                n_perm - batch_idx * batch_size
            } else {
                batch_size
            } as u32;

            let gen_start = std::time::Instant::now();
            let mut subsets = vec![0u32; current_batch_size as usize * k];

            use rand::rngs::SmallRng;

            subsets
                .par_chunks_mut(k)
                .for_each_with(pool.clone(), |local_pool, chunk| {
                    let mut local_rng = SmallRng::from_entropy();
                    for i in 0..k {
                        let j = local_rng.gen_range(i..n_total);
                        local_pool.swap(i, j);
                        chunk[i] = local_pool[i] as u32;
                    }
                    chunk.sort_unstable();
                });
            total_perm_gen_time += gen_start.elapsed();

            let comp_start = std::time::Instant::now();
            let batch_results = self.compute_es_batch_with_buffer(
                scores_buffer,
                &subsets,
                k as u32,
                n_total as u32,
                current_batch_size,
                score_type,
            )?;

            for result in batch_results {
                let perm_es = result.es as f64;
                if perm_es <= obs_es {
                    n_le_es += 1;
                }
                if perm_es >= obs_es {
                    n_ge_es += 1;
                }
                if perm_es <= 0.0 {
                    n_le_zero += 1;
                    le_zero_sum += perm_es;
                }
                if perm_es >= 0.0 {
                    n_ge_zero += 1;
                    ge_zero_sum += perm_es;
                }
            }
            total_gpu_comp_time += comp_start.elapsed();
        }

        println!("  (Permutation Gen Time: {:?})", total_perm_gen_time);
        println!("  (GPU Pure Comp Time: {:?})", total_gpu_comp_time);

        let le_zero_mean = if n_le_zero > 0 {
            le_zero_sum / n_le_zero as f64
        } else {
            0.0
        };
        let ge_zero_mean = if n_ge_zero > 0 {
            ge_zero_sum / n_ge_zero as f64
        } else {
            0.0
        };

        let nes = if obs_es > 0.0 {
            if ge_zero_mean != 0.0 {
                Some(obs_es / ge_zero_mean)
            } else {
                None
            }
        } else if le_zero_mean != 0.0 {
            Some(obs_es / le_zero_mean.abs())
        } else {
            None
        };

        let p_value = if obs_es > 0.0 {
            (n_ge_es + 1) as f64 / (n_ge_zero + 1) as f64
        } else {
            (n_le_es + 1) as f64 / (n_le_zero + 1) as f64
        };
        Ok(FgseaSimpleResult {
            es: obs_es,
            nes,
            p_value,
            n_perm,
        })
    }

    pub fn generate_null_distribution(
        &self,
        scores_buffer: &wgpu::Buffer,
        k: usize,
        n_total: usize,
        n_perm: usize,
        seed: u64,
        score_type: u32,
    ) -> Result<Vec<f32>> {
        use rand::SeedableRng;
        use rand::prelude::*;
        use rand::rngs::SmallRng;

        let batch_size = 200000;
        let num_batches = n_perm.div_ceil(batch_size);
        let pool: Vec<usize> = (0..n_total).collect();
        let mut all_es = Vec::with_capacity(n_perm);

        for batch_idx in 0..num_batches {
            let current_batch_size = if batch_idx == num_batches - 1 {
                n_perm - batch_idx * batch_size
            } else {
                batch_size
            } as u32;

            let mut subsets = vec![0u32; current_batch_size as usize * k];

            subsets.par_chunks_mut(k).enumerate().for_each_with(
                pool.clone(),
                |local_pool, (i, chunk)| {
                    let mut local_rng = SmallRng::seed_from_u64(
                        seed + batch_idx as u64 * batch_size as u64 + i as u64,
                    );
                    for i in 0..k {
                        let j = local_rng.gen_range(i..n_total);
                        local_pool.swap(i, j);
                        chunk[i] = local_pool[i] as u32;
                    }
                    chunk.sort_unstable();
                },
            );

            let batch_results = self.compute_es_batch_with_buffer(
                scores_buffer,
                &subsets,
                k as u32,
                n_total as u32,
                current_batch_size,
                score_type,
            )?;

            for res in batch_results {
                all_es.push(res.es);
            }
        }

        Ok(all_es)
    }

    pub fn calculate_null_stats(null_es: &[f32], obs_es: f64) -> (u64, u64, u64, u64, f64, f64) {
        let mut n_le_es = 0u64;
        let mut n_ge_es = 0u64;
        let mut n_le_zero = 0u64;
        let mut n_ge_zero = 0u64;
        let mut le_zero_sum = 0.0f64;
        let mut ge_zero_sum = 0.0f64;

        for &es in null_es {
            let perm_es = es as f64;
            if perm_es <= obs_es {
                n_le_es += 1;
            }
            if perm_es >= obs_es {
                n_ge_es += 1;
            }
            if perm_es <= 0.0 {
                n_le_zero += 1;
                le_zero_sum += perm_es;
            }
            if perm_es >= 0.0 {
                n_ge_zero += 1;
                ge_zero_sum += perm_es;
            }
        }

        (
            n_le_es,
            n_ge_es,
            n_le_zero,
            n_ge_zero,
            le_zero_sum,
            ge_zero_sum,
        )
    }

    pub fn fgsea_multilevel_pathway(
        &self,
        pathway_indices: &[usize],
        abs_scores: &[f32],
        n_perm: usize, // Base number of permutations (e.g. 1000)
        seed: u64,
        score_type: u32,
    ) -> Result<FgseaMultilevelResult> {
        use rand::SeedableRng;
        use rand::prelude::*;
        use rand::rngs::{SmallRng, StdRng};
        use rayon::prelude::*;

        let n_total = abs_scores.len();
        let k = pathway_indices.len();
        if k == 0 || k >= n_total {
            return Err(anyhow::anyhow!("Invalid pathway size"));
        }

        let mut sorted_pathway = pathway_indices.to_vec();
        sorted_pathway.sort_unstable();
        let obs_es = self.calculate_es_cpu(&sorted_pathway, abs_scores, score_type)?;
        let is_pos = obs_es >= 0.0;

        let scores_buffer = self.upload_scores(abs_scores);
        let _seed_rng = StdRng::seed_from_u64(seed);

        // Initial samples
        let pool: Vec<usize> = (0..n_total).collect();
        let mut current_samples = vec![0u32; n_perm * k];
        current_samples
            .par_chunks_mut(k)
            .for_each_with(pool.clone(), |local_pool, chunk| {
                let mut local_rng = SmallRng::from_entropy();
                for i in 0..k {
                    let j = local_rng.gen_range(i..n_total);
                    local_pool.swap(i, j);
                    chunk[i] = local_pool[i] as u32;
                }
                chunk.sort_unstable();
            });

        let mut log_p: f64 = 0.0;
        let sample_size = n_perm;

        for _level in 0..200 {
            // Evaluated current samples on GPU
            let batch_results = self.compute_es_batch_with_buffer(
                &scores_buffer,
                &current_samples,
                k as u32,
                n_total as u32,
                sample_size as u32,
                score_type,
            )?;

            let mut scores: Vec<(f32, usize)> = batch_results
                .iter()
                .enumerate()
                .map(|(i, r)| (r.es, i))
                .collect();

            // Sort to find threshold
            if is_pos {
                scores.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            } else {
                scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            }

            let mid = sample_size / 2;
            let threshold = scores[mid].0;

            // Check if we reached observed ES
            let reached = if is_pos {
                threshold >= obs_es as f32
            } else {
                threshold <= obs_es as f32
            };

            if reached {
                let count = scores
                    .iter()
                    .filter(|s| {
                        if is_pos {
                            s.0 >= obs_es as f32
                        } else {
                            s.0 <= obs_es as f32
                        }
                    })
                    .count();
                log_p += ((count + 1) as f64 / (sample_size + 1) as f64).ln();
                break;
            }

            // Probability factor for this level
            log_p += ((sample_size - mid + 1) as f64 / (sample_size + 1) as f64).ln();

            // Next samples generation (MCMC Splitting)
            let top_indices: Vec<usize> = scores[mid..].iter().map(|s| s.1).collect();
            let mut next_samples = vec![0u32; sample_size * k];

            let current_samples_ref = &current_samples;
            next_samples.par_chunks_mut(k).for_each(|chunk| {
                let mut local_rng = SmallRng::from_entropy();
                let src_idx = top_indices[local_rng.gen_range(0..top_indices.len())];
                let mut sample = current_samples_ref[src_idx * k..(src_idx + 1) * k].to_vec();

                // Perturb with MCMC
                let n_swaps = (k as f64 * 0.1).ceil() as usize;
                for _ in 0..n_swaps {
                    let hit_to_swap = local_rng.gen_range(0..k);
                    let old_gene = sample[hit_to_swap];
                    let mut new_gene = local_rng.gen_range(0..n_total) as u32;
                    while sample.binary_search(&new_gene).is_ok() {
                        new_gene = local_rng.gen_range(0..n_total) as u32;
                    }

                    sample[hit_to_swap] = new_gene;
                    sample.sort_unstable();

                    // We need ES to check threshold. Calculate locally for efficiency in inner loop.
                    let (new_es, _) = self.calculate_es_cpu_f32(&sample, abs_scores, score_type);
                    let reject = if is_pos {
                        new_es < threshold
                    } else {
                        new_es > threshold
                    };

                    if reject {
                        let idx = sample.binary_search(&new_gene).unwrap();
                        sample[idx] = old_gene;
                        sample.sort_unstable();
                    }
                }
                chunk.copy_from_slice(&sample);
            });
            current_samples = next_samples;
        }

        let p_value = log_p.exp().min(1.0);
        Ok(FgseaMultilevelResult {
            es: obs_es,
            p_value,
            log2err: (log_p / 2.0f64.ln()).abs() * 0.05, // Approximation for log2err
            n_perm,
        })
    }

    fn calculate_es_cpu_f32(
        &self,
        hits_u32: &[u32],
        weights: &[f32],
        score_type: u32,
    ) -> (f32, u32) {
        let n_total = weights.len();
        let k = hits_u32.len();
        let n_miss = (n_total - k) as f32;
        let mut sum_weights: f32 = 0.0;
        for &idx in hits_u32 {
            sum_weights += weights[idx as usize];
        }

        if sum_weights == 0.0 {
            return (0.0, 0);
        }

        let mut curr_max: f32 = 0.0;
        let mut curr_min: f32 = 0.0;
        let mut curr_sum_weight: f32 = 0.0;

        for (j, &hit_idx) in hits_u32.iter().enumerate() {
            let p_miss = (hit_idx as f32 - j as f32) / n_miss;
            let es_before = (curr_sum_weight / sum_weights) - p_miss;
            if es_before > curr_max {
                curr_max = es_before;
            }
            if es_before < curr_min {
                curr_min = es_before;
            }
            curr_sum_weight += weights[hit_idx as usize];
            let es_at = (curr_sum_weight / sum_weights) - p_miss;
            if es_at > curr_max {
                curr_max = es_at;
            }
            if es_at < curr_min {
                curr_min = es_at;
            }
        }

        let es = match score_type {
            1 => curr_max,
            2 => curr_min,
            _ => {
                if curr_max.abs() >= curr_min.abs() {
                    curr_max
                } else {
                    curr_min
                }
            }
        };
        (es, 0)
    }

    fn calculate_es_cpu(&self, hits: &[usize], weights: &[f32], score_type: u32) -> Result<f64> {
        if hits.is_empty() {
            return Ok(0.0);
        }
        let n_total = weights.len();
        let k = hits.len();
        let n_miss = (n_total - k) as f64;
        let sum_weights: f64 = hits.iter().map(|&idx| weights[idx] as f64).sum();
        if sum_weights == 0.0 {
            return Ok(0.0);
        }

        let mut curr_max = 0.0;
        let mut curr_min = 0.0;
        let mut curr_sum_weight = 0.0;

        for (j, &hit_idx) in hits.iter().enumerate() {
            let p_miss = (hit_idx - j) as f64 / n_miss;
            let es_before = (curr_sum_weight / sum_weights) - p_miss;
            if es_before > curr_max {
                curr_max = es_before;
            }
            if es_before < curr_min {
                curr_min = es_before;
            }
            curr_sum_weight += weights[hit_idx] as f64;
            let es_at = (curr_sum_weight / sum_weights) - p_miss;
            if es_at > curr_max {
                curr_max = es_at;
            }
            if es_at < curr_min {
                curr_min = es_at;
            }
        }

        Ok(match score_type {
            1 => curr_max,
            2 => curr_min,
            _ => {
                if curr_max.abs() >= curr_min.abs() {
                    curr_max
                } else {
                    curr_min
                }
            }
        })
    }
}

#[derive(Debug, Clone)]
pub struct FgseaSimpleResult {
    pub es: f64,
    pub nes: Option<f64>,
    pub p_value: f64,
    pub n_perm: usize,
}

#[derive(Debug, Clone)]
pub struct FgseaMultilevelResult {
    pub es: f64,
    pub p_value: f64,
    pub log2err: f64,
    pub n_perm: usize,
}
