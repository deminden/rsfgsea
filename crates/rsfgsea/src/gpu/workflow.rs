use crate::GpuEngine;
use anyhow::Result;
use rand::SeedableRng;
use rand::rngs::{SmallRng, StdRng};
use rand::seq::index::sample;
use rayon::prelude::*;

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

impl GpuEngine {
    fn sampled_subsets_batch(
        &self,
        n_total: usize,
        k: usize,
        batch_idx: usize,
        batch_size: usize,
        current_batch_size: usize,
        seed: u64,
    ) -> Vec<u32> {
        let mut subsets = vec![0u32; current_batch_size * k];
        subsets
            .par_chunks_mut(k)
            .enumerate()
            .for_each(|(i, chunk)| {
                let sample_seed = seed
                    .wrapping_add((batch_idx * batch_size) as u64)
                    .wrapping_add(i as u64);
                let mut local_rng = SmallRng::seed_from_u64(sample_seed);
                let picks = sample(&mut local_rng, n_total, k);
                for (dst, idx) in picks.into_iter().enumerate() {
                    chunk[dst] = idx as u32;
                }
                chunk.sort_unstable();
            });
        subsets
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
        let n_total = abs_scores.len();
        let k = pathway_indices.len();

        if k == 0 || k >= n_total {
            return Err(anyhow::anyhow!("Invalid pathway size"));
        }

        let mut sorted_pathway = pathway_indices.to_vec();
        sorted_pathway.sort_unstable();
        let obs_es = self.calculate_es_cpu(&sorted_pathway, abs_scores, score_type)?;

        let target_batch_size = 200000;
        let batch_size = self.capped_batch_size(k, target_batch_size)?;
        let num_batches = n_perm.div_ceil(batch_size);

        let mut n_le_es = 0u64;
        let mut n_ge_es = 0u64;
        let mut n_le_zero = 0u64;
        let mut n_ge_zero = 0u64;
        let mut le_zero_sum = 0.0f64;
        let mut ge_zero_sum = 0.0f64;

        let mut total_perm_gen_time = std::time::Duration::from_secs(0);
        let mut total_gpu_comp_time = std::time::Duration::from_secs(0);

        for batch_idx in 0..num_batches {
            let current_batch_size = if batch_idx == num_batches - 1 {
                n_perm - batch_idx * batch_size
            } else {
                batch_size
            } as u32;

            let gen_start = std::time::Instant::now();
            let subsets = self.sampled_subsets_batch(
                n_total,
                k,
                batch_idx,
                batch_size,
                current_batch_size as usize,
                seed,
            );
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

            let batch_es: Vec<f32> = batch_results.iter().map(|result| result.es).collect();
            let batch_stats = Self::calculate_null_stats(&batch_es, obs_es);
            n_le_es += batch_stats.0;
            n_ge_es += batch_stats.1;
            n_le_zero += batch_stats.2;
            n_ge_zero += batch_stats.3;
            le_zero_sum += batch_stats.4;
            ge_zero_sum += batch_stats.5;
            total_gpu_comp_time += comp_start.elapsed();
        }

        log::debug!("Permutation generation time: {:?}", total_perm_gen_time);
        log::debug!("GPU pure compute time: {:?}", total_gpu_comp_time);

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
        let target_batch_size = 200000;
        let batch_size = self.capped_batch_size(k, target_batch_size)?;
        let num_batches = n_perm.div_ceil(batch_size);
        let mut all_es = Vec::with_capacity(n_perm);

        for batch_idx in 0..num_batches {
            let current_batch_size = if batch_idx == num_batches - 1 {
                n_perm - batch_idx * batch_size
            } else {
                batch_size
            } as u32;

            let subsets = self.sampled_subsets_batch(
                n_total,
                k,
                batch_idx,
                batch_size,
                current_batch_size as usize,
                seed,
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
        n_perm: usize,
        seed: u64,
        score_type: u32,
    ) -> Result<FgseaMultilevelResult> {
        let scores_buffer = self.upload_scores(abs_scores);
        self.fgsea_multilevel_pathway_with_buffer(
            &scores_buffer,
            pathway_indices,
            abs_scores,
            n_perm,
            seed,
            score_type,
        )
    }

    pub fn fgsea_multilevel_pathway_with_buffer(
        &self,
        scores_buffer: &wgpu::Buffer,
        pathway_indices: &[usize],
        abs_scores: &[f32],
        n_perm: usize,
        seed: u64,
        score_type: u32,
    ) -> Result<FgseaMultilevelResult> {
        use rand::Rng;

        let n_total = abs_scores.len();
        let k = pathway_indices.len();
        if k == 0 || k >= n_total {
            return Err(anyhow::anyhow!("Invalid pathway size"));
        }

        let mut sorted_pathway = pathway_indices.to_vec();
        sorted_pathway.sort_unstable();
        let obs_es = self.calculate_es_cpu(&sorted_pathway, abs_scores, score_type)?;
        let is_pos = obs_es >= 0.0;

        let _seed_rng = StdRng::seed_from_u64(seed);
        let pool: Vec<usize> = (0..n_total).collect();
        let mut current_samples = vec![0u32; n_perm * k];
        current_samples
            .par_chunks_mut(k)
            .for_each_with(pool.clone(), |local_pool, chunk| {
                let mut local_rng = rand::thread_rng();
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
            let batch_results = self.compute_es_batch_with_buffer(
                scores_buffer,
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

            if is_pos {
                scores.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            } else {
                scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            }

            let mid = sample_size / 2;
            let threshold = scores[mid].0;
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

            log_p += ((sample_size - mid + 1) as f64 / (sample_size + 1) as f64).ln();

            let top_indices: Vec<usize> = scores[mid..].iter().map(|s| s.1).collect();
            let mut next_samples = vec![0u32; sample_size * k];
            let current_samples_ref = &current_samples;
            next_samples.par_chunks_mut(k).for_each(|chunk| {
                let mut local_rng = rand::thread_rng();
                let src_idx = top_indices[local_rng.gen_range(0..top_indices.len())];
                let mut sample = current_samples_ref[src_idx * k..(src_idx + 1) * k].to_vec();

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
            log2err: (log_p / 2.0f64.ln()).abs() * 0.05,
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
        if k == 0 {
            return (0.0, 0);
        }
        if k == n_total {
            return (0.0, hits_u32[0]);
        }

        let mut adj = Vec::with_capacity(k);
        let mut nr = 0.0f32;
        for &idx in hits_u32 {
            let value = weights[idx as usize].abs();
            adj.push(value);
            nr += value;
        }

        let mut max_p = f32::NEG_INFINITY;
        let mut min_p = f32::INFINITY;
        let mut max_i = 0usize;
        let mut min_i = 0usize;
        let mut csum = 0.0f32;

        for i in 0..k {
            csum += adj[i];
            let r_cum = if nr == 0.0 {
                (i + 1) as f32 / k as f32
            } else {
                csum / nr
            };
            let miss = (hits_u32[i] as usize - i) as f32 / (n_total - k) as f32;
            let top = r_cum - miss;
            let bottom = if nr == 0.0 {
                top - 1.0 / k as f32
            } else {
                top - adj[i] / nr
            };
            if top > max_p {
                max_p = top;
                max_i = i;
            }
            if bottom < min_p {
                min_p = bottom;
                min_i = i;
            }
        }

        let es = match score_type {
            1 => max_p,
            2 => min_p,
            _ => {
                if max_p == -min_p {
                    0.0
                } else if max_p > -min_p {
                    max_p
                } else {
                    min_p
                }
            }
        };
        let peak_idx = match score_type {
            1 => hits_u32[max_i],
            2 => hits_u32[min_i],
            _ => {
                if max_p == -min_p {
                    hits_u32[0]
                } else if max_p > -min_p {
                    hits_u32[max_i]
                } else {
                    hits_u32[min_i]
                }
            }
        };
        (es, peak_idx)
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
