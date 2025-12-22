use crate::core::{EnrichmentResult, Pathway, RankedList, ScoreType};
use rand::prelude::*;
use rand::rngs::SmallRng;
use rayon::prelude::*;
use special::Gamma;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GseaScore {
    pub ns: i64,
    pub coef_ns: i64,
    pub diff: i64, // n - k
    pub coef_const: i64,
}

impl GseaScore {
    pub fn new(ns: i64, coef_ns: i64, diff: i64, coef_const: i64) -> Self {
        Self {
            ns,
            coef_ns,
            diff,
            coef_const,
        }
    }

    pub fn get_double(&self) -> f64 {
        (self.coef_ns as f64 / self.ns as f64) - (self.coef_const as f64 / self.diff as f64)
    }

    // Comparison using i128 to avoid overflow
    pub fn compare(&self, other: &Self) -> std::cmp::Ordering {
        // score1 - score2 = (coef_ns1 * other.ns - ns1 * other.coef_ns) * diff - (coef_const1 - other.coef_const) * ns1 * other.ns
        let p1 =
            (self.coef_ns as i128 * other.ns as i128) - (self.ns as i128 * other.coef_ns as i128);
        let q1 = self.ns as i128 * other.ns as i128;
        let p2 = self.coef_const as i128 - other.coef_const as i128;
        let q2 = self.diff as i128; // diff is same for both in a pathway

        (p1 * q2).cmp(&(p2 * q1))
    }

    pub fn abs_num(&self) -> i128 {
        (self.coef_ns as i128 * self.diff as i128 - self.coef_const as i128 * self.ns as i128).abs()
    }
}

pub fn calculate_es(
    hits: &[usize],
    weights: &[f64],
    n_total: usize,
    score_type: ScoreType,
) -> (f64, usize) {
    if hits.is_empty() {
        return (0.0, 0);
    }

    let k = hits.len();
    let n_miss = (n_total - k) as f64;
    let sum_weights: f64 = hits.iter().map(|&idx| weights[idx]).sum();

    if sum_weights == 0.0 {
        return (0.0, 0);
    }

    let mut curr_max = 0.0;
    let mut curr_min = 0.0;
    let mut max_idx = hits[0];
    let mut min_idx = hits[0];

    let mut curr_sum_weight = 0.0;
    for (j, &hit_idx) in hits.iter().enumerate().take(k) {
        let p_miss = (hit_idx - j) as f64 / n_miss;

        let es_before = (curr_sum_weight / sum_weights) - p_miss;
        if es_before > curr_max {
            curr_max = es_before;
            max_idx = hit_idx;
        }
        if es_before < curr_min {
            curr_min = es_before;
            min_idx = hit_idx;
        }

        curr_sum_weight += weights[hit_idx];
        let es_at = (curr_sum_weight / sum_weights) - p_miss;
        if es_at > curr_max {
            curr_max = es_at;
            max_idx = hit_idx;
        }
        if es_at < curr_min {
            curr_min = es_at;
            min_idx = hit_idx;
        }
    }

    match score_type {
        ScoreType::Std => {
            if curr_max.abs() >= curr_min.abs() {
                (curr_max, max_idx)
            } else {
                (curr_min, min_idx)
            }
        }
        ScoreType::Pos => (curr_max, max_idx),
        ScoreType::Neg => (curr_min, min_idx),
    }
}

pub fn calculate_gsea_score(
    hits: &[usize],
    scaled_scores: &[i64],
    ns_total: i64,
    n_total: usize,
    score_type: ScoreType,
) -> (GseaScore, usize) {
    let k = hits.len();
    let diff = (n_total - k) as i64;

    let mut curr_max_num: i128 = 0;
    let mut curr_min_num: i128 = 0;
    let mut max_p = (0, 0); // (coef_ns, coef_const)
    let mut min_p = (0, 0);
    let mut m_idx = hits[0];
    let mut v_idx = hits[0];

    let mut curr_coef_ns: i64 = 0;
    for (j, &hit_idx) in hits.iter().enumerate().take(k) {
        let coef_const = (hit_idx - j) as i64;

        let num_before =
            (curr_coef_ns as i128 * diff as i128) - (coef_const as i128 * ns_total as i128);
        if num_before > curr_max_num {
            curr_max_num = num_before;
            max_p = (curr_coef_ns, coef_const);
            m_idx = hit_idx;
        }
        if num_before < curr_min_num {
            curr_min_num = num_before;
            min_p = (curr_coef_ns, coef_const);
            v_idx = hit_idx;
        }

        curr_coef_ns += scaled_scores[hit_idx];
        let num_at =
            (curr_coef_ns as i128 * diff as i128) - (coef_const as i128 * ns_total as i128);
        if num_at > curr_max_num {
            curr_max_num = num_at;
            max_p = (curr_coef_ns, coef_const);
            m_idx = hit_idx;
        }
        if num_at < curr_min_num {
            curr_min_num = num_at;
            min_p = (curr_coef_ns, coef_const);
            v_idx = hit_idx;
        }
    }

    match score_type {
        ScoreType::Std => {
            if curr_max_num.abs() >= curr_min_num.abs() {
                (GseaScore::new(ns_total, max_p.0, diff, max_p.1), m_idx)
            } else {
                (GseaScore::new(ns_total, min_p.0, diff, min_p.1), v_idx)
            }
        }
        ScoreType::Pos => (GseaScore::new(ns_total, max_p.0, diff, max_p.1), m_idx),
        ScoreType::Neg => (GseaScore::new(ns_total, min_p.0, diff, min_p.1), v_idx),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn run_gsea(
    ranks: &RankedList,
    pathways: &[Pathway],
    n_perm: usize,
    seed: u64,
    min_size: usize,
    max_size: usize,
    eps: f64,
    score_type: ScoreType,
    gsea_param: f64,
) -> Vec<EnrichmentResult> {
    let gene_to_idx: HashMap<String, usize> = ranks
        .genes
        .iter()
        .enumerate()
        .map(|(i, g)| (g.clone(), i))
        .collect();
    let n_total = ranks.len();
    let (abs_weights, scaled_scores, ns_total) = ranks.prepare(gsea_param);

    let mut final_results: Vec<EnrichmentResult> = pathways
        .par_iter()
        .filter_map(|pw| {
            let mut hits: Vec<usize> = pw
                .genes
                .iter()
                .filter_map(|g| gene_to_idx.get(g).copied())
                .collect();

            if hits.len() < min_size || hits.len() > max_size {
                return None;
            }

            hits.sort_unstable();

            // Exact ES using prepared weights
            let (es, peak_idx) = calculate_es(&hits, &abs_weights, n_total, score_type);
            let (obs_es, _) =
                calculate_gsea_score(&hits, &scaled_scores, ns_total, n_total, score_type);

            let mut n_le_es = 0;
            let mut n_ge_es = 0;
            let mut n_le_zero = 0;
            let mut n_ge_zero = 0;
            let mut le_zero_sum = 0.0;
            let mut ge_zero_sum = 0.0;

            if n_perm > 0 {
                let k = hits.len();
                let mut rand_hits = vec![0usize; k];
                let mut pool: Vec<usize> = (0..n_total).collect();
                let mut rng = SmallRng::seed_from_u64(seed + pw.name.len() as u64);

                for _ in 0..n_perm {
                    for i in 0..k {
                        let j = rng.gen_range(i..n_total);
                        pool.swap(i, j);
                        rand_hits[i] = pool[i];
                    }
                    rand_hits.sort_unstable();
                    let (rand_es, _) = calculate_es(&rand_hits, &abs_weights, n_total, score_type);

                    if rand_es <= es {
                        n_le_es += 1;
                    }
                    if rand_es >= es {
                        n_ge_es += 1;
                    }
                    if rand_es <= 0.0 {
                        n_le_zero += 1;
                        le_zero_sum += rand_es;
                    }
                    if rand_es >= 0.0 {
                        n_ge_zero += 1;
                        ge_zero_sum += rand_es;
                    }
                }
            }

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

            let nes = if es > 0.0 {
                if ge_zero_mean != 0.0 {
                    Some(es / ge_zero_mean)
                } else {
                    None
                }
            } else if le_zero_mean != 0.0 {
                Some(es / le_zero_mean.abs())
            } else {
                None
            };

            let (mut p_value, n_more_extreme) = if es > 0.0 {
                ((n_ge_es + 1) as f64 / (n_ge_zero + 1) as f64, n_ge_es)
            } else {
                ((n_le_es + 1) as f64 / (n_le_zero + 1) as f64, n_le_es)
            };

            let mut log2err = if n_perm > 0 {
                Some(
                    1.0 / 2.0_f64.ln()
                        * (((n_more_extreme + 1) as f64).trigamma()
                            - ((n_perm + 1) as f64).trigamma())
                        .sqrt(),
                )
            } else {
                None
            };

            // Multilevel p-value calculation
            if (n_more_extreme < 10) && (eps < 1.0 / (n_perm as f64)) && n_perm > 0 {
                let (m_p, m_err) = run_multilevel_gsea(
                    n_total,
                    &scaled_scores,
                    ns_total,
                    hits.len(),
                    obs_es,
                    score_type,
                    101,
                    seed + pw.name.len() as u64,
                    eps,
                );
                let denom_prob = if es > 0.0 {
                    (n_ge_zero + 1) as f64 / (n_perm + 1) as f64
                } else {
                    (n_le_zero + 1) as f64 / (n_perm + 1) as f64
                };
                p_value = (m_p / denom_prob).min(1.0);
                log2err = m_err;
            }

            let leading_edge: Vec<String> = if es >= 0.0 {
                hits.iter()
                    .filter(|&&idx| idx <= peak_idx)
                    .map(|&idx| ranks.genes[idx].clone())
                    .collect()
            } else {
                hits.iter()
                    .filter(|&&idx| idx >= peak_idx)
                    .map(|&idx| ranks.genes[idx].clone())
                    .collect()
            };

            Some(EnrichmentResult {
                pathway_name: pw.name.clone(),
                size: hits.len(),
                es,
                nes,
                p_value,
                padj: None,
                log2err,
                leading_edge,
            })
        })
        .collect();

    // BH adjustment
    if !final_results.is_empty() {
        let mut indices: Vec<usize> = (0..final_results.len()).collect();
        indices.sort_by(|&a, &b| {
            final_results[a]
                .p_value
                .partial_cmp(&final_results[b].p_value)
                .unwrap()
        });
        let m = final_results.len() as f64;
        let mut prev_padj = 1.0;
        for i in (0..indices.len()).rev() {
            let idx = indices[i];
            let p = final_results[idx].p_value;
            let padj = (p * m / (i + 1) as f64).min(prev_padj).min(1.0);
            final_results[idx].padj = Some(padj);
            prev_padj = padj;
        }
    }
    final_results
}

#[allow(clippy::too_many_arguments)]
pub fn run_multilevel_gsea(
    n_total: usize,
    scaled_scores: &[i64],
    ns_total: i64,
    k: usize,
    obs_es: GseaScore,
    score_type: ScoreType,
    sample_size: usize,
    seed: u64,
    _eps: f64,
) -> (f64, Option<f64>) {
    let mut rng = SmallRng::seed_from_u64(seed);

    // Initial samples (simple permutation)
    let mut current_samples: Vec<Vec<usize>> = (0..sample_size)
        .map(|_| {
            let mut s = (0..n_total).choose_multiple(&mut rng, k);
            s.sort_unstable();
            s
        })
        .collect();

    let mut log_p: f64 = 0.0;

    // We only care about matching the obs_es tail
    let is_pos = obs_es.get_double() >= 0.0;

    for _level in 0..100 {
        let mut scores: Vec<((GseaScore, usize), usize)> = current_samples
            .iter()
            .enumerate()
            .map(|(i, s)| {
                (
                    calculate_gsea_score(s, scaled_scores, ns_total, n_total, score_type),
                    i,
                )
            })
            .collect();

        // Sort by ES. If obs_es is negative, we want to look at the lower tail (more negative)
        if is_pos {
            scores.sort_by(|a, b| a.0.0.compare(&b.0.0));
        } else {
            scores.sort_by(|a, b| b.0.0.compare(&a.0.0));
        }

        let mid = sample_size / 2;
        let threshold = scores[mid].0.0;

        // Termination condition
        let reached = if is_pos {
            threshold.compare(&obs_es) != std::cmp::Ordering::Less
        } else {
            threshold.compare(&obs_es) != std::cmp::Ordering::Greater
        };

        if reached {
            let count = scores
                .iter()
                .filter(|s| {
                    if is_pos {
                        s.0.0.compare(&obs_es) != std::cmp::Ordering::Less
                    } else {
                        s.0.0.compare(&obs_es) != std::cmp::Ordering::Greater
                    }
                })
                .count();
            log_p += ((count + 1) as f64 / (sample_size + 1) as f64).ln();
            break;
        }

        log_p += ((sample_size - mid + 1) as f64 / (sample_size + 1) as f64).ln();

        // Resample via MCMC
        let top_indices: Vec<usize> = scores[mid..].iter().map(|s| s.1).collect();
        let mut next_samples = Vec::with_capacity(sample_size);
        for _ in 0..sample_size {
            let src_idx = top_indices[rng.gen_range(0..top_indices.len())];
            let mut sample = current_samples[src_idx].clone();

            // Perturbate
            let n_swaps = (k as f64 * 0.1).ceil() as usize;
            for _ in 0..n_swaps {
                let hit_pos = rng.gen_range(0..k);
                let old_gene = sample[hit_pos];
                let mut new_gene = rng.gen_range(0..n_total);
                while sample.binary_search(&new_gene).is_ok() {
                    new_gene = rng.gen_range(0..n_total);
                }

                sample[hit_pos] = new_gene;
                sample.sort_unstable();
                let (new_s, _) =
                    calculate_gsea_score(&sample, scaled_scores, ns_total, n_total, score_type);

                let reject = if is_pos {
                    new_s.compare(&threshold) == std::cmp::Ordering::Less
                } else {
                    new_s.compare(&threshold) == std::cmp::Ordering::Greater
                };

                if reject {
                    let idx = sample.binary_search(&new_gene).unwrap();
                    sample[idx] = old_gene;
                    sample.sort_unstable();
                }
            }
            next_samples.push(sample);
        }
        current_samples = next_samples;
    }

    let p_val = log_p.exp().min(1.0);
    // Statistical error estimation (matching fgsea formula)
    let log2err = (((log_p / 2.0_f64.ln()).abs().floor() + 1.0)
        * (((sample_size as f64 + 1.0) / 2.0).trigamma() - (sample_size as f64 + 1.0).trigamma()))
    .sqrt()
        / 2.0_f64.ln();

    (p_val, Some(log2err))
}
