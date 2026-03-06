use crate::core::{EnrichmentResult, Pathway, RankedList, ScoreType};
use crate::esruler_compat::EsRulerCompat;
use crate::fastgsea_compat::{
    calc_gsea_stat_cumulative_batch_f64,
    calc_gsea_stat_cumulative_batch_f64_thread_invariant_parallel,
};
use crate::rng_compat::{RLecuyerCmrgSeedCompat, RMt19937SeedCompat};
use rayon::prelude::*;
use special::Gamma;
use statrs::distribution::{Beta, ContinuousCDF};
use std::collections::{BTreeMap, HashMap, HashSet};
#[cfg(feature = "gpu")]
use std::sync::Arc;
#[cfg(feature = "gpu")]
use std::sync::atomic::{AtomicU64, Ordering};

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
            if curr_max.abs() == curr_min.abs() {
                (0.0, hits[0])
            } else if curr_max.abs() > curr_min.abs() {
                (curr_max, max_idx)
            } else {
                (curr_min, min_idx)
            }
        }
        ScoreType::Pos => (curr_max, max_idx),
        ScoreType::Neg => (curr_min, min_idx),
    }
}

// Rust port of fgsea::calcGseaStat behavior (for gseaParam=1 used in fgseaSimpleImpl).
pub fn calculate_es_fgsea(
    stats: &[f64],
    hits: &[usize],
    n_total: usize,
    score_type: ScoreType,
) -> (f64, usize) {
    if hits.is_empty() {
        return (0.0, 0);
    }

    let m = hits.len();
    if m == n_total {
        return (0.0, hits[0]);
    }

    // Match fgsea::calcGseaStat exactly (gseaParam=1):
    // rAdj <- abs(r[S]); NR <- sum(rAdj)
    let mut adj = Vec::with_capacity(m);
    let mut nr = 0.0_f64;
    for &idx in hits {
        let a = stats[idx].abs();
        adj.push(a);
        nr += a;
    }

    let mut max_p = f64::NEG_INFINITY;
    let mut min_p = f64::INFINITY;
    let mut max_i = 0usize;
    let mut min_i = 0usize;
    let mut csum = 0.0;

    for i in 0..m {
        csum += adj[i];
        let r_cum = if nr == 0.0 {
            (i + 1) as f64 / m as f64
        } else {
            csum / nr
        };
        let miss = (hits[i] - i) as f64 / (n_total - m) as f64;
        let top = r_cum - miss;
        let bottom = if nr == 0.0 {
            top - 1.0 / m as f64
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

    match score_type {
        ScoreType::Std => {
            if max_p == -min_p {
                (0.0, hits[0])
            } else if max_p > -min_p {
                (max_p, hits[max_i])
            } else {
                (min_p, hits[min_i])
            }
        }
        ScoreType::Pos => (max_p, hits[max_i]),
        ScoreType::Neg => (min_p, hits[min_i]),
    }
}

fn multilevel_error(pval: f64, sample_size: usize) -> f64 {
    if !(pval.is_finite()) || pval <= 0.0 {
        return f64::NAN;
    }
    let level = (-pval.log2() + 1.0).floor();
    (level
        * (((sample_size as f64 + 1.0) / 2.0).trigamma() - (sample_size as f64 + 1.0).trigamma()))
    .sqrt()
        / 2.0_f64.ln()
}

fn log2_qbeta(prob: f64, shape1: f64, shape2: f64) -> f64 {
    if shape1 <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if shape2 <= 0.0 {
        return 0.0;
    }
    match Beta::new(shape1, shape2) {
        Ok(beta) => beta.inverse_cdf(prob).log2(),
        Err(_) => f64::NAN,
    }
}

fn derive_fgsea_simple_seed(seed: u64) -> (u64, RMt19937SeedCompat) {
    // Mirrors first sample.int(1e9, 1) draw in fgseaMultilevel().
    let mut rng = RMt19937SeedCompat::from_r_set_seed(seed as u32);
    let simple_seed = rng.sample_int_one(1_000_000_000) as u64;
    (simple_seed, rng)
}

pub fn calculate_gsea_score(
    hits: &[usize],
    scaled_scores: &[i64],
    _ns_total: i64,
    n_total: usize,
    score_type: ScoreType,
) -> (GseaScore, usize) {
    let k = hits.len();
    let diff = (n_total - k) as i64;
    // Calculate local sum of weights for the pathway (hits)
    // The previous implementation incorrectly used the global sum passed as ns_total
    let ns: i64 = hits.iter().map(|&i| scaled_scores[i]).sum();

    let mut curr_max_num: i128 = 0;
    let mut curr_min_num: i128 = 0;
    let mut max_p = (0, 0); // (coef_ns, coef_const)
    let mut min_p = (0, 0);
    let mut m_idx = hits[0];
    let mut v_idx = hits[0];

    let mut curr_coef_ns: i64 = 0;
    for (j, &hit_idx) in hits.iter().enumerate().take(k) {
        let coef_const = (hit_idx - j) as i64;

        let num_before = (curr_coef_ns as i128 * diff as i128) - (coef_const as i128 * ns as i128);
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
        let num_at = (curr_coef_ns as i128 * diff as i128) - (coef_const as i128 * ns as i128);
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
            if curr_max_num.abs() == curr_min_num.abs() {
                (GseaScore::new(ns, 0, diff, 0), hits[0])
            } else if curr_max_num.abs() > curr_min_num.abs() {
                (GseaScore::new(ns, max_p.0, diff, max_p.1), m_idx)
            } else {
                (GseaScore::new(ns, min_p.0, diff, min_p.1), v_idx)
            }
        }
        ScoreType::Pos => (GseaScore::new(ns, max_p.0, diff, max_p.1), m_idx),
        ScoreType::Neg => (GseaScore::new(ns, min_p.0, diff, min_p.1), v_idx),
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
    run_gsea_with_sample_size(
        ranks, pathways, n_perm, seed, min_size, max_size, eps, score_type, gsea_param, 101,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn run_gsea_with_sample_size(
    ranks: &RankedList,
    pathways: &[Pathway],
    n_perm: usize,
    seed: u64,
    min_size: usize,
    max_size: usize,
    eps: f64,
    score_type: ScoreType,
    gsea_param: f64,
    sample_size: usize,
) -> Vec<EnrichmentResult> {
    run_gsea_internal(
        ranks,
        pathways,
        n_perm,
        seed,
        min_size,
        max_size,
        eps,
        score_type,
        gsea_param,
        true,
        sample_size,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn run_gsea_simple(
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
    run_gsea_simple_with_sample_size(
        ranks, pathways, n_perm, seed, min_size, max_size, eps, score_type, gsea_param, 101,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn run_gsea_simple_with_sample_size(
    ranks: &RankedList,
    pathways: &[Pathway],
    n_perm: usize,
    seed: u64,
    min_size: usize,
    max_size: usize,
    eps: f64,
    score_type: ScoreType,
    gsea_param: f64,
    sample_size: usize,
) -> Vec<EnrichmentResult> {
    run_gsea_internal(
        ranks,
        pathways,
        n_perm,
        seed,
        min_size,
        max_size,
        eps,
        score_type,
        gsea_param,
        false,
        sample_size,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn fgsea(
    ranks: &RankedList,
    pathways: &[Pathway],
    nperm: Option<usize>,
    n_perm_simple: usize,
    seed: u64,
    min_size: usize,
    max_size: usize,
    eps: f64,
    score_type: ScoreType,
    gsea_param: f64,
) -> Vec<EnrichmentResult> {
    fgsea_with_sample_size(
        ranks,
        pathways,
        nperm,
        n_perm_simple,
        seed,
        min_size,
        max_size,
        eps,
        score_type,
        gsea_param,
        101,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn fgsea_with_sample_size(
    ranks: &RankedList,
    pathways: &[Pathway],
    nperm: Option<usize>,
    n_perm_simple: usize,
    seed: u64,
    min_size: usize,
    max_size: usize,
    eps: f64,
    score_type: ScoreType,
    gsea_param: f64,
    sample_size: usize,
) -> Vec<EnrichmentResult> {
    if let Some(nperm_simple_mode) = nperm {
        run_gsea_simple_with_sample_size(
            ranks,
            pathways,
            nperm_simple_mode,
            seed,
            min_size,
            max_size,
            eps,
            score_type,
            gsea_param,
            sample_size,
        )
    } else {
        run_gsea_with_sample_size(
            ranks,
            pathways,
            n_perm_simple,
            seed,
            min_size,
            max_size,
            eps,
            score_type,
            gsea_param,
            sample_size,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn run_gsea_internal(
    ranks: &RankedList,
    pathways: &[Pathway],
    n_perm: usize,
    seed: u64,
    min_size: usize,
    max_size: usize,
    eps: f64,
    score_type: ScoreType,
    gsea_param: f64,
    allow_multilevel: bool,
    sample_size: usize,
) -> Vec<EnrichmentResult> {
    struct Working {
        pathway_name: String,
        size: usize,
        hits: Vec<usize>,
        es: f64,
        obs_es: f64,
        peak_idx: usize,
        n_le_es: usize,
        n_ge_es: usize,
        n_le_zero: usize,
        n_ge_zero: usize,
        le_zero_sum: f64,
        ge_zero_sum: f64,
        nes: Option<f64>,
        p_value: f64,
        padj: Option<f64>,
        log2err: Option<f64>,
    }

    let gene_to_idx: HashMap<String, usize> = ranks
        .genes
        .iter()
        .enumerate()
        .map(|(i, g)| (g.clone(), i))
        .collect();
    let n_total = ranks.len();

    // Match fgsea prepareStats warnings (improved wording).
    let mut seen_nonzero = HashSet::new();
    let mut tie_count = 0usize;
    for &s in &ranks.scores {
        if s != 0.0 && !seen_nonzero.insert(s.to_bits()) {
            tie_count += 1;
        }
    }
    if tie_count > 0 && n_total > 0 {
        let tie_pct = (tie_count as f64) * 100.0 / (n_total as f64);
        eprintln!(
            "Warning: detected {} tied non-zero ranking scores ({:.2}% of genes). Ties are resolved by input order and can slightly affect enrichment outcomes.",
            tie_count, tie_pct
        );
    }
    if matches!(score_type, ScoreType::Std) && ranks.scores.iter().all(|&s| s > 0.0) {
        eprintln!(
            "Warning: all ranking scores are positive while scoreType='std'. For one-tailed enrichment, consider scoreType='pos'."
        );
    }

    // Match fgsea::preparePathways() bounds behavior:
    // minSize <- max(minSize, 1)
    // maxSize <- min(maxSize, length(universe) - 1)
    let min_size = min_size.max(1);
    let sample_size = sample_size.max(1);
    let max_size = max_size.min(n_total.saturating_sub(1));
    let eps = eps.clamp(0.0, 1.0);
    let (_abs_weights, scaled_scores, ns_total) = ranks.prepare(gsea_param);
    // fgsea simple/multilevel wrapper path operates on prepareStats()-scaled integer stats.
    // Use the same scaled values (as f64) for observed ES and simple permutation stage.
    let simple_stats: Vec<f64> = scaled_scores.iter().map(|&v| v as f64).collect();
    let (simple_seed, mut r_seed_rng) = derive_fgsea_simple_seed(seed);

    // Heavy pathway preprocessing is independent per pathway; use parallel map
    // while preserving deterministic input order.
    let mut work: Vec<Working> = pathways
        .par_iter()
        .map(|pw| {
            let mut hits: Vec<usize> = pw
                .genes
                .iter()
                .filter_map(|g| gene_to_idx.get(g).copied())
                .collect();
            hits.sort_unstable();
            hits.dedup();
            if hits.len() < min_size || hits.len() > max_size {
                return None;
            }
            let (es, peak_idx) = calculate_es_fgsea(&simple_stats, &hits, n_total, score_type);
            Some(Working {
                pathway_name: pw.name.clone(),
                size: hits.len(),
                hits,
                es,
                obs_es: es,
                peak_idx,
                n_le_es: 0,
                n_ge_es: 0,
                n_le_zero: 0,
                n_ge_zero: 0,
                le_zero_sum: 0.0,
                ge_zero_sum: 0.0,
                nes: None,
                p_value: f64::NAN,
                padj: None,
                log2err: None,
            })
        })
        .collect::<Vec<_>>()
        .into_iter()
        .flatten()
        .collect();

    if n_perm > 0 && !work.is_empty() {
        if work.len() == 1 {
            // Match fgseaSimpleImpl(toKeepLength == 1) semantics:
            // - In fgseaSimple: draw seeds vector from MT, then inside bplapply
            //   `set.seed(seeds[i])` under SerialParam (L'Ecuyer-CMRG RNG kind).
            // - In fgseaMultilevel simple stage: one seed draw and one chunk.
            let (perm_chunks, chunk_seeds): (Vec<usize>, Vec<u64>) = if allow_multilevel {
                (vec![n_perm], vec![simple_seed])
            } else {
                let granularity = 1000usize.max(n_perm.div_ceil(128));
                let mut rem = n_perm;
                let mut chunks = Vec::new();
                while rem >= granularity {
                    chunks.push(granularity);
                    rem -= granularity;
                }
                if rem > 0 {
                    chunks.push(rem);
                }
                let mut seeds = Vec::with_capacity(chunks.len());
                if !chunks.is_empty() {
                    seeds.push(simple_seed);
                    for _ in 1..chunks.len() {
                        seeds.push(r_seed_rng.sample_int_one(1_000_000_000) as u64);
                    }
                }
                (chunks, seeds)
            };

            let k = work[0].size;
            let pathway_score = work[0].es;
            for (chunk_iters, chunk_seed) in perm_chunks.into_iter().zip(chunk_seeds.into_iter()) {
                let mut r_rng = RLecuyerCmrgSeedCompat::from_r_set_seed(chunk_seed as u32);
                for _ in 0..chunk_iters {
                    let mut rand_hits: Vec<usize> = r_rng
                        .sample_int_no_replace(n_total, k)
                        .into_iter()
                        .map(|x| x - 1)
                        .collect();
                    rand_hits.sort_unstable();
                    let (rand_es, _) =
                        calculate_es_fgsea(&simple_stats, &rand_hits, n_total, score_type);
                    if rand_es <= pathway_score {
                        work[0].n_le_es += 1;
                    }
                    if rand_es >= pathway_score {
                        work[0].n_ge_es += 1;
                    }
                    if rand_es <= 0.0 {
                        work[0].n_le_zero += 1;
                        work[0].le_zero_sum += rand_es;
                    }
                    if rand_es >= 0.0 {
                        work[0].n_ge_zero += 1;
                        work[0].ge_zero_sum += rand_es;
                    }
                }
            }
        } else {
            let pathway_scores: Vec<f64> = work.iter().map(|w| w.es).collect();
            let pathways_sizes: Vec<usize> = work.iter().map(|w| w.size).collect();
            let counts = if rayon::current_num_threads() > 1 && work.len() >= 128 {
                calc_gsea_stat_cumulative_batch_f64_thread_invariant_parallel(
                    &simple_stats,
                    1.0,
                    &pathway_scores,
                    &pathways_sizes,
                    n_perm,
                    simple_seed,
                    score_type,
                )
            } else {
                calc_gsea_stat_cumulative_batch_f64(
                    &simple_stats,
                    1.0,
                    &pathway_scores,
                    &pathways_sizes,
                    n_perm,
                    simple_seed,
                    score_type,
                )
            };
            for (i, w) in work.iter_mut().enumerate() {
                w.n_le_es = counts.le_es[i];
                w.n_ge_es = counts.ge_es[i];
                w.n_le_zero = counts.le_zero[i];
                w.n_ge_zero = counts.ge_zero[i];
                w.le_zero_sum = counts.le_zero_sum[i];
                w.ge_zero_sum = counts.ge_zero_sum[i];
            }
        }
    }

    let mut n_more_extreme_vec = vec![0usize; work.len()];
    let mut mode_fraction_vec = vec![0usize; work.len()];
    let mut simple_error_vec = vec![f64::NAN; work.len()];
    let mut mult_error_vec = vec![f64::NAN; work.len()];

    for (wi, w) in work.iter_mut().enumerate() {
        let le_zero_mean = if w.n_le_zero > 0 {
            w.le_zero_sum / w.n_le_zero as f64
        } else {
            0.0
        };
        let ge_zero_mean = if w.n_ge_zero > 0 {
            w.ge_zero_sum / w.n_ge_zero as f64
        } else {
            0.0
        };

        w.nes = match score_type {
            ScoreType::Std => {
                if w.es > 0.0 && ge_zero_mean != 0.0 {
                    Some(w.es / ge_zero_mean)
                } else if w.es <= 0.0 && le_zero_mean != 0.0 {
                    Some(w.es / le_zero_mean.abs())
                } else {
                    None
                }
            }
            ScoreType::Pos => {
                if w.es >= 0.0 && ge_zero_mean != 0.0 {
                    Some(w.es / ge_zero_mean)
                } else {
                    None
                }
            }
            ScoreType::Neg => {
                if w.es <= 0.0 && le_zero_mean != 0.0 {
                    Some(w.es / le_zero_mean.abs())
                } else {
                    None
                }
            }
        };

        if w.nes.is_some() {
            let p_le = (w.n_le_es + 1) as f64 / (w.n_le_zero + 1) as f64;
            let p_ge = (w.n_ge_es + 1) as f64 / (w.n_ge_zero + 1) as f64;
            w.p_value = p_le.min(p_ge);
        }

        let n_more_extreme = match score_type {
            ScoreType::Std => {
                if w.es > 0.0 {
                    w.n_ge_es
                } else {
                    w.n_le_es
                }
            }
            ScoreType::Pos => w.n_ge_es,
            ScoreType::Neg => w.n_le_es,
        };

        let mode_fraction = match score_type {
            ScoreType::Std => {
                if w.es >= 0.0 {
                    w.n_ge_zero
                } else {
                    w.n_le_zero
                }
            }
            ScoreType::Pos => w.n_ge_zero,
            ScoreType::Neg => w.n_le_zero,
        };

        n_more_extreme_vec[wi] = n_more_extreme;
        mode_fraction_vec[wi] = mode_fraction;

        w.log2err = if n_perm > 0 && w.p_value.is_finite() {
            Some(
                1.0 / 2.0_f64.ln()
                    * (((n_more_extreme + 1) as f64).trigamma() - ((n_perm + 1) as f64).trigamma())
                        .sqrt(),
            )
        } else {
            None
        };

        if allow_multilevel && mode_fraction < 10 {
            w.p_value = f64::NAN;
            w.nes = None;
            w.log2err = None;
        } else if allow_multilevel && n_perm > 0 && w.p_value.is_finite() {
            let n_more = n_more_extreme as f64;
            let n_perm_f = n_perm as f64;
            let left = log2_qbeta(0.025, n_more, n_perm_f - n_more + 1.0);
            let right = log2_qbeta(0.975, n_more + 1.0, n_perm_f - n_more);
            let crude = ((n_more + 1.0) / (n_perm_f + 1.0)).log2();
            let simple_error = 0.5 * (crude - left).max(right - crude);
            let mult_error = multilevel_error((n_more + 1.0) / (n_perm_f + 1.0), sample_size);
            simple_error_vec[wi] = simple_error;
            mult_error_vec[wi] = mult_error;
        }
    }

    if allow_multilevel && n_perm > 0 && !work.is_empty() {
        let mut multilevel_groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for i in 0..work.len() {
            if work[i].p_value.is_finite()
                && mode_fraction_vec[i] >= 10
                && mult_error_vec[i].is_finite()
                && mult_error_vec[i] < simple_error_vec[i]
            {
                multilevel_groups.entry(work[i].size).or_default().push(i);
            }
        }

        let multilevel_seed = if multilevel_groups.is_empty() {
            None
        } else {
            // fgseaMultilevel.R samples pathway-size group order before taking multilevel seed:
            // indxs <- sample(1:length(multilevelPathwaysList))
            // seed <- sample.int(1e9, 1)
            r_seed_rng.consume_sample_shuffle(multilevel_groups.len());
            Some(r_seed_rng.sample_int_one(1_000_000_000) as u64)
        };

        let multilevel_groups_vec: Vec<Vec<usize>> = multilevel_groups.into_values().collect();
        let group_seed = multilevel_seed.unwrap_or(simple_seed);
        let run_group = |idxs: Vec<usize>| {
            let k = work[idxs[0]].size;
            let denom_prob_min = idxs
                .iter()
                .map(|&i| (mode_fraction_vec[i] + 1) as f64 / (n_perm + 1) as f64)
                .fold(f64::INFINITY, f64::min);
            let eps_group = eps * denom_prob_min;
            let obs_es: Vec<f64> = idxs.iter().map(|&i| work[i].obs_es).collect();
            let ml = run_multilevel_gsea_group(
                n_total,
                &scaled_scores,
                ns_total,
                k,
                &obs_es,
                score_type,
                sample_size,
                group_seed,
                eps_group,
            );
            (idxs, ml)
        };

        type MultilevelGroupResult = (Vec<usize>, Vec<(f64, bool, Option<f64>)>);
        let multilevel_results: Vec<MultilevelGroupResult> =
            if rayon::current_num_threads() > 1 && multilevel_groups_vec.len() > 1 {
                multilevel_groups_vec
                    .into_par_iter()
                    .map(run_group)
                    .collect()
            } else {
                multilevel_groups_vec.into_iter().map(run_group).collect()
            };

        for (idxs, ml) in multilevel_results {
            for (local_i, &global_i) in idxs.iter().enumerate() {
                let (m_p, is_cp_ge_half, _m_err) = ml[local_i];
                let denom_prob = (mode_fraction_vec[global_i] + 1) as f64 / (n_perm + 1) as f64;
                work[global_i].p_value = (m_p / denom_prob).min(1.0);
                if work[global_i].p_value < eps {
                    work[global_i].p_value = eps;
                    work[global_i].log2err = None;
                } else if is_cp_ge_half {
                    work[global_i].log2err =
                        Some(multilevel_error(work[global_i].p_value, sample_size));
                } else {
                    work[global_i].log2err = None;
                }
            }
        }
    }

    let mut final_results: Vec<EnrichmentResult> = work
        .into_iter()
        .map(|w| {
            let leading_edge: Vec<String> = match score_type {
                ScoreType::Pos => w
                    .hits
                    .iter()
                    .filter(|&&idx| idx <= w.peak_idx)
                    .map(|&idx| ranks.genes[idx].clone())
                    .collect(),
                ScoreType::Neg => {
                    let mut le: Vec<String> = w
                        .hits
                        .iter()
                        .filter(|&&idx| idx >= w.peak_idx)
                        .map(|&idx| ranks.genes[idx].clone())
                        .collect();
                    le.reverse();
                    le
                }
                ScoreType::Std => {
                    if w.es > 0.0 {
                        w.hits
                            .iter()
                            .filter(|&&idx| idx <= w.peak_idx)
                            .map(|&idx| ranks.genes[idx].clone())
                            .collect()
                    } else if w.es < 0.0 {
                        let mut le: Vec<String> = w
                            .hits
                            .iter()
                            .filter(|&&idx| idx >= w.peak_idx)
                            .map(|&idx| ranks.genes[idx].clone())
                            .collect();
                        le.reverse();
                        le
                    } else {
                        Vec::new()
                    }
                }
            };

            EnrichmentResult {
                pathway_name: w.pathway_name,
                size: w.size,
                es: w.es,
                nes: w.nes,
                p_value: w.p_value,
                padj: w.padj,
                log2err: w.log2err,
                leading_edge,
            }
        })
        .collect();

    if !final_results.is_empty() {
        let mut indices: Vec<usize> = (0..final_results.len())
            .filter(|&i| final_results[i].p_value.is_finite())
            .collect();
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

    final_results.sort_by(|a, b| a.pathway_name.cmp(&b.pathway_name));
    final_results
}
#[allow(clippy::too_many_arguments)]
pub fn run_multilevel_gsea(
    n_total: usize,
    scaled_scores: &[i64],
    _ns_total: i64,
    k: usize,
    obs_es: f64,
    score_type: ScoreType,
    sample_size: usize,
    seed: u64,
    eps: f64,
) -> (f64, Option<f64>) {
    let (p, _cp_ge_half, err) = run_multilevel_gsea_impl(
        n_total,
        scaled_scores,
        k,
        obs_es,
        score_type,
        sample_size,
        seed,
        eps,
    );
    (p, err)
}

#[allow(clippy::too_many_arguments)]
fn run_multilevel_gsea_impl(
    n_total: usize,
    scaled_scores: &[i64],
    k: usize,
    obs_es: f64,
    score_type: ScoreType,
    sample_size: usize,
    seed: u64,
    eps: f64,
) -> (f64, bool, Option<f64>) {
    run_multilevel_gsea_esruler(
        n_total,
        scaled_scores,
        k,
        obs_es,
        score_type,
        sample_size,
        seed,
        eps,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_multilevel_gsea_group(
    n_total: usize,
    scaled_scores: &[i64],
    _ns_total: i64,
    k: usize,
    obs_es_list: &[f64],
    score_type: ScoreType,
    sample_size: usize,
    seed: u64,
    eps: f64,
) -> Vec<(f64, bool, Option<f64>)> {
    run_multilevel_gsea_esruler_group(
        n_total,
        scaled_scores,
        k,
        obs_es_list,
        score_type,
        sample_size,
        seed,
        eps,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_multilevel_gsea_esruler(
    n_total: usize,
    scaled_scores: &[i64],
    k: usize,
    obs_es: f64,
    score_type: ScoreType,
    sample_size: usize,
    seed: u64,
    eps: f64,
) -> (f64, bool, Option<f64>) {
    if k == 0 || n_total == 0 {
        return (1.0, true, Some(0.0));
    }
    let pos_ranks: Vec<i64> = scaled_scores.iter().map(|v| v.abs()).collect();
    let mut neg_ranks = pos_ranks.clone();
    neg_ranks.reverse();

    let mut es_pos = EsRulerCompat::new(pos_ranks, sample_size, k, 1.0, false);
    let mut es_neg = EsRulerCompat::new(neg_ranks, sample_size, k, 1.0, false);

    let cur_es = obs_es;
    if cur_es >= 0.0 {
        es_pos.extend(cur_es.abs(), seed, eps);
    } else {
        es_neg.extend(cur_es.abs(), seed, eps);
    }

    let sign = matches!(score_type, ScoreType::Pos | ScoreType::Neg);
    let (p, _is_cp_ge_half, err) = if cur_es >= 0.0 {
        es_pos.get_pvalue(cur_es.abs(), eps, sign)
    } else {
        es_neg.get_pvalue(cur_es.abs(), eps, sign)
    };

    if err.is_finite() {
        (p, _is_cp_ge_half, Some(err))
    } else {
        (p, _is_cp_ge_half, None)
    }
}

#[allow(clippy::too_many_arguments)]
fn run_multilevel_gsea_esruler_group(
    n_total: usize,
    scaled_scores: &[i64],
    k: usize,
    obs_es_list: &[f64],
    score_type: ScoreType,
    sample_size: usize,
    seed: u64,
    eps: f64,
) -> Vec<(f64, bool, Option<f64>)> {
    if k == 0 || n_total == 0 || obs_es_list.is_empty() {
        return vec![(1.0, true, Some(0.0)); obs_es_list.len()];
    }

    let pos_ranks: Vec<i64> = scaled_scores.iter().map(|v| v.abs()).collect();
    let mut neg_ranks = pos_ranks.clone();
    neg_ranks.reverse();

    let mut es_pos = EsRulerCompat::new(pos_ranks, sample_size, k, 1.0, false);
    let mut es_neg = EsRulerCompat::new(neg_ranks, sample_size, k, 1.0, false);

    let mut max_es = f64::NEG_INFINITY;
    let mut min_es = f64::INFINITY;
    for &es in obs_es_list {
        if es > max_es {
            max_es = es;
        }
        if es < min_es {
            min_es = es;
        }
    }

    if max_es >= 0.0 {
        es_pos.extend(max_es.abs(), seed, eps);
    }
    if min_es < 0.0 {
        es_neg.extend(min_es.abs(), seed, eps);
    }

    let sign = matches!(score_type, ScoreType::Pos | ScoreType::Neg);
    obs_es_list
        .iter()
        .copied()
        .map(|cur_es| {
            let (p, _is_cp_ge_half, err) = if cur_es >= 0.0 {
                es_pos.get_pvalue(cur_es.abs(), eps, sign)
            } else {
                es_neg.get_pvalue(cur_es.abs(), eps, sign)
            };
            if err.is_finite() {
                (p, _is_cp_ge_half, Some(err))
            } else {
                (p, _is_cp_ge_half, None)
            }
        })
        .collect()
}

#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
pub fn run_gsea_gpu(
    ranks: &RankedList,
    pathways: &[Pathway],
    n_perm: usize,
    seed: u64,
    min_size: usize,
    max_size: usize,
    score_type: ScoreType,
    gsea_param: f64,
) -> Result<Vec<EnrichmentResult>, anyhow::Error> {
    use rsfgsea_gpu::GpuEngine;
    use std::collections::BTreeMap;

    let sample_size = 101usize;
    let eps = 1e-50_f64;

    let (abs_weights, scaled_scores, _ns_total) = ranks.prepare(gsea_param);
    let abs_weights_f32: Vec<f32> = abs_weights.iter().map(|&w| w as f32).collect();

    let gene_to_idx: HashMap<String, usize> = ranks
        .genes
        .iter()
        .enumerate()
        .map(|(i, g)| (g.clone(), i))
        .collect();

    // 1. Group pathways by size
    let mut by_size: BTreeMap<usize, Vec<(usize, Vec<usize>)>> = BTreeMap::new();
    for (i, p) in pathways.iter().enumerate() {
        let hits: Vec<usize> = p
            .genes
            .iter()
            .filter_map(|g| gene_to_idx.get(g).copied())
            .collect();
        let k = hits.len();
        if k >= min_size && k <= max_size {
            by_size.entry(k).or_default().push((i, hits));
        }
    }

    let runtime = tokio::runtime::Runtime::new()?;
    let engine = runtime.block_on(GpuEngine::new())?;
    let scores_buffer = engine.upload_scores(&abs_weights_f32);
    let gpu_score_type = match score_type {
        ScoreType::Std => 0,
        ScoreType::Pos => 1,
        ScoreType::Neg => 2,
    };

    let mut results = vec![None; pathways.len()];
    let gpu_verbose = std::env::var("RSFGSEA_GPU_VERBOSE")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let total_groups = by_size.len();
    let mut group_idx = 0usize;

    let total_null_gen_time = Arc::new(AtomicU64::new(0));
    let total_pure_screening_time = Arc::new(AtomicU64::new(0));
    let total_multilevel_time = Arc::new(AtomicU64::new(0));

    for (k, group) in by_size {
        group_idx += 1;
        if gpu_verbose
            || group_idx == 1
            || group_idx == total_groups
            || group_idx.is_multiple_of(25)
        {
            println!(
                "GPU null group {}/{}: {} pathways (k={})",
                group_idx,
                total_groups,
                group.len(),
                k
            );
        }

        let gen_start = std::time::Instant::now();
        let mut null_distribution = engine.generate_null_distribution(
            &scores_buffer,
            k,
            ranks.len(),
            n_perm,
            seed,
            gpu_score_type,
        )?;
        total_null_gen_time.fetch_add(
            gen_start.elapsed().as_micros() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );

        // Sort null distribution for O(log N) lookup
        null_distribution.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

        // Pre-calculate stats for the null distribution
        let mut n_le_zero = 0u64;
        let mut n_ge_zero = 0u64;
        let mut le_zero_sum = 0.0f64;
        let mut ge_zero_sum = 0.0f64;
        for &es in &null_distribution {
            let perm_es = es as f64;
            if perm_es <= 0.0 {
                n_le_zero += 1;
                le_zero_sum += perm_es;
            }
            if perm_es >= 0.0 {
                n_ge_zero += 1;
                ge_zero_sum += perm_es;
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

        // Process pathways of this size in parallel
        let group_results: Vec<Option<EnrichmentResult>> = group
            .par_iter()
            .map(|(orig_idx, hits)| {
                let screening_start = std::time::Instant::now();
                let mut sorted_hits = hits.clone();
                sorted_hits.sort_unstable();
                let (obs_es, peak_idx) =
                    calculate_es(&sorted_hits, &abs_weights, ranks.len(), score_type);

                // Use binary search for fast p-value count
                let obs_es_f32 = obs_es as f32;
                let (n_le_es, n_ge_es) = {
                    let idx_le = null_distribution
                        .binary_search_by(|val| val.partial_cmp(&obs_es_f32).unwrap())
                        .map_or_else(|idx| idx, |idx| idx + 1); // Count elements <= obs_es_f32
                    let idx_ge = null_distribution
                        .binary_search_by(|val| val.partial_cmp(&obs_es_f32).unwrap())
                        .unwrap_or_else(|idx| idx); // Find first element >= obs_es_f32

                    (idx_le as u64, (null_distribution.len() - idx_ge) as u64)
                };

                let p_le = (n_le_es + 1) as f64 / (n_le_zero + 1) as f64;
                let p_ge = (n_ge_es + 1) as f64 / (n_ge_zero + 1) as f64;
                let p_value_simple = match score_type {
                    ScoreType::Std => p_le.min(p_ge),
                    ScoreType::Pos => p_ge,
                    ScoreType::Neg => p_le,
                };
                total_pure_screening_time.fetch_add(
                    screening_start.elapsed().as_micros() as u64,
                    Ordering::Relaxed,
                );

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

                let leading_edge: Vec<String> = match score_type {
                    ScoreType::Pos => sorted_hits
                        .iter()
                        .filter(|&&idx| idx <= peak_idx)
                        .map(|&idx| ranks.genes[idx].clone())
                        .collect(),
                    ScoreType::Neg => {
                        let mut le: Vec<String> = sorted_hits
                            .iter()
                            .filter(|&&idx| idx >= peak_idx)
                            .map(|&idx| ranks.genes[idx].clone())
                            .collect();
                        le.reverse();
                        le
                    }
                    ScoreType::Std => {
                        if obs_es > 0.0 {
                            sorted_hits
                                .iter()
                                .filter(|&&idx| idx <= peak_idx)
                                .map(|&idx| ranks.genes[idx].clone())
                                .collect()
                        } else if obs_es < 0.0 {
                            let mut le: Vec<String> = sorted_hits
                                .iter()
                                .filter(|&&idx| idx >= peak_idx)
                                .map(|&idx| ranks.genes[idx].clone())
                                .collect();
                            le.reverse();
                            le
                        } else {
                            Vec::new()
                        }
                    }
                };

                let n_more_extreme = match score_type {
                    ScoreType::Std => {
                        if obs_es > 0.0 {
                            n_ge_es
                        } else {
                            n_le_es
                        }
                    }
                    ScoreType::Pos => n_ge_es,
                    ScoreType::Neg => n_le_es,
                };
                let mode_fraction = match score_type {
                    ScoreType::Std => {
                        if obs_es >= 0.0 {
                            n_ge_zero
                        } else {
                            n_le_zero
                        }
                    }
                    ScoreType::Pos => n_ge_zero,
                    ScoreType::Neg => n_le_zero,
                };
                let simple_log2err = if p_value_simple.is_finite() {
                    Some(
                        1.0 / 2.0_f64.ln()
                            * (((n_more_extreme + 1) as f64).trigamma()
                                - ((n_perm + 1) as f64).trigamma())
                            .sqrt(),
                    )
                } else {
                    None
                };
                let should_run_multilevel = if mode_fraction < 10 || !p_value_simple.is_finite() {
                    false
                } else {
                    let n_more = n_more_extreme as f64;
                    let n_perm_f = n_perm as f64;
                    let left = log2_qbeta(0.025, n_more, n_perm_f - n_more + 1.0);
                    let right = log2_qbeta(0.975, n_more + 1.0, n_perm_f - n_more);
                    let crude = ((n_more + 1.0) / (n_perm_f + 1.0)).log2();
                    let simple_error = 0.5 * (crude - left).max(right - crude);
                    let mult_error =
                        multilevel_error((n_more + 1.0) / (n_perm_f + 1.0), sample_size);
                    mult_error < simple_error
                };
                if should_run_multilevel {
                    let ml_start = std::time::Instant::now();
                    let (m_p, is_cp_ge_half, _m_err) = run_multilevel_gsea_impl(
                        ranks.len(),
                        &scaled_scores,
                        k,
                        obs_es,
                        score_type,
                        sample_size,
                        seed + *orig_idx as u64,
                        eps,
                    );

                    total_multilevel_time
                        .fetch_add(ml_start.elapsed().as_micros() as u64, Ordering::Relaxed);

                    let denom_prob = (mode_fraction + 1) as f64 / (n_perm + 1) as f64;
                    let mut p_value_ml = (m_p / denom_prob).min(1.0);
                    let log2err = if p_value_ml < eps {
                        p_value_ml = eps;
                        None
                    } else if is_cp_ge_half {
                        Some(multilevel_error(p_value_ml, sample_size))
                    } else {
                        None
                    };

                    Some(EnrichmentResult {
                        pathway_name: pathways[*orig_idx].name.clone(),
                        p_value: p_value_ml,
                        padj: None,
                        es: obs_es,
                        nes,
                        log2err,
                        size: k,
                        leading_edge,
                    })
                } else {
                    Some(EnrichmentResult {
                        pathway_name: pathways[*orig_idx].name.clone(),
                        p_value: p_value_simple,
                        padj: None,
                        es: obs_es,
                        nes,
                        log2err: simple_log2err,
                        size: k,
                        leading_edge,
                    })
                }
            })
            .collect();

        for (i, res) in group_results.into_iter().enumerate() {
            if let Some(r) = res {
                results[group[i].0] = Some(r);
            }
        }
    }

    println!("\nGPU Execution Timings (Total Compute Across All Threads):");
    println!(
        "  Null Distribution Gen: {} ms",
        total_null_gen_time.load(Ordering::Relaxed) / 1000
    );
    println!(
        "  Pure Screening Pass:   {} ms",
        total_pure_screening_time.load(Ordering::Relaxed) / 1000
    );
    println!(
        "  Multilevel Pass:       {} ms",
        total_multilevel_time.load(Ordering::Relaxed) / 1000
    );

    let mut final_results: Vec<EnrichmentResult> = results.into_iter().flatten().collect();

    if !final_results.is_empty() {
        let mut indices: Vec<usize> = (0..final_results.len())
            .filter(|&i| final_results[i].p_value.is_finite())
            .collect();
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

    final_results.sort_by(|a, b| a.pathway_name.cmp(&b.pathway_name));

    Ok(final_results)
}
