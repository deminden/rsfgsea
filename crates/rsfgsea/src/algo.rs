use crate::core::{EnrichmentResult, Pathway, RankedList, ScoreType};
use crate::esruler_compat::EsRulerCompat;
use crate::fastgsea_compat::{calc_gsea_stat_cumulative_batch_f64, calc_gsea_stat_cumulative_f64};
use crate::rng_compat::{Mt19937Compat, RMt19937SeedCompat, combination, uid_wrapper};
use special::Gamma;
use statrs::distribution::{Beta, ContinuousCDF};
use std::collections::{BTreeMap, HashMap};
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MultilevelEngine {
    EsRulerCompat,
    Legacy,
}

impl MultilevelEngine {
    pub fn from_env() -> Self {
        match std::env::var("RSFGSEA_MULTILEVEL_ENGINE")
            .unwrap_or_else(|_| "esruler".to_string())
            .to_lowercase()
            .as_str()
        {
            "legacy" => Self::Legacy,
            _ => Self::EsRulerCompat,
        }
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
    struct Working {
        pathway_name: String,
        size: usize,
        hits: Vec<usize>,
        es: f64,
        obs_es: GseaScore,
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
    let (abs_weights, scaled_scores, ns_total) = ranks.prepare(gsea_param);
    let (simple_seed, mut r_seed_rng) = derive_fgsea_simple_seed(seed);

    let mut work = Vec::new();
    for pw in pathways {
        let mut hits: Vec<usize> = pw
            .genes
            .iter()
            .filter_map(|g| gene_to_idx.get(g).copied())
            .collect();
        if hits.len() < min_size || hits.len() > max_size {
            continue;
        }
        hits.sort_unstable();
        hits.dedup();
        let (obs_es, _) =
            calculate_gsea_score(&hits, &scaled_scores, ns_total, n_total, score_type);
        let (es, peak_idx) = calculate_es_fgsea(&abs_weights, &hits, n_total, score_type);
        work.push(Working {
            pathway_name: pw.name.clone(),
            size: hits.len(),
            hits,
            es,
            obs_es,
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
        });
    }

    if n_perm > 0 && !work.is_empty() {
        if work.len() == 1 {
            let mut rng = Mt19937Compat::new(simple_seed as u32);
            let k = work[0].size;
            let pathway_score = work[0].es;
            for _ in 0..n_perm {
                let mut rand_sample = combination(1, n_total, k, &mut rng);
                rand_sample.sort_unstable();
                let rand_es =
                    calc_gsea_stat_cumulative_f64(&abs_weights, &rand_sample, 1.0, score_type)
                        [k - 1];
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
        } else {
            let pathway_scores: Vec<f64> = work.iter().map(|w| w.es).collect();
            let pathways_sizes: Vec<usize> = work.iter().map(|w| w.size).collect();
            let counts = calc_gsea_stat_cumulative_batch_f64(
                &abs_weights,
                1.0,
                &pathway_scores,
                &pathways_sizes,
                n_perm,
                simple_seed,
                score_type,
            );
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

        if mode_fraction < 10 {
            w.p_value = f64::NAN;
            w.nes = None;
            w.log2err = None;
        } else if n_perm > 0 && w.p_value.is_finite() {
            let n_more = n_more_extreme as f64;
            let n_perm_f = n_perm as f64;
            let left = log2_qbeta(0.025, n_more, n_perm_f - n_more + 1.0);
            let right = log2_qbeta(0.975, n_more + 1.0, n_perm_f - n_more);
            let crude = ((n_more + 1.0) / (n_perm_f + 1.0)).log2();
            let simple_error = 0.5 * (crude - left).max(right - crude);
            let mult_error = multilevel_error((n_more + 1.0) / (n_perm_f + 1.0), 101);
            simple_error_vec[wi] = simple_error;
            mult_error_vec[wi] = mult_error;
        }
    }

    if n_perm > 0 && !work.is_empty() {
        let mut multilevel_groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for i in 0..work.len() {
            if work[i].p_value.is_finite()
                && mode_fraction_vec[i] >= 10
                && mult_error_vec[i].is_finite()
                && simple_error_vec[i].is_finite()
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

        for (_size, idxs) in multilevel_groups {
            let k = work[idxs[0]].size;
            let denom_prob_min = idxs
                .iter()
                .map(|&i| (mode_fraction_vec[i] + 1) as f64 / (n_perm + 1) as f64)
                .fold(f64::INFINITY, f64::min);
            let eps_group = eps * denom_prob_min;

            let obs_es: Vec<GseaScore> = idxs.iter().map(|&i| work[i].obs_es).collect();
            let ml = run_multilevel_gsea_group_with_engine(
                n_total,
                &scaled_scores,
                ns_total,
                k,
                &obs_es,
                score_type,
                101,
                multilevel_seed.unwrap_or(simple_seed),
                eps_group,
                MultilevelEngine::from_env(),
            );

            for (local_i, &global_i) in idxs.iter().enumerate() {
                let (m_p, is_cp_ge_half, _m_err) = ml[local_i];
                let denom_prob = (mode_fraction_vec[global_i] + 1) as f64 / (n_perm + 1) as f64;
                work[global_i].p_value = (m_p / denom_prob).min(1.0);
                if work[global_i].p_value < eps {
                    work[global_i].p_value = eps;
                    work[global_i].log2err = None;
                } else if is_cp_ge_half {
                    work[global_i].log2err = Some(multilevel_error(work[global_i].p_value, 101));
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

    final_results
}
pub fn run_multilevel_gsea(
    n_total: usize,
    scaled_scores: &[i64],
    ns_total: i64,
    k: usize,
    obs_es: GseaScore,
    score_type: ScoreType,
    sample_size: usize,
    seed: u64,
    eps: f64,
) -> (f64, Option<f64>) {
    let (p, _cp_ge_half, err) = run_multilevel_gsea_with_engine_details(
        n_total,
        scaled_scores,
        ns_total,
        k,
        obs_es,
        score_type,
        sample_size,
        seed,
        eps,
        MultilevelEngine::from_env(),
    );
    (p, err)
}

#[allow(clippy::too_many_arguments)]
pub fn run_multilevel_gsea_with_engine(
    n_total: usize,
    scaled_scores: &[i64],
    ns_total: i64,
    k: usize,
    obs_es: GseaScore,
    score_type: ScoreType,
    sample_size: usize,
    seed: u64,
    eps: f64,
    engine: MultilevelEngine,
) -> (f64, Option<f64>) {
    let (p, _cp_ge_half, err) = run_multilevel_gsea_with_engine_details(
        n_total,
        scaled_scores,
        ns_total,
        k,
        obs_es,
        score_type,
        sample_size,
        seed,
        eps,
        engine,
    );
    (p, err)
}

#[allow(clippy::too_many_arguments)]
fn run_multilevel_gsea_with_engine_details(
    n_total: usize,
    scaled_scores: &[i64],
    ns_total: i64,
    k: usize,
    obs_es: GseaScore,
    score_type: ScoreType,
    sample_size: usize,
    seed: u64,
    eps: f64,
    engine: MultilevelEngine,
) -> (f64, bool, Option<f64>) {
    match engine {
        MultilevelEngine::EsRulerCompat => run_multilevel_gsea_esruler(
            n_total,
            scaled_scores,
            k,
            obs_es,
            score_type,
            sample_size,
            seed,
            eps,
        ),
        MultilevelEngine::Legacy => run_multilevel_gsea_legacy(
            n_total,
            scaled_scores,
            ns_total,
            k,
            obs_es,
            score_type,
            sample_size,
            seed,
            eps,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_multilevel_gsea_group_with_engine(
    n_total: usize,
    scaled_scores: &[i64],
    ns_total: i64,
    k: usize,
    obs_es_list: &[GseaScore],
    score_type: ScoreType,
    sample_size: usize,
    seed: u64,
    eps: f64,
    engine: MultilevelEngine,
) -> Vec<(f64, bool, Option<f64>)> {
    match engine {
        MultilevelEngine::EsRulerCompat => run_multilevel_gsea_esruler_group(
            n_total,
            scaled_scores,
            k,
            obs_es_list,
            score_type,
            sample_size,
            seed,
            eps,
        ),
        MultilevelEngine::Legacy => obs_es_list
            .iter()
            .copied()
            .map(|obs_es| {
                run_multilevel_gsea_legacy(
                    n_total,
                    scaled_scores,
                    ns_total,
                    k,
                    obs_es,
                    score_type,
                    sample_size,
                    seed,
                    eps,
                )
            })
            .collect(),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_multilevel_gsea_esruler(
    n_total: usize,
    scaled_scores: &[i64],
    k: usize,
    obs_es: GseaScore,
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

    let cur_es = obs_es.get_double();
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
        (p, true, Some(err))
    } else {
        (p, true, None)
    }
}

#[allow(clippy::too_many_arguments)]
fn run_multilevel_gsea_esruler_group(
    n_total: usize,
    scaled_scores: &[i64],
    k: usize,
    obs_es_list: &[GseaScore],
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
    for &obs in obs_es_list {
        let es = obs.get_double();
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
        .map(|obs| {
            let cur_es = obs.get_double();
            let (p, _is_cp_ge_half, err) = if cur_es >= 0.0 {
                es_pos.get_pvalue(cur_es.abs(), eps, sign)
            } else {
                es_neg.get_pvalue(cur_es.abs(), eps, sign)
            };
            if err.is_finite() {
                (p, true, Some(err))
            } else {
                (p, true, None)
            }
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn run_multilevel_gsea_legacy(
    n_total: usize,
    scaled_scores: &[i64],
    ns_total: i64,
    k: usize,
    obs_es: GseaScore,
    score_type: ScoreType,
    sample_size: usize,
    seed: u64,
    _eps: f64,
) -> (f64, bool, Option<f64>) {
    let mut rng = Mt19937Compat::new(seed as u32);

    // Initial samples (simple permutation)
    let mut current_samples: Vec<Vec<usize>> = (0..sample_size)
        .map(|_| {
            let mut s = combination(0, n_total - 1, k, &mut rng);
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
            let src_idx = top_indices[uid_wrapper(0, top_indices.len() - 1, &mut rng)];
            let mut sample = current_samples[src_idx].clone();

            // Perturbate
            let n_swaps = (k as f64 * 0.1).ceil() as usize;
            for _ in 0..n_swaps {
                let hit_pos = uid_wrapper(0, k - 1, &mut rng);
                let old_gene = sample[hit_pos];
                let mut new_gene = uid_wrapper(0, n_total - 1, &mut rng);
                while sample.binary_search(&new_gene).is_ok() {
                    new_gene = uid_wrapper(0, n_total - 1, &mut rng);
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

    (p_val, true, Some(log2err))
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

    let (abs_weights, scaled_scores, ns_total) = ranks.prepare(gsea_param);
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

    let total_null_gen_time = Arc::new(AtomicU64::new(0));
    let total_pure_screening_time = Arc::new(AtomicU64::new(0));
    let total_multilevel_time = Arc::new(AtomicU64::new(0));

    for (k, group) in by_size {
        println!(
            "Processing {} pathways of size k={} using shared null distribution...",
            group.len(),
            k
        );

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

                let p_value_simple = if obs_es > 0.0 {
                    (n_ge_es + 1) as f64 / (n_perm + 1) as f64
                } else {
                    (n_le_es + 1) as f64 / (n_perm + 1) as f64
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

                let leading_edge: Vec<String> = if obs_es >= 0.0 {
                    sorted_hits
                        .iter()
                        .filter(|&&idx| idx <= peak_idx)
                        .map(|&idx| ranks.genes[idx].clone())
                        .collect()
                } else {
                    sorted_hits
                        .iter()
                        .filter(|&&idx| idx >= peak_idx)
                        .map(|&idx| ranks.genes[idx].clone())
                        .collect()
                };

                let n_more_extreme = if obs_es > 0.0 { n_ge_es } else { n_le_es };
                if n_more_extreme < 10 {
                    let ml_start = std::time::Instant::now();
                    // High precision pass for significant pathways
                    // Using CPU for multilevel pass as it is faster for small batches/pathways

                    // Actually we need the real obs_es as GseaScore for CPU multilevel
                    let s_hits = {
                        let mut h = hits.clone();
                        h.sort_unstable();
                        h
                    };
                    let (obs_gsea_score, _) = calculate_gsea_score(
                        &s_hits,
                        &scaled_scores,
                        ns_total,
                        ranks.len(),
                        score_type,
                    );
                    let (m_p, m_err) = run_multilevel_gsea(
                        ranks.len(),
                        &scaled_scores,
                        ns_total,
                        k,
                        obs_gsea_score,
                        score_type,
                        1000,
                        seed + *orig_idx as u64,
                        1e-10,
                    );

                    total_multilevel_time
                        .fetch_add(ml_start.elapsed().as_micros() as u64, Ordering::Relaxed);

                    let p_value_ml = m_p;

                    Some(EnrichmentResult {
                        pathway_name: pathways[*orig_idx].name.clone(),
                        p_value: p_value_ml,
                        padj: None,
                        es: obs_es,
                        nes,
                        log2err: m_err,
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
                        log2err: None,
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

    // Sort and calculate padj
    final_results.sort_by(|a, b| a.p_value.partial_cmp(&b.p_value).unwrap());
    let m = final_results.len();
    for (i, res) in final_results.iter_mut().enumerate() {
        res.padj = Some((res.p_value * m as f64 / (i + 1) as f64).min(1.0));
    }

    // Ensure padj is monotonic
    for i in (0..m - 1).rev() {
        let next_padj = final_results[i + 1].padj.unwrap_or(1.0);
        final_results[i].padj = Some(final_results[i].padj.unwrap_or(1.0).min(next_padj));
    }

    Ok(final_results)
}
