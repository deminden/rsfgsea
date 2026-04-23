use crate::algo_support::{
    apply_bh_adjustment, build_gene_index, compute_nes, extract_pathway_hits, leading_edge,
    mode_fraction_count, multilevel_error, selected_tail_count, should_refine_multilevel,
    simple_log2err, warn_prepare_stats,
};
use crate::core::{EnrichmentResult, Pathway, RankedList, ScoreType};
use crate::fastgsea_compat::{
    calc_gsea_stat_cumulative_batch_f64, calc_gsea_stat_cumulative_batch_f64_parallel,
    calc_gsea_stat_cumulative_batch_f64_thread_invariant_parallel,
};
#[cfg(feature = "gpu")]
use crate::gpu_algo::run_gsea_gpu_with_config_impl;
use crate::multilevel::run_multilevel_gsea_group_impl;
use crate::rng_compat::{RLecuyerCmrgSeedCompat, RMt19937SeedCompat, RSampleKind};
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::time::Instant;

pub fn resolve_rng_seed(seed: Option<u64>) -> u64 {
    seed.unwrap_or_else(rand::random)
}

pub trait IntoSeed {
    fn into_seed(self) -> Option<u64>;
}

impl IntoSeed for u64 {
    fn into_seed(self) -> Option<u64> {
        Some(self)
    }
}

impl IntoSeed for Option<u64> {
    fn into_seed(self) -> Option<u64> {
        self
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

fn derive_fgsea_simple_seed(seed: u64, sample_kind: RSampleKind) -> (u64, RMt19937SeedCompat) {
    // Mirrors first sample.int(1e9, 1) draw in fgseaMultilevel().
    let mut rng = RMt19937SeedCompat::from_r_set_seed(seed as u32);
    let simple_seed = rng.sample_int_one_with_kind(1_000_000_000, sample_kind) as u64;
    (simple_seed, rng)
}

fn fgsea_simple_perm_per_proc(n_perm: usize) -> Vec<usize> {
    if n_perm == 0 {
        return Vec::new();
    }

    // Mirrors fgseaSimple():
    // granularity <- max(1000, ceiling(nperm / 128))
    // permPerProc <- rep(granularity, floor(nperm / granularity))
    // if (nperm - sum(permPerProc) > 0) append remainder
    let granularity = 1000usize.max(n_perm.div_ceil(128));
    let mut perm_per_proc = vec![granularity; n_perm / granularity];
    let used = perm_per_proc.len() * granularity;
    if n_perm > used {
        perm_per_proc.push(n_perm - used);
    }
    perm_per_proc
}

fn derive_fgsea_simple_chunk_plan(
    seed: u64,
    n_perm: usize,
    sample_kind: RSampleKind,
) -> Vec<(usize, u64)> {
    let perm_per_proc = fgsea_simple_perm_per_proc(n_perm);
    let mut rng = RMt19937SeedCompat::from_r_set_seed(seed as u32);
    perm_per_proc
        .into_iter()
        .map(|chunk_perm| {
            let chunk_seed = rng.sample_int_one_with_kind(1_000_000_000, sample_kind) as u64;
            (chunk_perm, chunk_seed)
        })
        .collect()
}

fn timing_enabled() -> bool {
    std::env::var_os("RSFGSEA_STAGE_TIMING").is_some()
}

fn log_stage_timing(enabled: bool, stage: &str, start: Instant) {
    if enabled {
        eprintln!(
            "RSFGSEA_STAGE\t{stage}\t{:.6}",
            start.elapsed().as_secs_f64()
        );
    }
}

fn fast_nonexact_simple_enabled() -> bool {
    std::env::var_os("RSFGSEA_FAST_NONEXACT_SIMPLE").is_some()
}

#[allow(clippy::too_many_arguments)]
pub fn fgsea_multilevel_with_sample_size<S: IntoSeed>(
    ranks: &RankedList,
    pathways: &[Pathway],
    n_perm: usize,
    seed: S,
    min_size: usize,
    max_size: usize,
    eps: f64,
    score_type: ScoreType,
    gsea_param: f64,
    sample_size: usize,
) -> Vec<EnrichmentResult> {
    fgsea_multilevel_with_sample_size_and_kind(
        ranks,
        pathways,
        n_perm,
        seed,
        min_size,
        max_size,
        eps,
        score_type,
        gsea_param,
        sample_size,
        RSampleKind::Rejection,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn fgsea_multilevel_with_sample_size_and_kind<S: IntoSeed>(
    ranks: &RankedList,
    pathways: &[Pathway],
    n_perm: usize,
    seed: S,
    min_size: usize,
    max_size: usize,
    eps: f64,
    score_type: ScoreType,
    gsea_param: f64,
    sample_size: usize,
    sample_kind: RSampleKind,
) -> Vec<EnrichmentResult> {
    let seed = resolve_rng_seed(seed.into_seed());
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
        sample_kind,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn fgsea_simple_with_sample_size<S: IntoSeed>(
    ranks: &RankedList,
    pathways: &[Pathway],
    n_perm: usize,
    seed: S,
    min_size: usize,
    max_size: usize,
    eps: f64,
    score_type: ScoreType,
    gsea_param: f64,
    sample_size: usize,
) -> Vec<EnrichmentResult> {
    fgsea_simple_with_sample_size_and_kind(
        ranks,
        pathways,
        n_perm,
        seed,
        min_size,
        max_size,
        eps,
        score_type,
        gsea_param,
        sample_size,
        RSampleKind::Rejection,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn fgsea_simple_with_sample_size_and_kind<S: IntoSeed>(
    ranks: &RankedList,
    pathways: &[Pathway],
    n_perm: usize,
    seed: S,
    min_size: usize,
    max_size: usize,
    eps: f64,
    score_type: ScoreType,
    gsea_param: f64,
    sample_size: usize,
    sample_kind: RSampleKind,
) -> Vec<EnrichmentResult> {
    let seed = resolve_rng_seed(seed.into_seed());
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
        sample_kind,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn fgsea<S: IntoSeed + Copy>(
    ranks: &RankedList,
    pathways: &[Pathway],
    nperm: Option<usize>,
    n_perm_simple: usize,
    seed: S,
    min_size: usize,
    max_size: usize,
    eps: f64,
    score_type: ScoreType,
    gsea_param: f64,
) -> Vec<EnrichmentResult> {
    fgsea_with_sample_size_and_kind(
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
        RSampleKind::Rejection,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn fgsea_with_sample_size<S: IntoSeed + Copy>(
    ranks: &RankedList,
    pathways: &[Pathway],
    nperm: Option<usize>,
    n_perm_simple: usize,
    seed: S,
    min_size: usize,
    max_size: usize,
    eps: f64,
    score_type: ScoreType,
    gsea_param: f64,
    sample_size: usize,
) -> Vec<EnrichmentResult> {
    fgsea_with_sample_size_and_kind(
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
        sample_size,
        RSampleKind::Rejection,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn fgsea_with_sample_size_and_kind<S: IntoSeed + Copy>(
    ranks: &RankedList,
    pathways: &[Pathway],
    nperm: Option<usize>,
    n_perm_simple: usize,
    seed: S,
    min_size: usize,
    max_size: usize,
    eps: f64,
    score_type: ScoreType,
    gsea_param: f64,
    sample_size: usize,
    sample_kind: RSampleKind,
) -> Vec<EnrichmentResult> {
    if let Some(nperm_simple_mode) = nperm {
        fgsea_simple_with_sample_size_and_kind(
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
            sample_kind,
        )
    } else {
        fgsea_multilevel_with_sample_size_and_kind(
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
            sample_kind,
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
    sample_kind: RSampleKind,
) -> Vec<EnrichmentResult> {
    let timing = timing_enabled();
    let total_start = Instant::now();

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

    let stage_start = Instant::now();
    let gene_to_idx = build_gene_index(ranks);
    let n_total = ranks.len();

    warn_prepare_stats(ranks, score_type);

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
    let mut r_seed_rng = None;
    let mut simple_seed_for_multilevel = None;
    log_stage_timing(timing, "setup", stage_start);

    // Heavy pathway preprocessing is independent per pathway; use parallel map
    // while preserving deterministic input order.
    let stage_start = Instant::now();
    let mut work: Vec<Working> = pathways
        .par_iter()
        .map(|pw| {
            let hits = extract_pathway_hits(pw, &gene_to_idx);
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
    log_stage_timing(timing, "pathway_preprocessing", stage_start);

    if n_perm > 0 && !work.is_empty() {
        let stage_start = Instant::now();
        let simple_chunk_plan = if allow_multilevel {
            let (simple_seed, rng) = derive_fgsea_simple_seed(seed, sample_kind);
            r_seed_rng = Some(rng);
            simple_seed_for_multilevel = Some(simple_seed);
            vec![(n_perm, simple_seed)]
        } else {
            derive_fgsea_simple_chunk_plan(seed, n_perm, sample_kind)
        };

        if work.len() == 1 {
            // Upstream fgseaSimpleImpl special-cases a single pathway and runs
            // `sample.int()` inside `bplapply(..., SerialParam())`, which uses
            // R's L'Ecuyer-CMRG worker RNG rather than boost::mt19937. Each
            // simple-mode chunk uses its own sampled worker seed.
            let w = &mut work[0];
            for (chunk_perm, chunk_seed) in simple_chunk_plan {
                let mut rng = RLecuyerCmrgSeedCompat::from_r_set_seed(chunk_seed as u32);
                for _ in 0..chunk_perm {
                    let mut selected =
                        rng.sample_int_no_replace_with_kind(n_total, w.size, sample_kind);
                    for idx in &mut selected {
                        *idx -= 1;
                    }
                    selected.sort_unstable();
                    let (rand_es, _) =
                        calculate_es_fgsea(&simple_stats, &selected, n_total, score_type);
                    if rand_es <= w.es {
                        w.n_le_es += 1;
                    }
                    if rand_es >= w.es {
                        w.n_ge_es += 1;
                    }
                    if rand_es <= 0.0 {
                        w.n_le_zero += 1;
                        w.le_zero_sum += rand_es;
                    }
                    if rand_es >= 0.0 {
                        w.n_ge_zero += 1;
                        w.ge_zero_sum += rand_es;
                    }
                }
            }
        } else {
            let pathway_scores: Vec<f64> = work.iter().map(|w| w.es).collect();
            let pathways_sizes: Vec<usize> = work.iter().map(|w| w.size).collect();
            for (chunk_perm, chunk_seed) in simple_chunk_plan {
                let use_parallel_batch = rayon::current_num_threads() > 1 && work.len() >= 128;
                let counts = if use_parallel_batch && fast_nonexact_simple_enabled() {
                    // Experimental optimization route for scaling studies. It splits
                    // permutation work across workers, so it does not preserve fgsea/R
                    // RNG parity and must remain opt-in for correctness-sensitive runs.
                    calc_gsea_stat_cumulative_batch_f64_parallel(
                        &simple_stats,
                        1.0,
                        &pathway_scores,
                        &pathways_sizes,
                        chunk_perm,
                        chunk_seed,
                        score_type,
                    )
                } else if use_parallel_batch {
                    calc_gsea_stat_cumulative_batch_f64_thread_invariant_parallel(
                        &simple_stats,
                        1.0,
                        &pathway_scores,
                        &pathways_sizes,
                        chunk_perm,
                        chunk_seed,
                        score_type,
                    )
                } else {
                    calc_gsea_stat_cumulative_batch_f64(
                        &simple_stats,
                        1.0,
                        &pathway_scores,
                        &pathways_sizes,
                        chunk_perm,
                        chunk_seed,
                        score_type,
                    )
                };
                for (i, w) in work.iter_mut().enumerate() {
                    w.n_le_es += counts.le_es[i];
                    w.n_ge_es += counts.ge_es[i];
                    w.n_le_zero += counts.le_zero[i];
                    w.n_ge_zero += counts.ge_zero[i];
                    w.le_zero_sum += counts.le_zero_sum[i];
                    w.ge_zero_sum += counts.ge_zero_sum[i];
                }
            }
        }
        log_stage_timing(timing, "simple_stage", stage_start);
    }

    let stage_start = Instant::now();
    let mut n_more_extreme_vec = vec![0usize; work.len()];
    let mut mode_fraction_vec = vec![0usize; work.len()];

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

        w.nes = compute_nes(w.es, score_type, le_zero_mean, ge_zero_mean);

        if w.nes.is_some() {
            let p_le = (w.n_le_es + 1) as f64 / (w.n_le_zero + 1) as f64;
            let p_ge = (w.n_ge_es + 1) as f64 / (w.n_ge_zero + 1) as f64;
            w.p_value = p_le.min(p_ge);
        }

        let n_more_extreme =
            selected_tail_count(score_type, w.es, w.n_le_es as u64, w.n_ge_es as u64) as usize;
        let mode_fraction =
            mode_fraction_count(score_type, w.es, w.n_le_zero as u64, w.n_ge_zero as u64) as usize;

        n_more_extreme_vec[wi] = n_more_extreme;
        mode_fraction_vec[wi] = mode_fraction;

        w.log2err = if w.p_value.is_finite() {
            simple_log2err(n_more_extreme as u64, n_perm)
        } else {
            None
        };

        if allow_multilevel && mode_fraction < 10 {
            w.p_value = f64::NAN;
            w.nes = None;
            w.log2err = None;
        }
    }
    log_stage_timing(timing, "simple_postprocessing", stage_start);

    if allow_multilevel && n_perm > 0 && !work.is_empty() {
        let stage_start = Instant::now();
        let mut multilevel_groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for i in 0..work.len() {
            if work[i].p_value.is_finite()
                && should_refine_multilevel(
                    n_more_extreme_vec[i] as u64,
                    mode_fraction_vec[i] as u64,
                    n_perm,
                    sample_size,
                    work[i].p_value,
                )
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
            let r_seed_rng = r_seed_rng
                .as_mut()
                .expect("multilevel seed state must be present when allow_multilevel is true");
            r_seed_rng.consume_sample_shuffle_with_kind(multilevel_groups.len(), sample_kind);
            Some(r_seed_rng.sample_int_one_with_kind(1_000_000_000, sample_kind) as u64)
        };

        let multilevel_groups_vec: Vec<Vec<usize>> = multilevel_groups.into_values().collect();
        let group_seed =
            multilevel_seed
                .unwrap_or(simple_seed_for_multilevel.expect(
                    "simple multilevel seed must be present when allow_multilevel is true",
                ));
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
        log_stage_timing(timing, "multilevel_refinement", stage_start);
    }

    let stage_start = Instant::now();
    let mut final_results: Vec<EnrichmentResult> = work
        .into_iter()
        .map(|w| EnrichmentResult {
            pathway_name: w.pathway_name,
            size: w.size,
            es: w.es,
            nes: w.nes,
            p_value: w.p_value,
            padj: w.padj,
            log2err: w.log2err,
            leading_edge: leading_edge(&w.hits, w.peak_idx, w.es, score_type, ranks),
        })
        .collect();

    apply_bh_adjustment(&mut final_results);
    final_results.sort_by(|a, b| a.pathway_name.cmp(&b.pathway_name));
    log_stage_timing(timing, "finalize_results", stage_start);
    log_stage_timing(timing, "total", total_start);
    final_results
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
    run_multilevel_gsea_group_impl(
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

#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
pub fn run_gsea_gpu<S: IntoSeed>(
    ranks: &RankedList,
    pathways: &[Pathway],
    n_perm: usize,
    seed: S,
    min_size: usize,
    max_size: usize,
    score_type: ScoreType,
    gsea_param: f64,
) -> Result<Vec<EnrichmentResult>, anyhow::Error> {
    run_gsea_gpu_with_config(
        ranks, pathways, n_perm, seed, min_size, max_size, 1e-50, score_type, gsea_param, 101, true,
    )
}

#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
pub fn run_gsea_gpu_with_config<S: IntoSeed>(
    ranks: &RankedList,
    pathways: &[Pathway],
    n_perm: usize,
    seed: S,
    min_size: usize,
    max_size: usize,
    eps: f64,
    score_type: ScoreType,
    gsea_param: f64,
    sample_size: usize,
    allow_multilevel: bool,
) -> Result<Vec<EnrichmentResult>, anyhow::Error> {
    let seed = resolve_rng_seed(seed.into_seed());
    run_gsea_gpu_with_config_impl(
        ranks,
        pathways,
        n_perm,
        seed,
        min_size,
        max_size,
        eps,
        score_type,
        gsea_param,
        sample_size,
        allow_multilevel,
    )
}
