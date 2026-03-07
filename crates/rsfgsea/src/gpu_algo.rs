use crate::algo::calculate_es_fgsea;
use crate::algo_support::{
    apply_bh_adjustment, build_gene_index, compute_nes, extract_pathway_hits, leading_edge,
    mode_fraction_count, multilevel_error, selected_tail_count, should_refine_multilevel,
};
use crate::core::{EnrichmentResult, Pathway, RankedList, ScoreType};
use crate::multilevel::run_multilevel_gsea_impl;
use anyhow::Result;
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_gsea_gpu_with_config_impl(
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
    allow_multilevel: bool,
) -> Result<Vec<EnrichmentResult>> {
    use crate::gpu::GpuEngine;

    let sample_size = sample_size.max(1);
    let eps = eps.clamp(0.0, 1.0);

    let (abs_weights, scaled_scores, _ns_total) = ranks.prepare(gsea_param);
    let simple_stats: Vec<f64> = scaled_scores.iter().map(|&score| score as f64).collect();
    let abs_weights_f32: Vec<f32> = abs_weights.iter().map(|&w| w as f32).collect();
    let gene_to_idx = build_gene_index(ranks);

    let mut by_size: BTreeMap<usize, Vec<(usize, Vec<usize>)>> = BTreeMap::new();
    for (i, pathway) in pathways.iter().enumerate() {
        let hits = extract_pathway_hits(pathway, &gene_to_idx);
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
            log::info!(
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
        total_null_gen_time.fetch_add(gen_start.elapsed().as_micros() as u64, Ordering::Relaxed);
        null_distribution.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

        let (_n_le_all, _n_ge_all, n_le_zero, n_ge_zero, le_zero_sum, ge_zero_sum) =
            GpuEngine::calculate_null_stats(&null_distribution, 0.0);
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

        let group_results: Vec<Option<EnrichmentResult>> = group
            .par_iter()
            .map(|(orig_idx, hits)| {
                let screening_start = std::time::Instant::now();
                let mut sorted_hits = hits.clone();
                sorted_hits.sort_unstable();
                let (obs_es, peak_idx) =
                    calculate_es_fgsea(&simple_stats, &sorted_hits, ranks.len(), score_type);

                let (n_le_es, n_ge_es, _, _, _, _) =
                    GpuEngine::calculate_null_stats(&null_distribution, obs_es);

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

                let nes = compute_nes(obs_es, score_type, le_zero_mean, ge_zero_mean);
                let leading_edge = leading_edge(&sorted_hits, peak_idx, obs_es, score_type, ranks);
                let n_more_extreme = selected_tail_count(score_type, obs_es, n_le_es, n_ge_es);
                let mode_fraction = mode_fraction_count(score_type, obs_es, n_le_zero, n_ge_zero);
                let simple_log2err = crate::algo_support::simple_log2err(n_more_extreme, n_perm);

                if allow_multilevel
                    && should_refine_multilevel(
                        n_more_extreme,
                        mode_fraction,
                        n_perm,
                        sample_size,
                        p_value_simple,
                    )
                {
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
                        size: k,
                        es: obs_es,
                        nes,
                        p_value: p_value_ml,
                        padj: None,
                        log2err,
                        leading_edge,
                    })
                } else {
                    Some(EnrichmentResult {
                        pathway_name: pathways[*orig_idx].name.clone(),
                        size: k,
                        es: obs_es,
                        nes,
                        p_value: p_value_simple,
                        padj: None,
                        log2err: simple_log2err,
                        leading_edge,
                    })
                }
            })
            .collect();

        for (i, res) in group_results.into_iter().enumerate() {
            if let Some(result) = res {
                results[group[i].0] = Some(result);
            }
        }
    }

    log::info!("GPU execution timings:");
    log::info!(
        "null distribution generation: {} ms",
        total_null_gen_time.load(Ordering::Relaxed) / 1000
    );
    log::info!(
        "pure screening pass: {} ms",
        total_pure_screening_time.load(Ordering::Relaxed) / 1000
    );
    log::info!(
        "multilevel pass: {} ms",
        total_multilevel_time.load(Ordering::Relaxed) / 1000
    );

    let mut final_results: Vec<EnrichmentResult> = results.into_iter().flatten().collect();
    apply_bh_adjustment(&mut final_results);
    final_results.sort_by(|a, b| a.pathway_name.cmp(&b.pathway_name));
    Ok(final_results)
}
