use crate::core::{EnrichmentResult, Pathway, RankedList, ScoreType};
use special::Gamma;
use statrs::distribution::{Beta, ContinuousCDF};
use std::collections::HashMap;

pub(crate) fn build_gene_index(ranks: &RankedList) -> HashMap<String, usize> {
    ranks
        .genes
        .iter()
        .enumerate()
        .map(|(i, gene)| (gene.clone(), i))
        .collect()
}

pub(crate) fn extract_pathway_hits(
    pathway: &Pathway,
    gene_to_idx: &HashMap<String, usize>,
) -> Vec<usize> {
    let mut hits: Vec<usize> = pathway
        .genes
        .iter()
        .filter_map(|gene| gene_to_idx.get(gene).copied())
        .collect();
    hits.sort_unstable();
    hits.dedup();
    hits
}

pub(crate) fn warn_prepare_stats(ranks: &RankedList, score_type: ScoreType) {
    let n_total = ranks.len();
    let mut seen_nonzero = std::collections::HashSet::new();
    let mut tie_count = 0usize;
    for &score in &ranks.scores {
        if score != 0.0 && !seen_nonzero.insert(score.to_bits()) {
            tie_count += 1;
        }
    }

    if tie_count > 0 && n_total > 0 {
        let tie_pct = (tie_count as f64) * 100.0 / (n_total as f64);
        log::warn!(
            "detected {} tied non-zero ranking scores ({:.2}% of genes); ties are resolved by input order and can slightly affect enrichment outcomes",
            tie_count,
            tie_pct
        );
    }

    if matches!(score_type, ScoreType::Std) && ranks.scores.iter().all(|&score| score > 0.0) {
        log::warn!(
            "all ranking scores are positive while scoreType='std'; for one-tailed enrichment, consider scoreType='pos'"
        );
    }
}

pub(crate) fn compute_nes(
    es: f64,
    score_type: ScoreType,
    le_zero_mean: f64,
    ge_zero_mean: f64,
) -> Option<f64> {
    match score_type {
        ScoreType::Std => {
            if es > 0.0 && ge_zero_mean != 0.0 {
                Some(es / ge_zero_mean)
            } else if es <= 0.0 && le_zero_mean != 0.0 {
                Some(es / le_zero_mean.abs())
            } else {
                None
            }
        }
        ScoreType::Pos => {
            if es >= 0.0 && ge_zero_mean != 0.0 {
                Some(es / ge_zero_mean)
            } else {
                None
            }
        }
        ScoreType::Neg => {
            if es <= 0.0 && le_zero_mean != 0.0 {
                Some(es / le_zero_mean.abs())
            } else {
                None
            }
        }
    }
}

pub(crate) fn selected_tail_count(
    score_type: ScoreType,
    es: f64,
    n_le_es: u64,
    n_ge_es: u64,
) -> u64 {
    match score_type {
        ScoreType::Std => {
            if es > 0.0 {
                n_ge_es
            } else {
                n_le_es
            }
        }
        ScoreType::Pos => n_ge_es,
        ScoreType::Neg => n_le_es,
    }
}

pub(crate) fn mode_fraction_count(
    score_type: ScoreType,
    es: f64,
    n_le_zero: u64,
    n_ge_zero: u64,
) -> u64 {
    match score_type {
        ScoreType::Std => {
            if es >= 0.0 {
                n_ge_zero
            } else {
                n_le_zero
            }
        }
        ScoreType::Pos => n_ge_zero,
        ScoreType::Neg => n_le_zero,
    }
}

pub(crate) fn simple_log2err(n_more_extreme: u64, n_perm: usize) -> Option<f64> {
    if n_perm == 0 {
        return None;
    }
    Some(
        1.0 / 2.0_f64.ln()
            * (((n_more_extreme + 1) as f64).trigamma() - ((n_perm + 1) as f64).trigamma()).sqrt(),
    )
}

pub(crate) fn multilevel_error(pval: f64, sample_size: usize) -> f64 {
    if !(pval.is_finite()) || pval <= 0.0 {
        return f64::NAN;
    }
    let level = (-pval.log2() + 1.0).floor();
    (level
        * (((sample_size as f64 + 1.0) / 2.0).trigamma() - (sample_size as f64 + 1.0).trigamma()))
    .sqrt()
        / 2.0_f64.ln()
}

pub(crate) fn log2_qbeta(prob: f64, shape1: f64, shape2: f64) -> f64 {
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

pub(crate) fn should_refine_multilevel(
    n_more_extreme: u64,
    mode_fraction: u64,
    n_perm: usize,
    sample_size: usize,
    p_value_simple: f64,
) -> bool {
    if mode_fraction < 10 || !p_value_simple.is_finite() || n_perm == 0 {
        return false;
    }

    let n_more = n_more_extreme as f64;
    let n_perm_f = n_perm as f64;
    let left = log2_qbeta(0.025, n_more, n_perm_f - n_more + 1.0);
    let right = log2_qbeta(0.975, n_more + 1.0, n_perm_f - n_more);
    let crude = ((n_more + 1.0) / (n_perm_f + 1.0)).log2();
    let simple_error = 0.5 * (crude - left).max(right - crude);
    let mult_error = multilevel_error((n_more + 1.0) / (n_perm_f + 1.0), sample_size);
    mult_error < simple_error
}

pub(crate) fn leading_edge(
    hits: &[usize],
    peak_idx: usize,
    es: f64,
    score_type: ScoreType,
    ranks: &RankedList,
) -> Vec<String> {
    match score_type {
        ScoreType::Pos => hits
            .iter()
            .filter(|&&idx| idx <= peak_idx)
            .map(|&idx| ranks.genes[idx].clone())
            .collect(),
        ScoreType::Neg => {
            let mut le: Vec<String> = hits
                .iter()
                .filter(|&&idx| idx >= peak_idx)
                .map(|&idx| ranks.genes[idx].clone())
                .collect();
            le.reverse();
            le
        }
        ScoreType::Std => {
            if es > 0.0 {
                hits.iter()
                    .filter(|&&idx| idx <= peak_idx)
                    .map(|&idx| ranks.genes[idx].clone())
                    .collect()
            } else if es < 0.0 {
                let mut le: Vec<String> = hits
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
    }
}

pub(crate) fn apply_bh_adjustment(results: &mut [EnrichmentResult]) {
    if results.is_empty() {
        return;
    }

    let mut indices: Vec<usize> = (0..results.len())
        .filter(|&i| results[i].p_value.is_finite())
        .collect();
    indices.sort_by(|&a, &b| results[a].p_value.partial_cmp(&results[b].p_value).unwrap());

    let m = results.len() as f64;
    let mut prev_padj = 1.0;
    for i in (0..indices.len()).rev() {
        let idx = indices[i];
        let p = results[idx].p_value;
        let padj = (p * m / (i + 1) as f64).min(prev_padj).min(1.0);
        results[idx].padj = Some(padj);
        prev_padj = padj;
    }
}
