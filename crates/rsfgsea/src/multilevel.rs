use crate::core::ScoreType;
use crate::esruler_compat::EsRulerCompat;

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_multilevel_gsea_impl(
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
pub(crate) fn run_multilevel_gsea_group_impl(
    n_total: usize,
    scaled_scores: &[i64],
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

    if obs_es >= 0.0 {
        es_pos.extend(obs_es.abs(), seed, eps);
    } else {
        es_neg.extend(obs_es.abs(), seed, eps);
    }

    let sign = matches!(score_type, ScoreType::Pos | ScoreType::Neg);
    let (p, is_cp_ge_half, err) = if obs_es >= 0.0 {
        es_pos.get_pvalue(obs_es.abs(), eps, sign)
    } else {
        es_neg.get_pvalue(obs_es.abs(), eps, sign)
    };

    if err.is_finite() {
        (p, is_cp_ge_half, Some(err))
    } else {
        (p, is_cp_ge_half, None)
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
            let (p, is_cp_ge_half, err) = if cur_es >= 0.0 {
                es_pos.get_pvalue(cur_es.abs(), eps, sign)
            } else {
                es_neg.get_pvalue(cur_es.abs(), eps, sign)
            };
            if err.is_finite() {
                (p, is_cp_ge_half, Some(err))
            } else {
                (p, is_cp_ge_half, None)
            }
        })
        .collect()
}
