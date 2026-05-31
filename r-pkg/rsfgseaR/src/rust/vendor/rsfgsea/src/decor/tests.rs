use super::*;
use crate::algo::calculate_es_fgsea;
use tempfile::tempdir;

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-8,
        "actual={actual}, expected={expected}"
    );
}

fn formula_cache(values: &[f32]) -> DecorCache {
    DecorCache {
        metadata: DecorCacheMetadata {
            format: CACHE_FORMAT.to_string(),
            version: CACHE_VERSION.to_string(),
            created_by: "rsfgsea".to_string(),
            gmt_sha256: "gmt".to_string(),
            expression_sha256: "expr".to_string(),
            correlation: DecorCorrelation::Pearson,
            redundancy: DecorRedundancy::PositiveMean,
            expression_gene_axis: "rows".to_string(),
            expression_has_header: true,
            gene_id_mode: GENE_ID_MODE.to_string(),
            n_pathways: 1,
            n_rows: values.len(),
        },
        pathways: BTreeMap::from([(
            "PW".to_string(),
            DecorPathwayScores {
                genes: values
                    .iter()
                    .enumerate()
                    .map(|(idx, _)| format!("G{idx}"))
                    .collect(),
                redundancy: values.to_vec(),
            },
        )]),
    }
}

fn formula_context(formula: DecorWeightFormula) -> DecorFormulaContext {
    let cache = formula_cache(&[0.0, 0.2, 0.4, 0.8]);
    let options = DecorOptions {
        alpha: 2.0,
        weight_formula: formula,
        ..DecorOptions::default()
    };
    DecorFormulaContext::from_cache(&cache, &options).unwrap()
}

#[test]
fn decor_formula_raw_rational_matches_manual_values() {
    let ctx = formula_context(DecorWeightFormula::RawRational);
    assert_close(ctx.penalty(0.0).unwrap(), 1.0);
    assert_close(ctx.penalty(0.5).unwrap(), 0.5);

    let cache = formula_cache(&[0.5]);
    let options = DecorOptions {
        alpha: 0.0,
        weight_formula: DecorWeightFormula::RawRational,
        ..DecorOptions::default()
    };
    let ctx = DecorFormulaContext::from_cache(&cache, &options).unwrap();
    assert_close(ctx.penalty(0.9).unwrap(), 1.0);
}

#[test]
fn decor_formula_scaled_rational_uses_global_median() {
    let ctx = formula_context(DecorWeightFormula::ScaledRational);
    assert_close(ctx.global_median_redundancy.unwrap(), 0.3);
    let expected = 1.0 / (1.0 + 2.0 * (0.6 / (0.3 + 1e-12)));
    assert_close(ctx.penalty(0.6).unwrap(), expected);
}

#[test]
fn decor_formula_q75_scaled_rational_uses_global_q75() {
    let ctx = formula_context(DecorWeightFormula::Q75ScaledRational);
    assert_close(ctx.global_q75_redundancy.unwrap(), 0.5);
    let expected = 1.0 / (1.0 + 2.0 * (0.5 / (0.5 + 1e-12)));
    assert_close(ctx.penalty(0.5).unwrap(), expected);
}

#[test]
fn decor_formula_exp_scaled_matches_manual_value() {
    let ctx = formula_context(DecorWeightFormula::ExpScaled);
    let expected = (-2.0_f64 * (0.3_f64 / (0.3_f64 + 1e-12_f64))).exp();
    assert_close(ctx.penalty(0.3).unwrap(), expected);
}

#[test]
fn decor_formula_odds_rational_emphasizes_high_redundancy() {
    let ctx = formula_context(DecorWeightFormula::OddsRational);
    assert_close(ctx.penalty(0.0).unwrap(), 1.0);
    assert_close(
        ctx.penalty(0.5).unwrap(),
        1.0 / (1.0 + 2.0 * (0.5 / (0.5 + 1e-12))),
    );
    assert!(ctx.penalty(1.0).unwrap() < 1e-6);
}

#[test]
fn decor_formula_threshold_rational_ignores_scores_below_tau() {
    let cache = formula_cache(&[0.1, 0.2, 0.5]);
    let options = DecorOptions {
        alpha: 2.0,
        weight_formula: DecorWeightFormula::ThresholdRational,
        threshold_tau: 0.25,
        ..DecorOptions::default()
    };
    let ctx = DecorFormulaContext::from_cache(&cache, &options).unwrap();
    assert_close(ctx.penalty(0.2).unwrap(), 1.0);
    assert_close(ctx.penalty(0.5).unwrap(), 1.0 / (1.0 + 2.0 * 0.25));
}

#[test]
fn decor_formula_quantile_rational_uses_average_rank_for_ties() {
    let cache = formula_cache(&[0.1, 0.2, 0.2, 0.8]);
    let options = DecorOptions {
        alpha: 2.0,
        weight_formula: DecorWeightFormula::QuantileRational,
        ..DecorOptions::default()
    };
    let ctx = DecorFormulaContext::from_cache(&cache, &options).unwrap();
    let q = 1.5 / 3.0;
    assert_close(
        ctx.penalty(f64::from(0.2_f32)).unwrap(),
        1.0 / (1.0 + 2.0 * q),
    );
}

#[test]
fn decor_formula_floor_scaled_rational_respects_floor() {
    let cache = formula_cache(&[0.0, 0.2, 0.4, 0.8]);
    let options = DecorOptions {
        alpha: 1000.0,
        weight_formula: DecorWeightFormula::FloorScaledRational,
        penalty_floor: 0.25,
        ..DecorOptions::default()
    };
    let ctx = DecorFormulaContext::from_cache(&cache, &options).unwrap();
    assert!(ctx.penalty(0.8).unwrap() >= 0.25);
    assert_close(ctx.penalty(0.0).unwrap(), 1.0);
}

#[test]
fn decor_formula_power_retention_uses_gamma() {
    let cache = formula_cache(&[0.0, 0.5, 1.0]);
    let options = DecorOptions {
        alpha: 999.0,
        gamma: 1.0,
        weight_formula: DecorWeightFormula::PowerRetention,
        ..DecorOptions::default()
    };
    let ctx = DecorFormulaContext::from_cache(&cache, &options).unwrap();
    assert_close(ctx.penalty(0.5).unwrap(), 0.5);

    let options = DecorOptions {
        gamma: 0.0,
        weight_formula: DecorWeightFormula::PowerRetention,
        ..DecorOptions::default()
    };
    let ctx = DecorFormulaContext::from_cache(&cache, &options).unwrap();
    assert_close(ctx.penalty(0.9).unwrap(), 1.0);
}

#[test]
fn decor_formula_validation_rejects_invalid_parameters() {
    for options in [
        DecorOptions {
            gamma: -1.0,
            ..DecorOptions::default()
        },
        DecorOptions {
            threshold_tau: 1.0,
            ..DecorOptions::default()
        },
        DecorOptions {
            penalty_floor: 1.0,
            ..DecorOptions::default()
        },
        DecorOptions {
            scale_epsilon: 0.0,
            ..DecorOptions::default()
        },
    ] {
        assert!(DecorFormulaContext::from_cache(&formula_cache(&[0.1]), &options).is_err());
    }
}

#[test]
fn decor_distribution_formulas_error_without_finite_cache_scores() {
    let cache = formula_cache(&[]);
    let options = DecorOptions {
        weight_formula: DecorWeightFormula::ScaledRational,
        ..DecorOptions::default()
    };
    let err = DecorFormulaContext::from_cache(&cache, &options)
        .unwrap_err()
        .to_string();
    assert!(err.contains("no finite redundancy scores"));
}

#[test]
fn decor_es_matches_classic_when_redundancy_zero() {
    let stats = vec![4.0, 3.0, 2.0, -1.0, -2.0, -3.0];
    let hits = vec![0, 2, 5];
    let penalty = vec![1.0, 1.0, 1.0];
    let decor = calculate_es_decor(&stats, &hits, &penalty, stats.len(), ScoreType::Std)
        .unwrap()
        .0;
    let classic = calculate_es_fgsea(&stats, &hits, stats.len(), ScoreType::Std).0;
    assert!((decor - classic).abs() < 1e-12);
}

#[test]
fn decor_alpha_zero_matches_classic() {
    let stats = vec![4.0, 3.0, 2.0, -1.0, -2.0, -3.0];
    let hits = vec![0, 2, 5];
    let cache = formula_cache(&[0.9, 0.1, 0.4]);
    let options = DecorOptions {
        alpha: 0.0,
        weight_formula: DecorWeightFormula::RawRational,
        ..DecorOptions::default()
    };
    let ctx = DecorFormulaContext::from_cache(&cache, &options).unwrap();
    let penalty = ctx.penalties_for(&[0.9, 0.1, 0.4]).unwrap();
    let decor = calculate_es_decor(&stats, &hits, &penalty, stats.len(), ScoreType::Std)
        .unwrap()
        .0;
    let classic = calculate_es_fgsea(&stats, &hits, stats.len(), ScoreType::Std).0;
    assert!((decor - classic).abs() < 1e-12);
}

#[test]
fn decor_downweights_high_redundancy_gene() {
    let stats = vec![5.0, 4.0, 1.0, -1.0, -2.0, -3.0];
    let hits = vec![0, 1, 5];
    let no_penalty = vec![1.0, 1.0, 1.0];
    let penalized = vec![1.0 / 2.0, 1.0, 1.0];
    let classic = calculate_es_decor(&stats, &hits, &no_penalty, stats.len(), ScoreType::Std)
        .unwrap()
        .0;
    let decor = calculate_es_decor(&stats, &hits, &penalized, stats.len(), ScoreType::Std)
        .unwrap()
        .0;
    assert_ne!(decor, classic);
    assert!(decor.is_finite());
}

#[test]
fn decor_uniform_penalty_cancels_like_classic_weight_normalization() {
    let stats = vec![5.0, 4.0, 1.0, -1.0, -2.0, -3.0];
    let hits = vec![0, 1, 5];
    let uniform_penalty = vec![0.25, 0.25, 0.25];
    let decor = calculate_es_decor(&stats, &hits, &uniform_penalty, stats.len(), ScoreType::Std)
        .unwrap()
        .0;
    let classic = calculate_es_fgsea(&stats, &hits, stats.len(), ScoreType::Std).0;
    assert_close(decor, classic);
}

#[test]
fn decor_all_zero_adjusted_weights_fall_back_to_uniform_hit_weights() {
    let stats = vec![5.0, 4.0, 1.0, -1.0, -2.0, -3.0];
    let hits = vec![0, 1, 5];
    let zero_penalty = vec![0.0, 0.0, 0.0];
    let decor = calculate_es_decor(&stats, &hits, &zero_penalty, stats.len(), ScoreType::Std)
        .unwrap()
        .0;
    let uniform_stats = vec![1.0; stats.len()];
    let uniform = calculate_es_fgsea(&uniform_stats, &hits, stats.len(), ScoreType::Std).0;
    assert_close(decor, uniform);
}

#[test]
fn invalid_redundancy_length_errors_cleanly() {
    let err = calculate_es_decor(&[1.0, 2.0], &[0, 1], &[1.0], 2, ScoreType::Std)
        .unwrap_err()
        .to_string();
    assert!(err.contains("length mismatch"));
}

fn scalar_counts_for_selected(
    stats: &[f64],
    selected: &[usize],
    penalties: &[Vec<f64>],
    observed_es: &[f64],
    score_type: ScoreType,
) -> Vec<DecorNullCounts> {
    let mut expected = vec![DecorNullCounts::default(); penalties.len()];
    for (i, penalty) in penalties.iter().enumerate() {
        let (rand_es, _) =
            calculate_es_decor_prechecked(stats, selected, penalty, stats.len(), score_type);
        update_decor_null_count(&mut expected[i], rand_es, observed_es[i]);
    }
    expected
}

fn rank_major_penalties(penalties: &[Vec<f64>]) -> Vec<f64> {
    let size = penalties[0].len();
    let mut out = Vec::with_capacity(size * penalties.len());
    for rank_idx in 0..size {
        for penalty in penalties {
            out.push(penalty[rank_idx]);
        }
    }
    out
}

#[test]
fn decor_batched_counts_match_scalar_es_updates_for_all_score_types() {
    let stats = vec![4.0, 3.0, 2.0, -1.0, -2.0, -3.0];
    let stats_abs = stats.iter().map(|v: &f64| v.abs()).collect::<Vec<_>>();
    let selected = vec![0, 2, 5];
    let penalties = [vec![1.0, 0.5, 1.0], vec![0.2, 1.0, 0.8]];
    let observed_es = vec![0.4, -0.1];
    let penalties_rank_major = rank_major_penalties(&penalties);

    for score_type in [ScoreType::Std, ScoreType::Pos, ScoreType::Neg] {
        let mut counts = vec![DecorNullCounts::default(); 2];
        let mut scratch = DecorBatchScratch::default();
        update_decor_batched_counts(
            &stats_abs,
            &selected,
            &penalties_rank_major,
            &observed_es,
            stats.len(),
            score_type,
            &mut counts,
            &mut scratch,
        );
        let expected =
            scalar_counts_for_selected(&stats, &selected, &penalties, &observed_es, score_type);
        assert_eq!(counts, expected);
    }
}

#[test]
fn decor_runtime_size_group_forces_batched_layout_at_threshold() {
    let stats = vec![4.0, 3.0, 2.0, 1.0, -1.0, -2.0];
    let hits = vec![0, 2, 5];
    let work = (0..DECOR_BATCH_MIN_GROUP_PATHWAYS)
        .map(|i| {
            let penalty = vec![1.0 / (1.0 + i as f64 * 0.01), 0.8, 0.6 + i as f64 * 0.001];
            let (es, peak_idx) =
                calculate_es_decor_prechecked(&stats, &hits, &penalty, stats.len(), ScoreType::Std);
            DecorWorking {
                pathway_name: format!("PW_{i}"),
                size: hits.len(),
                hits: hits.clone(),
                penalty,
                es,
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
            }
        })
        .collect::<Vec<_>>();

    let indices = (0..work.len()).collect::<Vec<_>>();
    let group = DecorRuntimeSizeGroup::from_indices(hits.len(), indices, &work);
    assert!(group.use_batched);
    assert_eq!(
        group.penalties_rank_major.len(),
        hits.len() * DECOR_BATCH_MIN_GROUP_PATHWAYS
    );
    for rank_idx in 0..hits.len() {
        for (work_idx, w) in work.iter().enumerate().take(DECOR_BATCH_MIN_GROUP_PATHWAYS) {
            assert_eq!(
                group.penalties_rank_major[rank_idx * DECOR_BATCH_MIN_GROUP_PATHWAYS + work_idx],
                w.penalty[rank_idx]
            );
        }
    }
}

#[test]
fn decor_grouped_null_matches_scalar_reference_for_mixed_sizes() {
    let stats = vec![5.0, 4.0, 3.0, 2.0, -1.0, -2.0, -3.0];
    let stats_abs = stats.iter().map(|v: &f64| v.abs()).collect::<Vec<_>>();
    let selected_by_size = BTreeMap::from([
        (2usize, vec![0usize, 5usize]),
        (3usize, vec![0usize, 3usize, 6usize]),
    ]);
    let mut work = vec![
        test_working("PW2_A", &stats, vec![0, 2], vec![1.0, 0.7]),
        test_working("PW3_A", &stats, vec![0, 3, 6], vec![1.0, 0.8, 0.6]),
        test_working("PW2_B", &stats, vec![1, 5], vec![0.5, 1.0]),
        test_working("PW3_B", &stats, vec![2, 4, 6], vec![0.9, 0.4, 1.0]),
    ];

    let mut size_groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (idx, w) in work.iter().enumerate() {
        size_groups.entry(w.size).or_default().push(idx);
    }
    let runtime_groups = size_groups
        .into_iter()
        .map(|(size, indices)| DecorRuntimeSizeGroup::from_indices(size, indices, &work))
        .collect::<Vec<_>>();
    for group in &runtime_groups {
        let selected = selected_by_size.get(&group.size).unwrap();
        let mut counts = vec![DecorNullCounts::default(); group.work_indices.len()];
        if group.use_batched {
            let mut scratch = DecorBatchScratch::default();
            update_decor_batched_counts(
                &stats_abs,
                selected,
                &group.penalties_rank_major,
                &group.observed_es,
                stats.len(),
                ScoreType::Std,
                &mut counts,
                &mut scratch,
            );
        } else {
            for (count, work_idx) in counts.iter_mut().zip(group.work_indices.iter()) {
                let w = &work[*work_idx];
                let (rand_es, _) = calculate_es_decor_prechecked(
                    &stats,
                    selected,
                    &w.penalty,
                    stats.len(),
                    ScoreType::Std,
                );
                update_decor_null_count(count, rand_es, w.es);
            }
        }

        for (&work_idx, count) in group.work_indices.iter().zip(counts) {
            let w = &mut work[work_idx];
            w.n_le_es += count.n_le_es;
            w.n_ge_es += count.n_ge_es;
            w.n_le_zero += count.n_le_zero;
            w.n_ge_zero += count.n_ge_zero;
            w.le_zero_sum += count.le_zero_sum;
            w.ge_zero_sum += count.ge_zero_sum;
        }
    }

    for w in &work {
        let selected = selected_by_size.get(&w.size).unwrap();
        let expected = scalar_counts_for_selected(
            &stats,
            selected,
            std::slice::from_ref(&w.penalty),
            &[w.es],
            ScoreType::Std,
        );
        assert_eq!(
            DecorNullCounts {
                n_le_es: w.n_le_es,
                n_ge_es: w.n_ge_es,
                n_le_zero: w.n_le_zero,
                n_ge_zero: w.n_ge_zero,
                le_zero_sum: w.le_zero_sum,
                ge_zero_sum: w.ge_zero_sum,
            },
            expected[0],
            "{}",
            w.pathway_name
        );
    }
}

fn test_working(
    pathway_name: &str,
    stats: &[f64],
    hits: Vec<usize>,
    penalty: Vec<f64>,
) -> DecorWorking {
    let (es, peak_idx) =
        calculate_es_decor_prechecked(stats, &hits, &penalty, stats.len(), ScoreType::Std);
    DecorWorking {
        pathway_name: pathway_name.to_string(),
        size: hits.len(),
        hits,
        penalty,
        es,
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
    }
}

#[test]
fn cache_build_write_read_validate_round_trip() {
    let dir = tempdir().unwrap();
    let expr = dir.path().join("expression.tsv");
    fs::write(
            &expr,
            "gene\ts1\ts2\ts3\ts4\nA\t1\t2\t3\t4\nB\t1.1\t2.1\t3.1\t4.1\nC\t4\t3\t2\t1\nE\t1\t1\t1\t1\n",
        )
        .unwrap();
    let gmt = dir.path().join("test.gmt");
    fs::write(&gmt, "PW\tdesc\tA\tB\tC\tE\n").unwrap();
    let expected = DecorCacheExpectedMetadata {
        gmt_sha256: file_sha256(&gmt).unwrap(),
        expression_sha256: Some(file_sha256(&expr).unwrap()),
        correlation: DecorCorrelation::Pearson,
        redundancy: DecorRedundancy::PositiveMean,
        expression_gene_axis: "rows".to_string(),
        expression_has_header: true,
        gene_id_mode: GENE_ID_MODE.to_string(),
    };
    let pathways = vec![Pathway {
        name: "PW".to_string(),
        description: None,
        genes: vec!["A".into(), "B".into(), "C".into(), "E".into()],
    }];
    let cache = build_decor_cache_from_expression(&pathways, &expr, expected.clone()).unwrap();
    let path = dir.path().join("cache.decor.tsv");
    write_decor_cache_atomic(&path, &cache).unwrap();
    let loaded = read_decor_cache(&path).unwrap();
    assert!(validate_decor_cache(&loaded.metadata, &expected).is_compatible());
    let pw = loaded.pathways.get("PW").unwrap();
    assert_eq!(pw.genes, vec!["A", "B", "C", "E"]);
    assert!(pw.redundancy[0] > 0.3);
}

#[test]
fn formula_and_alpha_do_not_affect_cache_compatibility() {
    let cache = formula_cache(&[0.1, 0.2]);
    let expected = DecorCacheExpectedMetadata {
        gmt_sha256: "gmt".to_string(),
        expression_sha256: Some("expr".to_string()),
        correlation: DecorCorrelation::Pearson,
        redundancy: DecorRedundancy::PositiveMean,
        expression_gene_axis: "rows".to_string(),
        expression_has_header: true,
        gene_id_mode: GENE_ID_MODE.to_string(),
    };
    assert!(validate_decor_cache(&cache.metadata, &expected).is_compatible());

    let options = DecorOptions {
        alpha: 0.0,
        weight_formula: DecorWeightFormula::ExpScaled,
        ..DecorOptions::default()
    };
    let ctx = DecorFormulaContext::from_cache(&cache, &options).unwrap();
    assert_eq!(ctx.weight_formula, DecorWeightFormula::ExpScaled);
    assert!(validate_decor_cache(&cache.metadata, &expected).is_compatible());
}
