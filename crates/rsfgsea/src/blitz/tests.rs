use super::*;
use crate::io::{read_gmt, read_ranked_list};
use serde::Deserialize;

#[test]
fn numpy_mt19937_permutation_matches_reference() {
    let cases = [
        (0, vec![2, 8, 4, 9, 1]),
        (1, vec![2, 9, 6, 4, 0]),
        (42, vec![8, 1, 5, 0, 7]),
        (20_260_322, vec![3, 7, 9, 0, 4]),
    ];
    for (seed, expected) in cases {
        let mut rng = NumpyMt19937::new(seed);
        assert_eq!(&rng.choice_without_replacement(10, 5), &expected);
    }
}

#[test]
fn numpy_mt19937_standard_normals_match_reference() {
    let mut rng = NumpyMt19937::new(0);
    let observed = rng.standard_normals(10);
    let expected = [
        1.764052345967664,
        0.4001572083672233,
        0.9787379841057392,
        2.240893199201458,
        1.8675579901499675,
        -0.977277879876411,
        0.9500884175255894,
        -0.1513572082976979,
        -0.10321885179355784,
        0.41059850193837233,
    ];
    for (observed, expected) in observed.iter().zip(expected) {
        assert!((observed - expected).abs() < 1e-15);
    }
}

#[test]
fn python_int_set_iteration_matches_reference_cases() {
    assert_eq!(python_int_set_iteration_order(&[2, 9]), vec![9, 2]);
    assert_eq!(
        python_int_set_iteration_order(&(105..120).collect::<Vec<_>>()),
        (105..120).collect::<Vec<_>>()
    );
    assert_eq!(
        python_int_set_iteration_order(&[0, 16, 32, 48, 64]),
        vec![0, 32, 64, 16, 48]
    );
}

#[test]
fn python_string_hash_and_set_order_match_seed0_reference() {
    assert_eq!(python_ascii_hash_seed0("g001") as i64, 3270876322613014562);
    assert_eq!(python_ascii_hash_seed0("g004") as i64, -8038800456378607197);
    let set = PythonStringSet::from_iter(
        [
            "g001", "g004", "g009", "g008", "g007", "g003", "g012", "g010", "g002", "g011", "g006",
            "g005",
        ]
        .into_iter()
        .map(str::to_string),
    );
    assert_eq!(
        set.iter_values().cloned().collect::<Vec<_>>(),
        [
            "g001", "g004", "g008", "g007", "g003", "g012", "g010", "g002", "g006", "g011", "g009",
            "g005",
        ]
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>()
    );
}

#[test]
fn blitz_mode_runs_tiny_fixture() {
    let ranks = RankedList::new(
        vec!["g1", "g2", "g3", "g4", "g5", "g6"]
            .into_iter()
            .map(str::to_string)
            .collect(),
        vec![3.0, 2.0, 1.0, -1.0, -2.0, -3.0],
    );
    let pathways = vec![
        Pathway {
            name: "PW_A".to_string(),
            description: None,
            genes: vec!["g1", "g2", "g3"]
                .into_iter()
                .map(str::to_string)
                .collect(),
        },
        Pathway {
            name: "PW_B".to_string(),
            description: None,
            genes: vec!["g4", "g5", "g6"]
                .into_iter()
                .map(str::to_string)
                .collect(),
        },
    ];
    let options = BlitzOptions {
        permutations: 64,
        anchors: 8,
        min_size: 1,
        max_size: 6,
        processes: 1,
        ..BlitzOptions::default()
    };
    let res = fgsea_blitz_with_options(&ranks, &pathways, &options).unwrap();
    assert_eq!(res.len(), 2);
    assert!(res.iter().all(|row| row.p_value.is_finite()));
}

#[test]
fn blitz_prepare_signature_sorts_before_deduplicating() {
    let ranks = RankedList {
        genes: ["dup", "low", "dup", "high", "mid"]
            .into_iter()
            .map(str::to_string)
            .collect(),
        scores: vec![-2.0, -5.0, 4.0, 8.0, 1.0],
    };
    let signature = prepare_signature(&ranks, false);
    assert_eq!(
        signature.genes,
        ["high", "dup", "mid", "low"]
            .into_iter()
            .map(str::to_string)
            .collect::<Vec<_>>()
    );
    assert_eq!(signature.abs_scores, vec![8.0, 4.0, 1.0, 5.0]);
}

#[test]
fn blitz_clean_pathways_matches_python_string_set_order_reference() {
    let (_, cleaned) = reference_inputs();
    let top = cleaned
        .iter()
        .find(|pathway| pathway.name == "TOP_12")
        .map(|pathway| &pathway.genes)
        .unwrap();
    assert_eq!(
        top,
        &[
            "g001", "g004", "g009", "g008", "g007", "g003", "g010", "g002", "g006", "g011", "g012",
            "g005",
        ]
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>()
    );
}

#[test]
fn optimized_enrichment_score_extrema_match_running_sum_reference() {
    let abs_scores = vec![3.0, 1.5, 7.0, 2.25, 4.0, 5.5, 0.75, 6.25, 3.75];
    let hits = vec![7, 0, 4, 2];
    let mut sorted_hits = hits.clone();
    sorted_hits.sort_unstable();
    let mut scratch = BlitzScoreScratch::new(abs_scores.len());
    let observed = enrichment_score_for_indices(&abs_scores, &hits, &sorted_hits, &mut scratch);

    let mut hit_indicator = vec![0.0; abs_scores.len()];
    for &hit in &hits {
        hit_indicator[hit] = 1.0;
    }
    let number_hits = hits.len();
    let number_miss = abs_scores.len().saturating_sub(number_hits);
    let hit_scores = hits.iter().map(|&idx| abs_scores[idx]).collect::<Vec<_>>();
    let sum_hit_scores = numpy_hit_score_sum_f64(&hit_scores);
    let norm_hit = 1.0 / sum_hit_scores;
    let norm_no_hit = 1.0 / number_miss as f64;
    let mut running = Vec::new();
    let mut csum = 0.0;
    let mut best_idx = 0usize;
    let mut best_abs = f64::NEG_INFINITY;
    for i in 0..abs_scores.len() {
        csum +=
            hit_indicator[i] * abs_scores[i] * norm_hit - (1.0 - hit_indicator[i]) * norm_no_hit;
        running.push(csum);
        let cur_abs = csum.abs();
        if cur_abs > best_abs {
            best_abs = cur_abs;
            best_idx = i;
        }
    }
    let (rmax, max_value) = running
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, v)| (i, *v))
        .unwrap();
    let (rmin, min_value) = running
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, v)| (i, *v))
        .unwrap();

    assert_f64_bits("optimized ES", observed.es, running[best_idx]);
    assert_eq!(observed.peak_idx, best_idx);
    assert_eq!(observed.rmax, rmax);
    assert_eq!(observed.rmin, rmin);
    assert_f64_bits("optimized max", observed.max_value, max_value);
    assert_f64_bits("optimized min", observed.min_value, min_value);
}

#[test]
fn sorted_hit_anchor_es_matches_marker_reference_exactly() {
    let abs_scores = (0..257)
        .map(|i| (((i * 37 + 11) % 101) as f64 + 1.0) / 17.0)
        .collect::<Vec<_>>();
    let hit_sets = [
        vec![0, 3, 11, 87, 256],
        vec![255, 7, 128, 1, 64, 9, 200],
        vec![13, 14, 15, 16, 17, 18, 19, 20],
        vec![101, 5, 230, 44, 177, 6, 88, 2, 199],
    ];
    let mut sorted_hits = Vec::new();
    let mut sorted_hits_u32 = Vec::new();
    let mut marker = HitMarker::new(abs_scores.len());
    for hits in hit_sets {
        let observed = enrichment_score_for_hits(&abs_scores, &hits, &mut sorted_hits);
        let hits_u32 = hits.iter().map(|&idx| idx as u32).collect::<Vec<_>>();
        let observed_u32 =
            enrichment_score_for_hits_u32(&abs_scores, &hits_u32, &mut sorted_hits_u32);
        let expected = enrichment_score_for_hits_marker_reference(&abs_scores, &hits, &mut marker);
        assert_f64_bits("sorted-hit anchor ES", observed, expected);
        assert_f64_bits("u32 sorted-hit anchor ES", observed_u32, expected);
    }
}

#[test]
fn parallel_final_scoring_matches_sequential_scoring_exactly() {
    let (signature, cleaned) = reference_inputs();
    let options = BlitzOptions::default();
    let model = estimate_model(&signature, &cleaned, &options).unwrap();
    let params = model_params_by_size(&model, &cleaned, &options);
    let mut sequential_options = options.clone();
    sequential_options.processes = 1;
    let mut parallel_options = options.clone();
    parallel_options.processes = 4;

    let sequential =
        score_blitz_pathways(&signature, &cleaned, &params, &sequential_options).unwrap();
    let parallel = score_blitz_pathways(&signature, &cleaned, &params, &parallel_options).unwrap();
    assert_eq!(sequential.len(), parallel.len());
    for (seq, par) in sequential.iter().zip(parallel.iter()) {
        assert_eq!(seq.result.pathway_name, par.result.pathway_name);
        assert_eq!(seq.result.size, par.result.size);
        assert_eq!(seq.result.leading_edge, par.result.leading_edge);
        assert_f64_bits(
            &format!("{} ES", seq.result.pathway_name),
            seq.result.es,
            par.result.es,
        );
        assert_f64_bits(
            &format!("{} pval", seq.result.pathway_name),
            seq.result.p_value,
            par.result.p_value,
        );
        assert_f64_bits(
            &format!("{} NES", seq.result.pathway_name),
            seq.result.nes.unwrap(),
            par.result.nes.unwrap(),
        );
    }
}

#[test]
fn blitz_model_cache_reuses_exact_model() {
    clear_blitz_model_cache();
    let (signature, cleaned) = reference_inputs();
    let options = BlitzOptions::default();

    let (cold, first_hit) = estimate_model_with_cache(&signature, &cleaned, &options).unwrap();
    let (warm, second_hit) = estimate_model_with_cache(&signature, &cleaned, &options).unwrap();

    assert!(!first_hit);
    assert!(second_hit);
    assert_linear_interp_bits("alpha_pos", &warm.alpha_pos, &cold.alpha_pos);
    assert_linear_interp_bits("beta_pos", &warm.beta_pos, &cold.beta_pos);
    assert_linear_interp_bits("pos_ratio", &warm.pos_ratio, &cold.pos_ratio);
    assert_linear_interp_bits("alpha_neg", &warm.alpha_neg, &cold.alpha_neg);
    assert_linear_interp_bits("beta_neg", &warm.beta_neg, &cold.beta_neg);
}

#[test]
fn blitz_model_cache_key_changes_for_model_affecting_inputs() {
    let (signature, cleaned) = reference_inputs();
    let options = BlitzOptions::default();
    let key = blitz_model_cache_key(&signature, &cleaned, &options);

    let mut changed_seed = options.clone();
    changed_seed.seed += 1;
    assert_ne!(
        key,
        blitz_model_cache_key(&signature, &cleaned, &changed_seed)
    );

    let mut changed_permutations = options.clone();
    changed_permutations.permutations += 1;
    assert_ne!(
        key,
        blitz_model_cache_key(&signature, &cleaned, &changed_permutations)
    );

    let mut changed_processes = options.clone();
    changed_processes.processes += 1;
    assert_ne!(
        key,
        blitz_model_cache_key(&signature, &cleaned, &changed_processes)
    );

    let mut changed_library = cleaned.clone();
    let larger_size = max_cleaned_library_size(&cleaned) + 1;
    changed_library.push(CleanedPathway {
        name: "cache_key_larger_pathway".to_string(),
        genes: (0..larger_size)
            .map(|idx| format!("cache_gene_{idx}"))
            .collect(),
        hit_indices: Vec::new(),
        sorted_hit_indices: Vec::new(),
        leading_hits: PythonIntSet::new(),
    });
    assert_ne!(
        key,
        blitz_model_cache_key(&signature, &changed_library, &options)
    );
}

#[test]
fn blitz_model_cache_key_uses_score_distribution_not_gene_names() {
    let (signature, cleaned) = reference_inputs();
    let mut renamed_signature = signature.clone();
    for (idx, gene) in renamed_signature.genes.iter_mut().enumerate() {
        *gene = format!("renamed_gene_{idx}");
    }
    renamed_signature.gene_to_idx = renamed_signature
        .genes
        .iter()
        .enumerate()
        .map(|(idx, gene)| (gene.clone(), idx))
        .collect();
    let options = BlitzOptions::default();

    assert_eq!(
        blitz_model_cache_key(&signature, &cleaned, &options),
        blitz_model_cache_key(&renamed_signature, &cleaned, &options)
    );
}

#[derive(Debug, Deserialize)]
struct AnchorTraceRow {
    set_size: usize,
    alpha_pos: f64,
    beta_pos: f64,
    alpha_neg: f64,
    beta_neg: f64,
    pos_ratio: f64,
    alpha_pos_smooth: f64,
    beta_pos_smooth: f64,
    alpha_neg_smooth: f64,
    beta_neg_smooth: f64,
    pos_ratio_jittered: f64,
    pos_ratio_smooth: f64,
}

#[derive(Debug, Deserialize)]
struct ResultTraceRow {
    pathway: String,
    set_size: usize,
    es: f64,
    pos_alpha: f64,
    pos_beta: f64,
    pos_ratio_clipped: f64,
    neg_alpha: f64,
    neg_beta: f64,
    nes: f64,
    pval: f64,
}

#[derive(Debug, Deserialize)]
struct SignatureTraceRow {
    gene: String,
    centered_score: f64,
    abs_score: f64,
}

#[derive(Debug, Deserialize)]
struct TailTraceRow {
    case: String,
    branch: String,
    x: f64,
    alpha: f64,
    beta: f64,
    pos_ratio: f64,
    deep_accuracy: usize,
    fallback_used: bool,
    gamma_prob: f64,
    survival_prob: f64,
    prob_two_tailed: f64,
    pval: f64,
    nes: f64,
}

fn reference_inputs() -> (BlitzSignature, Vec<CleanedPathway>) {
    let root = env!("CARGO_MANIFEST_DIR");
    let ranks =
        read_ranked_list(format!("{root}/tests/data/blitz_reference/synthetic.rnk")).unwrap();
    let pathways = read_gmt(format!("{root}/tests/data/blitz_reference/synthetic.gmt"))
        .unwrap()
        .pathways;
    let signature = prepare_signature(&ranks, true);
    let cleaned = clean_pathways(&pathways, &signature);
    (signature, cleaned)
}

fn read_anchor_trace() -> Vec<AnchorTraceRow> {
    let root = env!("CARGO_MANIFEST_DIR");
    let mut reader = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .from_path(format!(
            "{root}/tests/data/blitz_reference/synthetic.trace_anchors.tsv"
        ))
        .unwrap();
    reader
        .deserialize::<AnchorTraceRow>()
        .collect::<Result<Vec<_>, _>>()
        .unwrap()
}

fn read_signature_trace() -> Vec<SignatureTraceRow> {
    let root = env!("CARGO_MANIFEST_DIR");
    let mut reader = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .from_path(format!(
            "{root}/tests/data/blitz_reference/synthetic.trace_signature.tsv"
        ))
        .unwrap();
    reader
        .deserialize::<SignatureTraceRow>()
        .collect::<Result<Vec<_>, _>>()
        .unwrap()
}

fn read_result_trace() -> Vec<ResultTraceRow> {
    let root = env!("CARGO_MANIFEST_DIR");
    let mut reader = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .from_path(format!(
            "{root}/tests/data/blitz_reference/synthetic.trace_results.tsv"
        ))
        .unwrap();
    reader
        .deserialize::<ResultTraceRow>()
        .collect::<Result<Vec<_>, _>>()
        .unwrap()
}

fn read_tail_trace() -> Vec<TailTraceRow> {
    let root = env!("CARGO_MANIFEST_DIR");
    let mut reader = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .from_path(format!(
            "{root}/tests/data/blitz_reference/tail_fallback.trace_gamma.tsv"
        ))
        .unwrap();
    reader
        .deserialize::<TailTraceRow>()
        .collect::<Result<Vec<_>, _>>()
        .unwrap()
}

fn assert_f64_bits(label: &str, observed: f64, expected: f64) {
    assert_eq!(
        observed.to_bits(),
        expected.to_bits(),
        "{label}: observed {observed:?}, expected {expected:?}"
    );
}

fn assert_linear_interp_bits(label: &str, observed: &LinearInterp, expected: &LinearInterp) {
    assert_eq!(observed.x.len(), expected.x.len(), "{label} x length");
    assert_eq!(observed.y.len(), expected.y.len(), "{label} y length");
    for (idx, (observed, expected)) in observed.x.iter().zip(&expected.x).enumerate() {
        assert_f64_bits(&format!("{label} x[{idx}]"), *observed, *expected);
    }
    for (idx, (observed, expected)) in observed.y.iter().zip(&expected.y).enumerate() {
        assert_f64_bits(&format!("{label} y[{idx}]"), *observed, *expected);
    }
}

#[test]
fn blitz_anchor_trace_matches_reference() {
    let (signature, cleaned) = reference_inputs();
    let options = BlitzOptions::default();
    let signature_trace = read_signature_trace();
    assert_eq!(signature.genes.len(), signature_trace.len());
    for (idx, expected) in signature_trace.iter().enumerate() {
        assert_eq!(signature.genes[idx], expected.gene);
        assert_f64_bits(
            &format!("{} abs(centered score)", expected.gene),
            signature.abs_scores[idx],
            expected.centered_score.abs(),
        );
        assert_f64_bits(
            &format!("{} abs score", expected.gene),
            signature.abs_scores[idx],
            expected.abs_score,
        );
    }
    let (anchor_sizes, fits) = estimate_anchor_fits(&signature, &cleaned, &options).unwrap();
    let trace = read_anchor_trace();
    assert_eq!(anchor_sizes.len(), trace.len());
    assert_eq!(fits.len(), trace.len());

    let x = anchor_sizes.iter().map(|&v| v as f64).collect::<Vec<_>>();
    let alpha_pos = fits.iter().map(|fit| fit.alpha_pos).collect::<Vec<_>>();
    let beta_pos = fits.iter().map(|fit| fit.beta_pos).collect::<Vec<_>>();
    let alpha_neg = fits.iter().map(|fit| fit.alpha_neg).collect::<Vec<_>>();
    let beta_neg = fits.iter().map(|fit| fit.beta_neg).collect::<Vec<_>>();
    let mut jitter_rng = NumpyMt19937::new(options.seed as u32);
    let jitters = jitter_rng.standard_normals(fits.len());
    let pos_ratio_jittered = fits
        .iter()
        .zip(jitters)
        .map(|(fit, jitter)| (fit.pos_ratio - (0.0001 * jitter).abs()).clamp(0.0, 1.0))
        .collect::<Vec<_>>();
    let alpha_pos_smooth = lowess(&alpha_pos, &x, 0.6);
    let beta_pos_smooth = lowess(&beta_pos, &x, 0.15);
    let alpha_neg_smooth = lowess(&alpha_neg, &x, 0.6);
    let beta_neg_smooth = lowess(&beta_neg, &x, 0.15);
    let pos_ratio_smooth = lowess(&pos_ratio_jittered, &x, 0.5);

    for (idx, ((&set_size, fit), expected)) in
        anchor_sizes.iter().zip(&fits).zip(&trace).enumerate()
    {
        assert_eq!(set_size, expected.set_size);
        assert_f64_bits(
            &format!("{set_size} alpha_pos"),
            fit.alpha_pos,
            expected.alpha_pos,
        );
        assert_f64_bits(
            &format!("{set_size} beta_pos"),
            fit.beta_pos,
            expected.beta_pos,
        );
        assert_f64_bits(
            &format!("{set_size} alpha_neg"),
            fit.alpha_neg,
            expected.alpha_neg,
        );
        assert_f64_bits(
            &format!("{set_size} beta_neg"),
            fit.beta_neg,
            expected.beta_neg,
        );
        assert_f64_bits(
            &format!("{set_size} pos_ratio"),
            fit.pos_ratio,
            expected.pos_ratio,
        );
        assert_f64_bits(
            &format!("{set_size} pos_ratio_jittered"),
            pos_ratio_jittered[idx],
            expected.pos_ratio_jittered,
        );
        assert_f64_bits(
            &format!("{set_size} alpha_pos_smooth"),
            alpha_pos_smooth[idx],
            expected.alpha_pos_smooth,
        );
        assert_f64_bits(
            &format!("{set_size} beta_pos_smooth"),
            beta_pos_smooth[idx],
            expected.beta_pos_smooth,
        );
        assert_f64_bits(
            &format!("{set_size} alpha_neg_smooth"),
            alpha_neg_smooth[idx],
            expected.alpha_neg_smooth,
        );
        assert_f64_bits(
            &format!("{set_size} beta_neg_smooth"),
            beta_neg_smooth[idx],
            expected.beta_neg_smooth,
        );
        assert_f64_bits(
            &format!("{set_size} pos_ratio_smooth"),
            pos_ratio_smooth[idx],
            expected.pos_ratio_smooth,
        );
    }
}

#[test]
fn blitz_result_trace_matches_reference() {
    for row in read_result_trace() {
        let (prob_two_tailed, nes) = if row.es > 0.0 {
            let gamma_prob = gamma_cdf(row.es, row.pos_alpha, row.pos_beta);
            let combined =
                (gamma_prob * row.pos_ratio_clipped + 1.0 - row.pos_ratio_clipped).min(1.0);
            let prob_two_tailed = (1.0 - combined).min(0.5);
            (prob_two_tailed, normal_isf(prob_two_tailed))
        } else {
            let gamma_prob = gamma_cdf(-row.es, row.neg_alpha, row.neg_beta);
            let combined = (gamma_prob - (gamma_prob * row.pos_ratio_clipped)
                + row.pos_ratio_clipped)
                .min(1.0);
            let mut prob_two_tailed = (1.0 - combined).min(0.5);
            if prob_two_tailed == 0.5 {
                prob_two_tailed -= gamma_prob;
            }
            (prob_two_tailed, -normal_isf(prob_two_tailed))
        };
        assert_f64_bits(&format!("{} nes", row.pathway), nes, row.nes);
        assert_f64_bits(
            &format!("{} pval", row.pathway),
            2.0 * prob_two_tailed,
            row.pval,
        );
        assert!(row.set_size > 0);
    }
}

#[test]
fn mpmath_gammacdf_matches_tail_trace_exact_bits() {
    let rows = read_tail_trace();
    assert!(rows.iter().any(|row| row.fallback_used));
    let sampled = rows
        .into_iter()
        .enumerate()
        .filter_map(|(idx, row)| {
            (idx % 19 == 0
                || row.case == "pos_lower_half"
                || row.case == "pos_upper_integer"
                || row.case == "pos_lower_noninteger"
                || row.case == "neg_lower_integer"
                || row.case == "neg_upper_integer"
                || row.case == "neg_upper_noninteger")
                .then_some(row)
        })
        .collect::<Vec<_>>();
    assert!(sampled.len() >= 10);
    for row in sampled {
        let observed =
            crate::blitz_mpmath::gammacdf(row.x, row.alpha, row.beta, row.deep_accuracy).unwrap();
        assert_f64_bits(
            &format!("{} gamma_prob", row.case),
            observed.cdf,
            row.gamma_prob,
        );
        assert_f64_bits(
            &format!("{} survival_prob", row.case),
            observed.survival,
            row.survival_prob,
        );
    }
}

#[test]
fn blitz_tail_probability_matches_trace_exact_bits() {
    for row in read_tail_trace() {
        let branch = match row.branch.as_str() {
            "pos" => crate::blitz_mpmath::TailBranch::Positive,
            "neg" => crate::blitz_mpmath::TailBranch::Negative,
            other => panic!("unknown tail trace branch {other}"),
        };
        let observed = crate::blitz_mpmath::tail_probability(
            branch,
            row.x,
            row.alpha,
            row.beta,
            row.pos_ratio,
            row.deep_accuracy,
        )
        .unwrap();
        assert_f64_bits(
            &format!("{} gamma_prob", row.case),
            observed.gamma_prob,
            row.gamma_prob,
        );
        if row.pval == 0.0 && !row.nes.is_finite() {
            assert!(
                observed.survival_prob.is_finite() && observed.survival_prob >= 0.0,
                "{} hidden survival_prob should stay finite for underflow sentinel",
                row.case
            );
        } else {
            assert_f64_bits(
                &format!("{} survival_prob", row.case),
                observed.survival_prob,
                row.survival_prob,
            );
        }
        assert_f64_bits(
            &format!("{} prob_two_tailed", row.case),
            observed.prob_two_tailed,
            row.prob_two_tailed,
        );
        assert_f64_bits(&format!("{} pval", row.case), observed.p_value, row.pval);
        let nes = match branch {
            crate::blitz_mpmath::TailBranch::Positive => normal_isf(observed.prob_two_tailed),
            crate::blitz_mpmath::TailBranch::Negative => {
                let mut nes = -normal_isf(observed.prob_two_tailed);
                if nes == 0.0 {
                    nes = -0.0;
                }
                nes
            }
        };
        assert_f64_bits(&format!("{} nes", row.case), nes, row.nes);
    }
}

#[test]
fn gamma_cdf_blitz_uses_fallback_without_error() {
    let rows = read_tail_trace()
        .into_iter()
        .filter(|row| {
            row.fallback_used
                && (row.case == "pos_lower_half"
                    || row.case == "pos_upper_integer"
                    || row.case == "neg_upper_noninteger")
        })
        .collect::<Vec<_>>();
    assert_eq!(rows.len(), 3);
    for row in rows {
        let observed = gamma_cdf_blitz(row.x, row.alpha, row.beta, row.deep_accuracy).unwrap();
        assert_f64_bits(
            &format!("{} fallback cdf", row.case),
            observed,
            row.gamma_prob,
        );
    }
}
