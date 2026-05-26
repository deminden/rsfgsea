use assert_cmd::Command;
use predicates::prelude::*;
use rsfgsea::prelude::*;
use std::fs;
use std::path::Path;
use tempfile::tempdir;

fn cli_bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_rsfgsea"))
}

fn fixture_paths() -> (&'static str, &'static str, &'static str) {
    (
        "tests/data/decor_ranks.rnk",
        "tests/data/decor_pathways.gmt",
        "tests/data/decor_expression.tsv",
    )
}

fn read_result_body(path: &std::path::Path) -> String {
    fs::read_to_string(path)
        .unwrap()
        .lines()
        .skip(1)
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn decor_dispatch_routes_wrapper_and_explicit_modes() {
    let (ranks, gmt, expression) = fixture_paths();
    let dir = tempdir().unwrap();
    let cache = dir.path().join("decor-cache.tsv");
    let wrapper_out = dir.path().join("wrapper.tsv");
    let explicit_multi_out = dir.path().join("explicit-multilevel.tsv");
    let wrapper_simple_out = dir.path().join("wrapper-simple.tsv");
    let explicit_simple_out = dir.path().join("explicit-simple.tsv");

    cli_bin()
        .args([
            "--method",
            "decor",
            "--mode",
            "fgsea",
            "--nPermSimple",
            "128",
            "--sampleSize",
            "11",
            "--seed",
            "42",
            "--ranks",
            ranks,
            "--gmt",
            gmt,
            "--output",
            wrapper_out.to_str().unwrap(),
            "--decor-cache",
            cache.to_str().unwrap(),
            "--decor-expression",
            expression,
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("method=decor mode=fgsea"))
        .stdout(predicate::str::contains("nperm=None"));

    cli_bin()
        .args([
            "--method",
            "decor",
            "--mode",
            "multilevel",
            "--nPermSimple",
            "128",
            "--sampleSize",
            "11",
            "--seed",
            "42",
            "--ranks",
            ranks,
            "--gmt",
            gmt,
            "--output",
            explicit_multi_out.to_str().unwrap(),
            "--decor-cache",
            cache.to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("mode=multilevel"));

    assert_eq!(
        read_result_body(&wrapper_out),
        read_result_body(&explicit_multi_out),
        "wrapper decor without --nperm should match explicit decor multilevel"
    );

    cli_bin()
        .args([
            "--method",
            "decor",
            "--mode",
            "fgsea",
            "--nperm",
            "128",
            "--seed",
            "42",
            "--ranks",
            ranks,
            "--gmt",
            gmt,
            "--output",
            wrapper_simple_out.to_str().unwrap(),
            "--decor-cache",
            cache.to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("nperm=Some(128)"));

    cli_bin()
        .args([
            "--method",
            "decor",
            "--mode",
            "simple",
            "--nperm",
            "128",
            "--seed",
            "42",
            "--ranks",
            ranks,
            "--gmt",
            gmt,
            "--output",
            explicit_simple_out.to_str().unwrap(),
            "--decor-cache",
            cache.to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("mode=simple"));

    assert_eq!(
        read_result_body(&wrapper_simple_out),
        read_result_body(&explicit_simple_out),
        "decor wrapper with --nperm should match fixed simple mode"
    );

    cli_bin()
        .args([
            "--method",
            "decor",
            "--mode",
            "multilevel",
            "--nperm",
            "128",
            "--ranks",
            ranks,
            "--gmt",
            gmt,
            "--output",
            dir.path().join("bad.tsv").to_str().unwrap(),
            "--decor-cache",
            cache.to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "--nperm is only valid with --mode fgsea or --mode simple",
        ));
}

#[test]
fn decor_release_presets_resolve_to_frozen_formulas() {
    let sensitive = resolve_decor_preset(DecorPreset::Sensitive);
    assert_eq!(sensitive.weight_formula, DecorWeightFormula::RawRational);
    assert_eq!(sensitive.alpha, 22.0);

    let balanced = resolve_decor_preset(DecorPreset::Balanced);
    assert_eq!(
        balanced.weight_formula,
        DecorWeightFormula::ThresholdRational
    );
    assert_eq!(balanced.threshold_tau, 0.04);
    assert_eq!(balanced.alpha, 60.0);

    let specific = resolve_decor_preset(DecorPreset::Specific);
    assert_eq!(
        specific.weight_formula,
        DecorWeightFormula::ThresholdRational
    );
    assert_eq!(specific.threshold_tau, 0.05);
    assert_eq!(specific.alpha, 65.0);

    let strict = resolve_decor_preset(DecorPreset::Strict);
    assert_eq!(strict.weight_formula, DecorWeightFormula::ExpScaled);
    assert!((strict.alpha - (-0.10_f64.ln())).abs() < 1e-15);
    assert_eq!(strict.target_median_penalty, Some(0.10));
}

#[test]
fn decor_default_balanced_matches_explicit_threshold_formula() {
    let (ranks, gmt, expression) = fixture_paths();
    let dir = tempdir().unwrap();
    let cache = dir.path().join("decor-cache.tsv");
    let default_out = dir.path().join("default.tsv");
    let explicit_out = dir.path().join("explicit.tsv");

    cli_bin()
        .args([
            "--method",
            "decor",
            "--mode",
            "simple",
            "--nperm",
            "128",
            "--seed",
            "7",
            "--ranks",
            ranks,
            "--gmt",
            gmt,
            "--output",
            default_out.to_str().unwrap(),
            "--decor-cache",
            cache.to_str().unwrap(),
            "--decor-expression",
            expression,
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("Decor preset=balanced"))
        .stdout(predicate::str::contains(
            "weight_formula=threshold-rational, alpha=60, threshold_tau=0.04",
        ));

    cli_bin()
        .args([
            "--method",
            "decor",
            "--mode",
            "simple",
            "--nperm",
            "128",
            "--seed",
            "7",
            "--ranks",
            ranks,
            "--gmt",
            gmt,
            "--output",
            explicit_out.to_str().unwrap(),
            "--decor-cache",
            cache.to_str().unwrap(),
            "--decor-weight-formula",
            "threshold-rational",
            "--decor-alpha",
            "60",
            "--decor-threshold",
            "0.04",
        ])
        .assert()
        .success();

    assert_eq!(
        read_result_body(&default_out),
        read_result_body(&explicit_out)
    );
}

#[test]
fn decor_simple_and_multilevel_agree_on_tiny_fixture_shape() {
    let (ranks_path, gmt_path, expression_path) = fixture_paths();
    let dir = tempdir().unwrap();
    let cache_path = dir.path().join("decor-cache.tsv");
    let ranks = read_ranked_list(ranks_path).unwrap();
    let pathway_db = read_gmt(gmt_path).unwrap();
    let mut options = DecorOptions {
        cache_path: Some(cache_path),
        expression_path: Some(expression_path.into()),
        ..DecorOptions::default()
    };
    options.apply_preset(DecorPreset::Balanced);
    let (cache, _) =
        ensure_decor_cache_for_paths(&pathway_db.pathways, Path::new(gmt_path), &options, true)
            .unwrap();

    let simple = fgsea_decor_simple_with_options(
        &ranks,
        &pathway_db.pathways,
        &cache,
        &options,
        2000,
        99,
        1,
        ranks.len() - 1,
        1e-10,
        ScoreType::Std,
        1.0,
        101,
    )
    .unwrap();
    let multi = fgsea_decor_multilevel_with_options(
        &ranks,
        &pathway_db.pathways,
        &cache,
        &options,
        1000,
        99,
        1,
        ranks.len() - 1,
        1e-10,
        ScoreType::Std,
        1.0,
        101,
    )
    .unwrap();

    assert_eq!(simple.len(), multi.len());
    for (s, m) in simple.iter().zip(multi.iter()) {
        assert_eq!(s.pathway_name, m.pathway_name);
        assert_eq!(s.es.signum(), m.es.signum());
        assert!(s.p_value.is_finite());
        assert!(m.p_value.is_finite());
        assert!(m.p_value > 0.0 && m.p_value <= 1.0);
        assert!(m.padj.is_some_and(f64::is_finite));
        assert!(m.nes.is_some_and(f64::is_finite));
        assert!(!m.leading_edge.is_empty());
        if let (Some(s_nes), Some(m_nes)) = (s.nes, m.nes) {
            assert!((s_nes - m_nes).abs() < 0.75);
        }
    }
}

#[test]
fn decor_multilevel_eps_and_sample_size_sanity() {
    let (ranks_path, gmt_path, expression_path) = fixture_paths();
    let dir = tempdir().unwrap();
    let cache_path = dir.path().join("decor-cache.tsv");
    let ranks = read_ranked_list(ranks_path).unwrap();
    let pathway_db = read_gmt(gmt_path).unwrap();
    let options = DecorOptions {
        cache_path: Some(cache_path),
        expression_path: Some(expression_path.into()),
        ..DecorOptions::default()
    };
    let (cache, _) =
        ensure_decor_cache_for_paths(&pathway_db.pathways, Path::new(gmt_path), &options, true)
            .unwrap();

    let loose = fgsea_decor_multilevel_with_options(
        &ranks,
        &pathway_db.pathways,
        &cache,
        &options,
        500,
        123,
        1,
        ranks.len() - 1,
        1e-5,
        ScoreType::Std,
        1.0,
        51,
    )
    .unwrap();
    let tight = fgsea_decor_multilevel_with_options(
        &ranks,
        &pathway_db.pathways,
        &cache,
        &options,
        500,
        123,
        1,
        ranks.len() - 1,
        1e-20,
        ScoreType::Std,
        1.0,
        201,
    )
    .unwrap();

    for (loose_row, tight_row) in loose.iter().zip(tight.iter()) {
        assert_eq!(loose_row.pathway_name, tight_row.pathway_name);
        assert!(loose_row.p_value.is_finite());
        assert!(tight_row.p_value.is_finite());
        assert!(tight_row.p_value <= (loose_row.p_value * 100.0).min(1.0));
        if let (Some(loose_err), Some(tight_err)) = (loose_row.log2err, tight_row.log2err) {
            assert!(tight_err <= loose_err + 0.25);
        }
    }
}

#[test]
fn decor_multilevel_handles_edge_case_fixture_without_nan_or_inf() {
    let dir = tempdir().unwrap();
    let ranks = dir.path().join("edge.rnk");
    let gmt = dir.path().join("edge.gmt");
    let expression = dir.path().join("edge-expression.tsv");
    let cache = dir.path().join("edge-cache.tsv");
    let output = dir.path().join("edge-output.tsv");

    fs::write(
        &ranks,
        "g1\t4\ng2\t4\ng3\t3\ng4\t2\ng5\t1\ng6\t-1\ng7\t-2\ng8\t-3\n",
    )
    .unwrap();
    fs::write(
        &gmt,
        "SIZE1\tna\tg1\nN_MINUS1\tna\tg1\tg2\tg3\tg4\tg5\tg6\tg7\nDUPLICATE\tna\tg1\tg1\tg2\nMISSING\tna\tg1\tghost\nBELOW_TAU\tna\tg4\tg5\tg6\n",
    )
    .unwrap();
    fs::write(
        &expression,
        "gene\ts1\ts2\ts3\ts4\n\
g1\t1\t2\t3\t4\n\
g2\t1.1\t2.1\t3.1\t4.1\n\
g3\t4\t3\t2\t1\n\
g4\t1\t1\t1\t1\n\
g5\t2\t1\t2\t1\n\
g6\t3\t2\t1\t0\n\
g7\t4\t4\t4\t4\n\
g8\t0\t1\t0\t1\n",
    )
    .unwrap();

    cli_bin()
        .args([
            "--method",
            "decor",
            "--mode",
            "multilevel",
            "--nPermSimple",
            "512",
            "--sampleSize",
            "51",
            "--eps",
            "1e-10",
            "--seed",
            "314",
            "--ranks",
            ranks.to_str().unwrap(),
            "--gmt",
            gmt.to_str().unwrap(),
            "--output",
            output.to_str().unwrap(),
            "--decor-cache",
            cache.to_str().unwrap(),
            "--decor-expression",
            expression.to_str().unwrap(),
            "--minSize",
            "1",
            "--maxSize",
            "7",
        ])
        .assert()
        .success();

    let content = fs::read_to_string(output).unwrap();
    assert!(content.contains("SIZE1"));
    assert!(content.contains("N_MINUS1"));
    assert!(content.contains("DUPLICATE"));
    assert!(content.contains("MISSING"));
    assert!(!content.contains("NaN"));
    assert!(!content.to_ascii_lowercase().contains("inf"));
}
