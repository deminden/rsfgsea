use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
use tempfile::tempdir;

fn write_test_inputs() -> (
    tempfile::TempDir,
    std::path::PathBuf,
    std::path::PathBuf,
    std::path::PathBuf,
) {
    let dir = tempdir().unwrap();
    let ranks = dir.path().join("test.rnk");
    let gmt = dir.path().join("test.gmt");
    let output = dir.path().join("out.tsv");

    fs::write(&ranks, "g1\t2.0\ng2\t1.0\ng3\t-1.0\ng4\t-2.0\n").unwrap();
    fs::write(&gmt, "PW_A\tdesc\tg1\tg2\nPW_B\tdesc\tg3\tg4\n").unwrap();

    (dir, ranks, gmt, output)
}

fn cli_bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_rsfgsea"))
}

fn plot_cli_bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_rsfgsea-plot-enrichment"))
}

fn plot_table_cli_bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_rsfgsea-plot-gsea-table"))
}

#[test]
fn cli_simple_mode_writes_results() {
    let (_dir, ranks, gmt, output) = write_test_inputs();

    cli_bin()
        .args([
            "--mode",
            "simple",
            "--nPermSimple",
            "100",
            "--ranks",
            ranks.to_str().unwrap(),
            "--gmt",
            gmt.to_str().unwrap(),
            "--output",
            output.to_str().unwrap(),
        ])
        .assert()
        .success();

    let content = fs::read_to_string(output).unwrap();
    assert!(content.contains("pathway\tsize\tes\tnes\tpval\tpadj\tlog2err\tleading_edge"));
    assert!(content.contains("PW_A"));
}

#[test]
fn decor_cli_builds_cache_then_reuses_it() {
    let dir = tempdir().unwrap();
    let ranks = "tests/data/decor_ranks.rnk";
    let gmt = "tests/data/decor_pathways.gmt";
    let expression = "tests/data/decor_expression.tsv";
    let cache = dir.path().join("decor-cache.tsv");
    let output = dir.path().join("decor.tsv");

    cli_bin()
        .args([
            "--method",
            "decor",
            "--mode",
            "simple",
            "--nperm",
            "100",
            "--seed",
            "42",
            "--ranks",
            ranks,
            "--gmt",
            gmt,
            "--output",
            output.to_str().unwrap(),
            "--decor-cache",
            cache.to_str().unwrap(),
            "--decor-expression",
            expression,
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("Decor cache built"));

    assert!(cache.exists());
    let first = fs::read_to_string(&output).unwrap();
    assert!(first.contains("PW_REDUNDANT"));

    cli_bin()
        .args([
            "--method",
            "decor",
            "--mode",
            "simple",
            "--nperm",
            "100",
            "--seed",
            "42",
            "--ranks",
            ranks,
            "--gmt",
            gmt,
            "--output",
            output.to_str().unwrap(),
            "--decor-cache",
            cache.to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("Reusing compatible decor cache"));
}

fn assert_tsv_snapshot(actual_path: &std::path::Path, expected_path: &str) {
    let actual = fs::read_to_string(actual_path)
        .unwrap()
        .replace("\r\n", "\n");
    let expected = fs::read_to_string(expected_path)
        .unwrap()
        .replace("\r\n", "\n");
    assert_eq!(actual, expected);
}

#[test]
fn synthetic_decor_simple_snapshot_preserves_conditional_null_parity() {
    let dir = tempdir().unwrap();
    let output = dir.path().join("decor.tsv");
    let cache = dir.path().join("decor-cache.tsv");

    cli_bin()
        .args([
            "--method",
            "decor",
            "--mode",
            "simple",
            "--nperm",
            "128",
            "--seed",
            "20260322",
            "--ranks",
            "tests/data/decor_ranks.rnk",
            "--gmt",
            "tests/data/decor_pathways.gmt",
            "--output",
            output.to_str().unwrap(),
            "--decor-cache",
            cache.to_str().unwrap(),
            "--decor-expression",
            "tests/data/decor_expression.tsv",
            "--minSize",
            "1",
            "--maxSize",
            "10",
        ])
        .assert()
        .success();

    assert_tsv_snapshot(
        &output,
        "tests/data/synthetic_decor_simple_seed20260322_nperm128_alpha23.expected.tsv",
    );
}

#[test]
fn decor_cli_rejects_multilevel() {
    let dir = tempdir().unwrap();
    let output = dir.path().join("decor.tsv");
    let cache = dir.path().join("decor-cache.tsv");

    cli_bin()
        .args([
            "--method",
            "decor",
            "--mode",
            "multilevel",
            "--ranks",
            "tests/data/decor_ranks.rnk",
            "--gmt",
            "tests/data/decor_pathways.gmt",
            "--output",
            output.to_str().unwrap(),
            "--decor-cache",
            cache.to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "decor currently supports CPU simple-mode null only",
        ));
}

#[test]
fn cli_gpu_rejects_non_fgsea_mode_before_adapter_init() {
    let (_dir, ranks, gmt, output) = write_test_inputs();
    let expected_stderr = if cfg!(feature = "gpu") {
        "--gpu currently supports only --mode fgsea."
    } else {
        "--gpu requires building the CLI with --features gpu."
    };

    cli_bin()
        .args([
            "--gpu",
            "--mode",
            "simple",
            "--ranks",
            ranks.to_str().unwrap(),
            "--gmt",
            gmt.to_str().unwrap(),
            "--output",
            output.to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains(expected_stderr));
}

#[test]
fn plot_cli_writes_png() {
    let (_dir, ranks, gmt, output) = write_test_inputs();
    let png = output.with_extension("png");

    plot_cli_bin()
        .args([
            "--ranks",
            ranks.to_str().unwrap(),
            "--gmt",
            gmt.to_str().unwrap(),
            "--pathway",
            "PW_A",
            "--output",
            png.to_str().unwrap(),
            "--dpi",
            "300",
        ])
        .assert()
        .success();

    let bytes = fs::read(&png).unwrap();
    assert!(bytes.starts_with(&[0x89, b'P', b'N', b'G']));
}

#[test]
fn plot_cli_transparent_background_writes_rgba_png() {
    let (_dir, ranks, gmt, output) = write_test_inputs();
    let png = output.with_extension("transparent.png");

    plot_cli_bin()
        .args([
            "--ranks",
            ranks.to_str().unwrap(),
            "--gmt",
            gmt.to_str().unwrap(),
            "--pathway",
            "PW_A",
            "--output",
            png.to_str().unwrap(),
            "--transparent-background",
        ])
        .assert()
        .success();

    let data = fs::read(&png).unwrap();
    assert!(data.starts_with(&[0x89, b'P', b'N', b'G']));
    assert_eq!(data[25], 6);
}

#[test]
fn plot_cli_dpi_controls_pixel_dimensions() {
    let (_dir, ranks, gmt, output) = write_test_inputs();
    let png = output.with_extension("png");

    plot_cli_bin()
        .args([
            "--ranks",
            ranks.to_str().unwrap(),
            "--gmt",
            gmt.to_str().unwrap(),
            "--pathway",
            "PW_A",
            "--output",
            png.to_str().unwrap(),
            "--width-in",
            "1.2",
            "--height-in",
            "1.0",
            "--dpi",
            "300",
        ])
        .assert()
        .success();

    let bytes = fs::read(&png).unwrap();
    let width = u32::from_be_bytes(bytes[16..20].try_into().unwrap());
    let height = u32::from_be_bytes(bytes[20..24].try_into().unwrap());
    assert_eq!(width, 360);
    assert_eq!(height, 300);
}

#[test]
fn plot_table_cli_writes_png() {
    let (_dir, ranks, gmt, output) = write_test_inputs();
    let png = output.with_extension("table.png");

    plot_table_cli_bin()
        .args([
            "--ranks",
            ranks.to_str().unwrap(),
            "--gmt",
            gmt.to_str().unwrap(),
            "--pathway",
            "PW_A",
            "PW_B",
            "--output",
            png.to_str().unwrap(),
            "--dpi",
            "300",
            "--nPermSimple",
            "100",
        ])
        .assert()
        .success();

    let bytes = fs::read(&png).unwrap();
    assert!(bytes.starts_with(&[0x89, b'P', b'N', b'G']));
    let width = u32::from_be_bytes(bytes[16..20].try_into().unwrap());
    let height = u32::from_be_bytes(bytes[20..24].try_into().unwrap());
    assert!(
        width >= 1680,
        "expected table plot width >= 1680 px, got {width}"
    );
    assert!(
        height >= 200,
        "expected table plot height >= 200 px, got {height}"
    );
    assert!(width > height, "expected table plot to be wider than tall");
}
