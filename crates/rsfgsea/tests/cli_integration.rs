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
