use assert_cmd::Command;
use std::fs;
use std::path::{Path, PathBuf};
use tempfile::tempdir;

fn cli_bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_rsfgsea"))
}

fn write_fixture(dir: &Path) -> (PathBuf, PathBuf, PathBuf, PathBuf) {
    let ranks = dir.join("ranks.rnk");
    let gmt = dir.join("pathways.gmt");
    let expr = dir.join("expression.tsv");
    let cache = dir.join("decor-cache.tsv");

    let rank_lines = (1..=36)
        .map(|idx| {
            let score = if idx <= 18 {
                40.0 - idx as f64
            } else {
                -(idx as f64 - 18.0)
            };
            format!("g{idx}\t{score:.3}")
        })
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(&ranks, format!("{rank_lines}\n")).unwrap();

    let pathway = |name: &str, genes: &[usize]| {
        let mut fields = vec![name.to_string(), "na".to_string()];
        fields.extend(genes.iter().map(|idx| format!("g{idx}")));
        fields.join("\t")
    };
    let gmt_lines = [
        pathway("PW_01", &[1]),
        pathway("PW_05", &[1, 2, 3, 4, 5]),
        pathway("PW_15", &(1..=15).collect::<Vec<_>>()),
        pathway("PW_20", &(1..=20).collect::<Vec<_>>()),
        pathway("PW_35", &(1..=35).collect::<Vec<_>>()),
    ]
    .join("\n");
    fs::write(&gmt, format!("{gmt_lines}\n")).unwrap();

    let mut expr_lines = vec!["gene\ts1\ts2\ts3\ts4".to_string()];
    for idx in 1..=36 {
        expr_lines.push(format!(
            "g{idx}\t{:.3}\t{:.3}\t{:.3}\t{:.3}",
            idx as f64,
            idx as f64 + 0.25,
            37.0 - idx as f64,
            (idx % 7) as f64
        ));
    }
    fs::write(&expr, format!("{}\n", expr_lines.join("\n"))).unwrap();

    (ranks, gmt, expr, cache)
}

#[derive(Default)]
struct RunDecorOptions<'a> {
    expr: Option<&'a Path>,
    mode: &'a str,
    nperm: Option<&'a str>,
    min_size: Option<&'a str>,
    explicit_formula: bool,
}

fn run_decor(ranks: &Path, gmt: &Path, cache: &Path, out: &Path, options: RunDecorOptions<'_>) {
    let mut cmd = cli_bin();
    cmd.args([
        "--method",
        "decor",
        "--mode",
        options.mode,
        "--nPermSimple",
        "128",
        "--sampleSize",
        "11",
        "--seed",
        "42",
        "--ranks",
        ranks.to_str().unwrap(),
        "--gmt",
        gmt.to_str().unwrap(),
        "--output",
        out.to_str().unwrap(),
        "--decor-cache",
        cache.to_str().unwrap(),
    ]);
    if let Some(expr) = options.expr {
        cmd.args(["--decor-expression", expr.to_str().unwrap()]);
    }
    if let Some(nperm) = options.nperm {
        cmd.args(["--nperm", nperm]);
    }
    if let Some(min_size) = options.min_size {
        cmd.args(["--minSize", min_size]);
    }
    if options.explicit_formula {
        cmd.args([
            "--decor-weight-formula",
            "threshold-rational",
            "--decor-alpha",
            "60",
            "--decor-threshold",
            "0.04",
        ]);
    }
    cmd.assert().success();
}

fn pathway_sizes(path: &Path) -> Vec<(String, usize)> {
    fs::read_to_string(path)
        .unwrap()
        .lines()
        .skip(1)
        .map(|line| {
            let fields = line.split('\t').collect::<Vec<_>>();
            (fields[0].to_string(), fields[1].parse::<usize>().unwrap())
        })
        .collect()
}

fn assert_same_pathway_set(simple: &Path, multilevel: &Path) {
    assert_eq!(pathway_sizes(simple), pathway_sizes(multilevel));
}

#[test]
fn decor_simple_and_multilevel_share_default_filtering() {
    let dir = tempdir().unwrap();
    let (ranks, gmt, expr, cache) = write_fixture(dir.path());
    let simple = dir.path().join("simple.tsv");
    let multilevel = dir.path().join("multilevel.tsv");

    run_decor(
        &ranks,
        &gmt,
        &cache,
        &simple,
        RunDecorOptions {
            expr: Some(&expr),
            mode: "simple",
            nperm: Some("128"),
            ..Default::default()
        },
    );
    run_decor(
        &ranks,
        &gmt,
        &cache,
        &multilevel,
        RunDecorOptions {
            mode: "multilevel",
            ..Default::default()
        },
    );

    assert_same_pathway_set(&simple, &multilevel);
}

#[test]
fn decor_simple_and_multilevel_share_explicit_min_size_filtering() {
    let dir = tempdir().unwrap();
    let (ranks, gmt, expr, cache) = write_fixture(dir.path());
    let simple_min15 = dir.path().join("simple-min15.tsv");
    let multilevel_min15 = dir.path().join("multilevel-min15.tsv");
    let simple_min1 = dir.path().join("simple-min1.tsv");
    let multilevel_min1 = dir.path().join("multilevel-min1.tsv");

    run_decor(
        &ranks,
        &gmt,
        &cache,
        &simple_min15,
        RunDecorOptions {
            expr: Some(&expr),
            mode: "simple",
            nperm: Some("128"),
            min_size: Some("15"),
            ..Default::default()
        },
    );
    run_decor(
        &ranks,
        &gmt,
        &cache,
        &multilevel_min15,
        RunDecorOptions {
            mode: "multilevel",
            min_size: Some("15"),
            ..Default::default()
        },
    );
    run_decor(
        &ranks,
        &gmt,
        &cache,
        &simple_min1,
        RunDecorOptions {
            mode: "simple",
            nperm: Some("128"),
            min_size: Some("1"),
            ..Default::default()
        },
    );
    run_decor(
        &ranks,
        &gmt,
        &cache,
        &multilevel_min1,
        RunDecorOptions {
            mode: "multilevel",
            min_size: Some("1"),
            ..Default::default()
        },
    );

    assert_same_pathway_set(&simple_min15, &multilevel_min15);
    assert_same_pathway_set(&simple_min1, &multilevel_min1);
    assert_eq!(
        pathway_sizes(&simple_min15),
        vec![
            ("PW_15".to_string(), 15),
            ("PW_20".to_string(), 20),
            ("PW_35".to_string(), 35),
        ]
    );
}

#[test]
fn decor_wrapper_multilevel_and_explicit_formula_share_filtering() {
    let dir = tempdir().unwrap();
    let (ranks, gmt, expr, cache) = write_fixture(dir.path());
    let wrapper = dir.path().join("wrapper.tsv");
    let explicit_multilevel = dir.path().join("explicit-multilevel.tsv");
    let explicit_formula = dir.path().join("explicit-formula.tsv");

    run_decor(
        &ranks,
        &gmt,
        &cache,
        &wrapper,
        RunDecorOptions {
            expr: Some(&expr),
            mode: "fgsea",
            min_size: Some("15"),
            ..Default::default()
        },
    );
    run_decor(
        &ranks,
        &gmt,
        &cache,
        &explicit_multilevel,
        RunDecorOptions {
            mode: "multilevel",
            min_size: Some("15"),
            ..Default::default()
        },
    );
    run_decor(
        &ranks,
        &gmt,
        &cache,
        &explicit_formula,
        RunDecorOptions {
            mode: "simple",
            nperm: Some("128"),
            min_size: Some("15"),
            explicit_formula: true,
            ..Default::default()
        },
    );

    assert_same_pathway_set(&wrapper, &explicit_multilevel);
    assert_same_pathway_set(&wrapper, &explicit_formula);
}
