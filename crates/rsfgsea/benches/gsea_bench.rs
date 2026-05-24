use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rsfgsea::blitz::{
    __bench_blitz_anchor_calibration, __bench_blitz_prepare_scoring, __bench_blitz_score_prepared,
    __bench_blitz_tail_microcases,
};
use rsfgsea::decor::DecorPathwayScores;
use rsfgsea::prelude::*;
use std::collections::BTreeMap;
use std::fs;
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::Duration;

struct SyntheticWorkload {
    name: &'static str,
    ranks: RankedList,
    pathways: Vec<Pathway>,
}

struct SyntheticDecorMatrixWorkload {
    workload: SyntheticWorkload,
    cache: DecorCache,
    expression_path: PathBuf,
    _temp_dir: tempfile::TempDir,
}

fn benchmark_es(c: &mut Criterion) {
    let n = 10000;
    let k = 100;
    let abs_scores: Vec<f64> = (0..n).map(|i| (n - i) as f64).collect();
    let mut hits: Vec<usize> = Vec::new();
    for i in 0..k {
        hits.push(i * (n / k));
    }

    c.bench_function("calculate_es_10k_100", |b| {
        b.iter(|| {
            calculate_es_fgsea(
                black_box(&abs_scores),
                black_box(&hits),
                black_box(n),
                black_box(ScoreType::Std),
            )
        })
    });

    let penalty: Vec<f64> = (0..k)
        .map(|i| 1.0 / (1.0 + 23.0 * (i as f64 / k as f64)))
        .collect();
    c.bench_function("calculate_es_decor_10k_100", |b| {
        b.iter(|| {
            calculate_es_decor(
                black_box(&abs_scores),
                black_box(&hits),
                black_box(&penalty),
                black_box(n),
                black_box(ScoreType::Std),
            )
        })
    });
}

fn synthetic_workload(
    name: &'static str,
    gene_count: usize,
    pathway_count: usize,
) -> SyntheticWorkload {
    let mut rng = StdRng::seed_from_u64(0x5eed_1234);
    let genes: Vec<String> = (0..gene_count).map(|i| format!("g{i:05}")).collect();
    let scores = real_shaped_scores(gene_count, &mut rng);
    let pathways = real_shaped_pathways(&genes, pathway_count, &mut rng);

    SyntheticWorkload {
        name,
        ranks: RankedList::new(genes, scores),
        pathways,
    }
}

fn real_shaped_scores(gene_count: usize, rng: &mut StdRng) -> Vec<f64> {
    let midpoint = gene_count as f64 / 2.0;
    let scale = gene_count as f64 / 7.0;

    (0..gene_count)
        .map(|i| {
            let rank_signal = ((midpoint - i as f64) / scale).tanh() * 4.0;
            let periodic = ((i as f64) / 47.0).sin() * 0.35 + ((i as f64) / 211.0).cos() * 0.2;
            let jitter = rng.random_range(-0.03..0.03);
            let enrichment_spike = match i {
                0..=249 => 2.0,
                250..=499 => 1.2,
                _ if i + 500 >= gene_count => -1.6,
                _ => 0.0,
            };
            rank_signal + periodic + jitter + enrichment_spike
        })
        .collect()
}

fn real_shaped_pathways(genes: &[String], pathway_count: usize, rng: &mut StdRng) -> Vec<Pathway> {
    let module_count = (genes.len() / 250).max(8);
    (0..pathway_count)
        .map(|pathway_idx| {
            let size = sample_realistic_pathway_size(rng).min(genes.len().saturating_sub(1));
            let module_idx = pathway_idx % module_count;
            let module_start = module_idx * genes.len() / module_count;
            let module_width = 650.min(genes.len());
            let module_end = (module_start + module_width).min(genes.len());
            let module_target = (size * 3 / 4).min(module_end - module_start);

            let mut selected = Vec::with_capacity(size);
            let mut used = std::collections::HashSet::with_capacity(size);

            while selected.len() < module_target {
                let idx = rng.random_range(module_start..module_end);
                if used.insert(idx) {
                    selected.push(genes[idx].clone());
                }
            }
            while selected.len() < size {
                let idx = rng.random_range(0..genes.len());
                if used.insert(idx) {
                    selected.push(genes[idx].clone());
                }
            }
            selected.shuffle(rng);

            Pathway {
                name: format!("SYN_PATHWAY_{pathway_idx:05}"),
                description: Some("real-shaped synthetic benchmark pathway".to_string()),
                genes: selected,
            }
        })
        .collect()
}

fn synthetic_decor_cache(workload: &SyntheticWorkload) -> DecorCache {
    let mut rows = 0usize;
    let pathways = workload
        .pathways
        .iter()
        .enumerate()
        .map(|(pathway_idx, pathway)| {
            rows += pathway.genes.len();
            let redundancy = pathway
                .genes
                .iter()
                .enumerate()
                .map(|(gene_idx, _)| {
                    let phase = ((pathway_idx + gene_idx) % 29) as f32 / 28.0;
                    (0.05 + 0.75 * phase).min(1.0)
                })
                .collect();
            (
                pathway.name.clone(),
                DecorPathwayScores {
                    genes: pathway.genes.clone(),
                    redundancy,
                },
            )
        })
        .collect::<BTreeMap<_, _>>();

    DecorCache {
        metadata: DecorCacheMetadata {
            format: "rsfgsea-decor-cache".to_string(),
            version: "1".to_string(),
            created_by: "rsfgsea-benchmark".to_string(),
            gmt_sha256: "synthetic".to_string(),
            expression_sha256: "synthetic".to_string(),
            correlation: DecorCorrelation::Pearson,
            redundancy: DecorRedundancy::PositiveMean,
            expression_gene_axis: "rows".to_string(),
            expression_has_header: true,
            gene_id_mode: "verbatim".to_string(),
            n_pathways: pathways.len(),
            n_rows: rows,
        },
        pathways,
    }
}

fn synthetic_decor_matrix_workload(
    name: &'static str,
    gene_count: usize,
    pathway_count: usize,
    sample_count: usize,
) -> SyntheticDecorMatrixWorkload {
    let workload = synthetic_workload(name, gene_count, pathway_count);
    let cache = synthetic_decor_cache(&workload);
    let temp_dir = tempfile::tempdir().expect("create benchmark temp dir");
    let expression_path = temp_dir.path().join("expression.tsv");
    write_synthetic_expression(&expression_path, &workload.ranks.genes, sample_count);
    SyntheticDecorMatrixWorkload {
        workload,
        cache,
        expression_path,
        _temp_dir: temp_dir,
    }
}

fn write_synthetic_expression(path: &PathBuf, genes: &[String], sample_count: usize) {
    let mut out = String::new();
    out.push_str("gene");
    for sample_idx in 0..sample_count {
        out.push_str(&format!("\ts{sample_idx}"));
    }
    out.push('\n');
    for (gene_idx, gene) in genes.iter().enumerate() {
        out.push_str(gene);
        let module = (gene_idx / 50) as f64;
        for sample_idx in 0..sample_count {
            let sample = sample_idx as f64;
            let value = (sample / 3.0 + module).sin()
                + (gene_idx as f64 / 113.0).cos() * 0.25
                + (sample * (gene_idx % 17) as f64 / 19.0).cos() * 0.15;
            out.push_str(&format!("\t{value:.6}"));
        }
        out.push('\n');
    }
    fs::write(path, out).expect("write synthetic expression");
}

fn sample_realistic_pathway_size(rng: &mut StdRng) -> usize {
    match rng.random_range(0..100) {
        0..=19 => rng.random_range(15..=30),
        20..=44 => rng.random_range(31..=75),
        45..=69 => rng.random_range(76..=150),
        70..=89 => rng.random_range(151..=300),
        _ => rng.random_range(301..=500),
    }
}

fn benchmark_workloads() -> Vec<SyntheticWorkload> {
    let default_workload = synthetic_workload("10k_genes_1k_pathways", 10_000, 1_000);
    let mut workloads = vec![default_workload];
    // Publication-shaped thread scaling runs:
    // for t in 1 2 4 8 16; do
    //   RAYON_NUM_THREADS=$t RSFGSEA_THREAD_MATRIX_BENCH=1 \
    //     cargo bench -p rsfgsea --bench gsea_bench
    // done
    if std::env::var_os("RSFGSEA_THREAD_MATRIX_BENCH").is_some() {
        workloads.push(synthetic_workload("10k_genes_5k_pathways", 10_000, 5_000));
    }
    if std::env::var_os("RSFGSEA_PERM_HEAVY_BENCH").is_some() {
        workloads.push(synthetic_workload(
            "10k_genes_5k_pathways_perm_heavy",
            10_000,
            5_000,
        ));
    }
    // Enable larger release-note-sized workloads with:
    // RSFGSEA_HEAVY_BENCH=1 cargo bench -p rsfgsea --bench gsea_bench
    if std::env::var_os("RSFGSEA_HEAVY_BENCH").is_some() {
        workloads.push(synthetic_workload("20k_genes_15k_pathways", 20_000, 15_000));
    }
    workloads
}

fn simple_permutations(workload_name: &str) -> usize {
    if workload_name.contains("perm_heavy") {
        100_000
    } else {
        10_000
    }
}

fn decor_simple_permutations(workload_name: &str) -> usize {
    if workload_name.contains("perm_heavy") {
        10_000
    } else {
        1_000
    }
}

fn multilevel_permutations(_workload_name: &str) -> usize {
    1000
}

fn wrapper_dispatch_enabled() -> bool {
    std::env::var_os("RSFGSEA_WRAPPER_BENCH").is_some()
}

fn blitz_bench_enabled() -> bool {
    std::env::var_os("RSFGSEA_BLITZ_BENCH").is_some()
}

fn configure_representative_group<'a>(
    c: &'a mut Criterion,
    name: &str,
) -> criterion::BenchmarkGroup<'a, criterion::measurement::WallTime> {
    let mut group = c.benchmark_group(name);
    group.sample_size(15);
    group.measurement_time(Duration::from_secs(10));
    group.warm_up_time(Duration::from_secs(1));
    group
}

fn configure_decor_matrix_group<'a>(
    c: &'a mut Criterion,
    name: &str,
) -> criterion::BenchmarkGroup<'a, criterion::measurement::WallTime> {
    let mut group = c.benchmark_group(name);
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(5));
    group.warm_up_time(Duration::from_secs(1));
    group
}

fn configure_blitz_group<'a>(
    c: &'a mut Criterion,
    name: &str,
) -> criterion::BenchmarkGroup<'a, criterion::measurement::WallTime> {
    let mut group = c.benchmark_group(name);
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(8));
    group.warm_up_time(Duration::from_secs(1));
    group
}

fn decor_options_for_matrix(
    matrix: &SyntheticDecorMatrixWorkload,
    formula: DecorWeightFormula,
) -> DecorOptions {
    DecorOptions {
        alpha: 23.0,
        expression_path: Some(matrix.expression_path.clone()),
        weight_formula: formula,
        penalty_floor: 0.2,
        ..DecorOptions::default()
    }
}

fn blitz_options_for_bench() -> BlitzOptions {
    BlitzOptions {
        permutations: 1000,
        anchors: 40,
        min_size: 5,
        max_size: 4000,
        processes: 4,
        symmetric: false,
        seed: 0,
        center: true,
        accuracy: 40,
        deep_accuracy: 50,
    }
}

fn load_file_workload(name: &'static str, ranks_path: &Path, gmt_path: &Path) -> SyntheticWorkload {
    let ranks = read_ranked_list(ranks_path).expect("read blitz benchmark ranks");
    let pathways = read_gmt(gmt_path)
        .expect("read blitz benchmark gmt")
        .pathways;
    SyntheticWorkload {
        name,
        ranks,
        pathways,
    }
}

fn optional_file_workload(
    name: &'static str,
    ranks_path: &Path,
    gmt_path: &Path,
) -> Option<SyntheticWorkload> {
    (ranks_path.exists() && gmt_path.exists())
        .then(|| load_file_workload(name, ranks_path, gmt_path))
}

fn blitz_benchmark_workloads() -> Vec<SyntheticWorkload> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let repo_root = manifest_dir
        .parent()
        .and_then(Path::parent)
        .expect("crate should live under workspace crates/");
    let mut workloads = vec![
        synthetic_workload("synthetic_10k_genes_1k_pathways", 10_000, 1_000),
        load_file_workload(
            "publication_fixture",
            &manifest_dir.join("tests/data/blitz_reference/publication_fgsea.rnk"),
            &manifest_dir.join("tests/data/blitz_reference/publication_fgsea.gmt"),
        ),
    ];
    if let Some(workload) = optional_file_workload(
        "lung_vs_muscle_go_bp",
        &repo_root.join("data/deseq2_positive_ranks/lung_vs_muscle.rnk"),
        &repo_root.join("data/GO_Biological_Process_2025.gmt"),
    ) {
        workloads.push(workload);
    }
    workloads
}

fn benchmark_blitz(c: &mut Criterion) {
    if !blitz_bench_enabled() {
        return;
    }

    let workloads = blitz_benchmark_workloads();
    let options = blitz_options_for_bench();

    {
        let mut group = configure_blitz_group(c, "blitz_full_cold");
        for workload in &workloads {
            group.throughput(Throughput::Elements(workload.pathways.len() as u64));
            group.bench_with_input(
                BenchmarkId::from_parameter(workload.name),
                workload,
                |b, workload| {
                    b.iter(|| {
                        fgsea_blitz_with_options(
                            black_box(&workload.ranks),
                            black_box(&workload.pathways),
                            black_box(&options),
                        )
                    })
                },
            );
        }
        group.finish();
    }

    {
        let mut group = configure_blitz_group(c, "blitz_anchor_calibration_only");
        for workload in &workloads {
            group.throughput(Throughput::Elements(workload.pathways.len() as u64));
            group.bench_with_input(
                BenchmarkId::from_parameter(workload.name),
                workload,
                |b, workload| {
                    b.iter(|| {
                        __bench_blitz_anchor_calibration(
                            black_box(&workload.ranks),
                            black_box(&workload.pathways),
                            black_box(&options),
                        )
                    })
                },
            );
        }
        group.finish();
    }

    {
        let mut group = configure_blitz_group(c, "blitz_final_scoring_only");
        for workload in &workloads {
            let prepared = __bench_blitz_prepare_scoring(
                black_box(&workload.ranks),
                black_box(&workload.pathways),
                black_box(&options),
            )
            .expect("prepare blitz scoring benchmark");
            group.throughput(Throughput::Elements(workload.pathways.len() as u64));
            group.bench_with_input(
                BenchmarkId::from_parameter(workload.name),
                &prepared,
                |b, prepared| b.iter(|| __bench_blitz_score_prepared(black_box(prepared))),
            );
        }
        group.finish();
    }

    {
        let mut group = configure_blitz_group(c, "blitz_gamma_tail_microcases");
        group.bench_function("scipy_plus_mpmath_thresholds", |b| {
            b.iter(|| __bench_blitz_tail_microcases(black_box(options.deep_accuracy)))
        });
        group.finish();
    }
}

fn benchmark_decor_matrix(c: &mut Criterion) {
    let matrix =
        synthetic_decor_matrix_workload("5k_genes_500_pathways_32_samples", 5_000, 500, 32);
    let n_perm = 200usize;

    {
        let mut group = configure_decor_matrix_group(c, "decor_matrix_formulas_profile");
        for formula in [
            DecorWeightFormula::RawRational,
            DecorWeightFormula::ScaledRational,
            DecorWeightFormula::ExpScaled,
            DecorWeightFormula::OddsRational,
            DecorWeightFormula::FloorScaledRational,
            DecorWeightFormula::PowerRetention,
        ] {
            let options = decor_options_for_matrix(&matrix, formula);
            group.throughput(Throughput::Elements(matrix.workload.pathways.len() as u64));
            group.bench_with_input(
                BenchmarkId::from_parameter(formula.to_string()),
                &options,
                |b, options| {
                    b.iter(|| {
                        fgsea_decor_simple_with_options(
                            black_box(&matrix.workload.ranks),
                            black_box(&matrix.workload.pathways),
                            black_box(&matrix.cache),
                            black_box(options),
                            black_box(n_perm),
                            black_box(42),
                            black_box(15),
                            black_box(500),
                            black_box(1e-10),
                            black_box(ScoreType::Std),
                            black_box(1.0),
                            black_box(101),
                        )
                    })
                },
            );
        }
        group.finish();
    }
}

fn benchmark_end_to_end(c: &mut Criterion) {
    let workloads = benchmark_workloads();
    let decor_caches = workloads
        .iter()
        .map(synthetic_decor_cache)
        .collect::<Vec<_>>();

    {
        let mut group = configure_representative_group(c, "representative_multilevel");
        for workload in &workloads {
            let permutations = multilevel_permutations(workload.name);
            group.throughput(Throughput::Elements(workload.pathways.len() as u64));
            group.bench_with_input(
                BenchmarkId::from_parameter(workload.name),
                workload,
                |b, workload| {
                    b.iter(|| {
                        fgsea_multilevel_with_sample_size(
                            black_box(&workload.ranks),
                            black_box(&workload.pathways),
                            black_box(permutations),
                            black_box(42),
                            black_box(15),
                            black_box(500),
                            black_box(1e-10),
                            black_box(ScoreType::Std),
                            black_box(1.0),
                            black_box(101),
                        )
                    })
                },
            );
        }
        group.finish();
    }

    {
        let mut group = configure_representative_group(c, "representative_simple");
        for workload in &workloads {
            let permutations = simple_permutations(workload.name);
            group.throughput(Throughput::Elements(workload.pathways.len() as u64));
            group.bench_with_input(
                BenchmarkId::from_parameter(workload.name),
                workload,
                |b, workload| {
                    b.iter(|| {
                        fgsea_simple_with_sample_size(
                            black_box(&workload.ranks),
                            black_box(&workload.pathways),
                            black_box(permutations),
                            black_box(42),
                            black_box(15),
                            black_box(500),
                            black_box(1e-10),
                            black_box(ScoreType::Std),
                            black_box(1.0),
                            black_box(101),
                        )
                    })
                },
            );
        }
        group.finish();
    }

    {
        let mut group = configure_representative_group(c, "representative_decor_simple_reuse");
        for (workload, cache) in workloads.iter().zip(decor_caches.iter()) {
            let permutations = decor_simple_permutations(workload.name);
            group.throughput(Throughput::Elements(workload.pathways.len() as u64));
            group.bench_with_input(
                BenchmarkId::from_parameter(workload.name),
                &(workload, cache),
                |b, (workload, cache)| {
                    b.iter(|| {
                        fgsea_decor_simple_with_sample_size(
                            black_box(&workload.ranks),
                            black_box(&workload.pathways),
                            black_box(cache),
                            black_box(23.0),
                            black_box(permutations),
                            black_box(42),
                            black_box(15),
                            black_box(500),
                            black_box(1e-10),
                            black_box(ScoreType::Std),
                            black_box(1.0),
                            black_box(101),
                        )
                    })
                },
            );
        }
        group.finish();
    }

    if wrapper_dispatch_enabled() {
        let mut group = configure_representative_group(c, "representative_wrapper_dispatch");
        for workload in &workloads {
            let permutations = multilevel_permutations(workload.name);
            group.throughput(Throughput::Elements(workload.pathways.len() as u64));
            group.bench_with_input(
                BenchmarkId::from_parameter(workload.name),
                workload,
                |b, workload| {
                    b.iter(|| {
                        fgsea_with_sample_size(
                            black_box(&workload.ranks),
                            black_box(&workload.pathways),
                            black_box(None),
                            black_box(permutations),
                            black_box(42),
                            black_box(15),
                            black_box(500),
                            black_box(1e-10),
                            black_box(ScoreType::Std),
                            black_box(1.0),
                            black_box(101),
                        )
                    })
                },
            );
        }
        group.finish();
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(15)
        .measurement_time(Duration::from_secs(10))
        .warm_up_time(Duration::from_secs(1));
    targets = benchmark_es, benchmark_end_to_end, benchmark_decor_matrix, benchmark_blitz
}
criterion_main!(benches);
