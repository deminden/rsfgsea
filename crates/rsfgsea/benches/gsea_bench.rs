use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rsfgsea::prelude::*;
use std::hint::black_box;
use std::time::Duration;

struct SyntheticWorkload {
    name: &'static str,
    ranks: RankedList,
    pathways: Vec<Pathway>,
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
        .map(|i| 1.0 / (1.0 + 0.5 * (i as f64 / k as f64)))
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

fn multilevel_permutations(_workload_name: &str) -> usize {
    1000
}

fn wrapper_dispatch_enabled() -> bool {
    std::env::var_os("RSFGSEA_WRAPPER_BENCH").is_some()
}

fn configure_representative_group<'a>(
    c: &'a mut Criterion,
    name: &str,
) -> criterion::BenchmarkGroup<'a, criterion::measurement::WallTime> {
    let mut group = c.benchmark_group(name);
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(5));
    group.warm_up_time(Duration::from_millis(500));
    group
}

fn benchmark_end_to_end(c: &mut Criterion) {
    let workloads = benchmark_workloads();

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
        .sample_size(10)
        .measurement_time(Duration::from_secs(5))
        .warm_up_time(Duration::from_millis(500));
    targets = benchmark_es, benchmark_end_to_end
}
criterion_main!(benches);
