use anyhow::Result;
use clap::Parser;
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::seq::index::sample;
use rsfgsea::prelude::*;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(author, version, about = "Synthetic CPU vs GPU simple-mode benchmark")]
struct Args {
    #[arg(long, default_value_t = 50_000)]
    genes: usize,
    #[arg(long, default_value_t = 10_000)]
    pathways: usize,
    #[arg(long = "min-size", default_value_t = 50)]
    min_size: usize,
    #[arg(long = "max-size", default_value_t = 500)]
    max_size: usize,
    #[arg(long = "nperm", default_value_t = 300_000)]
    n_perm: usize,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, default_value_t = false)]
    skip_cpu: bool,
    #[arg(long)]
    cpu_baseline_ms: Option<f64>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    assert!(args.min_size > 0, "min-size must be > 0");
    assert!(
        args.max_size <= args.genes,
        "max-size must be <= number of genes"
    );
    assert!(
        args.min_size <= args.max_size,
        "min-size must be <= max-size"
    );

    println!("=== Synthetic CPU vs GPU (simple mode) ===");
    println!(
        "Configuration: genes={}, pathways={}, sizes=[{}, {}], nperm={}, seed={}",
        args.genes, args.pathways, args.min_size, args.max_size, args.n_perm, args.seed
    );

    let mut rng = StdRng::seed_from_u64(args.seed);

    let genes: Vec<String> = (0..args.genes).map(|i| format!("GENE_{i}")).collect();
    let scores: Vec<f64> = (0..args.genes).map(|_| rng.gen_range(-5.0..5.0)).collect();
    let ranks = RankedList::new(genes.clone(), scores);

    let mut pathways = Vec::with_capacity(args.pathways);
    for i in 0..args.pathways {
        let size = rng.gen_range(args.min_size..=args.max_size);
        let idxs = sample(&mut rng, args.genes, size);
        let pw_genes: Vec<String> = idxs.into_iter().map(|idx| genes[idx].clone()).collect();
        pathways.push(Pathway {
            name: format!("PW_{i}"),
            description: None,
            genes: pw_genes,
        });
    }

    let mut cpu_ms: Option<f64> = args.cpu_baseline_ms;
    if !args.skip_cpu {
        println!("\n[1/2] CPU simple benchmark...");
        let cpu_start = Instant::now();
        let cpu_results = fgsea_simple_with_sample_size(
            &ranks,
            &pathways,
            args.n_perm,
            args.seed,
            1,
            args.genes.saturating_sub(1),
            1e-50,
            ScoreType::Std,
            1.0,
            101,
        );
        let cpu_elapsed = cpu_start.elapsed();
        let measured_cpu_ms = cpu_elapsed.as_secs_f64() * 1000.0;
        cpu_ms = Some(measured_cpu_ms);
        println!(
            "CPU simple: {} ms ({} pathways)",
            cpu_elapsed.as_millis(),
            cpu_results.len()
        );
    } else {
        println!("\n[1/2] CPU simple benchmark skipped (--skip-cpu).");
        if let Some(ms) = cpu_ms {
            println!("Using provided CPU baseline: {:.0} ms", ms);
        }
    }

    #[cfg(feature = "gpu")]
    {
        println!("\n[2/2] GPU benchmark...");
        let gpu_start = Instant::now();
        let gpu_results = rsfgsea::algo::run_gsea_gpu(
            &ranks,
            &pathways,
            args.n_perm,
            args.seed,
            1,
            args.genes.saturating_sub(1),
            ScoreType::Std,
            1.0,
        )?;
        let gpu_elapsed = gpu_start.elapsed();
        println!(
            "GPU simple+multilevel path: {} ms ({} pathways)",
            gpu_elapsed.as_millis(),
            gpu_results.len()
        );
        if let Some(ms) = cpu_ms {
            println!(
                "Speedup (CPU/GPU): {:.2}x",
                (ms / 1000.0) / gpu_elapsed.as_secs_f64()
            );
        } else {
            println!("Speedup (CPU/GPU): N/A (no CPU baseline available)");
        }
    }

    #[cfg(not(feature = "gpu"))]
    {
        println!("GPU feature not enabled; rebuild with --features gpu.");
    }

    Ok(())
}
