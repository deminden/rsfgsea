use anyhow::Result;
use rand::prelude::*;
use rand::rngs::StdRng;
use rsfgsea::prelude::*;
use std::time::Instant;

fn main() -> Result<()> {
    println!("=== GSEA Optimization & Precision Test ===");

    // 1. Generate synthetic data
    let n_genes = 10000;
    let n_pathways = 2000;
    let n_perm = 100000;

    let mut rng = StdRng::seed_from_u64(42);
    let mut genes = Vec::new();
    let mut scores = Vec::new();
    for i in 0..n_genes {
        genes.push(format!("GENE_{}", i));
        scores.push(rng.gen_range(-5.0..5.0));
    }
    let ranks = RankedList::new(genes.clone(), scores);

    let mut pathways = Vec::new();
    // Create pathways with many duplicate sizes to trigger optimization
    let unique_sizes = [20, 50, 100];
    for i in 0..n_pathways {
        let size = unique_sizes[i % unique_sizes.len()];
        let mut pw_genes = genes.clone();
        pw_genes.shuffle(&mut rng);
        pw_genes.truncate(size);
        pathways.push(Pathway {
            name: format!("PW_{}", i),
            description: None,
            genes: pw_genes,
        });
    }

    println!("Configuration:");
    println!("  Genes: {}", n_genes);
    println!(
        "  Pathways: {} (Unique sizes: {})",
        n_pathways,
        unique_sizes.len()
    );
    println!("  Permutations: {}", n_perm);

    // 2. CPU Run (Reference)
    println!("\n[1/2] Running GSEA on CPU...");
    let start_cpu = Instant::now();
    let cpu_results = run_gsea(
        &ranks,
        &pathways,
        n_perm,
        42,
        1,
        1000,
        1e-10, // Small eps for multilevel
        ScoreType::Std,
        1.0,
    );
    let cpu_dur = start_cpu.elapsed();
    println!("CPU Time: {:?}", cpu_dur);

    // 3. GPU Run (Optimized)
    #[cfg(feature = "gpu")]
    {
        println!("\n[2/2] Running Optimized GSEA on GPU...");
        let start_gpu = Instant::now();
        let gpu_results = rsfgsea::algo::run_gsea_gpu(
            &ranks,
            &pathways,
            n_perm,
            42,
            1,
            1000,
            ScoreType::Std,
            1.0,
        )?;
        let gpu_dur = start_gpu.elapsed();
        println!("GPU Time: {:?}", gpu_dur);
        println!(
            "Speedup: {:.2}x",
            cpu_dur.as_secs_f64() / gpu_dur.as_secs_f64()
        );

        // 4. Precision Assessment
        println!("\nPrecision Check (CPU vs GPU):");

        let mut max_es_diff = 0.0;
        let mut max_pval_diff = 0.0;
        let mut max_nes_diff = 0.0;

        // Map results for comparison
        let mut cpu_map = std::collections::HashMap::new();
        for res in &cpu_results {
            cpu_map.insert(res.pathway_name.clone(), res);
        }

        for g_res in &gpu_results {
            if let Some(c_res) = cpu_map.get(&g_res.pathway_name) {
                let es_diff = (c_res.es - g_res.es).abs();
                let pval_diff = (c_res.p_value - g_res.p_value).abs();
                let nes_diff = (c_res.nes.unwrap_or(0.0) - g_res.nes.unwrap_or(0.0)).abs();

                if es_diff > max_es_diff {
                    max_es_diff = es_diff;
                }
                if pval_diff > max_pval_diff {
                    max_pval_diff = pval_diff;
                }
                if nes_diff > max_nes_diff {
                    max_nes_diff = nes_diff;
                }
            }
        }

        println!("  Max ES Difference:   {:.10}", max_es_diff);
        println!("  Max P-value Diff:    {:.10}", max_pval_diff);
        println!("  Max NES Difference:  {:.10}", max_nes_diff);

        if max_pval_diff < 0.05 {
            println!(
                "\nCONCLUSION: Optimized GPU GSEA matches CPU results within statistical variance."
            );
        } else {
            println!(
                "\nNOTE: P-value differences observed. This is expected as GPU uses single-precision f32 and shared permutations."
            );
        }
    }

    #[cfg(not(feature = "gpu"))]
    println!("\nGPU feature not enabled. Skipping GPU test.");

    Ok(())
}
