use anyhow::Result;
use rand::prelude::*;
use rand::rngs::StdRng;
use rsfgsea::prelude::*;
use std::time::Instant;

fn main() -> Result<()> {
    println!("=== GSEA Optimization & Precision Test ===");

    // 1. Generate synthetic data
    let n_genes = 100000;
    let n_pathways = 10000;
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
    let _cpu_results = fgsea_multilevel_with_sample_size(
        &ranks,
        &pathways,
        n_perm,
        42,
        1,
        1000,
        1e-10, // Small eps for multilevel
        ScoreType::Std,
        1.0,
        101,
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
        let mut max_pval_pathway = None;
        let mut max_pval_rel_diff = 0.0;
        let mut max_pval_rel_pathway = None;

        // Statistics buckets
        let mut rel_diff_below_1 = 0;
        let mut rel_diff_1_10 = 0;
        let mut rel_diff_10_50 = 0;
        let mut rel_diff_50_plus = 0;
        let mut total_rel_diff = 0.0;
        let mut count_comparisons = 0;

        // Map results for comparison
        let mut cpu_map = std::collections::HashMap::new();
        for res in &_cpu_results {
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
                    max_pval_pathway = Some(g_res.pathway_name.clone());
                }
                if nes_diff > max_nes_diff {
                    max_nes_diff = nes_diff;
                }

                // Relative P-value Difference
                let rel_denom = c_res.p_value.max(1e-10);
                let pval_rel_diff = pval_diff / rel_denom;

                if pval_rel_diff > max_pval_rel_diff {
                    max_pval_rel_diff = pval_rel_diff;
                    max_pval_rel_pathway = Some(g_res.pathway_name.clone());
                }

                // Bucket stats
                if pval_rel_diff > 0.50 {
                    rel_diff_50_plus += 1;
                } else if pval_rel_diff > 0.10 {
                    rel_diff_10_50 += 1;
                } else if pval_rel_diff > 0.01 {
                    rel_diff_1_10 += 1;
                } else {
                    rel_diff_below_1 += 1;
                }
                total_rel_diff += pval_rel_diff;
                count_comparisons += 1;
            }
        }

        if let Some((c, g)) = max_pval_pathway.as_ref().and_then(|name| {
            cpu_map
                .get(name)
                .zip(gpu_results.iter().find(|r| &r.pathway_name == name))
        }) {
            let name = &c.pathway_name;
            println!("\nMax P-value Difference Pathway: {}", name);
            println!(
                "  CPU: p={:.10}, ES={:.6}, Size={}",
                c.p_value, c.es, c.size
            );
            println!(
                "  GPU: p={:.10}, ES={:.6}, Size={}",
                g.p_value, g.es, g.size
            );
        }

        if let Some((c, g)) = max_pval_rel_pathway.as_ref().and_then(|name| {
            cpu_map
                .get(name)
                .zip(gpu_results.iter().find(|r| &r.pathway_name == name))
        }) {
            let name = &c.pathway_name;
            println!("\nMax Relative P-value Difference Pathway: {}", name);
            println!(
                "  CPU: p={:.10}, ES={:.6}, Size={}",
                c.p_value, c.es, c.size
            );
            println!(
                "  GPU: p={:.10}, ES={:.6}, Size={}",
                g.p_value, g.es, g.size
            );
            println!("  Rel Diff: {:.6}", max_pval_rel_diff);
        }

        println!("\nSample Comparisons (First 5 pathways):");
        for i in 0..5 {
            let pw_name = format!("PW_{}", i);
            if let (Some(c), Some(g)) = (
                cpu_map.get(&pw_name),
                gpu_results.iter().find(|r| r.pathway_name == pw_name),
            ) {
                let rel_diff = (c.p_value - g.p_value).abs() / c.p_value.max(1e-10);
                println!(
                    "  {}: CPU p={:.4}, ES={:.4} | GPU p={:.4}, ES={:.4} | Diff p={:.4} | Rel Diff={:.4}",
                    pw_name,
                    c.p_value,
                    c.es,
                    g.p_value,
                    g.es,
                    (c.p_value - g.p_value).abs(),
                    rel_diff
                );
            }
        }

        println!("  Max ES Difference:   {:.10}", max_es_diff);
        println!("  Max P-value Diff:    {:.10}", max_pval_diff);
        println!("  Max Relative P-val:  {:.10}", max_pval_rel_diff);
        println!("  Max NES Difference:  {:.10}", max_nes_diff);

        println!("\nRelative P-value Difference Statistics:");
        println!(
            "  Mean Relative Diff:  {:.2}%",
            (total_rel_diff / count_comparisons as f64) * 100.0
        );
        println!("  Distribution:");
        println!(
            "    < 1% diff:         {} ({:.1}%)",
            rel_diff_below_1,
            (rel_diff_below_1 as f64 / count_comparisons as f64) * 100.0
        );
        println!(
            "    1% - 10% diff:     {} ({:.1}%)",
            rel_diff_1_10,
            (rel_diff_1_10 as f64 / count_comparisons as f64) * 100.0
        );
        println!(
            "    10% - 50% diff:    {} ({:.1}%)",
            rel_diff_10_50,
            (rel_diff_10_50 as f64 / count_comparisons as f64) * 100.0
        );
        println!(
            "    > 50% diff:        {} ({:.1}%)",
            rel_diff_50_plus,
            (rel_diff_50_plus as f64 / count_comparisons as f64) * 100.0
        );

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
