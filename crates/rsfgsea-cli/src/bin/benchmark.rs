use anyhow::Result;
use csv::ReaderBuilder;
use flate2::read::MultiGzDecoder;
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rsfgsea::prelude::*;
use rsfgsea_gpu::GpuEngine;
use std::fs::File;
use std::time::Instant;

fn calculate_spearman(v1: &[f64], v2_ranks: &[f64]) -> f64 {
    let v1_ranks = get_ranks(v1);
    pearson_correlation(&v1_ranks, v2_ranks)
}

fn get_ranks(v: &[f64]) -> Vec<f64> {
    let n = v.len();
    let mut indexed: Vec<(usize, f64)> = v.iter().enumerate().map(|(i, &x)| (i, x)).collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && indexed[j].1 == indexed[i].1 {
            j += 1;
        }
        let avg_rank = (i + j - 1) as f64 / 2.0 + 1.0;
        for k in i..j {
            ranks[indexed[k].0] = avg_rank;
        }
        i = j;
    }
    ranks
}

fn pearson_correlation(v1: &[f64], v2: &[f64]) -> f64 {
    let n = v1.len() as f64;
    let m1 = v1.iter().sum::<f64>() / n;
    let m2 = v2.iter().sum::<f64>() / n;
    let mut num = 0.0;
    let mut d1 = 0.0;
    let mut d2 = 0.0;
    for i in 0..v1.len() {
        let x = v1[i] - m1;
        let y = v2[i] - m2;
        num += x * y;
        d1 += x * x;
        d2 += y * y;
    }
    if d1 == 0.0 || d2 == 0.0 {
        return 0.0;
    }
    num / (d1.sqrt() * d2.sqrt())
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== GSEA Real Data Benchmark ===");

    let path = "../min-rust-corr/data/Muscle - Skeletal_normalized_counts.tsv.gz";
    println!("Loading data from: {}", path);
    let file = File::open(path)?;
    let decoder = MultiGzDecoder::new(file);
    let mut rdr = ReaderBuilder::new()
        .delimiter(b'\t')
        .has_headers(true)
        .from_reader(decoder);

    let headers = rdr.headers()?.clone();
    let n_samples = headers.len() - 1;

    let mut gene_names = Vec::new();
    let mut gene_data = Vec::new();
    let mut col_sums = vec![0.0; n_samples];

    for result in rdr.records() {
        let record = result?;
        let gene_id = record[0].to_string();
        let mut values = Vec::with_capacity(n_samples);
        for (i, s) in record.iter().skip(1).enumerate() {
            let parsed = s.parse::<f64>().ok().unwrap_or(0.0);
            values.push(parsed);
            col_sums[i] += parsed;
        }
        gene_names.push(gene_id);
        gene_data.push(values);
    }
    println!("Loaded {} genes.", gene_names.len());

    println!("Ranking genes by absolute Spearman correlation to sample depth...");
    let phenotype_ranks = get_ranks(&col_sums);
    let mut correlations: Vec<(String, f64)> = gene_names
        .into_iter()
        .zip(gene_data.iter())
        .map(|(name, data)| (name, calculate_spearman(data, &phenotype_ranks)))
        .collect();

    correlations.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
    let top_500_genes: Vec<String> = correlations
        .iter()
        .take(500)
        .map(|(n, _)| n.clone())
        .collect();

    let (names, scores): (Vec<String>, Vec<f64>) = correlations.into_iter().unzip();
    let ranks = RankedList::new(names, scores);

    let mut rng = StdRng::seed_from_u64(42);
    let mut pathway_genes = top_500_genes.clone();
    pathway_genes.shuffle(&mut rng);
    pathway_genes.truncate(100);

    let pathway = Pathway {
        name: "TOP_CORRELATED".to_string(),
        description: None,
        genes: pathway_genes,
    };

    // Increased permutations for better resolution
    let n_perm = 1_000_000;
    println!("\nBenchmark Configuration:");
    println!("  Permutations: {}", n_perm);
    println!("  Pathway: 100 genes (sampled from top 500)");

    println!("\n[1/2] Computing GSEA on CPU (Rayon)...");
    let start_cpu = Instant::now();
    let _cpu_res = run_gsea(
        &ranks,
        std::slice::from_ref(&pathway),
        n_perm,
        42,
        1,
        5000,
        1.0,
        ScoreType::Std,
        1.0,
    );
    let cpu_duration = start_cpu.elapsed();
    println!("CPU Time: {:?}", cpu_duration);

    println!("\n[2/2] Computing GSEA on GPU...");
    let engine = GpuEngine::new().await?;
    let (abs_weights, _, _) = ranks.prepare(1.0);
    let abs_weights_f32: Vec<f32> = abs_weights.iter().map(|&w| w as f32).collect();
    let scores_buffer = engine.upload_scores(&abs_weights_f32);
    let gene_to_idx: std::collections::HashMap<String, usize> = ranks
        .genes
        .iter()
        .enumerate()
        .map(|(i, g)| (g.clone(), i))
        .collect();
    let hits: Vec<usize> = pathway
        .genes
        .iter()
        .filter_map(|g| gene_to_idx.get(g).copied())
        .collect();

    let start_gpu = Instant::now();
    let _gpu_res = engine.fgsea_simple_pathway_with_buffer(
        &scores_buffer,
        &hits,
        &abs_weights_f32,
        n_perm,
        42,
        0,
    )?;
    let gpu_duration = start_gpu.elapsed();
    println!("GPU Simple Time: {:?}", gpu_duration);

    println!("\n[3/3] Computing Multilevel GSEA on GPU...");
    let n_perm_ml = 1000;
    let start_ml = Instant::now();
    let ml_res = engine.fgsea_multilevel_pathway(&hits, &abs_weights_f32, n_perm_ml, 42, 0)?;
    let ml_duration = start_ml.elapsed();
    println!("GPU Multilevel Time: {:?}", ml_duration);
    println!(
        "  P-value: {:.4e}, Log2err: {:.4}",
        ml_res.p_value, ml_res.log2err
    );

    println!("\nSummary:");
    println!("  CPU Simple: {}ms", cpu_duration.as_millis());
    println!("  GPU Simple: {}ms", gpu_duration.as_millis());
    println!("  GPU Multilevel: {}ms", ml_duration.as_millis());
    println!(
        "  Simple Speedup: {:.2}x",
        cpu_duration.as_secs_f64() / gpu_duration.as_secs_f64()
    );

    Ok(())
}
