use anyhow::Result;
use rsfgsea::prelude::*;
use rsfgsea_gpu::GpuEngine;
use rand::prelude::*;
use rand::rngs::StdRng;

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== GSEA Precision Verification (CPU f64 vs GPU f32) ===");
    
    let n_total = 10000;
    let k = 100;
    let mut rng = StdRng::seed_from_u64(42);
    
    let mut weights = vec![0.0f32; n_total];
    for w in weights.iter_mut() {
        *w = rng.gen_range(0.0..10.0);
    }
    
    let mut hits = (0..n_total).collect::<Vec<usize>>();
    hits.shuffle(&mut rng);
    hits.truncate(k);
    hits.sort_unstable();
    
    let engine = GpuEngine::new().await?;
    
    // 1. Calculate ES on CPU with f64
    let weights_f64: Vec<f64> = weights.iter().map(|&w| w as f64).collect();
    let (cpu_es, _) = rsfgsea::algo::calculate_es(&hits, &weights_f64, n_total, ScoreType::Std);
    
    // 2. Calculate ES on GPU with f32
    let hits_u32: Vec<u32> = hits.iter().map(|&h| h as u32).collect();
    let batch_results = engine.compute_es_batch(&weights, &hits_u32, k as u32, n_total as u32, 1, 0)?;
    let gpu_es = batch_results[0].es;
    
    println!("Results for single pathway:");
    println!("  CPU ES (f64): {:.10}", cpu_es);
    println!("  GPU ES (f32): {:.10}", gpu_es);
    println!("  Difference:   {:.10}", (cpu_es - gpu_es as f64).abs());
    
    // 3. Batch Verification
    println!("\nVerifying 1000 random permutations...");
    let n_batch = 1000;
    let mut all_subsets = Vec::new();
    let mut pool: Vec<usize> = (0..n_total).collect();
    for _ in 0..n_batch {
        pool.shuffle(&mut rng);
        let mut subset = pool[..k].to_vec();
        subset.sort_unstable();
        for &h in &subset { all_subsets.push(h as u32); }
    }
    
    let gpu_results = engine.compute_es_batch(&weights, &all_subsets, k as u32, n_total as u32, n_batch as u32, 0)?;
    
    let mut max_diff = 0.0f64;
    let mut sum_diff = 0.0f64;
    
    for (i, gpu_res) in gpu_results.iter().enumerate() {
        let start = i * k;
        let subset_usize: Vec<usize> = all_subsets[start..start+k].iter().map(|&h| h as usize).collect();
        let (cpu_es_batch, _) = rsfgsea::algo::calculate_es(&subset_usize, &weights_f64, n_total, ScoreType::Std);
        let gpu_es_batch = gpu_res.es as f64;
        
        let diff = (cpu_es_batch - gpu_es_batch).abs();
        if diff > max_diff { max_diff = diff; }
        sum_diff += diff;
    }
    
    println!("Batch Verification Summary:");
    println!("  Max Difference: {:.10}", max_diff);
    println!("  Avg Difference: {:.10}", sum_diff / n_batch as f64);
    
    if max_diff < 1e-5 {
        println!("\nCONCLUSION: GPU f32 precision is SUFFICIENT for GSEA rankings.");
    } else {
        println!("\nCONCLUSION: GPU f32 precision might show minor deviations from double precision.");
    }
    
    Ok(())
}
