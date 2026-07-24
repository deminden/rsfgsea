#![cfg(feature = "gpu")]

use rsfgsea::GpuEngine;
use std::fs;
use std::process::Command;

fn gl_backend_in_libtest() -> bool {
    let requested_gl = std::env::var("WGPU_BACKEND")
        .map(|value| value.eq_ignore_ascii_case("gl"))
        .unwrap_or(false);
    let gallium_d3d12 = std::env::var("GALLIUM_DRIVER")
        .map(|value| value.eq_ignore_ascii_case("d3d12"))
        .unwrap_or(false);
    requested_gl || gallium_d3d12
}

macro_rules! skip_if_no_gpu {
    ($engine:expr) => {{
        if gl_backend_in_libtest() {
            println!(
                "Skipping test: wgpu GL/D3D12 backend can crash during libtest thread teardown"
            );
            return;
        }
        match $engine {
            Ok(e) => e,
            Err(e) => {
                println!("Skipping test: {}", e);
                return;
            }
        }
    }};
}

/// Test basic GPU engine initialization
#[test]
fn test_gpu_engine_init() {
    if gl_backend_in_libtest() {
        println!("Skipping test: wgpu GL/D3D12 backend can crash during libtest thread teardown");
        return;
    }

    pollster::block_on(async {
        let engine_res = GpuEngine::new().await;
        if let Err(e) = engine_res {
            println!("Skipping test: {}", e);
            return;
        }
        let engine = engine_res.unwrap();
        // Just verify it exists
        drop(engine);
    });
}

#[test]
fn test_gpu_cli_smoke_for_gl_d3d12_backend() {
    if !gl_backend_in_libtest() {
        println!("Skipping test: GL/D3D12 backend was not requested");
        return;
    }

    let dir = tempfile::tempdir().unwrap();
    let ranks = dir.path().join("test.rnk");
    let gmt = dir.path().join("test.gmt");
    let output = dir.path().join("out.tsv");
    fs::write(
        &ranks,
        "A\t4.0\nB\t3.0\nC\t2.0\nD\t-1.0\nE\t-2.0\nF\t-3.0\n",
    )
    .unwrap();
    fs::write(&gmt, "PW_REDUNDANT\tna\tA\tB\tC\nPW_MIXED\tna\tA\tD\tF\n").unwrap();

    let status = Command::new(env!("CARGO_BIN_EXE_rsfgsea"))
        .args([
            "--gpu",
            "--mode",
            "fgsea",
            "--nperm",
            "64",
            "--seed",
            "42",
            "--minSize",
            "1",
            "--maxSize",
            "6",
            "--ranks",
            ranks.to_str().unwrap(),
            "--gmt",
            gmt.to_str().unwrap(),
            "--output",
            output.to_str().unwrap(),
        ])
        .status()
        .expect("failed to run rsfgsea CLI");
    assert!(status.success(), "rsfgsea GPU CLI smoke test failed");

    let content = fs::read_to_string(output).unwrap();
    assert!(content.contains("PW_REDUNDANT"));
}

/// Test compute_es_batch with a simple case
#[tokio::test]
async fn test_compute_es_batch_simple() {
    let engine = skip_if_no_gpu!(GpuEngine::new().await);

    // Create simple test data: 100 genes, pathway of size 10
    let n_total = 100u32;
    let k = 10u32;

    // All weights = 1.0 for simplicity
    let abs_scores: Vec<f32> = vec![1.0; n_total as usize];

    // Single permutation: indices 0..10
    let indices: Vec<u32> = (0..k).collect();

    let results = engine
        .compute_es_batch(&abs_scores, &indices, k, n_total, 1, 0)
        .expect("GPU compute failed");

    assert_eq!(results.len(), 1);
    println!("ES: {}, peak_idx: {}", results[0].es, results[0].peak_idx);
}

/// Test compute_es_batch with multiple permutations
#[tokio::test]
async fn test_compute_es_batch_multiple() {
    let engine = skip_if_no_gpu!(GpuEngine::new().await);

    let n_total = 100u32;
    let k = 10u32;
    let n_perm = 100usize;

    // Decreasing weights (like a typical ranked list)
    let abs_scores: Vec<f32> = (0..n_total).map(|i| (n_total - i) as f32).collect();

    // Generate random permutations
    use rand::prelude::*;
    let mut rng = rand::rng();
    let mut subsets_indices = Vec::with_capacity(n_perm * k as usize);

    for _ in 0..n_perm {
        let mut pool: Vec<usize> = (0..n_total as usize).collect();
        pool.shuffle(&mut rng);
        let mut subset = pool[..k as usize].to_vec();
        subset.sort_unstable();
        for item in subset.iter().take(k as usize) {
            subsets_indices.push(*item as u32);
        }
    }

    let results = engine
        .compute_es_batch(&abs_scores, &subsets_indices, k, n_total, n_perm as u32, 0)
        .expect("GPU compute failed");

    assert_eq!(results.len(), n_perm);
    println!("First 10 ES values:");
    for (i, r) in results.iter().take(10).enumerate() {
        println!("  Perm {}: ES = {:.4}", i, r.es);
    }
}

/// Test fgsea_simple_pathway with known pathway
#[tokio::test]
async fn test_fgsea_simple_pathway() {
    let engine = skip_if_no_gpu!(GpuEngine::new().await);

    let n_total = 1000;
    let k = 50;

    // Create ranked scores: decreasing from top
    let abs_scores: Vec<f32> = (0..n_total).map(|i| (n_total - i) as f32).collect();

    // Pathway enriched at the top: indices 0..50
    let pathway_top: Vec<usize> = (0..k).collect();

    let result = engine
        .fgsea_simple_pathway(&pathway_top, &abs_scores, 2000, 42, 1)
        .expect("fgsea_simple failed");

    println!("Top-enriched pathway:");
    println!("  ES: {:.4}", result.es);
    println!("  NES: {:?}", result.nes);
    println!("  P-value: {:.4}", result.p_value);

    // Should be significantly enriched (positive ES, low p-value)
    assert!(result.es > 0.0, "ES should be positive for top-enriched");
    assert!(result.p_value < 0.05, "Should be significant");
}

/// Test fgsea_simple_pathway with random pathway (should not be significant)
#[tokio::test]
async fn test_fgsea_simple_pathway_random() {
    let engine = skip_if_no_gpu!(GpuEngine::new().await);

    let n_total = 1000;
    let k = 50;

    // Create ranked scores
    let abs_scores: Vec<f32> = (0..n_total)
        .map(|i| ((n_total - i) as f32).powi(2))
        .collect();

    // Random pathway: evenly distributed
    use rand::prelude::*;
    let mut rng = rand::rng();
    let mut pathway_random: Vec<usize> = (0..n_total).collect();
    pathway_random.shuffle(&mut rng);
    pathway_random.truncate(k);
    pathway_random.sort_unstable();

    let result = engine
        .fgsea_simple_pathway(&pathway_random, &abs_scores, 1000, 42, 0)
        .expect("fgsea_simple failed");

    println!("Random pathway:");
    println!("  ES: {:.4}", result.es);
    println!("  NES: {:?}", result.nes);
    println!("  P-value: {:.4}", result.p_value);

    // ES can be positive or negative but should be small
    assert!(
        result.es.abs() < 0.5,
        "ES should be small for random pathway"
    );
}

/// Test fgsea_simple_pathway with bottom-enriched pathway
#[tokio::test]
async fn test_fgsea_simple_pathway_bottom() {
    let engine = skip_if_no_gpu!(GpuEngine::new().await);

    let n_total = 1000;
    let k = 50;

    // Create ranked scores
    let abs_scores: Vec<f32> = (0..n_total).map(|i| (n_total - i) as f32).collect();

    // Pathway enriched at the bottom: indices (n_total-k)..n_total
    let pathway_bottom: Vec<usize> = (n_total - k..n_total).collect();

    let result = engine
        .fgsea_simple_pathway(&pathway_bottom, &abs_scores, 2000, 42, 2)
        .expect("fgsea_simple failed");

    println!("Bottom-enriched pathway:");
    println!("  ES: {:.4}", result.es);
    println!("  NES: {:?}", result.nes);
    println!("  P-value: {:.4}", result.p_value);

    // Should be significantly enriched (negative ES, low p-value)
    assert!(result.es < 0.0, "ES should be negative for bottom-enriched");
    assert!(result.p_value < 0.05, "Should be significant");
}

/// Test with realistic gene expression data
#[tokio::test]
async fn test_fgsea_simple_realistic() {
    let engine = skip_if_no_gpu!(GpuEngine::new().await);

    // Simulate realistic gene expression: 5000 genes
    let n_total = 5000;

    // Create scores with some distribution
    use rand::prelude::*;
    use rand_distr::{Distribution, Normal};
    let mut rng = StdRng::seed_from_u64(12345);
    let normal = Normal::new(0.0, 1.0).unwrap();

    let mut scores: Vec<f64> = (0..n_total).map(|_| normal.sample(&mut rng)).collect();

    // Add strong signal to top 100 genes
    for score in scores.iter_mut().take(100) {
        *score += 2.0;
    }

    // Sort in descending order
    scores.sort_by(|a, b| b.partial_cmp(a).unwrap());

    // Convert to abs scores (as would be done in real GSEA)
    let abs_scores: Vec<f32> = scores.iter().map(|&s| s.abs() as f32).collect();

    // Create pathway enriched in top genes
    let pathway_enriched: Vec<usize> = (0..50).collect();

    let result = engine
        .fgsea_simple_pathway(&pathway_enriched, &abs_scores, 2000, 42, 1)
        .expect("fgsea_simple failed");

    println!("Realistic enriched pathway:");
    println!("  ES: {:.4}", result.es);
    println!("  NES: {:?}", result.nes);
    println!("  P-value: {:.6e}", result.p_value);

    assert!(result.es > 0.0, "Should have positive ES");
    assert!(result.p_value < 0.01, "Should be highly significant");
}
