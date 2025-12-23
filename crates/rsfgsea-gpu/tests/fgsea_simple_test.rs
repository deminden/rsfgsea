use rsfgsea_gpu::GpuEngine;

macro_rules! skip_if_no_gpu {
    ($engine:expr) => {
        match $engine {
            Ok(e) => e,
            Err(e) => {
                println!("Skipping test: {}", e);
                return;
            }
        }
    };
}

/// Test basic GPU engine initialization
#[test]
fn test_gpu_engine_init() {
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
    let mut rng = rand::thread_rng();
    let mut all_indices = Vec::with_capacity(n_perm * k as usize);

    for _ in 0..n_perm {
        let mut pool: Vec<usize> = (0..n_total as usize).collect();
        pool.shuffle(&mut rng);
        let mut subset = pool[..k as usize].to_vec();
        subset.sort_unstable();
        for i in 0..k as usize {
            all_indices.push(subset[i] as u32);
        }
    }

    let results = engine
        .compute_es_batch(&abs_scores, &all_indices, k, n_total, n_perm as u32, 0)
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
    let abs_scores: Vec<f32> = (0..n_total)
        .map(|i| (n_total - i) as f32)
        .collect();

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
    let mut rng = rand::thread_rng();
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
    assert!(result.es.abs() < 0.5, "ES should be small for random pathway");
}

/// Test fgsea_simple_pathway with bottom-enriched pathway
#[tokio::test]
async fn test_fgsea_simple_pathway_bottom() {
    let engine = skip_if_no_gpu!(GpuEngine::new().await);

    let n_total = 1000;
    let k = 50;

    // Create ranked scores
    let abs_scores: Vec<f32> = (0..n_total)
        .map(|i| (n_total - i) as f32)
        .collect();

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
    use rand_distr::{Normal, Distribution};
    let mut rng = StdRng::seed_from_u64(12345);
    let normal = Normal::new(0.0, 1.0).unwrap();

    let mut scores: Vec<f64> = (0..n_total)
        .map(|_| normal.sample(&mut rng))
        .collect();

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

/// Benchmark: compare GPU vs CPU (conceptual test)
#[tokio::test]
async fn test_gpu_performance() {
    let engine = skip_if_no_gpu!(GpuEngine::new().await);

    let n_total = 10000;
    let k = 100;
    let n_perm = 10000;

    // Create test data
    let abs_scores: Vec<f32> = (0..n_total)
        .map(|i| (n_total - i) as f32)
        .collect();

    let pathway: Vec<usize> = (0..k).collect();

    // Time the GPU implementation
    let start = std::time::Instant::now();
    let result = engine
        .fgsea_simple_pathway(&pathway, &abs_scores, n_perm, 42, 1)
        .expect("fgsea_simple failed");
    let duration = start.elapsed();

    println!("\nGPU Performance Test:");
    println!("  Genes: {}", n_total);
    println!("  Pathway size: {}", k);
    println!("  Permutations: {}", n_perm);
    println!("  Time: {:?}", duration);
    println!("  ES: {:.4}", result.es);
    println!("  P-value: {:.6e}", result.p_value);

    // Just check it completes in reasonable time (< 5 seconds)
    assert!(duration.as_secs() < 5, "Should complete in under 5 seconds");
}
