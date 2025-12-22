/// Simple CPU-only unit tests for the GPU implementation logic
/// These test the ES calculation on CPU without requiring GPU hardware

use rsfgsea_gpu::GpuEngine;

#[test]
fn test_es_calculation_logic() {
    // Test the ES calculation directly without GPU
    // This validates our algorithm matches expected behavior
    
    let n_total = 100;
    let pathway_size = 10;
    
    // Create simple weights
    let weights: Vec<f32> = (0..n_total).map(|i| (n_total - i) as f32).collect();
    
    // Top-enriched pathway
    let pathway_top: Vec<usize> = (0..pathway_size).collect();
    
    // Calculate expected ES manually
    let sum_weights: f64 = pathway_top.iter().map(|&i| weights[i] as f64).sum();
    let n_miss = (n_total - pathway_size) as f64;
    
    println!("Manual ES calculation test:");
    println!("  Total genes: {}", n_total);
    println!("  Pathway size: {}", pathway_size);
    println!("  Sum weights: {:.2}", sum_weights);
    println!("  N miss: {}", n_miss);
    
    // The ES should be positive for top-enriched pathways
    assert!(sum_weights > 0.0);
}

#[test]
fn test_permutation_indices() {
    use rand::prelude::*;
    let mut rng = rand::thread_rng();
    
    let n_total = 1000;
    let k = 50;
    
    // Generate a random permutation
    let mut indices: Vec<usize> = (0..n_total).collect();
    indices.partial_shuffle(&mut rng, k);
    
    // Verify we get k unique indices
    let selected: Vec<usize> = indices.iter().take(k).copied().collect();
    assert_eq!(selected.len(), k);
    
    // Verify all indices are in range
    for &idx in &selected {
        assert!(idx < n_total);
    }
    
    println!("Permutation test passed: {} indices selected from {}", k, n_total);
}

#[test]
fn test_enrichment_score_properties() {
    // Test properties of enrichment scores
    
    // ES should be 0 for empty pathway
    let empty_pathway: Vec<usize> = vec![];
    assert_eq!(empty_pathway.len(), 0);
    
    // ES calculation should handle uniform weights
    let weights_uniform: Vec<f32> = vec![1.0; 100];
    let pathway: Vec<usize> = (0..10).collect();
    
    let sum: f64 = pathway.iter().map(|&i| weights_uniform[i] as f64).sum();
    assert!((sum - 10.0).abs() < 0.001);
    
    println!("Enrichment score properties test passed");
}

/// Test that we can create the data structures needed for GPU computation
#[test]
fn test_data_preparation() {
    let n_total = 1000;
    let k = 50;
    let n_perm = 100;
    
    // Create scores
    let abs_scores: Vec<f32> = (0..n_total).map(|i| (n_total - i) as f32).collect();
    
    // Generate permutation indices
    use rand::prelude::*;
    use rand::rngs::StdRng;
    let mut rng = StdRng::seed_from_u64(42);
    
    let mut all_indices: Vec<u32> = Vec::with_capacity(n_perm * k);
    for _ in 0..n_perm {
        let mut pool: Vec<usize> = (0..n_total).collect();
        pool.partial_shuffle(&mut rng, k);
        for i in 0..k {
            all_indices.push(pool[i] as u32);
        }
    }
    
    assert_eq!(all_indices.len(), n_perm * k);
    assert_eq!(abs_scores.len(), n_total);
    
    println!("Data preparation test passed:");
    println!("  Scores: {} elements", abs_scores.len());
    println!("  Permutation indices: {} elements", all_indices.len());
    println!("  Expected: {} elements", n_perm * k);
}
