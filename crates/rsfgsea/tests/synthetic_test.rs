use rand::prelude::*;
use rand::rngs::StdRng;
use rsfgsea::prelude::*;

#[test]
fn test_synthetic_large_run() {
    // 1. Generate synthetic RankedList
    let n_genes = 2000;
    let mut rng = StdRng::seed_from_u64(12345);

    let mut genes = Vec::new();
    let mut scores = Vec::new();

    for i in 0..n_genes {
        genes.push(format!("GENE_{}", i));
        // Random scores roughly normal distributed
        let score: f64 = rng.gen_range(-3.0..3.0);
        scores.push(score);
    }

    let db = RankedList::new(genes.clone(), scores);

    // 2. Generate synthetic Pathways
    let n_pathways = 50;
    let mut pathways = Vec::new();

    for i in 0..n_pathways {
        let size = rng.gen_range(15..100);
        let mut pw_genes = Vec::new();
        // Randomly sample genes for the pathway
        let mut gene_indices: Vec<usize> = (0..n_genes).collect();
        gene_indices.shuffle(&mut rng);

        for &idx in gene_indices.iter().take(size) {
            pw_genes.push(genes[idx].clone());
        }

        pathways.push(Pathway {
            name: format!("PATHWAY_{}", i),
            description: Some("Synthetic pathway".to_string()),
            genes: pw_genes,
        });
    }

    // 3. Run GSEA
    // Use 1000 perms to be reasonably fast but thorough
    let results = run_gsea(
        &db,
        &pathways,
        1000,
        999,
        10,
        500,
        1e-10,
        ScoreType::Std,
        1.0,
    );

    // 4. Assertions
    // We expect some results, though we filter by min/max size
    // Our synthetic pathways are 15-100, limits are 10-500, so all should pass size filter
    assert_eq!(results.len(), n_pathways);

    for res in results {
        assert!(res.es >= -1.0 && res.es <= 1.0);
        assert!(res.p_value >= 0.0 && res.p_value <= 1.0);
        if let Some(padj) = res.padj {
            assert!((0.0..=1.0).contains(&padj));
        }
        // Check leading edge is subset of genes
        assert!(res.leading_edge.len() <= res.size);
    }
}
