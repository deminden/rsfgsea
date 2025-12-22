use rsfgsea::prelude::*;

#[test]
fn test_regression_against_known_values() {
    let genes = vec![
        "ZMYND10".to_string(),
        "SCIN".to_string(),
        "MLXIPL".to_string(),
        "NPC1L1".to_string(),
        "CYP3A43".to_string(),
    ];
    let scores = vec![1.0, 0.8, 0.6, 0.4, 0.2];
    let ranks = RankedList::new(genes, scores);

    let pathways = vec![Pathway {
        name: "TEST".to_string(),
        description: None,
        genes: vec!["ZMYND10".to_string(), "SCIN".to_string()],
    }];

    let results = run_gsea(
        &ranks,
        &pathways,
        100,
        42,
        1,
        100,
        1e-10,
        ScoreType::Std,
        1.0,
    );
    assert_eq!(results.len(), 1);
    let res = &results[0];

    // With absolute weights [1.0, 0.8] and N=5, k=2.
    // Normalized sum = 1.8.
    // Hits at 0, 1.
    // es_at(0) = 1.0/1.8 - 0/3 = 0.555...
    // es_at(1) = 1.8/1.8 - 0/3 = 1.0
    // Max ES = 1.0.
    assert!((res.es - 1.0).abs() < 1e-9);
}
