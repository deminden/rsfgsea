use rsfgsea::algo::calculate_es_fgsea;
use rsfgsea::prelude::*;
use std::fs;
use tempfile::tempdir;

fn ranked_list_from_scores(scores: &[f64]) -> RankedList {
    let genes = (0..scores.len())
        .map(|i| format!("g{}", i + 1))
        .collect::<Vec<_>>();
    RankedList::new(genes, scores.to_vec())
}

fn pathway_from_indices(name: &str, hits: &[usize], ranks: &RankedList) -> Pathway {
    Pathway {
        name: name.to_string(),
        description: None,
        genes: hits.iter().map(|&i| ranks.genes[i].clone()).collect(),
    }
}

#[test]
fn calc_gsea_stat_examples_match_upstream_fgsea_cases() {
    let stats = (-10..=10).rev().map(|x| x as f64).collect::<Vec<_>>();
    let n_total = stats.len();

    let top_hits = (0..5).collect::<Vec<_>>();
    let bottom_hits = (15..21).collect::<Vec<_>>();
    let mixed_hits = vec![1, 3, 5, 7, 9];

    let (top_es, _) = calculate_es_fgsea(&stats, &top_hits, n_total, ScoreType::Std);
    let (bottom_es, _) = calculate_es_fgsea(&stats, &bottom_hits, n_total, ScoreType::Std);
    let (mixed_es, _) = calculate_es_fgsea(&stats, &mixed_hits, n_total, ScoreType::Std);

    assert!((top_es - 1.0).abs() < 1e-12);
    assert!((bottom_es + 1.0).abs() < 1e-12);
    assert!((mixed_es - 0.71).abs() < 0.01);
}

#[test]
fn calc_gsea_stat_returns_zero_for_balanced_case() {
    let stats = (-10..=10).rev().map(|x| x as f64).collect::<Vec<_>>();
    let hits = vec![9, 10, 11];

    let (es, _) = calculate_es_fgsea(&stats, &hits, stats.len(), ScoreType::Std);

    assert_eq!(es, 0.0);
}

#[test]
fn calc_gsea_stat_leading_edge_matches_upstream_examples() {
    let ranks = ranked_list_from_scores(&[
        10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0,
        -7.0, -8.0, -9.0, -10.0,
    ]);
    let pathways = vec![
        pathway_from_indices("pos", &[0, 1, 2, 3, 4, 9], &ranks),
        pathway_from_indices("neg", &[9, 15, 16, 17, 18, 19, 20], &ranks),
        pathway_from_indices("zero", &[9, 10, 11], &ranks),
    ];

    let res = run_gsea_simple(
        &ranks,
        &pathways,
        10,
        1,
        1,
        ranks.len() - 1,
        1e-50,
        ScoreType::Std,
        1.0,
    );

    let pos = res.iter().find(|r| r.pathway_name == "pos").unwrap();
    let neg = res.iter().find(|r| r.pathway_name == "neg").unwrap();
    let zero = res.iter().find(|r| r.pathway_name == "zero").unwrap();

    assert!(pos.leading_edge.contains(&"g1".to_string()));
    assert!(!pos.leading_edge.contains(&"g10".to_string()));

    assert!(neg.leading_edge.contains(&"g21".to_string()));
    assert!(!neg.leading_edge.contains(&"g10".to_string()));

    assert_eq!(zero.es, 0.0);
    assert!(zero.leading_edge.is_empty());
}

#[test]
fn calc_gsea_stat_handles_zero_gene_level_stats_consistently() {
    let stats_zero = vec![
        10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0,
        -7.0, -8.0, -9.0, -10.0,
    ];
    let stats_eps = vec![
        10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, -1e-9, -1.0, -2.0, -3.0, -4.0, -5.0,
        -6.0, -7.0, -8.0, -9.0, -10.0,
    ];

    let hits = vec![10];
    let es_zero = calculate_es_fgsea(&stats_zero, &hits, stats_zero.len(), ScoreType::Std).0;
    let es_eps = calculate_es_fgsea(&stats_eps, &hits, stats_eps.len(), ScoreType::Std).0;

    assert!((es_zero - es_eps).abs() < 1e-12);
}

#[test]
fn simple_results_are_reproducible_for_fixed_seed() {
    let ranks = ranked_list_from_scores(&[4.0, 3.0, 2.0, 1.0, -1.0, -2.0, -3.0, -4.0]);
    let pathways = vec![
        pathway_from_indices("p1", &[0, 1, 2], &ranks),
        pathway_from_indices("p2", &[5, 6, 7], &ranks),
    ];

    let res1 = run_gsea_simple(
        &ranks,
        &pathways,
        2000,
        42,
        1,
        ranks.len() - 1,
        1e-50,
        ScoreType::Std,
        1.0,
    );
    let res2 = run_gsea_simple(
        &ranks,
        &pathways,
        2000,
        42,
        1,
        ranks.len() - 1,
        1e-50,
        ScoreType::Std,
        1.0,
    );

    assert_eq!(res1.len(), res2.len());
    for (a, b) in res1.iter().zip(res2.iter()) {
        assert_eq!(a.pathway_name, b.pathway_name);
        assert!((a.es - b.es).abs() < 1e-12);
        assert!((a.p_value - b.p_value).abs() < 1e-12);
        assert_eq!(a.leading_edge, b.leading_edge);
    }
}

#[test]
fn multilevel_results_are_reproducible_for_fixed_seed() {
    let ranks = ranked_list_from_scores(&[1.1, 1.0, 0.9, 0.5, 0.0, -0.1, -0.5, -0.9, -1.0]);
    let pathways = vec![Pathway {
        name: "p".to_string(),
        description: None,
        genes: vec!["g1".to_string(), "g2".to_string(), "g5".to_string()],
    }];

    let res1 = run_gsea_with_sample_size(
        &ranks,
        &pathways,
        1000,
        1,
        1,
        ranks.len() - 1,
        1e-50,
        ScoreType::Std,
        1.0,
        101,
    );
    let res2 = run_gsea_with_sample_size(
        &ranks,
        &pathways,
        1000,
        1,
        1,
        ranks.len() - 1,
        1e-50,
        ScoreType::Std,
        1.0,
        101,
    );

    assert_eq!(res1.len(), 1);
    assert_eq!(res2.len(), 1);
    assert!((res1[0].p_value - res2[0].p_value).abs() < 1e-12);
}

#[test]
fn leading_edge_is_consistent_under_sign_flip_and_positive_score_type() {
    let ranks = ranked_list_from_scores(&[10.0, 8.0, 6.0, 4.0, 2.0, -2.0, -4.0, -6.0, -8.0, -10.0]);
    let pathway = pathway_from_indices("p", &[0, 1, 4, 8], &ranks);
    let neg_ranks =
        ranked_list_from_scores(&[-10.0, -8.0, -6.0, -4.0, -2.0, 2.0, 4.0, 6.0, 8.0, 10.0]);
    let neg_pathway = Pathway {
        name: "p".to_string(),
        description: None,
        genes: pathway.genes.clone(),
    };

    let std_res = run_gsea_simple(
        &ranks,
        std::slice::from_ref(&pathway),
        50,
        1,
        1,
        9,
        1e-50,
        ScoreType::Std,
        1.0,
    );
    let neg_std_res = run_gsea_simple(
        &neg_ranks,
        std::slice::from_ref(&neg_pathway),
        50,
        1,
        1,
        9,
        1e-50,
        ScoreType::Std,
        1.0,
    );
    let pos_res = run_gsea_simple(
        &ranks,
        std::slice::from_ref(&pathway),
        50,
        1,
        1,
        9,
        1e-50,
        ScoreType::Pos,
        1.0,
    );

    assert_eq!(std_res[0].leading_edge, neg_std_res[0].leading_edge);
    assert_eq!(std_res[0].leading_edge, pos_res[0].leading_edge);
}

#[test]
fn duplicate_genes_in_gene_sets_are_deduplicated() {
    let ranks = ranked_list_from_scores(&[5.0, 4.0, 3.0, 2.0, 1.0, -1.0]);
    let pathways = vec![
        Pathway {
            name: "dedup".to_string(),
            description: None,
            genes: vec!["g1".to_string(), "g2".to_string(), "g3".to_string()],
        },
        Pathway {
            name: "dup".to_string(),
            description: None,
            genes: vec![
                "g1".to_string(),
                "g2".to_string(),
                "g2".to_string(),
                "g3".to_string(),
            ],
        },
    ];

    let simple = run_gsea_simple(
        &ranks,
        &pathways,
        500,
        42,
        1,
        ranks.len() - 1,
        1e-50,
        ScoreType::Std,
        1.0,
    );
    let multi = run_gsea(
        &ranks,
        &pathways,
        500,
        42,
        1,
        ranks.len() - 1,
        1e-50,
        ScoreType::Std,
        1.0,
    );

    assert_eq!(simple[0].size, simple[1].size);
    assert!((simple[0].es - simple[1].es).abs() < 1e-12);
    assert_eq!(multi[0].size, multi[1].size);
    assert!((multi[0].es - multi[1].es).abs() < 1e-12);
}

#[test]
fn fgsea_wrapper_routes_to_simple_when_nperm_is_provided() {
    let ranks = ranked_list_from_scores(&[4.0, 3.0, 2.0, 1.0, -1.0, -2.0, -3.0, -4.0]);
    let pathways = vec![
        pathway_from_indices("p1", &[0, 1, 2], &ranks),
        pathway_from_indices("p2", &[5, 6, 7], &ranks),
    ];

    let wrapper = fgsea(
        &ranks,
        &pathways,
        Some(500),
        1000,
        42,
        1,
        ranks.len() - 1,
        1e-50,
        ScoreType::Std,
        1.0,
    );
    let simple = run_gsea_simple(
        &ranks,
        &pathways,
        500,
        42,
        1,
        ranks.len() - 1,
        1e-50,
        ScoreType::Std,
        1.0,
    );

    assert_eq!(wrapper.len(), simple.len());
    for (a, b) in wrapper.iter().zip(simple.iter()) {
        assert_eq!(a.pathway_name, b.pathway_name);
        assert_eq!(a.size, b.size);
        assert!((a.es - b.es).abs() < 1e-12);
        assert!((a.p_value - b.p_value).abs() < 1e-12);
        assert_eq!(a.leading_edge, b.leading_edge);
    }
}

#[test]
fn single_pathway_simple_matches_batched_result_for_that_pathway() {
    let ranks = ranked_list_from_scores(&[4.0, 3.0, 2.0, 1.0, -1.0, -2.0, -3.0, -4.0]);
    let pathways = vec![
        pathway_from_indices("p1", &[0, 1, 2], &ranks),
        pathway_from_indices("p2", &[5, 6, 7], &ranks),
    ];

    let batched = run_gsea_simple(
        &ranks,
        &pathways,
        500,
        42,
        1,
        ranks.len() - 1,
        1e-50,
        ScoreType::Std,
        1.0,
    );
    let single = run_gsea_simple(
        &ranks,
        &pathways[1..2],
        500,
        42,
        1,
        ranks.len() - 1,
        1e-50,
        ScoreType::Std,
        1.0,
    );

    let from_batch = batched
        .iter()
        .find(|res| res.pathway_name == "p2")
        .expect("missing p2 result from batched run");
    assert_eq!(single.len(), 1);
    assert_eq!(single[0].pathway_name, from_batch.pathway_name);
    assert_eq!(single[0].size, from_batch.size);
    assert!((single[0].es - from_batch.es).abs() < 1e-12);
    assert_eq!(single[0].leading_edge, from_batch.leading_edge);
    assert!(single[0].p_value.is_finite());
}

#[test]
fn simple_and_multilevel_return_empty_for_zero_pathways() {
    let ranks = ranked_list_from_scores(&[4.0, 3.0, 2.0, 1.0, -1.0, -2.0]);
    let pathways = vec![pathway_from_indices("p", &[0, 1], &ranks)];

    let simple = run_gsea_simple(
        &ranks,
        &pathways,
        100,
        42,
        50,
        10,
        1e-50,
        ScoreType::Std,
        1.0,
    );
    let multi = run_gsea(
        &ranks,
        &pathways,
        100,
        42,
        50,
        10,
        1e-50,
        ScoreType::Std,
        1.0,
    );

    assert!(simple.is_empty());
    assert!(multi.is_empty());
}

#[test]
fn full_universe_gene_set_is_skipped() {
    let ranks = ranked_list_from_scores(&[3.0, 2.0, 1.0, -1.0, -2.0]);
    let pathways = vec![Pathway {
        name: "all".to_string(),
        description: None,
        genes: ranks.genes.clone(),
    }];

    let res = run_gsea_simple(
        &ranks,
        &pathways,
        1,
        1,
        1,
        usize::MAX,
        1e-50,
        ScoreType::Std,
        1.0,
    );

    assert!(res.is_empty());
}

#[test]
fn score_type_specific_zero_es_cases_do_not_break() {
    let ranks = ranked_list_from_scores(&[5.0, 4.0, 3.0, 2.0, 1.0, -1.0, -2.0, -3.0, -4.0, -5.0]);
    let tail = vec![Pathway {
        name: "tail".to_string(),
        description: None,
        genes: ranks.genes[5..].to_vec(),
    }];
    let head = vec![Pathway {
        name: "head".to_string(),
        description: None,
        genes: ranks.genes[..5].to_vec(),
    }];

    let pos = run_gsea_simple(&ranks, &tail, 1000, 1, 1, 9, 1e-50, ScoreType::Pos, 1.0);
    let neg = run_gsea_simple(&ranks, &head, 1000, 1, 1, 9, 1e-50, ScoreType::Neg, 1.0);

    assert_eq!(pos.len(), 1);
    assert_eq!(neg.len(), 1);
    assert!(pos[0].p_value.is_finite());
    assert!(neg[0].p_value.is_finite());
}

#[test]
fn ranked_list_reader_rejects_duplicate_and_non_finite_scores() {
    let dir = tempdir().unwrap();

    let dup_path = dir.path().join("dup.rnk");
    fs::write(&dup_path, "g1\t1.0\ng1\t2.0\n").unwrap();
    let dup_err = read_ranked_list(dup_path.to_str().unwrap()).unwrap_err();
    assert!(dup_err.to_string().contains("Duplicate gene"));

    let nan_path = dir.path().join("nan.rnk");
    fs::write(&nan_path, "g1\tNaN\n").unwrap();
    let nan_err = read_ranked_list(nan_path.to_str().unwrap()).unwrap_err();
    assert!(nan_err.to_string().contains("Non-finite score"));

    let inf_path = dir.path().join("inf.rnk");
    fs::write(&inf_path, "g1\tinf\n").unwrap();
    let inf_err = read_ranked_list(inf_path.to_str().unwrap()).unwrap_err();
    assert!(inf_err.to_string().contains("Non-finite score"));
}

#[test]
fn gmt_reader_loads_pathways() {
    let dir = tempdir().unwrap();
    let gmt = dir.path().join("test.gmt");
    fs::write(&gmt, "p1\tdesc\tg1\tg2\tg3\np2\t\tg4\tg5\n").unwrap();

    let db = read_gmt(gmt.to_str().unwrap()).unwrap();

    assert_eq!(db.pathways.len(), 2);
    assert_eq!(db.pathways[0].name, "p1");
    assert_eq!(db.pathways[0].genes, vec!["g1", "g2", "g3"]);
    assert_eq!(db.pathways[1].name, "p2");
    assert_eq!(db.pathways[1].description, None);
}
