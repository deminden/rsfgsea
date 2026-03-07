use rsfgsea::prelude::ScoreType;

fn cpu_es(hits: &[usize], weights: &[f32], n_total: usize, score_type: ScoreType) -> f64 {
    let weights_f64: Vec<f64> = weights.iter().map(|&w| w as f64).collect();
    rsfgsea::algo::calculate_es(hits, &weights_f64, n_total, score_type).0
}

#[test]
fn calculate_es_matches_reference_for_top_enriched_pathway() {
    let n_total = 100usize;
    let weights: Vec<f32> = (0..n_total).map(|i| (n_total - i) as f32).collect();
    let hits: Vec<usize> = (0..10).collect();

    let es = cpu_es(&hits, &weights, n_total, ScoreType::Std);

    assert!((es - 1.0).abs() < 1e-9, "top-enriched ES should be 1.0");
}

#[test]
fn calculate_es_respects_score_type_direction() {
    let n_total = 100usize;
    let weights: Vec<f32> = (0..n_total).map(|i| (n_total - i) as f32).collect();
    let top_hits: Vec<usize> = (0..10).collect();
    let bottom_hits: Vec<usize> = (n_total - 10..n_total).collect();

    let pos_top = cpu_es(&top_hits, &weights, n_total, ScoreType::Pos);
    let neg_top = cpu_es(&top_hits, &weights, n_total, ScoreType::Neg);
    let pos_bottom = cpu_es(&bottom_hits, &weights, n_total, ScoreType::Pos);
    let neg_bottom = cpu_es(&bottom_hits, &weights, n_total, ScoreType::Neg);

    assert!(pos_top > 0.0);
    assert!(neg_top <= 0.0);
    assert!(pos_bottom >= 0.0);
    assert!(neg_bottom < 0.0);
}

#[test]
fn calculate_es_stays_within_expected_range() {
    let n_total = 1000usize;
    let weights: Vec<f32> = (0..n_total)
        .map(|i| ((n_total - i) as f32).sqrt())
        .collect();
    let hits: Vec<usize> = (0..50).step_by(2).collect();

    let es_std = cpu_es(&hits, &weights, n_total, ScoreType::Std);
    let es_pos = cpu_es(&hits, &weights, n_total, ScoreType::Pos);
    let es_neg = cpu_es(&hits, &weights, n_total, ScoreType::Neg);

    assert!((-1.0..=1.0).contains(&es_std));
    assert!((-1.0..=1.0).contains(&es_pos));
    assert!((-1.0..=1.0).contains(&es_neg));
}
