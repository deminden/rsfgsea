use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rsfgsea::prelude::*;

fn benchmark_es(c: &mut Criterion) {
    let n = 10000;
    let k = 100;
    let abs_scores: Vec<f64> = (0..n).map(|i| (n - i) as f64).collect();
    let mut hits: Vec<usize> = Vec::new();
    for i in 0..k {
        hits.push(i * (n / k));
    }

    c.bench_function("calculate_es_10k_100", |b| {
        b.iter(|| {
            calculate_es(
                black_box(&hits),
                black_box(&abs_scores),
                black_box(n),
                black_box(ScoreType::Std),
            )
        })
    });
}

criterion_group!(benches, benchmark_es);
criterion_main!(benches);
