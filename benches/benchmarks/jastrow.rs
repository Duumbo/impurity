use criterion::{black_box, criterion_group, Criterion};
use impurity::{FockState, SIZE};
use impurity::jastrow::compute_jastrow_exp;

pub fn jastrow_long(c: &mut Criterion) {
    // Variationnal parameters:
    let params: Vec<f64> = vec![1.0; SIZE*SIZE];
    let mut res = 0.0;
    c.bench_function("Calcul Exponent Jastrow 8x8", |b| {
        b.iter(|| {
        let state = FockState {
            spin_up: 21,
            spin_down: 53,
        };
            res = compute_jastrow_exp(state, black_box(&params));
        })
    });
}

criterion_group!(benches, jastrow_long,);
