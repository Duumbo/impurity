use criterion::{black_box, criterion_group, Criterion};
use impurity::jastrow::compute_jastrow_exp;
use impurity::{FockState, SIZE};

pub fn jastrow_long(c: &mut Criterion) {
    // Variationnal parameters:
    let params: Vec<f64> = vec![1.0; SIZE * SIZE];
    let mut res = 0.0;
    c.bench_function("Calcul Exponent Jastrow 8x8", |b| {
        b.iter(|| {
            let state = FockState {
                spin_up: 21u8,
                spin_down: 53u8,
                n_sites: 8,
            };
            res = compute_jastrow_exp(state, black_box(&params), 8);
        })
    });
}

criterion_group!(benches, jastrow_long,);
