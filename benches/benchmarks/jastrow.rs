use criterion::{black_box, criterion_group, Criterion};
use impurity::jastrow::{compute_jastrow_exp, fast_update_jastrow};
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

pub fn jastrow_fast(c: &mut Criterion) {
    // Setup the same situation as before
    let params: Vec<f64> = vec![1.0; SIZE * SIZE];
    let mut res = 0.0;
    let state = FockState {
        spin_up: 21u8,
        spin_down: 53u8,
        n_sites: 8,
    };
    res = compute_jastrow_exp(state, black_box(&params), 8);
    // The update is up 3->4
    let newstate = FockState {
        spin_up: 13u8,
        spin_down: 53u8,
        n_sites: 8,
    };
    c.bench_function("Calcul Exponent FASTJastrow 8x8", |b| {
        b.iter(|| {
            fast_update_jastrow(&mut res, &params, &state, &newstate, state.n_sites, 3, 4);
        })
    });
}

criterion_group!(benches, jastrow_long, jastrow_fast);
