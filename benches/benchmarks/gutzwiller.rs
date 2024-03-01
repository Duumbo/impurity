use criterion::{black_box, criterion_group, Criterion};
use impurity::gutzwiller::{compute_gutzwiller_exp, fast_update_gutzwiller};
use impurity::{FockState, SIZE};

pub fn gutzwiller_long(c: &mut Criterion) {
    // Variationnal parameters:
    const N_SITES: usize = 8;
    let params: Vec<f64> = vec![1.0; SIZE];
    let mut res = 0.0;
    c.bench_function("Calcul Exponent Gutzwiller 8x8", |b| {
        b.iter(|| {
            let state = FockState {
                spin_up: 21u8,
                spin_down: 53u8,
                n_sites: 8,
            }; // Should give 3.0
            res = compute_gutzwiller_exp(black_box(state.clone()), black_box(&params), N_SITES);
        })
    });
}

pub fn fast_gutzwiller(c: &mut Criterion) {
    // Let's setup the same situation as before.
    const N_SITES: usize = 8;
    let params: Vec<f64> = vec![1.0; SIZE];
    let mut res = 0.0;
    let state = FockState {
        spin_up: 21u8,
        spin_down: 53u8,
        n_sites: 8,
    }; // Should give 3.0
    res = compute_gutzwiller_exp(black_box(state.clone()), black_box(&params), N_SITES);

    // Now the benchmark will test the fast-update
    // The update is up 3->4
    c.bench_function("Calcul Exponent FASTGutzwiller 8x8", |b| {
        b.iter(|| {
            fast_update_gutzwiller(&mut res, &params, &state.spin_up, 3, 4);
        })
    });
}

criterion_group!(benches, gutzwiller_long, fast_gutzwiller,);
