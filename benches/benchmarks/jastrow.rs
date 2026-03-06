use criterion::{black_box, criterion_group, Criterion, BenchmarkId};
use impurity::jastrow::{compute_jastrow_exp, fast_update_jastrow};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use impurity::{FockState, Hopper, RandomStateGeneration, Spin};

const MAX_N_SITES: usize = 128;
type DATATYPE = u128;

pub fn bench_jastrow_number_of_sites(c: &mut Criterion) {
    let mut group = c.benchmark_group("Calcul du projecteur Jastrow");
    let mut rng = SmallRng::seed_from_u64(42);

    // Setup the intial state
    let mut params: Vec<f64> = vec![0.0; MAX_N_SITES*MAX_N_SITES];
    for p in params.iter_mut() {
        *p = rng.gen_range(-1.0..1.0);
    }

    // Setup the loop over the number of sites
    for i in 2..=128 {
        // Setup the random state

        let mut rng = SmallRng::seed_from_u64(4298);
        group.bench_with_input(BenchmarkId::new("Calcul complet", i), &i,
            |b, i| {
                let state: FockState<DATATYPE> = if i % 2 == 0 {
                    FockState::generate_from_nelec(&mut rng, *i, *i)
                } else {
                    FockState::generate_from_nelec(&mut rng, *i + 1, *i)
                };
                let mut out_idx = (0, 0, Spin::Up);
                state.generate_hopping(&mut rng, *i as u32, &mut out_idx);
                let newstate = state.make_hopping(&out_idx);

                b.iter(||{
                    compute_jastrow_exp(newstate, black_box(&params), *i);
            });
            }
           );

        let mut rng = SmallRng::seed_from_u64(4298);
        group.bench_with_input(BenchmarkId::new("Fast update", i), &i,
            |b, i| {
                let state: FockState<DATATYPE> = if i % 2 == 0 {
                    FockState::generate_from_nelec(&mut rng, *i, *i)
                } else {
                    FockState::generate_from_nelec(&mut rng, *i + 1, *i)
                };
                let mut res = compute_jastrow_exp(state, black_box(&params), *i);
                let mut out_idx = (0, 0, Spin::Up);
                state.generate_hopping(&mut rng, *i as u32, &mut out_idx);
                let newstate = state.make_hopping(&out_idx);

                b.iter(||{
                    fast_update_jastrow(&mut res, &params, &state, &newstate, state.n_sites, 3, 4);
            })
            });
    }
    group.finish();
}

//pub fn jastrow_long(c: &mut Criterion) {
//    // Variationnal parameters:
//    let params: Vec<f64> = vec![1.0; SIZE * SIZE];
//    let mut res = 0.0;
//    c.bench_function("Calcul Exponent Jastrow 8x8", |b| {
//        b.iter(|| {
//            let state = FockState {
//                spin_up: 21u8,
//                spin_down: 53u8,
//                n_sites: 8,
//            };
//            res = compute_jastrow_exp(state, black_box(&params), 8);
//        })
//    });
//}

//pub fn jastrow_fast(c: &mut Criterion) {
//    // Setup the same situation as before
//    let params: Vec<f64> = vec![1.0; SIZE * SIZE];
//    let mut res = 0.0;
//    let state = FockState {
//        spin_up: 21u8,
//        spin_down: 53u8,
//        n_sites: 8,
//    };
//    res = compute_jastrow_exp(state, black_box(&params), 8);
//    // The update is up 3->4
//    let newstate = FockState {
//        spin_up: 13u8,
//        spin_down: 53u8,
//        n_sites: 8,
//    };
//    c.bench_function("Calcul Exponent FASTJastrow 8x8", |b| {
//        b.iter(|| {
//            fast_update_jastrow(&mut res, &params, &state, &newstate, state.n_sites, 3, 4);
//        })
//    });
//}

criterion_group!(benches, bench_jastrow_number_of_sites);
