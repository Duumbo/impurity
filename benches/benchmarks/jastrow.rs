use criterion::{black_box, criterion_group, Criterion, BenchmarkId};
use impurity::jastrow::{compute_jastrow_exp, fast_update_jastrow};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use impurity::{FockState, RandomStateGeneration, Spin};

const MAX_N_SITES: usize = 128;
type DATATYPE = u128;

pub fn bench_jastrow_number_of_sites(c: &mut Criterion) {
    let mut group = c.benchmark_group("Calcul du projecteur Jastrow");
    let mut rng = SmallRng::seed_from_u64(42);

    // Setup the intial state
    let mut params: [f64; MAX_N_SITES*MAX_N_SITES] = [0.0; MAX_N_SITES*MAX_N_SITES];
    for i in 0..MAX_N_SITES*MAX_N_SITES {
        params[i] = rng.gen();
    }
    let mut res;

    // Setup the loop over the number of sites
    for i in 2..129 {
        let mut hop = (0, 0, Spin::Up);
        // Setup the random state
        let state: FockState<DATATYPE> = FockState::generate_from_nelec(&mut rng, i + i%2, i);

        res = compute_jastrow_exp(state, black_box(&params), i);
        let newstate = state.generate_hopping(&mut rng, state.n_sites.try_into().unwrap(), &mut hop);
        group.bench_with_input(BenchmarkId::new("Calcul complet", i), &i,
        |b, i| b.iter(||{
            compute_jastrow_exp(newstate, black_box(&params), *i);
        }
        ));
        group.bench_with_input(BenchmarkId::new("Fast update", i), &i,
            |b, _| b.iter(||{
            fast_update_jastrow(&mut res, &params, &state, &newstate, state.n_sites, hop.0, hop.1);
            }));
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
