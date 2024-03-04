use criterion::{black_box, criterion_group, Criterion, BenchmarkId};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use impurity::gutzwiller::{compute_gutzwiller_exp, fast_update_gutzwiller};
use impurity::{FockState, RandomStateGeneration};

const MAX_N_SITES: usize = 128;
type DATATYPE = u128;

pub fn bench_gutzwiller_number_of_sites(c: &mut Criterion) {
    let mut group = c.benchmark_group("Calcul du projecteur Gutzwiller");
    let mut rng = SmallRng::seed_from_u64(42);

    // Setup the intial state
    let mut params: [f64; MAX_N_SITES] = [0.0; MAX_N_SITES];
    rng.fill(&mut params);
    let mut res;

    // Setup the loop over the number of sites
    for i in 2..129 {
        // Setup the random state
        let state: FockState<DATATYPE> = FockState::generate_from_nelec(&mut rng, i, MAX_N_SITES);

        res = compute_gutzwiller_exp(state.clone(), &params, i);
        group.bench_with_input(BenchmarkId::new("Calcul complet", i), &i,
            |b, i| b.iter(||{
                compute_gutzwiller_exp(black_box(state.clone()), black_box(&params), *i);
            }
           ));
        group.bench_with_input(BenchmarkId::new("Fast update", i), &i,
            |b, _| b.iter(||{
                fast_update_gutzwiller(&mut res, &params, &state.spin_up, 0, 1);
            }));
    }
    group.finish();
}

criterion_group!(benches, bench_gutzwiller_number_of_sites,);
