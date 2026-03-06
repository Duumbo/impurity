use criterion::{black_box, criterion_group, Criterion, BenchmarkId};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use impurity::gutzwiller::{compute_gutzwiller_exp, fast_update_gutzwiller};
use impurity::{FockState, RandomStateGeneration, Spin, Hopper};

const MAX_N_SITES: usize = 128;
type DATATYPE = u128;

pub fn bench_gutzwiller_number_of_sites(c: &mut Criterion) {
    let mut group = c.benchmark_group("Calcul du projecteur Gutzwiller");
    let mut rng = SmallRng::seed_from_u64(42);

    // Setup the intial state
    let mut params: [f64; MAX_N_SITES] = [0.0; MAX_N_SITES];
    rng.fill(&mut params);

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

                compute_gutzwiller_exp(black_box(newstate), black_box(&params), *i);
                }
               )
            });

        let mut rng = SmallRng::seed_from_u64(4298);
        group.bench_with_input(BenchmarkId::new("Fast update", i), &i,
            |b, i| {
                let state: FockState<DATATYPE> = if i % 2 == 0 {
                    FockState::generate_from_nelec(&mut rng, *i, *i)
                } else {
                    FockState::generate_from_nelec(&mut rng, *i + 1, *i)
                };
                let mut hop = (0, 0, Spin::Up);
                state.generate_hopping(&mut rng, *i as u32, &mut hop);
                let mut res = compute_gutzwiller_exp(state, black_box(&params), *i);

                b.iter(||{
                    match hop.2 {
                        Spin::Up => {
                            fast_update_gutzwiller(&mut res, &params, &state.spin_down, hop.0, hop.1);
                        },
                        Spin::Down => {
                            fast_update_gutzwiller(&mut res, &params, &state.spin_up, hop.0, hop.1);
                        }
                    }
                })
            });
    }
    group.finish();
}

criterion_group!(benches, bench_gutzwiller_number_of_sites,);
