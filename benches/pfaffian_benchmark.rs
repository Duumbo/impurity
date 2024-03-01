use criterion::{black_box, criterion_group, criterion_main, Criterion};
use impurity::pfaffian::{Spin, construct_matrix_a_from_state, get_pfaffian_ratio};
use impurity::{FockState, BitOps};
use rand::{rngs::SmallRng, Rng, SeedableRng};

fn convert_spin_to_array(state: FockState<u8>, n: usize) -> (Vec<usize>, Vec<usize>) {
    let mut indices: Vec<usize> = Vec::with_capacity(n);
    let mut indices2: Vec<usize> = Vec::with_capacity(n);
    let (mut spin_up, mut spin_down) = (state.spin_up, state.spin_down);
    let (mut i, mut j): (usize, usize) = (
        spin_up.leading_zeros() as usize,
        spin_down.leading_zeros() as usize,
    );
    while i < state.n_sites {
        indices.push(i);
        spin_up.set(i);
        i = spin_up.leading_zeros() as usize;
    }
    while j < state.n_sites {
        indices2.push(j);
        spin_down.set(j);
        j = spin_down.leading_zeros() as usize;
    }
    (indices, indices2)
}

pub fn bench_compute_pfaff(c: &mut Criterion) {
    // From test_pfaffian_update_random_no_sign_correction
    const SIZE: usize = 8;
    let mut rng = SmallRng::seed_from_u64(42);
    // Size of the system
    let mut params = vec![0.0; 4 * SIZE * SIZE];

    // Generate the variationnal parameters
    // params[i+8*j] = f_ij
    for j in 0..2*SIZE {
        for i in 0..2*SIZE {
            params[i + 2*j*SIZE] = rng.gen::<f64>();
        }
    }

    // Generate random initial state.
    let state = crate::FockState {
        spin_up: rng.gen::<u8>(),
        spin_down: rng.gen::<u8>(),
        n_sites: SIZE,
    };
    let n = state.spin_up.count_ones() + state.spin_down.count_ones();
    // Matrix needs to be even sized
    //
    // Initial State
    let pfstate = construct_matrix_a_from_state(params.clone(), state);

    // Generate random update
    // Spin up
    // What index from?
    // Get where there are electrons
    let (sups, sdowns) = convert_spin_to_array(state, n as usize);
    let initial_index =
            sups[rng.gen::<usize>() % sups.len()];
    // Where to?
    // It must not be occupied
    let mut final_index;
    final_index = rng.gen::<usize>() % SIZE;
    while sups.contains(&final_index) {
        final_index = rng.gen::<usize>() % SIZE;
    }

    // Now let's compute the pfaffian ratio
    let mut res: f64 = 0.0;
    c.bench_function("Calcul Update de ratio de pfaffian 8x8", |b| {
        b.iter(|| {
        res = get_pfaffian_ratio(&pfstate, initial_index, final_index, black_box(Spin::Up)).0;
        })
    });
}

criterion_group!(benches, bench_compute_pfaff);
criterion_main!(benches);
