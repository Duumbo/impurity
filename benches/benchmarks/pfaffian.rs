use criterion::{black_box, criterion_group, Criterion};
use impurity::pfaffian::compute_pfaffian_wq;
use impurity::pfaffian::{construct_matrix_a_from_state, get_pfaffian_ratio, update_pstate};
use impurity::{BitOps, FockState, SysParams, generate_bitmask, Spin};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use rand_mt::Mt64;

const SIZE_N: usize = 2;
const SIZE_M: usize = 4;
// SIZE = SIZE_N x SIZE_M
const SIZE: usize = SIZE_N*SIZE_M;
const NFIJ: usize = 4*SIZE*SIZE;
const NVIJ: usize = SIZE*(SIZE - 1) / 2;
const NGI: usize = SIZE;
const NELEC: usize = SIZE;
const NMCSAMP: usize = 1000;
const NBOOTSTRAP: usize = 1;
const NMCWARMUP: usize = 100;
const NWARMUPCHAINS: usize = 1;
const MCSAMPLE_INTERVAL: usize = 1;
const CLEAN_UPDATE_FREQUENCY: usize = 32;
const TOLERENCE_SHERMAN_MORRISSON: f64 = 1e-12;
const TOLERENCE_SINGULARITY: f64 = 1e-12;
const _CONS_U: f64 = 1.0;
const CONS_T: f64 = 1.0;
const PAIRWF: bool = false;

pub const HOPPINGS: [f64; SIZE*SIZE] = {
    // Constructs hopping matrix for SITES_N*SITES_M
    let mut tmp = [0.0; SIZE*SIZE];
    let mut i = 0;
    let mut j = 0;
    while i < SIZE_M {
        while j < SIZE_N {
            let next_inline = (i + 1) % SIZE_M;
            let prev_inline = (i + SIZE_M - 1) % SIZE_M;
            let next_column = (j + 1) % SIZE_N;
            let prev_column = (j +SIZE_N - 1) % SIZE_N;
            tmp[ next_inline + j * SIZE_M + (i + j * SIZE_M) * SIZE] = 1.0;
            tmp[ prev_inline + j * SIZE_M + (i + j * SIZE_M) * SIZE] = 1.0;
            tmp[ i + j * SIZE_M + (i + next_column * SIZE_M) * SIZE] = 1.0;
            tmp[ i + j * SIZE_M + (i + prev_column * SIZE_M) * SIZE] = 1.0;
            j += 1;
        }
        i += 1;
        j = 0;
    }
    i = 0;
    // RESET DIAGONAL (edge case for if SIZE_M==1 or SIZE_N==1)
    while i < SIZE {
        tmp[ i + i*SIZE] = 0.0;
        i += 1;
    }
    tmp
};

pub fn bench_compute_pfaff(c: &mut Criterion) {
    let mut rng = Mt64::new(34);
    let mut fij: Vec<f64> = Vec::new();
    for i in 0..8 {
        for j in 0..8 {
            fij.push(rng.gen());
            if i == j { fij[j + 8 * j] = 0.0;}
        }
    }
    c.bench_function("Calcul pfaffian 8x8", |b| {
        b.iter(|| compute_pfaffian_wq(black_box(&mut fij.clone()), 1))
    });
}


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

pub fn bench_compute_updated_pfaffian(c: &mut Criterion) {
    // From test_pfaffian_update_random_no_sign_correction
    const SIZE: usize = 8;
    let mut rng = SmallRng::seed_from_u64(42);
    let bitmask = generate_bitmask(&HOPPINGS, SIZE);
    let system_params = SysParams {
        size: SIZE,
        nelec: NELEC,
        array_size: (SIZE + 7) / 8,
        cons_t: -CONS_T,
        cons_u: CONS_T,
        nfij: NFIJ,
        nvij: NVIJ,
        ngi: NGI,
        transfert_matrix: &HOPPINGS,
        hopping_bitmask: &bitmask,
        clean_update_frequency: CLEAN_UPDATE_FREQUENCY,
        nmcsample: NMCSAMP,
        nbootstrap: NBOOTSTRAP,
        nmcwarmup: NMCWARMUP,
        nwarmupchains: NWARMUPCHAINS,
        mcsample_interval: MCSAMPLE_INTERVAL,
        tolerance_sherman_morrison: TOLERENCE_SHERMAN_MORRISSON,
        tolerance_singularity: TOLERENCE_SINGULARITY,
        pair_wavefunction: PAIRWF,
        _opt_iter: 0,
    };
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
    let state = FockState {
        spin_up: rng.gen::<u8>(),
        spin_down: rng.gen::<u8>(),
        n_sites: SIZE,
    };
    let n = state.spin_up.count_ones() + state.spin_down.count_ones();
    // Matrix needs to be even sized
    //
    // Initial State
    let pfstate = construct_matrix_a_from_state(&params, state, &system_params);

    // Generate random update
    // Spin up
    // What index from?
    // Get where there are electrons
    let (sups, _sdowns) = convert_spin_to_array(state, n as usize);
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
    c.bench_function("Calcul Update de ratio de pfaffian 8x8", |b| {
        b.iter(|| {
        let _tmp = get_pfaffian_ratio(&pfstate, initial_index, final_index, black_box(Spin::Up), &params);
        })
    });

    let res = get_pfaffian_ratio(&pfstate, initial_index, final_index, black_box(Spin::Up), &params);

    c.bench_function("Calcul Sherman-Morrison 8x8", |b| {
        b.iter(|| {
            let mut pfstate_work = pfstate.clone();
            let _new_pstate = update_pstate(&mut pfstate_work, (initial_index, final_index, Spin::Up), res.1.clone(), res.2);
        })
    });
}

criterion_group!(benches, bench_compute_pfaff, bench_compute_updated_pfaffian,);
