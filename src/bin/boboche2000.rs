use log::info;

use impurity::{FockState, RandomStateGeneration, VarParams, SysParams, generate_bitmask, DerivativeOperator};
use impurity::monte_carlo::compute_mean_energy;

/// Size of the system.
const SIZE: usize = 4;
/// Hubbard's model $U$ parameter
static CONS_U: f64 = 1.0;
/// Hubbard's model $t$ parameter
static CONS_T: f64 = -1.0;

const NFIJ: usize = 4*SIZE*SIZE;
const NVIJ: usize = SIZE*SIZE;
const NGI: usize = SIZE;

pub const HOPPINGS: [f64; SIZE*SIZE] = [
    0.0, 1.0, 1.0, 0.0,
    1.0, 0.0, 0.0, 1.0,
    1.0, 0.0, 0.0, 1.0,
    0.0, 1.0, 1.0, 0.0
];



const NELEC: usize = 4;
const NMCSAMP: usize = 100_000;
const NMCWARMUP: usize = 1000;
const CLEAN_UPDATE_FREQUENCY: usize = 8;
const TOLERENCE_SHERMAN_MORRISSON: f64 = 1e-15;
const TOLERANCE_SINGULARITY: f64 = 1e-15;
const _MONTE_CARLO_CONVERGENCE_TOLERANCE: f64 = 1e-1;

#[allow(dead_code)]
fn sq(a: f64) -> f64 {
    a * a
}

#[allow(dead_code)]
fn mean_energy_analytic_2sites(params: &VarParams, _sys: &SysParams) -> f64 {
    let f00 = params.fij[0 + SIZE*SIZE] - params.fij[0 + 2*SIZE*SIZE];
    let f01 = params.fij[1 + SIZE*SIZE] - params.fij[2 + 2*SIZE*SIZE];
    let f10 = params.fij[2 + SIZE*SIZE] - params.fij[1 + 2*SIZE*SIZE];
    let f11 = params.fij[3 + SIZE*SIZE] - params.fij[3 + 2*SIZE*SIZE];
    let f1 = f01 + f10;
    let f0 = f11 * <f64>::exp(params.gi[1] - params.vij[1]);
    let f2 = f00 * <f64>::exp(params.gi[0] - params.vij[1]);
    let f4 = f0 + f2;
    let exp0 = 2.0 * (params.gi[0] + params.vij[1]);
    let exp1 = 2.0 * (params.gi[1] + params.vij[1]);
    let num = 2.0 * f1 * f4 * CONS_T + sq(f0) * CONS_U + sq(f2) * CONS_U;
    let deno =
        sq(f00) * <f64>::exp(-exp0) +
        sq(f01) + sq(f10) +
        sq(f11) * <f64>::exp(-exp1);
    num / deno
}

fn log_system_parameters(sys: &SysParams) {
    info!("System parameter SIZE = {}", sys.size);
    info!("System parameter NELEC = {}", sys.nelec);
    info!("System parameter NMCSAMP = {}", sys.nmcsample);
    info!("System parameter NMCWARMUP = {}", sys.nmcwarmup);
    info!("System parameter CONS_U = {}", sys.cons_u);
    info!("System parameter CONS_T = {}", sys.cons_t);
}

fn main() {
    // Initialize logger
    env_logger::init();
    let bitmask = generate_bitmask(&HOPPINGS, SIZE);
    let sys = SysParams {
        size: SIZE,
        nelec: NELEC,
        array_size: (SIZE + 7) / 8,
        cons_t: CONS_T,
        cons_u: CONS_U,
        nfij: NFIJ,
        nvij: NVIJ,
        ngi: NGI,
        mcsample_interval: 1,
        transfert_matrix: &HOPPINGS,
        hopping_bitmask: &bitmask,
        clean_update_frequency: CLEAN_UPDATE_FREQUENCY,
        nmcwarmup: NMCWARMUP,
        nmcsample: NMCSAMP,
        tolerance_sherman_morrison: TOLERENCE_SHERMAN_MORRISSON,
        tolerance_singularity: TOLERANCE_SINGULARITY,
        pair_wavefunction: false,
    };
    log_system_parameters(&sys);

    let mut otilde: Vec<f64> = vec![0.0; (NFIJ + NVIJ + NGI) * NMCSAMP];
    let mut expvalo: Vec<f64> = vec![0.0; NFIJ + NVIJ + NGI];
    let mut expval_ho: Vec<f64> = vec![0.0; NFIJ + NVIJ + NGI];
    let mut visited: Vec<usize> = vec![0; NMCSAMP];
    let mut der = DerivativeOperator {
        o_tilde: &mut otilde,
        expval_o: &mut expvalo,
        ho: &mut expval_ho,
        n: (NFIJ + NVIJ + NGI) as i32,
        nsamp: NMCSAMP as f64,
        nsamp_int: 1,
        mu: -1,
        visited: &mut visited,
        pfaff_off: NGI + NVIJ,
        jas_off: NGI,
        epsilon: 0.0,
    };

    let mut fij: [f64; NFIJ] = [
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
    1.078313550425773	,
    0.007172274681240365,
    0.028714778076311877,
    0.09168843535310542	,
    0.04813118562079141	,
    1.0625398526882723	,
    0.08433353658389342	,
    0.002722470871706029,
    0.07270002762085896	,
    0.026989164590497917,
    0.007555596176108393,
    0.046284058565227465,
    0.011127921360085048,
    0.07287939415825727	,
    0.08138828369394709	,
    0.012799567556772274,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
    ];

    let mut vij: [f64; NVIJ] = [
        0.0, 2.0, 2.0, 2.0,
        2.0, 0.0, 2.0, 2.0,
        2.0, 2.0, 0.0, 2.0,
        2.0, 2.0, 2.0, 0.0,
    ];

    let mut gi: [f64; NGI] = [
        -1.0, -1.0, -1.0, -1.0
    ];

    let mut rng = rand::thread_rng();
    //let parameters = generate_random_params(&mut rng);
    let parameters = VarParams {
        size: SIZE,
        fij: &mut fij,
        gi: &mut gi,
        vij: &mut vij
    };

    let state: FockState<u8> = {
        let mut tmp: FockState<u8> = FockState::generate_from_nelec(&mut rng, NELEC, SIZE);
        while tmp.spin_up.count_ones() != tmp.spin_down.count_ones() {
            tmp = FockState::generate_from_nelec(&mut rng, NELEC, SIZE);
        }
        tmp
    };

    info!("Initial State: {}", state);
    info!("Initial Nelec: {}, {}", state.spin_down.count_ones(), state.spin_up.count_ones());
    info!("Nsites: {}", state.n_sites);

    let (energy, _, _, _) = compute_mean_energy(&mut rng, state, &parameters, &sys, &mut der);
    println!("energy: {}", energy);
    //close(energy, -0.35, MONTE_CARLO_CONVERGENCE_TOLERANCE);
    //close(energy, mean_energy_analytic_2sites(&parameters, &sys), MONTE_CARLO_CONVERGENCE_TOLERANCE);
}
