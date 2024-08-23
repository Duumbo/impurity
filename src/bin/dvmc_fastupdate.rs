use std::ptr::addr_of;
use std::slice::from_raw_parts as slice;
use blas::{daxpy, dcopy, dscal};
use impurity::optimisation::conjugate_gradiant;
use log::{debug, info};

use impurity::{generate_bitmask, DerivativeOperator, FockState, RandomStateGeneration, SysParams, VarParams};
use impurity::monte_carlo::compute_mean_energy;

const SIZE: usize = 4;
const NFIJ: usize = 4*SIZE*SIZE;
const NVIJ: usize = SIZE*SIZE;
const NGI: usize = SIZE;
const NPARAMS: usize = NFIJ + NGI + NVIJ;
const NELEC: usize = SIZE;
const NMCSAMP: usize = 1_000;
const NMCWARMUP: usize = 1_000;
const CLEAN_UPDATE_FREQUENCY: usize = 16;
const TOLERENCE_SHERMAN_MORRISSON: f64 = 1e-12;
const TOLERENCE_SINGULARITY: f64 = 1e-12;
const CONS_U: f64 = 1.0;
const CONS_T: f64 = -1.0;
const EPSILON_CG: f64 = 1e-10;

pub const HOPPINGS: [f64; SIZE*SIZE] = [
    0.0, 1.0, 1.0, 0.0,
    1.0, 0.0, 0.0, 1.0,
    1.0, 0.0, 0.0, 1.0,
    0.0, 1.0, 1.0, 0.0
];

pub static mut FIJ: [f64; NFIJ] = [
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

pub static mut VIJ: [f64; NVIJ] = [
    0.0, 2.0, 2.0, 2.0,
    2.0, 0.0, 2.0, 2.0,
    2.0, 2.0, 0.0, 2.0,
    2.0, 2.0, 2.0, 0.0,
];

pub static mut GI: [f64; NGI] = [
    -1.0, -1.0, -1.0, -1.0
];

fn log_system_parameters(sys: &SysParams) {
    info!("System parameter SIZE = {}", SIZE);
    info!("System parameter NELEC = {}", NELEC);
    info!("System parameter NMCSAMP = {}", NMCSAMP);
    info!("System parameter NMCWARMUP = {}", NMCWARMUP);
    info!("System parameter CONS_U = {}", sys.cons_u);
    info!("System parameter CONS_T = {}", sys.cons_t);
    debug!("System parameter CLEAN_UPDATE_FREQUENCY = {}", CLEAN_UPDATE_FREQUENCY);
    debug!("System parameter TOLERENCE_SHERMAN_MORRISSON = {}", TOLERENCE_SHERMAN_MORRISSON);
    for i in 0..4*SIZE*SIZE {
        unsafe {
            if FIJ[i] == 0.0 {continue;}
            if i < SIZE*SIZE {
                debug!("F_[{},{}]^[up, up]={}", i/SIZE, i%SIZE, FIJ[i]);
            }
            else if i < 2*SIZE*SIZE {
                debug!("F_[{},{}]^[up, down]={}", i/SIZE - SIZE, i%SIZE, FIJ[i]);
            }
            else if i < 3*SIZE*SIZE {
                debug!("F_[{},{}]^[down, up]={}", i/SIZE - 2*SIZE, i%SIZE, FIJ[i]);
            }
            else {
                debug!("F_[{},{}]^[down, down]={}", i/SIZE - 3*SIZE, i%SIZE, FIJ[i]);
            }
        }
    }
    for i in 0..SIZE*SIZE {
        unsafe {
            if VIJ[i] == 0.0 {continue;}
            debug!("V_[{},{}]={}", i/SIZE, i%SIZE, VIJ[i]);
        }
    }
    for i in 0..SIZE {
        unsafe {
            if GI[i] == 0.0 {continue;}
            debug!("G_[{}]={}", i, GI[i]);
        }
    }
}

fn main() {
    // Initialize logger
    env_logger::init();
    let bitmask = generate_bitmask(&HOPPINGS, SIZE);
    let system_params = SysParams {
        size: SIZE,
        nelec: NELEC,
        array_size: (SIZE + 7) / 8,
        cons_t: CONS_T,
        cons_u: CONS_U,
        nfij: NFIJ,
        nvij: NVIJ,
        ngi: NGI,
        transfert_matrix: &HOPPINGS,
        hopping_bitmask: &bitmask,
        clean_update_frequency: CLEAN_UPDATE_FREQUENCY,
        nmcsample: NMCSAMP,
        nmcwarmup: NMCWARMUP,
        tolerance_sherman_morrison: TOLERENCE_SHERMAN_MORRISSON,
        tolerance_singularity: TOLERENCE_SINGULARITY
    };
    log_system_parameters(&system_params);

    let mut rng = rand::thread_rng();
    //let parameters = generate_random_params(&mut rng);
    let parameters = unsafe { VarParams {
        fij: slice(addr_of!(FIJ) as *const f64, NFIJ),
        gi: slice(addr_of!(GI) as *const f64, NGI),
        vij: slice(addr_of!(VIJ) as *const f64, NVIJ)
    }};

    let state: FockState<u8> = {
        let mut tmp: FockState<u8> = FockState::generate_from_nelec(&mut rng, NELEC, SIZE);
        while tmp.spin_up.count_ones() != tmp.spin_down.count_ones() {
            tmp = FockState::generate_from_nelec(&mut rng, NELEC, SIZE);
        }
        tmp
    };

    let mut otilde: Vec<f64> = vec![0.0; (NFIJ + NVIJ + NGI) * NMCSAMP];
    let mut expvalo: Vec<f64> = vec![0.0; NFIJ + NVIJ + NGI];
    let mut expval_ho: Vec<f64> = vec![0.0; NFIJ + NVIJ + NGI];
    let mut visited: Vec<usize> = vec![0; NMCSAMP];
    let mut derivative = DerivativeOperator {
        o_tilde: &mut otilde,
        expval_o: &mut expvalo,
        ho: &mut expval_ho,
        n: (NFIJ + NVIJ + NGI) as i32,
        nsamp: NMCSAMP as f64,
        mu: -1,
        visited: &mut visited,
        pfaff_off: NGI + NVIJ,
        jas_off: NGI,
    };

    info!("Initial State: {}", state);
    info!("Initial Nelec: {}, {}", state.spin_down.count_ones(), state.spin_up.count_ones());
    info!("Nsites: {}", state.n_sites);

    let mean_energy = compute_mean_energy(&mut rng, state, &parameters, &system_params, &mut derivative);

    let mut x0 = vec![0.0; NFIJ + NVIJ + NGI];
    x0[(NGI + NVIJ)..(NGI + NVIJ + NFIJ)].copy_from_slice(parameters.fij);
    x0[NGI..(NGI + NVIJ)].copy_from_slice(parameters.vij);
    x0[0..NGI].copy_from_slice(parameters.gi);

    let mut b: Vec<f64> = vec![0.0; derivative.n as usize];
    unsafe {
        let incx = 1;
        let incy = 1;
        dscal(derivative.n, 1.0 / (NMCSAMP as f64), derivative.ho, incx);
        dscal(derivative.n, 1.0 / (NMCSAMP as f64), derivative.expval_o, incx);
        daxpy(derivative.n, -mean_energy, derivative.expval_o, incx, derivative.ho, incy);
        dcopy(derivative.n, derivative.ho, incx, &mut b, incy);
    }
    conjugate_gradiant(&derivative, &mut b, &mut x0, EPSILON_CG, 4, NPARAMS as i32);
    info!("Need to update parameters with: {:?}", x0);
    info!("Visited: {:?}", derivative.visited);
}
