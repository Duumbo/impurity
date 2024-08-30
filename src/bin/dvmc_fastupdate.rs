use blas::{daxpy, dcopy, ddot, dnrm2, dscal};
use impurity::optimisation::{conjugate_gradiant, spread_eigenvalues};
use log::{debug, error, info};

use impurity::{generate_bitmask, DerivativeOperator, FockState, RandomStateGeneration, SysParams, VarParams};
use impurity::monte_carlo::compute_mean_energy;

const SIZE: usize = 4;
const NFIJ: usize = 4*SIZE*SIZE;
const NVIJ: usize = SIZE*SIZE;
const NGI: usize = SIZE;
const NPARAMS: usize = NFIJ + NGI + NVIJ;
const NELEC: usize = SIZE;
const NMCSAMP: usize = 10_000;
const NMCWARMUP: usize = 1_000;
const CLEAN_UPDATE_FREQUENCY: usize = 8;
const TOLERENCE_SHERMAN_MORRISSON: f64 = 1e-12;
const TOLERENCE_SINGULARITY: f64 = 1e-12;
const CONS_U: f64 = 1.0;
const CONS_T: f64 = -1.0;
const EPSILON_CG: f64 = 1e-10;
const EPSILON_SPREAD: f64 = 0.0;
const OPTIMISATION_TIME_STEP: f64 = 1e-2;
const NOPTITER: usize = 1_000;

pub const HOPPINGS: [f64; SIZE*SIZE] = [
    0.0, 1.0, 1.0, 0.0,
    1.0, 0.0, 0.0, 1.0,
    1.0, 0.0, 0.0, 1.0,
    0.0, 1.0, 1.0, 0.0
];

fn log_system_parameters(params: &VarParams, sys: &SysParams) {
    let fij = &params.fij;
    let vij = &params.vij;
    let gi = &params.gi;
    info!("System parameter SIZE = {}", SIZE);
    info!("System parameter NELEC = {}", NELEC);
    info!("System parameter NMCSAMP = {}", NMCSAMP);
    info!("System parameter NMCWARMUP = {}", NMCWARMUP);
    info!("System parameter CONS_U = {}", sys.cons_u);
    info!("System parameter CONS_T = {}", sys.cons_t);
    debug!("System parameter CLEAN_UPDATE_FREQUENCY = {}", CLEAN_UPDATE_FREQUENCY);
    debug!("System parameter TOLERENCE_SHERMAN_MORRISSON = {}", TOLERENCE_SHERMAN_MORRISSON);
    for i in 0..4*SIZE*SIZE {
        if fij[i] == 0.0 {continue;}
        if i < SIZE*SIZE {
            debug!("F_[{},{}]^[up, up]={}", i/SIZE, i%SIZE, fij[i]);
        }
        else if i < 2*SIZE*SIZE {
            debug!("F_[{},{}]^[up, down]={}", i/SIZE - SIZE, i%SIZE, fij[i]);
        }
        else if i < 3*SIZE*SIZE {
            debug!("F_[{},{}]^[down, up]={}", i/SIZE - 2*SIZE, i%SIZE, fij[i]);
        }
        else {
            debug!("F_[{},{}]^[down, down]={}", i/SIZE - 3*SIZE, i%SIZE, fij[i]);
        }
    }
    for i in 0..SIZE*SIZE {
        if vij[i] == 0.0 {continue;}
        debug!("V_[{},{}]={}", i/SIZE, i%SIZE, vij[i]);
    }
    for i in 0..SIZE {
        if gi[i] == 0.0 {continue;}
        debug!("G_[{}]={}", i, gi[i]);
    }
}

fn zero_out_derivatives(der: &mut DerivativeOperator) {
    for i in 0.. (NFIJ + NVIJ + NGI) * NMCSAMP {
        der.o_tilde[i] = 0.0;
    }
    for i in 0..NFIJ + NVIJ + NGI {
        der.expval_o[i] = 0.0;
        der.ho[i] = 0.0;
    }
    for i in 0..NMCSAMP {
        der.visited[i] = 0;
    }
    der.mu = -1;
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

    let mut rng = rand::thread_rng();
    //let parameters = generate_random_params(&mut rng);
    let mut all_params = vec![
        -1.0, -1.0, -1.0, -1.0,
        0.0, 2.0, 2.0, 2.0,
        2.0, 0.0, 2.0, 2.0,
        2.0, 2.0, 0.0, 2.0,
        2.0, 2.0, 2.0, 0.0,
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
    let (gi, params) = all_params.split_at_mut(NGI);
    let (vij, fij) = params.split_at_mut(NVIJ);
    let mut parameters = VarParams {
        fij,
        gi,
        vij
    };
    println!("fij {} {}", parameters.fij.len(), NFIJ);
    log_system_parameters(&parameters, &system_params);

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
        epsilon: EPSILON_SPREAD,
    };

    info!("Initial State: {}", state);
    info!("Initial Nelec: {}, {}", state.spin_down.count_ones(), state.spin_up.count_ones());
    info!("Nsites: {}", state.n_sites);

    for opt_iter in 0..NOPTITER {
        let mean_energy = compute_mean_energy(&mut rng, state, &parameters, &system_params, &mut derivative);
        println!("{}", derivative.mu);

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
        //spread_eigenvalues(&mut derivative);
        conjugate_gradiant(&derivative, &mut b, &mut x0, EPSILON_CG, 4, NPARAMS as i32);
        info!("Need to update parameters with: {:?}", x0);
        unsafe {
            let incx = 1;
            let incy = 1;
            let alpha = - OPTIMISATION_TIME_STEP;
            daxpy(NGI as i32, alpha, &x0, incx, &mut parameters.gi, incy);
            daxpy(NVIJ as i32, alpha, &x0[NGI..NPARAMS], incx, &mut parameters.vij, incy);
            daxpy(NFIJ as i32, alpha, &x0[NGI + NVIJ..NPARAMS], incx, &mut parameters.fij, incy);
        }
        info!("Correctly finished optimisation iteration {}", opt_iter);
        //info!("Rescaling the parameters.");
        let scale: f64 = unsafe {
            let incx = 1;
            let incy = 1;
            ddot(derivative.n, derivative.expval_o, incx, parameters.gi, incy)
        };
        info!("Scale by : {}", scale);
        let ratio = 1.0 / (scale + 1.0);
        unsafe {
            let incx = 1;
            dscal(NPARAMS as i32, ratio, parameters.gi, incx)
        }
        info!("Scaled parameters by ratio = {}", ratio);
        log_system_parameters(&parameters, &system_params);
        zero_out_derivatives(&mut derivative);
        let opt_delta = unsafe {
            let incx = 1;
            dnrm2(derivative.n, &x0, incx)
        };
        error!("Changed parameters by norm {}", opt_delta);
    }
}
