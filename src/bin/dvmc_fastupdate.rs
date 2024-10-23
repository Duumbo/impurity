use blas::{daxpy, dcopy, ddot, dnrm2, dscal, idamax};
use log::{debug, error, info};
use rand_mt::Mt64;
use indicatif::{ProgressBar, ProgressStyle};
use std::fs::File;
use std::io::Write;

use impurity::optimisation::{conjugate_gradiant, spread_eigenvalues};
use impurity::{generate_bitmask, DerivativeOperator, FockState, RandomStateGeneration, SysParams, VarParams};
use impurity::monte_carlo::compute_mean_energy;

const SEED: u64 = 1434;
const SIZE: usize = 4;
const NFIJ: usize = 4*SIZE*SIZE;
const NVIJ: usize = SIZE*SIZE;
const NGI: usize = SIZE;
const NPARAMS: usize = NFIJ + NGI + NVIJ;
const NELEC: usize = SIZE;
const NMCSAMP: usize = 100_000;
const NMCWARMUP: usize = 1000;
const MCSAMPLE_INTERVAL: usize = 1;
const NTHREADS: usize = 6;
const CLEAN_UPDATE_FREQUENCY: usize = 8;
const TOLERENCE_SHERMAN_MORRISSON: f64 = 1e-12;
const TOLERENCE_SINGULARITY: f64 = 1e-12;
const CONS_U: f64 = 1.0;
const CONS_T: f64 = -1.0;
const EPSILON_CG: f64 = 1e-16;
const EPSILON_SPREAD: f64 = 0.0;
const OPTIMISATION_TIME_STEP: f64 = 1e-2;
const NOPTITER: usize = 10_000;

pub const HOPPINGS: [f64; SIZE*SIZE] = [
    //0.0, 1.0, 1.0, 0.0,
    0.0, 1.0, 1.0, 0.0,
    1.0, 0.0, 0.0, 1.0,
    1.0, 0.0, 0.0, 1.0,
    0.0, 1.0, 1.0, 0.0
];

fn sq(a: f64) -> f64 {
    <f64>::abs(a) * <f64>::abs(a)
}

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

fn log_system_parameters(e: f64, ae: f64, corr_time: f64, fp: &mut File, params: &VarParams, sys: &SysParams) {
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
    debug!("\n{}", params);
    let mut params = format!("{:+>.05e}  ", e).to_owned();
    params.push_str(&format!("{:+>.05e}  ", ae).to_owned());
    params.push_str(&format!("{:+>.05e}  ", corr_time).to_owned());
    for i in 0..4*SIZE*SIZE {
        let a = format!("{:+>.05e} ", fij[i]);
        params.push_str(&a);
    }
    for i in 0..SIZE*SIZE {
        let a = format!("{:+>.05e} ", vij[i]);
        params.push_str(&a);
    }
    for i in 0..SIZE {
        let a = format!("{:+>.05e} ", gi[i]);
        params.push_str(&a);
    }
    params.push_str(&"\n");
    fp.write(params.as_bytes()).unwrap();
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
    let mut fp = File::create("params").unwrap();
    let mut statesfp = File::create("states").unwrap();
    let mut save: bool = true;
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
        mcsample_interval: MCSAMPLE_INTERVAL,
        tolerance_sherman_morrison: TOLERENCE_SHERMAN_MORRISSON,
        tolerance_singularity: TOLERENCE_SINGULARITY
    };

    let mut rng = Mt64::new(SEED);
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
    //let mut all_params = vec![
    //-1.562407471562315485e-06,
    //-1.562407471562315485e-06,
    //0.0,
    //-8.118842052166648227e-01,
    //-8.118842052166648227e-01,
    //0.0,
    //0.0, 0.0, 0.0, 0.0,
    ////2.992823494391085859e-01,
    ////3.164984479189808504e-09,
    ////0.0,
    ////5.126018557775564588e-01,
    ////-2.992823494391085859e-01,
    ////0.0,
    ////-3.164984479189808504e-09,
    ////-5.126018557775564588e-01,
    //0.04813118562079141	,
    //1.0625398526882723	,
    //0.08433353658389342	,
    //0.002722470871706029,
    //0.07270002762085896	,
    //0.026989164590497917,
    //0.007555596176108393,
    //0.046284058565227465,
    //0.0, 0.0, 0.0, 0.0,
    //];
    // Optimised params 2sites
    //let mut all_params = vec![
    //     2.992823494391085859e-01,
    //    -8.118842052166648227e-01,
    //    0.0,
    //    -5.126018557775564588e-01,
    //    -5.126018557775564588e-01,
    //    0.0,
    //    //0.000000000000000000e+00,
    //    0.0, 0.0, 0.0, 0.0,
    //    1.085729148576013436e-01,
    //    3.715326522320877012e-01,
    //    3.716078461869162797e-01,
    //    3.298336802820764357e-01,
    //    0.0, 0.0, 0.0, 0.0,
    //    0.0, 0.0, 0.0, 0.0,
    //    ];
    let (gi, params) = all_params.split_at_mut(NGI);
    let (vij, fij) = params.split_at_mut(NVIJ);
    let mut parameters = VarParams {
        fij,
        gi,
        vij,
        size: SIZE
    };
    println!("{}", mean_energy_analytic_2sites(&parameters, &system_params));
    //log_system_parameters(0.0, &mut fp, &parameters, &system_params);

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
        nsamp_int: MCSAMPLE_INTERVAL,
        mu: -1,
        visited: &mut visited,
        pfaff_off: NGI + NVIJ,
        jas_off: NGI,
        epsilon: EPSILON_SPREAD,
    };

    info!("Initial State: {}", state);
    info!("Initial Nelec: {}, {}", state.spin_down.count_ones(), state.spin_up.count_ones());
    info!("Nsites: {}", state.n_sites);

    let opt_progress_bar = ProgressBar::new(NOPTITER as u64);
    opt_progress_bar.set_prefix("Optimisation Progress: ");
    opt_progress_bar.set_style(ProgressStyle::with_template("[{elapsed_precise}] {prefix} {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}")
    .unwrap()
    .progress_chars("##-"));

    for opt_iter in 0..NOPTITER {
        let (mean_energy, accumulated_states, correlation_time) = {
            compute_mean_energy(&mut rng, state, &parameters, &system_params, &mut derivative)
        };
        if save {
            let mut out_str: String = String::new();
            for s in accumulated_states.iter() {
                out_str.push_str(&format!("{}\n", s));
            }
            statesfp.write(out_str.as_bytes()).unwrap();
            save = false;
        }

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
        // Sorella Louche stuff
        //info!("Rescaling the parameters.");
        //let scale: f64 = unsafe {
        //    let incx = 1;
        //    let incy = 1;
        //    ddot(derivative.n, derivative.expval_o, incx, parameters.gi, incy)
        //};
        //info!("Scale by : {}", scale);
        //let ratio = 1.0 / (scale + 1.0);
        //unsafe {
        //    let incx = 1;
        //    dscal(NPARAMS as i32, ratio, parameters.gi, incx)
        //}
        //info!("Scaled parameters by ratio = {}", ratio);

        // JastrowGutzwiller Shifting
        let mut shift = 0.0;
        for i in 0..NGI {
            shift += parameters.gi[i];
        }
        for i in 0..NVIJ {
            shift += parameters.vij[i];
        }
        shift = shift / (NGI + NVIJ) as f64;
        for i in 0..NGI {
            parameters.gi[i] -= shift;
        }
        for i in 0..NVIJ {
            parameters.vij[i] -= shift;
        }
        // Slater Rescaling
        unsafe {
            let incx = 1;
            let max = parameters.fij[idamax(NFIJ as i32, parameters.fij, incx) - 1];
            info!("Max was: {}", max);
            dscal(NFIJ as i32, 1.0 / max, parameters.fij, incx);
        }
        let analytic_energy = mean_energy_analytic_2sites(&parameters, &system_params);
        log_system_parameters(mean_energy, analytic_energy, correlation_time, &mut fp, &parameters, &system_params);
        zero_out_derivatives(&mut derivative);
        let opt_delta = unsafe {
            let incx = 1;
            dnrm2(derivative.n, &x0, incx)
        };
        //error!("Changed parameters by norm {}", opt_delta);
        opt_progress_bar.inc(1);
        opt_progress_bar.set_message(format!("Changed parameters by norm: {:+>.05e} Current energy: {:+>.05e}", opt_delta, mean_energy));
    }
    opt_progress_bar.finish()
}
