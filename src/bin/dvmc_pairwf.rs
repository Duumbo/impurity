use blas::{daxpy, dcopy, dnrm2, dscal, idamax};
use log::{debug, error, info};
use rand_mt::Mt64;
use indicatif::{ProgressBar, ProgressStyle};
use std::fs::File;
use std::io::Write;

use impurity::optimisation::conjugate_gradiant;
use impurity::{generate_bitmask, mapto_pairwf, DerivativeOperator, FockState, RandomStateGeneration, SysParams, VarParams};
use impurity::monte_carlo::compute_mean_energy;

const SEED: u64 = 1434;
const SIZE: usize = 2;
const NFIJ: usize = SIZE*SIZE;
const NVIJ: usize = SIZE*(SIZE - 1) / 2;
const NGI: usize = SIZE;
const NPARAMS: usize = NFIJ + NGI + NVIJ;
const NELEC: usize = SIZE;
const NMCSAMP: usize = 40_000;
const NBOOTSTRAP: usize = 1;
const NMCWARMUP: usize = 1000;
const MCSAMPLE_INTERVAL: usize = 1;
const _NTHREADS: usize = 1;
const CLEAN_UPDATE_FREQUENCY: usize = 32;
const TOLERENCE_SHERMAN_MORRISSON: f64 = 1e-12;
const TOLERENCE_SINGULARITY: f64 = 1e-12;
const CONS_U: f64 = 1.0;
const CONS_T: f64 = -1.0;
const EPSILON_CG: f64 = 1e-16;
const EPSILON_SHIFT: f64 = 0.02;
const OPTIMISATION_TIME_STEP: f64 = 1e-1;
const OPTIMISATION_DECAY: f64 = 0.0;
const NOPTITER: usize = 1_000;
const KMAX: usize = NPARAMS;
const PARAM_THRESHOLD: f64 = 1e-1;
const OPTIMISE: bool = true;
const OPTIMISE_GUTZ: bool = true;
const OPTIMISE_JAST: bool = true;
const OPTIMISE_ORB: bool = true;

pub const HOPPINGS: [f64; SIZE*SIZE] = [
    0.0, 1.0, 1.0, 0.0,
    //0.0, 1.0, 1.0, 0.0,
    //1.0, 0.0, 0.0, 1.0,
    //1.0, 0.0, 0.0, 1.0,
    //0.0, 1.0, 1.0, 0.0
];

fn save_otilde(fp: &mut File, der: &DerivativeOperator) {
    let width = 16;
    let mut o_tilde = "".to_owned();
    for mu in 0..(der.mu + 1) as usize {
        for n in 0..der.n as usize {
            o_tilde.push_str(&format!("{:>width$.04e}", der.o_tilde[n + mu * der.n as usize]));
        }
        o_tilde.push_str("\n");
    }
    fp.write(&o_tilde.as_bytes()).unwrap();
}

fn sq(a: f64) -> f64 {
    <f64>::abs(a) * <f64>::abs(a)
}

fn norm(par: &VarParams) -> f64 {
    let f00ud = par.fij[0 + 0 * SIZE];
    let f11ud = par.fij[1 + 1 * SIZE];
    let f01ud = par.fij[0 + 1 * SIZE];
    let f10ud = par.fij[1 + 0 * SIZE];
    let g0 = par.gi[0];
    let g1 = par.gi[1];
    let v = par.vij[0];
    let a = <f64>::exp(2.0 * g0 - 2.0 * v)*sq(<f64>::abs(f00ud));
    let b = <f64>::exp(2.0 * g1 - 2.0 * v)*sq(<f64>::abs(f11ud));
    let e = sq(<f64>::abs(f10ud));
    let f = sq(<f64>::abs(f01ud));
    a + b + e + f
}

fn mean_energy_analytic_2sites(par: &VarParams, _sys: &SysParams) -> f64 {
    let a = par.fij[1 + 0 * SIZE]
        + par.fij[0 + 1 * SIZE];
    let b = par.fij[0 + 0 * SIZE]
        * <f64>::exp(par.gi[0]-par.vij[0]);
    let c = par.fij[1 + 1 * SIZE]
        * <f64>::exp(par.gi[1]-par.vij[0]);
    let d = 2.0 * CONS_T * (b + c) * a;
    let e = sq(par.fij[1 + 1 * SIZE])
        * <f64>::exp(2.0*par.gi[1]-2.0*par.vij[0]) * CONS_U;
    let f = sq(par.fij[0 + 0 * SIZE])
        * <f64>::exp(2.0*par.gi[0]-2.0*par.vij[0]) * CONS_U;
    (d + e + f) / norm(par)
}

fn log_system_parameters(e: f64, ae: f64, deltae: f64, corr_time: f64, fp: &mut File, params: &VarParams, sys: &SysParams, dpar: &[f64]) {
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
    params.push_str(&format!("{:+>.05e}  ", deltae).to_owned());
    params.push_str(&format!("{:+>.05e}  ", corr_time).to_owned());
    for i in 0..SIZE {
        let a = format!("{:+>.05e} ", gi[i]);
        params.push_str(&a);
    }
    for i in 0..NVIJ {
        let a = format!("{:+>.05e} ", vij[i]);
        params.push_str(&a);
    }
    for i in 0..4 {
        let a = format!("{:+>.05e} ", fij[3+i]);
        params.push_str(&a);
    }
    for i in 0..7 {
        let a = format!("{:+>.05e} ", dpar[i]);
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
    writeln!(fp, "{}", format!("# {} {} {}", SIZE, NMCSAMP, NOPTITER)).unwrap();
    let mut der_fp = File::create("derivative").unwrap();
    let mut wder_fp = File::create("work_derivative").unwrap();
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
        nbootstrap: NBOOTSTRAP,
        nmcwarmup: NMCWARMUP,
        mcsample_interval: MCSAMPLE_INTERVAL,
        tolerance_sherman_morrison: TOLERENCE_SHERMAN_MORRISSON,
        tolerance_singularity: TOLERENCE_SINGULARITY,
        pair_wavefunction: true,
    };

    let mut rng = Mt64::new(SEED);
    //let parameters = generate_random_params(&mut rng);
    //let mut all_params = Vec::with_capacity(NGI + NVIJ + NFIJ);
    //for _ in 0..(NGI + NVIJ + NFIJ) {
    //    all_params.push(rng.gen());
    //}
    //let (gi, params) = all_params.split_at_mut(NGI);
    //let (vij, mut fij) = params.split_at_mut(NVIJ);
    //   0.000000000000000000e+00  5.012713072533996646e-09
    let mut fij = [0.41, -0.18, 0.11, 0.84];
    let mut general_fij = [
        0.0, 0.0, 0.0, 0.0,
        1.093500753438337580e-01, 3.768419990611672210e-01, 3.769186909982900624e-01, 3.322533463612635796e-01,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
    ];
    let mut vij = [5.079558854017672820e-01];
    let mut gi = [3.016937239100276336e-01, -8.096496093117950821e-01];
    //let mut general_fij: Vec<f64> = vec![0.0; 4*NFIJ];
    let mut parameters = VarParams {
        fij: &mut general_fij,
        gi: &mut gi,
        vij: &mut vij,
        size: SIZE
    };
    unsafe {
        dcopy(NFIJ as i32, &fij[0..NFIJ], 1, &mut parameters.fij[NFIJ..2*NFIJ], 1);
    }
    println!("{}", mean_energy_analytic_2sites(&parameters, &system_params));
    //log_system_parameters(0.0, &mut fp, &parameters, &system_params);

    let state: FockState<u8> = {
        let mut tmp: FockState<u8> = FockState::generate_from_nelec(&mut rng, NELEC, SIZE);
        while tmp.spin_up.count_ones() != tmp.spin_down.count_ones() {
            tmp = FockState::generate_from_nelec(&mut rng, NELEC, SIZE);
        }
        tmp
    };

    let mut otilde: Vec<f64> = vec![0.0; (4*NFIJ + NVIJ + NGI) * NMCSAMP];
    let mut work_otilde: Vec<f64> = vec![0.0; (NFIJ + NVIJ + NGI) * NMCSAMP];
    let mut expvalo: Vec<f64> = vec![0.0; 4*NFIJ + NVIJ + NGI];
    let mut work_expvalo: Vec<f64> = vec![0.0; NFIJ + NVIJ + NGI];
    let mut expval_ho: Vec<f64> = vec![0.0; 4*NFIJ + NVIJ + NGI];
    let mut work_expval_ho: Vec<f64> = vec![0.0; NFIJ + NVIJ + NGI];
    let mut visited: Vec<usize> = vec![0; NMCSAMP];
    let mut work_visited: Vec<usize> = vec![0; NMCSAMP];
    let mut derivative = DerivativeOperator {
        o_tilde: &mut otilde,
        expval_o: &mut expvalo,
        ho: &mut expval_ho,
        n: (4*NFIJ + NVIJ + NGI) as i32,
        nsamp: NMCSAMP as f64,
        nsamp_int: MCSAMPLE_INTERVAL,
        mu: -1,
        visited: &mut visited,
        pfaff_off: NGI + NVIJ,
        jas_off: NGI,
        epsilon: EPSILON_SHIFT,
    };
    let mut work_derivative = DerivativeOperator {
        o_tilde: &mut work_otilde,
        expval_o: &mut work_expvalo,
        ho: &mut work_expval_ho,
        n: (NFIJ + NVIJ + NGI) as i32,
        nsamp: NMCSAMP as f64,
        nsamp_int: MCSAMPLE_INTERVAL,
        mu: -1,
        visited: &mut work_visited,
        pfaff_off: NGI + NVIJ,
        jas_off: NGI,
        epsilon: EPSILON_SHIFT,
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
        let (mean_energy, accumulated_states, deltae, correlation_time) = {
            compute_mean_energy(&mut rng, state, &parameters, &system_params, &mut derivative)
        };
        if true {
            let mut statesfp = File::create("states").unwrap();
            let mut out_str: String = String::new();
            for s in accumulated_states.iter() {
                out_str.push_str(&format!("{}\n", s));
            }
            statesfp.write(out_str.as_bytes()).unwrap();
            save = false;
        }
        // Copy out the relevant terms.
        work_derivative.mu = derivative.mu;
        let mut i = 0;
        for elem in derivative.visited.iter() {
            work_derivative.visited[i] = *elem;
            i += 1;
        }
        mapto_pairwf(&derivative, &mut work_derivative, &system_params);

        let mut x0 = vec![0.0; NFIJ + NVIJ + NGI];
        x0[(NGI + NVIJ)..(NGI + NVIJ + NFIJ)].copy_from_slice(&fij);
        x0[NGI..(NGI + NVIJ)].copy_from_slice(parameters.vij);
        x0[0..NGI].copy_from_slice(parameters.gi);

        // 68 misawa
        let mut b: Vec<f64> = vec![0.0; work_derivative.n as usize];
        unsafe {
            let incx = 1;
            let incy = 1;
            daxpy(work_derivative.n, -mean_energy, work_derivative.expval_o, incx, work_derivative.ho, incy);
            dcopy(work_derivative.n, work_derivative.ho, incx, &mut b, incy);
        }
        save_otilde(&mut der_fp, &derivative);
        save_otilde(&mut wder_fp, &work_derivative);
        //spread_eigenvalues(&mut derivative);
        //println!("x0 in = {:?}", x0);
        let mut flag: bool = true;
        let ignored_columns = conjugate_gradiant(&work_derivative, &mut b, &mut x0, EPSILON_SHIFT, KMAX, NPARAMS as i32, PARAM_THRESHOLD);
        let mut delta_alpha = vec![0.0; NPARAMS];
        let mut j: usize = 0;
        for i in 0..NPARAMS {
            if ignored_columns[i] {
                continue;
            }
            delta_alpha[i] = x0[j];
            j += 1;
            if !<f64>::is_finite(delta_alpha[i]) {
                flag = false;
            }
        }
        if OPTIMISE {
            unsafe {
                let incx = 1;
                let incy = 1;
                let alpha = - OPTIMISATION_TIME_STEP * <f64>::exp(- (opt_iter as f64) * OPTIMISATION_DECAY);
                if OPTIMISE_GUTZ {
                    daxpy(NGI as i32, alpha, &delta_alpha, incx, &mut parameters.gi, incy);
                }
                if OPTIMISE_JAST {
                    daxpy(NVIJ as i32, alpha, &delta_alpha[NGI..NPARAMS], incx, &mut parameters.vij, incy);
                }
                if OPTIMISE_ORB {
                    daxpy(NFIJ as i32, alpha, &delta_alpha[NGI + NVIJ..NPARAMS], incx, &mut fij, incy);
                }
            }
            info!("Correctly finished optimisation iteration {}", opt_iter);
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
        }
        // HARD CODE vij = vji
        // Slater Rescaling
        unsafe {
            let incx = 1;
            let max = fij[idamax(NFIJ as i32, &fij, incx) - 1];
            if <f64>::abs(max) < 1e-16 {
                error!("Pfaffian parameters are all close to 0.0. Rescaling might bring noise.");
                panic!("Undefined behavior.");
            }
            info!("Max was: {}", max);
            dscal(NFIJ as i32, 1.0 / max, &mut fij, incx);
        }
        unsafe {
            dcopy(
                NFIJ as i32,
                &fij,
                1,
                &mut parameters.fij[NFIJ..2*NFIJ],
                1
            );
        }
        let analytic_energy = mean_energy_analytic_2sites(&parameters, &system_params);
        log_system_parameters(mean_energy, analytic_energy, deltae, correlation_time, &mut fp, &parameters, &system_params, &delta_alpha);
        zero_out_derivatives(&mut derivative);
        let opt_delta = unsafe {
            let incx = 1;
            dnrm2(work_derivative.n, &delta_alpha, incx)
        };
        //error!("Changed parameters by norm {}", opt_delta);
        opt_progress_bar.inc(1);
        opt_progress_bar.set_message(format!("Changed parameters by norm: {:+>.05e} Current energy: {:+>.05e}", opt_delta, mean_energy));
    }
    opt_progress_bar.finish()
}
