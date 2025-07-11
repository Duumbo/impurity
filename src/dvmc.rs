use log::{info, error};
use rand::distributions::{Distribution, Standard};
use rand::Rng;
use std::fmt::{Display, Debug};
use blas::{idamax, daxpy, dcopy, dscal, dnrm2};
use std::thread;

use crate::{BitOps, DerivativeOperator, FockState, SysParams, VarParams};
use crate::monte_carlo::{compute_mean_energy, compute_mean_energy_exact};
use crate::optimisation::{conjugate_gradiant, ParameterMap};

pub const ADAMS_BASHFORTH_COEFS: [f64; 15] = [
    1.0, 3.0/2.0, -1.0/2.0, 23.0/12.0, -16.0/12.0, 5.0/12.0, 55.0/24.0, -59.0/24.0, 37.0/24.0, -9.0/24.0, 1901.0/720.0, -2774.0/720.0, 2616.0/720.0, -1274.0/720.0, 251.0/720.0
];

#[derive(Debug)]
pub enum EnergyOptimisationMethod {
    ExactInverse,
    ConjugateGradiant,
}

#[derive(Debug)]
pub enum EnergyComputationMethod {
    MonteCarlo,
    ExactSum,
}

#[derive(Debug)]
pub struct VMCParams {
    pub noptiter: usize,
    pub compute_energy_method: EnergyComputationMethod,
    pub optimise_energy_method: EnergyOptimisationMethod,
    pub epsilon: f64,
    pub epsilon_cg: f64,
    pub kmax: usize,
    pub dt: f64,
    pub optimisation_decay: f64,
    pub nthreads: usize,
    pub threshold: f64,
    pub optimise: bool,
    pub optimise_gutzwiller: bool,
    pub optimise_jastrow: bool,
    pub optimise_orbital: bool,
    pub conv_param_threshold: f64,
    pub filter_before_shift: bool,
    pub adams_bashforth_order: usize,
}

//fn merge_derivatives(der: &mut DerivativeOperator, der_vec: &mut [DerivativeOperator], param_map: &GenParameterMap, nmc: usize, nthreads: usize, x0: &mut [f64]){
//    let mut acc_expvalo = &mut der_vec[0].expval_o;
//    let mut acc_ho = &mut der_vec[0].ho;
//    let mut acc_visited = &mut der_vec[0].visited;
//    let mut k = der_vec[0].mu as usize;
//    // Allocate only if you need to
//    if nthreads != 1 {
//        k = 0;
//        let mut expvalo = vec![0.0; param_map.gendim as usize].into_boxed_slice();
//        let mut ho = vec![0.0; param_map.gendim as usize].into_boxed_slice();
//        let mut visited = vec![0; nmc * nthreads].into_boxed_slice();
//        acc_expvalo = &mut expvalo;
//        acc_ho = &mut ho;
//        acc_visited = &mut visited;
//        for i in 0..nthreads {
//            unsafe {
//                let incx = 1;
//                let incy = 1;
//                daxpy(der_vec[i].n, 1.0 / nthreads as f64, &der_vec[i].expval_o, incx, &mut acc_expvalo, incy);
//                daxpy(der_vec[i].n, 1.0 / nthreads as f64, &der_vec[i].ho, incx, &mut acc_ho, incy);
//            }
//            for j in 0..der_vec[i].mu as usize {
//                acc_visited[k] = der_vec[i].visited[j];
//                k += 1;
//            }
//            // TODO compactify otilde
//        }
//        param_map.update_reduced_representation_no_otilde(der, k as i32, &acc_expvalo, &acc_ho, &acc_visited, x0);
//    }
//    else {
//        param_map.update_reduced_representation_no_otilde(der, k as i32, &acc_expvalo, &acc_ho, &acc_visited, x0);
//    }
//}

fn zero_out_derivatives(der: &mut DerivativeOperator, sys: &SysParams) {
    for i in 0.. (der.n as usize) * sys.nmcsample {
        der.o_tilde[i] = 0.0;
    }
    for i in 0..der.n as usize {
        der.expval_o[i] = 0.0;
        der.ho[i] = 0.0;
    }
    for i in 0..sys.nmcsample {
        der.visited[i] = 0;
    }
    der.mu = -1;
}

fn update_initial_state<T: BitOps + From<u8> + Display + Debug + Send + Sync>(
    states: &mut [FockState<T>],
    acc_states: &[FockState<T>],
    nmcsample: usize,
    nthreads: usize
) {
    for i in 0..nthreads {
        states[i] = acc_states[(i + 1) * nmcsample - 1];
    }
}

fn parallel_monte_carlo<T, R>(
    rngs: &mut [&mut R],
    initial_state: &[FockState<T>],
    work_der_vec: &mut [DerivativeOperator],
    vmcparams: &VMCParams,
    params: & VarParams,
    sys: & SysParams,
    pmap: &ParameterMap
) -> (f64, Vec<FockState<T>>, f64, f64)
where
    T: BitOps + From<u8> + Display + Debug + Send + Sync,
    R: Rng + ?Sized + Send + Sync,
    Standard: Distribution<T> + Send
{
    // If NTHREADS == 1, don't waste time to spawn threads.
    if vmcparams.nthreads == 1 {
        let out = compute_mean_energy(
                        rngs[0],
                        initial_state[0],
                        params,
                        sys,
                        &mut work_der_vec[0],
                        pmap
                    );
        return out;
    }

    let mut res_vec = Vec::new();
    // Scoped thread because threads need to accept not static stack parameters
    thread::scope(|scope| {
        let threads: Vec<_> = (0..vmcparams.nthreads)
            .map(|idx| {
                // Wrestle the borrow checker
                let wder_ptr = &mut work_der_vec[idx] as *mut _;
                let rng_ptr = &mut *rngs[idx] as *mut R;
                let state_ptr = &initial_state[idx] as *const _;
                let rng = unsafe { &mut *rng_ptr};
                let wder = unsafe { &mut *wder_ptr};
                let state = unsafe { & *state_ptr};
                scope.spawn(
                || {
                    let out = compute_mean_energy(
                        rng,
                        *state,
                        params,
                        sys,
                        wder,
                        pmap
                    );
                    out
                }
                )
            })
        .collect();

        for handle in threads {
            res_vec.push(handle.join().unwrap());
        }
    });
    let mut out_energy = 0.0;
    let mut out_states_vec = Vec::new();
    let mut out_de = 0.0;
    let mut out_corrtime = 0.0;
    for thread in 0..vmcparams.nthreads {
        //println!("Energy of thread {} = {}", thread, res_vec[thread].0);
        out_energy += res_vec[thread].0;
        out_de += res_vec[thread].2;
        out_corrtime += res_vec[thread].3;
        out_states_vec.append(&mut res_vec[thread].1);
    }

    //panic!("Stop!");
    //(res_vec[0].0, res_vec[0].1.clone(), res_vec[0].2, res_vec[0].3)
    (out_energy / vmcparams.nthreads as f64, out_states_vec, out_de / vmcparams.nthreads as f64, out_corrtime / vmcparams.nthreads as f64)
}

//fn compactify_otilde(der: &mut DerivativeOperator, mu_vec: &[i32], nmc: usize, nthreads: usize) {
//    if nthreads == 1 {
//        return;
//    }
//    let mut current_mu = mu_vec[0];
//    for i in 1..nthreads {
//        for j in 0..mu_vec[i] as usize {
//            der.o_tilde[j + (der.n * current_mu) as usize] = der.o_tilde[j + der.n as usize * nmc * i];
//        }
//        current_mu += mu_vec[i];
//    }
//
//}

fn adams_bashforth_optimisation(
    delta_alpha: &[f64],
    parameter_steps: &mut [f64],
    vmcparams: &VMCParams,
    sys: &SysParams,
    adams_order: &mut usize,
    opt_iter: usize,
    params: &mut VarParams,
    nparams: i32
) {
    let n_full_params = sys.ngi + sys.nvij + sys.nfij;
    unsafe {
        dcopy(n_full_params as i32, &delta_alpha, 1, &mut
            parameter_steps[0..n_full_params],
            1);
    }
    // Adams-Bashforth 2order
    for adams_i in 0.. *adams_order {
        unsafe {
            let coef_off = *adams_order * (*adams_order - 1) / 2;
            let incx = 1;
            let incy = 1;
            let alpha = - ADAMS_BASHFORTH_COEFS[adams_i + coef_off] * vmcparams.dt * <f64>::exp(-
                (opt_iter as f64) * vmcparams.optimisation_decay);
            //let alpha = 1.0;
            if vmcparams.optimise_gutzwiller {
                daxpy(sys.ngi as i32, alpha,
                    &parameter_steps[adams_i*n_full_params..adams_i*n_full_params+sys.ngi],
                    incx, &mut params.gi, incy);
            }
            if vmcparams.optimise_jastrow {
                daxpy(sys.nvij as i32, alpha,
                    &parameter_steps[adams_i*n_full_params+sys.ngi..adams_i*n_full_params+sys.ngi+sys.nvij],
                    incx,
                    &mut params.vij, incy);
            }
            if vmcparams.optimise_orbital {
                daxpy(sys.nfij as i32, alpha,
                    &parameter_steps[adams_i*n_full_params+sys.ngi+sys.nvij..adams_i*n_full_params+sys.ngi+sys.nvij+sys.nfij],
                    incx, &mut params.fij, incy);
            }
        }
    }
    // Copy last params one space back
    for adams_i in 1..vmcparams.adams_bashforth_order {
        let begin_line_old = n_full_params*(vmcparams.adams_bashforth_order - adams_i - 1);
        let end_line_old = n_full_params*(vmcparams.adams_bashforth_order - adams_i);
        let begin_line_new = n_full_params*(vmcparams.adams_bashforth_order - adams_i);
        let end_line_new = n_full_params*(vmcparams.adams_bashforth_order - adams_i + 1);
        unsafe {
            let param_ptr =
                &parameter_steps[begin_line_old..end_line_old] as *const [f64];
            dcopy(nparams, &*param_ptr, 1, &mut
                parameter_steps[begin_line_new..end_line_new], 1);
        }
    }
    if *adams_order < vmcparams.adams_bashforth_order {
        *adams_order += 1;
    }
}

pub fn variationnal_monte_carlo<R: Rng + ?Sized + Send + Sync, T>(
    rng: &mut [&mut R],
    initial_state: &mut [FockState<T>],
    params: &mut VarParams,
    sys: &mut SysParams,
    vmcparams: &VMCParams,
    pmap: &ParameterMap,
) -> (Vec<f64>, usize, Vec<f64>)
where T: BitOps + From<u8> + Display + Debug + Send + Sync, Standard: Distribution<T> + std::marker::Send
{
    let mut output_energy_array = vec![0.0; vmcparams.noptiter * 3];
    let mut output_param_array = vec![0.0; vmcparams.noptiter * (sys.ngi + sys.nvij + sys.nfij)];
    let mut parameter_steps = vec![0.0; (sys.ngi + sys.nvij + sys.nfij) * vmcparams.adams_bashforth_order];
    let mut adams_order = 1;

    let mut x0 = vec![0.0; sys.ngi + sys.nvij + sys.nfij + 3].into_boxed_slice();
    let mut b = vec![0.0; sys.ngi + sys.nvij + sys.nfij + 3].into_boxed_slice();

    let mut work_der_vec = Vec::new();
    for _ in 0..vmcparams.nthreads {
        let work_der = DerivativeOperator::new(
            pmap.nparams as i32 + 3,
            -1,
            match vmcparams.compute_energy_method {
                EnergyComputationMethod::ExactSum => sys.nmcsample as f64,
                EnergyComputationMethod::MonteCarlo => sys.nmcsample as f64,
            },
            sys.mcsample_interval,
            vmcparams.epsilon
        );
        work_der_vec.push(work_der);
    }

    for opt_iter in 0..vmcparams.noptiter {
        sys._opt_iter = opt_iter;

        let (mean_energy, _accumulated_states, deltae, correlation_time) = {
            match vmcparams.compute_energy_method {
                EnergyComputationMethod::MonteCarlo => {
                    let (mean_energy, _accumulated_states, deltae, correlation_time) =
                    parallel_monte_carlo(rng, initial_state, &mut work_der_vec, vmcparams, params, sys, pmap);
                    update_initial_state(initial_state, &_accumulated_states, sys.nmcsample, vmcparams.nthreads);
                    (mean_energy, _accumulated_states, deltae, correlation_time)
                },
                EnergyComputationMethod::ExactSum => {
                    let (mean_energy, _accumulated_states, deltae, correlation_time) =
                    (compute_mean_energy_exact(params, sys, &mut work_der_vec[0], pmap), Vec::with_capacity(0), 0.0, 0.0);
                    (mean_energy, _accumulated_states, deltae, correlation_time)
                },
            }
        };
        // Watch out, derivatives operator are dirty. Exactly three columns are garbage

        // Save energy, error and correlation_time.
        output_energy_array[opt_iter * 3] = mean_energy;
        output_energy_array[opt_iter * 3 + 1] = deltae;
        output_energy_array[opt_iter * 3 + 2] = correlation_time;

        let mut mu_vec: Vec<i32> = vec![0; vmcparams.nthreads];
        for i in 0..vmcparams.nthreads {
            mu_vec[i] = work_der_vec[i].mu;
        }

        // 68 misawa
        for der in work_der_vec.iter_mut() {
            unsafe {
                let incx = 1;
                let incy = 1;
                // TODO correct the energy computed from thread.
                daxpy(der.n, -mean_energy, &der.expval_o, incx, &mut der.ho, incy);
                daxpy(der.n, 1.0, &der.ho, incx, &mut b, incy);
            }
        }
        let b_nrm = unsafe {
            let incx = 1;
            dnrm2(pmap.nparams as i32, &b, incx)
        };
        if b_nrm <= 0.0 {
            println!("Exit early, achieved convergence within {} iteration, reason: b vector norm <= 0.0 (derivatives too small)", opt_iter+1);
            return (output_energy_array, opt_iter, output_param_array);
        }

        let mut _flag: bool = true;
        let ignored_columns = match vmcparams.optimise_energy_method {
            EnergyOptimisationMethod::ExactInverse => {
                //exact_overlap_inverse(&work_der_vec, &mut b, vmcparams.epsilon, pmap.nparams as i32,
                //   vmcparams.threshold)
                todo!()
            },
            EnergyOptimisationMethod::ConjugateGradiant => {
                conjugate_gradiant(&work_der_vec, &mut b, &mut x0, vmcparams.epsilon, vmcparams.kmax,
                    pmap.nparams as i32 + 3, vmcparams.threshold, vmcparams.epsilon_cg,
                    vmcparams.filter_before_shift, vmcparams.nthreads, pmap.ngi, pmap.nvij)
            },
        };

        // Ignore truncated params in SR
        let mut delta_alpha = vec![0.0; pmap.dim as usize + 3];
        let mut j: usize = 0;
        let mut norm2 = 0.0;
        for i in 0..pmap.nparams {
            if ignored_columns[i] {
                continue;
            }
            delta_alpha[i] = b[j];
            norm2 += b[j] * b[j];
            j += 1;
            if !<f64>::is_finite(delta_alpha[i]) {
                _flag = false;
                error!("Parameter update contains NaN or Inf at iter {}.", opt_iter);
                panic!("Undefined behavior");
            }
        }

        // Remap da to parameter space
        let delta_alpha = pmap.reverse_map(&mut delta_alpha);

        // Update variationnal parameters
        if vmcparams.optimise {
            if vmcparams.adams_bashforth_order == 1 {
                unsafe {
                    let incx = 1;
                    let incy = 1;
                    let alpha = - vmcparams.dt * <f64>::exp(- (opt_iter as f64) * vmcparams.optimisation_decay);
                    //let alpha = 1.0;
                    if vmcparams.optimise_gutzwiller {
                        daxpy(sys.ngi as i32, alpha, &delta_alpha, incx, &mut params.gi, incy);
                    }
                    if vmcparams.optimise_jastrow {
                        daxpy(sys.nvij as i32, alpha, &delta_alpha[sys.ngi..pmap.dim], incx,
                            &mut params.vij, incy);
                    }
                    if vmcparams.optimise_orbital {
                        daxpy(sys.nfij as i32, alpha, &delta_alpha[sys.ngi +
                            sys.nvij..pmap.dim], incx, &mut params.fij, incy);
                    }
                }
            } else {
                adams_bashforth_optimisation(
                    &delta_alpha,
                    &mut parameter_steps,
                    vmcparams,
                    sys,
                    &mut adams_order,
                    opt_iter,
                    params,
                    pmap.dim as i32
                    );
            }
            info!("Correctly finished optimisation iteration {}", opt_iter);

            // JastrowGutzwiller Shifting
            let mut shift = 0.0;
            for i in 0..sys.ngi {
                shift += params.gi[i];
            }
            for i in 0..sys.nvij {
                shift += params.vij[i];
            }
            shift = shift / (sys.ngi + sys.nvij) as f64;
            for i in 0..sys.ngi {
                params.gi[i] -= shift;
            }
            for i in 0..sys.nvij {
                params.vij[i] -= shift;
            }
        }

        // Slater Rescaling
        if norm2 <= vmcparams.conv_param_threshold {
            println!("Exit early, achieved convergence within {} iteration, update now under supplied threshold.", opt_iter+1);
            return (output_energy_array, opt_iter, output_param_array);
        }
        unsafe {
            let incx = 1;
            let max = params.fij[idamax(sys.nfij as i32, &params.fij, incx) - 1];
            //let nrm2 = dnrm2(sys.nfij as i32, params.fij, 1);
            if <f64>::abs(max) < 1e-16 {
                error!("Pfaffian params are all close to 0.0 at iter {}. Rescaling might bring noise.", opt_iter);
                panic!("Undefined behavior.");
            }
            info!("Max was: {}", max);
            dscal(sys.nfij as i32, 4.0 / max, &mut params.fij, incx);
            dscal(pmap.dim as i32 + 3, 0.0, &mut x0, incx);
            //dscal(sys.nfij as i32, 2.0 / <f64>::sqrt(nrm2), &mut params.fij, incx);
        }

        // Zero out derivatives
        for i in 0..vmcparams.nthreads {
            zero_out_derivatives(&mut work_der_vec[i], sys);
        }
        // Zero out x0 and b
        unsafe {
            dscal(pmap.dim as i32 + 3, 0.0, &mut x0, 1);
            dscal(pmap.dim as i32 + 3, 0.0, &mut b, 1);
        }

        // Copy to output param vec
        unsafe {
            let incx = 1;
            let incy = 1;
            dcopy(sys.ngi as i32, &params.gi, incx, &mut output_param_array[opt_iter*(sys.ngi+sys.nvij+sys.nfij)..opt_iter*(sys.ngi+sys.nvij+sys.nfij)+sys.ngi], incy);
            dcopy(sys.nvij as i32, &params.vij, incx, &mut output_param_array[opt_iter*(sys.ngi+sys.nvij+sys.nfij)+sys.ngi..opt_iter*(sys.ngi+sys.nvij+sys.nfij)+sys.ngi+sys.nvij], incy);
            dcopy(sys.nfij as i32, &params.fij, incx, &mut output_param_array[opt_iter*(sys.ngi+sys.nvij+sys.nfij)+sys.ngi+sys.nvij..(opt_iter+1)*(sys.ngi+sys.nvij+sys.nfij)-1], incy);
        }
        //print_delta_alpha(&delta_alpha, sys.ngi, sys.nvij, sys.nfij);
        //let opt_delta = unsafe {
        //    let incx = 1;
        //    dnrm2(der.n, &delta_alpha, incx)
        //};
        //error!("Changed params by norm {}", opt_delta);
        //opt_progress_bar.inc(1);
        //opt_progress_bar.set_message(format!("Changed params by norm: {:+>.05e} Current energy: {:+>.05e}", opt_delta, mean_energy));
    }
    //opt_progress_bar.finish()
    (output_energy_array, vmcparams.noptiter, output_param_array)
}
