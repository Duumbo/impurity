use log::{info, error};
use rand::distributions::{Distribution, Standard};
use rand::Rng;
use std::fmt::{Display, Debug};
use blas::{idamax, daxpy, dcopy, dscal};

use crate::{VarParams, DerivativeOperator, SysParams, FockState, BitOps};
use crate::monte_carlo::{compute_mean_energy, compute_mean_energy_exact};
use crate::optimisation::{conjugate_gradiant, exact_overlap_inverse, GenParameterMap, ReducibleGeneralRepresentation};

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
    pub nparams: usize,
    pub threshold: f64,
    pub optimise: bool,
    pub optimise_gutzwiller: bool,
    pub optimise_jastrow: bool,
    pub optimise_orbital: bool,
}

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

pub fn variationnal_monte_carlo<R: Rng + ?Sized, T>(
    rng: &mut R,
    initial_state: FockState<T>,
    params: &mut VarParams,
    sys: &mut SysParams,
    vmcparams: &VMCParams,
    param_map: &GenParameterMap,
) -> Vec<f64>
where T: BitOps + From<u8> + Display + Debug, Standard: Distribution<T>
{
    let mut output_energy_array = vec![0.0; vmcparams.noptiter * 3];

    let mut x0 = vec![0.0; sys.ngi + sys.nvij + sys.nfij].into_boxed_slice();
    let mut b = vec![0.0; sys.ngi + sys.nvij + sys.nfij].into_boxed_slice();
    let mut der = DerivativeOperator::new(
        vmcparams.nparams as i32,
        -1,
        match vmcparams.compute_energy_method {
            EnergyComputationMethod::ExactSum => 1.0,
            EnergyComputationMethod::MonteCarlo => sys.nmcsample as f64,
        },
        sys.mcsample_interval,
        param_map.n_independant_jastrow + param_map.n_independant_gutzwiller,
        param_map.n_independant_gutzwiller,
        vmcparams.epsilon
    );
    let mut work_der = param_map.mapto_general_representation(&der, &mut x0);
    //let mut der = DerivativeOperator {
    //    o_tilde: &mut otilde,
    //    expval_o: &mut expvalo,
    //    ho: &mut expval_ho,
    //    n: (4*sys.nfij + sys.nvij + sys.ngi) as i32,
    //    nsamp: match vmcparams.compute_energy_method {
    //        EnergyComputationMethod::ExactSum => 1.0,
    //        EnergyComputationMethod::MonteCarlo => sys.nmcsample as f64,
    //    },
    //    nsamp_int: sys.mcsample_interval,
    //    mu: -1,
    //    visited: &mut visited,
    //    pfaff_off: sys.ngi + sys.nvij,
    //    jas_off: sys.ngi,
    //    epsilon: vmcparams.epsilon,
    //};
    //let mut work_derivative = DerivativeOperator {
    //    o_tilde: &mut work_otilde,
    //    expval_o: &mut work_expvalo,
    //    ho: &mut work_expval_ho,
    //    n: (sys.nfij + sys.nvij + sys.ngi) as i32,
    //    nsamp: match vmcparams.compute_energy_method {
    //        EnergyComputationMethod::ExactSum => 1.0,
    //        EnergyComputationMethod::MonteCarlo => sys.nmcsample as f64,
    //    },
    //    nsamp_int: sys.mcsample_interval,
    //    mu: -1,
    //    visited: &mut work_visited,
    //    pfaff_off: sys.ngi + sys.nvij,
    //    jas_off: sys.ngi,
    //    epsilon: vmcparams.epsilon,
    //};
    for opt_iter in 0..vmcparams.noptiter {
        //println!("{:?}", params.fij);
        //println!("x0 = {:?}", x0);
        //println!("b = {:?}", b);
        sys._opt_iter = opt_iter;
        let (mean_energy, _accumulated_states, deltae, correlation_time) = {
            match vmcparams.compute_energy_method {
                EnergyComputationMethod::MonteCarlo => compute_mean_energy(rng, initial_state, params, sys, &mut work_der),
                EnergyComputationMethod::ExactSum => {
                    (compute_mean_energy_exact(params, sys, &mut work_der), Vec::with_capacity(0), 0.0, 0.0)
                },
            }
        };
        // Save energy, error and correlation_time.
        output_energy_array[opt_iter * 3] = mean_energy;
        output_energy_array[opt_iter * 3 + 1] = deltae;
        output_energy_array[opt_iter * 3 + 2] = correlation_time;

        //// Copy out the relevant terms.
        //work_derivative.mu = derivative.mu;
        //let mut i = 0;
        //for elem in derivative.visited.iter() {
        //    work_derivative.visited[i] = *elem;
        //    i += 1;
        //}
        //mapto_pairwf(&derivative, &mut work_derivative, sys);

        x0[(sys.ngi + sys.nvij)..(sys.ngi + sys.nvij + sys.nfij)].copy_from_slice(params.fij);
        x0[sys.ngi..(sys.ngi + sys.nvij)].copy_from_slice(params.vij);
        x0[0..sys.ngi].copy_from_slice(params.gi);
        param_map.update_reduced_representation(&work_der, &mut der, &mut x0);
        //println!("Expval Full = {:?}", work_der.ho);
        //println!("Expval Reduced = {:?}", der.ho);
        //panic!("Stop.");

        // 68 misawa
        //let mut b: Vec<f64> = vec![0.0; der.n as usize];
        unsafe {
            let incx = 1;
            let incy = 1;
            daxpy(der.n, -mean_energy, &der.expval_o, incx, &mut der.ho, incy);
            dcopy(der.n, &der.ho, incx, &mut b, incy);
        }

        let mut _flag: bool = true;
        let ignored_columns = match vmcparams.optimise_energy_method {
            EnergyOptimisationMethod::ExactInverse => {
                exact_overlap_inverse(&der, &mut b, vmcparams.epsilon, vmcparams.nparams as i32, vmcparams.threshold)
            },
            EnergyOptimisationMethod::ConjugateGradiant => {
                conjugate_gradiant(&der, &mut b, &mut x0, vmcparams.epsilon, vmcparams.kmax, vmcparams.nparams as i32, vmcparams.threshold, vmcparams.epsilon_cg)
            },
        };
        let mut delta_alpha = vec![0.0; param_map.gendim as usize];
        let mut j: usize = 0;
        for i in 0..vmcparams.nparams {
            if ignored_columns[i] {
                continue;
            }
            delta_alpha[i] = b[j];
            j += 1;
            if !<f64>::is_finite(delta_alpha[i]) {
                _flag = false;
            }
        }
        //panic!("Stop!");
        param_map.update_delta_alpha_reduced_to_gen(&mut delta_alpha);
        //println!("b = {:?}", b);
        //println!("delta_alpha = {:?}", delta_alpha);
        //panic!("Stop.");
        if vmcparams.optimise {
            unsafe {
                let incx = 1;
                let incy = 1;
                let alpha = - vmcparams.dt * <f64>::exp(- (opt_iter as f64) * vmcparams.optimisation_decay);
                if vmcparams.optimise_gutzwiller {
                    daxpy(sys.ngi as i32, alpha, &delta_alpha, incx, &mut params.gi, incy);
                }
                if vmcparams.optimise_jastrow {
                    daxpy(sys.nvij as i32, alpha, &delta_alpha[sys.ngi..vmcparams.nparams], incx, &mut params.vij, incy);
                }
                if vmcparams.optimise_orbital {
                    daxpy(sys.nfij as i32, alpha, &delta_alpha[sys.ngi + sys.nvij..vmcparams.nparams], incx, &mut params.fij, incy);
                }
            }
            info!("Correctly finished optimisation iteration {}", opt_iter);
            //info!("Rescaling the params.");
            //let scale: f64 = unsafe {
            //    let incx = 1;
            //    let incy = 1;
            //    ddot(derivative.n, derivative.expval_o, incx, params.gi, incy)
            //};
            //info!("Scale by : {}", scale);
            //let ratio = 1.0 / (scale + 1.0);
            //unsafe {
            //    let incx = 1;
            //    dscal(vmcparams.nparams as i32, ratio, params.gi, incx)
            //}
            //info!("Scaled params by ratio = {}", ratio);

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
        // HARD CODE vij = vji
        // Slater Rescaling
        unsafe {
            let incx = 1;
            let max = params.fij[idamax(sys.nfij as i32, &params.fij, incx) - 1];
            if <f64>::abs(max) < 1e-16 {
                error!("Pfaffian params are all close to 0.0. Rescaling might bring noise.");
                panic!("Undefined behavior.");
            }
            info!("Max was: {}", max);
            dscal(sys.nfij as i32, 1.0 / max, &mut params.fij, incx);
        }
        //unsafe {
        //    dcopy(
        //        sys.nfij as i32,
        //        &params.fij,
        //        1,
        //        &mut params.fij[sys.nfij..2*sys.nfij],
        //        1
        //    );
        //}
        zero_out_derivatives(&mut der, sys);
        zero_out_derivatives(&mut work_der, sys);
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
    output_energy_array
}
