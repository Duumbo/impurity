use blas::daxpy;
use log::{error, info, trace, warn};
use rand::distributions::{Distribution, Standard};
use rand::Rng;
use std::fs::File;
use std::io::Write;

use crate::gutzwiller::compute_gutzwiller_der;
use crate::jastrow::compute_jastrow_der;
use crate::{BitOps, DerivativeOperator, FockState, RandomStateGeneration, Spin, SysParams, VarParams};
use crate::density::{compute_internal_product_parts, fast_internal_product};
use crate::pfaffian::{compute_pfaffian_derivative, update_pstate, PfaffianState};
use crate::hamiltonian::{kinetic, potential};

//fn propose_exchange
//<R: Rng + ?Sized,
//T: BitOps + std::fmt::Display + std::fmt::Debug + From<u8>>
//(
//    state: &FockState<T>,
//    pfaff_state: &PfaffianState,
//    previous_proj: &mut f64,
//    exchange: &mut (usize, usize),
//    rng: &mut R,
//    params: &VarParams,
//    sys: &SysParams,
//) -> (f64, FockState<T>, Vec<f64>, usize)
//    where Standard: Distribution<T>
//{
//    let state2 = state.generate_exchange(rng, exchange);
//    let (ratio_ip, updated_column, col_idx) = {
//        fast_internal_product(state, &state2, pfaff_state, &hop, previous_proj, params)
//    };
//    (ratio_ip, state2, updated_column, col_idx)
//}

fn propose_hopping
<R: Rng + ?Sized,
T: BitOps + std::fmt::Display + std::fmt::Debug + From<u8>>
(
    state: &FockState<T>,
    pfaff_state: &PfaffianState,
    previous_proj: &mut f64,
    hop: &mut (usize, usize, Spin),
    rng: &mut R,
    params: &VarParams,
    sys: &SysParams,
) -> (f64, FockState<T>, Vec<f64>, usize)
    where Standard: Distribution<T>
{
    let state2 = state.generate_hopping(rng, sys.size as u32, hop, sys);
    let (ratio_ip, updated_column, col_idx) = {
        fast_internal_product(state, &state2, pfaff_state, &hop, previous_proj, params)
    };
    (ratio_ip, state2, updated_column, col_idx)
}

fn compute_hamiltonian<T: BitOps + std::fmt::Display + std::fmt::Debug>(state: FockState<T>, pstate: &PfaffianState, proj: f64, params: &VarParams, sys: &SysParams) -> f64 {
    let kin = kinetic(state, pstate, proj, params, sys);
    let e = kin + potential(state, proj, pstate, sys);
    trace!("Hamiltonian application <x|H|psi> = {} for state: |x> = {}", e, state);
    e / (pstate.pfaff * <f64>::exp(proj))
}

fn compute_derivative_operator<T: BitOps + std::fmt::Debug + std::fmt::Display + From<u8>>
(state: FockState<T>, pstate: &PfaffianState, der: &mut DerivativeOperator, sys: &SysParams)
{
    compute_gutzwiller_der(state, sys.size, der);
    compute_jastrow_der(state, der, sys.size);
    compute_pfaffian_derivative(pstate, der, sys);
}

#[inline(always)]
fn make_update<T: BitOps + std::fmt::Debug + std::fmt::Display + From<u8>>(
    n: &mut usize,
    proj: &mut f64,
    proj_copy: &mut f64,
    proj_copy_persistent: &mut f64,
    ratio: &f64,
    ratio_prod: &mut f64,
    state: &mut FockState<T>,
    state2: &FockState<T>,
    hop: &(usize, usize, Spin),
    col: Vec<f64>,
    colidx: usize,
    pstate: &mut PfaffianState,
    params: &VarParams,
    sys: &SysParams
) {
    // Clean update once in a while
    // written in this way for branch prediction.
    if *n < sys.clean_update_frequency {
        *state = *state2;
        *proj = *proj_copy;
        update_pstate(pstate, *hop, col, colidx);
        *ratio_prod *= ratio;
    } else {
        let tmp_pfaff = pstate.pfaff;
        (*pstate, *proj) = compute_internal_product_parts(*state2, params);
        let inverse_proj = <f64>::exp(*proj_copy_persistent - *proj);
        let err = <f64>::abs(<f64>::abs(tmp_pfaff * *ratio * *ratio_prod * inverse_proj) - <f64>::abs(pstate.pfaff));
        if pstate.pfaff*pstate.pfaff < sys.tolerance_singularity {
            warn!("Updated matrix is probably singular, got pfaffian {:.2e} and Tolerence is : {:e}.", pstate.pfaff, sys.tolerance_singularity);
        }
        trace!("PfaffianState after clean update: {:?}", pstate);
        if err >= sys.tolerance_sherman_morrison {
            warn!("Sherman-Morrisson update error of {:.2e} on computed pfaffian. Tolerence is : {:e}. Expected {}, got {}", err, sys.tolerance_sherman_morrison, tmp_pfaff * *ratio * *ratio_prod * inverse_proj, pstate.pfaff);
        }
        *n = 0;
        *state = *state2;
        *proj_copy_persistent = *proj;
        *ratio_prod = 1.0;
    }
}

pub fn compute_mean_energy
<R: Rng + ?Sized,
T: BitOps + std::fmt::Debug + std::fmt::Display + From<u8>>
(rng: &mut R, initial_state: FockState<T>, params: &VarParams, sys: &SysParams, derivatives: &mut DerivativeOperator) -> (f64, Vec<FockState<T>>, f64)
where Standard: Distribution<T>
{
    let mut ratiofp = File::create("ratios").unwrap();
    if derivatives.mu != -1 {
        warn!("The derivative operator current row was mu = {} on entry, is it reinitialized?", derivatives.mu);
    }
    let mut state = initial_state;
    let mut accumulated_states: Vec<FockState<T>> = Vec::new();
    let (mut pstate, mut proj) = compute_internal_product_parts(state, params);
    let mut hop: (usize, usize, Spin) = (0, 0, Spin::Up);
    let mut _lip = <f64>::ln(<f64>::abs(<f64>::exp(proj) * pstate.pfaff)) * 2.0;
    let mut n_accepted_updates: usize = 0;
    let mut energy: f64 = 0.0;
    let mut _energy_sq: f64 = 0.0;
    let mut proj_copy_persistent = proj;
    let mut ratio_prod = 1.0;
    let mut sample_counter: usize = 0;
    if <f64>::log2(sys.nmcsample as f64) <= 5.0 {
        error!("Not enough monte-carlo sample for an accurate error estimation. NMCSAMPLE = {}", sys.nmcsample);
        panic!("NMCSAMPLE is less than 32.");
    }
    let error_estimation_level = <f64>::log2(sys.nmcsample as f64) as usize - 5;
    let mut energy_sums = vec![0.0; error_estimation_level];
    let mut energy_quad_sums = vec![0.0; error_estimation_level];
    let mut previous_energies = vec![0.0; error_estimation_level + 1];
    let mut n_values = vec![0; error_estimation_level];

    info!("Starting the warmup phase.");
    // Warmup
    for _ in 0..sys.nmcwarmup {
        let mut proj_copy = proj;
        let (mut ratio, state2, col, colidx) = propose_hopping(&state, &pstate, &mut proj_copy, &mut hop, rng, params, sys);
        trace!("Current state: {}", state);
        trace!("Proposed state: {}", state2);
        trace!("Ratio: {}", ratio);
        ratio *= <f64>::exp(proj_copy - proj);
        let w = rng.gen::<f64>();
        if <f64>::abs(ratio) * <f64>::abs(ratio) >= w {
            // We ACCEPT
            trace!("Accept.");
            n_accepted_updates += 1;
            make_update(
                &mut n_accepted_updates,
                &mut proj,
                &mut proj_copy,
                &mut proj_copy_persistent,
                &ratio,
                &mut ratio_prod,
                &mut state,
                &state2,
                &hop,
                col,
                colidx,
                &mut pstate,
                params,
                sys
            );

        }
    }

    info!("Starting the sampling phase.");
    // MC Sampling
    // We need to reset the counter that the warmup increased.
    derivatives.mu = 0;
    // Compute the derivative for the first element in the markov chain
    compute_derivative_operator(state, &pstate, derivatives, sys);
    for mc_it in 0..(sys.nmcsample * sys.mcsample_interval) {
        let mut proj_copy = proj;
        trace!("Before proposition: ~O_[0, {}] = {}", derivatives.mu + 1, derivatives.o_tilde[(derivatives.n * (derivatives.mu + 1)) as usize]);
        let (mut ratio, state2, col, colidx) = propose_hopping(&state, &pstate, &mut proj_copy, &mut hop, rng, params, sys);
        trace!("After proposition: ~O_[0, {}] = {}", derivatives.mu + 1, derivatives.o_tilde[(derivatives.n * (derivatives.mu + 1)) as usize]);
        trace!("Current state: {}", state);
        trace!("Proposed state: {}", state2);
        trace!("Ratio: {}", ratio);
        ratio *= <f64>::exp(proj_copy - proj);
        let w = rng.gen::<f64>();
        ratiofp.write(format!("De {}, à {} = {}\n", state, state2, <f64>::abs(ratio) * <f64>::abs(ratio)).as_bytes()).unwrap();
        if <f64>::abs(ratio) * <f64>::abs(ratio) >= w {
            // We ACCEPT
            trace!("Accept.");
            n_accepted_updates += 1;
            make_update(
                &mut n_accepted_updates,
                &mut proj,
                &mut proj_copy,
                &mut proj_copy_persistent,
                &ratio,
                &mut ratio_prod,
                &mut state,
                &state2,
                &hop,
                col,
                colidx,
                &mut pstate,
                params,
                sys
            );
            // Compute the derivative operator
            if sample_counter >= sys.mcsample_interval {
                derivatives.mu += 1;
                compute_derivative_operator(state, &pstate, derivatives, sys);
            }
        }
        if sample_counter >= sys.mcsample_interval {
            accumulated_states.push(state);
            // Accumulate energy
            derivatives.visited[derivatives.mu as usize] += 1;
            let state_energy = compute_hamiltonian(state, &pstate, proj, params, sys);
            energy += state_energy;
            _energy_sq += state_energy*state_energy;
            // Energy error estimation
            let accumulation_level = <f32>::log2((mc_it + 1) as f32) as usize;
            for i in 0..accumulation_level {
                if i >= error_estimation_level { break;}
                if i == accumulation_level - 1{
                    if i == 0 {break;}
                    previous_energies[i] = energy_sums[i - 1];
                } else {
                    n_values[i] += 1;
                    energy_sums[i] += 0.5 * (previous_energies[i] + state_energy);
                    energy_quad_sums[i] += 0.5 * (previous_energies[i]*previous_energies[i] + state_energy*state_energy);
                }
            }

            // Accumulate <HO_m>
            unsafe {
                let incx = 1;
                let incy = 1;
                daxpy(
                    derivatives.n,
                    state_energy,
                    &derivatives.o_tilde[(derivatives.n * derivatives.mu) as usize .. (derivatives.n * (derivatives.mu + 1)) as usize],
                    incx,
                    &mut derivatives.ho,
                    incy
                );
            }
            // Accumulate the derivative operators
            for i in 0 .. derivatives.n as usize {
                derivatives.expval_o[i] += derivatives.o_tilde[i + (derivatives.n * derivatives.mu) as usize];
            }
            sample_counter = 0;
        }
        sample_counter += 1;

    }
    derivatives.mu += 1;
    info!("Final Energy: {:.2}", energy);
    energy = energy / sys.nmcsample as f64;
    info!("Final Energy normalized: {:.2}", energy);
    // Error estimation
    let mut error = vec![0.0; error_estimation_level];
    for i in 0..error_estimation_level {
        error[i] = <f64>::sqrt(
            (energy_quad_sums[i] - energy_sums[i]*energy_sums[i] / (n_values[i] as f64)) /
            (n_values[i] * (n_values[i] - 1)) as f64
            )
    }
    let correlation_time = 0.5 * ((error[error_estimation_level-1]/error[0])*(error[error_estimation_level-1]/error[0]) - 1.0);
    (energy, accumulated_states, correlation_time)
}
