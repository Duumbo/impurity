use log::{error, info, trace, warn};
use rand::distributions::{Distribution, Standard};
use rand::Rng;

use crate::{BitOps, FockState, RandomStateGeneration, Spin, SysParams, VarParams};
use crate::density::{compute_internal_product_parts, fast_internal_product, fast_internal_product_spin_change};
use crate::pfaffian::{update_pstate, PfaffianState};
use crate::hamiltonian::{kinetic, potential};

pub enum Projector {
    Identity,
    ElectronExcitation {
        i: usize,
        sigmai: Spin,
    },
    PairExcitation {
        i: usize,
        sigmai: Spin,
        j: usize,
        sigmaj: Spin
    },
}

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

#[inline(always)]
fn propose_hopping
<R: Rng + ?Sized,
T: BitOps + std::fmt::Display + std::fmt::Debug + From<u8> + Send>
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
    let state2 = state.generate_hopping(rng, sys.size as u32, hop);
    let (ratio_ip, updated_column, col_idx) = {
        fast_internal_product(state, &state2, pfaff_state, &hop, previous_proj, params)
    };
    (ratio_ip, state2, updated_column, col_idx)
}

#[inline(always)]
fn compute_hamiltonian<T: BitOps + std::fmt::Display + std::fmt::Debug + Send>(state: FockState<T>, pstate: &PfaffianState, proj: f64, params: &VarParams, sys: &SysParams) -> f64 {
    let kin = kinetic(state, pstate, proj, params, sys);
    let e = kin + potential(state, proj, pstate, sys);
    trace!("Hamiltonian application <x|H|psi> = {} for state: |x> = {}", e, state);
    e / (pstate.pfaff * <f64>::exp(proj))
}

#[inline(always)]
fn make_update<T: BitOps + std::fmt::Debug + std::fmt::Display + From<u8> + Send>(
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
        (*pstate, *proj) = compute_internal_product_parts(*state2, params, sys);
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

#[inline(always)]
fn warmup<T, R>(
    rng: &mut R,
    state: &mut FockState<T>,
    hop: &mut (usize, usize, Spin),
    proj: &mut f64,
    proj_copy_persistent: &mut f64,
    ratio_prod: &mut f64,
    pstate: &mut PfaffianState,
    n_accepted_updates: &mut usize,
    params: &VarParams,
    sys: &SysParams
)
where T: BitOps + From<u8> + std::fmt::Debug + std::fmt::Display + Send,
      R: Rng + ?Sized,
      Standard: Distribution<T>
{
    // Warmup
    for _ in 0..sys.nmcwarmup {
        let mut proj_copy = *proj;
        let (mut ratio, state2, col, colidx) = propose_hopping(&state, &pstate, &mut proj_copy, hop, rng, params, sys);
        trace!("Current state: {}", state);
        trace!("Proposed state: {}", state2);
        trace!("Ratio: {}", ratio);
        ratio *= <f64>::exp(proj_copy - *proj);
        let w = rng.gen::<f64>();
        if <f64>::abs(ratio) * <f64>::abs(ratio) >= w {
            // We ACCEPT
            trace!("Accept.");
            *n_accepted_updates += 1;
            make_update(
                n_accepted_updates,
                proj,
                &mut proj_copy,
                proj_copy_persistent,
                &ratio,
                ratio_prod,
                state,
                &state2,
                &hop,
                col,
                colidx,
                pstate,
                params,
                sys
            );

        }
    }

}

#[inline(always)]
fn accumulate_expvals(energy: &mut f64, state_energy: f64, rho: f64) {
    // Accumulate Energy
    *energy += state_energy;
}

#[inline(always)]
fn normalize(energy: &mut f64, energy_bootstraped: &mut f64, nsample: f64, nbootstrap: f64) {
    *energy *= 1.0 / (nsample - 1.0);
    *energy_bootstraped *= 1.0 / nsample;
    *energy_bootstraped *= 1.0 / nbootstrap;
}

#[inline(always)]
fn sq(x: f64) -> f64
{
    x * x
}

fn general_hopping<T: BitOps + std::fmt::Display + std::fmt::Debug + From<u8> + Send>
(state: &FockState<T>, hop: &(usize, usize, Spin, Spin)) -> Option<FockState<T>>
{
    let i = hop.0;
    let j = hop.1;
    let mut state2 = state.clone();
    match hop.3 {
        Spin::Up => {
            if state.spin_up.check(j) {
                return None;
            }
            state2.spin_up.set(j);
        },
        Spin::Down => {
            if state.spin_down.check(j) {
                return None;
            }
            state2.spin_down.set(j);
        },
    }
    match hop.2 {
        Spin::Up => {
            if !state.spin_up.check(i) {
                return None;
            }
            state2.spin_up.set(i);
        },
        Spin::Down => {
            if !state.spin_down.check(i) {
                return None;
            }
            state2.spin_down.set(i);
        },
    }
    Some(state2)
}

// C^t_{i,sigmai} C_{j,sigmaj} C^t_{k,sigmak} C_{l,sigmal}
#[inline(always)]
fn opperate_by_correlator_twice
<T: BitOps + std::fmt::Display + std::fmt::Debug + From<u8> + Send>
(
    state: &FockState<T>,
    pfaff_state: &PfaffianState,
    previous_proj: &mut f64,
    hop: &(usize, usize, Spin, Spin),
    params: &VarParams,
) -> Option<(f64, FockState<T>, Vec<f64>, usize)>
{
    let state2 = general_hopping(state, hop);
    match state2 {
        None => {
            None
        },
        Some(x) => {
            let (ratio_ip, updated_column, col_idx) = {
                fast_internal_product_spin_change(state, &x, pfaff_state, &hop, previous_proj, params)
            };
            Some((ratio_ip, x, updated_column, col_idx))
        },
    }
}

// C^t_{i,sigmai} C_{j,sigmaj}
#[inline(always)]
fn opperate_by_correlator
<T: BitOps + std::fmt::Display + std::fmt::Debug + From<u8> + Send>
(
    state: &FockState<T>,
    pfaff_state: &PfaffianState,
    previous_proj: &mut f64,
    hop: &(usize, usize, Spin, Spin),
    params: &VarParams,
) -> Option<(f64, FockState<T>, Vec<f64>, usize)>
{
    let state2 = general_hopping(state, hop);
    match state2 {
        None => {
            None
        },
        Some(x) => {
            let (ratio_ip, updated_column, col_idx) = {
                fast_internal_product_spin_change(state, &x, pfaff_state, &hop, previous_proj, params)
            };
            Some((ratio_ip, x, updated_column, col_idx))
        },
    }
}

#[inline(always)]
fn accumulate_correlators<T: BitOps + std::fmt::Debug + std::fmt::Display + From<u8> + Send>
(state: FockState<T>, pstate: &PfaffianState, proj: f64, params: &VarParams, n_sites: usize, expval_corr: &mut [f64])
{
    // Loop over all correlators
    // These loops are unrolled to allow the compiler to optimise hard coded spins => less branches
    // since everything is inlined
    // up up
    for i in 0..n_sites {
        for j in 0..n_sites {
            let corr = (i, j, Spin::Up, Spin::Up);
            let mut proj_copy = proj;
            let (ratio, _, _, _) = match opperate_by_correlator(&state, pstate, &mut proj_copy, &corr, params) {
                None => {
                    // Accumulate nothing if hopping is impossible.
                    continue;
                },
                Some(r) => {
                    r
                }
            };
            let total_ratio = ratio * <f64>::exp(proj_copy - proj);
            expval_corr[j + i * n_sites] += total_ratio;
        }
    }
    // up down
    for i in 0..n_sites {
        for j in 0..n_sites {
            let corr = (i, j, Spin::Up, Spin::Down);
            let mut proj_copy = proj;
            let (ratio, _, _, _) = match opperate_by_correlator(&state, pstate, &mut proj_copy, &corr, params) {
                None => {
                    // Accumulate nothing if hopping is impossible.
                    continue;
                },
                Some(r) => {
                    r
                }
            };
            let total_ratio = ratio * <f64>::exp(proj_copy - proj);
            expval_corr[j + i * n_sites + n_sites*n_sites] += total_ratio;
        }
    }
    // down up
    for i in 0..n_sites {
        for j in 0..n_sites {
            let corr = (i, j, Spin::Down, Spin::Up);
            let mut proj_copy = proj;
            let (ratio, _, _, _) = match opperate_by_correlator(&state, pstate, &mut proj_copy, &corr, params) {
                None => {
                    // Accumulate nothing if hopping is impossible.
                    continue;
                },
                Some(r) => {
                    r
                }
            };
            let total_ratio = ratio * <f64>::exp(proj_copy - proj);
            expval_corr[j + i * n_sites + 2*n_sites*n_sites] += total_ratio;
        }
    }
    // down down
    for i in 0..n_sites {
        for j in 0..n_sites {
            let corr = (i, j, Spin::Down, Spin::Down);
            let mut proj_copy = proj;
            let (ratio, _, _, _) = match opperate_by_correlator(&state, pstate, &mut proj_copy, &corr, params) {
                None => {
                    // Accumulate nothing if hopping is impossible.
                    continue;
                },
                Some(r) => {
                    r
                }
            };
            let total_ratio = ratio * <f64>::exp(proj_copy - proj);
            expval_corr[j + i * n_sites + 3*n_sites*n_sites] += total_ratio;
        }
    }
}

// C^t C
pub fn compute_mean_correlator
<R: Rng + ?Sized,
T: BitOps + std::fmt::Debug + std::fmt::Display + From<u8> + Send>
(rng: &mut R, initial_state: FockState<T>, projection: Projector, params: &VarParams, sys: &SysParams) -> (f64, Vec<FockState<T>>)
where Standard: Distribution<T>
{
    let mut expval_correlators = vec![0.0; 4*sys.size*sys.size];
    let mut state = initial_state;
    let mut accumulated_states: Vec<FockState<T>> = Vec::new();
    let (mut pstate, mut proj) = compute_internal_product_parts(state, params, sys);
    let mut hop: (usize, usize, Spin) = (0, 0, Spin::Up);
    let mut _lip = <f64>::ln(<f64>::abs(<f64>::exp(proj) * pstate.pfaff)) * 2.0;
    let mut n_accepted_updates: usize = 0;
    let mut energy: f64 = 0.0;
    let mut _energy_sq: f64 = 0.0;
    let mut proj_copy_persistent = proj;
    let mut ratio_prod = 1.0;
    let mut sample_counter: usize = 0;
    if <f64>::log2(sys.nmcsample as f64) <= 6.0 {
        error!("Not enough monte-carlo sample for an accurate error estimation. NMCSAMPLE = {}", sys.nmcsample);
        panic!("NMCSAMPLE is less than 64.");
    }
    let mut energy_bootstraped = 0.0;

    if sys.nwarmupchains > sys._opt_iter {
        info!("Starting the warmup phase.");
        warmup(rng, &mut state, &mut hop, &mut proj, &mut proj_copy_persistent, &mut ratio_prod, &mut
            pstate, &mut n_accepted_updates, params, sys);
    }

    info!("Starting the sampling phase.");
    // MC Sampling
    // Accumulate the first state into the markov chain
    accumulated_states.push(state);
    let state_energy = compute_hamiltonian(state, &pstate, proj, params, sys);

    accumulate_expvals(&mut energy, state_energy, 1.0);
    for mc_it in 0..(sys.nmcsample * sys.mcsample_interval) {
        let mut proj_copy = proj;
        let (mut ratio, state2, col, colidx) = propose_hopping(&state, &pstate, &mut proj_copy, &mut hop, rng, params, sys);
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
        if sample_counter >= sys.mcsample_interval {
            accumulated_states.push(state);
            match projection {
                Projector::Identity => {
                    accumulate_correlators(state, &pstate, proj, &params, sys.size, &mut expval_correlators);
                },
                Projector::ElectronExcitation { i, sigmai } => {
                    let excitation = (i, i, sigmai, sigmai);
                    let x = general_hopping(&state, &excitation);
                    match x {
                        None => {
                            // If None, then n_isigmai |x> = 0. Accumulate nothing
                        },
                        Some(_) => {
                    accumulate_correlators(state, &pstate, proj, &params, sys.size, &mut expval_correlators);
                        },
                    }
                },
                Projector::PairExcitation { i, sigmai, j, sigmaj } => {
                    let excitation1 = (i, i, sigmai, sigmai);
                    let excitation2 = (j, j, sigmaj, sigmaj);
                    let res1 = general_hopping(&state, &excitation1);
                    let res2 = general_hopping(&state, &excitation2);
                    match res1 {
                        None => {
                            // If None, then n_isigmai |x> = 0. Accumulate nothing
                        },
                        Some(_) => {
                            match res2 {
                                None => {
                                    // If None, then n_jsigmaj |x> = 0. Accumulate nothing
                                },
                                Some(_) => {
                    accumulate_correlators(state, &pstate, proj, &params, sys.size, &mut expval_correlators);
                                },
                            }
                        },
                    }
                },
            }
            let state_energy = compute_hamiltonian(state, &pstate, proj, params, sys);

            accumulate_expvals(&mut energy, state_energy, 1.0);
            sample_counter = 0;
            if mc_it >= (sys.nmcsample - sys.nbootstrap) * sys.mcsample_interval {
                energy_bootstraped += energy;
            }
        }
        sample_counter += 1;

    }
    info!("Final Energy: {:.2}", energy);
    normalize(&mut energy, &mut energy_bootstraped, sys.nmcsample as f64, sys.nbootstrap as f64);
    info!("Final Energy normalized: {:.2}", energy);
    (energy_bootstraped, accumulated_states)
}
