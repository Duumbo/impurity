use blas::daxpy;
use log::{info, trace, warn};
use rand::distributions::{Distribution, Standard};
use rand::Rng;

use crate::gutzwiller::compute_gutzwiller_der;
use crate::jastrow::compute_jastrow_der;
use crate::{BitOps, DerivativeOperator, FockState, RandomStateGeneration, Spin, SysParams, VarParams};
use crate::density::{compute_internal_product_parts, fast_internal_product};
use crate::pfaffian::{compute_pfaffian_derivative, update_pstate, PfaffianState};
use crate::hamiltonian::{kinetic, potential};

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
    let e = kin + potential(state, sys);
    trace!("Hamiltonian application <x|H|psi> = {} for state: |x> = {}", e, state);
    e
}

fn get_sign<T: BitOps>(s1: &FockState<T>, hop: &(usize, usize, Spin)) -> usize {
    let (a, b) = if hop.0 < hop.1 {
        (hop.1, hop.0)
    } else {
        (hop.0, hop.1)
    };
    trace!("First mask: {:08b}, second mask: {:08b}", !(<u8>::MAX >> a), <u8>::MAX >> (b + 1));
    let mask = {
        !(<T>::ones() >> a) & (<T>::ones() >> (b + 1))
    };
    let n_ones = match hop.2 {
        Spin::Up => {
            (s1.spin_up & mask).count_ones()
        },
        Spin::Down => {
            (s1.spin_down & mask).count_ones()
        }
    };

    n_ones as usize
}

fn compute_derivative_operator<T: BitOps + std::fmt::Debug + std::fmt::Display + From<u8>>
(state: FockState<T>, pstate: &PfaffianState, der: &mut DerivativeOperator, params: &VarParams, sys: &SysParams)
{
    compute_gutzwiller_der(state, sys.size, der);
    compute_jastrow_der(state, der, sys.size);
    compute_pfaffian_derivative(pstate, der, sys);
}

pub fn compute_mean_energy
<R: Rng + ?Sized,
T: BitOps + std::fmt::Debug + std::fmt::Display + From<u8>>
(rng: &mut R, initial_state: FockState<T>, params: &VarParams, sys: &SysParams, derivatives: &mut DerivativeOperator) -> f64
where Standard: Distribution<T>
{
    let mut state = initial_state;
    let (mut pstate, mut proj) = compute_internal_product_parts(state, params);
    let mut hop: (usize, usize, Spin) = (0, 0, Spin::Up);
    let mut _lip = <f64>::ln(<f64>::abs(<f64>::exp(proj) * pstate.pfaff)) * 2.0;
    let mut n_accepted_updates: usize = 0;
    let mut energy: f64 = 0.0;
    let mut proj_copy_persistent = proj;
    let mut sign: usize = 0;

    info!("Starting the warmup phase.");
    // Warmup
    for _ in 0..sys.nmcwarmup {
        let mut proj_copy = proj;
        let (ratio, state2, col, colidx) = propose_hopping(&state, &pstate, &mut proj_copy, &mut hop, rng, params, sys);
        trace!("Current state: {}", state);
        trace!("Proposed state: {}", state2);
        trace!("Ratio: {}", ratio);
        let w = rng.gen::<f64>();
        if <f64>::abs(ratio) * <f64>::abs(ratio) >= w {
            // We ACCEPT
            trace!("Accept.");
            n_accepted_updates += 1;

            // Clean update once in a while
            if n_accepted_updates < sys.clean_update_frequency {
                state = state2;
                proj = proj_copy;
                update_pstate(&mut pstate, hop, col, colidx);
            } else {
                let tmp_pfaff = pstate.pfaff;
                trace!("PfaffianState before clean update: {:?}", pstate);
                sign = (sign + get_sign(&state, &hop)) % 2;
                trace!("Number of electrons between hopping: {}", get_sign(&state, &hop));
                (pstate, proj) = compute_internal_product_parts(state2, params);
                let inverse_proj = <f64>::exp(proj_copy_persistent - proj);
                if sign != 0 {
                    pstate.pfaff *= -1.0;
                }
                if pstate.pfaff*pstate.pfaff < sys.tolerance_singularity {
                    warn!("Updated matrix is probably singular, got pfaffian {:.2e} and Tolerence is : {:e}.", pstate.pfaff, sys.tolerance_singularity);
                }
                trace!("PfaffianState after clean update: {:?}", pstate);
                let err = <f64>::abs(tmp_pfaff * ratio * inverse_proj) - <f64>::abs(pstate.pfaff);
                if err >= sys.tolerance_sherman_morrison {
                    warn!("Sherman-Morrisson update error of {:.2e} on computed pfaffian. Tolerence is : {:e}. Ratio was {}, states were: {} -> {}", err, sys.tolerance_sherman_morrison, ratio, state, state2);
                }
                n_accepted_updates = 0;
                state = state2;
                proj_copy_persistent = proj;
            }
        }
    }

    info!("Starting the sampling phase.");
    // MC Sampling
    for i in 0..derivatives.n as usize {
        derivatives.o_tilde[i] *= 0.0;
    }
    for mc_it in 0..sys.nmcsample {
        let mut proj_copy = proj;
        trace!("Before proposition: ~O_[0, {}] = {}", derivatives.mu + 1, derivatives.o_tilde[(derivatives.n * (derivatives.mu + 1)) as usize]);
        let (ratio, state2, col, colidx) = propose_hopping(&state, &pstate, &mut proj_copy, &mut hop, rng, params, sys);
        trace!("After proposition: ~O_[0, {}] = {}", derivatives.mu + 1, derivatives.o_tilde[(derivatives.n * (derivatives.mu + 1)) as usize]);
        trace!("Current state: {}", state);
        trace!("Proposed state: {}", state2);
        trace!("Ratio: {}", ratio);
        let w = rng.gen::<f64>();
        if <f64>::abs(ratio) * <f64>::abs(ratio) >= w {
            // We ACCEPT
            trace!("Accept.");
            n_accepted_updates += 1;
            // Clean update once in a while
            // written in this way for branch prediction.
            if n_accepted_updates < sys.clean_update_frequency {
                derivatives.mu += 1;
                state = state2;
                proj = proj_copy;
                update_pstate(&mut pstate, hop, col, colidx);
            } else {
                derivatives.mu += 1;
                let tmp_pfaff = pstate.pfaff;
                sign = (sign + get_sign(&state, &hop)) % 2;
                trace!("Number of electrons between hopping: {}", get_sign(&state, &hop));
                (pstate, proj) = compute_internal_product_parts(state2, params);
                let inverse_proj = <f64>::exp(proj_copy_persistent - proj);
                if sign != 0 {
                    pstate.pfaff *= -1.0;
                }
                let err = <f64>::abs(tmp_pfaff * ratio * inverse_proj) - <f64>::abs(pstate.pfaff);
                if pstate.pfaff*pstate.pfaff < sys.tolerance_singularity {
                    warn!("Updated matrix is probably singular, got pfaffian {:.2e} and Tolerence is : {:e}.", pstate.pfaff, sys.tolerance_singularity);
                }
                trace!("PfaffianState after clean update: {:?}", pstate);
                if err >= sys.tolerance_sherman_morrison {
                    warn!("Sherman-Morrisson update error of {:.2e} on computed pfaffian. Tolerence is : {:e}. Ratio was {}", err, sys.tolerance_sherman_morrison, ratio);
                }
                n_accepted_updates = 0;
                state = state2;
                proj_copy_persistent = proj;
            }
            // Compute the derivative operator
            compute_derivative_operator(state, &pstate, derivatives, params, sys);
        }
        derivatives.visited[(derivatives.mu + 1) as usize] += 1;
        // Accumulate energy
        let state_energy = compute_hamiltonian(state, &pstate, proj, params, sys);
        energy += state_energy;
        // Accumulate <HO_m>
        unsafe {
            let incx = 1;
            let incy = 1;
            daxpy(
                derivatives.n,
                state_energy,
                &derivatives.o_tilde[(derivatives.n * (derivatives.mu + 1)) as usize .. (derivatives.n * (derivatives.mu + 2)) as usize],
                incx,
                &mut derivatives.ho,
                incy
            );
        }
        // Accumulate the derivative operators
        for i in 0 .. derivatives.n as usize {
            derivatives.expval_o[i] += derivatives.o_tilde[i + (derivatives.n * (derivatives.mu + 1)) as usize];
        }

    }
    info!("Final Energy: {:.2}", energy);
    energy = energy / sys.nmcsample as f64;
    info!("Final Energy normalized: {:.2}", energy);
    energy
}
