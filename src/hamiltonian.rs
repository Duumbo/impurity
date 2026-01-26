use log::trace;

use crate::density::{compute_internal_product_parts, fast_internal_product_no_otilde};
use crate::pfaffian::PfaffianState;
use crate::{BitOps, FockState, Hopper, SpinState};
use crate::{VarParams, Spin, SysParams};

/// Computes the potential term of the Hamiltonian.
/// # Arguments
/// * __`spin_up`__ - The encoded spin up part of the fock state.
/// * __`spin_down`__ - The encoded spin up part of the fock state.
/// # Returns
/// * __`pot_term`__ - The potential term of the Hamiltonian. Gives the diagonal
/// term of the Hamiltonian.
/// # Definition
/// The potential term is defined
/// $$
/// H_U=U\sum_i n_{i\uparrow}n_{i\downarrow}
/// $$
pub fn potential<T>(state: FockState<T>, proj: f64, pstate: &PfaffianState, sys: &SysParams) -> f64
where
    T: BitOps + std::fmt::Display + Send,
{
    let pot = ((state.spin_up & state.spin_down).count_ones() as f64) * sys.cons_u;
    trace!("Output potential <x|U|psi> = {:.2} for state |x> = {}", pot, state);
    //pot * pstate.pfaff * <f64>::exp(proj)
    pot * pstate.pfaff * <f64>::exp(proj)
}

fn compute_sign_permutation<T>(state: FockState<T>, hop: (usize, usize, Spin)) -> f64
where
    T: BitOps + From<u8> + From<SpinState> + std::fmt::Debug + std::fmt::Display + Send
{
    let (bigger_index, smaller_index) = if hop.0 > hop.1 {
            (hop.0, hop.1)
        } else {
            (hop.1, hop.0)
        };
    let mut left_mask = <T>::ones();
    let mut right_mask = <T>::ones();
    left_mask.mask_bits(state.n_sites);
    right_mask.mask_bits(state.n_sites);
    left_mask = left_mask << (state.n_sites - bigger_index);
    right_mask = right_mask >> (smaller_index  + 1);
    right_mask.mask_bits(state.n_sites);
    let mask = left_mask & right_mask;
    let n_perm = match hop.2 {
        Spin::Up => {
            (state.spin_up & mask).count_ones()
        },
        Spin::Down => {
            (state.spin_down & mask).count_ones()
        }
    };
    if n_perm % 2 == 0 {
        1.0
    } else {
        -1.0
    }
}

/// Computes the kinetic term of the Hamiltonian.
/// # Arguments
/// * __`spin_up`__ - The encoded spin up part of the fock state.
/// * __`spin_down`__ - The encoded spin up part of the fock state.
/// # Returns
/// * __`kin_term`__ - The kinetic term of the Hamiltonian. Contains all the
/// states that are accessible from the given state.
/// # Definition
/// The kinetic term is defined
/// $$
/// H_T=-t\sum_{<i,j>,\sigma}c^\dagger_{i\sigma}c_{j\sigma}+c^\dagger_{j\sigma}c_{i\sigma}
/// $$
pub fn kinetic<T>(state: FockState<T>, previous_pstate: &PfaffianState, previous_proj: f64, params: &VarParams, sys: &SysParams) -> f64
where
    T: BitOps + From<u8> + From<SpinState> + std::fmt::Debug + std::fmt::Display + Send
{
    let hops = state.generate_all_hoppings(&sys.hopping_bitmask);
    //println!("KINETIC FOR {}", state);

    let mut kin = 0.0;
    for hop in hops.into_iter() {
        let mut f_state = state;
        match hop.2 {
            Spin::Down => {
                f_state.spin_down.set(hop.0);
                f_state.spin_down.set(hop.1);
            },
            Spin::Up => {
                f_state.spin_up.set(hop.0);
                f_state.spin_up.set(hop.1);
            }
        };
        // Fast update
        let mut proj = previous_proj;
        let (ratio, _col, _colidx) = fast_internal_product_no_otilde(&state, &f_state, previous_pstate, &hop, &mut proj, params);
        let pfaff = previous_pstate.pfaff * ratio;
        let ip = pfaff * <f64>::exp(proj);
        // Clean update
        //let sign = compute_sign_permutation(state, hop);
        let sign = 1.0;
        //let (pstate, proj) = compute_internal_product_parts(f_state, params, sys);
        //let pfaff = pstate.pfaff;
        //let ip = pfaff * <f64>::exp(proj);
        //println!("{f_state} => {}",sign*ip*sys.cons_t*sys.transfert_matrix[hop.0 + hop.1*sys.size]/ (previous_pstate.pfaff * <f64>::exp(previous_proj)));
        //trace!("Projection state: |x'> = {}, z = {}", f_state, ratio);
        //trace!("Adding kinetic term t_[i,j]<x'|psi>: |x> = {}, |x'> = {}, hop = ({}, {}, {}) Computed <x'|psi>/<x|psi> = {}", state, f_state, hop.0, hop.1, hop.2, ratio);
        kin += sign*ip*sys.cons_t*sys.transfert_matrix[hop.0 + hop.1*sys.size];
    }

    //println!("KINETIC END");
    trace!("Output kinetic <x|K|psi> = {:.2} for state |x> = {}", kin, state);
    kin
}

pub fn kinetic_clean<T>(state: FockState<T>, params: &VarParams, sys: &SysParams) -> f64
where
    T: BitOps + From<SpinState> + From<u8> + std::fmt::Debug + std::fmt::Display + Send
{
    let hops = state.generate_all_hoppings(&sys.hopping_bitmask);

    let mut kin = 0.0;
    for hop in hops.into_iter() {
        let mut f_state = state;
        match hop.2 {
            Spin::Down => {
                f_state.spin_down.set(hop.0);
                f_state.spin_down.set(hop.1);
            },
            Spin::Up => {
                f_state.spin_up.set(hop.0);
                f_state.spin_up.set(hop.1);
            }
        };
        let (pstate, proj) = compute_internal_product_parts(f_state, params, sys);
        let pfaff = pstate.pfaff;
        let ip = pfaff * <f64>::exp(proj);
        kin += ip*sys.cons_t*sys.transfert_matrix[hop.0 + hop.1*sys.size];
    }

    trace!("Output kinetic <x|K|psi> = {:.2} for state |x> = {}", kin, state);
    kin
}

