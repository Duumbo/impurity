use log::trace;

use crate::density::fast_internal_product_no_otilde;
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
    T: BitOps + std::fmt::Display,
{
    let pot = ((state.spin_up & state.spin_down).count_ones() as f64) * sys.cons_u;
    trace!("Output potential <x|U|psi> = {:.2} for state |x> = {}", pot, state);
    pot * pstate.pfaff * <f64>::exp(proj)
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
    T: BitOps + From<SpinState> + std::fmt::Debug + std::fmt::Display
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
        let mut proj = previous_proj;
        let (ratio, _col, _colidx) = fast_internal_product_no_otilde(&state, &f_state, previous_pstate, &hop, &mut proj, params);
        let pfaff = previous_pstate.pfaff * ratio;
        let ip = pfaff * <f64>::exp(proj);
        trace!("Projection state: |x'> = {}, z = {}", f_state, ratio);
        trace!("Adding kinetic term t_[i,j]<x'|psi>: |x> = {}, |x'> = {}, hop = ({}, {}, {}) Computed <x'|psi>/<x|psi> = {}", state, f_state, hop.0, hop.1, hop.2, ratio);
        kin += ip*sys.cons_t*sys.transfert_matrix[hop.0 + hop.1*sys.size];
    }

    trace!("Output kinetic <x|K|psi> = {:.2} for state |x> = {}", kin, state);
    kin
}

