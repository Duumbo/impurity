use crate::{density::compute_internal_product, BitOps, FockState, CONS_U, Hopper};
use crate::{VarParams, CONS_T};

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
pub fn potential<T>(state: FockState<T>) -> f64
where
    T: BitOps,
{
    ((state.spin_up & state.spin_down).count_ones() as f64) * CONS_U
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
pub fn kinetic<T>(state: FockState<T>, params: &VarParams) -> f64
where
    T: BitOps + From<u8> + std::fmt::Debug + std::fmt::Display
{
    let hops = state.generate_all_hoppings();

    let mut kin = 0.0;
    for hop in hops.into_iter() {
        let mut f_state = state;
        if hop.2 == 0 {
            f_state.spin_down.set(hop.0);
            f_state.spin_down.set(hop.1);
        } else {
            f_state.spin_up.set(hop.0);
            f_state.spin_up.set(hop.1);
        }
        kin += compute_internal_product(f_state, params)*CONS_T;
    }

    kin
}

