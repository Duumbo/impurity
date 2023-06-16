use crate::{SIZE, FockState};

/// Computes the gutzwiller factor for a single fock state.
/// # Arguments
/// * __`fock_state`__ - The fock state encoding. Convention is to have the
/// first half of the bitstring be the spin up, for $i\in \[1,N\]$, and the other
/// half be the spin down, in the same order. Order of the $g_i$ matches the order
/// of sites in the bitstring.
/// * __`gutzwiller_params`__ - The variationnal parameters $g_i$ for the
/// Gutzwiller projector. The order is by sites.
/// * __[`SIZE`]__ - Constant. The number of sites in the system.
/// # Returns
/// * __`gutz_out`__ - The exponent of the Gutzwiller projector.
/// # Exemple
/// As an exemple, let's take the state $\lvert 5;5\rangle$ and variationnal parameters
/// all equal to one. Only $2$ bits are set on both bitstrings, so the result
/// should be $2$.
/// ```rust
/// use impurity::{FockState, SIZE};
/// use impurity::gutzwiller::compute_gutzwiller_exp;
/// let state = FockState { spin_up: 5, spin_down: 5 };
/// let gutzwiller_params: Vec<f64> = vec![1.0; SIZE];
/// assert_eq!(compute_gutzwiller_exp(state, &gutzwiller_params), 2.0);
/// ```
pub fn compute_gutzwiller_exp(fock_state: FockState, gutzwiller_params: &[f64]) -> f64 {
    // sum_i g_i n_i up n_i down
    let mut gutzwiller_sites = fock_state.spin_up & fock_state.spin_down;
    let mut gutz_out: f64 = 0.0;
    let mut i = gutzwiller_sites.leading_zeros() as usize;
    while i < SIZE {
        gutz_out += gutzwiller_params[i];
        gutzwiller_sites ^= 1 << (SIZE - 1 - i);
        i = gutzwiller_sites.leading_zeros() as usize;
    }
    gutz_out
}

/// Computes the gutzwiller fast update.
/// # Not Implemented
#[inline(always)]
pub fn fast_update_gutzwiller() {
    unimplemented!();
}
