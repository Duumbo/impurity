use crate::{SIZE, FockState};

/// Computes the Jastrow exponent for a single fock state.
/// # Arguments
/// * __`fock_state`__ - The fock state encoding. Convention is to have the
/// order $i,j\in\[1,N\]$ for both fields.
/// Order of the $v_{ij}$
/// matches the order of sites in the bitstring.
/// * __`jastrow_params`__ - The variationnal parameters $v_{ij}$ for the
/// Jastrow projector. Ordered by sites number, with small index $i$.
/// * __[`SIZE`]__ - Constant. The number of sites in the system.
/// # Returns
/// * __`jastrow_out`__ - The exponent of the Jastrow projector $P_J$.
/// # Exemple
/// As an exemple, we have the state $\lvert 5\;5\rangle$ and variationnal parameters
/// all equal to one. There are $6$ bits set to zero that can each see $5$ other
/// zeros. The sum should the equal $5\times6=30$ for each spin, totaling $60$.
/// ```rust
/// use impurity::{FockState, SIZE};
/// use impurity::jastrow::compute_jastrow_exp;
/// let state = FockState { spin_up: 5, spin_down: 5 };
/// let jastrow_params: Vec<f64> = vec![1.0; SIZE*SIZE];
/// assert_eq!(compute_jastrow_exp(state, &jastrow_params), 60.0);
/// ```
pub fn compute_jastrow_exp(fock_state: FockState, jastrow_params: &[f64]) -> f64 {
    // 1/2 sum {i\neq j} v_{ij}(n_i - 1)(n_j - 1)
    // Truth table for (n_i - 1)(n_j - 1)
    // n_i | n_j | result
    // 0   | 0   | 1
    // 0   | 1   | 0
    // 1   | 0   | 0
    // 1   | 1   | 0
    // Equiv NOR
    let mut jastrow_out = 0.0;
    for j in 1..SIZE {
        // Set all i positions that satisfy the condition
        let mut shifted_nor = !(fock_state.spin_up | fock_state.spin_up.rotate_right(j as u32));

        // Spend all the set bits on adding the correct jastrow param.
        let mut i = shifted_nor.leading_zeros() as usize;
        while i < SIZE {
            jastrow_out += jastrow_params[i + j*SIZE];
            shifted_nor ^= 1 << (SIZE - 1 - i);
            i = shifted_nor.leading_zeros() as usize;
        }
    }
    for j in 1..SIZE {
        // Set all i positions that satisfy the condition
        let mut shifted_nor = !(fock_state.spin_down | fock_state.spin_down.rotate_right(j as u32));

        // Spend all the set bits on adding the correct jastrow param.
        let mut i = shifted_nor.leading_zeros() as usize;
        while i < SIZE {
            jastrow_out += jastrow_params[i + j*SIZE];
            shifted_nor ^= 1 << (SIZE - 1 - i);
            i = shifted_nor.leading_zeros() as usize;
        }
    }
    jastrow_out
}

/// Computes the Jastrow fast update
/// # Not Implemented
#[inline(always)]
pub fn fast_update_jastrow() {
    unimplemented!();
}

