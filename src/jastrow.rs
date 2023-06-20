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
/// let state = FockState { spin_up: 5, spin_down: 4 };
/// let jastrow_params: Vec<f64> = vec![1.0; SIZE*SIZE];
/// assert_eq!(compute_jastrow_exp(state, &jastrow_params), 9.0);
/// ```
pub fn compute_jastrow_exp(fock_state: FockState, jastrow_params: &[f64]) -> f64 {
    // 1/2 sum {i\neq j} v_{ij}(n_i - 1)(n_j - 1)
    let mut jastrow_out = 0.0;
    for j in 1..SIZE {
        // Set all i positions that satisfy the condition
        let regular_nor = !(fock_state.spin_up ^ fock_state.spin_down);
        let shifted_nor = !(fock_state.spin_up.rotate_right(j as u32) ^ fock_state.spin_down.rotate_right(j as u32));
        let spin_mask = fock_state.spin_up ^ fock_state.spin_down.rotate_right(j as u32);
        let mut jastrow_set = regular_nor & shifted_nor;

        // Spend all the set bits on adding the correct jastrow param.
        let mut i = jastrow_set.leading_zeros() as usize;
        while i < SIZE {
            let looking_at_bit = 1 << (SIZE - 1 - i);
            if (spin_mask ^ looking_at_bit) & looking_at_bit == 0 {
                jastrow_out -= jastrow_params[i + j*SIZE];
            } else {
                jastrow_out += jastrow_params[i + j*SIZE];
            }
            jastrow_set ^= looking_at_bit;
            i = jastrow_set.leading_zeros() as usize;
        }
    }
    jastrow_out * 0.5
}

/// Computes the Jastrow fast update
/// # Not Implemented
#[inline(always)]
pub fn fast_update_jastrow() {
    unimplemented!();
}

#[cfg(test)]
mod test {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand::rngs::SmallRng;
    use crate::{FockState, SIZE, BitStruct};

    fn compute_jastrow_easy_to_follow(fock_state: FockState, jastrow_params: &[f64]) -> f64 {
        let mut jastrow_out = 0.0;
        for i in 0..SIZE {
            let bit_i = 1 << i;
            let ni_down = (fock_state.spin_down & bit_i).count_ones();
            let ni_up = (fock_state.spin_up & bit_i).count_ones();
            let ni: isize = (ni_down + ni_up) as isize;
            for j in 0..SIZE {
                if i == j {continue;}
                let bit_j = 1 << j;
                let nj_down = (fock_state.spin_down & bit_j).count_ones();
                let nj_up = (fock_state.spin_up & bit_j).count_ones();
                let nj: isize = (nj_down + nj_up) as isize;
                jastrow_out += jastrow_params[i + j*SIZE] * ((ni - 1) * (nj - 1)) as f64;
            }
        }
        jastrow_out * 0.5
    }

    #[test]
    fn test_jastrow() {
        let mut rng = SmallRng::seed_from_u64(42);
        for _ in 0..100 {
            let up = rng.gen::<BitStruct>();
            let down = rng.gen::<BitStruct>();
            let fock_state1 = FockState { spin_up: up, spin_down: down };
            let fock_state2 = FockState { spin_up: up, spin_down: down };
            let jastrow_params: Vec<f64> = vec![1.0; SIZE*SIZE];
            assert_eq!(
                compute_jastrow_exp(fock_state1, &jastrow_params),
                compute_jastrow_easy_to_follow(fock_state2, &jastrow_params)
            )
        }
    }
}

