use crate::{FockState, BitOps};

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
pub fn compute_jastrow_exp<T>(fock_state: FockState<T>, jastrow_params: &[f64], n_sites: usize, max_size: usize) -> f64
where T: BitOps + std::fmt::Display
{
    let mut jastrow_out = 0.0;
    let mut regular_nor = !(fock_state.spin_up ^ fock_state.spin_down);
    regular_nor &= <u64>::max_value() << (64 - n_sites);
    let mut i: usize = regular_nor.leading_zeros() as usize;
    let mut indices: Vec<usize> = Vec::with_capacity(n_sites);
    while i < n_sites {
        indices.push(i);
        regular_nor ^= 1 << (64 - 1 - i);
        for nk in 0..indices.len() - 1 {
            let (n1, n2) = (fock_state.spin_up, fock_state.spin_down);
            let k = indices[nk];
            if !(n1 & ( 1 << (64 - 1 - i)) == 0) ^ !(n2 & (1 << (64 - 1 - k)) == 0) {
               jastrow_out -= jastrow_params[i + k*n_sites];
            } else {
               jastrow_out += jastrow_params[i + k*n_sites];
            }
        }
        i = regular_nor.leading_zeros() as usize;
    }
    jastrow_out
}

/// Computes the Jastrow fast update
/// # Not Implemented
pub fn fast_update_jastrow<T>(
    previous_jas: &mut f64,
    jastrow_params: &[f64],
    previous_fock: &FockState<T>,
    new_fock: &FockState<T>,
    n_sites: usize,
    max_size: usize,
    previous_j: usize,
    new_j: usize)
where T: BitOps + std::fmt::Display
{
    println!("Call to test");
    // Undo previous fock state
    // 1/2 sum {i\neq j} v_{ij}(n_i - 1)(n_j - 1)
    let mut regular_nor = !(previous_fock.spin_up ^ previous_fock.spin_down);
    regular_nor.mask_bits(n_sites);
    // Set all i positions that satisfy the condition

    // Gotta undo
    if regular_nor.check(previous_j) {
        for i in 0..n_sites {
            let spin_mask = previous_fock.spin_up ^ previous_fock.spin_down.rotate_right(i as u32);
            if spin_mask.check(i) {
                println!("Undo jastrow add: ({}, {})", i, previous_j);
                *previous_jas += jastrow_params[previous_j + i*n_sites];
            } else {
                println!("Undo jastrow sub: ({}, {})", i, previous_j);
                *previous_jas -= jastrow_params[previous_j + i*n_sites];
            }
        }
    }

    let mut regular_nor = !(new_fock.spin_up ^ new_fock.spin_down);
    regular_nor.mask_bits(n_sites);
    for i in 0..n_sites {
        let spin_mask = previous_fock.spin_up ^ previous_fock.spin_down.rotate_right(i as u32);
        if regular_nor.check(previous_j) {
            if spin_mask.check(i) {
                println!("Update jastrow sub: ({}, {})", i, previous_j);
                *previous_jas -= jastrow_params[previous_j + i*n_sites];
            } else {
                println!("Update jastrow add: ({}, {})", i, previous_j);
                *previous_jas += jastrow_params[previous_j + i*n_sites];
            }
        }
        if regular_nor.check(new_j) {
            if spin_mask.check(i) {
                println!("Update jastrow sub: ({}, {})", i, new_j);
                *previous_jas -= jastrow_params[new_j + i*n_sites];
            } else {
                println!("Update jastrow add: ({}, {})", i, new_j);
                *previous_jas += jastrow_params[new_j + i*n_sites];
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand::rngs::SmallRng;
    use crate::{FockState, SIZE};
    use assert::close;

    fn compute_jastrow_easy_to_follow<T>(fock_state: FockState<T>, jastrow_params: &[f64], n_sites: usize, max_size: usize) -> f64
    where T: BitOps + From<u8>
    {
        let mut jastrow_out = 0.0;
        for i in 0..n_sites {
            let bit_i: T = ((1 as u8) << (max_size - 1 - i)).into();
            let ni_down = (fock_state.spin_down & bit_i).count_ones();
            let ni_up = (fock_state.spin_up & bit_i).count_ones();
            let ni: isize = (ni_down + ni_up) as isize;
            for j in 0..n_sites {
                if i == j {continue;}
                let bit_j: T = (1 << (max_size - 1 - j)).into();
                let nj_down = (fock_state.spin_down & bit_j).count_ones();
                let nj_up = (fock_state.spin_up & bit_j).count_ones();
                let nj: isize = (nj_down + nj_up) as isize;
                jastrow_out += jastrow_params[i + j*n_sites] * ((ni - 1) * (nj - 1)) as f64;
                if (ni != 1) && (nj != 1) {
                    println!("Added: {}, at ({}, {})",
                             jastrow_params[i + j*n_sites] * ((ni - 1) * (nj - 1)) as f64,
                             i, j);
                }
            }
        }
        jastrow_out * 0.5
    }

    #[test]
    fn test_jastrow_u8() {
        let mut rng = SmallRng::seed_from_u64(42);
        for _ in 0..100 {
            let up = rng.gen::<u8>();
            let down = rng.gen::<u8>();
            let fock_state1 = FockState { spin_up: up, spin_down: down };
            let fock_state2 = FockState { spin_up: up, spin_down: down };
            let mut jastrow_params: Vec<f64> = Vec::with_capacity(SIZE*SIZE);
            for _ in 0..SIZE*SIZE {jastrow_params.push(rng.gen::<f64>());}
            for i in 0..8 {
                for j in 0..8 {
                    jastrow_params[j + i*8] = jastrow_params[i + j*8];
                }
            }
            close(
                compute_jastrow_exp(fock_state1, &jastrow_params, 8, 8),
                compute_jastrow_easy_to_follow(fock_state2, &jastrow_params, 8, 8),
                1e-12
            )
        }
    }

    #[test]
    fn test_fast_update_jastrow_u8() {
        let mut rng = SmallRng::seed_from_u64(42);
        let mut up = rng.gen::<u8>();
        let down = rng.gen::<u8>();
        let mut fock_state = FockState { spin_up: up, spin_down: down };
        let mut jastrow_params: Vec<f64> = Vec::with_capacity(64);
        for _ in 0..64{jastrow_params.push(rng.gen());}
        for i in 0..8 {
            for j in 0..8 {
                jastrow_params[j + i*8] = jastrow_params[i + j*8];
            }
        }
        let mut jastrow = compute_jastrow_exp(fock_state.clone(), &jastrow_params, 8, 8);
        println!("previous_jas: {}", jastrow);
        for _ in 0..100 {
            let spin_update: f64 = rng.gen();
            // Spin up
            if spin_update < 0.5 {
                let old_index: usize = up.leading_zeros().try_into().unwrap();
                let new_index: usize = (rng.gen::<u8>() % 8).into();
                println!("old_j: {}, new_j: {}, old_up: {}, down: {}", old_index, new_index, up, down);
                if up.check(new_index) {
                    continue;
                }
                up.set(old_index);
                up.set(new_index);
                println!("new up: {}", up);
                let new_fock_state = FockState { spin_up: up, spin_down: down };
                let full_jastrow = compute_jastrow_exp(new_fock_state.clone(), &jastrow_params, 8, 8);
                fast_update_jastrow(&mut jastrow, &jastrow_params, &fock_state, &new_fock_state, 8, 8, old_index, new_index);
                assert_eq!(full_jastrow, jastrow);
                fock_state = new_fock_state;
            } // Spin down
            else {
            }
        }
    }

    #[test]
    fn test_jastrow_u8_5sites() {
        let mut rng = SmallRng::seed_from_u64(42);
        for _ in 0..100 {
            let up = rng.gen::<u8>();
            let down = rng.gen::<u8>();
            println!("up, {}, down, {}", up, down);
            let fock_state1 = FockState { spin_up: up, spin_down: down };
            let fock_state2 = FockState { spin_up: up, spin_down: down };
            let mut jastrow_params: Vec<f64> = Vec::with_capacity(SIZE*SIZE);
            for _ in 0..SIZE*SIZE {jastrow_params.push(rng.gen::<f64>());}
            assert_eq!(
                compute_jastrow_exp(fock_state1, &jastrow_params, 5, 8),
                compute_jastrow_easy_to_follow(fock_state2, &jastrow_params, 5, 8)
            )
        }
    }
}
