use std::fmt::Debug;
#[cfg(feature = "python-interface")]
use pyo3::prelude::*;

use crate::{BitOps, FockState};

/// Computes the gutzwiller factor for a single fock state.
/// # Arguments
/// * __`fock_state`__ - The fock state encoding. Convention is to have the
/// first half of the bitstring be the spin up, for $i\in \[1,N\]$, and the other
/// half be the spin down, in the same order. Order of the $g_i$ matches the order
/// of sites in the bitstring.
/// * __`gutzwiller_params`__ - The variationnal parameters $g_i$ for the
/// Gutzwiller projector. The order is by sites.
/// * __`n_sites`__ - The number of sites in the system.
/// # Returns
/// * __`gutz_out`__ - The exponent of the Gutzwiller projector.
/// # Example
/// As an exemple, let's take the state $\lvert 5;5\rangle$ and variationnal parameters
/// all equal to one. Only $2$ bits are set on both bitstrings, so the result
/// should be $2$.
/// ```rust
/// use impurity::{FockState, SIZE};
/// use impurity::gutzwiller::compute_gutzwiller_exp;
/// let state = FockState { spin_up: 5u8, spin_down: 5u8, n_sites: 8};
/// let gutzwiller_params: Vec<f64> = vec![1.0; SIZE];
/// assert_eq!(compute_gutzwiller_exp(state, &gutzwiller_params, state.n_sites), 2.0);
/// ```
pub fn compute_gutzwiller_exp<T>(
    fock_state: FockState<T>,
    gutzwiller_params: &[f64],
    n_sites: usize,
) -> f64
where
    T: BitOps + From<u8> + std::ops::Shl<usize, Output = T> + Debug,
{
    // sum_i g_i n_i up n_i down
    let mut gutzwiller_sites = fock_state.spin_up & fock_state.spin_down;
    gutzwiller_sites.mask_bits(n_sites);
    let mut gutz_out: f64 = 0.0;
    let mut i = gutzwiller_sites.leading_zeros() as usize;
    while i < n_sites {
        gutz_out += gutzwiller_params[i];
        gutzwiller_sites.set(i);
        i = gutzwiller_sites.leading_zeros() as usize;
    }
    gutz_out
}

/// Computes the gutzwiller fast update.
/// # Arguments
/// * __`previous_gutz`__ - The previous Gutzwiller coefficient $\ln P_{\text{G}}$.
/// `previous_gutz` is modified to contain the updated coefficient.
/// * __`gutzwiller_params`__ - The variationnal parameters $g_i$ for the
/// Gutzwiller projector. The order is by sites.
/// * __`previous_other_state`__ - The opposite spin part of the previous [FockState].
/// * __`previous_index`__ - The index before the hopping.
/// * __`new_index`__ - The index after the hopping.
/// # Example
///
/// ```rust
/// use impurity::{FockState, SIZE};
/// use impurity::gutzwiller::{fast_update_gutzwiller, compute_gutzwiller_exp};
/// const N_SITES: usize = 8;
/// let params: Vec<f64> = vec![1.0; SIZE];
/// let mut res = 0.0;
/// let state = FockState {
///     spin_up: 21u8,
///     spin_down: 53u8,
///     n_sites: 8,
/// }; // Should give 3.0
/// res = compute_gutzwiller_exp(state.clone(), &params, N_SITES);
/// // Now the benchmark will test the fast-update
/// // The update is up 3->4
/// fast_update_gutzwiller(&mut res, &params, &state.spin_up, 3, 4);
/// assert_eq!(res, 2.0);
/// ```
#[inline(always)]
pub fn fast_update_gutzwiller<T>(
    previous_gutz: &mut f64,
    gutzwiller_params: &[f64],
    previous_other_state: &T,
    previous_index: usize,
    new_index: usize,
) where
    T: BitOps,
{
    if previous_other_state.check(previous_index) {
        *previous_gutz -= gutzwiller_params[previous_index];
    }
    if previous_other_state.check(new_index) {
        *previous_gutz += gutzwiller_params[new_index];
    }
}

#[cfg(feature = "python-interface")]
#[pyfunction]
pub fn gutzwiller_exponent(
    spin_up: u8,
    spin_down: u8,
    gutzwiller_params: [f64; 8],
    n_sites: usize,
) -> PyResult<f64>
{
    let fock_state = FockState{
        spin_up,
        spin_down,
        n_sites: 8
    };
    Ok(compute_gutzwiller_exp(fock_state, &gutzwiller_params, n_sites))
}


#[cfg(feature = "python-interface")]
#[pyfunction]
pub fn gutzwiller_fastupdate(
    previous_coefficient: f64,
    gutzwiller_params: [f64; 8],
    previous_state_up: u8,
    previous_state_down: u8,
    previous_index: usize,
    new_index: usize,
    spin: bool
) -> PyResult<f64>
{
    let mut out = previous_coefficient;
    if spin {
        fast_update_gutzwiller(&mut out, &gutzwiller_params, &previous_state_down, previous_index, new_index)
    }
    else {
        fast_update_gutzwiller(&mut out, &gutzwiller_params, &previous_state_up, previous_index, new_index)
    }
    Ok(out)
}

#[cfg(test)]
mod test {
    use crate::gutzwiller::fast_update_gutzwiller;
    use crate::{BitOps, FockState, SpinState, ARRAY_SIZE, SIZE};
    use assert::close;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use super::compute_gutzwiller_exp;

    // This test choses a random state and random params and computes the
    // Gutzwiller exponent.

    #[test]
    fn test_gutzwiller_exp_u8() {
        let mut rng = SmallRng::seed_from_u64(42);
        // This is a random test, run it five times.
        for test_iter in 0..100 {
            // Generate random state.
            let mut state = rng.gen::<FockState<u8>>();
            state.n_sites = SIZE;

            // Get the index that are the same.
            let mut e_and = state.spin_up & state.spin_down;
            e_and.mask_bits(8);
            let mut ind: Vec<u8> = Vec::new();
            for i in 0..8 {
                if (e_and & (1 << (8 - 1 - i))) == (1 << (8 - 1 - i)) {
                    ind.push(i as u8);
                }
            }

            let mut rng_params: Vec<f64> = Vec::with_capacity(SIZE);
            for _ in 0..SIZE {
                rng_params.push(rng.gen::<f64>());
            }

            // Compute the result manually
            let mut manual = 0.0;
            for i in ind.iter() {
                manual += &rng_params[*i as usize];
            }

            // These should be exactly the same always.
            println!(
                "Test iteration: {}, Computed: {}, Test implementation: {}",
                test_iter,
                compute_gutzwiller_exp(state, &rng_params, 8),
                manual,
            );
            assert_eq!(compute_gutzwiller_exp(state, &rng_params, 8), manual);
        }
    }

    #[test]
    fn test_fast_update_gutzwiller_exp_u8() {
        let mut rng = SmallRng::seed_from_u64(42);
        for test_iter in 0..100 {
            // Generate random state.
            let mut state = rng.gen::<FockState<u8>>();
            state.n_sites = SIZE;

            let mut rng_params: Vec<f64> = Vec::with_capacity(SIZE);
            for _ in 0..SIZE {
                rng_params.push(rng.gen::<f64>());
            }

            let mut gutz = compute_gutzwiller_exp(state.clone(), &rng_params, 8);
            let spin_update: f64 = rng.gen();
            let mut e_up = state.spin_up;
            let mut e_down = state.spin_down;
            for j in 0..12 {
                if spin_update < 0.5 {
                    let old_index: usize = e_up.leading_zeros().try_into().unwrap();
                    let new_index: usize = (rng.gen::<u8>() % 8).into();
                    if e_up.check(new_index) {
                        continue;
                    }
                    e_up.set(old_index);
                    e_up.set(new_index);
                    let new_fock = FockState {
                        spin_up: e_up,
                        spin_down: e_down,
                        n_sites: SIZE,
                    };
                    fast_update_gutzwiller(
                        &mut gutz,
                        &rng_params,
                        &state.spin_down,
                        old_index,
                        new_index,
                    );
                    let long_gutz = compute_gutzwiller_exp(new_fock.clone(), &rng_params, 8);
                    close(gutz, long_gutz, 1e-12);
                    state = new_fock;
                    println!(
                        "Test iteration: {}, hopping: {}, Computed: {}, Test implementation: {}",
                        test_iter, j, gutz, long_gutz
                    );
                } else {
                    let old_index: usize = e_down.leading_zeros().try_into().unwrap();
                    let new_index: usize = (rng.gen::<u8>() % 8).into();
                    if e_down.check(new_index) {
                        continue;
                    }
                    e_down.set(old_index);
                    e_down.set(new_index);
                    let new_fock = FockState {
                        spin_up: e_up,
                        spin_down: e_down,
                        n_sites: SIZE,
                    };
                    fast_update_gutzwiller(
                        &mut gutz,
                        &rng_params,
                        &state.spin_up,
                        old_index,
                        new_index,
                    );
                    let long_gutz = compute_gutzwiller_exp(new_fock.clone(), &rng_params, 8);
                    close(gutz, long_gutz, 1e-12);
                    state = new_fock;
                    println!(
                        "Test iteration: {}, hopping: {}, Computed: {}, Test implementation: {}",
                        test_iter, j, gutz, long_gutz
                    );
                }
            }
        }
    }

    #[test]
    fn test_fast_update_gutzwiller_exp_u8_5sites() {
        const N_SITES: usize = 5;
        let mut rng = SmallRng::seed_from_u64(42);
        for test_iter in 0..100 {
            // Random up state.
            let mut e_up = rng.gen::<u8>();
            e_up.mask_bits(5);

            // Random down state.
            let mut e_down = rng.gen::<u8>();
            e_down.mask_bits(5);

            let mut state = FockState {
                spin_up: e_up,
                spin_down: e_down,
                n_sites: SIZE,
            };
            let mut rng_params: Vec<f64> = Vec::with_capacity(N_SITES);
            for _ in 0..SIZE {
                rng_params.push(rng.gen::<f64>());
            }

            let mut gutz = compute_gutzwiller_exp(state.clone(), &rng_params, N_SITES);
            let spin_update: f64 = rng.gen();
            for j in 0..12 {
                if spin_update < 0.5 {
                    let old_index: usize = e_up.leading_zeros().try_into().unwrap();
                    let new_index: usize = (rng.gen::<u8>() % N_SITES as u8).into();
                    if e_up.check(new_index) {
                        continue;
                    }
                    e_up.set(old_index);
                    e_up.set(new_index);
                    e_up.mask_bits(5);
                    let new_fock = FockState {
                        spin_up: e_up,
                        spin_down: e_down,
                        n_sites: SIZE,
                    };
                    fast_update_gutzwiller(
                        &mut gutz,
                        &rng_params,
                        &state.spin_down,
                        old_index,
                        new_index,
                    );
                    let long_gutz = compute_gutzwiller_exp(new_fock.clone(), &rng_params, N_SITES);
                    println!(
                        "Test iteration: {}, hopping up: {}, Computed: {}, Test implementation: {}",
                        test_iter, j, gutz, long_gutz
                    );
                    close(gutz, long_gutz, 1e-12);
                    state = new_fock;
                } else {
                    let old_index: usize = e_down.leading_zeros().try_into().unwrap();
                    let new_index: usize = (rng.gen::<u8>() % N_SITES as u8).into();
                    if e_down.check(new_index) {
                        continue;
                    }
                    e_down.set(old_index);
                    e_down.set(new_index);
                    e_down.mask_bits(5);
                    let new_fock = FockState {
                        spin_up: e_up,
                        spin_down: e_down,
                        n_sites: SIZE,
                    };
                    fast_update_gutzwiller(
                        &mut gutz,
                        &rng_params,
                        &state.spin_up,
                        old_index,
                        new_index,
                    );
                    let long_gutz = compute_gutzwiller_exp(new_fock.clone(), &rng_params, N_SITES);
                    println!("Test iteration: {}, hopping down: {}, Computed: {}, Test implementation: {}",
                             test_iter, j, gutz, long_gutz);
                    close(gutz, long_gutz, 1e-12);
                    state = new_fock;
                }
            }
        }
    }

    #[test]
    fn test_gutzwiller_exp_spin_state() {
        let mut rng = SmallRng::seed_from_u64(42);
        // This is a random test, run it five times.
        for test_iter in 0..10 {
            let mut e_up = [0; ARRAY_SIZE];
            let mut e_down = [0; ARRAY_SIZE];
            for i in 0..ARRAY_SIZE {
                // Random up state.
                e_up[i] = rng.gen::<u8>();
                // Random down state.
                e_down[i] = rng.gen::<u8>();
            }
            let e_up = SpinState {
                state: e_up,
                n_elec: SIZE,
            };
            let e_down = SpinState {
                state: e_down,
                n_elec: SIZE,
            };

            // Get the index that are the same.
            let mut e_and = e_down & e_up;
            e_and.mask_bits(SIZE);
            let mut ind: Vec<u8> = Vec::new();
            let mut i: u8 = e_and.leading_zeros().try_into().unwrap();
            while (i as usize) < SIZE {
                i = e_and.leading_zeros().try_into().unwrap();
                if (i as usize) < SIZE {
                    ind.push(i);
                }
                e_and.set(e_and.leading_zeros() as usize);
                i = e_and.leading_zeros().try_into().unwrap();
            }

            let state = FockState {
                spin_up: e_up,
                spin_down: e_down,
                n_sites: SIZE,
            };
            let mut rng_params: Vec<f64> = Vec::with_capacity(SIZE);
            for _ in 0..SIZE {
                rng_params.push(rng.gen::<f64>());
            }

            // Compute the result manually
            let mut manual = 0.0;
            for i in ind.iter() {
                manual += &rng_params[*i as usize];
            }

            // These should be exactly the same always.
            println!(
                "Test iteration: {}, Computed: {}, Test implementation: {}",
                test_iter,
                compute_gutzwiller_exp(state.clone(), &rng_params, SIZE),
                manual,
            );
            assert_eq!(compute_gutzwiller_exp(state, &rng_params, SIZE), manual);
        }
    }
}
