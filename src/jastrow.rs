#[cfg(feature = "python-interface")]
use pyo3::prelude::*;

use crate::{BitOps, FockState};

/// Computes the Jastrow exponent for a single fock state.
/// # Arguments
/// * __`fock_state`__ - The fock state encoding. Convention is to have the
/// order $i,j\in\[1,N\]$ for both fields.
/// Order of the $v_{ij}$
/// matches the order of sites in the bitstring.
/// * __`jastrow_params`__ - The variationnal parameters $v_{ij}$ for the
/// Jastrow projector. Ordered by sites number, with small index $i$.
/// * __`n_sites`__ - The number of sites in the system. Important because of the
/// garbage data created if not taken into account.
/// # Returns
/// * __`jastrow_out`__ - The coefficient $\ln P_\text{J}$.
/// # Exemple
/// As an exemple, we have the state $\lvert 5\;5\rangle$ and variationnal parameters
/// all equal to one. There are $6$ bits set to zero that can each see $5$ other
/// zeros. The sum should the equal $5\times6=30$ for each spin, totaling $60$.
/// ```rust
/// use impurity::{FockState};
/// use impurity::jastrow::compute_jastrow_exp;
/// const SIZE: usize = 8;
/// let state = FockState { spin_up: 5u8, spin_down: 4u8, n_sites: 8};
/// let jastrow_params: Vec<f64> = vec![1.0; SIZE*SIZE];
/// assert_eq!(compute_jastrow_exp(state, &jastrow_params, state.n_sites), 9.0);
/// ```
pub fn compute_jastrow_exp<T>(
    fock_state: FockState<T>,
    jastrow_params: &[f64],
    n_sites: usize,
) -> f64
where
    T: BitOps + std::fmt::Display,
{
    let mut jastrow_out = 0.0;
    let mut regular_nor = !(fock_state.spin_up ^ fock_state.spin_down);
    regular_nor.mask_bits(n_sites);
    let mut i: usize = regular_nor.leading_zeros() as usize;
    let mut indices: Vec<usize> = Vec::with_capacity(n_sites);
    while i < n_sites {
        indices.push(i);
        regular_nor.set(i);
        for nk in 0..indices.len() - 1 {
            let (n1, n2) = (fock_state.spin_up, fock_state.spin_down);
            let k = indices[nk];
            if n1.check(i) ^ n2.check(k) {
                jastrow_out -= jastrow_params[i + k * n_sites];
            } else {
                jastrow_out += jastrow_params[i + k * n_sites];
            }
        }
        i = regular_nor.leading_zeros() as usize;
    }
    jastrow_out
}


fn jastrow_undo_update<T>(
    spin_mask: &mut T,
    previous_jas: &mut f64,
    jastrow_params: &[f64],
    fock_state: &FockState<T>,
    index_j: usize,
    index_skip: usize,
    n_sites: usize,
) where
    T: BitOps,
{
    if spin_mask.check(index_j) {
        //*spin_mask = *spin_mask & (<T>::ones() >> (index_j + 1));
        let mut i: usize = spin_mask.leading_zeros() as usize;
        while i < n_sites {
            spin_mask.set(i);
            if (i == index_j) | (i == index_skip) {
                i = spin_mask.leading_zeros() as usize;
                continue;
            }
            let (n1, n2) = (fock_state.spin_up, fock_state.spin_down);
            if n1.check(i) ^ n2.check(index_j) {
                *previous_jas += jastrow_params[i + index_j * n_sites];
            } else {
                *previous_jas -= jastrow_params[i + index_j * n_sites];
            }
            i = spin_mask.leading_zeros() as usize;
        }
    }
}

fn jastrow_do_update<T>(
    spin_mask: &mut T,
    previous_jas: &mut f64,
    jastrow_params: &[f64],
    fock_state: &FockState<T>,
    index_j: usize,
    index_skip: usize,
    n_sites: usize,
) where
    T: BitOps,
{
    if spin_mask.check(index_j) {
        //*spin_mask = *spin_mask & (<T>::ones() >> (index_j + 1));
        let mut i: usize = spin_mask.leading_zeros() as usize;
        while i < n_sites {
            spin_mask.set(i);
            if (i == index_j) | (i == index_skip) {
                i = spin_mask.leading_zeros() as usize;
                continue;
            }
            let (n1, n2) = (fock_state.spin_up, fock_state.spin_down);
            if n1.check(i) ^ n2.check(index_j) {
                *previous_jas -= jastrow_params[i + index_j * n_sites];
            } else {
                *previous_jas += jastrow_params[i + index_j * n_sites];
            }
            i = spin_mask.leading_zeros() as usize;
        }
    }
}

#[inline(always)]
fn jastrow_single_update<T>(
    spin_mask: &mut T,
    previous_jas: &mut f64,
    jastrow_params: &[f64],
    fock_state: &FockState<T>,
    index_j: usize,
    index_i: usize,
    sign: bool,
    n_sites: usize,
) where
    T: BitOps,
{
    if spin_mask.check(index_j) & spin_mask.check(index_i) {
        let (n1, n2) = (fock_state.spin_up, fock_state.spin_down);
        if n1.check(index_i) ^ n2.check(index_j) {
            if sign {
                *previous_jas -= jastrow_params[index_i + index_j * n_sites];
            } else {
                *previous_jas += jastrow_params[index_i + index_j * n_sites];
            }
        } else {
            if sign {
                *previous_jas += jastrow_params[index_i + index_j * n_sites];
            } else {
                *previous_jas -= jastrow_params[index_i + index_j * n_sites];
            }
        }
    }
}

/// Computes the Jastrow fast update
/// # Arguments
/// * __`previous_jas`__ - The previous jastrow coefficient $\ln P_\text{J}$.
/// `previous_jas` is modified to contain the updated coefficient.
/// * __`jastrow_params`__ - The variationnal parameters $v_{ij}$ for the
/// Jastrow projector. Ordered by sites number, with small index $i$.
/// * __`previous_fock`__ - The [FockState] before the hopping.
/// * __`new_fock`__ - The [FockState] after the hopping.
/// * __`n_sites`__ - The number of sites to consider. Important in order to
/// clear the garbage data that would be created.
/// * __`previous_j`__ - The previous index of the hopped spin.
/// * __`new_j`__ - The new index of the hopped spin.
/// # Exemple
///
/// ```rust
/// use impurity::{FockState};
/// use impurity::jastrow::{compute_jastrow_exp, fast_update_jastrow};
/// const SIZE: usize = 8;
/// let params: Vec<f64> = vec![1.0; SIZE * SIZE];
/// let mut res = 0.0;
/// let state = FockState {
///     spin_up: 21u8,
///     spin_down: 53u8,
///     n_sites: 8,
/// };
/// res = compute_jastrow_exp(state, &params, 8);
/// // The update is up 3->4
/// let newstate = FockState {
///     spin_up: 13u8,
///     spin_down: 53u8,
///     n_sites: 8,
/// };
/// fast_update_jastrow(&mut res, &params, &state, &newstate, state.n_sites, 3, 4);
/// assert_eq!(res, -2.0);
/// ```
pub fn fast_update_jastrow<T>(
    previous_jas: &mut f64,
    jastrow_params: &[f64],
    previous_fock: &FockState<T>,
    new_fock: &FockState<T>,
    n_sites: usize,
    previous_j: usize,
    new_j: usize,
) where
    T: BitOps + std::fmt::Display,
{
    // Undo previous fock state
    // 1/2 sum {i\neq j} v_{ij}(n_i - 1)(n_j - 1)
    let mut regular_nor = !(previous_fock.spin_up ^ previous_fock.spin_down);
    regular_nor.mask_bits(n_sites);
    // Set all i positions that satisfy the condition

    // Gotta undo
    jastrow_undo_update(
        &mut regular_nor.clone(),
        previous_jas,
        jastrow_params,
        previous_fock,
        previous_j,
        new_j,
        n_sites,
    );
    jastrow_undo_update(
        &mut regular_nor.clone(),
        previous_jas,
        jastrow_params,
        previous_fock,
        new_j,
        previous_j,
        n_sites,
    );
    jastrow_single_update(
        &mut regular_nor,
        previous_jas,
        jastrow_params,
        previous_fock,
        new_j,
        previous_j,
        false,
        n_sites,
    );

    // Now do
    let mut regular_nor = !(new_fock.spin_up ^ new_fock.spin_down);
    regular_nor.mask_bits(n_sites);
    jastrow_do_update(
        &mut regular_nor.clone(),
        previous_jas,
        jastrow_params,
        new_fock,
        previous_j,
        new_j,
        n_sites,
    );
    jastrow_do_update(
        &mut regular_nor.clone(),
        previous_jas,
        jastrow_params,
        new_fock,
        new_j,
        previous_j,
        n_sites,
    );
    jastrow_single_update(
        &mut regular_nor,
        previous_jas,
        jastrow_params,
        new_fock,
        new_j,
        previous_j,
        true,
        n_sites,
    );
}

#[cfg(feature = "python-interface")]
#[pyfunction]
pub fn jastrow_exponent(
    spin_up: u8,
    spin_down: u8,
    jastrow_params: [f64; 64],
    n_sites: usize,
) -> PyResult<f64>
{
    let fock_state = FockState{
        spin_up,
        spin_down,
        n_sites: 8
    };
    Ok(compute_jastrow_exp(fock_state, &jastrow_params, n_sites))
}

#[cfg(feature = "python-interface")]
#[pyfunction]
pub fn jastrow_fastupdate(
    previous_coefficient: f64,
    jastrow_params: [f64; 64],
    previous_state_up: u8,
    previous_state_down: u8,
    new_state_up: u8,
    new_state_down: u8,
    n_sites: usize,
    previous_index: usize,
    new_index: usize
) -> PyResult<f64>
{
    let mut out = previous_coefficient;
    let prev = FockState{
        spin_up: previous_state_up,
        spin_down: previous_state_down,
        n_sites
    };
    let new = FockState{
        spin_up: new_state_up,
        spin_down: new_state_down,
        n_sites
    };
    fast_update_jastrow(
        &mut out,
        &jastrow_params,
        &prev,
        &new,
        n_sites,
        previous_index,
        new_index
    );
    Ok(out)
}

#[cfg(feature = "python-interface")]
#[pyfunction]
pub fn compute_jastrow_easy_to_follow(
    sup: u8,
    sdown: u8,
    jastrow_params: [f64; 64],
    n_sites: usize
) -> f64
{
    let mut zeta: Vec<f64> = vec![];
    for i in 0..n_sites {
        zeta.push(
            (sup.check(i) as isize +
            sdown.check(i) as isize - 1) as f64
            );
    }
    let mut zeta_prime = zeta.clone();
    unsafe{
        dgemv(
            b"N"[0],
            n_sites as i32,
            n_sites as i32,
            1.0,
            &jastrow_params,
            n_sites as i32,
            &zeta,
            1,
            0.0,
            &mut zeta_prime,
            1
        );
        ddot(n_sites as i32, &zeta, 1, &zeta_prime, 1) * 0.5
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{FockState};
    use assert::close;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    fn compute_jastrow_easy_to_follow<T>(
        fock_state: FockState<T>,
        jastrow_params: &[f64],
        n_sites: usize
    ) -> f64
    where
        T: BitOps + From<u8>,
    {
        let mut zeta: Vec<f64> = vec![];
        for i in 0..n_sites {
            zeta.push(
                (fock_state.spin_up.check(i) as isize +
                fock_state.spin_down.check(i) as isize - 1) as f64
                );
        }
        println!("Zeta: {:?}", zeta);
        let mut zeta_prime = zeta.clone();
        unsafe{
            dgemv(
                b"N"[0],
                n_sites as i32,
                n_sites as i32,
                1.0,
                jastrow_params,
                n_sites as i32,
                &zeta,
                1,
                0.0,
                &mut zeta_prime,
                1
            );
            ddot(n_sites as i32, &zeta, 1, &zeta_prime, 1) * 0.5
        }
    }

    const fn u8_to_spin(u: u8, n_sites: usize) -> u8 {
        u << (8 - n_sites)
    }

    #[test]
    fn test_jastrow_easy() {
        const NSITES: usize = 3;
        let jastrow_params: Vec<f64> = vec![
            0.0, 0.3, 0.7,
            0.4, 0.0, -0.2,
            0.5, -0.9, 0.0
        ];
        let state1 = FockState{
            spin_up: u8_to_spin(5, NSITES),
            spin_down: u8_to_spin(3, NSITES),
            n_sites: NSITES
        };
        let state2 = FockState{
            spin_up: u8_to_spin(5, NSITES),
            spin_down: u8_to_spin(7, NSITES),
            n_sites: NSITES
        };

        assert_eq!(
            compute_jastrow_easy_to_follow(state1, &jastrow_params, NSITES),
            0.0
        );
        assert_eq!(
            compute_jastrow_easy_to_follow(state2, &jastrow_params, NSITES),
            0.6
        );
    }

    #[test]
    fn test_jastrow_u8() {
        const SIZE: usize = 8;
        let mut rng = SmallRng::seed_from_u64(42);
        const NSITES: usize = 8;
        for _ in 0..100 {
            let up = rng.gen::<u8>();
            let down = rng.gen::<u8>();
            let fock_state1 = FockState {
                spin_up: up,
                spin_down: down,
                n_sites: SIZE,
            };
            let fock_state2 = FockState {
                spin_up: up,
                spin_down: down,
                n_sites: SIZE,
            };
            let mut jastrow_params: Vec<f64> = Vec::with_capacity(SIZE * SIZE);
            for _ in 0..SIZE * SIZE {
                jastrow_params.push(rng.gen::<f64>());
            }
            for i in 0..NSITES {
                for j in 0..NSITES {
                    jastrow_params[j + i * NSITES] = jastrow_params[i + j * NSITES];
                    if i == j {jastrow_params[i + j* NSITES] = 0.0;}
                }
            }
            close(
                compute_jastrow_exp(fock_state1, &jastrow_params, NSITES),
                compute_jastrow_easy_to_follow(fock_state2, &jastrow_params, NSITES),
                1e-12,
            )
        }
    }

    #[test]
    fn test_fast_update_jastrow_u8() {
        const SIZE: usize = 8;
        let mut rng = SmallRng::seed_from_u64(42);
        let mut up = rng.gen::<u8>();
        let down = rng.gen::<u8>();
        let mut fock_state = FockState {
            spin_up: up,
            spin_down: down,
            n_sites: SIZE,
        };
        let mut jastrow_params: Vec<f64> = Vec::with_capacity(64);
        for _ in 0..64 {
            jastrow_params.push(rng.gen());
        }
        for i in 0..8 {
            for j in 0..8 {
                jastrow_params[j + i * 8] = jastrow_params[i + j * 8];
            }
        }
        let mut jastrow = compute_jastrow_exp(fock_state.clone(), &jastrow_params, 8);
        println!("previous_jas: {}", jastrow);
        for _ in 0..100 {
            let spin_update: f64 = rng.gen();
            // Spin up
            if spin_update < 0.5 {
                let old_index: usize = up.leading_zeros().try_into().unwrap();
                let new_index: usize = (rng.gen::<u8>() % 8).into();
                println!(
                    "old_j: {}, new_j: {}, old_up: {}, down: {}",
                    old_index, new_index, up, down
                );
                if up.check(new_index) {
                    continue;
                }
                up.set(old_index);
                up.set(new_index);
                println!("new up: {}", up);
                let new_fock_state = FockState {
                    spin_up: up,
                    spin_down: down,
                    n_sites: SIZE,
                };
                let full_jastrow = compute_jastrow_exp(new_fock_state.clone(), &jastrow_params, 8);
                fast_update_jastrow(
                    &mut jastrow,
                    &jastrow_params,
                    &fock_state,
                    &new_fock_state,
                    8,
                    old_index,
                    new_index,
                );
                close(full_jastrow, jastrow, 1e-12);
                fock_state = new_fock_state;
            }
            // Spin down
            else {
            }
        }
    }

    #[test]
    fn test_jastrow_u8_5sites() {
        let mut rng = SmallRng::seed_from_u64(42);
        const NSITES: usize = 5;
        for _ in 0..100 {
            // Generate random state
            let mut fock_state1 = rng.gen::<FockState<u8>>();
            fock_state1.n_sites = NSITES;
            println!("State: {}", fock_state1);

            // Copy the state
            let fock_state2 = fock_state1.clone();

            let mut jastrow_params: Vec<f64> = Vec::with_capacity(NSITES * NSITES);
            for _ in 0..(NSITES * NSITES) {
                jastrow_params.push(rng.gen::<f64>());
            }
            for i in 0..NSITES {
                for j in 0..NSITES {
                    jastrow_params[j + i * NSITES] = jastrow_params[i + j * NSITES];
                    if i == j {jastrow_params[i + j* NSITES] = 0.0;}
                }
            }
            println!("Params: {:?}", jastrow_params);
            close(
                compute_jastrow_exp(fock_state1, &jastrow_params, NSITES),
                compute_jastrow_easy_to_follow(fock_state2, &jastrow_params, NSITES),
                1e-12,
            )
        }
    }
}
