use crate::{BitOps, FockState, SIZE};
use blas::{daxpy, ddot, dgemv, dger, dgemm};
use lapack::{dgetrf, dgetri};
use pfapack::skpfa;
use std::fmt;

pub enum Spin {
    Up,
    Down,
}

/// Represents the Pfaffian state $\lvert\phi_{\text{PF}}\rangle$.
/// #TODOC
/// # Fields
/// * __`coeffs`__ - The variationnal parameters $f_{ij}$.
/// * __`n_elec`__ - The number of electrons. This value is constant and
/// determines the size of the matrix $A$.
/// * __`matrix`__ - The matrix $A$. This is the matrix that we need the pfaffian
/// to get the inner product $\braket{x}{\phi_{\text{PF}}}$
/// * __`curr_state`__ - The current state of that the matrix $A$ is written in.
pub struct PfaffianState {
    pub n_elec: usize,
    pub n_sites: usize,
    pub pfaff: f64,
    pub coeffs: Vec<f64>,
    indices: (Vec<usize>, Vec<usize>),
    pub inv_matrix: Vec<f64>,
}

/// The pfaffian state implementation.
/// # Provides
/// * __`rebuild_matrix`__ - A function to rebuild the $A^{-1}$ matrix from the lower
/// triangle.
impl PfaffianState {
    pub fn rebuild_matrix(&mut self) {
        for i in 0..self.n_elec {
            for j in 0..self.n_elec {
                self.inv_matrix[j + i * self.n_elec] = -self.inv_matrix[i + j * self.n_elec];
            }
        }
    }
}

// Just a pretty display for the $A^{-1}$ matrix.
impl fmt::Display for PfaffianState {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> fmt::Result {
        let width = 8;
        for i in 0..self.n_elec {
            write!(f, "\n|").unwrap();
            for j in 0..self.n_elec {
                write!(
                    f, " {:width$} ",
                    (self.inv_matrix[j + self.n_elec * i] * 1000.0).round() / 1000.0
                ).unwrap();
            }
            write!(f, "|").unwrap();
        }
        Ok(())
    }
}

/// Inverts a matrix.
/// # Fields
/// * __`a`__ - The matrix $A$ of the variationnal parameters.
/// * __`n`__ - The dimension of the matrix, this correspond to the number of
/// electrons.
fn invert_matrix(a: &mut [f64], n: i32) {
    // Info output of lapack
    let mut info1: i32 = 0;
    let mut info2: i32 = 0;

    // Length of work vector
    let n_entry: i32 = n * n;
    // Workspaces
    let mut work: Vec<f64> = Vec::with_capacity(n_entry as usize);
    let mut ipiv: Vec<i32> = Vec::with_capacity(n as usize);

    // Inverse matrix `a` inplace using L*U decomposition.
    unsafe {
        dgetrf(n, n, a, n, &mut ipiv, &mut info1);
        dgetri(n, a, n, &ipiv, &mut work, n_entry, &mut info2);
    }

    // These should never be not 0.
    // If this panics, then a was not of size n, most probably.
    // This will panic if the matrix is singular
    // Refer to LAPACK error message.
    if !(info1 == 0) || !(info2 == 0) {
        println!(
            "The algorithm failed to invert the matrix. DGETRF: info={}, DGETRI: info={}",
            info1, info2
        );
        panic!("Matrix invertion fail.");
    }
}

fn matrix_product(a: &[f64], b: &[f64]) -> Vec<f64>{
    let n: i32 = <f64>::sqrt(a.len() as f64) as i32;
    let mut c = Vec::with_capacity((n*n) as usize);
    unsafe {
        c.set_len((n*n) as usize);
        dgemm(b"N"[0], b"N"[0], n, n, n, 1.0, a, n, b, n, 0.0, &mut c, n)
    }

    c
}

fn transpose(a: &Vec<f64>, n: usize) -> Vec<f64>{
    let mut b = a.clone();
    for i in 0..n {
        for j in 0..n {
            b[i*n + j] = a[j*n +i];
        }
    }
    b
}

/// Constructs pfaffian matrix from state.
/// # Fields
/// * __`fij`__ - All the variationnal parameters.
/// * __`state`__ - The state of the system.
pub fn construct_matrix_a_from_state<T>(fij: Vec<f64>, state: FockState<T>) -> PfaffianState
where
    T: BitOps + std::fmt::Display,
{
    // Fij upup, updown, downup, downdown
    let n = state.spin_up.count_ones() as usize + state.spin_down.count_ones() as usize;
    let mut a: Vec<f64> = vec![0.0; n * n];

    let mut indices: Vec<usize> = Vec::with_capacity(n);
    let mut indices2: Vec<usize> = Vec::with_capacity(n);
    let (mut spin_up, mut spin_down) = (state.spin_up, state.spin_down);
    let (mut i, mut j): (usize, usize) = (
        spin_up.leading_zeros() as usize,
        spin_down.leading_zeros() as usize,
    );
    while i < state.n_sites {
        indices.push(i);
        spin_up.set(i);
        i = spin_up.leading_zeros() as usize;
    }
    while j < state.n_sites {
        indices2.push(j);
        spin_down.set(j);
        j = spin_down.leading_zeros() as usize;
    }
    let off = indices.len();
    let size = state.n_sites;

    // X_{ij}=F_{ij}^{\sigma_i,\sigma_j} - F_{ji}^{\sigma_j,\sigma_i}, tahara2008
    // +0 -> upup, +SIZE^2 -> updown, +2*SIZE^2 -> downup, +3*SIZE^2 -> down down
    for jj in 0..indices2.len() {
        for ii in 0..indices.len() {
            a[ii + (jj + off) * n] =
                fij[indices[ii] + size * indices2[jj] + size*size]
                -fij[indices2[jj] + size * indices[ii] + 2*size*size];
            a[jj + off + ii * n] =
                fij[indices2[jj] + size * indices[ii] + 2*size*size]
                -fij[indices[ii] + size * indices2[jj] + size*size];
        }
    }
    for jj in 0..indices.len() {
        for ii in 0..indices.len() {
            if indices[ii] == indices[jj] {
                continue;
            }
            a[ii + jj * n] =
                fij[indices[ii] + size * indices[jj]]
                -fij[indices[jj] + size * indices[ii]];
            a[jj + ii * n] =
                fij[indices[jj] + size * indices[ii]]
                -fij[indices[ii] + size * indices[jj]];
        }
    }
    for jj in 0..indices2.len() {
        for ii in 0..indices2.len() {
            if indices2[ii] == indices2[jj] {
                continue;
            }
            a[ii + off + (jj + off) * n] =
                 fij[indices2[ii] + size * indices2[jj] + 3*size*size]
                 -fij[indices2[jj] + size * indices2[ii] + 3*size*size];
            a[jj + off + (ii + off) * n] =
                 fij[indices2[jj] + size * indices2[ii] + 3*size*size]
                 -fij[indices2[ii] + size * indices2[jj] + 3*size*size];
        }
    }

    // Invert matrix.
    let pfaffian_value = compute_pfaffian_wq(&mut a.clone(), n as i32);
    invert_matrix(&mut a, n as i32);

    PfaffianState {
        coeffs: fij,
        n_elec: n,
        n_sites: state.n_sites,
        inv_matrix: a,
        indices: (indices, indices2),
        pfaff: pfaffian_value,
    }
}

/// Gets the ratio of pfaffian after an update.
/// # Fields
/// * __`previous_pstate`__ - The pfaffian state to update.
/// * __`previous_i`__ - The initial index of the jumping electron.
/// * __`new_i`__ - The index of the electron after the jump.
/// * __`spin`__ - The spin of the jumping electron.
///
/// # Returns
/// * __`pfaff_up`__ - The new pfaffian ratio after the update.
/// * __`new_b`__ - The new column and row in the matrix $A$.
/// * __`col`__ - The index of the column and row that changed in the matrix $A$.
pub fn get_pfaffian_ratio(
    previous_pstate: &PfaffianState,
    previous_i: usize,
    new_i: usize,
    spin: Spin,
) -> (f64, Vec<f64>, usize) {

    // Rename
    let indx_up = &previous_pstate.indices.0;
    let indx_down = &previous_pstate.indices.1;
    let fij = &previous_pstate.coeffs;
    let n_sites = previous_pstate.n_sites;
    let n_elec = previous_pstate.n_elec;

    // Gen new vector b
    // X_{ij}=F_{ij}^{\sigma_i,\sigma_j} - F_{ji}^{\sigma_j,\sigma_i}, tahara2008
    // +0 -> upup, +SIZE^2 -> updown, +2*SIZE^2 -> downup, +3*SIZE^2 -> down down
    let mut new_b: Vec<f64> = Vec::with_capacity(n_elec);
    for iup in indx_up.iter() {
        match spin {
            Spin::Up => {
                if *iup == previous_i {
                    new_b.push(0.0);
                    continue;
                }
                new_b.push(
                    fij[new_i + n_sites * iup]
                    -fij[iup + n_sites * new_i]);
            }
            Spin::Down => {
                new_b.push(
                    fij[new_i + n_sites * iup + 2*n_sites*n_sites]
                    -fij[iup + n_sites * new_i + n_sites*n_sites]);
            }
        };
    }
    for idown in indx_down.iter() {
        match spin {
            Spin::Up => {
                new_b.push(
                    fij[new_i + n_sites * idown + n_sites*n_sites]
                    -fij[idown + n_sites * new_i + 2*n_sites*n_sites]);
            }
            Spin::Down => {
                if *idown == previous_i {
                    new_b.push(0.0);
                    continue;
                }
                new_b.push(
                    fij[new_i + n_sites * idown + 3*n_sites*n_sites]
                    -fij[idown + n_sites * new_i + 3*n_sites*n_sites]);
            }
        };
    }

    // Get the column to replace.
    let col = match spin {
        Spin::Up => indx_up.iter().position(|&r| r == previous_i).unwrap(),
        Spin::Down => indx_down.iter().position(|&r| r == previous_i).unwrap() + indx_up.len(),
    };

    // Compute the updated pfaffian.
    let pfaff_up;
    unsafe {
        pfaff_up = ddot(
            n_elec as i32,
            &new_b,
            1,
            &previous_pstate.inv_matrix[n_elec * col..n_elec + n_elec * col],
            1,
        )
    }
    (pfaff_up, new_b, col)
}

/// Updates the pfaffian state given a computed ratio from the vector b.
/// # Fields
/// * __`pstate`__ - The pfaffian state to update. It will update inplace the
/// inverse matrix $A^{-1}$.
/// * __`bm`__ - The vector $b$ needed to update the inverse matrix. This can
/// be acquired from the function [[get_pfaffian_ratio]].
/// * __`col`__ - The column and row in the matrix $A$ that changed. This is NOT
/// correlated to the index of the electron.
pub fn update_pstate(pstate: &mut PfaffianState, bm: Vec<f64>, col: usize) {
    // Rename and copy when necessary.
    let n = pstate.n_elec as i32;
    let mut new_inv = pstate.inv_matrix.clone();

    // Compute A^-1 * b_m
    let mut y = vec![0.0; n as usize];
    unsafe {
        dgemv(
            b"n"[0],
            n,
            n,
            1.0,
            &pstate.inv_matrix,
            n,
            &bm,
            1,
            0.0,
            &mut y,
            1,
        );
    }
    // We already got the pfaffian ratio
    let pfaff_ratio = 1.0 / y[col];

    unsafe {
        // B^-1 = B^-1 - 1./(X[alpha])) * X*Y^T where Y is the m column of A_inv
        dger(
            n,
            n,
            -pfaff_ratio,
            &y,
            1,
            &pstate.inv_matrix[col..col + n as usize * col],
            n,
            &mut new_inv,
            n,
        );

        // B^-1 = B^-1 + 1./(X[alpha])) * Y*X^T
        dger(
            n,
            n,
            pfaff_ratio,
            &pstate.inv_matrix[col..col + n as usize * col],
            n,
            &y,
            1,
            &mut new_inv,
            n,
        );

        // adding column alpha of A_inv to B_inv
        daxpy(
            n,
            -pfaff_ratio,
            &pstate.inv_matrix[col..n as usize * (col + 1)],
            n,
            &mut new_inv[col..n as usize * (col + 1)],
            n,
        );

        // adding row alpha of A_inv to B_inv
        daxpy(
            n,
            -pfaff_ratio,
            &pstate.inv_matrix[n as usize * col..(n + 1) as usize * col],
            1,
            &mut new_inv[n as usize * col..(n + 1) as usize * col],
            1,
        );
    }
    pstate.inv_matrix = new_inv;
}

/// Computes the Pfaffian with a workspace query.
/// # Arguments
/// * __`a`__ - The $A$ matix. This is the variationnal parameters of the
/// pfaffian state.
///
/// # Returns
/// * __`pfaff`__ - The computed pfaffian $\braket{x}{\phi_{\text{PF}}}$.
///
/// # Explanation of the function
/// This function computes the pfaffian in two parts. The first part of the
/// function calls the workspace query of Pfapack. The second part calls the
/// pfaffian computation routine with the optimal workspace size.
pub fn compute_pfaffian_wq(a: &mut [f64], n: i32) -> f64 {
    let mut pfaff: f64 = 0.0;
    let mut info: i32 = 0;
    let mut iwork: Vec<i32> = Vec::with_capacity(n as usize);

    let mut b: Vec<f64> = Vec::with_capacity(((n+1)*n) as usize);

    for x in a.iter() {
        b.push(*x);
    }

    // Workspace query
    let mut work: Vec<f64> = Vec::with_capacity(1);
    work.push(0.0);
    unsafe {
        skpfa::dskpfa(
            b'L',
            b'P',
            &n,
            &mut b,
            &n,
            &mut pfaff,
            &mut iwork,
            &mut work,
            &(-1),
            &mut info,
        )
    }
    assert_eq!(info, 0);
    let lwork: i32 = work[0] as i32;
    // Compute using the lower and p method.
    let mut work: Vec<f64> = Vec::with_capacity(lwork as usize);
    unsafe {
        skpfa::dskpfa(
            b'L', b'P', &n, &mut b, &n, &mut pfaff, &mut iwork, &mut work, &lwork, &mut info,
        )
    }
    assert_eq!(info, 0);
    pfaff
}

#[cfg(test)]
mod tests {
    use crate::pfaffian::{update_pstate, Spin, FockState, BitOps, invert_matrix};
    use rand::{rngs::SmallRng, Rng};
    use rand::SeedableRng;
    use assert::close;

    use super::{construct_matrix_a_from_state, get_pfaffian_ratio, PfaffianState};

    fn convert_spin_to_array(state: FockState<u8>, n: usize) -> (Vec<usize>, Vec<usize>) {
        let mut indices: Vec<usize> = Vec::with_capacity(n);
        let mut indices2: Vec<usize> = Vec::with_capacity(n);
        let (mut spin_up, mut spin_down) = (state.spin_up, state.spin_down);
        let (mut i, mut j): (usize, usize) = (
            spin_up.leading_zeros() as usize,
            spin_down.leading_zeros() as usize,
        );
        while i < state.n_sites {
            indices.push(i);
            spin_up.set(i);
            i = spin_up.leading_zeros() as usize;
        }
        while j < state.n_sites {
            indices2.push(j);
            spin_down.set(j);
            j = spin_down.leading_zeros() as usize;
        }
        (indices, indices2)
    }

    fn frobenius_norm(state: PfaffianState) -> f64 {
        let mut norm: f64 = 0.0;
        for x in state.inv_matrix.iter() {
            norm += x*x;
        }
        norm
    }

    #[test]
    fn test_pfaffian_update_random_no_sign_correction() {
        const SIZE: usize = 8;
        let mut rng = SmallRng::seed_from_u64(42);
        // Size of the system
        let mut params = vec![0.0; 4 * SIZE * SIZE];

        for test_iter in 0..1000 {
            println!("test_iter: {}", test_iter);
            // Generate the variationnal parameters
            // params[i+8*j] = f_ij
            for j in 0..2*SIZE {
                for i in 0..2*SIZE {
                    params[i + 2*j*SIZE] = rng.gen::<f64>();
                }
            }

            // Generate random initial state.
            let state = crate::FockState {
                spin_up: rng.gen::<u8>(),
                spin_down: rng.gen::<u8>(),
                n_sites: SIZE,
            };
            let n = state.spin_up.count_ones() + state.spin_down.count_ones();
            // Matrix needs to be even sized
            if n % 2 == 1 { continue;}
            println!("------------- Initial State ----------------");
            let mut pfstate = construct_matrix_a_from_state(params.clone(), state);
            println!("Inverse Matrix: {}", pfstate);

            // Generate random update
            // Spin up or down?
            let is_spin_up: bool = rng.gen::<u8>() % 2 == 0;
            // What index from?
            // Get where there are electrons
            let (sups, sdowns) = convert_spin_to_array(state, n as usize);
            let initial_index =
                if is_spin_up {
                    if sups.len() == 0 { continue;}
                    sups[rng.gen::<usize>() % sups.len()]
                }
                else {
                    if sdowns.len() == 0 { continue;}
                    sdowns[rng.gen::<usize>() % sdowns.len()]
                };
            // Where to?
            // It must not be occupied
            let mut final_index;
            if is_spin_up {
                if sups.len() == SIZE {
                    continue;
                }
                final_index = rng.gen::<usize>() % SIZE;
                while sups.contains(&final_index) {
                    final_index = rng.gen::<usize>() % SIZE;
                }
            }
            else {
                if sdowns.len() == SIZE {
                    continue;
                }
                final_index = rng.gen::<usize>() % SIZE;
                while sdowns.contains(&final_index) {
                    final_index = rng.gen::<usize>() % SIZE;
                }
            }
            let mut sup = state.spin_up.clone();
            let mut sdown = state.spin_down.clone();
            if is_spin_up {
                sup.set(initial_index);
                sup.set(final_index);
            }
            else {
                sdown.set(initial_index);
                sdown.set(final_index);
            }
            let state2 = crate::FockState {
                spin_up: sup,
                spin_down: sdown,
                n_sites: SIZE,
            };

            println!("------------- Updated State Long way ----------------");
            let mut pfstate2 = construct_matrix_a_from_state(params.clone(), state2);
            println!("Inverse Matrix: {}", pfstate2);
            println!("------------- Proposed Update ------------------");
            println!("Jumps from: {}, Lands on: {}", initial_index, final_index);
            println!("Spin is up: {}", is_spin_up);
            let tmp =
                if is_spin_up {
                    get_pfaffian_ratio(&pfstate, initial_index, final_index, Spin::Up)
                }
                else {
                    get_pfaffian_ratio(&pfstate, initial_index, final_index, Spin::Down)
                };
            println!("Ratio: {}", tmp.0);
            println!("B col: {:?}", tmp.1);
            println!("Column that changed: {}", tmp.2);
            println!("Difference: {}", pfstate.pfaff * tmp.0 - pfstate2.pfaff);
            close(<f64>::abs(pfstate.pfaff * tmp.0), <f64>::abs(pfstate2.pfaff), 1e-12);
            println!("Computed Pfaffian matches updated pfaffian.");

            println!("------------- Updated Inverse matrix ------------");
            update_pstate(&mut pfstate, tmp.1, tmp.2);
            println!("{}", pfstate);
            invert_matrix(&mut pfstate.inv_matrix, pfstate.n_elec as i32);
            invert_matrix(&mut pfstate2.inv_matrix, pfstate2.n_elec as i32);
            println!("Direct Matrix Long Way: {}", pfstate2);
            println!("Direct Matrix updated: {}", pfstate);
            println!("Testing frobenius norm:");
            close(frobenius_norm(pfstate), frobenius_norm(pfstate2), 1e-8);
            println!("Frobenius norm are equal!");
        }
    }

    #[test]
    fn test_pfaffian_8sites_u8() {
        const SIZE: usize = 8;
        let mut params = vec![0.0; 4 * SIZE * SIZE];
        // params[i+8*j] = f_ij
        params[7 + SIZE * 7] = 1.0;
        params[7 + SIZE * 6] = 0.8;
        params[6 + SIZE * 7] = 1.0;
        params[6 + SIZE * 6] = 0.5;
        params[7 + SIZE * 7 + 3*SIZE*SIZE] = 1.0;
        params[7 + SIZE * 6 + 3*SIZE*SIZE] = 0.8;
        params[6 + SIZE * 7 + 3*SIZE*SIZE] = 1.0;
        params[6 + SIZE * 6 + 3*SIZE*SIZE] = 0.5;
        let state = crate::FockState {
            spin_up: 3u8,
            spin_down: 3u8,
            n_sites: SIZE,
        };
        let pfstate = construct_matrix_a_from_state(params, state);
        println!("Inverse Matrix: {}", pfstate);
        close(pfstate.pfaff, 0.04, 1e-12);
    }

    #[test]
    fn test_pfaffian_8sites_u8_update_spin_up() {
        const SIZE: usize = 8;
        let mut params = vec![0.0; 4 * SIZE * SIZE];
        // params[i+8*j] = f_ij
        params[7 + SIZE * 7] = 1.0;
        params[7 + SIZE * 6] = 0.8;
        params[7 + SIZE * 5] = 0.9;
        params[6 + SIZE * 7] = 1.0;
        params[6 + SIZE * 6] = 0.5;
        params[6 + SIZE * 5] = 1.0;
        params[5 + SIZE * 7] = 1.0;
        params[5 + SIZE * 6] = 1.0;
        params[5 + SIZE * 5] = 0.5;
        params[7 + SIZE * 7 + 3*SIZE*SIZE] = 1.0;
        params[7 + SIZE * 6 + 3*SIZE*SIZE] = 0.8;
        params[7 + SIZE * 5 + 3*SIZE*SIZE] = 0.9;
        params[6 + SIZE * 7 + 3*SIZE*SIZE] = 1.0;
        params[6 + SIZE * 6 + 3*SIZE*SIZE] = 0.5;
        params[6 + SIZE * 5 + 3*SIZE*SIZE] = 1.0;
        params[5 + SIZE * 7 + 3*SIZE*SIZE] = 1.0;
        params[5 + SIZE * 6 + 3*SIZE*SIZE] = 1.0;
        params[5 + SIZE * 5 + 3*SIZE*SIZE] = 0.5;
        let state = crate::FockState {
            spin_up: 3u8,
            spin_down: 3u8,
            n_sites: SIZE,
        };
        let pfstate = construct_matrix_a_from_state(params.clone(), state);
        println!("Inverse Matrix: {}", pfstate);
        close(pfstate.pfaff, 0.04, 1e-12);
        let state2 = crate::FockState {
            spin_up: 5u8,
            spin_down: 3u8,
            n_sites: SIZE,
        };
        let pfstate2 = construct_matrix_a_from_state(params, state2);
        println!("Inverse Matrix: {}", pfstate2);
        let pfaff_ratio = get_pfaffian_ratio(&pfstate, 6, 5, Spin::Up).0;
        close(pfstate.pfaff * pfaff_ratio, pfstate2.pfaff, 1e-12);
    }

    #[test]
    fn test_pfaffian_8sites_u8_update_spin_down() {
        const SIZE: usize = 8;
        let mut params = vec![0.0; 4 * SIZE * SIZE];
        // params[i+8*j] = f_ij
        params[7 + SIZE * 7] = 1.0;
        params[7 + SIZE * 6] = 0.8;
        params[7 + SIZE * 5] = 0.9;
        params[6 + SIZE * 7] = 1.0;
        params[6 + SIZE * 6] = 0.5;
        params[6 + SIZE * 5] = 1.0;
        params[5 + SIZE * 7] = 1.0;
        params[5 + SIZE * 6] = 1.0;
        params[5 + SIZE * 5] = 0.5;
        params[7 + SIZE * 7 + SIZE*SIZE] = 1.0;
        params[7 + SIZE * 5 + SIZE*SIZE] = 0.9;
        params[6 + SIZE * 6 + SIZE*SIZE] = 0.5;
        params[6 + SIZE * 5 + SIZE*SIZE] = 1.0;
        params[5 + SIZE * 7 + SIZE*SIZE] = 1.0;
        params[5 + SIZE * 6 + SIZE*SIZE] = 1.0;
        params[5 + SIZE * 5 + SIZE*SIZE] = 0.5;
        params[7 + SIZE * 7 + 2*SIZE*SIZE] = 1.0;
        params[7 + SIZE * 6 + 2*SIZE*SIZE] = 0.8;
        params[7 + SIZE * 5 + 2*SIZE*SIZE] = 0.9;
        params[6 + SIZE * 7 + 2*SIZE*SIZE] = 1.0;
        params[6 + SIZE * 6 + 2*SIZE*SIZE] = 0.5;
        params[6 + SIZE * 5 + 2*SIZE*SIZE] = 1.0;
        params[5 + SIZE * 7 + 2*SIZE*SIZE] = 1.0;
        params[5 + SIZE * 6 + 2*SIZE*SIZE] = 1.0;
        params[5 + SIZE * 5 + 2*SIZE*SIZE] = 0.5;
        params[7 + SIZE * 7 + 3*SIZE*SIZE] = 1.0;
        params[7 + SIZE * 6 + 3*SIZE*SIZE] = 0.8;
        params[7 + SIZE * 5 + 3*SIZE*SIZE] = 0.9;
        params[6 + SIZE * 7 + 3*SIZE*SIZE] = 1.0;
        params[6 + SIZE * 6 + 3*SIZE*SIZE] = 0.5;
        params[6 + SIZE * 5 + 3*SIZE*SIZE] = 1.0;
        params[5 + SIZE * 7 + 3*SIZE*SIZE] = 1.0;
        params[5 + SIZE * 6 + 3*SIZE*SIZE] = 1.0;
        params[5 + SIZE * 5 + 3*SIZE*SIZE] = 0.5;
        let state = crate::FockState {
            spin_up: 3u8,
            spin_down: 3u8,
            n_sites: SIZE,
        };
        let pfstate = construct_matrix_a_from_state(params.clone(), state);
        println!("Inverse Matrix: {}", pfstate);
        close(pfstate.pfaff, 0.84, 1e-12);
        let state2 = crate::FockState {
            spin_up: 3u8,
            spin_down: 5u8,
            n_sites: SIZE,
        };
        let pfstate2 = construct_matrix_a_from_state(params, state2);
        println!("Inverse Matrix: {}", pfstate2);
        let tmp = get_pfaffian_ratio(&pfstate, 6, 5, Spin::Down);
        println!("B: {:?}", tmp.1);
        println!("Ratio: {}", tmp.0);
        close(pfstate.pfaff * tmp.0, pfstate2.pfaff, 1e-12);
    }

    #[test]
    fn test_pfaffian_8sites_u8_update_matrix() {
        const SIZE: usize = 8;
        let mut params = vec![0.0; 4 * SIZE * SIZE];
        // params[i+8*j] = f_ij
        params[7 + SIZE * 7] = 1.0;
        params[7 + SIZE * 6] = 0.8;
        params[7 + SIZE * 5] = 0.9;
        params[6 + SIZE * 7] = 1.0;
        params[6 + SIZE * 6] = 0.5;
        params[6 + SIZE * 5] = 1.0;
        params[5 + SIZE * 7] = 1.0;
        params[5 + SIZE * 6] = 1.0;
        params[5 + SIZE * 5] = 0.5;
        params[7 + SIZE * 7 + SIZE*SIZE] = 1.0;
        params[7 + SIZE * 5 + SIZE*SIZE] = 0.9;
        params[6 + SIZE * 6 + SIZE*SIZE] = 0.5;
        params[6 + SIZE * 5 + SIZE*SIZE] = 1.0;
        params[5 + SIZE * 7 + SIZE*SIZE] = 1.0;
        params[5 + SIZE * 6 + SIZE*SIZE] = 1.0;
        params[5 + SIZE * 5 + SIZE*SIZE] = 0.5;
        params[7 + SIZE * 7 + 2*SIZE*SIZE] = 1.0;
        params[7 + SIZE * 6 + 2*SIZE*SIZE] = 0.8;
        params[7 + SIZE * 5 + 2*SIZE*SIZE] = 0.9;
        params[6 + SIZE * 7 + 2*SIZE*SIZE] = 1.0;
        params[6 + SIZE * 6 + 2*SIZE*SIZE] = 0.5;
        params[6 + SIZE * 5 + 2*SIZE*SIZE] = 1.0;
        params[5 + SIZE * 7 + 2*SIZE*SIZE] = 1.0;
        params[5 + SIZE * 6 + 2*SIZE*SIZE] = 1.0;
        params[5 + SIZE * 5 + 2*SIZE*SIZE] = 0.5;
        params[7 + SIZE * 7 + 3*SIZE*SIZE] = 1.0;
        params[7 + SIZE * 6 + 3*SIZE*SIZE] = 0.8;
        params[7 + SIZE * 5 + 3*SIZE*SIZE] = 0.9;
        params[6 + SIZE * 7 + 3*SIZE*SIZE] = 1.0;
        params[6 + SIZE * 6 + 3*SIZE*SIZE] = 0.5;
        params[6 + SIZE * 5 + 3*SIZE*SIZE] = 1.0;
        params[5 + SIZE * 7 + 3*SIZE*SIZE] = 1.0;
        params[5 + SIZE * 6 + 3*SIZE*SIZE] = 1.0;
        params[5 + SIZE * 5 + 3*SIZE*SIZE] = 0.5;
        let state = crate::FockState {
            spin_up: 3u8,
            spin_down: 3u8,
            n_sites: SIZE,
        };
        println!("------------- Initial State ----------------");
        let mut pfstate = construct_matrix_a_from_state(params.clone(), state);
        println!("Inverse Matrix: {}", pfstate);
        close(pfstate.pfaff, 0.84, 1e-12);
        let state2 = crate::FockState {
            spin_up: 3u8,
            spin_down: 5u8,
            n_sites: SIZE,
        };
        println!("------------- Updated State Long way ----------------");
        let pfstate2 = construct_matrix_a_from_state(params, state2);
        println!("Inverse Matrix: {}", pfstate2);
        println!("------------- Proposed Update ------------------");
        let tmp = get_pfaffian_ratio(&pfstate, 6, 5, Spin::Down);
        println!("Ratio: {}", tmp.0);
        println!("B col: {:?}", tmp.1);
        close(pfstate.pfaff * tmp.0, pfstate2.pfaff, 1e-12);
        println!("Computed Pfaffian matches updated pfaffian.");
        update_pstate(&mut pfstate, tmp.1, tmp.2);
        println!("------------- Updated Inverse matrix ------------");
        println!("{}", pfstate);
        for (good, test) in pfstate2.inv_matrix.iter().zip(pfstate.inv_matrix) {
            close(*good, test, 1e-12);
        }
    }
}
