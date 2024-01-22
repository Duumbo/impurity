use crate::{BitOps, FockState, SIZE};
use blas::{daxpy, ddot, dgemv, dger};
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
    // Refer to LAPACK error message.
    if !(info1 == 0) || !(info2 == 0) {
        println!(
            "The algorithm failed to invert the matrix. DGETRF: info={}, DGETRI: info={}",
            info1, info2
        );
        panic!("Matrix invertion fail.");
    }
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
        println!("i: {}", i);
        i = spin_up.leading_zeros() as usize;
    }
    while j < state.n_sites {
        indices2.push(j);
        spin_down.set(j);
        println!("j: {}", j);
        j = spin_down.leading_zeros() as usize;
    }
    println!("indices: {:?}, indices2: {:?}", indices, indices2);
    let off = indices.len();
    for jj in 0..indices2.len() {
        for ii in 0..indices.len() {
            a[ii + (jj + off) * n] = fij[indices[ii] + state.n_sites * indices2[jj]];
            a[jj + off + ii * n] = -fij[indices[ii] + state.n_sites * indices2[jj]];
        }
    }
    for jj in 0..indices.len() {
        for ii in 0..indices.len() {
            if indices[ii] == indices[jj] {
                continue;
            }
            a[ii + jj * n] = fij[indices[ii] + state.n_sites * indices[jj]];
            a[jj + ii * n] = -fij[indices[ii] + state.n_sites * indices[jj]];
        }
    }
    for jj in 0..indices2.len() {
        for ii in 0..indices2.len() {
            if indices2[ii] == indices2[jj] {
                continue;
            }
            a[ii + off + (jj + off) * n] = fij[indices2[ii] + state.n_sites * indices2[jj]];
            a[jj + off + (ii + off) * n] = -fij[indices2[ii] + state.n_sites * indices2[jj]];
        }
    }

    // Invert matrix.
    let pfaffian_value = compute_pfaffian_wq(&mut a.clone(), n as i32);
    println!(
        "Direct Matrix: {}",
        PfaffianState {
            coeffs: fij.clone(),
            n_elec: n,
            n_sites: state.n_sites,
            inv_matrix: a.clone(),
            indices: (indices.clone(), indices2.clone()),
            pfaff: pfaffian_value,
        }
    );
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
    let mut new_b: Vec<f64> = Vec::with_capacity(n_elec);
    let off = indx_up.len();
    for iup in indx_up.iter() {
        match spin {
            Spin::Up => {
                if *iup == previous_i {
                    new_b.push(0.0);
                    continue;
                }
                new_b.push(fij[new_i + n_sites * iup]);
            }
            Spin::Down => {
                new_b.push(-fij[iup + n_sites * new_i]);
            }
        };
    }
    for idown in indx_down.iter() {
        match spin {
            Spin::Up => {
                new_b.push(fij[new_i + n_sites * idown]);
            }
            Spin::Down => {
                if *idown == previous_i {
                    new_b.push(0.0);
                    continue;
                }
                new_b.push(fij[new_i + n_sites * idown]);
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
    let mut iwork: Vec<i32> = Vec::with_capacity(SIZE);

    // Workspace query
    let mut work: Vec<f64> = Vec::with_capacity(1);
    work.push(0.0);
    unsafe {
        skpfa::dskpfa(
            b'L',
            b'P',
            &n,
            a,
            &n,
            &mut pfaff,
            &mut iwork,
            &mut work,
            &(-1),
            &mut info,
        )
    }
    let lwork: i32 = work[0] as i32;
    // Compute using the lower and p method.
    let mut work: Vec<f64> = Vec::with_capacity(lwork as usize);
    unsafe {
        skpfa::dskpfa(
            b'L', b'P', &n, a, &n, &mut pfaff, &mut iwork, &mut work, &lwork, &mut info,
        )
    }
    assert_eq!(info, 0);
    pfaff
}

#[cfg(test)]
mod tests {
    use crate::pfaffian::{update_pstate, Spin};
    use assert::close;

    use super::{construct_matrix_a_from_state, get_pfaffian_ratio};

    #[test]
    fn test_pfaffian_8sites_u8() {
        const SIZE: usize = 8;
        let mut params = vec![0.0; 4 * SIZE * SIZE];
        // params[i+8*j] = f_ij
        params[7 + SIZE * 7] = 1.0;
        params[7 + SIZE * 6] = 0.8;
        params[6 + SIZE * 7] = 1.0;
        params[6 + SIZE * 6] = 0.5;
        let state = crate::FockState {
            spin_up: 3u8,
            spin_down: 3u8,
            n_sites: SIZE,
        };
        let pfstate = construct_matrix_a_from_state(params, state);
        println!("Inverse Matrix: {}", pfstate);
        close(pfstate.pfaff, 1.3, 1e-12);
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
        let state = crate::FockState {
            spin_up: 3u8,
            spin_down: 3u8,
            n_sites: SIZE,
        };
        let pfstate = construct_matrix_a_from_state(params.clone(), state);
        println!("Inverse Matrix: {}", pfstate);
        close(pfstate.pfaff, 1.3, 1e-12);
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
        let state = crate::FockState {
            spin_up: 3u8,
            spin_down: 3u8,
            n_sites: SIZE,
        };
        let pfstate = construct_matrix_a_from_state(params.clone(), state);
        println!("Inverse Matrix: {}", pfstate);
        close(pfstate.pfaff, 1.3, 1e-12);
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
        let state = crate::FockState {
            spin_up: 3u8,
            spin_down: 3u8,
            n_sites: SIZE,
        };
        println!("------------- Initial State ----------------");
        let mut pfstate = construct_matrix_a_from_state(params.clone(), state);
        println!("Inverse Matrix: {}", pfstate);
        close(pfstate.pfaff, 1.3, 1e-12);
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
