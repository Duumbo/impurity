use crate::{BitOps, FockState, SIZE};
use pfapack::skpfa;
use lapack::{dgetrf, dgetri};
use blas::ddot;

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
    pub indices: (Vec<usize>, Vec<usize>),
    pub inv_matrix: Vec<f64>,
}

/// The pfaffian state implementation.
/// #TODOC
impl PfaffianState {
    pub fn rebuild_matrix(&mut self) {
        for i in 0..self.n_elec {
            for j in 0..self.n_elec {
                self.inv_matrix[j + i * self.n_elec] = -self.inv_matrix[i + j * self.n_elec];
            }
        }
    }
}

/// Inverts a matrix.
/// #TODOC
fn invert_matrix(a: &mut [f64], n: i32) {
    // Info output of lapack
    let mut info1: i32 = 0;
    let mut info2: i32 = 0;

    // Length of work vector
    let n_entry: i32 = n*n;
    // Workspaces
    let mut work: Vec<f64> = Vec::with_capacity(n_entry as usize);
    let mut ipiv: Vec<i32> = Vec::with_capacity(n as usize);

    // Inverse matrix `a` inplace using L*U decomposition.
    unsafe{
        dgetrf(n, n, a, n, &mut ipiv, &mut info1);
        dgetri(n, a, n, &ipiv, &mut work, n_entry, &mut info2);
    }

    // These should never be not 0.
    // If this panics, then there a was not of size n, most probably.
    // Refer to LAPACK error message.
    if !(info1 == 0) || !(info2 == 0)  {
        println!(
            "The algorithm failed to invert the matrix. DGETRF: info={}, DGETRI: info={}",
            info1,
            info2
        );
        panic!("Matrix invertion fail.");
    }
}

/// Constructs pfaffian matrix from state.
/// #TODOC
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

/// Get the ration of the pfaffian given the update.
/// #TODOC
pub fn get_pfaffian_ratio(previous_pstate: &PfaffianState, previous_i: usize, new_i: usize) -> f64
{
    // Rename
    let indx_up =  &previous_pstate.indices.0;
    let indx_down =  &previous_pstate.indices.1;
    let fij = &previous_pstate.coeffs;
    let n_sites = previous_pstate.n_sites;
    let n_elec = previous_pstate.n_elec;

    // Gen new vector b
    let mut new_b: Vec<f64> = Vec::with_capacity(n_elec);
    for iup in indx_up.iter() {
        new_b.push(fij[new_i + n_sites * iup]);
    }
    for idown in indx_down.iter() {
        new_b.push(fij[new_i + n_sites * idown]);
    }
    println!("New b: {:?}", new_b);

    let col = indx_up.iter().position(|&r| r == previous_i).unwrap();
    // Compute the updated pfaffian.
    let pfaff_up;
    unsafe {
        pfaff_up =
            ddot(n_elec as i32, &new_b, 1, &previous_pstate.inv_matrix[n_elec*col..n_elec + n_elec*col], 1)
    }
    pfaff_up
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
        println!("Inverse Matrix: {:?}", pfstate.inv_matrix);
        close(
            pfstate.pfaff,
            0.3,
            1e-12,
        );
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
        println!("Inverse Matrix: {:?}", pfstate.inv_matrix);
        close(
            pfstate.pfaff,
            0.3,
            1e-12,
        );
        let state2 = crate::FockState {
            spin_up: 5u8,
            spin_down: 3u8,
            n_sites: SIZE,
        };
        let pfstate2 = construct_matrix_a_from_state(params, state2);
        println!("Inverse Matrix: {:?}", pfstate2.inv_matrix);
        let pfaff_ratio = get_pfaffian_ratio(&pfstate, 6, 5);
        close(pfstate.pfaff * pfaff_ratio, pfstate2.pfaff, 1e-12);
    }
}
