use crate::{BitOps, FockState, SIZE};
use pfapack::skpfa;

/// Represents the Pfaffian state $\lvert\phi_{\text{PF}}\rangle$.
/// # Fields
/// * __`coeffs`__ - The variationnal parameters $f_{ij}$.
/// * __`n_elec`__ - The number of electrons. This value is constant and
/// determines the size of the matrix $A$.
/// * __`matrix`__ - The matrix $A$. This is the matrix that we need the pfaffian
/// to get the inner product $\braket{x}{\phi_{\text{PF}}}$
/// * __`curr_state`__ - The current state of that the matrix $A$ is written in.
pub struct PfaffianState<T> {
    pub coeffs: Vec<f64>,
    pub n_elec: usize,
    pub matrix: Vec<f64>,
    pub curr_state: FockState<T>,
}

impl<T> PfaffianState<T> {
    pub fn rebuild_matrix(&mut self) {
        for i in 0..self.n_elec {
            for j in 0..self.n_elec {
                self.matrix[j + i * self.n_elec] = -self.matrix[i + j * self.n_elec];
            }
        }
    }
}

pub fn construct_matrix_a_from_state<T>(fij: Vec<f64>, state: FockState<T>) -> PfaffianState<T>
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
    for ii in 0..indices.len() {
        for jj in 0..indices2.len() {
            a[ii + (jj + off) * n] = fij[indices[ii] + state.n_sites * indices2[jj]];
            a[jj + off + ii * n] = -fij[indices[ii] + state.n_sites * indices2[jj]];
        }
    }

    PfaffianState {
        coeffs: fij,
        n_elec: n,
        matrix: a,
        curr_state: state,
    }
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
    use crate::{pfaffian::compute_pfaffian_wq, SIZE};
    use assert::close;

    use super::construct_matrix_a_from_state;

    #[test]
    fn test_pfaffian() {
        let mut params = vec![0.0; 4 * SIZE * SIZE];
        // params[i+8*j] = f_ij
        params[7 + 8 * 7] = 1.0;
        params[7 + 8 * 6] = 0.8;
        params[6 + 8 * 7] = 1.0;
        params[6 + 8 * 6] = 0.5;
        let state = crate::FockState {
            spin_up: 3u8,
            spin_down: 3u8,
            n_sites: 8,
        };
        let mut pfstate = construct_matrix_a_from_state(params, state);
        println!("Matrix: {:?}", pfstate.matrix);
        close(
            compute_pfaffian_wq(&mut pfstate.matrix, pfstate.n_elec as i32),
            0.3,
            1e-12,
        );
    }
}
