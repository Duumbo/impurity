use crate::{SIZE, FockState};
use pfapack::skpfa;

/// Represents the Pfaffian state $\lvert\phi_{\text{PF}}\rangle$.
/// # Fields
/// * __`coeffs`__ - The variationnal parameters $f_{ij}$.
/// * __`n_elec`__ - The number of electrons. This value is constant and
/// determines the size of the matrix $A$.
/// * __`matrix`__ - The matrix $A$. This is the matrix that we need the pfaffian
/// to get the inner product $\braket{x}{\phi_{\text{PF}}}$
/// * __`curr_state`__ - The current state of that the matrix $A$ is written in.
pub struct PfaffianState {
    pub coeffs: Vec<f64>,
    pub n_elec: usize,
    pub matrix: Vec<f64>,
    pub curr_state: FockState,
}

impl PfaffianState {
    pub fn rebuild_matrix(&mut self) {
        for i in 0..self.n_elec {
            for j in 0..self.n_elec {
                self.matrix[j + i * self.n_elec] = - self.matrix[i + j * self.n_elec];
            }
        }
    }
}

pub fn construct_matrix_a_from_state(fij: Vec<f64>, state: FockState) -> PfaffianState {
    let n = state.spin_up.count_ones() as usize + state.spin_down.count_ones() as usize;
    let mut a: Vec<f64> = vec![0.0; n*n];

    // Get index i (position of the first set bit.)
    let mut i = state.spin_up.leading_zeros() as usize;
    let mut spin_up = state.spin_up;
    let (mut ii, mut jj) = (0, 1);
    while i < SIZE {
        println!("i: {}", i);
        let mut spin_down = state.spin_down;
        let mut j = spin_down.leading_zeros() as usize;
        while j < SIZE {
            println!("j: {}", j);
            a[jj + ii * n] = fij[i + j * SIZE];
            a[ii + jj * n] = - fij[i + j * SIZE];
            spin_down ^= 1 << (SIZE - 1 - j);
            jj += 1;
            j = spin_down.leading_zeros() as usize;
        }
        spin_up ^= 1 << (SIZE - 1 - i);
        ii += 1;
        jj = 0;
        i = spin_up.leading_zeros() as usize;
    }


    PfaffianState { coeffs: fij, n_elec: n, matrix: a, curr_state: state }
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
    use super::construct_matrix_a_from_state;

    #[test]
    fn test_pfaffian() {
        todo!();
    }
}
