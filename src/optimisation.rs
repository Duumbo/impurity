use core::panic;

use blas::{daxpy, dcopy, ddot, dgemm, dgemv, dger, dscal};
use lapack::dsyev;
use log::{error, trace};
use colored::Colorize;

use crate::{DerivativeOperator};

/// Encoding the index map to a reduced representation
pub struct ParameterMap {
    /// Number of independant parameters.
    pub nparams: usize,
    /// Dimension of the parameter map; Number of total parameters
    pub dim: usize,
    nfij: usize,
    pub nvij: usize,
    pub ngi: usize,
    size: usize,
    /// Index map. Indexes from general parameter to reduced parameter.
    pub map: Box<[usize]>,
}

impl ParameterMap {
    pub fn new(nparams: usize, size: usize) -> ParameterMap {
        let nfij = 4*size*size;
        let nvij = (size*size - size)/2;
        let ngi = size;
        let dim = nfij + nvij + ngi;
        let map = vec![0; dim].into_boxed_slice();
        ParameterMap { nparams, dim, map, nfij, nvij, ngi, size }
    }
    #[inline(always)]
    fn get_fij(self: &Self, i: usize, j: usize) -> usize {
        if i * self.size + j >= self.nfij {
            panic!("Index out of bounds. {}", i);
        }
        self.map[self.ngi + self.nvij + i * self.size + j] + self.ngi + self.nvij + 2
    }
    pub fn index_fij(self: &Self, otilde: &[f64], i: usize, j: usize, mu: usize) -> f64 {
        let new_index = self.get_fij(i, j);
        otilde[new_index + mu * self.nparams]
    }
    pub fn index_fij_mut<'a, 'b>(self: &'a Self, otilde: &'b mut [f64], i: usize, j: usize, mu: usize) -> &'b mut f64 {
        let new_index = self.get_fij(i, j);
        if new_index >= self.nparams + 3 {
            panic!("Index out of bounds f_{} {} at {}", i, j, new_index);
        }
        &mut otilde[new_index + mu * (self.nparams+3)]
    }
    #[inline(always)]
    fn get_vij(self: &Self, i: usize, k: usize) -> usize {
        if i + k*(k-1)/2 >= self.nvij {
            panic!("Index out of bounds. {}", i);
        }
        self.map[self.ngi + i + k*(k-1)/2] + self.ngi + 1
    }
    pub fn index_vij(self: &Self, otilde: &[f64], i: usize, k: usize, mu: usize) -> f64 {
        let new_index = self.get_vij(i, k);
        otilde[new_index + mu * self.nparams]
    }
    pub fn index_vij_mut<'a, 'b>(self: &'a Self, otilde: &'b mut [f64], i: usize, k: usize, mu: usize) -> &'b mut f64 {
        let new_index = self.get_vij(i, k);
        if new_index >= self.nparams + 2 {
            panic!("Index out of bounds");
        }
        &mut otilde[new_index + mu * (self.nparams+3)]
    }
    #[inline(always)]
    fn get_gi(self: &Self, i: usize) -> usize {
        if i >= self.ngi {
            panic!("Index out of bounds. {}", i);
        }
        self.map[i]
    }
    pub fn index_gi(self: &Self, otilde: &[f64], i: usize, mu: usize) -> f64 {
        let new_index = self.get_gi(i);
        otilde[new_index + mu * self.nparams]
    }
    pub fn index_gi_mut<'a, 'b>(self: &'a Self, otilde: &'b mut [f64], i: usize, mu: usize) -> &'b mut f64 {
        let new_index = self.get_gi(i);
        if new_index >= self.nparams + 2 {
            panic!("Index out of bounds");
        }
        &mut otilde[new_index + mu * (self.nparams+3)]
    }

    pub fn reverse_map(self: &Self, x: &mut [f64]) -> Box<[f64]> {
        let mut out = vec![0.0; self.dim].into_boxed_slice();
        for i in 0..self.ngi {
            let id = self.map[i];
            if id == 0 {continue;}
            out[i] = x[id];
        }
        for i in 0..self.nvij {
            let id = self.map[self.ngi + i];
            let od = i + self.ngi;
            if id == 0 {continue;}
            let id = id + self.ngi + 1;
            out[od] = x[id];
        }
        for i in 0..self.nfij {
            let id = self.map[self.ngi + self.nvij + i];
            let od = i + self.ngi + self.nvij;
            if id == 0 {continue;}
            let id = id + self.ngi + self.nvij + 2;
            out[od] = x[id];
        }
        out
    }
}

/// Encoding of a map to a reduced representation
/// # Map encoding
/// This struct encodes a Map to and from the general representation with
/// all the parameters to and from a reduced representation where some parameters
/// are multiples of other.
/// This struct is usefull to reduce the dimension of the optimisation and also
/// reduces the amount of linearly dependant columns, which helps with conditionning.
//pub struct GenParameterMap {
//    /// Number of independant parameters.
//    pub dim: i32,
//    /// Dimension of general representation
//    pub gendim: i32,
//    /// Number of general parameters
//    pub n_genparams: i32,
//    /// Number of independant gutzwiller parameters.
//    pub n_independant_gutzwiller: usize,
//    /// Number of independant jastrow parameters.
//    pub n_independant_jastrow: usize,
//    /// The projector from the general representation to the reduced one.
//    /// Should behave:
//    /// Ax => \tilde{x} where every exquivalent component of x are sumed in
//    /// the lowest index and zero elsewhere.
//    /// Example:
//    /// A = ((1, 1), (0, 0))
//    /// x = (a, b)
//    /// \tilde{x} = (a + b, 0)
//    /// A will be triangular, so only the upper triangle needs to be stored.
//    /// If you transpose this matrix, it will reverse this operation somewhat
//    /// A^T = ((1, 0), (1, 0)
//    /// A\tilde{x} = (a + b, a + b)
//    /// This indeed copies the parameter x_0 to x_1. If x_1 is set to half x_0,
//    /// then
//    /// A = ((1, 2), (0, 0))
//    /// x = (a, b), Ax = (a + 2b)
//    /// Then the coefficient wise-inverse of the transpose is usefull
//    /// A^T^{-1} = ((1, 0), (1/2, 0))
//    /// x = (a, 0)
//    /// \tilde{x} = (a, 1/2 a)
//    /// It is then usefull to store only the lower triangle of the "inverse" of
//    /// the transpose, as coefficient don't matter when projecting, but do
//    /// matter when recovering.
//    ///
//    /// Indexing
//    /// n = dim (dim - 1) / 2 + i - (dim - j) * ( dim - j - 1) / 2
//    /// Can only have one non zero value per row
//    ///
//    /// PERFORMANCE:
//    /// If performance is an issue, an idea would be to write this matrix inside
//    /// an integer. This is not done here as it removes to possibility to make
//    /// a parameter say half another one.
//    pub projector: Box<[f64]>,
//}
//
//pub trait ReducibleGeneralRepresentation {
//    /// Overrides b and x0
//    /// Safety:
//    /// b and x0 must have lenght self.gendim
//    fn mapto_general_representation(&self, der: &DerivativeOperator, x0: &mut [f64],) -> DerivativeOperator;
//    /// Overrides b and x0
//    /// Safety:
//    /// b and x0 must have lenght self.gendim
//    /// On exit, the range [self.dim..self.gendim] contains garbage.
//    fn mapto_reduced_representation(&self, der: &DerivativeOperator, x0: &mut [f64],) -> DerivativeOperator;
//    /// Overrides b and x0
//    /// Safety:
//    /// b and x0 must have lenght self.gendim
//    fn update_general_representation(&self, der: &DerivativeOperator, out_der: &mut DerivativeOperator, x0: &mut [f64],);
//    /// Overrides b and x0
//    /// Safety:
//    /// b and x0 must have lenght self.gendim
//    /// On exit, the range [self.dim..self.gendim] contains garbage.
//    fn update_reduced_representation_no_otilde(&self, out_der: &mut DerivativeOperator, mu: i32, expval_o: &[f64], ho: &[f64], visited: &[usize], x0: &mut [f64]);
//    fn update_delta_alpha_reduced_to_gen(&self, x0: &mut [f64]);
//    fn parallel_reduced_representation_otilde(&self, der: &DerivativeOperator, otilde: &mut [f64]);
//}
//
//impl<'a> ReducibleGeneralRepresentation for GenParameterMap {
//    fn mapto_general_representation(&self, der: &DerivativeOperator, x0: &mut [f64]) -> DerivativeOperator
//    {
//        if self.dim == 0 {
//            error!("Tried to map from an empty parameter representation, this will
//                break conjugate gradient or diagonalisation. Dim = {}", self.dim);
//            panic!("Undefined behavior.");
//        }
//
//        let mut out_x = vec![0.0; self.gendim as usize];
//        let n = self.gendim as usize;
//        let mut out_der = DerivativeOperator::new(self.gendim, der.mu, der.nsamp, der.nsamp_int,
//            self.n_independant_gutzwiller + self.n_independant_jastrow,
//            self.n_independant_gutzwiller, der.epsilon);
//        if der.mu < 0 {
//            return out_der;
//        }
//
//        // Copy the visited vector
//        for i in 0..der.mu as usize {
//            out_der.visited[i] = der.visited[i];
//        }
//
//        // Copy all other data
//        let mut k = 0;
//        let mut flag = false;
//        for j in 0..n {
//            for i in n..n {
//                let pij = self.projector[
//                    j + (i * (i+1) / 2)
//                ];
//                if pij != 0.0 {
//                    out_der.expval_o[i] = pij * der.expval_o[k];
//                    out_der.ho[i] = pij * der.ho[k];
//                    out_x[i] = pij * x0[k];
//                    unsafe {
//                        let incx = self.dim;
//                        let incy = self.gendim;
//                        dcopy(
//                            der.mu,
//                            &der.o_tilde[i..(self.dim*der.mu) as usize],
//                            incx,
//                            &mut out_der.o_tilde[k..(self.gendim * der.mu) as usize],
//                            incy
//                        );
//                    }
//                    flag = true;
//                    //break;
//                }
//            }
//            if flag{
//                k += 1;
//                flag = false;
//            }
//        }
//        unsafe {
//            let incx = 1;
//            let incy = 1;
//            dcopy(self.gendim, &out_x, incx, x0, incy);
//        }
//
//        // Should return b and x0 as well
//        out_der
//    }
//
//    fn mapto_reduced_representation(&self, der: &DerivativeOperator, x0: &mut [f64]) -> DerivativeOperator
//    {
//        if self.dim == 0 {
//            error!("Tried to map from an empty parameter representation, this will
//                break conjugate gradient or diagonalisation. Dim = {}", self.dim);
//            panic!("Undefined behavior.");
//        }
//        let n = self.gendim as usize;
//        let mut out_der = DerivativeOperator::new(self.dim, der.mu, der.nsamp, der.nsamp_int,
//            self.n_independant_gutzwiller + self.n_independant_jastrow,
//            self.n_independant_gutzwiller, der.epsilon);
//        if der.mu < 0 {
//            return out_der;
//        }
//
//        // Copy the visited vector
//        for i in 0..der.mu as usize {
//            out_der.visited[i] = der.visited[i];
//        }
//
//        // Copy all other data
//        // iter over transpose's diagonal
//        let mut i: usize = 0;
//        for j in 0..n {
//            let pjj = self.projector[
//                    j + (j * (j+1) / 2)
//            ];
//            if pjj != 0.0 { // If 0, parameter does not map to itself.
//                if pjj != 1.0 {
//                    error!("Parameters must map to themselfs with multiple 1.");
//                    panic!("Undefined behavior.");
//                }
//                out_der.expval_o[i] = der.expval_o[j];
//                out_der.ho[i] = der.ho[j];
//                // Modify the vectors inplace, we won't iterate on past values
//                x0[i] = x0[j];
//                unsafe {
//                    let incx = self.gendim;
//                    let incy = self.dim;
//                    dcopy(
//                        der.mu,
//                        &der.o_tilde[j..n*der.mu as usize],
//                        incx,
//                        &mut out_der.o_tilde[i..(self.dim * der.mu) as usize],
//                        incy
//                    );
//                }
//                // Filling the independant parameter index
//                i += 1;
//            }
//        }
//        if i != self.dim as usize {
//            error!("Parameter map did not yield a dimension of {}, instead got {}", self.dim, i);
//            panic!("Undefined behavior.");
//        }
//
//        out_der
//    }
//
//    fn update_general_representation(&self, der: &DerivativeOperator, out_der: &mut DerivativeOperator, x0: &mut [f64]) {
//        if self.dim == 0 {
//            error!("Tried to map from an empty parameter representation, this will
//                break conjugate gradient or diagonalisation. Dim = {}", self.dim);
//            panic!("Undefined behavior.");
//        }
//
//        let mut out_x = vec![0.0; self.gendim as usize];
//        let n = self.gendim as usize;
//        out_der.mu = der.mu;
//
//        // Copy the visited vector
//        for i in 0..der.mu as usize {
//            out_der.visited[i] = der.visited[i];
//        }
//
//        // Copy all other data
//        let mut k = 0;
//        let mut flag = false;
//        for j in 0..n {
//            for i in j..n {
//                let pij = self.projector[
//                    j + (i * (i+1) / 2)
//                ];
//                if pij != 0.0 {
//                    out_der.expval_o[i] = pij * der.expval_o[k];
//                    out_der.ho[i] = pij * der.ho[k];
//                    out_x[i] = pij * x0[k];
//                    unsafe {
//                        let incx = self.dim;
//                        let incy = self.gendim;
//                        dcopy(
//                            der.mu,
//                            &der.o_tilde[i..(self.dim*der.mu) as usize],
//                            incx,
//                            &mut out_der.o_tilde[k..(self.gendim * der.mu) as usize],
//                            incy
//                        );
//                    }
//                    flag = true;
//                    //break;
//                }
//            }
//            if flag{
//                k += 1;
//                flag = false;
//            }
//        }
//        unsafe {
//            let incx = 1;
//            let incy = 1;
//            dcopy(self.gendim, &out_x, incx, x0, incy);
//        }
//    }
//
//    fn update_delta_alpha_reduced_to_gen(&self, x0: &mut [f64]) {
//        if self.dim == 0 {
//            error!("Tried to map from an empty parameter representation, this will
//                break conjugate gradient or diagonalisation. Dim = {}", self.dim);
//            panic!("Undefined behavior.");
//        }
//        let n = self.gendim as usize;
//        let mut out_x0 = vec![0.0; n];
//
//        // Copy all other data
//        // iter over transpose's diagonal
//        // Copy all other data
//        let mut k = 0;
//        let mut flag = false;
//        for j in 0..n {
//            for i in j..n {
//                let pij = self.projector[
//                    j + (i * (i+1) / 2)
//                ];
//                if pij != 0.0 {
//                    out_x0[i] = pij * x0[k];
//                    flag = true;
//                    //break;
//                }
//            }
//            if flag{
//                k += 1;
//                flag = false;
//            }
//        }
//        unsafe {
//            let incx = 1;
//            let incy = 1;
//            dcopy(self.gendim, &out_x0, incx, x0, incy);
//        }
//    }
//
//    fn update_reduced_representation_no_otilde(&self, out_der: &mut DerivativeOperator, mu: i32, expval_o: &[f64], ho: &[f64], visited: &[usize], x0: &mut [f64]) {
//        if self.dim == 0 {
//            error!("Tried to map from an empty parameter representation, this will
//                break conjugate gradient or diagonalisation. Dim = {}", self.dim);
//            panic!("Undefined behavior.");
//        }
//        let n = self.gendim as usize;
//        out_der.mu = mu;
//
//        // Copy the visited vector
//        for i in 0..mu as usize {
//            out_der.visited[i] = visited[i];
//        }
//
//        // Copy all other data
//        // iter over transpose's diagonal
//        let mut i = 0;
//        for j in 0..n {
//            let pjj = self.projector[
//                    j + (j * (j+1) / 2)
//            ];
//            if pjj != 0.0 { // If 0, parameter does not map to itself.
//                if pjj != 1.0 {
//                    error!("Parameters must map to themselfs with multiple 1.");
//                    panic!("Undefined behavior.");
//                }
//                out_der.expval_o[i] = expval_o[j];
//                out_der.ho[i] = ho[j];
//                // Modify the vectors inplace, we won't iterate on past values
//                x0[i] = x0[j];
//                // Filling the independant parameter index
//                i += 1;
//            }
//        }
//        if i != self.dim as usize {
//            error!("Parameter map did not yield a dimension of {}, instead got {}", self.dim, i);
//            panic!("Undefined behavior.");
//        }
//    }
//
//    fn parallel_reduced_representation_otilde(&self, der: &DerivativeOperator, otilde: &mut [f64]) {
//        if self.dim == 0 {
//            error!("Tried to map from an empty parameter representation, this will
//                break conjugate gradient or diagonalisation. Dim = {}", self.dim);
//            panic!("Undefined behavior.");
//        }
//        let n = self.gendim as usize;
//
//        // iter over transpose's diagonal
//        let mut i = 0;
//        for j in 0..n {
//            let pjj = self.projector[
//                    j + (j * (j+1) / 2)
//            ];
//            if pjj != 0.0 { // If 0, parameter does not map to itself.
//                if pjj != 1.0 {
//                    error!("Parameters must map to themselfs with multiple 1.");
//                    panic!("Undefined behavior.");
//                }
//                // Test for collision
//                if otilde[i] != 0.0 {
//                    error!("O tilde is not empty. Was it reset? Is there pointer collision?");
//                    panic!("Undefined behavior.");
//                }
//                unsafe {
//                    let incx = self.gendim;
//                    let incy = self.dim;
//                    dcopy(
//                        der.mu,
//                        &der.o_tilde[j..n*der.mu as usize],
//                        incx,
//                        &mut otilde[i..(self.dim * der.mu) as usize],
//                        incy
//                    );
//                }
//                // Filling the independant parameter index
//                i += 1;
//            }
//        }
//        if i != self.dim as usize {
//            error!("Parameter map did not yield a dimension of {}, instead got {}", self.dim, i);
//            panic!("Undefined behavior.");
//        }
//    }
//}

fn gradient(x: &[f64], otilde: &[Box<[f64]>], visited: &[&[usize]], expval_o: &[Box<[f64]>], b: &mut [f64], diag_epsilon: &[f64], dim: i32, mu: &[i32], nsamp: f64) {
    let alpha = -1.0;
    let incx = 1;
    let incy = 1;
    let mut work: Vec<f64> = vec![0.0; dim as usize];
    // Compute Ax
    compute_w(&mut work, otilde, visited, expval_o, x, diag_epsilon, dim, mu, nsamp);
    //println!("Ax = {:?}", work);
    unsafe {
        // Compute b - Ax
        daxpy(dim, alpha, &work, incx, b, incy);
    }
}

fn update_x(x: &mut [f64], pk: &[f64], alpha: f64, dim: i32) {
    let incx = 1;
    let incy = 1;
    unsafe {
        daxpy(dim, alpha, &pk, incx, x, incy);
    };
}

fn compute_w(w: &mut [f64], otilde: &[Box<[f64]>], visited: &[&[usize]], expval_o: &[Box<[f64]>], p: &[f64], diag_epsilon: &[f64], dim: i32, mu: &[i32], nsamp: f64) {
    // Computes Ap
    if dim == 0 {
        error!("Cannot compute the matrix product of dimension 0. Something happened with the cutting of dimensions that is not accounted for");
        panic!("Undefined behavior.");
    }
    let incx = 1;
    let incy = 1;
    let alpha = 1.0;
    let beta = 0.0;
    let gamma = 1.0;
    unsafe {
        // Copy p in w, so we can add epsilon*p to the result
        dcopy(dim, p, incx, w, incy);
        // Reset w
        dscal(dim, 0.0, w, incx);
    }

    for (i, o) in otilde.iter().enumerate() {
        let mut work: Vec<f64> = vec![0.0; mu[i] as usize];
        unsafe {
            // 80 misawa
            dgemv(b"T"[0], dim, mu[i], alpha, o, dim, p, incx, beta, &mut work, incy);
            for j in 0..mu[i] as usize {
                work[j] *= visited[i][j] as f64;
            }
            dgemv(b"N"[0], dim, mu[i], 1.0 / nsamp, o, dim, &work, incx, gamma, w, incy);
            trace!("O_[m, mu] O^[T]_[mu, n] x_[n] = {:?}", w);
            let alpha = ddot(dim, &expval_o[i], incx, p, incy);
            // 81 misawa
            daxpy(dim, - alpha, &expval_o[i], incx, w, incy);
        }
        for i in 0..dim as usize {
            w[i] += p[i] * diag_epsilon[i];
        }
    }
}

fn update_r(rk: &mut [f64], w: &[f64], alphak: f64, dim: i32) {
    let incx = 1;
    let incy = 1;
    unsafe {
        daxpy(dim, - alphak, w, incx, rk, incy)
    }
}

fn alpha_k(r_k: &[f64], p_k: &[f64], w: &[f64], alphak: &mut f64, dim: i32) -> f64 {
    let incx = 1;
    let incy = 1;
    unsafe {
        let rkrk = ddot(dim, r_k, incx, r_k, incy);
        *alphak = rkrk / ddot(dim, p_k, incx, w, incy);
        rkrk
    }
}

fn beta_k(r_k: &[f64], rkrk: f64, dim: i32) -> f64 {
    let incx = 1;
    let incy = 1;
    unsafe {
        ddot(dim, r_k, incx, r_k, incy) / rkrk
    }
}

fn update_p(r_k: &[f64], p_k: &mut [f64], beta: f64, dim: i32) {
    let alpha = 1.0;
    let incx = 1;
    let incy = 1;
    unsafe {
        dscal(dim, beta, p_k, incx);
        daxpy(dim, alpha, r_k, incx, p_k, incy);
    }
}

/// Adds an offset to the diagonal of the overlap matrix S_[k,m].
/// TODOC
/// see tahara 2008 3.3.1
/// $$
/// \begin{align}
/// \widetilde{S}_{k,m}&=\epsilon\delta_{k,m}+S_{k,m}\\
/// &=\epsilon\delta_{k,\mu}\delta{\mu,m}+\widetilde{O}^*_{k,\mu}\widetilde{O}^T_{\mu,m}
/// \end{align}
/// $$
pub fn spread_eigenvalues(a: &mut DerivativeOperator) {
    let epsilon = a.epsilon;
    let n_elem = {
        if a.mu > a.n {a.n}
        else {a.mu}
    };
    let work = vec![epsilon; n_elem as usize];
    unsafe {
        let alpha = 1.0;
        let incx = 1;
        let incy = a.n + 1;
        daxpy(n_elem, alpha, &work, incx, &mut a.o_tilde, incy);
    }
}

fn prefilter_overlap_matrix(a_vec: &[DerivativeOperator], _ignore_idx: &mut [bool], dim: i32, _diag_threshold: f64, epsilon: f64, filter_before_shift: bool, nthreads: usize) -> (usize, Box<[f64]>) {
    // WARNING: 3 colums are always junk. Always ignore them. This is done after
    // this function call
    //
    // Loop over diagonal elements of S_km
    // Reminder, S_{kk} = 1/N_{\rm MC.} \sum_\mu \tilde{O}^*_{k\mu}\tilde{O}^T_{\mu k} -
    // \Re{\expval{O_k}}^2

    let mut skip_param_count: usize = 0;
    let mut diag_elem = vec![0.0; dim as usize].into_boxed_slice();
    let mut diag_eps = vec![0.0; dim as usize].into_boxed_slice();
    for k in 0..dim as usize {
        // Start with 1/N_{\rm MC.} \sum_\mu \tilde{O}^*_{k\mu}\tilde{O}^T_{\mu k}
        let z1: f64 = {
            let mut z = 0.0;
            for a in a_vec.iter() {
                let mut tmp = 0.0;
                for j in 0..a.mu as usize {
                    let elem = a.o_tilde[k + j*(dim as usize)];
                    tmp += elem * elem * a.visited[j] as f64;
                }
                //tmp
                z += tmp / a.mu as f64
            }
            z
        };

        // Now \Re{\expval{O_k}}^2
        let z2: f64 = {
            let mut z = 0.0;
            for a in a_vec.iter() {
                z += a.expval_o[k] * a.expval_o[k]
            }
            z / nthreads as f64
        };

        if filter_before_shift {
            diag_elem[k] = z1 - z2;
        } else {
            diag_elem[k] = (z1 - z2) * (1.0 + epsilon);
        }

    }

    let mut max_elem = <f64>::MIN;
    let mut min_elem = <f64>::MAX;
    for k in 0..dim as usize {
        if diag_elem[k] > max_elem {
            max_elem = diag_elem[k];
        }
        if diag_elem[k] < min_elem {
            min_elem = diag_elem[k];
        }
    }
    if min_elem < _diag_threshold * max_elem {
        for k in 0..dim as usize {
            if diag_elem[k] < _diag_threshold * max_elem {
                skip_param_count += 1;
                _ignore_idx[k] = true;
            }
        }
    }
    let mut n = 0;
    for i in 0..dim as usize {
        if filter_before_shift {
            diag_eps[n] = diag_elem[i] * epsilon;
        } else {
            diag_eps[n] = diag_elem[i] * epsilon / (1.0 + epsilon);
        }
        n += 1;
    }

    (dim as usize - skip_param_count, diag_eps)

}

fn cpy_segmented_matrix_to_dense(a_vec: &[DerivativeOperator], output_otilde: &mut [Box<[f64]>], output_expvalo: &mut [Box<[f64]>], ignore_idx: &[bool], dim: i32, nparams_opt: usize) {
    for (ii, a) in a_vec.iter().enumerate() {
        let mut j: usize = 0;
        if a.mu == 0 {
            error!("mu was 0 on entry, was it updated during copy?");
            panic!("Will panic during this call during the copy.");
        }
        for k in 0..dim as usize {
            if j > nparams_opt {
                error!("Supplied dimension does not match allocated memory.");
                panic!("Undefined behavior");
            }
            if ignore_idx[k] {
                continue;
            }
            unsafe {
                dcopy(
                    a.mu,
                    &a.o_tilde[k..(a.mu * a.n) as usize + k - a.n as usize],
                    a.n,
                    &mut output_otilde[ii][j..nparams_opt * a.mu as usize - nparams_opt + j],
                    nparams_opt as i32
                );
            }
            output_expvalo[ii][j] = a.expval_o[k];
            j += 1;

        }
    }
}

fn compute_s_explicit(otilde: &[f64], expval_o: &[f64], visited: &[usize], dim: i32, mu: i32, nsamp: f64, epsilon: f64) -> Vec<f64> {
    // Computes Ap
    let incx = 1;
    let alpha = 1.0/nsamp;
    // Duumbo
    //let alpha = 1.0;
    let beta = -(1.0);
    let mut otilde_w_visited = vec![0.0; (dim * mu) as usize];
    for i in 0..mu as usize {
        for j in 0..dim as usize {
            otilde_w_visited[j + i*dim as usize] = otilde[j + i*dim as usize] * visited[i] as f64;
        }
    }
    //println!("{}", _save_otilde(&otilde, mu as usize, dim as usize));
    //println!("{:?}", visited);

    // Temp work vector
    let mut work = vec![0.0; (dim * dim) as usize];
    unsafe {
        // 80 misawa
        dger(dim, dim, 1.0, &expval_o, incx, &expval_o, incx, &mut work, dim);
        dgemm(b"N"[0], b"T"[0], dim, dim, mu, alpha, &otilde, dim, &otilde_w_visited, dim, beta, &mut work, dim);
    }
    // Shift smallest eigenvalues
    //println!("Before diag filter");
    //println!("{}", _save_otilde(&work, dim as usize, dim as usize));
    for i in 0..dim as usize {
        work[i + dim as usize * i] *= 1.0 + epsilon;
        //work[i + dim as usize * i] += epsilon;
    }
    work
}

fn _save_otilde(der: &[f64], mu: usize, n: usize) -> String {
    let width = 16;
    let mut o_tilde = "".to_owned();
    for m in 0..mu {
        for i in 0..n  {
            if i == m {
                o_tilde.push_str(&format!("{:>width$.04e}", der[i + m * n]).yellow());
            } else {
                o_tilde.push_str(&format!("{:>width$.04e}", der[i + m * n]));
            }
        }
        o_tilde.push_str("\n");
    }
    o_tilde
}

fn diagonalize_dense_matrix(s: &mut [f64], dim: i32) -> Vec<f64> {
    let jobz = b"V"[0];
    let uplo = b"U"[0];
    let mut w = vec![0.0; dim as usize];
    let lwork = 3*(dim);
    let mut work = vec![0.0; lwork as usize];
    let mut info = 0;
    unsafe {
        dsyev(
            jobz,
            uplo,
            dim,
            s,
            dim,
            &mut w,
            &mut work,
            lwork,
            &mut info
        );
    }
    if info < 0{
        error!("Parameter {} had an illegal value in call to lapack::dsyev.", <i32>::abs(info));
    }
    else if info > 0{
        error!("Convergence was not achieved to diagonalize the overlap matrix.
            {} off-diagonal elements did not converge to 0.", info);
    }

    w
}

fn compute_delta_from_eigenvalues(x0: &mut [f64], eigenvectors: &[f64], eigenvalues: &[f64], dim: i32) {
    let trans = b"T"[0];
    let incx = 1;
    let incy = 1;
    let mut work = vec![0.0; dim as usize];
    unsafe {
        dgemv(trans, dim, dim, 1.0, eigenvectors, dim, x0, incx, 0.0, &mut work, incy);
    }
    for i in 0..dim as usize{
        work[i] *= eigenvalues[i];
    }
    let trans = b"N"[0];
    unsafe {
        dgemv(trans, dim, dim, 1.0, eigenvectors, dim, &work, incx, 0.0, x0, incy);
    }
}

fn _compute_matrix_product(s: &mut [f64], eigenvectors: &[f64], eigenvalues: &[f64], dim: i32) {
    let transb = b"T"[0];
    let incx = 1;
    let incy = 1;
    let mut work = vec![0.0; (dim * dim) as usize];
    let mut s_copy = vec![0.0; (dim * dim) as usize];
    let mut work_direct = vec![0.0; (dim * dim) as usize];
    let mut s_copy_direct = vec![0.0; (dim * dim) as usize];
    unsafe {
        dcopy(dim*dim, eigenvectors, incx, &mut work, incy);
        dcopy(dim*dim, eigenvectors, incx, &mut work_direct, incy);
        //dgemm(transa, transb, dim, dim, dim, 1.0, eigenvectors, dim, s, dim, 0.0, &mut s_copy, dim);
    }
    for i in 0..dim as usize{
        for j in 0..dim as usize {
           if eigenvalues[i] < 1e-6 {
           work[j + dim as usize * i] *= 0.0;
           work_direct[j + dim as usize * i] *= 0.0;
           continue;
           }
            work[j + dim as usize * i] *= 1.0 / eigenvalues[i];
            work_direct[j + dim as usize * i] *= eigenvalues[i];
        }
    }
    let transa = b"N"[0];
    unsafe {
        dgemm(transa, transb, dim, dim, dim, 1.0, eigenvectors, dim, &work, dim, 0.0, &mut s_copy, dim);
        dgemm(transa, transb, dim, dim, dim, 1.0, eigenvectors, dim, &work_direct, dim, 0.0, &mut s_copy_direct, dim);
        //dcopy(dim*dim, s, incx, &mut work, incy);
    let transa = b"N"[0];
    let transb = b"T"[0];
        dgemm(transa, transb, dim, dim, dim, 1.0, &s_copy, dim, &s_copy_direct, dim, 0.0, s, dim);
    }
}

//pub fn exact_overlap_inverse(a: &DerivativeOperator, b: &mut [f64], epsilon: f64, dim: i32, thresh: f64) -> Vec<bool>{
//    // PRE FILTER
//    let mut ignore = vec![false; dim as usize];
//    //println!("dim = {}, Unfiltered S = ", dim);
//    //println!("{}", save_otilde(&unfiltered_s, dim as usize, dim as usize));
//    let (new_dim, _) = prefilter_overlap_matrix(a, &mut ignore, dim, thresh, epsilon, true);
//    let mut otilde = vec![0.0; new_dim * a.mu as usize].into_boxed_slice();
//    let mut expvalo = vec![0.0; new_dim].into_boxed_slice();
//    cpy_segmented_matrix_to_dense(a, &mut otilde, &mut expvalo, &ignore, dim, new_dim);
//    //println!("{}", _save_otilde(&a.o_tilde, a.mu as usize, a.n as usize));
//    let mut filtered_s = compute_s_explicit(&otilde, &expvalo, &a.visited, new_dim as i32, a.mu, a.nsamp, epsilon);
//    let mut work = vec![0.0; new_dim];
//    unsafe {
//        dgemv(b"N"[0], new_dim as i32, new_dim as i32, 1.0, &filtered_s, new_dim as i32, &b, 1, 0.0, &mut work, 1);
//    }
//    println!("Ax = {:?}", work);
//    println!("{}", _save_otilde(&filtered_s, new_dim as usize, new_dim as usize));
//    println!("a.mu = {}, a.n = {}", a.mu, a.n);
//    println!("dim = {}, new_dim = {}", dim, new_dim);
//    let mut _unfiltered_s = compute_s_explicit(&a.o_tilde, &a.expval_o, &a.visited, dim as i32, a.mu, a.nsamp, epsilon);
//    //for i in 0..dim as usize {
//    //    if ignore[i] {
//    //        println!("{}", _save_otilde(&filtered_s, new_dim, new_dim));
//    //        println!("ignore : {:?}", ignore);
//    //        println!("{}", _save_otilde(&unfiltered_s, dim as usize, dim as usize));
//    //        panic!("stop");
//    //        break;
//
//    //    }
//    //}
//    // Test matrix are equal
//    //for i in 0..dim as usize {
//    //    for j in 0.. dim as usize {
//    //        if ! (filtered_s[j + i * dim as usize] == unfiltered_s[j + i * dim as usize]) {
//    //            panic!("Assertion");
//    //        }
//    //    }
//    //}
//    //println!("dim = {}, Unfiltered S = ", dim);
//    //println!("{}", _save_otilde(&unfiltered_s, dim as usize, dim as usize));
//    //let new_dim = prefilter_overlap_matrix(a, &mut ignore, dim, thresh);
//    ////println!("ignore : {:?}", ignore);
//    //let mut otilde = vec![0.0; new_dim * a.mu as usize];
//    //let mut expvalo = vec![0.0; new_dim];
//    //cpy_segmented_matrix_to_dense(a, &mut otilde, &mut expvalo, &ignore, dim, new_dim);
//    //let filtered_s = compute_s_explicit(&otilde, &expvalo, a.visited, new_dim as i32, a.mu, a.nsamp);
//    let eigenvalues = diagonalize_dense_matrix(&mut filtered_s, new_dim as i32);
//    let _eigenvalues = diagonalize_dense_matrix(&mut _unfiltered_s, dim as i32);
//    //_compute_matrix_product(&mut s_copy, &unfiltered_s, &eigenvalues, dim);
//    //println!("S^[-1] = \n{}", _save_otilde(&s_copy, dim as usize, dim as usize));
//    //let eigenvectors = &unfiltered_s;
//    //println!("S = \n{}", save_otilde(&s_copy, dim as usize, dim as usize));
//    //compute_matrix_product(&mut s_copy, &eigenvectors, &eigenvalues, dim);
//    //println!("S^[-1] = \n{}", save_otilde(&s_copy, dim as usize, dim as usize));
//    //panic!("stop");
//    //let mut x0_raw = vec![0.0; dim as usize];
//    //unsafe {
//    //    let incx = 1;
//    //    let incy = 1;
//    //    dgemv(b"T"[0], dim, dim, 1.0, &s_copy, dim, x0, incx, 0.0, &mut x0_raw, incy);
//    //}
//    //println!("UD^[-1]U^[T] = \n{}", save_otilde(&s_copy, dim as usize, dim as usize));
//    //println!("x0 = {:?}", x0_raw);
//    println!("filted eigenvalues: {:?}", eigenvalues);
//    println!("eigenvalues: {:?}", _eigenvalues);
//
//    // Remove problematic eigenvalue
//    //let mut max = <f64>::MIN;
//    ////println!("{:?}", eigenvalues);
//    //for e in eigenvalues.iter() {
//    //    if *e > max {
//    //        max = *e;
//    //    }
//    //}
//    //let threshold = thresh * max;
//    //let mut i = 0;
//    //for e in eigenvalues.iter_mut() {
//    //    if *e < threshold {
//    //        *e = 0.0;
//    //        //ignore[i] = true;
//    //    }
//    //    i+=1;
//    //}
//    ////println!("{:?}", ignore);
//    ////Invert matrix
//    //for e in eigenvalues.iter_mut() {
//    //    if *e == 0.0 {
//    //        continue;
//    //    }
//    //    *e = 1.0 / *e;
//    //}
//    compute_delta_from_eigenvalues(b, &filtered_s, &eigenvalues, new_dim as i32);
//    //println!("b = {:?}", b);
//
//    return ignore;
//}

/// Computes the solution of $A\mathbf{x}-\mathbf{b}=0$
/// TODOC
/// Output
/// b is the optimised parameter step.
pub fn conjugate_gradiant(a: &[DerivativeOperator], b: &mut [f64], x0: &mut [f64], epsilon: f64, kmax: usize, dim: i32, thresh: f64, epsilon_convergence: f64, filter_before_shift: bool, nthreads: usize, ngi: usize, nvij: usize) -> Vec<bool>
{
    // PRE FILTER
    let mut ignore = vec![false; dim as usize];
    let (mut new_dim, mut diag_epsilon) = prefilter_overlap_matrix(a, &mut ignore, dim, thresh, epsilon, filter_before_shift, nthreads);
    // ALWAYS DELETE THE FIRST COLUMN OR EACH PARAMETER TYPE
    // this houses garbage data, this data gets there because of the parameter map
    if !ignore[0] {
        ignore[0] = true;
        new_dim -= 1;
    }
    if !ignore[ngi + 1] {
        ignore[ngi + 1] = true;
        new_dim -= 1;
    }
    if !ignore[ngi + nvij + 2] {
        ignore[ngi + nvij + 2] = true;
        new_dim -= 1;
    }

    let mut otilde = Vec::new();
    let mut expvalo = Vec::new();
    let mut vis: Vec<&[usize]> = Vec::new();
    let mut mus = Vec::new();
    let mut nsamp = 0.0;
    for a in a.iter() {
        let otildei = vec![0.0; new_dim * a.mu as usize].into_boxed_slice();
        let expvaloi = vec![0.0; new_dim].into_boxed_slice();
        otilde.push(otildei);
        expvalo.push(expvaloi);
        vis.push(&a.visited);
        mus.push(a.mu);
        nsamp += a.nsamp;
    }
    cpy_segmented_matrix_to_dense(a, &mut otilde, &mut expvalo, &ignore, dim, new_dim);

    let mut w = vec![0.0; new_dim].into_boxed_slice();
    // Error threshold
    let mut e = 0.0;
    let mut j = 0;
    for i in 0..dim as usize {
        if ignore[i] {
            b[i] = 0.0;
            diag_epsilon[i] = 0.0;
            continue;
        }
        e += b[i] * b[i];
        b[j] = b[i];
        diag_epsilon[j] = diag_epsilon[i];
        j += 1;
    }
    // Sanity check
    for i in 0..dim as usize {
        if i >= j {
            b[i] = 0.0;
            diag_epsilon[i] = 0.0;
        }
    }
    e *= epsilon_convergence;
    trace!("Error threshold e = {}", e);
    //println!("Error threshold e = {}", e);
    gradient(x0, &otilde, &vis, &expvalo, b, &diag_epsilon, new_dim as i32, &mus, nsamp);
    let mut p = vec![0.0; new_dim].into_boxed_slice();
    unsafe {
        dcopy(new_dim as i32, b, 1, &mut p, 1);
    }
    let mut alpha = 0.0;

    for k in 0..kmax {
        trace!("r_{} : {:?}", k, b);
        trace!("p_{} : {:?}", k, p);
        //println!("r_{} : {:?}", k, b);
        //println!("p_{} : {:?}", k, p);
        compute_w(&mut w, &otilde, &vis, &expvalo, &p, &diag_epsilon, new_dim as i32, &mus, nsamp);
        //println!("w_{} : {:?}", k, w);
        let nrm2rk = alpha_k(b, &p, &w, &mut alpha, new_dim as i32);
        trace!("alpha_{} : {}", k, alpha);
        //println!("alpha_{} : {}", k, alpha);
        //if alpha < 0.0 {
        //    error!("Input overlap matrix S was not positive-definite.");
        //    //break;
        //    panic!("p^T S p < 0.0");
        //}
        update_x(x0, &p, alpha, new_dim as i32);
        trace!("x_{} : {:?}", k+1, x0);
        //println!("x_{} : {:?}", k+1, x0);
        update_r(b, &w, alpha, new_dim as i32);
        let beta = beta_k(b, nrm2rk, new_dim as i32);
        if beta * nrm2rk < e {
            trace!("Achieved convergence at {} iterations", k);
            break;
        }
        trace!("beta_{} : {}", k, beta);
        update_p(b, &mut p, beta, new_dim as i32);
    }
    unsafe {
        dcopy(new_dim as i32, x0, 1, b, 1);
    }
    //println!("b = {:?}", b);
    ignore
}
