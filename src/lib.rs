#[cfg(feature = "python-interface")]
use pyo3::prelude::*;

use blas::{dgemm};

mod strings;

// Have the FockState struct at the root.
include!("fock_state.rs");

// Include the params definition
include!("constants.rs");
include!("hoppings.rs");


/// Collection of the variationnal parameters
/// # Definition
/// This struct contains the reference to the variationnal parameters, in the
/// general representation. It is advised to place all three kind of variationnal
/// parameter contiguously in memory.
/// # Fields
/// * __`size`__ - $N_{\rm sites}$, the number of sites in the system.
/// * __`gi[i]`__ - Gutzwiller parameters $g_i$.
/// * __`vij[i + j*(j-1)/2]`__ - Jastrow parameters $v_{i,j}$. Symetric packed matrix.
/// * __`fij[i * N + j + (sigma_i + 2*sigma_j) * N * N]`__ - Orbital (Pfaffian)
/// parameters $F_{i,j}^{\sigma_i,\sigma_j}$.
pub struct VarParams<'a> {
    pub fij: &'a mut [f64],
    pub vij: &'a mut [f64],
    pub gi: &'a mut [f64],
    pub size: usize,
}

impl<'a> std::fmt::Display for VarParams<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::result::Result<(), std::fmt::Error> {
        let width = 8;
        for i in 0..4*self.size*self.size {
            if i < self.size*self.size {
                writeln!(f, "F_[{},{}]^[{UPARROW}, {UPARROW}]={:>width$.05e}", i/self.size, i%self.size, self.fij[i])?;
            }
            else if i < 2*self.size*self.size {
                writeln!(f, "F_[{},{}]^[{UPARROW}, {DOWNARROW}]={:>width$.05e}", i/self.size - self.size, i%self.size, self.fij[i])?;
            }
            else if i < 3*self.size*self.size {
                writeln!(f, "F_[{},{}]^[{DOWNARROW}, {UPARROW}]={:>width$.05e}", i/self.size - 2*self.size, i%self.size, self.fij[i])?;
            }
            else {
                writeln!(f, "F_[{},{}]^[{DOWNARROW}, {DOWNARROW}]={:>width$.05e}", i/self.size - 3*self.size, i%self.size, self.fij[i])?;
            }
        }
        for i in 0..self.size*self.size {
            writeln!(f, "V_[{},{}]={:>width$.05e}", i/self.size, i%self.size, self.vij[i])?;
        }
        for i in 0..self.size {
            writeln!(f, "G_[{}]={:>width$.05e}", i, self.gi[i])?;
        }
        Ok(())
    }
}

/// The operator $\tilde{O}_{k\mu}$
/// # Definition
/// This struct represents the derivative operator $\tilde{O}_{k\mu}$ defined
///
/// $$
/// \tilde{O}_{k\mu}=\vert x\_{\mu}\rangle\left(
/// \frac{1}{\langle x\_{\mu}\vert\psi\rangle}\frac{\partial}{\partial\alpha_k}\langle x\_{\mu}\vert\psi\rangle
/// \right)\langle x\_{\mu}\vert
/// $$
///
/// where the index $\mu$ represents the Monte-Carlo sampling iteration. This
/// struct also hold $\langle O_k\rangle$ and $\langle HO_k\rangle$.
pub struct DerivativeOperator {
    /// __`n`__ - Number of variationnal parameters to be optimised.
    pub n: i32,
    /// __`mu`__ - Monte-Carlo sample iteration. `-1` when struct is empty, else
    /// points to the last sample.
    pub mu: i32,
    /// __`nsamp`__ - Number of total samples, for normalisation.
    pub nsamp: f64,
    /// __`nsamp_int`__ - Number of total samples.
    pub nsamp_int: usize,
    /// __`epsilon`__ - Epsilon value to add to diagonal elements of the overlap
    /// matrix.
    pub epsilon: f64,
    /// __`o_tilde[k + mu * n]`__ - $\tilde{O}_{k,\mu}$.
    pub o_tilde: Box<[f64]>,
    /// __`expval_o[k]`__ - $\langle O_k\rangle$.
    pub expval_o: Box<[f64]>,
    /// __`ho[k]`__ - $\langle HO_k\rangle$.
    pub ho: Box<[f64]>,
    /// __`visited[mu]`__ - Number of times this row is duplicated in $\tilde{O}_{k\mu}$.
    pub visited: Box<[usize]>,
}

impl DerivativeOperator {
    fn new(n: i32, mu: i32, nsamp: f64, nsamp_int: usize, epsilon: f64) -> Self {
        DerivativeOperator {
            o_tilde: vec![0.0; n as usize * nsamp as usize].into_boxed_slice(),
            expval_o: vec![0.0; n as usize].into_boxed_slice(),
            ho: vec![0.0; n as usize].into_boxed_slice(),
            visited: vec![0; nsamp as usize].into_boxed_slice(),
            mu,
            n,
            nsamp,
            nsamp_int,
            epsilon
        }
    }
}

impl<'a> std::fmt::Display for DerivativeOperator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::result::Result<(), std::fmt::Error> {
        let width = 10;
        let mut expval = "<O> = ".to_owned();
        let mut ho = "<HO> = ".to_owned();
        let mut o_tilde = "O = ".to_owned();
        for n in 0..self.n as usize {
            expval.push_str(&format!("{:>width$.04e} ", self.expval_o[n]));
            ho.push_str(&format!("{:>width$.04e} ", self.ho[n]));
            for mu in 0..(self.mu + 1) as usize {
                o_tilde.push_str(&format!("{:>width$.04e}", self.o_tilde[n + n * mu]));
            }
            o_tilde.push_str("\n");
        }
        expval.push_str("\n");
        ho.push_str("\n");
        o_tilde.push_str("\n");
        write!(f, "{}", expval)?;
        write!(f, "{}", ho)?;
        write!(f, "{}", o_tilde)?;
        write!(f, "mu = {}", self.mu)?;
        Ok(())

    }
}


fn _save_otilde(der: &DerivativeOperator) {
    let width = 16;
    let mut c = vec![0.0; (der.n * der.n) as usize];
    println!("dim = {}", der.n * der.n);
    unsafe {
        dgemm(b"N"[0], b"T"[0], der.n, der.n, der.mu, 1.0, &der.o_tilde, der.n, &der.o_tilde, der.n, 0.0, &mut c, der.n);
    }
    let mut outstr = "".to_owned();
    outstr.push_str(&format!("<O_kO_m> = "));
    for i in 0..der.n as usize {
        outstr.push_str(&format!("\n           "));
        for j in 0..der.n as usize {
            outstr.push_str(&format!("{:>width$.04e}", c[i + der.n as usize * j]));
        }
    }
    println!("{}", outstr);
}

//pub fn mapto_pairwf(input: &DerivativeOperator, output: &mut DerivativeOperator, sys: &SysParams) {
//    if input.mu != output.mu {
//        error!("Input dimension does not match output dimension. Make sure to match mu for both structures.");
//        panic!("Undefined Behavior.");
//    }
//    if input.mu < 0 {
//        error!("Dimension cannot be negative; is it empty?");
//        panic!("Undefined Behavior");
//    }
//    let nfij = sys.size*sys.size;
//    // Copy and scale fij from FIJ
//    for i in sys.ngi+sys.nvij+nfij..sys.ngi+sys.nvij+2*nfij {
//        unsafe {
//            //println!("{}", output.mu);
//            dcopy(
//                input.mu,
//                &input.o_tilde[i..input.n as usize + input.mu as usize * (i + 1)],
//                input.n,
//                &mut output.o_tilde[(i - nfij)..output.n as usize + output.mu as usize * (i - nfij + 1)],
//                output.n
//            );
//            //for j in 0..nfij {
//            //    dscal(
//            //        output.mu,
//            //        der_facteur,
//            //        &mut output.o_tilde[j+sys.ngi+sys.nvij..output.n as usize + output.mu as usize * (j + 1 + sys.ngi + sys.nvij)],
//            //        output.n,
//            //    )
//            //}
//        }
//
//    }
//    // Copy Gutzwiller and jastrow
//    for i in 0..sys.ngi+sys.nvij {
//        unsafe {
//            dcopy(
//                input.mu,
//                &input.o_tilde[i..input.n as usize * (i + 1)],
//                input.n,
//                &mut output.o_tilde[i..output.n as usize * (i + 1)],
//                output.n
//            );
//        }
//
//    }
//    unsafe {
//        dcopy(
//            nfij as i32,
//            &input.expval_o[input.pfaff_off + nfij..input.pfaff_off + 2*nfij],
//            1,
//            &mut output.expval_o[input.pfaff_off..output.pfaff_off + nfij],
//            1
//        );
//        dcopy(
//            nfij as i32,
//            &input.ho[input.pfaff_off + nfij..input.pfaff_off + 2*nfij],
//            1,
//            &mut output.ho[input.pfaff_off..input.pfaff_off + nfij],
//            1
//        );
//        //dscal(nfij as i32, der_facteur, &mut output.expval_o[output.pfaff_off..output.pfaff_off + nfij], 1);
//        //dscal(nfij as i32, der_facteur, &mut output.ho[output.pfaff_off..output.pfaff_off + nfij], 1);
//        dcopy(
//            sys.nvij as i32,
//            &input.expval_o[input.jas_off..input.jas_off + sys.nvij],
//            1,
//            &mut output.expval_o[output.jas_off..output.jas_off + sys.nvij],
//            1
//        );
//        dcopy(
//            sys.nvij as i32,
//            &input.ho[input.jas_off..input.jas_off + sys.nvij],
//            1,
//            &mut output.ho[output.jas_off..output.jas_off + sys.nvij],
//            1
//        );
//        dcopy(
//            sys.ngi as i32,
//            &input.expval_o[0..sys.ngi],
//            1,
//            &mut output.expval_o[0..sys.ngi],
//            1
//        );
//        dcopy(
//            sys.ngi as i32,
//            &input.ho[0..sys.ngi],
//            1,
//            &mut output.ho[0..sys.ngi],
//            1
//        );
//    }
//}

/// Module to calculate pfaffian
/// # Usage
/// Call the workspace query function on the $F^{\sigma\sigma'}_{ij}$ matrix and compute the
/// pfaffian. Then the fast-update can be used to compute the ratio of probability
/// of the hopping.
/// # Definition
/// The Pfaffian wave function is defined such that
///
/// $$
/// \vert \phi_{\text{PF}}\rangle=
/// \left[
///     \sum_{i, j = 0}^{N_s-1}F_{ij}^{\sigma\sigma'}c_{i\sigma}^\dagger c_{j\sigma'}^\dagger
/// \right]^{N_e /2} \vert 0\rangle
/// $$
///
/// and the order of the column in this matrix are defined with the normal order.
/// The normal order is to have the first half be the spin up and the second be
/// the spin down. Each subpart are numericaly ordered.
/// When computing the pfaffian, it is the following scalar product that is
/// obtained:
///
/// $$
/// \langle x\vert\phi_{\text{PF}}\rangle=
/// \left(\frac{N_e}{2}\right)!\ \text{Pf}(X)
/// $$
///
/// $$
/// \text{Where } X_{ij}=F_{ij}^{\sigma\sigma'}-F_{ji}^{\sigma\sigma'}
/// $$
///
/// # Example
///
///
/// $$
/// X=\begin{pmatrix}
/// 0&F_{01}^{\uparrow\uparrow}&F_{00}^{\uparrow\down}&F_{01}^{\uparrow\down}\\\\
/// -F_{10}^{\uparrow\uparrow}&0&F_{10}^{\uparrow\down}&F_{11}^{\uparrow\down}\\\\
/// -F_{00}^{\down\uparrow}&-F_{01}^{\down\uparrow}&0&F_{01}^{\down\down}\\\\
/// -F_{10}^{\down\uparrow}&-F_{11}^{\down\uparrow}&-F_{10}^{\down\down}&0
/// \end{pmatrix}
/// $$
pub mod pfaffian;

/// Computation of the mean energy $\langle E\rangle$.
/// # Definition
/// This module computes the mean of observable quantities, currently only
///
/// $$
/// \langle E\rangle=\frac{\langle\psi\vert E\vert\psi\rangle}{\langle\psi\vert\psi\rangle}\\\\
/// \langle E\rangle=\sum_x
/// \frac{\langle\psi\vert x\rangle\langle x\vert
/// E\vert\psi\rangle}{\langle\psi\vert\psi\rangle}\\\\
/// \langle E\rangle=\sum_x\rho(x)
/// \frac{\langle x\vert
/// E\vert\psi\rangle}{\langle x\vert\psi\rangle}
/// $$
///
/// Computation for this value can be exact, meaning we sample the whole Hilbert
/// space, or can be Monte-Carlo, where we sample according to a Markov chain.
pub mod monte_carlo;

/// Computation of the internal product $\langle x\vert\psi\rangle$.
/// # Definition
/// Computing the internal product has two main parts to it:
/// * Computing the pfaffian of the configuration matrix for $\langle x\vert\phi_{\rm
/// pair}\rangle$.
/// * Computing the projector coefficients $\mathcal{P}_G\mathcal{P}_J$.
///
/// The functions inside this module call the appropriate routines for some
/// predefined use-cases.
pub mod density;

/// Calculate Jastrow coefficients
/// # Definition
/// The Jastrow correlation factors is defined as:
///
/// $$
/// P_J=e^{\frac12\sum_{i\neq j}v_{ij}(n_i-1)(n_j-1)}\newline
/// \ln P_J=\frac12\sum_{i\neq j}v_{ij}(n_i-1)(n_j-1)
/// $$
///
/// # Truth table
/// Using the definition $n_i=n_{i\uparrow}+n_{i\downarrow}$,
///
/// | $n_{i\uparrow}$ | $n_{i\downarrow}$ | $n_{j\uparrow}$ | $n_{j\downarrow}$ | $(n_i-1)(n_j-1)$ |
/// |-----------------|-------------------|-----------------|-------------------|------------------|
/// | 0   | 0   | 0 | 0 | 1                   |
/// | 0   | 1   | 0 | 0 | 0                   |
/// | 1   | 0   | 0 | 0 | 0                   |
/// | 1   | 1   | 0 | 0 | -1                   |
/// | 0   | 0   | 1 | 0 | 0                   |
/// | 0   | 1   | 1 | 0 | 0                   |
/// | 1   | 0   | 1 | 0 | 0                   |
/// | 1   | 1   | 1 | 0 | 0                   |
/// | 0   | 0   | 0 | 1 | 0                   |
/// | 0   | 1   | 0 | 1 | 0                  |
/// | 1   | 0   | 0 | 1 | 0                   |
/// | 1   | 1   | 0 | 1 | 0                   |
/// | 0   | 0   | 1 | 1 | -1                   |
/// | 0   | 1   | 1 | 1 | 0                   |
/// | 1   | 0   | 1 | 1 | 0                   |
/// | 1   | 1   | 1 | 1 | 1                   |
///
/// This projector encodes an interaction between holes and pairs of electrons.
///
pub mod jastrow;

/// Calculate Gutzwiller coefficients
/// # Definition
/// The Gutzwiller correlation factor is defined as:
///
/// $$
/// P_G=e^{\sum_ig_in_{i\uparrow}n_{i\downarrow}}\newline
/// \ln P_G=\sum_ig_in_{i\uparrow}n_{i\downarrow}
/// $$
///
/// # Truth table
/// | $n_{i\uparrow}$ | $n_{i\downarrow}$ | $n_{i\downarrow}n_{i\uparrow}$ |
/// |-----------------|-------------------|--------------------------------|
/// | 0 | 0 | 0 |
/// | 0 | 1 | 0 |
/// | 1 | 0 | 0 |
/// | 1 | 1 | 1 |
///
/// This projector encodes an interaction between electrons on the same site.
pub mod gutzwiller;

/// Hubbard's model terms for hopping.
/// # Definition
/// The Hubbard model Hamiltonian is defined
/// $$
/// H=U\sum_i n_{i\uparrow}n_{i\downarrow}
/// -t\sum_{<i,j>,\sigma}c^\dagger_{i\sigma}c_{j\sigma}+c^\dagger_{j\sigma}c_{i\sigma}
/// $$
pub mod hamiltonian;

/// The optimisation of the ground state
/// # Definition
/// The optimisation of the variationnal parameters follow the Stochastic-Reconfiguration (SR)
/// scheme. In this scheme, we update the variationnal parameters using the
/// imaginary time evolution of the Schrodigner equation.
///
/// $$
/// \frac{\partial\alpha_k}{\partial\tau}=
/// -S^{-1}\_{k\ell}\langle \partial\_{\alpha_\ell}\bar{\psi}\vert
/// (H-\langle H\rangle)\vert\bar{\psi}\rangle
/// $$
///
/// The main computation of this module is to inverse the overlap matrix $S$,
/// by solving the linear equation $S\mathbf{x}=\mathbf{b}$,
/// which is defined
///
/// $$
/// S_{km}=\langle O^*\_{k}O\_m\rangle-\langle O\_k\rangle\langle O\_m\rangle\\\\
/// b_m=\langle HO\_m\rangle-\langle H\rangle\langle O\_m\rangle
/// $$
pub mod optimisation;

/// Computation of the ground-state
/// # Definition
/// This is the main computation of this program. This module's task is to
/// iteratively compute the mean energy of a variationnal state and update its
/// variationnal parameters according to the optimisation scheme. It is also
/// tasked of making sense of the parameters that are optimised and not
/// optimised.
pub mod dvmc;

/// TODOC
pub mod green;

#[cfg(feature = "python-interface")]
#[pymodule]
fn impurity(m: &Bound<'_, PyModule>) -> PyResult<()> {
    use pyo3::wrap_pyfunction;

    m.add_function(wrap_pyfunction!(gutzwiller::gutzwiller_exponent, m)?)?;
    m.add_function(wrap_pyfunction!(gutzwiller::gutzwiller_fastupdate, m)?)?;
    m.add_function(wrap_pyfunction!(jastrow::jastrow_exponent, m)?)?;
    m.add_function(wrap_pyfunction!(jastrow::jastrow_fastupdate, m)?)?;
    m.add_function(wrap_pyfunction!(jastrow::compute_jastrow_easy_to_follow, m)?)?;
    m.add_function(wrap_pyfunction!(density::compute_internal_product_py, m)?)?;
    Ok(())
}
