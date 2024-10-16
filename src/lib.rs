#[cfg(feature = "python-interface")]
use pyo3::prelude::*;

/// Input file parsing util.
/// # Subfiles
/// * __`orbitale.csv`__ - Variationnal parameters for the orbital. In csv
/// format, 3 column, for $f_{ij}$. First column is `i`, second is `j`. The third
/// column is for the parameters identifier.
pub mod parse;

mod strings;

// Have the FockState struct at the root.
include!("fock_state.rs");

// Include the params definition
include!("constants.rs");
include!("hoppings.rs");


/// Collection of the variationnal parameters
/// TODOC
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

/// The operator $O_k$
/// TODOC
pub struct DerivativeOperator<'a> {
    pub o_tilde: &'a mut [f64],
    pub expval_o: &'a mut [f64],
    pub ho: &'a mut [f64],
    /// Number of variationnal parameters
    pub n: i32,
    /// Number of monte-carlo sampling
    pub mu: i32,
    pub nsamp: f64,
    pub nsamp_int: usize,
    pub visited: &'a mut [usize],
    pub pfaff_off: usize,
    pub jas_off: usize,
    pub epsilon: f64,
}

impl<'a> std::fmt::Display for DerivativeOperator<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::result::Result<(), std::fmt::Error> {
        let width = 10;
        let mut expval = "<O> = ".to_owned();
        let mut ho = "<HO> = ".to_owned();
        let mut o_tilde = "O = ".to_owned();
        for mu in 0..(self.mu + 1) as usize {
            expval.push_str(&format!("{:>width$.04e} ", self.expval_o[mu]));
            ho.push_str(&format!("{:>width$.04e} ", self.ho[mu]));
            for n in 0..self.n as usize {
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

pub mod monte_carlo;

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
/// TODOC
pub mod optimisation;

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
