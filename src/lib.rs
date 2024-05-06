#[cfg(feature = "python-interface")]
use pyo3::prelude::*;

/// Size of the system.
pub const SIZE: usize = 10;

/// Hubbard's model $U$ parameter
pub static CONS_U: f64 = 1.0;
/// Hubbard's model $t$ parameter
pub static CONS_T: f64 = 1.0;

/// Input file parsing util.
/// # Subfiles
/// * __`orbitale.csv`__ - Variationnal parameters for the orbital. In csv
/// format, 3 column, for $f_{ij}$. First column is `i`, second is `j`. The third
/// column is for the parameters identifier.
pub mod parse;

// Have the FockState struct at the root.
include!("fock_state.rs");

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
