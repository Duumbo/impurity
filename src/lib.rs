/// Size of the system.
type BitStruct = SpinState;
pub const SIZE: usize = 10;

pub static CONS_U: f64 = 1.0;
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
/// Call the workspace query function on the $f_{ij}$ matrix and compute the
/// pfaffian. When the fast update function will be implemented, it will be
/// here.
/// # Definition
/// The Pfaffian wave function is defined such that
///
/// $$
/// \vert \phi_{\text{PF}}\rangle=
/// \left[
///     \sum_{i, j = 0}^{N_s-1}f_{ij}c_{i\uparrow}^\dagger c_{j\downarrow}^\dagger
/// \right]^{N_e /2} \vert 0\rangle
/// $$
///
/// When computing the pfaffian, it is the following scalar product that is
/// obtained:
///
/// $$
/// \langle x\vert\phi_{\text{PF}}\rangle=
/// \left(\frac{N_e}{2}\right)!\ \text{Pf}(X)
/// $$
///
/// $$
/// \text{Where } X_{ij}=f_{ij}-f_{ji}
/// $$
///
pub mod pfaffian;

/// Calculate Jastrow coefficients
/// # Definition
/// The Jastrow correlation factors is defined as:
///
/// $$
/// P_J=e^{\frac12\sum_{i\neq j}v_{ij}(n_i-1)(n_j-1)}
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
pub mod jastrow;

/// Calculate Gutzwiller coefficients
/// # Definition
/// The Gutzwiller correlation factors is defined as:
///
/// $$
/// P_G=e^{\sum_ig_in_{i\uparrow}n_{i\downarrow}}
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
/// This is equivalent to $n_{i\downarrow}\\&n_{i\uparrow}$
pub mod gutzwiller;

pub mod hamiltonian;
