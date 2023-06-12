
/// Size of the system.
pub const SIZE: usize = 2;

/// Input file parsing util.
/// # Subfiles
/// * __`orbitale.csv`__ - Variationnal parameters for the orbital. In csv
/// format, 3 column, for $f_{ij}$. First column is `i`, second is `j`. The third
/// column is for the parameters identifier.
pub mod parse;

/// Convenient mod to calculate pfaffian
pub mod pfaffian;
