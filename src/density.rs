#[cfg(feature = "python-interface")]
use pyo3::{pyfunction, PyResult};

use crate::jastrow::compute_jastrow_exp;
use crate::gutzwiller::compute_gutzwiller_exp;
use crate::pfaffian::construct_matrix_a_from_state;
use crate::FockState;

pub fn compute_internal_product(
    state: FockState<u8>,
    fij: &[f64],
    vij: &[f64],
    gi: &[f64],
    nsites: usize,
) -> f64 {
    let mut pfaffian_state = construct_matrix_a_from_state(fij, state);
    let pfaffian = pfaffian_state.pfaff;
    pfaffian_state.rebuild_matrix();
    let jastrow_exp = compute_jastrow_exp(state, vij, nsites);
    let gutz_exp = compute_gutzwiller_exp(state, gi, nsites);
    let scalar_prod = <f64>::exp(jastrow_exp + gutz_exp) * pfaffian;
    scalar_prod
}

#[cfg(feature = "python-interface")]
#[pyfunction]
pub fn compute_internal_product_py(
    sup: u8,
    sdown: u8,
    fij: [f64; 36],
    n_sites: usize,
) -> PyResult<f64> {

    let state = FockState { spin_up: sup, spin_down: sdown, n_sites };
    let mut pfaffian_state = construct_matrix_a_from_state(&fij, state);
    let pfaffian = pfaffian_state.pfaff;
    pfaffian_state.rebuild_matrix();
    Ok(
        pfaffian * (1..=(sup.count_ones() + sdown.count_ones())/2).product::<u32>() as f64,
        )
}
