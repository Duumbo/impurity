#[cfg(feature = "python-interface")]
use pyo3::{pyfunction, PyResult};

use crate::jastrow::compute_jastrow_exp;
use crate::gutzwiller::compute_gutzwiller_exp;
use crate::pfaffian::construct_matrix_a_from_state;
use crate::{FockState, VarParams, BitOps};

pub fn compute_internal_product<T>(
    state: FockState<T>,
    params: &VarParams
) -> f64
where T: BitOps + std::fmt::Debug + std::fmt::Display + From<u8> + std::ops::Shl<usize, Output = T>
{
    let mut pfaffian_state = construct_matrix_a_from_state(&params.fij, state);
    let pfaffian = pfaffian_state.pfaff;
    pfaffian_state.rebuild_matrix();
    let jastrow_exp = compute_jastrow_exp(state, &params.vij, state.n_sites);
    let gutz_exp = compute_gutzwiller_exp(state, &params.gi, state.n_sites);
    let scalar_prod = <f64>::abs(<f64>::exp(jastrow_exp + gutz_exp) * pfaffian);
    <f64>::ln(scalar_prod) * 2.0
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
