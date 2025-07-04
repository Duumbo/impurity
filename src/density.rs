use log::trace;
#[cfg(feature = "python-interface")]
use pyo3::{pyfunction, PyResult};

use crate::jastrow::{compute_jastrow_exp, fast_update_jastrow};
use crate::gutzwiller::{compute_gutzwiller_exp, fast_update_gutzwiller};
use crate::pfaffian::{construct_matrix_a_from_state, get_pfaffian_ratio, PfaffianState};
use crate::{BitOps, FockState, Spin, SpinState, VarParams, SysParams};

pub fn compute_internal_product<T>(
    state: FockState<T>,
    params: &VarParams,
    sys: &SysParams,
) -> f64
where T: BitOps + std::fmt::Debug + std::fmt::Display + From<u8> + std::ops::Shl<usize, Output = T> + Send
{
    let (mut pfaffian_state, jastrow_exp, gutz_exp) = {
        (
            construct_matrix_a_from_state(params.fij, state, sys),
            compute_jastrow_exp(state, params.vij, state.n_sites),
            compute_gutzwiller_exp(state, params.gi, state.n_sites)
        )
    };
    let pfaffian = pfaffian_state.pfaff;
    pfaffian_state.rebuild_matrix();
    let scalar_prod = <f64>::abs(<f64>::exp(jastrow_exp + gutz_exp) * pfaffian);
    trace!("Projector value: {}, for state: {}", jastrow_exp + gutz_exp, state);
    trace!("Computed <x|psi>: {}, for state: {}", scalar_prod, state);
    <f64>::ln(scalar_prod) * 2.0
}

pub fn compute_internal_product_parts<T>(
    state: FockState<T>,
    params: &VarParams,
    sys: &SysParams,
) -> (PfaffianState, f64)
where T: BitOps + std::fmt::Debug + std::fmt::Display + From<u8> + std::ops::Shl<usize, Output = T> + Send
{
    let (mut pfaffian_state, jastrow_exp, gutz_exp) = {
        (
            construct_matrix_a_from_state(params.fij, state, sys),
            compute_jastrow_exp(state, params.vij, state.n_sites),
            compute_gutzwiller_exp(state, params.gi, state.n_sites)
        )
    };
    pfaffian_state.rebuild_matrix();
    (pfaffian_state, jastrow_exp + gutz_exp)
}

pub fn fast_internal_product_no_otilde<T>(
    previous_state: &FockState<T>,
    new_state: &FockState<T>,
    previous_pstate: &PfaffianState,
    hopping: &(usize, usize, Spin),
    previous_proj: &mut f64,
    params: &VarParams,
) -> (f64, Vec<f64>, usize)
where T: BitOps + std::fmt::Debug + std::fmt::Display + From<SpinState> + std::ops::Shl<usize, Output = T> + Send
{
    // Rename things.
    let previous_i = hopping.0;
    let new_i = hopping.1;
    let spin = hopping.2;
    let n_sites = new_state.n_sites;

    let (pfaffian_ratio, b_vec, col) = {
        let fij = &params.fij;
        let vij = &params.vij;
        let gi = &params.gi;
        fast_update_jastrow(previous_proj, vij, previous_state, new_state, n_sites, previous_i, new_i);
        match spin {
            Spin::Up => {
        fast_update_gutzwiller(previous_proj, gi, &previous_state.spin_down, previous_i, new_i);
            },
            Spin::Down => {
        fast_update_gutzwiller(previous_proj, gi, &previous_state.spin_up, previous_i, new_i);
            }

        }
        get_pfaffian_ratio(previous_pstate, previous_i, new_i, spin, fij)
    };

    // Combine to get the internal product.
    trace!("Fast Projector value: {}, for state: {}", previous_proj, new_state);
    trace!("Fast Computed <x'|pf>/<x|pf>: {}, |x'> = {}, |x> = {}", pfaffian_ratio, new_state, previous_state);
    (pfaffian_ratio, b_vec, col)
}

pub fn fast_internal_product<T>(
    previous_state: &FockState<T>,
    new_state: &FockState<T>,
    previous_pstate: &PfaffianState,
    hopping: &(usize, usize, Spin),
    previous_proj: &mut f64,
    params: &VarParams,
) -> (f64, Vec<f64>, usize)
where T: BitOps + std::fmt::Debug + std::fmt::Display + From<SpinState> + std::ops::Shl<usize, Output = T> + Send
{
    // Rename things.
    let previous_i = hopping.0;
    let new_i = hopping.1;
    let spin = hopping.2;
    let n_sites = new_state.n_sites;

    let (pfaffian_ratio, b_vec, col) = {
        let fij = &params.fij;
        let vij = &params.vij;
        let gi = &params.gi;
        fast_update_jastrow(previous_proj, vij, previous_state, new_state, n_sites, previous_i, new_i);
        match spin {
            Spin::Up => {
        fast_update_gutzwiller(previous_proj, gi, &previous_state.spin_down, previous_i, new_i);
            },
            Spin::Down => {
        fast_update_gutzwiller(previous_proj, gi, &previous_state.spin_up, previous_i, new_i);
            }

        }
        get_pfaffian_ratio(previous_pstate, previous_i, new_i, spin, fij)
    };

    // Combine to get the internal product.
    trace!("Fast Projector value: {}, for state: {}", previous_proj, new_state);
    trace!("Fast Computed <x'|pf>/<x|pf>: {}, |x'> = {}, |x> = {}", pfaffian_ratio, new_state, previous_state);
    (pfaffian_ratio, b_vec, col)
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
