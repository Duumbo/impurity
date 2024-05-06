use rand::Rng;
use std::path::Path;

use impurity::gutzwiller::compute_gutzwiller_exp;
use impurity::hamiltonian::{kinetic, potential};
use impurity::jastrow::compute_jastrow_exp;
use impurity::parse::orbitale::parse_orbitale_def;
use impurity::pfaffian::construct_matrix_a_from_state;
use impurity::{FockState, CONS_T, RandomStateGeneration};

const NELEC: usize = 6;
const SIZE: usize = 8;

fn compute_internal_product(
    state: FockState<u8>,
    fij: Vec<f64>,
    vij: Vec<f64>,
    gi: Vec<f64>,
) -> f64 {
    let mut pfaffian_state = construct_matrix_a_from_state(&fij, state);
    let pfaffian = pfaffian_state.pfaff;
    pfaffian_state.rebuild_matrix();
    let jastrow_exp = compute_jastrow_exp(state, &vij, 8);
    let gutz_exp = compute_gutzwiller_exp(state, &gi, 8);
    let scalar_prod = <f64>::exp(jastrow_exp + gutz_exp) * pfaffian;
    scalar_prod
}

fn main() {
    let mut rng = rand::thread_rng();
    let orbitale_fp = Path::new("data/orbitale.csv");
    let fij = parse_orbitale_def(&orbitale_fp.to_path_buf(), SIZE).unwrap();
    let mut vij: Vec<f64> = Vec::with_capacity(SIZE * SIZE);
    for _ in 0..SIZE * SIZE {
        vij.push(rng.gen())
    }
    let mut gi: Vec<f64> = Vec::with_capacity(SIZE);
    for _ in 0..SIZE {
        gi.push(rng.gen())
    }
    let state = FockState::generate_from_nelec(&mut rng, NELEC, SIZE);
    let internal_product = compute_internal_product(state, fij.clone(), vij.clone(), gi.clone());
    let rho_x = internal_product * internal_product;
    let pot = potential(state.spin_up, state.spin_down);
    let cin = kinetic(state.spin_up, state.spin_down, state.n_sites);
    println!("Terme cin to compute: {:?}", cin);
    println!("Terme pot to compute: {:?} * the internal product", pot);
    println!("rho(x): {}", rho_x);
    println!("Computing the Hamiltonian terms.");
    let mut energie = rho_x * pot * internal_product;
    for s in cin.into_iter() {
        energie +=
            rho_x * compute_internal_product(s, fij.clone(), vij.clone(), gi.clone()) * CONS_T;
    }
    println!("Énergie: {}", energie);
}
