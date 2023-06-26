use std::path::Path;

use impurity::parse::orbitale::parse_orbitale_def;
use impurity::pfaffian::{compute_pfaffian_wq, construct_matrix_a_from_state};
use impurity::{SIZE, FockState};

fn main() {
    let orbitale_fp = Path::new("data/orbitale.csv");
    let fij = parse_orbitale_def(&orbitale_fp.to_path_buf(), SIZE).unwrap();
    let state = FockState { spin_up: 3, spin_down: 4 };
    let mut pfaffian_state = construct_matrix_a_from_state(fij, state);
    println!("f matrix: {:?}", pfaffian_state.coeffs);
    println!("Pfaffian matrix: {:?}", pfaffian_state.matrix);
    let pfaffian = compute_pfaffian_wq(&mut pfaffian_state.matrix, pfaffian_state.n_elec as i32);
    pfaffian_state.rebuild_matrix();
    println!("Pfaffian matrix: {:?}", pfaffian_state.matrix);
    println!("{}", pfaffian);
}
