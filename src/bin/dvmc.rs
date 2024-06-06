use rand::Rng;

use impurity::gutzwiller::compute_gutzwiller_exp;
use impurity::jastrow::compute_jastrow_exp;
use impurity::pfaffian::construct_matrix_a_from_state;
use impurity::{FockState, RandomStateGeneration};
use impurity::hamiltonian::{potential, kinetic};

const NELEC: usize = 6;
const SIZE: usize = 8;
const NMCSAMP: usize = 100000;

struct VarParams {
    fij: Vec<f64>,
    vij: Vec<f64>,
    gi: Vec<f64>
}

fn compute_internal_product(
    state: FockState<u8>,
    params: &VarParams
) -> f64 {
    let mut pfaffian_state = construct_matrix_a_from_state(&params.fij, state);
    let pfaffian = pfaffian_state.pfaff;
    pfaffian_state.rebuild_matrix();
    let jastrow_exp = compute_jastrow_exp(state, &params.vij, 8);
    let gutz_exp = compute_gutzwiller_exp(state, &params.gi, 8);
    let scalar_prod = <f64>::exp(jastrow_exp + gutz_exp) * pfaffian;
    <f64>::ln(scalar_prod * scalar_prod)
}

fn propose_hopping<R: Rng + ?Sized>(state: &FockState<u8>, rng: &mut R, params: &VarParams) -> (f64, FockState<u8>) {
    let state2 = state.generate_hopping(rng, SIZE as u32);
    let ip2 = compute_internal_product(state2, params);
    (ip2, state2)
}

fn generate_random_params<R: Rng + ?Sized>(rng: &mut R) -> VarParams {
    let mut fij: Vec<f64> = Vec::with_capacity(4*SIZE*SIZE);
    let mut vij: Vec<f64> = Vec::with_capacity(SIZE * SIZE);
    for _ in 0..SIZE * SIZE {
        vij.push(rng.gen())
    }
    for _ in 0..4*SIZE * SIZE {
        fij.push(rng.gen())
    }
    let mut gi: Vec<f64> = Vec::with_capacity(SIZE);
    for _ in 0..SIZE {
        gi.push(rng.gen())
    }
    VarParams{fij, vij, gi}
}

fn compute_hamiltonian(state: FockState<u8>) -> f64 {
    potential(state)
}

fn main() {
    let mut rng = rand::thread_rng();
    let parameters = generate_random_params(&mut rng);
    let mut state: FockState<u8> = FockState::generate_from_nelec(&mut rng, NELEC, SIZE);
    //println!("State: {:?}", state);
    //println!("Nelec: {}, {}", state.spin_down.count_ones(), state.spin_up.count_ones());
    let lip = compute_internal_product(state, &parameters);
    let mut energy: f64 = compute_hamiltonian(state);
    println!("{}", energy);
    for _ in 0..NMCSAMP {
        let (lip2, state2) = propose_hopping(&state, &mut rng, &parameters);
        let ratio = <f64>::exp(lip2 - lip);
        //println!("Ratio: {}", ratio);
        let w = rng.gen::<f64>();
        if ratio >= w {
            // We ACCEPT
            //println!("Accept.");
            state = state2;
        }
        energy += compute_hamiltonian(state);
        println!("{}", energy);
    }
    energy = energy / NMCSAMP as f64;
    println!("Potential Energy: {}", energy);
}
