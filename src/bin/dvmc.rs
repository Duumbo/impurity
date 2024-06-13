use rand::Rng;
use std::ptr::addr_of;

use impurity::{FockState, RandomStateGeneration, VarParams};
use impurity::{FIJ, GI, VIJ, SIZE};
use impurity::density::compute_internal_product;
use impurity::hamiltonian::{potential, kinetic};
use impurity::HOP_BITMASKS;

const NELEC: usize = 8;
const NMCSAMP: usize = 1000;


fn propose_hopping<R: Rng + ?Sized>(state: &FockState<u8>, rng: &mut R, params: &VarParams) -> (f64, FockState<u8>) {
    let state2 = state.generate_hopping(rng, SIZE as u32);
    let ip2 = compute_internal_product(state2, params);
    (ip2, state2)
}

//fn generate_random_params<R: Rng + ?Sized>(rng: &mut R) -> VarParams {
//    let mut fij: Vec<f64> = Vec::with_capacity(4*SIZE*SIZE);
//    let mut vij: Vec<f64> = Vec::with_capacity(SIZE * SIZE);
//    for _ in 0..SIZE * SIZE {
//        vij.push(rng.gen())
//    }
//    for _ in 0..4*SIZE * SIZE {
//        fij.push(rng.gen())
//    }
//    let mut gi: Vec<f64> = Vec::with_capacity(SIZE);
//    for _ in 0..SIZE {
//        gi.push(rng.gen())
//    }
//    VarParams{fij, vij, gi}
//}

fn compute_hamiltonian(state: FockState<u8>, ip: f64, params: &VarParams) -> f64 {
    let kin = kinetic(state, params);
    (kin / <f64>::exp(ip)) + potential(state)
}

fn main() {
    let mut rng = rand::thread_rng();
    //let parameters = generate_random_params(&mut rng);
    let parameters = unsafe { VarParams {
        fij: addr_of!(FIJ) as *const f64,
        gi: addr_of!(GI) as *const f64,
        vij: addr_of!(VIJ) as *const f64
    }};

    let mut state: FockState<u8> = FockState::generate_from_nelec(&mut rng, NELEC, SIZE);
    println!("State: {}", state);
    println!("Nelec: {}, {}", state.spin_down.count_ones(), state.spin_up.count_ones());
    println!("Nsites: {}", state.n_sites);
    let mut lip = compute_internal_product(state, &parameters);
    let mut energy: f64 = compute_hamiltonian(state, lip, &parameters);
    for _ in 0..NMCSAMP {
        let (lip2, state2) = propose_hopping(&state, &mut rng, &parameters);
        println!("Current state: {}", state);
        println!("Proposed state: {}", state2);
        let ratio = <f64>::exp(lip2 - lip);
        println!("Ratio: {}", ratio);
        let w = rng.gen::<f64>();
        if ratio >= w {
            // We ACCEPT
            println!("Accept.");
            state = state2;
            lip = lip2;
        }
        //println!("State, up: {:b}, down: {:b}, energy: {}", state.spin_up, state.spin_down, compute_hamiltonian(state, lip, &parameters));
        energy += compute_hamiltonian(state, lip, &parameters);
    }
    energy = energy / NMCSAMP as f64;
    println!("Energy: {}", energy);
    println!("Hop bitmasks: {:08b}, {:08b}", HOP_BITMASKS[0], HOP_BITMASKS[1]);
}
