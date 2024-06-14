use rand::Rng;
use std::ptr::addr_of;
use log::{debug, info};

use impurity::{FockState, RandomStateGeneration, VarParams};
use impurity::{FIJ, GI, VIJ, SIZE};
use impurity::density::compute_internal_product;
use impurity::hamiltonian::{potential, kinetic};

const NELEC: usize = 4;
const NMCSAMP: usize = 100_000;
const NMCWARMUP: usize = 1000;

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
    let e = (kin / <f64>::exp(ip)) + potential(state);
    debug!("Inside compute hamiltonian State: {}", state);
    debug!("Inside compute hamiltonian Energy: {}", e);
    e
}

fn main() {
    // Initialize logger
    env_logger::init();

    let mut rng = rand::thread_rng();
    //let parameters = generate_random_params(&mut rng);
    let parameters = unsafe { VarParams {
        fij: addr_of!(FIJ) as *const f64,
        gi: addr_of!(GI) as *const f64,
        vij: addr_of!(VIJ) as *const f64
    }};

    let mut state: FockState<u8> = {
        let mut tmp: FockState<u8> = FockState::generate_from_nelec(&mut rng, NELEC, SIZE);
        while tmp.spin_up.count_ones() != tmp.spin_down.count_ones() {
            tmp = FockState::generate_from_nelec(&mut rng, NELEC, SIZE);
        }
        tmp
    };

    info!("Initial State: {}", state);
    info!("Initial Nelec: {}, {}", state.spin_down.count_ones(), state.spin_up.count_ones());
    info!("Nsites: {}", state.n_sites);

    let mut lip = compute_internal_product(state, &parameters);
    let mut energy: f64 = 0.0;

    info!("Starting the warmup phase.");
    // Warmup
    for _ in 0..NMCWARMUP {
        let (lip2, state2) = propose_hopping(&state, &mut rng, &parameters);
        debug!("Current state: {}", state);
        debug!("Proposed state: {}", state2);
        let ratio = <f64>::exp(lip2 - lip);
        debug!("Ratio: {}", ratio);
        let w = rng.gen::<f64>();
        if ratio >= w {
            // We ACCEPT
            debug!("Accept.");
            state = state2;
            lip = lip2;
        }
    }

    info!("Starting the sampling phase.");
    // MC Sampling
    for _ in 0..NMCSAMP {
        let (lip2, state2) = propose_hopping(&state, &mut rng, &parameters);
        debug!("Current state: {}", state);
        debug!("Proposed state: {}", state2);
        let ratio = <f64>::exp(lip2 - lip);
        debug!("Ratio: {}", ratio);
        let w = rng.gen::<f64>();
        if ratio >= w {
            // We ACCEPT
            debug!("Accept.");
            state = state2;
            lip = lip2;
        }
        energy += compute_hamiltonian(state, lip, &parameters);
    }
    energy = energy / NMCSAMP as f64;
    info!("Final Energy: {}", energy);
}
