use rand::Rng;
use std::ptr::addr_of;
use log::{debug, info, trace};

use impurity::{FockState, RandomStateGeneration, VarParams, Spin};
use impurity::{FIJ, GI, VIJ, SIZE, CONS_U, CONS_T};
use impurity::density::compute_internal_product;
use impurity::hamiltonian::potential;

const NELEC: usize = 4;
const NMCSAMP: usize = 10_000;
const NMCWARMUP: usize = 100;

fn propose_hopping<R: Rng + ?Sized>(state: &FockState<u8>, rng: &mut R, params: &VarParams) -> (f64, FockState<u8>) {
    let n0: usize = 0;
    let n1: usize = 0;
    let n2: Spin = Spin::Up;
    let state2 = state.generate_hopping(rng, SIZE as u32, &mut (n0, n1, n2));
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

fn compute_hamiltonian(state: FockState<u8>, _ip: f64, _params: &VarParams) -> f64 {
    //let kin = kinetic(state, params);
    //let e = (kin / <f64>::exp(ip)) + potential(state);
    let e = potential(state);
    trace!("Inside compute hamiltonian Energy: {} for state: {}", e, state);
    e
}

fn log_system_parameters() {
    info!("System parameter SIZE = {}", SIZE);
    info!("System parameter NELEC = {}", NELEC);
    info!("System parameter NMCSAMP = {}", NMCSAMP);
    info!("System parameter NMCWARMUP = {}", NMCWARMUP);
    info!("System parameter CONS_U = {}", CONS_U);
    info!("System parameter CONS_T = {}", CONS_T);
    for i in 0..4*SIZE*SIZE {
        unsafe {
            if FIJ[i] == 0.0 {continue;}
            if i < SIZE*SIZE {
                debug!("F_[{},{}]^[up, up]={}", i/SIZE, i%SIZE, FIJ[i]);
            }
            else if i < 2*SIZE*SIZE {
                debug!("F_[{},{}]^[up, down]={}", i/SIZE - SIZE, i%SIZE, FIJ[i]);
            }
            else if i < 3*SIZE*SIZE {
                debug!("F_[{},{}]^[down, up]={}", i/SIZE - 2*SIZE, i%SIZE, FIJ[i]);
            }
            else {
                debug!("F_[{},{}]^[down, down]={}", i/SIZE - 3*SIZE, i%SIZE, FIJ[i]);
            }
        }
    }
    for i in 0..SIZE*SIZE {
        unsafe {
            if VIJ[i] == 0.0 {continue;}
            debug!("V_[{},{}]={}", i/SIZE, i%SIZE, VIJ[i]);
        }
    }
    for i in 0..SIZE {
        unsafe {
            if GI[i] == 0.0 {continue;}
            debug!("G_[{}]={}", i, GI[i]);
        }
    }
}

fn main() {
    // Initialize logger
    env_logger::init();
    log_system_parameters();

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
        trace!("Current state: {}", state);
        trace!("Proposed state: {}", state2);
        let ratio = <f64>::exp(lip2 - lip);
        trace!("Ratio: {}", ratio);
        let w = rng.gen::<f64>();
        if ratio >= w {
            // We ACCEPT
            trace!("Accept.");
            state = state2;
            lip = lip2;
        }
    }

    info!("Starting the sampling phase.");
    // MC Sampling
    for _ in 0..NMCSAMP {
        let (lip2, state2) = propose_hopping(&state, &mut rng, &parameters);
        trace!("Current state: {}", state);
        trace!("Proposed state: {}", state2);
        let ratio = <f64>::exp(lip2 - lip);
        trace!("Ratio: {}", ratio);
        let w = rng.gen::<f64>();
        if ratio >= w {
            // We ACCEPT
            trace!("Accept.");
            state = state2;
            lip = lip2;
        }
        energy += compute_hamiltonian(state, lip, &parameters);
    }
    energy = energy / NMCSAMP as f64;
    info!("Final Energy: {}", energy);
}
