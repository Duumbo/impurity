use impurity::pfaffian::{update_pstate, PfaffianState};
use rand::Rng;
use std::ptr::addr_of;
use log::{debug, info, trace, warn};

use impurity::{FockState, RandomStateGeneration, VarParams, Spin};
use impurity::{FIJ, GI, VIJ, SIZE, CONS_U, CONS_T};
use impurity::density::{compute_internal_product_parts, fast_internal_product};
use impurity::hamiltonian::{potential, kinetic};

const NELEC: usize = SIZE;
const NMCSAMP: usize = 100_000;
const NMCWARMUP: usize = 10_000;
const CLEAN_UPDATE_FREQUENCY: usize = 0;
const TOLERENCE_SHERMAN_MORRISSON: f64 = 1e-12;
const TOLERENCE_SINGULARITY: f64 = 1e-12;

fn propose_hopping<R: Rng + ?Sized>(
    state: &FockState<u8>,
    pfaff_state: &PfaffianState,
    previous_proj: &mut f64,
    hop: &mut (usize, usize, Spin),
    rng: &mut R,
    params: &VarParams
) -> (f64, FockState<u8>, Vec<f64>, usize) {
    let state2 = state.generate_hopping(rng, SIZE as u32, hop);
    let (ratio_ip, updated_column, col_idx) = {
        fast_internal_product(state, &state2, pfaff_state, &hop, previous_proj, params)
    };
    (ratio_ip, state2, updated_column, col_idx)
}

fn compute_hamiltonian(state: FockState<u8>, pstate: &PfaffianState, proj: f64, params: &VarParams) -> f64 {
    let kin = kinetic(state, pstate, proj, params);
    let e = kin + potential(state);
    trace!("Hamiltonian application <x|H|psi> = {} for state: |x> = {}", e, state);
    e
}

fn log_system_parameters() {
    info!("System parameter SIZE = {}", SIZE);
    info!("System parameter NELEC = {}", NELEC);
    info!("System parameter NMCSAMP = {}", NMCSAMP);
    info!("System parameter NMCWARMUP = {}", NMCWARMUP);
    info!("System parameter CONS_U = {}", CONS_U);
    info!("System parameter CONS_T = {}", CONS_T);
    debug!("System parameter CLEAN_UPDATE_FREQUENCY = {}", CLEAN_UPDATE_FREQUENCY);
    debug!("System parameter TOLERENCE_SHERMAN_MORRISSON = {}", TOLERENCE_SHERMAN_MORRISSON);
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

fn get_sign(s1: &FockState<u8>, hop: &(usize, usize, Spin)) -> usize {
    let (a, b) = if hop.0 < hop.1 {
        (hop.1, hop.0)
    } else {
        (hop.0, hop.1)
    };
    trace!("First mask: {:08b}, second mask: {:08b}", !(<u8>::MAX >> a), <u8>::MAX >> (b + 1));
    let mask = {
        !(<u8>::MAX >> a) & (<u8>::MAX >> (b + 1))
    };
    let n_ones = match hop.2 {
        Spin::Up => {
            (s1.spin_up & mask).count_ones()
        },
        Spin::Down => {
            (s1.spin_down & mask).count_ones()
        }
    };

    n_ones as usize
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

    let (mut pstate, mut proj) = compute_internal_product_parts(state, &parameters);
    let mut hop: (usize, usize, Spin) = (0, 0, Spin::Up);
    let mut _lip = <f64>::ln(<f64>::abs(<f64>::exp(proj) * pstate.pfaff)) * 2.0;
    let mut n_accepted_updates: usize = 0;
    let mut energy: f64 = 0.0;
    let mut proj_copy_persistent = proj;
    let mut sign: usize = 0;

    info!("Starting the warmup phase.");
    // Warmup
    for _ in 0..NMCWARMUP {
        let mut proj_copy = proj;
        let (ratio, state2, col, colidx) = propose_hopping(&state, &pstate, &mut proj_copy, &mut hop, &mut rng, &parameters);
        trace!("Current state: {}", state);
        trace!("Proposed state: {}", state2);
        trace!("Ratio: {}", ratio);
        let w = rng.gen::<f64>();
        if <f64>::abs(ratio) * <f64>::abs(ratio) >= w {
            // We ACCEPT
            trace!("Accept.");
            n_accepted_updates += 1;

            // Clean update once in a while
            if n_accepted_updates < CLEAN_UPDATE_FREQUENCY {
                state = state2;
                proj = proj_copy;
                update_pstate(&mut pstate, hop, col, colidx);
            } else {
                let tmp_pfaff = pstate.pfaff;
                trace!("PfaffianState before clean update: {:?}", pstate);
                sign = (sign + get_sign(&state, &hop)) % 2;
                trace!("Number of electrons between hopping: {}", get_sign(&state, &hop));
                (pstate, proj) = compute_internal_product_parts(state2, &parameters);
                let inverse_proj = <f64>::exp(proj_copy_persistent - proj);
                if sign != 0 {
                    pstate.pfaff *= -1.0;
                }
                trace!("Number of electrons between hopping: {}", get_sign(&state, &hop));
                (pstate, proj) = compute_internal_product_parts(state2, &parameters);
                if pstate.pfaff*pstate.pfaff < TOLERENCE_SINGULARITY {
                    warn!("Updated matrix is probably singular, got pfaffian {:.2e} and Tolerence is : {:e}.", pstate.pfaff, TOLERENCE_SINGULARITY);
                }
                trace!("PfaffianState after clean update: {:?}", pstate);
                let err = <f64>::abs(tmp_pfaff * ratio * inverse_proj) - <f64>::abs(pstate.pfaff);
                if err >= TOLERENCE_SHERMAN_MORRISSON {
                    warn!("Sherman-Morrisson update error of {:.2e} on computed pfaffian. Tolerence is : {:e}. Ratio was {}, states were: {} -> {}", err, TOLERENCE_SHERMAN_MORRISSON, ratio, state, state2);
                }
                n_accepted_updates = 0;
                state = state2;
                proj_copy_persistent = proj;
            }
        }
    }

    info!("Starting the sampling phase.");
    // MC Sampling
    for _ in 0..NMCSAMP {
        let mut proj_copy = proj;
        let (ratio, state2, col, colidx) = propose_hopping(&state, &pstate, &mut proj_copy, &mut hop, &mut rng, &parameters);
        trace!("Current state: {}", state);
        trace!("Proposed state: {}", state2);
        trace!("Ratio: {}", ratio);
        let w = rng.gen::<f64>();
        if <f64>::abs(ratio) * <f64>::abs(ratio) >= w {
            // We ACCEPT
            trace!("Accept.");
            n_accepted_updates += 1;
            // Clean update once in a while
            // written in this way for branch prediction.
            if n_accepted_updates < CLEAN_UPDATE_FREQUENCY {
                state = state2;
                proj = proj_copy;
                update_pstate(&mut pstate, hop, col, colidx);
            } else {
                let tmp_pfaff = pstate.pfaff;
                sign = (sign + get_sign(&state, &hop)) % 2;
                trace!("Number of electrons between hopping: {}", get_sign(&state, &hop));
                (pstate, proj) = compute_internal_product_parts(state2, &parameters);
                let inverse_proj = <f64>::exp(proj_copy_persistent - proj);
                if sign != 0 {
                    pstate.pfaff *= -1.0;
                }
                trace!("Number of electrons between hopping: {}", get_sign(&state, &hop));
                (pstate, proj) = compute_internal_product_parts(state2, &parameters);
                let err = <f64>::abs(tmp_pfaff * ratio * inverse_proj) - <f64>::abs(pstate.pfaff);
                if err >= TOLERENCE_SHERMAN_MORRISSON {
                    warn!("Sherman-Morrisson update error of {:.2e} on computed pfaffian. Tolerence is : {:e}. Ratio was {}", err, TOLERENCE_SHERMAN_MORRISSON, ratio);
                }
                n_accepted_updates = 0;
                state = state2;
                proj_copy_persistent = proj;
            }
        }
        energy += compute_hamiltonian(state, &pstate, proj, &parameters);
    }
    info!("Final Energy: {:.2}", energy);
    energy = energy / NMCSAMP as f64;
    info!("Final Energy normalized: {:.2}", energy);
}
