use rand::Rng;
use std::ptr::addr_of;
use log::{debug, info, trace, warn};
use assert::close;

use impurity::{FockState, RandomStateGeneration, VarParams, Spin};
use impurity::pfaffian::{update_pstate, PfaffianState};
use impurity::density::{compute_internal_product_parts, fast_internal_product};
use impurity::hamiltonian::{potential, kinetic};

/// Size of the system.
const SIZE: usize = 4;
/// Hubbard's model $U$ parameter
static CONS_U: f64 = 1.0;
/// Hubbard's model $t$ parameter
static CONS_T: f64 = -1.0;

const NFIJ: usize = 4*SIZE*SIZE;
const NVIJ: usize = SIZE*SIZE;
const NGI: usize = SIZE;

static mut FIJ: [f64; NFIJ] = [
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
1.078313550425773	,
0.007172274681240365,
0.028714778076311877,
0.09168843535310542	,
0.04813118562079141	,
1.0625398526882723	,
0.08433353658389342	,
0.002722470871706029,
0.07270002762085896	,
0.026989164590497917,
0.007555596176108393,
0.046284058565227465,
0.011127921360085048,
0.07287939415825727	,
0.08138828369394709	,
0.012799567556772274,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
];

static mut VIJ: [f64; NVIJ] = [
    0.0, 2.0, 2.0, 2.0,
    2.0, 0.0, 2.0, 2.0,
    2.0, 2.0, 0.0, 2.0,
    2.0, 2.0, 2.0, 0.0,
];

static mut GI: [f64; NGI] = [
    -1.0, -1.0, -1.0, -1.0
];

const NELEC: usize = 4;
const NMCSAMP: usize = 100_000;
const NMCWARMUP: usize = 1000;
const CLEAN_UPDATE_FREQUENCY: usize = 0;
const TOLERENCE_SHERMAN_MORRISSON: f64 = 1e-15;
const MONTE_CARLO_CONVERGENCE_TOLERANCE: f64 = 1e-1;

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

#[test]
fn monte_carlo_first_iteration() {
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
                let inverse_proj = <f64>::exp(proj - proj_copy);
                trace!("PfaffianState before clean update: {:?}", pstate);
                trace!("Updated pfaffian got: {:.2e}", pstate.pfaff * ratio * inverse_proj);
                sign = (sign + get_sign(&state, &hop)) % 2;
                trace!("Number of electrons between hopping: {}", get_sign(&state, &hop));
                (pstate, proj) = compute_internal_product_parts(state2, &parameters);
                if sign != 0 {
                    pstate.pfaff *= -1.0;
                }
                trace!("PfaffianState after clean update: {:?}", pstate);
                let err = <f64>::abs((tmp_pfaff * ratio * inverse_proj) - pstate.pfaff);
                if err >= TOLERENCE_SHERMAN_MORRISSON {
                    warn!("Sherman-Morrisson update error of {:.2e} on computed pfaffian. Tolerence is : {:e}. Ratio was {}, states were: {} -> {}", err, TOLERENCE_SHERMAN_MORRISSON, ratio, state, state2);
                }
                n_accepted_updates = 0;
                state = state2;
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
            // Clean update once in a while
            // written in this way for branch prediction.
            if n_accepted_updates < CLEAN_UPDATE_FREQUENCY {
                state = state2;
                proj = proj_copy;
                update_pstate(&mut pstate, hop, col, colidx);
            } else {
                let tmp_pfaff = pstate.pfaff;
                let inverse_proj = <f64>::exp(proj - proj_copy);
                sign = (sign + get_sign(&state, &hop)) % 2;
                trace!("Number of electrons between hopping: {}", get_sign(&state, &hop));
                (pstate, proj) = compute_internal_product_parts(state2, &parameters);
                if sign != 0 {
                    pstate.pfaff *= -1.0;
                }
                let err = <f64>::abs((tmp_pfaff * ratio * inverse_proj) - pstate.pfaff);
                if err >= TOLERENCE_SHERMAN_MORRISSON {
                    warn!("Sherman-Morrisson update error of {:.2e} on computed pfaffian. Tolerence is : {:e}. Ratio was {}", err, TOLERENCE_SHERMAN_MORRISSON, ratio);
                }
                n_accepted_updates = 0;
                state = state2;
            }
        }
        energy += compute_hamiltonian(state, &pstate, proj, &parameters);
    }
    info!("Final Energy: {:.2}", energy);
    energy = energy / NMCSAMP as f64;
    info!("Final Energy normalized: {:.2}", energy);
    close(energy, -0.35, MONTE_CARLO_CONVERGENCE_TOLERANCE);
}
