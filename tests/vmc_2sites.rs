use assert::close;
use impurity::monte_carlo::compute_mean_energy;
use rand::prelude::*;
use impurity::gutzwiller::compute_gutzwiller_exp;
use impurity::jastrow::compute_jastrow_exp;
use impurity::pfaffian::construct_matrix_a_from_state;
use impurity::{FockState, VarParams, SysParams, generate_bitmask, RandomStateGeneration, DerivativeOperator};
use impurity::hamiltonian::{kinetic, potential};

// Number of sites
const SIZE: usize = 2;
// Hubbard's model $U$ parameter
const CONS_U: f64 = 4.0;
// Hubbard's model $t$ parameter
const CONS_T: f64 = -1.0;
// Number of electrons
const NELEC: usize = 2;
const NMCSAMP: usize = 10000;
const NMCWARMUP: usize = 1000;
const CLEAN_UPDATE_FREQUENCY: usize = 32;
const TOL_SHERMAN: f64 = 1e-12;
const TOL_SINGULARITY: f64 = 1e-12;

const NFIJ: usize = 4*SIZE*SIZE;
const NVIJ: usize = SIZE*SIZE;
const NGI: usize = SIZE;

pub const HOPPINGS: [f64; SIZE*SIZE] = [
    0.0, 1.0,
    1.0, 0.0,
];

#[derive(Debug)]
enum State {
    F3,
    F5,
    F6,
    F9,
    F10,
    F12
}

fn norm(par: &VarParams) -> f64 {
    let f00ud = par.fij[0 + 0 * SIZE + 1 * SIZE * SIZE];
    let f00du = par.fij[0 + 0 * SIZE + 2 * SIZE * SIZE];
    let f11ud = par.fij[1 + 1 * SIZE + 1 * SIZE * SIZE];
    let f11du = par.fij[1 + 1 * SIZE + 2 * SIZE * SIZE];
    let f01ud = par.fij[0 + 1 * SIZE + 1 * SIZE * SIZE];
    let f10ud = par.fij[1 + 0 * SIZE + 1 * SIZE * SIZE];
    let f01du = par.fij[0 + 1 * SIZE + 2 * SIZE * SIZE];
    let f10du = par.fij[1 + 0 * SIZE + 2 * SIZE * SIZE];
    let f01uu = par.fij[0 + 1 * SIZE + 0 * SIZE * SIZE];
    let f10uu = par.fij[1 + 0 * SIZE + 0 * SIZE * SIZE];
    let f01dd = par.fij[0 + 1 * SIZE + 3 * SIZE * SIZE];
    let f10dd = par.fij[1 + 0 * SIZE + 3 * SIZE * SIZE];
    let g0 = par.gi[0];
    let g1 = par.gi[1];
    let v = par.vij[1];
    let a = <f64>::exp(2.0 * g0 - 2.0 * v)*sq(<f64>::abs(f00ud - f00du));
    let b = <f64>::exp(2.0 * g1 - 2.0 * v)*sq(<f64>::abs(f11ud - f11du));
    let c = sq(<f64>::abs(f01uu - f10uu));
    let d = sq(<f64>::abs(f01dd - f10dd));
    let e = sq(<f64>::abs(f10ud - f01du));
    let f = sq(<f64>::abs(f01ud - f10du));
    a + b + e + f
}

fn print_ratios(par: &VarParams) {
    let f00ud = par.fij[0 + 0 * SIZE + 1 * SIZE * SIZE];
    let f00du = par.fij[0 + 0 * SIZE + 2 * SIZE * SIZE];
    let f11ud = par.fij[1 + 1 * SIZE + 1 * SIZE * SIZE];
    let f11du = par.fij[1 + 1 * SIZE + 2 * SIZE * SIZE];
    let f01ud = par.fij[1 + 0 * SIZE + 1 * SIZE * SIZE];
    let f10ud = par.fij[0 + 1 * SIZE + 1 * SIZE * SIZE];
    let f01du = par.fij[1 + 0 * SIZE + 2 * SIZE * SIZE];
    let f10du = par.fij[0 + 1 * SIZE + 2 * SIZE * SIZE];
    let g0 = par.gi[0];
    let g1 = par.gi[1];
    let v = par.vij[1];
    let _psi5 = (f11ud - f11du) * <f64>::exp(g1 - v);
    let _psi6 = f10ud - f01du;
    let _psi9 = f01ud - f10du;
    let _psi10 = (f00ud - f00du) * <f64>::exp(g0 - v);
    //let mut statesfp = File::create("ratios_th").unwrap();
    //statesfp.write(&format!("<5|psi>/<6|psi> = {}\n", sq(psi5 / psi6)).as_bytes()).unwrap();
    //statesfp.write(&format!("<5|psi>/<9|psi> = {}\n", sq(psi5 / psi9)).as_bytes()).unwrap();
    //statesfp.write(&format!("<10|psi>/<6|psi> = {}\n", sq(psi10 / psi6)).as_bytes()).unwrap();
    //statesfp.write(&format!("<10|psi>/<9|psi> = {}\n", sq(psi10 / psi9)).as_bytes()).unwrap();
    //statesfp.write(&format!("<6|psi>/<5|psi> = {}\n", sq(psi6 / psi5)).as_bytes()).unwrap();
    //statesfp.write(&format!("<6|psi>/<10|psi> = {}\n", sq(psi6 / psi10)).as_bytes()).unwrap();
    //statesfp.write(&format!("<9|psi>/<5|psi> = {}\n", sq(psi9 / psi5)).as_bytes()).unwrap();
    //statesfp.write(&format!("<9|psi>/<10|psi> = {}\n", sq(psi9 / psi10)).as_bytes()).unwrap();
}

fn print_ip(par: &VarParams) {
    let f00ud = par.fij[0 + 0 * SIZE + 1 * SIZE * SIZE];
    let f00du = par.fij[0 + 0 * SIZE + 2 * SIZE * SIZE];
    let f11ud = par.fij[1 + 1 * SIZE + 1 * SIZE * SIZE];
    let f11du = par.fij[1 + 1 * SIZE + 2 * SIZE * SIZE];
    let f01ud = par.fij[1 + 0 * SIZE + 1 * SIZE * SIZE];
    let f10ud = par.fij[0 + 1 * SIZE + 1 * SIZE * SIZE];
    let f01du = par.fij[1 + 0 * SIZE + 2 * SIZE * SIZE];
    let f10du = par.fij[0 + 1 * SIZE + 2 * SIZE * SIZE];
    let g0 = par.gi[0];
    let g1 = par.gi[1];
    let v = par.vij[1];
    let _psi5 = (f11ud - f11du) * <f64>::exp(g1 - v);
    let _psi6 = f10ud - f01du;
    let _psi9 = f01ud - f10du;
    let _psi10 = (f00ud - f00du) * <f64>::exp(g0 - v);
    //let mut statesipfp = File::create("statesip").unwrap();
    //statesipfp.write(&format!("<5|psi> = {}\n", psi5).as_bytes()).unwrap();
    //statesipfp.write(&format!("<9|psi> = {}\n", psi9).as_bytes()).unwrap();
    //statesipfp.write(&format!("<6|psi> = {}\n", psi6).as_bytes()).unwrap();
    //statesipfp.write(&format!("<10|psi> = {}\n", psi10).as_bytes()).unwrap();
}

fn energy_individual_state(state: &State, par: &VarParams) -> f64 {
    match state {
        State::F3 => {
            0.0
        },
        State::F5 => {
            let f01ud = par.fij[0 + 1 * SIZE + 1 * SIZE * SIZE];
            let f10ud = par.fij[1 + 0 * SIZE + 1 * SIZE * SIZE];
            let f01du = par.fij[0 + 1 * SIZE + 2 * SIZE * SIZE];
            let f10du = par.fij[1 + 0 * SIZE + 2 * SIZE * SIZE];
            let f11ud = par.fij[1 + 1 * SIZE + 1 * SIZE * SIZE];
            let f11du = par.fij[1 + 1 * SIZE + 2 * SIZE * SIZE];
            let g1 = par.gi[1];
            let v = par.vij[1];
            let a = CONS_T * (f01ud + f10ud - f01du - f10du);
            let b = CONS_U * (f11ud - f11du) * <f64>::exp(g1 - v);
            println!("|5> pot: {}, kin: {}", b, a);
            a + b
        },
        State::F6 => {
            let f00ud = par.fij[0 + 0 * SIZE + 1 * SIZE * SIZE];
            let f00du = par.fij[0 + 0 * SIZE + 2 * SIZE * SIZE];
            let f11ud = par.fij[1 + 1 * SIZE + 1 * SIZE * SIZE];
            let f11du = par.fij[1 + 1 * SIZE + 2 * SIZE * SIZE];
            let g0 = par.gi[0];
            let g1 = par.gi[1];
            let v = par.vij[1];
            let a = CONS_T * (f00ud - f00du) * <f64>::exp(g0 - v);
            let b = CONS_T * (f11ud - f11du) * <f64>::exp(g1 - v);
            println!("|6> pot: {}, kin: {}", 0, a + b);
            a + b
        },
        State::F9 => {
            let f00ud = par.fij[0 + 0 * SIZE + 1 * SIZE * SIZE];
            let f00du = par.fij[0 + 0 * SIZE + 2 * SIZE * SIZE];
            let f11ud = par.fij[1 + 1 * SIZE + 1 * SIZE * SIZE];
            let f11du = par.fij[1 + 1 * SIZE + 2 * SIZE * SIZE];
            let g0 = par.gi[0];
            let g1 = par.gi[1];
            let v = par.vij[1];
            let a = CONS_T * (f00ud - f00du) * <f64>::exp(g0 - v);
            let b = CONS_T * (f11ud - f11du) * <f64>::exp(g1 - v);
            println!("|9> pot: {}, kin: {}", 0, a + b);
            a + b
        },
        State::F10 => {
            let f01ud = par.fij[0 + 1 * SIZE + 1 * SIZE * SIZE];
            let f10ud = par.fij[1 + 0 * SIZE + 1 * SIZE * SIZE];
            let f01du = par.fij[0 + 1 * SIZE + 2 * SIZE * SIZE];
            let f10du = par.fij[1 + 0 * SIZE + 2 * SIZE * SIZE];
            let f00ud = par.fij[0 + 0 * SIZE + 1 * SIZE * SIZE];
            let f00du = par.fij[0 + 0 * SIZE + 2 * SIZE * SIZE];
            let g0 = par.gi[0];
            let v = par.vij[1];
            let a = CONS_T * (f01ud + f10ud - f01du - f10du);
            let b = CONS_U * (f00ud - f00du) * <f64>::exp(g0 - v);
            println!("|10> pot: {}, kin: {}", b, a);
            a + b
        },
        State::F12 => {
            0.0
        },
    }
}

fn sq(a: f64) -> f64 {
    a*a
}

fn analytic(par: &VarParams) -> f64 {
    let a = par.fij[1 + 0 * SIZE + 1 * SIZE * SIZE]
        - par.fij[0 + 1 * SIZE + 2 * SIZE * SIZE]
        + par.fij[0 + 1 * SIZE + 1 * SIZE * SIZE]
        - par.fij[1 + 0 * SIZE + 2 * SIZE * SIZE];
    let b = (par.fij[0 + 0 * SIZE + 1 * SIZE * SIZE]
        - par.fij[0 + 0 * SIZE + 2 * SIZE * SIZE])
        * <f64>::exp(par.gi[0]-par.vij[1]);
    let c = (par.fij[1 + 1 * SIZE + 1 * SIZE * SIZE]
        - par.fij[1 + 1 * SIZE + 2 * SIZE * SIZE])
        * <f64>::exp(par.gi[1]-par.vij[1]);
    let d = 2.0 * CONS_T * (b + c) * a;
    let e = sq(par.fij[1 + 1 * SIZE + 1 * SIZE * SIZE]
        - par.fij[1 + 1 * SIZE + 2 * SIZE * SIZE])
        * <f64>::exp(2.0*par.gi[1]-2.0*par.vij[1]) * CONS_U;
    let f = sq(par.fij[0 + 0 * SIZE + 1 * SIZE * SIZE]
        - par.fij[0 + 0 * SIZE + 2 * SIZE * SIZE])
        * <f64>::exp(2.0*par.gi[0]-2.0*par.vij[1]) * CONS_U;
    d + e + f
}


#[test]
fn comupte_energy_from_all_states() {
    //let mut statesfp = File::create("states").unwrap();
    //let mut energyfp = File::create("energy").unwrap();
    env_logger::init();
    let mut rng = SmallRng::seed_from_u64(42u64);
    //let mut rng = thread_rng();
    let bitmask = generate_bitmask(&HOPPINGS, SIZE);
    println!("bitmasks: {:?}", bitmask);
    let sys = SysParams {
        size: SIZE,
        nelec: NELEC,
        array_size: (SIZE + 7) / 8,
        cons_t: CONS_T,
        cons_u: CONS_U,
        nfij: NFIJ,
        nvij: NVIJ,
        ngi: NGI,
        mcsample_interval: 1,
        transfert_matrix: &HOPPINGS,
        hopping_bitmask: &bitmask,
        clean_update_frequency: CLEAN_UPDATE_FREQUENCY,
        nmcwarmup: NMCWARMUP,
        nmcsample: NMCSAMP,
        tolerance_sherman_morrison: TOL_SHERMAN,
        tolerance_singularity: TOL_SINGULARITY,
    };
    let mut fij = [
        0.0, 0.0, 0.0, 0.0,
        0.41, -0.18, 0.11, 0.84,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
    ];
    let mut vij = [0.0, 0.3, 0.3, 0.0];
    let mut gi = [-0.7, -0.5];
    println!("fij: {:?}", fij);
    println!("vij: {:?}", vij);
    println!("gi: {:?}", gi);
    let parameters = VarParams {
        size: SIZE,
        fij: &mut fij,
        gi: &mut gi,
        vij: &mut vij
    };
    print_ratios(&parameters);
    print_ip(&parameters);

    // Generate all the states for 2 sites S=0
    let states_names: [State; 4] = [State::F10, State::F9, State::F6, State::F5];
    let all_states: [FockState<u8>; 4] = [
        // \ket{10}
        FockState{spin_up: 128u8, spin_down: 128u8, n_sites: SIZE},
        // \ket{12}
        //FockState{spin_up: 192u8, spin_down: 0u8, n_sites: SIZE},
        // \ket{3}
        //FockState{spin_up: 0u8, spin_down: 192u8, n_sites: SIZE},
        // \ket{9}
        FockState{spin_up: 128u8, spin_down: 64u8, n_sites: SIZE},
        // \ket{6}
        FockState{spin_up: 64u8, spin_down: 128u8, n_sites: SIZE},
        // \ket{5}
        FockState{spin_up: 64u8, spin_down: 64u8, n_sites: SIZE},
    ];
    println!("Basis states.");
    for i in 0..4 {
        println!("State {} : {}", i, all_states[i]);
    }
    let mut mean_energy = 0.0;
    for i in 0..4 {
        let state = all_states[i];
        let pstate = construct_matrix_a_from_state(&parameters.fij, state);
        println!("X^[-1] = {}", pstate);
        println!("Pfaffian: {}", pstate.pfaff);
        let proj = compute_jastrow_exp(state, &parameters.vij, SIZE)
            +compute_gutzwiller_exp(state, &parameters.gi, SIZE);
        println!("Proj: {}", proj);
        let ip = pstate.pfaff * <f64>::exp(proj);
        println!("<{:?}|psi> = {}", states_names[i], ip);
        let pot = potential(state, proj, &pstate, &sys);
        println!("Potential energy: {}", pot);
        let kin = kinetic(state, &pstate, proj, &parameters, &sys);
        println!("Kinetic energy: {}", kin);
        energy_individual_state(&states_names[i], &parameters);
        mean_energy += (kin + pot) * pstate.pfaff * <f64>::exp(proj);
        println!("Energy: {}", (kin + pot) / norm(&parameters));
        close( kin + pot,
        energy_individual_state(&states_names[i], &parameters), 1e-12);
    }
    close(mean_energy, analytic(&parameters), 1e-12);
    let mut otilde: Vec<f64> = vec![0.0; (NFIJ + NVIJ + NGI) * NMCSAMP];
    let mut expvalo: Vec<f64> = vec![0.0; NFIJ + NVIJ + NGI];
    let mut expval_ho: Vec<f64> = vec![0.0; NFIJ + NVIJ + NGI];
    let mut visited: Vec<usize> = vec![0; NMCSAMP];
    let mut der = DerivativeOperator {
        o_tilde: &mut otilde,
        expval_o: &mut expvalo,
        ho: &mut expval_ho,
        n: (NFIJ + NVIJ + NGI) as i32,
        nsamp: NMCSAMP as f64,
        nsamp_int: 1,
        mu: -1,
        visited: &mut visited,
        pfaff_off: NGI + NVIJ,
        jas_off: NGI,
        epsilon: 0.0,
    };
    let initial_state: FockState<u8> = {
        let mut tmp: FockState<u8> = FockState::generate_from_nelec(&mut rng, NELEC, SIZE);
        while tmp.spin_up.count_ones() != tmp.spin_down.count_ones() {
            tmp = FockState::generate_from_nelec(&mut rng, NELEC, SIZE);
        }
        tmp
    };

    let (mc_mean_energy, accumulated_states, cor) = compute_mean_energy(&mut rng, initial_state, &parameters, &sys, &mut der);
    let mut out_str: String = String::new();
    for s in accumulated_states.iter() {
        out_str.push_str(&format!("{}\n", s));
    }
    //statesfp.write(out_str.as_bytes()).unwrap();

    println!("Correlation time: {}", cor);
    let error = <f64>::sqrt((1.0 + 2.0 * cor) / (NMCSAMP as f64));
    let mut energy_str: String = String::new();
    energy_str.push_str(&format!("{} {} {}\n", mean_energy, mc_mean_energy, error));
    //energyfp.write(energy_str.as_bytes()).unwrap();
    println!("Comparing monte-carlo energy, tol: {}", error);
    mean_energy = mean_energy / norm(&parameters);
    println!("Monte-Carlo: {}, Analytic: {}", mc_mean_energy, mean_energy);
    close(mc_mean_energy, mean_energy, mean_energy * 1e-2);
}
