use assert::close;
use blas::daxpy;
use blas::dcopy;
use impurity::monte_carlo::compute_mean_energy_exact;
use impurity::gutzwiller::compute_gutzwiller_exp;
use impurity::jastrow::compute_jastrow_exp;
use impurity::pfaffian::construct_matrix_a_from_state;
use impurity::{generate_bitmask, mapto_pairwf, DerivativeOperator, FockState, SysParams, VarParams};
use impurity::hamiltonian::{kinetic, potential};

// Number of sites
const SIZE: usize = 2;
// Hubbard's model $U$ parameter
const CONS_U: f64 = 8.0;
// Hubbard's model $t$ parameter
const CONS_T: f64 = -1.0;
// Number of electrons
const NELEC: usize = 2;
const NMCSAMP: usize = 10000;
const NMCWARMUP: usize = 1000;
const CLEAN_UPDATE_FREQUENCY: usize = 32;
const TOL_SHERMAN: f64 = 1e-12;
const TOL_SINGULARITY: f64 = 1e-12;
const _NOPTITER: usize = 100;
const _EPSILON_SHIFT: f64 = 0.01;
const _DT: f64 = -0.01;

const NFIJ: usize = 4*SIZE*SIZE;
const NVIJ: usize = SIZE*(SIZE-1)/2;
const NGI: usize = SIZE;

pub const HOPPINGS: [f64; SIZE*SIZE] = [
    0.0, 1.0,
    1.0, 0.0,
];

#[derive(Debug)]
pub enum State {
    F3,
    F5,
    F6,
    F9,
    F10,
    F12
}

fn individual_state(state: &State, par: &VarParams) -> f64 {
    match state {
        State::F3 => {
            let f01dd = par.fij[0 + 1 * SIZE + 3 * SIZE * SIZE];
            let f10dd = par.fij[1 + 0 * SIZE + 3 * SIZE * SIZE];
            f01dd - f10dd
        },
        State::F5 => {
            let f11ud = par.fij[1 + 1 * SIZE + 1 * SIZE * SIZE];
            let f11du = par.fij[1 + 1 * SIZE + 2 * SIZE * SIZE];
            let g1 = par.gi[1];
            let v = par.vij[0];
            (f11ud - f11du) * <f64>::exp(g1 - v)
        },
        State::F6 => {
            let f10ud = par.fij[1 + 0 * SIZE + 1 * SIZE * SIZE];
            let f01du = par.fij[0 + 1 * SIZE + 2 * SIZE * SIZE];
            f10ud - f01du
        },
        State::F9 => {
            let f01ud = par.fij[0 + 1 * SIZE + 1 * SIZE * SIZE];
            let f10du = par.fij[1 + 0 * SIZE + 2 * SIZE * SIZE];
            f01ud - f10du
        },
        State::F10 => {
            let f00ud = par.fij[0 + 0 * SIZE + 1 * SIZE * SIZE];
            let f00du = par.fij[0 + 0 * SIZE + 2 * SIZE * SIZE];
            let g0 = par.gi[0];
            let v = par.vij[0];
            (f00ud - f00du) * <f64>::exp(g0 - v)
        },
        State::F12 => {
            let f01uu = par.fij[0 + 1 * SIZE + 0 * SIZE * SIZE];
            let f10uu = par.fij[1 + 0 * SIZE + 0 * SIZE * SIZE];
            f01uu - f10uu
        },
    }
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
    let v = par.vij[0];
    let a = <f64>::exp(2.0 * g0 - 2.0 * v)*sq(<f64>::abs(f00ud - f00du));
    let b = <f64>::exp(2.0 * g1 - 2.0 * v)*sq(<f64>::abs(f11ud - f11du));
    let _c = sq(<f64>::abs(f01uu - f10uu));
    let _d = sq(<f64>::abs(f01dd - f10dd));
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
    let v = par.vij[0];
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
    let v = par.vij[0];
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
            let v = par.vij[0];
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
            let v = par.vij[0];
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
            let v = par.vij[0];
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
            let v = par.vij[0];
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
        * <f64>::exp(par.gi[0]-par.vij[0]);
    let c = (par.fij[1 + 1 * SIZE + 1 * SIZE * SIZE]
        - par.fij[1 + 1 * SIZE + 2 * SIZE * SIZE])
        * <f64>::exp(par.gi[1]-par.vij[0]);
    let d = 2.0 * CONS_T * (b + c) * a;
    let e = sq(par.fij[1 + 1 * SIZE + 1 * SIZE * SIZE]
        - par.fij[1 + 1 * SIZE + 2 * SIZE * SIZE])
        * <f64>::exp(2.0*par.gi[1]-2.0*par.vij[0]) * CONS_U;
    let f = sq(par.fij[0 + 0 * SIZE + 1 * SIZE * SIZE]
        - par.fij[0 + 0 * SIZE + 2 * SIZE * SIZE])
        * <f64>::exp(2.0*par.gi[0]-2.0*par.vij[0]) * CONS_U;
    d + e + f
}

fn analytic_ho_expval(par: &VarParams) -> Vec<f64> {
    let mut out_der = vec![0.0; SIZE + 1 + 4*SIZE*SIZE];
    out_der[0] = individual_state(&State::F10, par)
        * energy_individual_state(&State::F10, par);
    out_der[1] = individual_state(&State::F5, par)
        * energy_individual_state(&State::F5, par);
    out_der[2] = - out_der[0] - out_der[1];
    //out_der[3] =
    //out_der[4] =
    //out_der[5] =
    //out_der[6] =
    out_der[3] = 1.0 * {
        <f64>::exp(par.gi[0] - par.vij[0]) / individual_state(&State::F10, par)
            * individual_state(&State::F10, par)
            * energy_individual_state(&State::F10, par)
    };
    out_der[4] = 1.0 * energy_individual_state(&State::F6, par);
    out_der[5] = 1.0 * energy_individual_state(&State::F9, par);
    out_der[6] = 1.0 * {
        <f64>::exp(par.gi[1] - par.vij[0]) / individual_state(&State::F5, par)
            * individual_state(&State::F5, par)
            * energy_individual_state(&State::F5, par)
    };

    out_der
}

fn analytic_derivatives_expval(par: &VarParams) -> Vec<f64> {
    let mut out_der = vec![0.0; SIZE + 1 + 4*SIZE*SIZE];
    out_der[0] = sq(individual_state(&State::F10, par));
    out_der[1] = sq(individual_state(&State::F5, par));
    out_der[2] = - out_der[0] - out_der[1];
    // fij
    out_der[3] = 1.0 * {
        <f64>::exp(par.gi[0] - par.vij[0]) / individual_state(&State::F10, par)
            * sq(individual_state(&State::F10, par))
    };
    out_der[4] = 1.0 * individual_state(&State::F6, par);
    out_der[5] = 1.0 * individual_state(&State::F9, par);
    out_der[6] = 1.0 * {
        <f64>::exp(par.gi[1] - par.vij[0]) / individual_state(&State::F5, par)
            * sq(individual_state(&State::F5, par))
    };
    out_der
}

fn print_der(der1: &[f64], der2: &[f64], npar: usize) {
    println!("Monte-Carlo    Analytic       Ratio");
    for i in 0..npar {
        println!("{:11.4e}  {:10.4e}  {:10.4e}", der1[i], der2[i], der2[i] / der1[i]);
    }
}


#[test]
fn comupte_energy_from_all_states() {
    //let mut statesfp = File::create("states").unwrap();
    //let mut energyfp = File::create("energy").unwrap();
    env_logger::init();
    //let mut rng = thread_rng();
    let bitmask = generate_bitmask(&HOPPINGS, SIZE);
    println!("bitmasks: {:?}", bitmask);
    let sys = SysParams {
        size: SIZE,
        nelec: NELEC,
        array_size: (SIZE + 7) / 8,
        cons_t: CONS_T,
        cons_u: CONS_U,
        nfij: SIZE*SIZE,
        nvij: NVIJ,
        ngi: NGI,
        mcsample_interval: 1,
        nbootstrap: 1,
        transfert_matrix: &HOPPINGS,
        hopping_bitmask: &bitmask,
        clean_update_frequency: CLEAN_UPDATE_FREQUENCY,
        nmcwarmup: NMCWARMUP,
        nmcsample: NMCSAMP,
        nwarmupchains: 1,
        tolerance_sherman_morrison: TOL_SHERMAN,
        tolerance_singularity: TOL_SINGULARITY,
        pair_wavefunction: false,
        _opt_iter: 0,
    };
    let mut fij = [
        0.0, 0.0, 0.0, 0.0,
        1.093500753438337580e-01, 3.768419990611672210e-01, 3.769186909982900624e-01, 3.322533463612635796e-01,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
    ];
    let mut vij = [5.079558854017672820e-01];
    let mut gi = [3.016937239100276336e-01, -8.096496093117950821e-01];
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
        let pstate = construct_matrix_a_from_state(&parameters.fij, state, &sys);
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
    mean_energy = mean_energy / norm(&parameters);
    let otilde: Vec<f64> = vec![0.0; (NFIJ + NVIJ + NGI) * NMCSAMP];
    let expvalo: Vec<f64> = vec![0.0; NFIJ + NVIJ + NGI];
    let expval_ho: Vec<f64> = vec![0.0; NFIJ + NVIJ + NGI];
    let visited: Vec<usize> = vec![0; NMCSAMP];
    let mut der = DerivativeOperator {
        o_tilde: otilde.into_boxed_slice(),
        expval_o: expvalo.into_boxed_slice(),
        ho: expval_ho.into_boxed_slice(),
        n: (NFIJ + NVIJ + NGI) as i32,
        nsamp: NMCSAMP as f64,
        nsamp_int: 1,
        mu: -1,
        visited: visited.into_boxed_slice(),
        pfaff_off: NGI + NVIJ,
        jas_off: NGI,
        epsilon: 0.0,
    };
    let otilde_pair: Vec<f64> = vec![0.0; (NFIJ + NVIJ + NGI) * NMCSAMP];
    let expvalo_pair: Vec<f64> = vec![0.0; NFIJ + NVIJ + NGI];
    let expval_ho_pair: Vec<f64> = vec![0.0; NFIJ + NVIJ + NGI];
    let visited_pair: Vec<usize> = vec![0; NMCSAMP];
    let mut der_pair = DerivativeOperator {
        o_tilde: otilde_pair.into_boxed_slice(),
        expval_o: expvalo_pair.into_boxed_slice(),
        ho: expval_ho_pair.into_boxed_slice(),
        n: (sys.size*sys.size + NVIJ + NGI) as i32,
        nsamp: NMCSAMP as f64,
        nsamp_int: 1,
        mu: -1,
        visited: visited_pair.into_boxed_slice(),
        pfaff_off: NGI + NVIJ,
        jas_off: NGI,
        epsilon: 0.0,
    };

    let mean_energy_es = compute_mean_energy_exact(&parameters, &sys, &mut der);
    der_pair.mu = der.mu;
    //for s in accumulated_states.iter() {
    //    out_str.push_str(&format!("{}\n", s));
    //}
    //statesfp.write(out_str.as_bytes()).unwrap();

    //println!("Correlation time: {}", cor);
    //let error = <f64>::sqrt((1.0 + 2.0 * cor) / (NMCSAMP as f64));
    let mut energy_str: String = String::new();
    energy_str.push_str(&format!("{} {}\n", mean_energy, mean_energy_es));
    //energyfp.write(energy_str.as_bytes()).unwrap();
    //println!("Comparing monte-carlo energy, tol: {}", error);
    println!("Monte-Carlo: {}, Analytic: {}", mean_energy_es, mean_energy);
    close(mean_energy_es, mean_energy, 1e-12);
    mapto_pairwf(&der, &mut der_pair, &sys);

    // Test derivatives
    let exp_val = analytic_derivatives_expval(&parameters);
    println!("Checking <O>");
    print_der(&der_pair.expval_o, &exp_val, sys.ngi+sys.nvij+sys.nfij);
    let psi = norm(&parameters);
    println!("Norm: {:10.4e}", psi);
    for i in 0..sys.ngi+sys.nvij+sys.nfij {
        println!("{} == {}, tol = {}", der_pair.expval_o[i], exp_val[i] / psi, 2e-2);
        close(der_pair.expval_o[i], exp_val[i] / psi, 1e-12);
    }

    let exp_val_ho = analytic_ho_expval(&parameters);
    println!("Checking <HO>");
    print_der(&der_pair.ho, &exp_val_ho, sys.ngi+sys.nvij+sys.nfij);
    let psi = norm(&parameters);
    println!("Norm: {:10.4e}", psi);
    for i in 0..sys.ngi+sys.nvij+sys.nfij {
        println!("{} == {},  tol = {}", der_pair.ho[i] * psi, exp_val_ho[i], 2e-2);
        close(der_pair.ho[i] * psi, exp_val_ho[i], 2e-2);
    }
    let mut x0 = vec![0.0; SIZE * SIZE + NVIJ + NGI];
    x0[0] = parameters.gi[0];
    x0[1] = parameters.gi[1];
    x0[2] = parameters.vij[0];
    for i in 0.. SIZE * SIZE {
        x0[NGI + NVIJ + i] = parameters.fij[SIZE*SIZE + i];
    }

    // 68 misawa
    let mut b: Vec<f64> = vec![0.0; der_pair.n as usize];
    unsafe {
        let incx = 1;
        let incy = 1;
        daxpy(der_pair.n, -mean_energy, &der_pair.expval_o, incx, &mut der_pair.ho, incy);
        dcopy(der_pair.n, &der_pair.ho, incx, &mut b, incy);
    }
    println!("x0 = {:?}", x0);
}
