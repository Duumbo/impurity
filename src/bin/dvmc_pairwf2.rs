use impurity::optimisation::GenParameterMap;
//use blas::dcopy;
//use log::{debug, info};
use rand_mt::Mt64;
use rand::Rng;
//use indicatif::{ProgressBar, ProgressStyle};
use std::fs::File;
use std::io::Write;

use impurity::{VarParams, SysParams, generate_bitmask, FockState, RandomStateGeneration};
use impurity::dvmc::{variationnal_monte_carlo, EnergyOptimisationMethod, EnergyComputationMethod, VMCParams};

type BitSize = u32;

const SEED: u64 = 14;
const SIZE: usize = 16;
const NFIJ: usize = 4*SIZE*SIZE;
const NVIJ: usize = SIZE*(SIZE - 1) / 2;
const NGI: usize = SIZE;
const NPARAMS: usize = NFIJ + NGI + NVIJ;
const NELEC: usize = SIZE;
const NMCSAMP: usize = 1000;
const NBOOTSTRAP: usize = 1;
const NMCWARMUP: usize = 1000;
const NWARMUPCHAINS: usize = 1;
const MCSAMPLE_INTERVAL: usize = 1;
const _NTHREADS: usize = 1;
const CLEAN_UPDATE_FREQUENCY: usize = 32;
const TOLERENCE_SHERMAN_MORRISSON: f64 = 1e-12;
const TOLERENCE_SINGULARITY: f64 = 1e-12;
const CONS_U: f64 = 1.0;
const CONS_T: f64 = 1.0;
const INITIAL_RATIO_UT: f64 = 8.0;
const FINAL_RATIO_UT: f64 = 32.0;
const NRATIO_POINTS: usize = 1;
const EPSILON_CG: f64 = 1e-16;
const EPSILON_SHIFT: f64 = 1e-3;
const OPTIMISATION_TIME_STEP: f64 = 1e-2;
const OPTIMISATION_DECAY: f64 = 0.0;
const NOPTITER: usize = 1000;
const KMAX: usize = NPARAMS;
const PARAM_THRESHOLD: f64 = 1e-3;
//const PARAM_THRESHOLD: f64 = 0.0;
const OPTIMISE: bool = true;
const OPTIMISE_GUTZ: bool = true;
const OPTIMISE_JAST: bool = true;
const OPTIMISE_ORB: bool = true;
const SET_EXPVALO_ZERO: bool = false;
const COMPUTE_ENERGY_METHOD: EnergyComputationMethod = EnergyComputationMethod::MonteCarlo;
const OPTIMISE_ENERGY_METHOD: EnergyOptimisationMethod = EnergyOptimisationMethod::ConjugateGradiant;
const ENERGY_CONV_AVERAGE_SAMPLE: usize = 20;
const N_INDEP_PARAMS: usize = SIZE*SIZE + NGI + NVIJ;
const N_GUTZ: usize = NGI;
const N_JAST: usize = NVIJ;
const PAIRWF: bool = false;
const CONV_PARAM_THRESHOLD: f64 = 1e-10;

// 4 et 2 Sites
//pub const HOPPINGS: [f64; SIZE*SIZE] = [
//    0.0, 1.0, 1.0, 0.0,
//    //0.0, 1.0, 1.0, 0.0,
//    //1.0, 0.0, 0.0, 1.0,
//    //1.0, 0.0, 0.0, 1.0,
//    //0.0, 1.0, 1.0, 0.0
//];

// 7 Sites
//pub const HOPPINGS: [f64; SIZE*SIZE] = [
//    0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0,
//    1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
//    0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0,
//    1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
//    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
//    0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0,
//    0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
//];

// 8 Sites
//pub const HOPPINGS: [f64; SIZE*SIZE] = [
//    0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0,
//    1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
//    0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
//    1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
//    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0,
//    0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0,
//    0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0,
//    0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0,
//];

// 9 Sites
//pub const HOPPINGS: [f64; SIZE*SIZE] = [
//    0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
//    1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
//    0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
//    1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
//    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0,
//    0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0,
//    0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0,
//    0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0,
//    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
//];

// 10 Sites
//pub const HOPPINGS: [f64; SIZE*SIZE] = [
//    0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
//    1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
//    0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
//    1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
//    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0,
//    0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0,
//    0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0,
//    0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0,
//    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
//    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
//];

// 16 Sites
pub const HOPPINGS: [f64; SIZE*SIZE] = [
    0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
    1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
    1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0,
    0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0,
    0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0,
];
//

// Pairwf
//const PARAMS_PROJECTOR: [f64; (NFIJ + NVIJ + NGI) * (NFIJ + NVIJ + NGI - 1) / 2 + NFIJ + NVIJ + NGI] = [
//    /* g0 */    1.0, // 0,
//    /* g1 */    1.0, 0.0, //
//    /* v01 */   0.0, 0.0, 0.0,
//    /* f00uu */ 0.0, 0.0, 0.0, 0.0,
//    /* f01uu */ 0.0, 0.0, 0.0, 0.0, 0.0,
//    /* f10uu */ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
//    /* f11uu */ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
//    /* f00ud */ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
//    /* f01ud */ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
//    /* f10ud */ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
//    /* f11ud */ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
//    /* f00du */ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
//    /* f01du */ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
//    /* f10du */ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
//    /* f11du */ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
//    /* f00dd */ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
//    /* f01dd */ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
//    /* f10dd */ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
//    /* f11dd */ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
//];

// General rep
//const PARAMS_PROJECTOR: [f64; (NFIJ + NVIJ + NGI) * (NFIJ + NVIJ + NGI - 1) / 2 + NFIJ + NVIJ + NGI] = {
//    let mut param = [0.0; (NFIJ + NVIJ + NGI) * (NFIJ + NVIJ + NGI - 1) / 2 + NFIJ + NVIJ + NGI];
//    let mut j = 0;
//    let mut n = 0;
//    while j < NFIJ + NVIJ + NGI {
//        param[j + (j * (j+1) / 2)] = 1.0;
//        j += 1;
//        n += 1;
//    }
//  if n != N_INDEP_PARAMS {
//      panic!("Number of set independant params is not correct.");
//  }
//    param
//};


// General pairwf rep
const PARAMS_PROJECTOR: [f64; (NFIJ + NVIJ + NGI) * (NFIJ + NVIJ + NGI - 1) / 2 + NFIJ + NVIJ + NGI] = {
    let mut param = [0.0; (NFIJ + NVIJ + NGI) * (NFIJ + NVIJ + NGI - 1) / 2 + NFIJ + NVIJ + NGI];
    let mut j = 0;
    let mut n = 0;
    while j < NVIJ + NGI {
        param[j + (j * (j+1) / 2)] = 1.0;
        j += 1;
        n += 1;
    }
    let mut j = NVIJ + NGI;
    while j < SIZE * SIZE + NVIJ + NGI{
        j += SIZE * SIZE;
        param[j + (j * (j+1) / 2)] = 1.0;
        j -= SIZE * SIZE;
        j += 1;
        n += 1;
    }
    if n != N_INDEP_PARAMS {
        panic!("Number of set independant params is not correct.");
    }
    param
};


fn sq(a: f64) -> f64 {
    <f64>::abs(a) * <f64>::abs(a)
}

fn write_energy(fp: &mut File, e: &[f64]) {
    let mut mean_energy = 0.0;
    let mut mean_error = 0.0;
    let mut mean_corr = 0.0;
    for i in 0..ENERGY_CONV_AVERAGE_SAMPLE {
        mean_energy += e[NOPTITER*3 - 3*i - 3];
        mean_error += e[NOPTITER*3 - 3*i - 2];
        mean_corr += e[NOPTITER*3 - 3*i - 1];
    }
        mean_energy *=  1.0 / ENERGY_CONV_AVERAGE_SAMPLE as f64;
        mean_error *= 1.0 / ENERGY_CONV_AVERAGE_SAMPLE as f64;
        mean_corr *= 1.0 / ENERGY_CONV_AVERAGE_SAMPLE as f64;

    writeln!(fp, "{}", format!("{} {} {}", mean_energy, mean_error, mean_corr)).unwrap();
}

fn log_energy_convs(e: &[f64], fp: &mut File) {
    for i in 0..NOPTITER {
        writeln!(fp, "{}", format!("{} {} {}", e[3*i], e[3*i +1], e[3*i+2])).unwrap();
    }
}

fn main() {
    let mut fp = File::create("u_t_sweep").unwrap();
    writeln!(fp, "{}", format!("# {} {} {}", SIZE, NMCSAMP, NOPTITER)).unwrap();
    let mut paramsfp = File::create("params").unwrap();
    writeln!(paramsfp, "{}", format!("# {} {} {}", SIZE, NMCSAMP, NOPTITER)).unwrap();
    let mut _save: bool = true;
    // Initialize logger
    env_logger::init();
    let bitmask = generate_bitmask(&HOPPINGS, SIZE);
    let mut rng = Mt64::new(SEED);

    for nsweep_iter in 0..NRATIO_POINTS {
        let mut system_params = SysParams {
            size: SIZE,
            nelec: NELEC,
            array_size: (SIZE + 7) / 8,
            cons_t: -CONS_T,
            cons_u: CONS_T * INITIAL_RATIO_UT + (nsweep_iter as f64 / NRATIO_POINTS as f64) * CONS_T * (FINAL_RATIO_UT - INITIAL_RATIO_UT),
            nfij: NFIJ,
            nvij: NVIJ,
            ngi: NGI,
            transfert_matrix: &HOPPINGS,
            hopping_bitmask: &bitmask,
            clean_update_frequency: CLEAN_UPDATE_FREQUENCY,
            nmcsample: NMCSAMP,
            nbootstrap: NBOOTSTRAP,
            nmcwarmup: NMCWARMUP,
            nwarmupchains: NWARMUPCHAINS,
            mcsample_interval: MCSAMPLE_INTERVAL,
            tolerance_sherman_morrison: TOLERENCE_SHERMAN_MORRISSON,
            tolerance_singularity: TOLERENCE_SINGULARITY,
            pair_wavefunction: PAIRWF,
            _opt_iter: 0,
        };
        //println!("U = {}, T = {}", system_params.cons_u, system_params.cons_t);

        let mut all_params: Vec<f64> = Vec::with_capacity(NGI + NVIJ + NFIJ);
        for _ in 0..(NGI + NVIJ + NFIJ) {
            all_params.push(rng.gen());
        }
        let (gi, params) = all_params.split_at_mut(NGI);
        let (vij, fij) = params.split_at_mut(NVIJ);
        let mut parameters = VarParams {
            fij,
            gi,
            vij,
            size: SIZE
        };
        //for i in 0..SIZE*SIZE {
        //    parameters.fij[i] = 0.0;
        //    parameters.fij[i +SIZE*SIZE] = 0.5;
        //    parameters.fij[i +2*SIZE*SIZE] = 0.0;
        //    parameters.fij[i +3*SIZE*SIZE] = 0.0;
        //}
        //println!("{:?}", parameters.fij);

        let vmcparams = VMCParams {
            dt: OPTIMISATION_TIME_STEP,
            optimisation_decay: OPTIMISATION_DECAY,
            threshold: PARAM_THRESHOLD,
            kmax: KMAX,
            epsilon: EPSILON_SHIFT,
            epsilon_cg: EPSILON_CG,
            noptiter: NOPTITER,
            nparams: N_INDEP_PARAMS,
            optimise: OPTIMISE,
            optimise_gutzwiller: OPTIMISE_GUTZ,
            optimise_jastrow: OPTIMISE_JAST,
            optimise_orbital: OPTIMISE_ORB,
            compute_energy_method: COMPUTE_ENERGY_METHOD,
            optimise_energy_method: OPTIMISE_ENERGY_METHOD,
            conv_param_threshold: CONV_PARAM_THRESHOLD,
        };

        let state: FockState<BitSize> = {
            let mut tmp: FockState<BitSize> = FockState::generate_from_nelec(&mut rng, NELEC, SIZE);
            while tmp.spin_up.count_ones() != tmp.spin_down.count_ones() {
                tmp = FockState::generate_from_nelec(&mut rng, NELEC, SIZE);
            }
            tmp
        };
        let param_map = GenParameterMap {
            dim: N_INDEP_PARAMS as i32,
            gendim: (NFIJ + NGI + NVIJ) as i32,
            n_genparams: (NFIJ + NGI + NVIJ) as i32,
            n_independant_gutzwiller: N_GUTZ,
            n_independant_jastrow: N_JAST,
            projector: Box::new(PARAMS_PROJECTOR),
        };

        let e_array = variationnal_monte_carlo(&mut rng, state, &mut parameters, &mut system_params, &vmcparams, &param_map);
        //write_energy(&mut fp, &e_array);

        log_energy_convs(&e_array, &mut paramsfp);

    }
}
