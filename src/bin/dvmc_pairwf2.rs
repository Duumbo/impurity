use impurity::optimisation::GenParameterMap;
//use blas::dcopy;
//use log::{debug, info};
use rand_mt::Mt64;
use rand::Rng;
use std::mem;

//use indicatif::{ProgressBar, ProgressStyle};
use std::fs::File;
use std::io::Write;

use impurity::{VarParams, SysParams, generate_bitmask, FockState, RandomStateGeneration};
use impurity::dvmc::{variationnal_monte_carlo, EnergyOptimisationMethod, EnergyComputationMethod, VMCParams};

type BitSize = u128;

const SEED: u64 = 12458;
const SIZE_N: usize = 4;
const SIZE_M: usize = 4;
// SIZE = SIZE_N x SIZE_M
const SIZE: usize = SIZE_N*SIZE_M;
const NFIJ: usize = 4*SIZE*SIZE;
const NVIJ: usize = SIZE*(SIZE - 1) / 2;
const NGI: usize = SIZE;
const NPARAMS: usize = NFIJ + NGI + NVIJ;
const NELEC: usize = SIZE;
const NMCSAMP: usize = 1000;
const NBOOTSTRAP: usize = 1;
const NMCWARMUP: usize = NOPTITER;
const NWARMUPCHAINS: usize = 1000;
const MCSAMPLE_INTERVAL: usize = SIZE;
const NTHREADS: usize = 1;
const CLEAN_UPDATE_FREQUENCY: usize = 1;
const TOLERENCE_SHERMAN_MORRISSON: f64 = 1e-12;
const TOLERENCE_SINGULARITY: f64 = 1e-12;
const _CONS_U: f64 = 1.0;
const CONS_T: f64 = 1.0;
const INITIAL_RATIO_UT: f64 = 8.0;
const FINAL_RATIO_UT: f64 = 32.0;
const NRATIO_POINTS: usize = 1;
const EPSILON_CG: f64 = 1e-16;
const EPSILON_SHIFT: f64 = 1e-3;
const OPTIMISATION_TIME_STEP: f64 = 2e-2;
const OPTIMISATION_DECAY: f64 = 0.0;
const NOPTITER: usize = 1000;
const ADAMS_BASHFORTH_ORDER: usize = 1;
const KMAX: usize = NPARAMS;
const FILTER_BEFORE_SHIFT: bool = false; // Better false (16 sites)
//const PARAM_THRESHOLD: f64 = <f64>::EPSILON;
const PARAM_THRESHOLD: f64 = 1e-2;
//const PARAM_THRESHOLD: f64 = 0.0;
//const PARAM_THRESHOLD: f64 = -<f64>::INFINITY;
const OPTIMISE: bool = true;
const OPTIMISE_GUTZ: bool = true;
const OPTIMISE_JAST: bool = true;
const OPTIMISE_ORB: bool = true;
const COMPUTE_ENERGY_METHOD: EnergyComputationMethod = EnergyComputationMethod::MonteCarlo;
const OPTIMISE_ENERGY_METHOD: EnergyOptimisationMethod = EnergyOptimisationMethod::ConjugateGradiant;
const _ENERGY_CONV_AVERAGE_SAMPLE: usize = 20;
const N_GUTZ: usize = NGI;
const N_JAST: usize = NVIJ;
const PAIRWF: bool = false;
const CONV_PARAM_THRESHOLD: f64 = 1e-100;

//const N_INDEP_PARAMS: usize = NFIJ + NGI + NVIJ;
const N_INDEP_PARAMS: usize = SIZE*SIZE + NGI + NVIJ;
//const N_INDEP_PARAMS: usize = 3;
const SET_VIJ_ZERO: bool = true;
const SET_GI_ZERO: bool = true;
const SET_PAIR_PFAFFIAN: bool = true;

pub enum BoundCond {
    Periodic,
    Closed
}
const LATTICE_BOUNDARY_CONDITIONS: BoundCond = BoundCond::Closed;

pub const HOPPINGS: [f64; SIZE*SIZE] = {
    // Constructs hopping matrix for SITES_N*SITES_M
    let mut tmp = [0.0; SIZE*SIZE];
    let mut i = 0;
    let mut j = 0;
    while i < SIZE_M {
        while j < SIZE_N {
            let next_inline = (i + 1) % SIZE_M;
            let prev_inline = (i + SIZE_M - 1) % SIZE_M;
            let next_column = (j + 1) % SIZE_N;
            let prev_column = (j +SIZE_N - 1) % SIZE_N;
            match LATTICE_BOUNDARY_CONDITIONS {
                BoundCond::Closed => {
                    if next_inline > i {
                        tmp[ next_inline + j * SIZE_M + (i + j * SIZE_M) * SIZE] = 1.0;
                    }
                    if prev_inline < i {
                        tmp[ prev_inline + j * SIZE_M + (i + j * SIZE_M) * SIZE] = 1.0;
                    }
                    if next_column > j {
                        tmp[ i + j * SIZE_M + (i + next_column * SIZE_M) * SIZE] = 1.0;
                    }
                    if prev_column < j {
                        tmp[ i + j * SIZE_M + (i + prev_column * SIZE_M) * SIZE] = 1.0;
                    }
                },
                BoundCond::Periodic => {
                    tmp[ next_inline + j * SIZE_M + (i + j * SIZE_M) * SIZE] += 1.0;
                    tmp[ prev_inline + j * SIZE_M + (i + j * SIZE_M) * SIZE] += 1.0;
                    tmp[ i + j * SIZE_M + (i + next_column * SIZE_M) * SIZE] += 1.0;
                    tmp[ i + j * SIZE_M + (i + prev_column * SIZE_M) * SIZE] += 1.0;
                },
            };
            j += 1;
        }
        i += 1;
        j = 0;
    }
    i = 0;
    // RESET DIAGONAL (edge case for if SIZE_M==1 or SIZE_N==1)
    while i < SIZE {
        tmp[ i + i*SIZE] = 0.0;
        i += 1;
    }
    tmp
};

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


fn _sq(a: f64) -> f64 {
    <f64>::abs(a) * <f64>::abs(a)
}

fn _write_energy(fp: &mut File, e: &[f64]) {
    let mut mean_energy = 0.0;
    let mut mean_error = 0.0;
    let mut mean_corr = 0.0;
    for i in 0.._ENERGY_CONV_AVERAGE_SAMPLE {
        mean_energy += e[NOPTITER*3 - 3*i - 3];
        mean_error += e[NOPTITER*3 - 3*i - 2];
        mean_corr += e[NOPTITER*3 - 3*i - 1];
    }
        mean_energy *=  1.0 / _ENERGY_CONV_AVERAGE_SAMPLE as f64;
        mean_error *= 1.0 / _ENERGY_CONV_AVERAGE_SAMPLE as f64;
        mean_corr *= 1.0 / _ENERGY_CONV_AVERAGE_SAMPLE as f64;

    writeln!(fp, "{}", format!("{} {} {}", mean_energy, mean_error, mean_corr)).unwrap();
}

fn log_energy_convs(e: &[f64], fp: &mut File, n_true_iter: usize) {
    for i in 0..n_true_iter {
        writeln!(fp, "{}", format!("{} {} {}", e[3*i], e[3*i +1], e[3*i+2])).unwrap();
    }
}

fn _print_matrix(mat: &[f64], n: usize, m: usize) {
    let width = 16;
    println!("dim = ({}, {})", n, m);
    let mut outstr = "".to_owned();
    outstr.push_str(&format!("Matrix = "));
    for i in 0..n {
        outstr.push_str(&format!("\n           "));
        for j in 0..m {
            outstr.push_str(&format!("{:>width$.04e}", mat[i + n * j]));
        }
    }
    println!("{}", outstr);
}

//const ORB_MVMC: [f64; 256] = [
//    0.666757086512084,
//    0.286090169975276,
//    0.011540326441343,
//    0.020093037962211,
//    0.343010633098388,
//    -0.02437028151181,
//    -0.03641325137755,
//    0.144025553929876,
//    -0.04512673879211,
//    -0.08773820830744,
//    0.094972154703531,
//    0.091436397813703,
//    0.047004245831873,
//    0.113708798679599,
//    0.136692251175072,
//    -0.08977854137679,
//    0.286814624974364,
//    0.564143864392261,
//    0.378066857760119,
//    0.007116409426331,
//    0.109246818237017,
//    0.158273018960884,
//    0.090861525238776,
//    -0.13038564732068,
//    0.018588994240957,
//    0.026691188019809,
//    0.041670093072899,
//    -0.00608551311167,
//    0.134144058892849,
//    0.002896181520296,
//    0.076933776950295,
//    0.136479974062799,
//    -0.03969822430991,
//    0.348598300256572,
//    0.572539155047812,
//    0.272065985177365,
//    -0.13580744129527,
//    0.077674846058671,
//    0.269926879253226,
//    -0.00333975824996,
//    0.101277354589919,
//    -0.00945519973647,
//    0.001758405574274,
//    0.026334191838158,
//    0.227282979361719,
//    0.056230418434108,
//    0.036408482568390,
//    0.111542360311511,
//    0.047968424681364,
//    -0.04527011112852,
//    0.187048004768072,
//    0.545927794452178,
//    0.100822839048893,
//    0.014907846590197,
//    0.108395787352226,
//    0.443351479700969,
//    -0.00483796632896,
//    -0.00516662021476,
//    -0.11342771070779,
//    0.026736987087408,
//    0.055639099721483,
//    0.203983859170746,
//    0.118718863666411,
//    0.014589719585825,
//    0.278736790476632,
//    0.075756805193891,
//    -0.08084516624099,
//    0.050952429888204,
//    0.522905675826001,
//    0.338971687083658,
//    0.013381761655916,
//    0.168984766754409,
//    0.210909679943873,
//    0.070569891624482,
//    -0.04066489177404,
//    0.085760370880244,
//    -0.03054714516080,
//    0.043153088734279,
//    0.030725193129026,
//    0.145614152594314,
//    -0.00015727028993,
//    0.177035817851798,
//    0.116694579221741,
//    0.042749130912412,
//    0.323414981599964,
//    0.635930024627275,
//    0.248015291537895,
//    0.051214141682648,
//    0.129243256426765,
//    0.285762148185434,
//    -0.00982476770020,
//    -0.08345182024645,
//    -0.10985080827454,
//    -0.00904235870139,
//    0.097961371659280,
//    0.095840021565877,
//    -0.02489407680377,
//    0.161571268777002,
//    0.250413964755045,
//    0.136221442257539,
//    -0.02914609627066,
//    0.260418181482008,
//    0.374879466657312,
//    0.176226095113019,
//    -0.02557837720795,
//    0.154549252871044,
//    0.317663796894934,
//    0.095017540516748,
//    -0.05972323574767,
//    -0.04718260530426,
//    -0.02524212926563,
//    -0.01425301686732,
//    0.097500109288484,
//    -0.09381043521615,
//    0.043041460452022,
//    0.413810900867561,
//    0.141026381544753,
//    0.015264651547298,
//    0.241270140512857,
//    0.538827862592152,
//    0.051716278200180,
//    0.018070422138566,
//    0.097838035403967,
//    0.280216211380776,
//    0.039268470117464,
//    0.111212974785637,
//    -0.01332915522840,
//    0.022403785313613,
//    -0.03018424702422,
//    -0.01732169628132,
//    0.075848683042504,
//    0.023608330551455,
//    0.283789487081070,
//    0.111715241277135,
//    -0.02137131797218,
//    0.016880508080719,
//    0.530995787235326,
//    0.210616150588507,
//    -0.04129199253663,
//    0.132744854012485,
//    0.463099006405284,
//    0.003286355677279,
//    -0.11938764363175,
//    0.087068447036551,
//    -0.01060406200836,
//    -0.02129421896157,
//    -0.00572528490047,
//    -0.05135189633102,
//    0.145202265410351,
//    0.252624122066156,
//    0.198391377990429,
//    0.030887108421043,
//    0.208234165639156,
//    0.452202012421954,
//    0.216987709566990,
//    0.008527272164055,
//    0.121797210556754,
//    0.223954604889196,
//    0.141140296591585,
//    -0.03492450169467,
//    0.114944617374325,
//    0.127957733094044,
//    0.047042775767650,
//    -0.12300083641001,
//    -0.07819362039386,
//    -0.04029640280825,
//    0.294564939029030,
//    0.109926678511417,
//    0.011931917655935,
//    0.297200629152693,
//    0.562888724083025,
//    0.337619054068040,
//    0.021574082709380,
//    0.127870191943226,
//    0.115862533170469,
//    -0.08331038925188,
//    0.153184791569163,
//    0.026428924019018,
//    0.023390962078974,
//    -0.02693551702027,
//    0.038388388700673,
//    -0.03384928648002,
//    0.156348971130674,
//    0.256449734123796,
//    0.147738033211962,
//    -0.03582393096634,
//    0.268853954927088,
//    0.578545726572740,
//    0.058574371788179,
//    -0.11258388525537,
//    0.067228102307760,
//    0.281595208888845,
//    0.031974607393636,
//    0.126944259844287,
//    0.189156047416118,
//    0.089326035784310,
//    -0.02001532930066,
//    -0.13509264857321,
//    -0.02057376102528,
//    -0.03254917324965,
//    0.403196496534628,
//    0.085355477778346,
//    0.083907241661412,
//    0.143278854266006,
//    0.486332837698593,
//    0.208685272433464,
//    -0.03256227519965,
//    0.053627418028271,
//    0.107807962917929,
//    -0.00596656106780,
//    0.001974869520666,
//    0.193781588093484,
//    0.042928731501280,
//    -0.04369951175699,
//    0.003243240525903,
//    0.016395529472261,
//    -0.00223879170493,
//    0.201158931890878,
//    0.069459064774957,
//    -0.16752828571760,
//    0.190862992527834,
//    0.610065016093839,
//    0.378806794469981,
//    -0.04087076635290,
//    0.046231099321971,
//    0.068697188566222,
//    0.061176994848785,
//    0.111006298107891,
//    0.040633493650589,
//    0.045706122596658,
//    -0.03801047422289,
//    0.019922944799210,
//    -0.14802377531069,
//    0.164369703993762,
//    0.112411259193314,
//    0.034033227533131,
//    -0.03923837397985,
//    0.334056603157368,
//    0.549538477350001,
//    0.328149993156676,
//    -0.10105409529973,
//    0.059669347656295,
//    0.062293657706354,
//    0.069825378006894,
//    0.106029402586386,
//    0.146442560914971,
//    -0.00791214410019,
//    -0.00279126532850,
//    0.096016815262221,
//    -0.07424623945024,
//    -0.04545492549332,
//    0.314233553570198,
//    -0.01164169101462,
//    0.019089165369585,
//    0.373811293935293,
//    0.707060846369222,
//];


fn main() {
    let mut fp = File::create("u_t_sweep").unwrap();
    writeln!(fp, "{}", format!("# {} {} {}", SIZE, NMCSAMP, NOPTITER)).unwrap();
    let mut paramsfp = File::create("params").unwrap();
    writeln!(paramsfp, "{}", format!("# {} {} {}", SIZE, NMCSAMP, NOPTITER)).unwrap();
    let mut _save: bool = true;
    // Initialize logger
    env_logger::init();
    let bitmask = generate_bitmask(&HOPPINGS, SIZE);
    let mut rng = Vec::new();
    for i in 0..NTHREADS {
        rng.push(Mt64::new(SEED + i as u64));
    }
    let mut rngs: Vec<&mut Mt64> = rng.iter_mut().collect();

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
            all_params.push(rngs[0].gen());
            //all_params.push(1.0);
        }
        let (gi, params) = all_params.split_at_mut(NGI);
        let (vij, fij) = params.split_at_mut(NVIJ);
        let mut parameters = VarParams {
            fij,
            gi,
            vij,
            size: SIZE
        };
        //let g = parameters.gi[0];
        if SET_GI_ZERO {
            for i in 0..NGI {
                let g = 0.0;
                parameters.gi[i] = g;
            }
        }
        if SET_VIJ_ZERO {
            for i in 0..NVIJ {
                parameters.vij[i] = 0.0;
            }
        }
        if SET_PAIR_PFAFFIAN {
            for i in 0..SIZE*SIZE {
                parameters.fij[i] = 0.0;
                //parameters.fij[i +SIZE*SIZE] = 0.5;
                //parameters.fij[i +SIZE*SIZE] = ORB_MVMC[i];
                parameters.fij[i +2*SIZE*SIZE] = 0.0;
                parameters.fij[i +3*SIZE*SIZE] = 0.0;
            }
        }
        let mut max = <f64>::MIN;
        for i in 0..NGI {
            if max < parameters.gi[i] {
                max = parameters.gi[i];
            }
        }
        for i in 0..NVIJ {
            if max < parameters.vij[i] {
                max = parameters.vij[i];
            }
        }
        for i in 0..NFIJ {
            if max < parameters.fij[i] {
                max = parameters.fij[i];
            }
        }
        // Scale parameters
        for i in 0..NGI {
            parameters.gi[i] /= max;
        }
        for i in 0..NVIJ {
            parameters.vij[i] /= max;
        }
        for i in 0..NFIJ {
            parameters.fij[i] /= max;
        }
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
            nthreads: NTHREADS,
            filter_before_shift: FILTER_BEFORE_SHIFT,
            adams_bashforth_order: ADAMS_BASHFORTH_ORDER,
        };

        let state: FockState<BitSize> = {
            let mut tmp: FockState<BitSize> = FockState::generate_from_nelec(&mut rngs[0], NELEC, SIZE);
            while tmp.spin_up.count_ones() != tmp.spin_down.count_ones() {
                tmp = FockState::generate_from_nelec(&mut rngs[0], NELEC, SIZE);
            }
            tmp
        };
        let mut states_vec = Vec::with_capacity(NTHREADS);
        for _ in 0..NTHREADS {
            states_vec.push(state);
        }
        let param_map = GenParameterMap {
            dim: N_INDEP_PARAMS as i32,
            gendim: (NFIJ + NGI + NVIJ) as i32,
            n_genparams: (NFIJ + NGI + NVIJ) as i32,
            n_independant_gutzwiller: N_GUTZ,
            n_independant_jastrow: N_JAST,
            projector: Box::new(PARAMS_PROJECTOR),
        };

        println!("Before starting.");
        let (e_array, noptiter) = variationnal_monte_carlo(&mut rngs, &mut states_vec, &mut parameters, &mut system_params, &vmcparams, &param_map);
        //write_energy(&mut fp, &e_array);

        log_energy_convs(&e_array, &mut paramsfp, noptiter);

    }
    mem::drop(rng);
}
