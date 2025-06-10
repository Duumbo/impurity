use criterion::{criterion_group, Criterion, BenchmarkId};
use impurity::dvmc::{VMCParams, EnergyComputationMethod, EnergyOptimisationMethod};
use impurity::monte_carlo::compute_mean_energy;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use impurity::{DerivativeOperator, VarParams, SysParams, FockState, RandomStateGeneration, generate_bitmask, ARRAY_SIZE};

const MAX_SITES: usize = 128;
type DATATYPE = u128;

pub fn bench_monte_carlo(c: &mut Criterion) {
    let mut group = c.benchmark_group("Énergie moyenne monte-carlo");
    group.significance_level(0.1).sample_size(500);
    let mut rng = SmallRng::seed_from_u64(42);

    const NMCSAMPLE: f64 = 1000.0;

    // Setup the loop over the number of sites
    for i in (2..MAX_SITES).step_by(2) {
        let hoppings: Vec<f64> = {
            // Constructs hopping matrix for SITES_N*SITES_M
            let mut tmp = vec![0.0; i*i];
            let mut k = 0;
            let mut j = 0;
            while k < i {
                while j < 1 {
                    let next_inline = (k + 1) % i;
                    let prev_inline = (k + i - 1) % i;
                    let next_column = (j + 1) % 1;
                    let prev_column = (j + 1 - 1) % 1;
                    tmp[ next_inline + j * i + (k + j * i) * i] = 1.0;
                    tmp[ prev_inline + j * i + (k + j * i) * i] = 1.0;
                    tmp[ k + j * i + (k + next_column * i) * i] = 1.0;
                    tmp[ k + j * i + (k + prev_column * i) * i] = 1.0;
                    j += 1;
                }
                k += 1;
                j = 0;
            }
            k = 0;
            // RESET DIAGONAL (edge case for if SIZE_M==1 or SIZE_N==1)
            while k < i {
                tmp[ k + k*i] = 0.0;
                k += 1;
            }
            tmp
        };
        let initial_state: FockState<DATATYPE> = FockState::generate_from_nelec(&mut rng, i + i%2, i);
        let ngi = i;
        let nvij = i * ( i - 1 ) / 2;
        let nfij = 4* i * i;
        let nparams = ngi + nvij + nfij;
        let bitmask = generate_bitmask(&hoppings, i);
        let sys = SysParams {
            size: i,
            nelec: i - i%2,
            array_size: ARRAY_SIZE,
            cons_t: -1.0,
            cons_u: 1.0,
            nfij: nfij,
            nvij: nvij,
            ngi: ngi,
            transfert_matrix: &hoppings,
            hopping_bitmask: &bitmask,
            clean_update_frequency: 1,
            nmcsample: NMCSAMPLE as usize,
            nbootstrap: 1,
            nmcwarmup: 100,
            nwarmupchains: 1,
            mcsample_interval: 1,
            tolerance_sherman_morrison: 1e-16,
            tolerance_singularity: 1e-16,
            pair_wavefunction: false,
            _opt_iter: 0,
        };
        let mut all_params: Vec<f64> = Vec::with_capacity(ngi + nvij + nfij);
        for _ in 0..(ngi + nvij + nfij) {
            all_params.push(rng.gen());
        }
        let (gi, params) = all_params.split_at_mut(ngi);
        let (vij, fij) = params.split_at_mut(nvij);
        let params = VarParams {
            fij,
            gi,
            vij,
            size: i
        };
        let vmcparams = VMCParams {
            dt: 0.0,
            optimisation_decay: 0.0,
            threshold: 1e-2,
            kmax: nparams,
            epsilon: 1e-2,
            epsilon_cg: 1e-16,
            noptiter: 1,
            nparams: nparams,
            optimise: true,
            optimise_gutzwiller: true,
            optimise_jastrow: true,
            optimise_orbital: true,
            compute_energy_method: EnergyComputationMethod::MonteCarlo,
            optimise_energy_method: EnergyOptimisationMethod::ConjugateGradiant,
            conv_param_threshold: 1e-16,
            nthreads: 1
        };
        let mut der = DerivativeOperator::new(
            vmcparams.nparams as i32,
            NMCSAMPLE as i32,
            NMCSAMPLE,
            1,
            ngi + nvij,
            nvij,
            vmcparams.epsilon
        );

        group.bench_with_input(BenchmarkId::new("Calul <E> par sites", i), &i,
        |b, _| b.iter(||{
            compute_mean_energy(
                        &mut rng,
                        initial_state,
                        &params,
                        &sys,
                        &mut der
                    );
        }
        ));
    }
    group.finish();
}

criterion_group!(benches, bench_monte_carlo);
