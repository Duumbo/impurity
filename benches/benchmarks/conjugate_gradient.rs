use criterion::{criterion_group, Criterion, BenchmarkId};
use impurity::optimisation::conjugate_gradiant;
use impurity::dvmc::{VMCParams, EnergyComputationMethod, EnergyOptimisationMethod};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use impurity::DerivativeOperator;
const MAX_SITES: usize = 128;

pub fn bench_conjugate_gradient(c: &mut Criterion) {
    let mut group = c.benchmark_group("Inversion de la matrice de recouvrement");
    group.significance_level(0.1).sample_size(500);
    let mut rng = SmallRng::seed_from_u64(42);

    const NMCSAMPLE: f64 = 1000.0;

    // Setup the loop over the number of sites
    for i in 2..MAX_SITES {
        let ngi = i;
        let nvij = i * ( i - 1 ) / 2;
        let nfij = i * i;
        let nparams = ngi + nvij + nfij;
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
        for i in 0..nparams* NMCSAMPLE as usize {
            der.o_tilde[i] = rng.gen();
        }
        let mut b_vec = vec![0.0; nparams];
        let mut x0 = vec![0.0; nparams];
        for i in 0..nparams {
            der.expval_o[i] = rng.gen();
            der.ho[i] = rng.gen();
            b_vec[i] = rng.gen();
            x0[i] = rng.gen();
        }

        group.bench_with_input(BenchmarkId::new("Inversion par Gradient Conjugué", i), &i,
        |b, _| b.iter(||{
        conjugate_gradiant(&der, &mut b_vec, &mut x0, vmcparams.epsilon, vmcparams.kmax, vmcparams.nparams as i32, vmcparams.threshold, vmcparams.epsilon_cg)
        }
        ));
    }
    group.finish();
}

criterion_group!(benches, bench_conjugate_gradient);
