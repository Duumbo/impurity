use criterion::{black_box, criterion_group, Criterion};
use impurity::parse::orbitale::parse_orbitale_def;
use impurity::pfaffian::compute_pfaffian_wq;
use impurity::SIZE;
use std::path::Path;

pub fn bench_compute_pfaff(c: &mut Criterion) {
    let orbitale_fp = Path::new("data/orbitale.csv");
    let fij = parse_orbitale_def(&orbitale_fp.to_path_buf(), SIZE).unwrap();
    c.bench_function("Calcul pfaffian 8x8", |b| {
        b.iter(|| compute_pfaffian_wq(black_box(&mut fij.clone()), 1))
    });
}

criterion_group!(benches, bench_compute_pfaff,);
