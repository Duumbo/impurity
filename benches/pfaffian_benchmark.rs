use criterion::{black_box, criterion_group, criterion_main, Criterion};
use impurity::SIZE;
use impurity::pfaffian::compute_pfaffian_wq;
use impurity::parse::orbitale::parse_orbitale_def;
use std::path::Path;

pub fn bench_compute_pfaff(c: &mut Criterion) {
    let orbitale_fp = Path::new("data/orbitale.csv");
    c.bench_function("Calcul pfaffian 8x8", |b| b.iter(|| {
            let mut fij = parse_orbitale_def(&orbitale_fp.to_path_buf(), SIZE).unwrap();
            compute_pfaffian_wq(black_box(&mut fij))
        })
    );
}

criterion_group!(benches, bench_compute_pfaff);
criterion_main!(benches);
