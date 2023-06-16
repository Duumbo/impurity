use criterion::criterion_main;

mod benchmarks;

criterion_main!(
    benchmarks::pfaffian::benches,
    benchmarks::jastrow::benches,
    benchmarks::gutzwiller::benches,
);
