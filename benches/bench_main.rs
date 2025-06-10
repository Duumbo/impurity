use criterion::criterion_main;

mod benchmarks;

criterion_main!(
    benchmarks::monte_carlo::benches,
    benchmarks::conjugate_gradient::benches,
    benchmarks::pfaffian::benches,
    benchmarks::jastrow::benches,
    benchmarks::gutzwiller::benches,
);
