<div align="center">

# Impurity

![Rust](https://img.shields.io/badge/rust-%23000000.svg?style=for-the-badge&logo=rust&logoColor=white)

![Tests](https://img.shields.io/github/actions/workflow/status/Duumbo/impurity/rust.yml?color=%23a3d1af&style=for-the-badge)<!-- ![release](https://img.shields.io/github/v/tag/Duumbo/impurity?color=blue&style=for-the-badge) -->
![PULLREQUESTS](https://img.shields.io/github/issues-pr-closed/Duumbo/impurity?color=pink&style=for-the-badge) <!-- ![CRATES](https://img.shields.io/crates/v/pfapack?style=for-the-badge) -->

</div>

## Purpose
Implementation of an impurity solver for the Hubbard model.

### Requirements
- lapack
- blas
- pfapack or gfortran
You can choose your implementation for blas, I use intel-mkl. Simply modify the `Cargo.toml` for your preference.

## Build
To build, simply
```shell
cargo build -r
```
To run the main script,
```shell
cargo run -r
```

# Run Tests
To run the tests, this includes unit tests, tests inside the tests directory and
the exemple in the documentation, run
```shell
cargo test
```

# Speed Benchmarks
Speed micro-benchmarks are implemented in the benches directory. To see if an
implementation improved or regressed, you need to run both implementation with
```shell
cargo bench
```
To implement a new micro-bench, simply add a new benchmark in the appropriate
file in the `benches/benchmarks` directory, then add the function to the
benchmark main.

## Todo
See my [Trello](https://trello.com/b/hCw6jDse/impurity-solver).
