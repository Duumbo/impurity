use blas::{daxpy, dcopy, ddot, dgemv, dscal};
use log::trace;

use crate::DerivativeOperator;

fn gradient(x: &[f64], a: &DerivativeOperator, b: &mut [f64], dim: i32) {
    let alpha = -1.0;
    let incx = 1;
    let incy = 1;
    let mut work: Vec<f64> = vec![0.0; dim as usize];
    // Compute Ax
    compute_w(&mut work, a, x);
    unsafe {
        // Compute b - Ax
        daxpy(dim, alpha, &work, incx, b, incy);
    }
}

fn update_x(x: &mut [f64], pk: &[f64], alpha: f64, dim: i32) {
    let incx = 1;
    let incy = 1;
    unsafe {
        daxpy(dim, alpha, &pk, incx, x, incy);
    };
}

fn compute_w(w: &mut [f64], a: &DerivativeOperator, p: &[f64]) {
    // Computes Ap
    let incx = 1;
    let incy = 1;
    let alpha = 1.0;
    let beta = 0.0;

    // Temp work vector
    let mut work: Vec<f64> = vec![0.0; a.mu as usize];
    unsafe {
        trace!("x_[n] = {:?}", p);
        trace!("mu = {}, n = {}", a.mu, a.n);
        dgemv(b"T"[0], a.n, a.mu, alpha, a.o_tilde, a.n, p, incx, beta, &mut work, incy);
        for i in 0..a.mu as usize {
            work[i] *= a.visited[i] as f64;
        }
        trace!("O^[T]_[mu, n] x_[n] = {:?}", work);
        trace!("Len(work) = {}, a.n = {}, a.mu = {}", work.len(), a.n, a.mu);
        trace!("~O_[0, mu] = {:?}", a.o_tilde.iter().step_by(a.mu as usize).copied().collect::<Vec<f64>>());
        // Sometimes segfaults
        dgemv(b"N"[0], a.n, a.mu, 1.0 / a.nsamp, a.o_tilde, a.n, &work, incx, beta, w, incy);
        trace!("O_[m, mu] O^[T]_[mu, n] x_[n] = {:?}", w);
        let alpha = ddot(a.n, a.expval_o, incx, p, incy);
        daxpy(a.n, - alpha, a.expval_o, incx, w, incy);
    }
}

fn update_r(rk: &mut [f64], w: &[f64], alphak: f64, dim: i32) {
    let incx = 1;
    let incy = 1;
    unsafe {
        daxpy(dim, - alphak, w, incx, rk, incy)
    }
}

fn alpha_k(r_k: &[f64], p_k: &[f64], w: &[f64], alphak: &mut f64, dim: i32) -> f64 {
    let incx = 1;
    let incy = 1;
    unsafe {
        let rkrk = ddot(dim, r_k, incx, r_k, incy);
        *alphak = rkrk / ddot(dim, p_k, incx, w, incy);
        rkrk
    }
}

fn beta_k(r_k: &[f64], rkrk: f64, dim: i32) -> f64 {
    let incx = 1;
    let incy = 1;
    unsafe {
        ddot(dim, r_k, incx, r_k, incy) / rkrk
    }
}

fn update_p(r_k: &[f64], p_k: &mut [f64], beta: f64, dim: i32) {
    let alpha = 1.0;
    let incx = 1;
    let incy = 1;
    unsafe {
        dscal(dim, beta, p_k, incx);
        daxpy(dim, alpha, r_k, incx, p_k, incy);
    }
}

/// Computes the solution of $A\mathbf{x}-\mathbf{b}=0$
/// TODOC
pub fn conjugate_gradiant(a: &DerivativeOperator, b: &mut [f64], x0: &mut [f64], epsilon: f64, kmax: usize, dim: i32) {
    trace!("Initial guess x_0: {:?}", x0);
    trace!("Initial equation b: {:?}", b);
    let mut w: Vec<f64> = vec![0.0; dim as usize];
    // Error threshold
    let e = unsafe {
        let incx = 1;
        let incy = 1;
        ddot(dim, b, incx, b, incy) * epsilon
    };
    trace!("Error threshold e = {}", e);
    gradient(x0, a, b, dim);
    let mut p = Vec::with_capacity(dim as usize);
    unsafe {
        let incx = 1;
        let incy = 1;
        p.set_len(dim as usize);
        dcopy(dim, b, incx, &mut p, incy)
    }
    let mut alpha = 0.0;

    for k in 0..kmax {
        trace!("r_{} : {:?}", k, b);
        trace!("p_{} : {:?}", k, p);
        compute_w(&mut w, a, &p);
        let nrm2rk = alpha_k(b, &p, &w, &mut alpha, dim);
        trace!("alpha_{} : {}", k, alpha);
        update_x(x0, &p, alpha, dim);
        trace!("x_{} : {:?}", k+1, x0);
        update_r(b, &w, alpha, dim);
        let beta = beta_k(b, nrm2rk, dim);
        if beta * nrm2rk < e {
            trace!("Achieved convergence at {} iterations", k);
            break;
        }
        trace!("beta_{} : {}", k, beta);
        update_p(b, &mut p, beta, dim);
    }
}
