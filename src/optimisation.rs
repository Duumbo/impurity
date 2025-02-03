use blas::{daxpy, dcopy, ddot, dgemm, dgemv, dger, dscal};
use lapack::dsyev;
use log::{error, trace};
use colored::Colorize;
use std::fs::File;
use std::io::prelude::*;

use crate::DerivativeOperator;

fn gradient(x: &[f64], otilde: &[f64], visited: &[usize], expval_o: &[f64], b: &mut [f64], dim: i32, mu: i32, nsamp: f64) {
    let alpha = -1.0;
    let incx = 1;
    let incy = 1;
    let mut work: Vec<f64> = vec![0.0; dim as usize];
    // Compute Ax
    compute_w(&mut work, otilde, visited, expval_o, x, dim, mu, nsamp);
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

fn compute_w(w: &mut [f64], otilde: &[f64], visited: &[usize], expval_o: &[f64], p: &[f64], dim: i32, mu: i32, nsamp: f64) {
    // Computes Ap
    let incx = 1;
    let incy = 1;
    let alpha = 1.0;
    let beta = 0.0;

    // Temp work vector
    let mut work: Vec<f64> = vec![0.0; mu as usize];
    unsafe {
        trace!("x_[n] = {:?}", p);
        trace!("mu = {}, n = {}", mu, dim);
        // 80 misawa
        dgemv(b"T"[0], dim, mu, alpha, otilde, dim, p, incx, beta, &mut work, incy);
        for i in 0..mu as usize {
            work[i] *= visited[i] as f64;
        }
        trace!("O^[T]_[mu, n] x_[n] = {:?}", work);
        trace!("Len(work) = {}, a.n = {}, a.mu = {}", work.len(), dim, mu);
        trace!("~O_[0, mu] = {:?}", otilde.iter().step_by(mu as usize).copied().collect::<Vec<f64>>());
        // Sometimes segfaults
        dgemv(b"N"[0], dim, mu, 1.0 / nsamp, otilde, dim, &work, incx, beta, w, incy);
        trace!("O_[m, mu] O^[T]_[mu, n] x_[n] = {:?}", w);
        let alpha = ddot(dim, expval_o, incx, p, incy);
        // 81 misawa
        daxpy(dim, - alpha, expval_o, incx, w, incy);
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

/// Adds an offset to the diagonal of the overlap matrix S_[k,m].
/// TODOC
/// see tahara 2008 3.3.1
/// $$
/// \begin{align}
/// \widetilde{S}_{k,m}&=\epsilon\delta_{k,m}+S_{k,m}\\
/// &=\epsilon\delta_{k,\mu}\delta{\mu,m}+\widetilde{O}^*_{k,\mu}\widetilde{O}^T_{\mu,m}
/// \end{align}
/// $$
pub fn spread_eigenvalues(a: &mut DerivativeOperator) {
    let epsilon = a.epsilon;
    let n_elem = {
        if a.mu > a.n {a.n}
        else {a.mu}
    };
    let work = vec![epsilon; n_elem as usize];
    unsafe {
        let alpha = 1.0;
        let incx = 1;
        let incy = a.n + 1;
        daxpy(n_elem, alpha, &work, incx, &mut a.o_tilde, incy);
    }
}

fn prefilter_overlap_matrix(a: &DerivativeOperator, ignore_idx: &mut [bool], dim: i32, diag_threshold: f64) -> usize {
    // Loop over diagonal elements of S_km
    // Reminder, S_{kk} = 1/N_{\rm MC.} \sum_\mu \tilde{O}^*_{k\mu}\tilde{O}^T_{\mu k} -
    // \Re{\expval{O_k}}^2

    let mut skip_param_count: usize = 0;
    let mut diag_elem = vec![0.0; dim as usize];
    for k in 0..dim as usize {
        // Start with 1/N_{\rm MC.} \sum_\mu \tilde{O}^*_{k\mu}\tilde{O}^T_{\mu k}
        let z1: f64 = unsafe {
            ddot(
                a.mu,
                &a.o_tilde[k..a.n as usize + a.mu as usize * (k + 1)],
                a.n,
                &a.o_tilde[k..a.n as usize + a.mu as usize * (k + 1)],
                a.n
            )
        };

        // Now \Re{\expval{O_k}}^2
        let z2: f64 = a.expval_o[k] * a.expval_o[k];

        diag_elem[k] = z1 - z2;
    }

    let mut max_elem = <f64>::MIN;
    for k in 0..dim as usize {
        if diag_elem[k] > max_elem {
            max_elem = diag_elem[k];
        }
    }
    let threshold = diag_threshold * max_elem;
    for k in 0..dim as usize {
        if diag_elem[k] < threshold {
            skip_param_count += 1;
            ignore_idx[k] = true;
        }
    }
    dim as usize - skip_param_count

}

fn cpy_segmented_matrix_to_dense(a: &DerivativeOperator, output_otilde: &mut [f64], output_expvalo: &mut [f64], ignore_idx: &[bool], dim: i32, nparams_opt: usize) {
    let mut j: usize = 0;
    for k in 0..dim as usize {
        if ignore_idx[k] {
            continue;
        }
        unsafe {
            dcopy(
                a.mu,
                &a.o_tilde[k..a.n as usize + a.mu as usize * (k + 1)],
                a.n,
                &mut output_otilde[j..a.mu as usize * (j+1)],
                nparams_opt as i32
            );
        }
        output_expvalo[j] = a.expval_o[k];
        j += 1;

    }
}

fn compute_s_explicit(otilde: &[f64], expval_o: &[f64], visited: &[usize], dim: i32, mu: i32, nsamp: f64, epsilon: f64) -> Vec<f64> {
    // Computes Ap
    let incx = 1;
    let alpha = 1.0/nsamp;
    let beta = -1.0;
    let mut otilde_w_visited = vec![0.0; (dim * mu) as usize];
    for i in 0..mu as usize {
        for j in 0..dim as usize {
            otilde_w_visited[j + i*dim as usize] = otilde[j + i*dim as usize] * visited[i] as f64;
        }
    }

    // Temp work vector
    let mut work = vec![0.0; (dim * dim) as usize];
    unsafe {
        // 80 misawa
        dger(dim, dim, 1.0, &expval_o, incx, &expval_o, incx, &mut work, dim);
        dgemm(b"N"[0], b"T"[0], dim, dim, mu, alpha, &otilde_w_visited, dim, &otilde_w_visited, dim, beta, &mut work, dim);
    }
    // Shift smallest eigenvalues
    for i in 0..dim as usize {
        work[i + dim as usize * i] += epsilon;
    }
    work
}

fn save_otilde(der: &[f64], mu: usize, n: usize) -> String {
    let width = 16;
    let mut o_tilde = "".to_owned();
    for m in 0..mu {
        for i in 0..n  {
            if i == m {
                o_tilde.push_str(&format!("{:>width$.04e}", der[i + m * n]).yellow());
            } else {
                o_tilde.push_str(&format!("{:>width$.04e}", der[i + m * n]));
            }
        }
        o_tilde.push_str("\n");
    }
    o_tilde
}

fn diagonalize_dense_matrix(s: &mut [f64], dim: i32) -> Vec<f64> {
    let jobz = b"V"[0];
    let uplo = b"U"[0];
    let mut w = vec![0.0; dim as usize];
    let lwork = 3*(dim);
    let mut work = vec![0.0; lwork as usize];
    let mut info = 0;
    unsafe {
        dsyev(
            jobz,
            uplo,
            dim,
            s,
            dim,
            &mut w,
            &mut work,
            lwork,
            &mut info
        );
    }
    if info < 0{
        error!("Parameter {} had an illegal value in call to lapack::dsyev.", <i32>::abs(info));
    }
    else if info > 0{
        error!("Convergence was not achieved to diagonalize the overlap matrix.
            {} off-diagonal elements did not converge to 0.", info);
    }

    w
}

fn compute_delta_from_eigenvalues(x0: &mut [f64], eigenvectors: &[f64], eigenvalues: &[f64], dim: i32) {
    let trans = b"T"[0];
    let incx = 1;
    let incy = 1;
    let mut work = vec![0.0; dim as usize];
    unsafe {
        dgemv(trans, dim, dim, 1.0, eigenvectors, dim, x0, incx, 0.0, &mut work, incy);
    }
    for i in 0..dim as usize{
        work[i] *= eigenvalues[i];
    }
    let trans = b"N"[0];
    unsafe {
        dgemv(trans, dim, dim, 1.0, eigenvectors, dim, &work, incx, 0.0, x0, incy);
    }
}

fn compute_matrix_product(s: &mut [f64], eigenvectors: &[f64], eigenvalues: &[f64], dim: i32) {
    let transa = b"T"[0];
    let transb = b"T"[0];
    let incx = 1;
    let incy = 1;
    let mut work = vec![0.0; (dim * dim) as usize];
    let mut s_copy = vec![0.0; (dim * dim) as usize];
    let mut work_direct = vec![0.0; (dim * dim) as usize];
    let mut s_copy_direct = vec![0.0; (dim * dim) as usize];
    unsafe {
        dcopy(dim*dim, eigenvectors, incx, &mut work, incy);
        dcopy(dim*dim, eigenvectors, incx, &mut work_direct, incy);
        //dgemm(transa, transb, dim, dim, dim, 1.0, eigenvectors, dim, s, dim, 0.0, &mut s_copy, dim);
    }
    for i in 0..dim as usize{
        for j in 0..dim as usize {
           if eigenvalues[i] < 1e-1 {
           work[j + dim as usize * i] *= 0.0;
           work_direct[j + dim as usize * i] *= 0.0;
           continue;
           }
            work[j + dim as usize * i] *= 1.0 / eigenvalues[i];
            work_direct[j + dim as usize * i] *= eigenvalues[i];
        }
    }
    let transa = b"N"[0];
    unsafe {
        dgemm(transa, transb, dim, dim, dim, 1.0, eigenvectors, dim, &work, dim, 0.0, &mut s_copy, dim);
        dgemm(transa, transb, dim, dim, dim, 1.0, eigenvectors, dim, &work_direct, dim, 0.0, &mut s_copy_direct, dim);
        //dcopy(dim*dim, s, incx, &mut work, incy);
    let transa = b"N"[0];
    let transb = b"T"[0];
        dgemm(transa, transb, dim, dim, dim, 1.0, &s_copy, dim, &s_copy_direct, dim, 0.0, s, dim);
    }
}

/// Computes the solution of $A\mathbf{x}-\mathbf{b}=0$
/// TODOC
pub fn conjugate_gradiant(a: &DerivativeOperator, b: &mut [f64], x0: &mut [f64], epsilon: f64, kmax: usize, dim: i32, thresh: f64) -> Vec<bool>{
    // PRE FILTER
    let mut ignore = vec![false; dim as usize];
    //println!("{}", save_otilde(a.o_tilde, a.mu as usize, a.n as usize));
    let mut unfiltered_s = compute_s_explicit(a.o_tilde, a.expval_o, a.visited, dim, a.mu, a.nsamp, epsilon);
    let mut s_copy = unfiltered_s.clone();
    //println!("dim = {}, Unfiltered S = ", dim);
    //println!("{}", save_otilde(&unfiltered_s, dim as usize, dim as usize));
    let new_dim = prefilter_overlap_matrix(a, &mut ignore, dim, thresh);
    //println!("ignore : {:?}", ignore);
    let mut otilde = vec![0.0; new_dim * a.mu as usize];
    let mut expvalo = vec![0.0; new_dim];
    cpy_segmented_matrix_to_dense(a, &mut otilde, &mut expvalo, &ignore, dim, new_dim);
    //let filtered_s = compute_s_explicit(&otilde, &expvalo, a.visited, new_dim as i32, a.mu, a.nsamp);
    let mut eigenvalues = diagonalize_dense_matrix(&mut unfiltered_s, dim);
    //let eigenvectors = &unfiltered_s;
    //println!("S = \n{}", save_otilde(&s_copy, dim as usize, dim as usize));
    //compute_matrix_product(&mut s_copy, &eigenvectors, &eigenvalues, dim);
    //println!("S^[-1] = \n{}", save_otilde(&s_copy, dim as usize, dim as usize));
    //panic!("stop");
    //let mut x0_raw = vec![0.0; dim as usize];
    //unsafe {
    //    let incx = 1;
    //    let incy = 1;
    //    dgemv(b"T"[0], dim, dim, 1.0, &s_copy, dim, x0, incx, 0.0, &mut x0_raw, incy);
    //}
    //println!("UD^[-1]U^[T] = \n{}", save_otilde(&s_copy, dim as usize, dim as usize));
    //println!("x0 = {:?}", x0_raw);
    //println!("eigenvalues: {:?}", eigenvalues);

    // Remove problematic eigenvalue
    let mut max = <f64>::MIN;
    for e in eigenvalues.iter() {
        if *e > max {
            max = *e;
        }
    }
    let threshold = thresh * max;
    for e in eigenvalues.iter_mut() {
        if *e < threshold {
            *e = 0.0;
        }
    }
    //Invert matrix
    for e in eigenvalues.iter_mut() {
        if *e == 0.0 {
            continue;
        }
        *e = 1.0 / *e;
    }
    compute_delta_from_eigenvalues(x0, &unfiltered_s, &eigenvalues, dim);
    return ignore;
    let mut fp = File::create("overlap").unwrap();
    //fp.write_all(save_otilde(&filtered_s, new_dim as usize, new_dim as usize).as_bytes()).unwrap();
    println!("dim = {}, Filtered S = ", new_dim);
    //println!("{}", save_otilde(&filtered_s, new_dim as usize, new_dim as usize));
    //println!("{}", save_otilde(&otilde, a.mu as usize, new_dim as usize));

    trace!("Initial guess x_0: {:?}", x0);
    trace!("Initial equation b: {:?}", b);
    let mut w: Vec<f64> = vec![0.0; new_dim];
    // Error threshold
    let mut e = 0.0;
    for i in 0..dim as usize {
        if ignore[i] {
            continue;
        }
        e += b[i] * b[i];
    }
    e *= epsilon;
    trace!("Error threshold e = {}", e);
    //println!("Error threshold e = {}", e);
    gradient(x0, &otilde, a.visited, &expvalo, b, new_dim as i32, a.mu, a.nsamp);
    let mut p = vec![0.0; new_dim];
    let mut j: usize = 0;
    for i in 0..dim as usize {
        if ignore[i] {
            continue;
        }
        p[j] = b[i];
        j += 1;
    }
    unsafe {
        dcopy(new_dim as i32, &p, 1, b, 1);
    }
    let mut alpha = 0.0;

    for k in 0..kmax {
        trace!("r_{} : {:?}", k, b);
        trace!("p_{} : {:?}", k, p);
        //println!("r_{} : {:?}", k, b);
        //println!("p_{} : {:?}", k, p);
        compute_w(&mut w, &otilde, a.visited, &expvalo, &p, new_dim as i32, a.mu, a.nsamp);
        //println!("w_{} : {:?}", k, w);
        let nrm2rk = alpha_k(b, &p, &w, &mut alpha, new_dim as i32);
        trace!("alpha_{} : {}", k, alpha);
        //println!("alpha_{} : {}", k, alpha);
        if alpha < 0.0 {
            //error!("Input overlap matrix S was not positive-definite.");
            break;
            //panic!("p^T S p < 0.0");
        }
        update_x(x0, &p, alpha, new_dim as i32);
        trace!("x_{} : {:?}", k+1, x0);
        //println!("x_{} : {:?}", k+1, x0);
        update_r(b, &w, alpha, new_dim as i32);
        let beta = beta_k(b, nrm2rk, new_dim as i32);
        if beta * nrm2rk < e {
            trace!("Achieved convergence at {} iterations", k);
            break;
        }
        trace!("beta_{} : {}", k, beta);
        update_p(b, &mut p, beta, new_dim as i32);
    }
    ignore
}
