use std::{fmt::{Debug, Display}, ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign}};
use blas::{daxpy, dcopy, ddot, dgemm, dgemv, dger, dnrm2, dscal, dznrm2, idamax, izamax, zaxpy, zcopy, zdotc, zgemm, zgemv, zgerc, zscal, zdotu};
use num::complex::{Complex, ComplexFloat};
use lapack::{dgetrf, dgetri, dsyev, zgeev, zgetrf, zgetri};
use pfapack::skpfa;

pub trait Parameter:
    Add<Output = Self> +
    AddAssign +
    SubAssign +
    MulAssign +
    Mul<Output = Self> +
    Mul<f64, Output = Self> +
    Div<Output = Self> +
    Neg<Output = Self> +
    PartialEq +
    From<f64> +
    Copy +
    Sub<Output = Self> +
    Display +
    Debug +
    Send +
    Sync +
    Sized
{
    fn into_i32(&self) -> i32;
    fn nrmsq(&self) -> f64;
    fn is_finite(self) -> bool;
    fn abs(self) -> f64;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn sqrt(self) -> Self;
    fn re(self) -> f64;
    unsafe fn getrf(m: i32, n: i32, a: &mut [Self], lda: i32, ipiv: &mut [i32], info: &mut i32);
    unsafe fn getri(n: i32, a: &mut [Self], lda: i32, ipiv: &[i32], work: &mut [Self], lwork: i32, info: &mut i32);
    unsafe fn dot(n: i32, x: &[Self], incx: i32, y: &[Self], incy: i32) -> Self;
    unsafe fn gemv(trans: u8, m: i32, n: i32, alpha: Self, a: &[Self], lda: i32, x: &[Self], incx: i32, beta: Self, y: &mut [Self], incy: i32);
    unsafe fn ger(m: i32, n: i32, alpha: Self, x: &[Self], incx: i32, y: &[Self], incy: i32, a: &mut [Self], lda: i32);
    unsafe fn axpy(n: i32, alpha: Self, x: &[Self], incx: i32, y: &mut [Self], incy: i32);
    unsafe fn skpfa(uplo: u8, mthd: u8, n: &i32, a: &mut [Self], lda: &i32, pfaff: &mut Self, iwork: &mut [i32], work: &mut [Self], lwork: &i32, info: &mut i32);
    unsafe fn copy(n: i32, x: &[Self], incx: i32, y: &mut [Self], incy: i32);
    unsafe fn nrm2(n: i32, x: &[Self], incx: i32) -> f64;
    unsafe fn iamax(n: i32, x: &[Self], incx: i32) -> usize;
    unsafe fn scal(n: i32, a: Self, x: &mut [Self], incx: i32);
    unsafe fn gemm(transa: u8, transb: u8, m: i32, n: i32, k: i32, alpha: Self, a: &[Self], lda: i32, b: &[Self], ldb: i32, beta: Self, c: &mut [Self], ldc: i32);
    unsafe fn diag(jobz: u8, uplo: u8, n: i32, a: &mut [Self], lda: i32, w: &mut [Self], work: &mut [Self], lwork: i32, info: &mut i32);
}

pub trait Float32: Add<Output = Self> + Sized {}
pub trait Float64: Add<Output = Self> + Sized {}
pub trait Complex32: Add<Output = Self> + Sized {}
pub trait Complex64: Add<Output = Self> + Sized {}

impl Float32 for f32 {}
impl Float64 for f64 {}
impl Complex32 for Complex<f32> {}
impl Complex64 for Complex<f64> {}

//impl Parameter for f32 {}
impl Parameter for f64 {

    fn re(self) -> f64 {
        self
    }

    fn sqrt(self) -> Self {
        self.sqrt()
    }

    fn ln(self) -> Self {
        self.ln()
    }

    fn exp(self) -> Self {
        self.exp()
    }

    fn abs(self) -> f64 {
        self.abs()
    }

    fn is_finite(self) -> bool {
        <f64>::is_finite(self)
    }

    fn nrmsq(&self) -> f64 {
        *self * *self
    }

    fn into_i32(&self) -> i32 {
        *self as i32
    }

    unsafe fn copy(n: i32, x: &[f64], incx: i32, y: &mut [f64], incy: i32) {
        dcopy(n, x, incx, y, incy)
    }

    unsafe fn getrf(m: i32, n: i32, a: &mut [f64], lda: i32, ipiv: &mut [i32], info: &mut i32) {
        dgetrf(m, n, a, lda, ipiv, info)
    }

    unsafe fn getri(n: i32, a: &mut [f64], lda: i32, ipiv: &[i32], work: &mut [f64], lwork: i32, info: &mut i32) {
        dgetri(n, a, lda, ipiv, work, lwork, info)
    }

    unsafe fn dot(n: i32, x: &[f64], incx: i32, y: &[f64], incy: i32) -> f64 {
        ddot(n, x, incx, y, incy)
    }

    unsafe fn gemv(trans: u8, m: i32, n: i32, alpha: f64, a: &[f64], lda: i32, x: &[f64], incx: i32, beta: f64, y: &mut [f64], incy: i32) {
        dgemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy)
    }

    unsafe fn ger(m: i32, n: i32, alpha: f64, x: &[f64], incx: i32, y: &[f64], incy: i32, a: &mut [f64], lda: i32) {
        dger(m, n, alpha, x, incx, y, incy, a, lda)
    }

    unsafe fn axpy(n: i32, alpha: f64, x: &[f64], incx: i32, y: &mut [f64], incy: i32) {
        daxpy(n, alpha, x, incx, y, incy)
    }

    unsafe fn skpfa(uplo: u8, mthd: u8, n: &i32, a: &mut [f64], lda: &i32, pfaff: &mut f64, iwork: &mut [i32], work: &mut [f64], lwork: &i32, info: &mut i32) {
        skpfa::dskpfa(uplo, mthd, n, a, lda, pfaff, iwork, work, lwork, info)
    }

    unsafe fn nrm2(n: i32, x: &[Self], incx: i32) -> f64 {
        dnrm2(n, x, incx)
    }

    unsafe fn iamax(n: i32, x: &[Self], incx: i32) -> usize {
        idamax(n, x, incx)
    }

    unsafe fn scal(n: i32, a: Self, x: &mut [Self], incx: i32) {
        dscal(n, a, x, incx)
    }

    unsafe fn gemm(transa: u8, transb: u8, m: i32, n: i32, k: i32, alpha: Self, a: &[Self], lda: i32, b: &[Self], ldb: i32, beta: Self, c: &mut [Self], ldc: i32) {
        dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
    }

    unsafe fn diag(jobz: u8, uplo: u8, n: i32, a: &mut [Self], lda: i32, w: &mut [Self], work: &mut [Self], lwork: i32, info: &mut i32) {
        dsyev(jobz, uplo, n, a, lda, w, work, lwork, info)
    }

}
//impl Parameter for Complex<f32> {}
impl Parameter for Complex<f64> {

    fn re(self) -> f64 {
        ComplexFloat::re(self)
    }

    fn sqrt(self) -> Self {
        self.sqrt()
    }

    fn ln(self) -> Self {
        self.ln()
    }

    fn exp(self) -> Self {
        self.exp()
    }

    fn abs(self) -> f64 {
        ComplexFloat::abs(self) as f64
    }

    fn is_finite(self) -> bool {
        <Complex<f64>>::is_finite(self)
    }

    fn into_i32(&self) -> i32 {
        ComplexFloat::re(*self) as i32
    }

    unsafe fn getrf(m: i32, n: i32, a: &mut [Complex<f64>], lda: i32, ipiv: &mut [i32], info: &mut i32) {
        zgetrf(m, n, a, lda, ipiv, info)
    }

    unsafe fn getri(n: i32, a: &mut [Complex<f64>], lda: i32, ipiv: &[i32], work: &mut [Complex<f64>], lwork: i32, info: &mut i32) {
        zgetri(n, a, lda, ipiv, work, lwork, info)
    }

    unsafe fn dot(n: i32, x: &[Complex<f64>], incx: i32, y: &[Complex<f64>], incy: i32) -> Complex<f64> {
        let mut pres = vec![<Complex<f64>>::from(0.0); 1];
        //let mut pres = <Complex<f64>>::from(0.0);
        zdotc(&mut pres, n, x, incx, y, incy);
        println!("x = {:?}", x);
        println!("y = {:?}", y);
        println!("incx = {:?}", incx);
        println!("incy = {:?}", incy);
        println!("pres = {:?}", pres);
        pres[0]
    }

    unsafe fn gemv(trans: u8, m: i32, n: i32, alpha: Complex<f64>, a: &[Complex<f64>], lda: i32, x: &[Complex<f64>], incx: i32, beta: Complex<f64>, y: &mut [Complex<f64>], incy: i32) {
        zgemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy)
    }

    unsafe fn ger(m: i32, n: i32, alpha: Complex<f64>, x: &[Complex<f64>], incx: i32, y: &[Complex<f64>], incy: i32, a: &mut [Complex<f64>], lda: i32) {
        zgerc(m, n, alpha, x, incx, y, incy, a, lda)
    }

    unsafe fn axpy(n: i32, alpha: Complex<f64>, x: &[Complex<f64>], incx: i32, y: &mut [Complex<f64>], incy: i32) {
        zaxpy(n, alpha, x, incx, y, incy)
    }

    unsafe fn skpfa(uplo: u8, mthd: u8, n: &i32, a: &mut [Complex<f64>], lda: &i32, pfaff: &mut Complex<f64>, iwork: &mut [i32], work: &mut [Complex<f64>], lwork: &i32, info: &mut i32) {
        let mut rwork = vec![0.0; *n as usize];
        skpfa::zskpfa(uplo, mthd, n, a, lda, pfaff, iwork, work, lwork, &mut rwork, info)
    }

    unsafe fn copy(n: i32, x: &[Self], incx: i32, y: &mut [Self], incy: i32) {
        zcopy(n, x, incx, y, incy)
    }

    fn nrmsq(&self) -> f64 {
        self.norm_sqr()
    }

    unsafe fn nrm2(n: i32, x: &[Self], incx: i32) -> f64 {
        dznrm2(n, x, incx)
    }

    unsafe fn iamax(n: i32, x: &[Self], incx: i32) -> usize {
        izamax(n, x, incx)
    }

    unsafe fn scal(n: i32, a: Self, x: &mut [Self], incx: i32) {
        zscal(n, a, x, incx)
    }

    unsafe fn gemm(transa: u8, transb: u8, m: i32, n: i32, k: i32, alpha: Self, a: &[Self], lda: i32, b: &[Self], ldb: i32, beta: Self, c: &mut [Self], ldc: i32) {
        zgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
    }

    unsafe fn diag(jobz: u8, _uplo: u8, n: i32, a: &mut [Self], lda: i32, w: &mut [Self], work: &mut [Self], lwork: i32, info: &mut i32) {
        let jobvl = b"N"[0];
        // Not referenced
        let mut vl = Vec::new();
        let ldvl = 0;
        // Store temporarily the eigenvectors is vr
        let mut vr = vec![<Self>::from(0.0); (n*n) as usize];
        let ldvr = n;
        // Other work vector
        let mut rwork = vec![0.0; 2*n as usize];
        zgeev(jobvl, jobz, n, a, lda, w, &mut vl, ldvl, &mut vr, ldvr, work, lwork, &mut rwork, info);
        let incx = 1;
        let incy = 1;
        zcopy(n*n, &vr, incx, a, incy);
    }

}
