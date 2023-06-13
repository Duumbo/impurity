use crate::SIZE;
use pfapack::skpfa;

pub fn compute_pfaffian_wq(a: &mut [f64]) -> f64 {
    const N: i32 = SIZE as i32;
    let mut pfaff: f64 = 0.0;
    let mut info: i32 = 0;
    let mut iwork: Vec<i32> = Vec::with_capacity(SIZE);

    // Workspace query
    let mut work: Vec<f64> = Vec::with_capacity(1);
    work.push(0.0);
    unsafe {
        skpfa::dskpfa(
            'L' as u8,
            'P' as u8,
            &N,
            a,
            &N,
            &mut pfaff,
            &mut iwork,
            &mut work,
            &(-1),
            &mut info,
        )
    }
    let lwork: i32 = work[0] as i32;
    // Compute using the lower and p method.
    let mut work: Vec<f64> = Vec::with_capacity(lwork as usize);
    unsafe {
        skpfa::dskpfa(
            'L' as u8, 'P' as u8, &N, a, &N, &mut pfaff, &mut iwork, &mut work, &lwork, &mut info,
        )
    }
    assert_eq!(info, 0);
    pfaff
}
