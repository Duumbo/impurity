use crate::{BitStruct, FockState, SIZE, CONS_U, CONS_T};


/// Computes the potential energy term of the Hamiltonian.
pub fn terme_pot(spin_up: BitStruct, spin_down: BitStruct) -> f64 {
    ((spin_up & spin_down).count_ones() as f64) * CONS_U
}

pub fn terme_cin(
    spin_up: BitStruct,
    spin_down: BitStruct,
) -> Vec<FockState> {

    let mut out: Vec<FockState> = Vec::with_capacity(8);
    // Rotate the bits left. We don't have to do the right one, because it give
    // the same truth values, just shifted.
    let spin_up_shl1 = spin_up.rotate_left(1);
    let spin_down_shl1 = spin_down.rotate_left(1);
    let spin_up_shr1 = spin_up.rotate_right(1);
    let spin_down_shr1 = spin_down.rotate_right(1);

    // These values are litteraly e- at i can go right.
    let can_gor_spin_up = spin_up_shl1 ^ spin_up;
    let can_gor_spin_down = spin_down_shl1 ^ spin_down;
    let can_gol_spin_up = spin_up_shr1 ^ spin_up;
    let can_gol_spin_down = spin_down_shr1 ^ spin_down;

    out.append(
        &mut tm(spin_up, can_gor_spin_up, 2).into_iter()
        .map(|s| FockState{spin_up: s, spin_down})
        .collect::<Vec<FockState>>()
    );
    out.append(
        &mut tm(spin_down, can_gor_spin_down, 2)
        .into_iter()
        .map(|s| FockState{spin_up, spin_down: s})
        .collect::<Vec<FockState>>()
    );
    out.append(
        &mut tm(spin_up, can_gol_spin_up, 1)
        .into_iter()
        .map(|s| FockState{spin_up: s, spin_down})
        .collect::<Vec<FockState>>()
    );
    out.append(
        &mut tm(spin_down, can_gol_spin_down, 1)
        .into_iter()
        .map(|s| FockState{spin_up, spin_down: s})
        .collect::<Vec<FockState>>()
    );

    out
}

fn tm(spin: BitStruct, mut truth: BitStruct, shl_qt: usize) -> Vec<BitStruct> {
    let mut i = truth.leading_zeros();
    let mut out_vec: Vec<BitStruct> = Vec::with_capacity(8);
    while (i as usize) < SIZE {
        let n = SIZE as i32 - shl_qt as i32 - i as i32;
        let mask;
        if n < 0 {
            mask = <BitStruct>::from(3).rotate_right(n.abs() as u32);
        } else {
            mask = <BitStruct>::from(3).rotate_left(n as u32);
        }
        out_vec.push(spin ^ mask);
        truth ^= 1 << (SIZE - 1 - i as usize);
        i = truth.leading_zeros();
    }
    out_vec
}
