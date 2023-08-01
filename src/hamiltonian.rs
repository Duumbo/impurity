use crate::{BitOps, FockState, CONS_U, SIZE};

/// Computes the potential energy term of the Hamiltonian.
pub fn terme_pot<T>(spin_up: T, spin_down: T) -> f64
where
    T: BitOps,
{
    ((spin_up & spin_down).count_ones() as f64) * CONS_U
}

pub fn terme_cin<T>(spin_up: T, spin_down: T) -> Vec<FockState<T>>
where
    T: BitOps + From<u8>,
{
    let mut out: Vec<FockState<T>> = Vec::with_capacity(8);
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
        &mut tm(spin_up, can_gor_spin_up, 2)
            .into_iter()
            .map(|s| FockState {
                spin_up: s,
                spin_down,
                n_sites: SIZE,
            })
            .collect::<Vec<FockState<T>>>(),
    );
    out.append(
        &mut tm(spin_down, can_gor_spin_down, 2)
            .into_iter()
            .map(|s| FockState {
                spin_up,
                spin_down: s,
                n_sites: SIZE,
            })
            .collect::<Vec<FockState<T>>>(),
    );
    out.append(
        &mut tm(spin_up, can_gol_spin_up, 1)
            .into_iter()
            .map(|s| FockState {
                spin_up: s,
                spin_down,
                n_sites: SIZE,
            })
            .collect::<Vec<FockState<T>>>(),
    );
    out.append(
        &mut tm(spin_down, can_gol_spin_down, 1)
            .into_iter()
            .map(|s| FockState {
                spin_up,
                spin_down: s,
                n_sites: SIZE,
            })
            .collect::<Vec<FockState<T>>>(),
    );

    out
}

fn tm<T>(spin: T, mut truth: T, shl_qt: usize) -> Vec<T>
where
    T: BitOps + From<u8>,
{
    let mut i = truth.leading_zeros();
    let mut out_vec: Vec<T> = Vec::with_capacity(8);
    while (i as usize) < SIZE {
        let n = SIZE as i32 - shl_qt as i32 - i as i32;
        let mask;
        if n < 0 {
            mask = T::from(3).rotate_right(n.abs() as u32);
        } else {
            mask = T::from(3).rotate_left(n as u32);
        }
        out_vec.push(spin ^ mask);
        truth ^= T::from(1 << (SIZE - 1 - i as usize));
        i = truth.leading_zeros();
    }
    out_vec
}
