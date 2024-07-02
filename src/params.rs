/// Size of the system.
pub const SIZE: usize = 4;
pub const ARRAY_SIZE: usize = (SIZE + 7) / 8;
pub type BitStruct = u8;
pub const NBITS: usize = 8;

/// Hubbard's model $U$ parameter
pub static CONS_U: f64 = 1.0;
/// Hubbard's model $t$ parameter
pub static CONS_T: f64 = -1.0;

/// Collection of the variationnal parameters
pub struct VarParams {
    pub fij: *const f64,
    pub vij: *const f64,
    pub gi: *const f64,
}

pub const NFIJ: usize = 4*SIZE*SIZE;
pub const NVIJ: usize = SIZE*SIZE;
pub const NGI: usize = SIZE;

pub static mut FIJ: [f64; NFIJ] = [
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    //1.0, 2.0, 3.0, 4.0,
    //1.0, -0.5, 1.0, -0.5,
    //10.0, -1.5, 2.0, -3.5,
    //2.0, -3.5, 10.0, -4.5,
1.078313550425773	,
0.007172274681240365,
0.028714778076311877,
0.09168843535310542	,
0.04813118562079141	,
1.0625398526882723	,
0.08433353658389342	,
0.002722470871706029,
0.07270002762085896	,
0.026989164590497917,
0.007555596176108393,
0.046284058565227465,
0.011127921360085048,
0.07287939415825727	,
0.08138828369394709	,
0.012799567556772274,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
];

pub static mut VIJ: [f64; NVIJ] = [
    0.0, 2.0, 2.0, 2.0,
    2.0, 0.0, 2.0, 2.0,
    2.0, 2.0, 0.0, 2.0,
    2.0, 2.0, 2.0, 0.0,
];

pub static mut GI: [f64; NGI] = [
    -1.0, -1.0, -1.0, -1.0
];

pub const HOPPINGS: [f64; SIZE*SIZE] = [
    0.0, 1.0, 1.0, 0.0,
    1.0, 0.0, 0.0, 1.0,
    1.0, 0.0, 0.0, 1.0,
    0.0, 1.0, 1.0, 0.0
];

pub const HOP_BITMASKS: [BitStruct; SIZE / 2] = {
    let mut hop_tmp: [BitStruct; SIZE / 2] = [BitStruct::from_be(0); SIZE / 2];
    // Index for array
    let mut i: usize = 0;
    while i < SIZE / 2 {
        hop_tmp[i] = {
            let mut mask = BitStruct::from_be(0);
            let one = BitStruct::from_be(1);
            let mut j: usize = 0;
            while j < (SIZE - 1 - i) {
                if HOPPINGS[j + i + 1 + SIZE * j] != 0.0 {
                    if i == 0 {
                        mask ^= one << (NBITS - j - 2);
                    } else {
                        mask ^= one << (NBITS - j - 1);
                    }
                }
                j += 1;
            }
            if i == 0 && HOPPINGS[SIZE - 1] != 0.0 {
                mask ^= one << (NBITS - 1);
            }
            // Index for  HOPPINGS
            mask
        };
        i += 1;
    }
    hop_tmp
};
