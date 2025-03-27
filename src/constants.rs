const SIZE: usize = 9;
const ARRAY_SIZE: usize = (SIZE + 7) / 8;
/// System parameters
/// TODOC
#[derive(Debug)]
pub struct SysParams<'a> {
    pub size: usize,
    pub nelec: usize,
    pub array_size: usize,
    pub cons_u: f64,
    pub cons_t: f64,
    pub nfij: usize,
    pub nvij: usize,
    pub ngi: usize,
    pub transfert_matrix: &'a [f64],
    pub hopping_bitmask: &'a [SpinState],
    pub nmcwarmup: usize,
    pub nmcsample: usize,
    pub nbootstrap: usize,
    pub mcsample_interval: usize,
    pub clean_update_frequency: usize,
    pub tolerance_singularity: f64,
    pub tolerance_sherman_morrison: f64,
    pub pair_wavefunction: bool,
}

pub type BitStruct = u8;
pub const NBITS: usize = 8;
