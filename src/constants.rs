const SIZE: usize = 128;
pub const ARRAY_SIZE: usize = (SIZE + 7) / 8;
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
    pub nwarmupchains: usize,
    pub nbootstrap: usize,
    pub mcsample_interval: usize,
    pub clean_update_frequency: usize,
    pub tolerance_singularity: f64,
    pub tolerance_sherman_morrison: f64,
    pub pair_wavefunction: bool,
    pub _opt_iter: usize,
}

impl<'a> SysParams<'a> {
    pub fn new(
        size: usize,
        nelec: usize,
        array_size: usize,
        cons_u: f64,
        cons_t: f64,
        nfij: usize,
        nvij: usize,
        ngi: usize,
        transfert_matrix: &'a [f64],
        hopping_bitmask: &'a [SpinState],
        nmcwarmup: usize,
        nmcsample: usize,
        nwarmupchains: usize,
        nbootstrap: usize,
        mcsample_interval: usize,
        clean_update_frequency: usize,
        tolerance_singularity: f64,
        tolerance_sherman_morrison: f64,
        pair_wavefunction: bool,
    ) -> Self {
        SysParams {
            size,
            nelec,
            array_size,
            cons_t,
            cons_u,
            nfij,
            nvij,
            ngi,
            transfert_matrix,
            hopping_bitmask,
            nmcwarmup,
            nmcsample,
            nwarmupchains,
            nbootstrap,
            mcsample_interval,
            clean_update_frequency,
            tolerance_sherman_morrison,
            tolerance_singularity,
            pair_wavefunction,
            _opt_iter: 0,
        }
    }
}

pub type BitStruct = u8;
pub const NBITS: usize = 8;
