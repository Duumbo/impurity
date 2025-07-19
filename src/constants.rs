const SIZE: usize = 128;
const ARRAY_SIZE: usize = (SIZE + 7) / 8;
/// System parameters
///
/// Encodes the system specifications
#[derive(Debug)]
pub struct SysParams<'a> {
    /// Number of sites.
    pub size: usize,
    /// Number of electrons.
    pub nelec: usize,
    /// Array size for `SpinState`.
    pub array_size: usize,
    /// Hubbard $U$ value.
    pub cons_u: f64,
    /// Hubbard $t$ value.
    pub cons_t: f64,
    /// Number of total orbital (pfaffian) variationnal parameters.
    pub nfij: usize,
    /// Number of total Jastrow variationnal parameters.
    pub nvij: usize,
    /// Number of total Gutzwiller variationnal parameters.
    pub ngi: usize,
    /// The transfer matrix $t_{ij}$. This is used to define the hopping scheme
    /// for the Hubbard model.
    pub transfert_matrix: &'a [f64],
    /// The same transfer matrix, but written in bitmask form, all coefficient
    /// are $1$ in this.
    pub hopping_bitmask: &'a [SpinState],
    /// Number of warmup iteration before sampling Monte-Carlo.
    pub nmcwarmup: usize,
    /// Number of Monte-Carlo samples.
    pub nmcsample: usize,
    /// Number of optimisation iterations to include a warmup before the
    /// Monte-Carlo sampling.
    pub nwarmupchains: usize,
    /// Number of previous energies to bootstrap.
    pub nbootstrap: usize,
    /// Period of sampling Monte-Carlo. $0$ means sample every iterations, $1$
    /// means sample every other. Rule of thumb is to use `size` as this value.
    pub mcsample_interval: usize,
    /// Number of fast-update to be done before using a clean-update. $0$
    /// deactivate the fast-updates for the state propositions, but not for the
    /// energy computation. The energy computation is then always only one
    /// fast-update deep. Rule of thumb is to use $32$, it seems to work well.
    pub clean_update_frequency: usize,
    /// Tolerance of pfaffian being $0$ before emitting warnings.
    pub tolerance_singularity: f64,
    /// Tolerance of pfaffian update before emitting warnings.
    pub tolerance_sherman_morrison: f64,
    /// Pairwave-function hard coded. Set to `false`.
    pub pair_wavefunction: bool,
    /// Private value. Number of parameters optimisation iteration.
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

/// TODOC
pub type BitStruct = u8;
/// TODOC
pub const NBITS: usize = 8;
