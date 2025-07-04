extern crate num;

use num::PrimInt;
use rand::Rng;
use rand::distributions::{Distribution, Standard};
use std::fmt;
use log::error;
use std::marker::Send;

use strings::{UPARROW, DOWNARROW};

#[derive(Copy, Clone, Debug)]
pub enum Spin {
    Up,
    Down,
}

impl std::fmt::Display for Spin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::result::Result<(), std::fmt::Error> {
        match self {
            Spin::Up => write!(f, "{UPARROW}"),
            Spin::Down => write!(f, "{DOWNARROW}"),
        }?;
        Ok(())
    }
}

/// Abstraction layer for common bitwise operations.
/// # Purpose
/// The Bitops trait brings in scope an abstraction layer over some common bitwise
/// operations for a spin state. These operations make most functions compatible
/// with either a primitive integer type or an arbitrarily large byte array.
pub trait BitOps:
    std::ops::BitAnd<Output = Self> +
    Sized +
    std::ops::BitXorAssign +
    std::ops::BitXor<Output = Self> +
    Copy +
    std::ops::Not<Output = Self> +
    std::cmp::PartialEq +
    std::ops::Shr<usize, Output = Self> +
    std::ops::Shl<usize, Output = Self> +
    std::ops::BitOr<Output = Self> +
    From<SpinState> +
    Sync
{
    /// Provides the number of leading zeros in the bitstring. This gives the
    /// position of the first set bit in the string. This method is consistent
    /// with [BitOps::check] and [BitOps::set].
    fn leading_zeros(self) -> u32;
    /// Provides the number of set bits in the bitstirng. This gives the number
    /// of electrons in the bitstring.
    fn count_ones(self) -> u32;
    /// Compatibility layer for the std lib function of the same name.
    fn rotate_left(self, by: u32) -> Self;
    /// Compatibility layer for the std lib function of the same name.
    fn rotate_right(self, by: u32) -> Self;
    /// Set all the out of bounds bit to $0$. This is important if the bitstring
    /// is not full, to ensure that we don't processe garbage data.
    fn mask_bits(&mut self, by: usize);
    /// Set the $i$-th bit of the string, indexed from the left. This methods
    /// is consistent with [BitOps::check] and [BitOps::leading_zeros].
    fn set(&mut self, n: usize);
    /// Returns the truth value at index $i$, from the left. This methods is
    /// consistent with [BitOps::set] and [BitOps::leading_zeros].
    fn check(&self, i: usize) -> bool;
    /// Returns an owned instance of an all set bitstring.
    fn ones() -> Self;
    /// Returns an owned instance of an all unset bitstring.
    fn zeros() -> Self;
}

/// BitWise operations for all primitive ints. All methods are inlined and use
/// built-in  methods. [BitOps::set] and [BitOps::check] are implemented by
/// shifting a bitmask.
impl<I> BitOps for I
    where I: PrimInt + std::ops::BitXorAssign + std::ops::BitAndAssign + From<SpinState> + From<u8> + Sync
{
    #[inline(always)]
    fn leading_zeros(self) -> u32 {
        self.leading_zeros()
    }
    #[inline(always)]
    fn count_ones(self) -> u32 {
        self.count_ones()
    }
    #[inline(always)]
    fn rotate_left(self, by: u32) -> Self {
        self.rotate_left(by)
    }
    #[inline(always)]
    fn rotate_right(self, by: u32) -> Self {
        self.rotate_right(by)
    }
    #[inline(always)]
    fn mask_bits(&mut self, by: usize) {
        let n_bits: usize = std::mem::size_of::<I>() * u8::BITS as usize;
        *self &= <I>::max_value() << (n_bits - by);
    }
    #[inline(always)]
    fn set(&mut self, n: usize) {
        let n_bits: usize = std::mem::size_of::<I>() * u8::BITS as usize;
        if n >= n_bits {return;}
        let one: I = 1.into();
        *self = *self ^ (one << (n_bits - 1 - n));
    }
    #[inline(always)]
    fn check(&self, i: usize) -> bool {
        let n_bits: usize = std::mem::size_of::<I>() * u8::BITS as usize;
        if i >= n_bits {return false;}
        let one: I = 1.into();
        !(*self & (one << (n_bits - 1 - i)) == 0.into())
    }
    #[inline(always)]
    fn ones() -> Self {
        <I>::max_value()
    }
    #[inline(always)]
    fn zeros() -> Self {
        <I>::min_value()
    }
}

impl From<SpinState> for u8 {
    fn from(other: SpinState) -> Self {
        //Dumbo
        //if ARRAY_SIZE > 1 {error!("Casting SpinState of size {} to <u8> discards overflowing bits.", SIZE)}
        other.state[0]
    }
}

impl From<SpinState> for u16 {
    fn from(other: SpinState) -> Self {
        const NBYTES: usize = 2;
        if ARRAY_SIZE > NBYTES {error!("Casting SpinState of size {} to <u16> discards overflowing bits.", SIZE)}
        let mut tmp: [u8; NBYTES] = [0x00; NBYTES];
        for i in 0..ARRAY_SIZE {
            tmp[i] = other.state[i];
        }
        <u16>::from_be_bytes(tmp)
    }
}

impl From<SpinState> for u32 {
    fn from(other: SpinState) -> Self {
        const NBYTES: usize = 4;
        if ARRAY_SIZE > NBYTES {error!("Casting SpinState of size {} to <u32> discards overflowing bits.", SIZE)}
        let mut tmp: [u8; NBYTES] = [0x00; NBYTES];
        for i in 0..ARRAY_SIZE {
            tmp[i] = other.state[i];
        }
        <u32>::from_be_bytes(tmp)
    }
}

impl From<SpinState> for u64 {
    fn from(other: SpinState) -> Self {
        const NBYTES: usize = 8;
        if ARRAY_SIZE > NBYTES {error!("Casting SpinState of size {} to <u64> discards overflowing bits.", SIZE)}
        let mut tmp: [u8; NBYTES] = [0x00; NBYTES];
        for i in 0..ARRAY_SIZE {
            tmp[i] = other.state[i];
        }
        <u64>::from_be_bytes(tmp)
    }
}

impl From<SpinState> for u128 {
    fn from(other: SpinState) -> Self {
        const NBYTES: usize = 16;
        if ARRAY_SIZE > NBYTES {error!("Casting SpinState of size {} to <u128> discards overflowing bits.", SIZE)}
        let mut tmp: [u8; NBYTES] = [0x00; NBYTES];
        for i in 0..ARRAY_SIZE {
            tmp[i] = other.state[i];
        }
        <u128>::from_be_bytes(tmp)
    }
}

/// Encoding of the positions of a given spin.
/// # Definition
/// This structure fixes the number of electrons. The convention is to index
/// from the left to right. This structure is slower than using an primitive
/// integer type, but allows for an arbitrarily large number of spins, instead
/// of the maximum of $128$ sites.
/// # Usage
/// TODOC
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct SpinState {
    pub state: [u8; (SIZE + 7) / 8]
}

/// Bitwise implementations for the byte array structure [SpinState].
impl BitOps for SpinState {
    fn count_ones(self) -> u32 {
        let mut count: u32 = 0;
        for i in 0..ARRAY_SIZE {
            count += self.state[i].count_ones();
        }
        count
    }

    fn mask_bits(&mut self, by: usize) {
        let meta_by = by / 8;
        let granular_by = by % 8;
        for i in 0..meta_by {
            self.state[ARRAY_SIZE - 1 - i] &= 0;
        }
        self.state[meta_by] &= 0xff << (8 - granular_by - 1);
    }

    fn rotate_left(self, by: u32) -> Self {
        self << by as usize
    }

    fn rotate_right(self, by: u32) -> Self {
        self >> by as usize
    }

    fn leading_zeros(self) -> u32 {
        let mut count = 0;
        for i in 0..ARRAY_SIZE {
            count += self.state[i].leading_zeros();
            if count < (8 * (i + 1)) as u32 {break;}
        }
        count
    }

    fn set(&mut self, n: usize) {
        let meta_n = <usize>::min(n / 8, SIZE / 8);
        let granular_n = n % 8;
        self.state[meta_n] ^= 1 << (8 - granular_n - 1);
    }

    fn check(&self, i: usize) -> bool {
        let meta_n = <usize>::min(i / 8, SIZE / 8);
        let granular_n = i % 8;
        !(self.state[meta_n] & 1 << (8 - granular_n - 1) == 0)
    }

    fn ones() -> Self {
        let tmp = [0xff; ARRAY_SIZE];
        SpinState{state: tmp}
    }

    fn zeros() -> Self {
        let tmp = [0x00; ARRAY_SIZE];
        SpinState{state: tmp}
    }
}

impl Distribution<SpinState> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> SpinState {
        let mut state = [0; ARRAY_SIZE];
        rng.fill(&mut state);
        let state = SpinState{state};
        state
    }
}

impl From<u8> for SpinState {
    fn from(s: u8) -> SpinState {
        let mut tmp = [0; ARRAY_SIZE];
        tmp[0] = s;
        SpinState{state: tmp}
    }
}

impl std::ops::BitXorAssign<SpinState> for SpinState
{
    fn bitxor_assign(&mut self, rhs: SpinState) {
        for i in 0..ARRAY_SIZE {
            self.state[i] = self.state[i] ^ rhs.state[i];
        }
    }
}

impl std::ops::BitXor<SpinState> for SpinState {
    type Output = Self;

    fn bitxor(self, other: SpinState) -> Self::Output {
        let mut tmp: [u8; ARRAY_SIZE] = [0; ARRAY_SIZE];
        for i in 0..ARRAY_SIZE {
            tmp[i] = self.state[i] ^ other.state[i];
        }
        Self {state: tmp}
    }
}

impl std::ops::BitAnd<SpinState> for SpinState {
    type Output = Self;

    fn bitand(self, other: SpinState) -> Self::Output {
        let mut tmp: [u8; ARRAY_SIZE] = [0; ARRAY_SIZE];
        for i in 0..ARRAY_SIZE {
            tmp[i] = self.state[i] & other.state[i];
        }
        Self {state: tmp}
    }

}

impl std::ops::BitOr<SpinState> for SpinState {
    type Output = Self;

    fn bitor(self, other: SpinState) -> Self::Output {
        let mut tmp: [u8; ARRAY_SIZE] = [0; ARRAY_SIZE];
        for i in 0..ARRAY_SIZE {
            tmp[i] = self.state[i] | other.state[i];
        }
        Self {state: tmp}
    }

}

impl std::ops::Not for SpinState {
    type Output = Self;

    fn not(self) -> Self::Output {
        let mut tmp = [0; ARRAY_SIZE];
        for i in 0..ARRAY_SIZE {
            tmp[i] = ! self.state[i];
        }
        Self {state: tmp}
    }
}

impl std::ops::Shr<usize> for SpinState {
    type Output = Self;

    fn shr(self, by: usize) -> Self::Output {
        // Metashift.
        let meta_by = by as i32 / 8;
        let mut tmp: [u16; ARRAY_SIZE] = [0; ARRAY_SIZE];
        for i in 0..ARRAY_SIZE {
            let n = (i as i32- meta_by) % ARRAY_SIZE as i32;
            if n < 0 {
                tmp[i] = self.state[((n) + ARRAY_SIZE as i32) as usize] as u16;
            } else {
                tmp[i] = self.state[n as usize] as u16;
            }
            // Bitshift.
            tmp[i] = tmp[i] <<  8 - (by % 8);
        }

        // Xor
        let mut out_array: [u8; ARRAY_SIZE] = [0; ARRAY_SIZE];
        for i in 0..ARRAY_SIZE {
            out_array[i] = tmp[i].to_ne_bytes()[0];
        }
        for i in 0..ARRAY_SIZE {
            out_array[i] ^= tmp[(i + 1) % ARRAY_SIZE].to_ne_bytes()[1];
        }


        Self{ state: out_array}
    }
}

impl std::ops::Shl<usize> for SpinState {
    type Output = Self;

    // There are 3 steps to this algorithm.
    // 1. Metashift. This shift places the individual bytes in the right order.
    // 2. Bitshift on u16. This shits the individual bytes and store the
    // overflow in the u8 word buffer.
    // 3. Xor. This sets the newly padded-right bits that are 0 to the buffered
    // ones.
    fn shl(self, by: usize) -> Self::Output {
        // Metashift.
        let meta_by = by / 8;
        let mut tmp: [u16; ARRAY_SIZE] = [0; ARRAY_SIZE];
        for i in 0..(SIZE + 7) / 8 {
            tmp[i] = self.state[(i + meta_by) % ((SIZE + 7) / 8)] as u16;
            // Bitshift.
            tmp[i] = tmp[i] << (by % 8);
        }

        // Xor
        let mut out_array: [u8; ARRAY_SIZE] = [0; ARRAY_SIZE];
        for i in 0..ARRAY_SIZE {
            out_array[i] = tmp[i].to_ne_bytes()[1];
        }
        for i in 0..ARRAY_SIZE {
            out_array[i] ^= tmp[(i + 1) % ARRAY_SIZE].to_ne_bytes()[0];
        }

        Self{ state: out_array}
    }
}

impl fmt::Display for SpinState {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for i in 0..ARRAY_SIZE {
            write!(f, "{:0b} ", self.state[i])?;
        }
        write!(f, "")
    }
}

/// The Fock state structure. Encodes the spins positions.
/// # Definition
/// This structure has two different fields, the spin up and spin down component.
/// Each of these fields correspond to a physical bitstring that represent the
/// occupation of this state. It can either be a primitive integer type, like
/// [u64] or [u128], or it can be a byte array like [SpinState].
/// The convention is to place the sites in the order
/// $i\in\[0,N-1\]$ for both fields.
/// # Usage
/// This structure implements bitshifts to both fields.
/// ```rust
/// use impurity::FockState;
/// let state_5_both = FockState {spin_up: 5u8, spin_down: 5u8, n_sites: 8};
/// let state_20_both = FockState {spin_up: 20u8, spin_down: 20u8, n_sites: 8};
/// assert_eq!(state_5_both << 2, state_20_both);
/// ```
#[derive(Debug, Eq, PartialEq, Clone, Copy)]
pub struct FockState<T> where T: From<SpinState> + std::fmt::Display + Send
{
    pub spin_up: T,
    pub spin_down: T,
    pub n_sites: usize,
}

impl<T: std::ops::Shl<usize, Output = T> + From<SpinState> + std::fmt::Display + Send> std::ops::Shl<usize> for FockState<T> {
    type Output = Self;

    fn shl(self, u: usize) -> Self::Output {
        Self{
            spin_up: self.spin_up << u,
            spin_down: self.spin_down << u,
            n_sites: self.n_sites,
        }
    }
}

impl<T: std::ops::Shr<usize, Output = T> + From<SpinState> + std::fmt::Display + Send> std::ops::Shr<usize> for FockState<T> {
    type Output = Self;

    fn shr(self, u: usize) -> Self::Output {
        Self{
            spin_up: self.spin_up >> u,
            spin_down: self.spin_down >> u,
            n_sites: self.n_sites,
        }
    }
}

impl<T: BitOps + std::fmt::Display + Send> fmt::Display for FockState<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "|")?;
        for i in 0..self.n_sites {
            write!(f, "{}", self.spin_up.check(i) as usize)?;
        }
        write!(f, ":")?;
        for i in 0..self.n_sites {
            write!(f, "{}", self.spin_down.check(i) as usize)?;
        }
        write!(f, ">")
    }
}


// True bit rotation in a subset of the byte
#[inline(always)]
fn rot_right<T: BitOps>(s: T, n: usize, size: usize) -> T {
    (s >> n) | (s << (size - n))
}

pub trait Hopper {
    fn generate_all_hoppings(self: &Self, bitmask: &[SpinState]) -> Vec<(usize, usize, Spin)>;
    fn generate_all_exchange(self: &Self) -> Vec<(usize, usize)>;
}

impl<T: BitOps + From<SpinState> + std::fmt::Display + Send> Hopper for FockState<T> {
    fn generate_all_hoppings(self: &FockState<T>, bitmask: &[SpinState]) -> Vec<(usize, usize, Spin)> {
        let sup = self.spin_up;
        let sdo = self.spin_down;
        let nelec = (sup.count_ones() + sdo.count_ones()) as usize;
        let mut out_hoppings: Vec<(usize, usize, Spin)> = Vec::with_capacity(4 * nelec);

        // Hopping updates
        for j in 1..self.n_sites /2 + 1 {
            // See note 2024-06-08, rotate right gives all the horizontal links.
            // We need to go to j up to SIZE / 2
            //
            let bitmask = <T>::from(bitmask[j - 1]);
            let mut possible_hoppings_up = (rot_right(sup, j, self.n_sites) ^ sup) & bitmask;
            let mut possible_hoppings_do = (rot_right(sdo, j, self.n_sites) ^ sdo) & bitmask;

            // Consume the up possibilities
            let mut i: usize = possible_hoppings_up.leading_zeros() as usize;
            while i < self.n_sites {
                possible_hoppings_up.set(i);
                // Check where the electron came from
                let (came_from, went_to) = if sup.check(i) { // True: came from i, False: came from (i+1)%n
                    (i, ( i + self.n_sites - j ) % self.n_sites)
                } else {
                    (( i + self.n_sites - j ) % self.n_sites, i)
                };
                out_hoppings.push(
                    (came_from, went_to, Spin::Up)
                );
                i = possible_hoppings_up.leading_zeros() as usize;
            }

            // Consume the down possibilities
            let mut i: usize = possible_hoppings_do.leading_zeros() as usize;
            while i < self.n_sites {
                possible_hoppings_do.set(i);
                // Check where the electron came from
                let (came_from, went_to) = if sdo.check(i) { // True: came from i, False: came from (i+1)%n
                    (i, ( i + self.n_sites - j ) % self.n_sites)
                } else {
                    (( i + self.n_sites - j ) % self.n_sites, i)
                };
                out_hoppings.push(
                    (came_from, went_to, Spin::Down)
                );
                i = possible_hoppings_do.leading_zeros() as usize;
            }
        }

        out_hoppings
    }

    fn generate_all_exchange(self: &Self) -> Vec<(usize, usize)> {
        // TODO: UNOPTMISED
        // There are n choose 2 elements, so n(n-1)/2 terms, where n: nelec, roughly
        let sup = self.spin_up;
        let mut sup_copy = sup;
        let sdo = self.spin_down;
        let mut sdown_copy = sdo;
        let mut possible_exchange = sup ^ sdo;
        let mut i = possible_exchange.leading_zeros() as usize;
        let mut out_exchanges: Vec<(usize, usize)> = Vec::new();
        let mut j;

        while i < self.n_sites {
            let flag = sup.check(i);
            // Chosen spin up
            if flag {
                j = sdown_copy.leading_zeros() as usize;
                while j < self.n_sites {
                    out_exchanges.push((i, j));
                    sdown_copy.set(j);
                    j = sdown_copy.leading_zeros() as usize;
                }
                sdown_copy = sdo;
            }
            // Chose spin down
            else {
                j = sup_copy.leading_zeros() as usize;
                while j < self.n_sites {
                    out_exchanges.push((j, i));
                    sup_copy.set(j);
                    j = sup_copy.leading_zeros() as usize;
                }
                sup_copy = sup;
            }
            possible_exchange.set(i);
            i = possible_exchange.leading_zeros() as usize;
        }

        out_exchanges
    }
}

// Interface for random state generation
impl<T: BitOps + std::fmt::Display + Send> Distribution<FockState<T>> for Standard where Standard: Distribution<T> {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> FockState<T> {
        let su = rng.gen::<T>();
        let sd = rng.gen::<T>();
        FockState{
            spin_up: su,
            spin_down: sd,
            n_sites: SIZE,
        }
    }
}

pub trait RandomStateGeneration {
    fn generate_from_nelec<R: Rng + ?Sized>(rng: &mut R, nelec: usize, max_size: usize) -> Self;
    fn generate_hopping<R: Rng + ?Sized>(self: &Self, rng: &mut R, max_size: u32, out_idx: &mut (usize, usize, Spin)) -> Self;
    fn generate_exchange<R: Rng + ?Sized>(self: &Self, rng: &mut R, out_idx: &mut (usize, usize)) -> Self;
}

impl<T> RandomStateGeneration for FockState<T>
where T: BitOps + std::fmt::Display + Send,
    Standard: Distribution<T>
{
    fn generate_from_nelec<R: Rng + ?Sized>(rng: &mut R, nelec: usize, max_size: usize) -> FockState<T> {
        let mut state = FockState{spin_up: <T>::zeros(), spin_down: <T>::zeros(), n_sites: max_size};
        let mut i = 0;
        while i < nelec {
            let index: usize = rng.gen_range(0..max_size);
            let spin = rng.gen_bool(0.5);
            if spin {
                if !(state.spin_up.check(index)) {
                    state.spin_up.set(index);
                    i += 1;
                }
            }
            else {
                if !(state.spin_down.check(index)) {
                    state.spin_down.set(index);
                    i += 1;
                }
            }
        }
        state
    }


    fn generate_hopping<R: Rng + ?Sized>(self: &FockState<T>, rng: &mut R, max_size: u32, out_idx: &mut (usize, usize, Spin)) -> FockState<T> {
        // This is cheap, don't sweat it
        //let all_hops = self.generate_all_hoppings(&sys.hopping_bitmask);

        let spin = if rng.gen::<usize>() % 2 == 0 {
            Spin::Up
        } else {
            Spin::Down
        };
        let mut rand_from_idx;
        let mut rand_to_idx;
        match spin {
            Spin::Up => {
                // Choose spin from
                rand_from_idx = (rng.gen::<u32>() % max_size) as usize;
                while !self.spin_up.check(rand_from_idx) {
                    rand_from_idx = (rng.gen::<u32>() % max_size) as usize;
                }
                rand_to_idx = (rng.gen::<u32>() % max_size) as usize;
                while self.spin_up.check(rand_to_idx) {
                    rand_to_idx = (rng.gen::<u32>() % max_size) as usize;
                }
            },
            Spin::Down => {
                // Choose spin from
                rand_from_idx = (rng.gen::<u32>() % max_size) as usize;
                while !self.spin_down.check(rand_from_idx) {
                    rand_from_idx = (rng.gen::<u32>() % max_size) as usize;
                }
                rand_to_idx = (rng.gen::<u32>() % max_size) as usize;
                while self.spin_down.check(rand_to_idx) {
                    rand_to_idx = (rng.gen::<u32>() % max_size) as usize;
                }
            }
        }


        let mut sup = self.spin_up;
        let mut sdown = self.spin_down;

        match spin {
            Spin::Up => {
                // TODO: Combine these sets as a single bitmask. Maybe requires to modify the Trait BitOps
                sup.set(rand_from_idx);
                sup.set(rand_to_idx);
            },
            Spin::Down => {
                sdown.set(rand_from_idx);
                sdown.set(rand_to_idx);
            }
        }
        out_idx.0 = rand_from_idx;
        out_idx.1 = rand_to_idx;
        out_idx.2 = spin;

        FockState{
            n_sites: max_size as usize,
            spin_up: sup,
            spin_down: sdown,
        }

    }

    fn generate_exchange<R: Rng + ?Sized>(self: &FockState<T>, rng: &mut R, out_idx: &mut (usize, usize)) -> FockState<T> {
        // TODO: This is not so cheap. Perhaps change this to generate only a pair.
        let all_exchanges = self.generate_all_exchange();

        let rand_index = rng.gen_range(0..all_exchanges.len());
        let exchange = all_exchanges[rand_index];

        let mut sup = self.spin_up;
        let mut sdo = self.spin_up;
        // TODO: Combine these sets as a single bitmask. Maybe requires to modify the Trait BitOps
        sup.set(exchange.0);
        sup.set(exchange.1);
        sdo.set(exchange.0);
        sdo.set(exchange.1);

        out_idx.0 = exchange.0;
        out_idx.1 = exchange.1;

        FockState {
            n_sites: self.n_sites,
            spin_up: sup,
            spin_down: sdo,
        }
    }
}


#[cfg(test)]
mod test {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    #[test]
    fn test_random_state_generator() {
        let mut rng = SmallRng::seed_from_u64(42);
        for i in 0..100{
            let state: FockState<u128> = FockState::generate_from_nelec(&mut rng, i, 128);
            assert_eq!(state.spin_up.count_ones() + state.spin_down.count_ones(), i as u32);
        }
        for i in 0..50{
            let state: FockState<u64> = FockState::generate_from_nelec(&mut rng, i, 64);
            assert_eq!(state.spin_up.count_ones() + state.spin_down.count_ones(), i as u32);
        }
        for i in 0..20{
            let state: FockState<u32> = FockState::generate_from_nelec(&mut rng, i, 32);
            assert_eq!(state.spin_up.count_ones() + state.spin_down.count_ones(), i as u32);
        }
        for i in 0..8{
            let state: FockState<SpinState> = FockState::generate_from_nelec(&mut rng, i, SIZE);
            assert_eq!(state.spin_up.count_ones() + state.spin_down.count_ones(), i as u32);
        }
    }

}
