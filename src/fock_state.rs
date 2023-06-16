/// The Fock state structure.
/// # Definition
/// This structure has two different fields, the spin up and spin down component.
/// Each of these fields correspond to a physical `u128` that represent the
/// occupation of this state. The convention is to place the sites in the order
/// $i\in\[1,N\]$ for both fields.
/// # Usage
/// This structure implements bitshifts to both fields.
/// ```rust
/// use impurity::FockState;
/// let state_5_both = FockState {spin_up: 5, spin_down: 5};
/// let state_20_both = FockState {spin_up: 20, spin_down: 20};
/// assert_eq!(state_5_both << 2, state_20_both);
/// ```
#[derive(Debug, Eq, PartialEq)]
pub struct FockState {
    pub spin_up: u8,
    pub spin_down: u8,
}

impl std::ops::Shl<usize> for FockState {
    type Output = Self;

    fn shl(self, u: usize) -> Self::Output {
        Self{
            spin_up: self.spin_up << u,
            spin_down: self.spin_down << u,
        }
    }
}

impl std::ops::Shr<usize> for FockState {
    type Output = Self;

    fn shr(self, u: usize) -> Self::Output {
        Self{
            spin_up: self.spin_up >> u,
            spin_down: self.spin_down >> u,
        }
    }
}
