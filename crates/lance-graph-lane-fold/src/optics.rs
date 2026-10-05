//! Optics is a picture of the transform plane, not a substrate.
//!
//! The falsifier in `.grok/board/HANDOVER_FALSIFICATION.md`: an analog inner
//! product does not produce a DuckDB-matching popcount, and it does not carry
//! a generation-checked `u16`. This module locks that rejection. It does not
//! implement a lens.

/// A generation-checked ordinal. The handover the fold actually uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CheckedOrdinal {
    pub ordinal: u16,
    pub generation: u32,
}

/// What an optical plane can emit: a noisy energy, no address.
#[derive(Clone, Copy, Debug)]
pub struct AnalogEnergy {
    pub energy: f64,
}

/// Exact popcount of a mask word. The checksum side.
pub fn exact_popcount(word: u64) -> u32 {
    word.count_ones()
}

/// A toy optical inner product: each set bit contributes 1 plus a bias.
/// This is the smallest analog residual. A real lens is not better at the ordinal.
pub fn analog_inner(word: u64, bias: f64) -> AnalogEnergy {
    let ones = exact_popcount(word) as f64;
    AnalogEnergy {
        energy: ones * (1.0 + bias),
    }
}

impl AnalogEnergy {
    /// Round-trip to a count. Fails closed: a non-integral energy is not a popcount.
    pub fn as_popcount(self) -> Option<u32> {
        if self.energy.fract() != 0.0 || self.energy < 0.0 {
            return None;
        }
        Some(self.energy as u32)
    }

    /// An energy has no ordinal and no generation. The handover is not in the beam.
    pub fn as_ordinal(self, _generation: u32) -> Option<CheckedOrdinal> {
        let _ = self.energy;
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analog_energy_is_not_a_popcount() {
        let word = 0b0000_1111_u64;
        let exact = exact_popcount(word);
        let analog = analog_inner(word, 0.01);
        assert_eq!(exact, 4);
        assert_ne!(analog.as_popcount(), Some(exact));
        assert!(analog.as_popcount().is_none());
    }

    #[test]
    fn a_zero_bias_still_does_not_carry_the_ordinal() {
        let analog = analog_inner(0b0000_1111, 0.0);
        assert_eq!(analog.as_popcount(), Some(4));
        assert_eq!(
            analog.as_ordinal(1),
            None,
            "a matching energy is still not a generation-checked u16"
        );
    }

    #[test]
    fn the_handover_is_the_ordinal_not_the_energy() {
        let handed = CheckedOrdinal {
            ordinal: 7,
            generation: 3,
        };
        assert_eq!(handed.ordinal, 7);
        assert_eq!(handed.generation, 3);
    }
}
