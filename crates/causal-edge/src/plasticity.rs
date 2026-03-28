//! Per-plane plasticity: hot (learning), cold (stable), frozen (locked).
//!
//! 3 bits, one per SPO plane. Controls which planes accept
//! palette reassignment under evidence pressure.

/// Plasticity state for the three SPO planes.
///
/// Each bit: 0 = frozen/cold, 1 = hot (accepting updates).
/// - bit 0: S-plane plasticity
/// - bit 1: P-plane plasticity
/// - bit 2: O-plane plasticity
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlasticityState(u8);

impl PlasticityState {
    /// All planes frozen. Established clinical pattern.
    pub const ALL_FROZEN: Self = Self(0b000);
    /// All planes hot. New/uncertain edge, fully plastic.
    pub const ALL_HOT: Self = Self(0b111);
    /// Only S-plane hot (patient archetype uncertain).
    pub const S_HOT: Self = Self(0b001);
    /// Only P-plane hot (treatment archetype uncertain).
    pub const P_HOT: Self = Self(0b010);
    /// Only O-plane hot (outcome archetype uncertain).
    pub const O_HOT: Self = Self(0b100);

    /// Construct from raw 3-bit value.
    #[inline]
    pub fn from_bits(v: u8) -> Self {
        Self(v & 0b111)
    }

    /// Raw bits.
    #[inline]
    pub fn bits(self) -> u8 {
        self.0
    }

    /// Is the S-plane hot (plastic)?
    #[inline]
    pub fn s_hot(self) -> bool { self.0 & 0b001 != 0 }
    /// Is the P-plane hot?
    #[inline]
    pub fn p_hot(self) -> bool { self.0 & 0b010 != 0 }
    /// Is the O-plane hot?
    #[inline]
    pub fn o_hot(self) -> bool { self.0 & 0b100 != 0 }

    /// Freeze the S-plane.
    #[inline]
    pub fn freeze_s(self) -> Self { Self(self.0 & !0b001) }
    /// Freeze the P-plane.
    #[inline]
    pub fn freeze_p(self) -> Self { Self(self.0 & !0b010) }
    /// Freeze the O-plane.
    #[inline]
    pub fn freeze_o(self) -> Self { Self(self.0 & !0b100) }

    /// Heat the S-plane.
    #[inline]
    pub fn heat_s(self) -> Self { Self(self.0 | 0b001) }
    /// Heat the P-plane.
    #[inline]
    pub fn heat_p(self) -> Self { Self(self.0 | 0b010) }
    /// Heat the O-plane.
    #[inline]
    pub fn heat_o(self) -> Self { Self(self.0 | 0b100) }

    /// Number of hot (plastic) planes.
    #[inline]
    pub fn hot_count(self) -> u8 { self.0.count_ones() as u8 }
}
