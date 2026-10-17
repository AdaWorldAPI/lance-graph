//! CE64 bits 43..45: the orientation of a relation.
//!
//! One meaning everywhere (D-CGW-1): the function a relation is read in, on
//! Haspelmath's dative semantic map (Haspelmath 2003, Fig. 8.1). `Core` (0)
//! is the plain core argument, which is also every relation no reader has
//! oriented: a Moore lane writes 0 here, because its direction is its slot and
//! polarity, not a function on this map.
//!
//! These bits used to hold a per-plane S/P/O "pathological" sign triad. Nothing
//! in production wrote it, and the planner's confounder check computes its
//! directions from population masks, so the triad and its readers were removed
//! rather than moved (D-CGW-1).

/// The orientation code of a relation (bits 43..45 of a [`crate::CausalEdge64`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum Orientation {
    /// Plain core argument, or not oriented by any reader.
    #[default]
    Core = 0,
    /// Direction (allative: "to the town").
    Direction = 1,
    /// Recipient ("gave it to her").
    Recipient = 2,
    /// Experiencer ("it seems to me").
    Experiencer = 3,
    /// Purpose ("for the meal").
    Purpose = 4,
    /// Beneficiary ("bought it for him").
    Beneficiary = 5,
    /// Predicative possessor ("the book is mine").
    PredicativePossessor = 6,
    /// External possessor ("broke his arm" read as "broke the arm to him").
    ExternalPossessor = 7,
}

impl Orientation {
    /// All eight orientations, in code order.
    pub const ALL: [Self; 8] = [
        Self::Core,
        Self::Direction,
        Self::Recipient,
        Self::Experiencer,
        Self::Purpose,
        Self::Beneficiary,
        Self::PredicativePossessor,
        Self::ExternalPossessor,
    ];

    /// The 3-bit code written to bits 43..45.
    #[inline(always)]
    #[must_use]
    pub const fn code(self) -> u8 {
        self as u8
    }

    /// The orientation a code names. Only the low three bits are read, so every
    /// value names exactly one orientation.
    #[inline(always)]
    #[must_use]
    pub const fn from_code(code: u8) -> Self {
        Self::ALL[(code & 0b111) as usize]
    }
}
