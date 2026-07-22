//! `spo` — the subject-predicate-object triple, carrying palette word ids.
//!
//! v1 packed an SPO into 36 bits (three 12-bit COCA ids). v2's ids are the
//! 16-bit palette [`WordId`](crate::vocab::WordId)s, so a triple is `3 × 16 =
//! 48` bits — packed into a `u64` with a spare high 16 bits for a tag byte +
//! flags. The palette `(basin, identity)` pair of each slot is one `split`
//! away, so an SPO is directly addressable into the [`crate::space`] tables.

use crate::vocab::{split, WordId};

/// A semantic triple: who did what to whom, in palette word ids.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Spo {
    /// Subject word id.
    pub subject: WordId,
    /// Predicate (verb) word id.
    pub predicate: WordId,
    /// Object word id.
    pub object: WordId,
}

impl Spo {
    /// New triple.
    #[must_use]
    pub const fn new(subject: WordId, predicate: WordId, object: WordId) -> Self {
        Self {
            subject,
            predicate,
            object,
        }
    }

    /// Pack into the low 48 bits of a `u64` (`subject | predicate<<16 |
    /// object<<32`); the high 16 bits are left zero for a caller tag.
    #[must_use]
    pub const fn pack(self) -> u64 {
        (self.subject as u64) | ((self.predicate as u64) << 16) | ((self.object as u64) << 32)
    }

    /// Unpack from the low 48 bits of a `u64` (ignores the high 16).
    #[must_use]
    pub const fn unpack(bits: u64) -> Self {
        Self {
            subject: bits as u16,
            predicate: (bits >> 16) as u16,
            object: (bits >> 32) as u16,
        }
    }

    /// The three `(basin, identity)` palette pairs, in S/P/O order — the tile
    /// addresses [`crate::space::SemanticSpace`] scores.
    #[must_use]
    pub fn pairs(self) -> [(u8, u8); 3] {
        [
            split(self.subject),
            split(self.predicate),
            split(self.object),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_unpack_round_trip() {
        let t = Spo::new(0x0102, 0x2A2B, 0xFFFE);
        assert_eq!(Spo::unpack(t.pack()), t);
        // high 16 bits stay clear.
        assert_eq!(t.pack() >> 48, 0);
    }

    #[test]
    fn tag_bits_do_not_corrupt_the_triple() {
        let t = Spo::new(1, 2, 3);
        let tagged = t.pack() | (0xABCD << 48);
        assert_eq!(Spo::unpack(tagged), t);
    }

    #[test]
    fn pairs_are_the_palette_addresses() {
        let t = Spo::new(0x0100, 0x0201, 0x0302);
        assert_eq!(t.pairs(), [(1, 0), (2, 1), (3, 2)]);
    }
}
