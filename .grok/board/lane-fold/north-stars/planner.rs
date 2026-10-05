//! Map and teleport. The planner draws the map. The terminal is the teleport.
//!
//! A north star is a landing the teleport is allowed to make.
//! Check: `rustc --edition 2021 --test .grok/board/lane-fold/north-stars/planner.rs`

#![forbid(unsafe_code)]

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Star {
    Rail,
    Aperture,
    Homomorphism,
    Average,
    Complement,
    Span,
    Zone,
    Proof,
    Refusal,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MapOp {
    /// Tick i is tick i+d. No zipper.
    Shift { d: u16 },
    /// Once. Writes a mask, not pairs.
    AlignOnce,
    Gate,
    RangeSpan { lo: u16, hi: u16 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Landing {
    Count,
    Sum,
    Min,
    Max,
    Exists,
    Checksum,
    Refuse,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Map {
    pub ops: Vec<MapOp>,
    pub masks: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Teleport {
    pub star: Star,
    pub landing: Landing,
}

impl Map {
    pub const MASK_CAP: u32 = 8;

    pub fn draw(ops: Vec<MapOp>) -> Result<Self, Star> {
        let masks = ops.iter().filter(|o| matches!(o, MapOp::AlignOnce)).count() as u32;
        if masks > Self::MASK_CAP {
            return Err(Star::Refusal);
        }
        Ok(Self { ops, masks })
    }

    /// The teleport. One landing. The route is not a landing.
    pub fn teleport(&self, ask: Landing) -> Teleport {
        let star = match ask {
            Landing::Checksum => Star::Proof,
            Landing::Refuse => Star::Refusal,
            Landing::Count | Landing::Sum | Landing::Min | Landing::Max | Landing::Exists => {
                if self.ops.iter().any(|o| matches!(o, MapOp::RangeSpan { .. })) {
                    Star::Span
                } else if self.ops.iter().any(|o| matches!(o, MapOp::Shift { .. })) {
                    Star::Rail
                } else {
                    Star::Homomorphism
                }
            }
        };
        Teleport { star, landing: ask }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_shift_teleports_to_the_rail() {
        let map = Map::draw(vec![MapOp::Shift { d: 4 }, MapOp::Gate]).unwrap();
        let t = map.teleport(Landing::Sum);
        assert_eq!(t.star, Star::Rail);
        assert_eq!(t.landing, Landing::Sum);
    }

    #[test]
    fn a_span_is_not_a_scan() {
        let map = Map::draw(vec![MapOp::RangeSpan { lo: 10, hi: 20 }]).unwrap();
        assert_eq!(map.teleport(Landing::Count).star, Star::Span);
    }

    #[test]
    fn nine_alignments_are_a_refusal() {
        let ops = vec![MapOp::AlignOnce; 9];
        assert_eq!(Map::draw(ops), Err(Star::Refusal));
    }
}
