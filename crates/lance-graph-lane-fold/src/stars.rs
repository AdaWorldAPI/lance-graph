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
    Shift { d: u16 },
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
        assert_eq!(map.teleport(Landing::Sum).star, Star::Rail);
    }

    #[test]
    fn nine_alignments_are_a_refusal() {
        assert_eq!(Map::draw(vec![MapOp::AlignOnce; 9]), Err(Star::Refusal));
    }
}
