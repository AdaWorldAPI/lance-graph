use lance_graph_contract::algebra_law::{AlgebraDescriptor, AlgebraLaw, IdentityKind};
use std::convert::TryFrom;

pub const N: usize = 1 << 16;
pub const MASK_WORDS: usize = N / 64;
pub const APERTURE_CAP: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Row(u16);

impl TryFrom<u32> for Row {
    type Error = Refuse;
    fn try_from(v: u32) -> Result<Self, Refuse> {
        u16::try_from(v).map(Row).map_err(|_| Refuse::IndexWidened)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ApertureId(pub u16);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LaneId(pub u16);

#[derive(Clone, Copy, Debug)]
pub struct Aperture<'a> {
    pub id: ApertureId,
    pub words: &'a [u64],
    pub shift: u16,
}

impl<'a> Aperture<'a> {
    pub fn resident(id: ApertureId, words: &'a [u64]) -> Result<Self, Refuse> {
        if words.len() != MASK_WORDS {
            return Err(Refuse::LaneBound);
        }
        Ok(Self {
            id,
            words,
            shift: 0,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Hom {
    Count,
    Sum,
    Min,
    Max,
    Exists,
}

/// The law each homomorphism's merge obeys, as metadata for a planner.
///
/// Lane-fold keeps `Hom`; this only describes it. `Exists` is OR over
/// `false < true`: a join with the bottom (`false`) as identity, so it shares
/// `Max`'s law and needs no numeric identity. It has no report twin.
/// `invertible` is `false` throughout: no retraction path exists here.
impl AlgebraDescriptor for Hom {
    fn algebra_law(&self) -> AlgebraLaw {
        let (idempotent, identity_kind) = match self {
            Hom::Count | Hom::Sum => (false, IdentityKind::Zero),
            Hom::Min => (true, IdentityKind::Top),
            Hom::Max | Hom::Exists => (true, IdentityKind::Bottom),
        };
        AlgebraLaw {
            associative: true,
            commutative: true,
            idempotent,
            ordered: false,
            invertible: false,
            identity_kind,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Request {
    Reduce { hom: Hom, lane: LaneId },
    Avg { lane: LaneId },
    Repeat { n: u32 },
    Scale { n: u32 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refuse {
    IndexWidened,
    LaneBound,
    ApertureCap,
    PairList,
    FusedClaimsSkip,
    ReorderUnderPlane,
    UnstableAperture,
    RowsRequested,
    ScaleWithoutTerminal,
    GroupKnee,
}

#[derive(Clone, Debug)]
pub struct Query<'a> {
    pub aperture: Aperture<'a>,
    pub requests: Vec<Request>,
    pub fused: bool,
    pub claims_skip: bool,
    pub under_plane: bool,
    pub wants_rows: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Terminal {
    pub aperture: ApertureId,
    pub hom: Hom,
    pub lane: LaneId,
    pub consumers: u32,
    pub scale: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BestQuery {
    pub terminals: Vec<Terminal>,
    pub collapsed: u32,
    pub remaining: u32,
}

impl<'a> Query<'a> {
    pub fn check(&self) -> Result<(), Refuse> {
        if self.wants_rows {
            return Err(Refuse::RowsRequested);
        }
        if self.fused && self.claims_skip {
            return Err(Refuse::FusedClaimsSkip);
        }
        if self.under_plane && self.claims_skip {
            return Err(Refuse::ReorderUnderPlane);
        }
        Ok(())
    }

    pub fn collapse(&self) -> Result<BestQuery, Refuse> {
        self.check()?;
        let mut terminals: Vec<Terminal> = Vec::new();
        let mut collapsed = 0u32;
        for req in &self.requests {
            match req {
                Request::Reduce { hom, lane } => {
                    push_or_merge(&mut terminals, self.aperture.id, *hom, *lane, 1, 1, &mut collapsed);
                }
                Request::Avg { lane } => {
                    push_or_merge(&mut terminals, self.aperture.id, Hom::Sum, *lane, 1, 1, &mut collapsed);
                    push_or_merge(&mut terminals, self.aperture.id, Hom::Count, *lane, 1, 1, &mut collapsed);
                }
                Request::Repeat { n } => {
                    let last = terminals.last_mut().ok_or(Refuse::ScaleWithoutTerminal)?;
                    last.consumers = last.consumers.saturating_add(*n);
                    collapsed += n.saturating_sub(1);
                }
                Request::Scale { n } => {
                    let last = terminals.last_mut().ok_or(Refuse::ScaleWithoutTerminal)?;
                    last.scale = last.scale.saturating_mul(*n);
                    collapsed += n.saturating_sub(1);
                }
            }
        }
        Ok(BestQuery {
            remaining: terminals.len() as u32,
            terminals,
            collapsed,
        })
    }
}

fn push_or_merge(
    terminals: &mut Vec<Terminal>,
    aperture: ApertureId,
    hom: Hom,
    lane: LaneId,
    consumers: u32,
    scale: u32,
    collapsed: &mut u32,
) {
    if let Some(found) = terminals.iter_mut().find(|t| {
        t.aperture == aperture && t.hom == hom && t.lane == lane && t.scale == scale
    }) {
        found.consumers = found.consumers.saturating_add(consumers);
        *collapsed += consumers;
        return;
    }
    terminals.push(Terminal {
        aperture,
        hom,
        lane,
        consumers,
        scale,
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q(requests: Vec<Request>) -> Query<'static> {
        let words: &'static [u64] = Box::leak(vec![u64::MAX; MASK_WORDS].into_boxed_slice());
        Query {
            aperture: Aperture::resident(ApertureId(0), words).unwrap(),
            requests,
            fused: false,
            claims_skip: false,
            under_plane: false,
            wants_rows: false,
        }
    }

    #[test]
    fn a_thousand_sums_lower_to_one() {
        let best = q(vec![Request::Reduce { hom: Hom::Sum, lane: LaneId(1) }; 1000])
            .collapse()
            .unwrap();
        assert_eq!(best.remaining, 1);
        assert_eq!(best.terminals[0].consumers, 1000);
        assert_eq!(best.collapsed, 999);
    }

    #[test]
    fn avg_is_a_pair() {
        let best = q(vec![Request::Avg { lane: LaneId(2) }]).collapse().unwrap();
        assert_eq!(best.remaining, 2);
    }

    #[test]
    fn exists_is_a_join_with_bottom_identity() {
        let law = Hom::Exists.algebra_law();
        assert_eq!(law.identity_kind, IdentityKind::Bottom);
        assert!(law.associative && law.commutative && law.idempotent);
        assert!(!law.ordered && !law.invertible);
        // OR over `false < true` is MAX over the same order.
        assert_eq!(law, Hom::Max.algebra_law());
    }

    #[test]
    fn no_hom_claims_retraction_or_order() {
        for h in [Hom::Count, Hom::Sum, Hom::Min, Hom::Max, Hom::Exists] {
            let law = h.algebra_law();
            assert!(!law.invertible, "{h:?}: no remove path exists in lane-fold");
            assert!(!law.ordered, "{h:?} is order-insensitive");
        }
    }

    #[test]
    fn a_bare_scale_is_refused() {
        let err = q(vec![Request::Scale { n: 3 }]).collapse().unwrap_err();
        assert_eq!(err, Refuse::ScaleWithoutTerminal);
    }
}
