//! Scaffold for the best query. Not on the build. Not an executor.
//!
//! A query lowers to one borrowed aperture and one terminal per
//! homomorphism class. This module is the planner shape quack does not
//! have yet. Wire it in front of `lower`. Do not route it through the
//! cognitive layers.
//!
//! Build check, from the repo root:
//! `rustc --edition 2021 --test .grok/board/lane-fold/scaffold/best_query.rs`

#![forbid(unsafe_code)]

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

/// Borrowed. A fold receives it. A fold does not build it.
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
        Ok(Self { id, words, shift: 0 })
    }

    pub fn dead_word_fraction(self) -> f64 {
        let dead = self.words.iter().filter(|w| **w == 0).count();
        dead as f64 / self.words.len() as f64
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

/// `AVG` is not a homomorphism. The pair is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Request {
    Reduce { hom: Hom, lane: LaneId },
    Avg { lane: LaneId },
    /// Hand the same value to n consumers.
    Repeat { n: u32 },
    /// Add the sum to itself n times. Not the same as Repeat.
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

/// One terminal the executor is allowed to see.
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

    /// Partition by (aperture, operation, lane). One terminal per class.
    /// Avg becomes a sum and a count. Repeat and scale stay distinct.
    pub fn collapse(&self) -> Result<BestQuery, Refuse> {
        self.check()?;
        let mut terminals: Vec<Terminal> = Vec::new();
        let mut collapsed = 0u32;
        for req in &self.requests {
            let (hom, lane, scale) = match req {
                Request::Reduce { hom, lane } => (*hom, *lane, 1),
                Request::Avg { lane } => {
                    push_or_merge(&mut terminals, self.aperture.id, Hom::Sum, *lane, 1, 1, &mut collapsed);
                    push_or_merge(&mut terminals, self.aperture.id, Hom::Count, *lane, 1, 1, &mut collapsed);
                    continue;
                }
                Request::Repeat { n } => {
                    if let Some(last) = terminals.last_mut() {
                        last.consumers = last.consumers.saturating_add(*n);
                        collapsed += n.saturating_sub(1);
                    }
                    continue;
                }
                Request::Scale { n } => {
                    if let Some(last) = terminals.last_mut() {
                        last.scale = last.scale.saturating_mul(*n);
                        collapsed += n.saturating_sub(1);
                    }
                    continue;
                }
            };
            push_or_merge(&mut terminals, self.aperture.id, hom, lane, 1, scale, &mut collapsed);
        }
        let remaining = terminals.len() as u32;
        Ok(BestQuery { terminals, collapsed, remaining })
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
    if let Some(found) = terminals
        .iter_mut()
        .find(|t| t.aperture == aperture && t.hom == hom && t.lane == lane && t.scale == scale)
    {
        found.consumers = found.consumers.saturating_add(consumers);
        *collapsed += consumers;
        return;
    }
    terminals.push(Terminal { aperture, hom, lane, consumers, scale });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn full() -> Vec<u64> {
        vec![u64::MAX; MASK_WORDS]
    }

    #[test]
    fn a_thousand_sums_lower_to_one() {
        let words = full();
        let q = Query {
            aperture: Aperture::resident(ApertureId(0), &words).unwrap(),
            requests: vec![Request::Reduce { hom: Hom::Sum, lane: LaneId(1) }; 1000],
            fused: false,
            claims_skip: false,
            under_plane: false,
            wants_rows: false,
        };
        let best = q.collapse().unwrap();
        assert_eq!(best.remaining, 1);
        assert_eq!(best.terminals[0].consumers, 1000);
        assert_eq!(best.collapsed, 999);
    }

    #[test]
    fn avg_is_a_pair() {
        let words = full();
        let q = Query {
            aperture: Aperture::resident(ApertureId(0), &words).unwrap(),
            requests: vec![Request::Avg { lane: LaneId(2) }],
            fused: false,
            claims_skip: false,
            under_plane: false,
            wants_rows: false,
        };
        let best = q.collapse().unwrap();
        assert_eq!(best.remaining, 2);
        assert!(best.terminals.iter().any(|t| t.hom == Hom::Sum));
        assert!(best.terminals.iter().any(|t| t.hom == Hom::Count));
    }

    #[test]
    fn fused_cannot_claim_the_skip() {
        let words = full();
        let q = Query {
            aperture: Aperture::resident(ApertureId(0), &words).unwrap(),
            requests: vec![],
            fused: true,
            claims_skip: true,
            under_plane: false,
            wants_rows: false,
        };
        assert_eq!(q.collapse(), Err(Refuse::FusedClaimsSkip));
    }

    #[test]
    fn rows_are_not_a_fold() {
        let words = full();
        let q = Query {
            aperture: Aperture::resident(ApertureId(0), &words).unwrap(),
            requests: vec![Request::Reduce { hom: Hom::Count, lane: LaneId(0) }],
            fused: false,
            claims_skip: false,
            under_plane: false,
            wants_rows: true,
        };
        assert_eq!(q.collapse(), Err(Refuse::RowsRequested));
    }
}
