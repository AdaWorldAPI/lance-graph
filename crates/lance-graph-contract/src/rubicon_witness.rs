// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! `rubicon_witness` — reading which side of the Rubicon the thinking is on,
//! from the focus of attention alone (`D-ACR-8`).
//!
//! # What Heckhausen actually asserts, and why a focus mask can falsify it
//!
//! `unified-soa-rubikon-integration-v1.md` §3 maps the Heckhausen action phases
//! onto the kanban columns, and the crossing sits on
//! [`Planning`](KanbanColumn::Planning) →
//! [`CognitiveWork`](KanbanColumn::CognitiveWork). The claim the model makes
//! about that crossing is a claim about **attention**:
//!
//! > **Pre-Rubicon = deliberative mindset:** open, broad, impartial — many
//! > candidates held at once.
//! > **Post-Rubicon = implemental mindset:** narrow, partial, *shielding* — the
//! > intention is protected against reconsideration.
//!
//! A [`RowFocusMask`] measures exactly that, so the alpha channel does not
//! merely *accompany* the phase labels — **it is the instrument that can
//! falsify them**. If the two phases read the same on a genuinely deliberated
//! task, then either the kanban columns are decoration or the focus mask is not
//! recording attention, and both are findings worth having.
//!
//! # It READS. It never moves anything.
//!
//! Phase movement stays `advance_on_gate`, and the standing tombstone applies:
//! `E-PROGRESSION-IS-EXISTENCE-NOT-COMMAND-1` — progression is existence, not
//! command. An overlay that *drives* a transition from a focus reading has
//! rebuilt the scheduler this substrate removed. Nothing in this module takes
//! `&mut` anything, and the precedent is `PhaseCensus`: one read-only pass.
//!
//! # Breadth is a POPULATION, not an entry count
//!
//! [`RowFocusMask::len`] is the antichain size — and a single shallow entry can
//! cover an unbounded subtree, so a mask of one entry may be far broader than a
//! mask of ten. Breadth here is therefore the covered **population**: a focus of
//! depth `d` leaves `CASCADE_UNITS − d` unconstrained 256-ary units, so it
//! covers `256^(CASCADE_UNITS − d)` addresses. That is the `6×2×8bit↑n` reading
//! at the measurement — depth buys coverage exponentially, which is the whole
//! reason a flat index cannot express this.

use crate::attention_facet::RowFocusMask;
use crate::facet::CASCADE_UNITS;
use crate::kanban::KanbanColumn;

/// The two kanban columns the crossing sits between. Named once, here, so a
/// caller cannot sample the wrong pair and still typecheck.
pub const PRE_RUBICON: KanbanColumn = KanbanColumn::Planning;
/// The post-crossing column — the implemental, shielding side.
pub const POST_RUBICON: KanbanColumn = KanbanColumn::CognitiveWork;

/// Addresses a focus mask covers, as an exact `f64` population.
///
/// `256^12 ≈ 7.9e28` is comfortably inside `f64`, so this is exact for every
/// expressible focus and needs no log-space trickery. Entries are an antichain
/// (no entry covers another), so summing them double-counts nothing.
///
/// - empty mask → `0.0` (attention nowhere)
/// - one exact focus → `1.0` (one address)
/// - one whole-class focus → `256^12`
#[must_use]
pub fn coverage(mask: &RowFocusMask) -> f64 {
    mask.entries()
        .iter()
        .map(|e| 256f64.powi(i32::from(CASCADE_UNITS as u8 - e.depth())))
        .sum()
}

/// [`coverage`] on a log₂ scale, so a mean over samples is not swamped by the
/// broadest one.
///
/// `log2(1 + coverage)`, and the `1 +` is **finiteness, not distinctness**: an
/// empty mask covers `0.0`, and `log2(0)` is `-inf`, which would poison the
/// mean in [`FocusTrace::breadth`] — one unsampled moment would make a whole
/// phase read as infinitely narrow. With the offset, empty reads `0.0`, exact
/// reads `1.0`, and every trace stays finite.
///
/// (An earlier version of this comment claimed the offset was what kept empty
/// distinct from exact. That was wrong — without it they read `-inf` and `0.0`,
/// already distinct — and no test could falsify it, because the assertion it
/// justified passed either way. The mutation that exposed it is now the test
/// below.)
#[must_use]
pub fn breadth_bits(mask: &RowFocusMask) -> f64 {
    (1.0 + coverage(mask)).log2()
}

/// Overlap of two focuses as a Jaccard ratio over their **antichains**.
///
/// Deliberately entry-based rather than population-based: persistence asks
/// *"is the thinking still looking at the same things"*, and an entry is one
/// thing the thinking looked at. Two empty masks are unchanged (`1.0`) — a
/// focus that stays nowhere is still a focus that did not move.
#[must_use]
fn overlap(a: &RowFocusMask, b: &RowFocusMask) -> f64 {
    let union = a.union(b).len();
    if union == 0 {
        return 1.0;
    }
    a.intersect(b).len() as f64 / union as f64
}

/// A phase's sampled attention: the focus mask, sampled repeatedly while the
/// work sat in one kanban column.
///
/// The order of samples is the order they were taken — [`persistence`] reads
/// consecutive pairs, so a shuffled trace measures something else.
///
/// [`persistence`]: Self::persistence
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FocusTrace {
    column: Option<KanbanColumn>,
    samples: Vec<RowFocusMask>,
}

impl FocusTrace {
    /// An empty trace for a column — nothing sampled yet.
    #[must_use]
    pub fn new(column: KanbanColumn) -> Self {
        Self {
            column: Some(column),
            samples: Vec::new(),
        }
    }

    /// Record one sample. Takes `&mut self` on the TRACE — never on anything
    /// the substrate owns; a trace is the observer's own notebook.
    pub fn sample(&mut self, mask: RowFocusMask) {
        self.samples.push(mask);
    }

    /// The column this trace was taken in, if it was built with one.
    #[must_use]
    pub fn column(&self) -> Option<KanbanColumn> {
        self.column
    }

    /// How many samples the trace holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Is the trace empty of samples?
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Mean [`breadth_bits`] across samples; `None` with nothing sampled.
    ///
    /// `None` rather than `0.0`, because "nothing was measured" and "attention
    /// covered nothing" are different facts and a mean of zero samples must not
    /// impersonate the second.
    #[must_use]
    pub fn breadth(&self) -> Option<f64> {
        if self.samples.is_empty() {
            return None;
        }
        Some(self.samples.iter().map(breadth_bits).sum::<f64>() / self.samples.len() as f64)
    }

    /// Mean consecutive [`overlap`]; `None` with fewer than two samples.
    ///
    /// `1.0` = the focus never moved (maximally shielded); `0.0` = every sample
    /// looked somewhere new. Undefined for a single sample — persistence is a
    /// property of a SEQUENCE, and one point is not one.
    #[must_use]
    pub fn persistence(&self) -> Option<f64> {
        if self.samples.len() < 2 {
            return None;
        }
        let pairs = self.samples.windows(2).map(|w| overlap(&w[0], &w[1]));
        let n = self.samples.len() - 1;
        Some(pairs.sum::<f64>() / n as f64)
    }
}

/// What the two traces say about the crossing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RubiconVerdict {
    /// Both axes moved as Heckhausen predicts: narrower AND more persistent
    /// after the crossing.
    Crossed,
    /// One axis moved, the other did not. Reported as its own state rather than
    /// rounded to either neighbour — a half-signal is not a crossing and is not
    /// silence.
    Partial,
    /// Neither axis moved beyond `epsilon`. On a genuinely deliberated task
    /// this is the FINDING the module docs name (columns decoration, or the
    /// mask not recording attention); on a single-forced-candidate task it is
    /// the correct answer.
    Indistinguishable,
    /// An axis moved the WRONG way beyond `epsilon` — broader or less
    /// persistent after the crossing. Never folded into the other three: a
    /// contradicted model must be visible, not rounded to "no signal".
    Inverted,
}

/// The measured reading. The two deltas are always reported; the verdict only
/// classifies them, so a caller can disagree with the classification without
/// losing the numbers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RubiconReading {
    /// `pre.breadth − post.breadth`, in bits. **Positive = narrowing**, the
    /// direction the implemental mindset predicts.
    pub breadth_drop: f64,
    /// `post.persistence − pre.persistence`. **Positive = shielding**.
    pub persistence_gain: f64,
    /// The classification of the pair.
    pub verdict: RubiconVerdict,
}

/// Read the crossing from a pre- and post-Rubicon trace.
///
/// `None` when either side lacks the samples to measure — breadth needs one
/// sample, persistence needs two. An unmeasurable trace yields no reading at
/// all rather than a reading built on a default, which is the same refusal the
/// recipe ladder makes for an ungrounded input.
///
/// `epsilon` is the indifference band on BOTH axes, in their own units (bits
/// for breadth, ratio for persistence). It is a real knob: raising it turns
/// signals into [`Indistinguishable`](RubiconVerdict::Indistinguishable), and
/// lowering it admits smaller ones.
#[must_use]
pub fn read_crossing(pre: &FocusTrace, post: &FocusTrace, epsilon: f64) -> Option<RubiconReading> {
    let breadth_drop = pre.breadth()? - post.breadth()?;
    let persistence_gain = post.persistence()? - pre.persistence()?;

    let narrowed = breadth_drop > epsilon;
    let broadened = breadth_drop < -epsilon;
    let shielded = persistence_gain > epsilon;
    let loosened = persistence_gain < -epsilon;

    let verdict = if broadened || loosened {
        RubiconVerdict::Inverted
    } else if narrowed && shielded {
        RubiconVerdict::Crossed
    } else if narrowed || shielded {
        RubiconVerdict::Partial
    } else {
        RubiconVerdict::Indistinguishable
    };

    Some(RubiconReading {
        breadth_drop,
        persistence_gain,
        verdict,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention_facet::AttentionFocusFacet;
    use crate::facet::{FacetCascade, FacetTier};

    const CLASS: u32 = 0x0301_0000;

    /// A focus at `depth`, distinguished by `seed` in its coarsest unit — so
    /// two fixtures at the same depth are genuinely different addresses and not
    /// the same one counted twice.
    fn focus(seed: u8, depth: u8) -> AttentionFocusFacet {
        let mut tiers = [FacetTier { lo: 0, hi: 0 }; 6];
        tiers[0].hi = seed;
        AttentionFocusFacet::prefix(
            FacetCascade {
                facet_classid: CLASS,
                tiers,
            },
            depth,
        )
        .expect("depth within the cascade")
    }

    fn mask(entries: &[(u8, u8)]) -> RowFocusMask {
        let mut m = RowFocusMask::empty();
        for &(seed, depth) in entries {
            m.insert(focus(seed, depth));
        }
        m
    }

    /// **Breadth is a population, not a count.** One shallow entry outweighs
    /// many deep ones — the property that makes a flat index unusable here.
    #[test]
    fn one_shallow_focus_is_broader_than_many_deep_ones() {
        let broad = mask(&[(0, 2)]);
        let many_narrow = mask(&[(1, 11), (2, 11), (3, 11), (4, 11), (5, 11)]);
        assert!(
            broad.len() < many_narrow.len(),
            "the broad mask must have FEWER entries, or this proves nothing"
        );
        assert!(
            coverage(&broad) > coverage(&many_narrow) * 1e10,
            "and still cover vastly more: {} vs {}",
            coverage(&broad),
            coverage(&many_narrow)
        );
        // Empty and exact must not collide — the can-fire/can-stay-silent pair.
        assert_eq!(coverage(&RowFocusMask::empty()), 0.0);
        assert_eq!(coverage(&mask(&[(0, CASCADE_UNITS as u8)])), 1.0);
        assert!(
            breadth_bits(&RowFocusMask::empty()) < breadth_bits(&mask(&[(0, CASCADE_UNITS as u8)])),
            "attention nowhere must read narrower than attention on one address"
        );
    }

    /// **An unsampled moment must not make a whole phase read infinitely
    /// narrow.** `log2(0)` is `-inf`, and one `-inf` in a mean carries the
    /// entire trace with it — so a trace holding a real focus AND an empty
    /// sample must still report a finite breadth, strictly between the two.
    ///
    /// This is what the `1 +` in [`breadth_bits`] actually buys. Dropping it
    /// leaves every OTHER assertion in this module green.
    #[test]
    fn an_empty_sample_does_not_poison_a_traces_breadth() {
        assert!(
            breadth_bits(&RowFocusMask::empty()).is_finite(),
            "an empty focus must read finite, not -inf"
        );

        let mut t = FocusTrace::new(PRE_RUBICON);
        t.sample(mask(&[(1, 4)]));
        t.sample(RowFocusMask::empty());
        let b = t.breadth().expect("two samples");
        assert!(b.is_finite(), "one empty sample poisoned the mean: {b}");

        // …and non-trivially: it lands strictly between the two constituents,
        // so the empty sample is neither ignored nor dominant.
        let solo = breadth_bits(&mask(&[(1, 4)]));
        assert!(
            b > 0.0 && b < solo,
            "expected 0 < {b} < {solo} — the empty sample must pull the mean down without erasing it"
        );
    }

    /// **Can-fire.** A deliberated task: broad and roving before the crossing,
    /// narrow and held after it.
    #[test]
    fn a_deliberated_task_reads_as_a_crossing() {
        let mut pre = FocusTrace::new(PRE_RUBICON);
        // Weighing: several candidates at once, and the set keeps changing.
        pre.sample(mask(&[(1, 3), (2, 3), (3, 3)]));
        pre.sample(mask(&[(2, 3), (4, 3), (5, 3)]));
        pre.sample(mask(&[(5, 3), (6, 3), (7, 3)]));

        let mut post = FocusTrace::new(POST_RUBICON);
        // Shielding: one deep focus, held.
        for _ in 0..3 {
            post.sample(mask(&[(9, 10)]));
        }

        let r = read_crossing(&pre, &post, 0.05).expect("both traces are measurable");
        assert_eq!(r.verdict, RubiconVerdict::Crossed, "reading: {r:?}");
        assert!(r.breadth_drop > 0.0, "must narrow: {}", r.breadth_drop);
        assert!(
            r.persistence_gain > 0.0,
            "must shield: {}",
            r.persistence_gain
        );
        assert_eq!(pre.column(), Some(KanbanColumn::Planning));
        assert_eq!(post.column(), Some(KanbanColumn::CognitiveWork));
    }

    /// **Can-stay-silent, on NON-TRIVIAL input.** A single forced candidate:
    /// nothing to weigh, so no mindset shift exists to observe. Both traces are
    /// real, multi-sample and non-empty — an empty-input silence case would
    /// only prove the code handles emptiness.
    #[test]
    fn a_single_forced_candidate_shows_no_crossing() {
        let only = mask(&[(9, 10)]);
        let mut pre = FocusTrace::new(PRE_RUBICON);
        let mut post = FocusTrace::new(POST_RUBICON);
        for _ in 0..3 {
            pre.sample(only.clone());
            post.sample(only.clone());
        }
        assert!(!pre.is_empty() && pre.len() == 3, "non-trivial input");

        let r = read_crossing(&pre, &post, 0.05).expect("measurable");
        assert_eq!(
            r.verdict,
            RubiconVerdict::Indistinguishable,
            "nothing was deliberated, so nothing may be read as a crossing: {r:?}"
        );
        assert_eq!(r.breadth_drop, 0.0);
        assert_eq!(r.persistence_gain, 0.0);
    }

    /// A contradicted model must be VISIBLE. Broadening after the crossing is
    /// not "no signal" — it is the opposite of the prediction, and rounding it
    /// into `Indistinguishable` would hide a falsification.
    #[test]
    fn broadening_after_the_crossing_reads_as_inverted_not_as_silence() {
        let mut pre = FocusTrace::new(PRE_RUBICON);
        let mut post = FocusTrace::new(POST_RUBICON);
        for _ in 0..3 {
            pre.sample(mask(&[(9, 10)]));
            post.sample(mask(&[(1, 2), (2, 2)]));
        }
        let r = read_crossing(&pre, &post, 0.05).expect("measurable");
        assert_eq!(r.verdict, RubiconVerdict::Inverted, "reading: {r:?}");
        assert!(r.breadth_drop < 0.0, "it got broader: {}", r.breadth_drop);
    }

    /// One axis moving is `Partial` — neither a crossing nor silence.
    #[test]
    fn one_axis_alone_is_partial() {
        // Narrows, but persistence is identical (both held constant).
        let mut pre = FocusTrace::new(PRE_RUBICON);
        let mut post = FocusTrace::new(POST_RUBICON);
        for _ in 0..3 {
            pre.sample(mask(&[(1, 2)]));
            post.sample(mask(&[(9, 10)]));
        }
        let r = read_crossing(&pre, &post, 0.05).expect("measurable");
        assert_eq!(r.verdict, RubiconVerdict::Partial, "reading: {r:?}");
        assert!(r.breadth_drop > 0.0);
        assert_eq!(r.persistence_gain, 0.0, "persistence did not move");
    }

    /// **The threshold is not decoration.** Raising `epsilon` must silence a
    /// real signal and lowering it must admit it — the inertness test the
    /// falsifiability rule requires of any tolerance parameter.
    #[test]
    fn epsilon_is_a_real_knob_in_both_directions() {
        let mut pre = FocusTrace::new(PRE_RUBICON);
        let mut post = FocusTrace::new(POST_RUBICON);
        for _ in 0..3 {
            pre.sample(mask(&[(1, 9)]));
            post.sample(mask(&[(9, 10)]));
        }
        let drop = read_crossing(&pre, &post, 0.0).unwrap().breadth_drop;
        assert!(drop > 0.0, "there is a signal to silence: {drop}");

        // Below the signal: admitted.
        assert_ne!(
            read_crossing(&pre, &post, drop / 2.0).unwrap().verdict,
            RubiconVerdict::Indistinguishable,
            "a small epsilon must admit the signal"
        );
        // Above it: silenced.
        assert_eq!(
            read_crossing(&pre, &post, drop * 2.0).unwrap().verdict,
            RubiconVerdict::Indistinguishable,
            "a large epsilon must silence it"
        );
    }

    /// An unmeasurable trace yields NO reading — never one built on a default.
    #[test]
    fn an_unmeasurable_trace_refuses_rather_than_defaults() {
        let empty = FocusTrace::new(PRE_RUBICON);
        let mut single = FocusTrace::new(POST_RUBICON);
        single.sample(mask(&[(1, 4)]));
        let mut ok = FocusTrace::new(POST_RUBICON);
        ok.sample(mask(&[(1, 4)]));
        ok.sample(mask(&[(1, 4)]));

        assert_eq!(empty.breadth(), None, "no samples, no breadth");
        assert_eq!(single.persistence(), None, "one sample is not a sequence");
        assert!(single.breadth().is_some(), "but one sample IS a breadth");

        assert_eq!(read_crossing(&empty, &ok, 0.05), None);
        assert_eq!(read_crossing(&ok, &single, 0.05), None);
        // …and the pair that IS measurable does return a reading, so the three
        // `None`s above are refusals and not a function that always refuses.
        assert!(read_crossing(&ok, &ok, 0.05).is_some());
    }
}
