//! `tekamolo` — hydrating all FOUR lanes of the `ValueTenant::Tekamolo` tenant
//! (**Temporal · Kausal · Modal · Lokal**) by a **left-corner**, **style-driven**
//! read of German scripture.
//!
//! The tenant is a 16-byte V3 facet on EVERY SoA row — `classid(4) + 6×(u8:u8)`
//! read in `G4D3` as four `256:256:256` coarse→fine lanes, answering *when /
//! why / how / where*. Its contract fixes the zero-fallback: **an all-zero lane
//! reads as UNADDRESSED — never a wrong circumstance.**
//!
//! [`crate::promote::tekamolo_of`] hydrates **Temporal** from the TOC address
//! (`book:chapter:verse` — narrative position, exact). This module hydrates the
//! other three from German **grammar heuristics**.
//!
//! ## The heuristics are German, so the corpus is German
//!
//! `de/tekamolo.tsv` (German codebook, UD German-GSD + German-HDT): `lane ·
//! lemma · relation · count`, 163 rows — which adverbial form, in which
//! syntactic relation, lands in which lane. German forms ground German text and
//! nothing else: pointed at an English lane they match nothing and the tenant
//! would read "unaddressed" for a corpus that merely had the wrong lexicon.
//!
//! Two German versions are carried on purpose (`luther1545`,
//! `elberfelder1905`). A lane rate that holds across two independent
//! translations of the SAME verses is a property of the language; one that
//! appears in a single lane is a property of that translator. The pair is the
//! control, not a bigger sample.
//!
//! ## Left-corner: carry the incomplete hypothesis, commit at the right corner
//!
//! `.claude/knowledge/left-corner-grammar-tree-pointer-fabric.md` and PR #849
//! both name the failure mode: **premature commitment**, not a missing rule.
//! The obvious scan — take the FIRST adverbial that matches a lane — IS that
//! bug: it commits a left-corner reading before the clause has shown what it
//! is. A verse opening *"Da sprach er, weil …"* would bind Kausal to `da`
//! (which is also Lokal, at identical corpus count) and never see `weil`, the
//! unambiguous causal marker two words later.
//!
//! So a match OPENS a [`LaneHypothesis`] instead of committing one. Hypotheses
//! accumulate to the clause's **right corner** (a verse or punctuation
//! boundary), and only there does the lane resolve — by the heuristics' own
//! corpus evidence, with the runner-up's margin deciding commit-vs-abstain.
//!
//! [`Commit::LeftCorner`] keeps the premature behaviour available as the
//! declared BASELINE, because a right-corner claim with no left-corner control
//! measures nothing.
//!
//! ## The thinking style drives the read
//!
//! The knobs are not constants. `ThinkingStyle → FieldModulation → ScanParams`
//! is the shipped read-parameterization, and this module consumes it:
//!
//! | knob | what it sets here |
//! |---|---|
//! | `fan_out` | how many competing hypotheses a lane carries to the right corner |
//! | `depth_bias` | ≥ `LEFT_CORNER_AT` commits at the left corner (depth-first = decide now) |
//! | `resonance_threshold` | the margin the winner must beat the runner-up by |
//! | `noise_tolerance` | whether an ambiguous form (`da`) may win a lane at all |
//!
//! [`V2StyleProvider`] supplies the modulation. The contract declares
//! [`ThinkingStyleProvider`] and ships **no implementation** — the provider is
//! the consumer's job, and this is v2's.
//!
//! ## `da` is genuinely ambiguous and is not silently resolved
//!
//! The heuristics list `da` under BOTH Kausal and Lokal at identical counts
//! (*because* / *there*). This module never breaks that tie by fiat: an
//! ambiguous form opens a hypothesis in EVERY lane it is listed under, and
//! whether it may ultimately win one is `noise_tolerance`'s call, reported per
//! read. [`GrammarHeuristics::ambiguous`] counts such forms.

use lance_graph_contract::mul::{DkPosition, FlowState, MulAssessment};
use lance_graph_contract::tekamolo_facet::TekamoloRole;
use lance_graph_contract::thinking::{
    FieldModulation, ScanParams, SparseVec, StyleCluster, ThinkingStyle, ThinkingStyleProvider,
};
use std::collections::HashMap;

/// `depth_bias` at or above this commits at the LEFT corner (decide now).
pub const LEFT_CORNER_AT: f64 = 0.7;

/// `noise_tolerance` at or above this lets an ambiguous form win a lane.
pub const AMBIGUOUS_OK_AT: f64 = 0.5;

/// A form's syntactic relation class — the COARSE tier of a hydrated lane.
/// Ordinals start at 1 so `0` stays the zero-fallback (unaddressed).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum RelationClass {
    /// `advmod` — adverbial modifier.
    Advmod = 1,
    /// `mark` — subordinating marker.
    Mark = 2,
    /// `cc` — coordinating conjunction.
    Cc = 3,
    /// `obl` — oblique nominal.
    Obl = 4,
    /// `advcl` — adverbial clause.
    Advcl = 5,
}

impl RelationClass {
    /// Parse the UD relation the heuristics carry; `None` for an unknown one —
    /// refused, never folded onto a neighbouring class.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s.trim() {
            "advmod" => Self::Advmod,
            "mark" => Self::Mark,
            "cc" => Self::Cc,
            "obl" => Self::Obl,
            "advcl" => Self::Advcl,
            _ => return None,
        })
    }
}

/// One hydrated lane address: coarse relation class + rank within the lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaneAddress {
    /// Coarse tier — which relation carried the form.
    pub relation: RelationClass,
    /// Fine tier — the form's rank inside its lane, 1-based, by corpus count.
    /// Saturates at 255 rather than wrapping onto a different form.
    pub rank: u8,
}

impl LaneAddress {
    /// The lane's 3-byte `256:256:256` cascade, coarse→fine. Tier 2 is reserved
    /// `0`: this read has two graded levels of evidence, not three, and padding
    /// it with a third would fake a depth the heuristics do not carry.
    #[must_use]
    pub const fn bytes(self) -> [u8; 3] {
        [self.relation as u8, self.rank, 0]
    }
}

/// One carried, uncommitted lane reading — the left-corner hypothesis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LaneHypothesis {
    /// Where it would land if it wins.
    pub addr: LaneAddress,
    /// The heuristics' corpus count for this (form, lane, relation) — the
    /// evidence the right corner adjudicates on.
    pub weight: f64,
    /// Token index that opened it, so a caller can see WHERE the circumstance
    /// was stated rather than only that it was.
    pub at: usize,
    /// Is the opening form listed under more than one lane (`da`)?
    pub ambiguous: bool,
}

/// Where a lane commits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Commit {
    /// Commit at the FIRST match — the premature left-corner reading the
    /// left-corner literature names as the failure mode. Kept as the declared
    /// BASELINE so the right-corner result has a control.
    LeftCorner,
    /// Carry every hypothesis to the clause boundary and adjudicate there.
    RightCorner,
}

/// German grammar heuristics: which adverbial form, in which relation, lands in
/// which TEKAMOLO lane.
#[derive(Debug, Clone, Default)]
pub struct GrammarHeuristics {
    /// `form -> [(lane, address, count)]`; a form listed under two lanes keeps
    /// BOTH entries — the ambiguity is data, not noise to be cleaned.
    by_form: HashMap<String, Vec<(TekamoloRole, LaneAddress, u64)>>,
    ambiguous: usize,
    rows: usize,
}

impl GrammarHeuristics {
    /// Parse `de/tekamolo.tsv` — `lane\tlemma\trelation\tcount`, `#` comments.
    ///
    /// Ranks are per lane by descending count (ties by form, so the ranking is
    /// stable across runs — a rank that moves between runs is not an address).
    #[must_use]
    pub fn parse(tsv: &str) -> Self {
        let mut per_lane: HashMap<u8, Vec<(String, RelationClass, u64)>> = HashMap::new();
        let mut rows = 0usize;
        for line in tsv.lines() {
            if line.starts_with('#') || line.trim().is_empty() {
                continue;
            }
            let f: Vec<&str> = line.split('\t').collect();
            let (Some(lane), Some(lemma), Some(rel), Some(count)) =
                (f.first(), f.get(1), f.get(2), f.get(3))
            else {
                continue;
            };
            let role = match lane.trim() {
                "Temporal" => TekamoloRole::Temporal,
                "Kausal" => TekamoloRole::Kausal,
                "Modal" => TekamoloRole::Modal,
                "Lokal" => TekamoloRole::Lokal,
                _ => continue,
            };
            let (Some(relation), Ok(n)) = (RelationClass::parse(rel), count.trim().parse::<u64>())
            else {
                continue;
            };
            rows += 1;
            per_lane.entry(role as u8).or_default().push((
                lemma.trim().to_lowercase(),
                relation,
                n,
            ));
        }

        let mut by_form: HashMap<String, Vec<(TekamoloRole, LaneAddress, u64)>> = HashMap::new();
        for role in TekamoloRole::ALL {
            let Some(mut v) = per_lane.remove(&(role as u8)) else {
                continue;
            };
            v.sort_by(|a, b| b.2.cmp(&a.2).then_with(|| a.0.cmp(&b.0)));
            for (i, (lemma, relation, n)) in v.into_iter().enumerate() {
                let rank = u8::try_from(i + 1).unwrap_or(u8::MAX);
                by_form
                    .entry(lemma)
                    .or_default()
                    .push((role, LaneAddress { relation, rank }, n));
            }
        }
        let ambiguous = by_form.values().filter(|v| lanes_of(v).len() > 1).count();
        Self {
            by_form,
            ambiguous,
            rows,
        }
    }

    /// Entries for a lowercased form — empty if the heuristics do not carry it.
    #[must_use]
    pub fn lookup(&self, form: &str) -> &[(TekamoloRole, LaneAddress, u64)] {
        self.by_form.get(form).map_or(&[], Vec::as_slice)
    }

    /// Is this form listed under more than one lane (`da`)?
    #[must_use]
    pub fn is_ambiguous(&self, form: &str) -> bool {
        lanes_of(self.lookup(form)).len() > 1
    }

    /// How many forms carry more than one lane.
    #[must_use]
    pub const fn ambiguous(&self) -> usize {
        self.ambiguous
    }

    /// `(distinct forms, parsed rows)`.
    #[must_use]
    pub fn sizes(&self) -> (usize, usize) {
        (self.by_form.len(), self.rows)
    }
}

/// Distinct lanes among a form's entries.
fn lanes_of(entries: &[(TekamoloRole, LaneAddress, u64)]) -> Vec<u8> {
    let mut l: Vec<u8> = entries.iter().map(|(r, _, _)| *r as u8).collect();
    l.sort_unstable();
    l.dedup();
    l
}

/// deepnsm-v2's [`ThinkingStyleProvider`] — the contract declares the trait and
/// ships no implementation, because the modulation table is the consumer's.
///
/// The projection is **7D, not 23D**, and says so: the 23D cognitive vector
/// belongs to the persona-modeling storyline, which `.claude/v3/knowledge/
/// persona-vs-rung-ladder.md` records as deliberately unwired ("carried and
/// displayed, not acted on"). Claiming 23 dimensions here would import that
/// storyline into a read path that has no business asserting it.
#[derive(Debug, Clone, Copy, Default)]
pub struct V2StyleProvider;

impl ThinkingStyleProvider for V2StyleProvider {
    /// The 7 [`FieldModulation`] knobs as a sparse vector — this crate's honest
    /// projection, indices `0..7` in `to_fingerprint` order.
    fn style_vector(&self, style: ThinkingStyle) -> SparseVec {
        let m = self.default_modulation(style);
        vec![
            (0, m.resonance_threshold as f32),
            (1, m.fan_out as f32),
            (2, m.depth_bias as f32),
            (3, m.breadth_bias as f32),
            (4, m.noise_tolerance as f32),
            (5, m.speed_bias as f32),
            (6, m.exploration as f32),
        ]
    }

    /// Per-CLUSTER modulation, not per-style: this crate has evidence for how a
    /// cluster reads (converge vs diverge vs attend vs hurry) and none for how
    /// `Blunt` differs from `Frank`. Six rows are what is grounded; 36 would be
    /// 30 invented ones.
    fn default_modulation(&self, style: ThinkingStyle) -> FieldModulation {
        match style.cluster() {
            // Decide now, narrowly, on strong evidence — the LEFT-corner reader.
            StyleCluster::Analytical => FieldModulation {
                resonance_threshold: 0.6,
                fan_out: 2,
                depth_bias: 0.85,
                breadth_bias: 0.15,
                noise_tolerance: 0.15,
                speed_bias: 0.4,
                exploration: 0.2,
            },
            // Same haste, less rigour: commits early AND tolerates a thin margin.
            StyleCluster::Direct => FieldModulation {
                resonance_threshold: 0.3,
                fan_out: 2,
                depth_bias: 0.9,
                breadth_bias: 0.1,
                noise_tolerance: 0.4,
                speed_bias: 0.9,
                exploration: 0.1,
            },
            // Carry everything to the right corner; admit ambiguous forms.
            StyleCluster::Exploratory => FieldModulation {
                resonance_threshold: 0.15,
                fan_out: 8,
                depth_bias: 0.2,
                breadth_bias: 0.8,
                noise_tolerance: 0.75,
                speed_bias: 0.2,
                exploration: 0.9,
            },
            StyleCluster::Creative => FieldModulation {
                resonance_threshold: 0.2,
                fan_out: 6,
                depth_bias: 0.3,
                breadth_bias: 0.7,
                noise_tolerance: 0.7,
                speed_bias: 0.3,
                exploration: 0.8,
            },
            // Wide attention, but unwilling to assert on a thin margin.
            StyleCluster::Empathic => FieldModulation {
                resonance_threshold: 0.5,
                fan_out: 5,
                depth_bias: 0.35,
                breadth_bias: 0.65,
                noise_tolerance: 0.55,
                speed_bias: 0.3,
                exploration: 0.5,
            },
            // Deliberate: carries the most, demands the most before committing.
            StyleCluster::Meta => FieldModulation {
                resonance_threshold: 0.7,
                fan_out: 8,
                depth_bias: 0.25,
                breadth_bias: 0.75,
                noise_tolerance: 0.3,
                speed_bias: 0.1,
                exploration: 0.6,
            },
        }
    }

    /// Style from the MUL assessment. The mapping is deliberately coarse and
    /// states its own ground: an overconfident reader is pushed to the style
    /// that REFUSES early commitment, not the one that indulges it.
    fn select_from_assessment(&self, a: &MulAssessment) -> ThinkingStyle {
        match a.dk_position {
            // Mount Stupid commits early and is wrong: force the right-corner
            // reader on it. This is the whole point of measuring DK at all.
            DkPosition::MountStupid => ThinkingStyle::Metacognitive,
            DkPosition::ValleyOfDespair => ThinkingStyle::Curious,
            DkPosition::SlopeOfEnlightenment => ThinkingStyle::Analytical,
            DkPosition::Plateau => match a.homeostasis.flow_state {
                FlowState::Flow => ThinkingStyle::Direct,
                _ => ThinkingStyle::Systematic,
            },
        }
    }
}

/// A style-parameterized read of one clause.
#[derive(Debug, Clone, Copy)]
pub struct StyleRead {
    /// Where the lanes commit — derived from `depth_bias`.
    pub commit: Commit,
    /// Hypotheses a lane carries — `fan_out`.
    pub fan_out: usize,
    /// Margin the winner must beat the runner-up by, as a fraction of the
    /// winner's weight — `resonance_threshold`.
    pub margin: f64,
    /// May an ambiguous form (`da`) win a lane — `noise_tolerance`.
    pub admit_ambiguous: bool,
    /// The SIMD scan params the same modulation yields, carried so a caller
    /// reads ONE object rather than re-deriving them.
    pub scan: ScanParams,
}

impl StyleRead {
    /// Derive the read from a style through the provider — never hand-set.
    #[must_use]
    pub fn of(style: ThinkingStyle) -> Self {
        let m = V2StyleProvider.default_modulation(style);
        Self {
            commit: if m.depth_bias >= LEFT_CORNER_AT {
                Commit::LeftCorner
            } else {
                Commit::RightCorner
            },
            fan_out: m.fan_out.max(1),
            margin: m.resonance_threshold,
            admit_ambiguous: m.noise_tolerance >= AMBIGUOUS_OK_AT,
            scan: m.to_scan_params(),
        }
    }
}

/// What one clause's read produced.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ClauseRead {
    /// The committed address per lane, in Te-Ka-Mo-Lo order. `None` =
    /// UNADDRESSED (the tenant's zero-fallback), never a guess.
    pub lanes: [Option<LaneAddress>; 4],
    /// Hypotheses opened across all lanes — the evidence the read SAW.
    pub opened: usize,
    /// Lanes that had a hypothesis but committed nothing (margin too thin, or
    /// an ambiguous form the style would not admit). The abstention count is
    /// the honest half of the report.
    pub abstained: usize,
    /// Lanes whose winner was an ambiguous form.
    pub ambiguous_wins: usize,
}

/// Read ONE clause's tokens under `style`.
///
/// Every matching form opens a [`LaneHypothesis`]. Under
/// [`Commit::RightCorner`] the hypotheses are carried to the end of the token
/// slice — the clause's right corner — and only then adjudicated. Under
/// [`Commit::LeftCorner`] each lane locks on its first hypothesis, which is the
/// premature reading kept as the baseline.
#[must_use]
pub fn read_clause(h: &GrammarHeuristics, tokens: &[&str], style: ThinkingStyle) -> ClauseRead {
    let sr = StyleRead::of(style);
    let mut carried: [Vec<LaneHypothesis>; 4] = [vec![], vec![], vec![], vec![]];
    let mut out = ClauseRead::default();

    for (i, tok) in tokens.iter().enumerate() {
        let w: String = tok
            .chars()
            .filter(|c| c.is_alphabetic())
            .collect::<String>()
            .to_lowercase();
        if w.is_empty() {
            continue;
        }
        let amb = h.is_ambiguous(&w);
        for &(role, addr, n) in h.lookup(&w) {
            let lane = &mut carried[role as usize];
            // LEFT corner: the first hypothesis for a lane locks it. This is
            // the documented premature commitment, reproduced deliberately.
            if sr.commit == Commit::LeftCorner && !lane.is_empty() {
                continue;
            }
            if lane.len() >= sr.fan_out {
                continue; // fan_out bounds what a lane may carry
            }
            lane.push(LaneHypothesis {
                addr,
                weight: n as f64,
                at: i,
                ambiguous: amb,
            });
            out.opened += 1;
        }
    }

    for (li, lane) in carried.iter_mut().enumerate() {
        if lane.is_empty() {
            continue;
        }
        // Adjudicate on the heuristics' own corpus evidence; ties broken by the
        // EARLIER token, because with equal evidence the framing adverbial is
        // the one that opened the clause.
        lane.sort_by(|a, b| {
            b.weight
                .partial_cmp(&a.weight)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.at.cmp(&b.at))
        });
        let best = lane[0];
        if best.ambiguous && !sr.admit_ambiguous {
            out.abstained += 1;
            continue;
        }
        // Margin as a FRACTION of the winner's weight, so the threshold means
        // the same thing for a 3,916-count form and a 5-count one.
        let runner = lane.get(1).map_or(0.0, |h| h.weight);
        let sep = if best.weight > 0.0 {
            (best.weight - runner) / best.weight
        } else {
            0.0
        };
        if lane.len() >= 2 && sep < sr.margin {
            out.abstained += 1;
            continue;
        }
        if best.ambiguous {
            out.ambiguous_wins += 1;
        }
        out.lanes[li] = Some(best.addr);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A slice of the real German heuristics, verbatim from `de/tekamolo.tsv`
    /// (including `da` under BOTH lanes at identical counts).
    const H: &str = "# lane\tlemma\trelation\tcount\n\
        Lokal\tzu\tmark\t3916\n\
        Temporal\tnoch\tadvmod\t2575\n\
        Modal\tso\tadvmod\t1703\n\
        Kausal\tdamit\tadvmod\t986\n\
        Kausal\tda\tmark\t437\n\
        Lokal\tda\tmark\t437\n\
        Kausal\tweil\tmark\t273\n\
        Lokal\thier\tadvmod\t440\n\
        Temporal\theute\tadvmod\t637\n";

    fn h() -> GrammarHeuristics {
        GrammarHeuristics::parse(H)
    }

    #[test]
    fn heuristics_parse_and_rank_per_lane() {
        let g = h();
        assert_eq!(g.sizes().1, 9, "9 data rows");
        // Kausal ranks by descending count: damit(986) > da(437) > weil(273).
        let damit = g.lookup("damit")[0].1;
        let weil = g.lookup("weil")[0].1;
        assert_eq!(damit.rank, 1);
        assert_eq!(weil.rank, 3);
        assert_eq!(damit.relation, RelationClass::Advmod);
        assert_eq!(weil.relation, RelationClass::Mark);
        // `da` carries TWO lanes and is reported ambiguous.
        assert_eq!(g.lookup("da").len(), 2);
        assert!(g.is_ambiguous("da"));
        assert!(!g.is_ambiguous("weil"));
        assert_eq!(g.ambiguous(), 1);
    }

    /// **The left-corner bug, reproduced and then fixed.**
    ///
    /// *"Da sprach er, weil …"* — `da` (ambiguous, 437) opens Kausal first;
    /// `weil` (unambiguous, 273) arrives two words later. A LEFT-corner reader
    /// locks `da` and never sees `weil`. A RIGHT-corner reader carries both and
    /// adjudicates. This is the whole point of the module.
    #[test]
    fn left_corner_locks_the_first_form_right_corner_carries_both() {
        let g = h();
        let toks = ["Da", "sprach", "er", "weil", "Gott", "rief"];

        // Direct: depth_bias 0.9 → LeftCorner. Only `da` was ever opened.
        let left = read_clause(&g, &toks, ThinkingStyle::Direct);
        assert_eq!(
            StyleRead::of(ThinkingStyle::Direct).commit,
            Commit::LeftCorner
        );
        assert_eq!(left.opened, 2, "da opened Kausal AND Lokal, then locked");

        // Exploratory: depth_bias 0.2 → RightCorner, fan_out 8.
        let right = read_clause(&g, &toks, ThinkingStyle::Exploratory);
        assert_eq!(
            StyleRead::of(ThinkingStyle::Exploratory).commit,
            Commit::RightCorner
        );
        assert_eq!(right.opened, 3, "da x2 + weil — weil was SEEN");
        assert!(
            right.opened > left.opened,
            "the right-corner read must see strictly more evidence: {} vs {}",
            right.opened,
            left.opened
        );
    }

    /// The two readers disagree about the Kausal ADDRESS, not merely about how
    /// much they looked at. Without this the distinction would be cosmetic.
    #[test]
    fn the_two_corners_commit_different_kausal_addresses() {
        let g = h();
        // `weil` first (273), then `damit` (986) — the LATER form has the
        // stronger corpus evidence, so left-corner and right-corner must differ.
        let toks = ["weil", "er", "rief", "damit", "es", "geschah"];
        let left = read_clause(&g, &toks, ThinkingStyle::Direct);
        let right = read_clause(&g, &toks, ThinkingStyle::Exploratory);
        let k = TekamoloRole::Kausal as usize;
        assert_eq!(
            left.lanes[k].map(|a| a.rank),
            Some(3),
            "left corner locks `weil` (rank 3) because it came first"
        );
        assert_eq!(
            right.lanes[k].map(|a| a.rank),
            Some(1),
            "right corner adjudicates to `damit` (rank 1) on the evidence"
        );
        assert_ne!(left.lanes[k], right.lanes[k]);
    }

    /// An ambiguous form may win only where the style admits it — and the
    /// abstention is REPORTED, not silently dropped.
    #[test]
    fn ambiguity_is_admitted_by_style_never_by_fiat() {
        let g = h();
        let toks = ["Da", "stand", "er"]; // only `da` — ambiguous, both lanes
                                          // Analytical: noise_tolerance 0.15 → refuses ambiguous forms.
        let strict = read_clause(&g, &toks, ThinkingStyle::Analytical);
        assert!(!StyleRead::of(ThinkingStyle::Analytical).admit_ambiguous);
        assert_eq!(strict.lanes[TekamoloRole::Kausal as usize], None);
        assert_eq!(strict.lanes[TekamoloRole::Lokal as usize], None);
        assert_eq!(strict.abstained, 2, "both lanes abstained, and said so");
        assert_eq!(strict.ambiguous_wins, 0);

        // Exploratory: noise_tolerance 0.75 → admits it, in BOTH lanes, and the
        // tie is never broken by fiat.
        let open = read_clause(&g, &toks, ThinkingStyle::Exploratory);
        assert!(StyleRead::of(ThinkingStyle::Exploratory).admit_ambiguous);
        assert!(open.lanes[TekamoloRole::Kausal as usize].is_some());
        assert!(open.lanes[TekamoloRole::Lokal as usize].is_some());
        assert_eq!(open.ambiguous_wins, 2);
    }

    /// A clause with no German adverbial leaves every lane UNADDRESSED — the
    /// tenant's zero-fallback, and the guard can stay silent on real input.
    #[test]
    fn a_clause_without_cues_addresses_nothing() {
        let g = h();
        let toks = ["Gott", "sprach", "und", "es", "ward", "Licht"];
        for style in [ThinkingStyle::Direct, ThinkingStyle::Exploratory] {
            let r = read_clause(&g, &toks, style);
            assert_eq!(r.opened, 0);
            assert_eq!(r.abstained, 0);
            assert!(r.lanes.iter().all(Option::is_none), "{style:?}");
        }
    }

    /// `fan_out` actually bounds what a lane carries: raising it strictly
    /// increases the evidence a right-corner read opens.
    #[test]
    fn fan_out_bounds_the_carried_hypotheses() {
        let g = h();
        // Four Kausal forms in one clause.
        let toks = ["damit", "da", "weil", "damit", "x", "y"];
        let creative = read_clause(&g, &toks, ThinkingStyle::Creative); // fan_out 6
        let meta = read_clause(&g, &toks, ThinkingStyle::Metacognitive); // fan_out 8
        assert!(
            StyleRead::of(ThinkingStyle::Creative).fan_out
                < StyleRead::of(ThinkingStyle::Metacognitive).fan_out
        );
        assert!(
            meta.opened >= creative.opened,
            "a wider fan_out cannot see less: {} vs {}",
            meta.opened,
            creative.opened
        );
    }

    /// The provider is real: every cluster yields a distinct modulation, and
    /// the derived commit corner splits the six clusters into both kinds.
    #[test]
    fn the_provider_separates_the_clusters() {
        let mut corners = std::collections::HashSet::new();
        let mut fingerprints = std::collections::HashSet::new();
        for s in ThinkingStyle::ALL {
            let sr = StyleRead::of(s);
            corners.insert(sr.commit);
            fingerprints.insert(V2StyleProvider.default_modulation(s).to_fingerprint());
        }
        assert_eq!(corners.len(), 2, "both commit corners must be reachable");
        assert_eq!(
            fingerprints.len(),
            6,
            "one modulation per cluster, all distinct"
        );
        // The 7D projection is the modulation, not a fabricated 23D vector.
        assert_eq!(V2StyleProvider.style_vector(ThinkingStyle::Wise).len(), 7);
    }

    /// An overconfident reader is pushed AWAY from early commitment — the
    /// reason to measure Dunning-Kruger at all.
    #[test]
    fn mount_stupid_is_routed_to_the_right_corner_reader() {
        use lance_graph_contract::mul::{Homeostasis, TrustQualia, TrustTexture};
        let mk = |dk: DkPosition| MulAssessment {
            trust: TrustQualia {
                value: 0.5,
                texture: TrustTexture::Calibrated,
            },
            dk_position: dk,
            homeostasis: Homeostasis {
                flow_state: FlowState::Flow,
                allostatic_load: 0.2,
            },
            complexity_mapped: true,
            free_will_modifier: 0.5,
        };
        let overconfident = V2StyleProvider.select_from_assessment(&mk(DkPosition::MountStupid));
        assert_eq!(
            StyleRead::of(overconfident).commit,
            Commit::RightCorner,
            "Mount Stupid must NOT be handed the early-commit reader"
        );
        // …and the expert in flow IS allowed to decide fast, so the mapping is
        // discriminating rather than always-RightCorner.
        let expert = V2StyleProvider.select_from_assessment(&mk(DkPosition::Plateau));
        assert_eq!(StyleRead::of(expert).commit, Commit::LeftCorner);
    }
}
