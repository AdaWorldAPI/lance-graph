//! `wire` — every `CausalEdge64` field of a parsed triple, filled from the one
//! source that grounds it, at the address [`crate::hydrate`] gives it.
//!
//! The field contract is the CE64 semantic upper half (2026-10-16). Each row of
//! the table names the field's source. A field whose source says nothing stays
//! at its zero-fallback and the [`WireReport`] counts it; nothing is guessed to
//! make a field non-zero. [`crate::haspelmath::run_harness`] still emits the
//! grammar-only edge (bits 24..63 zero). This module is the layer above it that
//! adds annotations, each from a named source and never from the grammatical
//! fit.
//!
//! | bits | field | source |
//! |---|---|---|
//! | 0..23 | S, P, O codes | `run_harness`'s frozen operand ABI |
//! | 24..31 | frequency | `lance-graph-arm-discovery` [`arm_to_truth_u8`], rule `(S, P) → O` over independent verses: `f = cooccur / antecedent` |
//! | 32..39 | confidence | the same: `c = cooccur / (cooccur + k)`, `k = 1` (the NAL horizon) |
//! | 40..42 | Pearl ladder | the TEKAMOLO read: a Kausal lane stated on this event → `PO` (the claim sits at the intervention rung); otherwise `SO` (association) |
//! | 43..45 | orientation / syntax | the clause path ([`Origin`]) as a code on [`ORIENTATION_MAP`]; `0` = untraced |
//! | 46..49 | activation i4 | semantic pressure `pos - neg`, clamped to `-8..=7` |
//! | 50..52 | novelty → wisdom | the statement's rung after this update, `1` Staunen ..= `7` wisdom |
//! | 53..58 | belief-state update | the statement's last three updates: `W1` prior (53..54), `W2` (55..56), `W3` newest (57..58) |
//! | 59..63 | epistemic state | `EpistemicState5`: topology from the clause path, certification from the belief state |
//!
//! # The belief state is a Markov chain per statement
//!
//! A statement `(S, P, O)` has one state: `pos`, its independent supporting
//! verses; `tot`, the independent verses that gave `(S, P)` at all (so
//! `neg = tot - pos` gave another object); its rung; and its window. Each
//! observation updates the states it touches and nothing else, so the next
//! state depends only on the current state and the observation:
//!
//! - an observation of `(S, P, O)` from a verse not yet counted for it is a
//!   **corroboration** (`1`) of `(S, P, O)`;
//! - if that verse is also new for `(S, P)`, it is a **contest** (`2`) of every
//!   other `(S, P, O')`;
//! - an observation from a verse already counted is **dependent** (`3`): no new
//!   evidence, the NAL `DUPLICATE` / `OVERLAP` class.
//!
//! The window shifts the newest kind in at `W3`, and `W1` falls off. Each row
//! records the state of its own statement right after its own update, so the
//! rows of one statement, in stream order, are its time series.
//!
//! # The ladder (bits 50..52)
//!
//! A first sighting sits at rung 1 (Staunen). A corroboration climbs one rung
//! only while the observation is resolved and the pressure stays positive: a
//! rung is reached only when the one below holds. Resolved means single
//! readings, traced, no incomplete Haspelmath map, no TEKAMOLO abstention and,
//! for a relative clause, a bound antecedent. A contest drops the statement
//! back to rung 1. Six resolved corroborations after the first sighting,
//! without a contest, reach wisdom.
//!
//! # F/C against NAL revision
//!
//! `f` equals NAL revision of the unit observations (`pos` positive and `neg`
//! negative units at horizon 1). `c` equals it only for an uncontested
//! statement: arm-discovery counts confidence over the co-occurrence mass
//! (`c = pos / (pos + 1)`), NAL over all evidence (`c = tot / (tot + 1)`).
//! `f_and_c_against_nal_revision_of_units` pins both halves.
//!
//! # The orientation map (bits 43..45)
//!
//! A deterministic code on a semantic map of the syntactic functions the FSM
//! decides, in the Haspelmath sense: nodes are functions, and links say which
//! functions may share a form. The links follow the relativization
//! accessibility order (subject above object, Keenan & Comrie 1977):
//! `Main – Carried`, `Main – SubjRelative`, `SubjRelative – ObjRelative`.
//! Haspelmath's dative map is not encoded here: its eight functions would use
//! all eight codes and leave no unaddressed value, and [`crate::haspelmath`]
//! keeps its ordinals harness-local.

use std::collections::{HashMap, HashSet};

use causal_edge::{CausalEdge64, CausalMask};
use lance_graph_arm_discovery::{arm_to_truth_u8, CandidateRule, Item};
use lance_graph_contract::causal_witness::{CausalWitnessFacet, Locus};
use lance_graph_contract::epistemic_state5::{Certification3, Epi5Gen, EpistemicState5, Topology2};
use lance_graph_contract::hhtl::NiblePath;

use crate::fsm::{parse_readings, parse_to_spo_traced, Origin, Pos, Reading, Tagged, Traced};
use crate::haspelmath::{
    continuity, run_harness, FormAttestation, FrozenOperandAbi, HarnessError, MapVerdict,
};
use crate::spo::Spo;
use crate::tekamolo::{read_clause, GrammarHeuristics, LaneAddress, ReadParams};

/// The TEKAMOLO lane index of Kausal, in Te-Ka-Mo-Lo order.
pub const KAUSAL_LANE: usize = 1;

/// The orientation code of a clause path (bits 43..45). `0` is reserved for
/// "untraced", so a zero field never claims a path.
#[must_use]
pub const fn orientation_code(origin: Origin) -> u8 {
    match origin {
        Origin::Main => 1,
        Origin::Carried => 2,
        Origin::SubjRelative => 3,
        Origin::ObjRelative => 4,
    }
}

/// The syntactic-function map: for each code `0..=4`, its neighbours as a
/// bitmask over codes (bit `c` = code `c`). Code 0 is on no map.
pub const ORIENTATION_MAP: [u8; 5] = [
    0,
    (1 << 2) | (1 << 3), // Main: Carried, SubjRelative
    1 << 1,              // Carried: Main
    (1 << 1) | (1 << 4), // SubjRelative: Main, ObjRelative
    1 << 3,              // ObjRelative: SubjRelative
];

/// Is a set of orientation codes (bitmask over codes `1..=4`) one connected
/// region of [`ORIENTATION_MAP`]? The empty set, and any set containing code
/// 0 or a code above 4, is not.
#[must_use]
pub fn orientation_contiguous(codes: u8) -> bool {
    if codes == 0 || codes & 1 != 0 || codes >> 5 != 0 {
        return false;
    }
    let mut seen = 1u8 << codes.trailing_zeros();
    loop {
        let mut next = seen;
        for (c, links) in ORIENTATION_MAP.iter().enumerate() {
            if seen & (1 << c) != 0 {
                next |= links & codes;
            }
        }
        if next == seen {
            return seen == codes;
        }
        seen = next;
    }
}

/// The kind of one belief-state update (a 2-bit witness).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Update {
    /// An independent verse supports the statement.
    Corroborate = 1,
    /// An independent verse gave the same `(S, P)` with another object.
    Contest = 2,
    /// The verse was already counted: no new evidence.
    Dependent = 3,
}

/// Shift `kind` into a 6-bit window at `W3` (bits 4..5); `W1` falls off.
#[must_use]
pub const fn shift_window(window: u8, kind: Update) -> u8 {
    ((window & 0x3F) >> 2) | ((kind as u8) << 4)
}

/// The three updates of a window, `[W1 prior, W2, W3 newest]`; `0` = empty.
#[must_use]
pub const fn window_slots(window: u8) -> [u8; 3] {
    [window & 0b11, (window >> 2) & 0b11, (window >> 4) & 0b11]
}

/// One verse as the caller has it.
#[derive(Debug, Clone, Default)]
pub struct VerseIn<'a> {
    /// The verse's tokens with their admissible readings (no trailing stop).
    pub readings: Vec<Reading>,
    /// The surface forms, one per reading, for the TEKAMOLO read. Empty means
    /// "no surface text": the TEKAMOLO lanes and loci stay unaddressed.
    pub surface: Vec<&'a str>,
    /// Haspelmath form attestations cited for this verse.
    pub forms: Vec<FormAttestation>,
}

/// One wired triple.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WiredRow {
    /// The verse node's address.
    pub path: NiblePath,
    /// Corpus-order index (the Markov index of [`crate::hydrate`]).
    pub order: u32,
    /// Which input verse it was read in.
    pub verse: usize,
    /// The triple.
    pub spo: Spo,
    /// The 64-bit edge, every field set from its source.
    pub edge: CausalEdge64,
    /// The 24 × i4 loci.
    pub loci: CausalWitnessFacet,
    /// TEKAMOLO lane addresses, on the event that stated them only.
    pub tekamolo: [Option<LaneAddress>; 4],
    /// The clause path, when the verse could be traced.
    pub origin: Option<Origin>,
    /// Independent verses supporting the statement so far (`pos`).
    pub support: u32,
    /// Independent verses that gave `(S, P)` so far (`tot`).
    pub antecedent: u32,
}

/// What the wiring did, so a zero field is never mistaken for a reading.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct WireReport {
    /// Rows written.
    pub rows: usize,
    /// Verses with a multi-reading token or a parse mismatch: orientation 0.
    pub untraced_verses: usize,
    /// Relative-clause rows whose antecedent event was found within reach.
    pub antecedent_bound: usize,
    /// Relative-clause rows whose antecedent event was not found.
    pub antecedent_unbound: usize,
    /// TEKAMOLO loci written (a pointer to another event of the verse).
    pub tekamolo_loci: usize,
    /// TEKAMOLO lanes committed on a stating event.
    pub tekamolo_lanes: usize,
    /// Rows at the intervention rung (a Kausal lane stated on the event).
    pub intervention_rows: usize,
    /// Contest updates applied to other statements.
    pub contests: usize,
    /// Dependent observations (a verse already counted).
    pub dependent: usize,
    /// Rows certified `CausalCandidate`.
    pub causal_candidates: usize,
}

const REACH_BACK: i64 = -8;
const REACH_FWD: i64 = 7;

fn offset(from: usize, to: usize) -> Option<i8> {
    let d = to as i64 - from as i64;
    (d != 0 && (REACH_BACK..=REACH_FWD).contains(&d)).then_some(d as i8)
}

/// The traced clause paths of one verse, or `None` when a token has more than
/// one reading or the traced parse disagrees with the multi-reading one.
fn trace(readings: &[Reading], certain: &[Spo]) -> Option<Vec<Traced>> {
    let mut tagged = Vec::with_capacity(readings.len() + 1);
    for r in readings {
        if r.pos.len() != 1 {
            return None;
        }
        tagged.push(Tagged::new(r.id, r.pos.iter().next()?));
    }
    tagged.push(Tagged::new(0, Pos::Stop));
    let traced = parse_to_spo_traced(&tagged);
    let same =
        traced.len() == certain.len() && traced.iter().zip(certain).all(|(t, c)| t.spo == *c);
    same.then_some(traced)
}

/// The state of one statement `(S, P, O)`.
#[derive(Debug, Clone, Default)]
struct StmtState {
    verses: HashSet<usize>,
    causal_verses: HashSet<usize>,
    window: u8,
    rung: u8,
}

#[derive(Debug, Default)]
struct Group {
    verses: HashSet<usize>,
    objects: HashMap<u16, StmtState>,
}

/// The arm-discovery rule `(S, P) → O` over the counts so far.
fn rule(spo: Spo, pos: u32, tot: u32, window: u32) -> CandidateRule {
    CandidateRule {
        antecedent: vec![
            Item::new(0, u32::from(spo.subject)),
            Item::new(1, u32::from(spo.predicate)),
        ],
        consequent: vec![Item::new(2, u32::from(spo.object))],
        cooccur: pos,
        antecedent_count: tot,
        window,
    }
}

/// Wire every certain triple of the corpus.
///
/// # Errors
///
/// Exactly what [`run_harness`] refuses: unaligned verses, missing evidence
/// roots, a disconnected complete distribution, parser overflow, unaddressed
/// triples, operands outside the frozen ABI.
pub fn wire(
    markers: &[(u16, u16)],
    verses: &[VerseIn<'_>],
    basin: u8,
    abi: &FrozenOperandAbi,
    heuristics: Option<&GrammarHeuristics>,
    params: ReadParams,
) -> Result<(Vec<WiredRow>, WireReport), HarnessError> {
    let readings: Vec<Vec<Reading>> = verses.iter().map(|v| v.readings.clone()).collect();
    let forms: Vec<Vec<FormAttestation>> = verses.iter().map(|v| v.forms.clone()).collect();
    let harness = run_harness(markers, &readings, &forms, basin, abi)?;

    let mut rep = WireReport::default();
    let mut rows: Vec<WiredRow> = Vec::with_capacity(harness.encoded.len());
    let mut groups: HashMap<(u16, u16), Group> = HashMap::new();
    let mut next = 0usize;

    for (vi, verse) in verses.iter().enumerate() {
        let mut input = verse.readings.clone();
        input.push(Reading::stop());
        let parsed = parse_readings(&input);
        let n = parsed.certain.len();
        if n == 0 {
            continue;
        }
        let traced = trace(&verse.readings, &parsed.certain);
        if traced.is_none() {
            rep.untraced_verses += 1;
        }
        let incomplete = verse
            .forms
            .iter()
            .any(|f| continuity(f.functions, f.coverage_complete) == MapVerdict::Incomplete);
        let ambiguous = !parsed.ambiguous.is_empty();

        // TEKAMOLO: one read over the verse; each committed lane lands on the
        // event whose clause holds the stating token.
        let clause = match heuristics {
            Some(h) if !verse.surface.is_empty() && verse.surface.len() == verse.readings.len() => {
                Some(read_clause(h, &verse.surface, params))
            }
            _ => None,
        };
        let abstained = clause.is_some_and(|c| c.abstained > 0);
        let stating: [Option<usize>; 4] = std::array::from_fn(|lane| {
            let at = clause?.at[lane]?;
            let t = traced.as_ref()?;
            Some(
                t.iter()
                    .position(|x| x.closed_at >= at)
                    .unwrap_or(t.len() - 1),
            )
        });

        let first = rows.len();
        for k in 0..n {
            let h = harness.hydration.triples[next];
            let enc = harness.encoded[next];
            next += 1;
            rows.push(WiredRow {
                path: h.path,
                order: h.order,
                verse: vi,
                spo: h.spo,
                edge: enc.word,
                loci: CausalWitnessFacet::ZERO,
                tekamolo: [None; 4],
                origin: traced.as_ref().map(|t| t[k].origin),
                support: 0,
                antecedent: 0,
            });
        }

        // Relative-clause antecedent: the matrix event of the same sentence
        // whose subject is the antecedent word.
        let mut unbound = vec![false; n];
        if let Some(t) = traced.as_ref() {
            for (k, tr) in t.iter().enumerate() {
                if !tr.origin.is_relative() {
                    continue;
                }
                let ante_word = match tr.origin {
                    Origin::SubjRelative => tr.spo.subject,
                    _ => tr.spo.object,
                };
                let matrix = t.iter().enumerate().skip(k + 1).find(|(_, m)| {
                    m.sentence == tr.sentence
                        && !m.origin.is_relative()
                        && m.spo.subject == ante_word
                });
                match matrix.and_then(|(j, _)| offset(k, j)) {
                    Some(o) => {
                        rows[first + k].loci = rows[first + k].loci.with(Locus::Antecedent, o);
                        rep.antecedent_bound += 1;
                    }
                    None => {
                        unbound[k] = true;
                        rep.antecedent_unbound += 1;
                    }
                }
            }
        }

        // TEKAMOLO lanes on the stating event, loci elsewhere in the verse.
        for (lane, at) in stating.iter().enumerate() {
            let (Some(j), Some(c)) = (*at, clause) else {
                continue;
            };
            rows[first + j].tekamolo[lane] = c.lanes[lane];
            rep.tekamolo_lanes += 1;
            for k in 0..n {
                if let Some(o) = offset(k, j) {
                    rows[first + k].loci = rows[first + k].loci.with(Locus::ALL[lane], o);
                    rep.tekamolo_loci += 1;
                }
            }
        }

        // The belief-state update, row by row in stream order.
        for (k, row) in rows[first..].iter_mut().enumerate() {
            let spo = row.spo;
            let causal = row.tekamolo[KAUSAL_LANE].is_some();
            let resolved =
                traced.is_some() && !ambiguous && !incomplete && !abstained && !unbound[k];

            let group = groups.entry((spo.subject, spo.predicate)).or_default();
            if group.verses.insert(vi) {
                for (o, other) in &mut group.objects {
                    if *o != spo.object {
                        other.window = shift_window(other.window, Update::Contest);
                        if other.rung > 0 {
                            other.rung = 1;
                        }
                        rep.contests += 1;
                    }
                }
            }
            let tot = group.verses.len() as u32;
            let st = group.objects.entry(spo.object).or_default();
            let new_stmt = st.verses.is_empty();
            let own_fresh = st.verses.insert(vi);
            if causal {
                st.causal_verses.insert(vi);
            }
            let pos = st.verses.len() as u32;
            let pressure = (i64::from(pos) - i64::from(tot - pos)).clamp(-8, 7) as i8;
            let kind = if own_fresh {
                Update::Corroborate
            } else {
                rep.dependent += 1;
                Update::Dependent
            };
            st.window = shift_window(st.window, kind);
            if new_stmt {
                st.rung = 1;
            } else if kind == Update::Corroborate && resolved && pressure > 0 {
                st.rung = (st.rung + 1).min(7);
            }

            let truth = arm_to_truth_u8(&rule(spo, pos, tot, (vi + 1) as u32), 1);
            let pearl = if causal {
                rep.intervention_rows += 1;
                CausalMask::PO
            } else {
                CausalMask::SO
            };
            let topology = match row.origin {
                None => Topology2::Unknown,
                Some(Origin::Main) => Topology2::Direct,
                Some(_) if unbound[k] => Topology2::IndirectUnknown,
                Some(_) => Topology2::IndirectKnown,
            };
            let certification = if pos >= 2 && pressure > 0 {
                if st.causal_verses.len() >= 2 {
                    rep.causal_candidates += 1;
                    Certification3::CausalCandidate
                } else {
                    Certification3::Related
                }
            } else {
                Certification3::Associated
            };
            let state = EpistemicState5::new(Epi5Gen::V1, topology, certification);

            let mut edge = row.edge;
            edge.set_frequency_u8(truth.frequency);
            edge.set_confidence_u8(truth.confidence);
            edge.set_causal_mask(pearl);
            edge.set_direction(row.origin.map_or(0, orientation_code));
            row.edge = edge
                .with_activation(pressure)
                .with_novelty_wisdom(st.rung)
                .with_w_slot(st.window)
                .with_epistemic_raw5(state.raw());
            row.support = pos;
            row.antecedent = tot;
        }
        rep.rows = rows.len();
    }
    Ok((rows, rep))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fsm::PosSet;
    use lance_graph_contract::exploration::NarsTruth;

    const DOG: u16 = 10;
    const MAN: u16 = 11;
    const CAT: u16 = 12;
    const BIT: u16 = 20;
    const SAW: u16 = 21;
    const THE: u16 = 30;
    const WHO: u16 = 31;

    fn one(id: u16, pos: Pos) -> Reading {
        Reading::new(id, PosSet::single(pos))
    }

    /// `the S V the O`.
    fn svo(s: u16, v: u16, o: u16) -> Vec<Reading> {
        vec![
            one(THE, Pos::Det),
            one(s, Pos::Noun),
            one(v, Pos::Verb),
            one(THE, Pos::Det),
            one(o, Pos::Noun),
        ]
    }

    fn abi() -> FrozenOperandAbi {
        FrozenOperandAbi::new(1, &[DOG, MAN, CAT, BIT, SAW, THE, WHO]).unwrap()
    }

    fn run(verses: &[Vec<Reading>]) -> (Vec<WiredRow>, WireReport) {
        let markers: Vec<(u16, u16)> = (0..verses.len() as u16).map(|i| (1, i + 1)).collect();
        let ins: Vec<VerseIn<'_>> = verses
            .iter()
            .map(|r| VerseIn {
                readings: r.clone(),
                ..VerseIn::default()
            })
            .collect();
        wire(&markers, &ins, 1, &abi(), None, ReadParams::RIGHT_CORNER).unwrap()
    }

    fn epi(e: CausalEdge64) -> EpistemicState5 {
        EpistemicState5::decode(Epi5Gen::V1, e.epistemic_raw5()).unwrap()
    }

    #[test]
    fn orientation_map_is_contiguous_where_the_hierarchy_is() {
        let b = |c: u8| 1u8 << c;
        assert!(orientation_contiguous(b(1) | b(2)));
        assert!(orientation_contiguous(b(1) | b(3) | b(4)));
        assert!(
            !orientation_contiguous(b(1) | b(4)),
            "Main–ObjRelative skips SubjRelative"
        );
        assert!(
            !orientation_contiguous(b(2) | b(3)),
            "Carried–SubjRelative only via Main"
        );
        assert!(!orientation_contiguous(0));
        assert!(!orientation_contiguous(b(0) | b(1)));
    }

    #[test]
    fn window_shifts_newest_into_w3() {
        let w = shift_window(0, Update::Corroborate);
        assert_eq!(window_slots(w), [0, 0, 1]);
        let w = shift_window(w, Update::Contest);
        assert_eq!(window_slots(w), [0, 1, 2]);
        let w = shift_window(w, Update::Dependent);
        assert_eq!(window_slots(w), [1, 2, 3]);
        let w = shift_window(w, Update::Corroborate);
        assert_eq!(window_slots(w), [2, 3, 1], "W1 fell off");
    }

    /// Can fire: three independent verses climb `dog bit man` to rung 3 with
    /// arm truth f=1, c=3/4, pressure +3, three corroborations, Related.
    #[test]
    fn independent_corroboration_climbs_the_ladder() {
        let v = vec![svo(DOG, BIT, MAN); 3];
        let (rows, rep) = run(&v);
        assert_eq!(rep.rows, 3);
        let last = rows[2].edge;
        assert_eq!(last.frequency_u8(), 255);
        assert_eq!(last.confidence_u8(), (3 * 255 / 4) as u8);
        assert_eq!(last.activation(), 3);
        assert_eq!(last.novelty_wisdom(), 3);
        assert_eq!(window_slots(last.w_slot()), [1, 1, 1]);
        assert_eq!(epi(last).certification(), Certification3::Related);
        assert_eq!(epi(last).topology(), Topology2::Direct);
        assert_eq!(last.direction(), orientation_code(Origin::Main));
        assert_eq!(last.causal_mask(), CausalMask::SO);
        // The rows of one statement are its time series.
        let rungs: Vec<u8> = rows.iter().map(|r| r.edge.novelty_wisdom()).collect();
        assert_eq!(rungs, [1, 2, 3]);
        assert_eq!(
            epi(rows[0].edge).certification(),
            Certification3::Associated
        );
    }

    /// Stay silent: a repeat inside ONE verse is dependent — no new support,
    /// no climb, no certification beyond Associated.
    #[test]
    fn a_repeat_in_one_verse_is_dependent() {
        let mut v = svo(DOG, BIT, MAN);
        v.push(Reading::stop());
        v.extend(svo(DOG, BIT, MAN));
        let (rows, rep) = run(&[v]);
        assert_eq!(rows.len(), 2);
        assert_eq!(rep.dependent, 1);
        let e = rows[1].edge;
        assert_eq!(rows[1].support, 1);
        assert_eq!(e.novelty_wisdom(), 1, "no climb on dependent evidence");
        assert_eq!(window_slots(e.w_slot()), [0, 1, 3]);
        assert_eq!(epi(e).certification(), Certification3::Associated);
    }

    /// A contest drops the rival back to Staunen, and the pressure of a
    /// contested statement reads the balance.
    #[test]
    fn a_contest_resets_the_rival_and_presses_down() {
        let v = vec![
            svo(DOG, BIT, MAN),
            svo(DOG, BIT, MAN),
            svo(DOG, BIT, CAT),
            svo(DOG, BIT, CAT),
            svo(DOG, BIT, CAT),
        ];
        let (rows, rep) = run(&v);
        assert_eq!(
            rep.contests, 3,
            "each later verse contests the existing rival"
        );
        let cat = rows[4].edge;
        assert_eq!(cat.activation(), 1, "pos 3, neg 2");
        assert_eq!(cat.frequency_u8(), (3 * 255 / 5) as u8);
        assert_eq!(rows[2].edge.activation(), -1, "pos 1 vs man's 2");
        assert_eq!(rows[2].edge.novelty_wisdom(), 1);
    }

    /// F/C stays arm-discovery's law. `f` equals NAL revision of the unit
    /// observations; `c` equals it only while uncontested, because
    /// arm-discovery counts confidence over the co-occurrence mass.
    #[test]
    fn f_and_c_against_nal_revision_of_units() {
        let pos_u = NarsTruth::new(1.0, 0.5);
        let neg_u = NarsTruth::new(0.0, 0.5);
        for (pos, neg) in [(1u32, 0u32), (3, 0), (3, 2), (1, 4)] {
            let mut t = pos_u;
            for _ in 1..pos {
                t = t.revision(&pos_u);
            }
            for _ in 0..neg {
                t = t.revision(&neg_u);
            }
            let arm = arm_to_truth_u8(&rule(Spo::new(1, 2, 3), pos, pos + neg, 1), 1);
            let nal_f = (t.frequency * 255.0).floor() as i32;
            assert!(
                (i32::from(arm.frequency) - nal_f).abs() <= 1,
                "f({pos},{neg}): arm {} nal {nal_f}",
                arm.frequency
            );
            let nal_c = (t.confidence * 255.0).floor() as i32;
            let agree = (i32::from(arm.confidence) - nal_c).abs() <= 1;
            assert_eq!(agree, neg == 0, "c agrees iff uncontested ({pos},{neg})");
        }
    }

    /// A subject relative gets its orientation code and an Antecedent pointer
    /// to the matrix event; topology IndirectKnown.
    #[test]
    fn subject_relative_points_at_its_matrix_event() {
        // the dog who bit the man saw the cat
        let v = vec![
            one(THE, Pos::Det),
            one(DOG, Pos::Noun),
            one(WHO, Pos::Rel),
            one(BIT, Pos::Verb),
            one(THE, Pos::Det),
            one(MAN, Pos::Noun),
            one(SAW, Pos::Verb),
            one(THE, Pos::Det),
            one(CAT, Pos::Noun),
        ];
        let (rows, rep) = run(&[v]);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].origin, Some(Origin::SubjRelative));
        assert_eq!(rows[0].edge.direction(), 3);
        assert_eq!(rows[0].loci.at(Locus::Antecedent), 1);
        assert_eq!(rep.antecedent_bound, 1);
        assert_eq!(epi(rows[0].edge).topology(), Topology2::IndirectKnown);
        assert_eq!(
            rows[1].loci.at(Locus::Antecedent),
            0,
            "matrix event has no pointer"
        );
    }

    /// An ambiguous token leaves the verse untraced: orientation 0, topology
    /// Unknown, and no climb.
    #[test]
    fn an_untraced_verse_claims_no_path() {
        // A token both readings skip: the triple is certain, the verse is not
        // single-reading, so it cannot be traced.
        let mut amb = vec![Reading::new(WHO, PosSet::single(Pos::Det).with(Pos::Other))];
        amb.extend(svo(DOG, BIT, MAN));
        let (rows, rep) = run(&[svo(DOG, BIT, MAN), amb]);
        assert_eq!(rep.untraced_verses, 1);
        let r = rows
            .get(1)
            .expect("the certain triple survives the ambiguity");
        assert_eq!(r.edge.direction(), 0);
        assert_eq!(epi(r.edge).topology(), Topology2::Unknown);
        assert_eq!(r.edge.novelty_wisdom(), 1, "unresolved: no climb");
    }

    /// Pearl ↔ TEKAMOLO: a Kausal lane stated in two independent verses puts
    /// the event at the intervention rung and certifies CausalCandidate; the
    /// same statement without the lane stays SO / Related.
    #[test]
    fn kausal_lane_sets_the_intervention_rung() {
        let h = GrammarHeuristics::parse(
            "# lane\tlemma\trelation\tcount\nKausal\tweil\tmark\t273\nTemporal\theute\tadvmod\t637\n",
        );
        let surf = ["der", "hund", "biss", "den", "mann", "weil"];
        let mut tok = svo(DOG, BIT, MAN);
        tok.push(one(THE, Pos::Other)); // `weil`, skipped by the FSM
        let markers = [(1, 1), (1, 2)];
        let ins = vec![
            VerseIn {
                readings: tok.clone(),
                surface: surf.to_vec(),
                forms: vec![],
            },
            VerseIn {
                readings: tok,
                surface: surf.to_vec(),
                forms: vec![],
            },
        ];
        let (rows, rep) = wire(
            &markers,
            &ins,
            1,
            &abi(),
            Some(&h),
            ReadParams::RIGHT_CORNER,
        )
        .unwrap();
        assert_eq!(rep.intervention_rows, 2);
        assert_eq!(rows[1].edge.causal_mask(), CausalMask::PO);
        assert_eq!(
            epi(rows[1].edge).certification(),
            Certification3::CausalCandidate
        );
        let (plain, _) = run(&[svo(DOG, BIT, MAN), svo(DOG, BIT, MAN)]);
        assert_eq!(plain[1].edge.causal_mask(), CausalMask::SO);
        assert_eq!(epi(plain[1].edge).certification(), Certification3::Related);
    }
}
