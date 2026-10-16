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
//! | 46..49 | activation i4 | what this sentence's update caused: the change of the statement's arm frequency over the prior state, in eighths, clamped to `-8..=7`; `0` when the update changes nothing |
//! | 50..52 | novelty → wisdom | the statement's rung after this update, `1` Staunen ..= `7` wisdom |
//! | 53..58 | belief-state update | W slot into the cohort's [`WitnessTable`]: the entry that witnessed this statement's PRIOR state; `0` = first sighting or the prior is in another cohort |
//! | 59..63 | epistemic state | `EpistemicState5`: topology from the clause path, certification from the belief state and the causes arena |
//!
//! # The belief arenas
//!
//! Every row is also offered to [`crate::belief::BeliefArena`] as one unit
//! observation, stamped with its verse: the stamp is the independence gate,
//! so disjoint verses pool and a repeat inside one verse is CHOICE, never
//! counted twice. Two arenas, two term spaces:
//!
//! - **facts**: `S –Rel(P)→ O` over word ids. `Rel` never transits.
//! - **causes**: `cause ⇒ effect` over event terms. The cause is the event
//!   whose clause the Kausal marker (`weil`) opens; every matrix event of the
//!   same verse is an effect. `Impl` does not auto-transit.
//!
//! A cause event is certified `CausalCandidate` only when its `cause ⇒
//! effect` belief rests on two or more disjoint verse stamps (and the triple
//! itself is Related). Text never certifies `Causes`. The arena's stamps are
//! a 64-source bitset that folds by modulo: past 64 verses two sources can
//! collide, which turns a revision into a CHOICE (an undercount, never a
//! double count).
//!
//! # The belief state is a Markov chain per statement
//!
//! A statement `(S, P, O)` has one state: `pos`, its independent supporting
//! verses; `tot`, the independent verses that gave `(S, P)` at all (so
//! `neg = tot - pos` gave another object); and its rung. Each
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
//! Reading order is the chain. Each row records its statement's state right
//! after its own update, and registers one [`WitnessEntry`] (its verse, its
//! facts-arena belief) in the cohort's [`WitnessTable`]. Its W slot points at
//! the entry of the statement's PREVIOUS row, as the contract says
//! (`witness_table`: "each edge's W-slot resolves to the entry that witnessed
//! the prior state"). Walking W from a row and resolving each entry back to
//! its row walks the statement's updates newest → oldest: W1 prior, W2, W3 are
//! the last three hops. A cohort is 63 consecutive rows in reading order
//! (slots `1..=63`, slot 0 = none); a prior in an earlier cohort cannot be
//! addressed by 6 bits, so W is 0 there and the report counts an arc break.
//!
//! # The ladder (bits 50..52)
//!
//! A first sighting sits at rung 1 (Staunen). A corroboration climbs one rung
//! only while the observation is resolved and the support balance stays
//! positive (`pos > neg`): a
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
use lance_graph_contract::exploration::NarsTruth;
use lance_graph_contract::facet::{CascadeShape, FacetCascade};
use lance_graph_contract::grammar::verb_lexicon::classify_verb;
use lance_graph_contract::grammar::verb_table::morton_cell;
use lance_graph_contract::hhtl::NiblePath;
use lance_graph_contract::witness_table::{WitnessEntry, WitnessTable};

use crate::belief::{BeliefArena, CStmt, Copula, ReviseOutcome, Stamp};
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

/// Rows per witness cohort: slots `1..=63` of a `WitnessTable<64>`.
pub const COHORT_ROWS: usize = 63;

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
    /// Qualia palette index per token (`contract::qualia_profile`, `0` =
    /// unnamed), aligned with `readings`; empty = none. Where the index comes
    /// from is the caller's: this crate holds no word → qualia lexicon.
    pub qualia: Vec<u8>,
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
    /// The witness cohort this row's W slot is read in.
    pub cohort: u32,
    /// The qualia register, read under [`QUALIA_SHAPE`]: pairs 0..2 are
    /// `(operand code : qualia)` for S, P, O; pair 3 is `(verb cell : 1)` when
    /// the verb classifies, `(0 : 0)` when not. Class id 0 (default class).
    pub qualia: FacetCascade,
}

/// The carving the qualia register is read under: six `u8:u8` pairs.
pub const QUALIA_SHAPE: CascadeShape = CascadeShape::G6D2;

/// Pair index of the subject in the qualia register.
pub const Q_SUBJECT: u8 = 0;
/// Pair index of the predicate.
pub const Q_PREDICATE: u8 = 1;
/// Pair index of the object.
pub const Q_OBJECT: u8 = 2;
/// Pair index of the verb cell (`verb_table::morton_cell`, flag byte).
pub const Q_VERB_CELL: u8 = 3;

/// The qualia index on pair `pair` (level 1 of the `u8:u8`).
#[must_use]
pub fn qualia_at(f: FacetCascade, pair: u8) -> u8 {
    f.cascade_byte(QUALIA_SHAPE, pair, 1)
}

/// The operand code on pair `pair` (level 0 of the `u8:u8`).
#[must_use]
pub fn operand_at(f: FacetCascade, pair: u8) -> u8 {
    f.cascade_byte(QUALIA_SHAPE, pair, 0)
}

/// The verb cell, when the verb classified. Cell `0` is a real verb
/// (`Becomes`, `Present`), so presence is the flag byte, never the cell.
#[must_use]
pub fn verb_cell(f: FacetCascade) -> Option<u8> {
    (f.cascade_byte(QUALIA_SHAPE, Q_VERB_CELL, 1) == 1)
        .then(|| f.cascade_byte(QUALIA_SHAPE, Q_VERB_CELL, 0))
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
    /// `cause ⇒ effect` observations offered to the causes arena.
    pub implications: usize,
    /// Of those, observations the arena pooled (disjoint stamps).
    pub implications_revised: usize,
    /// Rows whose prior update sits in an earlier witness cohort (W = 0).
    pub arc_breaks: usize,
    /// Events not interned because the u16 event-term space was full.
    pub event_overflow: usize,
    /// Qualia tokens attached to a subject or object (adjectival).
    pub qualia_adjectival: usize,
    /// Qualia tokens attached to an event (adverbial).
    pub qualia_adverbial: usize,
    /// Qualia tokens with no slot to attach to (or in an untraced verse).
    pub qualia_unattached: usize,
    /// A second, different qualia for an already-qualified slot; the first is
    /// kept.
    pub qualia_conflicts: usize,
    /// Triples whose verb surface classified to a 144-cell.
    pub verb_classified: usize,
    /// Triples whose verb did not classify (unknown, non-English, or a
    /// homograph like `saw` the lexicon leaves unresolved).
    pub verb_unclassified: usize,
}

/// What [`wire`] produced: the rows, the report, and the two belief arenas
/// the rows were fed into.
#[derive(Debug)]
pub struct Wired {
    /// One row per certain triple, in stream order.
    pub rows: Vec<WiredRow>,
    /// What happened.
    pub report: WireReport,
    /// Every row as `S –Rel(P)→ O`, one unit observation per verse, stamped
    /// with the verse. `Rel` never transits.
    pub facts: BeliefArena,
    /// Every Kausal claim as `cause ⇒ effect` over EVENT terms (indices into
    /// [`Self::events`]), one unit observation per verse, stamped with the
    /// verse. `Impl` does not auto-transit (belief.rs V0).
    pub causes: BeliefArena,
    /// Event term `i` is the triple `events[i]`.
    pub events: Vec<Spo>,
    /// One `WitnessTable<64>` per cohort of [`COHORT_ROWS`] rows, in reading
    /// order. Entry = `(verse + 1, Some(facts-arena belief id))`.
    pub witnesses: Vec<WitnessTable<64>>,
}

/// One unit of positive evidence at horizon 1.
fn unit() -> NarsTruth {
    NarsTruth::new(1.0, 0.5)
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
    rung: u8,
    /// `(cohort, slot)` of this statement's latest row.
    last: Option<(u32, u8)>,
}

#[derive(Debug, Default)]
struct Group {
    verses: HashSet<usize>,
    objects: HashMap<u16, StmtState>,
}

/// Fill the qualia register of one traced verse's rows.
///
/// - The operand codes come from the edge (bits 0..23).
/// - A qualia on an adjective attaches to the next noun of its sentence, on
///   every triple of that sentence where the noun is the subject or object.
/// - A qualia on any other skipped token (an adverb) attaches to an event:
///   the preceding triple of the sentence when no noun or verb stands
///   between its close and the token, otherwise the next triple, otherwise
///   the last of the sentence.
/// - The verb cell is `morton_cell(classify_verb(surface))` of the verb token
///   that closed into the triple's predicate.
fn attach_qualia(verse: &VerseIn<'_>, t: &[Traced], rows: &mut [WiredRow], rep: &mut WireReport) {
    let r = &verse.readings;
    let pos_of = |i: usize| r[i].pos.iter().next();
    let mut sentence = vec![0u32; r.len()];
    let mut s = 0;
    for (i, slot) in sentence.iter_mut().enumerate() {
        *slot = s;
        if pos_of(i) == Some(Pos::Stop) {
            s += 1;
        }
    }
    // Pair g sits at bytes 4 + 2g (lo = qualia) and 5 + 2g (hi = operand).
    let mut bytes: Vec<[u8; 16]> = rows
        .iter()
        .map(|row| {
            let mut b = [0u8; 16];
            b[5] = row.edge.s_idx();
            b[7] = row.edge.p_idx();
            b[9] = row.edge.o_idx();
            b
        })
        .collect();
    let content = |i: usize| matches!(pos_of(i), Some(Pos::Noun | Pos::Verb));
    for (i, &q) in verse.qualia.iter().enumerate() {
        if q == 0 || i >= r.len() {
            continue;
        }
        let sent = sentence[i];
        let in_sent: Vec<usize> = (0..t.len()).filter(|&k| t[k].sentence == sent).collect();
        let mut put = |k: usize, pair: u8| {
            let at = 4 + 2 * pair as usize;
            if bytes[k][at] != 0 && bytes[k][at] != q {
                rep.qualia_conflicts += 1;
            } else {
                bytes[k][at] = q;
            }
        };
        let attached = match pos_of(i) {
            Some(Pos::Adj) => {
                let noun = (i + 1..r.len())
                    .take_while(|&j| sentence[j] == sent)
                    .find(|&j| pos_of(j) == Some(Pos::Noun));
                let mut hit = false;
                if let Some(n) = noun {
                    let w = r[n].id;
                    for &k in &in_sent {
                        if t[k].spo.subject == w {
                            put(k, Q_SUBJECT);
                            hit = true;
                        }
                        if t[k].spo.object == w {
                            put(k, Q_OBJECT);
                            hit = true;
                        }
                    }
                }
                hit.then_some(true)
            }
            Some(Pos::Other) => {
                let prev = in_sent.iter().rev().copied().find(|&k| t[k].closed_at < i);
                let next = in_sent.iter().copied().find(|&k| t[k].closed_at >= i);
                let target = match (prev, next) {
                    (Some(p), _) if !(t[p].closed_at + 1..i).any(content) => Some(p),
                    (_, Some(n)) => Some(n),
                    (p, None) => p,
                };
                target.map(|k| {
                    put(k, Q_PREDICATE);
                    false
                })
            }
            _ => None,
        };
        match attached {
            Some(true) => rep.qualia_adjectival += 1,
            Some(false) => rep.qualia_adverbial += 1,
            None => rep.qualia_unattached += 1,
        }
    }
    for (k, tr) in t.iter().enumerate() {
        let last = tr.closed_at.min(r.len().saturating_sub(1));
        let verb = (0..=last)
            .rev()
            .take_while(|&j| sentence[j] == tr.sentence)
            .find(|&j| pos_of(j) == Some(Pos::Verb) && r[j].id == tr.spo.predicate);
        let cell = verb
            .and_then(|j| verse.surface.get(j))
            .and_then(|w| classify_verb(w))
            .map(|(f, tense)| morton_cell(f, tense));
        match cell {
            Some(c) => {
                bytes[k][10] = 1;
                bytes[k][11] = c;
                rep.verb_classified += 1;
            }
            None => rep.verb_unclassified += 1,
        }
    }
    for (row, b) in rows.iter_mut().zip(bytes) {
        row.qualia = FacetCascade::from_bytes(&b);
    }
}

/// A frequency change in `u8` codes (`-255..=255`) as eighths, rounded half
/// away from zero, clamped to the i4 range.
fn eighths(delta: i32) -> i8 {
    let r = if delta >= 0 {
        (delta * 8 + 127) / 255
    } else {
        (delta * 8 - 127) / 255
    };
    r.clamp(-8, 7) as i8
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
) -> Result<Wired, HarnessError> {
    let readings: Vec<Vec<Reading>> = verses.iter().map(|v| v.readings.clone()).collect();
    let forms: Vec<Vec<FormAttestation>> = verses.iter().map(|v| v.forms.clone()).collect();
    let harness = run_harness(markers, &readings, &forms, basin, abi)?;

    let mut rep = WireReport::default();
    let mut rows: Vec<WiredRow> = Vec::with_capacity(harness.encoded.len());
    let mut groups: HashMap<(u16, u16), Group> = HashMap::new();
    let mut facts = BeliefArena::new();
    let mut causes = BeliefArena::new();
    let mut events: Vec<Spo> = Vec::new();
    let mut event_id: HashMap<Spo, u16> = HashMap::new();
    let mut witnesses: Vec<WitnessTable<64>> = Vec::new();
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
                cohort: 0,
                qualia: FacetCascade::default(),
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

        if let Some(t) = traced.as_ref() {
            attach_qualia(verse, t, &mut rows[first..], &mut rep);
        } else {
            rep.qualia_unattached += verse.qualia.iter().filter(|&&q| q != 0).count();
        }

        // Kausal claims: the stating event is the cause (the clause `weil`
        // opens), every matrix event of the verse is an effect. Each pair is
        // one unit observation of `cause ⇒ effect`, stamped with the verse, so
        // the arena pools independent verses and never double counts one.
        let mut cause_sources = vec![0u32; n];
        if let (Some(j), Some(t)) = (stating[KAUSAL_LANE], traced.as_ref()) {
            let mut intern = |spo: Spo, rep: &mut WireReport| -> Option<u16> {
                if let Some(&id) = event_id.get(&spo) {
                    return Some(id);
                }
                let Ok(id) = u16::try_from(events.len()) else {
                    rep.event_overflow += 1;
                    return None;
                };
                events.push(spo);
                event_id.insert(spo, id);
                Some(id)
            };
            for (e, tr) in t.iter().enumerate() {
                if e == j || tr.origin != Origin::Main {
                    continue;
                }
                let (Some(c), Some(f)) = (intern(t[j].spo, &mut rep), intern(tr.spo, &mut rep))
                else {
                    continue;
                };
                let stmt = CStmt {
                    s: c,
                    cop: Copula::Impl,
                    p: f,
                };
                rep.implications += 1;
                if matches!(
                    causes.observe(stmt, unit(), Stamp::source(vi as u32)),
                    ReviseOutcome::Revised { .. }
                ) {
                    rep.implications_revised += 1;
                }
                let sources = causes.get(stmt).map_or(0, |b| b.stamp.0.count_ones());
                cause_sources[j] = cause_sources[j].max(sources);
            }
        }

        // The belief-state update, row by row in stream order.
        for (k, row) in rows[first..].iter_mut().enumerate() {
            let spo = row.spo;
            let causal = row.tekamolo[KAUSAL_LANE].is_some();
            let resolved =
                traced.is_some() && !ambiguous && !incomplete && !abstained && !unbound[k];

            let group = groups.entry((spo.subject, spo.predicate)).or_default();
            let tot_prior = group.verses.len() as u32;
            if group.verses.insert(vi) {
                for (o, other) in &mut group.objects {
                    if *o != spo.object {
                        if other.rung > 0 {
                            other.rung = 1;
                        }
                        rep.contests += 1;
                    }
                }
            }
            let tot = group.verses.len() as u32;
            let st = group.objects.entry(spo.object).or_default();
            let pos_prior = st.verses.len() as u32;
            let new_stmt = pos_prior == 0;
            let own_fresh = st.verses.insert(vi);
            let pos = st.verses.len() as u32;
            // The support balance: more independent verses for this object than
            // for every other object of the same (S, P).
            let balance_pos = 2 * pos > tot;
            // The activation this sentence causes: the change of the arm
            // frequency over the prior state (unknown prior = arm's own 128).
            let f_prior =
                arm_to_truth_u8(&rule(spo, pos_prior, tot_prior, vi as u32), 1).frequency;
            let f_now = arm_to_truth_u8(&rule(spo, pos, tot, (vi + 1) as u32), 1).frequency;
            let activation = eighths(i32::from(f_now) - i32::from(f_prior));
            let kind = if own_fresh {
                Update::Corroborate
            } else {
                rep.dependent += 1;
                Update::Dependent
            };
            if new_stmt {
                st.rung = 1;
            } else if kind == Update::Corroborate && resolved && balance_pos {
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
            let fact = match facts.observe(
                CStmt {
                    s: spo.subject,
                    cop: Copula::Rel(spo.predicate),
                    p: spo.object,
                },
                unit(),
                Stamp::source(vi as u32),
            ) {
                ReviseOutcome::Admitted { id }
                | ReviseOutcome::Revised { id, .. }
                | ReviseOutcome::Chosen { id, .. } => id,
            };
            // Register this row's witness; W points at the prior row's entry.
            let row_ix = first + k;
            let (cohort, slot) = ((row_ix / COHORT_ROWS) as u32, (1 + row_ix % COHORT_ROWS) as u8);
            if witnesses.len() <= cohort as usize {
                witnesses.push(WitnessTable::new());
            }
            witnesses[cohort as usize]
                .set(
                    slot,
                    WitnessEntry {
                        mailbox_ref: vi as u32 + 1,
                        spo_fact_ref: Some(u64::from(fact)),
                    },
                )
                .expect("slot is 1..=63");
            let w_prior = match st.last {
                Some((c, s)) if c == cohort => s,
                Some(_) => {
                    rep.arc_breaks += 1;
                    0
                }
                None => 0,
            };
            st.last = Some((cohort, slot));
            row.cohort = cohort;
            let certification = if pos >= 2 && balance_pos {
                if cause_sources[k] >= 2 {
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
                .with_activation(activation)
                .with_novelty_wisdom(st.rung)
                .with_w_slot(w_prior)
                .with_epistemic_raw5(state.raw());
            row.support = pos;
            row.antecedent = tot;
        }
        rep.rows = rows.len();
    }
    Ok(Wired {
        rows,
        report: rep,
        facts,
        causes,
        events,
        witnesses,
    })
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
        let w = wire(&markers, &ins, 1, &abi(), None, ReadParams::RIGHT_CORNER).unwrap();
        (w.rows, w.report)
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

    /// Can fire: three independent verses climb `dog bit man` to rung 3 with
    /// arm truth f=1, c=3/4, Related. Only the first verse moves `f` (from the
    /// unknown 128 to 255), so only it activates.
    #[test]
    fn independent_corroboration_climbs_the_ladder() {
        let v = vec![svo(DOG, BIT, MAN); 3];
        let (rows, rep) = run(&v);
        assert_eq!(rep.rows, 3);
        let last = rows[2].edge;
        assert_eq!(last.frequency_u8(), 255);
        assert_eq!(last.confidence_u8(), (3 * 255 / 4) as u8);
        let acts: Vec<i8> = rows.iter().map(|r| r.edge.activation()).collect();
        assert_eq!(acts, [4, 0, 0], "a repeat of a settled belief changes nothing");
        assert_eq!(last.novelty_wisdom(), 3);
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
        assert_eq!(e.activation(), 0, "dependent evidence moves no frequency");
        assert_eq!(epi(e).certification(), Certification3::Associated);
    }

    /// A contest drops the rival back to Staunen; the activation is the change
    /// of the row's own frequency, and the climb waits for a positive balance.
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
        assert_eq!(cat.frequency_u8(), (3 * 255 / 5) as u8);
        let acts: Vec<i8> = rows.iter().map(|r| r.edge.activation()).collect();
        // man 128→255; man stays; cat 0→85; 85→127; 127→153.
        assert_eq!(acts, [4, 0, 3, 1, 1]);
        assert_eq!(rows[2].edge.novelty_wisdom(), 1);
        // pos 2 vs 2: no positive balance, no climb.
        assert_eq!(
            rows[3].edge.novelty_wisdom(),
            1,
            "no climb without a positive balance"
        );
        assert_eq!(rows[4].edge.novelty_wisdom(), 2, "3 vs 2 climbs");
    }

    /// The contested rival restarts at Staunen: `man` reached rung 2, a `cat`
    /// verse contests it, and the next `man` verse climbs from 1, not 2.
    #[test]
    fn a_contested_rival_restarts_at_staunen() {
        let v = vec![
            svo(DOG, BIT, MAN),
            svo(DOG, BIT, MAN),
            svo(DOG, BIT, CAT),
            svo(DOG, BIT, MAN),
        ];
        let (rows, _) = run(&v);
        assert_eq!(rows[1].edge.novelty_wisdom(), 2);
        assert_eq!(rows[3].edge.activation(), 1, "170 → 191");
        assert_eq!(
            rows[3].edge.novelty_wisdom(),
            2,
            "reset to 1, then one climb"
        );
        // The man arc skips the cat row: W points at the man row of verse 1.
        assert_eq!(rows[3].edge.w_slot(), 2);
    }

    fn run_wired(verses: &[Vec<Reading>]) -> Wired {
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

    /// The W slot is a Markov arc: from the newest row of a statement, each
    /// W resolves to the witness of the prior state of the SAME statement,
    /// skipping the rival's rows, and the walk ends at W = 0.
    #[test]
    fn w_walks_the_statement_arc_backwards() {
        let v = vec![
            svo(DOG, BIT, MAN),
            svo(DOG, BIT, CAT),
            svo(DOG, BIT, MAN),
            svo(DOG, BIT, CAT),
            svo(DOG, BIT, MAN),
        ];
        let w = run_wired(&v);
        let table = &w.witnesses[0];
        let mut verses = vec![w.rows[4].verse];
        let mut facts = Vec::new();
        let mut slot = w.rows[4].edge.w_slot();
        while slot != 0 {
            let e = table.get(slot).unwrap();
            let vi = (e.mailbox_ref - 1) as usize;
            verses.push(vi);
            facts.push(e.spo_fact_ref.unwrap());
            let prior = w.rows.iter().find(|r| r.verse == vi && r.spo.object == MAN).unwrap();
            slot = prior.edge.w_slot();
        }
        assert_eq!(verses, [4, 2, 0], "newest to oldest, cat rows skipped");
        assert!(facts.windows(2).all(|p| p[0] == p[1]), "one statement");
        let cat_fact = table.get(w.rows[3].edge.w_slot()).unwrap().spo_fact_ref.unwrap();
        assert_ne!(cat_fact, facts[0], "the rival is its own arc");
        assert_eq!(w.report.arc_breaks, 0);
    }

    /// A cohort holds 63 rows; a statement whose prior lies in the previous
    /// cohort starts a new arc (W = 0) and the break is counted.
    #[test]
    fn a_cohort_boundary_breaks_the_arc() {
        let v = vec![svo(DOG, BIT, MAN); COHORT_ROWS + 2];
        let w = run_wired(&v);
        assert_eq!(w.witnesses.len(), 2);
        assert_eq!(w.rows[COHORT_ROWS - 1].edge.w_slot(), (COHORT_ROWS - 1) as u8);
        assert_eq!(w.rows[COHORT_ROWS].cohort, 1);
        assert_eq!(w.rows[COHORT_ROWS].edge.w_slot(), 0, "prior is in cohort 0");
        assert_eq!(w.rows[COHORT_ROWS + 1].edge.w_slot(), 1);
        assert_eq!(w.report.arc_breaks, 1);
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

    fn heuristics() -> GrammarHeuristics {
        GrammarHeuristics::parse("# lane\tlemma\trelation\tcount\nKausal\tweil\tmark\t273\n")
    }

    /// `der Hund biss den Mann. weil die Katze sah den Hund`: effect clause,
    /// then the cause clause `weil` opens.
    fn weil_verse() -> VerseIn<'static> {
        let mut r = svo(DOG, BIT, MAN);
        r.push(Reading::stop());
        r.push(one(WHO, Pos::Other)); // `weil`, skipped by the FSM
        r.extend(svo(CAT, SAW, DOG));
        let surface = vec![
            "der", "hund", "biss", "den", "mann", ".", "weil", "die", "katze", "sah", "den", "hund",
        ];
        VerseIn {
            readings: r,
            surface,
            forms: vec![],
            qualia: vec![],
        }
    }

    fn run_in(ins: &[VerseIn<'_>]) -> Wired {
        let markers: Vec<(u16, u16)> = (0..ins.len() as u16).map(|i| (1, i + 1)).collect();
        wire(
            &markers,
            ins,
            1,
            &abi(),
            Some(&heuristics()),
            ReadParams::RIGHT_CORNER,
        )
        .unwrap()
    }

    fn impl_stmt(w: &Wired, cause: Spo, effect: Spo) -> CStmt {
        let id = |x: Spo| w.events.iter().position(|e| *e == x).unwrap() as u16;
        CStmt {
            s: id(cause),
            cop: Copula::Impl,
            p: id(effect),
        }
    }

    /// Can fire: two independent `weil` verses pool `cause ⇒ effect` in the
    /// causes arena (two disjoint stamps), the cause event sits at the
    /// intervention rung and is certified CausalCandidate; the effect is not.
    #[test]
    fn weil_feeds_the_causes_arena() {
        let w = run_in(&[weil_verse(), weil_verse()]);
        assert_eq!(w.report.implications, 2);
        assert_eq!(w.report.implications_revised, 1);
        let (cause, effect) = (Spo::new(CAT, SAW, DOG), Spo::new(DOG, BIT, MAN));
        let b = w.causes.get(impl_stmt(&w, cause, effect)).unwrap();
        assert_eq!(b.stamp.0.count_ones(), 2);
        assert!(b.truth.confidence > 0.6, "two units pool to c = 2/3");
        let row = |v: usize, x: Spo| w.rows.iter().find(|r| r.verse == v && r.spo == x).unwrap();
        assert_eq!(row(1, cause).edge.causal_mask(), CausalMask::PO);
        assert_eq!(
            epi(row(1, cause).edge).certification(),
            Certification3::CausalCandidate
        );
        assert_eq!(row(1, effect).edge.causal_mask(), CausalMask::SO);
        assert_eq!(
            epi(row(1, effect).edge).certification(),
            Certification3::Related
        );
        assert_eq!(w.report.causal_candidates, 1);
    }

    /// Stay silent: the cause clause seen twice, but the causal claim only
    /// once — the cause is Related (two supports), not a candidate.
    #[test]
    fn one_causal_claim_is_not_a_candidate() {
        let mut plain = svo(CAT, SAW, DOG);
        plain.push(Reading::stop());
        plain.extend(svo(DOG, BIT, MAN));
        // The plain verse first, so the weil verse's cause row already has two
        // supports and a positive balance: only the single implication
        // source keeps it from CausalCandidate.
        let w = run_in(&[
            VerseIn {
                readings: plain,
                ..VerseIn::default()
            },
            weil_verse(),
        ]);
        assert_eq!(w.report.implications, 1);
        assert_eq!(w.report.implications_revised, 0);
        assert_eq!(w.report.causal_candidates, 0);
        let cause = w
            .rows
            .iter()
            .find(|r| r.verse == 1 && r.spo == Spo::new(CAT, SAW, DOG))
            .unwrap();
        assert_eq!(cause.support, 2);
        assert!(2 * cause.support > cause.antecedent, "positive balance");
        assert_eq!(epi(cause.edge).certification(), Certification3::Related);
        assert_eq!(
            cause.edge.causal_mask(),
            CausalMask::PO,
            "the claim is causal"
        );
    }

    /// A `weil` clause with no matrix event states a cause of nothing: the
    /// claim sits at the intervention rung, but no implication is observed.
    #[test]
    fn weil_without_an_effect_observes_no_implication() {
        let mut r = svo(DOG, BIT, MAN);
        r.push(one(WHO, Pos::Other));
        let v = VerseIn {
            readings: r,
            surface: vec!["der", "hund", "biss", "den", "mann", "weil"],
            forms: vec![],
            qualia: vec![],
        };
        let w = run_in(&[v.clone(), v]);
        assert_eq!(w.report.intervention_rows, 2);
        assert_eq!(w.report.implications, 0);
        assert!(w.causes.entries().is_empty());
        assert_eq!(epi(w.rows[1].edge).certification(), Certification3::Related);
    }

    /// The facts arena pools independent verses exactly as arm-discovery's
    /// confidence does for an uncontested statement, and one verse repeated
    /// is CHOICE, never pooled.
    #[test]
    fn facts_arena_agrees_with_arm_and_refuses_a_repeat() {
        let (dog_bit_man, w) = (
            CStmt {
                s: DOG,
                cop: Copula::Rel(BIT),
                p: MAN,
            },
            run_in(&vec![
                VerseIn {
                    readings: svo(DOG, BIT, MAN),
                    ..VerseIn::default()
                };
                3
            ]),
        );
        let b = w.facts.get(dog_bit_man).unwrap();
        let arm = w.rows[2].edge.confidence_u8();
        assert!(((b.truth.confidence * 255.0) as i32 - i32::from(arm)).abs() <= 1);
        assert_eq!(b.stamp.0.count_ones(), 3);

        let mut twice = svo(DOG, BIT, MAN);
        twice.push(Reading::stop());
        twice.extend(svo(DOG, BIT, MAN));
        let w = run_in(&[VerseIn {
            readings: twice,
            ..VerseIn::default()
        }]);
        let b = w.facts.get(dog_bit_man).unwrap();
        assert_eq!(b.stamp.0.count_ones(), 1, "one verse is one source");
        assert!((b.truth.confidence - 0.5).abs() < 1e-6, "not pooled");
    }

    const QUICK: u8 = 42;

    /// One verse with surfaces and per-token qualia.
    fn qverse(tokens: &[(u16, Pos, &'static str, u8)]) -> VerseIn<'static> {
        VerseIn {
            readings: tokens.iter().map(|&(id, p, _, _)| one(id, p)).collect(),
            surface: tokens.iter().map(|&(_, _, w, _)| w).collect(),
            forms: vec![],
            qualia: tokens.iter().map(|&(_, _, _, q)| q).collect(),
        }
    }

    fn qrun(v: VerseIn<'static>) -> (WiredRow, WireReport) {
        let w = wire(&[(1, 1)], &[v], 1, &abi(), None, ReadParams::RIGHT_CORNER).unwrap();
        (w.rows[0], w.report)
    }

    /// The same qualia index is adjectival on the subject and adverbial on the
    /// event: the pair carrying it is the distinction, not a tag.
    #[test]
    fn one_qualia_index_adjective_or_adverb_by_pair() {
        let (adj, ra) = qrun(qverse(&[
            (THE, Pos::Det, "the", 0),
            (WHO, Pos::Adj, "quick", QUICK),
            (DOG, Pos::Noun, "dog", 0),
            (BIT, Pos::Verb, "caused", 0),
            (THE, Pos::Det, "the", 0),
            (MAN, Pos::Noun, "man", 0),
        ]));
        assert_eq!(adj.spo, Spo::new(DOG, BIT, MAN));
        assert_eq!(qualia_at(adj.qualia, Q_SUBJECT), QUICK);
        assert_eq!(qualia_at(adj.qualia, Q_PREDICATE), 0);
        assert_eq!(qualia_at(adj.qualia, Q_OBJECT), 0);
        assert_eq!(ra.qualia_adjectival, 1);

        let (adv, rv) = qrun(qverse(&[
            (THE, Pos::Det, "the", 0),
            (DOG, Pos::Noun, "dog", 0),
            (BIT, Pos::Verb, "caused", 0),
            (THE, Pos::Det, "the", 0),
            (MAN, Pos::Noun, "man", 0),
            (WHO, Pos::Other, "quickly", QUICK),
        ]));
        assert_eq!(qualia_at(adv.qualia, Q_PREDICATE), QUICK);
        assert_eq!(qualia_at(adv.qualia, Q_SUBJECT), 0);
        assert_eq!(rv.qualia_adverbial, 1);
        // The operand codes sit beside their qualia, equal to the edge's.
        assert_eq!(operand_at(adv.qualia, Q_SUBJECT), adv.edge.s_idx());
        assert_eq!(operand_at(adv.qualia, Q_OBJECT), adv.edge.o_idx());
    }

    /// An adjective on the object lands on the object pair; an adverb before
    /// the verb lands on the event; no qualia leaves every qualia byte 0.
    #[test]
    fn object_adjective_preverbal_adverb_and_silence() {
        let (obj, _) = qrun(qverse(&[
            (THE, Pos::Det, "the", 0),
            (DOG, Pos::Noun, "dog", 0),
            (BIT, Pos::Verb, "caused", 0),
            (THE, Pos::Det, "the", 0),
            (WHO, Pos::Adj, "quick", QUICK),
            (MAN, Pos::Noun, "man", 0),
        ]));
        assert_eq!(qualia_at(obj.qualia, Q_OBJECT), QUICK);
        assert_eq!(qualia_at(obj.qualia, Q_SUBJECT), 0);
        let (pre, _) = qrun(qverse(&[
            (THE, Pos::Det, "the", 0),
            (DOG, Pos::Noun, "dog", 0),
            (WHO, Pos::Other, "quickly", QUICK),
            (BIT, Pos::Verb, "caused", 0),
            (THE, Pos::Det, "the", 0),
            (MAN, Pos::Noun, "man", 0),
        ]));
        assert_eq!(qualia_at(pre.qualia, Q_PREDICATE), QUICK);
        let (none, r) = qrun(qverse(&[
            (THE, Pos::Det, "the", 0),
            (DOG, Pos::Noun, "dog", 0),
            (BIT, Pos::Verb, "caused", 0),
            (THE, Pos::Det, "the", 0),
            (MAN, Pos::Noun, "man", 0),
        ]));
        for pair in [Q_SUBJECT, Q_PREDICATE, Q_OBJECT] {
            assert_eq!(qualia_at(none.qualia, pair), 0);
        }
        assert_eq!(
            r.qualia_adjectival + r.qualia_adverbial + r.qualia_unattached,
            0
        );
    }

    /// An adverb after the first clause's object attaches to that clause, not
    /// to the next one: no noun or verb stands between.
    #[test]
    fn a_trailing_adverb_stays_with_its_clause() {
        let w = wire(
            &[(1, 1)],
            &[qverse(&[
                (THE, Pos::Det, "the", 0),
                (DOG, Pos::Noun, "dog", 0),
                (BIT, Pos::Verb, "caused", 0),
                (THE, Pos::Det, "the", 0),
                (MAN, Pos::Noun, "man", 0),
                (WHO, Pos::Other, "quickly", QUICK),
                (THE, Pos::Other, "and", 0),
                (THE, Pos::Det, "the", 0),
                (CAT, Pos::Noun, "cat", 0),
                (SAW, Pos::Verb, "saw", 0),
                (THE, Pos::Det, "the", 0),
                (DOG, Pos::Noun, "dog", 0),
            ])],
            1,
            &abi(),
            None,
            ReadParams::RIGHT_CORNER,
        )
        .unwrap();
        assert_eq!(w.rows.len(), 2);
        assert_eq!(qualia_at(w.rows[0].qualia, Q_PREDICATE), QUICK);
        assert_eq!(qualia_at(w.rows[1].qualia, Q_PREDICATE), 0);
    }

    /// An adjective with no noun after it in its sentence stays unattached;
    /// it never reaches the next sentence's subject.
    #[test]
    fn an_adjective_does_not_cross_a_sentence() {
        let w = wire(
            &[(1, 1)],
            &[qverse(&[
                (THE, Pos::Det, "the", 0),
                (DOG, Pos::Noun, "dog", 0),
                (BIT, Pos::Verb, "caused", 0),
                (THE, Pos::Det, "the", 0),
                (MAN, Pos::Noun, "man", 0),
                (WHO, Pos::Adj, "quick", QUICK),
                (0, Pos::Stop, ".", 0),
                // The next sentence opens with `dog`, the subject of the
                // first: an unbounded noun search would qualify it there.
                (THE, Pos::Det, "the", 0),
                (DOG, Pos::Noun, "dog", 0),
                (SAW, Pos::Verb, "saw", 0),
                (THE, Pos::Det, "the", 0),
                (CAT, Pos::Noun, "cat", 0),
            ])],
            1,
            &abi(),
            None,
            ReadParams::RIGHT_CORNER,
        )
        .unwrap();
        assert_eq!(w.rows.len(), 2);
        for row in &w.rows {
            assert_eq!(qualia_at(row.qualia, Q_SUBJECT), 0);
        }
        assert_eq!(w.report.qualia_unattached, 1);
        assert_eq!(w.report.qualia_adjectival, 0);
    }

    /// The verb cell is the 144-table byte; cell 0 (Becomes, Present) is real
    /// and told apart from "unclassified" by the flag; `saw` stays
    /// unclassified (a homograph the lexicon refuses).
    #[test]
    fn verb_cell_carries_a_presence_flag() {
        use lance_graph_contract::grammar::role_keys::Tense;
        use lance_graph_contract::grammar::verb_table::VerbFamily;
        let verse = |verb: &'static str| {
            qverse(&[
                (THE, Pos::Det, "the", 0),
                (DOG, Pos::Noun, "dog", 0),
                (BIT, Pos::Verb, verb, 0),
                (THE, Pos::Det, "the", 0),
                (MAN, Pos::Noun, "man", 0),
            ])
        };
        let (caused, _) = qrun(verse("caused"));
        assert_eq!(
            verb_cell(caused.qualia),
            Some(morton_cell(VerbFamily::Causes, Tense::Past))
        );
        let (becomes, _) = qrun(verse("becomes"));
        assert_eq!(verb_cell(becomes.qualia), Some(0), "cell 0 is a real verb");
        let (saw, r) = qrun(verse("saw"));
        assert_eq!(verb_cell(saw.qualia), None);
        assert_eq!(r.verb_unclassified, 1);
    }

    #[test]
    fn plain_rows_stay_at_association() {
        let (plain, _) = run(&[svo(DOG, BIT, MAN), svo(DOG, BIT, MAN)]);
        assert_eq!(plain[1].edge.causal_mask(), CausalMask::SO);
        assert_eq!(epi(plain[1].edge).certification(), Certification3::Related);
    }
}
