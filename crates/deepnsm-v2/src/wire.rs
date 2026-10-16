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
//! | 43..45 | orientation / syntax | the Haspelmath function the relation is read in ([`orientation_of`]): the dative-map functions the clause's attested forms allow, narrowed by its TEKAMOLO lanes; `0` = plain core argument or undecided. The clause path ([`Origin`]) goes to the topology in 59..63 |
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
//! state depends only on the current state and the observation.
//!
//! # The Gadamer gate decides; counting follows
//!
//! Each `(S, P)` group is an interpretive horizon
//! (`contract::revision::GadamerRevision`): its claims are the objects met,
//! its roots the independent verses. Every observation is an encounter, and
//! the gate's verdict decides whether anything is counted:
//!
//! - a new verse for the only object so far is `IndependentConfirmation` (or
//!   `HorizonExpansion` for a first sighting): counted;
//! - a new verse while another object is in view is `HorizonFusion`: counted,
//!   and both objects join the group's tension;
//! - a verse already counted is `Echo`: nothing counted;
//! - a rival object from a verse already counted is `ContradictionPreserved`:
//!   suspended, nothing counted, the tension kept.
//!
//! Claims are never withdrawn and the tension only grows (G3): a contest
//! never resets the rival. Only admitted observations reach the facts arena.
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
//! for a relative clause, a bound antecedent. A contest leaves the rival's
//! rung as it is. Six resolved, admitted observations with a positive balance
//! after the first sighting reach wisdom.
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

use causal_edge::{CausalEdge64, CausalMask, Orientation};
use lance_graph_arm_discovery::{arm_to_truth_u8, CandidateRule, Item};
use lance_graph_contract::causal_witness::{CausalWitnessFacet, Locus};
use lance_graph_contract::epistemic_state5::{Certification3, Epi5Gen, EpistemicState5, Topology2};
use lance_graph_contract::exploration::NarsTruth;
use lance_graph_contract::facet::{CascadeShape, FacetCascade};
use lance_graph_contract::grammar::verb_lexicon::classify_verb;
use lance_graph_contract::grammar::verb_table::morton_cell;
use lance_graph_contract::hhtl::NiblePath;
use lance_graph_contract::revision::{
    BasisView, CodebookId, EncounterEvidence, EvidenceMask, EvidentialEffect, GadamerRevision,
    GrammarId, HorizonId, InterpretiveHorizon, LanguageId, LensId, QuestionId, RevisionDelta,
    RevisionKind, RevisionPolicy,
};
use lance_graph_contract::witness_table::{WitnessEntry, WitnessTable};

use crate::belief::{BeliefArena, CStmt, Copula, ReviseOutcome, Stamp};
use crate::fsm::{parse_readings, parse_to_spo_traced, Origin, Pos, Reading, Tagged, Traced};
use crate::haspelmath::{
    continuity, run_harness, DativeFunction, FormAttestation, FrozenOperandAbi, HarnessError,
    MapVerdict,
};
use crate::spo::Spo;
use crate::tekamolo::{read_clause, GrammarHeuristics, LaneAddress, ReadParams};

/// The TEKAMOLO lane index of Kausal, in Te-Ka-Mo-Lo order.
pub const KAUSAL_LANE: usize = 1;

/// The orientation a Haspelmath function is written as (bits 43..45): the
/// function the relation is read in, on the dative semantic map
/// (Haspelmath 2003, Fig. 8.1). The code table is the contract's
/// [`Orientation`]; this only names which [`DativeFunction`] each one is.
/// `Judicantis` has no orientation.
#[must_use]
pub const fn orientation_of(f: DativeFunction) -> Option<Orientation> {
    match f {
        DativeFunction::Direction => Some(Orientation::Direction),
        DativeFunction::Recipient => Some(Orientation::Recipient),
        DativeFunction::Experiencer => Some(Orientation::Experiencer),
        DativeFunction::Purpose => Some(Orientation::Purpose),
        DativeFunction::Beneficiary => Some(Orientation::Beneficiary),
        DativeFunction::PredicativePossessor => Some(Orientation::PredicativePossessor),
        DativeFunction::ExternalPossessor => Some(Orientation::ExternalPossessor),
        DativeFunction::Judicantis => None,
    }
}

/// The function an orientation names; `None` for [`Orientation::Core`].
#[must_use]
pub const fn orientation_function(o: Orientation) -> Option<DativeFunction> {
    match o {
        Orientation::Core => None,
        Orientation::Direction => Some(DativeFunction::Direction),
        Orientation::Recipient => Some(DativeFunction::Recipient),
        Orientation::Experiencer => Some(DativeFunction::Experiencer),
        Orientation::Purpose => Some(DativeFunction::Purpose),
        Orientation::Beneficiary => Some(DativeFunction::Beneficiary),
        Orientation::PredicativePossessor => Some(DativeFunction::PredicativePossessor),
        Orientation::ExternalPossessor => Some(DativeFunction::ExternalPossessor),
    }
}

/// TEKAMOLO lanes that select a dative-map function when a form is
/// multifunctional: Kausal (final reading) → Purpose, Lokal → Direction.
const LANE_FUNCTION: [u8; 4] = [
    0,
    DativeFunction::Purpose.bit(),
    0,
    DativeFunction::Direction.bit(),
];

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
    /// The object is part of its `(S, P)` group's preserved tension: another
    /// object was met for the same `(S, P)` and neither was withdrawn.
    pub contested: bool,
    /// What the Gadamer gate made of this observation (`None`: the group ran
    /// out of gate bits and the observation was not admitted).
    pub revision: Option<RevisionKind>,
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
    /// Rows whose orientation names a dative-map function.
    pub orientation_coded: usize,
    /// Rows left at 0 because the forms allow several functions.
    pub orientation_ambiguous: usize,
    /// Rows whose one function has no code (Judicantis).
    pub orientation_unrepresented: usize,
    /// Rows whose Quorum locus points at an agreeing row.
    pub quorum_bound: usize,
    /// Rows whose Contradiction locus points at a disagreeing row.
    pub contradiction_bound: usize,
    /// Observations the gate suspended: a contradiction with no new
    /// independent root (`ContradictionPreserved`). Nothing is counted.
    pub suspended: usize,
    /// Observations not admitted because a group had more than 256 distinct
    /// verses or objects.
    pub gate_overflow: usize,
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

/// The gate's mask width: group-local roots and claims, 256 each.
type GateMask = [u64; 4];

/// Bit `i` of a [`GateMask`].
fn gate_bit(i: u16) -> GateMask {
    let mut m = [0u64; 4];
    m[usize::from(i / 64)] = 1 << (i % 64);
    m
}

const GATE_BITS: u16 = 256;

struct Group {
    verses: HashSet<usize>,
    objects: HashMap<u16, StmtState>,
    /// The `(S, P)` interpretive horizon: claims are this group's objects,
    /// roots its independent verses (group-local bit indices).
    horizon: InterpretiveHorizon<(), GateMask>,
    roots: HashMap<usize, u16>,
    claims: HashMap<u16, u16>,
}

impl Default for Group {
    fn default() -> Self {
        let empty = GateMask::empty();
        Self {
            verses: HashSet::new(),
            objects: HashMap::new(),
            horizon: InterpretiveHorizon {
                id: HorizonId(0),
                awareness: (),
                question: QuestionId(0),
                language: LanguageId(0),
                grammar: GrammarId(0),
                codebook: CodebookId(0),
                lens: LensId(0),
                projected_claims: empty,
                independent_roots: empty,
                inherited_roots: empty,
                unresolved_tension: empty,
                revision_index: 0,
            },
            roots: HashMap::new(),
            claims: HashMap::new(),
        }
    }
}

impl Group {
    /// Pass one observation of `(S, P, object)` from verse `vi` through the
    /// Gadamer gate and adopt the resulting horizon. `None` when the group has
    /// run out of root or claim bits (the observation is not admitted).
    fn encounter(&mut self, vi: usize, object: u16) -> Option<RevisionDelta<(), GateMask>> {
        let next_root = self.roots.len() as u16;
        let next_claim = self.claims.len() as u16;
        let root = *self.roots.entry(vi).or_insert(next_root);
        let claim = *self.claims.entry(object).or_insert(next_claim);
        if root >= GATE_BITS || claim >= GATE_BITS {
            return None;
        }
        let prior = &self.horizon;
        let own = gate_bit(claim);
        // Claims are never withdrawn: the working whole keeps every object
        // the group has met, so a rival stays in view as live tension.
        let proposed = prior.projected_claims.union(&own);
        let others = prior.projected_claims.difference(&own);
        let contradictions = if others.is_empty() {
            GateMask::empty()
        } else {
            others.union(&own)
        };
        let encounter = EncounterEvidence {
            proposed_claims: proposed,
            independent_roots: gate_bit(root),
            inherited_roots: GateMask::empty(),
            resistance: GateMask::empty(),
            contradictions,
            affected_parts: contradictions,
        };
        let ancestry = BasisView {
            ancestry_independent_roots: prior.independent_roots,
            ancestry_derived_roots: GateMask::empty(),
            ancestor_claims: prior.projected_claims,
            closes_cycle: false,
        };
        let delta = GadamerRevision.revise(prior, &encounter, &ancestry);
        self.horizon = delta.resulting.clone();
        Some(delta)
    }

    /// Whether `object` is part of this group's preserved tension.
    fn contested(&self, object: u16) -> bool {
        self.claims.get(&object).is_some_and(|&c| {
            c < GATE_BITS && self.horizon.unresolved_tension.intersects(&gate_bit(c))
        })
    }
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
            Some(Pos::Adv | Pos::Other) => {
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
    let mut fact_ids: HashMap<CStmt, u32> = HashMap::new();
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
                contested: false,
                revision: None,
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
                // Test every matrix event of the sentence within the i4 reach,
                // before or after the relative clause, nearest first (the
                // earlier one on a tie); the antecedent noun may be the
                // matrix subject or object.
                let matrix = t
                    .iter()
                    .enumerate()
                    .filter(|(j, m)| {
                        *j != k
                            && m.sentence == tr.sentence
                            && !m.origin.is_relative()
                            && (m.spo.subject == ante_word || m.spo.object == ante_word)
                            && offset(k, *j).is_some()
                    })
                    .min_by_key(|(j, _)| (j.abs_diff(k), *j > k));
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

        // Orientation: the dative-map functions of the forms attested inside
        // each traced clause span, narrowed by the row's TEKAMOLO lanes.
        let mut undecided = vec![false; n];
        if let Some(t) = traced.as_ref() {
            for (k, tr) in t.iter().enumerate() {
                // The clause span: after the previous triple of the same
                // sentence, never before the sentence's own start.
                let sentence_start = verse.readings[..tr.closed_at]
                    .iter()
                    .rposition(|r| r.pos.contains(Pos::Stop))
                    .map_or(0, |i| i + 1);
                let start = t[..k]
                    .iter()
                    .rev()
                    .find(|p| p.sentence == tr.sentence)
                    .map_or(sentence_start, |p| (p.closed_at + 1).max(sentence_start));
                let span = &verse.readings
                    [start.min(tr.closed_at)..=tr.closed_at.min(verse.readings.len() - 1)];
                let mut f = 0u8;
                for r in span {
                    for a in verse.forms.iter().filter(|a| a.word == r.id) {
                        f |= a.functions;
                    }
                }
                let lanes = rows[first + k]
                    .tekamolo
                    .iter()
                    .enumerate()
                    .fold(
                        0u8,
                        |m, (l, a)| {
                            if a.is_some() {
                                m | LANE_FUNCTION[l]
                            } else {
                                m
                            }
                        },
                    );
                if f & lanes != 0 {
                    f &= lanes;
                }
                let orientation = match f.count_ones() {
                    0 => Orientation::Core,
                    1 => {
                        let func = DativeFunction::ALL[f.trailing_zeros() as usize];
                        orientation_of(func).unwrap_or_else(|| {
                            rep.orientation_unrepresented += 1;
                            Orientation::Core
                        })
                    }
                    _ => {
                        rep.orientation_ambiguous += 1;
                        undecided[k] = true;
                        Orientation::Core
                    }
                };
                if orientation != Orientation::Core {
                    rep.orientation_coded += 1;
                }
                rows[first + k].edge.set_orientation(orientation);
            }
        }

        if let Some(t) = traced.as_ref() {
            attach_qualia(verse, t, &mut rows[first..], &mut rep);
        } else {
            rep.qualia_unattached += verse.qualia.iter().filter(|&&q| q != 0).count();
        }

        // The belief-state update, row by row in stream order.
        let mut admitted_rows = vec![false; n];
        for (k, row) in rows[first..].iter_mut().enumerate() {
            let spo = row.spo;
            let causal = row.tekamolo[KAUSAL_LANE].is_some();
            let resolved = traced.is_some()
                && !ambiguous
                && !incomplete
                && !abstained
                && !unbound[k]
                && !undecided[k];

            let group = groups.entry((spo.subject, spo.predicate)).or_default();
            // The gate decides whether and in which sense this observation
            // revises the (S, P) horizon; counting runs only if it admits.
            let delta = group.encounter(vi, spo.object);
            let effect = delta.as_ref().map(|d| d.evidential_effect);
            match delta.as_ref().map(|d| d.kind) {
                None => rep.gate_overflow += 1,
                Some(RevisionKind::HorizonFusion) => rep.contests += 1,
                Some(RevisionKind::ContradictionPreserved) => rep.suspended += 1,
                Some(RevisionKind::Echo) => rep.dependent += 1,
                Some(_) => {}
            }
            let admitted = effect == Some(EvidentialEffect::IncreaseEligible);
            let tot_prior = group.verses.len() as u32;
            if admitted {
                group.verses.insert(vi);
            }
            let tot = group.verses.len() as u32;
            let contested = group.contested(spo.object);
            let st = group.objects.entry(spo.object).or_default();
            let pos_prior = st.verses.len() as u32;
            let new_stmt = st.rung == 0;
            if admitted {
                st.verses.insert(vi);
            }
            let pos = st.verses.len() as u32;
            // The support balance: more independent verses for this object than
            // for every other object of the same (S, P).
            let balance_pos = 2 * pos > tot;
            // The activation this sentence causes: the change of the arm
            // frequency over the prior state (unknown prior = arm's own 128).
            let f_prior = arm_to_truth_u8(&rule(spo, pos_prior, tot_prior, vi as u32), 1).frequency;
            let f_now = arm_to_truth_u8(&rule(spo, pos, tot, (vi + 1) as u32), 1).frequency;
            let activation = eighths(i32::from(f_now) - i32::from(f_prior));
            if new_stmt {
                st.rung = 1;
            } else if admitted && resolved && balance_pos {
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
            // The facts arena counts, so it sees only admitted observations;
            // a suspended or echoed row points at the fact as it stands (none
            // for a claim never admitted).
            let stmt = CStmt {
                s: spo.subject,
                cop: Copula::Rel(spo.predicate),
                p: spo.object,
            };
            let fact = if admitted {
                let id = match facts.observe(stmt, unit(), Stamp::source(vi as u32)) {
                    ReviseOutcome::Admitted { id }
                    | ReviseOutcome::Revised { id, .. }
                    | ReviseOutcome::Chosen { id, .. } => id,
                };
                fact_ids.insert(stmt, id);
                Some(id)
            } else {
                fact_ids.get(&stmt).copied()
            };
            // Register this row's witness; W points at the prior row's entry.
            let row_ix = first + k;
            let (cohort, slot) = (
                (row_ix / COHORT_ROWS) as u32,
                (1 + row_ix % COHORT_ROWS) as u8,
            );
            if witnesses.len() <= cohort as usize {
                witnesses.push(WitnessTable::new());
            }
            witnesses[cohort as usize]
                .set(
                    slot,
                    WitnessEntry {
                        mailbox_ref: vi as u32 + 1,
                        spo_fact_ref: fact.map(u64::from),
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
            // CausalCandidate is decided after the Kausal claims are counted.
            let certification = if pos >= 2 && balance_pos {
                Certification3::Related
            } else {
                Certification3::Associated
            };
            let state = EpistemicState5::new(Epi5Gen::V1, topology, certification);

            let mut edge = row.edge;
            edge.set_frequency_u8(truth.frequency);
            edge.set_confidence_u8(truth.confidence);
            edge.set_causal_mask(pearl);
            row.edge = edge
                .with_activation(activation)
                .with_novelty_wisdom(st.rung)
                .with_w_slot(w_prior)
                .with_epistemic_raw5(state.raw());
            row.support = pos;
            row.antecedent = tot;
            row.contested = contested;
            row.revision = delta.map(|d| d.kind);
            admitted_rows[k] = admitted;
        }

        // Kausal claims: the stating event is the cause (the clause `weil`
        // opens), every matrix event of the verse is an effect. Each pair is
        // one unit observation of `cause ⇒ effect`, stamped with the verse, so
        // the arena pools independent verses and never double counts one.
        // It counts, so it runs after the gate: a pair is observed only when
        // both the cause row and the effect row were admitted (G5).
        if let (Some(j), Some(t)) = (stating[KAUSAL_LANE], traced.as_ref()) {
            let mut cause_sources = 0u32;
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
                if e == j || tr.origin != Origin::Main || !admitted_rows[j] || !admitted_rows[e] {
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
                cause_sources =
                    cause_sources.max(causes.get(stmt).map_or(0, |b| b.stamp.0.count_ones()));
            }
            // Two independent implication sources promote a Related cause.
            let row = &mut rows[first + j];
            let state = EpistemicState5::decode(Epi5Gen::V1, row.edge.epistemic_raw5())
                .expect("written above as V1");
            if cause_sources >= 2 && state.certification() == Certification3::Related {
                rep.causal_candidates += 1;
                let up = EpistemicState5::new(
                    Epi5Gen::V1,
                    state.topology(),
                    Certification3::CausalCandidate,
                );
                row.edge = row.edge.with_epistemic_raw5(up.raw());
            }
        }

        rep.rows = rows.len();
    }
    // Quorum and Contradiction loci over the whole row stream (a text is one
    // Markov chain): the nearest row within the i4 reach with the same
    // (S, P) and the same object agrees, with another object disagrees. The
    // contradiction is kept as a pointer, never dropped.
    for i in 0..rows.len() {
        let me = rows[i].spo;
        let near = |same: bool| {
            (i.saturating_sub(8)..(i + 8).min(rows.len()))
                .filter(|&j| j != i && offset(i, j).is_some())
                .filter(|&j| {
                    let o = rows[j].spo;
                    o.subject == me.subject
                        && o.predicate == me.predicate
                        && (o.object == me.object) == same
                })
                .min_by_key(|&j| (j.abs_diff(i), j > i))
        };
        let (agree, disagree) = (near(true), near(false));
        if let Some(j) = agree {
            rows[i].loci = rows[i].loci.with(Locus::Quorum, offset(i, j).unwrap());
            rep.quorum_bound += 1;
        }
        if let Some(j) = disagree {
            rows[i].loci = rows[i]
                .loci
                .with(Locus::Contradiction, offset(i, j).unwrap());
            rep.contradiction_bound += 1;
        }
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
    const TO: u16 = 32;
    const THERE: u16 = 33;

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
        FrozenOperandAbi::new(1, &[DOG, MAN, CAT, BIT, SAW, THE, WHO, TO, THERE]).unwrap()
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

    /// The seven listed functions round-trip through the contract's codes
    /// 1..=7; `Core` (0) is the plain core argument; Judicantis has none.
    #[test]
    fn orientation_codes_round_trip() {
        for f in DativeFunction::ALL {
            match orientation_of(f) {
                Some(o) => assert_eq!(orientation_function(o), Some(f)),
                None => assert_eq!(f, DativeFunction::Judicantis),
            }
        }
        let codes: Vec<u8> = DativeFunction::ALL
            .iter()
            .filter_map(|&f| orientation_of(f))
            .map(Orientation::code)
            .collect();
        assert_eq!(
            codes,
            [1, 2, 3, 4, 5, 6, 7],
            "all seven non-zero states used"
        );
        assert_eq!(orientation_function(Orientation::Core), None);
    }

    #[test]
    fn independent_corroboration_climbs_the_ladder() {
        let v = vec![svo(DOG, BIT, MAN); 3];
        let (rows, rep) = run(&v);
        assert_eq!(rep.rows, 3);
        let last = rows[2].edge;
        assert_eq!(last.frequency_u8(), 255);
        assert_eq!(last.confidence_u8(), (3 * 255 / 4) as u8);
        let acts: Vec<i8> = rows.iter().map(|r| r.edge.activation()).collect();
        assert_eq!(
            acts,
            [4, 0, 0],
            "a repeat of a settled belief changes nothing"
        );
        assert_eq!(last.novelty_wisdom(), 3);
        assert_eq!(epi(last).certification(), Certification3::Related);
        assert_eq!(epi(last).topology(), Topology2::Direct);
        assert_eq!(last.orientation(), Orientation::Core, "no attested form: plain core argument");
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
    fn a_contest_is_counted_and_the_balance_decides_the_climb() {
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

    /// A contest does not null the rival: `man` reached rung 2, a `cat` verse
    /// contests it, and the next `man` verse climbs on to 3. Both objects stay
    /// in the group's preserved tension.
    #[test]
    fn a_contested_rival_keeps_its_climb() {
        let v = vec![
            svo(DOG, BIT, MAN),
            svo(DOG, BIT, MAN),
            svo(DOG, BIT, CAT),
            svo(DOG, BIT, MAN),
        ];
        let (rows, rep) = run(&v);
        assert_eq!(rows[1].edge.novelty_wisdom(), 2);
        assert_eq!(rows[3].edge.activation(), 1, "170 → 191");
        assert_eq!(rows[3].edge.novelty_wisdom(), 3, "no reset, one climb");
        assert_eq!(rows[2].revision, Some(RevisionKind::HorizonFusion));
        assert_eq!(rows[3].revision, Some(RevisionKind::HorizonFusion));
        assert!(!rows[1].contested, "no rival yet");
        assert!(rows[2].contested && rows[3].contested);
        assert_eq!(rep.contests, 2);
        // The man arc skips the cat row: W points at the man row of verse 1.
        assert_eq!(rows[3].edge.w_slot(), 2);
    }

    /// A contradiction with no new independent root is suspended, not
    /// counted: one verse states `dog bit man` and `dog bit cat`. The cat row
    /// earns no support and no fact; the tension is kept.
    #[test]
    fn a_contradiction_from_one_source_is_suspended() {
        let mut v = svo(DOG, BIT, MAN);
        v.push(Reading::stop());
        v.extend(svo(DOG, BIT, CAT));
        let w = run_wired(&[v, svo(DOG, BIT, MAN)]);
        assert_eq!(w.report.suspended, 1);
        let cat = &w.rows[1];
        assert_eq!(cat.revision, Some(RevisionKind::ContradictionPreserved));
        assert_eq!(cat.support, 0);
        assert_eq!(cat.antecedent, 1, "the group counts its one source once");
        assert!(cat.contested);
        assert_eq!(w.witnesses[0].get(2).unwrap().spo_fact_ref, None);
        let cat_stmt = CStmt {
            s: DOG,
            cop: Copula::Rel(BIT),
            p: CAT,
        };
        assert!(
            w.facts.get(cat_stmt).is_none(),
            "the arena never counted it"
        );
        // A later independent man verse fuses (the tension is still live).
        assert_eq!(w.rows[2].revision, Some(RevisionKind::HorizonFusion));
        assert_eq!(w.rows[2].support, 2);
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
            let prior = w
                .rows
                .iter()
                .find(|r| r.verse == vi && r.spo.object == MAN)
                .unwrap();
            slot = prior.edge.w_slot();
        }
        assert_eq!(verses, [4, 2, 0], "newest to oldest, cat rows skipped");
        assert!(facts.windows(2).all(|p| p[0] == p[1]), "one statement");
        let cat_fact = table
            .get(w.rows[3].edge.w_slot())
            .unwrap()
            .spo_fact_ref
            .unwrap();
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
        assert_eq!(
            w.rows[COHORT_ROWS - 1].edge.w_slot(),
            (COHORT_ROWS - 1) as u8
        );
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

    /// A subject relative gets an Antecedent pointer to the matrix event;
    /// topology IndirectKnown. With no attested form, orientation stays 0.
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
        assert_eq!(rows[0].edge.orientation(), Orientation::Core);
        assert_eq!(rows[0].loci.at(Locus::Antecedent), 1);
        assert_eq!(rep.antecedent_bound, 1);
        assert_eq!(epi(rows[0].edge).topology(), Topology2::IndirectKnown);
        assert_eq!(
            rows[1].loci.at(Locus::Antecedent),
            0,
            "matrix event has no pointer"
        );
    }

    /// Right corner: `the cat bit the man who saw the dog`. The antecedent
    /// `man` is the matrix OBJECT and the matrix event comes FIRST, so the
    /// pointer goes back (-1).
    #[test]
    fn an_antecedent_behind_the_relative_is_found() {
        let v = vec![
            one(THE, Pos::Det),
            one(CAT, Pos::Noun),
            one(BIT, Pos::Verb),
            one(THE, Pos::Det),
            one(MAN, Pos::Noun),
            one(WHO, Pos::Rel),
            one(SAW, Pos::Verb),
            one(THE, Pos::Det),
            one(DOG, Pos::Noun),
        ];
        let (rows, rep) = run(&[v]);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].spo, Spo::new(CAT, BIT, MAN));
        assert!(rows[1].origin.is_some_and(Origin::is_relative));
        assert_eq!(rows[1].loci.at(Locus::Antecedent), -1);
        assert_eq!((rep.antecedent_bound, rep.antecedent_unbound), (1, 0));
    }

    fn form(word: u16, fs: &[DativeFunction]) -> FormAttestation {
        FormAttestation {
            word,
            functions: fs.iter().fold(0, |m, f| m | f.bit()),
            coverage_complete: true,
            evidence_root: 7,
        }
    }

    /// `the dog gave to the man`: the form `to` inside the clause decides the
    /// orientation when it allows one function, abstains (0, undecided) when
    /// it allows several, and the code stays out of a clause that lacks it.
    #[test]
    fn the_attested_form_sets_the_orientation() {
        let gave_to = |extra: Vec<Reading>| {
            let mut v = vec![
                one(THE, Pos::Det),
                one(DOG, Pos::Noun),
                one(BIT, Pos::Verb),
                one(TO, Pos::Other),
            ];
            v.extend([one(THE, Pos::Det), one(MAN, Pos::Noun)]);
            v.extend(extra);
            v
        };
        let run_forms = |v: Vec<Reading>, forms: Vec<FormAttestation>| {
            wire(
                &[(1, 1)],
                &[VerseIn {
                    readings: v,
                    forms,
                    ..VerseIn::default()
                }],
                1,
                &abi(),
                None,
                ReadParams::RIGHT_CORNER,
            )
            .unwrap()
        };
        let w = run_forms(
            gave_to(vec![]),
            vec![form(TO, &[DativeFunction::Recipient])],
        );
        assert_eq!(w.rows[0].edge.orientation(), Orientation::Recipient, "Recipient");
        assert_eq!(w.report.orientation_coded, 1);
        let w = run_forms(
            gave_to(vec![]),
            vec![form(
                TO,
                &[DativeFunction::Direction, DativeFunction::Recipient],
            )],
        );
        assert_eq!(w.rows[0].edge.orientation(), Orientation::Core);
        assert_eq!(w.report.orientation_ambiguous, 1);
        // A second clause without the form stays at 0.
        let mut second = vec![Reading::stop()];
        second.extend(svo(CAT, SAW, DOG));
        let w = run_forms(
            gave_to(second),
            vec![form(TO, &[DativeFunction::Recipient])],
        );
        assert_eq!(w.rows.len(), 2);
        assert_eq!(
            w.rows[1].edge.orientation(),
            Orientation::Core,
            "the form is in the other clause"
        );
    }

    /// A Lokal lane on the clause narrows a multifunctional form to
    /// Direction; without the lane the same form stays undecided.
    #[test]
    fn a_lokal_lane_narrows_the_form_to_direction() {
        let h =
            GrammarHeuristics::parse("# lane\tlemma\trelation\tcount\nLokal\tdort\tadvmod\t10\n");
        let readings = vec![
            one(THE, Pos::Det),
            one(DOG, Pos::Noun),
            one(BIT, Pos::Verb),
            one(TO, Pos::Other),
            one(THE, Pos::Det),
            one(MAN, Pos::Noun),
            one(THERE, Pos::Other),
        ];
        let surface = vec!["the", "dog", "bit", "to", "the", "man", "dort"];
        let forms = vec![form(
            TO,
            &[DativeFunction::Direction, DativeFunction::Recipient],
        )];
        let ins = [VerseIn {
            readings,
            surface,
            forms,
            ..VerseIn::default()
        }];
        let w = wire(
            &[(1, 1)],
            &ins,
            1,
            &abi(),
            Some(&h),
            ReadParams::RIGHT_CORNER,
        )
        .unwrap();
        assert!(w.rows[0].tekamolo[3].is_some(), "the Lokal lane landed");
        assert_eq!(w.rows[0].edge.orientation(), Orientation::Direction, "Direction");
        let w = wire(&[(1, 1)], &ins, 1, &abi(), None, ReadParams::RIGHT_CORNER).unwrap();
        assert_eq!(w.rows[0].edge.orientation(), Orientation::Core, "no lane: undecided");
    }

    /// Quorum points at the nearest agreeing row, Contradiction at the
    /// nearest disagreeing one; across verses, the earlier one on a tie.
    #[test]
    fn quorum_and_contradiction_point_at_peers() {
        let v = vec![svo(DOG, BIT, MAN), svo(DOG, BIT, CAT), svo(DOG, BIT, MAN)];
        let (rows, rep) = run(&v);
        assert_eq!(rows[2].loci.at(Locus::Quorum), -2);
        assert_eq!(rows[2].loci.at(Locus::Contradiction), -1);
        assert_eq!(rows[1].loci.at(Locus::Contradiction), -1, "tie: the prior");
        assert_eq!(rows[1].loci.at(Locus::Quorum), 0, "no other cat row");
        assert_eq!(rows[0].loci.at(Locus::Quorum), 2);
        assert_eq!((rep.quorum_bound, rep.contradiction_bound), (2, 3));
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
        assert_eq!(r.edge.orientation(), Orientation::Core);
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

    /// A token the lexicon tags as an adverb (`Pos::Adv`, COCA `r`) is
    /// adverbial like a skipped `Other` token (Codex review on #1591).
    #[test]
    fn a_pos_adv_token_is_adverbial() {
        let (adv, rv) = qrun(qverse(&[
            (THE, Pos::Det, "the", 0),
            (DOG, Pos::Noun, "dog", 0),
            (BIT, Pos::Verb, "caused", 0),
            (THE, Pos::Det, "the", 0),
            (MAN, Pos::Noun, "man", 0),
            (WHO, Pos::Adv, "quickly", QUICK),
        ]));
        assert_eq!(qualia_at(adv.qualia, Q_PREDICATE), QUICK);
        assert_eq!((rv.qualia_adverbial, rv.qualia_unattached), (1, 0));
    }

    /// A Kausal claim whose cause row the gate suspended is not counted: the
    /// verse states `dog bit man` and then `weil dog bit cat`, a rival from a
    /// source already counted. Twice. Nothing reaches the causes arena (Codex
    /// review on #1591).
    #[test]
    fn a_suspended_cause_is_not_counted() {
        let mut r = svo(DOG, BIT, MAN);
        r.push(Reading::stop());
        r.push(one(WHO, Pos::Other));
        r.extend(svo(DOG, BIT, CAT));
        let surface = vec![
            "der", "hund", "biss", "den", "mann", ".", "weil", "der", "hund", "biss", "die",
            "katze",
        ];
        let v = VerseIn {
            readings: r,
            surface,
            forms: vec![],
            qualia: vec![],
        };
        let w = run_in(&[v.clone(), v]);
        let cause = w
            .rows
            .iter()
            .find(|x| x.spo == Spo::new(DOG, BIT, CAT))
            .unwrap();
        assert!(
            cause.tekamolo[KAUSAL_LANE].is_some(),
            "the Kausal lane landed"
        );
        assert_eq!(cause.revision, Some(RevisionKind::ContradictionPreserved));
        assert_eq!(w.report.implications, 0);
        assert!(w.causes.entries().is_empty());
        assert_eq!(w.report.causal_candidates, 0);
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
