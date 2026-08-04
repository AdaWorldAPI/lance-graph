//! `stance` — the hermeneutic clause machine + the four-stance panel, lifted
//! out of `examples/probe_eyes_opened.rs` into the library so a non-example
//! consumer (the BLW cycle-loop-closure driver in `lance-graph-supervisor`,
//! per `.claude/plans/cycle-loop-closure-driver-v1.md` §12.3a) can reach it.
//!
//! This is a **behaviour-preserving lift**, not a rewrite: bodies are
//! byte-identical to the example except for the visibility changes the move
//! forces. `probe_eyes_opened.rs` keeps its own `main`, `report`,
//! `print_stance_panel`, fixtures, and every one of its assertions, and now
//! imports [`stream`], [`Interner`], [`Provenance`], [`RungLift`],
//! [`ReadOut`], [`FlipKind`], [`contradiction_ranking`], and [`stance_panel`]
//! from here instead of defining them. The lift's own falsifier is that the
//! probe's B1–B6 asserts stay green, unchanged.

use std::collections::HashMap;

use lance_graph_contract::grammar::clause_cues::{
    is_modal_aux, is_negation, pronoun_case, PronounCase,
};
use lance_graph_contract::grammar::verb_lexicon::{
    epistemic_reading, is_causal_cue, is_copula, read_verb,
};

use super::belief::{BeliefArena, CStmt, Copula, ReviseOutcome, Stamp};
use super::dissolution::staunen;
use super::insight::Snapshot;
use super::truth::TruthValue;

/// Function words the machine skips. `that` is handled BEFORE this list when
/// a perception verb has armed the complementizer wait. Discourse verbs
/// (`said`/`saying`) are stopped — they report speech, they are not clause
/// predicates for this machine.
const STOP: &[&str] = &[
    "the", "and", "but", "for", "unto", "upon", "into", "onto", "over", "under", "amongst", "with",
    "without", "from", "out", "all", "every", "which", "when", "now", "then", "also", "yea",
    "both", "more", "than", "any", "his", "their", "thy", "one", "that", "there", "let", "said",
    "saying", "spake", "surely", "freely", "together", "thereof",
];

/// Auxiliary predicate-armers beyond the contract modal list — KJV do-support
/// and have-forms ("did eat", "hath said"). Arm the next content word as
/// predicate, same as a copula.
const AUX: &[&str] = &[
    "did", "do", "doth", "dost", "hath", "hast", "have", "had", "mayest",
];

/// Interns strings to compact `u16` ids for statement storage; [`Interner::name`]
/// reverses the mapping for printing.
#[derive(Default)]
pub struct Interner {
    map: HashMap<String, u16>,
    names: Vec<String>,
}
impl Interner {
    /// A fresh interner with no strings assigned yet.
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
            names: Vec::new(),
        }
    }
    /// Returns `w`'s id, assigning the next free id on first sight.
    pub fn id(&mut self, w: &str) -> u16 {
        if let Some(&i) = self.map.get(w) {
            return i;
        }
        // Fail loudly rather than alias. The cast below is `as u16`, so past
        // 65,536 distinct strings the id wraps and two different words silently
        // share one id — corrupting every statement built from them. This is a
        // public library API now, so the bound is checked instead of assumed.
        // (Whole-book KJV interns ~12.5k, well under; a larger corpus would
        // otherwise corrupt quietly.)
        assert!(
            self.names.len() < u16::MAX as usize,
            "Interner exhausted: more than {} distinct strings",
            u16::MAX
        );
        let i = self.names.len() as u16;
        self.map.insert(w.to_string(), i);
        self.names.push(w.to_string());
        i
    }
    /// Reverses [`Interner::id`] — the string that was assigned `id`.
    pub fn name(&self, id: u16) -> &str {
        &self.names[id as usize]
    }
}

/// One emission's provenance — the arena holds truth; the probe holds WHERE.
#[derive(Debug, Clone)]
pub struct Provenance {
    /// The chapter:verse label the emission came from.
    pub verse: String,
    /// The emitted statement.
    pub stmt: CStmt,
    /// Whether the source text negated this emission (low-frequency
    /// invalidation vs an affirmation).
    pub negated: bool,
}

/// A rung-1 knows-that record: who knew, via which verb, that (subject was
/// object) — with the reflexivity bit the awareness blade hunts.
#[derive(Debug, Clone)]
pub struct RungLift {
    /// The chapter:verse label the lift came from.
    pub verse: String,
    /// The subject who knew/saw — the lift's outer subject.
    pub knower: u16,
    /// The perception/epistemic verb that licensed the lift.
    pub verb: u16,
    /// The inner statement's object (what the knower knew/saw to be true).
    pub object: u16,
    /// The 144-cell's tense-modulated modal prior — the lift's BLIND
    /// epistemic force (Abstracts/knew 0.85 > Mirrors/saw 0.70).
    pub modal: f32,
    /// The cell's one-byte Morton cascade address.
    pub cell: u8,
    /// Staunen of the arena AT the lift site — the felt CONTEXT
    /// (0.5·truth_entropy + 0.5·wonder, wonder = committed-contradiction
    /// tension).
    pub staunen_at: f32,
    /// Awareness quale = blind × context = modal × staunen_at.
    pub quale: f32,
    /// True only when the knower and the inner (overtly re-anchored)
    /// subject are the same referent — the awareness signature.
    pub self_referential: bool,
}

/// Accumulated read of one [`stream`] pass: raw emissions ([`ReadOut::provenance`]),
/// rung-1 lifts ([`ReadOut::lifts`]), causal edges ([`ReadOut::impls`]), and the
/// pass-2 admit/revise counts the hermeneutic-circle termination check uses.
#[derive(Default)]
pub struct ReadOut {
    /// Every observed (subject, Inh, predicate) emission from pass 1, in order.
    pub provenance: Vec<Provenance>,
    /// Every rung-1 knows-that record produced by [`stream`].
    pub lifts: Vec<RungLift>,
    /// Causal edges observed from `because`-cued text, as (verse, cause, effect).
    pub impls: Vec<(String, u16, u16)>, // (verse, cause, effect)
    /// Count of pass-2 emissions newly admitted (zero at a fixed point).
    pub pass2_admitted: usize,
    /// Count of pass-2 emissions that revised an existing belief (zero at a
    /// fixed point).
    pub pass2_revised: usize,
}

/// Stream verses through the cue-driven clause machine into `arena`.
///
/// The machine per token: pronoun-normalize (all Nominative/Accusative
/// personal pronouns → the single scene referent `they` — coarse
/// single-dialogue coreference, documented); negation/modal/aux/copula/
/// causal-cue/perception-verb catalogues; `verb_table` archetype consumer
/// for typed relational verbs; `-ed` (stem ≥ 3) morphology as the action
/// fallback. Subject anchoring is PRONOUN-STICKY: a noun never displaces a
/// pronoun subject (nouns mid-verse are mostly objects/appositions — this
/// is what keeps "the man and his wife" from stealing 2:25's subject before
/// "were not ashamed").
///
/// `pass2` re-presents identical stamps: the S4 overlap guard must route
/// every re-observation to CHOICE — the hermeneutic circle's termination.
pub fn stream(
    verses: &[(String, String)],
    arena: &mut BeliefArena,
    intern: &mut Interner,
    out: &mut ReadOut,
    pass2: bool,
) {
    let they = intern.id("they");
    let mut src: u32 = 0;

    for (verse, text) in verses {
        let mut subject: Option<u16> = None;
        let mut subject_is_pronoun = false;
        let mut armed = false; // copula/modal/aux/typed-verb armed a predicate
        let mut negated = false;
        // (knower, verb id, cell modal, cell address) — the 144 reading rides along.
        let mut await_that: Option<(u16, u16, f32, u8)> = None;
        let mut await_budget: u8 = 0; // content tokens left before the wait expires
        let mut lift_verb: Option<(u16, u16, f32, u8)> = None; // …after "that" is seen
                                                               // Did the inner clause re-anchor its OWN subject after "that"? A
                                                               // dropped inner subject ("God saw that [it] was good") inherits the
                                                               // knower — inherited identity is NOT evidence of reflexivity. The
                                                               // real-corpus blind run measured this: without the overt-subject
                                                               // requirement, the Genesis 1 refrain produced five degenerate
                                                               // "self-referential" lifts; with it, 3:7 stands alone.
        let mut inner_subject_seen = false;
        let mut causal_effect: Option<u16> = None; // effect predicate awaiting cause
        let mut last_pred: Option<u16> = None; // most recent emitted predicate

        for raw in text
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| !w.is_empty())
        {
            let w = raw.to_lowercase();

            // Complementizer: only special while a perception verb waits —
            // and only within a short window, else a DEMONSTRATIVE "that"
            // fifteen tokens later completes the lift ("to see what he would
            // call them … THAT was the name", 2:19 — measured false positive).
            if w == "that" {
                if let Some(kv) = await_that.take() {
                    lift_verb = Some(kv);
                    inner_subject_seen = false;
                }
                continue;
            }
            // Pronoun normalization — the single-scene referent.
            let w = match pronoun_case(&w) {
                Some(PronounCase::Nominative) | Some(PronounCase::Accusative) => {
                    subject = Some(they);
                    subject_is_pronoun = true;
                    if lift_verb.is_some() {
                        inner_subject_seen = true; // overt inner subject
                    }
                    continue;
                }
                _ => w,
            };
            if STOP.contains(&w.as_str()) {
                continue;
            }
            if is_negation(&w) {
                negated = true;
                continue;
            }
            if is_copula(&w) || is_modal_aux(&w) || AUX.contains(&w.as_str()) {
                armed = true;
                continue;
            }
            if is_causal_cue(&w) {
                causal_effect = last_pred;
                continue;
            }
            // Rails-shaped rung-lift gate: the verb must READ A 144 CELL
            // (cue gate licenses the that-complement; the matrix supplies the
            // reasoning — tense-modulated modal force + Morton address).
            if let Some(er) = epistemic_reading(&w) {
                if let Some(s) = subject {
                    let verb_id = intern.id(&w);
                    await_that = Some((s, verb_id, er.modal, er.cell));
                    await_budget = 3; // complementizer must be near
                }
                continue;
            }
            if w.len() <= 2 {
                continue;
            }
            // Spend the complementizer window on content tokens.
            if await_that.is_some() {
                if await_budget == 0 {
                    await_that = None; // too far — that "that" would be demonstrative
                } else {
                    await_budget -= 1;
                }
            }

            // Typed relational verb — the verb_table archetype consumer.
            let typed = read_verb(&w).is_some();
            // -ed action fallback (stem >= 3): "sewed" prints as an action
            // predicate on the current subject.
            let action_ed =
                !typed && w.ends_with("ed") && w.len() >= 5 && !armed && subject.is_some();

            if typed {
                armed = true;
                continue; // predicate = the verb's OBJECT, next content word
            }

            if armed || action_ed {
                // EMISSION: (subject, Inh, w) at f=0.9, or f=0.05 under negation.
                if let Some(s) = subject {
                    let p = intern.id(&w);
                    if s != p {
                        let f = if negated { 0.05 } else { 0.9 };
                        let stmt = CStmt {
                            s,
                            cop: Copula::Inh,
                            p,
                        };
                        let outcome =
                            arena.observe(stmt, TruthValue::new(f, 0.9), Stamp::source(src));
                        src += 1;
                        last_pred = Some(p);
                        if pass2 {
                            match outcome {
                                ReviseOutcome::Admitted { .. } => out.pass2_admitted += 1,
                                ReviseOutcome::Revised { .. } => out.pass2_revised += 1,
                                ReviseOutcome::Chosen { .. } => {}
                            }
                        } else {
                            out.provenance.push(Provenance {
                                verse: verse.clone(),
                                stmt,
                                negated,
                            });
                        }
                        // Rung lift: the armed knows-that consumes this
                        // emission as its inner statement. Fires in BOTH
                        // passes (pass-2 admit_derived on an unchanged
                        // derived statement is a no-op) so the stamp
                        // sequence stays identical across passes.
                        if let Some((knower, verb, modal, cell)) = lift_verb.take() {
                            if let Some(inner) = arena.get(stmt) {
                                let inner_truth = inner.truth;
                                let inner_id = arena
                                    .entries()
                                    .iter()
                                    .position(|b| b.stmt == stmt)
                                    .expect("just observed")
                                    as u32;
                                let meta = CStmt {
                                    s: knower,
                                    cop: Copula::Rel(verb),
                                    p,
                                };
                                // Context BEFORE output (codex P1): the
                                // snapshot must precede admit_derived, else
                                // the modal-scaled meta-belief sits inside
                                // its own context factor and `modal` leaks
                                // into BOTH sides of quale = modal × staunen
                                // (and duplicate lifts become incomparable).
                                // The inner emission IS stream context; the
                                // meta-belief is the lift's own output.
                                let staunen_at = if pass2 {
                                    0.0
                                } else {
                                    staunen(&Snapshot::of(arena, 0.0))
                                };
                                // Cell-graded epistemic force: the meta-truth
                                // discount IS the 144 cell's tense-modulated
                                // modal prior — knowing (Abstracts, 0.85)
                                // lifts harder than seeing (Mirrors, 0.70),
                                // graded by the matrix, never a constant.
                                let t = TruthValue::new(
                                    inner_truth.frequency * modal,
                                    inner_truth.confidence * modal,
                                );
                                arena.admit_derived(meta, t, &[inner_id], 1);
                                if !pass2 {
                                    // Blind × context: the cell's modal
                                    // (text-independent archetype) × the
                                    // arena's Staunen AT the lift site
                                    // (0.5·truth_entropy + 0.5·wonder; wonder
                                    // = committed-contradiction tension — the
                                    // felt stakes accumulated so far).
                                    out.lifts.push(RungLift {
                                        verse: verse.clone(),
                                        knower,
                                        verb,
                                        object: p,
                                        modal,
                                        cell,
                                        staunen_at,
                                        quale: modal * staunen_at,
                                        // Reflexive ONLY with an OVERT inner
                                        // subject: "they knew that THEY were
                                        // naked" — an inherited subject
                                        // ("saw that [it] was good") is the
                                        // knower by default, not by claim.
                                        self_referential: knower == s && inner_subject_seen,
                                    });
                                }
                            }
                        }
                        // Causal cue: "<effect> because <cause>" → the text's
                        // own Impl(cause → effect), observed — in BOTH passes
                        // (keeps the stamp sequence aligned; pass-2 hits the
                        // overlap guard like every other re-observation).
                        if let Some(effect) = causal_effect.take() {
                            if p != effect {
                                let imp = CStmt {
                                    s: p,
                                    cop: Copula::Impl,
                                    p: effect,
                                };
                                let imp_outcome = arena.observe(
                                    imp,
                                    TruthValue::new(0.9, 0.9),
                                    Stamp::source(src),
                                );
                                src += 1;
                                if pass2 {
                                    match imp_outcome {
                                        ReviseOutcome::Admitted { .. } => out.pass2_admitted += 1,
                                        ReviseOutcome::Revised { .. } => out.pass2_revised += 1,
                                        ReviseOutcome::Chosen { .. } => {}
                                    }
                                } else {
                                    out.impls.push((verse.clone(), p, effect));
                                }
                            }
                        }
                    }
                    armed = false;
                    negated = false;
                }
            } else {
                // Bare content word: subject anchoring, pronoun-sticky.
                if !subject_is_pronoun {
                    subject = Some(intern.id(&w));
                    if lift_verb.is_some() {
                        inner_subject_seen = true; // overt inner subject (noun)
                    }
                }
            }
        }
    }
}

/// Blind blade 1: rank every belief by preserved contradiction depth.
///
/// The floor is NOT decorative (inertness): consistent re-observation leaves
/// float-ε residue in `contradiction` (revision arithmetic on f32 — the
/// measured fixture shows `(they→naked)` at ~1e-8 after three consistent
/// f=0.9 observations). 0.05 admits every genuine polarity flip (≈0.85) and
/// silences ε-noise; dropping it to 0.0 re-admits the noise row (measured),
/// raising it past 0.85 silences the real reversals.
pub fn contradiction_ranking(arena: &BeliefArena) -> Vec<(CStmt, f32)> {
    let mut v: Vec<(CStmt, f32)> = arena
        .entries()
        .iter()
        .filter(|b| b.contradiction > 0.05)
        .map(|b| (b.stmt, b.contradiction))
        .collect();
    v.sort_by(|a, b| b.1.total_cmp(&a.1));
    v
}

/// Nietzschean genealogy: HOW did a held contradiction flip?
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlipKind {
    /// First observed NEGATED, later affirmed — the forbidden became done
    /// (Umwertung: low → high).
    Transvaluation,
    /// First affirmed, later NEGATED — the asserted was denied (high → low).
    Devaluation,
}

/// B6 — the ASPECT PANEL: four philosopher stances as PURE READS over ONE
/// unchanged arena. The operator's image: the corpus is a CRYSTAL — the
/// percipient's own knowledge and reflection shine a light through it,
/// creating reflections the crystal alone does not contain; thinking as a
/// Doppelspalt event, the reading an interference pattern between the
/// text-wave and the reader-wave. Wittgenstein's duck-rabbit gives the same
/// invariant operationally: "I see that it has not changed; and yet I see
/// it differently." Each stance takes `&BeliefArena` — mutation impossible
/// by signature; the caller asserts the runtime witness (entry count
/// unchanged). And unlike the physical Doppelspalt, the read is
/// NON-DESTRUCTIVE: nothing collapses (E-LC-SCARCITY-INVERSION-1 — the
/// substrate holds the distribution; stances are late-bound reads).
///
/// * **Hegel** — rank by Aufhebung. The three meanings of *aufheben* ARE
///   `revise_at`'s three fields: cancelled = pooled truth, preserved = the
///   `contradiction` field, lifted = the rung.
/// * **Nietzsche** — genealogy: partition the held contradictions by FLIP
///   DIRECTION read from provenance (first vs last emission's negation).
///   Transvaluation (forbidden → done) vs devaluation (asserted → denied)
///   — a distinction Hegel's symmetric depth ranking cannot see.
/// * **Kant** — critique: recompute the lift ranking with the reader's
///   modal grading ablated to uniform (0.5). The delta IS the reader's
///   a-priori contribution (the reader-wave, isolated); what survives — the
///   reversal set, pure text evidence — is a posteriori (the text-wave).
///   Doubles as the inertness test on the modal knob.
/// * **Wittgenstein** — meaning as use: rank concepts by DISTINCT
///   language-games participated in (Inh-subject, Inh-object, knows-that
///   object, Impl-cause, Impl-effect). No inner essence — breadth of
///   practice.
#[allow(clippy::type_complexity)]
pub fn stance_panel(
    arena: &BeliefArena,
    intern: &Interner,
    out: &ReadOut,
) -> (
    Vec<(CStmt, f32)>,       // Hegel: Aufhebung ranking
    Vec<(CStmt, FlipKind)>,  // Nietzsche: genealogy partition
    Vec<(String, f32, f32)>, // Kant: (lift label, graded quale, ablated quale)
    Vec<(u16, usize)>,       // Wittgenstein: (concept, distinct games)
) {
    // ── Hegel ──
    let hegel = contradiction_ranking(arena);

    // ── Nietzsche ──
    let mut nietzsche = Vec::new();
    for (stmt, _) in &hegel {
        let obs: Vec<&Provenance> = out.provenance.iter().filter(|p| p.stmt == *stmt).collect();
        if let (Some(first), Some(last)) = (obs.first(), obs.last()) {
            let kind = match (first.negated, last.negated) {
                (true, false) => Some(FlipKind::Transvaluation),
                (false, true) => Some(FlipKind::Devaluation),
                _ => None, // flip not legible from endpoints — no verdict
            };
            if let Some(k) = kind {
                nietzsche.push((*stmt, k));
            }
        }
    }

    // ── Kant ──
    const UNIFORM_MODAL: f32 = 0.5;
    let kant: Vec<(String, f32, f32)> = out
        .lifts
        .iter()
        .map(|l| {
            (
                format!("{} {}", l.verse, intern.name(l.verb)),
                l.quale,
                UNIFORM_MODAL * l.staunen_at,
            )
        })
        .collect();

    // ── Wittgenstein ──
    let mut games: HashMap<u16, std::collections::HashSet<&'static str>> = HashMap::new();
    for b in arena.entries() {
        // Observation-grounded Inh only — derived closure edges are the
        // arena's own inferences, not the text's usage.
        if b.stmt.cop == Copula::Inh && b.stamp != Stamp::default() {
            games.entry(b.stmt.s).or_default().insert("inh-subj");
            games.entry(b.stmt.p).or_default().insert("inh-obj");
        }
    }
    for l in &out.lifts {
        games.entry(l.knower).or_default().insert("rel-subj");
        games.entry(l.object).or_default().insert("rel-obj");
    }
    for (_, cause, effect) in &out.impls {
        games.entry(*cause).or_default().insert("impl-cause");
        games.entry(*effect).or_default().insert("impl-effect");
    }
    let mut wittgenstein: Vec<(u16, usize)> =
        games.into_iter().map(|(c, g)| (c, g.len())).collect();
    wittgenstein.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    (hegel, nietzsche, kant, wittgenstein)
}
