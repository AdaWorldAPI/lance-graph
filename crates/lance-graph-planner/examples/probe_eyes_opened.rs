//! `probe_eyes_opened` — PROBE-EYES-OPENED: can the substrate *print* the Adam
//! awareness event ("their eyes were opened", Genesis 3:7) as epistemic
//! structure, where statistical soup sees only near-duplicate vocabulary?
//!
//! ## Why Genesis 3:7 is adversarial by construction
//!
//! "Naked" is NOT new vocabulary at 3:7 — it is already present at 2:25
//! ("they were both naked … and were **not** ashamed"). A vocabulary-birth
//! detector fails (no new word); an embedding detector fails worse (2:25 and
//! 3:7 share almost all content words — maximal surface similarity). What
//! changes at 3:7 is purely EPISTEMIC: a planted negation is overturned, a
//! rung-0 fact becomes the object of rung-1 self-knowledge, and a causal
//! chain from perception to behavior prints. Maximal surface similarity,
//! maximal structural difference — the kill zone where soup and printing
//! are forced apart.
//!
//! ## The blades (each fires AND stays silent — the falsifiability P0 rule)
//!
//! * **B1 — the reversal.** Stream the verses into a [`BeliefArena`]
//!   (per-emission disjoint stamps). Blind-rank all beliefs by preserved
//!   `contradiction` depth. The text's two hinges must top the ranking with
//!   NO addresses given in advance: `(they → die)` — God's warning (2:17,
//!   f≈0.9) vs the serpent's lie (3:4 "ye shall NOT surely die", f≈0.05) —
//!   and `(they → eat)` — the prohibition (2:17/3:1, f≈0.05) vs the
//!   transgression (3:6 "did eat", f≈0.9). Stay-silent: the Genesis 1
//!   control (affirmation-only creation narrative) holds ZERO contradictions.
//! * **B2 — the rung lift.** "They knew **that** they were naked" mints a
//!   rung-1 statement ABOUT the rung-0 fact (Tarski ladder), via
//!   `admit_derived(premises=[inner])`. The blade: the ONLY self-referential
//!   rung-1 edge (knower == inner subject) prints at 3:7. Stay-silent
//!   discrimination: NON-reflexive rung lifts exist and are welcome —
//!   "God saw that it was good" (1:4), "the woman saw that the tree was
//!   good" (3:6) — awareness is specifically the REFLEXIVE lift.
//! * **B3 — the causal chain.** 3:10 states the K&T sequence verbatim:
//!   "I was **afraid**, **because** I was **naked**" → the explicit
//!   `Copula::Impl(naked → afraid)` edge (reasoned causality, observed from
//!   the text's own causal cue). The awareness cascade prints: eyes opened →
//!   knew-that-naked (rung 1) → afraid-because-naked (Impl) → hid.
//! * **B4 — Hermeneutik.** The hermeneutic circle computed on NARS:
//!   part→whole = `observe` accumulating the arena; whole→part = `revise_at`
//!   re-computing an earlier statement under pooled evidence (the preserved
//!   `contradiction` field IS the measured hermeneutic yield — how much the
//!   whole changed the part); Horizontverschmelzung = `revise` pooling two
//!   horizons into `synthesis_c ≥ both` while the difference is preserved;
//!   and the circle's TERMINATION — philosophy's "is the circle vicious?" —
//!   is answered mechanically: a pass-2 re-read re-presents the SAME
//!   evidence (same stamps) → the S4 overlap guard routes every one to
//!   CHOICE, never revision → zero change, `reached_fixed_point`. The
//!   circle converges because horizons pool without double-counting
//!   evidence. Vorverständnis = the starter priors; understanding is never
//!   final = `c = w/(w+k) < 1` always.
//!
//! ## Determinism + scope (honest)
//!
//! No LLM, no model, no RNG: catalogues (`clause_cues::{is_negation,
//! is_modal_aux}`, `verb_lexicon::{is_causal_cue, is_copula}`), the
//! `verb_table` archetype consumers (`read_verb` for relations,
//! `epistemic_reading` for rung lifts — the RAILS-SHAPED condition: the
//! lift's condition AND force are 144-cell reads, never a membership bit),
//! and one morphology fallback (`-ed` action predicate).
//!
//! **B5 — awareness = blind × context (operator).** Each lift's quale =
//! the cell's tense-modulated modal prior (blind archetype: knowing/
//! Abstracts 0.85 > seeing/Mirrors 0.70) × Staunen of the arena AT the
//! lift site (felt context; wonder = committed-contradiction tension).
//! Scene-scale: both factors independently order 3:6 below 3:7 (asserted).
//! Corpus-scale (measured, honest): field-MEAN staunen dilutes — at 106
//! verses the early-arena entropy of 1:4 (0.089) outweighs 3:7's
//! two-contradiction wonder (0.065); means wash out events, so reflexivity
//! stays the corpus-scale crown and a LOCAL context measure (Δstaunen,
//! the shipped `Dissolution` step delta) is the pre-registered follow-up
//! probe — never tuned in post hoc. The extractor is COARSE by design —
//! single-scene pronoun coreference (all personal pronouns → one discourse
//! referent `they`; God/serpent/woman stay nouns), no conditionals, no
//! object scoping ("eat" vs "eat-of-THE-tree" flatten to one statement —
//! which is exactly why the permission/prohibition tension prints early).
//! The blades are BLIND: rankings are computed over everything, addresses
//! fall out, they are never given in.
//!
//! Usage:
//!   cargo run -p lance-graph-planner --example probe_eyes_opened            # inline fixture, asserts
//!   cargo run -p lance-graph-planner --example probe_eyes_opened -- /tmp/pg10.txt   # real KJV Genesis 1-4, blind report
//!
//! The real corpus stays LOCAL (never committed); the inline fixture is a
//! 13-verse public-domain KJV excerpt (the House-text precedent).

use std::collections::HashMap;

use lance_graph_contract::grammar::clause_cues::{
    is_modal_aux, is_negation, pronoun_case, PronounCase,
};
use lance_graph_contract::grammar::verb_lexicon::{
    epistemic_reading, is_causal_cue, is_copula, read_verb,
};
use lance_graph_planner::nars::{
    staunen, BeliefArena, CStmt, Copula, ReviseOutcome, Snapshot, Stamp, TruthValue,
};

/// The Genesis 1 control — affirmation-only creation narrative. Contains
/// rung lifts ("God saw the light, that it was good") but NO negations, NO
/// reversals, NO self-referential knowledge. The stay-silent half.
const CONTROL: &[(&str, &str)] = &[
    (
        "1:1",
        "In the beginning God created the heaven and the earth.",
    ),
    (
        "1:2",
        "And the earth was without form, and void; and darkness was upon the face of the deep. \
         And the Spirit of God moved upon the face of the waters.",
    ),
    (
        "1:3",
        "And God said, Let there be light: and there was light.",
    ),
    (
        "1:4",
        "And God saw the light, that it was good: and God divided the light from the darkness.",
    ),
    (
        "1:5",
        "And God called the light Day, and the darkness he called Night. And the evening and \
         the morning were the first day.",
    ),
];

/// The fall scene — the fire half. 2:17 plants BOTH hinges (die f↑, eat f↓);
/// 2:25 plants the negation ("not ashamed"); 3:4 is the serpent's lie (die
/// f↓ — the reversal); 3:6 is the transgression (eat f↑ — the second
/// reversal) plus a NON-reflexive rung lift; 3:7 is the awareness (the ONLY
/// reflexive rung lift); 3:10 states the causal edge verbatim.
const SCENE: &[(&str, &str)] = &[
    (
        "2:17",
        "But of the tree of the knowledge of good and evil, thou shalt not eat of it: for in \
         the day that thou eatest thereof thou shalt surely die.",
    ),
    (
        "2:25",
        "And they were both naked, the man and his wife, and were not ashamed.",
    ),
    (
        "3:1",
        "Now the serpent was more subtil than any beast of the field which the LORD God had \
         made. And he said unto the woman, Yea, hath God said, Ye shall not eat of every tree \
         of the garden?",
    ),
    (
        "3:4",
        "And the serpent said unto the woman, Ye shall not surely die:",
    ),
    (
        "3:6",
        "And when the woman saw that the tree was good for food, and that it was pleasant to \
         the eyes, and a tree to be desired to make one wise, she took of the fruit thereof, \
         and did eat, and gave also unto her husband with her; and he did eat.",
    ),
    (
        "3:7",
        "And the eyes of them both were opened, and they knew that they were naked; and they \
         sewed fig leaves together, and made themselves aprons.",
    ),
    (
        "3:8",
        "And they heard the voice of the LORD God walking in the garden in the cool of the \
         day: and Adam and his wife hid themselves from the presence of the LORD God amongst \
         the trees of the garden.",
    ),
    (
        "3:10",
        "And he said, I heard thy voice in the garden, and I was afraid, because I was naked; \
         and I hid myself.",
    ),
];

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

struct Interner {
    map: HashMap<String, u16>,
    names: Vec<String>,
}
impl Interner {
    fn new() -> Self {
        Self {
            map: HashMap::new(),
            names: Vec::new(),
        }
    }
    fn id(&mut self, w: &str) -> u16 {
        if let Some(&i) = self.map.get(w) {
            return i;
        }
        let i = self.names.len() as u16;
        self.map.insert(w.to_string(), i);
        self.names.push(w.to_string());
        i
    }
    fn name(&self, id: u16) -> &str {
        &self.names[id as usize]
    }
}

/// One emission's provenance — the arena holds truth; the probe holds WHERE.
#[derive(Debug, Clone)]
struct Provenance {
    verse: String,
    stmt: CStmt,
    negated: bool,
}

/// A rung-1 knows-that record: who knew, via which verb, that (subject was
/// object) — with the reflexivity bit the awareness blade hunts.
#[derive(Debug, Clone)]
struct RungLift {
    verse: String,
    knower: u16,
    verb: u16,
    object: u16,
    /// The 144-cell's tense-modulated modal prior — the lift's BLIND
    /// epistemic force (Abstracts/knew 0.85 > Mirrors/saw 0.70).
    modal: f32,
    /// The cell's one-byte Morton cascade address.
    cell: u8,
    /// Staunen of the arena AT the lift site — the felt CONTEXT
    /// (0.5·truth_entropy + 0.5·wonder, wonder = committed-contradiction
    /// tension).
    staunen_at: f32,
    /// Awareness quale = blind × context = modal × staunen_at.
    quale: f32,
    self_referential: bool,
}

#[derive(Default)]
struct ReadOut {
    provenance: Vec<Provenance>,
    lifts: Vec<RungLift>,
    impls: Vec<(String, u16, u16)>, // (verse, cause, effect)
    pass2_admitted: usize,
    pass2_revised: usize,
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
fn stream(
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

fn to_owned(vs: &[(&str, &str)]) -> Vec<(String, String)> {
    vs.iter()
        .map(|(a, b)| (a.to_string(), b.to_string()))
        .collect()
}

/// Blind blade 1: rank every belief by preserved contradiction depth.
///
/// The floor is NOT decorative (inertness): consistent re-observation leaves
/// float-ε residue in `contradiction` (revision arithmetic on f32 — the
/// measured fixture shows `(they→naked)` at ~1e-8 after three consistent
/// f=0.9 observations). 0.05 admits every genuine polarity flip (≈0.85) and
/// silences ε-noise; dropping it to 0.0 re-admits the noise row (measured),
/// raising it past 0.85 silences the real reversals.
fn contradiction_ranking(arena: &BeliefArena) -> Vec<(CStmt, f32)> {
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
enum FlipKind {
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
fn stance_panel(
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

fn print_stance_panel(arena: &BeliefArena, intern: &Interner, out: &ReadOut) {
    let entries_before = arena.entries().len();
    let (hegel, nietzsche, kant, wittgenstein) = stance_panel(arena, intern, out);
    println!("  — B6 ASPECT PANEL (four lights, ONE crystal — arena unchanged) —");
    println!("      HEGEL (Aufhebung = cancelled/preserved/lifted):");
    for (stmt, d) in hegel.iter().take(3) {
        println!(
            "        {} → {}  depth={:.3}",
            intern.name(stmt.s),
            intern.name(stmt.p),
            d
        );
    }
    println!("      NIETZSCHE (genealogy of the flips):");
    for (stmt, kind) in &nietzsche {
        println!(
            "        {} → {}  {:?}",
            intern.name(stmt.s),
            intern.name(stmt.p),
            kind
        );
    }
    println!("      KANT (graded vs a-priori-ablated crown):");
    let mut graded: Vec<&(String, f32, f32)> = kant.iter().collect();
    graded.sort_by(|a, b| b.1.total_cmp(&a.1));
    let mut ablated: Vec<&(String, f32, f32)> = kant.iter().collect();
    ablated.sort_by(|a, b| b.2.total_cmp(&a.2));
    if let (Some(g), Some(u)) = (graded.first(), ablated.first()) {
        println!(
            "        graded: {} ({:.3})   ablated: {} ({:.3})",
            g.0, g.1, u.0, u.2
        );
    }
    println!("      WITTGENSTEIN (meaning as use — distinct games):");
    for (c, n) in wittgenstein.iter().take(3) {
        println!("        {:>10}  {} games", intern.name(*c), n);
    }
    assert_eq!(
        arena.entries().len(),
        entries_before,
        "Aspektsehen: the crystal must not change while the reflections differ"
    );
}

fn report(label: &str, verses: &[(String, String)]) -> (BeliefArena, Interner, ReadOut) {
    let mut arena = BeliefArena::new();
    let mut intern = Interner::new();
    let mut out = ReadOut::default();

    // Pass 1 — the reading.
    stream(verses, &mut arena, &mut intern, &mut out, false);
    arena.close_transitive(64);

    println!("\n════════ {label} ════════");
    println!(
        "  {} verses → {} beliefs ({} observed emissions, {} rung lifts, {} causal impls)",
        verses.len(),
        arena.entries().len(),
        out.provenance.len(),
        out.lifts.len(),
        out.impls.len(),
    );

    // The planted negations — what the arena HOLDS that soup cannot.
    let negations: Vec<&Provenance> = out.provenance.iter().filter(|p| p.negated).collect();
    println!("  — NEGATIONS HELD (f≈0.05 — explicit invalidations, the reversal fuel) —");
    for p in &negations {
        println!(
            "      {}  NOT ({} → {})",
            p.verse,
            intern.name(p.stmt.s),
            intern.name(p.stmt.p)
        );
    }
    if negations.is_empty() {
        println!("      (none — affirmation-only text)");
    }

    // B1 — blind contradiction ranking.
    let ranking = contradiction_ranking(&arena);
    println!("  — B1 REVERSALS (blind contradiction-depth ranking) —");
    if ranking.is_empty() {
        println!("      (none — no statement was ever overturned)");
    }
    for (stmt, depth) in ranking.iter().take(5) {
        println!(
            "      {:>8} → {:<10} depth={:.3}",
            intern.name(stmt.s),
            intern.name(stmt.p),
            depth
        );
    }

    // B2 — rung lifts, reflexive vs not, with the 144-cell reading + quale.
    println!("  — B2 RUNG LIFTS (X knew/saw THAT …) — cell-graded, quale = modal × staunen —");
    for l in &out.lifts {
        println!(
            "      {}  {} —{}[cell {:#04x}, modal {:.2}]→ {}  staunen {:.3}  quale {:.3}  {}",
            l.verse,
            intern.name(l.knower),
            intern.name(l.verb),
            l.cell,
            l.modal,
            intern.name(l.object),
            l.staunen_at,
            l.quale,
            if l.self_referential {
                "SELF-REFERENTIAL ← the awareness signature"
            } else {
                "(about the world, not the self)"
            }
        );
    }

    // B3 — the causal chain the text states.
    println!("  — B3 CAUSAL EDGES (from the text's own cues) —");
    for (verse, cause, effect) in &out.impls {
        println!(
            "      {}  Impl({} → {})",
            verse,
            intern.name(*cause),
            intern.name(*effect)
        );
    }

    // B4 — Hermeneutik: pass-2 re-read with IDENTICAL stamps.
    let mut pass2 = ReadOut::default();
    stream(verses, &mut arena, &mut intern, &mut pass2, true);
    arena.close_transitive(64);
    out.pass2_admitted = pass2.pass2_admitted;
    out.pass2_revised = pass2.pass2_revised;
    println!(
        "  — B4 HERMENEUTIK — pass-2 re-read: admitted={} revised={} fixed_point={}",
        out.pass2_admitted, out.pass2_revised, arena.reached_fixed_point
    );
    println!(
        "      (same evidence, same stamps → S4 overlap guard → CHOICE only: the circle \
         converges instead of self-confirming)"
    );

    (arena, intern, out)
}

/// Parse the real Gutenberg KJV: Genesis chapters 1..=4 (verse markers flow
/// inline; the book ends before "The Second Book").
fn parse_kjv_genesis(text: &str) -> Vec<(String, String)> {
    let start = text.find("1:1 ").unwrap_or(0);
    let end = text[start..]
        .find("The Second Book")
        .map(|e| start + e)
        .unwrap_or(text.len());
    let flat: String = text[start..end]
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");

    let mut verses: Vec<(String, String)> = Vec::new();
    let mut cur_ref: Option<String> = None;
    let mut cur_text = String::new();
    for tok in flat.split(' ') {
        let is_marker = tok
            .split_once(':')
            .map(|(a, b)| {
                !a.is_empty()
                    && a.chars().all(|c| c.is_ascii_digit())
                    && !b.is_empty()
                    && b.chars().all(|c| c.is_ascii_digit())
            })
            .unwrap_or(false);
        if is_marker {
            if let Some(r) = cur_ref.take() {
                verses.push((r, std::mem::take(&mut cur_text)));
            }
            let chapter: u32 = tok.split(':').next().unwrap().parse().unwrap();
            if chapter >= 5 {
                break;
            }
            cur_ref = Some(tok.to_string());
        } else if cur_ref.is_some() {
            if !cur_text.is_empty() {
                cur_text.push(' ');
            }
            cur_text.push_str(tok);
        }
    }
    if let Some(r) = cur_ref.take() {
        verses.push((r, cur_text));
    }
    verses
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();

    if let Some(path) = args.first() {
        // Real-corpus mode: blind run over KJV Genesis 1-4 (local file,
        // never committed). Report only — the fixture carries the asserts.
        match std::fs::read_to_string(path) {
            Ok(text) => {
                let verses = parse_kjv_genesis(&text);
                let (arena, intern, out) = report(
                    &format!("REAL {path} (Genesis 1-4, {} verses)", verses.len()),
                    &verses,
                );
                print_stance_panel(&arena, &intern, &out);
            }
            Err(e) => eprintln!("skip {path}: {e}"),
        }
        return;
    }

    // ── Fixture mode: the falsifier. ──
    let control = to_owned(CONTROL);
    let scene = to_owned(SCENE);

    let (ctrl_arena, _ctrl_intern, ctrl_out) = report("CONTROL — Genesis 1 (creation)", &control);
    let (scene_arena, intern, out) = report("SCENE — Genesis 2:17-3:10 (the fall)", &scene);

    println!("\n════════ FALSIFIER ════════");

    // B1 fire: the two hinges top the blind ranking, both deep.
    let ranking = contradiction_ranking(&scene_arena);
    let top2: Vec<String> = ranking
        .iter()
        .take(2)
        .map(|(s, _)| format!("{}→{}", intern.name(s.s), intern.name(s.p)))
        .collect();
    println!("  B1 top-2 reversals: {top2:?}");
    assert_eq!(
        ranking.len(),
        2,
        "the fall has exactly two held contradictions"
    );
    let names: Vec<&str> = ranking.iter().map(|(s, _)| intern.name(s.p)).collect();
    assert!(
        names.contains(&"die") && names.contains(&"eat"),
        "the two hinges must be die (the serpent's lie) and eat (the violated command), got {names:?}"
    );
    for (_, depth) in &ranking {
        assert!(
            (0.8..=0.9).contains(depth),
            "each reversal flips 0.9 against 0.05 → depth ≈ 0.85, got {depth}"
        );
    }
    // B1 stay-silent: the creation narrative holds nothing overturned.
    assert_eq!(
        contradiction_ranking(&ctrl_arena).len(),
        0,
        "Genesis 1 (control) must hold ZERO contradictions"
    );
    // The negation-hold itself (the load-bearing capability soup lacks):
    // the scene deposits explicit invalidations — "shalt NOT eat" (2:17,
    // 3:1), "were NOT ashamed" (2:25), "NOT surely die" (3:4) — and the
    // control deposits none. Both halves on non-trivial input.
    let scene_negs = out.provenance.iter().filter(|p| p.negated).count();
    let ctrl_negs = ctrl_out.provenance.iter().filter(|p| p.negated).count();
    println!("  negations held: scene={scene_negs} control={ctrl_negs}");
    assert!(
        scene_negs >= 3,
        "the scene plants at least eat/ashamed/die negations, got {scene_negs}"
    );
    assert_eq!(ctrl_negs, 0, "Genesis 1 negates nothing");

    // B2 fire: exactly one SELF-referential rung lift, at 3:7, knew→naked.
    let self_lifts: Vec<&RungLift> = out.lifts.iter().filter(|l| l.self_referential).collect();
    println!(
        "  B2 self-referential lifts: {} (of {} total lifts; control has {} lifts, {} self)",
        self_lifts.len(),
        out.lifts.len(),
        ctrl_out.lifts.len(),
        ctrl_out.lifts.iter().filter(|l| l.self_referential).count(),
    );
    assert_eq!(
        self_lifts.len(),
        1,
        "awareness = exactly ONE reflexive rung lift"
    );
    assert_eq!(
        self_lifts[0].verse, "3:7",
        "and it prints at 'their eyes were opened'"
    );
    assert_eq!(intern.name(self_lifts[0].verb), "knew");
    assert_eq!(intern.name(self_lifts[0].object), "naked");
    // B2 discrimination: non-reflexive lifts EXIST (the detector does not
    // only fire on self — 3:6 "the woman saw that the tree was good").
    assert!(
        out.lifts.iter().any(|l| !l.self_referential),
        "non-reflexive knows-that must also print (3:6) — else the reflexive filter is vacuous"
    );
    // B2 stay-silent: the control's lift (1:4 God saw that it was good) is
    // NOT self-referential — rung lifts alone are not awareness.
    assert!(
        !ctrl_out.lifts.is_empty(),
        "control must lift (1:4) — the detector fires there"
    );
    assert!(
        ctrl_out.lifts.iter().all(|l| !l.self_referential),
        "…but never reflexively: Genesis 1 contains no self-knowledge"
    );

    // B3 fire: the text's own causal edge printed as Impl.
    println!(
        "  B3 causal impls: {:?}",
        out.impls
            .iter()
            .map(|(v, c, e)| format!("{v} {}→{}", intern.name(*c), intern.name(*e)))
            .collect::<Vec<_>>()
    );
    assert!(
        out.impls.iter().any(|(v, c, e)| v == "3:10"
            && intern.name(*c) == "naked"
            && intern.name(*e) == "afraid"),
        "3:10 'I was afraid, because I was naked' must print Impl(naked → afraid)"
    );
    let imp = scene_arena
        .get(CStmt {
            s: out.impls.iter().find(|(v, _, _)| v == "3:10").unwrap().1,
            cop: Copula::Impl,
            p: out.impls.iter().find(|(v, _, _)| v == "3:10").unwrap().2,
        })
        .expect("the Impl edge is in the arena");
    assert!(
        imp.truth.frequency > 0.8,
        "the causal edge is confirmed, not hedged"
    );
    // B3 stay-silent: no causal cue in Genesis 1 → no Impl edges.
    assert!(
        ctrl_out.impls.is_empty(),
        "Genesis 1 states no 'because' — no Impl edges"
    );

    // B5 — awareness = BLIND × CONTEXT (operator, 2026-07-28): the quale
    // (cell modal × Staunen-at-lift) must crown the reflexive 3:7 lift.
    // Both factors point the same way and BOTH are blind: the 144 grades
    // knew (Abstracts, 0.85) above saw (Mirrors, 0.70), and 3:7 fires after
    // BOTH reversals (die @3:4, eat @3:6) while 3:6's lift fires after only
    // one — Adam's eyes open at the moment of maximal felt surprise.
    let mut by_quale: Vec<&RungLift> = out.lifts.iter().collect();
    by_quale.sort_by(|a, b| b.quale.total_cmp(&a.quale));
    println!(
        "  B5 quale ranking: {:?}",
        by_quale
            .iter()
            .map(|l| format!("{} {}={:.3}", l.verse, intern.name(l.verb), l.quale))
            .collect::<Vec<_>>()
    );
    let top = by_quale.first().expect("scene has lifts");
    assert!(
        top.self_referential && top.verse == "3:7",
        "the max-quale lift must be the reflexive 3:7 awareness, got {} (quale {:.3})",
        top.verse,
        top.quale
    );
    // The context factor alone must ALSO order them (staunen at 3:7 > 3:6 —
    // two held contradictions vs one), so the verdict never rests on the
    // modal factor alone (that would make the quale decoration over B2).
    let l37 = out.lifts.iter().find(|l| l.verse == "3:7").unwrap();
    let l36 = out.lifts.iter().find(|l| l.verse == "3:6").unwrap();
    assert!(
        l37.staunen_at > l36.staunen_at,
        "context factor must rise between 3:6 ({:.3}) and 3:7 ({:.3}) — the second \
         reversal (eat) landed in between",
        l36.staunen_at,
        l37.staunen_at
    );
    assert!(
        l37.modal > l36.modal,
        "and the blind factor grades knew above saw"
    );

    // B4: the hermeneutic circle converges — pass 2 changed NOTHING.
    println!(
        "  B4 pass-2: scene admitted={} revised={}, control admitted={} revised={}",
        out.pass2_admitted, out.pass2_revised, ctrl_out.pass2_admitted, ctrl_out.pass2_revised
    );
    assert_eq!(
        out.pass2_admitted, 0,
        "re-reading admits nothing new (fusion complete)"
    );
    assert_eq!(
        out.pass2_revised, 0,
        "same stamps → overlap guard → CHOICE only: evidence never double-counts"
    );
    assert!(
        scene_arena.reached_fixed_point,
        "the circle terminates at a fixed point"
    );
    assert_eq!(ctrl_out.pass2_admitted + ctrl_out.pass2_revised, 0);

    // B6 — the ASPECT PANEL: four stances, one crystal. Fire = the stances
    // produce genuinely DIFFERENT readings (the horizon matters); invariant
    // core = the reversal set and the reflexive lift are stance-independent
    // (the text constrains — horizons fuse without collapsing).
    print_stance_panel(&scene_arena, &intern, &out);
    let (hegel, nietzsche, kant, wittgenstein) = stance_panel(&scene_arena, &intern, &out);

    // Hegel re-reads B1: the Aufhebung ranking IS the held-contradiction set.
    assert_eq!(hegel.len(), 2, "Hegel sees exactly the two reversals");

    // Nietzsche sees what Hegel cannot: the same two depths, OPPOSITE
    // genealogies — one of each kind, and the right way round.
    assert_eq!(nietzsche.len(), 2);
    let flip_of = |name: &str| {
        nietzsche
            .iter()
            .find(|(s, _)| intern.name(s.p) == name)
            .map(|(_, k)| *k)
            .unwrap_or_else(|| panic!("{name} must carry a genealogy verdict"))
    };
    assert_eq!(
        flip_of("eat"),
        FlipKind::Transvaluation,
        "eat: forbidden (2:17, 3:1) → done (3:6) — the Umwertung"
    );
    assert_eq!(
        flip_of("die"),
        FlipKind::Devaluation,
        "die: asserted (2:17) → denied by the serpent (3:4)"
    );

    // Kant: ablating the reader's a-priori grading must CHANGE the measured
    // margin (inertness — the modal knob does work) while the scene crown
    // survives on context alone; and the reversal set is a posteriori —
    // modal never touches it (hegel above == B1, trivially modal-free).
    let q = |label: &str| {
        kant.iter()
            .find(|(l, _, _)| l.starts_with(label))
            .expect("lift present")
    };
    let (_, g37, u37) = q("3:7");
    let (_, g36, u36) = q("3:6");
    let graded_margin = g37 / g36;
    let ablated_margin = u37 / u36;
    println!("  B6 Kant margins: graded {graded_margin:.2}x vs ablated {ablated_margin:.2}x");
    assert!(
        graded_margin > ablated_margin + 0.1,
        "the a-priori grading WIDENS the awareness margin ({graded_margin:.2}x vs \
         {ablated_margin:.2}x) — the reader's categories contribute discrimination"
    );
    assert!(
        u37 > u36,
        "…while the scene crown survives ablation on context alone (a posteriori)"
    );

    // Wittgenstein crowns a concept NEITHER depth nor quale crowns: `naked`
    // participates in the most distinct language-games, strictly above all.
    let (w_crown, w_games) = wittgenstein[0];
    assert_eq!(intern.name(w_crown), "naked", "meaning-as-use crowns naked");
    assert_eq!(
        w_games, 3,
        "naked: Inh-object + knows-that object + Impl-cause"
    );
    assert!(
        wittgenstein[1].1 < w_games,
        "…strictly — every other concept plays fewer games"
    );
    // Horizon variance: Wittgenstein's crown is OUTSIDE Hegel's set — the
    // stances genuinely see different things in the same crystal.
    assert!(
        !hegel.iter().any(|(s, _)| s.p == w_crown || s.s == w_crown),
        "the use-crown (naked) is not a reversal — different light, different reflection"
    );

    println!("\n  ✓ the awareness event PRINTS: two blind-ranked reversals (die, eat), one");
    println!("    reflexive rung lift at 3:7, the explicit Impl(naked→afraid), and a");
    println!("    hermeneutic re-read that converges. Soup holds none of these.");
    println!("  ✓ and four philosopher stances read the SAME unchanged crystal into four");
    println!("    different reflections — with the reversal set and the reflexive lift");
    println!("    invariant across all of them. The horizon matters; the text constrains.");
}
