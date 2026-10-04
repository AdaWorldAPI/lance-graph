//! `fsm` — the part-of-speech finite-state machine that turns a tagged token
//! stream into [`Spo`] triples. It keeps v1's signature (tokens in, SPO out)
//! but not v1's machine: three clause states plus a one-level relative clause.
//!
//! Two entry points share one transition table:
//! - [`parse_to_spo`] takes one tag per token.
//! - [`parse_readings`] takes every admissible reading per token
//!   ([`PosSet`]) and keeps the alternatives the structure cannot separate
//!   (D-LXC-2). Frequency is never read here.
//!
//! The states track a minimal English clause: an optional determiner/modifier
//! run, a **subject** noun, a **verb** (predicate), an optional modifier run,
//! then an **object** noun that closes the triple. It is deliberately small and
//! deterministic — the semantics live in [`crate::space`], not here.
//!
//! ## Scope + known blind spot (paper-grounded, 2026-07-22)
//!
//! Full CFG parsing of natural language is combinatorially hostile (Moore 2000:
//! the Penn Treebank grammar averages **7.2×10²⁷ parses/sentence**); this FSM
//! deliberately commits to ONE coarse SPO reading and streams — determinism is
//! a feature at this scope (no recursion → cannot non-terminate). The
//! constituency the FSM omits is carried by the pointer fabric over the stream
//! ([`crate::wave`]; see
//! `.claude/knowledge/left-corner-grammar-tree-pointer-fabric.md`).
//!
//! **MOVEMENT constructions** (Liu 2025, JLM 13(2)) — object relatives ("the rat
//! that the cat bit"), topicalization, wh-fronting — invert canonical S/O order,
//! so a naive first-noun=subject emits the wrong triple. The cheapest
//! mitigation (logged fork, now BUILT 2026-07-23) is here: a
//! [`Pos::Rel`] relativizer/complementizer tag opens a single-level relative
//! clause whose embedded subject does NOT overwrite the matrix subject — the
//! relativizer's antecedent IS the matrix subject, exactly the FSM-side feeder
//! for the ±8 antecedent pointer in [`crate::wave`]. Object-relative
//! ("rat that cat bit ate cheese") and subject-relative ("dog that chased cat
//! barked man") both preserve the matrix subject through the embedded clause;
//! the embedded S-V-O emits its own triple. STILL out of scope (recency
//! heuristics, not attachment): coordination, nested/center-embedded relatives
//! (>1 level), topicalization, wh-fronting — the "last verb wins" / "re-anchor
//! subject" tie-breaks below will silently mis-bind those. This extracts a
//! coarse skeleton + the one commonest movement, not a parse.

use crate::spo::Spo;
use crate::vocab::WordId;
use causal_edge::pearl::CausalMask;

/// A coarse part-of-speech tag — the eight the FSM distinguishes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pos {
    /// Determiner / article (`the`, `a`) — skipped.
    Det,
    /// Adjective / modifier — attaches to the next noun (not part of the core SPO).
    Adj,
    /// Noun — a subject or object slot.
    Noun,
    /// Verb — the predicate slot.
    Verb,
    /// Relativizer / complementizer (`that`, `which`, `who`, `whom`, `whose`) —
    /// a clause-boundary marker. Promoted out of [`Pos::Other`] (2026-07-23) so
    /// the embedded clause's subject does NOT positionally overwrite the matrix
    /// subject (the movement blind spot). The relativizer's referent is the
    /// matrix subject (its antecedent); see `parse_to_spo`'s single-level
    /// relative-clause handling. This is the FSM-side feeder for the ±8
    /// antecedent pointer in [`crate::wave`] — a cheap positional tag, not the
    /// full gap→filler resolution.
    Rel,
    /// Other (preposition, conjunction, numeral, …) — skipped for the core
    /// triple.
    Other,
    /// End-of-sentence punctuation — flushes any partial clause.
    Stop,
    /// Adverb — skipped for the core triple like [`Pos::Other`], but kept
    /// apart so an adjective/adverb homograph can be decided by position
    /// (D-LXC-14) and an adverbial reading can feed the TEKAMOLO tenant.
    /// Unlike [`Pos::Adj`] it opens no nominal group. Last in bit order so
    /// the earlier tags keep their bits.
    Adv,
}

/// One tagged token: its palette [`WordId`] plus its [`Pos`].
#[derive(Debug, Clone, Copy)]
pub struct Tagged {
    /// Palette word id.
    pub id: WordId,
    /// Part of speech.
    pub pos: Pos,
}

impl Tagged {
    /// New tagged token.
    #[must_use]
    pub const fn new(id: WordId, pos: Pos) -> Self {
        Self { id, pos }
    }
}

/// The clause state as the FSM consumes tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    /// Nothing yet / after a flush — waiting for the subject noun.
    Start,
    /// Subject noun captured — waiting for the verb.
    HaveSubject,
    /// Subject + verb captured — waiting for the object noun.
    HaveVerb,
}

/// The relative-clause sub-machine (single level). Opened by a [`Pos::Rel`]
/// relativizer while a matrix subject is held; closed when the embedded clause
/// resolves, restoring the matrix subject.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Rel {
    /// No relative clause open.
    None,
    /// Relativizer just seen; the antecedent is the matrix subject. Awaiting an
    /// embedded subject noun (object-relative) or an embedded verb
    /// (subject-relative, antecedent = embedded subject).
    Open,
    /// Object-relative: the embedded subject noun is captured; awaiting the
    /// embedded verb (whose object is the antecedent).
    ObjSubject(WordId),
    /// The embedded verb is captured (subject-relative path); awaiting the
    /// embedded object noun, or the matrix verb which closes an intransitive
    /// embedded clause with no triple.
    HaveVerb { subj: WordId, verb: WordId },
}

/// The clause registers the FSM carries between tokens. One value of this is
/// one parser configuration; the single-reading [`parse_to_spo`] holds one,
/// the multi-reading [`parse_readings`] holds a small set of them.
///
/// After a `Stop` every register except `state`/`rel` is stale but never
/// read (a subject is always re-set before `HaveSubject` is entered), so a
/// fresh `Core` and a flushed one parse the rest of the stream identically.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Core {
    state: State,
    subject: WordId,
    predicate: WordId,
    // Single-level relative clause. When a relativizer opens one, `matrix`
    // parks the untouched matrix subject and `antecedent` records what the
    // relativizer refers to (the matrix subject). The embedded clause resolves
    // into its own triple without clobbering `subject`/`state`; on close, the
    // matrix subject resumes in `HaveSubject`.
    rel: Rel,
    matrix: WordId,
    antecedent: WordId,
    /// The previous token was a determiner or adjective: a nominal group is
    /// open and waiting for its head. Read only by [`parse_readings`]'s
    /// licensing and slot rules; [`parse_to_spo`] never consults it.
    nominal_open: bool,
    /// The open nominal group was opened by an adjective, not a determiner.
    /// Licensing reads it only where [`Typology::adjective_opens_nominal`].
    adjective_open: bool,
    /// The previous token was a determiner that came straight after the
    /// subject's head noun while the clause still waited for its verb: a
    /// quantifier floating off the subject ("the men **all** eat"), not an
    /// object's start. A determiner after any other word ("when **the** land")
    /// opens a group as usual. Read only by licensing.
    floating_quantifier: bool,
    /// The previous token was a noun. Read only to recognise a floating
    /// quantifier, which follows the subject's head directly.
    after_noun: bool,
    /// The subject slot holds an object carried over from the previous
    /// triple ("gave **him** the charge"). A determiner after it opens the
    /// next object's group, so it is never a floating quantifier.
    subject_carried: bool,
    /// The previous token opened the clause as its subject — a noun taken in
    /// [`State::Start`] with no determiner or adjective in front ("they",
    /// "men", "God"). An object carried into the subject slot ("gave him …")
    /// or a re-anchored noun ("mothers house") is not fresh. Read only by
    /// the slot rule.
    fresh_subject: bool,
    /// A verb has taken a predicate slot in this sentence (matrix or
    /// relative clause). Read only by the clause rule
    /// ([`Typology::predicate_required`]).
    predicated: bool,
}

impl Core {
    const START: Self = Self {
        state: State::Start,
        subject: 0,
        predicate: 0,
        rel: Rel::None,
        matrix: 0,
        antecedent: 0,
        nominal_open: false,
        adjective_open: false,
        floating_quantifier: false,
        after_noun: false,
        subject_carried: false,
        fresh_subject: false,
        predicated: false,
    };

    /// Consume one non-`Stop` token; returns the triple it closes, if any.
    /// This is the whole single-reading transition table, unchanged.
    fn step(&mut self, t: Tagged) -> Option<Spo> {
        debug_assert!(t.pos != Pos::Stop, "Stop is handled by the caller");
        let nominal_was_open = self.nominal_open;
        self.fresh_subject = false;
        self.nominal_open = matches!(t.pos, Pos::Det | Pos::Adj);
        self.adjective_open = t.pos == Pos::Adj;
        self.floating_quantifier = t.pos == Pos::Det
            && self.after_noun
            && !self.subject_carried
            && matches!(self.state, State::HaveSubject);
        self.after_noun = t.pos == Pos::Noun;

        // While a relative clause is open, its own tiny machine consumes the
        // embedded S-V-O; the matrix subject stays parked in `matrix`.
        if self.rel != Rel::None {
            match (self.rel, t.pos) {
                (_, Pos::Stop) => unreachable!("Stop handled by the caller"),
                (_, Pos::Det | Pos::Adj | Pos::Adv | Pos::Other) => {}
                // Object-relative: a noun after the relativizer is the embedded
                // subject ("rat that [cat] bit …"); the antecedent is its object.
                (Rel::Open, Pos::Noun) => self.rel = Rel::ObjSubject(t.id),
                // Subject-relative: a verb right after the relativizer means the
                // antecedent is the embedded subject ("dog that [chased] …").
                (Rel::Open, Pos::Verb) => {
                    self.predicated = true;
                    self.rel = Rel::HaveVerb {
                        subj: self.antecedent,
                        verb: t.id,
                    };
                }
                // Object-relative embedded verb: emit (embedded_subj, verb,
                // antecedent) and close — the antecedent IS the object.
                (Rel::ObjSubject(es), Pos::Verb) => {
                    self.predicated = true;
                    let out = Spo::new(es, t.id, self.antecedent);
                    self.subject = self.matrix;
                    self.state = State::HaveSubject;
                    self.rel = Rel::None;
                    return Some(out);
                }
                // Object-relative saw a second noun before its verb — treat the
                // newest as the embedded subject (recency), stay open.
                (Rel::ObjSubject(_), Pos::Noun) => self.rel = Rel::ObjSubject(t.id),
                (Rel::Open, Pos::Rel) => {} // stray relativizer — ignore
                (Rel::ObjSubject(_), Pos::Rel) => {}
                // Subject-relative embedded object closes the embedded triple.
                (Rel::HaveVerb { subj, verb }, Pos::Noun) => {
                    let out = Spo::new(subj, verb, t.id);
                    self.subject = self.matrix;
                    self.state = State::HaveSubject;
                    self.rel = Rel::None;
                    return Some(out);
                }
                // A second verb (no embedded object seen) is the MATRIX verb:
                // close the (intransitive → no triple) embedded clause and let
                // the matrix subject take this verb as its predicate.
                (Rel::HaveVerb { .. }, Pos::Verb) => {
                    self.predicated = true;
                    self.subject = self.matrix;
                    self.predicate = t.id;
                    self.state = State::HaveVerb;
                    self.rel = Rel::None;
                }
                (Rel::HaveVerb { .. }, Pos::Rel) => {}
                (Rel::None, _) => unreachable!("guarded by rel != Rel::None"),
            }
            return None;
        }

        match (self.state, t.pos) {
            // Skip determiners, modifiers, adverbs — they are not core slots.
            (_, Pos::Det | Pos::Adj | Pos::Adv | Pos::Other) => {}
            (_, Pos::Stop) => unreachable!("Stop handled by the caller"),

            // A relativizer only opens a relative clause when we already have a
            // matrix subject for it to modify; elsewhere it is inert (skipped
            // like Other), never resetting S/O.
            (State::HaveSubject, Pos::Rel) => {
                self.matrix = self.subject;
                self.antecedent = self.subject;
                self.rel = Rel::Open;
            }
            (State::Start | State::HaveVerb, Pos::Rel) => {}

            (State::Start, Pos::Noun) => {
                self.subject = t.id;
                self.state = State::HaveSubject;
                self.fresh_subject = !nominal_was_open;
                self.subject_carried = false;
            }
            (State::HaveSubject, Pos::Verb) => {
                self.predicated = true;
                self.predicate = t.id;
                self.state = State::HaveVerb;
            }
            (State::HaveVerb, Pos::Noun) => {
                let out = Spo::new(self.subject, self.predicate, t.id);
                // Serial-verb chain: the object seeds the next subject.
                self.subject = t.id;
                self.state = State::HaveSubject;
                self.subject_carried = true;
                return Some(out);
            }
            // A verb before a subject, or a second verb, restarts cleanly.
            (State::Start, Pos::Verb) => {}
            (State::HaveSubject, Pos::Noun) => {
                // Re-anchor the subject; it is no longer a carried object.
                self.subject = t.id;
                self.subject_carried = false;
            }
            (State::HaveVerb, Pos::Verb) => self.predicate = t.id, // last verb wins
        }
        None
    }
}

/// Parse a tagged token stream into SPO triples via the PoS FSM.
///
/// A triple is emitted whenever an object noun closes a `subject → verb →
/// object` clause; the object then becomes the subject of the next clause
/// (serial-verb chaining, as v1 did). A `Stop` resets to [`State::Start`].
///
/// One tag per token. [`parse_readings`] is the multi-reading form; on
/// input where every token has exactly one reading the two agree.
#[must_use]
pub fn parse_to_spo(tokens: &[Tagged]) -> Vec<Spo> {
    let mut out = Vec::new();
    let mut core = Core::START;
    for &t in tokens {
        // A stop flushes everything, matrix and embedded alike.
        if t.pos == Pos::Stop {
            core.state = State::Start;
            core.rel = Rel::None;
            continue;
        }
        out.extend(core.step(t));
    }
    out
}

/// For each token, the Pearl 2³ mask of SPO questions its clause has answered
/// BEFORE it arrives — the school question test (*Frageprobe*) as a position:
/// a word arriving after *who/what?* (subject) is answered and before *does
/// what?* (predicate) is, is asked *does what?*; *whom/what?* is the object.
/// The mask is [`CausalMask`], the same eight values `CausalEdge64` packs, so
/// the ladder has one bit order (S = `0b100`, P = `0b010`, O = `0b001`). Steps
/// the same transition table as [`parse_to_spo`]; a `Stop` starts a new
/// clause ([`CausalMask::None`]).
#[must_use]
pub fn answered_questions(tokens: &[Tagged]) -> Vec<CausalMask> {
    let mut out = Vec::with_capacity(tokens.len());
    let mut core = Core::START;
    let mut emitted = false;
    for &t in tokens {
        let mut mask = 0;
        if matches!(core.state, State::HaveSubject | State::HaveVerb) {
            mask |= CausalMask::S as u8;
        }
        if core.state == State::HaveVerb || emitted {
            mask |= CausalMask::P as u8;
        }
        if emitted {
            mask |= CausalMask::O as u8;
        }
        out.push(CausalMask::from_bits(mask));
        if t.pos == Pos::Stop {
            core.state = State::Start;
            core.rel = Rel::None;
            emitted = false;
            continue;
        }
        emitted |= core.step(t).is_some();
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Multi-reading input (D-LXC-2)
// ─────────────────────────────────────────────────────────────────────────────

/// A set of [`Pos`] readings for one token, as a bitmask.
///
/// This is the FSM's own alphabet, so it is language-neutral: a source
/// lexicon maps its tags onto [`Pos`] at its own boundary (see
/// [`crate::coca`] for COCA) and hands the FSM every admissible reading.
/// The empty set means "no lexical reading is known" — it is not a reading
/// and not a zero count.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct PosSet(u8);

impl Pos {
    /// Every tag, in bit order.
    pub const ALL: [Pos; 8] = [
        Pos::Det,
        Pos::Adj,
        Pos::Noun,
        Pos::Verb,
        Pos::Rel,
        Pos::Other,
        Pos::Stop,
        Pos::Adv,
    ];

    const fn bit(self) -> u8 {
        1 << (self as u8)
    }
}

impl PosSet {
    /// No known reading.
    pub const EMPTY: Self = Self(0);

    /// The set holding exactly `pos`.
    #[must_use]
    pub const fn single(pos: Pos) -> Self {
        Self(pos.bit())
    }

    /// This set with `pos` added.
    #[must_use]
    pub const fn with(self, pos: Pos) -> Self {
        Self(self.0 | pos.bit())
    }

    /// Whether `pos` is in the set.
    #[must_use]
    pub const fn contains(self, pos: Pos) -> bool {
        self.0 & pos.bit() != 0
    }

    /// Union.
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// This set with `pos` removed.
    #[must_use]
    pub const fn without(self, pos: Pos) -> Self {
        Self(self.0 & !pos.bit())
    }

    /// Number of readings.
    #[must_use]
    pub const fn len(self) -> usize {
        self.0.count_ones() as usize
    }

    /// Whether no reading is known.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// The readings, in [`Pos::ALL`] order.
    pub fn iter(self) -> impl Iterator<Item = Pos> {
        Pos::ALL.into_iter().filter(move |p| self.contains(*p))
    }
}

impl FromIterator<Pos> for PosSet {
    fn from_iter<I: IntoIterator<Item = Pos>>(iter: I) -> Self {
        iter.into_iter().fold(Self::EMPTY, Self::with)
    }
}

/// One token with every reading it is admitted under.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Reading {
    /// Palette word id.
    pub id: WordId,
    /// The admissible readings. [`PosSet::EMPTY`] = lexically unknown.
    pub pos: PosSet,
}

impl Reading {
    /// A token admitted under `pos`.
    #[must_use]
    pub const fn new(id: WordId, pos: PosSet) -> Self {
        Self { id, pos }
    }

    /// A sentence boundary.
    #[must_use]
    pub const fn stop() -> Self {
        Self {
            id: 0,
            pos: PosSet::single(Pos::Stop),
        }
    }
}

impl From<Tagged> for Reading {
    fn from(t: Tagged) -> Self {
        Self::new(t.id, PosSet::single(t.pos))
    }
}

/// What happened to one ambiguous token (two or more readings on entry).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Survivor {
    /// Index of the token in the input slice.
    pub index: usize,
    /// The readings it entered with.
    pub entered: PosSet,
    /// The readings still used by at least one surviving configuration when
    /// its sentence ended.
    pub survived: PosSet,
}

/// The result of [`parse_readings`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ReadingParse {
    /// Triples emitted on EVERY surviving configuration of their sentence,
    /// in stream order. With one reading per token this is exactly
    /// [`parse_to_spo`]'s output.
    pub certain: Vec<Spo>,
    /// Triples emitted on SOME but not all surviving configurations — the
    /// readings the available structure could not separate. Not truth, not
    /// ranked: a later layer (morphology, context) may eliminate them.
    pub alternative: Vec<Spo>,
    /// Every ambiguous token, in input order.
    pub ambiguous: Vec<Survivor>,
    /// Tokens that arrived with [`PosSet::EMPTY`] (no known reading). They are
    /// stepped as [`Pos::Other`] — skipped — and never reported as a reading.
    pub unknown: usize,
    /// Largest number of live configurations after merging.
    pub peak_configs: usize,
    /// Times a sentence was force-flushed because it exceeded
    /// [`MAX_CONFIGS`]. Each flush is a reported clause break, never a pick.
    pub overflow_flushes: usize,
    /// (configuration, reading) pairs the licensing rule dropped.
    pub unlicensed_dropped: usize,
    /// (configuration, reading) pairs the slot rule dropped.
    pub slot_dropped: usize,
    /// Tokens whose adjective/adverb readings the attribute rule narrowed.
    pub attribute_narrowed: usize,
    /// Configurations the clause rule dropped at a sentence end.
    pub unpredicated_dropped: usize,
}

/// Bound on live configurations per sentence. Exceeding it flushes the
/// sentence at that token (a reported clause break, see
/// [`ReadingParse::overflow_flushes`]) instead of choosing a reading.
pub const MAX_CONFIGS: usize = 256;

/// One live configuration: the parser registers, the triples this path has
/// emitted in the current sentence, and — for each ambiguous token so far —
/// the readings that reach this configuration.
#[derive(Debug, Clone)]
struct Config {
    core: Core,
    emitted: Vec<Spo>,
    support: Vec<PosSet>,
}

/// Word order of an attributive adjective relative to its noun (WALS 87A).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdjectiveOrder {
    /// "the old man", "der alte Mann".
    Before,
    /// "l'homme âgé".
    After,
    /// Both orders are common ("un bon homme", "un homme bon").
    Both,
}

/// The word-order facts the position rules read. A language is described by
/// these, never by rules of its own; [`Typology::ENGLISH`] is the default,
/// and a treebank's train split can measure them (see the `ud_pos_eval`
/// example).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Typology {
    /// Where an attributive adjective stands.
    pub adjective: AdjectiveOrder,
    /// An adjective is always followed by the rest of its nominal group, so a
    /// verb never comes straight after it. True for English; false where a
    /// predicative adjective precedes its verb ("weil es möglich ist") or an
    /// adjective follows its noun ("un projet important est").
    pub adjective_opens_nominal: bool,
    /// The [`AttributeRule`] clauses allowed to narrow a token. Empty means
    /// the attribute rule only reports ([`attribute_rule`]) and never drops
    /// a reading.
    pub attribute_rules: &'static [AttributeRule],
    /// The clause rule — "this sentence has no verb yet; which word could it
    /// be": at a sentence end, drop the reading combinations on which no verb
    /// took a predicate slot, if some combination's verb did. A rule switch,
    /// not a word-order fact; off in [`Typology::ENGLISH`] (KJV verses are
    /// often verbless fragments).
    pub predicate_required: bool,
}

impl Typology {
    /// English: adjectives before the noun, predicatives after the verb.
    pub const ENGLISH: Self = Self {
        adjective: AdjectiveOrder::Before,
        adjective_opens_nominal: true,
        attribute_rules: &[],
        predicate_required: false,
    };
}

impl Default for Typology {
    fn default() -> Self {
        Self::ENGLISH
    }
}

/// The licensing rule: a verb cannot follow a determiner, whose nominal
/// group is still waiting for its head — nor an adjective, where the
/// [`Typology`] says an adjective always opens one. [`parse_readings`]
/// applies it relatively: it never removes a token's last admissible
/// reading.
///
/// Except a determiner that arrived while the clause held a subject and still
/// waited for its verb ([`Core::floating_quantifier`]): it is a quantifier
/// floating off the subject ("the men **all** eat"), not the start of an
/// object, so the verb slot is still open. An adjective there still opens a
/// group.
const fn licensed(core: &Core, pos: Pos, typology: Typology) -> bool {
    let open = core.nominal_open
        && (!core.adjective_open || typology.adjective_opens_nominal)
        && !core.floating_quantifier;
    !(open && matches!(pos, Pos::Verb))
}

/// One clause of the attribute rule (D-LXC-14). Each names the position it
/// reads and the reading it keeps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AttributeRule {
    /// Right after a determiner-only word → adjective ("the **only** son").
    AfterDeterminer,
    /// Right before a word that can be an adjective → adverb, a degree
    /// modifier ("**very** good", "**pretty** high", "**sehr** gut").
    BeforeAdjective,
    /// Between a word that can be a subject and a word that can be a verb →
    /// adverb ("I **just** need", "airfare **alone** will").
    BetweenSubjectAndVerb,
    /// Next to a noun on the side [`Typology::adjective`] allows → adjective
    /// ("a **high** rate", "un homme **bon**").
    NextToNoun,
}

impl AttributeRule {
    /// Every clause, in the order [`attribute_rule`] tries them.
    pub const ALL: [AttributeRule; 4] = [
        AttributeRule::AfterDeterminer,
        AttributeRule::BeforeAdjective,
        AttributeRule::BetweenSubjectAndVerb,
        AttributeRule::NextToNoun,
    ];

    /// The reading this clause keeps (the other of Adj/Adv is dropped).
    #[must_use]
    pub const fn keeps(self) -> Pos {
        match self {
            AttributeRule::AfterDeterminer | AttributeRule::NextToNoun => Pos::Adj,
            AttributeRule::BeforeAdjective | AttributeRule::BetweenSubjectAndVerb => Pos::Adv,
        }
    }
}

/// The clause of the attribute rule that decides an adjective/adverb
/// homograph from its neighbours, never from which reading a corpus counts
/// more often — or `None` (the token lacks one of Adj/Adv, or no clause
/// matches: after a verb, "was **good**" and "ran **fast**" look the same).
/// Clauses are tried in [`AttributeRule::ALL`] order; `prev` / `next` are the
/// neighbours' readings within the sentence ([`PosSet::EMPTY`] at an edge).
#[must_use]
pub fn attribute_rule(
    prev: PosSet,
    this: PosSet,
    next: PosSet,
    typology: Typology,
) -> Option<AttributeRule> {
    if !(this.contains(Pos::Adj) && this.contains(Pos::Adv)) {
        return None;
    }
    let noun_after = next.contains(Pos::Noun)
        && matches!(
            typology.adjective,
            AdjectiveOrder::Before | AdjectiveOrder::Both
        );
    let noun_before = prev.contains(Pos::Noun)
        && matches!(
            typology.adjective,
            AdjectiveOrder::After | AdjectiveOrder::Both
        );
    AttributeRule::ALL.into_iter().find(|rule| match rule {
        AttributeRule::AfterDeterminer => prev == PosSet::single(Pos::Det),
        AttributeRule::BeforeAdjective => next.contains(Pos::Adj),
        AttributeRule::BetweenSubjectAndVerb => {
            prev.contains(Pos::Noun) && next.contains(Pos::Verb)
        }
        AttributeRule::NextToNoun => noun_after || noun_before,
    })
}

/// `this` narrowed by [`attribute_rule`]'s clause, if one is enabled in
/// `typology` and matches; otherwise `this` unchanged. Only one of Adj/Adv is
/// ever removed; every other reading stays.
#[must_use]
pub fn attribute_readings(prev: PosSet, this: PosSet, next: PosSet, typology: Typology) -> PosSet {
    match attribute_rule(prev, this, next, typology) {
        Some(rule) if typology.attribute_rules.contains(&rule) => {
            let drop = if rule.keeps() == Pos::Adj {
                Pos::Adv
            } else {
                Pos::Adj
            };
            this.without(drop)
        }
        _ => this,
    }
}

/// The slot rule: directly after a fresh subject noun (one that opened the
/// clause with no determiner or adjective in front of it), a word that can be
/// a verb fills the predicate slot — "they record", "he rose", "men sleep".
/// Its noun reading would make a compound with the subject instead, so it is
/// dropped, however much more often the corpus counts the word as a noun.
/// After a determined noun ("a sin offering"), a carried object ("gave him
/// charge") or a re-anchored noun ("mothers house") both readings stay.
/// Applied relatively, like licensing, and only when the token offers a verb.
fn slot_allows(core: &Core, pos: Pos, set: PosSet) -> bool {
    !(core.rel == Rel::None
        && core.state == State::HaveSubject
        && core.fresh_subject
        && pos == Pos::Noun
        && set.contains(Pos::Verb))
}

/// Parse a multi-reading token stream.
///
/// Each token enters with every reading in its [`PosSet`]. The parser keeps
/// a set of configurations, one per distinct path, and steps every
/// configuration by every reading with the same transition table as
/// [`parse_to_spo`]. Two configurations with the same registers and the same
/// triples so far are merged, and their reading histories joined.
///
/// **Elimination.** A (configuration, reading) pair is dropped when the
/// licensing rule forbids it — a verb right after a determiner or adjective
/// — but only if some other pair for the same token is allowed. When every
/// pair is forbidden the rule stands aside for that token, so a word with a
/// single reading is never rejected, and input with one reading per token
/// parses exactly as [`parse_to_spo`]. Frequency is never consulted.
///
/// **Slot rule.** Directly after a fresh subject noun, a token that can be a
/// verb loses its noun reading (see [`slot_allows`]): the position between
/// subject and object is the predicate, however often COCA counts the word
/// as a noun. Relative like licensing, so it never empties a token.
///
/// The "a sentence needs a predicate" clause rule
/// ([`Typology::predicate_required`]) exists but is off here: KJV verses are
/// often verbless fragments ("the goats for sin offering"), and on the KJV
/// that rule turned nouns into verbs about three times in four.
///
/// **Output.** At each `Stop` (and at the end of input) the sentence's
/// configurations are compared: triples found on all of them are
/// [`ReadingParse::certain`], the rest [`ReadingParse::alternative`].
#[must_use]
pub fn parse_readings(tokens: &[Reading]) -> ReadingParse {
    parse_readings_with(tokens, Typology::ENGLISH)
}

/// [`parse_readings`] for a language described by `typology`.
#[must_use]
pub fn parse_readings_with(tokens: &[Reading], typology: Typology) -> ReadingParse {
    let mut out = ReadingParse::default();
    let mut configs = vec![fresh()];
    // (input index, entered set) for each ambiguous token of this sentence.
    let mut pending: Vec<(usize, PosSet)> = Vec::new();

    for (index, tok) in tokens.iter().enumerate() {
        if tok.pos.contains(Pos::Stop) {
            finish_sentence(&mut out, &mut configs, &mut pending, typology);
            continue;
        }
        let set = if tok.pos.is_empty() {
            out.unknown += 1;
            PosSet::single(Pos::Other)
        } else {
            // Neighbours are read in place from the borrowed input; a stop
            // is a sentence edge.
            let edge = |t: Option<&Reading>| match t {
                Some(t) if !t.pos.contains(Pos::Stop) => t.pos,
                _ => PosSet::EMPTY,
            };
            let prev = edge(index.checked_sub(1).and_then(|i| tokens.get(i)));
            let next = edge(tokens.get(index + 1));
            let narrowed = attribute_readings(prev, tok.pos, next, typology);
            if narrowed != tok.pos {
                out.attribute_narrowed += 1;
            }
            narrowed
        };
        let ambiguous = tok.pos.len() > 1;
        if ambiguous {
            pending.push((index, tok.pos));
        }

        let allowed =
            |c: &Config, p: Pos| licensed(&c.core, p, typology) && slot_allows(&c.core, p, set);
        let any_allowed = configs.iter().any(|c| set.iter().any(|p| allowed(c, p)));
        let mut next: Vec<Config> = Vec::with_capacity(configs.len() * set.len());
        for c in &configs {
            for pos in set.iter() {
                if any_allowed && !allowed(c, pos) {
                    if licensed(&c.core, pos, typology) {
                        out.slot_dropped += 1;
                    } else {
                        out.unlicensed_dropped += 1;
                    }
                    continue;
                }
                let mut n = c.clone();
                n.emitted.extend(n.core.step(Tagged::new(tok.id, pos)));
                if typology.adjective_opens_nominal {
                    // Determiner and adjective open the same group here, so
                    // keep their paths mergeable.
                    n.core.adjective_open = false;
                }
                if ambiguous {
                    n.support.push(PosSet::single(pos));
                }
                merge_into(&mut next, n);
            }
        }
        configs = next;
        out.peak_configs = out.peak_configs.max(configs.len());
        if configs.len() > MAX_CONFIGS {
            out.overflow_flushes += 1;
            finish_sentence(&mut out, &mut configs, &mut pending, typology);
        }
    }
    finish_sentence(&mut out, &mut configs, &mut pending, typology);
    out
}

fn fresh() -> Config {
    Config {
        core: Core::START,
        emitted: Vec::new(),
        support: Vec::new(),
    }
}

/// Add `n` to `configs`, merging it into an equivalent configuration (same
/// registers, same triples) by joining their reading histories.
fn merge_into(configs: &mut Vec<Config>, n: Config) {
    if let Some(c) = configs
        .iter_mut()
        .find(|c| c.core == n.core && c.emitted == n.emitted)
    {
        for (a, b) in c.support.iter_mut().zip(&n.support) {
            *a = a.union(*b);
        }
    } else {
        configs.push(n);
    }
}

/// Close the current sentence: classify its triples, record the surviving
/// readings of its ambiguous tokens, and reset to one fresh configuration.
fn finish_sentence(
    out: &mut ReadingParse,
    configs: &mut Vec<Config>,
    pending: &mut Vec<(usize, PosSet)>,
    typology: Typology,
) {
    if typology.predicate_required && configs.iter().any(|c| c.core.predicated) {
        let before = configs.len();
        configs.retain(|c| c.core.predicated);
        out.unpredicated_dropped += before - configs.len();
    }
    if let Some((first, rest)) = configs.split_first() {
        for t in &first.emitted {
            if rest.iter().all(|c| c.emitted.contains(t)) {
                out.certain.push(*t);
            }
        }
        // Deduplicated within this sentence only: the same triple may be an
        // alternative of two different sentences.
        let mut alternative: Vec<Spo> = Vec::new();
        for c in configs.iter() {
            for t in &c.emitted {
                let everywhere = configs.iter().all(|o| o.emitted.contains(t));
                if !everywhere && !alternative.contains(t) {
                    alternative.push(*t);
                }
            }
        }
        out.alternative.extend(alternative);
    }
    for (k, (index, entered)) in pending.iter().enumerate() {
        let survived = configs
            .iter()
            .filter_map(|c| c.support.get(k).copied())
            .fold(PosSet::EMPTY, PosSet::union);
        out.ambiguous.push(Survivor {
            index: *index,
            entered: *entered,
            survived,
        });
    }
    pending.clear();
    configs.clear();
    configs.push(fresh());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn n(id: WordId) -> Tagged {
        Tagged::new(id, Pos::Noun)
    }
    fn v(id: WordId) -> Tagged {
        Tagged::new(id, Pos::Verb)
    }
    fn det() -> Tagged {
        Tagged::new(0, Pos::Det)
    }
    fn adj(id: WordId) -> Tagged {
        Tagged::new(id, Pos::Adj)
    }

    #[test]
    fn the_big_dog_bit_the_old_man() {
        // "the big dog bit the old man" → SPO(dog, bit, man).
        let toks = [det(), adj(11), n(101), v(202), det(), adj(12), n(303)];
        let spo = parse_to_spo(&toks);
        assert_eq!(spo, vec![Spo::new(101, 202, 303)]);
    }

    #[test]
    fn serial_verbs_chain_the_object_into_the_next_subject() {
        // "dog bit man saw cat" → (dog,bit,man) then (man,saw,cat).
        let toks = [n(1), v(2), n(3), v(4), n(5)];
        let spo = parse_to_spo(&toks);
        assert_eq!(spo, vec![Spo::new(1, 2, 3), Spo::new(3, 4, 5)]);
    }

    #[test]
    fn stop_flushes_a_partial_clause() {
        // "dog bit ." then "cat ran mouse" → only the second closes a triple.
        let toks = [n(1), v(2), Tagged::new(0, Pos::Stop), n(10), v(20), n(30)];
        let spo = parse_to_spo(&toks);
        assert_eq!(spo, vec![Spo::new(10, 20, 30)]);
    }

    fn rel() -> Tagged {
        Tagged::new(0, Pos::Rel)
    }

    #[test]
    fn object_relative_preserves_the_matrix_subject() {
        // "the rat that the cat bit ate the cheese"
        //   rat=1 cat=2 bit=3 ate=4 cheese=5
        // Correct: (cat, bit, rat) embedded, then (rat, ate, cheese) matrix.
        // The BUG this fixes: without Pos::Rel, `cat` re-anchors the subject and
        // the matrix triple becomes the wrong (cat, ate, cheese).
        let toks = [det(), n(1), rel(), det(), n(2), v(3), v(4), det(), n(5)];
        let spo = parse_to_spo(&toks);
        assert_eq!(spo, vec![Spo::new(2, 3, 1), Spo::new(1, 4, 5)]);
    }

    #[test]
    fn subject_relative_preserves_the_matrix_subject() {
        // "the dog that chased the cat barked the man" (at→dropped)
        //   dog=1 chased=2 cat=3 barked=4 man=5
        // Correct: (dog, chased, cat) embedded, then (dog, barked, man) matrix.
        let toks = [det(), n(1), rel(), v(2), det(), n(3), v(4), det(), n(5)];
        let spo = parse_to_spo(&toks);
        assert_eq!(spo, vec![Spo::new(1, 2, 3), Spo::new(1, 4, 5)]);
    }

    #[test]
    fn intransitive_subject_relative_emits_only_the_matrix_triple() {
        // "the man who slept woke the child"
        //   man=1 slept=2 woke=3 child=4
        // "slept" has no object → no embedded triple; matrix (man, woke, child).
        let toks = [det(), n(1), rel(), v(2), v(3), det(), n(4)];
        let spo = parse_to_spo(&toks);
        assert_eq!(spo, vec![Spo::new(1, 3, 4)]);
    }

    #[test]
    fn relativizer_without_a_matrix_subject_is_inert() {
        // A relativizer at Start (no antecedent) must not corrupt the next clause.
        let toks = [rel(), n(1), v(2), n(3)];
        let spo = parse_to_spo(&toks);
        assert_eq!(spo, vec![Spo::new(1, 2, 3)]);
    }

    // ── multi-reading parser (D-LXC-2) ──────────────────────────────────

    fn one(id: WordId, pos: Pos) -> Reading {
        Reading::new(id, PosSet::single(pos))
    }
    fn noun_or_verb(id: WordId) -> Reading {
        Reading::new(id, PosSet::single(Pos::Noun).with(Pos::Verb))
    }
    const NV: PosSet = PosSet::single(Pos::Noun).with(Pos::Verb);

    /// T1: a token with two readings enters with both.
    #[test]
    fn multiple_readings_survive_entry() {
        // "men record deeds": record = noun|verb.
        let p = parse_readings(&[one(1, Pos::Noun), noun_or_verb(2), one(3, Pos::Noun)]);
        assert_eq!(p.ambiguous.len(), 1);
        assert_eq!(p.ambiguous[0].index, 1);
        assert_eq!(p.ambiguous[0].entered, NV);
    }

    /// T2: structure removes a reading. After a determiner the nominal group
    /// waits for its head, so `record`'s verb reading has no licence.
    #[test]
    fn a_determiner_removes_the_verb_reading() {
        // "the record fell": the=det, record=noun|verb, fell=verb.
        let p = parse_readings(&[one(9, Pos::Det), noun_or_verb(2), one(4, Pos::Verb)]);
        assert_eq!(p.ambiguous[0].entered.len(), 2);
        assert_eq!(p.ambiguous[0].survived, PosSet::single(Pos::Noun));
        assert!(p.alternative.is_empty());
    }

    /// T3: when nothing in reach separates the readings, both stay, and the
    /// triple only one of them builds is an alternative, not certain. A
    /// decoder that always commits to one tag would report a certain triple
    /// or none.
    #[test]
    fn ambiguity_remains_when_structure_cannot_decide() {
        // "the men record deeds rot": verb path → (men, record, deeds) then
        // `rot`; noun path → `rot` is the predicate of "the men record
        // deeds". The subject is determined, so the slot rule does not apply.
        let p = parse_readings(&[
            one(9, Pos::Det),
            one(1, Pos::Noun),
            noun_or_verb(2),
            one(3, Pos::Noun),
            one(6, Pos::Verb),
        ]);
        assert_eq!(p.ambiguous[0].survived, NV);
        assert!(p.certain.is_empty());
        assert_eq!(p.alternative, vec![Spo::new(1, 2, 3)]);
        assert_eq!(p.slot_dropped, 0);
    }

    /// No "a sentence needs a predicate" rule: a verbless fragment keeps
    /// its noun reading. "the goats for sin offering" — `offering` follows a
    /// re-anchored noun, not a fresh subject, so neither rule touches it.
    #[test]
    fn a_verbless_fragment_keeps_its_noun_reading() {
        let p = parse_readings(&[
            one(9, Pos::Det),
            one(1, Pos::Noun),
            one(7, Pos::Other),
            one(3, Pos::Noun),
            noun_or_verb(2),
        ]);
        assert_eq!(p.ambiguous[0].survived, NV);
        assert_eq!(p.slot_dropped, 0);
    }

    /// The slot rule: right after a fresh subject noun, a homograph is the
    /// predicate, even where a later verb would give the sentence another
    /// one ("men record deeds rot").
    #[test]
    fn a_homograph_after_a_fresh_subject_is_the_predicate() {
        let p = parse_readings(&[
            one(1, Pos::Noun),
            noun_or_verb(2),
            one(3, Pos::Noun),
            one(6, Pos::Verb),
        ]);
        assert_eq!(p.ambiguous[0].survived, PosSet::single(Pos::Verb));
        assert_eq!(p.certain, vec![Spo::new(1, 2, 3)]);
        assert!(p.alternative.is_empty());
        assert_eq!(p.slot_dropped, 1);
        // Intransitive: "men sleep" — no triple, but the verb reading survives.
        let p = parse_readings(&[one(1, Pos::Noun), noun_or_verb(2)]);
        assert_eq!(p.ambiguous[0].survived, PosSet::single(Pos::Verb));
        assert!(p.certain.is_empty() && p.alternative.is_empty());
    }

    /// Silence twins of the slot rule: it does not fire on a word with no
    /// verb reading, after a determined subject, after an object carried
    /// into the subject slot, or after a re-anchored noun.
    #[test]
    fn the_slot_rule_needs_a_fresh_subject_and_a_verb_reading() {
        // "men stones rot": `stones` has no verb reading, so it stays.
        let p = parse_readings(&[one(1, Pos::Noun), one(3, Pos::Noun), one(6, Pos::Verb)]);
        assert_eq!(p.slot_dropped, 0);
        // "the men record deeds rot" keeps both readings (no slot drop).
        let p = parse_readings(&[
            one(9, Pos::Det),
            one(1, Pos::Noun),
            noun_or_verb(2),
            one(3, Pos::Noun),
            one(6, Pos::Verb),
        ]);
        assert_eq!(p.ambiguous[0].survived, NV);
        assert_eq!(p.slot_dropped, 0);
        // A homograph after a verb is an object, not a predicate:
        // "men saw record" keeps both readings for `record`.
        let p = parse_readings(&[one(1, Pos::Noun), one(4, Pos::Verb), noun_or_verb(2)]);
        assert_eq!(p.slot_dropped, 0);
        // "he gave him charge": `him` closes (he, gave, him) and is carried
        // into the subject slot; `charge` keeps both readings.
        let p = parse_readings(&[
            one(1, Pos::Noun),
            one(4, Pos::Verb),
            one(3, Pos::Noun),
            noun_or_verb(2),
        ]);
        assert_eq!(p.ambiguous[0].survived, NV);
        assert_eq!(p.slot_dropped, 0);
        // "mothers house": `house` follows a re-anchored noun.
        let p = parse_readings(&[one(1, Pos::Noun), one(3, Pos::Noun), noun_or_verb(2)]);
        assert_eq!(p.ambiguous[0].survived, NV);
        assert_eq!(p.slot_dropped, 0);
    }

    /// D-LXC-14: an adjective that may precede its verb ("weil es möglich
    /// ist", "un projet important est") does not license away a verb
    /// reading; in English word order it does.
    #[test]
    fn adjective_licensing_follows_the_typology() {
        // "the old ist N|V": verb reading after an adjective.
        let toks = [
            one(9, Pos::Det),
            one(1, Pos::Noun),
            one(5, Pos::Adj),
            noun_or_verb(2),
        ];
        let english = parse_readings_with(&toks, Typology::ENGLISH);
        assert_eq!(english.ambiguous[0].survived, PosSet::single(Pos::Noun));
        let verb_after_adjective = Typology {
            adjective_opens_nominal: false,
            ..Typology::ENGLISH
        };
        let german = parse_readings_with(&toks, verb_after_adjective);
        assert_eq!(german.ambiguous[0].survived, NV);
        // A determiner still licenses away the verb in every typology.
        let toks = [one(9, Pos::Det), noun_or_verb(2)];
        let p = parse_readings_with(&toks, verb_after_adjective);
        assert_eq!(p.ambiguous[0].survived, PosSet::single(Pos::Noun));
    }

    /// Bugbot (#1321): a quantifier floating after the subject ("the men
    /// **all** eat", "they **both** agree") opens no object group: the clause
    /// still waits for its verb, so licensing keeps the homograph's verb
    /// reading. A determiner after the verb ("men saw the record") or at the
    /// clause start ("the record") still drops it.
    #[test]
    fn a_floating_quantifier_keeps_the_verb_slot_open() {
        let p = parse_readings(&[
            one(9, Pos::Det),
            one(1, Pos::Noun),
            one(8, Pos::Det),
            noun_or_verb(2),
        ]);
        assert!(p.ambiguous[0].survived.contains(Pos::Verb));
        assert_eq!(p.unlicensed_dropped, 0);
        let p = parse_readings(&[
            one(1, Pos::Noun),
            one(4, Pos::Verb),
            one(9, Pos::Det),
            noun_or_verb(2),
        ]);
        assert_eq!(p.ambiguous[0].survived, PosSet::single(Pos::Noun));
        let p = parse_readings(&[one(9, Pos::Det), noun_or_verb(2)]);
        assert_eq!(p.ambiguous[0].survived, PosSet::single(Pos::Noun));
        // CodeRabbit (#1321): "he gave him the charge". "him" is an object
        // carried into the subject slot, so "the" opens the next object's
        // group and "charge" loses its verb reading.
        let p = parse_readings(&[
            one(1, Pos::Noun),
            one(4, Pos::Verb),
            one(3, Pos::Noun),
            one(9, Pos::Det),
            noun_or_verb(2),
        ]);
        assert_eq!(p.ambiguous[0].survived, PosSet::single(Pos::Noun));
        // "man when the land": a determiner after a non-noun opens a group.
        let p = parse_readings(&[
            one(1, Pos::Noun),
            one(7, Pos::Other),
            one(9, Pos::Det),
            noun_or_verb(2),
        ]);
        assert_eq!(p.ambiguous[0].survived, PosSet::single(Pos::Noun));
    }

    const ADJ_ADV: PosSet = PosSet::single(Pos::Adj).with(Pos::Adv);

    /// D-LXC-14: each attribute clause matches its own position, and none
    /// matches after a verb.
    #[test]
    fn each_attribute_clause_reads_its_position() {
        let t = Typology::ENGLISH;
        let det = PosSet::single(Pos::Det);
        let noun = PosSet::single(Pos::Noun);
        let verb = PosSet::single(Pos::Verb);
        let adj = PosSet::single(Pos::Adj);
        let e = PosSet::EMPTY;
        // "the [only] son"
        assert_eq!(
            attribute_rule(det, ADJ_ADV, noun, t),
            Some(AttributeRule::AfterDeterminer)
        );
        // "was [very] good"
        assert_eq!(
            attribute_rule(verb, ADJ_ADV, adj, t),
            Some(AttributeRule::BeforeAdjective)
        );
        // "I [just] need"
        assert_eq!(
            attribute_rule(noun, ADJ_ADV, verb, t),
            Some(AttributeRule::BetweenSubjectAndVerb)
        );
        // "of [high] rate" (`of` is Other)
        let other = PosSet::single(Pos::Other);
        assert_eq!(
            attribute_rule(other, ADJ_ADV, noun, t),
            Some(AttributeRule::NextToNoun)
        );
        // Silence: "was [good] ." and "ran [fast] ." — after a verb, nothing.
        assert_eq!(attribute_rule(verb, ADJ_ADV, e, t), None);
        // A token without both readings is never touched.
        assert_eq!(attribute_rule(det, adj, noun, t), None);
        // Post-nominal order: "homme [bon] ." is attributive only where the
        // typology allows an adjective after its noun.
        let both = Typology {
            adjective: AdjectiveOrder::Both,
            ..t
        };
        assert_eq!(
            attribute_rule(noun, ADJ_ADV, e, both),
            Some(AttributeRule::NextToNoun)
        );
        assert_eq!(attribute_rule(noun, ADJ_ADV, e, t), None);
    }

    /// D-LXC-14: as shipped no clause narrows (each was measured below
    /// frequency on gold tags); an enabled clause narrows exactly its token.
    #[test]
    fn attribute_clauses_narrow_only_when_enabled() {
        let toks = [
            one(9, Pos::Det),
            Reading::new(5, ADJ_ADV),
            one(1, Pos::Noun),
        ];
        let p = parse_readings(&toks);
        assert_eq!(p.ambiguous[0].survived, ADJ_ADV);
        assert_eq!(p.attribute_narrowed, 0);
        let enabled = Typology {
            attribute_rules: &[AttributeRule::AfterDeterminer],
            ..Typology::ENGLISH
        };
        let p = parse_readings_with(&toks, enabled);
        assert_eq!(p.ambiguous[0].entered, ADJ_ADV);
        assert_eq!(p.ambiguous[0].survived, PosSet::single(Pos::Adj));
        assert_eq!(p.attribute_narrowed, 1);
        // A clause that is not enabled does not fire: "was [very] good".
        let toks = [
            one(4, Pos::Verb),
            Reading::new(5, ADJ_ADV),
            one(6, Pos::Adj),
        ];
        let p = parse_readings_with(&toks, enabled);
        assert_eq!(p.ambiguous[0].survived, ADJ_ADV);
        // A stop is a sentence edge: "homme . [bon] ," does not read
        // `homme`, so the post-nominal clause cannot fire across it — while
        // without the stop it does.
        let both = Typology {
            adjective: AdjectiveOrder::Both,
            attribute_rules: &AttributeRule::ALL,
            ..Typology::ENGLISH
        };
        let across = [
            one(1, Pos::Noun),
            Reading::stop(),
            Reading::new(5, ADJ_ADV),
            one(7, Pos::Other),
        ];
        let p = parse_readings_with(&across, both);
        assert_eq!(p.ambiguous[0].survived, ADJ_ADV);
        let within = [
            one(1, Pos::Noun),
            Reading::new(5, ADJ_ADV),
            one(7, Pos::Other),
        ];
        let p = parse_readings_with(&within, both);
        assert_eq!(p.ambiguous[0].survived, PosSet::single(Pos::Adj));
    }

    /// The clause rule, when switched on: "the men record deeds" has no
    /// verb unless `record` is one, so its noun reading goes. Off (the
    /// default) both stay. With no path predicated it drops nothing.
    #[test]
    fn the_clause_rule_finds_the_only_possible_verb_when_enabled() {
        let toks = [
            one(9, Pos::Det),
            one(1, Pos::Noun),
            noun_or_verb(2),
            one(3, Pos::Noun),
        ];
        let off = parse_readings(&toks);
        assert_eq!(off.ambiguous[0].survived, NV);
        assert_eq!(off.unpredicated_dropped, 0);
        let on = Typology {
            predicate_required: true,
            ..Typology::ENGLISH
        };
        let p = parse_readings_with(&toks, on);
        assert_eq!(p.ambiguous[0].survived, PosSet::single(Pos::Verb));
        assert_eq!(p.certain, vec![Spo::new(1, 2, 3)]);
        assert_eq!(p.unpredicated_dropped, 1);
        // No path has a predicate: a sentence-initial homograph takes no
        // slot either way, so nothing is dropped.
        let p = parse_readings_with(&[noun_or_verb(2), one(3, Pos::Det)], on);
        assert_eq!(p.ambiguous[0].survived, NV);
        assert_eq!(p.unpredicated_dropped, 0);
    }

    /// The question test as a position: "dog bit man . cat" — before `dog`
    /// nothing is answered, before `bit` who?, before `man` who? + does what?,
    /// after the triple all three, and a stop starts over.
    #[test]
    fn answered_questions_follow_the_clause() {
        let toks = [n(1), v(2), n(3), Tagged::new(0, Pos::Stop), n(4)];
        let m = answered_questions(&toks);
        use CausalMask::{None as No, S, SP, SPO};
        assert_eq!(m, vec![No, S, SP, SPO, No]);
    }

    /// T4: a single-reading word is never rejected, even where the licensing
    /// rule would forbid it — a word is not dropped for having one reading.
    #[test]
    fn a_single_reading_is_never_rejected() {
        // "dog the bit man": a lone verb right after a determiner.
        let toks = [
            one(1, Pos::Noun),
            one(9, Pos::Det),
            one(2, Pos::Verb),
            one(3, Pos::Noun),
        ];
        let p = parse_readings(&toks);
        assert_eq!(p.certain, vec![Spo::new(1, 2, 3)]);
        assert_eq!(p.peak_configs, 1);
    }

    /// T5: paths that reach the same configuration merge, and the merged
    /// configuration remembers both readings.
    #[test]
    fn equivalent_configurations_merge() {
        // det|adj: both are skipped and both open a nominal group, so the
        // two paths end in the same registers with the same triples.
        let det_or_adj = Reading::new(5, PosSet::single(Pos::Det).with(Pos::Adj));
        let p = parse_readings(&[det_or_adj, one(1, Pos::Noun)]);
        assert_eq!(p.peak_configs, 1);
        assert_eq!(
            p.ambiguous[0].survived,
            PosSet::single(Pos::Det).with(Pos::Adj)
        );
    }

    /// T6 at the parser: an unknown token is counted, skipped like `Other`,
    /// and never reported as a reading.
    #[test]
    fn an_unknown_token_is_not_a_reading() {
        let p = parse_readings(&[
            one(1, Pos::Noun),
            Reading::new(7, PosSet::EMPTY),
            one(2, Pos::Verb),
            one(3, Pos::Noun),
        ]);
        assert_eq!(p.unknown, 1);
        assert!(p.ambiguous.is_empty());
        assert_eq!(p.certain, vec![Spo::new(1, 2, 3)]);
    }

    /// A stop closes a sentence: its alternatives and survivors are settled
    /// there, and the next sentence starts from one fresh configuration. The
    /// same homograph is ambiguous in the first sentence and decided by a
    /// determiner in the second.
    #[test]
    fn sentences_are_independent() {
        let toks = [
            one(9, Pos::Det),
            one(1, Pos::Noun),
            noun_or_verb(2),
            one(3, Pos::Noun),
            one(6, Pos::Verb),
            Reading::stop(),
            one(9, Pos::Det),
            noun_or_verb(2),
            one(4, Pos::Verb),
            one(5, Pos::Noun),
        ];
        let p = parse_readings(&toks);
        assert_eq!(p.alternative, vec![Spo::new(1, 2, 3)]);
        assert_eq!(p.certain, vec![Spo::new(2, 4, 5)]);
        assert_eq!(p.ambiguous[1].survived, PosSet::single(Pos::Noun));
    }

    /// T7, the negative twin: with one reading per token the multi-reading
    /// parser is the single-reading parser — same triples, no alternatives,
    /// one configuration. Random streams over every tag, including the
    /// det→verb adjacency the licensing rule targets.
    #[test]
    fn one_reading_per_token_parses_exactly_as_before() {
        let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let mut det_verb_seen = false;
        for _ in 0..2_000 {
            let len = (next() % 24) as usize;
            let toks: Vec<Tagged> = (0..len)
                .map(|_| {
                    let pos = Pos::ALL[(next() % Pos::ALL.len() as u64) as usize];
                    Tagged::new((next() % 6) as WordId, pos)
                })
                .collect();
            det_verb_seen |= toks
                .windows(2)
                .any(|w| w[0].pos == Pos::Det && w[1].pos == Pos::Verb);
            let readings: Vec<Reading> = toks.iter().map(|&t| t.into()).collect();
            let p = parse_readings(&readings);
            assert_eq!(p.certain, parse_to_spo(&toks), "{toks:?}");
            assert!(p.alternative.is_empty());
            assert!(p.ambiguous.is_empty());
            assert!(p.peak_configs <= 1);
        }
        assert!(det_verb_seen, "the sweep must exercise det→verb");
    }

    /// A pathological stream cannot grow configurations without bound: it
    /// is flushed at [`MAX_CONFIGS`], and the flush is reported.
    #[test]
    fn configuration_growth_is_bounded_and_reported() {
        // Alternating noun|verb with distinct ids: every path emits different
        // triples, so nothing merges.
        let toks: Vec<Reading> = (0..40).map(noun_or_verb).collect();
        let p = parse_readings(&toks);
        assert!(p.overflow_flushes > 0);
        assert!(p.peak_configs <= 2 * MAX_CONFIGS);
    }
}
