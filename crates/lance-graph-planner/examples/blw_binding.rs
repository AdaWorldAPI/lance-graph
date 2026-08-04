//! `blw_binding` — **D-BLW-2 REBUILD**: the binding rules ARE the instrument.
//!
//! This is a NEW harness, written against the measured KILL recorded in
//! `.claude/plans/cycle-loop-closure-driver-v1.md` §12.7. It does not modify,
//! import from, or resurrect its deleted predecessor (`blw_texture.rs`, removed
//! at `cbca9e6`), and it does not touch `blw_tenant.rs` (D-BLW-1, shipped) or
//! any `src/` file.
//!
//! # The KILL this file exists to repair (§12.7, read it before editing here)
//!
//! The predecessor adopted the right carrier — [`CausalWitnessFacet`], 24 signed
//! `i4` loci — and then **wrote three of them**: `Antecedent` (every stance),
//! `Quorum` (Hegel only), `Modal` (Kant only). Only `Antecedent` was shared, so
//! [`CausalWitnessFacet::agreement_count`] was **capped at 1 of 24 before a
//! single verse was read**, and 21 loci reported `0.0000` always. Measured means
//! were 0.0015–0.0825. §12.7's verdict, verbatim: *"The register was necessary
//! and is not sufficient — the binding rules ARE the instrument."*
//!
//! So the repair is **not** a new type and **not** a new statistic. It is a
//! rewrite of the rules by which loci are bound:
//!
//! 1. **One shared 9-locus menu**, computed by ONE function ([`mint`]) with
//!    exactly **9** `Locus` write sites. All four stances write the SAME nine
//!    loci; what differs between stances is the **focus selection** — which
//!    belief at a verse the stance reasons from. Agreement therefore means
//!    *"two stances that chose different focuses by different rules point at the
//!    same event on this dimension"*, which is a texture statement. The
//!    predecessor's private-locus design made that structurally impossible.
//! 2. **No private loci at all.** The ceiling is stated below and is 9, not 1.
//! 3. **The two headline quantities are reported separately and are never
//!    averaged** (§12.7's explicit instruction) — see the LEVER / TORQUE
//!    sections.
//!
//! # WRITE-SITE COUNT (stated before anything else, per the D-BLW-2 brief)
//!
//! **9 write sites of 24 slots ⇒ `agreement_count` ceiling = 9.**
//!
//! The nine, all in [`mint`], all shared by all four stances:
//! `SMeaning`(4) · `PMeaning`(5) · `OMeaning`(6) · `SupportedBy`(9) ·
//! `Supports`(10) · `QualiaReference`(12) · `MeaningLevel`(13) · `Quorum`(14) ·
//! `Contradiction`(15).
//!
//! The other **15 slots read `0` BY CONSTRUCTION** — this harness never writes
//! them, and that is a property of the source, not a measurement:
//!
//! | slot | why unwritten |
//! |---|---|
//! | 0–3 `Temporal`/`Kausal`/`Modal`/`Lokal` | the TEKAMOLO frame needs a TEKAMOLO parse; `stance::stream` is a cue-driven clause machine and produces none. Binding them would be fabrication. |
//! | 7 `Antecedent` | **deliberately retired from this instrument.** `stance::stream` normalizes EVERY nominative/accusative personal pronoun to one scene referent (`stance.rs:208-216`) — the coreference this locus names is *collapsed by design* upstream, so any binding here would invent a resolution the machine does not have. It is also the exact axis that carried the predecessor's entire (capped) signal. Reserved for the §12.6 A3′ cross-language falsifier, which needs morphology this repo does not have. |
//! | 8 `BasinAnchor` | needs an AriGraph `part_of:is_a` rail; none is wired to this corpus. |
//! | 11 `RunbookEvidence` | needs the 34-recipe dispatch; not consumed here. |
//! | 16–23 | reserved-empty by the register itself (`NAMED_LOCI = 16`). |
//!
//! Every OTHER zero this program prints is a **measurement**, and the report
//! distinguishes the two silence kinds per locus ([`Silence`]).
//!
//! # The mechanics being encoded (§12.3c / §12.6 A4) — two quantities, never one
//!
//! * **Sarcasm is TORQUE** — a real lever arm through a large angle: the
//!   binding is intact, but `QualiaReference`(12) points FAR from the local
//!   meaning loci `SMeaning`/`PMeaning`/`OMeaning`(4/5/6). Said and meant point
//!   apart. Reported as [`torque`] = `max |at(12) − at(l)|` over the bound
//!   `l ∈ {4,5,6}`; **undefined** (never `0`) when 12 or all of 4–6 are unbound.
//! * **Nihilism is a COLLAPSED LEVER ARM** — `Supports`(10) / `SupportedBy`(9)
//!   collapse, so nothing grounds anything and no torque is possible at any
//!   angle. Reported as [`Lever`], a four-way classification of *whether the
//!   evidence loci are bound at all*.
//!
//! These are (a) *is the evidence topology bound* and (b) *how far is the
//! qualia pointer from the meaning pointers*. **They are printed in two
//! separate tables and are never combined into a scalar.** A single mean over
//! both is the §12.3a defect (multi-axis collapsed to one coincidence number)
//! in its third costume.
//!
//! # The ±8 window is REAL — out-of-window reads UNBOUND, never saturated
//!
//! `causal_witness.rs`'s own header: a nibble is *"a signed offset ∈ [−8, +7]
//! naming WHERE in the ±8 `temporal.rs` Markov window that awareness
//! dimension's filler sits."* The predecessor **clamped** a five-digit verse
//! delta to `+7`, which manufactures agreement: two unrelated distant pointers
//! both land on `+7` and `agrees_at` reports convergence. This harness instead
//! leaves an out-of-window target **UNBOUND** and counts it as
//! [`Silence::OutOfWindow`], so the silence is disclosed rather than discovered.
//! Offset `0` is likewise never written — the register reads `0` as *unbound*,
//! so a same-verse target is a silence case, stated here in advance (that exact
//! confusion was a real defect in the predecessor's `Modal` rule).
//!
//! # Pre-registered falsifiers (§12.6) — indices verified against the corpus
//!
//! The verdict rule is fixed here, in source, BEFORE the run: an anchor pair
//! **separates** iff its structural distance strictly exceeds the LARGEST
//! distance measured on the three control pairs. The threshold is therefore the
//! corpus's own baseline churn, not a constant chosen by the author, and it
//! cannot be tuned toward a desired answer without changing the controls.
//!
//! * **A1 (must-have can-fire):** `55` (Gen 2:25, *"both naked … not ashamed"*)
//!   vs `62` (Gen 3:7, *"the eyes of them both were opened, and they knew that
//!   they were naked"*). The FACT is identical; only KNOWING changes. A lexical
//!   or polarity instrument scores them SIMILAR. **If this instrument cannot
//!   separate 55 from 62, that is a KILL of the instrument and is reported as
//!   one — the weights are NOT adjusted until they separate.**
//! * **A2:** `60` (Gen 3:5, the serpent's promise) vs `77` (Gen 3:22, God's
//!   confirmation). Proposition, lexis AND polarity are all held constant; only
//!   topology can separate them.
//! * **A3:** `60 → 62`, promise (*"eyes shall be opened"*) → fulfilment
//!   (*"eyes were opened"*): two verses apart, near-identical predicate.
//! * **Controls (can-stay-silent, mandatory):** `(96,97)`, `(98,99)`,
//!   `(100,101)` — adjacent Genesis-4 genealogy, no awareness inversion.
//!   Deliberately stopping short of Lamech's boast at 102–103, which carries
//!   real qualia. If A1's distance looks like these, the separation is noise and
//!   this program says so.
//!
//! # Claim ceiling (§12.4), binding on every line printed
//!
//! Structure and overlap only. This program may say two readings **differ**. It
//! may NOT say one is better, truer, more complete, more valid or more
//! accurate. No p-values (`jc::stats` p-values are classical independent-sample
//! values; verses within a book are domain-correlated — `I-NOISE-FLOOR-JIRAK`).
//! No fusion claim. No aphorism the measurement does not support.
//!
//! # Not compiled, not run
//!
//! The session that wrote this file is edit-only and cannot invoke `cargo`; the
//! orchestrator compiles and runs centrally. Nothing below is claimed green,
//! passing, or measured. Every number this program prints is computed at run
//! time from the real corpus; none is predicted here.
//!
//! # Usage
//!
//! ```text
//! cargo run -p lance-graph-planner --example blw_binding                 # 2,000 verses
//! cargo run -p lance-graph-planner --example blw_binding -- <tsv> 4000   # bounded
//! cargo run -p lance-graph-planner --example blw_binding -- <tsv> all    # whole book (slow)
//! ```
//!
//! The corpus is **bounded at 2,000 by default**: §12.7 measured 2,000 verses ≈
//! 1 s and the full 31,102 exceeding a 10-minute budget (`stance::stream` calls
//! `staunen(Snapshot::of(arena, 0.0))` once per rung lift, each an O(arena)
//! scan). All anchors (55/60/62/77) and controls (96–101) live in the first
//! ~110 rows.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use std::collections::{BTreeMap, HashMap, HashSet};

use lance_graph_contract::causal_witness::{CausalWitnessFacet, Locus, WITNESS_LOCI};
use lance_graph_planner::nars::stance::{stance_panel, stream, FlipKind, Interner, ReadOut};
use lance_graph_planner::nars::{BeliefArena, CStmt, Copula};

// ── constants, all pre-registered ───────────────────────────────────────────

/// Default corpus path (`index\ttext` per line, 0-based row index as the verse
/// label — `split_once('\t')` on a two-column file is not a parser).
const DEFAULT_TSV: &str = "/tmp/kjv_verses.tsv";

/// Default corpus bound in verses (§12.7's measured-good size).
const DEFAULT_VERSE_LIMIT: usize = 2_000;

/// Fixed-verse-set prefix for the horizon control (§12.3b), reused unchanged.
const HORIZON_K: usize = 1_000;

/// The register's window, inclusive: an offset outside it is UNBOUND.
const WINDOW_LO: isize = -8;
/// The register's window, inclusive (upper end).
const WINDOW_HI: isize = 7;

/// A stance whose facet fires on more than this fraction of verses is
/// DEGENERATE by prevalence (§12.7: an 88 % firing rate is the same tell as the
/// 99.61 % that killed §12.3a″) — excluded from the pairwise table and printed.
const DEGENERATE_RATE: f64 = 0.90;

/// A locus bound on more than this fraction of a stance's focused verses is
/// NEAR-CONSTANT: it carries about as little information as one that never
/// fires, so agreement is additionally reported with such loci masked out.
const NEAR_CONSTANT_RATE: f64 = 0.90;

/// Size of the shared binding menu — the write-site count, named so the
/// ceiling is a compile-time fact rather than a comment.
const MENU_LEN: usize = 9;

/// The shared binding menu — the ONLY loci this harness writes. Nine of 24.
const MENU: [Locus; MENU_LEN] = [
    Locus::SMeaning,
    Locus::PMeaning,
    Locus::OMeaning,
    Locus::SupportedBy,
    Locus::Supports,
    Locus::QualiaReference,
    Locus::MeaningLevel,
    Locus::Quorum,
    Locus::Contradiction,
];

/// A1 "before": Gen 2:25 — naked, not ashamed.
const A1_BEFORE: usize = 55;
/// A1 "after": Gen 3:7 — eyes opened, they KNEW that they were naked.
const A1_AFTER: usize = 62;
/// A2 "promise": Gen 3:5 — the serpent's *"your eyes shall be opened"*.
const A2_PROMISE: usize = 60;
/// A2 "confirmation": Gen 3:22 — God's *"become as one of us, to know"*.
const A2_CONFIRM: usize = 77;

/// Control pairs: adjacent Genesis-4 genealogy, no awareness inversion.
const CONTROLS: [(usize, usize); 3] = [(96, 97), (98, 99), (100, 101)];

// ── silence bookkeeping ─────────────────────────────────────────────────────

/// Why a locus reads `0` on one facet. The distinction is the whole point of
/// §12.7's *"by construction vs by measurement"* correction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Silence {
    /// The locus was written with a nonzero in-window offset.
    Bound,
    /// The rule does not apply to this focus at all (a stated precondition
    /// failed — e.g. `PMeaning` on a non-`Rel` focus). **By construction.**
    NotApplicable,
    /// The rule applied and found no target anywhere. **By measurement.**
    NoCandidate,
    /// A target exists but lies outside `[−8, +7]`, so the register cannot name
    /// it. **By measurement** — and disclosed rather than saturated.
    OutOfWindow,
}

/// Per-locus silence tally over one stance's whole run.
#[derive(Debug, Clone, Copy, Default)]
struct LocusStat {
    /// Facets where this locus bound.
    bound: usize,
    /// Facets where the rule did not apply (by construction).
    not_applicable: usize,
    /// Facets where the rule applied and found nothing (by measurement).
    no_candidate: usize,
    /// Facets where the only targets were outside the ±8 window.
    out_of_window: usize,
}

impl LocusStat {
    /// Fold one observation.
    fn add(&mut self, s: Silence) {
        match s {
            Silence::Bound => self.bound += 1,
            Silence::NotApplicable => self.not_applicable += 1,
            Silence::NoCandidate => self.no_candidate += 1,
            Silence::OutOfWindow => self.out_of_window += 1,
        }
    }
    /// Bind rate over all facets this stance minted (focused verses).
    fn rate(self, focused: usize) -> f64 {
        if focused == 0 {
            0.0
        } else {
            self.bound as f64 / focused as f64
        }
    }
}

// ── window-respecting pointer selection ─────────────────────────────────────

/// Nearest in-window position to `vi` among a SORTED, deduplicated position
/// vector, excluding `vi` itself. Ties on `|delta|` break toward the earlier
/// position (deterministic and disclosed, never hidden).
///
/// Returns [`Silence::OutOfWindow`] when candidates exist but none is inside
/// `[vi−8, vi+7]`, and [`Silence::NoCandidate`] when none exists at all.
fn pick(sorted: &[usize], vi: usize) -> (Option<i8>, Silence) {
    let lo = (vi as isize + WINDOW_LO).max(0) as usize;
    let hi = vi.saturating_add(WINDOW_HI as usize);
    let start = sorted.partition_point(|&p| p < lo);
    let mut best: Option<usize> = None;
    for &p in &sorted[start..] {
        if p > hi {
            break;
        }
        if p == vi {
            continue;
        }
        best = Some(match best {
            None => p,
            Some(b) if (p.abs_diff(vi), p) < (b.abs_diff(vi), b) => p,
            Some(b) => b,
        });
    }
    match best {
        Some(p) => (Some((p as isize - vi as isize) as i8), Silence::Bound),
        None if sorted.iter().any(|&p| p != vi) => (None, Silence::OutOfWindow),
        None => (None, Silence::NoCandidate),
    }
}

/// [`pick`] restricted to STRICTLY EARLIER positions — the nearest in-window
/// antecedent.
///
/// Used only by `QualiaReference`, whose register gloss is *"the event that
/// **set** my current texture"*: a texture-setting event is in the past by
/// definition. Keeping it backward-only is also what stops locus 12 from
/// degenerating into a copy of `Quorum` (which is nearest-either-direction over
/// the same occurrence list) — two loci that always agree would inflate
/// `agreement_count` with a redundancy rather than a convergence.
fn pick_backward(sorted: &[usize], vi: usize) -> (Option<i8>, Silence) {
    let earlier: Vec<usize> = sorted.iter().copied().filter(|&p| p < vi).collect();
    match pick(&earlier, vi) {
        // `pick`'s NoCandidate here means "nothing earlier"; distinguish it from
        // "nothing at all" so the report does not overstate absence.
        (None, Silence::NoCandidate) if sorted.iter().any(|&p| p != vi) => {
            (None, Silence::OutOfWindow)
        }
        other => other,
    }
}

/// [`pick`] over a `(position, negated)` occurrence list filtered to one
/// polarity. Same window and tie-break rules.
fn pick_polar(occs: &[(usize, bool)], vi: usize, want: bool) -> (Option<i8>, Silence) {
    let filtered: Vec<usize> = occs
        .iter()
        .filter(|&&(_, neg)| neg == want)
        .map(|&(p, _)| p)
        .collect();
    pick(&filtered, vi)
}

/// [`pick`] over an ad-hoc candidate list that was ALREADY window-restricted by
/// its caller (the two chain loci scan the window directly, because a global
/// chain search over a subject like the normalized pronoun referent is
/// quadratic and — since only in-window targets can ever bind — pointless).
///
/// **Disclosed consequence:** these two loci can report [`Silence::NoCandidate`]
/// but never [`Silence::OutOfWindow`]; their "no in-window chain link" and
/// "no chain link at all" cases are merged. Stated here rather than left for a
/// reader to infer from a suspiciously empty column.
fn pick_local(cands: &mut Vec<usize>, vi: usize) -> (Option<i8>, Silence) {
    cands.sort_unstable();
    cands.dedup();
    let (off, sil) = pick(cands, vi);
    // A local list is already window-bounded, so `OutOfWindow` is unreachable
    // here; normalize it so the report cannot imply a distinction it lacks.
    match sil {
        Silence::OutOfWindow => (off, Silence::NoCandidate),
        other => (off, other),
    }
}

// ── corpus ──────────────────────────────────────────────────────────────────

/// Read `index\ttext` rows. The verse LABEL is the row's own first column; the
/// verse POSITION is its 0-based index in the loaded slice, which is what every
/// binding rule uses.
fn load_tsv(path: &str) -> std::io::Result<Vec<(String, String)>> {
    let text = std::fs::read_to_string(path)?;
    Ok(text
        .lines()
        .filter_map(|l| l.split_once('\t'))
        .map(|(i, t)| (i.to_string(), t.to_string()))
        .collect())
}

/// Corpus guard: the pre-registered anchor INDICES must land on the
/// pre-registered anchor TEXTS. Without this, a differently-generated TSV would
/// silently shift every anchor and the A1/A2/A3 verdicts would be about
/// unrelated verses while still printing confidently.
fn assert_anchor_texts(verses: &[(String, String)]) {
    let want: [(usize, &[&str]); 4] = [
        (A1_BEFORE, &["naked", "ashamed"]),
        (A1_AFTER, &["eyes", "opened", "naked"]),
        (A2_PROMISE, &["eyes", "opened", "gods"]),
        (A2_CONFIRM, &["one of us", "know good and evil"]),
    ];
    for (idx, needles) in want {
        let text = verses
            .get(idx)
            .map(|(_, t)| t.to_ascii_lowercase())
            .unwrap_or_else(|| panic!("anchor index {idx} is not in the loaded corpus"));
        for n in needles {
            assert!(
                text.contains(n),
                "anchor index {idx} does not contain {n:?} — the corpus does not match \
                 the pre-registered indices; every anchor verdict would be about the \
                 wrong verse"
            );
        }
    }
}

// ── the built context ───────────────────────────────────────────────────────

/// Everything the binding rules read, built once per horizon.
///
/// Position vectors are sorted and deduplicated at build time so [`pick`] can
/// binary-search them; nothing here is recomputed per verse.
struct Ctx {
    /// Number of verses this context was built over.
    n: usize,
    /// Observed `(statement, negated)` per verse position, in emission order.
    stmts_at: Vec<Vec<(CStmt, bool)>>,
    /// `out.lifts` indices per verse position.
    lifts_at: Vec<Vec<usize>>,
    /// Concept → sorted positions where it is a SUBJECT of an observed statement.
    subj_at: HashMap<u16, Vec<usize>>,
    /// Concept → sorted positions where it is the PREDICATE TERM of one.
    obj_at: HashMap<u16, Vec<usize>>,
    /// Statement → sorted `(position, negated)` observations.
    stmt_at: HashMap<CStmt, Vec<(usize, bool)>>,
    /// Epistemic verb id → sorted positions where it licensed a rung lift.
    verb_at: HashMap<u16, Vec<usize>>,
    /// Every statement in the arena AFTER `close_transitive` — the presence set
    /// the two chain rules test against.
    present: HashSet<CStmt>,
    /// The stream's own read-out (provenance + lifts + causal edges).
    out: ReadOut,
    /// Hegel's ranked set (`contradiction_ranking`'s `> 0.05` filter), as a map.
    hegel: HashMap<CStmt, f32>,
    /// Nietzsche's genealogy partition, as a map.
    nietzsche: HashMap<CStmt, FlipKind>,
    /// Wittgenstein's concept → distinct-language-game count.
    games: HashMap<u16, usize>,
}

/// Run the REAL, UNMODIFIED `stance::stream` + `stance_panel` over `verses` and
/// index what the binding rules need.
///
/// Mirrors `probe_eyes_opened.rs::report()`'s pass-1 half: one
/// `stream(..., pass2 = false)` then `close_transitive(64)`. The pass-2 re-read
/// is skipped: `stream` resets its stamp counter at entry (`stance.rs:169`), so
/// a second pass over the same verses replays identical stamps and the arena's
/// S4 overlap guard routes every re-observation to CHOICE — it cannot change
/// anything `stance_panel` reads.
fn build(verses: &[(String, String)]) -> Ctx {
    let n = verses.len();
    let mut arena = BeliefArena::new();
    let mut intern = Interner::new();
    let mut out = ReadOut::default();
    stream(verses, &mut arena, &mut intern, &mut out, false);
    arena.close_transitive(64);

    let (hegel_v, nietzsche_v, kant_v, witt_v) = stance_panel(&arena, &intern, &out);
    // `stance_panel` builds its Kant vector by mapping over `out.lifts`; assert
    // the alignment rather than assume it, so a future change to that invariant
    // surfaces here instead of silently mis-indexing.
    assert_eq!(
        kant_v.len(),
        out.lifts.len(),
        "stance_panel's Kant vector must stay 1:1 with out.lifts"
    );

    let pos_of: HashMap<&str, usize> = verses
        .iter()
        .enumerate()
        .map(|(i, (v, _))| (v.as_str(), i))
        .collect();

    let mut stmts_at: Vec<Vec<(CStmt, bool)>> = vec![Vec::new(); n];
    let mut lifts_at: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut subj_at: HashMap<u16, Vec<usize>> = HashMap::new();
    let mut obj_at: HashMap<u16, Vec<usize>> = HashMap::new();
    let mut stmt_at: HashMap<CStmt, Vec<(usize, bool)>> = HashMap::new();
    let mut verb_at: HashMap<u16, Vec<usize>> = HashMap::new();

    for p in &out.provenance {
        let pos = *pos_of
            .get(p.verse.as_str())
            .expect("stream only emits verse labels it was given");
        stmts_at[pos].push((p.stmt, p.negated));
        stmt_at.entry(p.stmt).or_default().push((pos, p.negated));
        subj_at.entry(p.stmt.s).or_default().push(pos);
        obj_at.entry(p.stmt.p).or_default().push(pos);
    }
    for (i, l) in out.lifts.iter().enumerate() {
        let pos = *pos_of
            .get(l.verse.as_str())
            .expect("lift verse label must resolve");
        lifts_at[pos].push(i);
        verb_at.entry(l.verb).or_default().push(pos);
    }
    for v in subj_at.values_mut().chain(obj_at.values_mut()) {
        v.sort_unstable();
        v.dedup();
    }
    for v in verb_at.values_mut() {
        v.sort_unstable();
        v.dedup();
    }
    for v in stmt_at.values_mut() {
        v.sort_unstable();
        v.dedup();
    }

    let present: HashSet<CStmt> = arena.entries().iter().map(|b| b.stmt).collect();

    Ctx {
        n,
        stmts_at,
        lifts_at,
        subj_at,
        obj_at,
        stmt_at,
        verb_at,
        present,
        out,
        hegel: hegel_v.into_iter().collect(),
        nietzsche: nietzsche_v.into_iter().collect(),
        games: witt_v.into_iter().collect(),
    }
}

// ── focus selection: the ONLY thing that differs between stances ────────────

/// One stance's focus at one verse.
///
/// `stmt` is what the stance reasons ABOUT (its S/O terms and its copula);
/// `chain` is the OBSERVED statement whose evidence topology and polarity peers
/// are read. They differ only for Kant, whose focus is the lift's derived
/// meta-belief (`knower Rel(verb) object`, rung 1) while the chain runs through
/// that lift's inner, observed that-clause.
#[derive(Debug, Clone, Copy)]
struct Focus {
    /// The statement the stance reasons about.
    stmt: CStmt,
    /// The observed statement whose chain and polarity peers are read.
    chain: CStmt,
    /// `out.lifts` index when this focus came from a rung lift.
    lift: Option<usize>,
}

/// The four stances, as focus-selection rules over one unchanged arena read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stance {
    /// Aufhebung: focus = the verse's most deeply preserved contradiction.
    Hegel,
    /// Genealogy: focus = the verse's first statement with a legible flip.
    Nietzsche,
    /// Critique: focus = the verse's first rung lift (the a-priori seam).
    Kant,
    /// Meaning as use: focus = the verse's most language-game-connected statement.
    Wittgenstein,
}

impl Stance {
    /// All four, in report order.
    const ALL: [Stance; 4] = [
        Stance::Hegel,
        Stance::Nietzsche,
        Stance::Kant,
        Stance::Wittgenstein,
    ];

    /// Display name.
    fn name(self) -> &'static str {
        match self {
            Stance::Hegel => "Hegel",
            Stance::Nietzsche => "Nietzsche",
            Stance::Kant => "Kant",
            Stance::Wittgenstein => "Wittgenstein",
        }
    }

    /// Select this stance's focus at verse position `vi`, or `None` when the
    /// stance has nothing to read there.
    ///
    /// Silence conditions, stated per stance (not discovered post hoc):
    /// * **Hegel** — no statement observed at this verse survives
    ///   `contradiction_ranking`'s `> 0.05` filter. Ties on depth break toward
    ///   the earliest emission at the verse.
    /// * **Nietzsche** — no statement observed here carries a legible
    ///   `FlipKind` (an endpoint pair `(negated_first, negated_last)` that is
    ///   actually a flip).
    /// * **Kant** — this verse carries no rung lift, OR it carries one whose
    ///   inner that-clause emission cannot be located among the verse's own
    ///   provenance (the lift is then unreadable as a chain and is skipped
    ///   rather than guessed at).
    /// * **Wittgenstein** — this verse emits nothing at all through the clause
    ///   machine (e.g. a genealogy list the catalogues never arm a predicate
    ///   on), or none of its terms has a recorded language game.
    fn focus(self, vi: usize, ctx: &Ctx) -> Option<Focus> {
        let here = &ctx.stmts_at[vi];
        match self {
            Stance::Hegel => here
                .iter()
                .filter_map(|&(s, _)| ctx.hegel.get(&s).map(|&d| (s, d)))
                .enumerate()
                // max depth; earliest emission wins a tie (Reverse on the index)
                .max_by(|a, b| a.1 .1.total_cmp(&b.1 .1).then_with(|| b.0.cmp(&a.0)))
                .map(|(_, (s, _))| Focus {
                    stmt: s,
                    chain: s,
                    lift: None,
                }),
            Stance::Nietzsche => here
                .iter()
                .find(|&&(s, _)| ctx.nietzsche.contains_key(&s))
                .map(|&(s, _)| Focus {
                    stmt: s,
                    chain: s,
                    lift: None,
                }),
            Stance::Kant => {
                let &li = ctx.lifts_at[vi].first()?;
                let l = &ctx.out.lifts[li];
                // The inner that-clause emission: the verse's first observed
                // statement whose predicate term IS the lift's object. This is
                // how `stance::stream` builds the lift (the emission it consumes
                // is the inner statement), recovered here because `RungLift`
                // does not carry the inner subject. Disclosed tie-break: first
                // such emission at the verse.
                let inner = here
                    .iter()
                    .find(|&&(s, _)| s.p == l.object)
                    .map(|&(s, _)| s)?;
                Some(Focus {
                    stmt: CStmt {
                        s: l.knower,
                        cop: Copula::Rel(l.verb),
                        p: l.object,
                    },
                    chain: inner,
                    lift: Some(li),
                })
            }
            Stance::Wittgenstein => here
                .iter()
                .filter_map(|&(s, _)| {
                    let g = ctx.games.get(&s.s).copied().unwrap_or(0)
                        + ctx.games.get(&s.p).copied().unwrap_or(0);
                    (g > 0).then_some((s, g))
                })
                // most connected; smallest (s, p) breaks the tie deterministically
                .max_by_key(|&(s, g)| (g, std::cmp::Reverse((s.s, s.p))))
                .map(|(s, _)| Focus {
                    stmt: s,
                    chain: s,
                    lift: None,
                }),
        }
    }
}

// ── THE binding rules: nine write sites, one function, all stances ──────────

/// Mint one facet from one focus. **This is the instrument.**
///
/// Nine loci, nine rules, nine `Locus` write sites — every one shared by all
/// four stances, so [`CausalWitnessFacet::agreement_count`] has a ceiling of 9
/// rather than the predecessor's 1. Each rule states what it points at and what
/// makes it silent; every offset is a genuine pointer to another verse in the
/// register's ±8 window, never an encoded magnitude ("Loci, not magnitudes").
///
/// | locus | points at | silent when |
/// |---|---|---|
/// | 4 `SMeaning` | nearest other verse where the focus SUBJECT is a subject | subject appears nowhere else in window |
/// | 5 `PMeaning` | nearest other verse where the SAME relational verb licensed a lift | the focus copula is not `Rel` (**by construction** for `Inh`/`Impl`) |
/// | 6 `OMeaning` | nearest other verse where the focus PREDICATE TERM appears | term appears nowhere else in window |
/// | 9 `SupportedBy` | nearest in-window verse observing a link `A→B` such that `B→C` also exists, i.e. the chain BENEATH the focus | copula does not transit, or no such link in window |
/// | 10 `Supports` | nearest in-window verse observing a link `C→Z` such that the conclusion `A→Z` exists, i.e. the chain ABOVE | copula does not transit, or no such link in window |
/// | 12 `QualiaReference` | nearest EARLIER verse where THIS LIFT's inner that-clause was already observed — *the event that set the texture* (backward-only; see [`pick_backward`]) | no lift at this verse (**by construction** for the three non-lift focuses) |
/// | 13 `MeaningLevel` | nearest other verse where the SAME epistemic verb licensed a lift — the peer at my rung | no lift at this verse |
/// | 14 `Quorum` | nearest other verse re-observing the chain statement with the SAME polarity — the pooling peer | no same-polarity peer in window |
/// | 15 `Contradiction` | nearest other verse re-observing it with the OPPOSITE polarity — the preserved dissenting peer | no opposite-polarity peer in window |
///
/// Returns the facet and the per-menu-locus [`Silence`] vector, positionally
/// aligned to [`MENU`].
fn mint(vi: usize, focus: Focus, ctx: &Ctx) -> (CausalWitnessFacet, [Silence; MENU_LEN]) {
    let mut f = CausalWitnessFacet::ZERO;
    let mut sil = [Silence::NotApplicable; MENU_LEN];
    let empty: Vec<usize> = Vec::new();

    // 1 ── SMeaning(4): the subject's own grounding event.
    let (o, s) = pick(ctx.subj_at.get(&focus.stmt.s).unwrap_or(&empty), vi);
    if let Some(o) = o {
        f = f.with(Locus::SMeaning, o);
    }
    sil[0] = s;

    // 2 ── PMeaning(5): the RELATION's grounding event. Only a `Rel` copula has
    // a lexical predicate at all; `Inh`/`Impl` carry none, so this is
    // NotApplicable by construction for every non-lift focus.
    if let Copula::Rel(verb) = focus.stmt.cop {
        let (o, s) = pick(ctx.verb_at.get(&verb).unwrap_or(&empty), vi);
        if let Some(o) = o {
            f = f.with(Locus::PMeaning, o);
        }
        sil[1] = s;
    }

    // 3 ── OMeaning(6): the predicate TERM's own grounding event.
    let (o, s) = pick(ctx.obj_at.get(&focus.stmt.p).unwrap_or(&empty), vi);
    if let Some(o) = o {
        f = f.with(Locus::OMeaning, o);
    }
    sil[2] = s;

    // 4/5 ── the LEVER ARM: SupportedBy(9) / Supports(10).
    //
    // The arena's own composition rule (`close_transitive`: {A→B, B→C} ⊢ A→C,
    // gated on `Copula::transits`) is what "grounds" means here — this harness
    // re-implements no inference, it reads which links the arena holds.
    // Scanning only the ±8 window is not an approximation: a target outside the
    // window cannot be named by the register at all.
    let (a, cop, c) = (focus.chain.s, focus.chain.cop, focus.chain.p);
    if cop.transits() {
        let lo = (vi as isize + WINDOW_LO).max(0) as usize;
        let hi = vi
            .saturating_add(WINDOW_HI as usize)
            .min(ctx.n.saturating_sub(1));
        let mut below: Vec<usize> = Vec::new();
        let mut above: Vec<usize> = Vec::new();
        for (pos, cell) in ctx.stmts_at[lo..=hi].iter().enumerate() {
            let pos = lo + pos;
            if pos == vi {
                continue;
            }
            for &(st, _) in cell {
                if st.cop != cop {
                    continue;
                }
                // BELOW: this verse observes `A→B`, and `B→C` exists ⇒ the link
                // is the lower half of a derivation ending at my own statement.
                if st.s == a && st.p != a && st.p != c {
                    let mid = CStmt { s: st.p, cop, p: c };
                    if ctx.present.contains(&mid) {
                        below.push(pos);
                    }
                }
                // ABOVE: this verse observes `C→Z`, and the conclusion `A→Z`
                // exists ⇒ my own statement is the lower half of THAT chain.
                if st.s == c && st.p != c && st.p != a {
                    let concl = CStmt { s: a, cop, p: st.p };
                    if ctx.present.contains(&concl) {
                        above.push(pos);
                    }
                }
            }
        }
        let (o, s) = pick_local(&mut below, vi);
        if let Some(o) = o {
            f = f.with(Locus::SupportedBy, o);
        }
        sil[3] = s;
        let (o, s) = pick_local(&mut above, vi);
        if let Some(o) = o {
            f = f.with(Locus::Supports, o);
        }
        sil[4] = s;
    }

    // 6/7 ── QualiaReference(12) and MeaningLevel(13): the lift loci.
    if let Some(li) = focus.lift {
        let l = &ctx.out.lifts[li];
        // The event that set my texture: where this lift's INNER that-clause was
        // ALREADY observed — strictly earlier, because "set" is past tense. For
        // Gen 3:7 that is the statement Gen 2:25 emitted: knowing changed, the
        // fact did not.
        let (o, s) = pick_backward(
            &ctx.stmt_at
                .get(&focus.chain)
                .map(|v| v.iter().map(|&(p, _)| p).collect::<Vec<_>>())
                .unwrap_or_default(),
            vi,
        );
        if let Some(o) = o {
            f = f.with(Locus::QualiaReference, o);
        }
        sil[5] = s;
        // The context defining my level of meaning: the nearest peer lift
        // licensed by the SAME epistemic verb.
        let (o, s) = pick(ctx.verb_at.get(&l.verb).unwrap_or(&empty), vi);
        if let Some(o) = o {
            f = f.with(Locus::MeaningLevel, o);
        }
        sil[6] = s;
    }

    // 8/9 ── Quorum(14) / Contradiction(15): the polarity peers of the chain
    // statement, exactly as the register documents them (agreeing peer /
    // preserved dissenting peer).
    let occs = ctx
        .stmt_at
        .get(&focus.chain)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    let here_neg = ctx.stmts_at[vi]
        .iter()
        .find(|&&(st, _)| st == focus.chain)
        .map(|&(_, neg)| neg);
    if let Some(here_neg) = here_neg {
        let (o, s) = pick_polar(occs, vi, here_neg);
        if let Some(o) = o {
            f = f.with(Locus::Quorum, o);
        }
        sil[7] = s;
        let (o, s) = pick_polar(occs, vi, !here_neg);
        if let Some(o) = o {
            f = f.with(Locus::Contradiction, o);
        }
        sil[8] = s;
    }

    (f, sil)
}

// ── the two headline quantities, kept apart ─────────────────────────────────

/// The evidence lever arm: whether the two evidence loci are bound AT ALL.
/// **Nihilism's signature is [`Lever::Collapsed`]** — nothing grounds anything,
/// so no torque is possible at any angle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Lever {
    /// Both `SupportedBy` and `Supports` bound.
    Both,
    /// Only `SupportedBy` bound.
    BelowOnly,
    /// Only `Supports` bound.
    AboveOnly,
    /// Neither bound — the collapsed lever arm.
    Collapsed,
}

/// Classify a facet's evidence topology. Quantity (a) of the two.
fn lever(f: CausalWitnessFacet) -> Lever {
    match (f.is_bound(Locus::SupportedBy), f.is_bound(Locus::Supports)) {
        (true, true) => Lever::Both,
        (true, false) => Lever::BelowOnly,
        (false, true) => Lever::AboveOnly,
        (false, false) => Lever::Collapsed,
    }
}

/// The angular displacement: how far `QualiaReference` points from the local
/// meaning loci. **Sarcasm's signature is a LARGE torque** with the lever arm
/// intact — said and meant point apart. Quantity (b) of the two.
///
/// `None` when undefined (12 unbound, or none of 4/5/6 bound). Undefined is
/// **never** reported as `0`: a zero torque would mean "said and meant coincide",
/// which is the opposite claim.
fn torque(f: CausalWitnessFacet) -> Option<u8> {
    if !f.is_bound(Locus::QualiaReference) {
        return None;
    }
    let q = i16::from(f.at(Locus::QualiaReference));
    [Locus::SMeaning, Locus::PMeaning, Locus::OMeaning]
        .into_iter()
        .filter(|&l| f.is_bound(l))
        .map(|l| (q - i16::from(f.at(l))).unsigned_abs() as u8)
        .max()
}

// ── structural distance (the anchor/control comparison) ─────────────────────

/// Number of MENU loci on which two facets disagree, `0..=9`. This is the
/// comparison used for A1/A2/A3 and for the controls — the SAME function on
/// both, so an anchor cannot be graded on a scale the controls never faced.
fn menu_distance(a: CausalWitnessFacet, b: CausalWitnessFacet) -> usize {
    MENU.iter().filter(|&&l| a.at(l) != b.at(l)).count()
}

/// The MENU loci on which two facets disagree, by label.
fn differing_loci(a: CausalWitnessFacet, b: CausalWitnessFacet) -> Vec<&'static str> {
    MENU.iter()
        .filter(|&&l| a.at(l) != b.at(l))
        .map(|&l| l.label())
        .collect()
}

// ── per-stance run ──────────────────────────────────────────────────────────

/// One stance's minted facets plus its silence bookkeeping over one horizon.
struct StanceRun {
    /// Which stance produced this.
    stance: Stance,
    /// Facet per verse position (`ZERO` where the stance had no focus).
    facets: Vec<CausalWitnessFacet>,
    /// Per verse: did the stance have a focus at all? A verse with NO focus has
    /// no lever to speak of and must never be pooled in as "collapsed" — that
    /// would report the stance's silence as a nihilism reading.
    has_focus: Vec<bool>,
    /// Verses where the stance HAD a focus (the denominator for bind rates).
    focused: usize,
    /// Verses where the facet bound at least one locus.
    fired: usize,
    /// Per-MENU-locus silence tally.
    stats: [LocusStat; MENU_LEN],
}

impl StanceRun {
    /// Mint every verse for one stance.
    fn run(stance: Stance, ctx: &Ctx) -> Self {
        let mut facets = vec![CausalWitnessFacet::ZERO; ctx.n];
        let mut has_focus = vec![false; ctx.n];
        let mut stats = [LocusStat::default(); MENU_LEN];
        let mut focused = 0usize;
        let mut fired = 0usize;
        for ((vi, slot), flag) in facets.iter_mut().enumerate().zip(has_focus.iter_mut()) {
            let Some(focus) = stance.focus(vi, ctx) else {
                continue;
            };
            *flag = true;
            focused += 1;
            let (f, sil) = mint(vi, focus, ctx);
            for (st, s) in stats.iter_mut().zip(sil) {
                st.add(s);
            }
            if f.bound_count() > 0 {
                fired += 1;
            }
            *slot = f;
        }
        Self {
            stance,
            facets,
            has_focus,
            focused,
            fired,
            stats,
        }
    }

    /// Fraction of ALL verses on which this stance's facet fires.
    fn fire_rate(&self) -> f64 {
        if self.facets.is_empty() {
            0.0
        } else {
            self.fired as f64 / self.facets.len() as f64
        }
    }

    /// Is this stance degenerate by prevalence (§12.7's 88 % tell)?
    fn degenerate_by_prevalence(&self) -> bool {
        self.fire_rate() > DEGENERATE_RATE
    }

    /// Loci whose bind rate exceeds [`NEAR_CONSTANT_RATE`] over focused verses.
    fn near_constant(&self) -> [bool; MENU_LEN] {
        let mut out = [false; MENU_LEN];
        for (o, st) in out.iter_mut().zip(self.stats) {
            *o = st.rate(self.focused) > NEAR_CONSTANT_RATE;
        }
        out
    }
}

// ── reporting ───────────────────────────────────────────────────────────────

/// Print one facet as its BOUND slots, over all 24. A slot absent from the line
/// read `0`; whether that zero is by construction or by measurement is answered
/// by the per-locus silence table, not here.
fn print_vector(prefix: &str, f: CausalWitnessFacet) {
    let mut cells: Vec<String> = Vec::new();
    for slot in 0..WITNESS_LOCI {
        let v = f.get(slot);
        if v != 0 {
            let label = Locus::ALL.get(slot).map_or("reserved", |l| l.label());
            cells.push(format!("{label}={v:+}"));
        }
    }
    if cells.is_empty() {
        println!("{prefix} (all 24 slots unbound)");
    } else {
        println!("{prefix} {}", cells.join("  "));
    }
}

/// Mean and full distribution of [`CausalWitnessFacet::agreement_count`] over a
/// verse-aligned facet pair, plus the same restricted to DISCRIMINATING loci
/// (near-constant loci masked out on either side — a locus that fires on
/// everything carries as much information as one that never fires).
fn agreement(a: &StanceRun, b: &StanceRun) -> (f64, BTreeMap<usize, usize>, f64) {
    let na = a.near_constant();
    let nb = b.near_constant();
    let discriminating: Vec<Locus> = MENU
        .iter()
        .enumerate()
        .filter(|&(i, _)| !na[i] && !nb[i])
        .map(|(_, &l)| l)
        .collect();

    let mut hist: BTreeMap<usize, usize> = BTreeMap::new();
    let mut sum = 0usize;
    let mut disc_sum = 0usize;
    for (&fa, &fb) in a.facets.iter().zip(&b.facets) {
        // The SHIPPED comparison primitive, used exactly as written.
        let c = fa.agreement_count(fb);
        *hist.entry(c).or_insert(0) += 1;
        sum += c;
        disc_sum += discriminating
            .iter()
            .filter(|&&l| fa.agrees_at(fb, l))
            .count();
    }
    let n = a.facets.len().max(1) as f64;
    (sum as f64 / n, hist, disc_sum as f64 / n)
}

/// The three anchor distances (A1, A2, A3) and the three control distances for
/// one stance, computed with the SAME [`menu_distance`] on both, so an anchor is
/// never graded on a scale the controls did not face.
fn pair_distances(run: &StanceRun) -> (usize, usize, usize, Vec<usize>) {
    let d = |x: usize, y: usize| menu_distance(run.facets[x], run.facets[y]);
    let controls: Vec<usize> = CONTROLS.iter().map(|&(x, y)| d(x, y)).collect();
    (
        d(A1_BEFORE, A1_AFTER),
        d(A2_PROMISE, A2_CONFIRM),
        d(A2_PROMISE, A1_AFTER),
        controls,
    )
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = args
        .first()
        .cloned()
        .unwrap_or_else(|| DEFAULT_TSV.to_string());
    let limit: Option<usize> = match args.get(1).map(String::as_str) {
        Some("all") => None,
        Some(a) => Some(a.parse().unwrap_or(DEFAULT_VERSE_LIMIT)),
        None => Some(DEFAULT_VERSE_LIMIT),
    };

    println!("== D-BLW-2 REBUILD — the BINDING RULES are the instrument ==");
    println!(
        "write sites  : 9 of 24 slots, ALL shared by all four stances \
         ⇒ agreement_count ceiling = 9 (the predecessor's was 1; §12.7)"
    );
    println!(
        "by construct.: slots 0-3 (TEKAMOLO), 7 (Antecedent — pronouns are \
         collapsed upstream), 8 (BasinAnchor), 11 (RunbookEvidence), 16-23 \
         (reserved) are NEVER written here. Every other zero is a measurement."
    );
    println!(
        "window       : an out-of-window target reads UNBOUND, never saturated \
         to +7 — saturation manufactures agreement."
    );
    println!(
        "claim ceiling: this program may say two readings DIFFER. It may not \
         say one is better, truer or more complete (§12.4). No p-values."
    );

    let mut verses = match load_tsv(&path) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("blw_binding: cannot read {path}: {e}");
            return;
        }
    };
    let loaded = verses.len();
    if let Some(lim) = limit {
        if loaded > lim {
            verses.truncate(lim);
        }
    }
    let n = verses.len();
    println!("corpus       : {n} of {loaded} verses from {path}");
    // Every pre-registered index — anchors AND controls — must be inside the
    // loaded slice, or `pair_distances` would index out of range and the report
    // would be about verses that are not there.
    let highest_needed = CONTROLS
        .iter()
        .flat_map(|&(x, y)| [x, y])
        .chain([A1_BEFORE, A1_AFTER, A2_PROMISE, A2_CONFIRM])
        .max()
        .expect("the anchor/control index set is non-empty");
    if n <= highest_needed {
        println!(
            "blw_binding: corpus holds {n} verses but the pre-registered anchors and \
             controls need index {highest_needed} — stopping rather than reporting on \
             verses that are not loaded."
        );
        return;
    }
    assert_anchor_texts(&verses);
    println!("anchor guard : indices 55/60/62/77 carry the pre-registered texts — OK");

    let ctx = build(&verses);
    let runs: Vec<StanceRun> = Stance::ALL
        .iter()
        .map(|&s| StanceRun::run(s, &ctx))
        .collect();

    // ── per-stance focus / fire rates + the degeneracy discipline ──
    println!("\n— STANCE COVERAGE (focus rate, fire rate, degeneracy) —");
    for r in &runs {
        let deg = if r.degenerate_by_prevalence() {
            "  DEGENERATE BY PREVALENCE — excluded from the pairwise table"
        } else {
            ""
        };
        println!(
            "  {:<13} focus on {}/{} verses, fires on {} ({:.1}%){}",
            r.stance.name(),
            r.focused,
            n,
            r.fired,
            100.0 * r.fire_rate(),
            deg
        );
    }

    // ── per-locus bind rate + silence kind ──
    println!("\n— PER-LOCUS BIND RATE AND SILENCE KIND (denominator = focused verses) —");
    println!("  (n/a = by construction · none = no target anywhere · oow = target outside the ±8 window)");
    for r in &runs {
        println!("  {}:", r.stance.name());
        let nc = r.near_constant();
        for ((i, &l), st) in MENU.iter().enumerate().zip(r.stats) {
            let flag = if nc[i] { "   << NEAR-CONSTANT" } else { "" };
            println!(
                "      {:<17} bound {:.4}  (bound {}, n/a {}, none {}, oow {}){}",
                l.label(),
                st.rate(r.focused),
                st.bound,
                st.not_applicable,
                st.no_candidate,
                st.out_of_window,
                flag
            );
        }
    }

    // ── QUANTITY (a): the lever arm ──
    println!("\n— QUANTITY (a): EVIDENCE LEVER ARM — are loci 9/10 bound at all? —");
    println!("  Collapsed = nothing grounds anything = the nihilism signature (§12.3c).");
    for r in &runs {
        // Counted over FOCUSED verses only: a verse where the stance has nothing
        // to read has no lever arm, and pooling it in as "collapsed" would
        // report the stance's own silence as a nihilism reading of the text.
        let mut counts: BTreeMap<Lever, usize> = BTreeMap::new();
        for (&f, &focused) in r.facets.iter().zip(&r.has_focus) {
            if focused {
                *counts.entry(lever(f)).or_insert(0) += 1;
            }
        }
        let total: usize = counts.values().sum();
        let collapsed = counts.get(&Lever::Collapsed).copied().unwrap_or(0);
        println!(
            "  {:<13} {:?}  (collapsed {}/{} = {:.1}%)",
            r.stance.name(),
            counts,
            collapsed,
            total,
            if total == 0 {
                0.0
            } else {
                100.0 * collapsed as f64 / total as f64
            }
        );
        if total > 0 && (collapsed == 0 || collapsed == total) {
            println!(
                "      INERT: this stance's lever is constant across the corpus — \
                 it discriminates nothing here (a guard that always fires carries \
                 as much information as one that never does)."
            );
        }
    }

    // ── QUANTITY (b): torque ──
    println!("\n— QUANTITY (b): TORQUE — |locus 12 − loci 4/5/6|, the said-vs-meant angle —");
    println!("  Reported SEPARATELY from (a) and never averaged with it: they are");
    println!("  different mechanics (a real lever displaced, vs no lever at all).");
    for r in &runs {
        let mut hist: BTreeMap<u8, usize> = BTreeMap::new();
        let mut undefined = 0usize;
        // Focused verses only, for the same reason as (a): a verse the stance
        // never read is not a verse on which torque is "undefined by the text".
        for (&f, &focused) in r.facets.iter().zip(&r.has_focus) {
            if !focused {
                continue;
            }
            match torque(f) {
                Some(t) => *hist.entry(t).or_insert(0) += 1,
                None => undefined += 1,
            }
        }
        let defined: usize = hist.values().sum();
        println!(
            "  {:<13} defined on {} of {} focused verses, undefined on {} — distribution {:?}",
            r.stance.name(),
            defined,
            r.focused,
            undefined,
            hist
        );
        if defined == 0 {
            println!(
                "      UNDEFINED EVERYWHERE — this stance never binds locus 12 \
                 together with a meaning locus, so torque says nothing about it \
                 here. Printed, not smoothed over."
            );
        }
    }

    // ── pairwise texture, live stances only ──
    println!("\n— PAIRWISE TEXTURE (shipped agreement_count; ceiling 9) —");
    let live: Vec<&StanceRun> = runs
        .iter()
        .filter(|r| !r.degenerate_by_prevalence())
        .collect();
    if live.len() < 2 {
        println!("  fewer than two non-degenerate stances — no pairs to compare");
    }
    for (i, a) in live.iter().enumerate() {
        for b in live.iter().skip(i + 1) {
            let (mean, hist, disc) = agreement(a, b);
            println!(
                "  {} x {}: mean = {:.4} over {} verses; discriminating-loci mean = {:.4}; dist = {:?}",
                a.stance.name(),
                b.stance.name(),
                mean,
                n,
                disc,
                hist
            );
        }
    }

    // ── the pre-registered anchors ──
    println!("\n— PRE-REGISTERED ANCHORS (§12.6) — per-stance facet vectors —");
    for r in &runs {
        println!("  {}:", r.stance.name());
        for (label, vi) in [
            ("A1 before  55 (Gen 2:25)", A1_BEFORE),
            ("A1 after   62 (Gen 3:7)", A1_AFTER),
            ("A2 promise 60 (Gen 3:5)", A2_PROMISE),
            ("A2 confirm 77 (Gen 3:22)", A2_CONFIRM),
        ] {
            let f = r.facets[vi];
            let t = torque(f).map_or("undefined".to_string(), |t| t.to_string());
            print_vector(
                &format!("      {label}  lever={:?} torque={t}  |", lever(f)),
                f,
            );
        }
    }

    println!("\n— SEPARATION VERDICTS (rule fixed in source BEFORE the run) —");
    println!("  An anchor pair SEPARATES iff its structural distance over the 9 MENU");
    println!("  loci strictly exceeds the LARGEST distance on the three control pairs.");
    println!("  The threshold is the corpus's own baseline churn, not a chosen constant.");
    for r in &runs {
        let (d_a1, d_a2, d_a3, controls) = pair_distances(r);
        let ctrl_max = controls.iter().copied().max().unwrap_or(0);
        let ctrl_mean = controls.iter().sum::<usize>() as f64 / controls.len() as f64;
        println!(
            "  {} — controls {:?} (max {ctrl_max}, mean {ctrl_mean:.2}) of 9",
            r.stance.name(),
            controls
        );
        for (name, d, a, b) in [
            (
                "A1  55 vs 62 (naked → KNEW naked)",
                d_a1,
                A1_BEFORE,
                A1_AFTER,
            ),
            (
                "A2  60 vs 77 (promise vs confirmed)",
                d_a2,
                A2_PROMISE,
                A2_CONFIRM,
            ),
            (
                "A3  60 vs 62 (promise → fulfilment)",
                d_a3,
                A2_PROMISE,
                A1_AFTER,
            ),
        ] {
            let verdict = if d > ctrl_max {
                "SEPARATED (exceeds control baseline)"
            } else if d == 0 {
                "KILL — identical facets; this instrument cannot tell them apart"
            } else {
                "NOT SEPARATED — within control baseline, i.e. indistinguishable from ordinary churn"
            };
            println!(
                "      {name}: distance {d}/9 — {verdict}; loci differing: {:?}",
                differing_loci(r.facets[a], r.facets[b])
            );
        }
    }
    println!(
        "  A1 is the MUST-HAVE (§12.6): if no stance separates 55 from 62, that \
         is a KILL OF THE INSTRUMENT and is the reportable result. No weight in \
         this file was adjusted toward any of these outcomes — the rules were \
         fixed before the file could be run, and this session cannot run it."
    );

    // ── the horizon control (§12.3b/§12.3c falsifier 2) ──
    println!("\n— HORIZON CONTROL (§12.3b: fixed verse set, only the horizon moves) —");
    let k = HORIZON_K.min(n);
    if k >= n {
        println!("  Vk == Vm ({n} verses ≤ {HORIZON_K}) — needs Vm > Vk; skipped, not faked.");
    } else {
        let ctx_k = build(&verses[..k]);
        println!("  Vk={k} vs Vm={n}; sample growth excluded by construction (same {k} verses).");
        for stance in Stance::ALL {
            let rk = StanceRun::run(stance, &ctx_k);
            let rm = &runs[Stance::ALL
                .iter()
                .position(|&s| s == stance)
                .expect("stance")];
            let mut rebound = 0usize;
            let mut per_locus = [0usize; MENU_LEN];
            for (&a, &b) in rk.facets.iter().zip(&rm.facets).take(k) {
                if a != b {
                    rebound += 1;
                }
                for (slot, &l) in MENU.iter().enumerate() {
                    if a.at(l) != b.at(l) {
                        per_locus[slot] += 1;
                    }
                }
            }
            let detail: Vec<String> = MENU
                .iter()
                .zip(per_locus)
                .filter(|&(_, c)| c > 0)
                .map(|(l, c)| format!("{}={c}", l.label()))
                .collect();
            println!(
                "  {:<13} {rebound}/{k} verses rebind — {}",
                stance.name(),
                if detail.is_empty() {
                    "no locus moved (a real result, printed plainly)".to_string()
                } else {
                    detail.join(" ")
                }
            );
        }
    }

    println!("\n— WHAT THIS RUN DOES NOT SHOW —");
    println!("  · No cross-language arm (§12.6 A3′). Nine PD lanes are on disk, but");
    println!("    detection is not built: there is no morphological parser for Latin,");
    println!("    Greek, Syriac or Hebrew here, and hand-writing an `in quo`/`v němž`");
    println!("    matcher would fit the pre-registered answer instead of testing it.");
    println!("  · No substrate claim. This harness reads a TSV through the stance");
    println!("    library; the tenant/kanban/batch-writer substrate is D-BLW-1");
    println!("    (`blw_tenant.rs`) and is not exercised here.");
    println!("  · No validity claim of any kind (§12.4).");
    println!("\nblw_binding: done. Every number above was computed at run time.");
}
