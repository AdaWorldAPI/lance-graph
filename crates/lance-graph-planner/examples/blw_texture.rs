//! `blw_texture` — texture, not coincidence. Replaces the retired κ instrument
//! (`blw_lens_twin.rs`, D-BLW-2's discrimination twin) with the register
//! [`CausalWitnessFacet`] per `.claude/plans/cycle-loop-closure-driver-v1.md`
//! §12.3c (the instrument ruling) and §12.4 (the claim ceiling).
//!
//! # Why κ was retired (§12.3c, read before touching this file)
//!
//! κ over per-verse binaries measures how often two lenses *coincide*, and
//! discards *why*. Two lenses can agree on a verse for opposite reasons and κ
//! scores that as agreement — the clean falsifier is that nihilism and
//! sarcasm are both *negative*, so any sign/threshold/boolean collapses
//! them, yet they are different gestures. `CausalWitnessFacet` — 24 signed
//! `i4` loci, each a pointer to another stream position, never a magnitude
//! (`causal_witness.rs`'s own "Loci, not magnitudes" doctrine) — carries
//! *which* structure a reading binds, not merely whether it agrees.
//!
//! # What this harness does, precisely
//!
//! 1. Streams the real KJV verse TSV (`index\ttext`, no parser written here —
//!    `split_once('\t')` on a pre-tokenized 2-column file is not a parser)
//!    through the **REAL, UNMODIFIED** [`stance::stream`] + [`stance_panel`],
//!    mirroring `examples/probe_eyes_opened.rs::report()`'s pass-1 half (see
//!    [`build`] for exactly what is mirrored and what is deliberately
//!    skipped, and why the skip is safe).
//! 2. Mints ONE [`CausalWitnessFacet`] per (verse, stance) — four per verse —
//!    from what `stance_panel` actually produces for that verse. The four
//!    binding rules are below, each with its bind-nothing condition stated
//!    in advance (not discovered post hoc).
//! 3. Compares stances pairwise with [`CausalWitnessFacet::agreement_count`]
//!    — the shipped texture-comparison primitive, used exactly as written,
//!    never reimplemented.
//! 4. Runs the ONE runnable falsifier from §12.3c — the horizon intervention
//!    (§12.3b/§12.3c falsifier 2): hold the first `k` verses fixed, mint
//!    their facets from the arena as sealed at `Vk`, then again from the
//!    arena as sealed at `Vm > k`, and report which loci REBIND.
//! 5. Applies the degeneracy discipline unchanged from the κ era (a texture
//!    identical on every verse is the `closed_class_guess` 99.61 % defect in
//!    a new costume) — a DEGENERATE stance is excluded and printed, never
//!    reported as a stance.
//! 6. States plainly that the cross-language falsifier (§12.3c falsifier 1)
//!    is NOT ATTEMPTED here — and, since 2026-08-04, **not for want of
//!    corpora**: 9 Public-Domain lanes across 7 languages are on disk
//!    (§12.3c ⊘⊘⊘). It is not attempted because *detection* is not built —
//!    no morphological parser for Latin/Greek/Syriac/Hebrew exists here, and
//!    hand-writing a matcher for the split already pre-registered in §12.6
//!    A3′ would fit the answer instead of testing it. PROBE-BABEL-STANCES'
//!    "lanes" remain hand-authored `LaneLex` fixtures, not corpora. Not
//!    simulated, not substituted.
//!
//! # The four binding rules
//!
//! Every stance binds `Locus::Antecedent` — the ONE locus shared by all
//! four, so [`CausalWitnessFacet::agreement_count`] has a real chance to
//! fire (see the structural-ceiling note printed at runtime: giving each
//! stance its OWN private locus with no shared locus would make
//! `agreement_count` return `0` for every pair on every verse, always, by
//! construction — an instrument that can never speak is not an instrument).
//! `Locus::Antecedent` always means the same thing across stances — "the
//! nearest OTHER verse this reading's structure is grounded in" — computed
//! by a stance-specific rule. Two stances additionally use a PRIVATE second
//! locus for their own distinctive signal (documented per-stance below);
//! nothing here uses `Locus::QualiaReference` / `SMeaning` / `PMeaning` /
//! `OMeaning` / `BasinAnchor` / `SupportedBy` / `Supports` /
//! `RunbookEvidence` / `MeaningLevel` / `Contradiction` (Hegel's preserved
//! peer is `Quorum`, not `Contradiction` — see below) / the TEKAMOLO slots
//! `Temporal`/`Kausal`/`Lokal` — those loci read `0` (unbound) on every
//! facet this harness mints, always, by construction, not by measurement.
//!
//! Every offset is a genuine **pointer** (a signed distance in the
//! relevant index space, clamped into `[-8, +7]` by [`to_offset`]), never a
//! raw encoded scalar — this is the register's own operator-locked
//! constraint ("Loci, not magnitudes", `causal_witness.rs` header) and it
//! shaped every rule below, including the one deviation disclosed under
//! Kant's `Modal` binding.
//!
//! **Hegel — Aufhebung (`stance.rs`'s own reading: cancelled = pooled
//! truth, preserved = the `contradiction` field).** For the FIRST
//! provenance entry observed at this verse whose statement is hegel-ranked
//! (`contradiction_ranking`'s `> 0.05` filter):
//! - `Antecedent` = signed distance to the nearest OTHER-verse
//!   re-observation of the SAME statement with OPPOSITE polarity (the
//!   dissenting peer — "preserved").
//! - `Quorum` = signed distance to the nearest OTHER-verse re-observation
//!   with the SAME polarity (the pooling peer — "cancelled"), if any.
//! - **Binds nothing** when no provenance entry at this verse carries a
//!   hegel-ranked statement (the verse never contributes to any preserved
//!   contradiction). A verse with multiple hegel-ranked statements encodes
//!   only the first (documented tie-break, matching `blw_lens_twin.rs`'s
//!   own disclosed per-verse-projection precedent).
//!
//! **Nietzsche — genealogy (flip direction read from provenance
//! endpoints).** For the FIRST provenance entry at this verse whose
//! statement is nietzsche-ranked (has a legible `FlipKind`):
//! - `Antecedent` = if this verse is the FIRST occurrence of the statement,
//!   signed distance FORWARD to the LAST occurrence (positive — "this is
//!   where I get overturned"); if this verse IS the last occurrence, signed
//!   distance BACKWARD to the first (negative — "this is what I
//!   overturned"). The FlipKind itself (`Transvaluation`/`Devaluation`) is
//!   not re-encoded on the register — it already IS the ordering of
//!   `negated` at the two endpoints `stance_panel` computed; this pointer
//!   names WHERE the other endpoint is, not what kind of flip it was.
//! - **Binds nothing** when this verse is neither the first nor the last
//!   occurrence of a nietzsche-ranked statement (an intermediate
//!   re-observation carries no genealogy signal — `stance_panel`'s own
//!   genealogy only reads endpoints, correctly silent here too) OR when no
//!   provenance entry at this verse carries a nietzsche-ranked statement.
//!
//! **Kant — the §12.3a-corrected, RANK-based signal (never the raw
//! `quale > ablated` tautology, which is true for both shipped modals and
//! was the very defect §12.3a caught before it shipped).** For the FIRST
//! lift at this verse (`out.lifts`):
//! - `Antecedent` = signed distance to the nearest OTHER-verse
//!   re-observation of the lift's own INNER statement (the "that"-clause's
//!   grounded emission — e.g. 3:7's inner "they were naked" was already
//!   observed at 2:25; found by locating, among this verse's provenance
//!   entries, the one whose predicate equals the lift's `object`, then
//!   searching for the same statement elsewhere). Binds nothing if that
//!   inner statement was never observed anywhere else.
//! - `Modal` = **bound ONLY when a-priori grading actually moved this
//!   lift's rank** relative to the uniform-modal (0.5) ablation — i.e.
//!   `ablated_rank − graded_rank != 0` — pointing at the verse of the lift
//!   ranked immediately ABOVE this one in the GRADED ordering (the
//!   neighbor this lift's promotion/demotion is measured against). This is
//!   a DISCLOSED deviation from "pointer into the verse stream": the
//!   distance lives in rank-order space, then is mapped back onto the
//!   verse position of that rank-neighbor, so it stays a genuine pointer
//!   (to a real verse), just not one reached via `resolves_to`'s ±window
//!   semantics. **Binds nothing** when the verse holds no lift, when this
//!   lift is already rank 0 in the graded ordering (no neighbor above), or
//!   — the honest, non-fabricated silence case — when grading and ablation
//!   produce the SAME rank for this lift (a real, measurable tie, not
//!   assumed impossible: verses early in the stream can have
//!   `staunen_at == 0`, so `quale = modal * 0 == 0.5 * 0 == 0` under BOTH
//!   grading and ablation, producing exact ties among the zero-quale
//!   cluster).
//!
//! **Wittgenstein — meaning as use (distinct language-games).** Among the
//! concepts touched at this verse (provenance subject/predicate, lift
//! knower/object, causal-impl cause/effect — the SAME three sources
//! `stance_panel`'s Wittgenstein pools by concept; this harness groups them
//! by VERSE instead, a disclosed per-verse projection choice exactly like
//! `blw_lens_twin.rs`'s own Wittgenstein-reduced projection):
//! - `Antecedent` = pick the touched concept with the MOST distinct games
//!   globally (tie-break: smallest concept id), then signed distance to the
//!   nearest OTHER verse where that SAME concept is ALSO touched.
//! - **Binds nothing** when no concept touched at this verse has any
//!   recorded game (cannot happen for a verse producing ANY provenance,
//!   since every Inh emission registers ≥1 game for both its subject and
//!   predicate — so this fires only for a verse that emits NOTHING at all
//!   through `stream()`'s clause machine, e.g. a genealogy list the
//!   catalogues don't arm a predicate on) OR when the crowned concept's
//!   ONLY occurrence anywhere is this verse.
//!
//! # Honesty rules this file follows (non-negotiable, `CLAUDE.md` P0)
//!
//! No number in this program's output is invented — everything is computed
//! from the real corpus at run time; where a value is only knowable at
//! runtime (agreement means, rebind counts, degeneracy verdicts), this file
//! prints it and does not predict it. A null result (a stance that never
//! fires, two stances that never agree, zero rebinds under the horizon
//! intervention) is a result and is printed plainly, not tuned away. No
//! binding rule was adjusted to manufacture agreement or disagreement — the
//! rules above were fixed BEFORE this file could be run (this session
//! cannot run `cargo`; the orchestrator compiles and runs centrally), so
//! none of them could have been reverse-engineered from an observed output.
//!
//! # Claim ceiling (§12.4), binding on every line this program prints
//!
//! Structure and overlap only. Never "valid"/"accurate"/"better"/
//! "confirms". No p-values (`jc::stats` p-values are classical
//! independent-sample values; verses within a book are domain-correlated,
//! `I-NOISE-FLOOR-JIRAK`). The word "fusion" is used ONLY inside the
//! horizon-intervention section, per §12.3c's own restriction, and there it
//! names a measured change in binding topology, never a validity claim.
//!
//! # Usage
//!
//! `cargo run -p lance-graph-planner --example blw_texture [-- <tsv path>]`
//! — defaults to `/tmp/kjv_verses.tsv` (`index\ttext` per line, matching the
//! format this harness was told to expect; no chapter:verse labels needed
//! since verse POSITION is all the binding rules use).

use std::collections::HashMap;

use lance_graph_contract::causal_witness::{CausalWitnessFacet, Locus};
use lance_graph_planner::nars::stance::{stance_panel, stream, FlipKind, Interner, ReadOut};
use lance_graph_planner::nars::{BeliefArena, CStmt};

/// Default corpus path (`index\ttext` per line, 0-based row index as the
/// verse label — matches `/tmp/kjv_verses.tsv` as generated for this task).
const DEFAULT_TSV: &str = "/tmp/kjv_verses.tsv";

/// The horizon intervention's fixed prefix length (§12.3b's own corpus
/// floor, `1,000`, reused here rather than invented fresh — see §12.3a's
/// `CORPUS_FLOOR` in `blw_lens_twin.rs` for the same number's first use).
/// Pre-registered before any run: this file cannot be executed by the
/// session that wrote it, so this could not have been tuned to an observed
/// result.
const HORIZON_K: usize = 1_000;

/// Clamp a signed delta (which may be far outside `i8` range at whole-book
/// scale — verse positions run to five digits) into the register's `[-8,
/// +7]` window, correctly: clamping happens in `isize` BEFORE the cast, so
/// a delta of e.g. `30_000` saturates to `+7` rather than wrapping through
/// an out-of-range `as i8` cast (which would silently alias to an unrelated
/// small value instead of saturating).
fn to_offset(delta: isize) -> i8 {
    delta.clamp(-8, 7) as i8
}

/// Nearest position to `at` among `candidates`, excluding `at` itself. Ties
/// (equal `|distance|` on both sides) break toward the smaller (earlier)
/// position — a deterministic, disclosed tie-break, not a hidden one.
fn nearest_pos(candidates: impl Iterator<Item = usize>, at: usize) -> Option<usize> {
    candidates
        .filter(|&pos| pos != at)
        .min_by_key(|&pos| (pos.abs_diff(at), pos))
}

/// [`nearest_pos`] over a statement's `(position, negated)` occurrence list,
/// optionally filtered to one polarity (`Some(want)`) or any (`None`).
fn nearest_stmt_pos(occs: &[(usize, bool)], at: usize, want: Option<bool>) -> Option<usize> {
    nearest_pos(
        occs.iter()
            .filter(move |&&(_, neg)| want.is_none_or(|w| neg == w))
            .map(|&(pos, _)| pos),
        at,
    )
}

/// Read the TSV verse corpus. `index\ttext` per line — `split_once('\t')` on
/// an already-tokenized two-column file, not a parser: the inbound leg owns
/// text parsing, and this file was told not to write one.
fn load_tsv(path: &str) -> std::io::Result<Vec<(String, String)>> {
    let text = std::fs::read_to_string(path)?;
    Ok(text
        .lines()
        .filter_map(|line| line.split_once('\t'))
        .map(|(idx, verse)| (idx.to_string(), verse.to_string()))
        .collect())
}

/// Build the arena for `verses`. Mirrors `probe_eyes_opened.rs::report()`'s
/// pass-1 half EXACTLY: one `stream(..., pass2=false)` call, then
/// `close_transitive(64)` — the same call, same budget, same order `report`
/// uses before it prints B1-B3 and before `main()` hands the arena to
/// `print_stance_panel`.
///
/// Two disclosed divergences from `report()`:
/// 1. No printing — this harness only needs the panel machinery, not the
///    B1-B4 demonstration output.
/// 2. `report()`'s pass-2 re-read (used only for its own B4 Hermeneutik
///    demonstration) is SKIPPED. This is safe, not a shortcut:
///    `stance::stream` resets `src: u32 = 0` at entry (`stance.rs:169`), so
///    a second call over the SAME verses replays IDENTICAL stamps in
///    IDENTICAL order — a structural property of the stamp scheme, true
///    for ANY input, not an empirical fact only checked on the 8-verse
///    SCENE fixture where B4 happens to assert it. `BeliefArena`'s S4
///    overlap guard therefore routes every pass-2 re-observation to
///    `ReviseOutcome::Chosen` — no admission, no revision, no arena change
///    — so pass-2 cannot affect anything `stance_panel` reads. Skipping it
///    saves a second full `stream()` pass (and a second
///    `close_transitive`) at zero cost to correctness.
fn build(verses: &[(String, String)]) -> (BeliefArena, Interner, ReadOut) {
    let mut arena = BeliefArena::new();
    let mut intern = Interner::new();
    let mut out = ReadOut::default();
    stream(verses, &mut arena, &mut intern, &mut out, false);
    arena.close_transitive(64);
    (arena, intern, out)
}

/// Precomputed lookups over one [`ReadOut`], scoped to whatever verse
/// prefix `build` was called with (a "Vk" index and a "Vm" index are two
/// SEPARATE `VerseIndex` values over two separate `ReadOut`s — never mixed).
struct VerseIndex {
    /// Verse label -> position (0-based index into the verses slice this
    /// index was built from).
    pos_of: HashMap<String, usize>,
    /// Statement -> sorted `(position, negated)` occurrences, in
    /// verse-stream order (already non-decreasing since `out.provenance`
    /// is pushed in stream order).
    by_stmt: HashMap<CStmt, Vec<(usize, bool)>>,
    /// Verse position -> provenance entry indices observed there, in order.
    prov_at: HashMap<usize, Vec<usize>>,
    /// Verse position -> lift indices (`out.lifts`) observed there.
    lifts_at: HashMap<usize, Vec<usize>>,
    /// Verse position -> distinct concept ids touched there (provenance
    /// subject/predicate, lift knower/object, impl cause/effect).
    concepts_at: HashMap<usize, Vec<u16>>,
    /// Concept id -> sorted, deduplicated verse positions where it is
    /// touched (the inverse of `concepts_at`, built for the Wittgenstein
    /// rule's "nearest other verse touching this same concept" search).
    by_concept: HashMap<u16, Vec<usize>>,
}

impl VerseIndex {
    fn build(verses: &[(String, String)], out: &ReadOut) -> Self {
        let pos_of: HashMap<String, usize> = verses
            .iter()
            .enumerate()
            .map(|(i, (v, _))| (v.clone(), i))
            .collect();

        let mut by_stmt: HashMap<CStmt, Vec<(usize, bool)>> = HashMap::new();
        let mut prov_at: HashMap<usize, Vec<usize>> = HashMap::new();
        let mut concepts_at: HashMap<usize, Vec<u16>> = HashMap::new();

        for (idx, p) in out.provenance.iter().enumerate() {
            let pos = *pos_of
                .get(&p.verse)
                .expect("provenance verse label must resolve — stream() only emits labels it was given from `verses`");
            by_stmt.entry(p.stmt).or_default().push((pos, p.negated));
            prov_at.entry(pos).or_default().push(idx);
            concepts_at.entry(pos).or_default().push(p.stmt.s);
            concepts_at.entry(pos).or_default().push(p.stmt.p);
        }

        let mut lifts_at: HashMap<usize, Vec<usize>> = HashMap::new();
        for (idx, l) in out.lifts.iter().enumerate() {
            let pos = *pos_of.get(&l.verse).expect("lift verse label must resolve");
            lifts_at.entry(pos).or_default().push(idx);
            concepts_at.entry(pos).or_default().push(l.knower);
            concepts_at.entry(pos).or_default().push(l.object);
        }

        for (v, c, e) in &out.impls {
            let pos = *pos_of.get(v).expect("impl verse label must resolve");
            concepts_at.entry(pos).or_default().push(*c);
            concepts_at.entry(pos).or_default().push(*e);
        }

        for v in concepts_at.values_mut() {
            v.sort_unstable();
            v.dedup();
        }

        let mut by_concept: HashMap<u16, Vec<usize>> = HashMap::new();
        for (&pos, concepts) in &concepts_at {
            for &c in concepts {
                by_concept.entry(c).or_default().push(pos);
            }
        }
        for v in by_concept.values_mut() {
            v.sort_unstable();
            v.dedup();
        }

        Self {
            pos_of,
            by_stmt,
            prov_at,
            lifts_at,
            concepts_at,
            by_concept,
        }
    }
}

/// The four `stance_panel` outputs, bundled only so [`mint_all`] takes one
/// argument instead of four — no new machinery, purely a bundling
/// convenience over `stance_panel`'s own return shape.
struct Panel<'a> {
    hegel: &'a [(CStmt, f32)],
    nietzsche: &'a [(CStmt, FlipKind)],
    kant: &'a [(String, f32, f32)],
    wittgenstein: &'a [(u16, usize)],
}

/// One verse's texture: the four stances' independent readings of it.
#[derive(Clone, Copy)]
struct VerseTexture {
    hegel: CausalWitnessFacet,
    nietzsche: CausalWitnessFacet,
    kant: CausalWitnessFacet,
    wittgenstein: CausalWitnessFacet,
}

/// Hegel's binding rule — see the module doc's "Hegel" section for the full
/// statement of what this does and what makes it bind nothing.
fn mint_hegel(
    vi: usize,
    out: &ReadOut,
    index: &VerseIndex,
    hegel_set: &HashMap<CStmt, f32>,
) -> CausalWitnessFacet {
    let mut f = CausalWitnessFacet::ZERO;
    let Some(prov_idxs) = index.prov_at.get(&vi) else {
        return f;
    };
    for &pidx in prov_idxs {
        let p = &out.provenance[pidx];
        if !hegel_set.contains_key(&p.stmt) {
            continue;
        }
        let occs = &index.by_stmt[&p.stmt];
        if let Some(conflict) = nearest_stmt_pos(occs, vi, Some(!p.negated)) {
            f = f.with(
                Locus::Antecedent,
                to_offset(conflict as isize - vi as isize),
            );
        }
        if let Some(agree) = nearest_stmt_pos(occs, vi, Some(p.negated)) {
            f = f.with(Locus::Quorum, to_offset(agree as isize - vi as isize));
        }
        break; // first hegel-ranked statement at this verse wins (documented)
    }
    f
}

/// Nietzsche's binding rule — see the module doc's "Nietzsche" section.
fn mint_nietzsche(
    vi: usize,
    out: &ReadOut,
    index: &VerseIndex,
    nietzsche_set: &HashMap<CStmt, FlipKind>,
) -> CausalWitnessFacet {
    let mut f = CausalWitnessFacet::ZERO;
    let Some(prov_idxs) = index.prov_at.get(&vi) else {
        return f;
    };
    for &pidx in prov_idxs {
        let p = &out.provenance[pidx];
        if !nietzsche_set.contains_key(&p.stmt) {
            continue;
        }
        let occs = &index.by_stmt[&p.stmt];
        if let (Some(&(first, _)), Some(&(last, _))) = (occs.first(), occs.last()) {
            if first != last {
                if vi == first {
                    f = f.with(Locus::Antecedent, to_offset(last as isize - vi as isize));
                } else if vi == last {
                    f = f.with(Locus::Antecedent, to_offset(first as isize - vi as isize));
                }
                // an intermediate re-observation (neither first nor last)
                // carries no genealogy signal — stays unbound, correctly.
            }
        }
        break; // first nietzsche-ranked statement at this verse wins
    }
    f
}

/// Kant's binding rule — see the module doc's "Kant" section. `graded_rank`
/// / `ablated_rank` / `graded_order` are positionally aligned to
/// `out.lifts` (built once by [`kant_rank_vectors`]).
fn mint_kant(
    vi: usize,
    out: &ReadOut,
    index: &VerseIndex,
    graded_rank: &[usize],
    ablated_rank: &[usize],
    graded_order: &[usize],
) -> CausalWitnessFacet {
    let mut f = CausalWitnessFacet::ZERO;
    let Some(lift_idxs) = index.lifts_at.get(&vi) else {
        return f;
    };
    let Some(&li) = lift_idxs.first() else {
        return f;
    };
    let l = &out.lifts[li];

    // Antecedent: nearest OTHER-verse re-observation of the lift's own
    // inner statement.
    let inner_pidx = index.prov_at.get(&vi).and_then(|v| {
        v.iter()
            .copied()
            .find(|&pidx| out.provenance[pidx].stmt.p == l.object)
    });
    if let Some(pidx) = inner_pidx {
        let inner_stmt = out.provenance[pidx].stmt;
        if let Some(occs) = index.by_stmt.get(&inner_stmt) {
            if let Some(other) = nearest_stmt_pos(occs, vi, None) {
                f = f.with(Locus::Antecedent, to_offset(other as isize - vi as isize));
            }
        }
    }

    // Modal: bound ONLY when a-priori grading moved this lift's rank
    // relative to the uniform-modal ablation (the §12.3a-corrected signal).
    let rank_delta = ablated_rank[li] as isize - graded_rank[li] as isize;
    if rank_delta != 0 && graded_rank[li] > 0 {
        let neighbor_li = graded_order[graded_rank[li] - 1];
        let neighbor_pos = index.pos_of[&out.lifts[neighbor_li].verse];
        f = f.with(Locus::Modal, to_offset(neighbor_pos as isize - vi as isize));
    }
    f
}

/// Wittgenstein's binding rule — see the module doc's "Wittgenstein"
/// section.
fn mint_wittgenstein(
    vi: usize,
    index: &VerseIndex,
    concept_games: &HashMap<u16, usize>,
) -> CausalWitnessFacet {
    let mut f = CausalWitnessFacet::ZERO;
    let Some(touched) = index.concepts_at.get(&vi) else {
        return f;
    };
    let crown = touched
        .iter()
        .copied()
        .filter(|c| concept_games.contains_key(c))
        .max_by_key(|c| (concept_games[c], std::cmp::Reverse(*c)));
    let Some(crown) = crown else {
        return f;
    };
    if let Some(positions) = index.by_concept.get(&crown) {
        if let Some(other) = nearest_pos(positions.iter().copied(), vi) {
            f = f.with(Locus::Antecedent, to_offset(other as isize - vi as isize));
        }
    }
    f
}

/// Build the rank vectors Kant's binding needs, positionally aligned to
/// `out.lifts` (`stance_panel` builds `kant` via `out.lifts.iter().map(...)`
/// in the same order, `stance.rs` — asserted here rather than assumed
/// silently, since a mismatch would mean the panel's own invariant broke).
fn kant_rank_vectors(
    out: &ReadOut,
    kant: &[(String, f32, f32)],
) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    assert_eq!(
        kant.len(),
        out.lifts.len(),
        "stance_panel's Kant output must be positionally aligned 1:1 with out.lifts"
    );
    let n = kant.len();

    let mut graded_order: Vec<usize> = (0..n).collect();
    graded_order.sort_by(|&a, &b| kant[b].1.total_cmp(&kant[a].1)); // desc by graded quale
    let mut graded_rank = vec![0usize; n];
    for (rank, &idx) in graded_order.iter().enumerate() {
        graded_rank[idx] = rank;
    }

    let mut ablated_order: Vec<usize> = (0..n).collect();
    ablated_order.sort_by(|&a, &b| kant[b].2.total_cmp(&kant[a].2)); // desc by ablated quale
    let mut ablated_rank = vec![0usize; n];
    for (rank, &idx) in ablated_order.iter().enumerate() {
        ablated_rank[idx] = rank;
    }

    (graded_rank, ablated_rank, graded_order)
}

/// Mint all four stances' facets for verses `0..n` from one `(out, panel,
/// index)` triple.
fn mint_all(n: usize, out: &ReadOut, panel: &Panel, index: &VerseIndex) -> Vec<VerseTexture> {
    let hegel_set: HashMap<CStmt, f32> = panel.hegel.iter().cloned().collect();
    let nietzsche_set: HashMap<CStmt, FlipKind> = panel.nietzsche.iter().cloned().collect();
    let concept_games: HashMap<u16, usize> = panel.wittgenstein.iter().cloned().collect();
    let (graded_rank, ablated_rank, graded_order) = kant_rank_vectors(out, panel.kant);

    (0..n)
        .map(|vi| VerseTexture {
            hegel: mint_hegel(vi, out, index, &hegel_set),
            nietzsche: mint_nietzsche(vi, out, index, &nietzsche_set),
            kant: mint_kant(vi, out, index, &graded_rank, &ablated_rank, &graded_order),
            wittgenstein: mint_wittgenstein(vi, index, &concept_games),
        })
        .collect()
}

/// Degeneracy discipline, carried forward unchanged from the κ era (§12.3a,
/// §12.3c): a stance that never fires, or fires identically everywhere, is
/// DEGENERATE — excluded from the pairwise comparison and printed, never
/// silently reported as a live stance. Checked in this order so the printed
/// reason is specific rather than merely "degenerate".
fn degeneracy(facets: &[CausalWitnessFacet]) -> Option<&'static str> {
    if facets.is_empty() {
        return Some("no verses to evaluate");
    }
    if facets.iter().all(|f| f.bound_count() == 0) {
        return Some("bound_count == 0 on every verse — this stance never fires");
    }
    if facets.iter().all(|&f| f == facets[0]) {
        return Some("identical facet on every verse — constant texture (the 99.61% defect in a new costume)");
    }
    None
}

/// Mean [`CausalWitnessFacet::agreement_count`] and its full distribution
/// over `0..=24` (in practice concentrated at `{0,1}` given this file's
/// binding rules — see the structural-ceiling note printed alongside this
/// in `main`; the histogram is still computed and printed in full rather
/// than assumed, so the concentration is shown, not asserted).
fn agreement_stats(
    a: &[CausalWitnessFacet],
    b: &[CausalWitnessFacet],
) -> (f64, std::collections::BTreeMap<usize, usize>) {
    assert_eq!(a.len(), b.len(), "texture vectors must be verse-aligned");
    let mut hist: std::collections::BTreeMap<usize, usize> = std::collections::BTreeMap::new();
    let mut sum = 0usize;
    for (&fa, &fb) in a.iter().zip(b.iter()) {
        let c = fa.agreement_count(fb);
        *hist.entry(c).or_insert(0) += 1;
        sum += c;
    }
    let mean = if a.is_empty() {
        0.0
    } else {
        sum as f64 / a.len() as f64
    };
    (mean, hist)
}

/// Fraction of `facets` with `locus` bound (nonzero).
fn bind_rate(facets: &[CausalWitnessFacet], locus: Locus) -> f64 {
    if facets.is_empty() {
        return 0.0;
    }
    facets.iter().filter(|f| f.is_bound(locus)).count() as f64 / facets.len() as f64
}

/// The horizon intervention (§12.3b/§12.3c falsifier 2): `k` = facets minted
/// from the arena as sealed at `Vk` (`verses[..k]` only); `m` = facets
/// minted from the arena as sealed at `Vm > k`, for the SAME `k` verses
/// (`textures_m[..k]`, i.e. `m` here is already restricted to the fixed
/// verse set by the caller). Reports the count of verses whose facet
/// changed AT ALL, and — the texture-not-coincidence point — WHICH loci
/// rebound, per locus, rather than a single scalar delta.
fn rebind_report(name: &str, k: &[CausalWitnessFacet], m: &[CausalWitnessFacet]) {
    assert_eq!(
        k.len(),
        m.len(),
        "horizon comparison must be over the SAME fixed verse set"
    );
    let mut rebind_count = 0usize;
    let mut per_locus: Vec<(&'static str, usize)> =
        Locus::ALL.iter().map(|l| (l.label(), 0usize)).collect();
    for (&fk, &fm) in k.iter().zip(m.iter()) {
        if fk != fm {
            rebind_count += 1;
        }
        for (i, &locus) in Locus::ALL.iter().enumerate() {
            if fk.at(locus) != fm.at(locus) {
                per_locus[i].1 += 1;
            }
        }
    }
    println!(
        "  {name}: {rebind_count}/{} verses rebind SOMEWHERE between Vk and Vm",
        k.len()
    );
    for (label, count) in &per_locus {
        if *count > 0 {
            println!("      locus {label}: rebinds at {count} verse(s)");
        }
    }
    if rebind_count == 0 {
        println!(
            "      (zero rebinds — under this rule, the horizon made no measured \
             difference to {name}'s texture on these {} verses; a real result, \
             printed plainly, not smoothed over)",
            k.len()
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = args
        .first()
        .cloned()
        .unwrap_or_else(|| DEFAULT_TSV.to_string());

    let verses = match load_tsv(&path) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("blw_texture: cannot read {path}: {e}");
            return;
        }
    };
    let n = verses.len();
    println!("blw_texture: {n} verses loaded from {path}");
    if n == 0 {
        println!("blw_texture: empty corpus — nothing to do");
        return;
    }

    println!(
        "\n— CROSS-LANGUAGE FALSIFIER (§12.3c falsifier 1): NOT ATTEMPTED HERE —\n\
         This harness reads ONE English lane ({path}) and does not attempt the \
         cross-language arm. It is NOT blocked for want of corpora: 9 \
         Public-Domain lanes across 7 languages are on disk (§12.3c ⊘⊘⊘ — \
         KJV, Luther1545, Elberfelder1905, BKR, Tischendorf, Vulgate, \
         Peshitta, Aleppo, Westminster-Leningrad). It is not attempted because \
         detection is not built: this repo has no morphological parser for \
         Latin/Greek/Syriac/Hebrew, and hand-writing a matcher for a split \
         already pre-registered in §12.6 A3′ would fit the answer rather than \
         test it. Not simulated, not substituted.\n\
         (PROBE-BABEL-STANCES' \"lanes\" remain hand-authored LaneLex fixtures \
         in that probe's own source, not corpora — unchanged.)"
    );

    println!(
        "\n— EXPECTED RUNTIME —\n\
         `stance::stream` calls `staunen(Snapshot::of(arena, 0.0))` once per \
         rung lift (stance.rs:327), each an O(arena-size) scan; over a \
         {n}-verse book with O(lifts) lifts this is O(lifts x arena), and \
         this harness runs `stream` TWICE — once over the full {n}-verse \
         corpus, once over the {HORIZON_K}-verse horizon prefix.\n\
         MEASURED 2026-08-04 (superseding this file's earlier \"cannot run \
         cargo to measure it\"): 2,000 verses = 1 s wall; the full 31,102-verse \
         KJV exceeded a 10-minute budget and was killed. The cost is \
         superlinear as described above — a known shape, not a crash. Bound \
         the corpus or fix the O(arena) rescan before running whole-book."
    );

    // ── Vm = the full corpus ──
    let (arena_m, intern_m, out_m) = build(&verses);
    let (hegel_m, nietzsche_m, kant_m, witt_m) = stance_panel(&arena_m, &intern_m, &out_m);
    let index_m = VerseIndex::build(&verses, &out_m);
    let panel_m = Panel {
        hegel: &hegel_m,
        nietzsche: &nietzsche_m,
        kant: &kant_m,
        wittgenstein: &witt_m,
    };
    let textures_m = mint_all(n, &out_m, &panel_m, &index_m);

    // ── degeneracy discipline over the full corpus ──
    let hegel_facets: Vec<CausalWitnessFacet> = textures_m.iter().map(|t| t.hegel).collect();
    let nietzsche_facets: Vec<CausalWitnessFacet> =
        textures_m.iter().map(|t| t.nietzsche).collect();
    let kant_facets: Vec<CausalWitnessFacet> = textures_m.iter().map(|t| t.kant).collect();
    let witt_facets: Vec<CausalWitnessFacet> = textures_m.iter().map(|t| t.wittgenstein).collect();

    println!("\n— DEGENERACY DISCIPLINE (full {n}-verse corpus) —");
    let mut live: Vec<(&str, &[CausalWitnessFacet])> = Vec::new();
    for (name, facets) in [
        ("Hegel", hegel_facets.as_slice()),
        ("Nietzsche", nietzsche_facets.as_slice()),
        ("Kant", kant_facets.as_slice()),
        ("Wittgenstein", witt_facets.as_slice()),
    ] {
        match degeneracy(facets) {
            Some(reason) => println!("  DEGENERATE — {name} EXCLUDED: {reason}"),
            None => {
                let fired = facets.iter().filter(|f| f.bound_count() > 0).count();
                println!("  {name}: live — bound on {fired}/{} verses", facets.len());
                live.push((name, facets));
            }
        }
    }

    // ── pairwise texture comparison, LIVE stances only ──
    println!(
        "\n— PAIRWISE TEXTURE (agreement_count over the shipped register, live stances only) —"
    );
    if live.len() < 2 {
        println!("  fewer than two live stances — no pairs to compare");
    }
    for i in 0..live.len() {
        for j in (i + 1)..live.len() {
            let (na, fa) = live[i];
            let (nb, fb) = live[j];
            let (mean, hist) = agreement_stats(fa, fb);
            println!("  {na} x {nb}: mean agreement_count = {mean:.4} over {n} verses; distribution = {hist:?}");
        }
    }
    println!(
        "  Structural ceiling (stated in advance, not a measured finding): every \
         stance here binds `Antecedent` plus at most one PRIVATE second locus \
         (Hegel: Quorum; Kant: Modal); no two stances ever write the same \
         private locus, so `agreement_count` across ANY pair above can reach \
         at most 1 (via Antecedent alone) — the design's own ceiling, not the \
         register's ceiling of 24."
    );

    println!("\n— PER-LOCUS BIND RATE (which loci drive the texture) —");
    println!(
        "  Hegel:        Antecedent={:.4}  Quorum={:.4}",
        bind_rate(&hegel_facets, Locus::Antecedent),
        bind_rate(&hegel_facets, Locus::Quorum)
    );
    println!(
        "  Nietzsche:    Antecedent={:.4}",
        bind_rate(&nietzsche_facets, Locus::Antecedent)
    );
    println!(
        "  Kant:         Antecedent={:.4}  Modal={:.4}",
        bind_rate(&kant_facets, Locus::Antecedent),
        bind_rate(&kant_facets, Locus::Modal)
    );
    println!(
        "  Wittgenstein: Antecedent={:.4}",
        bind_rate(&witt_facets, Locus::Antecedent)
    );
    println!(
        "  Every other named locus (Temporal/Kausal/Modal[for non-Kant \
         stances]/Lokal/SMeaning/PMeaning/OMeaning/BasinAnchor/SupportedBy/ \
         Supports/RunbookEvidence/QualiaReference/MeaningLevel/Contradiction) \
         reads 0.0000 for every stance here, always, by construction — this \
         harness's binding rules never write them, not a measured absence."
    );

    // ── horizon intervention (§12.3b/§12.3c falsifier 2) ──
    let k = HORIZON_K.min(n);
    println!(
        "\n— HORIZON INTERVENTION (§12.3c falsifier 2: Vk={k} vs Vm={n}, fixed verse set 0..{k}) —"
    );
    if k == n {
        println!(
            "  Vk == Vm (the corpus has {n} <= {HORIZON_K} verses) — the horizon \
             intervention needs Vm > Vk to be meaningful; skipped, reported \
             honestly rather than run on a degenerate (k==m) pair."
        );
    } else {
        let (arena_k, intern_k, out_k) = build(&verses[..k]);
        let (hegel_k, nietzsche_k, kant_k, witt_k) = stance_panel(&arena_k, &intern_k, &out_k);
        let index_k = VerseIndex::build(&verses[..k], &out_k);
        let panel_k = Panel {
            hegel: &hegel_k,
            nietzsche: &nietzsche_k,
            kant: &kant_k,
            wittgenstein: &witt_k,
        };
        let textures_k = mint_all(k, &out_k, &panel_k, &index_k);

        println!(
            "  Fusion, in this section only, names a measured change in binding \
             topology under the horizon intervention — never a validity claim \
             (§12.3c/§12.4). The word does not appear elsewhere in this file."
        );

        let hegel_k_f: Vec<CausalWitnessFacet> = textures_k.iter().map(|t| t.hegel).collect();
        let hegel_m_f: Vec<CausalWitnessFacet> = textures_m[..k].iter().map(|t| t.hegel).collect();
        rebind_report("Hegel", &hegel_k_f, &hegel_m_f);

        let niet_k_f: Vec<CausalWitnessFacet> = textures_k.iter().map(|t| t.nietzsche).collect();
        let niet_m_f: Vec<CausalWitnessFacet> =
            textures_m[..k].iter().map(|t| t.nietzsche).collect();
        rebind_report("Nietzsche", &niet_k_f, &niet_m_f);

        let kant_k_f: Vec<CausalWitnessFacet> = textures_k.iter().map(|t| t.kant).collect();
        let kant_m_f: Vec<CausalWitnessFacet> = textures_m[..k].iter().map(|t| t.kant).collect();
        rebind_report("Kant", &kant_k_f, &kant_m_f);

        let witt_k_f: Vec<CausalWitnessFacet> = textures_k.iter().map(|t| t.wittgenstein).collect();
        let witt_m_f: Vec<CausalWitnessFacet> =
            textures_m[..k].iter().map(|t| t.wittgenstein).collect();
        rebind_report("Wittgenstein", &witt_k_f, &witt_m_f);
    }

    println!(
        "\nblw_texture: done. No number above was predicted — every one is computed at run time."
    );
}
