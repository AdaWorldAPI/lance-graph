//! `blw_lens_twin` — D-BLW-2's discrimination twin, over ONLY what is
//! reachable from `/tmp/kjv_spo.tsv` (the `deepnsm-v2::bible_wave --export`
//! output) plus a [`BeliefArena`] ingested from it exactly as
//! `examples/reason_whole_book.rs` does (lines ~55-96 of that file: one
//! `arena.observe(CStmt { s, cop, p: o }, TruthValue::new(1.0, 0.9),
//! Stamp::source(v))` per TSV row; `close_transitive` is deliberately NOT
//! run here — see "why no `close_transitive`" below).
//!
//! Design authority: `.claude/plans/cycle-loop-closure-driver-v1.md` §12.3a
//! (the four-stance adjudication) and §12.4 (the claim ceiling). Read those
//! before touching this file — this doc comment summarizes the parts that
//! bear on THIS harness, not the whole arm.
//!
//! # Reachability — the crux, confronted rather than routed around
//!
//! §12.3a already proves two of the four B6-panel stances dead on the TSV
//! path (Hegel constant-false; Nietzsche starved of `Provenance.negated`).
//! This harness independently re-verifies both AND extends the finding to
//! the other two — **three of four are unreachable**, not two, and the
//! fourth is reachable only in a reduced form:
//!
//! | stance | reachable? | why |
//! |---|---|---|
//! | **Hegel** | reachable, but **DEGENERATE** (proven constant-false) | `reason_whole_book.rs:92-96`-style ingestion always calls `TruthValue::new(1.0, 0.9)` — no polarity field exists in the TSV to vary it — so `revise_at`'s `depth = (b.truth.frequency - new.frequency).abs()` (`belief.rs:194`) is `0.0` for every re-observation. `Belief.contradiction` never leaves `0.0`; `contradiction_ranking`'s `> 0.05` filter (`stance.rs:411`) is empty for the whole book, always. |
//! | **Nietzsche** | **UNREACHABLE** | needs `Provenance.negated` (`stance.rs:87`), populated only inside `stance::stream()`'s cue-driven clause machine (`is_negation` over raw verse TEXT, `stance.rs:211-214`). `deepnsm_v2::Spo` carries no polarity field and the TSV's 7 columns (`subject_id, subject_word, predicate_id, predicate_word, object_id, object_word, verse_index`) have no negation column either — there is no `Provenance` list to build from the TSV at all, not a degraded one. Missing field: per-triple polarity. **Owning crate: `deepnsm-v2`** (the `Spo` triple type / the `bible_wave` FSM exporter). |
//! | **Kant** | **UNREACHABLE** (this is the finding beyond §12.3a's own two) | the §12.3a-corrected, rank-based Kant binary needs `RungLift` (`stance.rs:93-116`: knower, verb, inner statement, and — for the reflexivity read the panel actually uses — an OVERTLY re-anchored inner subject). Every `RungLift` is minted inside `stance::stream()`'s "that"-complementizer window (`stance.rs:185-233`), which requires labelled, raw verse TEXT (`verses: &[(String, String)]`, `stance.rs:151`). The TSV's flat `(subject, predicate, object, verse_index)` triples do not preserve clause nesting — there is no way to tell, from a TSV row alone, whether a predicate like "knew" introduced a that-complement versus a flat transitive relation. **Missing: labelled raw verse text (equivalently, the knows-that clause structure) in the TSV export.** The machinery to CONSUME it (`stance::stream`) already lives in `lance-graph-planner`; what's missing is the INPUT, and that input is `deepnsm-v2`'s to produce — `bible_wave`'s TSV export mode has no verse-text column, only interned word ids. **Owning crate: `deepnsm-v2`.** |
//! | **Wittgenstein** | reachable, **REDUCED** | `stance_panel`'s Wittgenstein reads THREE sources: `arena.entries()` for `"inh-subj"`/`"inh-obj"` games (needs only the arena — reachable), `out.lifts` for `"rel-subj"`/`"rel-obj"` (needs `RungLift` — unreachable, same as Kant), and `out.impls` for `"impl-cause"`/`"impl-effect"` (needs `stance::stream()`'s causal-cue detection over raw text — unreachable). Calling the REAL `stance_panel` with `ReadOut::default()` (the only `ReadOut` the TSV path can supply) naturally degrades it to the Inh-only 2-of-6-category form — no reimplementation, no invented logic, just the shipped function fed its honestly-available inputs. |
//!
//! **So: at most TWO of four lenses produce any per-verse signal from
//! TSV+arena alone (Hegel, Wittgenstein-reduced), and one of those two
//! (Hegel) is provably degenerate.** That leaves at most ONE non-degenerate
//! lens — with only one lens, there are ZERO pairs to run the discrimination
//! twin over. This is reported as a structural KILL, not papered over: see
//! "the twin" section of the output.
//!
//! # Why no `close_transitive`
//!
//! `reason_whole_book.rs` calls `arena.close_transitive(64)` after ingest.
//! This harness deliberately does NOT: `close_transitive`'s only effect on
//! the arena is `admit_derived`-path insertions, which (a) always set
//! `contradiction: 0.0` (`belief.rs`'s `admit_derived`, both branches) — so
//! it cannot un-degenerate Hegel — and (b) always carry `stamp:
//! Stamp::default()` — which `stance_panel`'s Wittgenstein Inh-game filter
//! explicitly excludes (`b.stamp != Stamp::default()`, `stance.rs:506`), so
//! derived entries are invisible to Wittgenstein too. Running closure here
//! would cost real time (146,676-entry arena on the full book, per the
//! `reason_whole_book` run this harness's TSV came from) for zero effect on
//! either reachable stance — so it is skipped, matching the ingest-only
//! scope the task specifies ("lines ~55-96").
//!
//! # The Wittgenstein per-verse projection is A DESIGN CHOICE, disclosed
//!
//! `stance_panel`'s Wittgenstein output is keyed by CONCEPT
//! (`Vec<(u16, usize)>` — a concept id and its distinct-game count), not by
//! verse. Neither §12.3 nor §12.3a specifies how to project a concept-keyed
//! panel output into a per-verse binary. This harness's choice, stated
//! plainly so it is never mistaken for a definition lifted verbatim from the
//! plan: **a verse's Wittgenstein(reduced) bit is `true` iff any concept
//! mentioned in that verse's triples (as subject or object id) carries ≥ 2
//! distinct games somewhere in the whole corpus** — i.e. the verse touches a
//! concept the corpus elsewhere shows playing BOTH the inh-subject and
//! inh-object role. Hegel's projection is more direct and needs no such
//! choice: a verse's bit is `true` iff any statement OBSERVED AT that verse
//! is present in `contradiction_ranking`'s output.
//!
//! # Two §12.3a diagnostics, and why only one applies to this path
//!
//! §12.3a names two mandatory diagnostics. (a) the pronoun-collision share
//! is a property of `stance::stream()`'s pronoun-to-`"they"` normalization
//! (`stance.rs:196-207`) — this harness's TSV-ingestion path performs NO
//! pronoun normalization at all (subjects/objects are `bible_wave`'s own
//! trained-codebook word ids, unrelated machinery), so the diagnostic has no
//! referent here and is reported as such rather than computed against the
//! wrong pipeline. (b) `Stamp::source(id) = 1 << (id % 64)` saturation
//! (`belief.rs:37`) DOES apply — verse indices collide mod 64 constantly
//! over a 31k-verse book, and IS computed below.
//!
//! # Claim ceiling (§12.4), binding on every line this program prints
//!
//! Overlap only. Never "valid"/"accurate"/"better"/"confirms". No p-value:
//! `jc::stats` p-values are classical independent-sample values and verses
//! within a book are domain-correlated (I-NOISE-FLOOR-JIRAK), so they do not
//! apply unmodified here. Never "Horizontverschmelzung"/"fusion" — that is
//! D-BLW-3, untouched by this harness. Bare κ is never printed without its
//! counts and both marginals.
//!
//! # Modes
//!
//! `cargo run -p lance-graph-planner --example blw_lens_twin [-- <tsv path>]`
//! — defaults to `/tmp/kjv_spo.tsv`. If that file is absent, runs a tiny
//! deterministic synthetic fixture instead, which exercises the degeneracy
//! machinery (DEGENERATE / UNSTABLE / `binary_association`-returns-`None`,
//! each proven to both fire AND stay silent) and explicitly does **not**
//! claim the twin — its verse count is far below the `N ≥ 1,000` corpus
//! floor.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use jc::stats::{binary_association, BinaryAssociation};
use lance_graph_planner::nars::stance::{stance_panel, Interner, ReadOut};
use lance_graph_planner::nars::{BeliefArena, CStmt, Copula, Stamp, TruthValue};

/// Landis–Koch "almost perfect" floor — the can-discriminate ceiling on κ.
/// Pre-registered in §12.3a; non-adjustable after any run.
const KAPPA_DISCRIMINATE_MAX: f64 = 0.80;
/// Landis–Koch slight/fair boundary — the can-agree floor on κ.
const KAPPA_AGREE_MIN: f64 = 0.20;
/// Corpus floor (§12.3a): below this the marginals are too noisy to read and
/// the twin is not reported at all.
const CORPUS_FLOOR: usize = 1_000;
/// can-discriminate's count clause: the discordant share must clear 5% of N.
const DISCORD_SHARE_MIN: f64 = 0.05;
/// A lens whose positive rate falls outside this band is DEGENERATE —
/// excluded from BOTH ∃-quantifiers, exclusion always printed.
const DEGENERATE_LOW: f64 = 0.01;
const DEGENERATE_HIGH: f64 = 0.99;
/// can-agree's own marginal guard — tighter than the DEGENERATE band, and a
/// DIFFERENT band (§12.3a states both explicitly; they must not be conflated).
const CAN_AGREE_MARGIN_LOW: f64 = 0.05;
const CAN_AGREE_MARGIN_HIGH: f64 = 0.95;
/// A pair whose expected agreement clears this is UNSTABLE — barred from
/// can-agree (a near-constant match is not evidence of agreement).
const UNSTABLE_EXPECTED_AGREEMENT: f64 = 0.95;

/// Which B6-panel stance a per-verse binary came from. `Nietzsche` and
/// `Kant` never reach a [`LensVerdict`] — they are UNREACHABLE and are
/// reported as such, not computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Lens {
    /// Aufhebung ranking, projected per-verse: reachable, proven degenerate.
    Hegel,
    /// Inh-subject/Inh-object breadth only (2 of 6 B6 categories):
    /// reachable in reduced form.
    WittgensteinReduced,
}

/// The copula test `reason_whole_book.rs` uses (its `is_copular`,
/// reproduced verbatim — a fixed keyword-membership catalogue, not a
/// parser). Copular predicates are `Inh` (transitive `is_a`); everything
/// else is a stored, never-transitive `Rel` verb (the S3 gate).
fn is_copular(word: &str) -> bool {
    matches!(
        word,
        "is" | "was"
            | "are"
            | "were"
            | "be"
            | "been"
            | "being"
            | "am"
            | "art"
            | "wast"
            | "become"
            | "became"
            | "becometh"
    )
}

/// The result of ingesting a `bible_wave --export` TSV into a
/// [`BeliefArena`], plus the per-verse index this harness needs (and that
/// `reason_whole_book.rs` does not build, since it never projects per-verse).
struct Ingested {
    arena: BeliefArena,
    /// Every distinct verse index, sorted, paired with the [`CStmt`]s
    /// OBSERVED at it (a triple's own verse — not derived, not revised-into).
    by_verse: Vec<(u32, Vec<CStmt>)>,
    n_rows: u64,
}

/// Ingest TSV text (already read into memory) exactly as
/// `reason_whole_book.rs:55-96` ingests a file — one `arena.observe` per
/// row, `TruthValue::new(1.0, 0.9)`, `Stamp::source(verse_index)` — plus the
/// per-verse grouping this harness needs for the per-verse projection.
fn ingest_str(raw: &str) -> Ingested {
    let mut arena = BeliefArena::new();
    let mut per_verse: HashMap<u32, Vec<CStmt>> = HashMap::new();
    let mut n_rows = 0u64;
    for line in raw.lines() {
        let mut f = line.split('\t');
        let (Some(s), Some(_sw), Some(pid), Some(pw), Some(o), Some(_ow), Some(v)) = (
            f.next(),
            f.next(),
            f.next(),
            f.next(),
            f.next(),
            f.next(),
            f.next(),
        ) else {
            continue;
        };
        let (Ok(s), Ok(o), Ok(v)) = (s.parse::<u16>(), o.parse::<u16>(), v.parse::<u32>()) else {
            continue;
        };
        let cop = if is_copular(pw) {
            Copula::Inh
        } else {
            // SKIP an unparsable predicate id — do NOT fold it into `Rel(0)`.
            // `unwrap_or(0)` collapses every malformed row into ONE statement
            // identity, so distinct garbage rows read as re-observations of the
            // same statement and inflate the very re-observation counts the
            // stances are computed from. Same treatment as the s/o/v columns
            // above.
            let Ok(p) = pid.parse::<u16>() else {
                continue;
            };
            Copula::Rel(p)
        };
        let stmt = CStmt { s, cop, p: o };
        arena.observe(stmt, TruthValue::new(1.0, 0.9), Stamp::source(v));
        per_verse.entry(v).or_default().push(stmt);
        n_rows += 1;
    }
    let mut by_verse: Vec<(u32, Vec<CStmt>)> = per_verse.into_iter().collect();
    by_verse.sort_by_key(|(v, _)| *v);
    Ingested {
        arena,
        by_verse,
        n_rows,
    }
}

/// Read `path` and ingest it via [`ingest_str`].
fn ingest_file(path: &str) -> Ingested {
    let raw = std::fs::read_to_string(path).expect("read SPO tsv");
    ingest_str(&raw)
}

/// Hegel's per-verse binary: `true` iff any [`CStmt`] OBSERVED at that verse
/// is present in `contradiction_ranking`'s output (i.e. its final
/// `contradiction > 0.05`, §12.3a's floor — see `stance.rs`'s doc comment
/// for why 0.05 is not decorative).
fn hegel_bits(by_verse: &[(u32, Vec<CStmt>)], hegel: &[(CStmt, f32)]) -> Vec<bool> {
    let positive: HashSet<CStmt> = hegel.iter().map(|(stmt, _)| *stmt).collect();
    by_verse
        .iter()
        .map(|(_, stmts)| stmts.iter().any(|s| positive.contains(s)))
        .collect()
}

/// Wittgenstein(reduced)'s per-verse binary, per the DISCLOSED projection
/// choice above: `true` iff any concept mentioned in the verse (as subject
/// or object id) carries ≥ 2 distinct games in `stance_panel`'s output.
fn wittgenstein_bits(by_verse: &[(u32, Vec<CStmt>)], wittgenstein: &[(u16, usize)]) -> Vec<bool> {
    let breadth: HashMap<u16, usize> = wittgenstein.iter().copied().collect();
    by_verse
        .iter()
        .map(|(_, stmts)| {
            stmts.iter().any(|s| {
                breadth.get(&s.s).copied().unwrap_or(0) >= 2
                    || breadth.get(&s.p).copied().unwrap_or(0) >= 2
            })
        })
        .collect()
}

/// One lens's per-verse binary vector plus its pre-registered degeneracy
/// status, computed BEFORE any pairing (§12.3a: "compute each lens's
/// positive rate before pairing").
struct LensVerdict {
    lens: Lens,
    bits: Vec<bool>,
    positive_rate: f64,
    degenerate: bool,
}

/// Build a [`LensVerdict`]: a sanity check that the rate is a genuine
/// probability (never silently `NaN`/out-of-range — a real assert, distinct
/// from the DEGENERATE *classification*, which is a soft [0.01, 0.99] band
/// that a proven-constant lens like Hegel is EXPECTED to fall outside of;
/// §12.3a's prose states both in one breath but they are not the same test —
/// see the module doc comment's note on this if it recurs).
fn lens_verdict(lens: Lens, bits: Vec<bool>) -> LensVerdict {
    let n = bits.len();
    assert!(n > 0, "{lens:?}: empty per-verse vector");
    let positives = bits.iter().filter(|&&b| b).count();
    let rate = positives as f64 / n as f64;
    assert!(
        rate.is_finite() && (0.0..=1.0).contains(&rate),
        "{lens:?}: positive rate out of range: {rate}"
    );
    let degenerate = !(DEGENERATE_LOW..=DEGENERATE_HIGH).contains(&rate);
    if degenerate {
        println!(
            "  DEGENERATE: {lens:?} positive rate {rate:.6} outside [{DEGENERATE_LOW}, {DEGENERATE_HIGH}] — excluded from both ∃-quantifiers"
        );
    }
    LensVerdict {
        lens,
        bits,
        positive_rate: rate,
        degenerate,
    }
}

/// One pair's full contingency table plus the two gate flags §12.3a's
/// degeneracy handling requires: `eligible` (neither lens DEGENERATE) and
/// `unstable` (`expected_agreement` clears the ceiling — barred from
/// can-agree regardless of `eligible`).
struct PairReport {
    a: Lens,
    b: Lens,
    table: BinaryAssociation,
    eligible: bool,
    unstable: bool,
}

fn fmt_kappa(k: Option<f64>) -> String {
    match k {
        Some(v) => format!("{v:.4}"),
        None => "undefined(p_e=1)".to_string(),
    }
}

fn fmt_phi(p: Option<f64>) -> String {
    match p {
        Some(v) => format!("{v:.4}"),
        None => "undefined(constant)".to_string(),
    }
}

/// Cross-tabulate two lenses. `binary_association` returning `None` is a
/// KILL naming the pair (§12.3a) — never a silently-skipped row.
fn evaluate_pair(a: &LensVerdict, b: &LensVerdict) -> Option<PairReport> {
    let Some(table) = binary_association(&a.bits, &b.bits) else {
        println!(
            "  KILL: binary_association({:?}, {:?}) returned None — lengths {} vs {}",
            a.lens,
            b.lens,
            a.bits.len(),
            b.bits.len()
        );
        return None;
    };
    let unstable = table.expected_agreement > UNSTABLE_EXPECTED_AGREEMENT;
    if unstable {
        println!(
            "  UNSTABLE: {:?}×{:?} expected_agreement {:.4} > {UNSTABLE_EXPECTED_AGREEMENT} — barred from can-agree",
            a.lens, b.lens, table.expected_agreement
        );
    }
    let eligible = !a.degenerate && !b.degenerate;
    Some(PairReport {
        a: a.lens,
        b: b.lens,
        table,
        eligible,
        unstable,
    })
}

/// Print the full per-pair table (§12.3a: "never a bare κ") and fold the
/// pre-registered can-discriminate / can-agree ∃-quantifiers over every
/// pair, respecting `eligible` and `unstable`.
fn evaluate_and_print(pairs: &[PairReport]) -> (bool, bool) {
    let mut can_discriminate = false;
    let mut can_agree = false;
    let mut kappas: Vec<f64> = Vec::new();

    for pr in pairs {
        let t = &pr.table;
        let n = (t.n00 + t.n01 + t.n10 + t.n11) as f64;
        let discordant_share = (t.n01 + t.n10) as f64 / n;

        println!(
            "  {:?} x {:?}: n00={} n01={} n10={} n11={} N={:.0}",
            pr.a, pr.b, t.n00, t.n01, t.n10, t.n11, n
        );
        println!(
            "    positive_rate_a={:.4} positive_rate_b={:.4} p_o={:.4} p_e={:.4} kappa={} phi={} discordant_share={:.4}",
            t.positive_rate_a,
            t.positive_rate_b,
            t.observed_agreement,
            t.expected_agreement,
            fmt_kappa(t.kappa),
            fmt_phi(t.phi),
            discordant_share,
        );

        if let Some(k) = t.kappa {
            kappas.push(k);
        }

        let discriminate_math = t.kappa.is_some_and(|k| k <= KAPPA_DISCRIMINATE_MAX)
            && discordant_share >= DISCORD_SHARE_MIN;
        let agree_math = t.kappa.is_some_and(|k| k >= KAPPA_AGREE_MIN)
            && (CAN_AGREE_MARGIN_LOW..=CAN_AGREE_MARGIN_HIGH).contains(&t.positive_rate_a)
            && (CAN_AGREE_MARGIN_LOW..=CAN_AGREE_MARGIN_HIGH).contains(&t.positive_rate_b);
        let discriminate_pass = discriminate_math && pr.eligible;
        let agree_pass = agree_math && pr.eligible && !pr.unstable;

        println!(
            "    can-discriminate={discriminate_pass} can-agree={agree_pass} eligible(both non-degenerate)={} unstable={}",
            pr.eligible, pr.unstable
        );

        can_discriminate |= discriminate_pass;
        can_agree |= agree_pass;
    }

    match (
        kappas.iter().cloned().reduce(f64::min),
        kappas.iter().cloned().reduce(f64::max),
    ) {
        (Some(min), Some(max)) => {
            println!(
                "  kappa range across {} pair(s) with a defined kappa: min={min:.4} max={max:.4}",
                kappas.len()
            );
        }
        _ => println!("  no pair produced a defined kappa"),
    }

    (can_discriminate, can_agree)
}

/// The corpus run: ingest, reachability report, the two lens verdicts, the
/// diagnostics, the twin, and the verdict.
fn run_corpus(path: &str) {
    println!("=== BLW discrimination twin — corpus run over {path} ===");
    let Ingested {
        arena,
        by_verse,
        n_rows,
    } = ingest_file(path);
    let n_verses = by_verse.len();
    println!(
        "ingested {n_rows} rows across {n_verses} distinct verses; arena has {} observed statements (close_transitive NOT run — see module doc comment)",
        arena.entries().len()
    );

    if n_verses < CORPUS_FLOOR {
        println!(
            "N={n_verses} < the {CORPUS_FLOOR}-verse corpus floor (§12.3a) — the twin is NOT reported."
        );
        return;
    }

    println!("\n--- reachability ---");
    println!(
        "UNREACHABLE: Nietzsche — needs Provenance.negated (per-triple polarity). The TSV's 7 columns carry no negation field and deepnsm_v2::Spo has none either; there is no Provenance list to build from the TSV at all. Owning crate: deepnsm-v2."
    );
    println!(
        "UNREACHABLE: Kant — needs RungLift (knower/verb/inner-statement/overtly-reanchored-subject). RungLifts are minted only inside stance::stream()'s \"that\"-complementizer window, which requires labelled raw verse TEXT as input; the TSV's flat (subject,predicate,object,verse) triples do not preserve clause nesting. The consuming machinery (stance::stream) already lives in lance-graph-planner — the missing piece is the INPUT (labelled verse text), which bible_wave's TSV export mode does not carry. Owning crate: deepnsm-v2."
    );

    let intern = Interner::new();
    let out = ReadOut::default();
    let (hegel, nietzsche_gated, kant_gated, wittgenstein) = stance_panel(&arena, &intern, &out);
    // Sanity check on the gating claim above, not a fresh computation: with
    // an empty ReadOut (the only ReadOut the TSV path can supply), Nietzsche
    // (iterates the Hegel ranking) and Kant (maps over out.lifts) MUST come
    // back empty from the real, unmodified stance_panel.
    assert!(
        nietzsche_gated.is_empty(),
        "sanity: stance_panel's Nietzsche output must be empty given an empty ReadOut"
    );
    assert!(
        kant_gated.is_empty(),
        "sanity: stance_panel's Kant output must be empty given an empty ReadOut"
    );

    println!("\n--- Hegel (reachable, DEGENERATE by construction) ---");
    println!(
        "  raw Hegel-positive statements (contradiction > 0.05): {} (§12.3a point 1 predicts 0 — uniform TruthValue::new(1.0, _) means revise_at's |f1-f2| depth is always 0)",
        hegel.len()
    );
    let hegel_v = lens_verdict(Lens::Hegel, hegel_bits(&by_verse, &hegel));
    println!("  Hegel positive rate: {:.6}", hegel_v.positive_rate);
    assert_eq!(
        hegel_v.bits.len(),
        n_verses,
        "Hegel per-verse vector must cover every verse"
    );

    println!("\n--- Wittgenstein(reduced) — Inh-subject/Inh-object only, 2 of the panel's 6 categories ---");
    println!("  concepts carrying >= 1 game: {}", wittgenstein.len());
    let max_games = wittgenstein.iter().map(|(_, g)| *g).max().unwrap_or(0);
    println!(
        "  max distinct games observed: {max_games} (ceiling here is 2, not the panel's full 6 — rel-*/impl-* categories are empty by construction on this path)"
    );
    let witt_v = lens_verdict(
        Lens::WittgensteinReduced,
        wittgenstein_bits(&by_verse, &wittgenstein),
    );
    println!(
        "  Wittgenstein(reduced) positive rate: {:.6}",
        witt_v.positive_rate
    );
    assert_eq!(
        witt_v.bits.len(),
        n_verses,
        "Wittgenstein(reduced) per-verse vector must cover every verse"
    );

    println!("\n--- diagnostics (§12.3a) ---");
    println!(
        "  (a) pronoun-collision share: N/A on this path — pronoun-to-\"they\" normalization is a stance::stream()-only mechanism; the TSV-ingestion path (bible_wave's own trained codebook ids) performs no such normalization, so this diagnostic has no referent here."
    );
    let saturated = arena
        .entries()
        .iter()
        .filter(|b| b.stamp.0.count_ones() == 64)
        .count();
    println!(
        "  (b) beliefs with a fully-saturated stamp (all 64 source bits set): {saturated} / {}",
        arena.entries().len()
    );

    println!("\n--- the discrimination twin ---");
    let lenses = [&hegel_v, &witt_v];
    let mut pairs: Vec<PairReport> = Vec::new();
    for i in 0..lenses.len() {
        for j in (i + 1)..lenses.len() {
            if let Some(pr) = evaluate_pair(lenses[i], lenses[j]) {
                pairs.push(pr);
            }
        }
    }
    println!(
        "  {} of 4 candidate lenses are reachable (Hegel, Wittgenstein-reduced) → {} pair(s); Nietzsche and Kant contribute zero pairs.",
        lenses.len(),
        pairs.len()
    );
    // The prose names SIX pairs (all 4 lenses reachable = C(4,2)); the guard
    // must use that number. At `< 2` a 3-lens run (3 pairs) printed nothing
    // while the full-table discipline still did not apply.
    const FULL_PANEL_PAIRS: usize = 6;
    if pairs.len() < FULL_PANEL_PAIRS {
        println!(
            "  note: §12.3a's \"assert the six tables are not all identical\" / full-table discipline assumes all 4 lenses reachable (6 pairs). With {} pair(s) reachable that comparison does not apply and is not attempted here.",
            pairs.len()
        );
    }
    let (can_discriminate, can_agree) = evaluate_and_print(&pairs);

    println!("\n--- verdict ---");
    println!("  can-discriminate: {can_discriminate}");
    println!("  can-agree: {can_agree}");
    if !can_discriminate && !can_agree {
        println!(
            "  KILL — structural, not a threshold miss on real data: the twin needs >= 2 reachable, non-degenerate lenses to form a pair. Hegel is DEGENERATE by construction (positive rate 0.0); only Wittgenstein(reduced) survives, and one lens cannot be paired with itself. 0 eligible pairs → both ∃-quantifiers are FALSE by construction."
        );
    }

    println!("\n--- claim ceiling (§12.4) ---");
    println!(
        "  kappa/phi above measure OVERLAP, not validity. No p-value is reported: jc::stats p-values are classical independent-sample values, and verses within a book are domain-correlated (I-NOISE-FLOOR-JIRAK), so they do not apply unmodified here."
    );
}

/// The synthetic mode: NOT a corpus claim. Proves each degeneracy path can
/// both FIRE and STAY SILENT, per the workspace's falsifiability rule (a
/// guard that always fires carries the same zero information as one that
/// never fires).
fn run_synthetic_smoke_test() {
    println!("=== BLW discrimination twin — SYNTHETIC smoke test (NOT a corpus claim) ===");
    println!(
        "this fixture is far below the N >= {CORPUS_FLOOR} verse corpus floor (§12.3a); it exercises the degeneracy machinery ONLY and asserts nothing about the KJV corpus"
    );

    // ── DEGENERATE: can-fire (Hegel — constant-false on ANY TSV-ingested
    // arena, §12.3a point 1) and can-stay-silent (Wittgenstein-reduced — a
    // genuinely mixed positive rate on this fixture). ──
    let tsv = "1\tone\t900\tis\t2\ttwo\t0\n\
               2\ttwo\t900\tis\t3\tthree\t1\n\
               4\tfour\t901\tchased\t5\tfive\t2\n\
               3\tthree\t900\tis\t4\tfour\t3\n\
               6\tsix\t900\tis\t7\tseven\t4\n\
               7\tseven\t900\tis\t8\teight\t5\n";
    let Ingested {
        arena,
        by_verse,
        n_rows,
    } = ingest_str(tsv);
    assert_eq!(n_rows, 6, "fixture must parse to exactly 6 rows");
    assert_eq!(by_verse.len(), 6, "fixture must span exactly 6 verses");

    let intern = Interner::new();
    let out = ReadOut::default();
    let (hegel, nietzsche, kant, wittgenstein) = stance_panel(&arena, &intern, &out);
    assert!(
        hegel.is_empty(),
        "Hegel must be constant-false on ANY uniform-frequency TSV ingest"
    );
    assert!(
        nietzsche.is_empty() && kant.is_empty(),
        "both must be gated to empty by the empty ReadOut"
    );

    let hegel_v = lens_verdict(Lens::Hegel, hegel_bits(&by_verse, &hegel));
    assert!(
        hegel_v.degenerate,
        "can-fire: DEGENERATE must fire on Hegel's constant-false vector"
    );
    assert!(
        hegel_v.positive_rate.abs() < f64::EPSILON,
        "Hegel positive rate must be exactly 0.0, got {}",
        hegel_v.positive_rate
    );

    let witt_v = lens_verdict(
        Lens::WittgensteinReduced,
        wittgenstein_bits(&by_verse, &wittgenstein),
    );
    assert!(
        !witt_v.degenerate,
        "can-stay-silent: DEGENERATE must NOT fire on Wittgenstein-reduced's mixed rate ({:.4})",
        witt_v.positive_rate
    );
    println!(
        "  DEGENERATE can-fire (Hegel, rate {:.4}) and can-stay-silent (Wittgenstein-reduced, rate {:.4}) both verified",
        hegel_v.positive_rate, witt_v.positive_rate
    );

    // ── UNSTABLE: can-fire (near-constant matching vectors) and
    // can-stay-silent (a balanced pair). ──
    let rare_a: Vec<bool> = (0..40).map(|i| i == 0).collect();
    let rare_b = rare_a.clone();
    let unstable_table =
        binary_association(&rare_a, &rare_b).expect("well-formed input must return Some");
    assert!(
        unstable_table.expected_agreement > UNSTABLE_EXPECTED_AGREEMENT,
        "can-fire: UNSTABLE must fire on near-constant matching vectors (p_e={:.4})",
        unstable_table.expected_agreement
    );

    let balanced_a = vec![true, false, true, false];
    let balanced_b = vec![true, true, false, false];
    // `.expect` here is itself the can-stay-silent proof for the
    // binary_association None-path below: a well-formed pair returning
    // `Some` is exactly "did not spuriously KILL".
    let balanced_table =
        binary_association(&balanced_a, &balanced_b).expect("well-formed input must return Some");
    assert!(
        balanced_table.expected_agreement <= UNSTABLE_EXPECTED_AGREEMENT,
        "can-stay-silent: UNSTABLE must NOT fire on a balanced pair (p_e={:.4})",
        balanced_table.expected_agreement
    );
    println!(
        "  UNSTABLE can-fire (p_e={:.4}) and can-stay-silent (p_e={:.4}) both verified",
        unstable_table.expected_agreement, balanced_table.expected_agreement
    );

    // ── binary_association's None path: can-fire (KILL naming the pair).
    // can-stay-silent is already proven above — `balanced_table` exists
    // only because that same call returned `Some`. ──
    let empty: Vec<bool> = Vec::new();
    assert!(
        binary_association(&empty, &empty).is_none(),
        "can-fire: binary_association(empty, empty) must KILL with None"
    );
    println!("  binary_association's None-path can-fire and can-stay-silent both verified");

    println!(
        "\nsynthetic smoke test PASSED — this does NOT constitute a corpus-scale twin claim (N={} << {CORPUS_FLOOR})",
        by_verse.len()
    );
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/kjv_spo.tsv".to_string());
    if Path::new(&path).is_file() {
        run_corpus(&path);
    } else {
        println!(
            "no TSV at {path} — running the synthetic degeneracy-machinery smoke test instead\n"
        );
        run_synthetic_smoke_test();
    }
}
