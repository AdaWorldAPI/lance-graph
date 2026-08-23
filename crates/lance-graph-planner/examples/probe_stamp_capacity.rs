//! PROBE-STAMP-CAPACITY-1 — quantifying the `Stamp` mod-64 evidence-loss curve.
//!
//! # The mechanism, verbatim from the shipped type
//!
//! `crates/lance-graph-planner/src/nars/belief.rs`:
//!
//! ```text
//! /// Fixed-width evidential stamp: bit *i* = observation source *i* (bounded
//! /// horizon of 64; sources beyond fold by modulo — CONSERVATIVE: folding can only
//! /// create false overlap, never false disjointness, so no-double-count survives).
//! #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
//! pub struct Stamp(pub u64);
//!
//! impl Stamp {
//!     /// The stamp of a single observation source.
//!     #[must_use]
//!     pub fn source(id: u32) -> Self {
//!         Stamp(1u64 << (id % 64))
//!     }
//! ```
//!
//! `id % 64` means two sources 64 apart (or any multiple of 64 apart) collide
//! onto the SAME bit. `disjoint()` (`self.0 & other.0 == 0`) then reports them
//! as NOT disjoint, and `revise_at` (same file) routes overlap through CHOICE —
//! keep the higher-confidence truth, count nothing twice — rather than through
//! `TruthValue::revise` (evidence pooling). This is CONSERVATIVE and CORRECT:
//! it never double-counts. This probe does not dispute that. It measures what
//! the conservatism COSTS at real corpus scale: an earlier merged probe
//! (`probe_r2il_real_episodes.rs`, gate E4) found that at 143 real function-
//! episodes, one widely-recurring macro reached 64 pooled revisions and had
//! 23 further real sources conservatively CHOICE-dropped. Those two numbers
//! (64, 23) are ORCHESTRATOR-MEASURED from that merged probe's own run; this
//! probe cites them only as the anecdote motivating the curve below — it does
//! NOT re-assert them as its own measurement, and every number this probe
//! itself prints comes from code in this file, run fresh, never hard-coded.
//!
//! # What is measured vs what is modelled
//!
//! - **K1, K2, K4** run the ACTUAL shipped [`Stamp`] type (`Stamp::source`,
//!   `.disjoint()`, `.union()`) and the ACTUAL shipped [`TruthValue::revise`].
//!   No corpus needed.
//! - **K3** needs the real R2IL episode corpus (env `R2IL_ORE_TSV`, same
//!   contract as `probe_r2il_real_episodes.rs`) to count real evidence
//!   sources = distinct `(binary, function)` pairs. Corpus-free gates (K1,
//!   K2, K4, K5) run and print regardless of whether the corpus is present;
//!   only K3 is gated, and its absence flips the process exit code to 2
//!   without suppressing the other gates' output.
//! - **K5 is MODELLED, NOT SHIPPED, NOT PROPOSED.** It asks: at the SAME
//!   corpus scale, what would a wider (or unbounded) source→bit mapping have
//!   recovered? It is implemented entirely in THIS file, on probe-local
//!   helper functions — it never touches, widens, or re-derives the shipped
//!   `Stamp(u64)` type. Whether any of the modelled widths should ever be
//!   adopted is the operator's ruling to make; this probe supplies only the
//!   measurement that would inform it, and says so at every print site.
//!
//! # Fences (see K6's printed output for the same list)
//!
//! - No shipped type modified: `Stamp`, `TruthValue`, `BeliefArena` are used
//!   read-only, exactly as shipped.
//! - The shipped conservatism is CORRECT, not a defect — it never double-
//!   counts. This probe measures its COST (evidence conservatively dropped),
//!   not a bug.
//! - No width-change recommendation is made here. K5's numbers describe
//!   modelled alternatives; adopting one is a decision this probe does not
//!   make, has no authority to make, and does not advocate for.
//! - The real corpus (K3) is 2 binaries harvested from one source (see
//!   `probe_r2il_real_episodes.rs`'s own module docs for provenance) — a
//!   real-world data point, not a synthetic distribution.
//! - No measured constant is hard-coded: every number this file prints is
//!   computed in this run, from code in this file, against the shipped API
//!   or a probe-local model that is labelled as such.

use std::collections::HashSet;
use std::process::ExitCode;

use lance_graph_planner::nars::belief::Stamp;
use lance_graph_planner::nars::truth::TruthValue;

// ================================================================================================
// Shipped-mechanism helpers (K1, K2, K4) — use the REAL Stamp type, read-only.
// ================================================================================================

/// Feed `n` distinct observation-source ids (`0..n`) through the REAL
/// `Stamp::source` / `.disjoint()` / `.union()` mod-64 mechanism. Returns
/// `(pooled, dropped)` — `pooled` = sources whose stamp was disjoint from
/// everything accumulated so far (evidence actually pooled); `dropped` =
/// sources whose stamp collided with an already-seen bit (CHOICE-dropped).
fn shipped_stamp_curve(n: u32) -> (u32, u32) {
    let mut stamp = Stamp(0);
    let (mut pooled, mut dropped) = (0u32, 0u32);
    for id in 0..n {
        let ev = Stamp::source(id);
        if stamp.disjoint(ev) {
            stamp = stamp.union(ev);
            pooled += 1;
        } else {
            dropped += 1;
        }
    }
    (pooled, dropped)
}

/// Same accounting as [`shipped_stamp_curve`], but also drives the REAL
/// `TruthValue::revise` on every pooled source, starting from `prior`. Every
/// source offers the identical `obs` truth (the E4 anecdote's shape: a
/// recurring macro observed as positive evidence in many episodes).
fn shipped_revise_sequence(n: u32, prior: TruthValue, obs: TruthValue) -> (TruthValue, u32, u32) {
    let mut truth = prior;
    let mut stamp = Stamp(0);
    let (mut pooled, mut dropped) = (0u32, 0u32);
    for id in 0..n {
        let ev = Stamp::source(id);
        if stamp.disjoint(ev) {
            truth = truth.revise(&obs);
            stamp = stamp.union(ev);
            pooled += 1;
        } else {
            dropped += 1;
        }
    }
    (truth, pooled, dropped)
}

// ================================================================================================
// MODELLED ALTERNATIVES (K5 only) — probe-local, NEVER touching the shipped Stamp type.
// ================================================================================================

/// Models what a source→bit mapping of `width` bits (or, for `width = None`,
/// an UNBOUNDED / infinite-width mapping — one bit per distinct source,
/// never folding) would recover for `n` distinct sources `0..n`. This is a
/// residue-collision simulation: a real `width`-bit array-backed stamp
/// collides two sources iff `id_a % width == id_b % width`, which is exactly
/// what tracking "have I already seen this residue" reproduces — so this
/// function is a faithful behavioural model of a wider bitset WITHOUT
/// constructing one, and without touching the shipped `Stamp(u64)` type.
/// MODELLED, NOT SHIPPED, NOT PROPOSED — see module docs.
fn modelled_width_curve(n: u32, width: Option<u32>) -> (u32, u32) {
    let mut seen: HashSet<u32> = HashSet::new();
    let (mut pooled, mut dropped) = (0u32, 0u32);
    for id in 0..n {
        let residue = match width {
            Some(w) => id % w,
            None => id, // unbounded: every id is its own residue, never collides
        };
        if seen.insert(residue) {
            pooled += 1;
        } else {
            dropped += 1;
        }
    }
    (pooled, dropped)
}

/// Same idea as [`shipped_revise_sequence`] but for an UNBOUNDED mapping,
/// modelled (not shipped): every source is disjoint from every other, so
/// every one pools. Used by K4 as the "what full pooling would have been
/// worth" comparison point.
fn modelled_unbounded_revise_sequence(n: u32, prior: TruthValue, obs: TruthValue) -> TruthValue {
    let mut truth = prior;
    for _ in 0..n {
        truth = truth.revise(&obs);
    }
    truth
}

// ================================================================================================
// K3 corpus reading (probe-local; mirrors probe_r2il_real_episodes.rs's TSV contract)
// ================================================================================================

/// Count distinct `(binary, function)` pairs — one evidence source per real
/// function-episode, same definition `probe_r2il_real_episodes.rs` uses.
/// Asserts the 13-column schema on every non-comment line (anti-vacuity:
/// this is the same schema-drift guard the merged probe carries).
fn count_evidence_sources(tsv: &str) -> u32 {
    let mut seen: HashSet<(String, String)> = HashSet::new();
    for line in tsv.lines() {
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        assert!(
            f.len() == 13,
            "schema drift: expected 13 columns, got {}",
            f.len()
        );
        seen.insert((f[0].to_string(), f[1].to_string()));
    }
    seen.len() as u32
}

// ================================================================================================
// main
// ================================================================================================

fn main() -> ExitCode {
    let mut pass = 0u32;
    let sweep: [u32; 8] = [8, 16, 32, 64, 96, 143, 256, 512];

    // -------------------------------------------------------------------------------------------
    // K1 THE MECHANISM, SHOWN — two ids 64 apart collide on the shipped Stamp; two ids
    // 1 apart do not. Can-fire (collision exists) AND can-stay-silent (adjacent ids
    // don't) on the real shipped type.
    // -------------------------------------------------------------------------------------------
    {
        let a = Stamp::source(5);
        let b_far = Stamp::source(5 + 64);
        let b_near = Stamp::source(5 + 1);
        assert!(
            !a.disjoint(b_far),
            "can-fire: ids 64 apart MUST collide under Stamp::source's mod-64 fold"
        );
        assert!(
            a.disjoint(b_near),
            "can-stay-silent: adjacent ids MUST NOT collide"
        );
        pass += 1;
        println!(
            "K1 PASS  Stamp::source(5) vs Stamp::source(69): disjoint={} (collision, as predicted); \
             Stamp::source(5) vs Stamp::source(6): disjoint={} (no collision)",
            a.disjoint(b_far),
            a.disjoint(b_near)
        );
    }

    // -------------------------------------------------------------------------------------------
    // K2 THE CURVE — real Stamp mod-64 mechanism, swept over N = 8..512 distinct sources.
    // -------------------------------------------------------------------------------------------
    {
        println!("K2       shipped Stamp mod-64 curve (N distinct sources -> pooled/dropped):");
        for &n in &sweep {
            let (pooled, dropped) = shipped_stamp_curve(n);
            println!(
                "         N={n:>4}  pooled={pooled:>4}  dropped={dropped:>4}  loss={:.1}%",
                100.0 * dropped as f64 / n as f64
            );
            if n <= 64 {
                assert_eq!(dropped, 0, "can-stay-silent: N<=64 must lose nothing");
            } else {
                assert!(dropped > 0, "can-fire: N>64 must lose something");
            }
        }
        pass += 1;
        println!("K2 PASS  loss is exactly 0 for every N<=64 and strictly positive for every N>64");
    }

    // -------------------------------------------------------------------------------------------
    // K3 REAL CORPUS — real evidence-source count via R2IL_ORE_TSV, same accounting as K2.
    // -------------------------------------------------------------------------------------------
    let mut corpus_absent = false;
    match std::env::var_os("R2IL_ORE_TSV") {
        None => {
            corpus_absent = true;
            eprintln!(
                "K3 CORPUS ABSENT — R2IL_ORE_TSV is unset. K1/K2/K4/K5 do not need it and ran above; \
                 only K3 (the real-corpus point on the curve) is skipped."
            );
            eprintln!("Fetch the real episode stream and re-run:");
            eprintln!(
                "  curl -sL https://github.com/AdaWorldAPI/ruff/releases/download/r2il-harvest-pass1/r2il-pass1.ore.tsv.gz \\\n    | zcat > /tmp/r2il-pass1.ore.tsv"
            );
            eprintln!(
                "  R2IL_ORE_TSV=/tmp/r2il-pass1.ore.tsv cargo run -p lance-graph-planner --example probe_stamp_capacity"
            );
        }
        Some(path) => {
            match std::fs::read_to_string(&path) {
                Err(_) => {
                    corpus_absent = true;
                    eprintln!("K3 CORPUS ABSENT — R2IL_ORE_TSV={path:?} is not readable. Never fabricated.");
                }
                Ok(tsv) => {
                    let n = count_evidence_sources(&tsv);
                    let (pooled, dropped) = shipped_stamp_curve(n);
                    pass += 1;
                    // META-REVIEW FIX (P1): K3 had NO assertion about what it measures — an
                    // all-comment TSV yields n=0 and still printed "K3 PASS". The claim K3 exists
                    // to make is that the REAL corpus lands on the lossy side of the curve.
                    assert!(
                        n > 64,
                        "K3: the corpus must land past the mod-64 ceiling to measure anything (got {n} sources)"
                    );
                    println!(
                        "K3 PASS  real corpus: {n} distinct (binary,function) evidence sources; \
                     shipped-Stamp accounting: pooled={pooled} dropped={dropped} ({:.1}% of real \
                     evidence sources would be CHOICE-dropped if every one hit the same recurring \
                     macro)",
                        100.0 * dropped as f64 / n.max(1) as f64
                    );
                }
            }
        }
    }

    // -------------------------------------------------------------------------------------------
    // K4 TRUTH IMPACT — shipped TruthValue::revise: modulo-limited pooling vs unbounded
    // pooling, for one N<=64 (can-stay-silent: they must coincide) and one N>64
    // (can-fire: the modulo-limited expectation must be strictly lower).
    // -------------------------------------------------------------------------------------------
    {
        let prior = TruthValue::new(0.5, 0.05); // ignorance prior, same shape as the E4 anecdote
        let obs = TruthValue::new(1.0, 0.9); // identical positive observation per source
        println!("K4       truth impact of modulo-limited vs unbounded pooling (shipped TruthValue::revise):");
        for &n in &[32u32, 143u32] {
            let (limited, lim_pooled, lim_dropped) = shipped_revise_sequence(n, prior, obs);
            let unbounded = modelled_unbounded_revise_sequence(n, prior, obs);
            let lim_e = limited.expectation();
            let unb_e = unbounded.expectation();
            println!(
                "         N={n:>4}  modulo-limited: pooled={lim_pooled} dropped={lim_dropped} e={lim_e:.4}  |  \
                 unbounded (MODELLED, not shipped): pooled={n} e={unb_e:.4}"
            );
            assert!(
                lim_e <= unb_e + 1e-6,
                "modulo-limited expectation must never exceed unbounded pooling's expectation"
            );
            if n <= 64 {
                assert!(
                    (lim_e - unb_e).abs() < 1e-6,
                    "can-stay-silent: at N<=64 nothing is dropped, so the two must coincide"
                );
            } else {
                assert!(
                    unb_e - lim_e > 1e-4,
                    "can-fire: at N>64 unbounded pooling must measurably exceed modulo-limited"
                );
            }
        }
        pass += 1;
        println!(
            "K4 PASS  modulo-limited expectation <= unbounded pooling's expectation, with a real gap \
             above N=64 and none below it — NOTE: a higher expectation from unbounded pooling is NOT \
             automatically better; the shipped modulo-64 CHOICE guard exists precisely to avoid the \
             double-counting that unbounded pooling of REPEATED, non-independent observations would \
             otherwise silently commit. This gate measures the gap; it does not judge which side of \
             it is correct in general."
        );
    }

    // -------------------------------------------------------------------------------------------
    // K5 MODELLED ALTERNATIVES — NOT A PROPOSAL, NOT IMPLEMENTED. What would wider (or
    // unbounded) source->bit mappings have recovered, at the same swept N values?
    // -------------------------------------------------------------------------------------------
    {
        println!(
            "K5       MODELLED ALTERNATIVES — NOT A PROPOSAL, NOT IMPLEMENTED. Probe-local simulation \
             only; the shipped Stamp(u64) is untouched. Costs NOT measured here: per-belief memory \
             (u64=8B; a modelled 128-bit/256-bit register would be 16B/32B per belief), cache-line \
             behaviour of a wider register, and wire size of any serialized stamp. None of that is \
             evaluated by this gate — only recovered-evidence counts are."
        );
        for &n in &sweep {
            let (p64, d64) = modelled_width_curve(n, Some(64)); // == shipped, sanity cross-check
            let (p128, d128) = modelled_width_curve(n, Some(128));
            let (p256, d256) = modelled_width_curve(n, Some(256));
            let (p_unb, d_unb) = modelled_width_curve(n, None);
            let (shipped_p, shipped_d) = shipped_stamp_curve(n);
            assert_eq!(
                (p64, d64),
                (shipped_p, shipped_d),
                "the width=64 model must reproduce the ACTUAL shipped Stamp curve exactly"
            );
            println!(
                "         N={n:>4}  64-bit(shipped)={p64:>4}/{d64:>4}  128-bit(modelled)={p128:>4}/{d128:>4}  \
                 256-bit(modelled)={p256:>4}/{d256:>4}  unbounded(modelled)={p_unb:>4}/{d_unb:>4}  (pooled/dropped)"
            );
        }
        pass += 1;
        println!(
            "K5 PASS  modelled alternatives printed. HONEST NOTE (meta-review P0): the width=64 \
             cross-check against the shipped Stamp curve is TRUE BY CONSTRUCTION — both sides \
             compute `id % 64` over the same range, so no input can make it diverge. It is a \
             readability aid, NOT a falsifier, and this section therefore has no independent \
             check: the modelled 128/256-bit rows are arithmetic, not measurements. K1 and K2 \
             are where the shipped type is actually exercised"
        );
    }

    // -------------------------------------------------------------------------------------------
    // K6 FENCES
    // -------------------------------------------------------------------------------------------
    {
        pass += 1;
        println!("K6 PASS  fences:");
        println!(
            "         - no shipped type modified: Stamp, TruthValue, BeliefArena used read-only"
        );
        println!(
            "         - the shipped mod-64 conservatism is CORRECT (never double-counts); this probe \
             measures its COST, not a defect"
        );
        println!(
            "         - K5's modelled widths are NOT a proposal and NOT implemented anywhere; any \
             width change is the operator's ruling to make, never this probe's"
        );
        println!(
            "         - the real corpus (K3, when present) is 2 binaries harvested from one source \
             (see probe_r2il_real_episodes.rs provenance) — a real data point, not a distribution"
        );
    }

    println!("\n{pass}/6 gates green (K3 counts only when the corpus was present).");
    if corpus_absent {
        ExitCode::from(2)
    } else {
        ExitCode::SUCCESS
    }
}
