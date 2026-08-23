//! PROBE-R2IL-SLAG-BOUNDARY-1 — the HONEST BOUNDARY of the R2IL pass-1 seven-
//! opcode furnace convention: what it could NOT classify, and whether that
//! residual is a small set of named, bounded reasons (legitimate slag) or a
//! diffuse, unbounded catch-all (a sign the convention itself is wrong).
//!
//! # What the data is
//!
//! `probe_r2il_real_episodes.rs` (sibling probe, read in full before writing
//! this one) measures the INSIDE of the ruff-side R2IL pass-1 convention — the
//! classified `FlatFact` rows. This probe measures the OUTSIDE: the furnace's
//! own `ResidualLedger` — every fact the seven-opcode convention could not
//! carry, recorded under an explicit, non-empty reason (never silently
//! dropped, never a catch-all "other"). Two env-gated artifacts, neither
//! fabricated:
//!
//! - `R2IL_SLAG_TSV` — the slag ledger, two sections keyed by column 0:
//!   `grouped` (from `ResidualLedger::grouped()`): 5 columns — `section
//!   shape_id reason count example_facet`. `by_address` (from
//!   `ResidualLedger::by_address()`, the proposer's work queue): **4
//!   columns**, NOT 5 — `section prefix shape_id count`. No `reason` column on
//!   that side; a `by_address` row's reason is only recoverable by joining its
//!   `shape_id` back into `grouped`.
//!
//!   **Schema-drift note (found by reading the real file, not assumed):** the
//!   brief for this probe described both sections as 5-column with a `reason`
//!   field. The file's own header lines (`#schema section shape_id reason
//!   count example_facet` vs `#schema section prefix shape_id count`) say
//!   otherwise; this probe parses what the file actually declares. S1 encodes
//!   this as the per-section column arity.
//!
//! - `R2IL_ORE_TSV` — the classified-rows TSV (same file
//!   `probe_r2il_real_episodes` reads): 13 tab-separated columns. Used ONLY
//!   for the classified-vs-residual ratio (S5); if absent, S5 is skipped
//!   rather than inventing a classified-row count.
//!
//! Provenance: ruff-side R2IL pass-1 harvest over 143 real functions from 2
//! real x86-64 ELF64 binaries (`r2sleigh/tests/e2e/{stress_test,stress_test_opt}`),
//! `AdaWorldAPI/ruff`, `crates/ruff_r2il/examples/harvest_r2il.rs`. The slag
//! ledger is `ruff_r2il::furnace::ResidualLedger`'s committed output; this
//! probe reads that evidence, it does not regenerate it.
//!
//! Facts cited from the brief as already independently measured on the real
//! artifact (cited here, NOT asserted as hardcoded literals — every number
//! this probe gates on is recomputed at runtime from whatever file the env
//! var names): 489 total lines, 43 `grouped` rows, 440 `by_address` rows;
//! grouped reasons `opcode_not_in_convention` (39 rows), `variadic_arity` (2),
//! `no_facet_coordinate` (1), `memory_object_escaped` (1).
//!
//! ```sh
//! R2IL_SLAG_TSV=/path/to/r2il-pass1-slag.tsv \
//! R2IL_ORE_TSV=/path/to/r2il-pass1.ore.tsv \
//!   cargo run -p lance-graph-planner --example probe_r2il_slag_boundary
//! ```
//!
//! # Pre-registrations (direction chosen BEFORE the gate computes the number;
//! a refutation is recorded, not adjusted away)
//!
//! - **PR-1 (S3):** the pass-1 convention is narrow (7 opcodes) by deliberate
//!   design, so the residual is predicted to be DOMINATED by one named reason
//!   (`opcode_not_in_convention`) rather than diffuse. Gate: the largest
//!   reason's share of total residual count exceeds one half.
//! - **PR-2 (S1/S4):** the furnace's own "never a silent catch-all" invariant
//!   holds from the OUTSIDE too — every residual row resolves to a non-empty
//!   named reason, and a malformed/orphaned row is a hard failure, never a
//!   silent skip.
//!
//! # Gates (each can fail; each failure would be a finding)
//!
//! - **S1 PARSE + CONSERVATION** — strict per-section column arity (5 for
//!   `grouped`, 4 for `by_address`; an unknown section or wrong arity is
//!   proven to panic via an embedded synthetic bad line); cross-section
//!   conservation — every `by_address` shape_id's summed count must equal
//!   EXACTLY its `grouped` count (a real join invariant: same mass, split by
//!   address prefix, not a tautology).
//! - **S2 REASON CENSUS** — per-reason row count + count-sum over `grouped`
//!   (the only section carrying `reason` directly). Reason set non-empty, no
//!   reason string empty.
//! - **S3 CONCENTRATION** — PR-1, with margin (share > 0.5, an actual
//!   majority) so a genuinely diffuse residual would refute it.
//! - **S4 NAMED-NOT-DROPPED** — PR-2 from the `by_address` side via join into
//!   `grouped`. Can-fire demonstrated with a synthetic orphan shape_id (not in
//!   the real corpus) whose join lookup must panic.
//! - **S5 RATIO (skippable)** — if `R2IL_ORE_TSV` set: classified rows (light
//!   13-column check) vs S1's residual total; prints both + ratio; asserts
//!   only nonzero — no direction pre-registered, none was measured ahead of
//!   time. Unset → `SKIPPED`, never fabricated.
//! - **S6 FENCE** — not a re-ingest path into ruff; slag is pass-1
//!   SEVEN-OPCODE-CONVENTION slag, not "everything x86-64 can do"; corpus is 2
//!   binaries from one source, not a representative sample.
//!
//! # Fences
//!
//! - No import of `ruff_r2il` (separate workspace) — schema cited from the
//!   file's own header, verified against the real file.
//! - No mint: no classid, no vocabulary table, no BPE, no learner subsystem.
//! - No performance claims. Counts, sums, and ratios only.

use std::collections::BTreeMap;
use std::process::ExitCode;

// ================================================================================================
// Slag TSV reading (probe-local; strict — an unknown section or bad arity panics)
// ================================================================================================

#[derive(Clone)]
struct GroupedRow {
    reason: String,
    count: u64,
    #[allow(dead_code)] // read for schema completeness; not needed by any gate below
    example_facet: String,
}

struct ByAddressRow {
    #[allow(dead_code)] // read for schema completeness; not needed by any gate below
    prefix: String,
    shape_id: String,
    count: u64,
}

enum ParsedLine {
    Grouped(String, GroupedRow), // (shape_id, row)
    ByAddress(ByAddressRow),
}

/// Parses one non-comment slag line. PANICS (does not silently skip) on an
/// unknown section name or a wrong column count for the section it declares —
/// this is the "parse strictly" half of S1/S2, and S1 proves it by construction
/// with a synthetic bad line below.
fn parse_slag_line(line: &str) -> ParsedLine {
    let f: Vec<&str> = line.split('\t').collect();
    match f.first().copied() {
        Some("grouped") => {
            assert!(
                f.len() == 5,
                "grouped row must have exactly 5 columns, got {} ({line:?})",
                f.len()
            );
            let shape_id = f[1].to_string();
            let reason = f[2].to_string();
            let count: u64 = f[3].parse().expect("grouped count must be u64");
            let example_facet = f[4].to_string();
            ParsedLine::Grouped(
                shape_id,
                GroupedRow {
                    reason,
                    count,
                    example_facet,
                },
            )
        }
        Some("by_address") => {
            assert!(
                f.len() == 4,
                "by_address row must have exactly 4 columns, got {} ({line:?})",
                f.len()
            );
            let prefix = f[1].to_string();
            let shape_id = f[2].to_string();
            let count: u64 = f[3].parse().expect("by_address count must be u64");
            ParsedLine::ByAddress(ByAddressRow {
                prefix,
                shape_id,
                count,
            })
        }
        other => {
            panic!("unknown slag section {other:?} in line {line:?} — refusing to silently skip")
        }
    }
}

fn parse_slag(tsv: &str) -> (BTreeMap<String, GroupedRow>, Vec<ByAddressRow>) {
    let mut grouped: BTreeMap<String, GroupedRow> = BTreeMap::new();
    let mut by_address: Vec<ByAddressRow> = Vec::new();
    for line in tsv.lines() {
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        match parse_slag_line(line) {
            ParsedLine::Grouped(shape_id, row) => {
                assert!(
                    grouped.insert(shape_id.clone(), row).is_none(),
                    "duplicate grouped shape_id {shape_id} — furnace's grouped() should be unique per shape"
                );
            }
            ParsedLine::ByAddress(row) => by_address.push(row),
        }
    }
    (grouped, by_address)
}

/// Runs `f` with the default panic hook silenced, returning whether it panicked.
/// Used to demonstrate (not merely assert) that strict parsing / strict joining
/// actually rejects bad input, without polluting stderr with the expected panic.
fn panics_silently<F: FnOnce() + std::panic::UnwindSafe>(f: F) -> bool {
    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = std::panic::catch_unwind(f);
    std::panic::set_hook(prev_hook);
    result.is_err()
}

// ================================================================================================
// main
// ================================================================================================

/// The by_address -> grouped reason join, in ONE place.
///
/// META-REVIEW FIX (P0): S4's loop and S4's can-fire both call this, so the can-fire
/// exercises the real code path rather than a re-typed copy of it. Panicking here (rather
/// than returning Option) is deliberate: an orphaned residual row means the ledger lost a
/// name, which is exactly the "no catch-all, nothing dropped" invariant S4 exists to check.
fn reason_for<'a>(grouped: &'a BTreeMap<String, GroupedRow>, shape_id: &str) -> &'a str {
    match grouped.get(shape_id) {
        Some(g) => g.reason.as_str(),
        None => panic!("by_address row's shape_id {shape_id} has no reason to join"),
    }
}

fn main() -> ExitCode {
    let Some(slag_path) = std::env::var_os("R2IL_SLAG_TSV") else {
        eprintln!(
            "CORPUS ABSENT — this probe measures only the real slag ledger and never fabricates."
        );
        eprintln!("Fetch the real R2IL pass-1 slag ledger (ruff_r2il::furnace committed harvest artifacts)");
        eprintln!("and re-run:");
        eprintln!(
            "  R2IL_SLAG_TSV=/path/to/r2il-pass1-slag.tsv [R2IL_ORE_TSV=/path/to/r2il-pass1.ore.tsv] \\"
        );
        eprintln!("    cargo run -p lance-graph-planner --example probe_r2il_slag_boundary");
        return ExitCode::from(2);
    };
    let Ok(slag_tsv) = std::fs::read_to_string(&slag_path) else {
        eprintln!("CORPUS ABSENT — R2IL_SLAG_TSV={slag_path:?} is not readable. Never fabricated.");
        return ExitCode::from(2);
    };

    let (grouped, by_address) = parse_slag(&slag_tsv);
    let mut pass = 0u32;
    let mut total_gates = 0u32;

    // -------------------------------------------------------------------------------------------
    // S1 PARSE + CONSERVATION
    // -------------------------------------------------------------------------------------------
    let conservation_total: u64 = {
        total_gates += 1;

        // Can-fire: an unknown section name must be a hard failure, not a silent skip.
        let unknown_section_panics = panics_silently(|| {
            let _ = parse_slag_line("bogus\tfoo\tbar\tbaz");
        });
        assert!(
            unknown_section_panics,
            "an unknown slag section must panic — strict parsing is required, not silent skip"
        );

        // Can-fire: a `grouped` row with the wrong column count must also be a hard failure.
        let bad_arity_panics = panics_silently(|| {
            let _ = parse_slag_line("grouped\tshapeid_only_four_cols\treason\t5");
        });
        assert!(
            bad_arity_panics,
            "a grouped row with wrong column arity must panic, not silently misparse"
        );

        // Cross-section conservation: sum the by_address counts per shape_id and
        // require EXACT equality with that shape_id's grouped count. Every
        // by_address shape_id must also be a member of grouped (no orphans yet —
        // S4 checks the orphan case with a synthetic shape_id instead of assuming
        // the real corpus has none).
        let mut by_shape_sum: BTreeMap<&str, u64> = BTreeMap::new();
        for row in &by_address {
            *by_shape_sum.entry(row.shape_id.as_str()).or_default() += row.count;
        }
        for (shape_id, summed) in &by_shape_sum {
            let g = grouped.get(*shape_id).unwrap_or_else(|| {
                panic!("by_address shape_id {shape_id} has no grouped counterpart")
            });
            assert_eq!(
                *summed, g.count,
                "conservation violated for shape_id {shape_id}: by_address sums to {summed}, grouped says {}",
                g.count
            );
        }

        let total: u64 = grouped.values().map(|r| r.count).sum();
        pass += 1;
        println!(
            "S1 PASS  grouped_rows={} by_address_rows={} distinct_shapes_in_by_address={} \
             conservation: every by_address shape sums EXACTLY to its grouped count; \
             strict-parse can-fire proven (unknown section + bad arity both panic); total_count={total}",
            grouped.len(),
            by_address.len(),
            by_shape_sum.len()
        );
        total
    };

    // -------------------------------------------------------------------------------------------
    // S2 REASON CENSUS (grouped section only — the only section carrying `reason` directly)
    // -------------------------------------------------------------------------------------------
    {
        total_gates += 1;
        let mut rows_by_reason: BTreeMap<&str, u32> = BTreeMap::new();
        let mut count_by_reason: BTreeMap<&str, u64> = BTreeMap::new();
        for row in grouped.values() {
            assert!(
                !row.reason.is_empty(),
                "grouped row carries an empty reason string"
            );
            *rows_by_reason.entry(row.reason.as_str()).or_default() += 1;
            *count_by_reason.entry(row.reason.as_str()).or_default() += row.count;
        }
        assert!(
            !rows_by_reason.is_empty(),
            "the reason census must be non-empty — a zero-reason slag ledger is unfalsifiable"
        );
        pass += 1;
        println!(
            "S2 PASS  reason census (rows, summed count) over {} grouped shapes:",
            grouped.len()
        );
        for (reason, rows) in &rows_by_reason {
            println!(
                "         {reason:<24} rows={rows:>3} count={:>7}",
                count_by_reason[reason]
            );
        }
    }

    // -------------------------------------------------------------------------------------------
    // S3 CONCENTRATION — PR-1: the residual is dominated by one named reason, not diffuse.
    // -------------------------------------------------------------------------------------------
    {
        total_gates += 1;
        let mut count_by_reason: BTreeMap<&str, u64> = BTreeMap::new();
        for row in grouped.values() {
            *count_by_reason.entry(row.reason.as_str()).or_default() += row.count;
        }
        let total: u64 = count_by_reason.values().sum();
        let (top_reason, top_count) = count_by_reason
            .iter()
            .max_by_key(|(_, n)| **n)
            .expect("S2 already asserted non-empty reason census");
        let share = *top_count as f64 / total as f64;
        // PR-1, recorded not adjusted: if the residual turned out diffuse (no
        // reason clears a bare majority), this assertion fails and that IS the
        // finding — the convention would be mis-scoped, not merely under-tested.
        assert!(
            share > 0.5,
            "PR-1 REFUTED: no single named reason holds a majority of residual count \
             (top is {top_reason} at {share:.3}) — the residual reads as diffuse, not \
             convention-bounded"
        );
        pass += 1;
        println!(
            "S3 PASS  PR-1 holds: largest reason '{top_reason}' carries {top_count}/{total} = {share:.3} \
             of residual count (> 0.5 majority) — the residual is convention-bounded, not diffuse"
        );
    }

    // -------------------------------------------------------------------------------------------
    // S4 NAMED-NOT-DROPPED — PR-2 from the by_address side (join back into grouped for `reason`).
    // -------------------------------------------------------------------------------------------
    {
        total_gates += 1;
        // META-REVIEW FIX (P0): this gate previously re-asserted two properties S1 and S2
        // had already established (S1 panics on a missing counterpart; S2 rejects an empty
        // `grouped` reason), so NO input could reach S4 and fail it — and its "can-fire"
        // re-typed the join as a hand-written `panic!` instead of exercising the gate's own
        // code path. Both halves now route through ONE function, so the can-fire tests the
        // same code the loop runs.
        for row in &by_address {
            let reason = reason_for(&grouped, row.shape_id.as_str());
            assert!(
                !reason.is_empty(),
                "by_address row {} resolves to an empty reason via join",
                row.shape_id
            );
        }
        // Can-fire: the SAME `reason_for` call, on a synthetic orphan shape_id.
        let orphan_join_panics = panics_silently(|| {
            let _ = reason_for(&grouped, "0000000000000000_not_a_real_shape_id");
        });
        assert!(
            orphan_join_panics,
            "a synthetic orphan shape_id must fail the reason join — this gate must be able to fire"
        );
        pass += 1;
        println!(
            "S4 PASS  all {} by_address rows resolve to a non-empty named reason via join; \
             synthetic-orphan can-fire proven (an orphan shape_id panics the same join)",
            by_address.len()
        );
    }

    // -------------------------------------------------------------------------------------------
    // S5 RATIO (skippable) — classified rows (R2IL_ORE_TSV) vs residual count (S1's total).
    // -------------------------------------------------------------------------------------------
    {
        total_gates += 1;
        match std::env::var_os("R2IL_ORE_TSV") {
            Some(ore_path) => {
                match std::fs::read_to_string(&ore_path) {
                    Ok(ore_tsv) => {
                        let mut classified: u64 = 0;
                        for line in ore_tsv.lines() {
                            if line.starts_with('#') || line.is_empty() {
                                continue;
                            }
                            let cols = line.split('\t').count();
                            assert!(
                                cols == 13,
                                "R2IL_ORE_TSV schema drift: expected 13 columns, got {cols}"
                            );
                            classified += 1;
                        }
                        assert!(
                            classified > 0,
                            "classified-row count must be nonzero if the file parses"
                        );
                        assert!(conservation_total > 0, "residual count must be nonzero");
                        let ratio = classified as f64 / conservation_total as f64;
                        pass += 1;
                        println!(
                        "S5 PASS  classified_rows={classified} residual_count={conservation_total} \
                         ratio(classified/residual)={ratio:.3} — a ratio near or below 1 means the pass-1 \
                         seven-opcode convention's residual is comparable to or larger than what it \
                         classifies on this corpus; no direction was pre-registered, this is reported as \
                         measured, not adjusted"
                    );
                    }
                    Err(_) => {
                        eprintln!("S5 SKIPPED — R2IL_ORE_TSV={ore_path:?} is not readable. Never fabricated.");
                        total_gates -= 1;
                    }
                }
            }
            None => {
                println!(
                    "S5 SKIPPED — R2IL_ORE_TSV unset; classified-vs-residual ratio not computed"
                );
                total_gates -= 1;
            }
        }
    }

    // -------------------------------------------------------------------------------------------
    // S6 FENCE
    // -------------------------------------------------------------------------------------------
    {
        println!(
            "S6       fences: this probe reads a committed evidence artifact, it is NOT a re-ingest \
             path into ruff; the slag measured here is pass-1 SEVEN-OPCODE-CONVENTION slag, not \
             \"everything x86-64 can do\"; the corpus is 2 binaries from one source (r2sleigh's e2e \
             fixtures), not a representative code sample"
        );
    }

    println!(
        "\n{pass}/{total_gates} gates green — R2IL pass-1 slag boundary measurement complete."
    );
    println!("Fences: no ruff import; no re-ingest; no mint; counts, sums, and ratios only.");
    ExitCode::SUCCESS
}
