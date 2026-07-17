//! Style-table agreement probe — do the three shipped 12-family
//! style-parameter tables measure the SAME construct?
//!
//! Run: `cargo run --manifest-path crates/jc/Cargo.toml --example style_table_agreement`
//!
//! # The three tables
//!
//! - **Table A (driver)** — `cognitive-shader-driver::engine_bridge::UNIFIED_STYLES`
//!   // SOURCE: crates/cognitive-shader-driver/src/engine_bridge.rs:545-726
//!   Ordinal-aligned to `StyleFamily::ALL` (pinned by the crate's own test
//!   `unified_styles_align_with_style_family`, engine_bridge.rs:778-787).
//! - **Table B (engine)** — `thinking-engine::cognitive_stack::EngineStyleExt::params`
//!   // SOURCE: crates/thinking-engine/src/cognitive_stack.rs:55-142
//!   Keyed by `StyleFamily` match arms; re-ordered here to `StyleFamily::ALL`
//!   ordinal order by matching variant name (the match arms in the source are
//!   NOT in ordinal order).
//! - **Table C (planner)** — `lance-graph-planner::thinking::style::PlannerStyleExt::default_modulation`
//!   // SOURCE: crates/lance-graph-planner/src/thinking/style.rs:74-142 (7 explicit
//!   arms) + `FieldModulation::default()` at style.rs:165-177 (5 fallback families:
//!   convergent, systematic, divergent, diffuse, peripheral — all get the SAME
//!   default (0.7, 6, 0.3), which is a distinct value from any of their real
//!   per-family calibrations in Tables A/B).
//!
//! All three tables share three overlapping dimensions: `resonance_threshold`,
//! `fan_out` (cast to `f64`), and `exploration`. This probe treats the 12 (or
//! 7) `StyleFamily` variants as SUBJECTS and the 3 tables as RATERS, and asks
//! via `jc::reliability`'s merged battery (ICC / Cronbach α / Pearson /
//! Spearman) whether the three raters agree.
//!
//! # Two modes, because the planner's defaults distort agreement
//!
//! - **Mode A — all-12**: every `StyleFamily` variant, including the 5 the
//!   planner defaults (`FieldModulation::default()`). Those 5 get an
//!   IDENTICAL default triple (0.7, 6, 0.3) in Table C regardless of what the
//!   family actually is, so this mode mixes real per-family disagreement (the
//!   7 explicit families) with default-vs-calibrated noise (the other 5).
//! - **Mode B — 7-explicit-only**: drop the 5 defaulted families, leaving the
//!   7 the planner author actually wrote an arm for (deliberate, analytical,
//!   creative, exploratory, focused, intuitive, metacognitive). This is the
//!   fairer test of whether Table C's real calibration agrees with A/B.
//!
//! **Measured direction (this run, not assumed):** Mode A's ICC(2,1) came out
//! LOWER than Mode B's on all three dimensions (e.g. resonance_threshold:
//! 0.7134 all-12 vs 0.9326 7-explicit) — the 5 defaulted families' values
//! diverge sharply from their real A/B calibrations (e.g. divergent/
//! peripheral are extreme in A/B but flattened to the default in C), so
//! folding them in adds disagreement rather than inflating it. Do not assume
//! the direction ahead of running this — it depends on how far the default
//! sits from the true per-family spread, which is exactly why both modes are
//! reported side by side instead of asserting one a priori.
//!
//! # Pre-registered verdict thresholds (hand-tuned, NOT Jirak-derived)
//!
//! Per dimension, using `ICC(2,1)` (two-way random, absolute agreement —
//! Shrout & Fleiss 1979):
//!   - `ICC(2,1) >= 0.75` → **IDENTITY** (the tables agree closely)
//!   - `ICC(2,1) <  0.50` → **DISTINCT** (the tables measure different things)
//!   - otherwise         → **AMBIGUOUS**
//!
//! These cutoffs are point-estimate, hand-tuned convention (the same 0.5/0.75
//! bands commonly used for "poor/moderate/good" ICC bands in the reliability
//! literature), NOT derived from `jc::jirak`'s weak-dependence bound. Per
//! `I-NOISE-FLOOR-JIRAK` (`lance-graph/CLAUDE.md`), significance calibration
//! for THESE 12/7-subject samples is a separate job this probe does not do —
//! it reports point estimates only. **DISTINCT is a valid, useful finding,
//! not a failure of the probe** — if Table C's own defaults diverge from the
//! calibrated per-family values in A/B, that is exactly the kind of
//! type-duplication drift `docs/TYPE_DUPLICATION_MAP.md` already flags for
//! `ThinkingStyle` (4 copies, contract canonical, not yet fully adopted).
//!
//! This example never panics or asserts on the measured outcome — it always
//! exits 0. The reliability battery's `Option<f64>` values are printed as
//! `None` when undefined (e.g. zero variance in a column) rather than
//! unwrapped.

use jc::reliability::{cronbach_alpha, icc, pearson, spearman, IccForm};

/// One row per `StyleFamily` variant, in `StyleFamily::ALL` ordinal order:
/// (name, resonance_threshold, fan_out as f64, exploration).
type StyleRow = (&'static str, f64, f64, f64);

/// Table A — driver `UNIFIED_STYLES`.
// SOURCE: crates/cognitive-shader-driver/src/engine_bridge.rs:545-726
const TABLE_A_DRIVER: [StyleRow; 12] = [
    ("deliberate", 0.70, 7.0, 0.20),
    ("analytical", 0.85, 3.0, 0.05),
    ("convergent", 0.75, 4.0, 0.10),
    ("systematic", 0.70, 5.0, 0.10),
    ("creative", 0.35, 12.0, 0.80),
    ("divergent", 0.40, 10.0, 0.70),
    ("exploratory", 0.30, 15.0, 0.90),
    ("focused", 0.90, 1.0, 0.00),
    ("diffuse", 0.45, 8.0, 0.40),
    ("peripheral", 0.20, 20.0, 0.60),
    ("intuitive", 0.50, 3.0, 0.30),
    ("metacognitive", 0.50, 5.0, 0.30),
];

/// Table B — thinking-engine `StyleParams` (re-keyed to `StyleFamily::ALL`
/// ordinal order by matching variant name; the source match arms are not in
/// ordinal order).
// SOURCE: crates/thinking-engine/src/cognitive_stack.rs:55-142
const TABLE_B_ENGINE: [StyleRow; 12] = [
    ("deliberate", 0.70, 7.0, 0.20),
    ("analytical", 0.85, 3.0, 0.05),
    ("convergent", 0.75, 4.0, 0.10),
    ("systematic", 0.70, 5.0, 0.10),
    ("creative", 0.35, 12.0, 0.80),
    ("divergent", 0.40, 10.0, 0.70),
    ("exploratory", 0.30, 15.0, 0.90),
    ("focused", 0.90, 1.0, 0.00),
    ("diffuse", 0.45, 8.0, 0.40),
    ("peripheral", 0.20, 20.0, 0.60),
    ("intuitive", 0.50, 3.0, 0.30),
    ("metacognitive", 0.50, 5.0, 0.30),
];

/// Table C — planner `FieldModulation` (7 explicit arms + `default()` for the
/// other 5). Rows marked `// default` are `FieldModulation::default()`
/// (0.7, 6, 0.3) at style.rs:165-177, NOT a per-family calibration.
// SOURCE: crates/lance-graph-planner/src/thinking/style.rs:74-142 (explicit) +
// crates/lance-graph-planner/src/thinking/style.rs:165-177 (default)
const TABLE_C_PLANNER: [StyleRow; 12] = [
    ("deliberate", 0.75, 6.0, 0.20),
    ("analytical", 0.85, 4.0, 0.10),
    ("convergent", 0.70, 6.0, 0.30),   // default
    ("systematic", 0.70, 6.0, 0.30),   // default
    ("creative", 0.50, 12.0, 0.80),
    ("divergent", 0.70, 6.0, 0.30),    // default
    ("exploratory", 0.30, 20.0, 1.00),
    ("focused", 0.90, 2.0, 0.05),
    ("diffuse", 0.70, 6.0, 0.30),      // default
    ("peripheral", 0.70, 6.0, 0.30),   // default
    ("intuitive", 0.60, 8.0, 0.40),
    ("metacognitive", 0.70, 8.0, 0.50),
];

/// Whether the planner wrote an explicit `default_modulation()` arm for this
/// `StyleFamily::ALL`-ordinal family, vs falling through to `_ =>
/// FieldModulation::default()`.
const PLANNER_HAS_EXPLICIT: [bool; 12] = [
    true, true, false, false, true, false, true, true, false, false, true, true,
];

fn fmt_opt(v: Option<f64>) -> String {
    match v {
        Some(x) => format!("{x:.4}"),
        None => "None".to_string(),
    }
}

fn verdict(icc21: Option<f64>) -> &'static str {
    match icc21 {
        Some(v) if v >= 0.75 => "IDENTITY",
        Some(v) if v < 0.50 => "DISTINCT",
        Some(_) => "AMBIGUOUS",
        None => "UNDEFINED (ICC None — likely zero variance in a column)",
    }
}

/// Select the column (resonance / fan_out / exploration) from a table,
/// restricted to the given row indices.
fn column(table: &[StyleRow; 12], indices: &[usize], which: usize) -> Vec<f64> {
    indices
        .iter()
        .map(|&i| match which {
            0 => table[i].1,
            1 => table[i].2,
            _ => table[i].3,
        })
        .collect()
}

/// Run the merged reliability battery for one dimension across the three
/// tables, restricted to `indices`, and print the results.
fn report_dimension(dim_name: &str, indices: &[usize], which: usize) {
    let col_a = column(&TABLE_A_DRIVER, indices, which);
    let col_b = column(&TABLE_B_ENGINE, indices, which);
    let col_c = column(&TABLE_C_PLANNER, indices, which);

    let n = col_a.len();
    let ratings: Vec<Vec<f64>> = (0..n)
        .map(|i| vec![col_a[i], col_b[i], col_c[i]])
        .collect();
    let icc21 = icc(&ratings, IccForm::Icc2_1);
    let icc31 = icc(&ratings, IccForm::Icc3_1);

    let items = vec![col_a.clone(), col_b.clone(), col_c.clone()];
    let alpha = cronbach_alpha(&items);

    let p_ab = pearson(&col_a, &col_b);
    let p_ac = pearson(&col_a, &col_c);
    let p_bc = pearson(&col_b, &col_c);
    let s_ab = spearman(&col_a, &col_b);
    let s_ac = spearman(&col_a, &col_c);
    let s_bc = spearman(&col_b, &col_c);

    println!("--- Dimension: {dim_name} (n={n} families) ---");
    println!("  ICC(2,1) absolute-agreement      : {}", fmt_opt(icc21));
    println!("  ICC(3,1) consistency              : {}", fmt_opt(icc31));
    println!("  Cronbach α (3-item scale)          : {}", fmt_opt(alpha));
    println!("  Pearson  r  A(driver)–B(engine)   : {}", fmt_opt(p_ab));
    println!("  Pearson  r  A(driver)–C(planner)  : {}", fmt_opt(p_ac));
    println!("  Pearson  r  B(engine)–C(planner)  : {}", fmt_opt(p_bc));
    println!("  Spearman ρ  A(driver)–B(engine)   : {}", fmt_opt(s_ab));
    println!("  Spearman ρ  A(driver)–C(planner)  : {}", fmt_opt(s_ac));
    println!("  Spearman ρ  B(engine)–C(planner)  : {}", fmt_opt(s_bc));
    println!("  Verdict (ICC(2,1) vs 0.75/0.50)    : {}", verdict(icc21));
    println!();
}

fn main() {
    println!("═══ Style-table agreement probe (D-TRI-2/4/5-adjacent) ═══");
    println!();
    println!("Three shipped 12-family style-parameter tables, checked for agreement");
    println!("via jc::reliability (ICC/Cronbach α/Pearson/Spearman). Point estimates");
    println!("only — NOT Jirak-derived significance (I-NOISE-FLOOR-JIRAK is a separate");
    println!("job). Verdict thresholds are hand-tuned + pre-registered:");
    println!("  ICC(2,1) >= 0.75 → IDENTITY   ICC(2,1) < 0.50 → DISTINCT   else → AMBIGUOUS");
    println!("DISTINCT is a valid finding, not a failure of this probe.");
    println!();

    println!("Table A (driver)  = cognitive-shader-driver::engine_bridge::UNIFIED_STYLES");
    println!("Table B (engine)  = thinking-engine::cognitive_stack StyleParams");
    println!("Table C (planner) = lance-graph-planner::thinking::style FieldModulation");
    println!("                    (7 explicit arms + FieldModulation::default() for the");
    println!("                     other 5: convergent, systematic, divergent, diffuse,");
    println!("                     peripheral)");
    println!();
    for (i, row) in TABLE_A_DRIVER.iter().enumerate() {
        let explicit = if PLANNER_HAS_EXPLICIT[i] {
            "explicit"
        } else {
            "DEFAULT"
        };
        println!(
            "  [{i:>2}] {:<14} A=({:.2},{:>4.0},{:.2})  B=({:.2},{:>4.0},{:.2})  C=({:.2},{:>4.0},{:.2}) [{explicit}]",
            row.0,
            row.1, row.2, row.3,
            TABLE_B_ENGINE[i].1, TABLE_B_ENGINE[i].2, TABLE_B_ENGINE[i].3,
            TABLE_C_PLANNER[i].1, TABLE_C_PLANNER[i].2, TABLE_C_PLANNER[i].3,
        );
    }
    println!();

    let all_12: Vec<usize> = (0..12).collect();
    let seven_explicit: Vec<usize> = (0..12).filter(|&i| PLANNER_HAS_EXPLICIT[i]).collect();

    println!("╔══ MODE A — all-12 (planner's 5 fallback families use FieldModulation::default) ══╗");
    println!();
    report_dimension("resonance_threshold", &all_12, 0);
    report_dimension("fan_out", &all_12, 1);
    report_dimension("exploration", &all_12, 2);

    println!("╔══ MODE B — 7-explicit-only (drop the 5 planner-default families) ══╗");
    println!();
    report_dimension("resonance_threshold", &seven_explicit, 0);
    report_dimension("fan_out", &seven_explicit, 1);
    report_dimension("exploration", &seven_explicit, 2);

    println!("═══ Done. Exit 0 regardless of verdicts above (no panics/asserts on outcome). ═══");
}
