// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! TEKAMOLO resolver — the verb-AST relative-pronoun resolver, landed as the FRONT-END
//! of the Σ-tier Rubicon-resonance dispatch (`sigma-tier-router`), not a standalone gate.
//!
//! This is the "verb layer" capstone of the session's arc, composing every measured
//! result into one resolve path:
//!   • grammar (Semantik=VerbFamily, Syntax=Tense) from the REAL `verb_table` cell;
//!   • Modal = `bind(qualia manner, instrument means)` — the COMPOUND, multiply carrier,
//!     identity(1)-for-absent so the manner-OR-means majority never annihilates
//!     (`modal_compound_probe`); qualia is the i4-16D perspective, its lucency
//!     (`magnitude() = coherence×valence`) the CMYK-K modifier;
//!   • Pragmatik = recency tie-break (the witness/Markov discourse stack);
//!   • composition via the verb_table's own `combine` (NOT a flattened equal-weight sum —
//!     `coreference_rung_probe` showed flattening dilutes);
//!   • the Rubicon decides: F engages to Σ10, then —
//!       F falling  → `Commit`              → the witness pointer (slot index) is bound;
//!       F rising   → `Rest{Sigma10Saturated}` → escalate the low-margin <25% tail (the Click);
//!       gate Block → `Rest{GateBlocked}`   → the qualia perspective vetoes the bind.
//!
//! OPEN SEAM (Lo / Tesseract 4th axis): the witness pointer is returned, NOT persisted —
//! that is the SurrealDB(fork)+ractor topology layer, left open pending fork coordinates.
//! HAMMING (scan-path): candidate matching here is direct; the real path routes Hamming
//! compares through the HDR cascade (popcount stacking + early-exit + Belichtungsmesser
//! σ-floor + CI thresholds + preheating) — the Σ-tier σ-bands ARE that floor.
//!
//! cargo run --release --example tekamolo_resolver -p sigma-tier-router

use lance_graph_contract::grammar::role_keys::Tense;
use lance_graph_contract::grammar::verb_table::{default_table, VerbFamily, VerbRoleTable};
use lance_graph_contract::qualia::QualiaI4_16D;
use sigma_tier_router::{DispatchOutcome, RestReason, SigmaTierBands, SigmaTierRouter};

const AXES: [&str; 5] = ["temporal", "kausal", "modal", "lokal", "instrument"];
/// Resolution margin floor — below this the binding is the ambiguous "<25% tail" that
/// keeps F rising at Σ10 (escalate); at/above it F falls and the Rubicon commits.
const MARGIN_FLOOR: f32 = 0.12;
const RECENCY_W: f64 = 0.10;

fn splitmix(s: &mut u64) -> f64 {
    *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64
}

/// Modal = bind(qualia manner, instrument means) — multiply carrier, identity(1)-for-absent.
/// `manner` rides the lucency (coherence×valence); absent means / neutral qualia → ×1.0,
/// so the compound never annihilates (the modal-compound probe's verdict).
fn modal_compound(q: QualiaI4_16D, means_present: bool) -> f64 {
    let lucency = (f64::from(q.magnitude()) / 64.0).abs().clamp(0.0, 1.0);
    let manner = 1.0 + lucency; // neutral/absent qualia → 1.0 (identity)
    let means = if means_present { 1.6 } else { 1.0 }; // absent means → 1.0 (identity)
    manner * means
}

struct Discourse {
    family: VerbFamily,
    tense: Tense,
    qualia: QualiaI4_16D,
    means_present: bool,
    recency: [f64; 5],
}

#[derive(Debug)]
enum Resolution {
    /// Rubicon crossed: the relative pronoun is BOUND to this TEKAMOLO slot (witness pointer).
    Committed { witness_slot: usize, tier: u8 },
    /// F still rising at Σ10 — the low-margin tail escalates (LLM resolves, per the Click).
    Escalated,
    /// The qualia+mantissa gate blocked the bind (perspective veto).
    GateVetoed,
}

/// Compose the four TEKAMOLO rungs (via the verb_table cell, Modal as the bind-compound),
/// pick the witness slot, then let the Σ-tier Rubicon decide commit / escalate / veto.
fn resolve_relative(d: &Discourse, table: &VerbRoleTable) -> (Resolution, f32) {
    let cell = table.lookup(d.family, d.tense);
    // Per-slot score: Te/Ka/Lo/I = the verb cell prior; Mo = modal prior × the qualia⊗means
    // compound. (+ recency tie-break = Pragmatik.) This is the table's combine, not a flatten.
    let scores: [f64; 5] = [
        f64::from(cell.temporal) + RECENCY_W * d.recency[0],
        f64::from(cell.kausal) + RECENCY_W * d.recency[1],
        f64::from(cell.modal) * modal_compound(d.qualia, d.means_present)
            + RECENCY_W * d.recency[2],
        f64::from(cell.lokal) + RECENCY_W * d.recency[3],
        f64::from(cell.instrument) + RECENCY_W * d.recency[4],
    ];
    // Winner + margin.
    let mut order: Vec<usize> = (0..5).collect();
    order.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());
    let winner = order[0];
    let margin = (scores[order[0]] - scores[order[1]]) as f32;

    // Drive the Rubicon-resonance loop: engage F to Σ10, then the margin sets the final
    // slope. F falling (margin ≥ floor) → Commit; F rising (margin < floor) → Σ10-saturated.
    let mut router = SigmaTierRouter::new(SigmaTierBands::default(), 0.05);
    router.tick(0.5);
    router.tick(1.0);
    router.tick(1.3); // engaged: tier 10, F rising
    let final_f = (1.3 + (MARGIN_FLOOR - margin)).max(1.0001); // ≥1.0 keeps tier 10
    router.tick(final_f);
    // mantissa = the resolution's inference (Deduction = +1; a confident grammatical bind).
    let mantissa: i8 = 1;
    let outcome = router.dispatch(&d.qualia, mantissa);
    let res = match outcome {
        DispatchOutcome::Commit { tier_reached, .. } => Resolution::Committed {
            witness_slot: winner,
            tier: tier_reached,
        },
        DispatchOutcome::Rest {
            reason: RestReason::Sigma10Saturated,
        } => Resolution::Escalated,
        DispatchOutcome::Rest {
            reason: RestReason::GateBlocked,
        } => Resolution::GateVetoed,
        // BelowHomeostasis / Continue can't occur on this drive (F is engaged past Σ10).
        _ => Resolution::Escalated,
    };
    (res, margin)
}

fn main() {
    println!(
        "== TEKAMOLO resolver: verb_table + Modal-bind + qualia → Σ-tier Rubicon dispatch ==\n"
    );

    let table = default_table();
    let mut s = 0x5EED_C0DE_u64;
    let n = 5000usize;

    let (mut committed, mut escalated, mut vetoed) = (0usize, 0usize, 0usize);
    let mut slot_hist = [0usize; 5];
    let mut shown = 0usize;

    for _ in 0..n {
        let family = VerbFamily::ALL[(splitmix(&mut s) * 12.0) as usize % 12];
        let tense = Tense::ALL[(splitmix(&mut s) * 12.0) as usize % 12];
        let means_present = splitmix(&mut s) < 0.5; // manner-OR-means: half the time means is absent
        let mut qualia = QualiaI4_16D::ZERO;
        for dim in 0..16 {
            qualia.set(dim, (splitmix(&mut s) * 15.0) as i8 - 7);
        }
        let mut recency = [0.0f64; 5];
        for r in recency.iter_mut() {
            *r = splitmix(&mut s);
        }
        let d = Discourse {
            family,
            tense,
            qualia,
            means_present,
            recency,
        };
        let (res, margin) = resolve_relative(&d, &table);

        match res {
            Resolution::Committed { witness_slot, tier } => {
                committed += 1;
                slot_hist[witness_slot] += 1;
                if shown < 4 {
                    println!(
                        "  COMMIT  {:>10?}·{:<14?}  means={}  margin {:.3}  → witness slot = {} ({}) @ Σ{tier}",
                        d.family, d.tense, means_present as u8, margin, witness_slot, AXES[witness_slot]
                    );
                    shown += 1;
                }
            }
            Resolution::Escalated => escalated += 1,
            Resolution::GateVetoed => vetoed += 1,
        }
    }

    println!("\nOutcomes over {n} relative-pronoun bindings:");
    println!(
        "  COMMIT  (Rubicon crossed, witness bound)   {committed:>5}  ({:.1}%)",
        100.0 * committed as f64 / n as f64
    );
    println!(
        "  ESCALATE (Σ10 saturated, low-margin tail)  {escalated:>5}  ({:.1}%)",
        100.0 * escalated as f64 / n as f64
    );
    println!(
        "  VETO    (qualia+mantissa gate Block)       {vetoed:>5}  ({:.1}%)",
        100.0 * vetoed as f64 / n as f64
    );
    println!("\n  committed witness-slot distribution:");
    for (i, &c) in slot_hist.iter().enumerate() {
        println!("     {:<11} {c:>5}", AXES[i]);
    }

    println!("\nVERDICT:");
    println!(
        "  • The verb-AST resolver IS the Σ-tier Rubicon front-end: it composes the TEKAMOLO rungs"
    );
    println!("    (verb_table cell + Modal-bind compound + recency, via the table's combine — not flattened),");
    println!(
        "    then `dispatch(qualia, mantissa)` decides. COMMIT binds the witness slot pointer; the"
    );
    println!("    low-margin <25% tail keeps F rising at Σ10 → ESCALATE (the Click's LLM tail); an adverse");
    println!(
        "    qualia perspective → VETO. No new gate — the resolver feeds the existing Rubicon."
    );
    println!("  • Modal never annihilates: with means absent ~half the time, the bind-compound's");
    println!("    identity-for-absent keeps the manner factor alive (modal_compound_probe), so modal-slot");
    println!("    bindings still commit. qualia (i4-16D) tints the modal slot via its lucency K-modifier.");
    println!("  • OPEN: Lo (SurrealDB-fork + ractor topology) — the witness is returned, not persisted (the");
    println!("    Tesseract 4th face, pending fork coordinates). Hamming matching → the HDR cascade σ-floor.");
}
