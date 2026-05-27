//! Orchestration deep-dive #2 — Odoo savant delegation (woa-rs AXIS-A guard → lance-graph AXIS-B reason).
//!
//! Run it:  `cargo run -p lance-graph-contract --example savant_dispatch`
//!
//! THE IDEA (Elixir-style):
//!   * a Savant is a delegated reasoner declared as a *tuple* (family · ReasoningKind ·
//!     InferenceType · SemiringChoice · StyleCluster) — the tuple FULLY determines dispatch.
//!   * dispatch is **pattern-matching** on `ReasoningKind` (like an Elixir `case`/multi-clause
//!     function head), and the `InferenceType` resolves O(1) to a `QueryStrategy`.
//!   * the deterministic guard (AXIS-A) stays in woa-rs; only the *ambiguous core* is delegated
//!     here (AXIS-B), and the answer is a NARS-truth-weighted **suggestion**, never an
//!     un-guarded write (Iron Rule 7).

use lance_graph_contract::reasoning::ReasoningKind;
use lance_graph_contract::savants::{savant_by_name, Savant};

/// A concrete situation arriving from the ERP.
struct Situation {
    headline: &'static str,
    /// The AXIS-A guard already ran in woa-rs; `true` means it could NOT decide deterministically
    /// and is delegating the ambiguous core to the savant.
    ambiguous: bool,
    savant: &'static str,
}

/// Pattern-match the kind → the reasoning approach (the Elixir multi-clause `case`).
fn approach(kind: ReasoningKind) -> &'static str {
    match kind {
        ReasoningKind::CustomerCategory => "classify against the family codebook (deductive lookup)",
        ReasoningKind::PostingAnomaly => "abduce the most likely cause from the evidence trail",
        ReasoningKind::NextBestAction => "induce the action with the highest expected value",
        ReasoningKind::InvoiceCompleteness => "check required-field coverage, score the gaps",
        ReasoningKind::MailIntent => "resonate the message against intent prototypes",
        ReasoningKind::Other(code) => match code {
            5 | 6 => "match open items / bank lines by evidence fusion (reconcile)",
            _ => "domain-specific Other(code) reasoner",
        },
    }
}

fn dispatch(s: &Savant) {
    println!("  savant     {} (#{}, lane {})", s.name, s.id, s.lane);
    println!("  family     {}", s.family.map(|f| format!("0x{f:02X}")).unwrap_or_else(|| "None (needs alignment axiom)".into()));
    println!("  tuple      kind={:?} · infer={:?} · semiring={:?} · style={:?}", s.kind, s.inference, s.semiring, s.style);
    // The InferenceType resolves O(1) to the runtime query strategy.
    println!("  → strategy {:?}   (InferenceType::default_strategy)", s.query_strategy());
    println!("  → approach {}", approach(s.kind));
    println!("  → output   NARS (frequency, confidence) suggestion — woa-rs applies it behind its AXIS-A guard\n");
}

fn main() {
    println!("== Odoo savant delegation: AXIS-A guard (woa-rs) → AXIS-B reason (lance-graph) ==\n");

    let inbox = [
        Situation { headline: "€1,200 payment arrived — does it fully reconcile the partner's open invoices?", ambiguous: true, savant: "PaymentToInvoiceMatcher" },
        Situation { headline: "3rd identical bill from this vendor, unmodified — auto-post it?",              ambiguous: true, savant: "AutopostRecommender" },
        Situation { headline: "new B2B partner in AT — which fiscal position (tax mapping)?",                 ambiguous: true, savant: "FiscalPositionResolver" },
        Situation { headline: "journal sequence jumps 1042 → 1044 — is 1043 a deleted posted entry?",         ambiguous: true, savant: "SequenceGapAnomalyDetector" },
        Situation { headline: "invoice with a perfectly matching single open item",                           ambiguous: false, savant: "ReconcileMatchSelector" },
    ];

    for s in &inbox {
        println!("• {}", s.headline);
        if !s.ambiguous {
            println!("  AXIS-A (woa-rs): deterministic match — applied directly, NO delegation.\n");
            continue;
        }
        let sv = savant_by_name(s.savant).expect("savant in roster");
        dispatch(sv);
    }

    println!("Same delegation tuple for all 25 savants; dispatch is data, not branches.");
    println!("Pattern-match on ReasoningKind picks the approach; the family's StyleCluster colours it.");
}
