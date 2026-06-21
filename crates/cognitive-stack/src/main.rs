//! # cognitive-stack — the Cognitive Compilation runtime in one binary
//!
//! Links the **new stack** (the Elixir-shaped template crates) on top of the
//! **old stack** (every AdaWorldAPI fork: lance-graph + ndarray + ractor +
//! surrealdb-kv-lance + OGAR). It is the golden-image proof that the compiled-
//! cognition reflex path links and runs with no LLM present — the LLM (rig) is
//! the teacher/compiler/critic at learning time only and is deliberately absent
//! from this runtime binary.
//!
//! See `README.md` (purpose + usage) and `INTEGRATION.md` (integration plan).

// ── OLD stack: force-link every AdaWorldAPI fork into the binary ──
use lance_graph as _; // the spine: query / codec / contract
use lance_graph_ogar as _; // OGAR Active-Record bridge (impl ClassView)
use ndarray as _; // the foundation: SIMD / HPC / Fingerprint / CAM-PQ
use ogar_vocab as _; // OGAR semantic type system (class codebook)
use ractor as _; // control-plane ownership fence
use surrealdb_core as _; // kv-lance provenance / timeline view

// ── NEW stack: the Cognitive Compilation Elixir-template crates ──
use cognitive_compiler as _; // trace → template synthesis (learning-time)
use elixir_template::source_ranking_v1;
use template_equivalence::EquivalenceConfig;
use template_runtime::ActionRegistry;

fn main() {
    // The first vertical slice's compiled template (no LLM to build it here —
    // it is the already-compiled reflex).
    let template = source_ranking_v1();

    println!("cognitive-stack — linked golden image");
    println!("  NEW: elixir-template · template-runtime · template-equivalence · cognitive-compiler");
    println!("  OLD: lance-graph + ndarray + ractor + surrealdb(kv-lance) + OGAR  [AdaWorldAPI forks]");
    println!();
    println!(
        "reflex template `{}` v{} → {} OGAR actions (runs deterministically, no LLM in hot path):",
        template.name,
        template.version,
        template.steps.len()
    );
    for action in template.ogar_action_names() {
        println!("    · {action}");
    }

    // The reflex executes via an ActionRegistry of OGAR actions; promotion is
    // gated by replay equivalence (§18). Both surfaces are linked and ready.
    let registry = ActionRegistry::new();
    let cfg = EquivalenceConfig::default();
    println!();
    println!(
        "runtime ready: {} OGAR actions registered; equivalence gate rank_tolerance={}, no_new_claims={}",
        registry.len(),
        cfg.rank_tolerance,
        cfg.require_no_new_claims
    );
}
