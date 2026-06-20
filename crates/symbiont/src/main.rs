//! Golden-image probe binary.
//!
//! Declaring each repo as a dependency forces the entire unified dependency
//! graph to compile and link into one binary. That successful build IS the
//! golden image — the known-good foundation for the kanban thinking, whose
//! first test workload is the perturbation simulation (`perturbation_sim`).
//!
//! Build:  cargo build --manifest-path crates/symbiont/Cargo.toml
//! Or:     docker build -f crates/symbiont/Dockerfile -t symbiont .

mod bridge;
mod domino;
mod kanban_loop;

fn main() {
    println!(
        "symbiont golden image: lance-graph + lance7/lancedb0.30 + ndarray + ractor + surrealdb(kv-lance) + OGAR linked"
    );
    // D1 — the first real runtime edge: each bus → one SoA NodeRow, its f64
    // perturbation → the Energy tenant (a thinking-style cascade onto the SoA).
    bridge::run_demo();
    // Scale: the 16k-board / 8-MiB ceiling, zero-copy.
    bridge::run_scale_demo(bridge::MAX_BOARDS);
    // The SoA-orchestration POC: 16-board AMX BF16 Morton-tile Domino batches.
    domino::run_poc(256, 3);
    // D2 — the kanban loop: version-tick → NextPhaseScheduler → try_advance_phase,
    // with the Domino sweep as the CognitiveWork phase over the SoA.
    kanban_loop::run_demo();
}
