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

fn main() {
    println!(
        "symbiont golden image: lance-graph + lance7/lancedb0.30 + ndarray + ractor + surrealdb(kv-lance) + OGAR linked"
    );
    // D1 — the first real runtime edge: run the perturbation cascade and encode
    // the result onto canonical SoA NodeRows (the degenerate Spain-grid gate).
    bridge::run_demo();
}
