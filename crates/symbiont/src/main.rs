//! Golden-image probe binary.
//!
//! Declaring each repo as a dependency forces the entire unified dependency
//! graph to compile and link into one binary. That successful build IS the
//! golden image — the known-good foundation for the kanban thinking, whose
//! first test workload is the perturbation simulation (`perturbation_sim`).
//!
//! Build:  cargo build --manifest-path crates/symbiont/Cargo.toml

fn main() {
    println!(
        "symbiont golden image: lance-graph + lance7/lancedb0.30 + ndarray + ractor + surrealdb(kv-lance) + OGAR linked"
    );
}
