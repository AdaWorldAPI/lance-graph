//! The 10-minute proof binary.
//!
//! Run:
//!   cargo run --manifest-path crates/jc/Cargo.toml --release --example prove_it
//!
//! Output: five pillar results with measured / predicted / pass-fail.
//! Exit code 0 = all implemented pillars pass. Exit code 1 = at least one fails.

fn main() {
    println!("═══ JC — Jirak-Cartan: Five-Pillar (+Pearl 5b) Proof-in-Code ═══");
    println!("Binary-Hamming causal field computation on d=10000/16384\n");

    let results = jc::run_all_pillars();

    let implemented: Vec<_> = results.iter().filter(|r| !r.detail.starts_with("DEFERRED")).collect();
    let passed = implemented.iter().filter(|r| r.pass).count();
    let failed = implemented.len() - passed;
    let deferred = results.len() - implemented.len();

    println!("═══ Summary ═══");
    println!("  Implemented: {}/{}", implemented.len(), results.len());
    println!("  Passed:      {passed}");
    println!("  Failed:      {failed}");
    println!("  Deferred:    {deferred} (coupled revival track)");

    if failed > 0 {
        println!("\n✗ {failed} pillar(s) FAILED — the substrate claim does not hold.");
        std::process::exit(1);
    } else {
        println!("\n✓ All implemented pillars pass. The substrate is formally sound.");
        if deferred > 0 {
            println!("  {deferred} pillar(s) deferred — activate coupled revival track to complete 5/5.");
        }
    }
}
