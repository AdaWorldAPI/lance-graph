//! Runs the 1BRC monoid-aggregation certification probe.
//!
//!   cargo run --manifest-path crates/jc/Cargo.toml --release --example onebrc_prove
//!
//! Certifies the algebra behind `ndarray/examples/onebrc_cascade_probe.rs`
//! (branch `claude/1brc-lance-graph-xfx5tu`): partition/regroup invariance of
//! the `(min, max, Σ, n)` group-by monoid, and the exhaustive BF16 hi/lo
//! decomposition exactness that makes the AMX TDPBF16PS leg exact.

fn main() {
    println!("== JC diagnostic probe: 1BRC monoid aggregation ==\n");
    let r = jc::onebrc_agg::prove();
    println!("{}", r.name);
    r.report();
    std::process::exit(i32::from(!r.pass));
}
