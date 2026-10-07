//! Cypher stage probe — fork specimen.
//!
//! `common.rs` is the shared body; the upstream specimen compiles the same
//! file against upstream `lance-graph` (see `common.rs`'s header). This host
//! adds the Quack arm, which exists only in the fork.
//!
//! ```bash
//! cargo run --release -p lance-graph-benches --example cypher_stage_probe -- df
//! cargo run --release -p lance-graph-benches --example cypher_stage_probe -- parser
//! cargo run --release -p lance-graph-benches --example cypher_stage_probe -- quack
//! cargo run --release -p lance-graph-benches --example cypher_stage_probe -- reexec
//! ```

mod common;
mod quack;

#[global_allocator]
static GLOBAL: common::CountingAlloc = common::CountingAlloc;

fn main() {
    let mode = std::env::args().nth(1).unwrap_or_else(|| "df".into());
    let rt = tokio::runtime::Runtime::new().unwrap();
    match mode.as_str() {
        "df" => rt.block_on(common::suite(
            "fork",
            &[2, 5, 100, 10_000, 1_000_000],
            |n| if n >= 1_000_000 { 3 } else { 15 },
        )),
        "parser" => common::parser_bench(),
        "quack" => quack::suite(&[2, 5, 100, 10_000, 1_000_000], |n| {
            if n >= 1_000_000 {
                20
            } else {
                200
            }
        }),
        "prepared" => {
            for n in [10_000usize, 1_000_000] {
                // in-range ids, an absent id, and the u32 extremes
                let mut vals: Vec<u32> = (0..1000u32)
                    .map(|i| (i.wrapping_mul(2_654_435_761)) % n as u32)
                    .collect();
                vals.extend([0, n as u32 - 1, n as u32, u32::MAX]);
                let (ns, k) = quack::prepared_param_probe(n, &vals);
                println!("quack prepared Q5 n={n}: {k} values, patched == fresh lowering, answers == oracle; median patch+exec {} us", common::fmt_us(ns));
                rt.block_on(common::df_param_rebind(n, &vals[..50]));
            }
        }
        "reexec" => rt.block_on(common::reexec_probe()),
        m => panic!("unknown mode {m}"),
    }
}
