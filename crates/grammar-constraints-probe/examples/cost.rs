//! Cold vs hot cost of the D-GRAMMAR-CONSTRAINTS-0 rules, reported separately.
//!   cargo run --release --manifest-path crates/grammar-constraints-probe/Cargo.toml --example cost
//! Cold phases are one-off (reported, not gates); the hot run is the five
//! prepared programs over the 40-token corpus.

use std::time::Instant;

use grammar_constraints_probe::fixtures;
use grammar_constraints_probe::*;

fn ns(f: impl FnOnce()) -> u128 {
    let t = Instant::now();
    f();
    t.elapsed().as_nanos()
}

fn main() {
    let corpus = fixtures::corpus();
    let gids = fixtures::gids(&corpus);
    let mut vocab = Vocab::new();
    let mut ids = None;
    let t_vocab = ns(|| ids = Some(Ids::bind(&mut vocab)));
    let ids = ids.unwrap();
    let mut tf = None;
    let t_dr = ns(|| tf = Some(tile(7, &gids)));
    let (table, frame) = tf.unwrap();
    let mut pop = None;
    let t_pop = ns(|| pop = Some(Population::bind(&corpus, &mut vocab, &table, &frame).unwrap()));
    let pop = pop.unwrap();
    let mut rules = None;
    let t_rules = ns(|| rules = Some(Rules::prepare(&ids, Operands::ALL)));
    let mut rules = rules.unwrap();

    for _ in 0..10_000 {
        std::hint::black_box(rules.run(&pop, &table, &frame).unwrap());
    }
    let reps = 100_000u32;
    let mut best = u128::MAX;
    for _ in 0..5 {
        let t = Instant::now();
        for _ in 0..reps {
            std::hint::black_box(
                rules
                    .run(std::hint::black_box(&pop), &table, &frame)
                    .unwrap(),
            );
        }
        best = best.min(t.elapsed().as_nanos() / u128::from(reps));
    }
    println!("tokens={} programs=5 mask_ops={}", pop.n, rules.op_count());
    println!("cold (reported, not gates): vocab_bind={t_vocab} ns  dr_map={t_dr} ns  population_bind={t_pop} ns  rule_prepare={t_rules} ns");
    println!("hot: five programs incl. frame re-check = {best} ns/run (best of 5 x {reps})");
}
