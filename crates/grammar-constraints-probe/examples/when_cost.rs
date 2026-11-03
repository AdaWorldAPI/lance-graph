//! D-GRAMMAR-WHEN-1 costs, cold and hot kept apart, for both physical forms.
//!   cargo run --release --manifest-path crates/grammar-constraints-probe/Cargo.toml --example when_cost
//! Cold phases are one-off (reported, not gates).

use std::hint::black_box;
use std::time::Instant;

use grammar_constraints_probe::fixtures;
use grammar_constraints_probe::when::*;
use grammar_constraints_probe::{tile, Ids, Vocab};
use lance_graph_dt64::{BindStore, MethodRef};

fn ns<T>(f: impl FnOnce() -> T) -> (T, u128) {
    let t = Instant::now();
    let r = f();
    (r, t.elapsed().as_nanos())
}

fn main() {
    let corpus = fixtures::corpus();
    let lex = Lexicon::v0();
    let mut v = Vocab::new();
    let (_, t_vocab) = ns(|| Ids::bind(&mut v));
    let ((table, frame), t_dr) = ns(|| tile(9, &fixtures::gids(&corpus)));
    let (pop, t_prod) =
        ns(|| FramedPopulation::produce(&corpus, &mut v, &table, &frame, &lex).unwrap());
    let rules = [RuleDesc::recipient(), RuleDesc::experiencer()];
    let mut store = BindStore::new();
    let (_, t_reg) = ns(|| {
        register(&mut store, CLASS_VIA, Physical::Via, &rules, &v, &lex).unwrap();
        register(&mut store, CLASS_COADDR, Physical::CoAddressed, &rules, &v, &lex).unwrap();
    });
    let n = pop.base.n;
    let mut all = vec![0u64; n.div_ceil(64)];
    for i in 0..n {
        all[i / 64] |= 1 << (i % 64);
    }
    println!("tokens={n} producer_lookups={}", pop.producer_lookups);
    println!(
        "cold (reported, not gates): vocab_bind={t_vocab} ns  dr_map={t_dr} ns  produce_head_frame={t_prod} ns  register_4_methods={t_reg} ns"
    );
    for (phys, class) in [(Physical::Via, CLASS_VIA), (Physical::CoAddressed, CLASS_COADDR)] {
        let mut composed = Vec::new();
        let mut t_comp = 0;
        for d in &rules {
            let r = MethodRef::of(class, &d.binding(phys, &v, &lex));
            let (c, t) = ns(|| compose(&store, &r, d, phys, &v, &lex).unwrap());
            t_comp += t;
            composed.push(c);
        }
        let ops: usize = composed.iter().map(|c| c.ops()).sum();
        for c in composed.iter_mut() {
            for _ in 0..10_000 {
                black_box(c.run(&pop, &table, &frame, &all).unwrap());
            }
        }
        let reps = 100_000u32;
        let mut best = u128::MAX;
        for _ in 0..5 {
            let t = Instant::now();
            for _ in 0..reps {
                for c in composed.iter_mut() {
                    black_box(c.run(black_box(&pop), &table, &frame, &all).unwrap());
                }
            }
            best = best.min(t.elapsed().as_nanos() / u128::from(reps));
        }
        println!(
            "{phys:?}: compose(admit+witness+lower) x2 = {t_comp} ns cold; programs=2 mask_ops={ops}; hot both methods incl. frame + lexicon-rev check = {best} ns/run (best of 5 x {reps})"
        );
    }
}
