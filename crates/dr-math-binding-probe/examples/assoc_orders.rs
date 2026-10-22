//! Per-order work and time for one 5-chain, plus the cold egglog proof.
//! Fold times are per evaluation (ns); the proof is a cold, out-of-fold cost,
//! reported in µs and not a gate.
use std::hint::black_box;
use std::time::Instant;

use dr_math_binding_probe::{
    Axis, AxisKind, Focus, FrameRevisions, FrameStore, Guards, MethodSignature, Operand, Scope,
    bind_dr64,
    recipe::{LawShape, Primitive},
    relcomp::{Chain, RelExpr, Rows, admit_composition_assoc, all_orders, prove_reassociation},
};

fn fill(seed: u64, keep_of_64: u32) -> Rows {
    let mut x = seed | 1;
    let mut rows = [0u64; 64];
    for r in rows.iter_mut() {
        for b in 0..64 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            if (x % 64) < u64::from(keep_of_64) {
                *r |= 1 << b;
            }
        }
    }
    rows
}

fn main() {
    let axis = |i: u16| Axis {
        class: 10 + i,
        kind: AxisKind::Identity,
    };
    let method = |i: u16| MethodSignature::presence(10 + i, i, axis(i), axis(i + 1));
    let st = FrameStore::new(FrameRevisions {
        source: 1,
        relation_map: 1,
        coordinates: 1,
    });
    let (focus, scope) = (Focus { revision: 1 }, Scope { id: 1 });
    // Dense on the left, sparse on the right: the order matters.
    let ops: Vec<Rows> = [48, 32, 16, 4, 1]
        .iter()
        .enumerate()
        .map(|(i, d)| fill(i as u64 + 3, *d))
        .collect();
    let bound = ops
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let m = method(i as u16);
            let at = Guards {
                frame: st.seal(),
                focus,
                scope,
            };
            bind_dr64(
                &st,
                m,
                m,
                &at,
                (focus, scope),
                Operand {
                    rows: r,
                    validity: None,
                },
            )
            .unwrap()
        })
        .collect();
    let chain = Chain::new(&st, bound).unwrap();
    let reference = chain.eval(&all_orders(5)[0]).unwrap().0;
    println!("{:<40} {:>8} {:>10}", "order", "row_ors", "ns/eval");
    for o in all_orders(5) {
        let (rows, work) = chain.eval(&o).unwrap();
        assert_eq!(rows, reference);
        let mut samples = Vec::new();
        for _ in 0..7 {
            let t = Instant::now();
            for _ in 0..1000 {
                black_box(chain.eval(black_box(&o)).unwrap());
            }
            samples.push(t.elapsed().as_nanos() / 1000);
        }
        samples.sort();
        println!("{:<40} {:>8} {:>10}", show(&o), work.row_ors, samples[3]);
    }
    let law = admit_composition_assoc(LawShape::for_method(method(0), Primitive::BooleanCompose))
        .unwrap();
    let orders = all_orders(5);
    let (from, to) = (&orders[0], orders.last().unwrap());
    let mut us = Vec::new();
    for _ in 0..5 {
        let t = Instant::now();
        prove_reassociation(&law, from, to).unwrap();
        us.push(t.elapsed().as_micros());
    }
    us.sort();
    println!(
        "\ncold egglog proof {} = {} (reported, not a gate): median {} µs",
        show(from),
        show(to),
        us[2]
    );
}

fn show(o: &RelExpr) -> String {
    match o {
        RelExpr::Leaf(i) => format!("r{i}"),
        RelExpr::Comp(a, b) => format!("({} {})", show(a), show(b)),
    }
}
