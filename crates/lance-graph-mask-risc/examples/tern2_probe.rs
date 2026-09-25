//! The two-level ternlog lowering (`Lowering::Tern2`) through the PRODUCTION
//! path, against the same chain forced onto the tiled path.
//!
//! For 4- and 5-plane chains, `Count` and `Keep` (into an `Out::Mask`), both
//! via `execute_compiled`. Every arm is asserted equal to the tiled result.
//!
//! `RUSTFLAGS="-C target-cpu=native" cargo run --release -p lance-graph-mask-risc --example tern2_probe`

use std::time::Instant;

use lance_graph_mask_risc::exec::{execute_compiled, Scratch};
use lance_graph_mask_risc::{
    words_for, Foreign, Lowering, MaskOp, Operand, Out, Planes, Program, Terminal, FUSED_SLOT_CAP,
};

fn lcg(seed: &mut u64) -> u64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *seed >> 11
}

fn median<T>(reps: usize, mut f: impl FnMut() -> T) -> (f64, T) {
    let mut ts = Vec::with_capacity(reps);
    let mut last = None;
    for _ in 0..reps {
        let t = Instant::now();
        last = Some(std::hint::black_box(f()));
        ts.push(t.elapsed().as_nanos() as f64);
    }
    ts.sort_by(|a, b| a.total_cmp(b));
    (ts[reps / 2], last.expect("reps > 0"))
}

fn p(i: u16) -> Operand {
    Operand::Plane(i)
}
fn s(i: u16) -> Operand {
    Operand::Scratch(i)
}

fn main() {
    let n = 1usize << 20;
    let reps = 41;
    let mut seed = 0x7E52;
    let ms: Vec<Vec<u64>> = (0..5)
        .map(|_| {
            (0..words_for(n))
                .map(|_| lcg(&mut seed) ^ (lcg(&mut seed) << 21))
                .collect()
        })
        .collect();
    let masks: Vec<&[u64]> = ms.iter().map(|v| v.as_slice()).collect();
    let planes = Planes {
        n_rows: n,
        masks: &masks,
        lanes: &[],
    };
    let chains: [(&str, Vec<MaskOp>, u16); 2] = [
        (
            "4p (a&b)|(c&d)",
            vec![
                MaskOp::And {
                    a: p(0),
                    b: p(1),
                    dst: 0,
                },
                MaskOp::And {
                    a: p(2),
                    b: p(3),
                    dst: 1,
                },
                MaskOp::Or {
                    a: s(0),
                    b: s(1),
                    dst: 2,
                },
            ],
            2,
        ),
        (
            "5p ((a|b)&c)^(d&!e)",
            vec![
                MaskOp::Or {
                    a: p(0),
                    b: p(1),
                    dst: 0,
                },
                MaskOp::And {
                    a: s(0),
                    b: p(2),
                    dst: 0,
                },
                MaskOp::AndNot {
                    a: p(3),
                    b: p(4),
                    dst: 1,
                },
                MaskOp::Xor {
                    a: s(0),
                    b: s(1),
                    dst: 2,
                },
            ],
            2,
        ),
    ];
    let far = FUSED_SLOT_CAP as u16;
    println!(
        "{:>22}  {:>10} {:>10} {:>6}   {:>10} {:>10} {:>6}",
        "chain", "count.tiled", "count.t2", "x", "keep.tiled", "keep.t2", "x"
    );
    for (name, ops, last) in chains {
        let mut tops = ops.clone();
        tops.push(MaskOp::Or {
            a: s(last),
            b: s(last),
            dst: far,
        });
        let mk = |ops: Vec<MaskOp>, slot: u16, keep: bool| {
            Program::new(
                ops,
                if keep {
                    Terminal::Keep { mask: s(slot) }
                } else {
                    Terminal::Count { mask: s(slot) }
                },
            )
        };
        let (fc, tc) = (mk(ops.clone(), last, false), mk(tops.clone(), far, false));
        let (fk, tk) = (mk(ops.clone(), last, true), mk(tops, far, true));
        let (cfc, ctc, cfk, ctk) = (fc.compile(), tc.compile(), fk.compile(), tk.compile());
        assert!(matches!(cfc.lowering(), Lowering::Tern2(_)));
        assert!(matches!(cfk.lowering(), Lowering::Tern2(_)));
        assert!(matches!(ctc.lowering(), Lowering::Tiled));
        let mut sc = Scratch::for_program(&tc, n).expect("scratch");
        let (t_tiled, v_tiled) = median(reps, || {
            execute_compiled(&ctc, &planes, &Foreign::NONE, &mut sc, Out::None, 0..n).unwrap()
        });
        let (t_t2, v_t2) = median(reps, || {
            execute_compiled(
                &cfc,
                &planes,
                &Foreign::NONE,
                &mut Scratch::new(0, 0),
                Out::None,
                0..n,
            )
            .unwrap()
        });
        assert_eq!(v_tiled, v_t2, "{name}: count disagrees");
        let mut ot = vec![0u64; words_for(n)];
        let mut of = vec![0u64; words_for(n)];
        let mut sk = Scratch::for_program(&tk, n).expect("scratch");
        let (k_tiled, _) = median(reps, || {
            execute_compiled(
                &ctk,
                &planes,
                &Foreign::NONE,
                &mut sk,
                Out::Mask(&mut ot),
                0..n,
            )
            .unwrap()
        });
        let (k_t2, _) = median(reps, || {
            execute_compiled(
                &cfk,
                &planes,
                &Foreign::NONE,
                &mut Scratch::new(0, 0),
                Out::Mask(&mut of),
                0..n,
            )
            .unwrap()
        });
        assert_eq!(ot, of, "{name}: keep disagrees");
        println!(
            "{name:>22}  {:>10.1} {:>10.1} {:>5.2}x   {:>10.1} {:>10.1} {:>5.2}x",
            t_tiled / 1e3,
            t_t2 / 1e3,
            t_tiled / t_t2,
            k_tiled / 1e3,
            k_t2 / 1e3,
            k_tiled / k_t2
        );
    }
}
