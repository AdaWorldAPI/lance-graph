//! Absolute execution extent.
//!
//! `execute_extent(.., lo..hi)` evaluates the SAME program over the rows of
//! `[lo, hi)`, in the same absolute row coordinates as `Planes`. It is an
//! outer restriction: a program `Range [1000, 2000)` over the extent
//! `[1500, 1700)` means `[1000, 2000) ∩ [1500, 1700)`, and lane element `r`
//! is row `r` whatever the extent.
//!
//! The gates:
//! - the edge matrix: every result equals a scalar oracle that knows nothing
//!   about tiles, words or edges;
//! - split composition: whole == the merge of any partition, in any order;
//! - the non-rebasing falsifier: an extent that does not start at row 0 must
//!   read the absolute rows, and the rebased reading is shown to differ;
//! - the structural gate (tiles visited scale with the extent, not with
//!   `n_rows`) is in-crate: `exec::extent_tile_tests`, since the tile plan
//!   is executor detail, not API;
//! - foreign planes are addressed by key and are never sliced by the extent.

use lance_graph_mask_risc::exec::{execute_extent, execute_into, Scratch};
use lance_graph_mask_risc::{
    ExecError, Foreign, ForeignPlane, LaneRef, MaskOp, Operand, Out, Planes, Pred, Program,
    Terminal, Value,
};

fn lcg(seed: &mut u64) -> u64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *seed >> 11
}

fn words(n: usize) -> usize {
    n.div_ceil(64)
}

fn plane(n: usize, set: impl Fn(usize) -> bool) -> Vec<u64> {
    let mut w = vec![0u64; words(n)];
    for r in (0..n).filter(|&r| set(r)) {
        w[r / 64] |= 1 << (r % 64);
    }
    w
}

fn bit(p: &[u64], r: usize) -> bool {
    p[r / 64] >> (r % 64) & 1 == 1
}

/// The two physical shapes a `Range ∩ plane` program can take: the #1268
/// fused fold, and the tiled path that writes the relation into scratch.
#[derive(Clone, Copy, Debug)]
enum Shape {
    Fused,
    Tiled,
}

#[derive(Clone, Copy, Debug)]
enum Term {
    Count,
    Any,
    All,
    Keep,
}

fn program(shape: Shape, lo: u32, hi: u32, term: Term) -> Program {
    let (ops, m) = match shape {
        Shape::Fused => (
            vec![MaskOp::Pred {
                pred: Pred::Range { lo, hi },
                under: Some(Operand::Plane(0)),
                dst: 0,
            }],
            Operand::Scratch(0),
        ),
        Shape::Tiled => (
            vec![
                MaskOp::Pred {
                    pred: Pred::Range { lo, hi },
                    under: None,
                    dst: 0,
                },
                MaskOp::And {
                    a: Operand::Scratch(0),
                    b: Operand::Plane(0),
                    dst: 1,
                },
            ],
            Operand::Scratch(1),
        ),
    };
    let terminal = match term {
        Term::Count => Terminal::Count { mask: m },
        Term::Any => Terminal::Any { mask: m },
        Term::All => Terminal::All { mask: m },
        Term::Keep => Terminal::Keep { mask: m },
    };
    Program::new(ops, terminal)
}

const POISON: u64 = 0xA5A5_5A5A_C3C3_3C3C;

/// Run `p` over `ext`. `Keep` writes into a poisoned buffer, which is
/// returned so the caller can check which bits were touched.
fn run(p: &Program, planes: &Planes<'_>, ext: std::ops::Range<usize>) -> (Value, Option<Vec<u64>>) {
    let n = planes.n_rows;
    let mut s = Scratch::for_program(p, n).expect("scratch");
    if matches!(p.terminal, Terminal::Keep { .. }) {
        let mut o = vec![POISON; words(n)];
        let v = execute_extent(p, planes, &Foreign::NONE, &mut s, Out::Mask(&mut o), ext)
            .expect("keep");
        (v, Some(o))
    } else {
        let v = execute_extent(p, planes, &Foreign::NONE, &mut s, Out::None, ext).expect("execute");
        (v, None)
    }
}

/// Check one (program range, extent, plane, shape, terminal) case against a
/// scalar oracle over absolute rows.
fn check(pl: &[u64], n: usize, prog: (u32, u32), ext: (usize, usize), shape: Shape, term: Term) {
    let masks: [&[u64]; 1] = [pl];
    let planes = Planes {
        n_rows: n,
        masks: &masks,
        lanes: &[],
    };
    let p = program(shape, prog.0, prog.1, term);
    let member = |r: usize| (prog.0 as usize..prog.1 as usize).contains(&r) && bit(pl, r);
    let in_ext = |r: usize| (ext.0..ext.1).contains(&r);
    let tag = format!("{shape:?} {term:?} n={n} prog={prog:?} ext={ext:?}");
    let (v, out) = run(&p, &planes, ext.0..ext.1);
    match term {
        Term::Count => {
            let want = (ext.0..ext.1).filter(|&r| member(r)).count();
            assert_eq!(v, Value::Count(want), "{tag}");
        }
        Term::Any => {
            let want = (ext.0..ext.1).any(member);
            assert_eq!(v, Value::Bool(want), "{tag}");
        }
        Term::All => {
            let want = (ext.0..ext.1).all(member);
            assert_eq!(v, Value::Bool(want), "{tag}");
        }
        Term::Keep => {
            let o = out.expect("keep writes out");
            for r in 0..n {
                let want = if in_ext(r) {
                    member(r)
                } else {
                    // Outside the extent the caller's bits survive.
                    POISON >> (r % 64) & 1 == 1
                };
                assert_eq!(bit(&o, r), want, "{tag} row {r}");
            }
        }
    }
}

#[test]
fn the_edge_matrix_agrees_with_the_scalar_oracle() {
    let n = 1317; // not a multiple of 64, three 8-word tiles
    let mut seed = 0xE17E_u64;
    let scattered: Vec<bool> = (0..n).map(|_| lcg(&mut seed).is_multiple_of(3)).collect();
    let planes: [(&str, Vec<u64>); 3] = [
        ("zero", plane(n, |_| false)),
        ("ones", plane(n, |_| true)),
        ("scattered", plane(n, |r| scattered[r])),
    ];
    let extents: &[(usize, usize)] = &[
        (0, 0),      // empty at zero
        (65, 65),    // empty mid-word
        (0, n),      // whole
        (5, 6),      // one row
        (3, 64),     // unaligned start, aligned end
        (0, 70),     // aligned start, unaligned end
        (3, 70),     // both unaligned, cross-word
        (70, 75),    // single word
        (64, 128),   // one aligned word pair
        (1, 511),    // inside one tile
        (5, 600),    // cross-tile
        (n - 30, n), // tail extent, N % 64 != 0
        (n - 1, n),  // last row
        (300, 500),  // the named program-relation cases below
        (100, 900),
    ];
    let progs: &[(u32, u32)] = &[
        (0, n as u32), // program contains every extent
        (100, 900),    // contains (300,500); partial vs most
        (100, 200),    // disjoint from (300,500)
        (300, 500),    // equal to one extent, inside (100,900)
        (250, 1300),   // partial overlap with (100,900)
        (65, 65),      // empty program range
    ];
    let mut cases = 0usize;
    for (_, pl) in &planes {
        for &prog in progs {
            for &ext in extents {
                for shape in [Shape::Fused, Shape::Tiled] {
                    let terms: &[Term] = match shape {
                        // The fused seam is Count/Any only; All and Keep
                        // stay on the tiled path by construction.
                        Shape::Fused => &[Term::Count, Term::Any],
                        Shape::Tiled => &[Term::Count, Term::Any, Term::All, Term::Keep],
                    };
                    for &term in terms {
                        check(pl, n, prog, ext, shape, term);
                        cases += 1;
                    }
                }
            }
        }
    }
    assert_eq!(cases, 3 * 6 * 15 * 6);
}

#[test]
fn a_tiny_extent_in_a_million_rows_reads_absolute_rows() {
    let n = 1 << 20;
    let pl = plane(n, |r| r % 5 != 0);
    for shape in [Shape::Fused, Shape::Tiled] {
        for ext in [(500_001, 500_002), (500_001, 500_065), (777_000, 777_512)] {
            for prog in [(0, n as u32), (500_000, 600_000), (1, 500_001)] {
                check(&pl, n, prog, ext, shape, Term::Count);
                check(&pl, n, prog, ext, shape, Term::Any);
            }
        }
    }
}

/// Program `Range [1000, 2000)` over the extent `[1500, 1700)` is exactly
/// rows `1500..1700`. Both rebased readings — the program's range taken
/// relative to the extent start, or the extent taken relative to the
/// program's start — select a different set and are shown to.
#[test]
fn the_extent_restricts_and_never_rebases_the_program() {
    let n = 4133;
    let ones = plane(n, |_| true);
    let masks: [&[u64]; 1] = [&ones];
    let planes = Planes {
        n_rows: n,
        masks: &masks,
        lanes: &[],
    };
    let absolute = (1500..1700).filter(|r| (1000..2000).contains(r)).count();
    let program_rebased = (1500..1700).filter(|r| (2500..3500).contains(r)).count();
    let extent_rebased = (1000..1200).filter(|r| (1000..2000).contains(r)).count();
    assert_eq!(absolute, 200);
    assert_ne!(program_rebased, absolute);
    for shape in [Shape::Fused, Shape::Tiled] {
        let (v, _) = run(
            &program(shape, 1000, 2000, Term::Count),
            &planes,
            1500..1700,
        );
        assert_eq!(v, Value::Count(absolute), "{shape:?}");
    }
    // Keep: exactly rows 1500..1700 carry the relation, in absolute words.
    let (_, out) = run(
        &program(Shape::Tiled, 1000, 2000, Term::Keep),
        &planes,
        1500..1700,
    );
    let o = out.expect("keep");
    for r in 0..n {
        let want = if (1500..1700).contains(&r) {
            true
        } else {
            POISON >> (r % 64) & 1 == 1
        };
        assert_eq!(bit(&o, r), want, "row {r}");
    }
    // The extent-rebased reading would put the selected bits at 1000..1200.
    assert_eq!(extent_rebased, 200);
    assert!(
        (1000..1200).any(|r| bit(&o, r) != ((1000..1200).contains(&r))),
        "the Keep result must not look like the extent-rebased reading"
    );
}

/// Distinct lane values at the word seams: an extent starting at 65 or 129
/// must sum those ABSOLUTE rows. A worker-local reading (lane element 0 is
/// the extent's first row) sums a different window, and the test proves the
/// two differ, so a rebasing executor cannot pass.
#[test]
fn values_are_read_at_absolute_rows_not_worker_local_ones() {
    let n = 1317;
    let mut lane: Vec<i32> = (0..n as i32).map(|r| r % 7).collect();
    for (i, r) in [63usize, 64, 65, 127, 128, 129].into_iter().enumerate() {
        lane[r] = 1000 * (i as i32 + 1);
    }
    let ones = plane(n, |_| true);
    let masks: [&[u64]; 1] = [&ones];
    let lanes = [LaneRef::I32(&lane)];
    let planes = Planes {
        n_rows: n,
        masks: &masks,
        lanes: &lanes,
    };
    let p = Program::new(
        vec![MaskOp::Pred {
            pred: Pred::Range {
                lo: 0,
                hi: n as u32,
            },
            under: Some(Operand::Plane(0)),
            dst: 0,
        }],
        Terminal::MaskedSumI32 {
            mask: Operand::Scratch(0),
            lane: 0,
        },
    );
    let sum = |r: std::ops::Range<usize>| lane[r].iter().map(|&v| i64::from(v)).sum::<i64>();
    for (lo, hi) in [(65usize, 200usize), (129, 300), (63, 66), (127, 130)] {
        let absolute = sum(lo..hi);
        let worker_local = sum(0..hi - lo);
        assert_ne!(
            absolute, worker_local,
            "the falsifier must separate the readings at {lo}"
        );
        let mut s = Scratch::for_program(&p, n).expect("scratch");
        let v =
            execute_extent(&p, &planes, &Foreign::NONE, &mut s, Out::None, lo..hi).expect("sum");
        assert_eq!(v, Value::SumI64(absolute), "extent {lo}..{hi}");
    }
}

/// Merge the partial results of a partition with the terminal's shipped law.
fn merge(a: Value, b: Value) -> Value {
    match (a, b) {
        (Value::Count(x), Value::Count(y)) => Value::Count(x + y),
        (Value::SumI64(x), Value::SumI64(y)) => Value::SumI64(x + y),
        _ => panic!("merge: use merge_bool / merge_opt for {a:?} {b:?}"),
    }
}

#[derive(Clone, Copy)]
enum Law {
    Add,
    Or,
    And,
    Min,
    Max,
}

fn merge_with(law: Law, a: Value, b: Value) -> Value {
    match (law, a, b) {
        (Law::Add, a, b) => merge(a, b),
        (Law::Or, Value::Bool(x), Value::Bool(y)) => Value::Bool(x || y),
        (Law::And, Value::Bool(x), Value::Bool(y)) => Value::Bool(x && y),
        (Law::Min, Value::OptI32(x), Value::OptI32(y)) => Value::OptI32(match (x, y) {
            (Some(x), Some(y)) => Some(x.min(y)),
            (x, None) => x,
            (None, y) => y,
        }),
        (Law::Max, Value::OptI32(x), Value::OptI32(y)) => Value::OptI32(match (x, y) {
            (Some(x), Some(y)) => Some(x.max(y)),
            (x, None) => x,
            (None, y) => y,
        }),
        _ => panic!("law/value mismatch"),
    }
}

#[test]
fn whole_execution_equals_the_merge_of_any_partition_in_any_order() {
    for n in [1317usize, 4096 + 37] {
        let mut seed = 0x5_1117 ^ n as u64;
        let scattered: Vec<bool> = (0..n).map(|_| lcg(&mut seed).is_multiple_of(4)).collect();
        let pl = plane(n, |r| scattered[r]);
        let vals: Vec<i32> = (0..n)
            .map(|_| (lcg(&mut seed) % 2001) as i32 - 1000)
            .collect();
        let masks: [&[u64]; 1] = [&pl];
        let lanes = [LaneRef::I32(&vals)];
        let planes = Planes {
            n_rows: n,
            masks: &masks,
            lanes: &lanes,
        };
        let (plo, phi) = (37u32, n as u32 - 11);
        let tiled_on = |t: Terminal| {
            let mut p = program(Shape::Tiled, plo, phi, Term::Count);
            p.terminal = t;
            p
        };
        let m = Operand::Scratch(1);
        let programs: Vec<(Program, Law)> = vec![
            (program(Shape::Fused, plo, phi, Term::Count), Law::Add),
            (program(Shape::Fused, plo, phi, Term::Any), Law::Or),
            (program(Shape::Tiled, plo, phi, Term::Count), Law::Add),
            (program(Shape::Tiled, plo, phi, Term::Any), Law::Or),
            (program(Shape::Tiled, plo, phi, Term::All), Law::And),
            (
                tiled_on(Terminal::MaskedSumI32 { mask: m, lane: 0 }),
                Law::Add,
            ),
            (
                tiled_on(Terminal::MaskedMinI32 { mask: m, lane: 0 }),
                Law::Min,
            ),
            (
                tiled_on(Terminal::MaskedMaxI32 { mask: m, lane: 0 }),
                Law::Max,
            ),
        ];
        // Two-way splits at the named seams; three-way splits at random.
        let mut partitions: Vec<Vec<usize>> = [0, 1, 63, 64, 65, 127, 128, 129, n / 2, n - 1, n]
            .into_iter()
            .map(|k| vec![0, k, n])
            .collect();
        for _ in 0..40 {
            let a = (lcg(&mut seed) as usize) % (n + 1);
            let b = (lcg(&mut seed) as usize) % (n + 1);
            let (a, b) = (a.min(b), a.max(b));
            partitions.push(vec![0, a, b, n]);
        }
        for (p, law) in &programs {
            let (whole, _) = run(p, &planes, 0..n);
            for cuts in &partitions {
                let parts: Vec<Value> = cuts
                    .windows(2)
                    .map(|w| run(p, &planes, w[0]..w[1]).0)
                    .collect();
                // Forward, reverse and a rotated order all merge to `whole`.
                for order in [
                    (0..parts.len()).collect::<Vec<_>>(),
                    (0..parts.len()).rev().collect(),
                    (0..parts.len()).map(|i| (i + 1) % parts.len()).collect(),
                ] {
                    let merged = order
                        .iter()
                        .map(|&i| parts[i])
                        .reduce(|a, b| merge_with(*law, a, b))
                        .expect("non-empty");
                    assert_eq!(merged, whole, "n={n} cuts={cuts:?} order={order:?}");
                }
            }
        }
        // Keep: disjoint extents write disjoint bits of ONE absolute buffer;
        // any order reproduces the whole-population Keep.
        let keep = program(Shape::Tiled, plo, phi, Term::Keep);
        let mut s = Scratch::for_program(&keep, n).expect("scratch");
        let mut whole = vec![0u64; words(n)];
        execute_into(
            &keep,
            &planes,
            &Foreign::NONE,
            &mut s,
            Out::Mask(&mut whole),
        )
        .expect("whole keep");
        for cuts in &partitions {
            let spans: Vec<(usize, usize)> = cuts.windows(2).map(|w| (w[0], w[1])).collect();
            for order in [
                (0..spans.len()).collect::<Vec<_>>(),
                (0..spans.len()).rev().collect(),
            ] {
                let mut o = vec![0u64; words(n)];
                for &i in &order {
                    let (lo, hi) = spans[i];
                    execute_extent(
                        &keep,
                        &planes,
                        &Foreign::NONE,
                        &mut s,
                        Out::Mask(&mut o),
                        lo..hi,
                    )
                    .expect("part keep");
                }
                assert_eq!(o, whole, "keep n={n} cuts={cuts:?} order={order:?}");
            }
        }
    }
}

#[test]
fn bad_extents_and_unmergeable_terminals_are_refused_before_execution() {
    let n = 200;
    let pl = plane(n, |r| r % 2 == 0);
    let keys: Vec<u32> = (0..n as u32).map(|r| r % 4).collect();
    let vals: Vec<i32> = (0..n as i32).collect();
    let masks: [&[u64]; 1] = [&pl];
    let lanes = [LaneRef::U32(&keys), LaneRef::I32(&vals)];
    let planes = Planes {
        n_rows: n,
        masks: &masks,
        lanes: &lanes,
    };
    let count = program(Shape::Tiled, 0, n as u32, Term::Count);
    let mut s = Scratch::for_program(&count, n).expect("scratch");
    for (lo, hi) in [(10usize, 9usize), (0, n + 1), (n + 1, n + 2)] {
        #[allow(clippy::reversed_empty_ranges)]
        let r = execute_extent(&count, &planes, &Foreign::NONE, &mut s, Out::None, lo..hi);
        assert_eq!(r, Err(ExecError::ExtentOutOfRange { lo, hi, n_rows: n }));
    }
    let group = Program::new(
        vec![],
        Terminal::GroupSumI32 {
            mask: Operand::Plane(0),
            key: 0,
            val: 1,
        },
    );
    let mut sink = vec![0i64; 4];
    let mut s = Scratch::for_program(&group, n).expect("scratch");
    assert_eq!(
        execute_extent(
            &group,
            &planes,
            &Foreign::NONE,
            &mut s,
            Out::I64(&mut sink),
            10..20
        ),
        Err(ExecError::ExtentUnsupported {
            what: "GroupSumI32"
        })
    );
    assert_eq!(sink, vec![0; 4], "a refusal writes nothing");
    // The whole extent accepts every terminal: it IS `execute_into`.
    assert_eq!(
        execute_extent(
            &group,
            &planes,
            &Foreign::NONE,
            &mut s,
            Out::I64(&mut sink),
            0..n
        ),
        Ok(Value::GroupSummed)
    );
    // A partial Keep needs its population-addressed sink.
    let keep = program(Shape::Tiled, 0, n as u32, Term::Keep);
    let mut s = Scratch::for_program(&keep, n).expect("scratch");
    assert_eq!(
        execute_extent(&keep, &planes, &Foreign::NONE, &mut s, Out::None, 10..20),
        Err(ExecError::TerminalNeedsOut { what: "Keep" })
    );
}

/// `Gather` reads its foreign plane by KEY. Slicing the local table must not
/// slice the foreign one: here every local row of the extent names a foreign
/// key below the extent's own row numbers, so a foreign plane restricted to
/// the extent would see none of them.
#[test]
fn the_extent_never_slices_a_foreign_plane() {
    let n = 1317;
    let foreign_rows = 100;
    let fwords = plane(foreign_rows, |k| k % 3 == 0);
    let fk: Vec<u32> = (0..n as u32)
        .map(|r| (r * 7) % foreign_rows as u32)
        .collect();
    let lanes = [LaneRef::U32(&fk)];
    let planes = Planes {
        n_rows: n,
        masks: &[],
        lanes: &lanes,
    };
    let fplanes = [ForeignPlane {
        words: &fwords,
        rows: foreign_rows,
    }];
    let foreign = Foreign {
        planes: &fplanes,
        lanes: &[],
    };
    let p = Program::new(
        vec![MaskOp::Gather {
            lane: 0,
            foreign: 0,
            dst: 0,
        }],
        Terminal::Count {
            mask: Operand::Scratch(0),
        },
    );
    let (lo, hi) = (600usize, 900usize);
    let want = (lo..hi).filter(|&r| bit(&fwords, fk[r] as usize)).count();
    assert!(want > 0 && want < hi - lo, "the case must be non-trivial");
    let mut s = Scratch::for_program(&p, n).expect("scratch");
    let v = execute_extent(&p, &planes, &foreign, &mut s, Out::None, lo..hi).expect("gather");
    assert_eq!(v, Value::Count(want));
}
