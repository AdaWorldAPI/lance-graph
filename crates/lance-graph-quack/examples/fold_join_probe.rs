//! D-RPF-9 — Fold-Join deforestation probe: when two address-aligned
//! membership sources meet in one terminal, which existing lowering already
//! avoids the intermediate, and what does each arm actually write?
//!
//! Nothing here is a production primitive. Every arm runs the SHIPPED
//! `lance-graph-mask-risc` executor (or `lance-graph-quack`'s lowering into
//! it). The one probe-local loop (`H`) is the row-at-a-time oracle every arm
//! is diffed against; its timing is printed as a scalar reference, never as
//! evidence about a backend.
//!
//! Arms, all answering `COUNT(A ∧ B)` (or the named Boolean variant):
//!
//! | arm | execution |
//! |---|---|
//! | A  | `Keep(A)` and `Keep(B)` into caller masks, then `And → Count` over them |
//! | B  | two RESIDENT masks, `And → Count` (`Lowering::Ternlog`, no slot) |
//! | C  | `Pred A`, `Pred B`, `And`, `Count` in one program (tiled) |
//! | D  | `Pred A`, then `Pred B under A`, `Count` (tiled, accumulator gate) |
//! | Dp | RESIDENT mask A gates `Pred B` (`under: Plane`), `Count` |
//! | E  | `AndNot` / `Xor` / three-input `Ternlog` over resident masks |
//! | F  | `lance-graph-quack` `lower` / `lower_fused` of `A AND B` |
//! | H  | per-row oracle (no `ndarray`, no mask) |
//!
//! The CE64 section runs C and D over `ValueTenant::MaterializedEdges` read in
//! place through `Pred::MatchFacet16Strided` and diffs them against the
//! canonical `causal_edge::CausalEdge64` accessors.
//!
//! ```text
//! CARGO_PROFILE_RELEASE_DEBUG=0 cargo run --release -p lance-graph-quack --example fold_join_probe
//! ```
//!
//! Timings are printed, never asserted. Every count is asserted.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use causal_edge::CausalEdge64;
use lance_graph_contract::canonical_node::{ValueTenant, NODE_ROW_STRIDE, VALUE_SLAB_ROW_OFFSET};
use lance_graph_mask_risc::exec::{execute_extent, execute_into, Scratch};
use lance_graph_mask_risc::{
    tile_words_for, words_for, ExecError, Foreign, LaneRef, Lowering, MaskOp, Operand, Out, Planes,
    Pred, Program, StridedRef, Terminal, Value, TILE_WORDS,
};
use lance_graph_quack::{lower, lower_fused, Agg, Cmp, Col, Filter, Query};

// ───────────────────────── allocation counter ─────────────────────────

struct Counting;
static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);

// SAFETY: a pure pass-through to `System`; the counters are the only addition.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        // SAFETY: same layout, same contract as the caller's.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from `alloc` above with this `layout`.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn alloc_snapshot() -> (usize, usize) {
    (
        ALLOCS.load(Ordering::Relaxed),
        BYTES.load(Ordering::Relaxed),
    )
}

// ───────────────────────── data ─────────────────────────

/// SplitMix64. An LCG's low bytes are periodic, and the first version of
/// this probe used them: every field came out uniform to the row, so a care
/// shifted by one bit matched exactly as often as the right one and the
/// can-fire check could not fire.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

const SCALE: i32 = 1_000_000;

/// Two `i32` lanes. `x < sel_a·SCALE` is predicate A, `y < sel_b·SCALE` is B.
/// `clustered` sorts `x` so A's survivors form one run (dead words become dead
/// tiles); otherwise survivors are scattered.
fn lanes(n: usize, seed: u64, clustered: bool) -> (Vec<i32>, Vec<i32>) {
    let mut r = Rng(seed);
    let mut x: Vec<i32> = (0..n).map(|_| (r.next() % SCALE as u64) as i32).collect();
    let y: Vec<i32> = (0..n).map(|_| (r.next() % SCALE as u64) as i32).collect();
    if clustered {
        x.sort_unstable();
    }
    (x, y)
}

fn thresh(sel: f64) -> i32 {
    (sel * f64::from(SCALE)) as i32
}

/// The oracle: per row, no mask, no `ndarray`.
fn oracle(x: &[i32], y: &[i32], ta: i32, tb: i32) -> usize {
    x.iter()
        .zip(y)
        .filter(|(a, b)| **a < ta && **b < tb)
        .count()
}

fn bits_of(lane: &[i32], t: i32) -> Vec<u64> {
    let mut m = vec![0u64; words_for(lane.len())];
    for (i, v) in lane.iter().enumerate() {
        if *v < t {
            m[i / 64] |= 1 << (i % 64);
        }
    }
    m
}

fn dead_fractions(m: &[u64]) -> (f64, f64) {
    let dead_words = m.iter().filter(|w| **w == 0).count();
    let tiles: Vec<&[u64]> = m.chunks(TILE_WORDS).collect();
    let dead_tiles = tiles.iter().filter(|t| t.iter().all(|w| *w == 0)).count();
    (
        dead_words as f64 / m.len().max(1) as f64,
        dead_tiles as f64 / tiles.len().max(1) as f64,
    )
}

// ───────────────────────── one arm ─────────────────────────

struct Report {
    name: &'static str,
    lowering: String,
    predicates: usize,
    mask_passes: usize,
    scratch_bytes: usize,
    population_bytes: usize,
    allocs_per_exec: f64,
    ns: f64,
}

fn lowering_name(p: &Program) -> String {
    match p.compile().lowering() {
        Lowering::Range(_) => "Range".into(),
        Lowering::Ternlog(_) => "Ternlog(no slot)".into(),
        Lowering::TernlogKeep(_) => "TernlogKeep".into(),
        Lowering::Tern2(_) => "Tern2".into(),
        Lowering::Tern3(_) => "Tern3".into(),
        Lowering::Tiled => "Tiled".into(),
    }
}

fn count(v: Result<Value, ExecError>) -> usize {
    match v {
        Ok(Value::Count(c)) => c,
        other => panic!("expected a Count, got {other:?}"),
    }
}

/// Run `p` `reps` times over `planes` with a pre-built scratch; return the
/// count and a [`Report`] (scratch is sized BEFORE the timer and printed).
fn run_arm(
    name: &'static str,
    p: &Program,
    planes: &Planes<'_>,
    reps: usize,
    population_bytes: usize,
) -> (usize, Report) {
    let mut scratch = Scratch::for_program(p, planes.n_rows).expect("scratch");
    let scratch_bytes = p.scratch_slots as usize * tile_words_for(planes.n_rows) * 8;
    let first = count(execute_into(
        p,
        planes,
        &Foreign::NONE,
        &mut scratch,
        Out::None,
    ));
    let (a0, _) = alloc_snapshot();
    let t = Instant::now();
    let mut c = 0;
    for _ in 0..reps {
        c = count(execute_into(
            p,
            planes,
            &Foreign::NONE,
            &mut scratch,
            Out::None,
        ));
    }
    let ns = t.elapsed().as_nanos() as f64 / reps as f64;
    let (a1, _) = alloc_snapshot();
    assert_eq!(c, first);
    let h = p.op_histogram();
    (
        c,
        Report {
            name,
            lowering: lowering_name(p),
            predicates: h.predicates,
            mask_passes: h.mask_passes(),
            scratch_bytes,
            population_bytes,
            allocs_per_exec: (a1 - a0) as f64 / reps as f64,
            ns,
        },
    )
}

fn print(r: &Report) {
    println!(
        "  {:<4} {:<17} preds {} passes {}  scratch {:>6} B  population-intermediate {:>6} B  allocs/exec {:.1}  {:>9.0} ns",
        r.name,
        r.lowering,
        r.predicates,
        r.mask_passes,
        r.scratch_bytes,
        r.population_bytes,
        r.allocs_per_exec,
        r.ns
    );
}

// ───────────────────────── programs ─────────────────────────

fn pred_a(ta: i32, under: Option<Operand>, dst: u16) -> MaskOp {
    MaskOp::Pred {
        pred: Pred::LtI32 { lane: 0, t: ta },
        under,
        dst,
    }
}
fn pred_b(tb: i32, under: Option<Operand>, dst: u16) -> MaskOp {
    MaskOp::Pred {
        pred: Pred::LtI32 { lane: 1, t: tb },
        under,
        dst,
    }
}
fn count_of(m: Operand) -> Terminal {
    Terminal::Count { mask: m }
}
fn keep_of(m: Operand) -> Terminal {
    Terminal::Keep { mask: m }
}

/// Arm A, all three steps: two `Keep`s into caller masks, then the fold.
/// Returns the count; `ma`/`mb` are the materialised population masks.
/// The three programs arm A runs, built once outside the timer.
struct ArmA {
    keep_a: Program,
    keep_b: Program,
    fold: Program,
}

impl ArmA {
    fn new(ta: i32, tb: i32) -> Self {
        Self {
            keep_a: Program::new(vec![pred_a(ta, None, 0)], keep_of(Operand::Scratch(0))),
            keep_b: Program::new(vec![pred_b(tb, None, 0)], keep_of(Operand::Scratch(0))),
            fold: Program::new(
                vec![MaskOp::And {
                    a: Operand::Plane(0),
                    b: Operand::Plane(1),
                    dst: 0,
                }],
                count_of(Operand::Scratch(0)),
            ),
        }
    }

    /// All three steps: two `Keep`s into the caller's masks, then the fold.
    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        lanes: &[LaneRef<'_>],
        n: usize,
        ma: &mut [u64],
        mb: &mut [u64],
        sa: &mut Scratch<'_>,
        sb: &mut Scratch<'_>,
        sf: &mut Scratch<'_>,
    ) -> usize {
        let none: [&[u64]; 0] = [];
        let planes = Planes {
            n_rows: n,
            masks: &none,
            lanes,
        };
        execute_into(&self.keep_a, &planes, &Foreign::NONE, sa, Out::Mask(ma)).expect("keep A");
        execute_into(&self.keep_b, &planes, &Foreign::NONE, sb, Out::Mask(mb)).expect("keep B");
        let both: [&[u64]; 2] = [ma, mb];
        let fold_planes = Planes {
            n_rows: n,
            masks: &both,
            lanes: &[],
        };
        count(execute_into(
            &self.fold,
            &fold_planes,
            &Foreign::NONE,
            sf,
            Out::None,
        ))
    }
}

// ───────────────────────── sections ─────────────────────────

/// The selectivity sweep: every arm, sparse to dense, scattered and clustered.
fn sweep(n: usize, reps: usize) {
    println!("\n== sweep: COUNT(x < a AND y < b), n = {n}, B selectivity 0.5 ==");
    let tb = thresh(0.5);
    for clustered in [false, true] {
        let (x, y) = lanes(n, 0xF01D, clustered);
        let lanes_ = [LaneRef::I32(&x), LaneRef::I32(&y)];
        for sel in [0.001, 0.01, 0.1, 0.5, 0.9] {
            let ta = thresh(sel);
            let want = oracle(&x, &y, ta, tb);
            let ma = bits_of(&x, ta);
            let mb = bits_of(&y, tb);
            let (dw, dt) = dead_fractions(&ma);
            println!(
                "\n -- {} A sel {sel}: oracle {want}, A dead-word {:.3}, dead-tile {:.3}",
                if clustered { "clustered" } else { "scattered" },
                dw,
                dt
            );
            let words = words_for(n);
            let none: [&[u64]; 0] = [];
            let lane_planes = Planes {
                n_rows: n,
                masks: &none,
                lanes: &lanes_,
            };
            let resident: [&[u64]; 2] = [&ma, &mb];
            let res_planes = Planes {
                n_rows: n,
                masks: &resident,
                lanes: &lanes_,
            };

            // A — materialise both masks, then fold.
            {
                let arm = ArmA::new(ta, tb);
                let mut sa = Scratch::for_program(&arm.keep_a, n).unwrap();
                let mut sb = Scratch::for_program(&arm.keep_b, n).unwrap();
                let mut sf = Scratch::for_program(&arm.fold, n).unwrap();
                let mut bufa = vec![0u64; words];
                let mut bufb = vec![0u64; words];
                let c0 = arm.run(&lanes_, n, &mut bufa, &mut bufb, &mut sa, &mut sb, &mut sf);
                let (a0, _) = alloc_snapshot();
                let t = Instant::now();
                let mut c = 0;
                for _ in 0..reps {
                    c = arm.run(&lanes_, n, &mut bufa, &mut bufb, &mut sa, &mut sb, &mut sf);
                }
                let ns = t.elapsed().as_nanos() as f64 / reps as f64;
                let (a1, _) = alloc_snapshot();
                assert_eq!(c, want);
                assert_eq!(c0, want);
                print(&Report {
                    name: "A",
                    lowering: "Keep,Keep,Ternlog".into(),
                    predicates: 2,
                    mask_passes: 1,
                    scratch_bytes: 2 * tile_words_for(n) * 8,
                    population_bytes: 2 * words * 8,
                    allocs_per_exec: (a1 - a0) as f64 / reps as f64,
                    ns,
                });
            }
            // B — two resident masks, one fused fold.
            let pb = Program::new(
                vec![MaskOp::And {
                    a: Operand::Plane(0),
                    b: Operand::Plane(1),
                    dst: 0,
                }],
                count_of(Operand::Scratch(0)),
            );
            let (cb, rb) = run_arm("B", &pb, &res_planes, reps, 0);
            assert_eq!(cb, want);
            assert_eq!(rb.lowering, "Ternlog(no slot)");
            print(&rb);
            // C — two predicates, And, Count.
            let pc = Program::new(
                vec![
                    pred_a(ta, None, 0),
                    pred_b(tb, None, 1),
                    MaskOp::And {
                        a: Operand::Scratch(0),
                        b: Operand::Scratch(1),
                        dst: 2,
                    },
                ],
                count_of(Operand::Scratch(2)),
            );
            let (cc, rc) = run_arm("C", &pc, &lane_planes, reps, 0);
            assert_eq!(cc, want);
            print(&rc);
            // D — B under the accumulated A.
            let pd = Program::new(
                vec![
                    pred_a(ta, None, 0),
                    pred_b(tb, Some(Operand::Scratch(0)), 1),
                ],
                count_of(Operand::Scratch(1)),
            );
            let (cd, rd) = run_arm("D", &pd, &lane_planes, reps, 0);
            assert_eq!(cd, want);
            print(&rd);
            // Dp — resident mask A gates predicate B.
            let pdp = Program::new(
                vec![pred_b(tb, Some(Operand::Plane(0)), 0)],
                count_of(Operand::Scratch(0)),
            );
            let (cdp, rdp) = run_arm("Dp", &pdp, &res_planes, reps, 0);
            assert_eq!(cdp, want);
            print(&rdp);
            // F — quack's own lowering of `x < a AND y < b`.
            let q = Query {
                filter: Filter::and([
                    Filter::cmp(Col(0), Cmp::LtI32(ta)),
                    Filter::cmp(Col(1), Cmp::LtI32(tb)),
                ]),
                agg: Agg::Count,
            };
            let pf = lower(&q).expect("lower");
            let (cf, rf) = run_arm("F", &pf, &lane_planes, reps, 0);
            assert_eq!(cf, want);
            print(&rf);
            let pff = lower_fused(&q).expect("lower_fused");
            let (cff, rff) = run_arm("Ff", &pff, &lane_planes, reps, 0);
            assert_eq!(cff, want);
            print(&rff);
            // H — the scalar oracle, timed for scale only.
            let t = Instant::now();
            let mut h = 0;
            for _ in 0..reps {
                h = oracle(std::hint::black_box(&x), &y, ta, tb);
            }
            let ns = t.elapsed().as_nanos() as f64 / reps as f64;
            assert_eq!(h, want);
            println!("  H    scalar oracle (reference only)                                                     {ns:>9.0} ns");
        }
    }
}

/// E — the other two- and three-input folds over resident masks, each against
/// a per-word reference, each confirmed to take the no-slot lowering.
fn boolean_variants(n: usize) {
    println!("\n== E: AndNot / Xor / Or / Ternlog(maj) over resident masks ==");
    let (x, y) = lanes(n, 0xE, false);
    let ma = bits_of(&x, thresh(0.3));
    let mb = bits_of(&y, thresh(0.6));
    let mc: Vec<u64> = bits_of(&x, thresh(0.7))
        .iter()
        .zip(&bits_of(&y, thresh(0.2)))
        .map(|(a, b)| a ^ b)
        .collect();
    let planes_data: [&[u64]; 3] = [&ma, &mb, &mc];
    let planes = Planes {
        n_rows: n,
        masks: &planes_data,
        lanes: &[],
    };
    let pc = |f: fn(u64, u64, u64) -> u64| -> usize {
        ma.iter()
            .zip(&mb)
            .zip(&mc)
            .map(|((a, b), c)| f(*a, *b, *c).count_ones() as usize)
            .sum()
    };
    let p = |o: MaskOp| Program::new(vec![o], count_of(Operand::Scratch(0)));
    let (pa, pb, pcc) = (Operand::Plane(0), Operand::Plane(1), Operand::Plane(2));
    let cases: [(&str, Program, usize); 4] = [
        (
            "a & !b",
            p(MaskOp::AndNot {
                a: pa,
                b: pb,
                dst: 0,
            }),
            pc(|a, b, _| a & !b),
        ),
        (
            "a ^ b",
            p(MaskOp::Xor {
                a: pa,
                b: pb,
                dst: 0,
            }),
            pc(|a, b, _| a ^ b),
        ),
        (
            "a | b",
            p(MaskOp::Or {
                a: pa,
                b: pb,
                dst: 0,
            }),
            pc(|a, b, _| a | b),
        ),
        (
            "maj(a,b,c)",
            p(MaskOp::Ternlog {
                imm: 0xE8,
                a: pa,
                b: pb,
                c: pcc,
                dst: 0,
            }),
            pc(|a, b, c| (a & b) | (a & c) | (b & c)),
        ),
    ];
    for (name, prog, want) in cases {
        let (got, r) = run_arm("E", &prog, &planes, 200, 0);
        assert_eq!(got, want, "{name}");
        assert_eq!(r.lowering, "Ternlog(no slot)", "{name}");
        println!("  {name:<11} = {got:>6}  {} {:>7.0} ns", r.lowering, r.ns);
    }
    // anti-vacuity: the four answers are pairwise distinct on this fixture
    let a = pc(|a, b, _| a & !b);
    let b = pc(|a, b, _| a ^ b);
    let c = pc(|a, b, _| a | b);
    assert!(
        a != b && b != c && a != c,
        "the Boolean variants must differ"
    );
}

/// `(x, y)` for row `i` of an edge-case fixture.
type RowGen = Box<dyn Fn(usize) -> (i32, i32)>;

/// Edge cases: every arm must agree with the oracle.
fn edge_cases() {
    println!("\n== edge cases (every arm equals the oracle) ==");
    let cases: Vec<(&str, usize, RowGen)> = vec![
        ("all zero", 4096, Box::new(|_| (SCALE, SCALE))),
        ("all one", 4096, Box::new(|_| (0, 0))),
        (
            "disjoint",
            4096,
            Box::new(|i| if i % 2 == 0 { (0, SCALE) } else { (SCALE, 0) }),
        ),
        (
            "identical",
            4096,
            Box::new(|i| if i % 3 == 0 { (0, 0) } else { (SCALE, SCALE) }),
        ),
        (
            "one live bit",
            4096,
            Box::new(|i| if i == 1777 { (0, 0) } else { (0, SCALE) }),
        ),
        (
            "partial word n=100",
            100,
            Box::new(|i| if i % 5 == 0 { (0, 0) } else { (0, SCALE) }),
        ),
        (
            "n = 65536+37",
            65_573,
            Box::new(|i| {
                if i % 7 < 3 {
                    (0, i as i32 % 2)
                } else {
                    (SCALE, 0)
                }
            }),
        ),
    ];
    let (ta, tb) = (1, 1);
    for (name, n, f) in cases {
        let (x, y): (Vec<i32>, Vec<i32>) = (0..n).map(&f).unzip();
        let want = oracle(&x, &y, ta, tb);
        let lanes_ = [LaneRef::I32(&x), LaneRef::I32(&y)];
        let ma = bits_of(&x, ta);
        let mb = bits_of(&y, tb);
        let resident: [&[u64]; 2] = [&ma, &mb];
        let planes = Planes {
            n_rows: n,
            masks: &resident,
            lanes: &lanes_,
        };
        let progs = [
            Program::new(
                vec![MaskOp::And {
                    a: Operand::Plane(0),
                    b: Operand::Plane(1),
                    dst: 0,
                }],
                count_of(Operand::Scratch(0)),
            ),
            Program::new(
                vec![
                    pred_a(ta, None, 0),
                    pred_b(tb, None, 1),
                    MaskOp::And {
                        a: Operand::Scratch(0),
                        b: Operand::Scratch(1),
                        dst: 2,
                    },
                ],
                count_of(Operand::Scratch(2)),
            ),
            Program::new(
                vec![
                    pred_a(ta, None, 0),
                    pred_b(tb, Some(Operand::Scratch(0)), 1),
                ],
                count_of(Operand::Scratch(1)),
            ),
            Program::new(
                vec![pred_b(tb, Some(Operand::Plane(0)), 0)],
                count_of(Operand::Scratch(0)),
            ),
        ];
        let got: Vec<usize> = progs
            .iter()
            .map(|p| {
                let mut s = Scratch::for_program(p, n).unwrap();
                count(execute_into(p, &planes, &Foreign::NONE, &mut s, Out::None))
            })
            .collect();
        assert!(got.iter().all(|g| *g == want), "{name}: {got:?} vs {want}");
        println!("  {name:<20} n {n:>6}  count {want:>6}  arms B,C,D,Dp agree");
    }
}

/// The intermediates that must NOT be eliminated, and the refusals.
fn must_not_eliminate(n: usize) {
    println!("\n== reuse, demanded bitmap, refusal ==");
    let (x, y) = lanes(n, 0xBEEF, false);
    let (ta, tb) = (thresh(0.2), thresh(0.5));
    let lanes_ = [LaneRef::I32(&x), LaneRef::I32(&y)];
    let none: [&[u64]; 0] = [];
    let planes = Planes {
        n_rows: n,
        masks: &none,
        lanes: &lanes_,
    };
    // 1. A terminal that DEMANDS the bitmap writes it, and it is the right one.
    let keep = Program::new(
        vec![
            pred_a(ta, None, 0),
            pred_b(tb, Some(Operand::Scratch(0)), 1),
        ],
        keep_of(Operand::Scratch(1)),
    );
    let mut s = Scratch::for_program(&keep, n).unwrap();
    let mut kept = vec![0u64; words_for(n)];
    execute_into(&keep, &planes, &Foreign::NONE, &mut s, Out::Mask(&mut kept)).unwrap();
    let want: Vec<u64> = bits_of(&x, ta)
        .iter()
        .zip(&bits_of(&y, tb))
        .map(|(a, b)| a & b)
        .collect();
    assert_eq!(kept, want, "a demanded bitmap must be written in full");
    println!(
        "  Keep(A∧B) writes the demanded {} B bitmap: equal to the oracle",
        kept.len() * 8
    );
    // 2. A reused intermediate: one kept A feeds two folds; their sum is |A|.
    let ma = bits_of(&x, ta);
    let mb = bits_of(&y, tb);
    let both: [&[u64]; 2] = [&ma, &mb];
    let rp = Planes {
        n_rows: n,
        masks: &both,
        lanes: &[],
    };
    let and = Program::new(
        vec![MaskOp::And {
            a: Operand::Plane(0),
            b: Operand::Plane(1),
            dst: 0,
        }],
        count_of(Operand::Scratch(0)),
    );
    let andnot = Program::new(
        vec![MaskOp::AndNot {
            a: Operand::Plane(0),
            b: Operand::Plane(1),
            dst: 0,
        }],
        count_of(Operand::Scratch(0)),
    );
    let mut s0 = Scratch::new(0, 0);
    let c_and = count(execute_into(&and, &rp, &Foreign::NONE, &mut s0, Out::None));
    let c_andnot = count(execute_into(
        &andnot,
        &rp,
        &Foreign::NONE,
        &mut s0,
        Out::None,
    ));
    let pop_a: usize = ma.iter().map(|w| w.count_ones() as usize).sum();
    assert_eq!(c_and + c_andnot, pop_a);
    assert!(
        c_and > 0 && c_andnot > 0,
        "both consumers must see survivors"
    );
    println!("  A reused by two folds: |A∧B| {c_and} + |A∧¬B| {c_andnot} = |A| {pop_a}");
    // 3. A terminal with no merge law is refused on a partial extent.
    let blend = Program::new(
        vec![pred_a(ta, None, 0)],
        Terminal::BlendI32 {
            mask: Operand::Scratch(0),
            then: 0,
            els: 1,
        },
    );
    let mut sb = Scratch::for_program(&blend, n).unwrap();
    let mut out = vec![0i32; n];
    let r = execute_extent(
        &blend,
        &planes,
        &Foreign::NONE,
        &mut sb,
        Out::I32(&mut out),
        0..n / 2,
    );
    assert!(
        matches!(r, Err(ExecError::ExtentUnsupported { what: "BlendI32" })),
        "{r:?}"
    );
    println!("  BlendI32 over a partial extent: refused (no merge law)");
}

// ───────────────────────── CE64 over MaterializedEdges ─────────────────────────

/// Bit layout of the two fields under test (`causal-edge` v2 layout).
const PEARL_SHIFT: u32 = 40; // bits 40..42
const EPI5_SHIFT: u32 = 59; // bits 59..63

/// A 16-byte `(pattern, care)` matching `value` in the `width`-bit field at
/// `shift` of the CE64 in half `half` (0 = first eight bytes) of the window.
fn ce64_pattern(half: usize, shift: u32, width: u32, value: u64) -> ([u8; 16], [u8; 16]) {
    let care = ((1u64 << width) - 1) << shift;
    let pat = (value << shift) & care;
    let mut p = [0u8; 16];
    let mut c = [0u8; 16];
    p[half * 8..half * 8 + 8].copy_from_slice(&pat.to_le_bytes());
    c[half * 8..half * 8 + 8].copy_from_slice(&care.to_le_bytes());
    (p, c)
}

fn edge(rows: &[u8], r: usize, k: usize) -> CausalEdge64 {
    let off = r * NODE_ROW_STRIDE
        + VALUE_SLAB_ROW_OFFSET
        + ValueTenant::MaterializedEdges.value_offset()
        + 8 * k;
    CausalEdge64(u64::from_le_bytes(rows[off..off + 8].try_into().unwrap()))
}

fn ce64(n: usize, reps: usize) {
    println!("\n== CE64 in place: COUNT(edge0.pearl == p AND edge2.epi5 == q), n = {n} ==");
    let vo = VALUE_SLAB_ROW_OFFSET + ValueTenant::MaterializedEdges.value_offset();
    assert_eq!(
        ValueTenant::MaterializedEdges.byte_len(),
        32,
        "the tenant is 4 × u64"
    );
    let mut r = Rng(0xCE64);
    let mut rows = vec![0u8; n * NODE_ROW_STRIDE];
    for b in rows.iter_mut() {
        *b = r.next() as u8;
    }
    let (p, q) = (5u64, 9u64);
    // Window 0 = edges 0|1, window 1 = edges 2|3: both inside the 32-byte tenant.
    let win0 = StridedRef {
        bytes: &rows,
        first_offset: vo,
        stride: NODE_ROW_STRIDE,
        records: n,
    };
    let win1 = StridedRef {
        bytes: &rows,
        first_offset: vo + 16,
        stride: NODE_ROW_STRIDE,
        records: n,
    };
    // No window leaves the tenant: window 1 ends exactly where the tenant does.
    let tenant_end = vo + ValueTenant::MaterializedEdges.byte_len();
    assert_eq!(win1.first_offset + 16, tenant_end);
    assert_eq!(win0.first_offset, vo);
    let lanes_ = [LaneRef::Strided(win0), LaneRef::Strided(win1)];
    let none: [&[u64]; 0] = [];
    let planes = Planes {
        n_rows: n,
        masks: &none,
        lanes: &lanes_,
    };
    let (pa, ca) = ce64_pattern(0, PEARL_SHIFT, 3, p); // edge 0, half 0 of window 0
    let (pb, cb) = ce64_pattern(0, EPI5_SHIFT, 5, q); // edge 2, half 0 of window 1
    let ma = |under, dst| MaskOp::Pred {
        pred: Pred::MatchFacet16Strided {
            lane: 0,
            pattern: pa,
            care: ca,
        },
        under,
        dst,
    };
    let mb = |under, dst| MaskOp::Pred {
        pred: Pred::MatchFacet16Strided {
            lane: 1,
            pattern: pb,
            care: cb,
        },
        under,
        dst,
    };
    let want = (0..n)
        .filter(|&i| {
            edge(&rows, i, 0).causal_mask() as u8 as u64 == p
                && u64::from(edge(&rows, i, 2).epistemic_raw5()) == q
        })
        .count();
    let only_a = (0..n)
        .filter(|&i| edge(&rows, i, 0).causal_mask() as u8 as u64 == p)
        .count();
    assert!(want > 0 && want * 3 < n, "anti-vacuity: {want} of {n}");
    println!("  accessor oracle: {want} (A alone {only_a})");
    // One strided predicate on its own: what each of C's two passes costs.
    let p_one = Program::new(vec![ma(None, 0)], count_of(Operand::Scratch(0)));
    let (c1, r1) = run_arm("A1", &p_one, &planes, reps, 0);
    assert_eq!(c1, only_a);
    print(&r1);
    let pc = Program::new(
        vec![
            ma(None, 0),
            mb(None, 1),
            MaskOp::And {
                a: Operand::Scratch(0),
                b: Operand::Scratch(1),
                dst: 2,
            },
        ],
        count_of(Operand::Scratch(2)),
    );
    let (cc, rc) = run_arm("C", &pc, &planes, reps, 0);
    assert_eq!(cc, want, "C vs canonical accessors");
    print(&rc);
    let pd = Program::new(
        vec![ma(None, 0), mb(Some(Operand::Scratch(0)), 1)],
        count_of(Operand::Scratch(1)),
    );
    let (cd, rd) = run_arm("D", &pd, &planes, reps, 0);
    assert_eq!(cd, want, "D vs canonical accessors");
    print(&rd);
    // The skip a strided `_under` kernel would give: B's bytes are loaded only
    // on A's survivors. Scalar, probe-local, a REFERENCE for what the missing
    // primitive could save — not a backend arm and not a proposal for one.
    let gated_scalar = |rows: &[u8]| -> usize {
        let mut c = 0;
        for i in 0..n {
            let base = i * NODE_ROW_STRIDE + vo;
            let e0 = u64::from_le_bytes(rows[base..base + 8].try_into().unwrap());
            if (e0 >> PEARL_SHIFT) & 0b111 == p {
                let e2 = u64::from_le_bytes(rows[base + 16..base + 24].try_into().unwrap());
                if (e2 >> EPI5_SHIFT) & 0b11111 == q {
                    c += 1;
                }
            }
        }
        c
    };
    let both_scalar = |rows: &[u8]| -> usize {
        let mut c = 0;
        for i in 0..n {
            let base = i * NODE_ROW_STRIDE + vo;
            let e0 = u64::from_le_bytes(rows[base..base + 8].try_into().unwrap());
            let e2 = u64::from_le_bytes(rows[base + 16..base + 24].try_into().unwrap());
            c += usize::from((e0 >> PEARL_SHIFT) & 0b111 == p && (e2 >> EPI5_SHIFT) & 0b11111 == q);
        }
        c
    };
    for (name, f) in [
        ("H-gated", &gated_scalar as &dyn Fn(&[u8]) -> usize),
        ("H-both", &both_scalar),
    ] {
        let t = Instant::now();
        let mut c = 0;
        for _ in 0..reps {
            c = f(std::hint::black_box(&rows));
        }
        let ns = t.elapsed().as_nanos() as f64 / reps as f64;
        assert_eq!(c, want);
        println!(
            "  {name:<8} scalar reference (A's row read first; B read {}) {ns:>12.0} ns",
            if name == "H-gated" {
                "only on A's survivors"
            } else {
                "on every row"
            }
        );
    }
    let line = |b: usize| b / 64;
    println!(
        "  bytes: resident {} B; tenant at row bytes {vo}..{}; named per row 32 B (two 16 B windows); \
         64 B lines per row: window 0 {:?}, window 1 {:?}",
        rows.len(),
        vo + 32,
        line(vo)..=line(vo + 15),
        line(vo + 16)..=line(vo + 31)
    );

    // Pattern merge: two field predicates inside ONE 16-byte window are one
    // ternary match — `match(p1, c1) ∧ match(p2, c2) = match(p1 | p2, c1 | c2)`
    // when the patterns agree on `c1 & c2` (here the cares are disjoint).
    let (pe1, ce1) = ce64_pattern(1, EPI5_SHIFT, 5, q); // edge 1, half 1 of window 0
    let merged_p: [u8; 16] = core::array::from_fn(|k| pa[k] | pe1[k]);
    let merged_c: [u8; 16] = core::array::from_fn(|k| ca[k] | ce1[k]);
    assert!((0..16).all(|k| ca[k] & ce1[k] == 0), "disjoint cares");
    let want_w = (0..n)
        .filter(|&i| {
            edge(&rows, i, 0).causal_mask() as u8 as u64 == p
                && u64::from(edge(&rows, i, 1).epistemic_raw5()) == q
        })
        .count();
    assert!(
        want_w > 0 && want_w * 3 < n,
        "anti-vacuity: {want_w} of {n}"
    );
    let two = Program::new(
        vec![
            ma(None, 0),
            MaskOp::Pred {
                pred: Pred::MatchFacet16Strided {
                    lane: 0,
                    pattern: pe1,
                    care: ce1,
                },
                under: None,
                dst: 1,
            },
            MaskOp::And {
                a: Operand::Scratch(0),
                b: Operand::Scratch(1),
                dst: 2,
            },
        ],
        count_of(Operand::Scratch(2)),
    );
    let one = Program::new(
        vec![MaskOp::Pred {
            pred: Pred::MatchFacet16Strided {
                lane: 0,
                pattern: merged_p,
                care: merged_c,
            },
            under: None,
            dst: 0,
        }],
        count_of(Operand::Scratch(0)),
    );
    let (c2, r2) = run_arm("C2", &two, &planes, reps, 0);
    let (c1m, r1m) = run_arm("M1", &one, &planes, reps, 0);
    assert_eq!(c2, want_w);
    assert_eq!(c1m, want_w);
    println!("  same window, edge0.pearl AND edge1.epi5 (oracle {want_w}):");
    print(&r2);
    print(&r1m);

    // The precondition is load-bearing: two patterns that DISAGREE on a shared
    // care bit have an empty conjunction, and an unconditional `p1 | p2` merge
    // would answer something else.
    let (px, cx) = ce64_pattern(0, PEARL_SHIFT, 3, p ^ 1); // same field, other value
    let conflict = Program::new(
        vec![
            ma(None, 0),
            MaskOp::Pred {
                pred: Pred::MatchFacet16Strided {
                    lane: 0,
                    pattern: px,
                    care: cx,
                },
                under: None,
                dst: 1,
            },
            MaskOp::And {
                a: Operand::Scratch(0),
                b: Operand::Scratch(1),
                dst: 2,
            },
        ],
        count_of(Operand::Scratch(2)),
    );
    let naive_p: [u8; 16] = core::array::from_fn(|k| pa[k] | px[k]);
    let naive_c: [u8; 16] = core::array::from_fn(|k| ca[k] | cx[k]);
    let naive = Program::new(
        vec![MaskOp::Pred {
            pred: Pred::MatchFacet16Strided {
                lane: 0,
                pattern: naive_p,
                care: naive_c,
            },
            under: None,
            dst: 0,
        }],
        count_of(Operand::Scratch(0)),
    );
    let (cc0, _) = run_arm("Cx", &conflict, &planes, 1, 0);
    let (cn, _) = run_arm("Mx", &naive, &planes, 1, 0);
    assert_eq!(cc0, 0, "conflicting patterns have no common row");
    assert_ne!(
        cn, 0,
        "an unconditional OR-merge answers a different question"
    );
    println!("  conflicting cares: two predicates {cc0}, unconditional OR-merge {cn} (refuse or constant-false, never merge)");

    // can fire: the wrong bit range must disagree
    let (pw, cw) = ce64_pattern(0, PEARL_SHIFT + 1, 3, p);
    let pwrong = Program::new(
        vec![
            MaskOp::Pred {
                pred: Pred::MatchFacet16Strided {
                    lane: 0,
                    pattern: pw,
                    care: cw,
                },
                under: None,
                dst: 0,
            },
            mb(Some(Operand::Scratch(0)), 1),
        ],
        count_of(Operand::Scratch(1)),
    );
    let (cw_, _) = run_arm("Cw", &pwrong, &planes, 1, 0);
    assert_ne!(cw_, want, "a care on the wrong bits must disagree");
    // stay silent: vary everything OUTSIDE care — edge1, edge3, the other
    // fields of edges 0 and 2, and every byte outside the tenant.
    let mut rows2 = rows.clone();
    for i in 0..n {
        let base = i * NODE_ROW_STRIDE;
        for (j, b) in rows2[base..base + NODE_ROW_STRIDE].iter_mut().enumerate() {
            let in_tenant = j >= vo && j < vo + 32;
            let k = (j.wrapping_sub(vo)) / 8;
            let bit_safe = if in_tenant && (k == 0 || k == 2) {
                let byte = (j - vo) % 8;
                let field_mask: u64 = if k == 0 {
                    0b111 << PEARL_SHIFT
                } else {
                    0b11111 << EPI5_SHIFT
                };
                ((field_mask >> (8 * byte)) & 0xFF) as u8
            } else {
                0
            };
            *b = (*b & bit_safe) | (!bit_safe & (r.next() as u8));
        }
    }
    let win0b = StridedRef {
        bytes: &rows2,
        ..win0
    };
    let win1b = StridedRef {
        bytes: &rows2,
        ..win1
    };
    let lanes2 = [LaneRef::Strided(win0b), LaneRef::Strided(win1b)];
    let planes2 = Planes {
        n_rows: n,
        masks: &none,
        lanes: &lanes2,
    };
    let (cd2, _) = run_arm("D", &pd, &planes2, 1, 0);
    assert_eq!(cd2, want, "bits outside care must not change the answer");
    println!("  can-fire (care shifted by one bit): {cw_} ≠ {want}; silent (all non-care bits rewritten): {cd2} = {want}");
}

fn main() {
    let n = 65_536;
    edge_cases();
    must_not_eliminate(n);
    boolean_variants(n);
    ce64(n, 200);
    sweep(n, 400);
    println!("\nall counts agree with the oracle");
}
