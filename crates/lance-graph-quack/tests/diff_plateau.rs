//! **Round 5 — difference and plateau as an expression, not a population.**
//!
//! Two resident states `a` (state_t) and `b` (state_t+1) over the same `n`
//! rows. The difference is `a XOR b`; its size is `POPCOUNT(a XOR b)`; the
//! plateau is "nothing changed", `NOT ANY(a XOR b)`.
//!
//! Four ways to compute it, every one checked against a bit-at-a-time row
//! oracle:
//!
//! | path | what it does |
//! |---|---|
//! | A — materialized reference | builds `a ^ b` as a `Vec<u64>` population, then popcounts it |
//! | B — mask-risc program | `Xor(plane0, plane1)` folded by `Count` / `Any` |
//! | C — quack `lower` | `(a AND NOT b) OR (NOT a AND b)` |
//! | D — quack `lower_fused` | the same filter, fused lowering |
//!
//! Measured per path: mask-risc's lowering class, the scratch slot words the
//! program needs at all (`Program::requires_scratch`), the non-zero slot words
//! left after a dense-diff execution (a derived word actually written), and
//! the heap bytes the execution itself allocates.
//!
//! The question is whether the fused path writes ZERO population-sized derived
//! words. Plateau is claimed only for what is measured here: two resident
//! bit-planes over one row space.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use lance_graph_mask_risc::{
    execute_into, words_for, Foreign, Lowering, MaskOp, Operand, Out, Planes, Program, Scratch,
    Terminal, Value, TILE_WORDS,
};
use lance_graph_quack::{lower, lower_fused, Agg, Filter, Mask, Query};

// ── thread-local heap meter (the `fused_ternlog.rs` pattern) ─────────────

struct Counting;

thread_local! {
    static BYTES: Cell<usize> = const { Cell::new(0) };
}

fn bytes() -> usize {
    BYTES.with(Cell::get)
}

// SAFETY: a pure pass-through to `System`; the counter is the only addition.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = BYTES.try_with(|b| b.set(b.get() + layout.size()));
        // SAFETY: same layout, same contract as the caller's.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from `alloc` above with this `layout`.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static A: Counting = Counting;

// ── fixtures ─────────────────────────────────────────────────────────────

const TILE_ROWS: usize = TILE_WORDS * 64;

fn lcg(seed: &mut u64) -> u64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *seed >> 11
}

fn set(words: &mut [u64], i: usize) {
    words[i / 64] ^= 1 << (i % 64);
}

fn bit(words: &[u64], i: usize) -> bool {
    words[i / 64] >> (i % 64) & 1 == 1
}

/// A random plane over `n` rows, tail bits zero.
fn random_plane(n: usize, seed: u64) -> Vec<u64> {
    let mut s = seed;
    let mut w = vec![0u64; words_for(n)];
    for i in 0..n {
        if lcg(&mut s) & 1 == 1 {
            set(&mut w, i);
        }
    }
    w
}

/// `b` = `a` with the rows in `flip` toggled.
fn with_flips(a: &[u64], flip: &[usize]) -> Vec<u64> {
    let mut b = a.to_vec();
    for &i in flip {
        set(&mut b, i);
    }
    b
}

/// The cases the round requires: empty, one bit, word boundary, tile
/// boundary, ragged tail, sparse scattered, dense.
fn cases() -> Vec<(&'static str, usize, Vec<usize>)> {
    let ragged = 3 * TILE_ROWS + 77;
    vec![
        ("empty, 1 row", 1, vec![]),
        ("empty, ragged", ragged, vec![]),
        ("one bit, row 0", 65, vec![0]),
        ("one bit, last row of word 0", 65, vec![63]),
        ("one bit, first row of word 1", 65, vec![64]),
        ("one bit, ragged tail", 65, vec![64]),
        (
            "one bit, last row of tile 0",
            TILE_ROWS + 1,
            vec![TILE_ROWS - 1],
        ),
        (
            "one bit, first row of tile 1",
            TILE_ROWS + 1,
            vec![TILE_ROWS],
        ),
        ("one bit, last row, ragged", ragged, vec![ragged - 1]),
        (
            "sparse scattered",
            ragged,
            (0..ragged).step_by(997).collect(),
        ),
        (
            "dense",
            ragged,
            (0..ragged).filter(|i| i % 3 != 0).collect(),
        ),
    ]
}

/// The bit-at-a-time oracle: compares rows, never words.
fn oracle(a: &[u64], b: &[u64], n: usize) -> usize {
    (0..n).filter(|&i| bit(a, i) != bit(b, i)).count()
}

/// Path A: the diff as a materialized population, then counted.
fn materialized(a: &[u64], b: &[u64]) -> (usize, usize) {
    let diff: Vec<u64> = a.iter().zip(b).map(|(x, y)| x ^ y).collect();
    let count = diff.iter().map(|w| w.count_ones() as usize).sum();
    (count, diff.len() * 8)
}

fn program_b(terminal: fn(Operand) -> Terminal) -> Program {
    Program::new(
        vec![MaskOp::Xor {
            a: Operand::Plane(0),
            b: Operand::Plane(1),
            dst: 0,
        }],
        terminal(Operand::Scratch(0)),
    )
}

fn diff_filter() -> Filter {
    Filter::or([
        Filter::and([
            Filter::plane(Mask(0)),
            Filter::negate(Filter::plane(Mask(1))),
        ]),
        Filter::and([
            Filter::negate(Filter::plane(Mask(0))),
            Filter::plane(Mask(1)),
        ]),
    ])
}

/// What one execution measured.
#[derive(Debug)]
struct Run {
    value: Value,
    lowering: Lowering,
    /// Scratch slots the program asks for at all.
    scratch_slots: usize,
    /// Slot-arena words left non-zero after execution.
    written_words: usize,
    /// Heap bytes allocated by the execution itself.
    heap: usize,
}

fn run(program: &Program, a: &[u64], b: &[u64], n: usize) -> Run {
    let masks: [&[u64]; 2] = [a, b];
    let planes = Planes {
        n_rows: n,
        masks: &masks,
        lanes: &[],
    };
    let slots = if program.requires_scratch() {
        program.scratch_slots as usize
    } else {
        0
    };
    let tile = lance_graph_mask_risc::tile_words_for(n);
    let mut buf = vec![0u64; lance_graph_mask_risc::scratch_words_for(tile, slots).expect("fits")];
    let (value, heap) = {
        let mut scratch =
            Scratch::over_for_program(&mut buf, program, n).expect("borrowed scratch");
        let before = bytes();
        let value =
            execute_into(program, &planes, &Foreign::NONE, &mut scratch, Out::None).expect("runs");
        (value, bytes() - before)
    };
    Run {
        value,
        lowering: program.lowering(),
        scratch_slots: slots,
        written_words: buf[..slots * tile].iter().filter(|&&w| w != 0).count(),
        heap,
    }
}

fn count_of(v: &Value) -> usize {
    match v {
        Value::Count(c) => *c,
        other => panic!("not a count: {other:?}"),
    }
}

fn any_of(v: &Value) -> bool {
    match v {
        Value::Bool(b) => *b,
        other => panic!("not a bool: {other:?}"),
    }
}

// ── tests ────────────────────────────────────────────────────────────────

/// FAILS IF: any path disagrees with the row oracle on the diff COUNT, or on
/// the PLATEAU, at any case — including the one-bit cases at a word
/// boundary, a tile boundary and the ragged tail, where a word-at-a-time
/// implementation most easily drops or invents a row.
#[test]
fn every_path_agrees_with_the_row_oracle() {
    let b_count = program_b(|mask| Terminal::Count { mask });
    let b_any = program_b(|mask| Terminal::Any { mask });
    let q = |agg| Query {
        filter: diff_filter(),
        agg,
    };
    let c_count = lower(&q(Agg::Count)).expect("lowers");
    let c_any = lower(&q(Agg::Any)).expect("lowers");
    let d_count = lower_fused(&q(Agg::Count)).expect("lowers");
    let d_any = lower_fused(&q(Agg::Any)).expect("lowers");

    let mut saw_plateau = false;
    let mut saw_change = false;
    for (i, (name, n, flips)) in cases().into_iter().enumerate() {
        let a = random_plane(n, 17 + i as u64);
        let b = with_flips(&a, &flips);
        let expect = oracle(&a, &b, n);
        assert_eq!(
            expect,
            flips.len(),
            "{name}: fixture flips are distinct rows"
        );
        let plateau = expect == 0;
        saw_plateau |= plateau;
        saw_change |= !plateau;

        assert_eq!(materialized(&a, &b).0, expect, "{name}: A");
        for (path, count, any) in [
            ("B", &b_count, &b_any),
            ("C", &c_count, &c_any),
            ("D", &d_count, &d_any),
        ] {
            assert_eq!(
                count_of(&run(count, &a, &b, n).value),
                expect,
                "{name}: {path} count"
            );
            assert_eq!(
                !any_of(&run(any, &a, &b, n).value),
                plateau,
                "{name}: {path} plateau"
            );
        }
    }
    assert!(
        saw_plateau && saw_change,
        "anti-vacuity: both outcomes occur"
    );
}

/// FAILS IF: the physical shape drifts. Pinned from measurement on the dense
/// ragged case (`3 × 16384 + 77` rows, 769 words):
///
/// | path | ops | lowering | scratch slots | slot words written | heap | population bytes |
/// |---|---|---|---|---|---|---|
/// | A | — | — | — | — | — | 6152 (the diff `Vec<u64>`) |
/// | B count / any | 1 | `Ternlog` imm `0x3C` | 0 | 0 | 0 | 0 |
/// | C count / any | 5 | `Ternlog` imm `0x3C` | 0 | 0 | 0 | 0 |
/// | D count / any | 1 | `Ternlog` imm `0x3C` | 0 | 0 | 0 | 0 |
///
/// `0x3C` is `a XOR b` in the VPTERNLOG convention. Quack's 5-op spelling
/// `(a AND NOT b) OR (NOT a AND b)` collapses to the same single table as the
/// 1-op `Xor`: the plateau and the diff count are each one ternlog fold
/// straight from the two resident planes.
#[test]
fn the_fused_path_writes_no_derived_words() {
    let n = 3 * TILE_ROWS + 77;
    let a = random_plane(n, 5);
    let b = with_flips(&a, &(0..n).filter(|i| i % 3 != 0).collect::<Vec<_>>());
    let words = words_for(n);

    let (_, a_bytes) = materialized(&a, &b);
    assert_eq!(
        a_bytes,
        words * 8,
        "A materializes one word per population word"
    );

    let q = |agg| Query {
        filter: diff_filter(),
        agg,
    };
    let paths = [
        ("B count", program_b(|mask| Terminal::Count { mask }), 1),
        ("B any", program_b(|mask| Terminal::Any { mask }), 1),
        ("C count", lower(&q(Agg::Count)).expect("lowers"), 5),
        ("C any", lower(&q(Agg::Any)).expect("lowers"), 5),
        ("D count", lower_fused(&q(Agg::Count)).expect("lowers"), 1),
        ("D any", lower_fused(&q(Agg::Any)).expect("lowers"), 1),
    ];
    for (name, p, ops) in &paths {
        let r = run(p, &a, &b, n);
        println!(
            "R5: {name:8} ops={} lowering={:?} scratch_slots={} written_words={} heap={}",
            p.ops.len(),
            r.lowering,
            r.scratch_slots,
            r.written_words,
            r.heap
        );
        assert_eq!(p.ops.len(), *ops, "{name}: op count");
        assert!(
            matches!(r.lowering, Lowering::Ternlog(f) if f.imm == 0x3C),
            "{name}: {:?}",
            r.lowering
        );
        assert_eq!(r.scratch_slots, 0, "{name}: needs no scratch at all");
        assert_eq!(r.written_words, 0, "{name}: wrote a derived word");
        assert_eq!(r.heap, 0, "{name}: allocated during execution");
    }
}

/// FAILS IF: the write meter cannot see a write, which would make every zero
/// above an instrument that measures nothing.
///
/// The twin is the same diff count with a `Range` over every row stacked on
/// the `Xor` slot. The fuser declines any program containing a `Pred`, so
/// this runs tiled: the `Xor` result is written into a scratch slot, and the
/// meter must see it. The count must still match the oracle.
#[test]
fn the_write_meter_sees_a_tiled_write() {
    let n = 3 * TILE_ROWS + 77;
    let a = random_plane(n, 5);
    let b = with_flips(&a, &(0..n).filter(|i| i % 3 != 0).collect::<Vec<_>>());
    let p = Program::new(
        vec![
            MaskOp::Xor {
                a: Operand::Plane(0),
                b: Operand::Plane(1),
                dst: 0,
            },
            MaskOp::Pred {
                pred: lance_graph_mask_risc::Pred::Range {
                    lo: 0,
                    hi: n as u32,
                },
                under: Some(Operand::Scratch(0)),
                dst: 1,
            },
        ],
        Terminal::Count {
            mask: Operand::Scratch(1),
        },
    );
    let r = run(&p, &a, &b, n);
    println!(
        "R5: tiled twin lowering={:?} scratch_slots={} written_words={}",
        r.lowering, r.scratch_slots, r.written_words
    );
    assert_eq!(r.lowering, Lowering::Tiled);
    assert!(r.scratch_slots > 0);
    assert!(
        r.written_words > 0,
        "the meter saw no write on the tiled path"
    );
    assert_eq!(count_of(&r.value), oracle(&a, &b, n));
}
