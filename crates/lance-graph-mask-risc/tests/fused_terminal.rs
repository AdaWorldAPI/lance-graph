//! The terminal decides whether membership becomes bits.
//!
//! `Range[lo, hi) ∩ resident plane` has two legal physical endings over ONE
//! membership relation:
//!
//! - FOLD — `Count` / `Any`: the executor reads the resident plane's touched
//!   words and two register-masked edge words. No scratch slot is carved, no
//!   derived membership word is written, no row id exists.
//! - MATERIALIZE — `Keep`: the relation is written to the demanded
//!   `Out::Mask`, because keeping the set is what `Keep` asks for.
//!
//! Every case below checks both arms against each other and against a scalar
//! oracle, and the gates check that the fold arm materializes NOTHING — not
//! "less than O(N)", nothing: a tile-sized scratch mask written only to be
//! counted is still a materialization.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use lance_graph_mask_risc::exec::{execute_into, Scratch};
use lance_graph_mask_risc::{
    touched_words, Foreign, FusedFold, MaskOp, Operand, Out, Planes, Pred, Program, Terminal, Value,
};

struct Counting;

thread_local! {
    // Per THREAD, not per process: the test harness runs tests on parallel
    // threads, and a process-wide counter picks up their allocations inside
    // this test's window (measured: 120 stray bytes on an otherwise clean run).
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

fn lcg(seed: &mut u64) -> u64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *seed >> 11
}

fn words_for(n: usize) -> usize {
    n.div_ceil(64)
}

/// A resident plane of `n` rows with `set(row)` bits, tail bits clear.
fn plane(n: usize, set: impl Fn(usize) -> bool) -> Vec<u64> {
    let mut w = vec![0u64; words_for(n)];
    for r in 0..n {
        if set(r) {
            w[r / 64] |= 1u64 << (r % 64);
        }
    }
    w
}

fn range_program(lo: u32, hi: u32, gate: Option<Operand>, terminal: Terminal) -> Program {
    Program::new(
        vec![MaskOp::Pred {
            pred: Pred::Range { lo, hi },
            under: gate,
            dst: 0,
        }],
        terminal,
    )
}

fn oracle(p: &[u64], lo: usize, hi: usize) -> usize {
    (lo..hi).filter(|&r| p[r / 64] >> (r % 64) & 1 == 1).count()
}

/// Run the FOLD arm with a scratch of zero slots — a scratch that cannot
/// hold a single derived word, so a fold that needed one could not run.
fn fold(p: &Program, planes: &Planes<'_>) -> Value {
    let mut s = Scratch::new(0, 0);
    execute_into(p, planes, &Foreign::NONE, &mut s, Out::None).expect("fold arm")
}

/// Run the MATERIALIZE arm and return the kept mask.
fn keep(lo: u32, hi: u32, planes: &Planes<'_>) -> Vec<u64> {
    let p = range_program(
        lo,
        hi,
        Some(Operand::Plane(0)),
        Terminal::Keep {
            mask: Operand::Scratch(0),
        },
    );
    assert!(p.fused_terminal().is_none(), "Keep is never fused");
    let mut s = Scratch::for_program(&p, planes.n_rows).expect("scratch");
    let mut out = vec![0u64; words_for(planes.n_rows)];
    execute_into(&p, planes, &Foreign::NONE, &mut s, Out::Mask(&mut out)).expect("keep arm");
    out
}

/// The flagship cases the capstone names, clipped to `n`.
fn ranges(n: u32) -> Vec<(u32, u32)> {
    let mut v = vec![
        (0, 0),
        (65.min(n), 65.min(n)),
        (0, 1),
        (n - 1, n),
        (0, 64.min(n)),             // aligned lo, aligned hi
        (3, 60.min(n)),             // inside one word
        (60.min(n - 1), n.min(70)), // across one word boundary
        (64.min(n), n),             // aligned lo, unaligned-or-whole hi
        (0, n),                     // whole population
    ];
    if n > 700 {
        v.push((1, 700)); // crosses more than one 512-row tile
        v.push((130, 1100));
    }
    v.retain(|&(lo, hi)| lo <= hi && hi <= n);
    v
}

/// FAILS IF: the fold arm and the materialize arm disagree with each other or
/// with the scalar oracle for any range × plane shape — including an empty
/// range, a single row, word-straddling edges, a sub-64-row tail and the
/// whole population.
#[test]
fn fold_and_keep_arms_agree_with_the_oracle() {
    let mut seed = 0x5eed_u64;
    let mut cases = 0;
    for n in [37usize, 64, 1024, 1317] {
        let mut shapes: Vec<(&str, Vec<u64>)> = vec![
            ("zero", plane(n, |_| false)),
            ("one", plane(n, |_| true)),
            ("clustered", plane(n, |r| (r / 97) % 3 == 1)),
        ];
        let scattered: Vec<bool> = (0..n).map(|_| lcg(&mut seed).is_multiple_of(5)).collect();
        shapes.push(("scattered", plane(n, |r| scattered[r])));
        for (name, p) in &shapes {
            let masks: [&[u64]; 1] = [p];
            let planes = Planes {
                n_rows: n,
                masks: &masks,
                lanes: &[],
            };
            for (lo, hi) in ranges(n as u32) {
                let want = oracle(p, lo as usize, hi as usize);
                let gate = Some(Operand::Plane(0));
                let count = range_program(
                    lo,
                    hi,
                    gate,
                    Terminal::Count {
                        mask: Operand::Scratch(0),
                    },
                );
                let any = range_program(
                    lo,
                    hi,
                    gate,
                    Terminal::Any {
                        mask: Operand::Scratch(0),
                    },
                );
                assert_eq!(
                    count.fused_terminal().map(|f| f.fold),
                    Some(FusedFold::Count)
                );
                assert_eq!(any.fused_terminal().map(|f| f.fold), Some(FusedFold::Any));
                let kept = keep(lo, hi, &planes);
                let kept_count: usize = kept.iter().map(|w| w.count_ones() as usize).sum();
                let ctx = format!("{name} n={n} [{lo},{hi})");
                assert_eq!(fold(&count, &planes), Value::Count(want), "count {ctx}");
                assert_eq!(kept_count, want, "keep→popcount {ctx}");
                assert_eq!(fold(&any, &planes), Value::Bool(want > 0), "any {ctx}");
                assert_eq!(kept.iter().any(|&w| w != 0), want > 0, "keep→any {ctx}");
                cases += 1;
            }
        }
    }
    // Anti-vacuity: the grid really covers the named shapes.
    assert!(cases >= 100, "only {cases} cases ran");
}

/// FAILS IF: a bare range (no gate) is not folded to `hi - lo` / `lo < hi`
/// without a scratch — the W2a case, where not even the plane is read.
#[test]
fn a_bare_range_folds_to_its_own_width() {
    let n = 1317usize;
    let p = plane(n, |_| false);
    let masks: [&[u64]; 1] = [&p];
    let planes = Planes {
        n_rows: n,
        masks: &masks,
        lanes: &[],
    };
    for (lo, hi) in ranges(n as u32) {
        let count = range_program(
            lo,
            hi,
            None,
            Terminal::Count {
                mask: Operand::Scratch(0),
            },
        );
        let any = range_program(
            lo,
            hi,
            None,
            Terminal::Any {
                mask: Operand::Scratch(0),
            },
        );
        assert!(!count.requires_scratch());
        assert_eq!(fold(&count, &planes), Value::Count((hi - lo) as usize));
        assert_eq!(fold(&any, &planes), Value::Bool(lo < hi));
    }
}

/// THE LOAD-BEARING GATE. FAILS IF: the fold arm carves a scratch slot or
/// writes a single word into the caller's arena.
///
/// The caller's buffer is poisoned to `u64::MAX` and handed to
/// `Scratch::over_for_program`, which zero-fills whatever it carves. After
/// construction AND execution the buffer must still be all `u64::MAX`: any
/// carved slot, and any derived membership word, would show up as a changed
/// word. The twin half runs the `Keep` program through the same buffer and
/// proves the probe can see a carve.
#[test]
fn the_fold_arm_writes_no_derived_membership_word() {
    let n = 4096usize;
    let p = plane(n, |r| r % 3 == 0);
    let masks: [&[u64]; 1] = [&p];
    let planes = Planes {
        n_rows: n,
        masks: &masks,
        lanes: &[],
    };
    let (lo, hi) = (70u32, 3000u32);
    for terminal in [
        Terminal::Count {
            mask: Operand::Scratch(0),
        },
        Terminal::Any {
            mask: Operand::Scratch(0),
        },
    ] {
        let prog = range_program(lo, hi, Some(Operand::Plane(0)), terminal);
        assert!(!prog.requires_scratch(), "a fused fold needs no scratch");
        let mut buf = vec![u64::MAX; 64];
        {
            let mut s = Scratch::over_for_program(&mut buf, &prog, n).expect("carve");
            assert_eq!(s.slots(), 0, "no slot is carved for a fold");
            execute_into(&prog, &planes, &Foreign::NONE, &mut s, Out::None).expect("fold");
        }
        assert!(
            buf.iter().all(|&w| w == u64::MAX),
            "the fold arm touched the caller's arena"
        );
    }

    // Can-it-fire: the same probe sees the materialize arm's carve.
    let keep_prog = range_program(
        lo,
        hi,
        Some(Operand::Plane(0)),
        Terminal::Keep {
            mask: Operand::Scratch(0),
        },
    );
    assert!(keep_prog.requires_scratch());
    let mut buf = vec![u64::MAX; 64];
    {
        let mut s = Scratch::over_for_program(&mut buf, &keep_prog, n).expect("carve");
        let mut out = vec![0u64; words_for(n)];
        execute_into(
            &keep_prog,
            &planes,
            &Foreign::NONE,
            &mut s,
            Out::Mask(&mut out),
        )
        .expect("keep");
        assert!(out.iter().any(|&w| w != 0), "Keep materialized the set");
    }
    assert!(
        buf.iter().any(|&w| w != u64::MAX),
        "the probe cannot see a carve, so its silence above proves nothing"
    );
}

/// FAILS IF: `Scratch::for_program` allocates for a fold, or the counter is
/// inert (the `Keep` half must allocate).
#[test]
fn sizing_a_fold_allocates_nothing() {
    let count = range_program(
        5,
        900,
        Some(Operand::Plane(0)),
        Terminal::Count {
            mask: Operand::Scratch(0),
        },
    );
    let keep_prog = range_program(
        5,
        900,
        Some(Operand::Plane(0)),
        Terminal::Keep {
            mask: Operand::Scratch(0),
        },
    );
    let before = bytes();
    let s = Scratch::for_program(&count, 1 << 20).expect("scratch");
    let fold_bytes = bytes() - before;
    drop(s);
    let before = bytes();
    let s = Scratch::for_program(&keep_prog, 1 << 20).expect("scratch");
    let keep_bytes = bytes() - before;
    drop(s);
    assert_eq!(fold_bytes, 0, "a fold carved a scratch arena");
    assert!(keep_bytes > 0, "the counter cannot see an allocation");
}

/// FAILS IF: a shape that holds derived membership elsewhere is wrongly
/// fused — a scratch gate, a second op, or a terminal reading another slot.
#[test]
fn only_the_exact_shape_is_fused() {
    let count = Terminal::Count {
        mask: Operand::Scratch(0),
    };
    assert!(range_program(0, 9, Some(Operand::Scratch(1)), count)
        .fused_terminal()
        .is_none());
    let two_ops = Program::new(
        vec![
            MaskOp::Pred {
                pred: Pred::Range { lo: 0, hi: 9 },
                under: None,
                dst: 0,
            },
            MaskOp::And {
                a: Operand::Scratch(0),
                b: Operand::Plane(0),
                dst: 1,
            },
        ],
        Terminal::Count {
            mask: Operand::Scratch(1),
        },
    );
    assert!(two_ops.fused_terminal().is_none());
    assert!(two_ops.requires_scratch());
    assert!(range_program(
        0,
        9,
        None,
        Terminal::All {
            mask: Operand::Scratch(0)
        }
    )
    .fused_terminal()
    .is_none());
}

/// FAILS IF: the touched-word span departs from the law
/// `lo == hi → 0` else `floor((hi-1)/64) - floor(lo/64) + 1`.
#[test]
fn touched_words_obey_the_law() {
    let mut checked = 0;
    for lo in [0u32, 1, 63, 64, 65, 127, 128, 500, 511, 512, 513] {
        for width in [0u32, 1, 2, 63, 64, 65, 128, 600] {
            let hi = lo + width;
            let want = if lo == hi {
                0
            } else {
                ((hi - 1) / 64 - lo / 64 + 1) as usize
            };
            assert_eq!(touched_words(lo, hi).len(), want, "[{lo},{hi})");
            checked += 1;
        }
    }
    assert_eq!(touched_words(65, 65), 0..0);
    assert_eq!(touched_words(0, 0), 0..0);
    assert!(checked > 80);
}

/// FAILS IF: validation is skipped on the fused path — an out-of-bounds range
/// or an absent plane must still be refused, not read.
#[test]
fn the_fused_path_still_validates() {
    let n = 100usize;
    let p = plane(n, |_| true);
    let masks: [&[u64]; 1] = [&p];
    let planes = Planes {
        n_rows: n,
        masks: &masks,
        lanes: &[],
    };
    let mut s = Scratch::new(0, 0);
    let too_far = range_program(
        0,
        101,
        Some(Operand::Plane(0)),
        Terminal::Count {
            mask: Operand::Scratch(0),
        },
    );
    assert!(execute_into(&too_far, &planes, &Foreign::NONE, &mut s, Out::None).is_err());
    let no_plane = range_program(
        0,
        10,
        Some(Operand::Plane(3)),
        Terminal::Count {
            mask: Operand::Scratch(0),
        },
    );
    assert!(execute_into(&no_plane, &planes, &Foreign::NONE, &mut s, Out::None).is_err());
}
