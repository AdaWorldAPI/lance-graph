//! D-HOLO H1: one coordinate-aligned Boolean constraint answered for K thought
//! branches at once.
//!
//! Pre-registered in `.claude/plans/d-holo-address-indexed-algebra-v1.md` (H1).
//! Query: for each branch k in T, count(P ∧ Q ∧ M_k), with P an address
//! aperture, Q = v < t and M_k branch k's coverage.
//!
//! - A: mask-risc, one program per branch (P ∧ Q recomputed each time).
//! - B: mask-risc, G = P ∧ Q kept once into a caller mask, then one
//!   `And` + `Count` program per branch.
//! - C: outside the contract: one pass over the rows of G adding the
//!   per-row written-by word into 64 position counters.
//! - D: snapshot baseline: per branch a materialized row-id list, filtered by
//!   P ∧ Q.
//!
//! Every answer is checked against a row-at-a-time oracle. Input
//! construction (thought planes, row words, snapshot lists) is timed and
//! charged.
//!
//! `cargo run --release -p lance-graph-mask-risc --example holo_k_branch_probe`
//! `cargo test -p lance-graph-mask-risc --example holo_k_branch_probe`

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use lance_graph_mask_risc::exec::{execute_compiled, Scratch};
use lance_graph_mask_risc::{
    words_for, CompiledProgram, Foreign, LaneRef, MaskOp, Operand, Out, Planes, Pred, Program,
    Terminal, Value,
};

struct Counting;
static ALLOCS: AtomicUsize = AtomicUsize::new(0);
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: forwards the caller's layout unchanged to the system allocator.
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        // SAFETY: `p` was returned by `System.alloc` with this layout.
        unsafe { System.dealloc(p, l) }
    }
}
#[global_allocator]
static A: Counting = Counting;

const N: usize = 65_536;
const W: usize = N / 64;
const THOUGHTS: usize = 64;

fn tier() -> &'static str {
    if cfg!(target_feature = "avx512f") {
        "v4 (avx512f)"
    } else if cfg!(target_feature = "avx2") {
        "v3 (avx2): timings not reportable"
    } else {
        "below v3: timings not reportable"
    }
}

fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

struct Fixture {
    v: Vec<i32>,
    /// Canonical coverage: bit t of `w[r]` = thought t covers row r.
    w: Vec<u64>,
}

fn fixture(seed: u64) -> Fixture {
    let v = (0..N)
        .map(|r| (mix(seed ^ r as u64) % 1000) as i32)
        .collect();
    let mut w = vec![0u64; N];
    for (r, word) in w.iter_mut().enumerate() {
        for t in 0..THOUGHTS {
            if mix(seed.rotate_left(17) ^ ((r as u64) << 8) ^ t as u64).is_multiple_of(10) {
                *word |= 1 << t;
            }
        }
    }
    Fixture { v, w }
}

/// Per-thought planes M_t (representation maintenance, charged).
fn build_planes(w: &[u64]) -> Vec<Vec<u64>> {
    let mut m = vec![vec![0u64; W]; THOUGHTS];
    for (r, &word) in w.iter().enumerate() {
        let mut b = word;
        while b != 0 {
            let t = b.trailing_zeros() as usize;
            b &= b - 1;
            m[t][r / 64] |= 1 << (r % 64);
        }
    }
    m
}

/// Row words built from per-thought planes: the other orientation of the
/// same coverage (charged when planes are the stored form).
fn build_words(m: &[Vec<u64>]) -> Vec<u64> {
    let mut w = vec![0u64; N];
    for (t, plane) in m.iter().enumerate() {
        for (wi, &word) in plane.iter().enumerate() {
            let mut b = word;
            while b != 0 {
                let r = wi * 64 + b.trailing_zeros() as usize;
                b &= b - 1;
                w[r] |= 1 << t;
            }
        }
    }
    w
}

/// Per-branch materialized row lists (snapshot baseline input, charged).
fn build_lists(w: &[u64]) -> Vec<Vec<u32>> {
    let mut l = vec![Vec::new(); THOUGHTS];
    for (r, &word) in w.iter().enumerate() {
        let mut b = word;
        while b != 0 {
            let t = b.trailing_zeros() as usize;
            b &= b - 1;
            l[t].push(r as u32);
        }
    }
    l
}

#[derive(Clone, Copy)]
struct Cell {
    lo: usize,
    hi: usize,
    t: i32,
    k: usize,
}

fn sel_mask(k: usize) -> u64 {
    if k >= 64 {
        u64::MAX
    } else {
        (1u64 << k) - 1
    }
}

fn oracle(f: &Fixture, c: Cell) -> Vec<u64> {
    let mut out = vec![0u64; c.k];
    for r in c.lo..c.hi {
        if f.v[r] < c.t {
            for (t, o) in out.iter_mut().enumerate() {
                *o += (f.w[r] >> t) & 1;
            }
        }
    }
    out
}

// ---------- arm A: one program per branch ----------

fn prog_a(c: Cell, plane: u16) -> Program {
    Program::new(
        vec![
            MaskOp::Pred {
                pred: Pred::LtI32 { lane: 0, t: c.t },
                under: Some(Operand::Plane(plane)),
                dst: 0,
            },
            MaskOp::Pred {
                pred: Pred::Range {
                    lo: c.lo as u32,
                    hi: c.hi as u32,
                },
                under: Some(Operand::Scratch(0)),
                dst: 1,
            },
        ],
        Terminal::Count {
            mask: Operand::Scratch(1),
        },
    )
}

// ---------- arm B: gate once, then one And + Count per branch ----------

fn prog_gate(c: Cell) -> Program {
    Program::new(
        vec![
            MaskOp::Pred {
                pred: Pred::Range {
                    lo: c.lo as u32,
                    hi: c.hi as u32,
                },
                under: None,
                dst: 0,
            },
            MaskOp::Pred {
                pred: Pred::LtI32 { lane: 0, t: c.t },
                under: Some(Operand::Scratch(0)),
                dst: 1,
            },
        ],
        Terminal::Keep {
            mask: Operand::Scratch(1),
        },
    )
}

fn prog_b(plane: u16) -> Program {
    // Plane 0 is G; branch k is plane k + 1.
    Program::new(
        vec![MaskOp::And {
            a: Operand::Plane(0),
            b: Operand::Plane(plane),
            dst: 0,
        }],
        Terminal::Count {
            mask: Operand::Scratch(0),
        },
    )
}

fn count(v: Value) -> u64 {
    match v {
        Value::Count(n) => n as u64,
        other => panic!("expected a count, got {other:?}"),
    }
}

struct Arms<'a> {
    f: &'a Fixture,
    m: &'a [Vec<u64>],
    lists: &'a [Vec<u32>],
}

/// Prepared programs and scratch for one cell: planning happens once,
/// outside the timed region, like a prepared statement.
struct Prepared {
    a: Vec<CompiledProgram>,
    gate: CompiledProgram,
    b: Vec<CompiledProgram>,
    sa: Scratch<'static>,
    sg: Scratch<'static>,
    sb: Scratch<'static>,
}

fn prepare(c: Cell) -> Prepared {
    let a: Vec<Program> = (0..c.k).map(|k| prog_a(c, k as u16)).collect();
    let b: Vec<Program> = (0..c.k).map(|k| prog_b(k as u16 + 1)).collect();
    let gate = prog_gate(c);
    let sa = Scratch::for_program(&a[0], N).expect("scratch");
    let sb = Scratch::for_program(&b[0], N).expect("scratch");
    let sg = Scratch::for_program(&gate, N).expect("scratch");
    // Lowering recognition happens here, once, not in the timed path.
    Prepared {
        a: a.into_iter().map(CompiledProgram::new).collect(),
        gate: CompiledProgram::new(gate),
        b: b.into_iter().map(CompiledProgram::new).collect(),
        sa,
        sg,
        sb,
    }
}

impl Arms<'_> {
    fn thought_planes(&self) -> [&[u64]; THOUGHTS] {
        std::array::from_fn(|i| self.m[i].as_slice())
    }

    fn a(&self, c: Cell, p: &mut Prepared, out: &mut [u64]) {
        let masks = self.thought_planes();
        let lanes = [LaneRef::I32(&self.f.v)];
        let planes = Planes {
            n_rows: N,
            masks: &masks,
            lanes: &lanes,
        };
        for (prog, o) in p.a.iter().zip(out.iter_mut()) {
            *o = count(
                execute_compiled(
                    &prog.compiled(),
                    &planes,
                    &Foreign::NONE,
                    &mut p.sa,
                    Out::None,
                    c.lo..c.hi,
                )
                .expect("A"),
            );
        }
    }

    fn gate(&self, c: Cell, p: &mut Prepared, g: &mut [u64]) {
        let lanes = [LaneRef::I32(&self.f.v)];
        let planes = Planes {
            n_rows: N,
            masks: &[],
            lanes: &lanes,
        };
        g.fill(0);
        execute_compiled(
            &p.gate.compiled(),
            &planes,
            &Foreign::NONE,
            &mut p.sg,
            Out::Mask(g),
            c.lo..c.hi,
        )
        .expect("gate");
    }

    fn b(&self, c: Cell, p: &mut Prepared, g: &mut [u64], out: &mut [u64]) {
        self.gate(c, p, g);
        let g: &[u64] = g;
        let masks: [&[u64]; THOUGHTS + 1] =
            std::array::from_fn(|i| if i == 0 { g } else { self.m[i - 1].as_slice() });
        let planes = Planes {
            n_rows: N,
            masks: &masks,
            lanes: &[],
        };
        for (prog, o) in p.b.iter().zip(out.iter_mut()) {
            *o = count(
                execute_compiled(
                    &prog.compiled(),
                    &planes,
                    &Foreign::NONE,
                    &mut p.sb,
                    Out::None,
                    c.lo..c.hi,
                )
                .expect("B"),
            );
        }
    }

    /// Outside the contract: positional popcount of the written-by words
    /// over the rows of G. Fixed 64 counters, whatever K is.
    fn c(&self, c: Cell, p: &mut Prepared, g: &mut [u64], out: &mut [u64]) {
        self.gate(c, p, g);
        let sel = sel_mask(c.k);
        let mut cnt = [0u32; 64];
        let words = g.iter().enumerate();
        for (wi, &gw) in words.take(c.hi.div_ceil(64)).skip(c.lo / 64) {
            let mut b = gw;
            while b != 0 {
                let r = wi * 64 + b.trailing_zeros() as usize;
                b &= b - 1;
                let x = self.f.w[r] & sel;
                for (j, ct) in cnt.iter_mut().enumerate() {
                    *ct += ((x >> j) & 1) as u32;
                }
            }
        }
        for (o, &ct) in out.iter_mut().zip(cnt.iter()).take(c.k) {
            *o = u64::from(ct);
        }
    }

    /// Snapshot baseline: each branch is a materialized list of its rows.
    fn d(&self, c: Cell, out: &mut [u64]) {
        for (k, o) in out.iter_mut().enumerate().take(c.k) {
            let l = &self.lists[k];
            let a = l.partition_point(|&r| (r as usize) < c.lo);
            let b = l.partition_point(|&r| (r as usize) < c.hi);
            *o = l[a..b]
                .iter()
                .filter(|&&r| self.f.v[r as usize] < c.t)
                .count() as u64;
        }
    }
}

/// Median of 11 runs, ns per call, plus the allocation count of one call.
fn time<F: FnMut()>(mut f: F) -> (f64, usize) {
    f();
    let before = ALLOCS.load(Ordering::Relaxed);
    f();
    let allocs = ALLOCS.load(Ordering::Relaxed) - before;
    let mut v: Vec<f64> = (0..11)
        .map(|_| {
            let t = Instant::now();
            let mut n = 0u32;
            while n < 3 || t.elapsed().as_micros() < 2_000 {
                f();
                n += 1;
            }
            t.elapsed().as_nanos() as f64 / f64::from(n)
        })
        .collect();
    v.sort_by(|a, b| a.total_cmp(b));
    (v[5], allocs)
}

fn main() {
    println!("tier: {}", tier());
    let f = fixture(0x5EED);
    let t0 = Instant::now();
    let m = black_box(build_planes(&f.w));
    let m_ns = t0.elapsed().as_nanos() as f64 / N as f64;
    let t0 = Instant::now();
    let lists = black_box(build_lists(&f.w));
    let l_ns = t0.elapsed().as_nanos() as f64 / N as f64;
    let t0 = Instant::now();
    let w2 = black_box(build_words(&m));
    let w_ns = t0.elapsed().as_nanos() as f64 / N as f64;
    assert_eq!(w2, f.w, "row words rebuilt from planes");
    println!(
        "# construction (charged): thought planes from row words {m_ns:.2} ns/row ({} KiB); row words from planes {w_ns:.2} ns/row ({} KiB); snapshot lists {l_ns:.2} ns/row ({} KiB)",
        THOUGHTS * W * 8 / 1024,
        N * 8 / 1024,
        lists.iter().map(|l| l.len() * 4).sum::<usize>() / 1024
    );
    let arms = Arms {
        f: &f,
        m: &m,
        lists: &lists,
    };
    let mut g = vec![0u64; words_for(N)];
    let mut out = vec![0u64; THOUGHTS];
    println!("s\taperture\tK\tA_ns\tB_ns\tC_ns\tD_ns\tA_alloc\tB_alloc\tC_alloc\tD_alloc\tC_K/C_1");
    for &(sname, t) in &[("1%", 10), ("10%", 100), ("50%", 500)] {
        for &(aname, lo, hi) in &[("65536", 0usize, N), ("4096", 7 * 4096, 8 * 4096)] {
            let mut c1 = 0.0;
            for &k in &[1usize, 8, 64] {
                let c = Cell { lo, hi, t, k };
                let want = oracle(&f, c);
                let mut p = prepare(c);
                arms.a(c, &mut p, &mut out);
                assert_eq!(&out[..k], &want[..], "A {sname} {aname} K={k}");
                arms.b(c, &mut p, &mut g, &mut out);
                assert_eq!(&out[..k], &want[..], "B");
                arms.c(c, &mut p, &mut g, &mut out);
                assert_eq!(&out[..k], &want[..], "C");
                arms.d(c, &mut out);
                assert_eq!(&out[..k], &want[..], "D");
                let (a, aa) = time(|| arms.a(c, &mut p, &mut out));
                let (b, ba) = time(|| arms.b(c, &mut p, &mut g, &mut out));
                let (cc, ca) = time(|| arms.c(c, &mut p, &mut g, &mut out));
                let (d, da) = time(|| arms.d(c, &mut out));
                if k == 1 {
                    c1 = cc;
                }
                println!(
                    "{sname}\t{aname}\t{k}\t{a:.0}\t{b:.0}\t{cc:.0}\t{d:.0}\t{aa}\t{ba}\t{ca}\t{da}\t{:.2}",
                    cc / c1
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arms_check(seed: u64) {
        let f = fixture(seed);
        let m = build_planes(&f.w);
        let lists = build_lists(&f.w);
        let arms = Arms {
            f: &f,
            m: &m,
            lists: &lists,
        };
        let mut g = vec![0u64; words_for(N)];
        let mut out = vec![0u64; THOUGHTS];
        for &(lo, hi) in &[(0, N), (4096, 8192), (100, 3_000)] {
            for &t in &[10, 500] {
                for &k in &[1usize, 8, 64] {
                    let c = Cell { lo, hi, t, k };
                    let want = oracle(&f, c);
                    let mut p = prepare(c);
                    // Poison the shared gate buffer before each arm, so no arm
                    // can pass by reading a mask another arm left behind.
                    g.fill(u64::MAX);
                    arms.a(c, &mut p, &mut out);
                    assert_eq!(&out[..k], &want[..], "A");
                    g.fill(u64::MAX);
                    arms.b(c, &mut p, &mut g, &mut out);
                    assert_eq!(&out[..k], &want[..], "B");
                    g.fill(u64::MAX);
                    arms.c(c, &mut p, &mut g, &mut out);
                    assert_eq!(&out[..k], &want[..], "C");
                    arms.d(c, &mut out);
                    assert_eq!(&out[..k], &want[..], "D");
                }
            }
        }
    }

    /// FAILS IF: any arm differs from the oracle on any cell, including an
    /// aperture that is not word-aligned (100..3000).
    #[test]
    fn every_arm_matches_the_oracle() {
        arms_check(3);
    }

    /// FAILS IF: the fixture is degenerate: branches identical, or the
    /// aperture or the predicate not excluding anything.
    #[test]
    fn the_fixture_discriminates() {
        let f = fixture(3);
        let c = Cell {
            lo: 4096,
            hi: 8192,
            t: 100,
            k: 64,
        };
        let o = oracle(&f, c);
        let full = oracle(
            &f,
            Cell {
                lo: 0,
                hi: N,
                t: 1000,
                k: 64,
            },
        );
        assert!(o.iter().any(|&x| x != o[0]), "branches must differ");
        assert!(o.iter().all(|&x| x > 0));
        assert!(
            o.iter().zip(&full).all(|(a, b)| a * 20 < *b),
            "P ∧ Q must exclude most rows"
        );
    }
}
