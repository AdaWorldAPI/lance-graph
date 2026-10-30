//! Gate I cost probe. Cold (XLOG parse+compile, admission, fact extraction,
//! address maps, register fill) is reported in µs and is NOT a gate: it runs
//! once per program. Hot (bind + chain typing + propagate) is reported in ns.
//!
//!   cargo run --release --example cost --manifest-path crates/xlog-relcomp-probe/Cargo.toml
use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::Instant;

use dr_math_binding_probe::relcomp::{Rows, Work, compose};
use dr_math_binding_probe::{
    Axis, AxisKind, Focus, FrameRevisions, FrameStore, Guards, MethodSignature, Scope,
};
use xlog_logic::{Compiler, parse_program};
use xlog_relcomp_probe::{AddressMap, admit, admit_program, facts, propagate, register};

struct Counting;
static ALLOCS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Relaxed);
        BYTES.fetch_add(l.size() as u64, Relaxed);
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
}
#[global_allocator]
static A: Counting = Counting;

const X: Axis = Axis {
    class: 10,
    kind: AxisKind::Identity,
};
const Y: Axis = Axis {
    class: 11,
    kind: AxisKind::Property,
};
const Z: Axis = Axis {
    class: 12,
    kind: AxisKind::Community,
};

fn allocs() -> (u64, u64) {
    (ALLOCS.load(Relaxed), BYTES.load(Relaxed))
}

type Tuples = Vec<(u32, u32)>;

fn program(name: &str) -> String {
    let (r, s): (Tuples, Tuples) = match name {
        "selective" => (vec![(1, 2), (1, 3), (9, 7)], vec![(2, 4), (3, 5), (7, 4)]),
        "dense" => {
            let all: Vec<(u32, u32)> = (0..64).flat_map(|a| (0..64).map(move |b| (a, b))).collect();
            (all.clone(), all)
        }
        // One hub: x=1 reaches every Y, every Y reaches z=4.
        "skewed" => (
            (0..64).map(|b| (1, b)).collect(),
            (0..64).map(|a| (a, 4)).collect(),
        ),
        // R's Y values never appear as S's rows.
        "disconnected" => (
            (0..32).map(|b| (1, b)).collect(),
            (32..64).map(|a| (a, 4)).collect(),
        ),
        // Every edge present except the ones the answer would need.
        "adversarial" => (
            (0..64).map(|b| (1, b)).collect(),
            (0..64)
                .flat_map(|a| (0..64).filter(|&c| c != 4).map(move |c| (a, c)))
                .chain([(5, 4)])
                .collect(),
        ),
        _ => unreachable!(),
    };
    let mut src = String::from("pred r(u32, u32).\npred s(u32, u32).\npred q(u32).\n");
    for (a, b) in &r {
        src += &format!("r({a}, {b}).\n");
    }
    for (a, b) in &s {
        src += &format!("s({a}, {b}).\n");
    }
    src += "q(Y) :- r(1, Y), s(Y, 4).\n?- q(Y).\n";
    src
}

fn main() {
    let isa = std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|c| {
            c.lines()
                .find(|l| l.starts_with("model name"))
                .map(str::to_string)
        })
        .unwrap_or_default();
    println!("host: {isa}");
    println!(
        "avx512f={} avx2={}",
        std::arch::is_x86_feature_detected!("avx512f"),
        std::arch::is_x86_feature_detected!("avx2")
    );
    println!(
        "{:<13} {:>9} {:>8} {:>9} | {:>7} {:>6} {:>6} {:>5} | {:>8} {:>8} {:>8} | ans",
        "fixture",
        "cold_us",
        "c_alloc",
        "c_bytes",
        "hot_ns",
        "h_alc",
        "words",
        "rORs",
        "scal_ns",
        "prep_ns",
        "full_ns"
    );
    for name in [
        "selective",
        "dense",
        "skewed",
        "disconnected",
        "adversarial",
    ] {
        let src = program(name);

        // ── cold ──────────────────────────────────────────────────────────
        let (a0, b0) = allocs();
        let t = Instant::now();
        let prog = parse_program(&src).unwrap();
        let mut c = Compiler::new();
        let plan = c.compile(&src).unwrap();
        admit_program(&prog).unwrap();
        let body = &plan
            .rules_by_scc
            .iter()
            .flatten()
            .find(|r| r.head == "q")
            .unwrap()
            .body;
        let q = admit(body).unwrap();
        let r = facts(&prog, "r").unwrap();
        let s = facts(&prog, "s").unwrap();
        let xs = AddressMap::new(X, r.iter().map(|t| t.0).chain(q.left_bound)).unwrap();
        let ys = AddressMap::new(Y, r.iter().map(|t| t.1).chain(s.iter().map(|t| t.0))).unwrap();
        let zs = AddressMap::new(Z, s.iter().map(|t| t.1).chain(q.right_bound)).unwrap();
        let rr: Rows = register(&r, &xs, &ys).unwrap();
        let ss: Rows = register(&s, &ys, &zs).unwrap();
        let x = xs.slot(1).unwrap();
        let z = zs.slot(4).unwrap();
        let cold = t.elapsed();
        let (a1, b1) = allocs();

        // ── hot ───────────────────────────────────────────────────────────
        let st = FrameStore::new(FrameRevisions {
            source: 1,
            relation_map: 1,
            coordinates: 1,
        });
        let g = Guards {
            frame: st.seal(),
            focus: Focus { revision: 1 },
            scope: Scope { id: 2 },
        };
        let now = (Focus { revision: 1 }, Scope { id: 2 });
        let (sr, sx) = (
            MethodSignature::presence(20, 0, X, Y),
            MethodSignature::presence(21, 0, Y, Z),
        );
        const N: u32 = 20_000;
        let run = || propagate(&st, &g, now, (sr, &rr), (sx, &ss), Some(x), Some(z), &ys).unwrap();
        let p = run();
        for _ in 0..1000 {
            black_box(run());
        }
        let (h0, _) = allocs();
        let t = Instant::now();
        for _ in 0..N {
            black_box(run());
        }
        let hot = t.elapsed().as_nanos() as f64 / f64::from(N);
        let h_alloc = (allocs().0 - h0) as f64 / f64::from(N);

        // Scalar enumeration over the tuples (the reference, not a rival).
        let t = Instant::now();
        for _ in 0..200 {
            let mut m = 0u64;
            for &(a, b) in black_box(&r) {
                if a == 1 {
                    for &(c2, d) in black_box(&s) {
                        if b == c2 && d == 4 {
                            m |= 1 << ys.slot(b).unwrap();
                        }
                    }
                }
            }
            black_box(m);
        }
        let scal = t.elapsed().as_nanos() as f64 / 200.0;

        // Prepared inverse: the z column packed once at prepare time, so the
        // hot step is two word reads and one AND. Costs a per-z prepare.
        let mut col = 0u64;
        for (i, w) in ss.iter().enumerate() {
            col |= (w >> z & 1) << i;
        }
        let t = Instant::now();
        for _ in 0..N {
            black_box(black_box(rr[x]) & black_box(col));
        }
        let prep = t.elapsed().as_nanos() as f64 / f64::from(N);

        // Full composition R∘S (answers "does x reach z", NOT which Y): the
        // existing kernel's cost on the same registers, for scale only.
        let t = Instant::now();
        for _ in 0..2000 {
            let mut w = Work::default();
            black_box(compose(black_box(&rr), black_box(&ss), &mut w));
        }
        let full = t.elapsed().as_nanos() as f64 / 2000.0;

        println!(
            "{:<13} {:>9.1} {:>8} {:>9} | {:>7.1} {:>6.2} {:>6} {:>5} | {:>8.1} {:>8.2} {:>8.1} | {:?}",
            name,
            cold.as_secs_f64() * 1e6,
            a1 - a0,
            b1 - b0,
            hot,
            h_alloc,
            p.words_read,
            p.work.row_ors,
            scal,
            prep,
            full,
            ys.ids(p.answer).len(),
        );
    }
}
