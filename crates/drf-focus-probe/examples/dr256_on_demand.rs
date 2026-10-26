//! D-DR256-OD lab: on-demand global DR256 mask, native u16 scatter, bounded
//! reuse across thoughts, and L1 interference with a resident DR64 fold. Lab
//! only; pre-registered in `.claude/board/entries/2026-10-26-dr256-on-demand-l1.md`.
//!
//! 64 registers x 4,096 slots, each a sealed u16 map + a 4,096-bit selection
//! (the #1790 fixtures). Thought t brings its own 8 KiB global plane `X_t` and
//! asks `any_t = (G AND X_t) != 0` or `count_t = |G AND X_t|`.
//!
//! Arms: A local-only (any), B u16 rebuild per thought, C u32 rebuild per
//! thought (widening charged once), D bounded reuse under a witness, E count
//! without a shared G (per-thought scatter of `L AND gather(X_t)`).
//!
//! `cargo run --release --example dr256_on_demand` from this crate. ns per
//! workload, median of 15 batches; reported, not gated. Needs ndarray with
//! `mask_scatter_or_u16` (the crate patches ndarray to the sibling checkout).
use drf_focus_probe::{MapId, Scope, SealedMap, SourceId, SLOTS, TILE_WORDS};
use ndarray::simd::{
    mask_scatter_or_u16, mask_scatter_or_u32, mask_ternlog_any, mask_ternlog_popcount,
    masked_sum_i32,
};
use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::BTreeSet;
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

const U: usize = 65_536;
const GW: usize = U / 64;
const R: usize = 64;
const AND: i32 = 0xC0; // ternlog(a, b, c) = a AND b
const NS: [usize; 5] = [1, 2, 4, 8, 64];

// ------------------------------------------------------------ allocations

struct Counting;
static ALLOCS: AtomicUsize = AtomicUsize::new(0);
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: forwards the caller's layout unchanged to the system allocator.
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        // SAFETY: `p` came from `alloc` above with the same layout.
        unsafe { System.dealloc(p, l) }
    }
}
#[global_allocator]
static A: Counting = Counting;

fn allocs_during(f: impl FnOnce()) -> usize {
    let before = ALLOCS.load(Ordering::Relaxed);
    f();
    ALLOCS.load(Ordering::Relaxed) - before
}

// ------------------------------------------------------------ fixtures

fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn permutation(seed: u64) -> Vec<u16> {
    let mut p: Vec<u16> = (0..U as u32).map(|x| x as u16).collect();
    for i in (1..U).rev() {
        let j = (mix(seed ^ i as u64) % (i as u64 + 1)) as usize;
        p.swap(i, j);
    }
    p
}

fn source() -> SourceId {
    SourceId { dataset: 7, branch: 0, version: 3, index_seal: 0xD256 }
}

fn map_of(r: usize, seal: u64, ord: &[u16]) -> SealedMap {
    let ord: Box<[u16; SLOTS]> = ord.to_vec().into_boxed_slice().try_into().expect("4096 slots");
    SealedMap::new(MapId(r as u64), seal, ord)
}

/// The resident register file: 64 sealed maps, 64 local selections (512 B
/// each, contiguous), and a selection generation per register.
struct Regs {
    name: &'static str,
    maps: Vec<SealedMap>,
    local: Vec<u64>,
    gens: [u64; R],
}

impl Regs {
    fn l(&self, r: usize) -> &[u64] {
        &self.local[r * TILE_WORDS..(r + 1) * TILE_WORDS]
    }
    fn select(&mut self, r: usize, s: usize) {
        self.local[r * TILE_WORDS + s / 64] |= 1 << (s % 64);
        self.gens[r] += 1;
    }
}

fn chaotic(seed: u64) -> Vec<SealedMap> {
    (0..R).map(|r| map_of(r, source().index_seal, &permutation(seed ^ (r as u64) << 20)[..SLOTS])).collect()
}

fn place(name: &'static str, targets: &[u16], seed: u64) -> Regs {
    let mut ords: Vec<Vec<u16>> =
        (0..R).map(|r| permutation(seed ^ 0xA5A5 ^ (r as u64) << 20)[..SLOTS].to_vec()).collect();
    let mut local = vec![0u64; R * TILE_WORDS];
    let per = targets.len().div_ceil(R).max(1);
    for (r, chunk) in targets.chunks(per).enumerate() {
        let slots = permutation(seed ^ 0x51 ^ (r as u64) << 24);
        for (&g, &s) in chunk.iter().zip(slots.iter().filter(|&&s| (s as usize) < SLOTS)) {
            ords[r][s as usize] = g;
            local[r * TILE_WORDS + s as usize / 64] |= 1 << (s % 64);
        }
    }
    let maps = ords.iter().enumerate().map(|(r, o)| map_of(r, source().index_seal, o)).collect();
    Regs { name, maps, local, gens: [0; R] }
}

fn fixtures() -> Vec<Regs> {
    let mut v = Vec::new();
    let mut local = vec![0u64; R * TILE_WORDS];
    for (i, w) in local.iter_mut().enumerate() {
        *w = mix(0x50 ^ i as u64);
    }
    v.push(Regs { name: "chaotic-50%", maps: chaotic(0xC4A0), local, gens: [0; R] });
    let block: Vec<u16> =
        (128..192u32).flat_map(|y| (64..128u32).map(move |x| (y * 256 + x) as u16)).collect();
    v.push(place("clustered-64x64", &block, 0xB10C));
    v.push(place("scattered-4096", &permutation(0x5CA7)[..4096], 0x5CA7));
    v.push(Regs { name: "empty", maps: chaotic(0xE), local: vec![0; R * TILE_WORDS], gens: [0; R] });
    let mut maps = chaotic(0xF);
    for (r, m) in maps.iter_mut().enumerate().take(16) {
        let o: Vec<u16> = (r * SLOTS..(r + 1) * SLOTS).map(|g| g as u16).collect();
        *m = map_of(r, source().index_seal, &o);
    }
    v.push(Regs { name: "full", maps, local: vec![u64::MAX; R * TILE_WORDS], gens: [0; R] });
    v
}

fn x_planes(n: usize) -> Vec<Vec<u64>> {
    (0..n).map(|t| (0..GW).map(|i| mix(0xA11 ^ (t as u64) << 32 ^ i as u64)).collect()).collect()
}

// ------------------------------------------------------------ witness

/// Everything G depends on. Any difference means the derived mask is stale.
#[derive(Clone, PartialEq, Eq)]
struct Witness {
    source: SourceId,
    scope: Scope,
    maps: [(u64, u64); R],
    gens: [u64; R],
}

fn witness(reg: &Regs) -> Witness {
    let mut maps = [(0, 0); R];
    for (m, s) in maps.iter_mut().zip(&reg.maps) {
        *m = (s.id.0, s.seal);
    }
    Witness { source: source(), scope: Scope(1), maps, gens: reg.gens }
}

/// A caller-owned, disposable derived mask: valid only while `witness` holds.
struct Derived {
    g: Vec<u64>,
    witness: Option<Witness>,
}

// ------------------------------------------------------------ arms

fn build_u16(reg: &Regs, g: &mut [u64]) {
    g.fill(0);
    for r in 0..R {
        mask_scatter_or_u16(reg.l(r), &reg.maps[r].ord[..], g, U);
    }
}

fn widen(reg: &Regs, lane: &mut [u32]) {
    for (r, m) in reg.maps.iter().enumerate() {
        for (d, &o) in lane[r * SLOTS..(r + 1) * SLOTS].iter_mut().zip(m.ord.iter()) {
            *d = u32::from(o);
        }
    }
}

fn build_u32(reg: &Regs, lane: &[u32], g: &mut [u64]) {
    g.fill(0);
    for r in 0..R {
        mask_scatter_or_u32(reg.l(r), &lane[r * SLOTS..(r + 1) * SLOTS], g, U);
    }
}

#[inline]
fn bit(w: &[u64], g: usize) -> bool {
    w[g / 64] >> (g % 64) & 1 == 1
}

/// A: any_t without a global mask: walk selected slots, test X_t at ord.
fn any_local(reg: &Regs, x: &[u64]) -> bool {
    for r in 0..R {
        let ord = &reg.maps[r].ord;
        for (w, &word) in reg.l(r).iter().enumerate() {
            let mut b = word;
            while b != 0 {
                let s = w * 64 + b.trailing_zeros() as usize;
                b &= b - 1;
                if bit(x, ord[s] as usize) {
                    return true;
                }
            }
        }
    }
    false
}

/// E: count_t without a shared G: per register gather X_t under L, scatter the
/// survivors, popcount. `m` (512 B) and `g` (8 KiB) are caller scratch.
fn count_e(reg: &Regs, x: &[u64], m: &mut [u64; TILE_WORDS], g: &mut [u64]) -> u64 {
    g.fill(0);
    for r in 0..R {
        let ord = &reg.maps[r].ord;
        for (w, (&word, mw)) in reg.l(r).iter().zip(m.iter_mut()).enumerate() {
            let mut out = 0u64;
            let mut b = word;
            while b != 0 {
                let lane = b.trailing_zeros() as usize;
                b &= b - 1;
                if bit(x, ord[w * 64 + lane] as usize) {
                    out |= 1 << lane;
                }
            }
            *mw = out;
        }
        mask_scatter_or_u16(&m[..], &ord[..], g, U);
    }
    g.iter().map(|w| u64::from(w.count_ones())).sum()
}

fn count_g(g: &[u64], x: &[u64]) -> u64 {
    mask_ternlog_popcount::<AND>(g, x, g)
}

fn any_g(g: &[u64], x: &[u64]) -> bool {
    mask_ternlog_any::<AND>(g, x, g)
}

/// D: reuse the derived mask only while the witness holds; rebuild otherwise.
fn ensure(reg: &Regs, d: &mut Derived) -> bool {
    let w = witness(reg);
    if d.witness.as_ref() == Some(&w) {
        return false;
    }
    build_u16(reg, &mut d.g);
    d.witness = Some(w);
    true
}

// ------------------------------------------------------------ oracle

fn oracle(reg: &Regs) -> Vec<u64> {
    let mut s = BTreeSet::new();
    for r in 0..R {
        for slot in 0..SLOTS {
            if bit(reg.l(r), slot) {
                s.insert(reg.maps[r].ord[slot]);
            }
        }
    }
    let mut g = vec![0u64; GW];
    for a in s {
        g[a as usize / 64] |= 1 << (a % 64);
    }
    g
}

fn and_count(a: &[u64], b: &[u64]) -> u64 {
    a.iter().zip(b).map(|(x, y)| u64::from((x & y).count_ones())).sum()
}

// ------------------------------------------------------------ timing

fn median_ns(mut f: impl FnMut()) -> f64 {
    let mut v = Vec::with_capacity(15);
    for _ in 0..15 {
        let reps = 5;
        let t = Instant::now();
        for _ in 0..reps {
            f();
        }
        v.push(t.elapsed().as_nanos() as f64 / reps as f64);
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    println!("tier: {}", if cfg!(target_feature = "avx512f") { "avx512f" } else { "below v4" });
    let xs = x_planes(64);
    let mut fx = fixtures();

    // ---------------------------------------------------- correctness
    let mut lane = vec![0u32; R * SLOTS];
    let mut g = vec![0u64; GW];
    let mut g32 = vec![0u64; GW];
    let mut m = [0u64; TILE_WORDS];
    let mut e = vec![0u64; GW];
    for reg in &fx {
        let want = oracle(reg);
        build_u16(reg, &mut g);
        widen(reg, &mut lane);
        build_u32(reg, &lane, &mut g32);
        assert_eq!(g, want, "{}: u16 mask", reg.name);
        assert_eq!(g32, want, "{}: u32 mask", reg.name);
        for x in xs.iter().take(8) {
            let c = and_count(&want, x);
            assert_eq!(count_g(&g, x), c, "{}: count over G", reg.name);
            assert_eq!(count_e(reg, x, &mut m, &mut e), c, "{}: count E", reg.name);
            let want_e: Vec<u64> = want.iter().zip(x).map(|(a, b)| a & b).collect();
            assert_eq!(e, want_e, "{}: E mask = G AND X", reg.name);
            assert_eq!(any_local(reg, x), c > 0, "{}: any A", reg.name);
            assert_eq!(any_g(&g, x), c > 0, "{}: any over G", reg.name);
        }
    }
    witness_controls(&fx[1], &xs);
    println!("correctness: masks and answers equal the oracle on all fixtures; witness controls hold");

    // ---------------------------------------------------- allocations
    let mut d = Derived { g: vec![0u64; GW], witness: None };
    let reg = &fx[0];
    let hot = allocs_during(|| {
        for x in &xs {
            black_box(any_local(reg, x));
            build_u16(reg, &mut g);
            black_box(count_g(&g, x));
            black_box(count_e(reg, x, &mut m, &mut e));
            ensure(reg, &mut d);
            black_box(count_g(&d.g, x));
            build_u32(reg, &lane, &mut g32);
        }
    });
    println!("hot-path allocations after setup: {hot}");

    // ---------------------------------------------------- workloads over N
    println!("csv,fixture,question,arm,N,total_ns,per_thought_ns,global_masks_live,rebuilds");
    for reg in &fx {
        for &n in &NS {
            let x = &xs[..n];
            let a = median_ns(|| {
                for xt in x {
                    black_box(any_local(reg, xt));
                }
            });
            let b_any = median_ns(|| {
                for xt in x {
                    build_u16(reg, &mut g);
                    black_box(any_g(&g, xt));
                }
            });
            let b_cnt = median_ns(|| {
                for xt in x {
                    build_u16(reg, &mut g);
                    black_box(count_g(&g, xt));
                }
            });
            let c_cnt = median_ns(|| {
                widen(reg, &mut lane);
                for xt in x {
                    build_u32(reg, &lane, &mut g32);
                    black_box(count_g(&g32, xt));
                }
            });
            let d_any = median_ns(|| {
                d.witness = None;
                for xt in x {
                    ensure(reg, &mut d);
                    black_box(any_g(&d.g, xt));
                }
            });
            let d_cnt = median_ns(|| {
                d.witness = None;
                for xt in x {
                    ensure(reg, &mut d);
                    black_box(count_g(&d.g, xt));
                }
            });
            let e_cnt = median_ns(|| {
                for xt in x {
                    black_box(count_e(reg, xt, &mut m, &mut e));
                }
            });
            for (q, arm, t, live, rebuilds) in [
                ("any", "A-local", a, 0, 0),
                ("any", "B-u16-per-thought", b_any, 1, n),
                ("any", "D-reuse", d_any, 1, 1),
                ("count", "B-u16-per-thought", b_cnt, 1, n),
                ("count", "C-u32-per-thought", c_cnt, 1, n),
                ("count", "D-reuse", d_cnt, 1, 1),
                ("count", "E-no-shared-G", e_cnt, 1, 0),
            ] {
                println!(
                    "csv,{},{q},{arm},{n},{t:.0},{:.0},{live},{rebuilds}",
                    reg.name,
                    t / n as f64
                );
            }
        }
    }

    // ---------------------------------------------------- incompatible thoughts
    // Every thought changes one register's selection: D must rebuild each time.
    println!("csv_incompat,fixture,N,d_total_ns,b_total_ns,d_rebuilds");
    for reg in fx.iter_mut().take(3) {
        for &n in &NS {
            let mut rebuilds = 0;
            let d_t = median_ns(|| {
                d.witness = None;
                rebuilds = 0;
                for (t, xt) in xs[..n].iter().enumerate() {
                    reg.gens[t % R] += 1; // a selection change (bits unchanged, witness moves)
                    if ensure(reg, &mut d) {
                        rebuilds += 1;
                    }
                    black_box(count_g(&d.g, xt));
                }
            });
            let b_t = median_ns(|| {
                for xt in &xs[..n] {
                    build_u16(reg, &mut g);
                    black_box(count_g(&g, xt));
                }
            });
            println!("csv_incompat,{},{n},{d_t:.0},{b_t:.0},{rebuilds}", reg.name);
        }
    }

    // ---------------------------------------------------- L1 interference
    l1(&fx[0], &xs);
    l1(&fx[1], &xs);
}

/// The resident DR64 fold F (one register: 512 B selection + 16 KiB i32 column)
/// timed inside 64-thought loops that run nothing, D's fold over a live G, B's
/// rebuild, or A's local walk between consecutive F calls.
fn l1(reg: &Regs, xs: &[Vec<u64>]) {
    let col: Vec<i32> = (0..SLOTS).map(|i| (mix(0xF0 ^ i as u64) % 2001) as i32 - 1000).collect();
    let sel = reg.l(0).to_vec();
    let mut g = vec![0u64; GW];
    build_u16(reg, &mut g);
    let mut f_ns: [Vec<f64>; 4] = Default::default();
    for _round in 0..40 {
        for (k, f) in f_ns.iter_mut().enumerate() {
            for xt in xs {
                let t = Instant::now();
                black_box(masked_sum_i32(black_box(&col), black_box(&sel)));
                f.push(t.elapsed().as_nanos() as f64);
                match k {
                    0 => {}
                    1 => {
                        black_box(count_g(&g, xt));
                    }
                    2 => {
                        build_u16(reg, &mut g);
                        black_box(count_g(&g, xt));
                    }
                    _ => {
                        black_box(any_local(reg, xt));
                    }
                }
            }
        }
    }
    let med = |v: &mut Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        (v[v.len() / 2], v[v.len() * 9 / 10])
    };
    let names = ["F alone", "F + D fold over live G", "F + B rebuild", "F + A local walk"];
    for (name, v) in names.iter().zip(f_ns.iter_mut()) {
        let (p50, p90) = med(v);
        println!("csv_l1,{},{name},{p50:.0},{p90:.0}", reg.name);
    }
}

/// Witness controls: a reused mask is stale after a selection change or a
/// rebinding, the witness catches both, and skipping it gives a wrong answer.
fn witness_controls(base: &Regs, xs: &[Vec<u64>]) {
    let mut reg = Regs {
        name: "witness",
        maps: base
            .maps
            .iter()
            .enumerate()
            .map(|(r, m)| map_of(r, m.seal, &m.ord[..]))
            .collect(),
        local: base.local.clone(),
        gens: base.gens,
    };
    let x = &xs[0];
    let mut d = Derived { g: vec![0u64; GW], witness: None };
    assert!(ensure(&reg, &mut d), "first demand builds");
    assert!(!ensure(&reg, &mut d), "a compatible second thought reuses");
    assert_eq!(d.g, oracle(&reg));

    // Selection change: select a slot whose address is outside G and inside X.
    let g0 = oracle(&reg);
    let (r, s) = (0..R)
        .flat_map(|r| (0..SLOTS).map(move |s| (r, s)))
        .find(|&(r, s)| {
            let o = reg.maps[r].ord[s] as usize;
            !bit(reg.l(r), s) && !bit(&g0, o) && bit(x, o)
        })
        .expect("a slot that changes the answer");
    let stale = count_g(&d.g, x);
    reg.select(r, s);
    let want = and_count(&oracle(&reg), x);
    assert_ne!(stale, want, "a stale mask gives a wrong count (the witness is load-bearing)");
    assert!(ensure(&reg, &mut d), "a selection change rebuilds");
    assert_eq!(count_g(&d.g, x), want);

    // Rebinding: map 7 at a new seal with different ordinals.
    let shifted: Vec<u16> = reg.maps[7].ord.iter().map(|o| o.wrapping_add(1)).collect();
    reg.maps[7] = map_of(7, reg.maps[7].seal + 1, &shifted);
    assert!(ensure(&reg, &mut d), "a rebinding rebuilds");
    assert_eq!(d.g, oracle(&reg));

    // Release and regenerate: a cleared witness rebuilds the identical mask.
    let kept = d.g.clone();
    d.g.fill(0);
    d.witness = None;
    assert!(ensure(&reg, &mut d));
    assert_eq!(d.g, kept, "regeneration is exact");
}
