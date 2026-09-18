//! D-DIAMOND-1 — the dual-fold substrate probe. Library half: the synthetic
//! ontology, the oracle, the four arms as functions, and the F4 guard. The
//! binary (`main.rs`) runs them at N = 1M and prints the report; the tests
//! run them at small N against the oracle.
//!
//! Everything timed here is first checked against the row oracle
//! (`SemanticPrefix::matches` per row) — oracle-first, like
//! `facet_axis_lcp_probe.rs`.

use lance_graph_contract::facet::{FacetCascade, SemanticPrefix};
use lance_graph_contract::ordered_lane::{OrderedLaneWitness, SealedFacetLane, WitnessError};
use lance_graph_mask_risc::words_for;
use ndarray::simd::{mask_and, mask_and_assign, mask_set_range, ternary_match_u64_to_mask};
use std::hint::black_box;
use std::sync::{Arc, RwLock};
use std::time::Instant;

pub const SEED: u64 = 0x9E37_79B9_7F4A_7C15;

// ─────────────────────────────────────────────────────────────────────────────
// Deterministic generation
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct SplitMix64(pub u64);

impl SplitMix64 {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Skewed index in `0..n`: `floor(n · u^k)` — the mass piles onto low
    /// indices (k = 3 gives a ~Zipf-like head; the head index is drawn ~40%
    /// of the time for n = 40).
    pub fn skewed(&mut self, n: u64, k: i32) -> u64 {
        let u = self.unit().powi(k);
        ((n as f64) * u) as u64
    }
}

/// The ontology lane's semantic tiles: canon (concept) · custom (app) · six
/// cascade tiers with branching that widens coarse→fine and skew at every
/// tile so subtree sizes differ by orders of magnitude.
pub fn ontology_key(r: &mut SplitMix64) -> FacetCascade {
    let canon = 0x0100 + r.skewed(48, 3) as u16;
    let custom = 1 + r.skewed(6, 2) as u16;
    let t0 = r.skewed(40, 3) as u16;
    let t1 = r.skewed(64, 2) as u16;
    let t2 = r.skewed(256, 2) as u16;
    let t3 = (r.next_u64() % 256) as u16;
    let t4 = (r.next_u64() % 4096) as u16;
    let t5 = (r.next_u64() & 0xFFFF) as u16;
    FacetCascade::from_semantic_tiles([canon, custom, t0, t1, t2, t3, t4, t5])
}

/// Tenant classid for the second lane (a V3-style `canon:gen` pair).
pub const TENANT_CANON: u16 = 0x0401;
pub const TENANT_CUSTOM: u16 = 0x1000;

/// The correlated L4-shaped tenant lane: six `(8:8)` palette pairs, each tile
/// derived from the ontology key's coarse tiles with probability `p_dep` and
/// random otherwise — correlated, never nested, so an intersection is
/// non-trivial and the tenant lane is NOT sorted along the ontology ordinal.
pub fn tenant_key(a: FacetCascade, r: &mut SplitMix64, p_dep: f64) -> FacetCascade {
    let t = a.semantic_tiles();
    let mut b = [0u16; 8];
    b[0] = TENANT_CANON;
    b[1] = TENANT_CUSTOM;
    // palette pair = (hi:lo); hi from the concept, lo from the first tier.
    // (hi:lo) palette pairs from the SKEWED coarse tiles (canon, custom, t0,
    // t1, t2), never from the uniform fine ones — so head pairs carry
    // populations and a tenant prefix intersects an ontology prefix
    // non-trivially.
    let dep = [
        ((((t[0] as u32) * 13 + (t[2] as u32) * 7) & 0xFF) << 8 | (((t[3] as u32) * 5) & 0xFF))
            as u16,
        ((((t[1] as u32) * 31 + (t[2] as u32)) & 0xFF) << 8 | (((t[4] as u32) * 3) & 0xFF)) as u16,
        ((((t[2] as u32) * 17 + (t[3] as u32)) & 0xFF) << 8 | (((t[4] as u32) * 7) & 0xFF)) as u16,
    ];
    for i in 0..6 {
        let dependent = r.unit() < p_dep;
        b[2 + i] = if i < 3 && dependent {
            dep[i]
        } else {
            (r.next_u64() & 0xFFFF) as u16
        };
    }
    FacetCascade::from_semantic_tiles(b)
}

/// The two lanes over one ordinal: `ontology` sorted (sealed), `tenant[i]`
/// belonging to row `i`. Plus the semantic `u64` planes the sweep reads.
pub struct World {
    pub lane: Arc<SealedFacetLane>,
    pub witness: OrderedLaneWitness,
    pub tenant: Vec<FacetCascade>,
    pub a_hi: Vec<u64>,
    pub a_lo: Vec<u64>,
    pub b_hi: Vec<u64>,
    pub b_lo: Vec<u64>,
    /// Seal-sort cost of the ontology lane (`SealedFacetLane::seal`), ns.
    pub seal_ns: f64,
}

pub fn build_world(n: usize, seed: u64, p_dep: f64) -> World {
    let mut r = SplitMix64(seed);
    let raw: Vec<FacetCascade> = (0..n).map(|_| ontology_key(&mut r)).collect();
    let t0 = Instant::now();
    let lane = SealedFacetLane::seal(raw, 1).expect("seals");
    let seal_ns = t0.elapsed().as_nanos() as f64;
    let witness = lane.witness();
    let tenant: Vec<FacetCascade> = lane
        .keys()
        .iter()
        .map(|&a| tenant_key(a, &mut r, p_dep))
        .collect();
    let (a_hi, a_lo): (Vec<u64>, Vec<u64>) =
        lane.keys().iter().map(|k| k.semantic_u64_halves()).unzip();
    let (b_hi, b_lo): (Vec<u64>, Vec<u64>) = tenant.iter().map(|k| k.semantic_u64_halves()).unzip();
    World {
        lane: Arc::new(lane),
        witness,
        tenant,
        a_hi,
        a_lo,
        b_hi,
        b_lo,
        seal_ns,
    }
}

impl World {
    pub fn n(&self) -> usize {
        self.lane.keys().len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Oracle + F4 guard
// ─────────────────────────────────────────────────────────────────────────────

pub fn oracle_mask(keys: &[FacetCascade], p: &SemanticPrefix) -> Vec<u64> {
    let mut m = vec![0u64; words_for(keys.len())];
    for (i, k) in keys.iter().enumerate() {
        if p.matches(*k) {
            m[i / 64] |= 1u64 << (i % 64);
        }
    }
    m
}

pub fn popcount(m: &[u64]) -> usize {
    m.iter().map(|w| w.count_ones() as usize).sum()
}

/// F4 — anti-vacuity: a timed population must be non-empty and must exclude
/// at least two thirds of the lane (`kept · 3 < total`).
pub fn nontrivial(kept: usize, total: usize) -> Result<(), String> {
    if kept == 0 {
        return Err("vacuous: kept == 0".into());
    }
    if kept * 3 >= total {
        return Err(format!("vacuous: kept {kept} · 3 >= total {total}"));
    }
    Ok(())
}

/// F4 for an intersection: neither input contains the other, and the AND is
/// neither empty nor equal to either input.
pub fn nontrivial_intersection(a: &[u64], b: &[u64], and: &[u64]) -> Result<(), String> {
    let (pa, pb, pand) = (popcount(a), popcount(b), popcount(and));
    if pand == 0 {
        return Err("vacuous ∩: empty".into());
    }
    if pand == pa {
        return Err("vacuous ∩: A ⊆ B (AND == A)".into());
    }
    if pand == pb {
        return Err("vacuous ∩: B ⊆ A (AND == B)".into());
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Timing
// ─────────────────────────────────────────────────────────────────────────────

/// ns per call: min over `rounds` of the mean over `reps` calls.
pub fn time_ns(rounds: usize, reps: usize, mut f: impl FnMut()) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..rounds {
        let t0 = Instant::now();
        for _ in 0..reps {
            f();
        }
        let ns = t0.elapsed().as_nanos() as f64 / reps as f64;
        if ns < best {
            best = ns;
        }
    }
    best
}

/// Stream through a 64 MiB buffer so nothing of ours survives in L1/L2
/// (8 MiB here). L3 on this box is 260 MiB and is NOT evicted by this.
pub fn evict_l2(scratch: &mut [u64]) -> u64 {
    let mut acc = 0u64;
    for (i, w) in scratch.iter_mut().enumerate() {
        *w = w.wrapping_add(i as u64);
        acc ^= *w;
    }
    black_box(acc)
}

// ─────────────────────────────────────────────────────────────────────────────
// P1 — point universe
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PairClass {
    Equal,
    CanonMismatch,
    CustomMismatch,
    EarlyTier,
    LateTier,
    Unrelated,
}

impl PairClass {
    pub const ALL: [PairClass; 6] = [
        PairClass::Equal,
        PairClass::CanonMismatch,
        PairClass::CustomMismatch,
        PairClass::EarlyTier,
        PairClass::LateTier,
        PairClass::Unrelated,
    ];
    pub fn label(self) -> &'static str {
        match self {
            PairClass::Equal => "fully equal",
            PairClass::CanonMismatch => "mismatch @ classid canon",
            PairClass::CustomMismatch => "mismatch @ classid custom",
            PairClass::EarlyTier => "mismatch @ early tier (t0)",
            PairClass::LateTier => "mismatch @ late tier (t5)",
            PairClass::Unrelated => "unrelated (random pair)",
        }
    }
    /// The semantic LCP every pair of this class must have (None = varies).
    pub fn expected_lcp(self) -> Option<u8> {
        match self {
            PairClass::Equal => Some(8),
            PairClass::CanonMismatch => Some(0),
            PairClass::CustomMismatch => Some(1),
            PairClass::EarlyTier => Some(2),
            PairClass::LateTier => Some(7),
            PairClass::Unrelated => None,
        }
    }
}

pub fn make_pairs(
    keys: &[FacetCascade],
    class: PairClass,
    n: usize,
    r: &mut SplitMix64,
) -> Vec<(FacetCascade, FacetCascade)> {
    (0..n)
        .map(|_| {
            let a = keys[(r.next_u64() % keys.len() as u64) as usize];
            let mut t = a.semantic_tiles();
            let b = match class {
                PairClass::Equal => a,
                PairClass::CanonMismatch => {
                    t[0] ^= 0x0001;
                    FacetCascade::from_semantic_tiles(t)
                }
                PairClass::CustomMismatch => {
                    t[1] ^= 0x0001;
                    FacetCascade::from_semantic_tiles(t)
                }
                PairClass::EarlyTier => {
                    t[2] ^= 0x0001;
                    FacetCascade::from_semantic_tiles(t)
                }
                PairClass::LateTier => {
                    t[7] ^= 0x0001;
                    FacetCascade::from_semantic_tiles(t)
                }
                PairClass::Unrelated => keys[(r.next_u64() % keys.len() as u64) as usize],
            };
            (a, b)
        })
        .collect()
}

/// Arm 1 — the corrected 8-tile lens: `vpxor` + classid-tile swap + `tzcnt`.
#[inline(never)]
pub fn lcp_tzcnt(a: &FacetCascade, b: &FacetCascade) -> u8 {
    a.shared_prefix_tiles(*b)
}

/// Arm 2 — the peek chain over the semantic tiles: eight `u16` compares in
/// coarse→fine order with early exit. Nothing assembled.
#[inline(never)]
pub fn lcp_peek(a: &FacetCascade, b: &FacetCascade) -> u8 {
    let (x, y) = (a.semantic_tiles(), b.semantic_tiles());
    let mut n = 0u8;
    while n < 8 && x[n as usize] == y[n as usize] {
        n += 1;
    }
    n
}

/// Arm 3 — the peek chain straight over the LE byte image at the SEMANTIC
/// byte offsets (canon = bytes 2..4, custom = 0..2, tiers = 4+2i..), the
/// `movzbl`/`cmp` shape the 1.72 ns axis chain measured. Two byte compares per
/// tile, early exit at the first differing tile.
#[inline(never)]
pub fn lcp_peek_bytes(a: &FacetCascade, b: &FacetCascade) -> u8 {
    const OFF: [usize; 8] = [2, 0, 4, 6, 8, 10, 12, 14];
    let (x, y) = (a.as_bytes(), b.as_bytes());
    let mut n = 0u8;
    while n < 8 {
        let o = OFF[n as usize];
        if x[o] != y[o] || x[o + 1] != y[o + 1] {
            break;
        }
        n += 1;
    }
    n
}

pub struct P1Row {
    pub class: PairClass,
    pub tzcnt_ns: f64,
    pub peek_u16_ns: f64,
    pub peek_bytes_ns: f64,
    pub ancestor_frac: f64,
}

/// Verify all three arms agree with each other and with the class's expected
/// LCP on every pair, then time each. `depth` is `depth(a)` for the
/// `is_ancestor` derivation (reported as a fraction, not timed separately: it
/// is one compare on the LCP).
pub fn run_p1(
    keys: &[FacetCascade],
    pairs_per_class: usize,
    depth: u8,
    r: &mut SplitMix64,
) -> Vec<P1Row> {
    let mut out = Vec::new();
    for class in PairClass::ALL {
        let pairs = make_pairs(keys, class, pairs_per_class, r);
        let mut anc = 0usize;
        for (a, b) in &pairs {
            let (t, p, pb) = (lcp_tzcnt(a, b), lcp_peek(a, b), lcp_peek_bytes(a, b));
            assert_eq!(t, p, "tzcnt vs peek_u16 disagree on {class:?}");
            assert_eq!(t, pb, "tzcnt vs peek_bytes disagree on {class:?}");
            if let Some(e) = class.expected_lcp() {
                assert_eq!(t, e, "{class:?}: LCP must be {e}");
            }
            if t >= depth {
                anc += 1;
            }
        }
        let time = |f: fn(&FacetCascade, &FacetCascade) -> u8| {
            time_ns(7, 1, || {
                let mut acc = 0u32;
                for (a, b) in &pairs {
                    acc += f(black_box(a), black_box(b)) as u32;
                }
                black_box(acc);
            }) / pairs.len() as f64
        };
        out.push(P1Row {
            class,
            tzcnt_ns: time(lcp_tzcnt),
            peek_u16_ns: time(lcp_peek),
            peek_bytes_ns: time(lcp_peek_bytes),
            ancestor_frac: anc as f64 / pairs.len() as f64,
        });
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// P2 — field universe
// ─────────────────────────────────────────────────────────────────────────────

/// `MatchU64` sweep over the semantic planes — the shipped kernel, in the
/// exact shape `quack::Filter::prefix_facet` lowers to without a witness.
pub fn sweep_mask(
    a_hi: &[u64],
    a_lo: &[u64],
    p: &SemanticPrefix,
    dst: &mut [u64],
    tmp: &mut [u64],
) {
    let (p_hi, p_lo) = p.lo_key().semantic_u64_halves();
    let d = u32::from(p.depth());
    if d <= 4 {
        let care = if d == 0 { 0 } else { u64::MAX << (64 - 16 * d) };
        ternary_match_u64_to_mask(a_hi, p_hi & care, care, dst);
    } else {
        ternary_match_u64_to_mask(a_hi, p_hi, u64::MAX, tmp);
        let care = u64::MAX << (64 - 16 * (d - 4));
        ternary_match_u64_to_mask(a_lo, p_lo & care, care, dst);
        mask_and_assign(dst, tmp);
    }
}

/// Witnessed bound + paint, over a destination sized to the WHOLE lane
/// (`dst.len() == words_for(n_rows)`). Used only where a full-lane mask is
/// actually needed (correctness checks against the oracle) — never on the
/// timed fold path, where [`touched_write`] is used instead.
pub fn bound_mask(
    lane: &SealedFacetLane,
    w: &OrderedLaneWitness,
    p: &SemanticPrefix,
    dst: &mut [u64],
) -> Result<(u32, u32), WitnessError> {
    let (lo, hi) = lane.bound(w, p)?;
    // `mask_set_range` writes every word (zero before, ones inside, zero
    // after) — no separate fill.
    mask_set_range(dst, lo as usize, hi as usize);
    Ok((lo, hi))
}

/// The TOUCHED-ONLY write: a destination sized to `words_for(hi)`, not
/// `words_for(n_rows)`. `mask_set_range` zeroes `dst[..lo_word]` and
/// `dst[hi_word+1..]` and only ever needs to reach `dst.len()` — so a
/// destination whose length depends on `hi` (not on the lane's row count)
/// bounds `mask_set_range`'s own work to `words_for(hi)` words, never to the
/// whole lane. This is the fix for the materialization bug: the old P2 write
/// path allocated `dst` sized to `words_for(n_rows)` regardless of how narrow
/// `[lo, hi)` was, so `mask_set_range` always did O(n_rows/64) work. Returns
/// the touched slice `dst[w0..w1]` where `w0 = lo/64`.
pub fn touched_write(lo: u32, hi: u32) -> Vec<u64> {
    let (lo, hi) = (lo as usize, hi as usize);
    let w1 = words_for(hi);
    let mut dst = vec![0u64; w1];
    mask_set_range(&mut dst, lo, hi);
    dst
}

#[derive(Debug, Clone)]
pub struct P2Row {
    pub n: usize,
    pub depth: u8,
    pub kept: usize,
    pub bound_ns: f64,
    pub touched_write_ns: f64,
    /// The `MatchU64` sweep over the WHOLE lane — a reference comparator,
    /// never part of any fold's own lowering (`no sweep, we said
    /// fold`). Kept because the original spec asked to compare the witnessed
    /// bound against this exact kernel.
    pub reference_sweep_ns: f64,
}

impl P2Row {
    pub fn bound_fold_total(&self) -> f64 {
        self.bound_ns + self.touched_write_ns
    }
    pub fn speedup(&self) -> f64 {
        self.reference_sweep_ns / self.bound_fold_total()
    }
}

/// Pick, per depth, a probe key whose prefix population passes F4 on this lane.
pub fn pick_prefixes(
    lane: &SealedFacetLane,
    w: &OrderedLaneWitness,
    depths: &[u8],
    r: &mut SplitMix64,
) -> Vec<(SemanticPrefix, usize)> {
    let n = lane.keys().len();
    let mut out = Vec::new();
    for &d in depths {
        let mut found = None;
        for _ in 0..2000 {
            let k = lane.keys()[(r.next_u64() % n as u64) as usize];
            let p = SemanticPrefix::of(k, d);
            let (lo, hi) = lane.bound(w, &p).expect("witnessed");
            let kept = (hi - lo) as usize;
            if nontrivial(kept, n).is_ok() {
                found = Some((p, kept));
                break;
            }
        }
        if let Some(f) = found {
            out.push(f);
        }
    }
    out
}

/// Stride-sample a sealed lane down to `n` rows (keeps order and distribution).
pub fn subsample(lane: &SealedFacetLane, n: usize) -> SealedFacetLane {
    let all = lane.keys();
    let step = (all.len() / n).max(1);
    let keys: Vec<FacetCascade> = all.iter().step_by(step).take(n).copied().collect();
    SealedFacetLane::attest_sorted(keys, lane.version())
        .expect("a subsequence of an ordered lane is ordered")
}

/// Run P2 on one lane: for each prefix, verify bound == sweep == oracle, then
/// time bound / write / sweep. `evict` streams a 64 MiB buffer before each
/// timed round (L2-evicted regime).
pub fn run_p2(
    lane: &SealedFacetLane,
    w: &OrderedLaneWitness,
    prefixes: &[(SemanticPrefix, usize)],
    evict: Option<&mut Vec<u64>>,
) -> Vec<P2Row> {
    let n = lane.keys().len();
    let (a_hi, a_lo): (Vec<u64>, Vec<u64>) =
        lane.keys().iter().map(|k| k.semantic_u64_halves()).unzip();
    let words = words_for(n);
    let mut dst = vec![0u64; words];
    let mut tmp = vec![0u64; words];
    let mut dst2 = vec![0u64; words];
    let mut evict = evict;
    let mut out = Vec::new();
    for (p, kept) in prefixes {
        // Oracle-first.
        let truth = oracle_mask(lane.keys(), p);
        let (lo, hi) = bound_mask(lane, w, p, &mut dst).expect("witnessed");
        assert_eq!(dst, truth, "bound mask != oracle at depth {}", p.depth());
        assert_eq!((hi - lo) as usize, *kept);
        sweep_mask(&a_hi, &a_lo, p, &mut dst2, &mut tmp);
        assert_eq!(dst2, truth, "sweep mask != oracle at depth {}", p.depth());
        nontrivial(*kept, n).expect("F4");

        let ev = |e: &mut Option<&mut Vec<u64>>| {
            if let Some(buf) = e.as_deref_mut() {
                evict_l2(buf);
            }
        };
        // bound: the two partition_points + O(1) validate
        let bound_ns = {
            let mut best = f64::INFINITY;
            for _ in 0..7 {
                ev(&mut evict);
                let t0 = Instant::now();
                let r = black_box(lane.bound(black_box(w), black_box(p)).unwrap());
                best = best.min(t0.elapsed().as_nanos() as f64);
                black_box(r);
            }
            best
        };
        // TOUCHED-ONLY write: the destination is sized to `words_for(hi)`,
        // never to `words_for(n)` — so cost scales with `hi`, not with the
        // lane's row count. Allocation is INSIDE the timed loop deliberately:
        // it is the size of the allocation (bounded by `hi`, not `n`) that is
        // under test, not amortized-away allocator cost.
        let touched_write_ns = {
            let mut best = f64::INFINITY;
            for _ in 0..7 {
                ev(&mut evict);
                let t0 = Instant::now();
                let d = touched_write(black_box(lo), black_box(hi));
                best = best.min(t0.elapsed().as_nanos() as f64);
                black_box(&d);
            }
            best
        };
        let reference_sweep_ns = {
            let mut best = f64::INFINITY;
            for _ in 0..7 {
                ev(&mut evict);
                let t0 = Instant::now();
                sweep_mask(
                    black_box(&a_hi),
                    black_box(&a_lo),
                    black_box(p),
                    &mut dst2,
                    &mut tmp,
                );
                best = best.min(t0.elapsed().as_nanos() as f64);
                black_box(&dst2);
            }
            best
        };
        out.push(P2Row {
            n,
            depth: p.depth(),
            kept: *kept,
            bound_ns,
            touched_write_ns,
            reference_sweep_ns,
        });
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// P3 — fold intersection over one ordinal, via a Morton-interleaved joint key
// ─────────────────────────────────────────────────────────────────────────────
//
// **No sweep in the fold arm.** The old arm A called `narrowed_sweep`, a
// `ternary_match_u64_to_mask` narrowed to a row range — still a per-row sweep,
// since narrowing the range a sweep runs over does not change what it is. The
// fix: build a THIRD sorted sequence — the joint key interleaving `d` tiles
// from the ontology key and `d` tiles from the tenant key, tile-by-tile
// (A0 B0 A1 B1 … A_{d-1} B_{d-1}) — and bound THAT with two `partition_point`s,
// exactly like `SealedFacetLane::bound` does for one key. The fold arm is then
// nothing but that one bound: no per-row predicate anywhere in it.
//
// **Equal depths only.** A joint key packs `d` tiles from each side, 16 bits
// per tile per side, so `2·d·16` bits must fit a `u128` — `d <= 4`. Unequal
// depths would need a padding/truncation scheme for the shorter side; rather
// than invent one, this probe restricts its TIMED P3 rows to `a_depth ==
// b_depth` and states that limitation here, per the task's own fallback
// instruction.

/// Minimum `kept ∩` for a P3 pair to count as non-trivial (F4 in spirit, not
/// only in letter: a one-row overlap satisfies the set conditions vacuously).
pub const INTERSECTION_FLOOR: usize = 32;

/// Max tile depth a joint key can carry: `2 · depth · 16 <= 128`.
pub const JOINT_MAX_DEPTH: u8 = 4;

/// Interleave the first `depth` semantic tiles of `a` and `b`
/// (`A0 B0 A1 B1 …`) into one `u128` joint key, always at width
/// [`JOINT_MAX_DEPTH`] (unused trailing tile-pairs are zero) so every joint
/// key in an index compares on the same bit positions regardless of the
/// query depth that will bound it. `depth` beyond `JOINT_MAX_DEPTH` is
/// clamped.
#[inline]
pub fn joint_key(a: FacetCascade, b: FacetCascade, depth: u8) -> u128 {
    let d = depth.min(JOINT_MAX_DEPTH);
    let (ta, tb) = (a.semantic_tiles(), b.semantic_tiles());
    let mut k = 0u128;
    for i in 0..JOINT_MAX_DEPTH as usize {
        let (av, bv) = if (i as u8) < d {
            (ta[i], tb[i])
        } else {
            (0u16, 0u16)
        };
        k = (k << 32) | ((av as u128) << 16) | bv as u128;
    }
    k
}

/// The same interleaving as [`joint_key`], but with every unfixed tile-pair
/// (position `>= depth`) set to the MAX `u16` on both sides — the joint
/// key's own `hi_key`, mirroring [`SemanticPrefix::hi_key`].
#[inline]
fn joint_key_hi(a: FacetCascade, b: FacetCascade, depth: u8) -> u128 {
    let d = depth.min(JOINT_MAX_DEPTH);
    let (ta, tb) = (a.semantic_tiles(), b.semantic_tiles());
    let mut k = 0u128;
    for i in 0..JOINT_MAX_DEPTH as usize {
        let (av, bv) = if (i as u8) < d {
            (ta[i], tb[i])
        } else {
            (0xFFFFu16, 0xFFFFu16)
        };
        k = (k << 32) | ((av as u128) << 16) | bv as u128;
    }
    k
}

/// One row of the joint-sorted index: the fixed-width joint key at
/// [`JOINT_MAX_DEPTH`] and the original row index (the ontology/tenant pair
/// at that index in `World`).
#[derive(Debug, Clone, Copy)]
struct JointRow {
    key: u128,
    row: u32,
}

/// The joint index: `(ontology, tenant)` pairs sorted by their
/// [`JOINT_MAX_DEPTH`]-wide joint key. Built ONCE per world (the equivalent
/// of `SealedFacetLane::seal` for the joint key) — a real, reported cost, not
/// hidden and not part of any timed fold.
pub struct JointIndex {
    rows: Vec<JointRow>,
}

impl JointIndex {
    /// Build the index and report the construction time (ns) separately —
    /// this is real one-shot cost, amortized only if the index is reused
    /// across many P3 queries at the same depth family.
    pub fn build(world: &World) -> (Self, f64) {
        let t0 = Instant::now();
        let mut rows: Vec<JointRow> = world
            .lane
            .keys()
            .iter()
            .zip(&world.tenant)
            .enumerate()
            .map(|(i, (&a, &b))| JointRow {
                key: joint_key(a, b, JOINT_MAX_DEPTH),
                row: i as u32,
            })
            .collect();
        rows.sort_unstable_by_key(|r| r.key);
        let build_ns = t0.elapsed().as_nanos() as f64;
        (JointIndex { rows }, build_ns)
    }

    /// **The fold arm.** Two `partition_point`s over the joint-sorted
    /// sequence, bounding the contiguous range whose joint key carries the
    /// first `depth` tile-pairs of `(a, b)`. No sweep, no per-row predicate —
    /// this is the entire timed cost of the fold.
    pub fn bound(&self, a: FacetCascade, b: FacetCascade, depth: u8) -> (u32, u32) {
        let lo = joint_key(a, b, depth);
        let hi = joint_key_hi(a, b, depth);
        let lo_idx = self.rows.partition_point(|r| r.key < lo);
        let hi_idx = self.rows.partition_point(|r| r.key <= hi);
        (lo_idx as u32, hi_idx as u32)
    }

    /// Materialize the original row indices in `[lo, hi)` of the
    /// joint-sorted sequence — done ONLY for oracle verification, never
    /// counted as part of the timed fold cost.
    pub fn materialize_rows(&self, lo: u32, hi: u32) -> Vec<usize> {
        self.rows[lo as usize..hi as usize]
            .iter()
            .map(|r| r.row as usize)
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct P3Row {
    pub depth: u8,
    pub kept_a: usize,
    pub kept_b: usize,
    pub kept_and: usize,
    /// The fold arm: ONE `JointIndex::bound` call. No sweep anywhere in it.
    pub fold_ns: f64,
    /// The non-fold reference comparator: sweep(A) + sweep(B) + AND, kept
    /// exactly as the original spec asked ("two sweeps + AND") — never
    /// labelled or treated as part of any fold's own lowering.
    pub reference_two_sweeps_ns: f64,
}

/// Find an `(A prefix, B prefix)` pair at equal `depth` whose intersection
/// passes F4, then verify the fold arm against the oracle intersection and
/// time both the fold and the reference comparator. `depth` must be
/// `<= JOINT_MAX_DEPTH` (asserted).
pub fn run_p3(world: &World, joint: &JointIndex, depth: u8, r: &mut SplitMix64) -> Option<P3Row> {
    assert!(
        depth <= JOINT_MAX_DEPTH,
        "run_p3: depth {depth} exceeds JOINT_MAX_DEPTH {JOINT_MAX_DEPTH}"
    );
    let n = world.n();
    let words = words_for(n);
    let lane = &*world.lane;
    let w = &world.witness;
    // Search for a non-trivial pair. The cheap witnessed bound gates the A
    // side before either oracle mask is built.
    let mut chosen = None;
    for _ in 0..20_000 {
        let i = (r.next_u64() % n as u64) as usize;
        let pa = SemanticPrefix::of(lane.keys()[i], depth);
        let pb = SemanticPrefix::of(world.tenant[i], depth);
        let (lo, hi) = lane.bound(w, &pa).expect("witnessed");
        if nontrivial((hi - lo) as usize, n).is_err() {
            continue;
        }
        let mb = oracle_mask(&world.tenant, &pb);
        if nontrivial(popcount(&mb), n).is_err() {
            continue;
        }
        let ma = oracle_mask(lane.keys(), &pa);
        let mut and = vec![0u64; words];
        mask_and(&ma, &mb, &mut and);
        if popcount(&and) >= INTERSECTION_FLOOR && nontrivial_intersection(&ma, &mb, &and).is_ok() {
            chosen = Some((lane.keys()[i], world.tenant[i], ma, mb, and));
            break;
        }
    }
    let (a, b, ma, mb, truth) = chosen?;

    // Verify the fold arm against the oracle intersection (row-set equality,
    // order-independent — the joint-sorted range need not preserve the
    // ontology ordinal's own order).
    let (lo, hi) = joint.bound(a, b, depth);
    let mut got: Vec<usize> = joint.materialize_rows(lo, hi);
    got.sort_unstable();
    let mut oracle_rows: Vec<usize> = (0..n)
        .filter(|&i| (truth[i / 64] >> (i % 64)) & 1 == 1)
        .collect();
    oracle_rows.sort_unstable();
    assert_eq!(got, oracle_rows, "fold arm row set != oracle intersection");

    // Reference comparator, verified against the same oracle masks.
    let words = words_for(n);
    let mut d1 = vec![0u64; words];
    let mut d2 = vec![0u64; words];
    let mut tmp = vec![0u64; words];
    let mut out = vec![0u64; words];
    let pa = SemanticPrefix::of(a, depth);
    let pb = SemanticPrefix::of(b, depth);
    sweep_mask(&world.a_hi, &world.a_lo, &pa, &mut d1, &mut tmp);
    assert_eq!(d1, ma);
    sweep_mask(&world.b_hi, &world.b_lo, &pb, &mut d2, &mut tmp);
    assert_eq!(d2, mb);
    mask_and(&d1, &d2, &mut out);
    assert_eq!(out, truth, "reference comparator != oracle");

    let fold_ns = time_ns(7, 1, || {
        let (lo, hi) = joint.bound(black_box(a), black_box(b), black_box(depth));
        black_box((lo, hi));
    });
    let reference_two_sweeps_ns = time_ns(7, 1, || {
        sweep_mask(
            black_box(&world.a_hi),
            black_box(&world.a_lo),
            black_box(&pa),
            &mut d1,
            &mut tmp,
        );
        sweep_mask(
            black_box(&world.b_hi),
            black_box(&world.b_lo),
            black_box(&pb),
            &mut d2,
            &mut tmp,
        );
        mask_and(&d1, &d2, &mut out);
        black_box(&out);
    });
    Some(P3Row {
        depth,
        kept_a: popcount(&ma),
        kept_b: popcount(&mb),
        kept_and: popcount(&truth),
        fold_ns,
        reference_two_sweeps_ns,
    })
}

/// The finding P3 records: the tenant lane, aligned to the ontology ordinal,
/// is not in numeric projection order, so it cannot be attested and its
/// prefix cannot be a witnessed bound over this ordinal.
pub fn tenant_attest_over_ontology_ordinal(world: &World) -> Result<SealedFacetLane, WitnessError> {
    SealedFacetLane::attest_sorted(world.tenant.clone(), world.lane.version())
}

// ─────────────────────────────────────────────────────────────────────────────
// P4 — async writer / sealed reader
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Default)]
pub struct WriterStats {
    pub seals: usize,
    pub appended: usize,
    pub sort_ns: Vec<f64>,
    pub attest_ns: Vec<f64>,
    pub publish_ns: Vec<f64>,
}

/// The published sealed image: readers pin an `Arc` (the `at(version)` path);
/// the writer swaps a new one in under a write lock.
pub type Published = Arc<RwLock<Arc<SealedFacetLane>>>;

/// Run `f` (the reader's measurement) while a writer appends out-of-order
/// keys to its own open image and re-seals every `batch` appends. Returns the
/// reader's result and the writer's costs.
pub fn with_open_writer<R>(
    published: Published,
    batch: usize,
    seed: u64,
    f: impl FnOnce() -> R,
) -> (R, WriterStats) {
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let stop_w = stop.clone();
    let pub_w = published.clone();
    let writer = std::thread::spawn(move || {
        let mut r = SplitMix64(seed);
        let mut open: Vec<FacetCascade> = pub_w.read().unwrap().keys().to_vec();
        let mut version = pub_w.read().unwrap().version();
        let mut stats = WriterStats::default();
        while !stop_w.load(std::sync::atomic::Ordering::Relaxed) {
            for _ in 0..batch {
                open.push(ontology_key(&mut r)); // arrival order: random, never sorted
            }
            stats.appended += batch;
            version += 1;
            let t0 = Instant::now();
            // Sort `open` IN PLACE (not a clone-then-sort): `open` must
            // still be independently owned after this iteration (the writer
            // keeps appending to it next loop), so a clone is structurally
            // required to hand an owned Vec to `attest_sorted` — that clone
            // cannot be eliminated. What CAN be eliminated: sorting a fresh
            // copy of the UNSORTED buffer every iteration. Because `open`
            // stays sorted from here on, each iteration's appended batch
            // lands as a short unsorted tail on an otherwise-sorted prefix,
            // which `sort_unstable_by`'s pattern-detecting (pdqsort) sort
            // handles cheaper than a full random-order sort of the whole
            // buffer — a real cost reduction on repeated seals, not merely a
            // reordering of the same two operations.
            open.sort_unstable_by(FacetCascade::cmp_numeric_projection);
            let t1 = Instant::now();
            let sealed = SealedFacetLane::attest_sorted(open.clone(), version).expect("sorted");
            let t2 = Instant::now();
            *pub_w.write().unwrap() = Arc::new(sealed);
            let t3 = Instant::now();
            stats.sort_ns.push((t1 - t0).as_nanos() as f64);
            stats.attest_ns.push((t2 - t1).as_nanos() as f64);
            stats.publish_ns.push((t3 - t2).as_nanos() as f64);
            stats.seals += 1;
        }
        stats
    });
    let out = f();
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    let stats = writer.join().unwrap();
    (out, stats)
}

/// Reader measurement: P1-style peek over `pairs` and P2-style bound over
/// `prefixes`, both against the pinned sealed lane.
/// min / median / p90 of per-round timings.
#[derive(Debug, Clone, Copy, Default)]
pub struct Dist {
    pub min: f64,
    pub median: f64,
    pub p90: f64,
    pub rounds: usize,
}

impl Dist {
    pub fn of(mut v: Vec<f64>) -> Dist {
        if v.is_empty() {
            return Dist::default();
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Dist {
            min: v[0],
            median: v[v.len() / 2],
            p90: v[(v.len() * 9 / 10).min(v.len() - 1)],
            rounds: v.len(),
        }
    }
}

/// Reader measurement against the PINNED sealed lane: P1-style peek over
/// `pairs` and P2-style bound over `prefixes`, repeated for at least `for_ms`
/// wall-clock milliseconds so an open writer has time to seal and publish
/// several times underneath. Returns per-pair and per-prefix distributions
/// over the rounds — a min would HIDE a perturbation, so P4 is judged on the
/// median and p90.
pub fn reader_measure(
    lane: &SealedFacetLane,
    w: &OrderedLaneWitness,
    pairs: &[(FacetCascade, FacetCascade)],
    prefixes: &[SemanticPrefix],
    for_ms: u64,
) -> (Dist, Dist) {
    let (mut peeks, mut bounds) = (Vec::new(), Vec::new());
    let start = Instant::now();
    while start.elapsed().as_millis() < u128::from(for_ms) || peeks.len() < 7 {
        let t0 = Instant::now();
        let mut acc = 0u32;
        for (a, b) in pairs {
            acc += lcp_peek_bytes(black_box(a), black_box(b)) as u32;
        }
        black_box(acc);
        peeks.push(t0.elapsed().as_nanos() as f64 / pairs.len() as f64);
        let t1 = Instant::now();
        let mut acc = 0u64;
        for p in prefixes {
            let (lo, hi) = lane.bound(black_box(w), black_box(p)).unwrap();
            acc += (hi - lo) as u64;
        }
        black_box(acc);
        bounds.push(t1.elapsed().as_nanos() as f64 / prefixes.len() as f64);
    }
    (Dist::of(peeks), Dist::of(bounds))
}
