//! 1BRC monoid-aggregation certification — diagnostic probe (not a numbered
//! pillar; same status as `sigma_codebook_probe`).
//!
//! Certifies the algebra that `ndarray/examples/onebrc_cascade_probe.rs`
//! (branch `claude/1brc-lance-graph-xfx5tu`) relies on — kernels there,
//! proof here, per the architecture rule (ndarray = hardware, jc = proof).
//! Zero-dep by the jc constitution: the proof is the proof regardless of
//! SIMD path.
//!
//! Four claims:
//!
//! 1. **Partition invariance** — the `(min, max, Σ, n)` group-by is a
//!    commutative monoid fold, so morsel-batched aggregation (any batch
//!    sizes, any merge order) equals the sequential fold EXACTLY. This is
//!    the correctness core of the Morton scatter path. Checked for the i64
//!    integer sums AND the f64 sums (integer tenths: every partial sum is an
//!    exact f64 integer < 2^53, so f64 merge order cannot matter — compared
//!    by bit pattern, not tolerance).
//! 2. **Regroup invariance** — folding per-station aggregates through an
//!    arbitrary binary tree (the Morton aggregate pyramid is one such tree)
//!    equals the linear fold exactly. Same monoid argument, one level up.
//! 3. **BF16 hi/lo decomposition exactness** — for EVERY integer-tenths
//!    temperature t ∈ [-999, 999]: `hi = (t/256)·256` and `lo = t − hi`
//!    both survive a bf16 round-trip unchanged, and `hi + lo == t`.
//!    Exhaustive over all 1999 values — proof by enumeration, not sampling.
//!    This is what makes the AMX TDPBF16PS group-by leg exact.
//! 4. **BF16-direct quantization bound** — `|bf16_rne(t) − t| ≤ 2` tenths
//!    over the same exhaustive range: the bf16 ulp at |t| ∈ [512, 1024) is
//!    4 tenths, and round-to-nearest-even errs by at most HALF an ulp = 2
//!    tenths, with equality attained. I.e. skipping the hi/lo split costs
//!    up to 0.2 °C on individual readings, while remaining unbiased enough
//!    that measured per-station mean error at N≈24k samples was 0.0123
//!    tenths (see the ndarray probe's bf16-direct row).

use crate::PillarResult;

const SEED: u64 = 0x1BC_0FFEE;
const N_ROWS: usize = 100_000;
const N_STATIONS: usize = 413;
const N_TRIALS: usize = 20;

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

// ── The monoid ──────────────────────────────────────────────────────────────

/// The 1BRC aggregation element. `sum_f` mirrors `sum_t` in f64 to certify
/// that the floating-point sums the Morton SoA carries are order-independent
/// for integer tenths (bit-compared, never tolerance-compared).
#[derive(Clone, Copy, PartialEq, Debug)]
struct Agg {
    min_t: i16,
    max_t: i16,
    sum_t: i64,
    sum_f: f64,
    cnt: u64,
}

impl Agg {
    const IDENTITY: Agg = Agg {
        min_t: i16::MAX,
        max_t: i16::MIN,
        sum_t: 0,
        sum_f: 0.0,
        cnt: 0,
    };

    fn fold(mut self, t: i16) -> Agg {
        self.min_t = self.min_t.min(t);
        self.max_t = self.max_t.max(t);
        self.sum_t += t as i64;
        self.sum_f += t as f64;
        self.cnt += 1;
        self
    }

    fn merge(mut self, o: Agg) -> Agg {
        self.min_t = self.min_t.min(o.min_t);
        self.max_t = self.max_t.max(o.max_t);
        self.sum_t += o.sum_t;
        self.sum_f += o.sum_f;
        self.cnt += o.cnt;
        self
    }

    /// Bit-exact equality: f64 compared by bit pattern (an exact-integer sum
    /// must be the SAME exact integer regardless of fold shape).
    fn bit_eq(&self, o: &Agg) -> bool {
        self.min_t == o.min_t
            && self.max_t == o.max_t
            && self.sum_t == o.sum_t
            && self.sum_f.to_bits() == o.sum_f.to_bits()
            && self.cnt == o.cnt
    }
}

// ── BF16 encode/decode (zero-dep local mirrors) ─────────────────────────────

/// f32 → bf16 round-to-nearest-even (the hardware conversion).
fn bf16_rne(v: f32) -> u16 {
    let bits = v.to_bits();
    ((bits.wrapping_add(0x7FFF).wrapping_add((bits >> 16) & 1)) >> 16) as u16
}

fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

// ── The probe ───────────────────────────────────────────────────────────────

pub fn prove() -> PillarResult {
    let start = std::time::Instant::now();
    let mut st = SEED;
    let mut invariance_ok = 0u64;
    let mut invariance_total = 0u64;

    for _trial in 0..N_TRIALS {
        // Deterministic 1BRC-shaped rows.
        let rows: Vec<(usize, i16)> = (0..N_ROWS)
            .map(|_| {
                let sid = (splitmix64(&mut st) % N_STATIONS as u64) as usize;
                let t = (splitmix64(&mut st) % 1999) as i16 - 999;
                (sid, t)
            })
            .collect();

        // Ground truth: sequential fold.
        let mut seq = vec![Agg::IDENTITY; N_STATIONS];
        for &(sid, t) in &rows {
            seq[sid] = seq[sid].fold(t);
        }

        // Claim 1a: random morsel partition, merged in shuffled order.
        let mut cuts: Vec<usize> = (0..8).map(|_| (splitmix64(&mut st) % N_ROWS as u64) as usize).collect();
        cuts.push(0);
        cuts.push(N_ROWS);
        cuts.sort_unstable();
        cuts.dedup();
        let mut morsels: Vec<Vec<Agg>> = cuts
            .windows(2)
            .map(|w| {
                let mut acc = vec![Agg::IDENTITY; N_STATIONS];
                for &(sid, t) in &rows[w[0]..w[1]] {
                    acc[sid] = acc[sid].fold(t);
                }
                acc
            })
            .collect();
        // Fisher-Yates on the morsel merge order.
        for i in (1..morsels.len()).rev() {
            let j = (splitmix64(&mut st) % (i as u64 + 1)) as usize;
            morsels.swap(i, j);
        }
        let mut part = vec![Agg::IDENTITY; N_STATIONS];
        for m in &morsels {
            for s in 0..N_STATIONS {
                part[s] = part[s].merge(m[s]);
            }
        }
        invariance_total += 1;
        if (0..N_STATIONS).all(|s| part[s].bit_eq(&seq[s])) {
            invariance_ok += 1;
        }

        // Claim 1b: full row permutation.
        let mut perm = rows.clone();
        for i in (1..perm.len()).rev() {
            let j = (splitmix64(&mut st) % (i as u64 + 1)) as usize;
            perm.swap(i, j);
        }
        let mut per = vec![Agg::IDENTITY; N_STATIONS];
        for &(sid, t) in &perm {
            per[sid] = per[sid].fold(t);
        }
        invariance_total += 1;
        if (0..N_STATIONS).all(|s| per[s].bit_eq(&seq[s])) {
            invariance_ok += 1;
        }

        // Claim 2: regroup — fold the per-station aggregates through a random
        // binary tree (repeatedly merge two random nodes) vs the linear fold.
        let linear = seq.iter().fold(Agg::IDENTITY, |a, &b| a.merge(b));
        let mut nodes: Vec<Agg> = seq.clone();
        while nodes.len() > 1 {
            let i = (splitmix64(&mut st) % nodes.len() as u64) as usize;
            let a = nodes.swap_remove(i);
            let j = (splitmix64(&mut st) % nodes.len() as u64) as usize;
            let b = nodes.swap_remove(j);
            nodes.push(a.merge(b));
        }
        invariance_total += 1;
        if nodes[0].bit_eq(&linear) {
            invariance_ok += 1;
        }
    }

    // Claims 3 + 4: exhaustive over every integer-tenths temperature.
    let mut hi_lo_exact = true;
    let mut max_direct_err = 0.0f32;
    for t in -999i32..=999 {
        let hi = (t / 256) * 256;
        let lo = t - hi;
        let hi_rt = bf16_to_f32(bf16_rne(hi as f32));
        let lo_rt = bf16_to_f32(bf16_rne(lo as f32));
        if hi_rt != hi as f32 || lo_rt != lo as f32 || hi + lo != t {
            hi_lo_exact = false;
        }
        let direct = bf16_to_f32(bf16_rne(t as f32));
        max_direct_err = max_direct_err.max((direct - t as f32).abs());
    }

    let invariance_rate = invariance_ok as f64 / invariance_total as f64;
    let pass = invariance_rate == 1.0 && hi_lo_exact && max_direct_err <= 2.0;

    PillarResult {
        name: "1BRC-MONOID: partition/regroup invariance + bf16 hi/lo exactness",
        pass,
        measured: max_direct_err as f64,
        predicted: 2.0, // half-ulp of bf16 at |t| ∈ [512, 1024) under RNE, in tenths
        detail: format!(
            "invariance {invariance_ok}/{invariance_total} bit-exact ({N_TRIALS} trials × \
             {{morsel-partition, permutation, regroup-tree}}, {N_ROWS} rows, {N_STATIONS} \
             stations, f64 sums compared by BIT PATTERN); hi/lo bf16 round-trip exact for \
             all 1999 tenths: {hi_lo_exact}; bf16-direct max quantization error \
             {max_direct_err} tenths (half-ulp bound 2, attained). Consumed by \
             ndarray/examples/onebrc_cascade_probe.rs (Morton scatter + AMX TDPBF16PS legs)."
        ),
        runtime_ms: start.elapsed().as_millis() as u64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_passes() {
        let r = prove();
        assert!(r.pass, "{}", r.detail);
    }

    #[test]
    fn bf16_helpers_agree_with_known_values() {
        assert_eq!(bf16_rne(1.0), 0x3F80);
        assert_eq!(bf16_to_f32(0x3F80), 1.0);
        // 999 is NOT bf16-exact (needs 10 significand bits) — rounds to 1000.
        assert_eq!(bf16_to_f32(bf16_rne(999.0)), 1000.0);
        // 768 and 255 (the hi/lo extremes) ARE exact.
        assert_eq!(bf16_to_f32(bf16_rne(768.0)), 768.0);
        assert_eq!(bf16_to_f32(bf16_rne(-255.0)), -255.0);
    }
}
