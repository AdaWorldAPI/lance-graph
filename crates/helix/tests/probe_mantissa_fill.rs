//! PROBE-MANTISSA-FILL + PROBE-PHASE-1 — Wave-0 probes against shipped code.
//!
//! Per `OGAR/docs/INTEGRATION-TEST-PLAN.md` §1 (the probe-first rule: no
//! integration brick lands before its probe is green) and the
//! volumetric-field-edge proposal (implicit centroids placed by the golden
//! mantissa, pairwise-weighted by the 256×256 attention LUT, splat-ranked).
//!
//! ## PROBE-MANTISSA-FILL
//!
//! Question: does the shipped golden-mantissa generator
//! ([`HemispherePoint::lift`] — azimuth `n·φ`, equal-area `r = √u`) place k
//! implicit centroids over a 256×256 tile MORE uniformly than seeded
//! uniform-random placement on the same support (the unit disk)?
//!
//! Metric (discrepancy proxy, 16×16 binning over the tile, in-disk bins only):
//!   - `occupied`: distinct in-disk bins hit (higher = better coverage)
//!   - `max_bin`:  worst-case pile-up (lower = better spread)
//!
//! PASS: at k ∈ {256, 1024}, golden beats EVERY one of three independently
//! seeded uniform-random baselines on BOTH metrics (no cherry-picked seed).
//!
//! KILL (per the proposal's kill-condition): golden loses → the "golden
//! mantissa places implicit centroids" leg falls back to an explicit grid.
//!
//! ## PROBE-PHASE-1 (regeneration determinism)
//!
//! Question: is the deterministic-phase generator bit-exact — same address ⟹
//! same sequence, across independent constructions? ([`CurveRuler`] is the
//! D-QUANTGATE-mandated coprime-integer walk; a float recurrence could drift,
//! an integer walk must not.)
//!
//! PASS: two independent `CurveRuler`s from the same `(path, depth)` produce
//! identical full arcs; the arc is a full permutation of 0..17; and the
//! permutation property holds for every one of the 17 possible offsets.

use helix::{CurveRuler, HemispherePoint};

const TILE: usize = 256;
const BINS: usize = 16; // 16×16 bins over the 256×256 tile
const BIN_W: usize = TILE / BINS;

/// xorshift64 — the workspace's zero-dep seeded RNG test pattern.
struct XorShift64(u64);
impl XorShift64 {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    /// Uniform f64 in [0, 1).
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Map a unit-disk point (x, z ∈ [-1, 1]) to a tile bin index, or None if the
/// pixel falls outside the 256×256 tile after scaling.
fn disk_to_bin(x: f64, z: f64) -> Option<usize> {
    let ux = (x + 1.0) / 2.0;
    let uz = (z + 1.0) / 2.0;
    if !(0.0..1.0).contains(&ux) || !(0.0..1.0).contains(&uz) {
        return None;
    }
    let px = (ux * TILE as f64) as usize;
    let pz = (uz * TILE as f64) as usize;
    Some((pz / BIN_W) * BINS + (px / BIN_W))
}

/// Whether a bin's center lies inside the unit disk (both generators share
/// the disk as support; corner bins are unreachable for both, so the metric
/// only counts bins both COULD hit).
fn bin_in_disk(bin: usize) -> bool {
    let bx = (bin % BINS) as f64;
    let bz = (bin / BINS) as f64;
    let cx = (bx + 0.5) / BINS as f64 * 2.0 - 1.0;
    let cz = (bz + 0.5) / BINS as f64 * 2.0 - 1.0;
    cx * cx + cz * cz <= 1.0
}

/// (occupied in-disk bins, max single-bin count) for a set of disk points.
fn fill_metrics(points: impl Iterator<Item = (f64, f64)>) -> (usize, usize) {
    let mut counts = [0usize; BINS * BINS];
    for (x, z) in points {
        if let Some(b) = disk_to_bin(x, z) {
            counts[b] += 1;
        }
    }
    let occupied = (0..BINS * BINS)
        .filter(|&b| bin_in_disk(b) && counts[b] > 0)
        .count();
    let max_bin = counts.iter().copied().max().unwrap_or(0);
    (occupied, max_bin)
}

fn golden_points(k: usize) -> Vec<(f64, f64)> {
    (0..k)
        .map(|n| {
            let p = HemispherePoint::lift(n, k);
            let (x, z, _y) = p.cartesian();
            (x, z)
        })
        .collect()
}

fn random_disk_points(k: usize, seed: u64) -> Vec<(f64, f64)> {
    // Rejection-sample uniform points in the unit disk (same support as the
    // golden generator) so the comparison is geometry-fair.
    let mut rng = XorShift64(seed);
    let mut out = Vec::with_capacity(k);
    while out.len() < k {
        let x = rng.unit() * 2.0 - 1.0;
        let z = rng.unit() * 2.0 - 1.0;
        if x * x + z * z < 1.0 {
            out.push((x, z));
        }
    }
    out
}

#[test]
fn probe_mantissa_fill_golden_beats_uniform_random() {
    // Three independent baseline seeds — golden must beat ALL of them on
    // BOTH metrics at BOTH sample counts; no cherry-picking.
    const SEEDS: [u64; 3] = [0x9E37_79B9_7F4A_7C15, 0xD1B5_4A32_D192_ED03, 0x2545_F491_4F6C_DD1D];

    for &k in &[256usize, 1024] {
        let (g_occ, g_max) = fill_metrics(golden_points(k).into_iter());
        for &seed in &SEEDS {
            let (r_occ, r_max) = fill_metrics(random_disk_points(k, seed).into_iter());
            assert!(
                g_occ >= r_occ,
                "k={k} seed={seed:#x}: golden occupied {g_occ} < random {r_occ} — \
                 MANTISSA-FILL RED: golden mantissa does not out-cover uniform random"
            );
            assert!(
                g_max <= r_max,
                "k={k} seed={seed:#x}: golden max-bin {g_max} > random {r_max} — \
                 MANTISSA-FILL RED: golden mantissa piles up worse than uniform random"
            );
        }
        // Print the receipt numbers so the probe run is quotable.
        println!("MANTISSA-FILL k={k}: golden occupied={g_occ} max_bin={g_max}");
        for &seed in &SEEDS {
            let (r_occ, r_max) = fill_metrics(random_disk_points(k, seed).into_iter());
            println!("  random seed={seed:#x}: occupied={r_occ} max_bin={r_max}");
        }
    }
}

#[test]
fn probe_mantissa_fill_no_empty_inner_region_at_1024() {
    // Stronger coverage claim at k=1024: every in-disk bin whose center is
    // comfortably interior (radius ≤ 0.9) must be occupied by golden points.
    let mut counts = [0usize; BINS * BINS];
    for (x, z) in golden_points(1024) {
        if let Some(b) = disk_to_bin(x, z) {
            counts[b] += 1;
        }
    }
    for b in 0..BINS * BINS {
        let bx = (b % BINS) as f64;
        let bz = (b / BINS) as f64;
        let cx = (bx + 0.5) / BINS as f64 * 2.0 - 1.0;
        let cz = (bz + 0.5) / BINS as f64 * 2.0 - 1.0;
        if cx * cx + cz * cz <= 0.81 {
            assert!(
                counts[b] > 0,
                "interior bin {b} (center {cx:.2},{cz:.2}) EMPTY at k=1024 — \
                 golden mantissa leaves interior holes"
            );
        }
    }
}

#[test]
fn probe_phase1_curve_ruler_regeneration_is_bit_exact() {
    // Same address ⟹ same sequence, across independent constructions.
    for path in [0u64, 1, 0x1234, u64::MAX, 0xDEAD_BEEF_CAFE_F00D] {
        for depth in [0u8, 1, 7, 16] {
            let a = CurveRuler::from_hhtl(path, depth);
            let b = CurveRuler::from_hhtl(path, depth);
            assert_eq!(a.arc(), b.arc(), "regeneration drift at ({path:#x},{depth})");
        }
    }
}

#[test]
fn probe_phase1_full_permutation_for_every_offset() {
    // The stride-4-over-17 arc must be a full permutation of 0..17 from every
    // possible start offset (coprimality must hold everywhere, not just at 0).
    for place in 0u64..17 {
        let ruler = CurveRuler::from_place(place);
        let arc = ruler.arc();
        let mut seen = [false; 17];
        for &v in &arc {
            assert!(v < 17, "residue {v} out of range");
            assert!(!seen[v as usize], "residue {v} repeated at place {place}");
            seen[v as usize] = true;
        }
        assert!(seen.iter().all(|&s| s), "incomplete permutation at place {place}");
    }
}
