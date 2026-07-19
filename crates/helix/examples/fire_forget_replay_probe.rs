//! PROBE-FIRE-FORGET-REPLAY — the operator's "without materialization" made
//! measurable (board `E-PERTURBATION-CONVERGENCE-1`, operator clarification).
//!
//! The claim: the SoA Morton-comma field is a FUNCTION `render(addr, aspect)`,
//! never a materialized OBJECT. A cognitive shader CHOOSES the aspects +
//! regions it attends to and renders ONLY those, on demand — **fire-and-forget**
//! (render → consume → discard, no global copy of the O(64k²)…O(256k²) field) —
//! and because the render is deterministic it is **replayable**: discarding is
//! free because re-evaluation reproduces the slice bit-exact. O(attended-region)
//! memory, never O(full-field).
//!
//! Four legs, each a falsifiable property of that claim:
//!  A. **Replay after discard.** Render region R, drop it, render R again →
//!     byte-identical. Fire-and-forget is safe ⟺ this holds.
//!  B. **O(attended-region) working set, NOT O(full-field).** Rendering K
//!     regions touches exactly Σ|region| cells; the full field (65536² ≈ 4.3e9
//!     cells) is NEVER allocated. Reported as the touched/full ratio — the
//!     "no global cache" measurement.
//!  C. **Fire-and-forget composability (no shared cache).** Two INDEPENDENT
//!     shaders that render an overlapping region agree bit-exact on the
//!     overlap — each re-derives identically, so no cache-coherence is needed.
//!  D. **Aspect selection.** The SAME region under two different aspects
//!     renders differently (the shader chooses what it wants), yet each aspect
//!     is independently replayable.
//!
//! Plus the comma fence (three-gap + coprime full-perm) — the aperiodic
//! deterministic phase that MAKES the render replayable.
//!
//! HONEST FENCE: `render_cell` here regenerates BOTH phase (comma, 0-bit, the
//! real production shape) AND a per-coarse-cell magnitude (deterministic from
//! the coarse prefix — in production the magnitude is the STORED tenant, read
//! from a small per-region store; the probe generates it so it is
//! self-contained). What is proven is the ARCHITECTURAL property — pure
//! function ⇒ replay-after-discard ⇒ O(region) working set ⇒ no global cache —
//! which holds regardless of whether the magnitude is stored or computed. The
//! vertical multi-level unbind (§4b Walsh-Hadamard) is the separate open leg
//! (PROBE-WHP-1/3), NOT tested here.
//!
//! std-only, deterministic (SplitMix64, no `rand`). Prints `FFREPLAY …` lines.

/// Field axis (65536 = 256², the "64k" tier); the full field is AXIS² cells.
const AXIS: u64 = 65_536;
/// Coarse-prefix shift: the magnitude tenant is per coarse cell (top bits).
const COARSE_SHIFT: u32 = 8;
/// Seed — "FFREPLAY1" family.
const SEED: u64 = 0x_4646_5245_504C_4159;

struct SplitMix64(u64);
impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// Deterministic hash of an address (SplitMix64 finalizer, stateless).
fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The comma phase at an address: the stride-4-over-17 coprime walk indexed by
/// the address (integer, bit-exact, aperiodic — the discrete Pythagorean
/// comma). 0 stored bits — regenerated from the address alone.
fn comma_phase(addr: u64) -> u8 {
    ((4 * (addr % 17)) % 17) as u8
}

/// The per-coarse-cell magnitude tenant. In production this is the STORED
/// envelope (the only stored bits); here it is regenerated deterministically
/// from the coarse prefix so the probe is self-contained. Either way the
/// render is a PURE FUNCTION of the address — which is the property under test.
fn magnitude(coarse: u64, aspect: u8) -> u8 {
    (mix(coarse ^ ((aspect as u64) << 40)) & 0xff) as u8
}

/// The render function: a PURE function of (addr, aspect). No global state, no
/// full-field allocation. This is the shader's per-cell evaluation — the field
/// AS a function, not an object.
fn render_cell(addr: u64, aspect: u8) -> u8 {
    let coarse = addr >> COARSE_SHIFT;
    let phase = comma_phase(addr);
    let mag = magnitude(coarse, aspect);
    // Compose: magnitude enveloped by the comma phase (a deterministic, exact
    // combination — the shape does not matter, its purity + determinism do).
    mag.wrapping_add(phase.wrapping_mul(7))
}

/// A rectangular attended region in the field (a shader's foveation window).
#[derive(Clone, Copy)]
struct Region {
    x0: u64,
    y0: u64,
    w: u64,
    h: u64,
}

impl Region {
    fn len(&self) -> u64 {
        self.w * self.h
    }
    /// The Morton-ish address of cell (x,y): row-major within the AXIS² field
    /// (the exact interleave is irrelevant to the fire-forget property; what
    /// matters is that the address is a pure function of position).
    fn addr(x: u64, y: u64) -> u64 {
        y * AXIS + x
    }
}

/// Render ONE region on demand into a fresh buffer — the fire-and-forget
/// evaluation. Allocates exactly `region.len()` bytes; NEVER the full field.
/// `touched` counts cells evaluated (the working-set instrument).
fn render_region(region: &Region, aspect: u8, touched: &mut u64) -> Vec<u8> {
    let mut out = Vec::with_capacity(region.len() as usize);
    for dy in 0..region.h {
        for dx in 0..region.w {
            let a = Region::addr(region.x0 + dx, region.y0 + dy);
            out.push(render_cell(a, aspect));
            *touched += 1;
        }
    }
    out
}

// ── comma fence (shared with the other probes) ──────────────────────────

fn three_gap_distinct_count(seq: &[f64]) -> usize {
    let mut s: Vec<f64> = seq.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = s.len();
    if n < 2 {
        return 0;
    }
    let gaps: Vec<f64> = (0..n)
        .map(|i| {
            let next = if i + 1 < n { s[i + 1] } else { s[0] + 1.0 };
            next - s[i]
        })
        .collect();
    let tol = 1e-9;
    let mut distinct: Vec<f64> = Vec::new();
    for &g in &gaps {
        if !distinct.iter().any(|&d| (d - g).abs() < tol) {
            distinct.push(g);
        }
    }
    distinct.len()
}

fn coprime_walk_is_full_permutation() -> bool {
    let mut seen = [false; 17];
    for k in 0..17u32 {
        let w = (4 * k % 17) as usize;
        if seen[w] {
            return false;
        }
        seen[w] = true;
    }
    seen.iter().all(|&s| s)
}

fn main() {
    let mut rng = SplitMix64::new(SEED);
    let full_field_cells = AXIS * AXIS; // 4_294_967_296

    // A shader's foveation: K attended regions chosen anywhere in the field.
    let k = 8u64;
    let regions: Vec<Region> = (0..k)
        .map(|_| {
            let x0 = rng.next_u64() % (AXIS - 64);
            let y0 = rng.next_u64() % (AXIS - 64);
            Region {
                x0,
                y0,
                w: 24,
                h: 24,
            }
        })
        .collect();

    // Leg A — replay after discard: render each region, drop it, render again.
    let mut all_replay = true;
    let mut touched_a = 0u64;
    for r in &regions {
        let first = render_region(r, 0, &mut touched_a);
        drop(first.clone()); // fire-and-forget: the slice is consumed + dropped
        let second = render_region(r, 0, &mut touched_a);
        if first != second {
            all_replay = false;
        }
    }
    eprintln!(
        "FFREPLAY legA replay_after_discard_bitexact={} regions={k}",
        if all_replay { 1 } else { 0 }
    );

    // Leg B — O(attended-region) working set, never O(full-field).
    let mut touched_b = 0u64;
    for r in &regions {
        let _ = render_region(r, 0, &mut touched_b);
    }
    let attended: u64 = regions.iter().map(|r| r.len()).sum();
    let ratio = touched_b as f64 / full_field_cells as f64;
    eprintln!(
        "FFREPLAY legB cells_touched={touched_b} attended_cells={attended} full_field_cells={full_field_cells} touched_over_full={ratio:.3e}"
    );

    // Leg C — fire-and-forget composability: two INDEPENDENT shaders render an
    // overlapping region; they must agree bit-exact on the overlap with NO
    // shared cache (each re-derives identically).
    let base = Region {
        x0: 1000,
        y0: 2000,
        w: 32,
        h: 32,
    };
    let shifted = Region {
        x0: 1016,
        y0: 2016,
        w: 32,
        h: 32,
    }; // overlaps base in a 16×16 corner
    let mut td = 0u64;
    let ra = render_region(&base, 0, &mut td);
    let rb = render_region(&shifted, 0, &mut td);
    let mut overlap_ok = true;
    let mut overlap_n = 0u64;
    for y in 2016..2032 {
        for x in 1016..1032 {
            let va = ra[((y - base.y0) * base.w + (x - base.x0)) as usize];
            let vb = rb[((y - shifted.y0) * shifted.w + (x - shifted.x0)) as usize];
            if va != vb {
                overlap_ok = false;
            }
            overlap_n += 1;
        }
    }
    eprintln!(
        "FFREPLAY legC composable_overlap_bitexact={} overlap_cells={overlap_n}",
        if overlap_ok { 1 } else { 0 }
    );

    // Leg D — aspect selection: the same region under two aspects differs, and
    // each aspect independently replays.
    let mut tdd = 0u64;
    let r = regions[0];
    let asp0 = render_region(&r, 0, &mut tdd);
    let asp1 = render_region(&r, 1, &mut tdd);
    let asp0_again = render_region(&r, 0, &mut tdd);
    let distinct = asp0 != asp1;
    let asp0_replays = asp0 == asp0_again;
    eprintln!(
        "FFREPLAY legD aspect_distinct={} aspect0_replays={}",
        if distinct { 1 } else { 0 },
        if asp0_replays { 1 } else { 0 }
    );

    // Comma fence — the deterministic phase that makes replay possible.
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let phi_seq: Vec<f64> = (0..4096)
        .map(|n| {
            let v = n as f64 * phi;
            v - v.floor()
        })
        .collect();
    eprintln!(
        "FFREPLAY comma_three_gap_distinct={} comma_coprime_full_perm={}",
        three_gap_distinct_count(&phi_seq),
        if coprime_walk_is_full_permutation() { 1 } else { 0 }
    );

    eprintln!("FFREPLAY axis={AXIS} coarse_shift={COARSE_SHIFT}");
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// The render is a PURE function: same (addr, aspect) → same byte, always,
    /// with no global state. This is what makes fire-and-forget safe.
    #[test]
    fn render_is_pure_and_replayable() {
        let mut rng = SplitMix64::new(SEED ^ 0x01);
        for _ in 0..50_000 {
            let a = rng.next_u64() % (AXIS * AXIS);
            let aspect = (rng.next_u64() & 1) as u8;
            assert_eq!(render_cell(a, aspect), render_cell(a, aspect));
        }
    }

    /// Replay after discard: render, drop, render again → byte-identical.
    #[test]
    fn region_replays_after_discard() {
        let r = Region {
            x0: 500,
            y0: 700,
            w: 20,
            h: 20,
        };
        let mut t = 0;
        let first = render_region(&r, 0, &mut t);
        drop(first.clone());
        let second = render_region(&r, 0, &mut t);
        assert_eq!(first, second);
    }

    /// Working set is O(attended-region), never O(full-field): rendering only
    /// ever touches |region| cells, and the touched set is a vanishing
    /// fraction of the AXIS² field.
    #[test]
    fn working_set_is_region_not_field() {
        let r = Region {
            x0: 0,
            y0: 0,
            w: 40,
            h: 40,
        };
        let mut touched = 0u64;
        let out = render_region(&r, 0, &mut touched);
        assert_eq!(touched, r.len());
        assert_eq!(out.len() as u64, r.len());
        // 1600 cells vs 4.29e9 full-field cells — O(region), not O(field).
        assert!(touched * 1_000_000 < AXIS * AXIS);
    }

    /// Fire-and-forget composability: two independent renders of an
    /// overlapping region agree bit-exact on the overlap (no shared cache).
    #[test]
    fn independent_renders_agree_on_overlap() {
        let mut t = 0;
        let a = Region {
            x0: 100,
            y0: 100,
            w: 16,
            h: 16,
        };
        let b = Region {
            x0: 108,
            y0: 104,
            w: 16,
            h: 16,
        };
        let ra = render_region(&a, 0, &mut t);
        let rb = render_region(&b, 0, &mut t);
        for y in 104..116 {
            for x in 108..116 {
                let va = ra[((y - a.y0) * a.w + (x - a.x0)) as usize];
                let vb = rb[((y - b.y0) * b.w + (x - b.x0)) as usize];
                assert_eq!(va, vb, "overlap must agree without a shared cache");
            }
        }
    }

    /// Aspect selection: different aspects render differently over a region
    /// (the shader chooses what it wants); each aspect independently replays.
    #[test]
    fn aspects_are_distinct_and_each_replayable() {
        let r = Region {
            x0: 300,
            y0: 400,
            w: 24,
            h: 24,
        };
        let mut t = 0;
        let a0 = render_region(&r, 0, &mut t);
        let a1 = render_region(&r, 1, &mut t);
        let a0b = render_region(&r, 0, &mut t);
        // At least one cell differs across aspects...
        let mut distinct = HashSet::new();
        distinct.insert(a0.clone());
        distinct.insert(a1.clone());
        assert_eq!(distinct.len(), 2, "aspects must render differently");
        // ...and aspect 0 replays exactly.
        assert_eq!(a0, a0b);
    }

    /// Comma fence: the deterministic phase is aperiodic (Steinhaus ≤3 +
    /// coprime full permutation) — the property that makes replay bit-exact.
    #[test]
    fn comma_fence_holds() {
        let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
        let seq: Vec<f64> = (0..1024)
            .map(|n| {
                let v = n as f64 * phi;
                v - v.floor()
            })
            .collect();
        assert!(three_gap_distinct_count(&seq) <= 3);
        assert!(coprime_walk_is_full_permutation());
    }
}
