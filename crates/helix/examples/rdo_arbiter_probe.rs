//! PROBE-RDO-ARBITER — turning "self-optimizing" into a measurement
//! (the [H] rung toward the space-time-aware-graph endgame; follow-on to
//! `E-X265-MORTON-SHIFT-1`).
//!
//! The claim: a free-energy / RDO **mode arbiter**, given a MortonShift mode
//! (motion = address delta, cheap for rigid translation) and a Residual mode
//! (code the change directly, no motion model), picks the cheaper mode
//! **per region** by DETECTING where rigid motion applies — and the adaptive
//! composite beats a fixed single-mode pipeline (what x265 lacks the address
//! substrate to do) even after paying the per-region mode-signal overhead.
//!
//! Why this is not a tautology (`argmin` is trivially ≤ each option): the
//! non-trivial content is (a) the arbiter must DETECT applicability from the
//! content — a real rigid-match search, not a hardcoded label — and its
//! choice must FLIP when the content flips; (b) the net win over the fixed
//! baseline must survive the mode-signal overhead (1 byte/region); (c) the
//! savings must come specifically from the rigid regions where the 2-byte
//! address delta replaces a large motion-less residual. This is the
//! CLAUDE.md free-energy loop made concrete: find the model that explains
//! the region with least surprise (bits); if none matches, escalate to the
//! residual path.
//!
//! HONEST FENCE: the residual RATE is a `bits/px` MODEL, not a real entropy
//! coder (the #1 parity gap named in this session's parity write-up). Leg 1
//! measures the ARBITRATION LOGIC — correct content-adaptive dispatch + net
//! win over a fixed pipeline. A real entropy-coded residual + a λ-swept
//! rate-distortion frontier (the full RDO test) is the named v2.
//!
//! std-only, deterministic (SplitMix64, no `rand`). Prints `RDOARB …` lines.

/// Region tile size (16×16 = 256 px).
const RW: usize = 16;
const RH: usize = 16;
/// Grid of regions (8×8 = 64 regions per frame).
const NX: usize = 8;
const NY: usize = 8;
/// Rigid-match search window (leading-edge, non-negative deltas 0..=MAXD).
const MAXD: usize = 4;
/// Residual cost model: bits per coded pixel (a STATED approximation —
/// a real entropy coder is v2). 4 bits/px = 0.5 byte/px.
const R_BITS_PER_PX: usize = 4;
/// MortonShift motion delta cost (the 2-byte address delta, per
/// `E-X265-MORTON-SHIFT-1`).
const MOTION_BYTES: usize = 2;
/// Per-region mode-signal overhead (which mode was chosen).
const SIGNAL_BYTES: usize = 1;
/// Seed — "RDOARBIT" family.
const SEED: u64 = 0x_5244_4F41_5242_4954;

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
    fn byte(&mut self) -> u8 {
        (self.next_u64() & 0xff) as u8
    }
}

/// One region's pixels, row-major `RW×RH`.
type Region = Vec<u8>;

fn noise_region(rng: &mut SplitMix64) -> Region {
    (0..(RW * RH)).map(|_| rng.byte()).collect()
}

/// Build `curr` as a rigid translate of `prev` by `(dx,dy)` (non-negative,
/// leading-edge): the overlap copies `prev`, the revealed L-strip is fresh
/// noise. This is the content a MortonShift mode explains with one delta.
fn rigid_translate(prev: &Region, dx: usize, dy: usize, rng: &mut SplitMix64) -> Region {
    let mut curr = vec![0u8; RW * RH];
    for y in 0..RH {
        for x in 0..RW {
            curr[y * RW + x] = if x >= dx && y >= dy {
                prev[(y - dy) * RW + (x - dx)]
            } else {
                rng.byte() // revealed leading-edge: genuinely new content
            };
        }
    }
    curr
}

/// Search the `0..=MAXD` non-negative delta window for a rigid match: the
/// smallest `(dx,dy)` whose translate of `prev` reproduces `curr` EXACTLY on
/// the overlap. Returns `(dx, dy, disocclusion_px)` or `None`. This is the
/// arbiter's applicability detector — the free-energy "does a motion model
/// explain this region?" test, read from content, never hardcoded.
fn rigid_match(prev: &Region, curr: &Region) -> Option<(usize, usize, usize)> {
    let mut best: Option<(usize, usize, usize)> = None;
    for dy in 0..=MAXD {
        for dx in 0..=MAXD {
            let mut ok = true;
            'chk: for y in dy..RH {
                for x in dx..RW {
                    if curr[y * RW + x] != prev[(y - dy) * RW + (x - dx)] {
                        ok = false;
                        break 'chk;
                    }
                }
            }
            if ok {
                // Overlap matched; the disocclusion is everything outside it.
                let overlap = (RW - dx) * (RH - dy);
                let disocc = RW * RH - overlap;
                // Prefer the smallest disocclusion (the true, tightest delta).
                if best.is_none_or(|(_, _, d)| disocc < d) {
                    best = Some((dx, dy, disocc));
                }
            }
        }
    }
    best
}

/// Bytes to code `n` pixels of residual at the model rate (ceil).
fn residual_bytes(n_px: usize) -> usize {
    (n_px * R_BITS_PER_PX).div_ceil(8)
}

/// Residual-mode cost: code every pixel that differs from `prev` (no motion
/// model — the frame-diff / intra fallback, the only thing a fixed pipeline
/// without an address substrate has).
fn residual_mode_bytes(prev: &Region, curr: &Region) -> usize {
    let changed = prev.iter().zip(curr).filter(|(a, b)| a != b).count();
    residual_bytes(changed)
}

/// MortonShift-mode cost when a rigid match with `disocc` revealed pixels was
/// found: the 2-byte address delta plus a residual for ONLY the revealed
/// strip (the interior is bit-exact, per `E-X265-MORTON-SHIFT-1`).
fn mortonshift_mode_bytes(disocc: usize) -> usize {
    MOTION_BYTES + residual_bytes(disocc)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mode {
    MortonShift,
    Residual,
}

/// The arbiter: per region, detect rigid applicability, cost both eligible
/// modes, pick argmin. A PURE function of the two candidate rates — it never
/// consults a ground-truth-optimal oracle, only the measured bytes.
fn arbitrate(prev: &Region, curr: &Region) -> (Mode, usize) {
    let residual = residual_mode_bytes(prev, curr);
    match rigid_match(prev, curr) {
        Some((_, _, disocc)) => {
            let ms = mortonshift_mode_bytes(disocc);
            if ms < residual {
                (Mode::MortonShift, ms)
            } else {
                (Mode::Residual, residual)
            }
        }
        None => (Mode::Residual, residual),
    }
}

/// One region's ground-truth kind, used ONLY to build content + to check the
/// arbiter's dispatch decision (never fed into the arbiter itself).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Rigid,
    NewContent,
}

/// Build a (prev, curr, kind) triple for region index `i`. Half the regions
/// are rigid translations (MortonShift-favorable), half are fresh unrelated
/// content (residual-favorable) — a mixed frame a fixed pipeline can't adapt to.
fn build_region(rng: &mut SplitMix64, i: usize) -> (Region, Region, Kind) {
    let prev = noise_region(rng);
    if i.is_multiple_of(2) {
        // Rigid: a small leading-edge translate (dx,dy in 1..=3).
        let dx = 1 + (rng.next_u64() % 3) as usize;
        let dy = 1 + (rng.next_u64() % 3) as usize;
        let curr = rigid_translate(&prev, dx, dy, rng);
        (prev, curr, Kind::Rigid)
    } else {
        // New content: unrelated fresh noise (no rigid model explains it).
        let curr = noise_region(rng);
        (prev, curr, Kind::NewContent)
    }
}

fn main() {
    let mut rng = SplitMix64::new(SEED);
    let regions: Vec<(Region, Region, Kind)> =
        (0..(NX * NY)).map(|i| build_region(&mut rng, i)).collect();

    let mut composite = 0usize; // adaptive arbiter + signal
    let mut all_residual = 0usize; // fixed pipeline: residual everywhere
    let mut correct_dispatch = 0usize;
    let mut ms_picks = 0usize;
    let mut res_picks = 0usize;

    for (prev, curr, kind) in &regions {
        let (mode, bytes) = arbitrate(prev, curr);
        composite += bytes + SIGNAL_BYTES;
        all_residual += residual_mode_bytes(prev, curr);
        match mode {
            Mode::MortonShift => ms_picks += 1,
            Mode::Residual => res_picks += 1,
        }
        // The arbiter should pick MortonShift exactly on the rigid regions.
        let expected = match kind {
            Kind::Rigid => Mode::MortonShift,
            Kind::NewContent => Mode::Residual,
        };
        if mode == expected {
            correct_dispatch += 1;
        }
    }

    let n = regions.len();
    eprintln!(
        "RDOARB regions={n} dispatch_correct={correct_dispatch}/{n} ms_picks={ms_picks} res_picks={res_picks}"
    );
    eprintln!(
        "RDOARB composite_bytes={composite} all_residual_bytes={all_residual} signal_overhead={}",
        n * SIGNAL_BYTES
    );
    let ratio = composite as f64 / all_residual as f64;
    eprintln!("RDOARB composite_over_fixed_ratio={ratio:.4} (adaptive < fixed ⇔ ratio < 1)");

    // Flip leg: take a rigid region, replace curr with fresh noise → the
    // arbiter's decision must FLIP from MortonShift to Residual (proves the
    // choice reads content, not a label).
    let mut frng = SplitMix64::new(SEED ^ 0xF117);
    let base = noise_region(&mut frng);
    let rigid_curr = rigid_translate(&base, 2, 1, &mut frng);
    let (m_rigid, _) = arbitrate(&base, &rigid_curr);
    let new_curr = noise_region(&mut frng);
    let (m_new, _) = arbitrate(&base, &new_curr);
    eprintln!(
        "RDOARB flip rigid_mode={m_rigid:?} newcontent_mode={m_new:?} flipped={}",
        (m_rigid == Mode::MortonShift && m_new == Mode::Residual) as u8
    );

    eprintln!("RDOARB r_bits_per_px={R_BITS_PER_PX} motion_bytes={MOTION_BYTES} maxd={MAXD}");
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Rigid content is detected and dispatched to MortonShift; unrelated
    /// content is dispatched to Residual — content-adaptive, not hardcoded.
    #[test]
    fn dispatch_matches_content_kind() {
        let mut rng = SplitMix64::new(SEED ^ 0x01);
        for i in 0..64 {
            let (prev, curr, kind) = build_region(&mut rng, i);
            let (mode, _) = arbitrate(&prev, &curr);
            let expected = match kind {
                Kind::Rigid => Mode::MortonShift,
                Kind::NewContent => Mode::Residual,
            };
            assert_eq!(mode, expected, "region {i}: wrong mode for its content");
        }
    }

    /// The decision FLIPS when the content flips: same `prev`, a rigid `curr`
    /// picks MortonShift, an unrelated `curr` picks Residual.
    #[test]
    fn choice_flips_with_content() {
        let mut rng = SplitMix64::new(SEED ^ 0x02);
        let prev = noise_region(&mut rng);
        let rigid = rigid_translate(&prev, 2, 1, &mut rng);
        let unrelated = noise_region(&mut rng);
        assert_eq!(arbitrate(&prev, &rigid).0, Mode::MortonShift);
        assert_eq!(arbitrate(&prev, &unrelated).0, Mode::Residual);
    }

    /// The adaptive composite (including signal overhead) beats the fixed
    /// residual-everywhere pipeline on mixed content — the self-optimizing
    /// win a fixed pipeline structurally cannot get.
    #[test]
    fn adaptive_beats_fixed_pipeline() {
        let mut rng = SplitMix64::new(SEED ^ 0x03);
        let regions: Vec<_> = (0..64).map(|i| build_region(&mut rng, i)).collect();
        let mut composite = 0usize;
        let mut fixed = 0usize;
        for (prev, curr, _) in &regions {
            composite += arbitrate(prev, curr).1 + SIGNAL_BYTES;
            fixed += residual_mode_bytes(prev, curr);
        }
        assert!(
            composite < fixed,
            "adaptive {composite} must beat fixed {fixed} even with signal overhead"
        );
    }

    /// The arbiter is a pure function of content: same input → same decision.
    #[test]
    fn arbiter_is_deterministic() {
        let mut rng = SplitMix64::new(SEED ^ 0x04);
        let prev = noise_region(&mut rng);
        let curr = rigid_translate(&prev, 3, 2, &mut rng);
        assert_eq!(arbitrate(&prev, &curr), arbitrate(&prev, &curr));
    }

    /// A rigid match is found for a translate and reports the correct
    /// disocclusion (the leading-edge L-strip).
    #[test]
    fn rigid_match_finds_translate() {
        let mut rng = SplitMix64::new(SEED ^ 0x05);
        let prev = noise_region(&mut rng);
        let curr = rigid_translate(&prev, 2, 1, &mut rng);
        let (dx, dy, disocc) = rigid_match(&prev, &curr).expect("must find the translate");
        assert_eq!((dx, dy), (2, 1));
        // disocc = W·H − (W−2)(H−1) = 256 − 14·15 = 256 − 210 = 46.
        assert_eq!(disocc, 46);
    }
}
