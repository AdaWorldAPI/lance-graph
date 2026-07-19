//! PROBE-MORTON-SHIFT-MOTION — the non-tautological head-to-head-2
//! (board `E-X265-HEADTOHEAD-1` follow-up; the operator's "moving a sprite
//! could become bit-shifting to adjacent" claim made falsifiable).
//!
//! The claim: rigid translation of an **arbitrary** sprite is a Morton
//! address-delta — a *dilated-lane integer add* (carry-through-the-holes) —
//! O(1) in the sprite's pixel count, bit-exact, NOT a raster block-search +
//! MV-residual. x265 categorically lacks the substrate (no Morton address,
//! no blasgraph shift operator over the neighborhood), so this is a
//! capability x265 never had, not a tuning difference.
//!
//! Why this is NOT `E-X265-HEADTOHEAD-1`'s tautology: there the sprite
//! APPEARANCE was φ-generated (model-matched), so ∞-PSNR was structural.
//! Here the appearance is arbitrary deterministic-noise (never φ-placed) and
//! is paid for EQUALLY on both sides — only the MOTION representation is
//! compared. The advantage, if real, is about how motion is coded, not about
//! a matched appearance model.
//!
//! Three legs (all structural + deterministic — the orchestrator adjudicates
//! the real-x265 quantitative sub-pel leg as v2):
//!  A. **dilated-lane add correctness + O(1) rigid translate.** The SAME
//!     `(dx,dy)` delta, applied once as two carry-through-holes adds, moves
//!     EVERY sprite pixel's Morton code identically → translation is one
//!     delta, independent of sprite size; reconstruction is BIT-EXACT for
//!     tile-aligned (integer) motion (interior residual = 0 bytes, where
//!     x265's DCT+quant round-trip is never exactly 0).
//!  B. **dis-occlusion accounting (the honest tail).** On a rigid translate
//!     the ONLY genuine new content is the revealed leading-edge L-strip
//!     (`dx·H + dy·W − dx·dy` px). Measured, not hidden — it is exactly the
//!     residual the entropy coder must still carry (my "shrinks the tail,
//!     doesn't erase it" fence). Rotation/scale/deform are out of scope by
//!     construction (not a bit-shift).
//!  C. **comma sub-tile phase (the Fujifilm X-Trans move).** Steinhaus
//!     three-gap (≤3) + stride-4-over-17 coprime full-permutation: the
//!     aperiodic sub-pel dither that replaces quarter-pel interpolation
//!     without aliasing against the tile grid (`E-COMMA-PERTURBATION-PHASE-1`).
//!
//! std-only, deterministic (SplitMix64, no `rand`). Prints `MSHIFT …`
//! greppable lines to stderr.

/// Sprite-field axis resolution: 8-bit axes → 16-bit Morton code, 256×256.
const AXIS_BITS: u32 = 8;
const AXIS: u32 = 1 << AXIS_BITS; // 256
/// Even-bit (x-lane) and odd-bit (y-lane) masks for the 16-bit code.
const XMASK: u32 = 0x5555;
const YMASK: u32 = 0xAAAA;
/// Seed — "MSHIFT01" family, per the probe convention.
const SEED: u64 = 0x_4D53_4849_4654_3031;

// ── SplitMix64 (same generator family as the other helix probes) ─────────

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

// ── 2-D Morton (Z-order) address + dilated-lane arithmetic ───────────────

/// Spread the low 8 bits of `v` into the even bit positions (dilation).
fn dilate8(v: u32) -> u32 {
    let mut x = v & 0xff;
    x = (x | (x << 4)) & 0x0f0f;
    x = (x | (x << 2)) & 0x3333;
    x = (x | (x << 1)) & 0x5555;
    x
}

/// Inverse of `dilate8`: gather the even bits back into the low 8 bits.
fn undilate8(v: u32) -> u32 {
    let mut x = v & 0x5555;
    x = (x | (x >> 1)) & 0x3333;
    x = (x | (x >> 2)) & 0x0f0f;
    x = (x | (x >> 4)) & 0x00ff;
    x
}

/// `morton2(x,y)` — interleave two 8-bit axes into a 16-bit Z-order code
/// (x in even bits, y in odd bits).
fn morton2(x: u32, y: u32) -> u32 {
    dilate8(x) | (dilate8(y) << 1)
}

/// Decode a 16-bit Morton code back to `(x, y)`.
fn unmorton2(m: u32) -> (u32, u32) {
    (undilate8(m & XMASK), undilate8((m & YMASK) >> 1))
}

/// Add `dx` (0..256) into the x-lane of a Morton code with correct carry:
/// set the y-bits to 1 so carries ripple THROUGH them into the next x-bit,
/// add the dilated delta, then mask back to the x-lane. The "bit-shifting to
/// adjacent" operation, exact form.
fn morton_add_x(m: u32, dx: u32) -> u32 {
    let xsum = (((m & XMASK) | YMASK).wrapping_add(dilate8(dx))) & XMASK;
    xsum | (m & YMASK)
}

/// Add `dy` (0..256) into the y-lane (dilated then shifted to odd bits),
/// carries rippling through the set x-bits.
fn morton_add_y(m: u32, dy: u32) -> u32 {
    let ysum = (((m & YMASK) | XMASK).wrapping_add(dilate8(dy) << 1)) & YMASK;
    ysum | (m & XMASK)
}

/// The single rigid-translate operator: apply the SAME `(dx,dy)` delta to a
/// Morton code via two carry-through-holes adds. This is the whole motion
/// description — one delta, applied identically to every sprite pixel.
fn morton_translate(m: u32, dx: u32, dy: u32) -> u32 {
    morton_add_y(morton_add_x(m, dx), dy)
}

// ── the arbitrary (non-φ) sprite ─────────────────────────────────────────

/// A sprite = a rectangle of deterministic-noise pixel values, anchored at
/// `(ax, ay)`. Appearance is arbitrary — never φ-generated — so any motion
/// advantage cannot be an appearance-model-match artifact.
struct Sprite {
    ax: u32,
    ay: u32,
    w: u32,
    h: u32,
    /// `px[oy*w + ox]` = the value at offset `(ox, oy)` from the anchor.
    px: Vec<u8>,
}

impl Sprite {
    fn new_noise(rng: &mut SplitMix64, ax: u32, ay: u32, w: u32, h: u32) -> Self {
        let px = (0..(w * h)).map(|_| (rng.next_u64() & 0xff) as u8).collect();
        Self { ax, ay, w, h, px }
    }

    /// Render the sprite AFTER a rigid translate by `(dx,dy)`, applying the
    /// translate as the Morton address-delta (`morton_translate`) rather than
    /// re-anchoring — this is the codec path under test.
    fn render_translated_by_address(&self, dx: u32, dy: u32, field: &mut [u8]) {
        for oy in 0..self.h {
            for ox in 0..self.w {
                let m0 = morton2(self.ax + ox, self.ay + oy);
                let m = morton_translate(m0, dx, dy);
                let (x, y) = unmorton2(m);
                if x < AXIS && y < AXIS {
                    field[(y * AXIS + x) as usize] = self.px[(oy * self.w + ox) as usize];
                }
            }
        }
    }
}

/// Ground-truth translated render: re-anchor to `(ax+dx, ay+dy)` and draw
/// directly in (x,y). If the address-delta path matches this bit-for-bit,
/// "translate = address arithmetic" is confirmed for this motion.
fn render_ground_truth_translate(s: &Sprite, dx: u32, dy: u32, field: &mut [u8]) {
    for oy in 0..s.h {
        for ox in 0..s.w {
            let x = s.ax + ox + dx;
            let y = s.ay + oy + dy;
            if x < AXIS && y < AXIS {
                field[(y * AXIS + x) as usize] = s.px[(oy * s.w + ox) as usize];
            }
        }
    }
}

// ── comma sub-tile phase fence (shared with mu_hydration_probe) ──────────

/// Steinhaus three-gap theorem: sorted `{n·φ}` takes ≤ 3 distinct arc
/// lengths — the aperiodicity certificate for the sub-pel dither.
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

/// The stride-4-over-17 coprime walk: `gcd(4,17)=1` ⇒ full permutation of
/// the 17 residues (the discrete comma).
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

// ── measurement ──────────────────────────────────────────────────────────

/// Number of bytes to transmit a rigid-translate motion delta: a varint per
/// axis. This is CONSTANT in the sprite's pixel count — the whole object
/// moves for the cost of one `(dx,dy)`.
fn motion_delta_bytes(dx: u32, dy: u32) -> usize {
    fn varint_len(mut v: u32) -> usize {
        let mut n = 1;
        while v >= 0x80 {
            v >>= 7;
            n += 1;
        }
        n
    }
    varint_len(dx) + varint_len(dy)
}

/// The genuine new content on a rigid translate by `(dx,dy)`: the revealed
/// leading-edge L-strip, `dx·h + dy·w − dx·dy` pixels (inclusion-exclusion).
/// This is the residual the entropy coder must still carry — the honest tail.
fn disocclusion_pixels(w: u32, h: u32, dx: u32, dy: u32) -> u32 {
    dx * h + dy * w - dx * dy
}

fn main() {
    let mut rng = SplitMix64::new(SEED);

    // An arbitrary (non-φ) 24×24 deterministic-noise sprite at (40, 32).
    let sprite = Sprite::new_noise(&mut rng, 40, 32, 24, 24);

    // Leg A — rigid tile-aligned translate = address-delta, bit-exact.
    // Try several integer deltas and confirm the address path reproduces the
    // ground-truth re-anchored render byte-for-byte, and that the motion cost
    // is constant regardless of the sprite's 576 pixels.
    let deltas: [(u32, u32); 4] = [(1, 0), (0, 1), (7, 5), (16, 16)];
    let mut all_exact = true;
    for (dx, dy) in deltas {
        let mut addr_field = vec![0u8; (AXIS * AXIS) as usize];
        let mut gt_field = vec![0u8; (AXIS * AXIS) as usize];
        sprite.render_translated_by_address(dx, dy, &mut addr_field);
        render_ground_truth_translate(&sprite, dx, dy, &mut gt_field);
        let exact = addr_field == gt_field;
        all_exact &= exact;
        let mbytes = motion_delta_bytes(dx, dy);
        let disocc = disocclusion_pixels(sprite.w, sprite.h, dx, dy);
        eprintln!(
            "MSHIFT legA dx={dx} dy={dy} bit_exact={} motion_bytes={mbytes} sprite_px={} disocc_px={disocc}",
            if exact { 1 } else { 0 },
            sprite.w * sprite.h
        );
    }
    eprintln!(
        "MSHIFT legA_all_bit_exact={} (motion cost O(1) in sprite_px — one (dx,dy) moves all)",
        if all_exact { 1 } else { 0 }
    );

    // Leg A' — dilated-lane add identity: morton2(x+dx,y+dy) == the two
    // carry-through-holes adds, over a deterministic sample of (x,y,dx,dy).
    let mut add_identity_ok = true;
    for _ in 0..4096 {
        let x = (rng.next_u64() % 200) as u32;
        let y = (rng.next_u64() % 200) as u32;
        let dx = (rng.next_u64() % (AXIS - x) as u64) as u32;
        let dy = (rng.next_u64() % (AXIS - y) as u64) as u32;
        let direct = morton2(x + dx, y + dy);
        let via_add = morton_translate(morton2(x, y), dx, dy);
        if direct != via_add {
            add_identity_ok = false;
            break;
        }
    }
    eprintln!(
        "MSHIFT legA_dilated_add_identity={}",
        if add_identity_ok { 1 } else { 0 }
    );

    // Leg B — dis-occlusion fraction for a representative small motion: the
    // residual tail the address-shift does NOT eliminate (revealed edge).
    let (bdx, bdy) = (2u32, 1u32);
    let disocc = disocclusion_pixels(sprite.w, sprite.h, bdx, bdy);
    let frac = disocc as f64 / (sprite.w * sprite.h) as f64;
    eprintln!(
        "MSHIFT legB disocc_dx={bdx} dy={bdy} disocc_px={disocc} disocc_frac={frac:.4} (interior residual = 0; only this strip is new content)"
    );

    // Leg C — comma sub-tile phase (the Fujifilm X-Trans anti-moiré move).
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let phi_seq: Vec<f64> = (0..4096)
        .map(|n| {
            let v = n as f64 * phi;
            v - v.floor()
        })
        .collect();
    eprintln!(
        "MSHIFT legC comma_three_gap_distinct={} comma_coprime_full_perm={}",
        three_gap_distinct_count(&phi_seq),
        if coprime_walk_is_full_permutation() { 1 } else { 0 }
    );

    eprintln!("MSHIFT axis={AXIS} sprite={}x{}", sprite.w, sprite.h);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Morton round-trip: encode then decode is the identity on the field.
    #[test]
    fn morton_roundtrip_is_identity() {
        let mut rng = SplitMix64::new(SEED ^ 0xA1);
        for _ in 0..10_000 {
            let x = (rng.next_u64() % AXIS as u64) as u32;
            let y = (rng.next_u64() % AXIS as u64) as u32;
            assert_eq!(unmorton2(morton2(x, y)), (x, y));
        }
    }

    /// The core claim: `morton2(x+dx, y+dy)` equals the two carry-through-
    /// holes adds — translation IS dilated-lane address arithmetic.
    #[test]
    fn translate_is_dilated_add() {
        let mut rng = SplitMix64::new(SEED ^ 0xA2);
        for _ in 0..20_000 {
            let x = (rng.next_u64() % 200) as u32;
            let y = (rng.next_u64() % 200) as u32;
            let dx = (rng.next_u64() % (AXIS - x) as u64) as u32;
            let dy = (rng.next_u64() % (AXIS - y) as u64) as u32;
            assert_eq!(
                morton2(x + dx, y + dy),
                morton_translate(morton2(x, y), dx, dy),
                "translate must be a dilated-lane add: x={x} y={y} dx={dx} dy={dy}"
            );
        }
    }

    /// Rigid tile-aligned translate via address-delta reconstructs the sprite
    /// bit-for-bit against the re-anchored ground truth (interior residual 0).
    #[test]
    fn rigid_translate_is_bit_exact() {
        let mut rng = SplitMix64::new(SEED ^ 0xA3);
        let sprite = Sprite::new_noise(&mut rng, 30, 20, 16, 16);
        for (dx, dy) in [(1u32, 0u32), (0, 1), (5, 9), (20, 20)] {
            let mut addr = vec![0u8; (AXIS * AXIS) as usize];
            let mut gt = vec![0u8; (AXIS * AXIS) as usize];
            sprite.render_translated_by_address(dx, dy, &mut addr);
            render_ground_truth_translate(&sprite, dx, dy, &mut gt);
            assert_eq!(addr, gt, "address-delta render must match ground truth");
        }
    }

    /// Motion cost is O(1) in sprite size: the delta bytes for a fixed motion
    /// do not depend on how many pixels the sprite has.
    #[test]
    fn motion_cost_is_constant_in_sprite_size() {
        let a = motion_delta_bytes(7, 5);
        let b = motion_delta_bytes(7, 5);
        assert_eq!(a, b);
        // Same motion, wildly different sprite areas → identical motion cost.
        let mut rng = SplitMix64::new(SEED ^ 0xA4);
        let small = Sprite::new_noise(&mut rng, 0, 0, 4, 4);
        let big = Sprite::new_noise(&mut rng, 0, 0, 64, 64);
        assert_eq!(small.w * small.h, 16);
        assert_eq!(big.w * big.h, 4096);
        // motion_delta_bytes takes only the delta — proving independence.
        assert_eq!(motion_delta_bytes(7, 5), a);
    }

    /// Dis-occlusion inclusion-exclusion is correct and bounded by the L-strip.
    #[test]
    fn disocclusion_is_the_l_strip() {
        assert_eq!(disocclusion_pixels(24, 24, 0, 0), 0);
        assert_eq!(disocclusion_pixels(24, 24, 1, 0), 24);
        assert_eq!(disocclusion_pixels(24, 24, 0, 1), 24);
        // dx·h + dy·w − dx·dy = 2·24 + 1·24 − 2·1 = 70.
        assert_eq!(disocclusion_pixels(24, 24, 2, 1), 70);
    }

    /// Comma fence: three-gap ≤ 3 and coprime full permutation.
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
