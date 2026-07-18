//! HEVC moving-scene anchor renderer (plan §5 external anchor, made visual).
//!
//! Renders the sprite-replay scene — 8 gaussian sprites tracing φ-spiral
//! (golden-angle hemisphere) paths, alternating hemispheres by index parity,
//! exactly as `helix/src/sprite_replay.rs` seeds them — to a Y4M (I420) clip.
//! x265 then encodes it (the arc's "replay x265's GOP grammar" made literal:
//! x265 runs its own I/P/B GOP over OUR moving scene) and reports bits/frame +
//! PSNR; ffmpeg decodes frames back for the screenshot montage.
//!
//! std-only, deterministic (SplitMix64, no rand) — matches the probe discipline.

use std::io::{BufWriter, Write};

const W: usize = 320;
const H: usize = 240;
const TOTAL: usize = 240; // frames == sprite-replay TOTAL
const NUM_SPRITES: usize = 8; // sprite-replay NUM_SPRITES

/// SplitMix64 — same generator + seed family as sprite_replay.rs.
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
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + self.unit() * (hi - lo)
    }
}

#[derive(Clone, Copy)]
struct Sprite {
    cx: f64,     // screen center x (px)
    cy: f64,     // screen center y (px)
    radius: f64, // spiral radius (px)
    sign: f64,   // +1 upper hemisphere, -1 lower (parity, like sprite-replay)
    bright: f64, // peak luma
    sigma: f64,  // gaussian spread (px)
}

/// φ-spiral (golden-angle Fibonacci hemisphere) point n of TOTAL, signed.
/// Returns the canonical `(x, z, y)` cartesian of the hemisphere point — the
/// SAME axis order `sprite_replay::sprite_position` uses (`HemispherePoint::
/// cartesian` → `(x, z, y)`, position = center + scale·[x, z, y]). The **signed
/// height is `z`** (the 2nd element), so `sign` genuinely selects the
/// hemisphere; the caller must project a signed axis to screen (not `abs`) or
/// the two hemispheres collapse onto one trajectory.
fn phi_spiral_cart(n: usize, total: usize, sign: f64) -> (f64, f64, f64) {
    // Golden angle ≈ 2.399963 rad — the same irrational winding the arc's
    // φ-spiral / CurveRuler uses (stride-4-over-17 is its integer cousin).
    let ga = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    let t = (n as f64 + 0.5) / total as f64; // 0..1 along the path
    let z = sign * (1.0 - t); // signed hemisphere height (upper for +, lower for −)
    let r = (1.0 - z * z).sqrt(); // disk radius at height z
    let theta = n as f64 * ga;
    (r * theta.cos(), z, r * theta.sin()) // (x, z=signed height, y)
}

fn seed_sprites() -> [Sprite; NUM_SPRITES] {
    // Same seed constant as sprite_replay::seed_sprites (0x5350_5249_5445_5F31 = "SPRITE_1").
    let mut rng = SplitMix64::new(0x5350_5249_5445_5F31);
    core::array::from_fn(|i| {
        // CANONICAL draw sequence — byte-for-byte the order sprite_replay uses:
        // place (u64), then center[0..3] (3 range draws), then scale. Screen/render
        // params are DERIVED from these canonical world values with NO extra RNG
        // draws, so sprite i's place/center/scale stream matches the probe exactly.
        let _place = rng.next_u64();
        let c0 = rng.range(-50.0, 50.0); // center[0]
        let c1 = rng.range(-50.0, 50.0); // center[1]
        let c2 = rng.range(-50.0, 50.0); // center[2]
        let scale_w = rng.range(5.0, 25.0); // scale
        let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
        // Derive the on-screen envelope from the canonical world values.
        let cx = W as f64 * 0.5 + (c0 / 50.0) * (W as f64 * 0.38); // [-50,50] → screen x
        let cy = H as f64 * 0.5 + (c1 / 50.0) * (H as f64 * 0.30); // [-50,50] → screen y
        let radius = 16.0 + (scale_w - 5.0) / 20.0 * 18.0; // [5,25] → [16,34] px
        let bright = 175.0 + ((c2 + 50.0) / 100.0) * 70.0; // [-50,50] → [175,245]
        let sigma = 6.0 + (scale_w - 5.0) / 20.0 * 5.0; // [5,25] → [6,11] px
        Sprite {
            cx,
            cy,
            radius,
            sign,
            bright,
            sigma,
        }
    })
}

fn main() -> std::io::Result<()> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "scene.y4m".into());
    let sprites = seed_sprites();
    let f = std::fs::File::create(&path)?;
    let mut out = BufWriter::new(f);

    // Y4M header — I420, 25 fps, progressive.
    write!(out, "YUV4MPEG2 W{W} H{H} F25:1 Ip A1:1 C420jpeg\n")?;

    let mut y = vec![0u8; W * H];
    let cw = W / 2;
    let ch = H / 2;
    let uv = vec![128u8; cw * ch]; // neutral chroma (grayscale scene)

    for frame in 0..TOTAL {
        // Background: a faint moving gradient so inter-frame prediction has
        // global motion to track (a static bg would make every P-frame near-zero).
        let pan = (frame as f64 / TOTAL as f64) * 40.0;
        for py in 0..H {
            for px in 0..W {
                let g = 24.0
                    + 10.0 * (((px as f64 + pan) * 0.03).sin())
                    + 6.0 * ((py as f64 * 0.05).cos());
                y[py * W + px] = g.clamp(0.0, 60.0) as u8;
            }
        }
        // Splat each sprite at its φ-spiral point for this frame.
        for s in &sprites {
            // Canonical (x, z, y): x → screen-x, SIGNED z → screen-y (so the
            // hemisphere sign mirrors the sprite vertically), y → depth.
            let (cx_off, cz_signed, cy_depth) = phi_spiral_cart(frame, TOTAL, s.sign);
            let px0 = s.cx + s.radius * cx_off;
            let py0 = s.cy + s.radius * cz_signed;
            // The remaining axis (y) modulates size: nearer = larger/brighter.
            let depth = 0.6 + 0.4 * cy_depth.abs();
            let sigma = s.sigma * depth;
            let peak = s.bright * depth;
            let rad = (sigma * 3.0).ceil() as i64;
            let inv2s2 = 1.0 / (2.0 * sigma * sigma);
            let cxi = px0.round() as i64;
            let cyi = py0.round() as i64;
            for dy in -rad..=rad {
                let yy = cyi + dy;
                if yy < 0 || yy >= H as i64 {
                    continue;
                }
                for dx in -rad..=rad {
                    let xx = cxi + dx;
                    if xx < 0 || xx >= W as i64 {
                        continue;
                    }
                    let d2 = (dx * dx + dy * dy) as f64;
                    let v = peak * (-d2 * inv2s2).exp();
                    let idx = yy as usize * W + xx as usize;
                    let cur = y[idx] as f64;
                    y[idx] = (cur + v).clamp(0.0, 255.0) as u8; // additive splat
                }
            }
        }
        out.write_all(b"FRAME\n")?;
        out.write_all(&y)?;
        out.write_all(&uv)?; // U
        out.write_all(&uv)?; // V
    }
    out.flush()?;
    eprintln!(
        "rendered {TOTAL} frames {W}x{H} ({} sprites, φ-spiral motion) → {path}",
        NUM_SPRITES
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The SplitMix64 stream is deterministic for the sprite_replay seed — the
    /// canonical-sequence contract the scene relies on.
    #[test]
    fn splitmix64_is_deterministic_for_the_sprite_seed() {
        let mut a = SplitMix64::new(0x5350_5249_5445_5F31);
        let mut b = SplitMix64::new(0x5350_5249_5445_5F31);
        for _ in 0..8 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
        // Distinct successive outputs (not a stuck generator).
        let mut c = SplitMix64::new(0x5350_5249_5445_5F31);
        let x = c.next_u64();
        let y = c.next_u64();
        assert_ne!(x, y);
    }

    /// `sign` MUST select the hemisphere: the signed-height axis flips with it,
    /// so opposite signs give different projected positions (the bug CodeRabbit
    /// caught — `abs(z)` had cancelled the sign).
    #[test]
    fn phi_spiral_sign_separates_hemispheres() {
        for n in [0usize, 37, 120, 239] {
            let (xp, zp, _) = phi_spiral_cart(n, TOTAL, 1.0);
            let (xn, zn, _) = phi_spiral_cart(n, TOTAL, -1.0);
            // x (azimuth) is sign-independent; the height z is the discriminator.
            assert!(
                (xp - xn).abs() < 1e-12,
                "azimuth is sign-independent at n={n}"
            );
            assert!(
                zp > 0.0 && zn < 0.0,
                "pos=upper / neg=lower hemisphere at n={n}"
            );
            assert!(
                (zp - zn).abs() > 1e-9,
                "sign must produce distinct projected height at n={n}"
            );
        }
    }

    /// seed_sprites is deterministic and alternates hemisphere by index parity.
    #[test]
    fn seed_sprites_deterministic_and_alternating() {
        let a = seed_sprites();
        let b = seed_sprites();
        for i in 0..NUM_SPRITES {
            assert_eq!(a[i].sign, b[i].sign);
            assert_eq!(a[i].cx.to_bits(), b[i].cx.to_bits());
            assert_eq!(a[i].sign, if i % 2 == 0 { 1.0 } else { -1.0 });
        }
    }
}
