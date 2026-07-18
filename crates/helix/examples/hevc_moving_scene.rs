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
    cx: f64,      // screen center x (px)
    cy: f64,      // screen center y (px)
    radius: f64,  // spiral radius (px)
    sign: f64,    // +1 upper hemisphere, -1 lower (parity, like sprite-replay)
    bright: f64,  // peak luma
    sigma: f64,   // gaussian spread (px)
}

/// φ-spiral (golden-angle Fibonacci hemisphere) point n of TOTAL, signed.
/// Returns the (x, y) of the spiral in the unit disk; z (depth) modulates size.
fn phi_spiral_xy(n: usize, total: usize, sign: f64) -> (f64, f64, f64) {
    // Golden angle ≈ 2.399963 rad — the same irrational winding the arc's
    // φ-spiral / CurveRuler uses (stride-4-over-17 is its integer cousin).
    let ga = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    let t = (n as f64 + 0.5) / total as f64; // 0..1 along the path
    let z = sign * (1.0 - t); // hemisphere height, sign flips hemisphere
    let r = (1.0 - z * z).sqrt(); // disk radius at height z
    let theta = n as f64 * ga;
    (r * theta.cos(), r * theta.sin(), z)
}

fn seed_sprites() -> [Sprite; NUM_SPRITES] {
    // Same seed constant as sprite_replay::seed_sprites (0x5350_5249_5445_5F31 = "SPRITE_1").
    let mut rng = SplitMix64::new(0x5350_5249_5445_5F31);
    core::array::from_fn(|i| {
        let _place = rng.next_u64(); // consume one u64 (mirrors the probe's field order)
        let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
        // Spread 8 centers over the frame (2 rows × 4 cols), jittered.
        let col = (i % 4) as f64;
        let row = (i / 4) as f64;
        let cx = 40.0 + col * (W as f64 - 80.0) / 3.0 + rng.range(-8.0, 8.0);
        let cy = 70.0 + row * (H as f64 - 140.0) + rng.range(-8.0, 8.0);
        let radius = rng.range(18.0, 34.0);
        let bright = rng.range(170.0, 245.0);
        let sigma = rng.range(6.0, 11.0);
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
    let path = std::env::args().nth(1).unwrap_or_else(|| "scene.y4m".into());
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
                let g = 24.0 + 10.0 * (((px as f64 + pan) * 0.03).sin())
                    + 6.0 * ((py as f64 * 0.05).cos());
                y[py * W + px] = g.clamp(0.0, 60.0) as u8;
            }
        }
        // Splat each sprite at its φ-spiral point for this frame.
        for s in &sprites {
            let (sx, sy, sz) = phi_spiral_xy(frame, TOTAL, s.sign);
            let px0 = s.cx + s.radius * sx;
            let py0 = s.cy + s.radius * sy;
            // Depth (z) modulates size: nearer (z→1) = larger/brighter.
            let depth = 0.6 + 0.4 * (sz.abs());
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
