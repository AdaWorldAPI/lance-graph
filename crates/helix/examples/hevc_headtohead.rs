//! HEAD-TO-HEAD: object-level helix motion codes vs x265's per-block MV field,
//! on the SAME moving scene.
//!
//! Renders the sprite-replay scene (the identical scene model + renderer as
//! `hevc_moving_scene.rs` — copied verbatim, see `render_frame` below) to a
//! Y4M for **lane A** (x265's own I/P/B GOP over our pixels — the orchestrator
//! runs x265 on the emitted file), and computes **lane B** in-process: each
//! sprite's motion is encoded as one [`helix::residue::Signed360`] code per
//! P-anchor (the object-level "one helix code per sprite per anchor" motion
//! representation), decoded back, and re-rasterized through the SAME
//! renderer used for the ground truth, so lane B's bitstream size and
//! reconstruction PSNR are measured against the real pixels x265 also sees.
//!
//! ## THE LOAD-BEARING CAVEAT — read before citing any number this prints
//!
//! Lane B is a **model-based codec handed the exact generative model of a
//! self-generated scene** — the renderer, the φ-spiral motion template, and
//! the encoder all agree by construction (the scene's sprites literally move
//! along the same golden-angle hemisphere lift that `Signed360::azimuth`
//! encodes). A large win over x265 on THIS scene is therefore **expected and
//! tautological**, not a general-codec result: it measures the concrete
//! bit-cost of the amortization and its lower bound **on model-matched
//! content**, never a claim that helix motion codes beat x265 on arbitrary,
//! independently-captured motion. That harder question — helix-manifold
//! codes vs arbitrary motion the encoder did NOT generate — is the named
//! `[H]` follow-up in `PROBE-SPRITE-REPLAY` (see `crates/helix/src/
//! sprite_replay.rs` module doc, "Why Signed360 should beat ResidueEdge").
//!
//! Cross-ref: `crates/helix/src/sprite_replay.rs` (the encode/decode
//! primitives this file reuses, and the honest-inverse discussion of
//! `decode_signed_to_n`), `.claude/board/EPIPHANIES.md` `E-SPRITE-IPB-HELIX-1`,
//! plan `x265-sprite-replay-probe-v1.md`.
//!
//! std-only + the `helix` public API. Deterministic (SplitMix64, no `rand`).

use std::io::{BufWriter, Write};

use helix::placement::Sign;
use helix::residue::{ResidueEncoder, Signed360};

const W: usize = 320;
const H: usize = 240;
const TOTAL: usize = 240; // frames == sprite-replay TOTAL
const NUM_SPRITES: usize = 8; // sprite-replay NUM_SPRITES

/// 1 I-anchor + 5 P-anchors, evenly spaced along the path — the GOP grammar
/// `sprite_replay.rs` uses (`I B B P B B P …`, B-frames un-stored here).
const NUM_ANCHORS: usize = 6;
/// Constant spacing between anchors in `n`-space (240 / 6 = 40). Known to
/// both encoder and decoder as part of the GOP structure, not smuggled from
/// the ground truth (mirrors `sprite_replay::ANCHOR_STEP`'s framing).
const ANCHOR_STEP: usize = TOTAL / NUM_ANCHORS;

/// SplitMix64 — same generator + seed family as `hevc_moving_scene.rs` /
/// `sprite_replay.rs`.
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
    // ── canonical world-space identity (lane B's I-frame appearance bytes) ──
    place: u64,   // HHTL place anchor — the sprite's fixed helix identity
    c0: f32,      // canonical center[0]
    c1: f32,      // canonical center[1]
    c2: f32,      // canonical center[2]
    scale_w: f32, // canonical scale
    // ── screen-space render params, derived from the canonical values above ──
    cx: f64,     // screen center x (px)
    cy: f64,     // screen center y (px)
    radius: f64, // spiral radius (px)
    sign: f64,   // +1 upper hemisphere, -1 lower (parity, like sprite-replay)
    bright: f64, // peak luma
    sigma: f64,  // gaussian spread (px)
}

/// `helix::placement::Sign` for this sprite — same parity the `sign: f64`
/// field encodes (`Pos` for `i % 2 == 0`, `Neg` otherwise).
fn sign_enum(sign_f64: f64) -> Sign {
    if sign_f64 >= 0.0 {
        Sign::Pos
    } else {
        Sign::Neg
    }
}

/// φ-spiral (golden-angle Fibonacci hemisphere) point n of TOTAL, signed.
/// Returns the canonical `(x, z, y)` cartesian of the hemisphere point — the
/// SAME axis order `sprite_replay::sprite_position` uses (`HemispherePoint::
/// cartesian` → `(x, z, y)`, position = center + scale·[x, z, y]). The
/// **signed height is `z`** (the 2nd element), so `sign` genuinely selects
/// the hemisphere; the caller must project a signed axis to screen (not
/// `abs`) or the two hemispheres collapse onto one trajectory.
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
        let place = rng.next_u64();
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
            place,
            c0: c0 as f32,
            c1: c1 as f32,
            c2: c2 as f32,
            scale_w: scale_w as f32,
            cx,
            cy,
            radius,
            sign,
            bright,
            sigma,
        }
    })
}

/// One sprite's I-frame "appearance" payload, LITERALLY serialized (LE) —
/// not a hardcoded byte count: `place` (8B) + `center` as 3×f32 (12B) +
/// `scale` as f32 (4B) = 24B. This is the identity lane B stores ONCE per
/// sprite; P-anchors carry motion only (via [`Signed360`]).
fn iframe_appearance_bytes(s: &Sprite) -> [u8; 24] {
    let mut b = [0u8; 24];
    b[0..8].copy_from_slice(&s.place.to_le_bytes());
    b[8..12].copy_from_slice(&s.c0.to_le_bytes());
    b[12..16].copy_from_slice(&s.c1.to_le_bytes());
    b[16..20].copy_from_slice(&s.c2.to_le_bytes());
    b[20..24].copy_from_slice(&s.scale_w.to_le_bytes());
    b
}

/// Wrapping `u16` distance (handles the azimuth's `[0, 65536)` wraparound).
fn azimuth_wrap_distance(a: u16, b: u16) -> u32 {
    let d = (a as i32 - b as i32).unsigned_abs();
    d.min(65536 - d)
}

/// Decode a [`Signed360`] back to the best-matching `n` — exhaustive search
/// over `0..total`, re-encoding each candidate `n` with the SAME `place` and
/// `sign` (both known to the decoder as the sprite's stored identity — the
/// I-frame appearance bytes, never re-derived from the residue itself) and
/// comparing the resulting `azimuth` field via wrapping `u16` distance. This
/// is the literal inverse of [`ResidueEncoder::encode_signed`]'s azimuth
/// stage — the same honest-inverse construction `sprite_replay.rs`'s
/// `decode_signed_to_n` uses, just built directly off the public
/// `encode_signed` round-trip instead of re-deriving the private formula.
fn decode_signed_to_n(place: u64, s: Signed360, total: usize) -> usize {
    let enc = ResidueEncoder::new(total);
    let mut best_n = 0usize;
    let mut best_d = u32::MAX;
    for n in 0..total {
        let cand = enc.encode_signed(place, n, s.sign());
        let d = azimuth_wrap_distance(cand.azimuth, s.azimuth);
        if d < best_d {
            best_d = d;
            best_n = n;
        }
    }
    best_n
}

/// Lane B encode + decode for every sprite: the I-frame appearance bytes,
/// the P-anchor motion bytes (both literally serialized via `to_bytes()` /
/// `iframe_appearance_bytes`, never a hardcoded formula), and the decoded
/// anchor `n` values (`anchors_n[sprite][0]` stays `0` — the I-anchor's `n`
/// is a GOP-structure constant, not a transmitted code; `[1..NUM_ANCHORS)`
/// are decoded from the transmitted [`Signed360`] codes).
fn lane_b_encode_decode(
    sprites: &[Sprite; NUM_SPRITES],
) -> (usize, usize, [[usize; NUM_ANCHORS]; NUM_SPRITES]) {
    let encoder = ResidueEncoder::new(TOTAL);
    let iframe_bytes: usize = sprites
        .iter()
        .map(|s| iframe_appearance_bytes(s).len())
        .sum();
    let mut motion_bytes = 0usize;
    let mut anchors_n = [[0usize; NUM_ANCHORS]; NUM_SPRITES];
    for (i, s) in sprites.iter().enumerate() {
        let sign = sign_enum(s.sign);
        // P-anchors 1..NUM_ANCHORS; anchor 0 is the I-frame (n=0), no motion code.
        for (k, slot) in anchors_n[i].iter_mut().enumerate().skip(1) {
            let n_true = k * ANCHOR_STEP;
            let s360 = encoder.encode_signed(s.place, n_true, sign);
            motion_bytes += s360.to_bytes().len();
            *slot = decode_signed_to_n(s.place, s360, TOTAL);
        }
    }
    (iframe_bytes, motion_bytes, anchors_n)
}

/// Reconstruct one sprite's residue index `n` at an arbitrary `frame`, from
/// its decoded anchor `n` values. Between two anchors: linear (parametric)
/// interpolation — the B-frame path, zero bits stored. Beyond the LAST
/// anchor (`n = 200..239` — the 6-anchor GOP's open tail, since 6 anchors at
/// step 40 only close the interval `[0, 200]`): extrapolate forward using
/// the last segment's slope, the decoder's only available signal past its
/// final anchor.
fn reconstruct_n(anchors_n: &[usize; NUM_ANCHORS], frame: usize) -> usize {
    let last = NUM_ANCHORS - 1;
    let last_anchor_frame = last * ANCHOR_STEP;
    if frame >= last_anchor_frame {
        let n0 = anchors_n[last - 1] as f64;
        let n1 = anchors_n[last] as f64;
        let slope = (n1 - n0) / ANCHOR_STEP as f64;
        let n = n1 + slope * (frame - last_anchor_frame) as f64;
        return n.round().clamp(0.0, (TOTAL - 1) as f64) as usize;
    }
    let seg = frame / ANCHOR_STEP;
    let f0 = seg * ANCHOR_STEP;
    let n0 = anchors_n[seg] as f64;
    let n1 = anchors_n[seg + 1] as f64;
    let t = (frame - f0) as f64 / ANCHOR_STEP as f64;
    (n0 + t * (n1 - n0)).round().clamp(0.0, (TOTAL - 1) as f64) as usize
}

/// Ground-truth render: every sprite at `n = frame` (its true φ-spiral
/// position for this frame). Thin wrapper over [`render_frame_with_n`] so
/// ground truth and lane-B reconstruction share ONE rasterizer.
fn render_frame(sprites: &[Sprite; NUM_SPRITES], frame: usize, y: &mut [u8]) {
    let per_sprite_n = [frame; NUM_SPRITES];
    render_frame_with_n(sprites, &per_sprite_n, frame, y);
}

/// The shared rasterizer: panning background (driven by `frame_for_bg`) +
/// one additive gaussian splat per sprite, each placed at its OWN
/// `per_sprite_n[i]` (ground truth passes `n = frame` for every sprite via
/// [`render_frame`]; lane B passes its reconstructed, per-sprite `n`).
/// Fills `y` (len `W * H`) in place.
fn render_frame_with_n(
    sprites: &[Sprite; NUM_SPRITES],
    per_sprite_n: &[usize; NUM_SPRITES],
    frame_for_bg: usize,
    y: &mut [u8],
) {
    // Background: a faint moving gradient so inter-frame prediction has
    // global motion to track (a static bg would make every P-frame near-zero).
    let pan = (frame_for_bg as f64 / TOTAL as f64) * 40.0;
    for py in 0..H {
        for px in 0..W {
            let g =
                24.0 + 10.0 * (((px as f64 + pan) * 0.03).sin()) + 6.0 * ((py as f64 * 0.05).cos());
            y[py * W + px] = g.clamp(0.0, 60.0) as u8;
        }
    }
    // Splat each sprite at its φ-spiral point for this frame.
    for (i, s) in sprites.iter().enumerate() {
        let n = per_sprite_n[i];
        // Canonical (x, z, y): x → screen-x, SIGNED z → screen-y (so the
        // hemisphere sign mirrors the sprite vertically), y → depth.
        let (cx_off, cz_signed, cy_depth) = phi_spiral_cart(n, TOTAL, s.sign);
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
}

fn main() -> std::io::Result<()> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "scene.y4m".into());
    let sprites = seed_sprites();

    // Lane B: encode + decode every sprite's P-anchor motion up front (needed
    // before the per-frame loop, so every frame's reconstructed n is ready).
    let (iframe_bytes, motion_bytes, anchors_n) = lane_b_encode_decode(&sprites);
    let our_total_bytes = iframe_bytes + motion_bytes;

    let f = std::fs::File::create(&path)?;
    let mut out = BufWriter::new(f);

    // Y4M header — I420, 25 fps, progressive (lane A: x265's input).
    writeln!(out, "YUV4MPEG2 W{W} H{H} F25:1 Ip A1:1 C420jpeg")?;

    let cw = W / 2;
    let ch = H / 2;
    let uv = vec![128u8; cw * ch]; // neutral chroma (grayscale scene)

    let mut y_true = vec![0u8; W * H];
    let mut y_lane_b = vec![0u8; W * H];
    let mut se_sum = 0.0f64;

    for frame in 0..TOTAL {
        // Ground truth: also lane A's Y4M payload.
        render_frame(&sprites, frame, &mut y_true);
        out.write_all(b"FRAME\n")?;
        out.write_all(&y_true)?;
        out.write_all(&uv)?; // U
        out.write_all(&uv)?; // V

        // Lane B: reconstruct this frame from the decoded anchor motion and
        // rasterize through the SAME renderer, then accumulate squared error
        // against the ground truth for the global PSNR.
        let mut per_sprite_n = [0usize; NUM_SPRITES];
        for (i, slot) in per_sprite_n.iter_mut().enumerate() {
            *slot = reconstruct_n(&anchors_n[i], frame);
        }
        render_frame_with_n(&sprites, &per_sprite_n, frame, &mut y_lane_b);

        for (a, b) in y_true.iter().zip(y_lane_b.iter()) {
            let d = *a as f64 - *b as f64;
            se_sum += d * d;
        }
    }
    out.flush()?;

    let mean_mse = se_sum / (W * H * TOTAL) as f64;
    let psnr_line = if mean_mse == 0.0 {
        "inf (bit-exact)".to_string()
    } else {
        format!("{:.2}", 10.0 * (255.0f64 * 255.0 / mean_mse).log10())
    };

    eprintln!("rendered {TOTAL} frames {W}x{H} ({NUM_SPRITES} sprites, φ-spiral motion) → {path}");
    eprintln!("HEADTOHEAD lane_b_bytes={our_total_bytes}");
    eprintln!("HEADTOHEAD lane_b_iframe_bytes={iframe_bytes}");
    eprintln!("HEADTOHEAD lane_b_motion_bytes={motion_bytes}");
    eprintln!("HEADTOHEAD lane_b_frames={TOTAL}");
    eprintln!(
        "HEADTOHEAD lane_b_bits_per_frame={:.1}",
        (our_total_bytes * 8) as f64 / TOTAL as f64
    );
    eprintln!("HEADTOHEAD lane_b_psnr_db={psnr_line}");
    eprintln!("HEADTOHEAD lane_a_y4m={path}");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// (1) `seed_sprites` is deterministic — same place/center/scale/sign
    /// stream on every call, alternating hemisphere by index parity.
    #[test]
    fn seed_sprites_is_deterministic() {
        let a = seed_sprites();
        let b = seed_sprites();
        for i in 0..NUM_SPRITES {
            assert_eq!(a[i].place, b[i].place);
            assert_eq!(a[i].c0.to_bits(), b[i].c0.to_bits());
            assert_eq!(a[i].c1.to_bits(), b[i].c1.to_bits());
            assert_eq!(a[i].c2.to_bits(), b[i].c2.to_bits());
            assert_eq!(a[i].scale_w.to_bits(), b[i].scale_w.to_bits());
            assert_eq!(a[i].sign, if i % 2 == 0 { 1.0 } else { -1.0 });
        }
    }

    /// (2) `decode_signed_to_n(encode_signed(place, n, sign), ...)` recovers
    /// `n` exactly or within a small tolerance, across both hemispheres and
    /// including the boundary indices (0, last).
    #[test]
    fn signed360_decode_recovers_n_within_tolerance() {
        let enc = ResidueEncoder::new(TOTAL);
        let cases: [(u64, usize, Sign); 6] = [
            (0x1234_5678, 0, Sign::Pos),
            (0x1234_5678, 40, Sign::Neg),
            (0x9ABC_DEF0, 120, Sign::Pos),
            (0x9ABC_DEF0, 200, Sign::Neg),
            (0x55, 160, Sign::Pos),
            (0xDEAD_BEEF, TOTAL - 1, Sign::Neg),
        ];
        for (place, n, sign) in cases {
            let s360 = enc.encode_signed(place, n, sign);
            let recovered = decode_signed_to_n(place, s360, TOTAL);
            let diff = (recovered as i64 - n as i64).unsigned_abs();
            assert!(
                diff <= 1,
                "place={place:#x} n={n} sign={sign:?} recovered={recovered} diff={diff}"
            );
        }
    }

    /// (3) Lane B's honest byte accounting: `NUM_SPRITES*24 (I-frame) +
    /// 5*NUM_SPRITES*6 (motion) == 432`, derived from real serialized bytes
    /// (`iframe_appearance_bytes` + `Signed360::to_bytes`), not a formula.
    #[test]
    fn lane_b_bytes_match_expected_total() {
        let sprites = seed_sprites();
        let (iframe_bytes, motion_bytes, _anchors_n) = lane_b_encode_decode(&sprites);
        assert_eq!(iframe_bytes, NUM_SPRITES * 24);
        // (NUM_ANCHORS - 1) P-anchors × NUM_SPRITES × one 6-byte Signed360.
        assert_eq!(motion_bytes, (NUM_ANCHORS - 1) * NUM_SPRITES * 6);
        assert_eq!(iframe_bytes + motion_bytes, 432);
    }
}
