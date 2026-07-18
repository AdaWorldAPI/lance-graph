//! PROBE-SPRITE-REPLAY-CORE — crawl-first core of the x265 sprite-replay probe.
//!
//! Cross-ref: `.claude/board/EPIPHANIES.md` `E-SPRITE-IPB-HELIX-1` (lance-graph
//! repo) and `.claude/plans/x265-sprite-replay-probe-v1.md` (ndarray repo,
//! §"Test spec (minimal, deterministic)"). This module is the CPU-native
//! core-claim test ONLY — splat3d EWA rasterization and the wgpu render tier
//! are explicitly OUT of scope here (deferred; see the plan's "Decode tiers
//! (b)/(c)" and the "GPU is render-grade only" scope guard). Accordingly this
//! probe tracks one anchor POINT per sprite (its φ-spiral position), not a
//! `K=64` Gaussian splat set — the render tier is where splats would attach.
//!
//! ## The core claim under test
//!
//! Object-level HELIX MOTION (one [`ResidueEdge`]/[`Signed360`] code per
//! sprite per P-frame) replaces x265's per-block MV field: does a single
//! helix code reproduce a sprite's motion within the register's quantization
//! bound, and does the I/P/B state sequence replay deterministically?
//!
//! ## API grounding (read this before touching the file)
//!
//! [`ResidueEncoder::encode`] / [`ResidueEncoder::encode_signed`] are
//! ONE-WAY: `(place, n) -> ResidueEdge` / `Signed360`. The crate ships NO
//! inverse — there is no `decode(edge) -> n` anywhere in `helix::residue`.
//! This module builds a decode by **re-deriving** the encoder's *private*
//! `aligned_for_residue` formula from PUBLIC primitives only
//! ([`HemispherePoint::lift`], [`Similarity::fisher_z`], [`STRIDE`],
//! [`EULER_GAMMA`], [`LN_17`]) — see [`aligned_for_residue_pub`], which
//! mirrors `residue.rs`'s private `aligned_for_residue` line for line — then
//! dequantizing the stored byte via the encoder's own
//! `RollingFloor::bucket_center` (also public, via [`ResidueEncoder::floor`])
//! and doing an exhaustive nearest-match search over `n in 0..total` (total
//! is small here, so this is cheap and exact-enough; no numerical solver
//! needed). This is NOT an invented round-trip API bolted onto helix — every
//! primitive used is public, and the reconstruction is the literal
//! mathematical inverse of the shipped quantization pipeline. See
//! [`decode_edge_to_n`] / [`decode_signed_to_n`].
//!
//! ## Ground-truth motion
//!
//! [`HemispherePoint::lift`] (or [`HemispherePoint::signed_lift`] for the
//! full sphere) IS ITSELF a golden-angle spiral winding from pole to rim as
//! `n` grows — literally a helix traced on the hemisphere by the crate's own
//! placement template. So the ground-truth path for each sprite is built
//! FROM that template (scaled + translated per sprite), rather than an
//! independently-invented 3-D curve the encoder has no way to represent:
//! `HemispherePoint::lift` couples azimuth and elevation together through a
//! single index `n` — there is no free 2-DOF direction codec in this crate,
//! so treating an arbitrary external direction as encodable would be the
//! "invented round-trip" the task instructions warn against. Using the
//! template's own `n`-indexed path keeps the probe honest to what helix
//! actually encodes: "the n-th point on the φ-spiral template", scaled to
//! the sprite's world-space envelope.
//!
//! ## Why Signed360 should beat ResidueEdge (the hemisphere-limitation test)
//!
//! [`ResidueEdge::to_bytes`]/[`ResidueEncoder::encode`] only ever represents
//! the UPPER hemisphere lift (`Signed360::rim` reuses the same
//! sign-independent unsigned pipeline — see `signed360_rim_matches_unsigned_
//! encode` in `residue.rs`'s own tests). So a 24-bit `ResidueEdge` cannot
//! carry which hemisphere a sprite's motion is in at all; sprites seeded with
//! [`Sign::Neg`] (lower hemisphere) will always be reconstructed via the
//! `ResidueEdge` path as if [`Sign::Pos`] — a real, measurable structural
//! error, not a rounding artifact. [`Signed360`] carries the sign exactly
//! (via the partition-encoded `polar` byte) plus a 16-bit azimuth field, so
//! it should reconstruct such sprites far more accurately. This is the
//! concrete content behind the plan's "measure both widths' quantization
//! error" instruction.
//!
//! ## GOP grammar
//!
//! `I B B P B B P …` over [`NUM_ANCHORS`] anchors (1 I + 5 P), with
//! [`B_FRACTIONS`] (two B-frames) of parametric interpolation between each
//! anchor pair — the x265 *operational* grammar the plan asks to replay, NOT
//! bitstream/byte parity with x265 itself (explicit plan scope guard).
//!
//! ## Pass/KILL adjudication
//!
//! This module asserts ONLY structural sanity (counts, finite values) and
//! the determinism requirement (the whole GOP is built twice from the same
//! seed; the two runs must be bit-identical). The printed table is what a
//! reviewer adjudicates against the plan's PASS/NEUTRAL/KILL bands — no
//! verdict is asserted here.
#![cfg(test)]

use crate::constants::{EULER_GAMMA, GOLDEN_RATIO, LN_17, STRIDE};
use crate::fisher_z::Similarity;
use crate::placement::{HemispherePoint, Sign};
use crate::residue::{ResidueEdge, ResidueEncoder, Signed360};

const TAU: f64 = core::f64::consts::TAU;

/// Sprite count — plan §Test spec: "N=8 sprites".
const NUM_SPRITES: usize = 8;
/// Resolution of each sprite's φ-spiral ground-truth path: `n ranges over
/// 0..TOTAL`. Kept modest so the exhaustive decode search (see module doc)
/// stays cheap and the anchor step below divides evenly.
const TOTAL: usize = 240;
/// 1 I-anchor + 5 P-anchors, evenly spaced along the path.
const NUM_ANCHORS: usize = 6;
/// Constant spacing between anchors in `n`-space (240 / 6 = 40). Known to
/// both "encoder" and "decoder" sides as part of the GOP structure — this is
/// NOT information smuggled from the ground truth; a real codec's GOP length
/// is a codec-configuration constant, not a per-frame secret.
const ANCHOR_STEP: usize = TOTAL / NUM_ANCHORS;
/// Two B-frames between each anchor pair — the "B B" of the "I B B P" GOP.
const B_FRACTIONS: [f64; 2] = [1.0 / 3.0, 2.0 / 3.0];

/// Deterministic seeded PRNG — SplitMix64 (no `rand` dependency, per the
/// edit-only/no-new-deps discipline for this drafting pass).
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

    /// Uniform `f64` in `[0, 1)`.
    fn next_f64_unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn next_f64_range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + self.next_f64_unit() * (hi - lo)
    }
}

/// Per-sprite seeded identity + placement parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
struct SpriteParams {
    /// The HHTL "place" this sprite's helix codes are anchored at — the
    /// object identity that stays fixed across the whole GOP (only `n`
    /// varies frame to frame, matching the plan's "ONE helix motion code per
    /// sprite" — the place is the sprite, `n` is where along its path).
    place: u64,
    /// Which hemisphere this sprite's motion occupies. Alternated by sprite
    /// index parity so the probe exercises BOTH the case `ResidueEdge`
    /// handles natively (Pos) and the case it structurally cannot (Neg).
    sign: Sign,
    center: [f64; 3],
    scale: f64,
}

fn seed_sprites() -> [SpriteParams; NUM_SPRITES] {
    let mut rng = SplitMix64::new(0x5350_5249_5445_5F31);
    core::array::from_fn(|i| {
        let place = rng.next_u64();
        let sign = if i % 2 == 0 { Sign::Pos } else { Sign::Neg };
        let center = [
            rng.next_f64_range(-50.0, 50.0),
            rng.next_f64_range(-50.0, 50.0),
            rng.next_f64_range(-50.0, 50.0),
        ];
        let scale = rng.next_f64_range(5.0, 25.0);
        SpriteParams {
            place,
            sign,
            center,
            scale,
        }
    })
}

/// Re-derivation of [`ResidueEncoder`]'s *private* `aligned_for_residue`,
/// built ONLY from public primitives (see module doc — mirrors
/// `residue.rs`'s private fn line for line: hemisphere rim → Fisher-Z →
/// Euler hand-off, pre-quantization).
fn aligned_for_residue_pub(n: usize, total: usize) -> f64 {
    let p = HemispherePoint::lift(n, total);
    let z = Similarity(p.rim()).fisher_z();
    let rank_frac = n as f64 / total as f64;
    z * STRIDE as f64 + EULER_GAMMA * (rank_frac - LN_17)
}

/// World-space position of sprite `n`-index `n`: the φ-spiral template point
/// (via [`HemispherePoint::signed_lift`]), scaled and translated into the
/// sprite's own envelope. Shared by ground truth AND decode-side
/// reconstruction so the only thing under test is `n`/`sign` recovery.
fn sprite_position(center: [f64; 3], scale: f64, n: usize, total: usize, sign: Sign) -> [f64; 3] {
    let p = HemispherePoint::signed_lift(n, total, sign);
    let (x, z, y) = p.cartesian();
    [
        center[0] + scale * x,
        center[1] + scale * z,
        center[2] + scale * y,
    ]
}

fn euclid(a: [f64; 3], b: [f64; 3]) -> f64 {
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}

/// Decode a [`ResidueEdge`] back to the best-matching `n` — exhaustive search
/// over `0..total` against the dequantized `end_idx` bucket center. `total`
/// is small (240) so this is cheap; see module doc for why this is the
/// honest inverse of the shipped quantization, not a shortcut.
fn decode_edge_to_n(edge: ResidueEdge, enc: &ResidueEncoder, total: usize) -> usize {
    let target = enc.floor().bucket_center(edge.end_idx);
    let mut best_n = 0usize;
    let mut best_err = f64::INFINITY;
    for n in 0..total {
        let err = (aligned_for_residue_pub(n, total) - target).abs();
        if err < best_err {
            best_err = err;
            best_n = n;
        }
    }
    best_n
}

/// Decode a [`Signed360`] back to the best-matching `n`, using the 16-bit
/// `azimuth` field (`n·φ mod 2π`) as the discriminant — far finer resolution
/// than the 8-bit rim alone, which is why Signed360 should out-perform
/// ResidueEdge on motion fidelity even setting the sign question aside.
fn decode_signed_to_n(s: Signed360, total: usize) -> usize {
    let az_target = (s.azimuth as f64 / 65536.0) * TAU;
    let mut best_n = 0usize;
    let mut best_err = f64::INFINITY;
    for n in 0..total {
        let az = (n as f64 * GOLDEN_RATIO).rem_euclid(TAU);
        let mut d = (az - az_target).abs();
        if d > TAU / 2.0 {
            d = TAU - d;
        }
        if d < best_err {
            best_err = d;
            best_n = n;
        }
    }
    best_n
}

/// One sprite's full-GOP replay result — the "sprite state" whose byte
/// identity across two independent builds is the determinism check.
#[derive(Debug, Clone, PartialEq)]
struct SpriteReplay {
    /// Ground-truth anchor `n` values (I + 5 P), ascending.
    anchors_true: [usize; NUM_ANCHORS],
    /// `n` recovered via the 24-bit `ResidueEdge` decode, per anchor.
    anchors_dec_edge: [usize; NUM_ANCHORS],
    /// `n` recovered via the 48-bit `Signed360` decode, per anchor.
    anchors_dec_signed: [usize; NUM_ANCHORS],
    /// Sign recovered via `Signed360::sign()`, per anchor (exact — the
    /// partition encoding carries no quantization loss on the sign bit).
    anchors_dec_sign: [Sign; NUM_ANCHORS],
    /// Position reconstruction error (Euclidean, world units) per anchor,
    /// ResidueEdge path (always reconstructed as `Sign::Pos` — the
    /// hemisphere-only structural limitation under test).
    pos_err_edge: [f64; NUM_ANCHORS],
    /// Position reconstruction error per anchor, Signed360 path (uses the
    /// recovered sign).
    pos_err_signed: [f64; NUM_ANCHORS],
    /// Per-B-frame (2 per anchor-gap × (NUM_ANCHORS-1) gaps) bidirectional
    /// consistency delta (forward-from-earlier-anchor vs
    /// backward-from-later-anchor), ResidueEdge-decoded anchors.
    b_bidir_edge: Vec<f64>,
    /// Same, Signed360-decoded anchors.
    b_bidir_signed: Vec<f64>,
    /// Per-B-frame error vs ground truth (average of forward/backward),
    /// ResidueEdge-decoded anchors.
    b_err_edge: Vec<f64>,
    /// Same, Signed360-decoded anchors.
    b_err_signed: Vec<f64>,
}

/// Encode + decode the full I/P/B GOP for one sprite.
fn replay_sprite(sp: &SpriteParams, enc: &ResidueEncoder, total: usize) -> SpriteReplay {
    let mut anchors_true = [0usize; NUM_ANCHORS];
    let mut anchors_dec_edge = [0usize; NUM_ANCHORS];
    let mut anchors_dec_signed = [0usize; NUM_ANCHORS];
    let mut anchors_dec_sign = [Sign::Pos; NUM_ANCHORS];
    let mut pos_err_edge = [0.0f64; NUM_ANCHORS];
    let mut pos_err_signed = [0.0f64; NUM_ANCHORS];

    for (k, slot) in anchors_true.iter_mut().enumerate() {
        *slot = (k * ANCHOR_STEP).min(total - 1);
    }

    for k in 0..NUM_ANCHORS {
        let n_true = anchors_true[k];

        // I-frame (k=0): full anchor dump. P-frame (k>0): ONE helix motion
        // code per sprite. Both are encoded identically here — the encoder
        // has no I/P distinction of its own; the GOP-grammar distinction is
        // that only k=0's anchor is *also* treated as the splat-dump base
        // (out of scope for this core probe — see module doc).
        let edge = enc.encode(sp.place, n_true);
        let s360 = enc.encode_signed(sp.place, n_true, sp.sign);

        let n_dec_edge = decode_edge_to_n(edge, enc, total);
        let n_dec_signed = decode_signed_to_n(s360, total);
        let dec_sign = s360.sign();

        let truth_pos = sprite_position(sp.center, sp.scale, n_true, total, sp.sign);
        // ResidueEdge structurally cannot carry hemisphere sign — always
        // reconstructed as Sign::Pos (this is the point of the test).
        let edge_pos = sprite_position(sp.center, sp.scale, n_dec_edge, total, Sign::Pos);
        let signed_pos = sprite_position(sp.center, sp.scale, n_dec_signed, total, dec_sign);

        anchors_dec_edge[k] = n_dec_edge;
        anchors_dec_signed[k] = n_dec_signed;
        anchors_dec_sign[k] = dec_sign;
        pos_err_edge[k] = euclid(truth_pos, edge_pos);
        pos_err_signed[k] = euclid(truth_pos, signed_pos);
    }

    // B-frames: parametric interpolation between surrounding anchors, no
    // stored motion of their own. Bidirectional check: extrapolate forward
    // from the earlier anchor's decoded n (+ the known constant anchor
    // spacing) vs backward from the later anchor's decoded n; the two must
    // agree to a tolerance a reviewer adjudicates from the printed table.
    let mut b_bidir_edge = Vec::with_capacity((NUM_ANCHORS - 1) * B_FRACTIONS.len());
    let mut b_bidir_signed = Vec::with_capacity((NUM_ANCHORS - 1) * B_FRACTIONS.len());
    let mut b_err_edge = Vec::with_capacity((NUM_ANCHORS - 1) * B_FRACTIONS.len());
    let mut b_err_signed = Vec::with_capacity((NUM_ANCHORS - 1) * B_FRACTIONS.len());

    for k in 0..NUM_ANCHORS - 1 {
        for &frac in &B_FRACTIONS {
            let n_b_true_f = anchors_true[k] as f64 + frac * ANCHOR_STEP as f64;
            let n_b_true = n_b_true_f.round().clamp(0.0, (total - 1) as f64) as usize;
            let truth_b_pos = sprite_position(sp.center, sp.scale, n_b_true, total, sp.sign);

            // ResidueEdge width (Sign::Pos reconstruction throughout).
            let n_fwd_edge = ((anchors_dec_edge[k] as f64) + frac * ANCHOR_STEP as f64)
                .round()
                .clamp(0.0, (total - 1) as f64) as usize;
            let n_bwd_edge = ((anchors_dec_edge[k + 1] as f64) - (1.0 - frac) * ANCHOR_STEP as f64)
                .round()
                .clamp(0.0, (total - 1) as f64) as usize;
            let pos_fwd_edge = sprite_position(sp.center, sp.scale, n_fwd_edge, total, Sign::Pos);
            let pos_bwd_edge = sprite_position(sp.center, sp.scale, n_bwd_edge, total, Sign::Pos);
            b_bidir_edge.push(euclid(pos_fwd_edge, pos_bwd_edge));
            b_err_edge.push(
                0.5 * (euclid(pos_fwd_edge, truth_b_pos) + euclid(pos_bwd_edge, truth_b_pos)),
            );

            // Signed360 width (recovered sign at each anchor).
            let n_fwd_signed = ((anchors_dec_signed[k] as f64) + frac * ANCHOR_STEP as f64)
                .round()
                .clamp(0.0, (total - 1) as f64) as usize;
            let n_bwd_signed = ((anchors_dec_signed[k + 1] as f64)
                - (1.0 - frac) * ANCHOR_STEP as f64)
                .round()
                .clamp(0.0, (total - 1) as f64) as usize;
            let sign_fwd = anchors_dec_sign[k];
            let sign_bwd = anchors_dec_sign[k + 1];
            let pos_fwd_signed =
                sprite_position(sp.center, sp.scale, n_fwd_signed, total, sign_fwd);
            let pos_bwd_signed =
                sprite_position(sp.center, sp.scale, n_bwd_signed, total, sign_bwd);
            b_bidir_signed.push(euclid(pos_fwd_signed, pos_bwd_signed));
            b_err_signed.push(
                0.5 * (euclid(pos_fwd_signed, truth_b_pos) + euclid(pos_bwd_signed, truth_b_pos)),
            );
        }
    }

    SpriteReplay {
        anchors_true,
        anchors_dec_edge,
        anchors_dec_signed,
        anchors_dec_sign,
        pos_err_edge,
        pos_err_signed,
        b_bidir_edge,
        b_bidir_signed,
        b_err_edge,
        b_err_signed,
    }
}

/// Build the whole GOP replay for all [`NUM_SPRITES`] sprites — the unit
/// re-run twice for the determinism check.
fn build_replay() -> Vec<SpriteReplay> {
    let sprites = seed_sprites();
    let enc = ResidueEncoder::new(TOTAL);
    sprites
        .iter()
        .map(|sp| replay_sprite(sp, &enc, TOTAL))
        .collect()
}

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        0.0
    } else {
        xs.iter().sum::<f64>() / xs.len() as f64
    }
}

fn max_of(xs: &[f64]) -> f64 {
    xs.iter().cloned().fold(0.0f64, f64::max)
}

#[test]
fn probe_sprite_replay_core() {
    let run_a = build_replay();
    let run_b = build_replay();

    // ── determinism: bit-identical sprite states across two independent
    // builds from the same seed, same ops, same order. ──
    assert_eq!(
        run_a, run_b,
        "PROBE-SPRITE-REPLAY-CORE determinism check failed: rebuilding the \
         GOP from the same seed produced different sprite states"
    );

    // ── structural sanity only (not the pass/KILL verdict) ──
    assert_eq!(run_a.len(), NUM_SPRITES);
    for r in &run_a {
        assert_eq!(r.anchors_true.len(), NUM_ANCHORS);
        assert_eq!(r.b_bidir_edge.len(), (NUM_ANCHORS - 1) * B_FRACTIONS.len());
        assert_eq!(
            r.b_bidir_signed.len(),
            (NUM_ANCHORS - 1) * B_FRACTIONS.len()
        );
        for &v in r
            .pos_err_edge
            .iter()
            .chain(r.pos_err_signed.iter())
            .chain(r.b_bidir_edge.iter())
            .chain(r.b_bidir_signed.iter())
            .chain(r.b_err_edge.iter())
            .chain(r.b_err_signed.iter())
        {
            assert!(v.is_finite(), "non-finite error value encountered: {v}");
        }
    }

    // ── the printed table — the reviewer's adjudication surface, not an
    // asserted verdict (per plan: PASS/NEUTRAL/KILL bands are read by a
    // human/agent reviewer against this table, never self-asserted here). ──
    eprintln!(
        "=== PROBE-SPRITE-REPLAY-CORE (N={NUM_SPRITES} sprites, TOTAL={TOTAL}, \
         NUM_ANCHORS={NUM_ANCHORS} [I+5P], ANCHOR_STEP={ANCHOR_STEP}, GOP=I B B P ...) ==="
    );

    eprintln!("-- per-sprite P/I-anchor motion fidelity (world-unit Euclidean position error) --");
    eprintln!(
        "{:<4} {:<5} {:<8} {:>10} {:>10} {:>10} {:>10}",
        "spr", "sign", "anchor", "n_true", "n_edge", "n_s360", "sign_s360"
    );
    let mut all_edge_err = Vec::new();
    let mut all_signed_err = Vec::new();
    for (i, r) in run_a.iter().enumerate() {
        let sprites = seed_sprites();
        let sign_label = if sprites[i].sign == Sign::Pos {
            "Pos"
        } else {
            "Neg"
        };
        for k in 0..NUM_ANCHORS {
            let kind = if k == 0 { "I" } else { "P" };
            eprintln!(
                "{i:<4} {sign_label:<5} {kind:<8} {:>10} {:>10} {:>10} {:>10?}",
                r.anchors_true[k],
                r.anchors_dec_edge[k],
                r.anchors_dec_signed[k],
                r.anchors_dec_sign[k]
            );
            eprintln!(
                "     err_edge(24b)={:>10.6}  err_signed(48b)={:>10.6}",
                r.pos_err_edge[k], r.pos_err_signed[k]
            );
            all_edge_err.push(r.pos_err_edge[k]);
            all_signed_err.push(r.pos_err_signed[k]);
        }
    }

    eprintln!("-- aggregate motion fidelity (I+P anchors, all sprites) --");
    eprintln!(
        "  ResidueEdge  (24-bit hemisphere): mean={:.6} max={:.6}",
        mean(&all_edge_err),
        max_of(&all_edge_err)
    );
    eprintln!(
        "  Signed360    (48-bit full-sphere): mean={:.6} max={:.6}",
        mean(&all_signed_err),
        max_of(&all_signed_err)
    );

    // Split by true sign to surface the hemisphere-limitation effect.
    let mut edge_pos_sign = Vec::new();
    let mut edge_neg_sign = Vec::new();
    let mut signed_pos_sign = Vec::new();
    let mut signed_neg_sign = Vec::new();
    let sprites = seed_sprites();
    for (i, r) in run_a.iter().enumerate() {
        for k in 0..NUM_ANCHORS {
            if sprites[i].sign == Sign::Pos {
                edge_pos_sign.push(r.pos_err_edge[k]);
                signed_pos_sign.push(r.pos_err_signed[k]);
            } else {
                edge_neg_sign.push(r.pos_err_edge[k]);
                signed_neg_sign.push(r.pos_err_signed[k]);
            }
        }
    }
    eprintln!("-- split by TRUE sign (the hemisphere-limitation test) --");
    eprintln!(
        "  Sign::Pos sprites — ResidueEdge mean={:.6} max={:.6} | Signed360 mean={:.6} max={:.6}",
        mean(&edge_pos_sign),
        max_of(&edge_pos_sign),
        mean(&signed_pos_sign),
        max_of(&signed_pos_sign)
    );
    eprintln!(
        "  Sign::Neg sprites — ResidueEdge mean={:.6} max={:.6} | Signed360 mean={:.6} max={:.6}",
        mean(&edge_neg_sign),
        max_of(&edge_neg_sign),
        mean(&signed_neg_sign),
        max_of(&signed_neg_sign)
    );

    eprintln!("-- B-frame bidirectional consistency (forward-from-earlier-anchor vs backward-from-later-anchor) --");
    let mut all_bidir_edge = Vec::new();
    let mut all_bidir_signed = Vec::new();
    let mut all_berr_edge = Vec::new();
    let mut all_berr_signed = Vec::new();
    for r in &run_a {
        all_bidir_edge.extend_from_slice(&r.b_bidir_edge);
        all_bidir_signed.extend_from_slice(&r.b_bidir_signed);
        all_berr_edge.extend_from_slice(&r.b_err_edge);
        all_berr_signed.extend_from_slice(&r.b_err_signed);
    }
    eprintln!(
        "  ResidueEdge  bidir-delta: mean={:.6} max={:.6} | vs-truth: mean={:.6} max={:.6}",
        mean(&all_bidir_edge),
        max_of(&all_bidir_edge),
        mean(&all_berr_edge),
        max_of(&all_berr_edge)
    );
    eprintln!(
        "  Signed360    bidir-delta: mean={:.6} max={:.6} | vs-truth: mean={:.6} max={:.6}",
        mean(&all_bidir_signed),
        max_of(&all_bidir_signed),
        mean(&all_berr_signed),
        max_of(&all_berr_signed)
    );

    eprintln!("=== end PROBE-SPRITE-REPLAY-CORE — reviewer adjudicates PASS/NEUTRAL/KILL from the table above ===");
}
