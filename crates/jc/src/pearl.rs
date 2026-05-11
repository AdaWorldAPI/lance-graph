//! Pillar 5b — direct Pearl 2³ mask-classification accuracy.
//!
//! Pillar 5 measures Berry-Esseen *sup-error* inflation under weak
//! dependence (an upstream proxy). This module measures the
//! **downstream task effect** that matters: how often does Pearl-mask
//! classification (SPO 2³ configuration) disagree between
//!
//!   (A) three disjoint-plane Hamming popcounts (Index regime —
//!       each role lives in its own lossless plane, no interference), vs
//!   (B) single-bundle VSA superposition (Argmax / CAM-PQ-shaped regime
//!       — the three role contributions superpose in one shared plane,
//!       with unbind-interference AND shared-codebook contamination).
//!
//! Ground truth: for each trial, an SPO mask m ∈ {0..8} is drawn. A
//! content fingerprint c is generated, plus three role-identity
//! fingerprints R_s, R_p, R_o. Active role k ∈ {S,P,O} binds content
//! into its plane as c ⊕ R_k; inactive role carries independent noise.
//!
//! Method A — **three lossless planes** (Index regime):
//!
//!   plane_s = (c ⊕ R_s) if S active else noise
//!   plane_p = (c ⊕ R_p) if P active else noise
//!   plane_o = (c ⊕ R_o) if O active else noise
//!
//! Classification: `hamming(plane_k ⊕ R_k, c) < threshold` per role.
//! No interference — each plane carries exactly one role's signal.
//!
//! Method B — **one bundled plane** (Argmax regime):
//!
//!   bundle = Σ_active (c ⊕ R_k)   (XOR superposition)
//!   bundle ⊕= codebook_bias         (CAM-PQ shared-codebook noise)
//!
//! Classification: `hamming(bundle ⊕ R_k, c) < threshold` per role.
//! Interference from other active roles + codebook bias corrupts
//! unbind. This is exactly the VSA bundled-binding failure mode at
//! high multiplicity, which the I-VSA-IDENTITIES iron rule warns
//! against.
//!
//! The mask-misclassification GAP is the direct task cost of
//! collapsing three lossless identity planes into one compressed code.

use crate::PillarResult;

const D_BITS: usize = 16_384;
const D_BYTES: usize = D_BITS / 8;
const N_TRIALS: usize = 4_000;

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn fingerprint(seed: u64) -> Vec<u8> {
    let mut fp = vec![0u8; D_BYTES];
    let mut s = seed;
    for chunk in fp.chunks_exact_mut(8) {
        let r = splitmix64(&mut s);
        chunk.copy_from_slice(&r.to_le_bytes());
    }
    fp
}

fn xor(a: &[u8], b: &[u8]) -> Vec<u8> {
    a.iter().zip(b).map(|(&x, &y)| x ^ y).collect()
}

fn xor_into(dst: &mut [u8], src: &[u8]) {
    for (d, &s) in dst.iter_mut().zip(src) {
        *d ^= s;
    }
}

fn hamming(a: &[u8], b: &[u8]) -> u32 {
    a.iter().zip(b).map(|(&x, &y)| (x ^ y).count_ones()).sum()
}

/// Classify Pearl mask from three independent lossless planes.
/// Each plane carries exactly one role's binding — no interference.
fn classify_three_planes(
    plane_s: &[u8],
    plane_p: &[u8],
    plane_o: &[u8],
    content: &[u8],
    r_s: &[u8],
    r_p: &[u8],
    r_o: &[u8],
    threshold: u32,
) -> u8 {
    // Unbind: XOR role identity out of the plane. If plane = c ⊕ R_k,
    // then plane ⊕ R_k = c → Hamming to content ≈ 0. Otherwise noise.
    let s_active = hamming(&xor(plane_s, r_s), content) < threshold;
    let p_active = hamming(&xor(plane_p, r_p), content) < threshold;
    let o_active = hamming(&xor(plane_o, r_o), content) < threshold;
    (s_active as u8) | ((p_active as u8) << 1) | ((o_active as u8) << 2)
}

/// Classify Pearl mask from single bundled plane. All three roles
/// superpose in the same register — unbind carries interference from
/// the other active roles, scaled by the codebook contamination.
fn classify_bundled(
    bundle: &[u8],
    content: &[u8],
    r_s: &[u8],
    r_p: &[u8],
    r_o: &[u8],
    threshold: u32,
) -> u8 {
    let s_active = hamming(&xor(bundle, r_s), content) < threshold;
    let p_active = hamming(&xor(bundle, r_p), content) < threshold;
    let o_active = hamming(&xor(bundle, r_o), content) < threshold;
    (s_active as u8) | ((p_active as u8) << 1) | ((o_active as u8) << 2)
}

pub fn prove() -> PillarResult {
    // Role identity fingerprints — full-size, pseudorandom. These are
    // NOT disjoint slice masks; they're the real VSA role-key identities.
    let r_s = fingerprint(0xA1);
    let r_p = fingerprint(0xA2);
    let r_o = fingerprint(0xA3);

    // Shared-codebook bias — represents CAM-PQ's 4096-centroid shared
    // codebook contamination when multiple roles project through it.
    let codebook_bias = fingerprint(0xDEAD_BEEF_CAFE);

    let mut three_plane_hits = 0usize;
    let mut bundled_hits = 0usize;
    let mut three_plane_bits_correct = 0usize;
    let mut bundled_bits_correct = 0usize;

    // Threshold: random Hamming ~ D_BITS / 2 = 8192.
    // c ⊕ c = 0 → Hamming 0. c ⊕ noise → Hamming ~8192.
    // Bundle interference with k active roles pushes distance toward
    // ~D/2. Threshold at 35 % of D separates "matched" from "noise".
    let threshold = (D_BITS as u32 * 35) / 100;

    for trial in 0..N_TRIALS {
        // Mask cycles 0..7 deterministically across trials.
        // Promote to u32 first to avoid u8 overflow at trial=7 (7*37=259 > 255)
        // in debug builds; final value is bounded to 3 bits regardless.
        let mask = ((trial as u32 * 37 + 11) & 0b111) as u8;
        let content = fingerprint(trial as u64 * 8 + 100);

        // Method A — three disjoint lossless planes.
        let plane_s = if mask & 0b001 != 0 {
            xor(&content, &r_s)
        } else {
            fingerprint(trial as u64 * 8 + 101)
        };
        let plane_p = if mask & 0b010 != 0 {
            xor(&content, &r_p)
        } else {
            fingerprint(trial as u64 * 8 + 102)
        };
        let plane_o = if mask & 0b100 != 0 {
            xor(&content, &r_o)
        } else {
            fingerprint(trial as u64 * 8 + 103)
        };

        // Method B — single bundled plane. Active roles XOR-superpose
        // (c ⊕ R_k); contamination XOR'd in last.
        let mut bundle = vec![0u8; D_BYTES];
        if mask & 0b001 != 0 {
            xor_into(&mut bundle, &xor(&content, &r_s));
        }
        if mask & 0b010 != 0 {
            xor_into(&mut bundle, &xor(&content, &r_p));
        }
        if mask & 0b100 != 0 {
            xor_into(&mut bundle, &xor(&content, &r_o));
        }
        xor_into(&mut bundle, &codebook_bias);

        let recovered_a = classify_three_planes(
            &plane_s, &plane_p, &plane_o, &content, &r_s, &r_p, &r_o, threshold,
        );
        if recovered_a == mask {
            three_plane_hits += 1;
        }
        three_plane_bits_correct += 3 - (recovered_a ^ mask).count_ones() as usize;

        let recovered_b = classify_bundled(&bundle, &content, &r_s, &r_p, &r_o, threshold);
        if recovered_b == mask {
            bundled_hits += 1;
        }
        bundled_bits_correct += 3 - (recovered_b ^ mask).count_ones() as usize;
    }

    let n = N_TRIALS as f64;
    let three_plane_acc = three_plane_hits as f64 / n;
    let bundled_acc = bundled_hits as f64 / n;
    let three_plane_bit_acc = three_plane_bits_correct as f64 / (3.0 * n);
    let bundled_bit_acc = bundled_bits_correct as f64 / (3.0 * n);
    let mask_accuracy_gap = three_plane_acc - bundled_acc;
    let bit_accuracy_gap = three_plane_bit_acc - bundled_bit_acc;

    // Pass criteria:
    // (1) Three-plane mask accuracy > 0.95 (lossless regime works cleanly)
    // (2) Three-plane beats bundled at full mask classification.
    //     The gap IS the Pearl 2³ decomposition premium.
    let pass = three_plane_acc > 0.95 && three_plane_acc >= bundled_acc;

    PillarResult {
        name: "Pearl 2³ mask-accuracy",
        pass,
        measured: mask_accuracy_gap,
        predicted: 0.0,
        detail: format!(
            "N={N_TRIALS}, d={D_BITS}, threshold={threshold} (~35 % of D): \
             three-plane mask-acc = {three_plane_acc:.4} (bit-acc {three_plane_bit_acc:.4}), \
             bundled mask-acc = {bundled_acc:.4} (bit-acc {bundled_bit_acc:.4}). \
             Mask gap = {mask_accuracy_gap:+.4} ({pct:+.1} percentage points). \
             Bit gap = {bit_accuracy_gap:+.4}. \
             Three-plane ≥ bundled ⇒ Index regime wins at Pearl 2³ addressability. \
             Interpretation: three lossless planes carry one role each (no VSA \
             superposition interference); the bundled plane must disentangle via \
             unbind, which is corrupted by other active roles + shared-codebook \
             bias. Pillar 5 quantified the upstream sup-error inflation; this \
             pillar quantifies the downstream mask-misclassification rate.",
            pct = mask_accuracy_gap * 100.0,
        ),
        runtime_ms: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn three_plane_wins_or_ties_pearl_mask() {
        let r = prove();
        assert!(r.pass, "Pearl 2³ pillar failed: {}", r.detail);
    }
}
