//! Calibrated SoA member specs — what the substrate must carry to reproduce the
//! study, derived from the `calibrate` + `resilience` examples (operator-authorized
//! additive design, 2026-06-16).
//!
//! This is a **spec**, not a runtime encoder: each [`SoaMemberSpec`] records the
//! member width + encoding + normalization that the calibration certified for one
//! study axis (ICC/Spearman/Cronbach against the deterministic study as ground
//! truth). It is the bridge artifact a contract-side change would consume; nothing
//! here serializes or touches the operator-locked `canonical_node` spine.
//!
//! **Orthogonality to topology is structural, not earned (operator, 2026-06-16).**
//! In the HHTL-OGAR GUID model, **topology lives in the KEY** — the HEEL/HIP/TWIG
//! cascade tiers of the `canonical_node` GUID — and the magnitude axes are **helix
//! value members hung off that key**. So any helix-residue value member is
//! orthogonal to topology *by the key/value split itself*; the resilience study's
//! measured `Spearman(λ₂, buffer) ≈ 0` only **confirms** what the GUID addressing
//! already enforces, it does not establish a new axis. Consequence: `inertia_buffer`
//! is NOT a novel "orthogonal column" — it is just **another helix value slot on
//! the HHTL key**, additive in the trivial sense (one more value member), with its
//! topology-orthogonality free.
//!
//! Two findings shape it:
//! 1. **The existing value tenants suffice.** All five contingency factors certify
//!    by value at **2-bit linear, stored normalized** (ICC ≥ 0.96) — a 2-bit
//!    turbovec/palette slot per factor preserves the study's per-axis values. The
//!    cross-axis structure (α / discriminant) wants ≥6-bit, so the *read budget*
//!    where orthogonality is judged is wider than the *store budget* per value.
//! 2. **The "new" member is just a helix value slot.** `inertia_buffer`
//!    ([`INERTIA`]) is added as one more helix-residue value member on the HHTL-OGAR
//!    key; its orthogonality to the topology (which the key carries) is structural,
//!    confirmed by the study, not introduced by it.
//! 3. **It maps onto two existing carriers (operator, 2026-06-16).** The substrate
//!    already offers the two tiers the calibration asks for: **16 × 8-bit
//!    `ResidueEdge` slots** (the helix-residue value members of the EdgeBlock) for
//!    the structure *read* budget (`read_bits ≤ 8`), and **32 × 4-bit turbovec
//!    lanes** (pairwise turboquant) for the per-value *store* (`store_bits ≤ 4` —
//!    `infight`'s certified 4-bit is exactly one lane). Six members fit either
//!    carrier with headroom (6/16 ResidueEdges or 6/32 turbovec lanes), so no new
//!    layout is needed — only the slot assignments (asserted in tests).

/// How a member quantizes its normalized value. Every variant is a **value tenant
/// hung off the HHTL-OGAR GUID key** — topology is the key, so all of these are
/// orthogonal to topology by construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Encoding {
    /// Min-max linear bins (palette / Signed360 rim) — sufficient when the
    /// normalized distribution is not pathologically skewed.
    Linear,
    /// Equal-population / codebook bins (turbovec, CAM-PQ) — resolution follows the
    /// data; preferred when the raw distribution is heavy-tailed.
    DataAdaptive,
    /// Helix `Signed360` residue value member — the canonical magnitude tenant on
    /// the HHTL-OGAR key.
    HelixResidue,
}

/// One calibrated SoA member: the width + encoding the study certifies for an axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SoaMemberSpec {
    /// Axis name (the study factor it carries).
    pub name: &'static str,
    /// Store width in bits (per-value fidelity certified at this budget).
    pub store_bits: u32,
    /// Read budget where the cross-axis structure (α / discriminant) is certified.
    pub read_bits: u32,
    /// Member encoding.
    pub encoding: Encoding,
    /// Members store normalized `[0,1]` values, not raw physical units (also lifts
    /// tiny-magnitude axes out of the ICC variance-underflow guard).
    pub normalized: bool,
    /// `true` if this is a NEW value slot to add to the substrate. Orthogonality to
    /// topology is NOT a property of the member — it is structural, given by the
    /// HHTL-OGAR key/value split (topology in the key, this in the value).
    pub additive: bool,
}

/// The five contingency factors as EXISTING value tenants, normalized, read at
/// ≥6-bit to keep the orthogonality crisp. Store width = the **robustly certified**
/// width (ICC ≥ 0.95 across inputs), NOT the cheapest that squeaks by on one sample:
/// four factors certify at 2-bit, but `infight` is marginal — its 2-bit ICC ranges
/// 0.93 (synthetic grid) to 0.96 (ES core), straddling the threshold, so its spec is
/// the robust **4-bit** (≥0.99 on both). (Codex #511 P2.)
pub const CONTINGENCY_FACTORS: [SoaMemberSpec; 5] = [
    spec("d_lambda2", 2, false),
    spec("dk_rotation", 2, false),
    spec("d_conductance", 2, false),
    spec("infight", 4, false), // marginal at 2-bit (0.93–0.96) → certified width is 4-bit
    spec("raumgewinn", 2, false),
];

/// The one additive member: the inertia/buffer axis (resilience study), added as a
/// helix-residue value slot on the HHTL-OGAR key. Its orthogonality to topology is
/// STRUCTURAL (topology is the key; this is a value) — the study's `Spearman ≈ 0`
/// confirms it. `additive` here means "one more value slot", not "a new axis type".
pub const INERTIA: SoaMemberSpec = SoaMemberSpec {
    name: "inertia_buffer",
    store_bits: 2,
    read_bits: 6,
    encoding: Encoding::HelixResidue,
    normalized: true,
    additive: true,
};

const fn spec(name: &'static str, store_bits: u32, additive: bool) -> SoaMemberSpec {
    SoaMemberSpec {
        name,
        store_bits,
        read_bits: 6,
        encoding: Encoding::Linear,
        normalized: true,
        additive,
    }
}

/// The full calibrated member set the substrate needs to reproduce the study: the
/// five existing-tenant factors + the one additive inertia member.
pub fn study_member_specs() -> Vec<SoaMemberSpec> {
    let mut v = CONTINGENCY_FACTORS.to_vec();
    v.push(INERTIA);
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn value_members_carry_their_robustly_certified_width() {
        // The calibration finding: normalized members certify at low bits — 2-bit
        // for four factors, but `infight` is marginal (2-bit ICC 0.93–0.96 across
        // inputs) so it carries the robust 4-bit. All map to existing tenants.
        for s in CONTINGENCY_FACTORS {
            let expect = if s.name == "infight" { 4 } else { 2 };
            assert_eq!(s.store_bits, expect, "{} certified store width", s.name);
            assert!(s.normalized, "{} must be normalized", s.name);
            assert!(!s.additive, "{} maps to an existing tenant", s.name);
        }
    }

    #[test]
    fn structure_read_budget_exceeds_store_budget() {
        // Cross-axis orthogonality wants more bits than per-value fidelity.
        for s in study_member_specs() {
            assert!(s.read_bits >= s.store_bits, "{} read ≥ store", s.name);
            assert!(s.read_bits >= 6, "{} structure read ≥ 6-bit", s.name);
        }
    }

    #[test]
    fn fits_substrate_carriers() {
        // store fits a 4-bit turbovec lane; read fits an 8-bit ResidueEdge slot;
        // the whole set fits both carriers (≤32 turbovec lanes, ≤16 ResidueEdges).
        let specs = study_member_specs();
        assert!(specs.len() <= 16, "fits the 16 ResidueEdge slots");
        assert!(specs.len() <= 32, "fits the 32 turbovec lanes");
        for s in &specs {
            assert!(
                s.store_bits <= 4,
                "{} store fits a 4-bit turbovec lane",
                s.name
            );
            assert!(
                s.read_bits <= 8,
                "{} read fits an 8-bit ResidueEdge slot",
                s.name
            );
        }
    }

    #[test]
    fn inertia_is_the_one_additive_member() {
        let specs = study_member_specs();
        let additive: Vec<_> = specs.iter().filter(|s| s.additive).collect();
        assert_eq!(additive.len(), 1, "exactly one new member required");
        assert_eq!(additive[0].name, "inertia_buffer");
    }
}
