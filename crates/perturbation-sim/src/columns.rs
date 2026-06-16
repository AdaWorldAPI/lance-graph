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
//! Two findings shape it:
//! 1. **The existing value tenants suffice.** All five contingency factors certify
//!    by value at **2-bit linear, stored normalized** (ICC ≥ 0.96) — a 2-bit
//!    turbovec/palette slot per factor preserves the study's per-axis values. The
//!    cross-axis structure (α / discriminant) wants ≥6-bit, so the *read budget*
//!    where orthogonality is judged is wider than the *store budget* per value.
//! 2. **One genuinely additive column.** The resilience study measured the
//!    inertia/buffer axis ORTHOGONAL to topology (`Spearman(λ₂, buffer) ≈ 0`), so
//!    no existing connectivity member can carry it — it requires its own member
//!    ([`INERTIA`], flagged [`additive`](SoaMemberSpec::additive)).

/// How a member quantizes its normalized value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Encoding {
    /// Min-max linear bins (palette / Signed360 rim) — sufficient when the
    /// normalized distribution is not pathologically skewed.
    Linear,
    /// Equal-population / codebook bins (turbovec, CAM-PQ) — resolution follows the
    /// data; preferred when the raw distribution is heavy-tailed.
    DataAdaptive,
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
    /// `true` if this member does NOT exist in the current substrate and must be
    /// added (it is statistically orthogonal to every existing column).
    pub additive: bool,
}

/// The five contingency factors — all certify at 2-bit linear, normalized; read at
/// ≥6-bit to keep the orthogonality crisp. These map onto EXISTING value tenants.
pub const CONTINGENCY_FACTORS: [SoaMemberSpec; 5] = [
    spec("d_lambda2", false),
    spec("dk_rotation", false),
    spec("d_conductance", false),
    spec("infight", false),
    spec("raumgewinn", false),
];

/// The one genuinely additive member: the inertia/buffer axis (resilience study),
/// orthogonal to topology — no existing connectivity column carries it.
pub const INERTIA: SoaMemberSpec = SoaMemberSpec {
    name: "inertia_buffer",
    store_bits: 2,
    read_bits: 6,
    encoding: Encoding::Linear,
    normalized: true,
    additive: true,
};

const fn spec(name: &'static str, additive: bool) -> SoaMemberSpec {
    SoaMemberSpec {
        name,
        store_bits: 2,
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
    fn value_members_certify_at_two_bits_normalized() {
        // The calibration finding: 2-bit linear normalized suffices per value.
        for s in CONTINGENCY_FACTORS {
            assert_eq!(s.store_bits, 2, "{} store width", s.name);
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
    fn inertia_is_the_one_additive_member() {
        let specs = study_member_specs();
        let additive: Vec<_> = specs.iter().filter(|s| s.additive).collect();
        assert_eq!(additive.len(), 1, "exactly one new member required");
        assert_eq!(additive[0].name, "inertia_buffer");
    }
}
