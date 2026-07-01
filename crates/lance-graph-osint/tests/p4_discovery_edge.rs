//! P4 · discovery determinism — ARM mining joins the palette.
//!
//! P1–P3 unified the distance sources, the edge carrier, and the Pearl-mask
//! semantics onto one 256×256 palette. P4 joins the *discovery* arm: the
//! Aerial+ codebook probe (`arm-discovery`) mines association rules off the same
//! integer oracle, and its output translates into the same `CausalEdge64` wire.
//!
//! The OSINT fixture encodes a dual-use signal — `militaryUse ⟹ impact` — as a
//! perfect correlation over 1000 rows. The probe must:
//!   1. mine the known rule with **exact** integer support/confidence ppm;
//!   2. be **deterministic** — same data + same θ ⇒ byte-identical rules, no
//!      seed (the float-free discovery-path guarantee);
//!   3. translate `arm_to_truth_u8` → `CausalEdge64::pack_v2`, and the packed
//!      edge's `frequency_u8()/confidence_u8()` equal the mined `TruthU8`.
//!
//! Live-`SpoStore` promotion stays gated on D-ARM-7 (jc Pillar 5, Jirak floor);
//! this probe exercises the pure in-memory mine → truth → edge path only.

use causal_edge::{CausalEdge64, CausalMask, PlasticityState};
use lance_graph_arm_discovery::{
    arm_to_truth_u8, AerialParams, AerialProposer, Dataset, FeatureSpec, Item, MatrixDistance,
    NARS_PERSONALITY_K,
};

// Two OSINT features, cardinality 2 each: militaryUse ∈ {no,yes}, impact ∈ {low,high}.
// Global slots (block offsets): militaryUse=no→0, yes→1; impact=low→2, high→3.
const MILITARY_USE: u32 = 0; // feature index
const IMPACT: u32 = 1; // feature index

/// Oracle: militaryUse=yes(slot 1) is near impact=high(slot 3); militaryUse=no
/// near impact=low. Off-diagonal defaults to "far" so an unset cell never reads
/// as nearest. dim = 4.
fn oracle(spec: &FeatureSpec) -> MatrixDistance {
    let mut table = vec![50u32; 16];
    for d in 0..4 {
        table[d * 4 + d] = 0;
    }
    let mut set = |a: usize, b: usize, v: u32| {
        table[a * 4 + b] = v;
        table[b * 4 + a] = v;
    };
    set(0, 2, 1); // militaryUse=no  ~ impact=low
    set(1, 3, 1); // militaryUse=yes ~ impact=high
    MatrixDistance::new(spec, table)
}

/// 1000 rows, perfect dual-use correlation: militaryUse == impact.
/// → militaryUse=yes ⟹ impact=high with confidence 1.0, support 0.5.
fn osint_dataset(spec: &FeatureSpec) -> Dataset {
    let rows: Vec<Vec<u32>> = (0..1000)
        .map(|i| {
            let mu = (i % 2) as u32; // 500 no, 500 yes
            vec![mu, mu] // impact == militaryUse
        })
        .collect();
    Dataset::new(spec.clone(), rows)
}

fn params() -> AerialParams {
    AerialParams {
        theta: 2,
        max_antecedent: 1,
        min_support_ppm: 50_000,
        min_confidence_ppm: 700_000,
    }
}

#[test]
fn p4_mines_the_known_dual_use_rule_with_exact_ppm() {
    let spec = FeatureSpec::new(vec![2, 2]);
    let rules = AerialProposer::new(osint_dataset(&spec), oracle(&spec), params()).mine();
    assert!(!rules.is_empty(), "probe must mine at least one rule");

    // The known rule: militaryUse=yes ⟹ impact=high.
    let known = rules
        .iter()
        .find(|r| {
            r.antecedent == vec![Item::new(MILITARY_USE, 1)]
                && r.consequent == vec![Item::new(IMPACT, 1)]
        })
        .expect("militaryUse=yes ⟹ impact=high must be mined");

    // Exact integer evidence: 500 rows have militaryUse=yes, all 500 also
    // impact=high; window = 1000.
    assert_eq!(known.antecedent_count, 500, "|militaryUse=yes|");
    assert_eq!(known.cooccur, 500, "|militaryUse=yes ∧ impact=high|");
    assert_eq!(known.window, 1000, "window");
    assert_eq!(known.support_ppm(), 500_000, "support = 500/1000");
    assert_eq!(known.confidence_ppm(), 1_000_000, "confidence = 500/500");
}

#[test]
fn p4_mining_is_deterministic_no_seed() {
    let spec = FeatureSpec::new(vec![2, 2]);
    let r1 = AerialProposer::new(osint_dataset(&spec), oracle(&spec), params()).mine();
    let r2 = AerialProposer::new(osint_dataset(&spec), oracle(&spec), params()).mine();
    assert_eq!(r1, r2, "same data + same θ ⇒ byte-identical rules");
}

#[test]
fn p4_rule_truth_packs_into_causal_edge() {
    let spec = FeatureSpec::new(vec![2, 2]);
    let rules = AerialProposer::new(osint_dataset(&spec), oracle(&spec), params()).mine();
    let known = rules
        .iter()
        .find(|r| {
            r.antecedent == vec![Item::new(MILITARY_USE, 1)]
                && r.consequent == vec![Item::new(IMPACT, 1)]
        })
        .expect("known rule present");

    // Stage B: integer ARM evidence → quantized NARS truth (the CausalEdge64 wire).
    let truth = arm_to_truth_u8(known, NARS_PERSONALITY_K);
    // frequency = cooccur*255/antecedent_count = 500*255/500 = 255 (P=1.0).
    assert_eq!(truth.frequency, 255);
    // confidence = m*255/(m+k) = 500*255/501 = 254.
    assert_eq!(truth.confidence, ((500u64 * 255) / 501) as u8);

    // The mined rule becomes a CausalEdge64 on the same palette: subject =
    // militaryUse slot, predicate = "implies" (fixed OSINT vocab slot), object =
    // impact slot. Palette indices are illustrative here (P1 pinned their metric);
    // P4's teeth are that the discovery truth round-trips into the edge wire.
    let edge = CausalEdge64::pack_v2(
        1,  // militaryUse=yes palette index
        0,  // "implies" predicate slot
        3,  // impact=high palette index
        truth.frequency,
        truth.confidence,
        CausalMask::SPO,
        0,
        PlasticityState::ALL_FROZEN,
    );
    assert_eq!(edge.frequency_u8(), truth.frequency, "freq → edge wire");
    assert_eq!(edge.confidence_u8(), truth.confidence, "conf → edge wire");
    assert_eq!(edge.causal_mask(), CausalMask::SPO);
}
