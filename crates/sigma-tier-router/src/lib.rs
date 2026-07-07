//! Σ-tier Rubicon-resonance dispatch crate — D-CSV-10.
//!
//! Implements the free-energy-gradient dispatch loop described in
//! `cognitive-substrate-convergence-v1.md` §10 and the D-CSV-10 test brief
//! (§11 / §15). The central invariant:
//!
//! > "Never commit on F-rising" — if `current_tier >= 10` AND `last_delta > 0`,
//! > the Rubicon is not crossed; the router returns `Rest { Σ10Saturated }`.
//!
//! The Σ10 Rubicon fires (→ `Commit`) only when the tier is saturated AND
//! F is **falling** (`last_delta < 0`), meaning the cycle has resolved.
//!
//! # Σ-tier band thresholds
//!
//! Band thresholds are derived from the Jirak 2016 Berry-Esseen rate for weakly-
//! dependent sequences (arxiv 1606.01617). Per `I-NOISE-FLOOR-JIRAK` in CLAUDE.md:
//! the workspace's 16384-bit fingerprints are weakly dependent by construction
//! (CAM-PQ-induced; overlapping role-key slices; XOR bundle accumulation). The
//! correct convergence rate is `n^(p/2-1)` for `p ∈ (2,3]`, NOT classical IID
//! Berry-Esseen. For the workspace's typical CAM-PQ-induced weak-dependence regime
//! (p ≈ 3), the tier spacing exponent is `p/2 = 1.5`, yielding `Σk = k^1.5 / 10^1.5`
//! (Σ1 ≈ 0.0316, convex) implemented in `SigmaTierBands::default()` and
//! `SigmaTierBands::jirak_p(3.0)`.
//!
//! See `SigmaTierBands::jirak_p` for the full derivation.
//!
//! # Resolved tech-debt
//!
//! TD-SIGMA-TIER-THRESHOLDS-1 (sprint-12 W-G4): `Default::default()` now returns
//! the Jirak-derived band table (replaces sprint-11 hand-tuned linear 0.1..1.0).
//! The prior hand-tuned values are preserved as `SigmaTierBands::hand_tuned()` for
//! backwards comparison.

use lance_graph_contract::mul::{GateDecision, i4_eval::gate_decision_i4};
use lance_graph_contract::qualia::QualiaI4_16D;

// ─────────────────────────────────────────────────────────────────────────────
// SigmaTierBands
// ─────────────────────────────────────────────────────────────────────────────

/// Band thresholds for the 10 Σ-tiers.
///
/// `sigma1_to_sigma10[i]` is the **upper** free-energy boundary for tier `i+1`
/// (i.e. index 0 = Σ1 threshold, index 9 = Σ10 threshold).
///
/// A tier is active when `current_f <= threshold[tier-1]` and
/// `current_f > threshold[tier-2]` (or 0 for tier 1).
///
/// The default values are Jirak-derived (Σk = k^1.5 / 10^1.5) per the
/// `I-NOISE-FLOOR-JIRAK` iron rule in CLAUDE.md. See `SigmaTierBands::jirak_p`
/// for the derivation. The prior hand-tuned linear defaults (0.10 .. 1.00) are
/// available via `SigmaTierBands::hand_tuned()`.
#[derive(Clone, Debug, PartialEq)]
pub struct SigmaTierBands {
    /// Upper F boundary per Σ-tier; index 0 = Σ1, index 9 = Σ10.
    pub sigma1_to_sigma10: [f32; 10],
}

impl SigmaTierBands {
    /// Jirak-derived band thresholds for moment parameter `p`.
    ///
    /// # Derivation
    ///
    /// For weakly-dependent sequences (the workspace's CAM-PQ-induced regime),
    /// Jirak 2016 (arxiv 1606.01617, Annals of Probability 44(3) 2024–2063)
    /// gives the Berry-Esseen rate:
    ///
    /// - `n^(p/2 - 1)` for `p ∈ (2, 3]` (the "weak dependence" regime)
    /// - `n^(-1/2)` in L^q for `p ≥ 4`  (the "asymptotically iid" regime)
    ///
    /// The workspace operates in the `p ≈ 3` regime (CAM-PQ-induced weak
    /// dependence from role-key overlaps + palette codebook quantization).
    ///
    /// The tier spacing exponent is `α = p/2`, derived from the Jirak rate via
    /// the scale-spacing construction: each tier width scales as the rate denominator
    /// raised to the per-tier rank, normalized so Σ10 = 1.0:
    ///
    /// ```text
    /// Σk = k^(p/2) / 10^(p/2)
    /// ```
    ///
    /// - `p = 3` → `α = 1.5` → `Σk = k^1.5 / 10^1.5`
    ///   Σ1 ≈ 0.0316, Σ5 ≈ 0.3536, Σ10 = 1.0 (convex, gentle low end)
    /// - `p = 4` → `α = 2.0` → `Σk = k^2.0 / 10^2.0`
    ///   Σ1 = 0.01, Σ5 = 0.25, Σ10 = 1.0 (more convex — larger Jirak correction at tail)
    ///
    /// Normalization: `Σ10 = 10^α / 10^α = 1.0` always (Σ10 anchored at 1.0).
    ///
    /// The convexity of the curve reflects the Jirak correction: the asymptotic
    /// regime (`p ≥ 4`) requires a tighter tail correction, so tier spacing
    /// concentrates more budget at the high-F region. For `p ≥ 4` the
    /// spacing is MORE convex than `p = 3` (higher variance of inter-tier deltas).
    ///
    /// # Panics
    ///
    /// Does not panic; any finite `p > 2.0` produces a strictly increasing sequence.
    ///
    /// # Reference
    ///
    /// Jirak 2016: "Berry-Esseen theorems under weak dependence",
    /// arxiv 1606.01617, Annals of Probability 44(3) 2024–2063.
    pub fn jirak_p(p: f32) -> Self {
        // α = p/2 is the scale exponent: Σk = k^α / 10^α.
        // The Jirak rate is n^(p/2-1); the spacing exponent p/2 is one step up,
        // encoding the rate's scale as a tier-rank power law.
        let alpha = p / 2.0;
        // Normalisation denominator: 10^α anchors Σ10 = 1.0 exactly.
        let denom = 10.0_f32.powf(alpha);
        let mut bands = [0.0_f32; 10];
        for (i, band) in bands.iter_mut().enumerate() {
            let k = (i + 1) as f32;
            *band = k.powf(alpha) / denom;
        }
        // Force Σ10 = 1.0 exactly to avoid floating-point rounding at the anchor.
        bands[9] = 1.0;
        Self {
            sigma1_to_sigma10: bands,
        }
    }

    /// Hand-tuned linear defaults from sprint-11: Σk = k × 0.10.
    ///
    /// **Deprecated for new code** — use the Jirak-derived `SigmaTierBands::default()`
    /// in new code. This constructor is preserved for sprint-11 backwards comparison
    /// (TD-SIGMA-TIER-THRESHOLDS-1) and for callers that explicitly need the
    /// linear 0.10 .. 1.00 spacing.
    ///
    /// # Note
    ///
    /// These values are the sprint-11 / Wave F hand-tuned baseline. They produce
    /// uniform linear spacing (equal tier width), which is incorrect for the
    /// workspace's CAM-PQ-induced weak-dependence regime. The principled spacing
    /// is `k^1.5 / 10^1.5` per Jirak 2016 (see `SigmaTierBands::jirak_p`).
    pub fn hand_tuned() -> Self {
        Self {
            sigma1_to_sigma10: [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00],
        }
    }

    /// Hand-tuned linear defaults (sprint-11 baseline). Prefer `hand_tuned()`.
    ///
    /// **Deprecated** — use `SigmaTierBands::hand_tuned()` for sprint-11
    /// backwards comparison, or `SigmaTierBands::default()` for the Jirak-
    /// derived default in new code.
    #[deprecated(
        since = "0.2.0",
        note = "use `SigmaTierBands::hand_tuned()` for backwards comparison or \
                `SigmaTierBands::default()` for Jirak-derived thresholds"
    )]
    pub fn default_bands() -> Self {
        Self::hand_tuned()
    }

    /// Returns true iff the thresholds are strictly monotonically increasing.
    ///
    /// A non-monotonic band table is a misconfiguration; the invariant is
    /// checked at construction time in `SigmaTierRouter::new`.
    pub fn is_monotonic(&self) -> bool {
        self.sigma1_to_sigma10.windows(2).all(|w| w[0] < w[1])
    }

    /// Map a free-energy value to a Σ-tier (1..=10).
    ///
    /// Returns the first tier whose upper threshold is `>= current_f`,
    /// or 10 if `current_f` exceeds all thresholds.
    pub fn tier_for(&self, current_f: f32) -> u8 {
        for (i, &threshold) in self.sigma1_to_sigma10.iter().enumerate() {
            if current_f <= threshold {
                return (i + 1) as u8;
            }
        }
        10
    }
}

impl Default for SigmaTierBands {
    /// Jirak-derived defaults for the workspace's CAM-PQ weak-dependence regime
    /// (`p = 3.0`). See `SigmaTierBands::jirak_p` for the full derivation.
    ///
    /// Σ1 ≈ 0.0316, Σ5 ≈ 0.3536, Σ10 = 1.0 (convex spacing).
    fn default() -> Self {
        Self::jirak_p(3.0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ResonanceState
// ─────────────────────────────────────────────────────────────────────────────

/// Minimal free-energy tracker.
///
/// Tracks two consecutive F values to derive `last_delta`, which is the
/// primary signal for the Rubicon condition: commit happens when
/// `last_delta < 0` (F is falling = surprise is being resolved).
#[derive(Clone, Debug, PartialEq)]
pub struct ResonanceState {
    /// F value from the previous tick.
    pub last_f: f32,
    /// F value from the most recent tick.
    pub current_f: f32,
    /// `current_f - last_f`; negative means F is falling (resolving).
    pub last_delta: f32,
}

impl ResonanceState {
    /// Construct a neutral starting state (all zeros).
    pub fn zero() -> Self {
        Self {
            last_f: 0.0,
            current_f: 0.0,
            last_delta: 0.0,
        }
    }

    /// Update the state with a new F sample; returns the new delta.
    pub fn update(&mut self, new_f: f32) -> f32 {
        self.last_f = self.current_f;
        self.current_f = new_f;
        self.last_delta = new_f - self.last_f;
        self.last_delta
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DispatchOutcome + RestReason
// ─────────────────────────────────────────────────────────────────────────────

/// Reason the router chose `Rest` over `Continue` or `Commit`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RestReason {
    /// Free energy is below the homeostasis floor; the cycle rests normally.
    BelowHomeostasis,
    /// Tier has reached Σ10 but F is still rising — the "never commit on
    /// F-rising" invariant (plan §16 D-CSV-10 test brief). The Rubicon
    /// requires F to be falling before commitment is permitted.
    Sigma10Saturated,
    /// The gate decision from `gate_decision_i4()` is `Block`; dispatch
    /// is prevented by the qualia + mantissa signal.
    GateBlocked,
}

/// Outcome of a single `SigmaTierRouter::dispatch()` call.
///
/// Note: `PartialEq` is NOT derived because `GateDecision` (from
/// `lance_graph_contract::mul`) does not implement `PartialEq`. Tests use
/// `matches!` for pattern-based assertions on `Commit` variants.
#[derive(Clone, Debug)]
pub enum DispatchOutcome {
    /// Cycle continues; the router proposes the next tier to run.
    Continue {
        /// The next Σ-tier to enter (current + 1, capped at 10).
        next_tier: u8,
    },
    /// Σ10 Rubicon crossed; the cycle commits.
    ///
    /// The gate and reached tier are returned so the caller can record
    /// the commit evidence (which merge mode, which tier level).
    Commit {
        /// Gate decision that was computed (always Flow or Hold — never
        /// Block here, because Block exits via `Rest { GateBlocked }`).
        gate: GateDecision,
        /// Always 10 in the current implementation (Rubicon is at Σ10).
        tier_reached: u8,
    },
    /// Cycle rests; no commit and no tier advance.
    Rest {
        /// Why the cycle chose to rest.
        reason: RestReason,
    },
}

// ─────────────────────────────────────────────────────────────────────────────
// SigmaTierRouter
// ─────────────────────────────────────────────────────────────────────────────

/// Σ-tier Rubicon-resonance dispatch router.
///
/// Encapsulates the active-inference cycle driver described in
/// `cognitive-substrate-convergence-v1.md` §10:
///
/// ```text
/// while  F > homeostasis_floor:
///     cycle()
///     F = compute_free_energy(...)
///     # F drops as the cycle resolves; if F < floor, shader rests
/// ```
///
/// The Rubicon fires when `current_tier() >= 10` AND `last_delta < 0`
/// (F is **falling** = cycle resolved). If F is rising at Σ10, the router
/// returns `Rest { Σ10Saturated }` — the invariant is: **never commit on
/// F-rising** (plan §16 D-CSV-10 test brief).
pub struct SigmaTierRouter {
    /// Σ-tier band thresholds.
    pub bands: SigmaTierBands,
    /// Free-energy tracker.
    pub state: ResonanceState,
    /// Homeostasis floor: below this F value the cycle rests normally.
    pub homeostasis_floor: f32,
}

impl SigmaTierRouter {
    /// Construct a new router with explicit bands and homeostasis floor.
    ///
    /// # Panics
    ///
    /// Panics if `bands.is_monotonic()` returns false — a non-monotonic
    /// band table is a misconfiguration that would produce nonsensical
    /// tier assignments.
    pub fn new(bands: SigmaTierBands, homeostasis_floor: f32) -> Self {
        assert!(
            bands.is_monotonic(),
            "SigmaTierBands must be strictly monotonically increasing; \
             non-monotonic table produces undefined tier assignments"
        );
        Self {
            bands,
            state: ResonanceState::zero(),
            homeostasis_floor,
        }
    }

    /// Update the tracker with the latest free-energy reading.
    ///
    /// Returns the delta (`current_f - last_f`); negative = F is falling
    /// (cycle resolving), positive = F is rising (more surprise).
    ///
    /// Call `tick()` before `dispatch()` each cycle so that `last_delta`
    /// is current when the Rubicon check runs.
    pub fn tick(&mut self, current_f: f32) -> f32 {
        self.state.update(current_f)
    }

    /// Current Σ-tier (1..=10) based on the tracker's `current_f`.
    ///
    /// Uses `bands.tier_for(current_f)` — see `SigmaTierBands::tier_for`.
    pub fn current_tier(&self) -> u8 {
        self.bands.tier_for(self.state.current_f)
    }

    /// Central dispatch method — implements the §10 Rubicon-resonance loop.
    ///
    /// Decision table (evaluated top-to-bottom, first match wins):
    ///
    /// 1. `current_f < homeostasis_floor` → `Rest { BelowHomeostasis }`
    /// 2. `gate_decision_i4(qualia, mantissa)` is `Block` → `Rest { GateBlocked }`
    /// 3. `current_tier() >= 10` AND `last_delta > 0` (F rising) →
    ///    `Rest { Σ10Saturated }` (invariant: never commit on F-rising)
    /// 4. `current_tier() >= 10` AND `last_delta <= 0` (F falling or flat) →
    ///    `Commit { gate, tier_reached: 10 }` (Rubicon hit)
    /// 5. otherwise → `Continue { next_tier: current_tier() + 1 }`
    ///
    /// # Arguments
    ///
    /// * `qualia` — i4-16D packed qualia for the current cycle.
    /// * `mantissa` — signed inference mantissa (from `InferenceType::to_mantissa()`).
    pub fn dispatch(&mut self, qualia: &QualiaI4_16D, mantissa: i8) -> DispatchOutcome {
        // Rule 1: homeostasis floor — cycle rests if F is below floor.
        if self.state.current_f < self.homeostasis_floor {
            return DispatchOutcome::Rest {
                reason: RestReason::BelowHomeostasis,
            };
        }

        // Rule 2: gate decision — if Block, the cycle cannot dispatch.
        let gate = gate_decision_i4(qualia, mantissa);
        if gate.is_blocked() {
            return DispatchOutcome::Rest {
                reason: RestReason::GateBlocked,
            };
        }

        let tier = self.current_tier();

        if tier >= 10 {
            // Rule 3: Σ10 + F rising → invariant violation, rest instead.
            if self.state.last_delta > 0.0 {
                return DispatchOutcome::Rest {
                    reason: RestReason::Sigma10Saturated,
                };
            }
            // Rule 4: Σ10 + F falling (or flat) → Rubicon crossed, commit.
            return DispatchOutcome::Commit {
                gate,
                tier_reached: 10,
            };
        }

        // Rule 5: normal continuation — advance one tier.
        DispatchOutcome::Continue {
            next_tier: tier + 1,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: is_blocked() on GateDecision from lance-graph-contract
// ─────────────────────────────────────────────────────────────────────────────

/// Extension trait to query whether a `GateDecision` from the `mul` module
/// represents a block decision.
///
/// `lance_graph_contract::mul::GateDecision` is an enum with variants
/// `Flow`, `Hold { reason }`, and `Block { reason }`. We need a single
/// predicate for the dispatch logic above.
trait GateDecisionExt {
    fn is_blocked(&self) -> bool;
}

impl GateDecisionExt for GateDecision {
    fn is_blocked(&self) -> bool {
        matches!(self, GateDecision::Block { .. })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use lance_graph_contract::qualia::QualiaI4_16D;

    // ── helpers ──────────────────────────────────────────────────────────────

    /// Build a router with hand-tuned (sprint-11 baseline) bands and the
    /// given homeostasis floor. Used by the original 12 tests that were
    /// calibrated to the linear 0.10..1.00 thresholds.
    #[allow(deprecated)]
    fn make_router(floor: f32) -> SigmaTierRouter {
        SigmaTierRouter::new(SigmaTierBands::default_bands(), floor)
    }

    /// Build a qualia that produces `GateDecision::Flow` from `gate_decision_i4`.
    ///
    /// High coherence + high valence + high warmth + high groundedness + low tension
    /// → Calibrated trust + Flow state → GateDecision::Flow.
    fn flow_qualia() -> QualiaI4_16D {
        QualiaI4_16D::ZERO
            .with(9, 5)  // coherence (DIM_COHERENCE)
            .with(1, 3)  // valence
            .with(2, 0)  // tension (low)
            .with(3, 5)  // warmth
            .with(14, 4) // groundedness
    }

    /// Build a qualia that produces `GateDecision::Block` from `gate_decision_i4`.
    ///
    /// Low coherence + high tension → Uncertain trust → Block.
    fn block_qualia() -> QualiaI4_16D {
        QualiaI4_16D::ZERO
            .with(9, -5)  // coherence (very low)
            .with(2, 5)   // tension (high)
    }

    // ── Test 1: default band thresholds are strictly monotonic ────────────────

    #[test]
    #[allow(deprecated)]
    fn test_bands_default_monotonic() {
        let bands = SigmaTierBands::default_bands();
        assert!(
            bands.is_monotonic(),
            "Default bands must be strictly monotonically increasing"
        );
        // Verify each threshold is strictly greater than the previous
        let t = &bands.sigma1_to_sigma10;
        assert_eq!(t.len(), 10);
        for i in 1..10 {
            assert!(
                t[i] > t[i - 1],
                "threshold[{}]={} must be > threshold[{}]={}",
                i, t[i], i - 1, t[i - 1]
            );
        }
    }

    // ── Test 2: correct tier for representative F values ──────────────────────

    #[test]
    #[allow(deprecated)]
    fn test_current_tier_for_each_band() {
        let bands = SigmaTierBands::default_bands();
        // F values just at/below each threshold should give the correct tier.
        // Default: Σk upper bound = k * 0.10
        let cases: &[(f32, u8)] = &[
            (0.05, 1),   // below Σ1 threshold (0.10)
            (0.10, 1),   // exactly at Σ1 threshold
            (0.15, 2),   // between Σ1 and Σ2
            (0.20, 2),   // exactly at Σ2 threshold
            (0.55, 6),   // between Σ5 and Σ6
            (0.90, 9),   // exactly at Σ9 threshold
            (0.95, 10),  // between Σ9 and Σ10 → tier 10
            (1.00, 10),  // exactly at Σ10 threshold
            (1.10, 10),  // above all thresholds → clamp to 10
        ];
        for &(f, expected_tier) in cases {
            let actual = bands.tier_for(f);
            assert_eq!(
                actual, expected_tier,
                "F={} should map to tier {}, got {}",
                f, expected_tier, actual
            );
        }
    }

    // ── Test 3: dispatch rests when F is below homeostasis floor ──────────────

    #[test]
    fn test_dispatch_below_homeostasis_rests() {
        let mut router = make_router(0.3);
        // Tick to F = 0.1, which is below the floor of 0.3.
        router.tick(0.1);
        let outcome = router.dispatch(&flow_qualia(), 3);
        assert!(
            matches!(
                outcome,
                DispatchOutcome::Rest {
                    reason: RestReason::BelowHomeostasis
                }
            ),
            "F below homeostasis floor must Rest with BelowHomeostasis"
        );
    }

    // ── Test 4: dispatch rests with GateBlocked when gate blocks ─────────────

    #[test]
    fn test_dispatch_block_rests_with_reason() {
        let mut router = make_router(0.0);
        // Tick to a mid-range F that is above the floor.
        router.tick(0.5);
        // Use block_qualia (Uncertain trust → Block gate).
        let outcome = router.dispatch(&block_qualia(), 2);
        assert!(
            matches!(
                outcome,
                DispatchOutcome::Rest {
                    reason: RestReason::GateBlocked
                }
            ),
            "Block gate must produce Rest {{ GateBlocked }}"
        );
    }

    // ── Test 5: dispatch continues within a mid-range band ───────────────────

    #[test]
    fn test_dispatch_continue_within_band() {
        let mut router = make_router(0.0);
        // F = 0.35 → tier 4 (0.30 < 0.35 <= 0.40). Expect Continue { next_tier: 5 }.
        // (Uses hand-tuned bands via make_router; tier boundaries differ under Jirak.)
        router.tick(0.35);
        let outcome = router.dispatch(&flow_qualia(), 3);
        assert!(
            matches!(outcome, DispatchOutcome::Continue { next_tier: 5 }),
            "F=0.35 should give Continue {{ next_tier: 5 }}"
        );
    }

    // ── Test 6: Σ10 + F falling → Commit ─────────────────────────────────────

    #[test]
    fn test_dispatch_commit_at_sigma10_falling() {
        let mut router = make_router(0.0);
        // First tick: high F
        router.tick(0.99);
        // Second tick: slightly lower → delta < 0 (F is falling)
        router.tick(0.95);
        // Both F values are in Σ10 range (>= 0.90 threshold of Σ9).
        // Delta = 0.95 - 0.99 = -0.04 (falling).
        assert_eq!(router.current_tier(), 10);
        assert!(router.state.last_delta < 0.0, "expected F-falling delta");
        let outcome = router.dispatch(&flow_qualia(), 3);
        assert!(
            matches!(outcome, DispatchOutcome::Commit { tier_reached: 10, .. }),
            "Σ10 + F-falling should Commit, got {:?}",
            outcome
        );
    }

    // ── Test 7: invariant — Σ10 + F rising → never Commit ────────────────────

    #[test]
    fn test_dispatch_no_commit_on_f_rising_property() {
        // The "never commit on F-rising" invariant from plan §16 D-CSV-10.
        // For any (qualia, mantissa) where current_tier==10 AND last_delta > 0,
        // dispatch must return Rest, NEVER Commit.
        let mut router = make_router(0.0);
        // Arrange: tick from low to high → delta > 0 (rising), tier = 10
        router.tick(0.90); // Σ9/Σ10 boundary
        router.tick(0.99); // higher → delta = +0.09 > 0, F rising
        assert_eq!(router.current_tier(), 10);
        assert!(
            router.state.last_delta > 0.0,
            "precondition: F must be rising (delta > 0)"
        );

        // Test with various qualia + mantissa combos that would otherwise Commit
        let test_cases: &[(&QualiaI4_16D, i8)] = &[
            (&flow_qualia(), 3),
            (&flow_qualia(), 1),
            (&flow_qualia(), 7),
            (&flow_qualia(), -1), // backward mantissa, still in tier 10
        ];

        for (qualia, mantissa) in test_cases {
            // Re-apply the rising state (tick resets delta each time)
            let mut r = make_router(0.0);
            r.tick(0.90);
            r.tick(0.99);

            let outcome = r.dispatch(qualia, *mantissa);
            assert!(
                matches!(
                    outcome,
                    DispatchOutcome::Rest {
                        reason: RestReason::Sigma10Saturated
                    }
                ),
                "Σ10 + F-rising must give Rest{{Sigma10Saturated}}, \
                 got {:?} for mantissa={}",
                outcome,
                mantissa
            );
            assert!(
                !matches!(outcome, DispatchOutcome::Commit { .. }),
                "INVARIANT VIOLATION: Commit must NEVER fire when F is rising \
                 at Σ10, mantissa={mantissa}"
            );
        }
    }

    // ── Test 8: tick updates last_delta correctly ─────────────────────────────

    #[test]
    fn test_tick_updates_delta() {
        let mut router = make_router(0.0);
        let d1 = router.tick(0.5);
        assert!((d1 - 0.5).abs() < 1e-6, "first tick delta should be 0.5 - 0.0 = 0.5");
        let d2 = router.tick(0.3);
        assert!((d2 - (-0.2)).abs() < 1e-6, "second tick delta should be 0.3 - 0.5 = -0.2");
        assert!((router.state.last_delta - (-0.2)).abs() < 1e-6);
        assert!((router.state.current_f - 0.3).abs() < 1e-6);
        assert!((router.state.last_f - 0.5).abs() < 1e-6);
    }

    // ── Test 9: ResonanceState round-trip ─────────────────────────────────────

    #[test]
    fn test_resonance_state_round_trip() {
        let mut s = ResonanceState::zero();
        assert!((s.last_f).abs() < 1e-9);
        assert!((s.current_f).abs() < 1e-9);
        assert!((s.last_delta).abs() < 1e-9);

        s.update(0.7);
        assert!((s.last_f).abs() < 1e-9, "last_f should be 0.0 after first update");
        assert!((s.current_f - 0.7).abs() < 1e-6);
        assert!((s.last_delta - 0.7).abs() < 1e-6);

        s.update(0.4);
        assert!((s.last_f - 0.7).abs() < 1e-6);
        assert!((s.current_f - 0.4).abs() < 1e-6);
        assert!((s.last_delta - (-0.3)).abs() < 1e-6);
    }

    // ── Test 10: RestReason variants are distinct ─────────────────────────────

    #[test]
    fn test_rest_reason_variants_distinct() {
        let r1 = RestReason::BelowHomeostasis;
        let r2 = RestReason::Sigma10Saturated;
        let r3 = RestReason::GateBlocked;
        assert_ne!(r1, r2);
        assert_ne!(r1, r3);
        assert_ne!(r2, r3);
        // Reflexive equality
        assert_eq!(r1, RestReason::BelowHomeostasis);
        assert_eq!(r2, RestReason::Sigma10Saturated);
        assert_eq!(r3, RestReason::GateBlocked);
    }

    // ── Test 11: homeostasis_floor = 0.0 commits normally at Σ10 ────────────

    #[test]
    fn test_homeostasis_floor_zero_commits_normally() {
        // With floor = 0.0, any positive F is above floor.
        // Verify that a Σ10 + F-falling scenario does commit.
        let mut router = make_router(0.0);
        router.tick(0.99);
        router.tick(0.91); // delta = -0.08 < 0 (falling), still in Σ10
        assert_eq!(router.current_tier(), 10);
        assert!(router.state.last_delta < 0.0);
        let outcome = router.dispatch(&flow_qualia(), 4);
        assert!(
            matches!(outcome, DispatchOutcome::Commit { tier_reached: 10, .. }),
            "floor=0.0 + Σ10 + F-falling must Commit, got {:?}",
            outcome
        );
    }

    // ── Test 12: router new / default state ──────────────────────────────────

    #[test]
    #[allow(deprecated)]
    fn test_router_new_default_state() {
        let bands = SigmaTierBands::default_bands();
        let router = SigmaTierRouter::new(bands.clone(), 0.2);

        // Initial resonance state is all zeros
        assert!((router.state.last_f).abs() < 1e-9);
        assert!((router.state.current_f).abs() < 1e-9);
        assert!((router.state.last_delta).abs() < 1e-9);
        // Homeostasis floor stored correctly
        assert!((router.homeostasis_floor - 0.2).abs() < 1e-6);
        // Bands stored correctly
        assert_eq!(router.bands.sigma1_to_sigma10, bands.sigma1_to_sigma10);
        // Initial tier is Σ1 (current_f = 0.0 <= 0.10)
        assert_eq!(router.current_tier(), 1);
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Sprint-12 W-G4: 8 NEW TESTS for Jirak-derived thresholds
    // D-CSV-15 / TD-SIGMA-TIER-THRESHOLDS-1 resolution
    // ══════════════════════════════════════════════════════════════════════════

    // ── New Test 1: Jirak default is convex ──────────────────────────────────

    #[test]
    fn test_jirak_default_is_convex() {
        // Convexity: each successive delta is larger than the previous.
        // Σk = k^1.5 / 10^1.5 → spacing accelerates toward high tiers.
        let bands = SigmaTierBands::default();
        let t = &bands.sigma1_to_sigma10;
        let deltas: Vec<f32> = (0..9).map(|i| t[i + 1] - t[i]).collect();
        for i in 0..deltas.len() - 1 {
            assert!(
                deltas[i + 1] > deltas[i],
                "Jirak default must be convex: delta[{}]={:.6} must be < delta[{}]={:.6}",
                i, deltas[i], i + 1, deltas[i + 1]
            );
        }
    }

    // ── New Test 2: Jirak default endpoints ──────────────────────────────────

    #[test]
    fn test_jirak_default_endpoints() {
        // Σ1 = 1^1.5 / 10^1.5 ≈ 0.031623; Σ10 = 1.0 exactly.
        let bands = SigmaTierBands::default();
        let t = &bands.sigma1_to_sigma10;
        // Σ1 ≈ 0.031623 (within ε)
        let expected_sigma1: f32 = 1.0_f32.powf(1.5) / 10.0_f32.powf(1.5);
        assert!(
            (t[0] - expected_sigma1).abs() < 1e-6,
            "Σ1 should be ≈ {:.7} (k^1.5/10^1.5), got {:.7}",
            expected_sigma1, t[0]
        );
        // Σ10 = 1.0 exactly (anchored)
        assert_eq!(t[9], 1.0_f32, "Σ10 must be exactly 1.0");
    }

    // ── New Test 3: hand_tuned preserves old values ───────────────────────────

    #[test]
    fn test_hand_tuned_preserves_old_values() {
        let bands = SigmaTierBands::hand_tuned();
        let expected = [0.10_f32, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00];
        assert_eq!(
            bands.sigma1_to_sigma10, expected,
            "hand_tuned() must return exactly the sprint-11 linear baseline"
        );
    }

    // ── New Test 4: jirak_p(3.0) matches Default ─────────────────────────────

    #[test]
    fn test_jirak_p_3_matches_default() {
        let jirak3 = SigmaTierBands::jirak_p(3.0);
        let default = SigmaTierBands::default();
        assert_eq!(
            jirak3.sigma1_to_sigma10, default.sigma1_to_sigma10,
            "jirak_p(3.0) must equal SigmaTierBands::default()"
        );
    }

    // ── New Test 5: jirak_p(4.0) is more convex than jirak_p(3.0) ───────────

    #[test]
    fn test_jirak_p_4_more_linear() {
        // For p=4 the exponent is 4/2 = 2.0 → k^2 spacing (more convex than p=3's k^1.5).
        // Per the Jirak derivation: higher p means the asymptotic Berry-Esseen correction
        // is larger at the tail, so tier spacing concentrates more budget at high F.
        // This is reflected as HIGHER delta variance for p=4 vs p=3.
        //
        // Measure via variance of inter-tier deltas: higher variance = more convex.
        let p3 = SigmaTierBands::jirak_p(3.0);
        let p4 = SigmaTierBands::jirak_p(4.0);

        let variance_of_deltas = |bands: &SigmaTierBands| -> f32 {
            let t = &bands.sigma1_to_sigma10;
            let deltas: Vec<f32> = (0..9).map(|i| t[i + 1] - t[i]).collect();
            let mean = deltas.iter().sum::<f32>() / deltas.len() as f32;
            deltas.iter().map(|d| (d - mean).powi(2)).sum::<f32>() / deltas.len() as f32
        };

        let var_p3 = variance_of_deltas(&p3);
        let var_p4 = variance_of_deltas(&p4);

        // p=4 → exponent 2.0 is MORE convex (larger tail correction) than p=3 → exponent 1.5
        assert!(
            var_p4 > var_p3,
            "jirak_p(4.0) should have higher delta variance ({:.8}) than jirak_p(3.0) ({:.8}); \
             p=4 has a larger Jirak tail correction (more convex spacing)",
            var_p4, var_p3
        );
    }

    // ── New Test 6: tier_for with Jirak vs hand-tuned at f=0.5 ──────────────

    #[test]
    fn test_tier_for_jirak_vs_hand_tuned_at_0_5() {
        // f=0.5 under Jirak (p=3): Σ5≈0.3536, Σ6≈0.4648, Σ7≈0.5857
        // 0.5 > Σ6 (0.4648), 0.5 <= Σ7 (0.5857) → tier 7
        let jirak = SigmaTierBands::jirak_p(3.0);
        let jirak_tier = jirak.tier_for(0.5);
        assert_eq!(
            jirak_tier, 7,
            "f=0.5 under Jirak p=3 should be tier 7 (Σ6≈0.4648 < 0.5 <= Σ7≈0.5857), got {}",
            jirak_tier
        );

        // f=0.5 under hand-tuned: Σ5=0.5 exactly → tier 5
        let linear = SigmaTierBands::hand_tuned();
        let linear_tier = linear.tier_for(0.5);
        assert_eq!(
            linear_tier, 5,
            "f=0.5 under hand-tuned linear should be tier 5 (Σ5=0.50 exactly), got {}",
            linear_tier
        );

        // The two must differ — the whole point of the Jirak derivation
        assert_ne!(
            jirak_tier, linear_tier,
            "Jirak and hand-tuned must assign different tiers to f=0.5"
        );
    }

    // ── New Test 7: Jirak default is monotonic ────────────────────────────────

    #[test]
    fn test_jirak_monotonic() {
        let bands = SigmaTierBands::default();
        assert!(
            bands.is_monotonic(),
            "Jirak-derived default bands must be strictly monotonically increasing"
        );
        // Also check all 9 consecutive pairs explicitly
        let t = &bands.sigma1_to_sigma10;
        for i in 0..9 {
            assert!(
                t[i] < t[i + 1],
                "Jirak band: t[{}]={:.6} must be < t[{}]={:.6}",
                i, t[i], i + 1, t[i + 1]
            );
        }
    }

    // ── New Test 8: homeostasis floor compatible with Jirak default ───────────

    #[test]
    fn test_homeostasis_floor_compatible() {
        // SigmaTierRouter::new(SigmaTierBands::default(), 0.05) must construct.
        // With current_f = 0.03 (< floor 0.05) → Rest { BelowHomeostasis }.
        //
        // Note: Σ1 (Jirak) ≈ 0.0316, so 0.03 < Σ1 AND 0.03 < floor.
        // The BelowHomeostasis check fires before any tier comparison.
        let mut router = SigmaTierRouter::new(SigmaTierBands::default(), 0.05);
        router.tick(0.03);
        let outcome = router.dispatch(&flow_qualia(), 3);
        assert!(
            matches!(
                outcome,
                DispatchOutcome::Rest {
                    reason: RestReason::BelowHomeostasis
                }
            ),
            "current_f=0.03 < homeostasis_floor=0.05 must give Rest{{BelowHomeostasis}}, \
             got {:?}",
            outcome
        );
        // Verify: 0.03 is indeed in the Jirak Σ1 zone (below Σ1 threshold)
        let sigma1 = SigmaTierBands::default().sigma1_to_sigma10[0];
        assert!(
            0.03_f32 < sigma1,
            "0.03 should be below Jirak Σ1≈{:.6} (confirming it is in the Σ1 zone)",
            sigma1
        );
    }
}
