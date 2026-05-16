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
//! Band thresholds are hand-tuned starting values per `I-NOISE-FLOOR-JIRAK`
//! TECH_DEBT note below. A principled Jirak-derived calibration is planned
//! for sprint-13+ (VAMPE + Jirak coupled revival, per plan §14 OQ-CSV-6).
//!
//! # TECH_DEBT — I-NOISE-FLOOR-JIRAK (hand-tuned thresholds)
//!
//! The `SigmaTierBands` default values (Σ1=0.10 .. Σ10=1.00) are hand-tuned.
//! Per the `I-NOISE-FLOOR-JIRAK` iron rule in CLAUDE.md, σ-threshold
//! calibration should cite Jirak 2016 (arxiv 1606.01617) derived bounds.
//! The weak-dependence correction is NOT applied here; these are acceptable
//! initial values for sprint-12 and must be replaced by principled derivation
//! in sprint-13+ once the VAMPE + Jirak coupled-revival pair activates.
//! See `.claude/board/TECH_DEBT.md` for tracking.

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
/// Default values are hand-tuned (Σ1=0.10, Σ2=0.20, …, Σ10=1.00).
/// See module-level TECH_DEBT note for the `I-NOISE-FLOOR-JIRAK` deferral.
#[derive(Clone, Debug, PartialEq)]
pub struct SigmaTierBands {
    /// Upper F boundary per Σ-tier; index 0 = Σ1, index 9 = Σ10.
    pub sigma1_to_sigma10: [f32; 10],
}

impl SigmaTierBands {
    /// Hand-tuned defaults: Σk threshold = k × 0.10.
    ///
    /// TECH_DEBT — I-NOISE-FLOOR-JIRAK: replace with Jirak-derived bounds
    /// in sprint-13+ (VAMPE + Jirak coupled-revival track). Until then these
    /// linear starting values are acceptable for sprint-12 integration tests.
    pub fn default_bands() -> Self {
        Self {
            sigma1_to_sigma10: [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00],
        }
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
    fn default() -> Self {
        Self::default_bands()
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

    /// Build a router with default bands and the given homeostasis floor.
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
}
