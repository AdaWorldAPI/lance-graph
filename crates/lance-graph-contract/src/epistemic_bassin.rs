// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! `epistemic_bassin` — **`EpistemicBassin24`** (operator-named: das *Bassin*, the accumulation pool): 24 named epistemic axes as an
//! `agree_u4[24] + disagree_u4[24]` PAIR (`D-DCR-2b`, operator co-architect
//! ruling 2026-09-01). Supersedes `basin_lanes::BasinLanes`' signed-net
//! representation.
//!
//! # Why the signed net was FALSIFIED, not merely limited
//!
//! In one signed register, `+3` support and `−3` refutation sum to `0` —
//! maximal balanced conflict indistinguishable from silence. If agreement
//! and disagreement are the interesting information, a representation that
//! destroys their coincidence is wrong, not constrained; `basin_lanes`'
//! own pinned collapse test is the evidence. The pair keeps both sides:
//! **net, polarity, conflict, masks and entropy are all DERIVED**, and
//! equal support/refutation is a first-class state (`Contested`), distinct
//! from silence by construction. Cost: one extra 12-byte register.
//!
//! # A BASSIN of values on a named BASIS — never an address
//!
//! The name carries both halves deliberately: the 24 axes are a named
//! epistemic **basis** (which axes, and their version, named by the facet
//! classid), and the register pair is the **Bassin** — the pool where one
//! hop of children's stance mass collects. Distinct from
//! [`EpisodicBasin`](crate::canonical_node::ValueTenant::EpisodicBasin),
//! which stores durable episodic REFERENCES; this type stores no reference
//! of any kind.
//!
//! The facet's 4-byte classid names the **24-axis basis** (which named
//! epistemic axes, and their version — "epistemic witness v3" is a classid,
//! not a Rust name). Each nibble is a **quantized value on a named axis**,
//! never an address:
//!
//! - parent/children topology lives in the HHTL KEY (+ an indexed 16-bit
//!   child mask reconstructs every direct child by appending each set
//!   nibble) — storing child pointers here would duplicate the key;
//! - durable episodic identity lives in `EpisodicBasin` / stable
//!   references, not in ±8 stream offsets;
//! - exact proof and authority live in their exact carriers — premises/W,
//!   `CausalEdge64`, NARS truth, revision/provenance;
//! - exact EWA covariance lives in the Σ/SPD carrier.
//!
//! What a lane MAY carry is **proprioception and pressure**: local
//! support/falsifier pressure (Tarski's exact depth stays derivable from
//! premise ancestry), ΔH / expected information gain (Shannon H itself
//! stays a readout of the candidate distribution), transformed
//! tension/residual/coherence (EWA's exact Σ stays elsewhere).
//! Counterfactual/revision axes describe pressure and change, never
//! empirical truth.
//!
//! # Storage is NOT minted here — the #1127 law was scoped too wide
//!
//! `loci-never-magnitude` is a law of the EXPERIMENTAL A9 ContextLoci
//! *reading* (`causal_witness::CausalWitnessFacet`), not of tenant 14's
//! physical bytes: the 12-byte register is content-blind, its
//! interpretation selected per row by the 4-byte classid, and multiple
//! readings of the same bytes are explicitly allowed (`causal_witness.rs`
//! itself ships as "the THIRD ClassView reading of the same register").
//! #1127 accidentally promoted a reading-specific invariant into a storage
//! invariant. So this basis can be a classid-selected reading of the SAME
//! physical lane; a separate tenant is minted only when one real row
//! demonstrably needs ContextLoci AND the basis simultaneously.
//!
//! # One hop, and an honest associativity note
//!
//! A node's pair expresses its DIRECT children accumulated; whether the
//! next hop continues or cancels what it reads is the second hop's
//! decision. [`accumulate_children`](EpistemicBassin24::accumulate_children)
//! is exact-sum-then-clamp and therefore order-independent **within one
//! call** — but recursively composing already-clamped child registers is
//! NOT associative (a clamped 15 means "at least 15"). Saturation is
//! monotone and conflict-preserving — a saturated count can understate
//! mass, never convert conflict into silence — which is precisely what the
//! superseded signed net could not guarantee.

/// Named axes in one basis register pair.
pub const BASIS_AXES: usize = 24;

/// Bytes per side (24 × u4).
pub const BASIS_REGISTER_BYTES: usize = 12;

/// Wire width of the pair: agree register then disagree register.
pub const BASIS_PAIR_BYTES: usize = 2 * BASIS_REGISTER_BYTES;

/// Per-axis count ceiling (u4).
pub const AXIS_COUNT_MAX: u8 = 15;

/// What one axis reads as, derived from the pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AxisState {
    /// No recorded support and no recorded refutation.
    Silent,
    /// Support only.
    Agree,
    /// Refutation only.
    Disagree,
    /// Both sides recorded — the state the signed net destroyed.
    Contested,
}

/// The 24-axis epistemic basis pair: unsigned support and refutation
/// counts per named axis, two lanes per byte (axis `2k` → low nibble,
/// `2k+1` → high).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct EpistemicBassin24 {
    agree: [u8; BASIS_REGISTER_BYTES],
    disagree: [u8; BASIS_REGISTER_BYTES],
}

fn pack_u4(counts: &[u8; BASIS_AXES]) -> [u8; BASIS_REGISTER_BYTES] {
    let mut bytes = [0u8; BASIS_REGISTER_BYTES];
    for (k, b) in bytes.iter_mut().enumerate() {
        let lo = counts[2 * k].min(AXIS_COUNT_MAX);
        let hi = counts[2 * k + 1].min(AXIS_COUNT_MAX);
        *b = (hi << 4) | lo;
    }
    bytes
}

fn unpack_u4(bytes: &[u8; BASIS_REGISTER_BYTES]) -> [u8; BASIS_AXES] {
    let mut out = [0u8; BASIS_AXES];
    for (k, b) in bytes.iter().enumerate() {
        out[2 * k] = b & 0x0F;
        out[2 * k + 1] = b >> 4;
    }
    out
}

impl EpistemicBassin24 {
    /// Epistemic silence: both sides zero on every axis. What mechanical
    /// hydration leaves behind, and what an unwritten pair reads as.
    pub const SILENT: Self = Self {
        agree: [0; BASIS_REGISTER_BYTES],
        disagree: [0; BASIS_REGISTER_BYTES],
    };

    /// Pack per-axis support and refutation counts, each saturating to
    /// `0..=15`.
    #[must_use]
    pub fn pack(agree: &[u8; BASIS_AXES], disagree: &[u8; BASIS_AXES]) -> Self {
        Self {
            agree: pack_u4(agree),
            disagree: pack_u4(disagree),
        }
    }

    /// Per-axis support counts.
    #[must_use]
    pub fn agree_counts(&self) -> [u8; BASIS_AXES] {
        unpack_u4(&self.agree)
    }

    /// Per-axis refutation counts.
    #[must_use]
    pub fn disagree_counts(&self) -> [u8; BASIS_AXES] {
        unpack_u4(&self.disagree)
    }

    /// DERIVED net stance on one axis: `agree − disagree`, in `−15..=15`.
    /// The superseded representation STORED this and nothing else; here it
    /// is a projection that forgets nothing underneath.
    #[must_use]
    pub fn net(&self, axis: usize) -> i8 {
        assert!(axis < BASIS_AXES, "axis out of range");
        self.agree_counts()[axis] as i8 - self.disagree_counts()[axis] as i8
    }

    /// DERIVED contested mass on one axis: the two-sided overlap
    /// `min(agree, disagree)` — `0` unless BOTH sides are recorded.
    #[must_use]
    pub fn contest(&self, axis: usize) -> u8 {
        assert!(axis < BASIS_AXES, "axis out of range");
        self.agree_counts()[axis].min(self.disagree_counts()[axis])
    }

    /// The axis's derived state.
    #[must_use]
    pub fn axis_state(&self, axis: usize) -> AxisState {
        assert!(axis < BASIS_AXES, "axis out of range");
        match (self.agree_counts()[axis], self.disagree_counts()[axis]) {
            (0, 0) => AxisState::Silent,
            (_, 0) => AxisState::Agree,
            (0, _) => AxisState::Disagree,
            _ => AxisState::Contested,
        }
    }

    /// Is every axis silent on both sides?
    #[must_use]
    pub fn is_silent(&self) -> bool {
        *self == Self::SILENT
    }

    /// Census over the 24 axes: `[silent, agree, disagree, contested]`.
    #[must_use]
    pub fn state_counts(&self) -> [usize; 4] {
        let mut out = [0usize; 4];
        for axis in 0..BASIS_AXES {
            out[match self.axis_state(axis) {
                AxisState::Silent => 0,
                AxisState::Agree => 1,
                AxisState::Disagree => 2,
                AxisState::Contested => 3,
            }] += 1;
        }
        out
    }

    /// Shannon entropy (bits) of the four-state axis distribution — the
    /// register-level proprioception readout ("a value tenant must add to
    /// entropy work"). `0.0` for a pure register; maximal `log2 4 = 2.0`
    /// at the uniform 6/6/6/6 mix. Unlike the superseded net, `Contested`
    /// is a cell of its own here, so balanced conflict RAISES this readout
    /// instead of vanishing from it.
    #[must_use]
    pub fn entropy_bits(&self) -> f32 {
        shannon(&self.state_counts())
    }

    /// One-hop contested-ness across the DIRECT children's states on one
    /// axis — Shannon entropy (bits) of the four-state distribution of
    /// `children[i].axis_state(axis)`. Empty children → `0.0`.
    #[must_use]
    pub fn stance_entropy_bits(children: &[Self], axis: usize) -> f32 {
        assert!(axis < BASIS_AXES, "axis out of range");
        let mut counts = [0usize; 4];
        for c in children {
            counts[match c.axis_state(axis) {
                AxisState::Silent => 0,
                AxisState::Agree => 1,
                AxisState::Disagree => 2,
                AxisState::Contested => 3,
            }] += 1;
        }
        shannon(&counts)
    }

    /// One-hop accumulation: per-axis exact `u32` sums PER SIDE, one clamp
    /// to `0..=15` at pack — order-independent within one call (see the
    /// module doc for the honest recursion caveat). A child's refutation
    /// accumulates on the disagree side and can never cancel another
    /// child's support — conflict is preserved, not netted away.
    /// Empty → [`SILENT`](Self::SILENT).
    #[must_use]
    pub fn accumulate_children(children: &[Self]) -> Self {
        let mut agree = [0u32; BASIS_AXES];
        let mut disagree = [0u32; BASIS_AXES];
        for c in children {
            let (a, d) = (c.agree_counts(), c.disagree_counts());
            for i in 0..BASIS_AXES {
                agree[i] += u32::from(a[i]);
                disagree[i] += u32::from(d[i]);
            }
        }
        let mut ap = [0u8; BASIS_AXES];
        let mut dp = [0u8; BASIS_AXES];
        for i in 0..BASIS_AXES {
            ap[i] = agree[i].min(u32::from(AXIS_COUNT_MAX)) as u8;
            dp[i] = disagree[i].min(u32::from(AXIS_COUNT_MAX)) as u8;
        }
        Self::pack(&ap, &dp)
    }

    /// Wire bytes: agree register, then disagree register.
    #[must_use]
    pub fn to_le_bytes(&self) -> [u8; BASIS_PAIR_BYTES] {
        let mut b = [0u8; BASIS_PAIR_BYTES];
        b[..BASIS_REGISTER_BYTES].copy_from_slice(&self.agree);
        b[BASIS_REGISTER_BYTES..].copy_from_slice(&self.disagree);
        b
    }

    /// Read a pair from its wire bytes — total, every pattern representable.
    #[must_use]
    pub fn from_le_bytes(b: &[u8; BASIS_PAIR_BYTES]) -> Self {
        let mut agree = [0u8; BASIS_REGISTER_BYTES];
        let mut disagree = [0u8; BASIS_REGISTER_BYTES];
        agree.copy_from_slice(&b[..BASIS_REGISTER_BYTES]);
        disagree.copy_from_slice(&b[BASIS_REGISTER_BYTES..]);
        Self { agree, disagree }
    }
}

/// Shannon entropy in bits over a count distribution; `0.0` on an empty one.
fn shannon(counts: &[usize]) -> f32 {
    let n: usize = counts.iter().sum();
    if n == 0 {
        return 0.0;
    }
    let n = n as f32;
    let mut h = 0.0f32;
    for &k in counts {
        if k > 0 {
            let p = k as f32 / n;
            h -= p * p.log2();
        }
    }
    h
}

// ═══════════════════════════════════════════════════════════════════════════
// Readout adapters — each lane family grounded in a SHIPPED certificate,
// checked against the tree rather than invented (2026-09-01):
//
// | lane family | exact carrier (stays exact) | lane carries (u4) | shipped surface |
// |---|---|---|---|
// | support / falsifier pressure (Tarski) | premise ancestry / W / CE64 | the pair's own counts | this type |
// | ΔH / expected info gain (Shannon) | the candidate distribution | `info_gain_u4` | `dismech_candidates::Evaluation` counts |
// | tension / residual (EWA) | `sigma_propagation::Spd2` Σ | `sigma_tension_u4` | `ewa_sandwich` + `log_norm_growth` + `pillar_5plus_bound` (Pillar 6, jc-verified 10000/10000 PSD) |
// | path-signature identity (Hambly-Lyons) | — | **NO LANE YET** | sigker's classification is ASSERTED, gated on jc Pillar 11 (DEFERRED) — same rule as the red-pillar mint gate: no lane while the pillar is red |
// ═══════════════════════════════════════════════════════════════════════════

/// Shannon expected-information-gain readout, quantized to a u4 lane:
/// `log2(before / after)` bits of candidate-set narrowing, floored per
/// whole bit, saturated at [`AXIS_COUNT_MAX`].
///
/// Proprioception, NOT evidence: the exact quantity stays derivable from
/// the candidate distribution itself (`dismech_candidates::Evaluation`
/// carries the counts); this lane only lets a node FEEL how sharply its
/// question is narrowing. `after == 0` — complete elimination, the
/// contradiction case — reads as maximal pressure (15). No narrowing
/// (`after >= before`) reads `0`.
#[must_use]
pub fn info_gain_u4(before: usize, after: usize) -> u8 {
    if after == 0 {
        return if before == 0 { 0 } else { AXIS_COUNT_MAX };
    }
    if after >= before {
        return 0;
    }
    let bits = (before as f64 / after as f64).log2();
    (bits.floor() as u64).min(u64::from(AXIS_COUNT_MAX)) as u8
}

/// EWA-tension readout, quantized to a u4 lane: `|log_norm_growth|`
/// measured in QUARTERS of the runtime concentration bound
/// (`sigma_propagation::pillar_5plus_bound`-derived), saturated at 15.
/// `4` = exactly at the certificate; `> 7` = past the 1.75× PASS slack
/// Pillar 6 rejects at.
///
/// The exact Σ stays in the SPD carrier; this lane is transformed
/// tension/residual only. A degenerate (`<= 0`) bound with nonzero
/// growth reads as maximal pressure — fail-loud, never fail-silent.
#[must_use]
pub fn sigma_tension_u4(growth: f64, bound: f64) -> u8 {
    let g = growth.abs();
    if bound <= 0.0 {
        return if g == 0.0 { 0 } else { AXIS_COUNT_MAX };
    }
    let quarters = (g / bound) * 4.0;
    if !quarters.is_finite() {
        return AXIS_COUNT_MAX;
    }
    (quarters.ceil() as u64).min(u64::from(AXIS_COUNT_MAX)) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_and_saturates_per_side() {
        let mut a = [0u8; BASIS_AXES];
        let mut d = [0u8; BASIS_AXES];
        for i in 0..BASIS_AXES {
            a[i] = (i % 16) as u8;
            d[i] = ((i + 5) % 16) as u8;
        }
        let p = EpistemicBassin24::pack(&a, &d);
        assert_eq!(p.agree_counts(), a);
        assert_eq!(p.disagree_counts(), d);
        assert_eq!(EpistemicBassin24::from_le_bytes(&p.to_le_bytes()), p);
        // over-range saturates, never wraps
        let over = EpistemicBassin24::pack(&[200; BASIS_AXES], &[16; BASIS_AXES]);
        assert_eq!(over.agree_counts(), [15; BASIS_AXES]);
        assert_eq!(over.disagree_counts(), [15; BASIS_AXES]);
    }

    /// **The REVERSED falsifier** (operator co-architect ruling): equal
    /// support and refutation MUST be distinguishable from silence. This is
    /// the exact case the superseded signed net provably collapsed
    /// (`basin_lanes`' pinned test) — here it is a first-class state.
    #[test]
    fn equal_support_and_refutation_is_distinguishable_from_silence() {
        let mut a = [0u8; BASIS_AXES];
        let mut d = [0u8; BASIS_AXES];
        a[5] = 3;
        d[5] = 3;
        let p = EpistemicBassin24::pack(&a, &d);
        assert!(!p.is_silent(), "balanced conflict is NOT silence");
        assert_eq!(p.axis_state(5), AxisState::Contested);
        assert_eq!(p.net(5), 0, "the net projection still exists…");
        assert_eq!(
            p.contest(5),
            3,
            "…but no longer forgets the conflict under it"
        );
        assert_eq!(p.axis_state(6), AxisState::Silent);
        assert_ne!(p, EpistemicBassin24::SILENT);
    }

    /// Conflict survives ACCUMULATION too — the children whose stances the
    /// signed net cancelled at the parent now land as (3,3) Contested.
    #[test]
    fn accumulation_preserves_conflict_instead_of_netting_it_away() {
        let mut sup = [0u8; BASIS_AXES];
        sup[5] = 3;
        let mut refu = [0u8; BASIS_AXES];
        refu[5] = 3;
        let child_a = EpistemicBassin24::pack(&sup, &[0; BASIS_AXES]);
        let child_b = EpistemicBassin24::pack(&[0; BASIS_AXES], &refu);
        let parent = EpistemicBassin24::accumulate_children(&[child_a, child_b]);
        assert_eq!(parent.axis_state(5), AxisState::Contested);
        assert_eq!(parent.agree_counts()[5], 3);
        assert_eq!(parent.disagree_counts()[5], 3);
        assert!(EpistemicBassin24::accumulate_children(&[]).is_silent());
        // order-independent within one call (exact sums, one clamp)
        assert_eq!(
            EpistemicBassin24::accumulate_children(&[child_a, child_b]),
            EpistemicBassin24::accumulate_children(&[child_b, child_a]),
        );
    }

    #[test]
    fn entropy_is_zero_for_pure_and_maximal_at_the_uniform_four_state_mix() {
        assert_eq!(EpistemicBassin24::SILENT.entropy_bits(), 0.0);
        let mut a = [0u8; BASIS_AXES];
        let mut d = [0u8; BASIS_AXES];
        for i in 0..BASIS_AXES {
            match i % 4 {
                1 => a[i] = 2,
                2 => d[i] = 2,
                3 => {
                    a[i] = 2;
                    d[i] = 2;
                }
                _ => {}
            }
        }
        let p = EpistemicBassin24::pack(&a, &d);
        assert_eq!(p.state_counts(), [6, 6, 6, 6]);
        assert!(
            (p.entropy_bits() - 2.0).abs() < 1e-6,
            "uniform 4-state = 2 bits"
        );
        // and Contested RAISES the readout instead of vanishing from it:
        // the same axes under the signed net would read 6 agree, 6 disagree,
        // 12 silent (the contested cell folded into silence).
    }

    #[test]
    fn stance_entropy_sees_a_two_way_contest_across_children() {
        let mut sup = [0u8; BASIS_AXES];
        sup[7] = 1;
        let mut refu = [0u8; BASIS_AXES];
        refu[7] = 1;
        let kids = [
            EpistemicBassin24::pack(&sup, &[0; BASIS_AXES]),
            EpistemicBassin24::pack(&[0; BASIS_AXES], &refu),
        ];
        assert_eq!(EpistemicBassin24::stance_entropy_bits(&kids, 7), 1.0);
        assert_eq!(EpistemicBassin24::stance_entropy_bits(&kids, 8), 0.0);
        assert_eq!(EpistemicBassin24::stance_entropy_bits(&[], 0), 0.0);
    }

    #[test]
    fn info_gain_reads_whole_bits_of_narrowing_and_saturates() {
        assert_eq!(info_gain_u4(8, 8), 0, "no narrowing");
        assert_eq!(info_gain_u4(8, 4), 1);
        assert_eq!(info_gain_u4(8, 1), 3);
        assert_eq!(
            info_gain_u4(1 << 20, 1),
            15,
            "saturates at the lane ceiling"
        );
        assert_eq!(
            info_gain_u4(5, 0),
            15,
            "complete elimination = max pressure"
        );
        assert_eq!(info_gain_u4(0, 0), 0, "no candidates, no gain");
        assert_eq!(info_gain_u4(4, 8), 0, "widening is not gain");
    }

    #[test]
    fn sigma_tension_is_calibrated_to_the_pillar_bound() {
        assert_eq!(sigma_tension_u4(0.0, 1.0), 0);
        assert_eq!(
            sigma_tension_u4(1.0, 1.0),
            4,
            "at the certificate = 4 quarters"
        );
        assert_eq!(
            sigma_tension_u4(-1.0, 1.0),
            4,
            "sign of growth is not tension"
        );
        assert_eq!(
            sigma_tension_u4(1.75, 1.0),
            7,
            "exactly the 1.75x PASS slack = 7 quarters"
        );
        assert!(sigma_tension_u4(1.76, 1.0) > 7, "past the slack");
        assert_eq!(sigma_tension_u4(100.0, 1.0), 15);
        assert_eq!(
            sigma_tension_u4(0.5, 0.0),
            15,
            "degenerate bound fails loud"
        );
        assert_eq!(sigma_tension_u4(0.0, 0.0), 0);
        // grounded against the REAL shipped certificate, not a made-up bound
        let b = crate::sigma_propagation::pillar_5plus_bound(4);
        assert!(b > 0.0);
        assert_eq!(sigma_tension_u4(b, b), 4);
    }
}
