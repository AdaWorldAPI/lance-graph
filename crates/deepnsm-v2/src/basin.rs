//! `basin` — Layer 5 basin self-codes + the **"where am I uncertain" self-report**.
//!
//! This is the D-SRS-3 keystone of `self-reasoning-substrate-v1`: the graph
//! measuring, from its OWN stored meaning codes, which regions of its knowledge
//! are diffuse — and being **right about it out-of-sample**.
//!
//! ## What a basin is (structural, not routing)
//!
//! A basin = one **subject's outgoing-object neighborhood** over the whole-book
//! KG: basin `s` is the set of object words `{o : (s, p, o) ∈ base}`. This is the
//! deepnsm-v2 realization of the le-contract L1–L3 `part_of:is_a` episodic rail —
//! a subject anchors a neighborhood of what it points to. It is deliberately NOT
//! the [`crate::vocab`] routing basin-byte: routing is measured ORTHOGONAL to
//! meaning (ρ≈−0.07 vs Jina, see `lib.rs`), so a routing-partition would give
//! meaning-incoherent basins and a degenerate gate.
//!
//! ## The self-code + the width instrument
//!
//! A basin's **self-code** (Layer 5) is the [`Cam96`] of its members' centroid:
//! reconstruct each member code to its concatenated-centroid point
//! ([`Cam96Space::reconstruct`]), average the points, re-encode. The **width** is
//! the mean squared-L2 of member points to that centroid — a diffuse neighborhood
//! reads wide (= uncertain), a tight one reads narrow (= confident).
//!
//! ## The held-out gate (never in-sample) — G-SRS3-1
//!
//! Split-half by index parity: `width_A` from the even members' own centroid,
//! `width_B` from the odd members' own — the halves never see each other.
//! [`heldout_split_gate`] Spearman-correlates `width_A` vs `width_B` across all
//! basins with ≥ `min_members` members. A basin the graph reports wide on half
//! its evidence being wide on the *other* half is the self-report being reliable
//! out-of-sample. The DISTRIBUTION's edge is **algebraic** (independently-
//! addressable rails, exact additive-decomposition distance), per
//! `E-CAM96-REVIEW-CORRECTIONS-1` — not raw fidelity.

use crate::space::{Cam96, Cam96Space};
use std::collections::HashMap;

/// A basin's Layer-5 self-measurement: its centroid self-code, its distribution
/// width, and the MUL-facing competence/curiosity the width induces.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BasinCode {
    /// The subject entity that anchors this basin (its outgoing neighborhood).
    pub subject: u16,
    /// The Cam96 centroid of the member object codes — the basin's self-code.
    pub self_code: Cam96,
    /// Distribution width: mean squared-L2 of member points to the centroid.
    /// Larger = more diffuse = the graph is *less* certain about this region.
    pub width: f32,
    /// Member count (objects in the neighborhood).
    pub members: usize,
    /// Contradiction density (REPORTED, not gated): fraction of `(s,p)` slots
    /// carrying > 1 distinct object — structural ambiguity in the neighborhood.
    pub contradiction: f32,
}

impl BasinCode {
    /// MUL self-measurement: competence ∈ [0,1] = `1 − width/max_width`, the
    /// signal `lance-graph-planner/mul` (Dunning-Kruger / compass) consumes.
    /// A derived READ over `max_width`, not a new tenant (plan §3).
    #[must_use]
    pub fn competence(&self, max_width: f32) -> f32 {
        // A non-positive / non-finite max_width means "no uncertainty scale" →
        // fully competent (nothing to explore). Written positively to keep the
        // partial-order comparison unambiguous (clippy neg_cmp_op_on_partial_ord).
        if max_width.is_finite() && max_width > 0.0 {
            (1.0 - self.width / max_width).clamp(0.0, 1.0)
        } else {
            1.0
        }
    }

    /// `curiosity = 1 − competence` — the exact `mul::compass CompassNeedles`
    /// value: the graph is drawn to explore the basins it is least certain of.
    #[must_use]
    pub fn curiosity(&self, max_width: f32) -> f32 {
        1.0 - self.competence(max_width)
    }
}

/// Compute the centroid point of a set of member codes (mean of reconstructed
/// points). Returns `None` for an empty set. All member points share the fixed
/// length `12·axis_dim` ([`Cam96Space::reconstruct`]), so the mean is well-defined.
fn centroid_point(space: &Cam96Space, members: &[Cam96]) -> Option<Vec<f32>> {
    let first = members.first()?;
    let d = space.reconstruct(first);
    let n = members.len() as f32;
    let mut acc = vec![0.0f32; d.len()];
    for m in members {
        for (a, x) in acc.iter_mut().zip(space.reconstruct(m)) {
            *a += x;
        }
    }
    for a in &mut acc {
        *a /= n;
    }
    Some(acc)
}

/// Mean squared-L2 of member points to a fixed centroid point — the basin width.
fn spread_about(space: &Cam96Space, members: &[Cam96], centroid: &[f32]) -> f32 {
    if members.is_empty() {
        return 0.0;
    }
    let total: f32 = members
        .iter()
        .map(|m| {
            space
                .reconstruct(m)
                .iter()
                .zip(centroid)
                .map(|(&x, &c)| (x - c) * (x - c))
                .sum::<f32>()
        })
        .sum();
    total / members.len() as f32
}

/// Compute a basin self-code from its member object codes and its `(predicate,
/// object)` edge list (for the contradiction-density report). `edges` is the
/// subject's outgoing `(predicate, object)` pairs; `members` the object codes.
#[must_use]
pub fn basin_self_code(
    space: &Cam96Space,
    subject: u16,
    members: &[Cam96],
    edges: &[(u16, u16)],
) -> Option<BasinCode> {
    let centroid = centroid_point(space, members)?;
    let width = spread_about(space, members, &centroid);
    // Contradiction density: fraction of predicates whose object set has > 1
    // distinct object under this subject.
    let mut by_p: HashMap<u16, std::collections::HashSet<u16>> = HashMap::new();
    for &(p, o) in edges {
        by_p.entry(p).or_default().insert(o);
    }
    let contradiction = if by_p.is_empty() {
        0.0
    } else {
        by_p.values().filter(|os| os.len() > 1).count() as f32 / by_p.len() as f32
    };
    Some(BasinCode {
        subject,
        self_code: space.encode(&centroid),
        width,
        members: members.len(),
        contradiction,
    })
}

/// The result of the held-out split-half reliability gate (G-SRS3-1).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeldOutGate {
    /// Basins with ≥ `min_members` members (both halves non-trivial).
    pub basins: usize,
    /// Spearman ρ across those basins between the even-half width and the
    /// odd-half width — the out-of-sample reliability of the width self-report.
    pub rho: f32,
    /// The pre-registered PASS floor (ρ ≥ `floor`).
    pub floor: f32,
    /// `min_members` used (≥ 6 per registration: each half ≥ 3).
    pub min_members: usize,
}

impl HeldOutGate {
    /// PASS ⇔ ρ ≥ floor (and enough basins to correlate). KILL ⇔ ρ ≤ 0.
    #[must_use]
    pub fn passed(&self) -> bool {
        self.basins >= 3 && self.rho >= self.floor
    }
    /// The registered KILL condition: uncorrelated or inverted.
    #[must_use]
    pub fn killed(&self) -> bool {
        self.basins >= 3 && self.rho <= 0.0
    }
}

/// Run the G-SRS3-1 held-out gate over `groups` (subject → member object codes).
///
/// For each basin with ≥ `min_members` members, split members by index parity
/// into even (A) and odd (B) halves, compute each half's width about its OWN
/// centroid, then Spearman-correlate `width_A` vs `width_B` across basins. The
/// two halves never share evidence, so a positive ρ cannot arise in-sample.
#[must_use]
pub fn heldout_split_gate(
    space: &Cam96Space,
    groups: &[(u16, Vec<Cam96>)],
    min_members: usize,
    floor: f32,
) -> HeldOutGate {
    let (mut wa, mut wb) = (Vec::new(), Vec::new());
    for (_s, members) in groups {
        if members.len() < min_members {
            continue;
        }
        let a: Vec<Cam96> = members.iter().step_by(2).copied().collect();
        let b: Vec<Cam96> = members.iter().skip(1).step_by(2).copied().collect();
        // min_members ≥ 6 ⇒ each half has ≥ 3; guard anyway.
        let (Some(ca), Some(cb)) = (centroid_point(space, &a), centroid_point(space, &b)) else {
            continue;
        };
        wa.push(spread_about(space, &a, &ca));
        wb.push(spread_about(space, &b, &cb));
    }
    HeldOutGate {
        basins: wa.len(),
        rho: spearman(&wa, &wb),
        floor,
        min_members,
    }
}

/// Run the G-SRS3-2 **constant-n** held-out gate: fix the per-half sample size
/// to `k` so member-count cannot vary across basins (removing the plug-in-width
/// n-bias that confounds [`heldout_split_gate`], `E[width] ≈ σ²(1 − 1/n)`).
///
/// For each basin with ≥ `2k` members, take the first `2k` codes, split by index
/// parity into A/B (each exactly `k`), width about each half's own centroid,
/// then Spearman across basins. With n fixed at `k`, a positive ρ on the REAL
/// binding vs ≈0 on a size-preserving label-shuffle (the caller's null) is a
/// SEMANTIC self-measurement, not an artifact. `min_members` on the returned
/// gate reports `2k`.
#[must_use]
pub fn heldout_constant_n_gate(
    space: &Cam96Space,
    groups: &[(u16, Vec<Cam96>)],
    k: usize,
    floor: f32,
) -> HeldOutGate {
    let (mut wa, mut wb) = (Vec::new(), Vec::new());
    for (_s, members) in groups {
        if k == 0 || members.len() < 2 * k {
            continue;
        }
        let head = &members[..2 * k];
        let a: Vec<Cam96> = head.iter().step_by(2).copied().collect();
        let b: Vec<Cam96> = head.iter().skip(1).step_by(2).copied().collect();
        let (Some(ca), Some(cb)) = (centroid_point(space, &a), centroid_point(space, &b)) else {
            continue;
        };
        wa.push(spread_about(space, &a, &ca));
        wb.push(spread_about(space, &b, &cb));
    }
    HeldOutGate {
        basins: wa.len(),
        rho: spearman(&wa, &wb),
        floor,
        min_members: 2 * k,
    }
}

/// A **Bessel-corrected** variable-n held-out gate: like [`heldout_split_gate`]
/// but each half's width is multiplied by `m/(m−1)` (its own half size `m`),
/// which removes the `E[width] ≈ σ²(1 − 1/n)` plug-in bias ANALYTICALLY while
/// keeping ALL members (full statistical power, unlike the constant-n gate that
/// discards evidence). Across basins the corrected width no longer tracks `n`
/// through the bias, so a size-preserving label-shuffle null should collapse to
/// ≈ 0. Exploratory power-check that distinguishes "weak because underpowered"
/// (constant-n k too small) from "weak because no signal" — NOT a registered
/// gate; the pre-registered verdict is [`heldout_constant_n_gate`]'s.
#[must_use]
pub fn heldout_bessel_gate(
    space: &Cam96Space,
    groups: &[(u16, Vec<Cam96>)],
    min_members: usize,
    floor: f32,
) -> HeldOutGate {
    let (mut wa, mut wb) = (Vec::new(), Vec::new());
    let bessel = |members: &[Cam96]| -> Option<f32> {
        let m = members.len();
        if m < 2 {
            return None;
        }
        let c = centroid_point(space, members)?;
        Some(spread_about(space, members, &c) * (m as f32) / (m as f32 - 1.0))
    };
    for (_s, members) in groups {
        if members.len() < min_members {
            continue;
        }
        let a: Vec<Cam96> = members.iter().step_by(2).copied().collect();
        let b: Vec<Cam96> = members.iter().skip(1).step_by(2).copied().collect();
        let (Some(va), Some(vb)) = (bessel(&a), bessel(&b)) else {
            continue;
        };
        wa.push(va);
        wb.push(vb);
    }
    HeldOutGate {
        basins: wa.len(),
        rho: spearman(&wa, &wb),
        floor,
        min_members,
    }
}

/// Spearman rank correlation = Pearson on average-rank-transformed values.
/// Returns `0.0` for < 2 points or a zero-variance side (no signal).
pub(crate) fn spearman(x: &[f32], y: &[f32]) -> f32 {
    if x.len() != y.len() || x.len() < 2 {
        return 0.0;
    }
    pearson(&average_ranks(x), &average_ranks(y))
}

/// Average (fractional) ranks — ties share the mean of the ranks they span.
fn average_ranks(v: &[f32]) -> Vec<f32> {
    let n = v.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut ranks = vec![0.0f32; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && v[idx[j]] == v[idx[i]] {
            j += 1;
        }
        // ranks i..j (0-based) share the average of (i+1..=j) 1-based ranks.
        let avg = ((i + 1 + j) as f32) / 2.0; // mean of i+1 .. j inclusive
        for &k in &idx[i..j] {
            ranks[k] = avg;
        }
        i = j;
    }
    ranks
}

/// Pearson correlation. `0.0` when either side has zero variance.
fn pearson(x: &[f32], y: &[f32]) -> f32 {
    let n = x.len() as f32;
    let mx = x.iter().sum::<f32>() / n;
    let my = y.iter().sum::<f32>() / n;
    let mut sxy = 0.0f32;
    let mut sxx = 0.0f32;
    let mut syy = 0.0f32;
    for (&a, &b) in x.iter().zip(y) {
        let dx = a - mx;
        let dy = b - my;
        sxy += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
    }
    let denom = (sxx * syy).sqrt();
    if denom <= 0.0 {
        0.0
    } else {
        sxy / denom
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A wide neighborhood (scattered object codes) reads wider than a tight one.
    #[test]
    fn tight_basin_narrower_than_diffuse_basin() {
        let space = Cam96Space::demo(4);
        // Tight: 4 near-identical codes. Diffuse: 4 far-apart codes.
        let tight = [
            [1u8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1u8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
            [1u8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1],
            [1u8, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1],
        ];
        let diffuse = [
            [0u8, 40, 80, 120, 160, 200, 10, 50, 90, 130, 170, 210],
            [200u8, 10, 150, 30, 90, 250, 40, 5, 199, 12, 77, 3],
            [5u8, 250, 12, 200, 33, 8, 240, 100, 2, 180, 44, 255],
            [128u8, 64, 32, 200, 16, 240, 8, 248, 4, 252, 2, 254],
        ];
        let bt = basin_self_code(&space, 1, &tight, &[]).unwrap();
        let bd = basin_self_code(&space, 2, &diffuse, &[]).unwrap();
        assert!(
            bt.width < bd.width,
            "tight width {} !< diffuse width {}",
            bt.width,
            bd.width
        );
    }

    /// A single-member basin has zero width (the point IS its own centroid).
    #[test]
    fn singleton_basin_has_zero_width() {
        let space = Cam96Space::demo(4);
        let b = basin_self_code(
            &space,
            7,
            &[[3u8, 9, 200, 4, 0, 255, 1, 2, 3, 4, 5, 6]],
            &[],
        )
        .unwrap();
        assert_eq!(b.width, 0.0);
        assert_eq!(b.members, 1);
    }

    /// Contradiction density: two objects under one predicate ⇒ 1.0; distinct
    /// predicates each with one object ⇒ 0.0.
    #[test]
    fn contradiction_density_counts_multi_object_predicates() {
        let space = Cam96Space::demo(4);
        let members = vec![[0u8; 12], [1u8; 12]];
        let clash = basin_self_code(&space, 1, &members, &[(5, 10), (5, 11)]).unwrap();
        assert_eq!(clash.contradiction, 1.0);
        let clean = basin_self_code(&space, 1, &members, &[(5, 10), (6, 11)]).unwrap();
        assert_eq!(clean.contradiction, 0.0);
    }

    /// competence/curiosity are complementary and bounded, and the widest basin
    /// (width == max_width) has zero competence / full curiosity.
    #[test]
    fn competence_and_curiosity_are_complementary() {
        let space = Cam96Space::demo(4);
        let b = basin_self_code(&space, 1, &[[0u8; 12], [200u8; 12]], &[]).unwrap();
        let mw = b.width; // this basin IS the widest
        assert!((b.competence(mw) - 0.0).abs() < 1e-6);
        assert!((b.curiosity(mw) - 1.0).abs() < 1e-6);
        // zero max_width ⇒ fully competent (no uncertainty to explore).
        assert_eq!(b.competence(0.0), 1.0);
    }

    /// Spearman is +1 for a monotone-agreeing pair, −1 for reversed, and the
    /// average-rank path handles ties without NaN.
    #[test]
    fn spearman_monotone_reversed_and_ties() {
        assert!((spearman(&[1.0, 2.0, 3.0, 4.0], &[10.0, 20.0, 30.0, 40.0]) - 1.0).abs() < 1e-6);
        assert!((spearman(&[1.0, 2.0, 3.0, 4.0], &[40.0, 30.0, 20.0, 10.0]) + 1.0).abs() < 1e-6);
        // all-tied side ⇒ zero variance ⇒ 0.0, not NaN.
        assert_eq!(spearman(&[1.0, 1.0, 1.0], &[3.0, 1.0, 2.0]), 0.0);
    }

    /// The held-out gate: a set of basins whose per-half widths agree (built so
    /// that width is basin-intrinsic and split-stable) passes; a set whose widths
    /// are assigned by parity noise does not. This is a MECHANISM test on
    /// synthetic groups — the real falsifier runs in `examples/bible_wave.rs`.
    #[test]
    fn heldout_gate_passes_on_reliable_widths_only() {
        let space = Cam96Space::demo(4);
        // Build 6 basins with graded intrinsic spread: basin k scatters its
        // members over a k-scaled code range, so both halves see the same
        // ordering. Each basin has 8 members (≥6, each half ≥3).
        let mut reliable: Vec<(u16, Vec<Cam96>)> = Vec::new();
        for k in 1..=6u16 {
            let scale = k as usize;
            let members: Vec<Cam96> = (0..8)
                .map(|i| std::array::from_fn(|ax| ((i * scale * 3 + ax) % 251) as u8))
                .collect();
            reliable.push((k, members));
        }
        let g = heldout_split_gate(&space, &reliable, 6, 0.35);
        assert_eq!(g.basins, 6);
        assert!(
            g.passed(),
            "graded-spread basins must pass the reliability gate: ρ={}",
            g.rho
        );
        assert!(!g.killed());
    }

    /// STRUCTURAL guarantees of the constant-n gate: it counts exactly the
    /// basins with ≥ 2k members, reports `min_members = 2k`, and never NaNs on a
    /// zero-variance (all-identical) input. (The SEMANTIC separation from a
    /// shuffled null is proven on the real corpus in `examples/bible_wave.rs`,
    /// not on the mod-13-periodic demo codebook, which cannot carry graded
    /// synthetic spread faithfully.)
    #[test]
    fn constant_n_gate_structural_guarantees() {
        let space = Cam96Space::demo(4);
        // 4 basins: sizes 12, 10, 9, 4. With k=5 (2k=10), only the first two
        // qualify. Codes are arbitrary but distinct so widths are finite.
        let mk = |seed: u16, n: usize| -> (u16, Vec<Cam96>) {
            let members = (0..n)
                .map(|i| std::array::from_fn(|ax| ((i * 3 + ax * 5 + seed as usize) % 200) as u8))
                .collect();
            (seed, members)
        };
        let groups = vec![mk(1, 12), mk(2, 10), mk(3, 9), mk(4, 4)];
        let g = heldout_constant_n_gate(&space, &groups, 5, 0.30);
        assert_eq!(g.basins, 2, "only basins with ≥2k=10 members qualify");
        assert_eq!(g.min_members, 10, "min_members reports 2k");
        assert!(g.rho.is_finite(), "ρ is never NaN");

        // k=0 ⇒ no basins; all-identical members ⇒ zero-variance ⇒ ρ=0 not NaN.
        assert_eq!(heldout_constant_n_gate(&space, &groups, 0, 0.30).basins, 0);
        let flat = vec![(1u16, vec![[7u8; 12]; 12]), (2u16, vec![[9u8; 12]; 12])];
        let gf = heldout_constant_n_gate(&space, &flat, 5, 0.30);
        assert_eq!(gf.rho, 0.0, "zero-variance widths ⇒ ρ=0, no NaN");
    }
}
