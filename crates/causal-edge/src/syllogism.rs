//! Hardwired NAL syllogism resolution on `CausalEdge64` — the **figure decomposition**.
//!
//! # The missing capstone
//!
//! `CausalEdge64::forward` composes two edges *positionally* and applies the
//! weight edge's **pre-set** inference type. It never asks *which term the two
//! edges share* — and that question is the whole of NAL syllogism. This module
//! adds the missing piece: given two premise edges, detect the **syllogistic
//! figure** by integer SPO-palette term-matching, select the matching NARS
//! rule, and emit the conclusion edge.
//!
//! This is hardwired exactly like **Pearl's 2³ causal mask** (`pearl.rs`): a
//! small, fixed, branch-minimal structural decomposition resolved by integer
//! equality — `O(1)`, no allocation, no float on the structural path. Where the
//! Pearl mask answers *"which SPO planes condition this edge?"*, the figure
//! answers *"which term bridges these two edges?"*.
//!
//! # The four figures (the `{premises} |- conclusion` notation)
//!
//! An edge `(s, p, o)` is read as the directed statement `s --[p]--> o` (the
//! subject inheres in / maps to the object; `p` labels the relation). Two
//! premises share exactly one **middle term** `M`; *where* `M` sits decides the
//! figure and therefore the rule (mirrors `RuleTables.syllogism` in OpenNARS and
//! the `{M-->P, S-->M} |- S-->P` notation carried by
//! `cognitive_codebook::NarsInference`):
//!
//! ```text
//! self = s1->o1     other = s2->o2     shared term      figure          rule
//! ────────────────────────────────────────────────────────────────────────────
//! o1 == s2          (M = o1)           {S->M, M->P}     Chain           Deduction   ⊢ s1->o2
//! s1 == o2          (M = s1)           {M->P, S->M}     ChainRev        Deduction   ⊢ s2->o1
//! s1 == s2          (M = s1)           {M->P, M->S}     SharedSubject   Induction   ⊢ o1->o2
//! o1 == o2          (M = o1)           {P->M, S->M}     SharedObject    Abduction   ⊢ s1->s2
//! ```
//!
//! # Firewall
//!
//! The figure is resolved by **integer palette equality** — it *proposes* which
//! rule applies (deterministic, no float, no language). The conclusion's truth
//! is then the canonical NARS truth-function, the same math as
//! `ndarray::hpc::nars` (the hardware) and `CausalEdge64::forward` (the
//! protocol). similarity never enters the hot path; addressing (palette index)
//! decides the figure.
//!
//! # Wiring EW64
//!
//! The premises arrive as `EpisodicEdges64` (`EdgeRef` slots) from the hot
//! episodic tier; the driver resolves each `EdgeRef` to its co-addressed
//! `CausalEdge64` (basin row) and folds [`CausalEdge64::syllogize`] across the
//! ≤4 hot edges — the relational/witness reasoning step. That resolution rides
//! the driver (it needs the basin store); this module owns only the pure,
//! zero-dep figure kernel.

use crate::edge::{CausalEdge64, InferenceType};
use crate::pearl::CausalMask;
use crate::plasticity::PlasticityState;

/// The NAL syllogistic figure — which term two SPO edges share.
///
/// The hardwired analogue of [`CausalMask`](crate::pearl::CausalMask): a
/// structural decomposition resolved by integer palette-index equality. Each
/// figure pins one NARS rule (see [`Figure::rule`]) and one conclusion shape
/// (see [`CausalEdge64::syllogize`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Figure {
    /// `o1 == s2`: the forward chain `s1->o1, o1->o2 ⊢ s1->o2`. **Deduction.**
    Chain,
    /// `s1 == o2`: the reverse chain `s2->o2, o2->o1 ⊢ s2->o1`. **Deduction.**
    ///
    /// The same transitive chain as [`Chain`](Figure::Chain) with the premises
    /// encountered in the opposite order; kept distinct so the conclusion shape
    /// is explicit.
    ChainRev,
    /// `s1 == s2`: shared subject `M->o1, M->o2 ⊢ o1->o2`. **Induction.**
    SharedSubject,
    /// `o1 == o2`: shared object `s1->M, s2->M ⊢ s1->s2`. **Abduction.**
    SharedObject,
}

impl Figure {
    /// The NARS rule this figure resolves to.
    #[inline]
    pub const fn rule(self) -> InferenceType {
        match self {
            Figure::Chain | Figure::ChainRev => InferenceType::Deduction,
            Figure::SharedSubject => InferenceType::Induction,
            Figure::SharedObject => InferenceType::Abduction,
        }
    }

    /// The canonical `{premise1, premise2} |- conclusion` NAL notation string,
    /// in `S/P/M` term names (`M` = the shared middle term).
    #[inline]
    pub const fn notation(self) -> &'static str {
        match self {
            Figure::Chain => "{S-->M, M-->P} |- S-->P",
            Figure::ChainRev => "{M-->P, S-->M} |- S-->P",
            Figure::SharedSubject => "{M-->P, M-->S} |- S-->P",
            Figure::SharedObject => "{P-->M, S-->M} |- S-->P",
        }
    }
}

/// A resolved syllogism: the detected [`Figure`] plus the conclusion edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Syllogism {
    /// Which term the premises shared.
    pub figure: Figure,
    /// The derived conclusion edge (conclusion SPO + composed NARS truth +
    /// signed inference mantissa + AND-ed Pearl mask).
    pub conclusion: CausalEdge64,
}

impl CausalEdge64 {
    /// Detect the NAL syllogistic figure shared with `other` by SPO palette
    /// term-matching — the hardwired analogue of the Pearl 2³ mask. `O(1)`.
    ///
    /// Returns `None` when the two edges share no term (no valid syllogism), or
    /// when they are the **same statement** (`s1==s2 && o1==o2`) — identical
    /// statements are merged by NARS *revision* ([`CausalEdge64::learn`]), not a
    /// syllogism. When several terms match, the figures are tried in the fixed
    /// canonical order `Chain → ChainRev → SharedSubject → SharedObject` and the
    /// first match wins.
    #[inline]
    pub fn figure(self, other: Self) -> Option<Figure> {
        let (s1, o1) = (self.s_idx(), self.o_idx());
        let (s2, o2) = (other.s_idx(), other.o_idx());
        // Same statement → revision, not syllogism.
        if s1 == s2 && o1 == o2 {
            return None;
        }
        if o1 == s2 {
            Some(Figure::Chain)
        } else if s1 == o2 {
            Some(Figure::ChainRev)
        } else if s1 == s2 {
            Some(Figure::SharedSubject)
        } else if o1 == o2 {
            Some(Figure::SharedObject)
        } else {
            None
        }
    }

    /// Resolve the NAL syllogism between `self` and `other`: detect the
    /// [`Figure`], apply the matching NARS truth-function, and emit the
    /// conclusion `CausalEdge64`.
    ///
    /// The conclusion carries:
    /// - **SPO**: the two outer terms (the middle term `M` is consumed), per the
    ///   figure table in the module docs. The predicate is carried from the
    ///   premise that contributes the conclusion's subject; *relational*
    ///   composition of the predicate palette is the complementary concern of
    ///   [`CausalEdge64::forward`]'s `compose_p` table.
    /// - **truth**: the canonical NARS rule for the figure (Deduction /
    ///   Induction / Abduction), identical to `ndarray::hpc::nars` and to
    ///   `forward`'s inline math.
    /// - **inference mantissa**: the signed v2 mantissa for the rule
    ///   (Deduction `+1`, Induction `+2`, Abduction `−1`) — forward-chain
    ///   positive, backward-chain negative.
    /// - **Pearl mask**: the bitwise AND of the two premise masks (only planes
    ///   active in *both* survive), matching `forward`.
    /// - **plasticity**: `ALL_HOT` — a freshly derived conclusion is learnable.
    ///
    /// Returns `None` exactly when [`figure`](CausalEdge64::figure) does.
    #[inline]
    pub fn syllogize(self, other: Self) -> Option<Syllogism> {
        let figure = self.figure(other)?;
        let (s1, p1, o1) = (self.s_idx(), self.p_idx(), self.o_idx());
        let (s2, p2, o2) = (other.s_idx(), other.p_idx(), other.o_idx());

        // Conclusion SPO + which premise supplies the predicate (subject side).
        let (cs, cp, co) = match figure {
            Figure::Chain => (s1, p1, o2),
            Figure::ChainRev => (s2, p2, o1),
            Figure::SharedSubject => (o1, p1, o2),
            Figure::SharedObject => (s1, p1, s2),
        };

        // Conclusion truth — canonical NARS rule for the figure (premise order
        // self, other). Mirrors `ndarray::hpc::nars` and `forward`.
        let (f1, c1) = (self.frequency(), self.confidence());
        let (f2, c2) = (other.frequency(), other.confidence());
        let (f, c) = match figure {
            Figure::Chain | Figure::ChainRev => deduction_truth(f1, c1, f2, c2),
            Figure::SharedSubject => induction_truth(f1, c1, f2, c2),
            Figure::SharedObject => abduction_truth(f1, c1, f2, c2),
        };

        // Pearl mask: AND (only planes active in both survive) — as `forward`.
        let mask = CausalMask::from_bits((self.causal_mask() as u8) & (other.causal_mask() as u8));

        // Build the conclusion. `pack` writes the signed mantissa under the v2
        // layout via `InferenceType::to_mantissa` (Deduction→+1, Induction→+2,
        // Abduction→−1), so the conclusion round-trips its rule + chain
        // direction. Temporal is structural (dropped under v2); direction is
        // neutral (recomputed downstream from composed palette dim0 signs).
        #[allow(deprecated)] // v2 `pack` drops temporal; mantissa carried via to_mantissa
        let conclusion = CausalEdge64::pack(
            cs,
            cp,
            co,
            (f.clamp(0.0, 1.0) * 255.0).round() as u8,
            (c.clamp(0.0, 1.0) * 255.0).round() as u8,
            mask,
            0, // direction: neutral; recomputed from composed palette downstream
            figure.rule(),
            PlasticityState::ALL_HOT,
            0, // temporal: structural under v2 (chain-position / AriGraph anchor)
        );

        Some(Syllogism { figure, conclusion })
    }
}

// ─── Truth-functions (mirror `ndarray::hpc::nars` + `CausalEdge64::forward`) ──
//
// Kept private to this module. The formulas are byte-identical to the canonical
// `ndarray` hardware functions and to `forward`'s inline arms; the intentional
// mirror keeps `causal-edge` zero-dep (it cannot import `ndarray`). A later DRY
// pass may factor `forward`'s arms onto these. The hot-path u8→u8 table form
// lives in `tables.rs` (deduction shipped; induction/abduction tables follow).

/// Deduction `A->B, B->C ⊢ A->C`: `f = f1·f2`, `c = f1·f2·c1·c2`.
#[inline]
fn deduction_truth(f1: f32, c1: f32, f2: f32, c2: f32) -> (f32, f32) {
    let f = f1 * f2;
    (f, f * c1 * c2)
}

/// Induction `A->B, A->C ⊢ B->C`: `f = f2`, `c = w/(w+1)`, `w = f1·c1·c2`.
#[inline]
fn induction_truth(f1: f32, c1: f32, f2: f32, c2: f32) -> (f32, f32) {
    let w = f1 * c1 * c2;
    (f2, w / (w + 1.0))
}

/// Abduction `A->B, C->B ⊢ A->C`: `f = f1`, `c = w/(w+1)`, `w = f2·c1·c2`.
#[inline]
fn abduction_truth(f1: f32, c1: f32, f2: f32, c2: f32) -> (f32, f32) {
    let w = f2 * c1 * c2;
    (f1, w / (w + 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build an edge with the given SPO palette + truth (f,c as u8). Pearl mask
    /// SPO, neutral direction, deduction stamp, all-hot.
    fn edge(s: u8, p: u8, o: u8, f: u8, c: u8) -> CausalEdge64 {
        #[allow(deprecated)]
        CausalEdge64::pack(
            s,
            p,
            o,
            f,
            c,
            CausalMask::SPO,
            0,
            InferenceType::Deduction,
            PlasticityState::ALL_HOT,
            0,
        )
    }

    #[test]
    fn figure_chain_when_o1_eq_s2() {
        // s1->o1(=M), M->o2  ⇒  Chain (deduction).
        let a = edge(10, 1, 20, 200, 200); // 10 -> 20
        let b = edge(20, 2, 30, 200, 200); // 20 -> 30   (shares M=20)
        assert_eq!(a.figure(b), Some(Figure::Chain));
        assert_eq!(Figure::Chain.rule(), InferenceType::Deduction);
    }

    #[test]
    fn figure_shared_subject_is_induction() {
        let a = edge(10, 1, 20, 200, 200); // M=10 -> 20
        let b = edge(10, 2, 30, 200, 200); // M=10 -> 30
        assert_eq!(a.figure(b), Some(Figure::SharedSubject));
        assert_eq!(Figure::SharedSubject.rule(), InferenceType::Induction);
    }

    #[test]
    fn figure_shared_object_is_abduction() {
        let a = edge(10, 1, 30, 200, 200); // 10 -> M=30
        let b = edge(20, 2, 30, 200, 200); // 20 -> M=30
        assert_eq!(a.figure(b), Some(Figure::SharedObject));
        assert_eq!(Figure::SharedObject.rule(), InferenceType::Abduction);
    }

    #[test]
    fn figure_chain_rev_when_s1_eq_o2() {
        let a = edge(20, 1, 30, 200, 200); // M=20 -> 30
        let b = edge(10, 2, 20, 200, 200); // 10 -> M=20
        // o1(30)!=s2(10); s1(20)==o2(20) ⇒ ChainRev.
        assert_eq!(a.figure(b), Some(Figure::ChainRev));
        assert_eq!(Figure::ChainRev.rule(), InferenceType::Deduction);
    }

    #[test]
    fn no_shared_term_is_none() {
        let a = edge(10, 1, 20, 200, 200);
        let b = edge(30, 2, 40, 200, 200);
        assert_eq!(a.figure(b), None);
        assert!(a.syllogize(b).is_none());
    }

    #[test]
    fn identical_statement_is_revision_not_syllogism() {
        // Same S and O ⇒ revision territory ⇒ None (even if p/truth differ).
        let a = edge(10, 1, 20, 200, 200);
        let b = edge(10, 9, 20, 100, 100);
        assert_eq!(a.figure(b), None);
    }

    #[test]
    fn chain_conclusion_links_outer_terms() {
        // 10 -> 20, 20 -> 30  ⊢  10 -> 30.
        let a = edge(10, 7, 20, 204, 204); // f=0.8 c=0.8
        let b = edge(20, 8, 30, 204, 204);
        let syl = a.syllogize(b).expect("chain resolves");
        assert_eq!(syl.figure, Figure::Chain);
        assert_eq!(syl.conclusion.s_idx(), 10, "conclusion subject = s1");
        assert_eq!(syl.conclusion.o_idx(), 30, "conclusion object = o2");
    }

    #[test]
    fn shared_subject_conclusion_links_objects() {
        // M->20, M->30  ⊢  20 -> 30 (induction).
        let a = edge(10, 1, 20, 200, 200);
        let b = edge(10, 2, 30, 200, 200);
        let syl = a.syllogize(b).unwrap();
        assert_eq!(syl.figure, Figure::SharedSubject);
        assert_eq!(syl.conclusion.s_idx(), 20);
        assert_eq!(syl.conclusion.o_idx(), 30);
    }

    #[test]
    fn shared_object_conclusion_links_subjects() {
        // 10->M, 20->M  ⊢  10 -> 20 (abduction).
        let a = edge(10, 1, 30, 200, 200);
        let b = edge(20, 2, 30, 200, 200);
        let syl = a.syllogize(b).unwrap();
        assert_eq!(syl.figure, Figure::SharedObject);
        assert_eq!(syl.conclusion.s_idx(), 10);
        assert_eq!(syl.conclusion.o_idx(), 20);
    }

    #[test]
    fn deduction_attenuates_confidence() {
        // Strong premises: deduction confidence must be below both inputs.
        let a = edge(10, 1, 20, 230, 230); // f=c≈0.90
        let b = edge(20, 2, 30, 230, 230);
        let syl = a.syllogize(b).unwrap();
        assert!(
            syl.conclusion.confidence() < a.confidence(),
            "deduction must weaken confidence: {} !< {}",
            syl.conclusion.confidence(),
            a.confidence()
        );
        // Deduction frequency = f1·f2 ≈ 0.9·0.9 = 0.81.
        let expected_f = a.frequency() * b.frequency();
        assert!((syl.conclusion.frequency() - expected_f).abs() < 0.02);
    }

    #[test]
    fn induction_and_abduction_are_weak() {
        let a = edge(10, 1, 20, 230, 230);
        let ind = a.syllogize(edge(10, 2, 30, 230, 230)).unwrap();
        let abd = a.syllogize(edge(40, 2, 20, 230, 230)).unwrap();
        assert!(ind.conclusion.confidence() < a.confidence(), "induction weak");
        assert!(abd.conclusion.confidence() < a.confidence(), "abduction weak");
    }

    #[test]
    fn pearl_mask_is_anded() {
        // SPO premise ∧ PO premise ⇒ PO conclusion.
        #[allow(deprecated)]
        let a = CausalEdge64::pack(
            10, 1, 20, 200, 200, CausalMask::SPO, 0,
            InferenceType::Deduction, PlasticityState::ALL_HOT, 0,
        );
        #[allow(deprecated)]
        let b = CausalEdge64::pack(
            20, 2, 30, 200, 200, CausalMask::PO, 0,
            InferenceType::Deduction, PlasticityState::ALL_HOT, 0,
        );
        let syl = a.syllogize(b).unwrap();
        assert_eq!(syl.conclusion.causal_mask(), CausalMask::PO);
    }

    #[test]
    fn notation_strings_are_canonical() {
        assert_eq!(Figure::SharedObject.notation(), "{P-->M, S-->M} |- S-->P");
        assert_eq!(Figure::SharedSubject.notation(), "{M-->P, M-->S} |- S-->P");
    }

    // The signed inference mantissa is a v2-layout concept; assert it only when
    // the (default) feature is on.
    #[cfg(feature = "causal-edge-v2-layout")]
    #[test]
    fn conclusion_carries_signed_chain_direction() {
        // Deduction / Induction are forward-chain (+); Abduction is backward (−).
        let a = edge(10, 1, 20, 200, 200);
        let chain = a.syllogize(edge(20, 2, 30, 200, 200)).unwrap();
        assert_eq!(chain.conclusion.inference_mantissa(), 1, "Deduction = +1");
        assert_eq!(chain.conclusion.inference_direction(), 1, "forward chain");

        let ind = a.syllogize(edge(10, 2, 30, 200, 200)).unwrap();
        assert_eq!(ind.conclusion.inference_mantissa(), 2, "Induction = +2");

        let abd = a.syllogize(edge(40, 2, 20, 200, 200)).unwrap();
        assert_eq!(abd.conclusion.inference_mantissa(), -1, "Abduction = −1");
        assert_eq!(abd.conclusion.inference_direction(), -1, "backward chain");
    }
}
