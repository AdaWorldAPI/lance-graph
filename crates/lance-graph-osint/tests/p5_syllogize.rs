//! P5 · syllogize chain — multi-hop reasoning on the palette edge.
//!
//! P4 made a single mined implication a `CausalEdge64`. P5 composes two edges
//! that share a term into a derived conclusion edge — NAL transitive deduction
//! `is_a(A,B) ∧ is_a(B,C) ⊢ is_a(A,C)`, carried entirely on the SPO palette.
//!
//! This is the `part_of`/`is_a` chaining the OSINT ontology needs (an actor
//! `part_of` a unit `part_of` a command ⊢ `part_of` the command; a system
//! `is_a` drone `is_a` dual-use-tech ⊢ dual-use-tech). The predicate is a
//! carried placeholder slot, so `is_a` and `part_of` chain by the identical
//! mechanism — the middle term `M` (shared palette index) is consumed, the two
//! outer terms survive, and the NARS deduction truth-function composes the
//! premise truths. `Figure::Chain`, `InferenceType::Deduction` (mantissa +1),
//! Pearl mask = AND of the premises.
//!
//! Integer-exact after the single f32 truth-composition (the NARS edge).

use causal_edge::{CausalEdge64, CausalMask, Figure, PlasticityState};

/// Palette slots for the OSINT ontology chain.
const A: u8 = 0x10; // e.g. a specific drone model
const B: u8 = 0x20; // its class: "reconnaissance UAV"
const C: u8 = 0x30; // its category: "dual-use technology"
const IS_A: u8 = 0x07; // predicate placeholder slot (shared)

/// Build an `is_a`-style premise edge on the palette with the given truth bytes.
fn premise(s: u8, o: u8, f: u8, c: u8) -> CausalEdge64 {
    CausalEdge64::pack_v2(s, IS_A, o, f, c, CausalMask::SPO, 0, PlasticityState::ALL_FROZEN)
}

#[test]
fn p5_is_a_chain_deduces_the_transitive_conclusion() {
    // is_a(A,B) ∧ is_a(B,C).  o1 == s2 == B  ⇒  Figure::Chain.
    let ab = premise(A, B, 255, 204); // f=1.0, c=0.8
    let bc = premise(B, C, 255, 204); // f=1.0, c=0.8

    assert_eq!(ab.figure(bc), Some(Figure::Chain), "o1==s2==B ⇒ Chain");

    let syl = ab.syllogize(bc).expect("shared middle term ⇒ a syllogism");
    assert_eq!(syl.figure, Figure::Chain);

    // Conclusion is is_a(A,C): the middle term B is consumed, outer terms survive.
    let concl = syl.conclusion;
    assert_eq!(concl.s_idx(), A, "conclusion subject = A");
    assert_eq!(concl.o_idx(), C, "conclusion object = C");

    // Deduction truth: f = f1·f2 = 1.0 → 255; c = f1·f2·c1·c2 = 1·1·0.8·0.8 = 0.64
    // → round(0.64·255) = 163.
    assert_eq!(concl.frequency_u8(), 255, "deduction f = f1·f2");
    assert_eq!(concl.confidence_u8(), 163, "deduction c = f1·f2·c1·c2");

    // Deduction stamps a positive (forward-chain) mantissa.
    assert_eq!(concl.inference_mantissa(), 1, "Deduction ⇒ mantissa +1");

    // Pearl mask = AND of the premise masks: SPO & SPO = SPO.
    assert_eq!(concl.causal_mask(), CausalMask::SPO);
}

#[test]
fn p5_no_shared_term_is_not_a_syllogism() {
    // is_a(A,B) and is_a(C,A-adjacent) with no shared S/O term ⇒ None.
    let ab = premise(A, B, 255, 200);
    let unrelated = premise(0x40, 0x50, 255, 200);
    assert_eq!(ab.figure(unrelated), None, "no shared term ⇒ no figure");
    assert!(ab.syllogize(unrelated).is_none());
}

#[test]
fn p5_identical_statement_is_revision_not_syllogism() {
    // Same S and O ⇒ NARS revision territory, not a syllogism.
    let e1 = premise(A, B, 255, 100);
    let e2 = premise(A, B, 255, 200);
    assert_eq!(e1.figure(e2), None, "identical S,O ⇒ revision, not syllogism");
    assert!(e1.syllogize(e2).is_none());
}

#[test]
fn p5_chain_is_deterministic() {
    let ab = premise(A, B, 255, 204);
    let bc = premise(B, C, 255, 204);
    let c1 = ab.syllogize(bc).unwrap().conclusion;
    let c2 = ab.syllogize(bc).unwrap().conclusion;
    assert_eq!(c1, c2, "syllogize is a pure function of its premises");
}
