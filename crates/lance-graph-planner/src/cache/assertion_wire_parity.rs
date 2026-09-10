//! **D-BBB-NARS-2 cross-crate fuse** — `lance_graph_contract::assertion_wire`
//! against `causal_edge::{CausalEdge64, CausalEdgeV3, layout}`.
//!
//! The contract's `AssertionWire` reads the 16-byte edge facet at byte
//! positions that MIRROR `causal_edge::edge_v3`'s documented layout, and its
//! two vocabularies (`AssertionTopology`, `AssertionBand`) MIRROR
//! `causal_edge::layout::{CausalTopology, ReasoningBand}`. Both crates are
//! zero-dep and cannot import each other, so neither mirror can be checked
//! where it lives. This planner is the only crate holding both; this module is
//! the fuse: if a byte moves or an ordinal/name drifts on either side, a test
//! here goes red before any consumer reads a plausible wrong assertion.
//!
//! Measurement only; compiled out of every non-test build (same footing as
//! [`super::stage26_v3_parity`]).

use causal_edge::edge::InferenceType;
use causal_edge::layout::{CausalTopology, ReasoningBand};
use causal_edge::pearl::CausalMask;
use causal_edge::plasticity::PlasticityState;
use causal_edge::{CausalEdge64, CausalEdgeV3};
use lance_graph_contract::assertion_wire::{
    AssertionBand, AssertionTopology, AssertionWire, BandPresence, BandReading, EdgeProvenance,
    TruthLens, WitnessKind,
};

const CLASSID: u32 = 0x0902_0011;

fn topology_present() -> BandReading {
    BandReading {
        truth_lens: TruthLens::Topology,
        band: BandPresence::Present,
        witness: WitnessKind::CausalFacet,
    }
}

/// A deterministic sweep of CE64-v2 edges covering every topology × band and
/// varied SPO / truth / mask / direction / mantissa / plasticity / w_slot.
fn sweep() -> Vec<(CausalEdge64, u16)> {
    let mut out = Vec::new();
    let mut k: u32 = 0;
    for topo in TOPOLOGIES {
        for band in BANDS {
            k += 1;
            let s = (k * 37 % 251) as u8;
            let p = (k * 59 % 253) as u8;
            let o = (k * 83 % 241) as u8;
            let f = (k * 101 % 256) as u8;
            let c = (k * 131 % 256) as u8;
            let mask = CausalMask::from_bits((k % 8) as u8);
            let dir = (k % 7) as u8;
            let plast = PlasticityState::from_bits((k % 7) as u8);
            let e =
                CausalEdge64::pack(s, p, o, f, c, mask, dir, InferenceType::Deduction, plast, 0)
                    .with_w_slot((k % 64) as u8)
                    .with_topology(topo)
                    .with_reasoning_band(band);
            let mut e = e;
            // Exercise the raw signed mantissa, including the 8 states
            // `InferenceType` cannot round-trip (edge_v3 module doc).
            e.set_inference_mantissa(((k % 16) as i8) - 8);
            out.push((e, (k * 977 % 65_536) as u16));
        }
    }
    out
}

/// The edge crate's vocabularies in wire order — spelled out here so the fuse
/// cannot pass by reading the mirror through itself.
const TOPOLOGIES: [CausalTopology; 4] = [
    CausalTopology::Direct,
    CausalTopology::IndirectKnownIntermediates,
    CausalTopology::IndirectUnknownIntermediates,
    CausalTopology::Unknown,
];
const BANDS: [ReasoningBand; 8] = [
    ReasoningBand::Surface,
    ReasoningBand::Association,
    ReasoningBand::Relation,
    ReasoningBand::Causal,
    ReasoningBand::Counterfactual,
    ReasoningBand::Perspective,
    ReasoningBand::Meta,
    ReasoningBand::Transcendent,
];

#[test]
fn the_two_wire_vocabularies_mirror_the_edge_crate_ordinal_and_name_exact() {
    for (i, t) in TOPOLOGIES.iter().enumerate() {
        let w = AssertionTopology::from_bits_2(i as u8);
        assert_eq!(w.to_bits_2(), t.to_bits_2(), "topology ordinal {i}");
        assert_eq!(w.label(), format!("{t:?}"), "topology name at ordinal {i}");
    }
    for (i, b) in BANDS.iter().enumerate() {
        let w = AssertionBand::from_bits_3(i as u8);
        assert_eq!(w.to_bits_3(), b.to_bits_3(), "band ordinal {i}");
        assert_eq!(w.label(), format!("{b:?}"), "band name at ordinal {i}");
    }
    assert_eq!(AssertionTopology::ALL.len(), 4);
    assert_eq!(AssertionBand::ALL.len(), 8);
}

#[test]
fn every_wire_byte_position_matches_the_v3_register_across_the_sweep() {
    let edges = sweep();
    assert_eq!(edges.len(), 32, "4 topologies × 8 bands");
    for (e, target) in edges {
        let v3 = CausalEdgeV3::from_v1(e, target);
        let w = AssertionWire::from_parts(CLASSID, v3.to_le_bytes());
        assert_eq!(w.classid(), CLASSID);
        assert_eq!(w.payload(), v3.to_le_bytes());
        assert_eq!(w.frequency_u8(), v3.frequency());
        assert_eq!(w.confidence_u8(), v3.confidence());
        assert_eq!(w.causal_mask_bits(), v3.causal_mask() as u8 & 0b111);
        assert_eq!(w.direction_bits(), v3.direction());
        assert_eq!(w.inference_mantissa(), v3.inference_mantissa());
        assert_eq!(w.plasticity_bits(), v3.plasticity().bits());
        assert_eq!(w.target(), v3.target());
        assert_eq!(w.w_slot(), v3.w_slot());
        assert_eq!(w.topology_raw(), v3.truth_raw());
        assert_eq!(w.band_raw(), v3.spare_raw());
        // …and against the CE64 the V3 was lifted from.
        assert_eq!(w.frequency_u8(), e.frequency_u8());
        assert_eq!(w.confidence_u8(), e.confidence_u8());
        assert_eq!(w.topology_raw(), e.topology().to_bits_2());
        assert_eq!(w.band_raw(), e.reasoning_band().to_bits_3());
        assert_eq!(w.w_slot(), e.w_slot());
        // The defining read projects to the SAME labels the edge crate reads.
        let view = w
            .read(topology_present(), EdgeProvenance::V3Register)
            .expect("declared + asserted ⇒ readable");
        assert_eq!(view.topology.label(), format!("{:?}", e.topology()));
        assert_eq!(view.band.label(), format!("{:?}", e.reasoning_band()));
        assert_eq!(view.frequency, e.frequency_u8());
        assert_eq!(view.confidence, e.confidence_u8());
        assert_eq!(view.causal_mask_bits, e.causal_mask() as u8 & 0b111);
        assert_eq!(view.target, target);
        // Rehydrating the payload the wire carries restores the CE64 bit-exact.
        let back =
            CausalEdgeV3::from_le_bytes(w.payload()).rehydrate(e.s_idx(), e.p_idx(), e.o_idx());
        assert_eq!(
            back.0, e.0,
            "CE64 → V3 → wire → V3 → CE64 must be bit-exact"
        );
    }
}

/// The aliasing pair, end to end through the substrate: two CE64 edges equal
/// in S,P,O and (f,c), differing only in topology × band, must stay two claims
/// on the wire, and the substrate must agree on which is which.
#[test]
fn the_aliasing_pair_survives_ce64_to_wire_and_back_as_two_claims() {
    let base = CausalEdge64::pack(
        7,
        0x90, // "causes" — dismech_evidence::DISMECH_PREDICATES 0x90
        42,
        192,
        217,
        CausalMask::SPO,
        0,
        InferenceType::Deduction,
        PlasticityState::from_bits(0),
        0,
    );
    let a = base
        .with_topology(CausalTopology::IndirectUnknownIntermediates)
        .with_reasoning_band(ReasoningBand::Relation);
    let b = base
        .with_topology(CausalTopology::IndirectKnownIntermediates)
        .with_reasoning_band(ReasoningBand::Causal);
    assert_eq!(
        (a.frequency_u8(), a.confidence_u8()),
        (b.frequency_u8(), b.confidence_u8())
    );
    assert_eq!(
        (a.s_idx(), a.p_idx(), a.o_idx()),
        (b.s_idx(), b.p_idx(), b.o_idx())
    );
    assert_ne!(a.0, b.0);

    let wa = AssertionWire::from_parts(CLASSID, CausalEdgeV3::from_v1(a, 9).to_le_bytes());
    let wb = AssertionWire::from_parts(CLASSID, CausalEdgeV3::from_v1(b, 9).to_le_bytes());
    let va = wa
        .read(topology_present(), EdgeProvenance::V3Register)
        .unwrap();
    let vb = wb
        .read(topology_present(), EdgeProvenance::V3Register)
        .unwrap();
    assert_ne!(va, vb, "epistemic aliasing on the wire");
    assert_eq!(va.topology, AssertionTopology::IndirectUnknownIntermediates);
    assert_eq!(va.band, AssertionBand::Relation);
    assert_eq!(vb.topology, AssertionTopology::IndirectKnownIntermediates);
    assert_eq!(vb.band, AssertionBand::Causal);

    // The substrate reads the same two claims back from the wire payloads.
    let ra = CausalEdgeV3::from_le_bytes(wa.payload()).rehydrate(7, 0x90, 42);
    let rb = CausalEdgeV3::from_le_bytes(wb.payload()).rehydrate(7, 0x90, 42);
    assert_eq!(ra.0, a.0);
    assert_eq!(rb.0, b.0);
    assert_eq!(ra.topology(), CausalTopology::IndirectUnknownIntermediates);
    assert_eq!(rb.reasoning_band(), ReasoningBand::Causal);
}

/// The silent twin of the refusal rule, across the crates: a lift that
/// DROPPED the tail (`from_v1_tail_unstated`) yields a wire whose defining
/// fields read as zero — and the contract refuses to hand that out unless the
/// caller asserts provenance. Asserting `V3Register` over an unstated lift is
/// the caller's lie, not the contract's.
#[test]
fn an_unstated_tail_lift_reads_as_direct_surface_only_if_the_caller_asserts_it() {
    let e = CausalEdge64::ZERO
        .with_topology(CausalTopology::Unknown)
        .with_reasoning_band(ReasoningBand::Meta);
    let w = AssertionWire::from_parts(
        CLASSID,
        CausalEdgeV3::from_v1_tail_unstated(e, 1).to_le_bytes(),
    );
    assert_eq!(w.topology_raw(), 0);
    assert_eq!(w.band_raw(), 0);
    assert!(w.read(topology_present(), EdgeProvenance::Unknown).is_err());
    // A truthful lift keeps the claim.
    let w2 = AssertionWire::from_parts(CLASSID, CausalEdgeV3::from_v1(e, 1).to_le_bytes());
    let v = w2
        .read(topology_present(), EdgeProvenance::V3Register)
        .unwrap();
    assert_eq!(v.topology, AssertionTopology::Unknown);
    assert_eq!(v.band, AssertionBand::Meta);
}
