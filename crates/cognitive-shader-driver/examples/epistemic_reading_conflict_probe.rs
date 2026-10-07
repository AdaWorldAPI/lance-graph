//! D-EPI-CONFLICT-0 → D-EPI-MIG-0: the conformance probe of the canonical
//! Cartesian `EpistemicState5` reading of `CausalEdge64` bits 59..63.
//!
//! # What it pinned (#1378, kept as history)
//!
//! P7a (#1369) wrote its certification into bits 61..63 and read it through
//! `band_reading::project_band`; the affordance law (#1370) read bits 59..63
//! as one code under a hand-assigned, deliberately unordered codebook. A
//! P7a-certified `Causes` over topology 0 was raw code 20, which that codebook
//! read as `Direct × Related`; 3 of 8 shared cells agreed.
//!
//! # What holds now (operator decision 2026-10-07)
//!
//! `EpistemicState5 = Topology2 × Certification3`, `raw5 = topology |
//! certification << 2`, with the ordinals already on the wire
//! (`CausalTopology` and P7a's codes). The #1370 codebook was the only thing
//! out of place; it is gone. So the canonical projection and the two old
//! split readings now agree by construction, and this probe proves it:
//!
//! | test | pin |
//! |---|---|
//! | `the_canonical_reading_is_the_two_old_lenses_read_together` | over every topology × 3-bit certification: canonical == `topology()` lens + P7a decode — 24 agree, 0 disagree, 8 reserved refuse |
//! | `undeclared_classes_refuse` | the law never measures a class without a canonical declaration |
//! | `a_certified_causes_reads_as_causes_under_every_topology` | F3: 20..23, `CAUSES` asserted, `COUNTERFACTUAL_PROBE` eligible |
//! | `the_1378_fixture_now_agrees` | P7a `Causes` over topology 0 is code 20 = `Direct × Causes` to both readers |
//! | `the_derived_table_equals_the_rules` | the factor-compiled table equals per-code rule evaluation (F6) |
//! | `one_factor_moves_only_its_recipes` | F7 + the frontier: 10↔9, 9↔13, 22, 23 |
//! | `a_legacy_reading_class_is_not_consumed_as_canonical` | F2: a historical-band class refuses even though its bits decode |
//! | `the_state_survives_restart_from_le_bytes` | F5 |
//! | `projections_never_change_the_edge` | F8 |
//!
//! Run: `cargo run -p cognitive-shader-driver --example epistemic_reading_conflict_probe`
//! Tests: `cargo test -p cognitive-shader-driver --example epistemic_reading_conflict_probe`

#[path = "shared/affordance_law.rs"]
mod affordance_law;
#[path = "shared/certification_reading.rs"]
mod certification_reading;

use affordance_law::raw5;
use causal_edge::edge::CausalEdge64;
use causal_edge::pearl::CausalMask;
use causal_edge::PlasticityState;
use certification_reading::{read, Contract, Refusal as P7aRefusal, CERT_CLASS, RAIL};
use lance_graph_contract::band_reading::EdgeProvenance;
use lance_graph_contract::epistemic_state5::{
    Epi5Declarations, Epi5Gen, Epi5Reading, EpistemicState5, Topology2,
};

/// The declarations both readers share: P7a's class, declared canonical.
fn decl() -> Epi5Declarations {
    let mut d = Epi5Declarations::new();
    d.declare(
        CERT_CLASS,
        RAIL,
        Epi5Reading {
            generation: Epi5Gen::V1,
        },
    );
    d
}

/// An edge with all three Pearl planes, so the plane rule never hides a
/// recipe and only the EpistemicState5 facts decide.
fn base() -> CausalEdge64 {
    CausalEdge64::pack_v2(
        1,
        2,
        3,
        200,
        150,
        CausalMask::SPO,
        0,
        PlasticityState::from_bits(0),
    )
}

/// The canonical projection of an edge of P7a's class.
fn canonical(edge: CausalEdge64) -> Option<EpistemicState5> {
    decl()
        .project_state5(
            CERT_CLASS,
            RAIL,
            Epi5Gen::V1,
            raw5(edge),
            EdgeProvenance::V2Stamped,
        )
        .ok()
}

/// The two OLD split readings, independent of the contract: the edge crate's
/// `CausalTopology` lens on 59..60 and P7a's own decode of 61..63.
fn split(edge: CausalEdge64) -> Option<(u8, Contract)> {
    let topology = edge.topology().to_bits_2();
    Contract::from_code(edge.spare())
        .ok()
        .map(|c| (topology, c))
}

/// (agree, disagree, refused) over every topology × 3-bit certification.
fn census() -> (usize, usize, usize) {
    let (mut agree, mut disagree, mut refused) = (0, 0, 0);
    for t in 0..4u8 {
        for c in 0..8u8 {
            let e = base().with_epistemic_raw5(t | c << 2);
            match (canonical(e), split(e)) {
                (Some(s), Some((ot, oc)))
                    if s.topology().ordinal() == ot
                        && Contract::from_certification(s.certification()) == oc =>
                {
                    agree += 1
                }
                (None, None) => refused += 1,
                _ => disagree += 1,
            }
        }
    }
    (agree, disagree, refused)
}

fn main() {
    println!("D-EPI-MIG-0: EpistemicState5 = Topology2 × Certification3 (raw5 = t | c << 2)");
    for t in Topology2::ALL {
        let row: Vec<String> = Contract::ALL
            .iter()
            .map(|c| format!("{:>2}", t.ordinal() | c.code() << 2))
            .collect();
        println!("  {t:<16?} {}", row.join(" "));
    }
    let (a, d, r) = census();
    println!(
        "agree {a}, disagree {d}, reserved/refused {r} of 32 (was 3 / 5 / 16 shared cells in #1378)"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use affordance_law::{
        measure_declared, LawGen, Refusal, CAUSAL_IDENTIFICATION, COUNTERFACTUAL_PROBE, EPI_LAW,
        HYDRATE_INTERMEDIATE, MECHANISM_FOLD,
    };
    use causal_edge::layout::ReasoningBand;
    use lance_graph_contract::class_view::ClassId;
    use lance_graph_contract::epistemic_state5::fact::{CAUSES, CERTIFICATION_MASK, TOPOLOGY_MASK};
    use lance_graph_contract::epistemic_state5::Epi5ReadError;

    /// A class never declared under the canonical reading.
    const UNDECLARED_CLASS: ClassId = 0x0904;

    fn eligible(class: ClassId, edge: CausalEdge64) -> Result<u64, Refusal> {
        measure_declared(
            LawGen::V1,
            &decl(),
            class,
            RAIL,
            edge,
            EdgeProvenance::V2Stamped,
        )
    }

    fn at(code: u8) -> u64 {
        eligible(CERT_CLASS, base().with_epistemic_raw5(code)).unwrap()
    }

    fn bits(recipes: &[u8]) -> u64 {
        recipes.iter().fold(0, |m, r| m | 1 << r)
    }

    /// F10: the canonical reading IS the two old lenses read together.
    #[test]
    fn the_canonical_reading_is_the_two_old_lenses_read_together() {
        assert_eq!(census(), (24, 0, 8));
    }

    /// FAILS IF: the law measures a class with no canonical declaration.
    #[test]
    fn undeclared_classes_refuse() {
        let e = base().with_epistemic_raw5(20);
        assert_eq!(
            eligible(UNDECLARED_CLASS, e),
            Err(Refusal::Reading(Epi5ReadError::UndeclaredClass(
                UNDECLARED_CLASS
            )))
        );
        assert!(eligible(CERT_CLASS, e).is_ok());
    }

    /// F3: a certified `Causes` is `Causes` under every topology.
    #[test]
    fn a_certified_causes_reads_as_causes_under_every_topology() {
        for t in Topology2::ALL {
            let e = certification_reading::stamp(base(), Contract::Causes, t);
            assert_eq!(raw5(e), 20 + t.ordinal());
            assert_eq!(
                read(&decl(), CERT_CLASS, e, EdgeProvenance::V2Stamped),
                Ok(Contract::Causes)
            );
            assert!(canonical(e).unwrap().asserts(CAUSES));
            assert_ne!(at(raw5(e)) & 1 << COUNTERFACTUAL_PROBE, 0);
        }
    }

    /// The #1378 fixture (P7a band 5 over topology 0) now means the same to
    /// both readers: `Direct × Causes`.
    #[test]
    fn the_1378_fixture_now_agrees() {
        let old = base().with_epistemic_raw5(5 << 2);
        assert_eq!(raw5(old), 20);
        let s = canonical(old).unwrap();
        assert_eq!(s.topology(), Topology2::Direct);
        assert_eq!(split(old), Some((0, Contract::Causes)));
        assert_eq!(
            Contract::from_certification(s.certification()),
            Contract::Causes
        );
    }

    /// F6: the factor-compiled per-code table equals evaluating the rules on
    /// each code's facts directly, for both law generations.
    #[test]
    fn the_derived_table_equals_the_rules() {
        for law in [LawGen::V1, LawGen::V2] {
            let rules = law.rules();
            for code in 0u8..32 {
                let want = EPI_LAW[code as usize].map(|f| {
                    (0..64).fold(0u64, |m, r| {
                        let rl = rules[r];
                        if rl.active && f & rl.requires == rl.requires && f & rl.forbids == 0 {
                            m | 1 << r
                        } else {
                            m
                        }
                    })
                });
                match want {
                    Some(w) => assert_eq!(law.tables().state[code as usize], w, "code {code}"),
                    None => assert!(code >= 24),
                }
            }
        }
    }

    /// F7 + the operator's frontier cases: one coordinate moves only the
    /// recipes that read it.
    #[test]
    fn one_factor_moves_only_its_recipes() {
        // IndirectUnknown × Related (10) vs IndirectKnown × Related (9):
        // exactly hydrate ↔ mechanism fold.
        assert_eq!(
            at(10) ^ at(9),
            bits(&[HYDRATE_INTERMEDIATE, MECHANISM_FOLD])
        );
        // Related → Supports under IndirectKnown (9 → 13): exactly causal
        // identification.
        assert_eq!(at(9) ^ at(13), bits(&[CAUSAL_IDENTIFICATION]));
        // IndirectUnknown × Causes (22): certified, mechanism open — may
        // hydrate, may not be "identified" again.
        assert_ne!(at(22) & 1 << HYDRATE_INTERMEDIATE, 0);
        assert_eq!(at(22) & 1 << CAUSAL_IDENTIFICATION, 0);
        assert_ne!(at(22) & 1 << COUNTERFACTUAL_PROBE, 0);
        // Unknown × Causes (23): causality known, topology unresolved.
        assert_eq!(at(23) & bits(&[HYDRATE_INTERMEDIATE, MECHANISM_FOLD]), 0);
        assert_ne!(at(23) & 1 << COUNTERFACTUAL_PROBE, 0);

        // Generally: recipes whose rules never mention a topology fact are
        // unchanged by any topology move, and likewise for certification.
        let rules = LawGen::V1.rules();
        let reads = |mask: u32| -> u64 {
            (0..64).fold(0, |m, r| {
                let rl = rules[r];
                if rl.active && (rl.requires | rl.forbids) & mask != 0 {
                    m | 1 << r
                } else {
                    m
                }
            })
        };
        let (topo_recipes, cert_recipes) = (reads(TOPOLOGY_MASK), reads(CERTIFICATION_MASK));
        for c in 0..6u8 {
            for (t1, t2) in [(0u8, 1u8), (1, 2), (2, 3), (0, 3)] {
                assert_eq!((at(t1 | c << 2) ^ at(t2 | c << 2)) & !topo_recipes, 0);
            }
        }
        for t in 0..4u8 {
            for c in 0..5u8 {
                assert_eq!((at(t | c << 2) ^ at(t | (c + 1) << 2)) & !cert_recipes, 0);
            }
        }
    }

    /// F2: what must disappear is consuming legacy-reading bits as canonical.
    /// The historical `ReasoningBand::Causal` writer puts 3 into 61..63, which
    /// decodes as `Supports`; a class that declared that historical reading
    /// has no canonical declaration, so it refuses instead of being read as a
    /// support claim nobody made.
    #[test]
    fn a_legacy_reading_class_is_not_consumed_as_canonical() {
        #[allow(deprecated)]
        let e = base().with_reasoning_band(ReasoningBand::Causal);
        assert_eq!(raw5(e) >> 2, 3, "the bits decode as Supports");
        const HISTORICAL_CLASS: ClassId = 0x0901; // P7a's LEGACY_CLASS
        assert_eq!(
            eligible(HISTORICAL_CLASS, e),
            Err(Refusal::Reading(Epi5ReadError::UndeclaredClass(
                HISTORICAL_CLASS
            )))
        );
        assert_eq!(
            read(&decl(), HISTORICAL_CLASS, e, EdgeProvenance::V2Stamped),
            Err(P7aRefusal::NotCertificationClass)
        );
    }

    /// F5: same edge + class/rail + generation + provenance → same state
    /// after a restart from the LE image.
    #[test]
    fn the_state_survives_restart_from_le_bytes() {
        for code in 0u8..32 {
            let e = base().with_epistemic_raw5(code);
            let restarted = CausalEdge64::from_le_bytes(e.to_le_bytes());
            assert_eq!(canonical(e), canonical(restarted));
            assert_eq!(canonical(e).is_some(), code < 24);
        }
    }

    /// F8: reading and projecting never write the edge; a factor transition
    /// exists only as a value until written jointly.
    #[test]
    fn projections_never_change_the_edge() {
        let e = base().with_epistemic_raw5(9);
        let before = e.0;
        let s = canonical(e).unwrap();
        let _ = (
            split(e),
            at(raw5(e)),
            read(&decl(), CERT_CLASS, e, EdgeProvenance::V2Stamped),
        );
        let moved = s.with_topology(Topology2::IndirectUnknown);
        assert_eq!(e.0, before, "a projection wrote the edge");
        assert_eq!(moved.raw(), 10);
        assert_eq!(
            e.with_epistemic_raw5(moved.raw()).0 ^ before,
            (9u64 ^ 10) << 59
        );
    }
}
