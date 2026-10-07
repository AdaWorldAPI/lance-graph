//! D-EPI-CONFLICT-0 → D-EPI-MIG-0: the disagreement this probe pinned is
//! resolved by having ONE reading of `CausalEdge64` bits 59..63, and the probe
//! is now the conformance check of that reading.
//!
//! # What it pinned (#1378, kept as history)
//!
//! P7a (#1369) wrote its certification into bits 61..63 alone and read it
//! through `band_reading::project_band`; the affordance law (#1370) read bits
//! 59..63 as one `EpistemicState5` code without any class declaration. A
//! P7a-certified `Causes` with topology 0 was raw code 20, which the law read
//! as `Direct × Related`; codes 21..23 refused; 3 of 8 shared cells agreed.
//!
//! # What holds now (operator decision 2026-10-07)
//!
//! Bits 59..63 are owned by `lance_graph_contract::epistemic_state5`: one
//! codebook (`CODEBOOK_V1`), declared per `(classid, rail, generation)`,
//! projected with refusals. P7a stamps by translating `(grounding, contract)`
//! into a canonical code and writing all five bits; it reads by projecting the
//! state and taking the legacy certification projection. The affordance law
//! measures only a projected state. So both are views of the same state, and:
//!
//! | test | pin |
//! |---|---|
//! | `undeclared_classes_refuse_under_both_readings` | the law no longer measures a class it has no declaration for |
//! | `a_certified_causes_reads_as_causes` | F3: `Causes` → code 1 (Direct) / 17 (IndirectKnown), carries `CAUSES`, makes `COUNTERFACTUAL_PROBE` eligible |
//! | `every_legacy_cell_translates_in_agreement_or_refuses` | F10: 11 of 24 cells map, 0 disagree, 13 refuse (was 3 / 5 / 16) |
//! | `both_readings_agree_on_every_raw_code` | F10: over all 32 codes the P7a reading and the law agree or both refuse |
//! | `the_historical_split_pattern_is_read_by_its_canonical_meaning` | the old `(band 5, topology 0)` bit pattern is code 20 to both readers alike |
//! | `measurement_is_unchanged_by_the_refactor` | F6: `measure` = the pre-migration body on every code × Pearl × law × provenance |
//! | `one_factor_moves_only_its_recipes` | F7: 25↔5 differ only in hydrate/mechanism; 5↔30 only in causal identification |
//! | `the_state_survives_restart_from_le_bytes` | F5 |
//! | `projections_never_change_the_edge` | F8 |
//!
//! Run: `cargo run -p cognitive-shader-driver --example epistemic_reading_conflict_probe`
//! Tests: `cargo test -p cognitive-shader-driver --example epistemic_reading_conflict_probe`

#[path = "shared/affordance_law.rs"]
mod affordance_law;
#[path = "shared/certification_reading.rs"]
mod certification_reading;

use affordance_law::{raw5, ASSOCIATED, CAUSES, RELATED, SUPPORTS};
use causal_edge::edge::CausalEdge64;
use causal_edge::pearl::CausalMask;
use causal_edge::PlasticityState;
use certification_reading::{read, stamp, Contract, Refusal as P7aRefusal, CERT_CLASS, RAIL};
use lance_graph_contract::band_reading::EdgeProvenance;
use lance_graph_contract::epistemic_state5::legacy::LegacyTopology;
use lance_graph_contract::epistemic_state5::{Epi5Declarations, Epi5Gen, Epi5Reading};

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

/// What P7a reads off an edge of its class.
fn p7a_reads(edge: CausalEdge64) -> Result<Contract, P7aRefusal> {
    read(&decl(), CERT_CLASS, edge, EdgeProvenance::V2Stamped)
}

/// The facts P7a's contract entails, in the affordance vocabulary.
/// `Contributes` is `SUPPORTS` (operator decision: "Contributes / Supports").
fn p7a_entails(contract: Contract) -> u32 {
    let mut f = 0;
    if contract.entails(Contract::Associated) {
        f |= ASSOCIATED;
    }
    if contract.entails(Contract::Related) {
        f |= RELATED;
    }
    if contract.entails(Contract::Contributes) {
        f |= SUPPORTS;
    }
    if contract.entails(Contract::Causes) {
        f |= CAUSES;
    }
    f
}

const SHARED: u32 = ASSOCIATED | RELATED | SUPPORTS | CAUSES;

/// The affordance facts of an edge of P7a's class, or `None` if refused.
fn affordance_facts(edge: CausalEdge64) -> Option<u32> {
    decl()
        .project_state5(
            CERT_CLASS,
            RAIL,
            Epi5Gen::V1,
            raw5(edge),
            EdgeProvenance::V2Stamped,
        )
        .ok()
        .map(|s| s.facts())
}

/// (agree, disagree, refused) over every legacy grounding × P7a contract.
fn census() -> (usize, usize, usize) {
    let (mut agree, mut disagree, mut refused) = (0, 0, 0);
    for t in LegacyTopology::ALL {
        for c in Contract::ALL {
            match stamp(base(), c, t) {
                Err(_) => refused += 1,
                Ok(e) => match affordance_facts(e) {
                    Some(f) if f & SHARED == p7a_entails(c) && p7a_reads(e) == Ok(c) => agree += 1,
                    _ => disagree += 1,
                },
            }
        }
    }
    (agree, disagree, refused)
}

fn main() {
    println!("D-EPI-MIG-0: legacy grounding x P7a contract -> canonical EpistemicState5");
    for t in LegacyTopology::ALL {
        for c in Contract::ALL {
            match stamp(base(), c, t) {
                Ok(e) => println!(
                    "  {t:<16?} {:<16} -> code {:>2}  facts {:#05x}",
                    format!("{c:?}"),
                    raw5(e),
                    affordance_facts(e).unwrap_or(0)
                ),
                Err(r) => println!("  {t:<16?} {:<16} -> refuses ({r:?})", format!("{c:?}")),
            }
        }
    }
    let (a, d, r) = census();
    println!("agree {a}, disagree {d}, refused {r} of 24 cells (was 3 / 5 / 16 in #1378)");
}

#[cfg(test)]
mod tests {
    use super::*;
    use affordance_law::EPI_LAW;
    use affordance_law::{
        measure_declared, LawGen, Refusal, CAUSAL_IDENTIFICATION, COUNTERFACTUAL_PROBE,
        HYDRATE_INTERMEDIATE, MECHANISM_FOLD, ROBUSTNESS_TEST, STRATIFY,
    };
    use lance_graph_contract::class_view::ClassId;
    use lance_graph_contract::epistemic_state5::{legacy, Epi5ReadError};

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

    fn bits(recipes: &[u8]) -> u64 {
        recipes.iter().fold(0, |m, r| m | 1 << r)
    }

    /// FAILS IF: the law measures a class with no canonical declaration.
    #[test]
    fn undeclared_classes_refuse_under_both_readings() {
        let e = stamp(base(), Contract::Causes, LegacyTopology::Direct).unwrap();
        assert_eq!(
            eligible(UNDECLARED_CLASS, e),
            Err(Refusal::Reading(Epi5ReadError::UndeclaredClass(
                UNDECLARED_CLASS
            )))
        );
        // Silence twin: the declared class measures the same edge.
        assert!(eligible(CERT_CLASS, e).is_ok());
    }

    /// F3: a certified `Causes` is canonical `Causes`, never `Related`.
    #[test]
    fn a_certified_causes_reads_as_causes() {
        for (t, code) in [
            (LegacyTopology::Direct, 1),
            (LegacyTopology::IndirectKnown, 17),
        ] {
            let e = stamp(base(), Contract::Causes, t).unwrap();
            assert_eq!(raw5(e), code);
            assert_eq!(p7a_reads(e), Ok(Contract::Causes));
            let facts = affordance_facts(e).unwrap();
            assert_ne!(facts & CAUSES, 0);
            assert_ne!(
                eligible(CERT_CLASS, e).unwrap() & 1 << COUNTERFACTUAL_PROBE,
                0
            );
        }
        // An unknown or unknown-intermediate grounding cannot carry Causes.
        for t in [LegacyTopology::IndirectUnknown, LegacyTopology::Unknown] {
            assert!(matches!(
                stamp(base(), Contract::Causes, t),
                Err(P7aRefusal::NoCanonicalState(_))
            ));
        }
    }

    /// F10 (legacy direction): every cell either translates into a state both
    /// readers agree on, or refuses. No disagreement remains.
    #[test]
    fn every_legacy_cell_translates_in_agreement_or_refuses() {
        assert_eq!(census(), (11, 0, 13));
    }

    /// F10 (bit direction): over every raw code, the P7a reading and the law
    /// either agree on the shared facts or both refuse.
    #[test]
    fn both_readings_agree_on_every_raw_code() {
        let mut agreed = 0;
        for code in 0u8..32 {
            let e = base().with_epistemic_raw5(code);
            match (p7a_reads(e), affordance_facts(e)) {
                (Ok(c), Some(f)) => {
                    agreed += 1;
                    // The certification projection is the strongest claim;
                    // the facts must entail exactly what it entails.
                    assert_eq!(f & SHARED, p7a_entails(c), "code {code}");
                }
                (Err(P7aRefusal::Reading(_)), None) => {}
                other => panic!("code {code}: readers diverge: {other:?}"),
            }
        }
        assert_eq!(agreed, EPI_LAW.iter().filter(|f| f.is_some()).count());
        assert_eq!(agreed, 10);
    }

    /// The bit pattern #1378 was built on (P7a band 5 over topology 0) is
    /// code 20. It is no longer what a P7a `Causes` produces, and both readers
    /// read it as the same thing: `Direct × Related`.
    #[test]
    fn the_historical_split_pattern_is_read_by_its_canonical_meaning() {
        let old = base().with_epistemic_raw5((Contract::Causes.code() << 2) | 0);
        assert_eq!(raw5(old), 20);
        assert_eq!(p7a_reads(old), Ok(Contract::Related));
        assert_eq!(
            affordance_facts(old).unwrap() & SHARED,
            ASSOCIATED | RELATED
        );
        let now = stamp(base(), Contract::Causes, LegacyTopology::Direct).unwrap();
        assert_ne!(raw5(now), raw5(old));
    }

    /// The pre-migration `measure` body, verbatim (classless, own codebook
    /// lookup), as the regression oracle for F6.
    fn pre_migration_measure(
        law: LawGen,
        edge: CausalEdge64,
        provenance: EdgeProvenance,
    ) -> Result<u64, Refusal> {
        if !matches!(
            provenance,
            EdgeProvenance::V2Stamped | EdgeProvenance::V3Register
        ) {
            return Err(Refusal::Provenance(provenance));
        }
        let code = (edge.spare() << 2) | edge.truth_raw();
        if EPI_LAW[code as usize].is_none() {
            return Err(Refusal::Undeclared(code));
        }
        let t = law.tables();
        Ok(t.state[code as usize] & t.pearl[edge.causal_mask() as usize])
    }

    /// F6: on a declared class, the projected measurement equals the
    /// pre-migration measurement everywhere.
    #[test]
    fn measurement_is_unchanged_by_the_refactor() {
        let mut compared = 0;
        for law in [LawGen::V1, LawGen::V2] {
            for pearl in 0u8..8 {
                for code in 0u8..32 {
                    for prov in [
                        EdgeProvenance::V2Stamped,
                        EdgeProvenance::V3Register,
                        EdgeProvenance::V1Legacy,
                        EdgeProvenance::Unknown,
                    ] {
                        let e = CausalEdge64::pack_v2(
                            1,
                            2,
                            3,
                            9,
                            9,
                            CausalMask::from_bits(pearl),
                            0,
                            PlasticityState::from_bits(0),
                        )
                        .with_epistemic_raw5(code);
                        let new = measure_declared(law, &decl(), CERT_CLASS, RAIL, e, prov);
                        assert_eq!(new, pre_migration_measure(law, e, prov), "{code} {prov:?}");
                        compared += 1;
                    }
                }
            }
        }
        assert_eq!(compared, 2 * 8 * 32 * 4);
    }

    /// F7 + the operator examples: changing one factor changes only the
    /// recipes that read that factor.
    #[test]
    fn one_factor_moves_only_its_recipes() {
        let at = |code| eligible(CERT_CLASS, base().with_epistemic_raw5(code)).unwrap();
        let (unknown, known, supports) = (at(25), at(5), at(30));
        // 25 = Indirect × IntermediateUnknown × Related: search, not fold.
        assert_eq!(
            unknown
                & bits(&[
                    HYDRATE_INTERMEDIATE,
                    MECHANISM_FOLD,
                    STRATIFY,
                    ROBUSTNESS_TEST
                ]),
            bits(&[HYDRATE_INTERMEDIATE, STRATIFY, ROBUSTNESS_TEST])
        );
        assert_eq!(unknown & 1 << COUNTERFACTUAL_PROBE, 0);
        // Intermediate unknown → known: exactly hydrate ↔ mechanism fold.
        assert_eq!(
            unknown ^ known,
            bits(&[HYDRATE_INTERMEDIATE, MECHANISM_FOLD])
        );
        // Related → Supports: exactly causal identification becomes eligible.
        assert_eq!(known ^ supports, bits(&[CAUSAL_IDENTIFICATION]));
        assert_ne!(supports & 1 << MECHANISM_FOLD, 0);
    }

    /// F5: same edge + class/rail + generation + provenance → same state
    /// after a restart from the LE image.
    #[test]
    fn the_state_survives_restart_from_le_bytes() {
        for code in (0u8..32).filter(|c| EPI_LAW[*c as usize].is_some()) {
            let e = base().with_epistemic_raw5(code);
            let restarted = CausalEdge64::from_le_bytes(e.to_le_bytes());
            let a = decl().project_state5(
                CERT_CLASS,
                RAIL,
                Epi5Gen::V1,
                raw5(e),
                EdgeProvenance::V2Stamped,
            );
            let b = decl().project_state5(
                CERT_CLASS,
                RAIL,
                Epi5Gen::V1,
                raw5(restarted),
                EdgeProvenance::V2Stamped,
            );
            assert_eq!(a, b);
            assert_eq!(p7a_reads(e), p7a_reads(restarted));
        }
    }

    /// F8: reading, projecting and computing a compatibility transition never
    /// writes the edge; only an explicit joint write does.
    #[test]
    fn projections_never_change_the_edge() {
        let e = base().with_epistemic_raw5(5);
        let before = e.0;
        let s = decl()
            .project_state5(
                CERT_CLASS,
                RAIL,
                Epi5Gen::V1,
                raw5(e),
                EdgeProvenance::V2Stamped,
            )
            .unwrap();
        let _ = (
            legacy::project_topology(s),
            legacy::project_certification(s),
        );
        let _ = p7a_reads(e);
        let _ = eligible(CERT_CLASS, e);
        let moved = legacy::retopologize(s, LegacyTopology::IndirectUnknown).unwrap();
        assert_eq!(e.0, before, "a projection wrote the edge");
        // The transition exists only as a value until it is written jointly.
        assert_eq!(moved.raw(), 25);
        assert_eq!(
            e.with_epistemic_raw5(moved.raw()).0 ^ before,
            (5u64 ^ 25) << 59
        );
    }
}
