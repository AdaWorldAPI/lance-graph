//! D-GSO-7a (P7a): the reasoning band as a relational certification.
//!
//! Supersedes the semantics of D-GSO-7 (`reasoning_band_probe.rs`, #1360)
//! without editing it. #1360 stays as shipped: a probe of how a band is earned
//! and lowered. What this probe replaces is the meaning of the rungs.
//!
//! # The claim under test
//!
//! Bits 61..63 hold the strongest relational statement the sealed model is
//! allowed to assert. They do not hold confidence, effect size, the reasoning
//! operation, or how much work the path took. Under one declared reading the
//! raw codes are:
//!
//! | code | contract | obligation (all integer folds over unit masks) |
//! |---|---|---|
//! | 0 | `Open` | none |
//! | 1 | `Associated` | A raises Y in some declared population, ≥ 2 distinct sources |
//! | 2 | `Related` | the association holds in a declared population **and** after the robustness mask |
//! | 3 | `Contributes` | in **every** declared stratum A does not lower Y, before and after the robustness mask, and raises it in at least one |
//! | 4 | `CausalCandidate` | `Contributes`, and exposure precedes outcome on every co-occurring unit |
//! | 5 | `Causes` | ≥ 2 distinct `InterventionBacked` sources **and** the executed randomized arms show the treated rate higher |
//! | 6, 7 | reserved | refuse |
//!
//! The names are working names. The obligations are the content, and the
//! thresholds (2 sources, the rate comparison) are policy pins.
//!
//! # What was measured, not assumed
//!
//! - **Ordinal comparison is legal on codes 0..=5 only.** `entails` is an
//!   explicit table; a test proves it equals `code >= code` on the six
//!   contracts. Codes 6 and 7 are not contracts, so they satisfy nothing.
//! - **The chain is monotone only with scoped association.** If `Associated`
//!   means the marginal association, an exhaustive family of small models finds
//!   relations that satisfy `Contributes` but not `Associated` (Simpson's
//!   reversal). Measured counts are printed by `main`. With `Associated`
//!   existential over the declared populations (the universe plus the declared
//!   strata, fixed before the data are read), the family has no violation.
//! - **`Causes` entails the lower contracts in the population that certified
//!   it**: the randomized arms. It does not entail them in the observational
//!   population, and a confounded observational population does not demote an
//!   intervention-certified relation (#1360's confounding cap would).
//! - **Sibling specificity is not a rung.** A and its sibling can both cause Y
//!   equally; comparing A against siblings then fails while `Causes` holds. So
//!   "discriminative against siblings" cannot sit below `Causes` in a chain; it
//!   is a statistic beside the band.
//! - **Removal is not the causal path.** Under overdetermination (Y = A or B)
//!   removing A from a unit with B leaves Y unchanged, while the population
//!   trial still certifies `Causes`.
//!
//! # Carrier (D-EPI-MIG-0)
//!
//! P7a's codes are the certification coordinate of the canonical Cartesian
//! `EpistemicState5` (`raw5 = topology | certification << 2`, bits 59..63).
//! A certification is stamped by identity packing `(MODEL_GROUNDING, code)`
//! with `with_epistemic_raw5` and read back through
//! `Epi5Declarations::project_state5`. All six contracts are meaningful under
//! every topology; 6 and 7 are reserved and refuse. Provenance is `causal_audit::SupportLedger` /
//! `SupportReceipt` / `SupportBasis` / `EvidenceSourceId`. Seals are
//! `scheduler::DatasetVersion`.
//!
//! # Not decided here
//!
//! - Whether the reading becomes a `band_reading` lens. Here the reading is
//!   probe-local and keyed by class: a class declared under the historical
//!   reading refuses, even for the same bits.
//! - `CausalCandidate`'s obligations beyond ordering. They are deliberately
//!   thin until real data exist.
//! - The robustness mask's source. It is an input (in practice an anomaly
//!   detector such as ndarray's CLAM/CHAODA). It can only refute: a mask can
//!   never create an association that the full population lacks.
//! - No stored rows are written.
//!
//! Run: `cargo run -p cognitive-shader-driver --example relational_certification_probe`
//! Tests: `cargo test -p cognitive-shader-driver --example relational_certification_probe`

use causal_edge::edge::CausalEdge64;
use causal_edge::layout::ReasoningBand;
use lance_graph_contract::band_reading::EdgeProvenance;
use lance_graph_contract::causal_audit::SupportBasis;
#[cfg(test)]
use lance_graph_contract::{causal_audit::EvidenceSourceId, scheduler::DatasetVersion};

#[path = "shared/certification_reading.rs"]
mod certification_reading;

use certification_reading::*;

#[path = "shared/certification_model.rs"]
mod certification_model;

use certification_model::*;

fn main() {
    let decl = declarations();
    let mut with_trial = robust_association();
    add_positive_trial(&mut with_trial);
    let mut simpson_trial = simpson_population();
    add_positive_trial(&mut simpson_trial);
    let fixtures: [(&str, Model); 4] = [
        (
            "robust association, observational",
            robust_association().build(),
        ),
        (
            "Simpson population, observational",
            simpson_population().build(),
        ),
        ("robust association + randomized trial", with_trial.build()),
        (
            "Simpson population + randomized trial",
            simpson_trial.build(),
        ),
    ];
    println!(
        "relational certification -> canonical EpistemicState5 (bits 59..63), grounding {MODEL_GROUNDING:?}"
    );
    let codebook: Vec<String> = Contract::ALL
        .iter()
        .map(|c| format!("{}={c:?}", c.code()))
        .collect();
    println!("  codebook: {} (6, 7 reserved)", codebook.join(", "));
    for (name, m) in &fixtures {
        let c = m.certify();
        let e = stamp(CausalEdge64::ZERO, c, MODEL_GROUNDING);
        let back = (
            e.epistemic_raw5(),
            read(&decl, CERT_CLASS, e, EdgeProvenance::V2Stamped),
        );
        println!(
            "  {name:<40} -> {c:?} (P7a code {}, marginal association {:?}, canonical {back:?})",
            c.code(),
            m.associated_marginal()
        );
    }
    let t = fixtures[3].1.trial_view();
    println!(
        "  Simpson + trial, read in the trial population: associated {:?}, contributes {:?}",
        t.associated(),
        t.contributes()
    );
    // The historical `Meta` band ordinal in the high half, as a joint code.
    let meta = CausalEdge64::ZERO.with_epistemic_raw5(ReasoningBand::Meta.to_bits_3() << 2);
    println!(
        "  historical Meta bits asked to satisfy Causes: {:?}",
        satisfies(&decl, CERT_CLASS, meta, Contract::Causes)
    );
    let census = monotonicity_census();
    println!(
        "monotonicity over {} models: scoped violations {}, marginal violations {} (Contributes without marginal association); certified per code {:?}",
        census.models, census.scoped_violations, census.marginal_violations, census.reached
    );
}

/// Exhaustive family: two strata, two cells each, n ∈ {1, 2}, every hit count,
/// two robustness masks (keep all, drop the first exposed outcome unit) and two
/// orderings (all ordered, that unit's outcome not preceded by exposure).
struct Census {
    models: usize,
    scoped_violations: usize,
    marginal_violations: usize,
    /// How many models certify each code.
    reached: [usize; 6],
}

fn monotonicity_census() -> Census {
    let options: Vec<(u32, u32)> = (1..=2u32)
        .flat_map(|n| (0..=n).map(move |h| (n, h)))
        .collect();
    let mut c = Census {
        models: 0,
        scoped_violations: 0,
        marginal_violations: 0,
        reached: [0; 6],
    };
    for &a0 in &options {
        for &b0 in &options {
            for &a1 in &options {
                for &b1 in &options {
                    for (drop, unordered) in
                        [(false, false), (true, false), (false, true), (true, true)]
                    {
                        let mut b = Builder::new();
                        let (_, hit) = b.cell(0, true, a0.0, a0.1);
                        b.cell(0, false, b0.0, b0.1);
                        b.cell(1, true, a1.0, a1.1);
                        b.cell(1, false, b1.0, b1.1);
                        b.sources(SupportBasis::DirectlyObserved, &[1, 2]);
                        if drop && hit != 0 {
                            b.m.clean &= !(hit.isolate_lowest_one());
                        }
                        if unordered && hit != 0 {
                            b.m.ordered &= !(hit.isolate_lowest_one());
                        }
                        let m = b.build();
                        c.models += 1;
                        c.reached[m.certify().code() as usize] += 1;
                        let holds = |s: Sat| s == Ok(true);
                        let (cc, co, re, asc) = (
                            holds(m.causal_candidate()),
                            holds(m.contributes()),
                            holds(m.related()),
                            holds(m.associated()),
                        );
                        if (cc && !co) || (co && !re) || (re && !asc) {
                            c.scoped_violations += 1;
                        }
                        if co && !holds(m.associated_marginal()) {
                            c.marginal_violations += 1;
                        }
                    }
                }
            }
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;
    use causal_edge::layout::EPISTEMIC_MASK;
    use lance_graph_contract::epistemic_state5::Epi5ReadError;

    /// P7a.1 + P7a.8: the reading is declared per class; reserved codes and
    /// historically-declared classes refuse; the bits alone decide nothing.
    #[test]
    fn the_reading_is_declared_and_reserved_codes_refuse() {
        let decl = declarations();
        for c in Contract::ALL {
            let edge = stamp(CausalEdge64::ZERO, c, MODEL_GROUNDING);
            assert_eq!(edge.epistemic_raw5(), c.code() << 2, "identity packing");
            assert_eq!(
                read(&decl, CERT_CLASS, edge, EdgeProvenance::V2Stamped),
                Ok(c)
            );
            // Same bits, a class without a canonical declaration: refused.
            assert_eq!(
                read(&decl, LEGACY_CLASS, edge, EdgeProvenance::V2Stamped),
                Err(Refusal::NotCertificationClass)
            );
        }
        // P7a's reserved bands 6, 7 in the high half: not canonical codes.
        for raw in [6u8, 7] {
            let edge = CausalEdge64::ZERO.with_epistemic_raw5(raw << 2);
            assert_eq!(
                read(&decl, CERT_CLASS, edge, EdgeProvenance::V2Stamped),
                Err(Refusal::Reading(Epi5ReadError::UndeclaredCode(raw << 2)))
            );
        }
        let edge = stamp(CausalEdge64::ZERO, Contract::Causes, MODEL_GROUNDING);
        assert_eq!(
            read(&decl, CERT_CLASS, edge, EdgeProvenance::Unknown),
            Err(Refusal::Reading(Epi5ReadError::UnknownProvenance(
                EdgeProvenance::Unknown
            )))
        );
    }

    /// The `>=` reading is legal on the six contracts and only there. The old
    /// `Meta` (6) and `Transcendent` (7) codes are numerically above `Causes`
    /// and must not satisfy it.
    #[test]
    fn ordinal_order_is_legal_only_on_the_declared_prefix() {
        for a in Contract::ALL {
            for b in Contract::ALL {
                assert_eq!(a.entails(b), a.code() >= b.code(), "{a:?} vs {b:?}");
            }
        }
        let decl = declarations();
        for historical in [ReasoningBand::Meta, ReasoningBand::Transcendent] {
            let edge = CausalEdge64::ZERO.with_epistemic_raw5(historical.to_bits_3() << 2);
            assert!(historical.to_bits_3() > Contract::Causes.code());
            assert!(satisfies(&decl, CERT_CLASS, edge, Contract::Causes).is_err());
            assert!(satisfies(&decl, CERT_CLASS, edge, Contract::Open).is_err());
        }
    }

    /// P7a.2: one source repeated a thousand times is one source.
    #[test]
    fn repetition_alone_cannot_climb() {
        let mut once = Builder::new();
        once.cell(0, true, 5, 4);
        once.cell(0, false, 5, 1);
        let mut many = Builder::new();
        many.cell(0, true, 5, 4);
        many.cell(0, false, 5, 1);
        once.sources(SupportBasis::DirectlyObserved, &[7]);
        for _ in 0..1000 {
            many.sources(SupportBasis::DirectlyObserved, &[7]);
        }
        let (once, many) = (once.build(), many.build());
        assert_eq!(many.associated(), Err(NotGrounded::TooFewSources));
        assert_eq!(once.certify(), Contract::Open);
        assert_eq!(many.certify(), once.certify());
        // One genuinely independent source changes it.
        let mut two = Builder::new();
        two.cell(0, true, 5, 4);
        two.cell(0, false, 5, 1);
        two.sources(SupportBasis::DirectlyObserved, &[7, 8]);
        assert!(two.build().certify().entails(Contract::Associated));
        // The same holds for identification: 1000 receipts, one lab.
        let mut lab = robust_association();
        lab.arm(true, 6, 5);
        lab.arm(false, 6, 1);
        for _ in 0..1000 {
            lab.sources(SupportBasis::InterventionBacked, &[10]);
        }
        assert_eq!(lab.build().causes(), Err(NotGrounded::TooFewSources));
        assert_ne!(lab.build().certify(), Contract::Causes);
    }

    /// P7a.3: `Related` comes from population masks. An association carried by
    /// anomalous units stops at `Associated`; one that appears only after the
    /// mask is not created by it.
    #[test]
    fn related_comes_from_population_masking_and_masks_only_refute() {
        assert!(robust_association().build().related() == Ok(true));

        // Outlier-driven: the exposed hits are exactly the anomalous units.
        let mut b = Builder::new();
        let (_, hits) = b.cell(0, true, 6, 3);
        b.cell(0, false, 6, 1);
        b.sources(SupportBasis::DirectlyObserved, &[1, 2]);
        b.m.clean &= !hits;
        let outlier = b.build();
        assert_eq!(outlier.associated(), Ok(true));
        assert_eq!(outlier.related(), Ok(false));
        assert_eq!(outlier.certify(), Contract::Associated);

        // Hidden-by-anomaly: no association until the unexposed hits are
        // masked. The mask does not raise it.
        let mut h = Builder::new();
        h.cell(0, true, 4, 2);
        let (_, hidden) = h.cell(0, false, 4, 3);
        h.sources(SupportBasis::DirectlyObserved, &[1, 2]);
        h.m.clean &= !hidden;
        let hidden = h.build();
        assert_eq!(hidden.assoc_in(hidden.universe & hidden.clean), Ok(true));
        assert_eq!(hidden.related(), Ok(false));
        assert_eq!(hidden.certify(), Contract::Open);
    }

    /// P7a.4 + P7a.5: a stable conditional effect certifies `Contributes` /
    /// `CausalCandidate`; no observational effect, however large, reaches
    /// `Causes`.
    #[test]
    fn observational_effect_stops_below_causes() {
        let mut big = Builder::new();
        big.cell(0, true, 10, 10);
        big.cell(0, false, 10, 0);
        big.cell(1, true, 10, 10);
        big.cell(1, false, 10, 0);
        big.sources(SupportBasis::DirectlyObserved, &[1, 2, 3, 4, 5]);
        big.sources(SupportBasis::CrossEnvironmentInvariant, &[6, 7]);
        let mut m = big.build();
        assert_eq!(m.certify(), Contract::CausalCandidate);
        assert_eq!(m.causes(), Err(NotGrounded::NoIdentificationDesign));
        // Ordering violated on one co-occurring unit: Contributes, not candidate.
        let co = m.exposed & m.outcome;
        m.ordered &= !(co.isolate_lowest_one());
        assert_eq!(m.certify(), Contract::Contributes);
    }

    /// P7a.6: `Causes` needs both attested identification and executed arms.
    /// Withdrawing a source at the next seal lowers the band; nothing sticks.
    #[test]
    fn causes_requires_identification_and_executed_arms() {
        let mut no_receipts = robust_association();
        no_receipts.arm(true, 6, 5);
        no_receipts.arm(false, 6, 1);
        assert_eq!(
            no_receipts.build().causes(),
            Err(NotGrounded::NoIdentificationDesign)
        );

        let mut no_arms = robust_association();
        no_arms.sources(SupportBasis::InterventionBacked, &[10, 11]);
        assert_eq!(no_arms.build().causes(), Err(NotGrounded::EmptyArm));

        let mut null_trial = robust_association();
        null_trial.arm(true, 6, 2);
        null_trial.arm(false, 6, 2);
        null_trial.sources(SupportBasis::InterventionBacked, &[10, 11]);
        assert_eq!(null_trial.build().causes(), Ok(false));
        assert_eq!(null_trial.build().certify(), Contract::CausalCandidate);

        let mut full = robust_association();
        add_positive_trial(&mut full);
        let sealed = full.build();
        assert_eq!(sealed.certify(), Contract::Causes);

        let mut next = sealed.clone();
        next.seal = DatasetVersion(2);
        assert_eq!(next.ledger.withdraw_source(EvidenceSourceId(11)), 1);
        assert_ne!(next.certify(), Contract::Causes);
    }

    /// P7a.7: running a counterfactual changes nothing it was not asked to
    /// write. The mantissa survives stamping, the band does not depend on it,
    /// and evaluating an alternate world leaves the sealed model's
    /// certification as it was.
    #[test]
    fn a_counterfactual_operation_does_not_move_the_band() {
        let decl = declarations();
        let mut b = robust_association();
        add_positive_trial(&mut b);
        let m = b.build();
        let certified = m.certify();
        for mantissa in -8i8..=7 {
            let edge = CausalEdge64::ZERO.with_inference_mantissa(mantissa);
            let stamped = stamp(edge, certified, MODEL_GROUNDING);
            assert_eq!(stamped.inference_mantissa(), mantissa);
            assert_eq!(
                (stamped.0 ^ edge.0) & !EPISTEMIC_MASK,
                0,
                "only bits 59..63 move"
            );
            assert_eq!(
                read(&decl, CERT_CLASS, stamped, EdgeProvenance::V2Stamped),
                Ok(certified)
            );
        }
        // Alternate world do(not A): exposure removed everywhere.
        let mut world = m.clone();
        world.exposed = 0;
        world.assigned = 0;
        assert_eq!(world.certify(), Contract::Open);
        assert_eq!(m.certify(), certified);
    }

    /// The five-state chain is monotone with association scoped to the declared
    /// populations, and is not with the marginal reading.
    #[test]
    fn the_chain_is_monotone_only_under_scoped_association() {
        let c = monotonicity_census();
        assert_eq!(c.models, 2500);
        assert_eq!(c.scoped_violations, 0);
        assert!(
            c.marginal_violations > 0,
            "the family must contain a Simpson case"
        );
        // Not vacuous: the family reaches every observational contract, and
        // no observational model reaches `Causes`.
        for code in 0..5 {
            assert!(
                c.reached[code] > 0,
                "code {code} never certified: {:?}",
                c.reached
            );
        }
        assert_eq!(c.reached[5], 0);
    }

    /// Simpson: marginal association is negative, every stratum positive. The
    /// observational chain certifies `CausalCandidate` through the declared
    /// strata; a trial certifies `Causes`, and the confounded observational
    /// population does not demote it. `Causes` entails the lower contracts in
    /// the trial population.
    #[test]
    fn confounding_does_not_demote_an_intervention_certified_relation() {
        let obs = simpson_population().build();
        assert_eq!(obs.associated_marginal(), Ok(false));
        assert_eq!(obs.contributes(), Ok(true));
        assert_eq!(obs.certify(), Contract::CausalCandidate);

        let mut b = simpson_population();
        add_positive_trial(&mut b);
        let m = b.build();
        assert_eq!(m.certify(), Contract::Causes);
        let t = m.trial_view();
        for (rung, sat) in [
            (Contract::Associated, t.associated()),
            (Contract::Related, t.related()),
            (Contract::Contributes, t.contributes()),
            (Contract::CausalCandidate, t.causal_candidate()),
        ] {
            assert_eq!(
                sat,
                Ok(true),
                "{rung:?} must hold in the certifying population"
            );
        }
    }

    /// Overdetermination: Y = A or B. Removing A from a unit with B leaves Y,
    /// yet the randomized population certifies `Causes`. Necessity is not the
    /// causal test.
    #[test]
    fn overdetermination_removal_says_dispensable_population_says_causes() {
        let mut b = Builder::new();
        let (a_and_b, _) = b.arm(true, 3, 3); // A and B -> Y
        b.arm(true, 3, 3); // A only -> Y
        let (b_only, _) = b.arm(false, 3, 3); // B only -> Y
        b.arm(false, 3, 0); // neither -> no Y
        b.sources(SupportBasis::InterventionBacked, &[10, 11]);
        let with_b = a_and_b | b_only;
        let m = b.build();
        // Removal diagnostic on one unit with A and B: Y = A or B still holds.
        let unit = a_and_b.isolate_lowest_one();
        assert_ne!(
            unit & with_b,
            0,
            "removing A leaves Y: A is dispensable here"
        );
        assert_eq!(m.causes(), Ok(true));
        assert_eq!(m.certify(), Contract::Causes);
    }

    /// Specificity against siblings is orthogonal to `Causes`: A and its
    /// sibling are equally effective, so A is not discriminative against the
    /// sibling population while `Causes` holds. It cannot be a rung below.
    #[test]
    fn sibling_specificity_is_not_a_rung() {
        let mut b = robust_association();
        let (a_arm, _) = b.arm(true, 6, 5);
        let (sib_arm, _) = b.arm(false, 6, 5); // the sibling treatment
        b.arm(false, 6, 1); // untreated control
        b.sources(SupportBasis::InterventionBacked, &[10, 11]);
        let mut m = b.build();
        m.trial &= !sib_arm; // the identification design compares A with untreated
        assert_eq!(m.certify(), Contract::Causes);
        assert_eq!(compare(m.outcome, a_arm, sib_arm, true), Ok(false));
    }

    /// Receipts recorded after the model's seal do not count: certification
    /// replays from the evidence the seal held, whatever the ledger holds now.
    #[test]
    fn receipts_after_the_seal_do_not_count() {
        let mut b = Builder::new();
        b.cell(0, true, 5, 4);
        b.cell(0, false, 5, 1);
        b.sources(SupportBasis::DirectlyObserved, &[1]);
        b.sources_at(SupportBasis::DirectlyObserved, &[2], DatasetVersion(2));
        b.arm(true, 6, 5);
        b.arm(false, 6, 1);
        b.sources_at(
            SupportBasis::InterventionBacked,
            &[10, 11],
            DatasetVersion(2),
        );
        let sealed = b.build();
        assert_eq!(sealed.seal, DatasetVersion(1));
        assert_eq!(sealed.associated(), Err(NotGrounded::TooFewSources));
        assert_eq!(sealed.causes(), Err(NotGrounded::NoIdentificationDesign));
        assert_eq!(sealed.certify(), Contract::Open);
        // The same ledger read at the later seal does count them.
        let later = Model {
            seal: DatasetVersion(2),
            ..sealed.clone()
        };
        assert_eq!(later.certify(), Contract::Causes);
        assert_eq!(
            sealed.certify(),
            Contract::Open,
            "replay of the earlier seal is unchanged"
        );
    }

    /// The same seal certifies to the same bits on replay.
    #[test]
    fn replay_gives_the_same_bits() {
        let decl = declarations();
        let seals = [
            robust_association().build(),
            simpson_population().build(),
            {
                let mut b = simpson_population();
                add_positive_trial(&mut b);
                b.build()
            },
        ];
        let run = || -> Vec<u64> {
            seals
                .iter()
                .map(|m| stamp(CausalEdge64::ZERO, m.certify(), MODEL_GROUNDING).0)
                .collect()
        };
        assert_eq!(run(), run());
        for bits in run() {
            assert!(read(
                &decl,
                CERT_CLASS,
                CausalEdge64(bits),
                EdgeProvenance::V2Stamped
            )
            .is_ok());
        }
    }
}
