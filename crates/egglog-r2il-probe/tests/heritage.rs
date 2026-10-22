//! Heritage study cannot confuse attribution with a checked proof.
use std::collections::BTreeSet;
use egglog_r2il_probe::heritage::{
    Atom27, HeritageEntry, HeritageRefusal, PROPOSED_ATOMS, analyse, sample_corpus,
};

#[test]
fn exactly_27_candidate_atoms_without_a_universality_claim() {
    let unique: BTreeSet<Atom27> = PROPOSED_ATOMS.into_iter().collect();
    assert_eq!(PROPOSED_ATOMS.len(), 27);
    assert_eq!(unique.len(), 27);
}

#[test]
fn historical_catalogue_leads_do_not_execute_as_proofs() {
    for entry in [
        HeritageEntry::leibniz_product_rule(),
        HeritageEntry::euler_identity(),
    ] {
        assert!(entry.source.is_some());
        assert!(!entry.is_mathematically_certified());
        assert_eq!(
            entry.replay(7),
            Err(HeritageRefusal::NoCheckedMathematicalProof)
        );
    }
}

#[test]
fn synthetic_e1_is_replayable_but_not_misattributed_to_leibniz() {
    let corpus = sample_corpus().expect("E1 proof certified");
    assert_eq!(corpus[0].id, "synthetic-e1-1690");
    assert!(corpus[0].source.is_none());
    assert_eq!(corpus[0].replay(7), Ok(56));
    let report = analyse(&corpus);
    assert_eq!(report.total_entries, 3);
    assert_eq!(report.historical_leads, 2);
    assert_eq!(report.mathematically_certified, 1);
    assert_eq!(report.distinct_compiled_programs, 1);
    assert_eq!(
        report.observed_atoms,
        vec![Atom27::Const, Atom27::Var, Atom27::Multiply]
    );
}
