//! D-EPI-CONFLICT-0: the affordance law reads `CausalEdge64` bits 59..63
//! without the unified 59..63 contract, and disagrees with the reading a
//! class declares through it. This probe pins the disagreement; it does not
//! resolve it.
//!
//! Issue: `ISS-EPISTEMIC-READINGS-DISAGREE-ON-BITS-61-63` in
//! `.claude/board/ISSUES.md`.
//!
//! # The unified contract
//!
//! `lance_graph_contract::band_reading` (D-ACR-7) is the one 59..63 reading
//! contract: per `(classid, rail)` a class declares which lens its producers
//! wrote into bits 59..60 (`TruthLens`) and whether bits 61..63 carry a band
//! (`BandPresence`). Projection is fallible: undeclared class, band absent and
//! untrusted provenance refuse. It declares two split fields; it has no joint
//! 5-bit lens.
//!
//! # The two readings
//!
//! - **P7a** (`relational_certification_probe.rs`, D-GSO-7a, #1369) writes a
//!   relational contract into bits 61..63 through `with_reasoning_band`:
//!   `0 Open, 1 Associated, 2 Related, 3 Contributes, 4 CausalCandidate,
//!   5 Causes`, 6 and 7 refuse. Comparison is ordinal on 0..=5 (`entails`).
//!   It reads the band through `BandDeclarations::project_band` with the class
//!   declared `BandPresence::Present`: it goes through the contract.
//! - **The affordance law** (`shared/affordance_law.rs`, D-GSO-AFF-0, #1370;
//!   read by four probes) reads bits 59..63 as ONE 5-bit EpistemicState5 code,
//!   `raw5 = spare() << 2 | truth_raw()`. Bits 61..63 are `spare()`, the same
//!   three bits P7a writes, so `raw5 >> 2` IS the P7a code. The law's codebook
//!   is deliberately not ordered and gives those bits other meanings. It takes
//!   no class and never consults the contract: it checks provenance only.
//!
//! The affordance probe says so itself ("a joint 5-bit reading needs its own
//! declaration there before it can be relied on; this probe declares it
//! locally"). That declaration does not exist, and
//! `ISS-NO-EVIDENCE-WRITER-FOR-EPISTEMIC-STATE` cannot be closed with P7a as
//! the writer until the two agree.
//!
//! # What is pinned
//!
//! | test | pin |
//! |---|---|
//! | `the_affordance_law_reads_bits_the_contract_refuses` | an undeclared class and a band-`Absent` class refuse under the contract; the affordance law measures both |
//! | `a_certified_causes_never_reads_as_causes` | the contract projects `Causes` (5); the affordance law reads `Related` (topology 0) or refuses (1..3); `COUNTERFACTUAL_PROBE` is never eligible |
//! | `the_affordance_causes_codes_are_not_p7a_causes` | the two codes that assert `CAUSES` sit on P7a `Open` and `CausalCandidate` |
//! | `affordance_codes_occupy_p7a_reserved_bands` | two declared codes sit on P7a's refused bands 6 and 7 |
//! | `the_readings_agree_on_three_of_eight_shared_cells` | exhaustive count over every P7a contract × topology |
//!
//! Both readings are exercised from shared code, never copied:
//! `shared/affordance_law.rs` and `shared/certification_reading.rs` (P7a's
//! classes, declarations, contract table, `entails` and `stamp`). Each test is
//! a pin of the current state: it FAILS when either codebook changes. A reconciliation should turn these into agreement tests and close
//! the issue, not edit the numbers to match.
//!
//! The agreement test compares only the three facts that share a name with a
//! P7a contract (`ASSOCIATED`, `RELATED`, `CAUSES`). P7a's `Contributes` and
//! `CausalCandidate` have no same-named fact, and `SUPPORTS` is not assumed to
//! mean `Contributes`.
//!
//! Run: `cargo run -p cognitive-shader-driver --example epistemic_reading_conflict_probe`
//! Tests: `cargo test -p cognitive-shader-driver --example epistemic_reading_conflict_probe`

#[path = "shared/affordance_law.rs"]
mod affordance_law;
#[path = "shared/certification_reading.rs"]
mod certification_reading;

use affordance_law::{raw5, ASSOCIATED, CAUSES, EPI_LAW, RELATED};
use causal_edge::edge::CausalEdge64;
use causal_edge::layout::TrustTexture;
use causal_edge::pearl::CausalMask;
use causal_edge::PlasticityState;
use certification_reading::{declarations, read, Contract, CERT_CLASS, RAIL};
use lance_graph_contract::band_reading::{BandReadError, BandReading, EdgeProvenance};
use lance_graph_contract::class_view::ClassId;

/// A class declared band-free: its bits 61..63 are spare.
const SPARE_CLASS: ClassId = 0x0903;
/// A class never declared.
const UNDECLARED_CLASS: ClassId = 0x0904;

/// P7a's contract as P7a's own reader returns it.
fn p7a_reads(edge: CausalEdge64) -> Result<Contract, certification_reading::Refusal> {
    read(&declarations(), CERT_CLASS, edge, EdgeProvenance::V2Stamped)
}

/// The band as the unified contract projects it for `class`: P7a's
/// declarations plus one band-free class.
fn contract_band(class: ClassId, edge: CausalEdge64) -> Result<u8, BandReadError> {
    let mut d = declarations();
    d.declare(SPARE_CLASS, RAIL, BandReading::ZERO_FALLBACK);
    d.project_band(
        class,
        RAIL,
        edge.reasoning_band().to_bits_3(),
        EdgeProvenance::V2Stamped,
    )
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

/// What P7a writes (its own `stamp`) plus a topology in bits 59..60.
fn stamp(contract: Contract, topology: u8) -> CausalEdge64 {
    certification_reading::stamp(base(), contract).with_truth(TrustTexture::from_bits_2(topology))
}

/// The facts P7a's contract entails (its own `entails` table), in the
/// affordance vocabulary, for the three facts that share a name with a
/// contract.
fn p7a_entails(contract: Contract) -> u32 {
    let mut f = 0;
    if contract.entails(Contract::Associated) {
        f |= ASSOCIATED;
    }
    if contract.entails(Contract::Related) {
        f |= RELATED;
    }
    if contract.entails(Contract::Causes) {
        f |= CAUSES;
    }
    f
}

const SHARED: u32 = ASSOCIATED | RELATED | CAUSES;

/// The affordance reading of an edge: its facts, or `None` if it refuses.
fn affordance_facts(edge: CausalEdge64) -> Option<u32> {
    EPI_LAW[raw5(edge) as usize]
}

/// (agree, disagree, refused) over every P7a contract × topology.
fn census() -> (usize, usize, usize) {
    let (mut agree, mut disagree, mut refused) = (0, 0, 0);
    for c in Contract::ALL {
        for t in 0..4u8 {
            match affordance_facts(stamp(c, t)) {
                None => refused += 1,
                Some(f) if f & SHARED == p7a_entails(c) => agree += 1,
                Some(_) => disagree += 1,
            }
        }
    }
    (agree, disagree, refused)
}

fn main() {
    println!(
        "D-EPI-CONFLICT-0: P7a contract (bits 61..63) x topology (59..60) under the affordance law"
    );
    for c in Contract::ALL {
        for t in 0..4u8 {
            let e = stamp(c, t);
            let shown = match affordance_facts(e) {
                None => "refuses".to_string(),
                Some(f) => format!(
                    "facts {:#05x}{}",
                    f,
                    if f & SHARED == p7a_entails(c) {
                        ""
                    } else {
                        "  <- disagrees"
                    }
                ),
            };
            let declared = p7a_reads(e).expect("P7a reads its own stamp");
            println!(
                "  P7a {:<16} topo {t}  raw5 {:>2}  {shown}",
                format!("{declared:?}"),
                raw5(e)
            );
        }
    }
    let (a, d, r) = census();
    println!("agree {a}, disagree {d}, refused {r} of 24 cells");
    let e = stamp(Contract::Causes, 0);
    println!(
        "same edge under the contract: spare class {:?}, undeclared class {:?}; the affordance law takes no class",
        contract_band(SPARE_CLASS, e),
        contract_band(UNDECLARED_CLASS, e)
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use affordance_law::{measure, LawGen, Refusal, COUNTERFACTUAL_PROBE};

    /// The affordance measurement. It takes no class: there is nothing to
    /// look up a declaration with.
    fn eligible(edge: CausalEdge64) -> Result<u64, Refusal> {
        measure(LawGen::V1, edge, EdgeProvenance::V2Stamped)
    }

    /// FAILS IF: the affordance law starts refusing what the contract refuses,
    /// i.e. it begins to consult the declaration.
    #[test]
    fn the_affordance_law_reads_bits_the_contract_refuses() {
        let e = stamp(Contract::Causes, 0);
        assert_eq!(
            contract_band(UNDECLARED_CLASS, e),
            Err(BandReadError::UndeclaredClass(UNDECLARED_CLASS))
        );
        assert_eq!(
            contract_band(SPARE_CLASS, e),
            Err(BandReadError::BandAbsent)
        );
        // Same edge, any class: the affordance law measures it.
        assert!(eligible(e).is_ok());
        // Silence twin: the contract does admit the band on P7a's class.
        assert_eq!(contract_band(CERT_CLASS, e), Ok(Contract::Causes.code()));
    }

    /// FAILS IF: a P7a-certified `Causes` starts reading as `CAUSES` under the
    /// affordance law, or `COUNTERFACTUAL_PROBE` becomes eligible on it.
    #[test]
    fn a_certified_causes_never_reads_as_causes() {
        // Topology 0: code 20, read as Related.
        let e = stamp(Contract::Causes, 0);
        assert_eq!(raw5(e), 20);
        assert_eq!(p7a_reads(e), Ok(Contract::Causes), "P7a reads Causes");
        let facts = affordance_facts(e).expect("code 20 is declared");
        assert_eq!(facts & CAUSES, 0, "P7a Causes carries no CAUSES fact");
        assert_eq!(facts & SHARED, ASSOCIATED | RELATED);
        let ok = eligible(e).expect("code 20 is measurable");
        assert_eq!(ok & (1 << COUNTERFACTUAL_PROBE), 0);
        // Anti-vacuity: the plane rule does not hide the recipe; a code that
        // does carry CAUSES makes it eligible on the same edge.
        let with_causes = stamp(Contract::Open, 1);
        assert_ne!(
            eligible(with_causes).unwrap() & (1 << COUNTERFACTUAL_PROBE),
            0
        );

        // Topologies 1..3: codes 21..23, not declared by the affordance law.
        for t in 1..4u8 {
            let e = stamp(Contract::Causes, t);
            assert_eq!(p7a_reads(e), Ok(Contract::Causes));
            assert_eq!(eligible(e), Err(Refusal::Undeclared(20 + t)));
        }
    }

    /// FAILS IF: the codes that assert `CAUSES` move off P7a `Open` (code 1)
    /// and `CausalCandidate` (code 17).
    #[test]
    fn the_affordance_causes_codes_are_not_p7a_causes() {
        let causes: Vec<u8> = (0..32u8)
            .filter(|&c| EPI_LAW[c as usize].is_some_and(|f| f & CAUSES != 0))
            .collect();
        assert_eq!(causes, vec![1, 17]);
        let contracts: Vec<_> = causes.iter().map(|c| Contract::from_code(c >> 2)).collect();
        assert_eq!(
            contracts,
            vec![Ok(Contract::Open), Ok(Contract::CausalCandidate)]
        );
    }

    /// FAILS IF: no declared affordance code sits on a band P7a refuses, or a
    /// different set does.
    #[test]
    fn affordance_codes_occupy_p7a_reserved_bands() {
        let reserved: Vec<u8> = (0..32u8)
            .filter(|&c| EPI_LAW[c as usize].is_some() && Contract::from_code(c >> 2).is_err())
            .collect();
        assert_eq!(reserved, vec![25, 30]);
    }

    /// FAILS IF: the agreement census over all 24 P7a contract × topology
    /// cells changes.
    #[test]
    fn the_readings_agree_on_three_of_eight_shared_cells() {
        assert_eq!(census(), (3, 5, 16));
    }
}
