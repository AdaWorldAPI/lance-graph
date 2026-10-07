//! D-GSO-4 (P4): entropy × topology routing probe.
//!
//! Plan: `.claude/plans/2026-10-06-global-sudoku-replayable-orchestration-v1.md`
//! §8, §10 and §18 P4; it is falsifier F-ECG-1 (with F-ECG-2) of
//! `.claude/plans/entropy-closure-causal-ground-v1.md`.
//!
//! F-ECG-3 is NOT covered here. It requires the census to discriminate on a
//! real corpus; no stored corpus carries topology bits on real edges yet, and
//! the hand-built fixtures below are chosen to cross the midpoint, so a
//! fixture test cannot show the census avoids collapsing to one class.
//!
//! Claim under test: two basins with IDENTICAL low field entropy, one whose
//! edges are mostly `Direct`, one mostly `Unknown` (CausalEdge64 bits 59..60
//! read through the `CausalTopology` lens), must land in different settlement
//! cells and be routed differently. If they route the same, entropy is being
//! read as mastery.
//!
//! What it reuses, and what it adds:
//!
//! - **Bits 59..60:** `CausalEdge64::with_topology` / `truth_raw`, unchanged.
//! - **Reading contract:** `band_reading::BandDeclarations::project_truth`
//!   (declared lens per `(classid, rail)`, asserted provenance). Projection is
//!   fallible; a refusal produces no cell.
//! - **Cell:** `settlement::SettlementSignals::cell`. Closure × competence
//!   decide; entropy and eigenvalue concentration are carried but do not.
//! - **New (the D-ECG-2 census):** the share of a basin's edges whose
//!   projected topology is known (`Direct` or `IndirectKnownIntermediates`)
//!   is used as evidence competence. Projected (`IndirectUnknownIntermediates`)
//!   and unknown edges are not counted as earned. This weighting is a policy
//!   choice for the probe, not a measured one.
//! - **Route:** the walker decision table of the entropy-closure plan §4b, one
//!   route per cell.
//!
//! Field entropy is measured, not assigned: the Shannon entropy of each
//! basin's edge-target histogram. Both basins have the same targets, so their
//! entropy is equal by measurement.
//!
//! Out of scope: Glass routing into counterfactual + revision (D-ECG-3,
//! F-ECG-4) and the band gate on 61..63 (D-ECG-6). No new bits.
//!
//! Run: `cargo run -p cognitive-shader-driver --example entropy_topology_probe`
//! Tests: `cargo test -p cognitive-shader-driver --example entropy_topology_probe`
//!
//! D-EPI-MIG-0 (2026-10-07): PROBE-ONLY legacy. Reads/writes the
//! historical `CausalTopology` half of CE64 bits 59..63 through
//! `band_reading`; that split reading no longer owns the field
//! (`contract::epistemic_state5` does). Kept as a record of the lens.
#![allow(deprecated)]

use causal_edge::edge::CausalEdge64;
use causal_edge::layout::CausalTopology;
use lance_graph_contract::band_reading::{
    BandDeclarations, BandReadError, BandReading, EdgeProvenance, TruthLens,
};
use lance_graph_contract::class_view::ClassId;
use lance_graph_contract::rail_geometry::RailAxis;
use lance_graph_contract::settlement::{SettlementCell, SettlementScope, SettlementSignals};

/// The fixture class whose producers write the topology lens.
const CLASS: ClassId = 0x0901;
/// The rail the declaration is made for.
const RAIL: RailAxis = RailAxis::Taxonomy;

/// One basin: its edges (target ordinal, edge) and the provenance the caller
/// asserts for the register they were read from.
struct Basin {
    edges: Vec<(u8, CausalEdge64)>,
    provenance: EdgeProvenance,
    closure_density: f32,
}

/// Topology census of one basin.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct Census {
    known: u32,
    projected: u32,
    hole: u32,
}

impl Census {
    fn total(self) -> u32 {
        self.known + self.projected + self.hole
    }

    /// Share of edges whose topology is known.
    fn competence(self) -> f32 {
        if self.total() == 0 {
            0.0
        } else {
            self.known as f32 / self.total() as f32
        }
    }
}

/// Count each edge's projected topology. Refuses as a whole if any edge
/// cannot be projected under the declaration and provenance.
fn census(decl: &BandDeclarations, basin: &Basin) -> Result<Census, BandReadError> {
    let mut c = Census::default();
    for &(_, edge) in &basin.edges {
        let raw = decl.project_truth(
            CLASS,
            RAIL,
            TruthLens::Topology,
            edge.truth_raw(),
            basin.provenance,
        )?;
        match CausalTopology::from_bits_2(raw) {
            CausalTopology::Direct | CausalTopology::IndirectKnownIntermediates => c.known += 1,
            CausalTopology::IndirectUnknownIntermediates => c.projected += 1,
            CausalTopology::Unknown => c.hole += 1,
        }
    }
    Ok(c)
}

/// Shannon entropy (bits) of the edge-target histogram.
fn field_entropy(basin: &Basin) -> f32 {
    let mut counts = [0u32; 256];
    for &(target, _) in &basin.edges {
        counts[target as usize] += 1;
    }
    let n = basin.edges.len() as f32;
    counts
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f32 / n;
            -p * p.log2()
        })
        .sum()
}

/// The walker's next move for a cell (entropy-closure plan §4b).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Route {
    /// Crystal: stable and grounded, continue normally.
    Continue,
    /// Glass: stable but causally under-grounded, seek the missing mediator.
    SeekMissingMediator,
    /// GroundedUnresolved: constrained neighbourhood, search the candidate basin.
    SearchCandidateBasin,
    /// Fog: no constraint, acquire means or evidence instead.
    AcquireMeans,
}

fn route(cell: SettlementCell) -> Route {
    match cell {
        SettlementCell::Crystal => Route::Continue,
        SettlementCell::Glass => Route::SeekMissingMediator,
        SettlementCell::GroundedUnresolved => Route::SearchCandidateBasin,
        SettlementCell::Fog => Route::AcquireMeans,
    }
}

/// Census → signals. An error when the census refuses; no cell is produced.
fn signals(decl: &BandDeclarations, basin: &Basin) -> Result<SettlementSignals, BandReadError> {
    let c = census(decl, basin)?;
    Ok(SettlementSignals {
        scope: SettlementScope {
            arena_id: 1,
            basin_id: Some(0),
            version: 1,
            branch_id: 0,
            witness_horizon: 1,
        },
        closure_density: basin.closure_density,
        evidence_competence: c.competence(),
        field_entropy: field_entropy(basin),
        eigenvalue_concentration: 0.5,
    })
}

fn declarations() -> BandDeclarations {
    let mut d = BandDeclarations::new();
    d.declare(
        CLASS,
        RAIL,
        BandReading {
            truth_lens: TruthLens::Topology,
            ..BandReading::ZERO_FALLBACK
        },
    );
    d
}

/// Twelve edges, ten to target 0 and one each to targets 1 and 2 (low
/// entropy), with the topology of each edge given in order.
fn basin(topologies: [CausalTopology; 12], closure_density: f32) -> Basin {
    let targets = [0u8, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0];
    Basin {
        edges: targets
            .iter()
            .zip(topologies)
            .map(|(&t, topo)| (t, CausalEdge64::ZERO.with_topology(topo)))
            .collect(),
        provenance: EdgeProvenance::V2Stamped,
        closure_density,
    }
}

fn direct_dominant(closure: f32) -> Basin {
    use CausalTopology::*;
    basin(
        [
            Direct,
            Direct,
            IndirectKnownIntermediates,
            Direct,
            Direct,
            Direct,
            IndirectKnownIntermediates,
            Direct,
            Direct,
            Unknown,
            Direct,
            IndirectUnknownIntermediates,
        ],
        closure,
    )
}

fn unknown_dominant(closure: f32) -> Basin {
    use CausalTopology::*;
    basin(
        [
            Unknown,
            Unknown,
            IndirectUnknownIntermediates,
            Unknown,
            Unknown,
            Direct,
            Unknown,
            IndirectUnknownIntermediates,
            Unknown,
            Unknown,
            IndirectKnownIntermediates,
            Unknown,
        ],
        closure,
    )
}

fn main() {
    let decl = declarations();
    println!("D-GSO-4 entropy x topology probe");
    for (name, b) in [
        ("direct-dominant, closed ", direct_dominant(0.9)),
        ("unknown-dominant, closed", unknown_dominant(0.9)),
        ("direct-dominant, open   ", direct_dominant(0.2)),
        ("unknown-dominant, open  ", unknown_dominant(0.2)),
    ] {
        let s = signals(&decl, &b).expect("V2Stamped provenance projects");
        println!(
            "  {name}  H={:.4}  census={:?}  competence={:.3}  -> {:?} / {:?}",
            s.field_entropy,
            census(&decl, &b).unwrap(),
            s.evidence_competence,
            s.cell(),
            route(s.cell()),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// F-ECG-1. FAILS IF: two basins with identical low entropy and identical
    /// closure land in the same cell or take the same route.
    #[test]
    fn equal_entropy_different_topology_routes_differently() {
        let decl = declarations();
        let earned = signals(&decl, &direct_dominant(0.9)).unwrap();
        let thin = signals(&decl, &unknown_dominant(0.9)).unwrap();

        // Anti-vacuity: entropy and closure really are identical, and low.
        assert_eq!(earned.field_entropy.to_bits(), thin.field_entropy.to_bits());
        assert!(earned.field_entropy < 1.0);
        assert_eq!(earned.closure_density, thin.closure_density);

        assert_eq!(earned.cell(), SettlementCell::Crystal);
        assert_eq!(thin.cell(), SettlementCell::Glass);
        assert_ne!(route(earned.cell()), route(thin.cell()));
        // Both look settled; only the census separates them.
        assert!(earned.cell().appears_settled() && thin.cell().appears_settled());
    }

    /// FAILS IF: a selector that reads entropy alone could tell the two apart.
    /// This is what the probe exists to refuse: equal entropy, equal answer.
    #[test]
    fn an_entropy_only_selector_cannot_separate_them() {
        let entropy_only = |b: &Basin| field_entropy(b) < 1.0;
        assert_eq!(
            entropy_only(&direct_dominant(0.9)),
            entropy_only(&unknown_dominant(0.9))
        );
    }

    /// FAILS IF: the census miscounts. Exact counts pinned for both basins.
    #[test]
    fn census_counts_exactly() {
        let decl = declarations();
        assert_eq!(
            census(&decl, &direct_dominant(0.9)).unwrap(),
            Census {
                known: 10,
                projected: 1,
                hole: 1
            }
        );
        assert_eq!(
            census(&decl, &unknown_dominant(0.9)).unwrap(),
            Census {
                known: 2,
                projected: 2,
                hole: 8
            }
        );
    }

    /// F-ECG-2. FAILS IF: a cell emerges from bits that cannot be read: unknown
    /// or v1 provenance, a class declared under the trust lens, or a class
    /// that was never declared.
    #[test]
    fn unreadable_ground_produces_no_cell() {
        let decl = declarations();
        for p in [EdgeProvenance::Unknown, EdgeProvenance::V1Legacy] {
            let mut b = direct_dominant(0.9);
            b.provenance = p;
            assert_eq!(
                signals(&decl, &b).unwrap_err(),
                BandReadError::UnknownProvenance
            );
        }

        let mut trust = BandDeclarations::new();
        trust.declare(CLASS, RAIL, BandReading::ZERO_FALLBACK);
        assert!(matches!(
            signals(&trust, &direct_dominant(0.9)),
            Err(BandReadError::LensMismatch { .. })
        ));

        assert_eq!(
            signals(&BandDeclarations::new(), &direct_dominant(0.9)).unwrap_err(),
            BandReadError::UndeclaredClass(CLASS)
        );

        let mut asserted = direct_dominant(0.9);
        asserted.provenance = EdgeProvenance::V3Register;
        assert!(
            signals(&decl, &asserted).is_ok(),
            "an asserted register reads"
        );
    }

    /// FAILS IF: one of the four routes is unreachable from the census. The
    /// fixtures are hand-built to cross the midpoint, so this is reachability
    /// only, not F-ECG-3's real-corpus anti-vacuity check.
    #[test]
    fn every_route_is_reachable_from_the_fixtures() {
        let decl = declarations();
        let routes = [
            direct_dominant(0.9),
            unknown_dominant(0.9),
            direct_dominant(0.2),
            unknown_dominant(0.2),
        ]
        .map(|b| route(signals(&decl, &b).unwrap().cell()));
        assert_eq!(
            routes,
            [
                Route::Continue,
                Route::SeekMissingMediator,
                Route::SearchCandidateBasin,
                Route::AcquireMeans,
            ]
        );
    }

    /// FAILS IF: entropy moves the cell. The same census under any entropy
    /// stays in the same cell.
    #[test]
    fn entropy_refines_but_never_routes() {
        let decl = declarations();
        let base = signals(&decl, &unknown_dominant(0.9)).unwrap();
        for h in [0.0_f32, 0.5, 1.0, 4.0, 8.0] {
            let s = SettlementSignals {
                field_entropy: h,
                ..base
            };
            assert_eq!(route(s.cell()), Route::SeekMissingMediator, "H = {h}");
        }
    }
}
