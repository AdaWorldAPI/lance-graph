//! `shape` — the DATA SHAPE DETECTOR: **the graph reasoning about the best
//! possible representation of its own knowledge.**
//!
//! The operator's "brutal" move (`self-reasoning-substrate-v1` D-SRS-2
//! reshaped): per predicate, measure the edge-set's SHAPE and route it to the
//! right carrier — instead of materializing every relation as derived triples
//! and hoping the closure stays small. This module is the first mechanical
//! **rung-2 meta-awareness** citizen: its subject matter is not the world but
//! the graph's own representation of the world.
//!
//! ## Two detectors: [`detect_measured`] (shipped) vs [`detect`] (record)
//!
//! [`detect_measured`] is the SHIPPED router: it MEASURES the candidate trie
//! (coverage + amortization) and routes on the measured fit. [`detect`] is the
//! v1 STRUCTURAL router — it guessed shape from degree statistics and its
//! *routing verdict* was falsified on the real book (a `max_in ≤ 1` purity gate
//! mis-routes a noisy near-forest, `D-SRS-2` RESULT). `detect` is retained as
//! that falsified record — but note its *function* is NOT dead: `detect_measured`
//! calls it every invocation for the shared structural stats (`edges`,
//! `entities`, `cyclic`, `closure_pressure`). Only v1's routing VERDICT is
//! retired; [`detect_all`] (the v1 census wrapper) is the genuinely test-only part.
//!
//! ## The structural stats
//!
//! Per predicate `p` over its (deduplicated) edge set: `edges`, `entities`,
//! `max_in`, `max_out`, `cyclic` (directed DFS), and `closure_pressure = Σ_v
//! in(v)·out(v)` — the count of length-2 directed composition PATHS. This is an
//! **upper bound / proxy** for a transitive closure's first-pass growth, NOT the
//! exact addition count: the closure dedups (two paths through different
//! intermediates collapse to one endpoint; a path whose shortcut already exists
//! adds nothing). It is used only as a routing threshold, never as a correctness
//! gate — it was the O(N²)-risk signal that fired in `D-SRS-1`'s whole-book finding.
//!
//! v1 classification order (the falsified record): `Empty` → `Cyclic` → `Flat`
//! (pressure 0; a star is Flat) → `Forest` (`max_in ≤ 1`) → `Dag`. The v2
//! measured routing lives on [`detect_measured`].

use crate::ancestry::FamilyTrie;
use crate::spo::Spo;
use std::collections::HashMap;

/// The measured-fit thresholds for routing a predicate to a trie (v2 router).
/// A trie must cover ≥ this share of its entities and pay ≥ this amortization
/// (ancestor pairs per stored pointer) or it is not worth relocating.
const MIN_TRIE_COVERAGE: f64 = 0.8;
const MIN_TRIE_AMORTIZATION: f64 = 2.0;

/// The shape class of one predicate's edge set (pre-registered order).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShapeClass {
    /// No edges.
    Empty,
    /// The predicate subgraph contains a directed cycle.
    Cyclic,
    /// `closure_pressure == 0` — no entity is both an object and a subject;
    /// transitive composition can never fire (a star is Flat).
    Flat,
    /// Every entity has at most one parent (`max_in ≤ 1`) — a forest.
    Forest,
    /// Acyclic with multi-parent entities.
    Dag,
}

/// Where the detector routes a predicate's knowledge (pre-registered).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Representation {
    /// Leave as plain edges — closure adds nothing (Empty / Flat).
    EdgeTable,
    /// The DN / HHTL family radix-trie codebook ([`crate::ancestry::FamilyTrie`]);
    /// ancestry = prefix containment; the closure is NEVER materialized.
    RadixTrie,
    /// A small materialized derivation fabric is fine (Dag with low pressure).
    MaterializedFabric,
    /// Primary-parent trie + the multi-parent residue as pointers/Escalate
    /// (Dag with high pressure).
    TriePlusEscalate,
    /// Cyclic — bounded fabric + global-graph Escalate; no trie can ground it.
    BoundedEscalate,
}

impl Representation {
    /// The **G byte of an SPOG quad** (S·P·O·**G**raph). A SoA "SPOG tenant"
    /// stamps each triple with its predicate's `G`, so a reader routes by `G` —
    /// *which shape-graph the SPO participates in* — without re-detecting. The
    /// detector's per-predicate census is thus not an ephemeral report but the
    /// **materialized G lane**: the shape verdict linked to the SPO role. Fits
    /// the `4×(u8:u8:u8)` SPO-triplet carving of the 12-byte facet extended with
    /// this `G` (`le-contract` §3). Codes are stable — persisted rows depend on
    /// them (append new variants, never renumber).
    #[must_use]
    pub const fn graph_id(self) -> u8 {
        match self {
            Self::EdgeTable => 0,
            Self::RadixTrie => 1,
            Self::TriePlusEscalate => 2,
            Self::MaterializedFabric => 3,
            Self::BoundedEscalate => 4,
        }
    }
}

/// The per-predicate shape report — the detector's rung-2 verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShapeReport {
    /// The predicate word id this report is about.
    pub predicate: u16,
    /// Edge count (deduplicated `(subject, object)` pairs).
    pub edges: usize,
    /// Distinct entities touched.
    pub entities: usize,
    /// Max in-degree (parents per entity).
    pub max_in: usize,
    /// Max out-degree (children per entity).
    pub max_out: usize,
    /// Whether the subgraph contains a directed cycle.
    pub cyclic: bool,
    /// `Σ_v in(v)·out(v)` — length-2 composition PATHS; an upper bound on a
    /// transitive closure's first-pass growth (the closure dedups), the O(N²)
    /// routing signal.
    pub closure_pressure: u64,
    /// The shape class.
    pub class: ShapeClass,
    /// The routed representation.
    pub recommend: Representation,
}

/// Classify one predicate's deduplicated edge set.
#[must_use]
pub fn detect(predicate: u16, edges_in: &[(u16, u16)]) -> ShapeReport {
    // Dedup edges — repeated observations are frequency, not structure.
    let mut edges: Vec<(u16, u16)> = edges_in.to_vec();
    edges.sort_unstable();
    edges.dedup();

    let mut in_deg: HashMap<u16, usize> = HashMap::new();
    let mut out_deg: HashMap<u16, usize> = HashMap::new();
    let mut adj: HashMap<u16, Vec<u16>> = HashMap::new();
    for &(s, o) in &edges {
        *out_deg.entry(s).or_insert(0) += 1;
        *in_deg.entry(o).or_insert(0) += 1;
        adj.entry(s).or_default().push(o);
    }
    let mut entities: Vec<u16> = in_deg.keys().chain(out_deg.keys()).copied().collect();
    entities.sort_unstable();
    entities.dedup();

    let max_in = in_deg.values().copied().max().unwrap_or(0);
    let max_out = out_deg.values().copied().max().unwrap_or(0);
    let closure_pressure: u64 = entities
        .iter()
        .map(|e| {
            in_deg.get(e).copied().unwrap_or(0) as u64 * out_deg.get(e).copied().unwrap_or(0) as u64
        })
        .sum();

    // Directed-cycle detection: iterative 3-color DFS.
    let cyclic = has_cycle(&entities, &adj);

    let (class, recommend) = if edges.is_empty() {
        (ShapeClass::Empty, Representation::EdgeTable)
    } else if cyclic {
        (ShapeClass::Cyclic, Representation::BoundedEscalate)
    } else if closure_pressure == 0 {
        (ShapeClass::Flat, Representation::EdgeTable)
    } else if max_in <= 1 {
        (ShapeClass::Forest, Representation::RadixTrie)
    } else if closure_pressure <= 4 * edges.len() as u64 {
        (ShapeClass::Dag, Representation::MaterializedFabric)
    } else {
        (ShapeClass::Dag, Representation::TriePlusEscalate)
    };

    ShapeReport {
        predicate,
        edges: edges.len(),
        entities: entities.len(),
        max_in,
        max_out,
        cyclic,
        closure_pressure,
        class,
        recommend,
    }
}

/// The v2 MEASURED report: structural stats plus the trie fit actually
/// measured on the candidate representation. `PartialEq` only (carries `f64`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeasuredShape {
    /// The predicate word id.
    pub predicate: u16,
    /// Deduplicated edge count.
    pub edges: usize,
    /// Distinct entities.
    pub entities: usize,
    /// Contains a directed cycle.
    pub cyclic: bool,
    /// `Σ_v in(v)·out(v)` — length-2 paths; an upper-bound O(N²) routing signal.
    pub closure_pressure: u64,
    /// Entities the primary-parent trie grounds to a root.
    pub covered: usize,
    /// `covered / (covered + cycle_residue)` — the share the trie can ground.
    pub coverage: f64,
    /// `ancestor_pairs / covered` — pairs answered per stored pointer.
    pub amortization: f64,
    /// No multi-parent and no cycle residue (a pure forest).
    pub residue_free: bool,
    /// The routed representation under the v2 measured rule.
    pub recommend: Representation,
}

/// v2 detector — **measures** the candidate trie instead of guessing shape from
/// degree statistics (v1's [`detect`] purity gate was falsified on the real
/// book: one FSM mis-parse demotes a 99%-forest). Builds the residue-tolerant
/// [`FamilyTrie`], measures coverage + amortization, and routes on the measured
/// fit — a trie that pays (`coverage ≥ 0.8`, `amortization ≥ 2.0`) wins even
/// with residue; one that does not is bounded, never forced.
#[must_use]
pub fn detect_measured(predicate: u16, edges_in: &[(u16, u16)]) -> MeasuredShape {
    let base = detect(predicate, edges_in); // structural stats (dedups internally)
    let mut edges: Vec<(u16, u16)> = edges_in.to_vec();
    edges.sort_unstable();
    edges.dedup();
    let trie = FamilyTrie::build(&edges);

    let covered = trie.covered();
    let cyc = trie.cycle_residue();
    let coverage = if covered + cyc == 0 {
        0.0
    } else {
        covered as f64 / (covered + cyc) as f64
    };
    let amortization = if covered == 0 {
        0.0
    } else {
        trie.pair_count() as f64 / covered as f64
    };
    let residue_free = trie.multi_parent_residue() == 0 && cyc == 0;

    let recommend = if base.edges == 0 || base.closure_pressure == 0 {
        Representation::EdgeTable
    } else if coverage >= MIN_TRIE_COVERAGE && amortization >= MIN_TRIE_AMORTIZATION {
        if residue_free {
            Representation::RadixTrie
        } else {
            Representation::TriePlusEscalate
        }
    } else if base.cyclic {
        Representation::BoundedEscalate
    } else if base.closure_pressure <= 4 * base.edges as u64 {
        Representation::MaterializedFabric
    } else {
        // High-pressure acyclic that the trie can't pay for: a trie that does
        // not amortize is not a fallback — bound it.
        Representation::BoundedEscalate
    };

    MeasuredShape {
        predicate,
        edges: base.edges,
        entities: base.entities,
        cyclic: base.cyclic,
        closure_pressure: base.closure_pressure,
        covered,
        coverage,
        amortization,
        residue_free,
        recommend,
    }
}

/// The whole-KG MEASURED census (v2), sorted by edge count descending.
#[must_use]
pub fn detect_all_measured(triples: &[Spo]) -> Vec<MeasuredShape> {
    let mut by_p: HashMap<u16, Vec<(u16, u16)>> = HashMap::new();
    for t in triples {
        by_p.entry(t.predicate)
            .or_default()
            .push((t.subject, t.object));
    }
    let mut out: Vec<MeasuredShape> = by_p
        .into_iter()
        .map(|(p, edges)| detect_measured(p, &edges))
        .collect();
    out.sort_by(|a, b| b.edges.cmp(&a.edges).then(a.predicate.cmp(&b.predicate)));
    out
}

/// Group a triple stream by predicate and classify each — the whole-KG shape
/// census, sorted by edge count descending. v1 structural detector, retained
/// as the falsified, regression-pinned record (see `detect_measured`).
#[must_use]
pub fn detect_all(triples: &[Spo]) -> Vec<ShapeReport> {
    let mut by_p: HashMap<u16, Vec<(u16, u16)>> = HashMap::new();
    for t in triples {
        by_p.entry(t.predicate)
            .or_default()
            .push((t.subject, t.object));
    }
    let mut out: Vec<ShapeReport> = by_p
        .into_iter()
        .map(|(p, edges)| detect(p, &edges))
        .collect();
    out.sort_by(|a, b| b.edges.cmp(&a.edges).then(a.predicate.cmp(&b.predicate)));
    out
}

/// Iterative 3-color DFS over the directed adjacency; true iff a back edge
/// exists.
fn has_cycle(entities: &[u16], adj: &HashMap<u16, Vec<u16>>) -> bool {
    #[derive(Clone, Copy, PartialEq)]
    enum Color {
        White,
        Gray,
        Black,
    }
    let mut color: HashMap<u16, Color> = entities.iter().map(|&e| (e, Color::White)).collect();
    for &start in entities {
        if color[&start] != Color::White {
            continue;
        }
        // Stack of (node, next-child-index).
        let mut stack: Vec<(u16, usize)> = vec![(start, 0)];
        color.insert(start, Color::Gray);
        while let Some(&mut (node, ref mut idx)) = stack.last_mut() {
            let children = adj.get(&node).map_or(&[][..], Vec::as_slice);
            if *idx < children.len() {
                let child = children[*idx];
                *idx += 1;
                match color[&child] {
                    Color::Gray => return true, // back edge — cycle
                    Color::White => {
                        color.insert(child, Color::Gray);
                        stack.push((child, 0));
                    }
                    Color::Black => {}
                }
            } else {
                color.insert(node, Color::Black);
                stack.pop();
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    /// G-SRS2-c: a chain is Forest → RadixTrie.
    #[test]
    fn chain_is_forest_radix_trie() {
        let r = detect(1, &[(1, 2), (2, 3), (3, 4)]);
        assert_eq!(r.class, ShapeClass::Forest);
        assert_eq!(r.recommend, Representation::RadixTrie);
        assert!(!r.cyclic);
        assert_eq!(r.closure_pressure, 2); // entities 2 and 3 compose
    }

    /// G-SRS2-c: a directed cycle is Cyclic → BoundedEscalate.
    #[test]
    fn cycle_is_bounded_escalate() {
        let r = detect(2, &[(1, 2), (2, 3), (3, 1)]);
        assert_eq!(r.class, ShapeClass::Cyclic);
        assert_eq!(r.recommend, Representation::BoundedEscalate);
        assert!(r.cyclic);
    }

    /// G-SRS2-c: disjoint pairs are Flat → EdgeTable (pressure 0).
    #[test]
    fn disjoint_pairs_are_flat_edge_table() {
        let r = detect(3, &[(1, 2), (3, 4), (5, 6)]);
        assert_eq!(r.class, ShapeClass::Flat);
        assert_eq!(r.recommend, Representation::EdgeTable);
        assert_eq!(r.closure_pressure, 0);
    }

    /// G-SRS2-c: a star (one root, N children) is Flat → EdgeTable — relocation
    /// buys nothing at depth 1 (pressure 0), even though it is technically a
    /// forest.
    #[test]
    fn star_is_flat_edge_table() {
        let edges: Vec<(u16, u16)> = (1..=10).map(|c| (0, c)).collect();
        let r = detect(4, &edges);
        assert_eq!(r.class, ShapeClass::Flat);
        assert_eq!(r.recommend, Representation::EdgeTable);
        assert_eq!(r.max_in, 1);
    }

    /// G-SRS2-c: a dense multi-parent DAG routes by pressure — low pressure →
    /// MaterializedFabric; high pressure → TriePlusEscalate.
    #[test]
    fn dag_routes_by_pressure() {
        // Low pressure: diamond 1→2, 1→3, 2→4, 3→4 (max_in=2; pressure =
        // in(2)*out(2) + in(3)*out(3) = 1 + 1 = 2 ≤ 4×4).
        let low = detect(5, &[(1, 2), (1, 3), (2, 4), (3, 4)]);
        assert_eq!(low.class, ShapeClass::Dag);
        assert_eq!(low.recommend, Representation::MaterializedFabric);

        // High pressure: a 10×10 bipartite hub through a single waist node w:
        // 10 parents → w, w → 10 children ⇒ pressure at w = 10·10 = 100 >
        // 4×20 = 80.
        let mut edges: Vec<(u16, u16)> = Vec::new();
        let w = 100u16;
        for p in 1..=10u16 {
            edges.push((p, w));
        }
        for c in 200..210u16 {
            edges.push((w, c));
        }
        let high = detect(6, &edges);
        assert_eq!(high.class, ShapeClass::Dag);
        assert_eq!(high.recommend, Representation::TriePlusEscalate);
        assert_eq!(high.closure_pressure, 100);
    }

    /// Repeated observations of the same edge are frequency, not structure —
    /// deduplicated before classification.
    #[test]
    fn duplicate_edges_dedup_before_classification() {
        let r = detect(7, &[(1, 2), (1, 2), (1, 2), (2, 3)]);
        assert_eq!(r.edges, 2);
        assert_eq!(r.class, ShapeClass::Forest);
    }

    /// The census sorts by edge count descending.
    #[test]
    fn census_sorts_by_edges_desc() {
        let mut triples = Vec::new();
        for k in 0..5u16 {
            triples.push(Spo::new(k, 9, k + 1)); // predicate 9: 5 edges
        }
        triples.push(Spo::new(1, 8, 2)); // predicate 8: 1 edge
        let census = detect_all(&triples);
        assert_eq!(census[0].predicate, 9);
        assert_eq!(census[1].predicate, 8);
    }

    // ── v2 measured router (G-SRS2v2-c) ──

    /// A 10-node chain measures coverage 1.0, amortization 4.5 → RadixTrie.
    #[test]
    fn v2_chain_measures_to_radix_trie() {
        let edges: Vec<(u16, u16)> = (0..9).map(|k| (k, k + 1)).collect(); // 10 nodes
        let m = detect_measured(1, &edges);
        assert_eq!(m.covered, 10);
        assert_eq!(m.coverage, 1.0);
        assert!((m.amortization - 4.5).abs() < 1e-9, "{m:?}"); // 45/10
        assert!(m.residue_free);
        assert_eq!(m.recommend, Representation::RadixTrie);
    }

    /// A directed cycle → BoundedEscalate (trie can't ground it).
    #[test]
    fn v2_cycle_is_bounded_escalate() {
        let m = detect_measured(2, &[(1, 2), (2, 3), (3, 1)]);
        assert!(m.cyclic);
        assert_eq!(m.recommend, Representation::BoundedEscalate);
    }

    /// Disjoint pairs and a star are EdgeTable (pressure 0).
    #[test]
    fn v2_flat_shapes_are_edge_table() {
        assert_eq!(
            detect_measured(3, &[(1, 2), (3, 4)]).recommend,
            Representation::EdgeTable
        );
        let star: Vec<(u16, u16)> = (1..=10).map(|c| (0, c)).collect();
        assert_eq!(
            detect_measured(4, &star).recommend,
            Representation::EdgeTable
        );
    }

    /// A diamond has fit amortization 1.0 (< 2) but low pressure →
    /// MaterializedFabric.
    #[test]
    fn v2_diamond_is_materialized_fabric() {
        let m = detect_measured(5, &[(1, 2), (1, 3), (2, 4), (3, 4)]);
        assert!(m.amortization < 2.0, "{m:?}");
        assert_eq!(m.recommend, Representation::MaterializedFabric);
    }

    /// A 10×10 waist DAG: fit fails AND pressure 100 > 4×20 → BoundedEscalate.
    #[test]
    fn v2_high_pressure_dag_without_fit_is_bounded() {
        let mut edges: Vec<(u16, u16)> = Vec::new();
        let w = 100u16;
        for p in 1..=10u16 {
            edges.push((p, w));
        }
        for c in 200..210u16 {
            edges.push((w, c));
        }
        let m = detect_measured(6, &edges);
        assert!(!m.cyclic);
        assert!(m.amortization < 2.0, "{m:?}");
        assert_eq!(m.closure_pressure, 100);
        assert_eq!(m.recommend, Representation::BoundedEscalate);
    }

    /// The SPOG G-lane codes are stable and distinct — a persisted row's `G`
    /// must decode to the same representation forever.
    #[test]
    fn spog_graph_ids_are_stable_and_distinct() {
        let all = [
            Representation::EdgeTable,
            Representation::RadixTrie,
            Representation::TriePlusEscalate,
            Representation::MaterializedFabric,
            Representation::BoundedEscalate,
        ];
        let ids: Vec<u8> = all.iter().map(|r| r.graph_id()).collect();
        assert_eq!(ids, vec![0, 1, 2, 3, 4]); // pinned — never renumber
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), all.len(), "G codes must be distinct");
    }

    /// THE case v1 was falsified on: a 99%-forest with ONE multi-parent noise
    /// edge and a detached 2-cycle. v1 demoted it to Dag/MaterializedFabric;
    /// v2 MEASURES coverage ≥ 0.8 and amortization ≥ 2 → TriePlusEscalate.
    #[test]
    fn v2_noisy_near_forest_routes_to_trie_where_v1_failed() {
        let mut edges: Vec<(u16, u16)> = (0..15).map(|k| (k, k + 1)).collect(); // 16-node chain
        edges.push((100, 8)); // noise: 8 already has parent 7 → multi-parent residue
        edges.push((200, 201)); // detached 2-cycle → uncovered
        edges.push((201, 200));

        // v1: the multi-parent noise edge makes max_in = 2 → Dag, and low
        // pressure → MaterializedFabric. The trie route never reaches it.
        let v1 = detect(9, &edges);
        assert_eq!(v1.class, ShapeClass::Cyclic); // the 2-cycle also trips v1 to Cyclic
                                                  // (either way, v1 never routes this begat-shaped predicate to a trie.)

        // v2: measure it.
        let m = detect_measured(9, &edges);
        assert!(m.coverage >= 0.8, "coverage {}", m.coverage);
        assert!(m.amortization >= 2.0, "amort {}", m.amortization);
        assert!(!m.residue_free, "has residue");
        assert_eq!(m.recommend, Representation::TriePlusEscalate);
    }
}
