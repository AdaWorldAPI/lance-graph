//! D-PUZZLE-0: the domain-free half of every puzzle probe.
//!
//! One edge carries a coordinate (`raw5`, CE64 bits 59..63). One question is
//! a population, a `u32` over the 32 codes (`facts_population`). Many edges
//! are answered by a fold over the edge words. The class declaration is
//! checked once per lane, never per edge.
//!
//! What a domain supplies: its claims, its law (which claims survive, which are
//! forced), and its own counters. What lives here: the propagation reading, the
//! questions, the folds and the oracle. The fence test at the bottom fails if
//! domain vocabulary appears above it: a domain needing code here would be
//! evidence against "same algebra".
//!
//! # The propagation reading (probe-local, not canon)
//!
//! | a claim at the snapshot | `Topology2 × Certification3` | raw5 |
//! |---|---|---|
//! | given | `Direct × Causes` | 20 |
//! | forced by propagation | `IndirectKnown × Causes` | 21 |
//! | true, but not yet forced | `IndirectUnknown × Causes` | 22 |
//! | any other surviving candidate | `Direct × Associated` | 4 |
//!
//! "True, not yet forced" needs a unique solution, which each domain checks.
//! Refuted claims are absent: the 5-bit state has no "refuted" coordinate.

#![allow(dead_code)]

use std::hint::black_box;
use std::time::Instant;

use causal_edge::edge::CausalEdge64;
use causal_edge::layout::EPISTEMIC_SHIFT;
use lance_graph_contract::band_reading::EdgeProvenance;
use lance_graph_contract::class_view::ClassId;
use lance_graph_contract::epistemic_state5::fact::{
    ASSOCIATED, CAUSES, DIRECT, IND_KNOWN, IND_UNKNOWN, RELATED,
};
use lance_graph_contract::epistemic_state5::{
    facts_population, Certification3, Epi5Declarations, Epi5Gen, Epi5Reading, EpistemicState5,
    Population, Topology2,
};
use lance_graph_contract::rail_geometry::RailAxis;

pub const RAIL: RailAxis = RailAxis::Taxonomy;

const fn state(t: Topology2, c: Certification3) -> EpistemicState5 {
    EpistemicState5::new(Epi5Gen::V1, t, c)
}
pub const GIVEN: EpistemicState5 = state(Topology2::Direct, Certification3::Causes);
pub const FORCED: EpistemicState5 = state(Topology2::IndirectKnown, Certification3::Causes);
pub const ENTAILED: EpistemicState5 = state(Topology2::IndirectUnknown, Certification3::Causes);
pub const CANDIDATE: EpistemicState5 = state(Topology2::Direct, Certification3::Associated);
/// Meaningful in general, never emitted by this reading.
pub const UNKNOWN_CAUSES: EpistemicState5 = state(Topology2::Unknown, Certification3::Causes);

/// The four states in emission order; `Lane::expected` is indexed alike.
pub const STATES: [EpistemicState5; 4] = [GIVEN, FORCED, ENTAILED, CANDIDATE];

/// A question: its population (fast path) and the same question asked of one
/// decoded state (oracle). Both are written from facts, never from codes.
pub struct Query {
    pub name: &'static str,
    pub population: Population,
    pub asks: fn(EpistemicState5) -> bool,
}

pub fn queries() -> [Query; 5] {
    [
        Query {
            name: "every asserted claim (any topology, Causes)",
            population: facts_population(CAUSES),
            asks: |s| s.asserts(CAUSES),
        },
        Query {
            name: "given (Direct x Causes)",
            population: facts_population(DIRECT | CAUSES),
            asks: |s| s.asserts(DIRECT | CAUSES),
        },
        Query {
            name: "forced, chain known (IndirectKnown x Causes)",
            population: facts_population(IND_KNOWN | CAUSES),
            asks: |s| s.asserts(IND_KNOWN | CAUSES),
        },
        Query {
            name: "entailed, not yet forced (IndirectUnknown x Causes)",
            population: facts_population(IND_UNKNOWN | CAUSES),
            asks: |s| s.asserts(IND_UNKNOWN | CAUSES),
        },
        Query {
            name: "live candidates (Direct, Associated, not Related)",
            population: facts_population(DIRECT | ASSOCIATED) & !facts_population(RELATED),
            asks: |s| s.asserts(DIRECT | ASSOCIATED) && !s.asserts(RELATED),
        },
    ]
}

/// Edges, the group each belongs to, and the domain's own counts per state
/// (`expected[i]` counts `STATES[i]`), kept apart from the edges so the folds
/// have something independent to agree with.
#[derive(Default)]
pub struct Lane {
    pub edges: Vec<CausalEdge64>,
    pub group_of: Vec<u32>,
    pub groups: u32,
    pub expected: [usize; 4],
}

impl Lane {
    pub fn push(&mut self, s: EpistemicState5) {
        self.edges
            .push(CausalEdge64::ZERO.with_epistemic_raw5(s.raw()));
        self.group_of.push(self.groups);
    }

    pub fn close_group(&mut self) {
        self.groups += 1;
    }

    /// The domain's count for question `q` of `queries()`: question 0 is the
    /// union of the three asserted states, 1..=4 are the states themselves.
    pub fn expected_for(&self, q: usize) -> usize {
        match q {
            0 => self.expected[0] + self.expected[1] + self.expected[2],
            q => self.expected[q - 1],
        }
    }
}

/// The declaration gate, checked once per lane: a lane whose class does not
/// declare the canonical reading is refused before any edge is read.
pub fn admit(decl: &Epi5Declarations, class: ClassId) -> Option<Epi5Reading> {
    decl.get(class, RAIL)
}

pub fn declarations(class: ClassId) -> Epi5Declarations {
    let mut d = Epi5Declarations::new();
    d.declare(class, RAIL, Epi5Reading::default());
    d
}

pub fn raw5(e: CausalEdge64) -> u32 {
    (e.0 >> EPISTEMIC_SHIFT) as u32
}

/// Fast path, one question: a shift and a mask per edge.
pub fn count_in(edges: &[CausalEdge64], population: Population) -> usize {
    edges
        .iter()
        .filter(|&&e| population >> raw5(e) & 1 == 1)
        .count()
}

/// Fast path, every question at once: one pass builds a 32-bin histogram; a
/// population's count is the sum of its bins.
pub fn histogram(edges: &[CausalEdge64]) -> [usize; 32] {
    let mut h = [0usize; 32];
    for &e in edges {
        h[raw5(e) as usize] += 1;
    }
    h
}

pub fn count_from(h: &[usize; 32], population: Population) -> usize {
    (0..32)
        .filter(|&r| population >> r & 1 == 1)
        .map(|r| h[r])
        .sum()
}

/// Oracle: decode every edge through the declaration, then ask the state.
pub fn count_decoded(
    edges: &[CausalEdge64],
    decl: &Epi5Declarations,
    class: ClassId,
    asks: fn(EpistemicState5) -> bool,
) -> usize {
    edges
        .iter()
        .filter(|e| {
            let s = decl
                .project_state5(
                    class,
                    RAIL,
                    Epi5Gen::V1,
                    e.epistemic_raw5(),
                    EdgeProvenance::V2Stamped,
                )
                .expect("every edge in a declared lane decodes");
            asks(s)
        })
        .count()
}

/// Group fold: one question per group.
pub fn per_group(lane: &Lane, population: Population) -> Vec<u32> {
    let mut out = vec![0u32; lane.groups as usize];
    for (&e, &g) in lane.edges.iter().zip(&lane.group_of) {
        out[g as usize] += population >> raw5(e) & 1;
    }
    out
}

/// Every question three ways (filter, histogram, decode) against the domain's
/// counts. Panics on the first disagreement; returns the counts.
pub fn check_three_ways(lane: &Lane, decl: &Epi5Declarations, class: ClassId) -> [usize; 5] {
    let h = histogram(&lane.edges);
    let mut out = [0usize; 5];
    for (i, q) in queries().iter().enumerate() {
        let want = lane.expected_for(i);
        assert_eq!(
            count_in(&lane.edges, q.population),
            want,
            "filter: {}",
            q.name
        );
        assert_eq!(count_from(&h, q.population), want, "histogram: {}", q.name);
        assert_eq!(
            count_decoded(&lane.edges, decl, class, q.asks),
            want,
            "decode: {}",
            q.name
        );
        out[i] = want;
    }
    out
}

pub fn median_ns_per_edge(edges: usize, mut f: impl FnMut() -> usize) -> f64 {
    let mut times: Vec<f64> = (0..7)
        .map(|_| {
            let t = Instant::now();
            black_box(f());
            t.elapsed().as_nanos() as f64 / edges as f64
        })
        .collect();
    times.sort_by(f64::total_cmp);
    times[times.len() / 2]
}

/// Print the counts and the three timed paths for one lane.
pub fn report(lane: &Lane, decl: &Epi5Declarations, class: ClassId, counts: &[usize; 5]) {
    for (q, n) in queries().iter().zip(counts) {
        println!("  {:<56} pop {:#010x}  {:>8}", q.name, q.population, n);
    }
    println!(
        "  build: debug_assertions={} avx2={} avx512f={}",
        cfg!(debug_assertions),
        cfg!(target_feature = "avx2"),
        cfg!(target_feature = "avx512f")
    );
    let n = lane.edges.len();
    let pop = facts_population(IND_UNKNOWN | CAUSES);
    let filter = median_ns_per_edge(n, || count_in(black_box(&lane.edges), pop));
    let hist = median_ns_per_edge(n, || histogram(black_box(&lane.edges))[22]);
    let decoded = median_ns_per_edge(n, || {
        count_decoded(black_box(&lane.edges), decl, class, |s| {
            s.asserts(IND_UNKNOWN | CAUSES)
        })
    });
    println!("  median of 7, ns per edge:");
    println!("    one population, filter fold        {filter:.3}");
    println!("    32-bin histogram (every question)   {hist:.3}");
    println!("    per-edge decode + asserts (oracle)  {decoded:.3}");
}

/// SplitMix64: deterministic, seedable.
pub struct Rng(pub u64);
impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    pub fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    pub fn shuffle<T>(&mut self, xs: &mut [T]) {
        for i in (1..xs.len()).rev() {
            xs.swap(i, self.below(i as u64 + 1) as usize);
        }
    }
}

#[cfg(test)]
pub mod fence {
    /// The shared half carries no domain vocabulary. Checked on the source
    /// above this module, so the list below does not match itself.
    #[test]
    fn the_shared_fold_names_no_domain() {
        let src = include_str!("population_fold.rs");
        let shared = &src[..src.find("#[cfg(test)]\npub mod fence").unwrap()];
        let lower = shared.to_lowercase();
        for banned in [
            "sudoku",
            "crossword",
            "chess",
            "digit",
            "slot",
            "cell",
            "letter",
            "grid",
            "board",
            "move",
        ] {
            assert!(!lower.contains(banned), "shared fold mentions `{banned}`");
        }
    }

    /// The four states are the stated codes, and the five questions are
    /// exactly the stated code sets.
    #[test]
    fn the_reading_and_the_questions_are_the_stated_codes() {
        use super::*;
        let raws: Vec<u8> = STATES.iter().map(|s| s.raw()).collect();
        assert_eq!(raws, [20, 21, 22, 4]);
        let pops: Vec<Population> = queries().iter().map(|q| q.population).collect();
        assert_eq!(pops, [0xF0_0000, 1 << 20, 1 << 21, 1 << 22, 1 << 4]);
    }
}
