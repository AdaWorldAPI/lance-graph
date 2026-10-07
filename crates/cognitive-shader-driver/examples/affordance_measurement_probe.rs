//! D-GSO-AFF-0: recipe eligibility is a measurement over resident edge state.
//!
//! Claim under test:
//!
//! ```text
//! CausalEdge64 × RecipeLaw[0..63]  →  EligibleRecipes : u64
//! ```
//!
//! falls out of the existing substrate as two const-table lookups and one AND,
//! with no scheduler, task, capability object or heap allocation. The `u64` is
//! transient: it is measured from the edge each time and never written back.
//!
//! # The readings the law declares (law generation V1 / V2)
//!
//! - **EpistemicState5.** Bits 59..63 are read as ONE 5-bit code. ⊘
//!   D-EPI-MIG-0: the code is the canonical Cartesian product
//!   `Topology2 × Certification3`, `raw5 = topology | certification << 2`
//!   (`contract::epistemic_state5`); its facts are the union of the two
//!   factors' facts (`DIRECT`, `INDIRECT`, `INTERMEDIATE_*`,
//!   `TOPOLOGY_UNKNOWN`; `ASSOCIATED`, `RELATED`, `SUPPORTS`,
//!   `CAUSAL_CANDIDATE`, `CAUSES`). 24 codes are meaningful; 24..31 (reserved
//!   certifications) refuse. `OBSERVED` is no longer a fact: observation is
//!   evidence, not a coordinate. (#1370's hand-assigned ten-code codebook is
//!   gone.) Raw code order is still not epistemic order: topology is not
//!   ordered, so `raw_a > raw_b` across topologies means nothing.
//!   Bits 59..63 are read only under an asserted v2-stamped or V3-register
//!   provenance (the `band_reading` rule): on a v1 row they are old
//!   `temporal` bits, and v1 or unknown provenance refuses.
//! - **Pearl3.** Bits 40..42 under their shipped reading (which S/P/O planes
//!   participate). Only one recipe reads it.
//!
//! Every other field (S/P/O bytes, F/C, bits 43..45, the inference mantissa,
//! plasticity, the witness slot) is irrelevant to every recipe here, and a test
//! sweeps each of them to prove it.
//!
//! # The recipe law
//!
//! | bit | recipe | requires | forbids | Pearl planes |
//! |---|---|---|---|---|
//! | 0 | `HYDRATE_INTERMEDIATE` | `INDIRECT`, `INTERMEDIATE_UNKNOWN` | `INTERMEDIATE_KNOWN` | — |
//! | 1 | `MECHANISM_FOLD` | `INDIRECT`, `INTERMEDIATE_KNOWN` | — | — |
//! | 2 | `STRATIFY` | `ASSOCIATED` | (V2: `CAUSES`) | — |
//! | 3 | `ROBUSTNESS_TEST` | `RELATED` | — | — |
//! | 4 | `CAUSAL_IDENTIFICATION` | `SUPPORTS` | `INTERMEDIATE_UNKNOWN`, `CAUSES` | — |
//! | 5 | `OBSERVE_FOLD` | — | — | — |
//! | 6 | `COUNTERFACTUAL_PROBE` | `CAUSES` | — | S, P and O |
//! | 7..63 | not in the law | | | |
//!
//! `CAUSAL_IDENTIFICATION` only makes the attempt legal. Certifying `Causes`
//! stays the job of the certification obligations (D-GSO-7a); nothing here
//! writes the edge.
//!
//! The rules are compiled at build time per factor (`[u64; 4]` topology,
//! `[u64; 6]` certification), the per-code `[u64; 32]` is DERIVED as their AND,
//! plus `[u64; 8]` per Pearl projection. At run time the facts disappear:
//! `eligible = STATE[raw5] & PEARL[pearl3]`.
//!
//! # Layers kept apart
//!
//! 1. facts: what the code means (the law's fact table);
//! 2. static affordances: the compiled tables;
//! 3. dynamic gates: not in this probe (D-GSO-AFF-1);
//! 4. preference: `eligible & preference`. It can only clear bits.
//!
//! # Not decided here
//!
//! - The recipe set is a probe pin. ⊘ D-EPI-MIG-0: the state layout and fact
//!   vocabulary are now the contract's (`epistemic_state5`), declared per
//!   class; this probe's edges belong to `AFF_CLASS`.
//! - Dynamic gates, Shannon preference and executing a recipe are out of scope.
//!
//! Run: `cargo run -p cognitive-shader-driver --example affordance_measurement_probe`
//! Tests: `cargo test -p cognitive-shader-driver --example affordance_measurement_probe`

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use causal_edge::edge::CausalEdge64;
use causal_edge::pearl::CausalMask;
use causal_edge::PlasticityState;
use lance_graph_contract::band_reading::EdgeProvenance;

// ── allocation counter (the recipe_quartet_probe pattern) ─────────────────

/// Counts allocations on the current thread; test threads run in parallel.
struct CountingAlloc;

thread_local! {
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

// SAFETY: forwards every call to `System` unchanged; it only adds a counter.
unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = ALLOCATIONS.try_with(|c| c.set(c.get() + 1));
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let _ = ALLOCATIONS.try_with(|c| c.set(c.get() + 1));
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

/// Run `f` and return its result with the allocations it made on this thread.
fn counting<T>(f: impl FnOnce() -> T) -> (T, usize) {
    let before = ALLOCATIONS.with(Cell::get);
    let out = f();
    (out, ALLOCATIONS.with(Cell::get) - before)
}

#[path = "shared/affordance_law.rs"]
mod affordance_law;
use affordance_law::*;

/// Fact names by bit position (bit 5, formerly `OBSERVED`, is unused).
const FACT_NAMES: [&str; 12] = [
    "DIRECT",
    "INDIRECT",
    "INTERMEDIATE_PRESENT",
    "INTERMEDIATE_UNKNOWN",
    "INTERMEDIATE_KNOWN",
    "-",
    "ASSOCIATED",
    "RELATED",
    "SUPPORTS",
    "CAUSES",
    "TOPOLOGY_UNKNOWN",
    "CAUSAL_CANDIDATE",
];

const RECIPE_NAMES: [&str; 7] = [
    "HYDRATE_INTERMEDIATE",
    "MECHANISM_FOLD",
    "STRATIFY",
    "ROBUSTNESS_TEST",
    "CAUSAL_IDENTIFICATION",
    "OBSERVE_FOLD",
    "COUNTERFACTUAL_PROBE",
];

/// Preference may only remove legal recipes.
fn prefer(eligible: u64, preference: u64) -> u64 {
    eligible & preference
}

/// One deterministic ordinal from a population: the lowest set bit.
fn select(population: u64) -> Option<u8> {
    (population != 0).then(|| population.trailing_zeros() as u8)
}

// ── cold-path explanation (independent of the compiled tables) ──────────────

/// Why a recipe is or is not legal. Diagnostics only; never on the hot path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Verdict {
    Eligible,
    UntrustedProvenance,
    Unreadable,
    NotInLaw,
    MissingFact(u32),
    ForbiddenFact(u32),
    PearlPlanesMissing(u8),
}

/// Re-derive one recipe's legality from the rules, not from the tables.
fn explain(law: LawGen, edge: CausalEdge64, provenance: EdgeProvenance, recipe: u8) -> Verdict {
    if !admitted(provenance) {
        return Verdict::UntrustedProvenance;
    }
    let Some(facts) = EPI_LAW[raw5(edge) as usize] else {
        return Verdict::Unreadable;
    };
    let rl = law.rules()[recipe as usize];
    if !rl.active {
        return Verdict::NotInLaw;
    }
    let missing = rl.requires & !facts;
    if missing != 0 {
        return Verdict::MissingFact(missing);
    }
    let forbidden = rl.forbids & facts;
    if forbidden != 0 {
        return Verdict::ForbiddenFact(forbidden);
    }
    let planes = rl.pearl & !(edge.causal_mask() as u8);
    if planes != 0 {
        return Verdict::PearlPlanesMissing(planes);
    }
    Verdict::Eligible
}

// ── fixtures ────────────────────────────────────────────────────────────────

/// Measure an edge this probe stamped itself under the v2 layout.
fn stamped(law: LawGen, edge: CausalEdge64) -> Result<u64, Refusal> {
    measure(law, edge, EdgeProvenance::V2Stamped)
}

/// Explain a recipe on an edge this probe stamped itself.
fn explain_stamped(law: LawGen, edge: CausalEdge64, recipe: u8) -> Verdict {
    explain(law, edge, EdgeProvenance::V2Stamped, recipe)
}

/// Stamp a raw 5-bit code through the canonical joint writer. Raw on
/// purpose: fixtures also stamp undeclared codes to prove they refuse.
fn with_code(edge: CausalEdge64, code: u8) -> CausalEdge64 {
    edge.with_epistemic_raw5(code)
}

/// An edge with every irrelevant field non-zero.
fn busy_edge(pearl: CausalMask) -> CausalEdge64 {
    CausalEdge64::pack_v2(
        0xA5,
        0x3C,
        0x7E,
        200,
        150,
        pearl,
        0b101,
        PlasticityState::from_bits(0b011),
    )
    .with_inference_mantissa(-6)
    .with_w_slot(41)
}

fn fact_names(mask: u32) -> String {
    (0..12)
        .filter(|i| mask & (1 << i) != 0)
        .map(|i| FACT_NAMES[i])
        .collect::<Vec<_>>()
        .join("|")
}

fn main() {
    println!("EligibleRecipes per EpistemicState5 code (law V1, Pearl SPO):");
    for code in 0u8..32 {
        let Some(facts) = EPI_LAW[code as usize] else {
            continue;
        };
        let e = stamped(LawGen::V1, with_code(busy_edge(CausalMask::SPO), code)).unwrap();
        let names: Vec<&str> = (0..7)
            .filter(|r| e & (1 << r) != 0)
            .map(|r| RECIPE_NAMES[r])
            .collect();
        println!(
            "  {code:>2} {:<70} -> {:#04x} {:?}",
            fact_names(facts),
            e,
            names
        );
    }
    let unreadable = (0u8..32).filter(|c| EPI_LAW[*c as usize].is_none()).count();
    println!("  {unreadable} codes refuse (reserved certifications 6, 7)");
    let edge = with_code(busy_edge(CausalMask::SO), 10);
    let (e, allocs) = counting(|| stamped(LawGen::V1, edge));
    let eligible = e.unwrap_or(0);
    println!(
        "Indirect × IntermediateUnknown × Related under SO: {eligible:#04x}, {allocs} allocations"
    );
    for r in 0..7u8 {
        if eligible & 1 << r == 0 {
            println!(
                "  refused {:<22} {:?}",
                RECIPE_NAMES[r as usize],
                explain_stamped(LawGen::V1, edge, r)
            );
        }
    }
    let cheap = 1 << OBSERVE_FOLD | 1 << STRATIFY | 1 << COUNTERFACTUAL_PROBE;
    let preferred = prefer(eligible, cheap);
    println!(
        "  preference {cheap:#04x} -> {preferred:#04x}, first recipe {:?} (COUNTERFACTUAL_PROBE stays illegal)",
        select(preferred).map(|r| RECIPE_NAMES[r as usize])
    );
    let causes = with_code(busy_edge(CausalMask::SPO), 21);
    println!(
        "Causes code under V1 {:#04x}, under V2 {:#04x} (V2 forbids STRATIFY once CAUSES holds)",
        stamped(LawGen::V1, causes).unwrap(),
        stamped(LawGen::V2, causes).unwrap()
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL_PEARL: [CausalMask; 8] = [
        CausalMask::None,
        CausalMask::O,
        CausalMask::P,
        CausalMask::PO,
        CausalMask::S,
        CausalMask::SO,
        CausalMask::SP,
        CausalMask::SPO,
    ];
    const ACTIVE: u64 = 0b111_1111;

    fn valid_codes() -> impl Iterator<Item = u8> {
        (0u8..32).filter(|c| EPI_LAW[*c as usize].is_some())
    }

    /// F1 + F8: same bits and same law generation give the same population, on
    /// every valid code and projection, and the generation itself matters.
    #[test]
    fn f1_same_bits_and_generation_give_the_same_population() {
        let mut generations_differ = false;
        for law in [LawGen::V1, LawGen::V2] {
            for code in valid_codes() {
                for p in ALL_PEARL {
                    let e = with_code(busy_edge(p), code);
                    let a = stamped(law, e);
                    assert_eq!(a, stamped(law, e));
                    assert_eq!(a, stamped(law, CausalEdge64(e.0)), "replay from raw bits");
                    if law == LawGen::V1 && a != stamped(LawGen::V2, e) {
                        generations_differ = true;
                    }
                }
            }
        }
        assert!(generations_differ, "V2 must change at least one population");
    }

    /// F2: changing a relevant factor flips the affected bit.
    #[test]
    fn f2_a_relevant_factor_flips_the_affected_bit() {
        let unknown = stamped(LawGen::V1, with_code(busy_edge(CausalMask::SO), 10)).unwrap();
        let known = stamped(LawGen::V1, with_code(busy_edge(CausalMask::SO), 9)).unwrap();
        assert_ne!(unknown & 1 << HYDRATE_INTERMEDIATE, 0);
        assert_eq!(unknown & 1 << MECHANISM_FOLD, 0);
        assert_eq!(known & 1 << HYDRATE_INTERMEDIATE, 0);
        assert_ne!(known & 1 << MECHANISM_FOLD, 0);

        let causes_so = stamped(LawGen::V1, with_code(busy_edge(CausalMask::SO), 21)).unwrap();
        let causes_spo = stamped(LawGen::V1, with_code(busy_edge(CausalMask::SPO), 21)).unwrap();
        assert_eq!(causes_so & 1 << COUNTERFACTUAL_PROBE, 0);
        assert_ne!(causes_spo & 1 << COUNTERFACTUAL_PROBE, 0);
        assert_eq!(
            causes_so ^ causes_spo,
            1 << COUNTERFACTUAL_PROBE,
            "only that bit moves"
        );
    }

    /// F3: every field no recipe reads leaves every bit alone; the Pearl
    /// projection moves only the one recipe that reads it.
    #[test]
    fn f3_irrelevant_fields_change_no_bit() {
        for code in valid_codes() {
            let base = with_code(busy_edge(CausalMask::SPO), code);
            let want = stamped(LawGen::V1, base).unwrap();
            let same =
                |e: CausalEdge64| assert_eq!(stamped(LawGen::V1, e).unwrap(), want, "code {code}");
            for v in 0..=255u8 {
                let (mut s, mut p, mut o) = (base, base, base);
                s.set_s_idx(v);
                p.set_p_idx(v);
                o.set_o_idx(v);
                same(s);
                same(p);
                same(o);
                let fc = CausalEdge64::pack_v2(
                    0xA5,
                    0x3C,
                    0x7E,
                    v,
                    255 - v,
                    CausalMask::SPO,
                    0b101,
                    PlasticityState::from_bits(0b011),
                );
                same(with_code(
                    fc.with_inference_mantissa(-6).with_w_slot(41),
                    code,
                ));
            }
            for v in 0..8u8 {
                let mut d = base;
                d.set_direction(v);
                same(d);
                let mut pl = base;
                pl.set_plasticity(PlasticityState::from_bits(v));
                same(pl);
            }
            for m in -8i8..=7 {
                same(base.with_inference_mantissa(m));
            }
            for w in 0..64u8 {
                same(base.with_w_slot(w));
            }
            for p in ALL_PEARL {
                let moved = want ^ stamped(LawGen::V1, with_code(busy_edge(p), code)).unwrap();
                assert_eq!(
                    moved & !(1 << COUNTERFACTUAL_PROBE),
                    0,
                    "Pearl moved another recipe"
                );
            }
        }
    }

    /// F4 + F5: the compiled tables agree, bit for bit, with an independent
    /// re-derivation from the rules; every refused recipe names its failed
    /// obligation; and every active recipe both fires and stays silent.
    #[test]
    fn f4_f5_every_bit_matches_its_named_obligation() {
        for law in [LawGen::V1, LawGen::V2] {
            let (mut fired, mut silent) = (0u64, 0u64);
            for code in valid_codes() {
                for p in ALL_PEARL {
                    let e = with_code(busy_edge(p), code);
                    let pop = stamped(law, e).unwrap();
                    for r in 0..64u8 {
                        let v = explain_stamped(law, e, r);
                        if pop & 1 << r != 0 {
                            assert_eq!(v, Verdict::Eligible, "{law:?} code {code} recipe {r}");
                            fired |= 1 << r;
                        } else {
                            assert!(
                                matches!(
                                    v,
                                    Verdict::NotInLaw
                                        | Verdict::MissingFact(_)
                                        | Verdict::ForbiddenFact(_)
                                        | Verdict::PearlPlanesMissing(_)
                                ),
                                "{law:?} code {code} recipe {r}: refused without a reason ({v:?})"
                            );
                            silent |= 1 << r;
                        }
                    }
                }
            }
            assert_eq!(
                fired, ACTIVE,
                "{law:?}: every recipe in the law fires somewhere"
            );
            assert_eq!(
                silent & ACTIVE & !(1 << OBSERVE_FOLD),
                ACTIVE & !(1 << OBSERVE_FOLD)
            );
            assert_eq!(fired & !ACTIVE, 0, "recipes outside the law never fire");
        }
    }

    /// F6 + F7: preference only removes bits, never adds one, and dropping it
    /// leaves legality unchanged. Selection stays inside the legal set.
    #[test]
    fn f6_f7_preference_can_only_remove() {
        let mut lcg: u64 = 0x9E37_79B9_7F4A_7C15;
        for code in valid_codes() {
            for p in ALL_PEARL {
                let eligible = stamped(LawGen::V1, with_code(busy_edge(p), code)).unwrap();
                assert_eq!(
                    prefer(eligible, u64::MAX),
                    eligible,
                    "no preference = legality"
                );
                for _ in 0..64 {
                    lcg = lcg.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                    let preferred = prefer(eligible, lcg);
                    assert_eq!(preferred & !eligible, 0, "preference set an illegal bit");
                    if let Some(r) = select(preferred) {
                        assert_ne!(eligible & 1 << r, 0);
                    }
                }
            }
        }
    }

    /// The raw code is not a strength: topology is not ordered, so a larger
    /// code can lack what a smaller one licenses. `MECHANISM_FOLD` is legal at
    /// 5 (`IndirectKnown × Associated`) and refused at 7 (`Unknown ×
    /// Associated`), although 7 > 5. Within one topology the certification
    /// chain IS cumulative (20 `Causes` ⊇ 8 `Related`).
    #[test]
    fn raw_code_order_is_not_epistemic_order() {
        let (causes, related) = (EPI_LAW[20].unwrap(), EPI_LAW[8].unwrap());
        assert_eq!(causes & related, related);
        assert_ne!(causes, related);
        let at = |c: u8| stamped(LawGen::V1, with_code(busy_edge(CausalMask::SPO), c)).unwrap();
        assert_ne!(at(5) & 1 << MECHANISM_FOLD, 0);
        assert_eq!(at(7) & 1 << MECHANISM_FOLD, 0);
    }

    /// Reserved codes refuse instead of reading as a weak state. A topology
    /// write is a FACTOR update under the Cartesian layout: from 21
    /// (`IndirectKnown × Causes`) it yields 22 (`IndirectUnknown × Causes`), a
    /// meaningful state, and only the topology recipes move.
    #[test]
    fn reserved_codes_refuse_and_a_topology_write_is_a_factor_update() {
        let reserved = !lance_graph_contract::epistemic_state5::facts_population(0);
        assert_eq!(reserved, 0xFF00_0000, "reserved = certifications 6 and 7");
        for code in (0u8..32).filter(|c| reserved >> c & 1 == 1) {
            assert!(EPI_LAW[code as usize].is_none());
            let e = with_code(busy_edge(CausalMask::SPO), code);
            assert_eq!(stamped(LawGen::V1, e), Err(Refusal::Undeclared(code)));
            assert_eq!(
                explain_stamped(LawGen::V1, e, OBSERVE_FOLD),
                Verdict::Unreadable
            );
        }
        let causes = with_code(busy_edge(CausalMask::SPO), 21);
        let moved = causes.with_topology(causal_edge::layout::CausalTopology::from_bits_2(0b10));
        assert_eq!(raw5(moved), 22);
        let (a, b) = (
            stamped(LawGen::V1, causes).unwrap(),
            stamped(LawGen::V1, moved).unwrap(),
        );
        assert_eq!(a ^ b, 1 << HYDRATE_INTERMEDIATE | 1 << MECHANISM_FOLD);
    }

    /// Bits 59..63 are only read under an asserted v2/V3 provenance. A v1 row
    /// whose old `temporal` was 2560 has bits 61 and 63 set, which would read
    /// as code 20 (`Direct × Causes`) and make `COUNTERFACTUAL_PROBE`
    /// eligible; it refuses instead.
    #[test]
    fn v1_and_unknown_provenance_refuse() {
        let v1_row = CausalEdge64((2560u64 << 52) | (CausalMask::SPO as u64) << 40);
        assert_eq!(
            raw5(v1_row),
            20,
            "the old temporal bit lands on a declared code"
        );
        for prov in [EdgeProvenance::V1Legacy, EdgeProvenance::Unknown] {
            assert_eq!(
                measure(LawGen::V1, v1_row, prov),
                Err(Refusal::Provenance(prov))
            );
            assert_eq!(
                explain(LawGen::V1, v1_row, prov, COUNTERFACTUAL_PROBE),
                Verdict::UntrustedProvenance
            );
        }
        // Can fire: the same bits under an asserted v2 stamp do measure.
        for prov in [EdgeProvenance::V2Stamped, EdgeProvenance::V3Register] {
            let e = measure(LawGen::V1, v1_row, prov).unwrap();
            assert_ne!(e & 1 << COUNTERFACTUAL_PROBE, 0);
        }
    }

    /// F9: the measurement allocates nothing, over every valid code and
    /// projection. The output is a `u64`; there is no other object.
    #[test]
    fn f9_the_measurement_allocates_nothing() {
        let edges: [CausalEdge64; 192] = {
            let mut a = [CausalEdge64::ZERO; 192];
            let mut i = 0;
            for code in valid_codes() {
                for p in ALL_PEARL {
                    a[i] = with_code(busy_edge(p), code);
                    i += 1;
                }
            }
            assert_eq!(i, 192);
            a
        };
        let (acc, allocs) = counting(|| {
            edges.iter().fold(0u64, |acc, e| {
                acc ^ stamped(LawGen::V1, *e)
                    .unwrap()
                    .rotate_left(acc as u32 & 63)
            })
        });
        assert_eq!(allocs, 0);
        assert_ne!(acc, 0);
    }
}
