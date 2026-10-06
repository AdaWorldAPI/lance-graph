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
//! - **EpistemicState5.** Bits 59..63 are read as ONE 5-bit code,
//!   `raw5 = spare() << 2 | truth_raw()`, through the existing accessors. The
//!   law maps a code to a set of semantic facts (`DIRECT`, `INDIRECT`,
//!   `INTERMEDIATE_*`, `OBSERVED`, `ASSOCIATED`, `RELATED`, `SUPPORTS`,
//!   `CAUSES`). Only ten codes are valid; the rest refuse. The codes are
//!   deliberately not ordered by strength: `raw_a > raw_b` means nothing, and
//!   "or higher" is written into each code's fact set by the law.
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
//! The rules are compiled at build time into `[u64; 32]` (per code) and
//! `[u64; 8]` (per Pearl projection). At run time the facts disappear:
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
//! - The codebook, the fact vocabulary and the recipe set are probe pins.
//! - Bits 59..63 have shipped split readings (`truth`/`topology` on 59..60,
//!   `ReasoningBand` on 61..63, read separately by `band_reading`). A joint
//!   5-bit reading needs its own declaration there before it can be relied on;
//!   this probe declares it locally.
//! - Dynamic gates, Shannon preference and executing a recipe are out of scope.
//!
//! Run: `cargo run -p cognitive-shader-driver --example affordance_measurement_probe`
//! Tests: `cargo test -p cognitive-shader-driver --example affordance_measurement_probe`

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use causal_edge::edge::CausalEdge64;
use causal_edge::pearl::CausalMask;
use causal_edge::PlasticityState;

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

// ── facts (the EpistemicState5 reading) ────────────────────────────────────

const DIRECT: u32 = 1 << 0;
const INDIRECT: u32 = 1 << 1;
const INTERMEDIATE_PRESENT: u32 = 1 << 2;
const INTERMEDIATE_UNKNOWN: u32 = 1 << 3;
const INTERMEDIATE_KNOWN: u32 = 1 << 4;
const OBSERVED: u32 = 1 << 5;
const ASSOCIATED: u32 = 1 << 6;
const RELATED: u32 = 1 << 7;
const SUPPORTS: u32 = 1 << 8;
const CAUSES: u32 = 1 << 9;

const FACT_NAMES: [&str; 10] = [
    "DIRECT",
    "INDIRECT",
    "INTERMEDIATE_PRESENT",
    "INTERMEDIATE_UNKNOWN",
    "INTERMEDIATE_KNOWN",
    "OBSERVED",
    "ASSOCIATED",
    "RELATED",
    "SUPPORTS",
    "CAUSES",
];

const IND_UNKNOWN: u32 = INDIRECT | INTERMEDIATE_PRESENT | INTERMEDIATE_UNKNOWN;
const IND_KNOWN: u32 = INDIRECT | INTERMEDIATE_PRESENT | INTERMEDIATE_KNOWN;
const UP_TO_RELATED: u32 = ASSOCIATED | RELATED;
const UP_TO_SUPPORTS: u32 = UP_TO_RELATED | SUPPORTS;

/// EpistemicState5 law: code → facts. Ten valid codes, deliberately not ordered
/// by strength (`Causes` sits at 1, `Open` at 0, `Related` at 20, …).
const EPI_LAW: [Option<u32>; 32] = {
    let mut t = [None; 32];
    t[0] = Some(0); // Open
    t[3] = Some(DIRECT | OBSERVED);
    t[7] = Some(DIRECT | OBSERVED | ASSOCIATED);
    t[12] = Some(DIRECT | ASSOCIATED);
    t[20] = Some(DIRECT | UP_TO_RELATED);
    t[25] = Some(IND_UNKNOWN | UP_TO_RELATED);
    t[5] = Some(IND_KNOWN | UP_TO_RELATED);
    t[30] = Some(IND_KNOWN | UP_TO_SUPPORTS);
    t[1] = Some(DIRECT | UP_TO_SUPPORTS | CAUSES);
    t[17] = Some(IND_KNOWN | UP_TO_SUPPORTS | CAUSES);
    t
};

// ── the recipe law ──────────────────────────────────────────────────────────

/// One recipe's static obligations. `pearl` names the planes that must
/// participate (S = 0b100, P = 0b010, O = 0b001).
#[derive(Clone, Copy)]
struct Rule {
    active: bool,
    requires: u32,
    forbids: u32,
    pearl: u8,
}

const NOT_IN_LAW: Rule = Rule {
    active: false,
    requires: 0,
    forbids: 0,
    pearl: 0,
};

const fn rule(requires: u32, forbids: u32, pearl: u8) -> Rule {
    Rule {
        active: true,
        requires,
        forbids,
        pearl,
    }
}

const HYDRATE_INTERMEDIATE: u8 = 0;
const MECHANISM_FOLD: u8 = 1;
const STRATIFY: u8 = 2;
const ROBUSTNESS_TEST: u8 = 3;
const CAUSAL_IDENTIFICATION: u8 = 4;
const OBSERVE_FOLD: u8 = 5;
const COUNTERFACTUAL_PROBE: u8 = 6;
const RECIPE_NAMES: [&str; 7] = [
    "HYDRATE_INTERMEDIATE",
    "MECHANISM_FOLD",
    "STRATIFY",
    "ROBUSTNESS_TEST",
    "CAUSAL_IDENTIFICATION",
    "OBSERVE_FOLD",
    "COUNTERFACTUAL_PROBE",
];

const fn law(stratify_forbids: u32) -> [Rule; 64] {
    let mut t = [NOT_IN_LAW; 64];
    t[HYDRATE_INTERMEDIATE as usize] = rule(INDIRECT | INTERMEDIATE_UNKNOWN, INTERMEDIATE_KNOWN, 0);
    t[MECHANISM_FOLD as usize] = rule(INDIRECT | INTERMEDIATE_KNOWN, 0, 0);
    t[STRATIFY as usize] = rule(ASSOCIATED, stratify_forbids, 0);
    t[ROBUSTNESS_TEST as usize] = rule(RELATED, 0, 0);
    t[CAUSAL_IDENTIFICATION as usize] = rule(SUPPORTS, INTERMEDIATE_UNKNOWN | CAUSES, 0);
    t[OBSERVE_FOLD as usize] = rule(0, 0, 0);
    t[COUNTERFACTUAL_PROBE as usize] = rule(CAUSES, 0, 0b111);
    t
}

/// The recipe law in force. A generation names the whole declared reading:
/// the EpistemicState5 codebook and the recipe rules together.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LawGen {
    V1,
    /// V2 forbids `STRATIFY` once `CAUSES` is certified.
    V2,
}

const LAW_V1: [Rule; 64] = law(0);
const LAW_V2: [Rule; 64] = law(CAUSES);

/// Compiled static affordances for one law generation.
struct Tables {
    /// Per EpistemicState5 code: recipes whose fact obligations hold. Invalid
    /// codes have no entry here; they refuse before lookup.
    state: [u64; 32],
    /// Per Pearl projection: recipes whose plane obligations hold.
    pearl: [u64; 8],
}

const fn compile(law: &[Rule; 64]) -> Tables {
    let mut state = [0u64; 32];
    let mut code = 0;
    while code < 32 {
        if let Some(facts) = EPI_LAW[code] {
            let mut r = 0;
            while r < 64 {
                let rl = law[r];
                if rl.active && facts & rl.requires == rl.requires && facts & rl.forbids == 0 {
                    state[code] |= 1 << r;
                }
                r += 1;
            }
        }
        code += 1;
    }
    let mut pearl = [0u64; 8];
    let mut p = 0;
    while p < 8 {
        let mut r = 0;
        while r < 64 {
            let rl = law[r];
            if rl.active && (p as u8) & rl.pearl == rl.pearl {
                pearl[p] |= 1 << r;
            }
            r += 1;
        }
        p += 1;
    }
    Tables { state, pearl }
}

static TABLES_V1: Tables = compile(&LAW_V1);
static TABLES_V2: Tables = compile(&LAW_V2);

impl LawGen {
    fn tables(self) -> &'static Tables {
        match self {
            LawGen::V1 => &TABLES_V1,
            LawGen::V2 => &TABLES_V2,
        }
    }
    fn rules(self) -> &'static [Rule; 64] {
        match self {
            LawGen::V1 => &LAW_V1,
            LawGen::V2 => &LAW_V2,
        }
    }
}

// ── the measurement ─────────────────────────────────────────────────────────

/// The edge's bits 59..63 do not form a code this law declares.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Unreadable(u8);

/// Bits 59..63 as one code, through the shipped accessors.
fn raw5(edge: CausalEdge64) -> u8 {
    (edge.spare() << 2) | edge.truth_raw()
}

/// `CausalEdge64 × RecipeLaw → EligibleRecipes`. Two lookups and one AND.
fn measure(law: LawGen, edge: CausalEdge64) -> Result<u64, Unreadable> {
    let code = raw5(edge);
    if EPI_LAW[code as usize].is_none() {
        return Err(Unreadable(code));
    }
    let t = law.tables();
    Ok(t.state[code as usize] & t.pearl[edge.causal_mask() as usize])
}

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
    Unreadable,
    NotInLaw,
    MissingFact(u32),
    ForbiddenFact(u32),
    PearlPlanesMissing(u8),
}

/// Re-derive one recipe's legality from the rules, not from the tables.
fn explain(law: LawGen, edge: CausalEdge64, recipe: u8) -> Verdict {
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

/// Stamp an EpistemicState5 code through the shipped writers.
fn with_code(edge: CausalEdge64, code: u8) -> CausalEdge64 {
    use causal_edge::layout::TrustTexture;
    edge.with_spare(code >> 2)
        .with_truth(TrustTexture::from_bits_2(code & 0b11))
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
    (0..10)
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
        let e = measure(LawGen::V1, with_code(busy_edge(CausalMask::SPO), code)).unwrap();
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
    println!("  {unreadable} codes refuse (not declared by the law)");
    let edge = with_code(busy_edge(CausalMask::SO), 25);
    let (e, allocs) = counting(|| measure(LawGen::V1, edge));
    let eligible = e.unwrap_or(0);
    println!(
        "Indirect × IntermediateUnknown × Related under SO: {eligible:#04x}, {allocs} allocations"
    );
    for r in 0..7u8 {
        if eligible & 1 << r == 0 {
            println!(
                "  refused {:<22} {:?}",
                RECIPE_NAMES[r as usize],
                explain(LawGen::V1, edge, r)
            );
        }
    }
    let cheap = 1 << OBSERVE_FOLD | 1 << STRATIFY | 1 << COUNTERFACTUAL_PROBE;
    let preferred = prefer(eligible, cheap);
    println!(
        "  preference {cheap:#04x} -> {preferred:#04x}, first recipe {:?} (COUNTERFACTUAL_PROBE stays illegal)",
        select(preferred).map(|r| RECIPE_NAMES[r as usize])
    );
    let causes = with_code(busy_edge(CausalMask::SPO), 17);
    println!(
        "Causes code under V1 {:#04x}, under V2 {:#04x} (V2 forbids STRATIFY once CAUSES holds)",
        measure(LawGen::V1, causes).unwrap(),
        measure(LawGen::V2, causes).unwrap()
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
                    let a = measure(law, e);
                    assert_eq!(a, measure(law, e));
                    assert_eq!(a, measure(law, CausalEdge64(e.0)), "replay from raw bits");
                    if law == LawGen::V1 && a != measure(LawGen::V2, e) {
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
        let unknown = measure(LawGen::V1, with_code(busy_edge(CausalMask::SO), 25)).unwrap();
        let known = measure(LawGen::V1, with_code(busy_edge(CausalMask::SO), 5)).unwrap();
        assert_ne!(unknown & 1 << HYDRATE_INTERMEDIATE, 0);
        assert_eq!(unknown & 1 << MECHANISM_FOLD, 0);
        assert_eq!(known & 1 << HYDRATE_INTERMEDIATE, 0);
        assert_ne!(known & 1 << MECHANISM_FOLD, 0);

        let causes_so = measure(LawGen::V1, with_code(busy_edge(CausalMask::SO), 17)).unwrap();
        let causes_spo = measure(LawGen::V1, with_code(busy_edge(CausalMask::SPO), 17)).unwrap();
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
            let want = measure(LawGen::V1, base).unwrap();
            let same =
                |e: CausalEdge64| assert_eq!(measure(LawGen::V1, e).unwrap(), want, "code {code}");
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
                let moved = want ^ measure(LawGen::V1, with_code(busy_edge(p), code)).unwrap();
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
                    let pop = measure(law, e).unwrap();
                    for r in 0..64u8 {
                        let v = explain(law, e, r);
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
                let eligible = measure(LawGen::V1, with_code(busy_edge(p), code)).unwrap();
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

    /// The raw code is not a strength: code 1 (`Causes`) carries strictly more
    /// facts than code 20 (`Related`), so comparing codes would admit wrongly.
    #[test]
    fn raw_code_order_is_not_epistemic_order() {
        let (causes, related) = (EPI_LAW[1].unwrap(), EPI_LAW[20].unwrap());
        assert_eq!(causes & related, related);
        assert_ne!(causes, related);
        // `ROBUSTNESS_TEST` is legal at code 1 (Causes) and refused at code 12
        // (Associated only), although 12 > 1.
        let at = |c: u8| measure(LawGen::V1, with_code(busy_edge(CausalMask::SPO), c)).unwrap();
        assert_ne!(at(1) & 1 << ROBUSTNESS_TEST, 0);
        assert_eq!(at(12) & 1 << ROBUSTNESS_TEST, 0);
    }

    /// Undeclared codes refuse instead of reading as a weak state, including a
    /// code produced by a writer that touches only half of bits 59..63.
    #[test]
    fn undeclared_codes_refuse_including_half_field_writes() {
        for code in (0u8..32).filter(|c| EPI_LAW[*c as usize].is_none()) {
            let e = with_code(busy_edge(CausalMask::SPO), code);
            assert_eq!(measure(LawGen::V1, e), Err(Unreadable(code)));
            assert_eq!(explain(LawGen::V1, e, OBSERVE_FOLD), Verdict::Unreadable);
        }
        // 17 = 0b100_01; rewriting only bits 59..60 through the shipped
        // topology writer leaves bits 61..63 alone and yields 0b100_10 = 18.
        let causes = with_code(busy_edge(CausalMask::SPO), 17);
        let half = causes.with_topology(causal_edge::layout::CausalTopology::from_bits_2(0b10));
        assert_eq!(raw5(half), 18);
        assert_eq!(measure(LawGen::V1, half), Err(Unreadable(18)));
    }

    /// F9: the measurement allocates nothing, over every valid code and
    /// projection. The output is a `u64`; there is no other object.
    #[test]
    fn f9_the_measurement_allocates_nothing() {
        let edges: [CausalEdge64; 80] = {
            let mut a = [CausalEdge64::ZERO; 80];
            let mut i = 0;
            for code in valid_codes() {
                for p in ALL_PEARL {
                    a[i] = with_code(busy_edge(p), code);
                    i += 1;
                }
            }
            assert_eq!(i, 80);
            a
        };
        let (acc, allocs) = counting(|| {
            edges.iter().fold(0u64, |acc, e| {
                acc ^ measure(LawGen::V1, *e)
                    .unwrap()
                    .rotate_left(acc as u32 & 63)
            })
        });
        assert_eq!(allocs, 0);
        assert_ne!(acc, 0);
    }
}
