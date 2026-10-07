//! The D-GSO-AFF-0 recipe law (merged in #1370), shared by the examples that
//! measure eligibility: `affordance_measurement_probe` (where it is tested)
//! and `ce64_cycle_survival_probe` (which measures eligibility across cycles
//! with the identical law instead of a copy).
//!
//! D-EPI-MIG-0: the codebook and facts now come from
//! `lance_graph_contract::epistemic_state5`; measurement projects bits 59..63
//! through a class declaration (`measure_declared`; `measure` uses the
//! probes' declared `AFF_CLASS`). Tables and rules are unchanged. Each
//! including example uses a subset, hence the `dead_code` allowance.
#![allow(dead_code)]

use causal_edge::edge::CausalEdge64;
use lance_graph_contract::band_reading::EdgeProvenance;
use lance_graph_contract::class_view::ClassId;
use lance_graph_contract::epistemic_state5::{
    Epi5Declarations, Epi5Gen, Epi5ReadError, Epi5Reading, EpistemicState5, CODEBOOK_V1,
};
use lance_graph_contract::rail_geometry::RailAxis;

// ── facts (the canonical EpistemicState5 reading) ──────────────────────────
//
// D-EPI-CANON-0: the fact vocabulary and the codebook are the contract's
// (`lance_graph_contract::epistemic_state5`), not this probe's. Re-exported so
// the probes that name them keep compiling against the ONE source.

#[allow(unused_imports)] // each including probe uses a subset
pub use lance_graph_contract::epistemic_state5::fact::{
    ASSOCIATED, CAUSES, DIRECT, INDIRECT, IND_KNOWN, IND_UNKNOWN, INTERMEDIATE_KNOWN,
    INTERMEDIATE_PRESENT, INTERMEDIATE_UNKNOWN, OBSERVED, RELATED, SUPPORTS, UP_TO_RELATED,
    UP_TO_SUPPORTS,
};

/// The codebook the law compiles against: the contract's `CODEBOOK_V1`,
/// aliased (was a probe-local table until D-EPI-CANON-0; the values are
/// identical).
pub const EPI_LAW: [Option<u32>; 32] = CODEBOOK_V1;

/// The class the affordance probes' edges belong to, declared under the
/// canonical reading. Measurement is per class: an undeclared class refuses.
pub const AFF_CLASS: ClassId = 0x0905;
pub const AFF_RAIL: RailAxis = RailAxis::Taxonomy;

/// `AFF_CLASS`'s declaration: bits 59..63 read as `EpistemicState5` V1.
pub const AFF_READING: Epi5Reading = Epi5Reading {
    generation: Epi5Gen::V1,
};

/// The probes' declaration table (a builder for callers that want one).
pub fn declarations() -> Epi5Declarations {
    let mut d = Epi5Declarations::new();
    d.declare(AFF_CLASS, AFF_RAIL, AFF_READING);
    d
}

/// Write a declared state onto an edge: all of bits 59..63, nothing else.
pub fn write_state(edge: CausalEdge64, state: EpistemicState5) -> CausalEdge64 {
    edge.with_epistemic_raw5(state.raw())
}

/// Write a V1 code that must be declared (fixture convenience).
pub fn write_code(edge: CausalEdge64, code: u8) -> CausalEdge64 {
    let state = EpistemicState5::decode(Epi5Gen::V1, code).expect("a declared V1 code");
    write_state(edge, state)
}

// ── the recipe law ──────────────────────────────────────────────────────────

/// One recipe's static obligations. `pearl` names the planes that must
/// participate (S = 0b100, P = 0b010, O = 0b001).
#[derive(Clone, Copy)]
pub struct Rule {
    pub active: bool,
    pub requires: u32,
    pub forbids: u32,
    pub pearl: u8,
}

pub const NOT_IN_LAW: Rule = Rule {
    active: false,
    requires: 0,
    forbids: 0,
    pearl: 0,
};

pub const fn rule(requires: u32, forbids: u32, pearl: u8) -> Rule {
    Rule {
        active: true,
        requires,
        forbids,
        pearl,
    }
}

pub const HYDRATE_INTERMEDIATE: u8 = 0;
pub const MECHANISM_FOLD: u8 = 1;
pub const STRATIFY: u8 = 2;
pub const ROBUSTNESS_TEST: u8 = 3;
pub const CAUSAL_IDENTIFICATION: u8 = 4;
pub const OBSERVE_FOLD: u8 = 5;
pub const COUNTERFACTUAL_PROBE: u8 = 6;
pub const fn law(stratify_forbids: u32) -> [Rule; 64] {
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
pub enum LawGen {
    V1,
    /// V2 forbids `STRATIFY` once `CAUSES` is certified.
    V2,
}

pub const LAW_V1: [Rule; 64] = law(0);
pub const LAW_V2: [Rule; 64] = law(CAUSES);

/// Compiled static affordances for one law generation.
pub struct Tables {
    /// Per EpistemicState5 code: recipes whose fact obligations hold. Invalid
    /// codes have no entry here; they refuse before lookup.
    pub state: [u64; 32],
    /// Per Pearl projection: recipes whose plane obligations hold.
    pub pearl: [u64; 8],
}

pub const fn compile(law: &[Rule; 64]) -> Tables {
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

pub static TABLES_V1: Tables = compile(&LAW_V1);
pub static TABLES_V2: Tables = compile(&LAW_V2);

impl LawGen {
    /// The codebook generation this law is compiled against.
    pub fn generation(self) -> Epi5Gen {
        Epi5Gen::V1
    }
    pub fn tables(self) -> &'static Tables {
        match self {
            LawGen::V1 => &TABLES_V1,
            LawGen::V2 => &TABLES_V2,
        }
    }
    pub fn rules(self) -> &'static [Rule; 64] {
        match self {
            LawGen::V1 => &LAW_V1,
            LawGen::V2 => &LAW_V2,
        }
    }
}

// ── the measurement ─────────────────────────────────────────────────────────

/// Why an edge cannot be measured under this law.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refusal {
    /// The caller did not assert that bits 59..63 were written under the v2
    /// layout. On a v1 row they are old `temporal` bits: `temporal = 128`
    /// sets bit 59 and would read as code 1 (`Causes`).
    Provenance(EdgeProvenance),
    /// Bits 59..63 do not form a code the declared generation declares.
    Undeclared(u8),
    /// The class has no canonical declaration, or declares another
    /// generation than the law was compiled against.
    Reading(Epi5ReadError),
}

impl From<Epi5ReadError> for Refusal {
    fn from(e: Epi5ReadError) -> Self {
        match e {
            Epi5ReadError::UnknownProvenance(p) => Refusal::Provenance(p),
            Epi5ReadError::UndeclaredCode(c) => Refusal::Undeclared(c),
            other => Refusal::Reading(other),
        }
    }
}

/// Only an asserted v2-stamped edge or a clean V3 register is readable.
pub fn admitted(provenance: EdgeProvenance) -> bool {
    provenance.trusted()
}

/// Bits 59..63 as one code, through the canonical joint accessor.
pub fn raw5(edge: CausalEdge64) -> u8 {
    edge.epistemic_raw5()
}

/// The core: a projected state × the edge's Pearl planes → eligible recipes.
/// Two lookups and one AND.
pub fn measure_state(law: LawGen, state: EpistemicState5, pearl: u8) -> u64 {
    let t = law.tables();
    t.state[state.raw() as usize] & t.pearl[pearl as usize & 0b111]
}

/// `CausalEdge64 × RecipeLaw → EligibleRecipes` for an edge of `class`:
/// the canonical projection first (declaration, provenance, generation,
/// code), then the compiled tables.
pub fn measure_declared(
    law: LawGen,
    decl: &Epi5Declarations,
    class: ClassId,
    rail: RailAxis,
    edge: CausalEdge64,
    provenance: EdgeProvenance,
) -> Result<u64, Refusal> {
    let state = decl.project_state5(class, rail, law.generation(), raw5(edge), provenance)?;
    Ok(measure_state(law, state, edge.causal_mask() as u8))
}

/// `measure_declared` for an edge of the probes' declared `AFF_CLASS`.
pub fn measure(
    law: LawGen,
    edge: CausalEdge64,
    provenance: EdgeProvenance,
) -> Result<u64, Refusal> {
    // The class's declaration is a constant: no table, no allocation (F9).
    let state = AFF_READING.project(law.generation(), raw5(edge), provenance)?;
    Ok(measure_state(law, state, edge.causal_mask() as u8))
}
