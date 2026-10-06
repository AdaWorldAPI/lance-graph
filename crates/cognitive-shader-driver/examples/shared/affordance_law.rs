//! The D-GSO-AFF-0 recipe law (merged in #1370), shared by the examples that
//! measure eligibility: `affordance_measurement_probe` (where it is tested)
//! and `ce64_cycle_survival_probe` (which measures eligibility across cycles
//! with the identical law instead of a copy).
//!
//! Moved here unchanged. Each including example uses a subset, hence the
//! `dead_code` allowance.
#![allow(dead_code)]

use causal_edge::edge::CausalEdge64;
use lance_graph_contract::band_reading::EdgeProvenance;

// ── facts (the EpistemicState5 reading) ────────────────────────────────────

pub const DIRECT: u32 = 1 << 0;
pub const INDIRECT: u32 = 1 << 1;
pub const INTERMEDIATE_PRESENT: u32 = 1 << 2;
pub const INTERMEDIATE_UNKNOWN: u32 = 1 << 3;
pub const INTERMEDIATE_KNOWN: u32 = 1 << 4;
pub const OBSERVED: u32 = 1 << 5;
pub const ASSOCIATED: u32 = 1 << 6;
pub const RELATED: u32 = 1 << 7;
pub const SUPPORTS: u32 = 1 << 8;
pub const CAUSES: u32 = 1 << 9;

pub const IND_UNKNOWN: u32 = INDIRECT | INTERMEDIATE_PRESENT | INTERMEDIATE_UNKNOWN;
pub const IND_KNOWN: u32 = INDIRECT | INTERMEDIATE_PRESENT | INTERMEDIATE_KNOWN;
pub const UP_TO_RELATED: u32 = ASSOCIATED | RELATED;
pub const UP_TO_SUPPORTS: u32 = UP_TO_RELATED | SUPPORTS;

/// EpistemicState5 law: code → facts. Ten valid codes, deliberately not ordered
/// by strength (`Causes` sits at 1, `Open` at 0, `Related` at 20, …).
pub const EPI_LAW: [Option<u32>; 32] = {
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
    /// sets bit 59 and would read as code 1 (`Causes`). Same rule as
    /// `band_reading`: v1 and unknown provenance refuse.
    Provenance(EdgeProvenance),
    /// Bits 59..63 do not form a code this law declares.
    Undeclared(u8),
}

/// Only an asserted v2-stamped edge or a clean V3 register is readable.
pub fn admitted(provenance: EdgeProvenance) -> bool {
    matches!(
        provenance,
        EdgeProvenance::V2Stamped | EdgeProvenance::V3Register
    )
}

/// Bits 59..63 as one code, through the shipped accessors.
pub fn raw5(edge: CausalEdge64) -> u8 {
    (edge.spare() << 2) | edge.truth_raw()
}

/// `CausalEdge64 × RecipeLaw → EligibleRecipes`. Two lookups and one AND.
pub fn measure(
    law: LawGen,
    edge: CausalEdge64,
    provenance: EdgeProvenance,
) -> Result<u64, Refusal> {
    if !admitted(provenance) {
        return Err(Refusal::Provenance(provenance));
    }
    let code = raw5(edge);
    if EPI_LAW[code as usize].is_none() {
        return Err(Refusal::Undeclared(code));
    }
    let t = law.tables();
    Ok(t.state[code as usize] & t.pearl[edge.causal_mask() as usize])
}
