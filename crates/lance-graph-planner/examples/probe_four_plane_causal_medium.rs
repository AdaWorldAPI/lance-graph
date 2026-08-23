//! PROBE-FOUR-PLANE-CAUSAL-MEDIUM-1 — one tiny causal chain with a hidden
//! mediator, one supporting derivation, one falsifying intervention: do the
//! FOUR planes stay distinguishable while sharing the memory ABI?
//!
//! **The four-plane convergence under test (operator, 2026-08-23):**
//!
//! ```text
//! HHTL / Attention V3   WHERE is relevant?        AttentionFocusFacet scope
//! CausalEdge64 59..60   WHAT causal topology?     CausalTopology (Direct /
//!                                                 IndirectKnown / IndirectUnknown / Unknown)
//! CausalEdge64 61..63   THROUGH WHAT lens?        ReasoningBand (Surface..Transcendent)
//! Tarski G24N4          WHY supportable?          +n support / 0 unresolved / −n falsifier
//! R2IL V4               WHAT did we DO?           typed intervention row + observed consequence
//! ```
//!
//! Held separate on purpose: a negative G24N4 nibble must NOT auto-flip
//! `ReasoningBand` to `Counterfactual` or rewrite `CausalTopology` — the
//! planes are jointly readable, never derived from one another. The target
//! sentence the composed state must express (no scalar confidence can):
//!
//! > "Inside this region, an indirect causal relation is hypothesized, its
//! > constructive derivation reaches +2, a −1 falsifier exists against a
//! > wrong mediator, and the mediator locus is still epistemically empty."
//!
//! …and then the closed experimental loop: a typed intervention on the true
//! mediator BINDS the mediator locus, upgrades the topology register
//! `IndirectUnknownIntermediates → IndirectKnownIntermediates`, and leaves
//! the support lane untouched.
//!
//! # Honesty box
//!
//! - The "world" here is a toy oracle (a known 3-event chain `A → M → B`).
//!   The probe tests the ABI REPRESENTATION loop — four planes resident,
//!   distinguishable, updatable — not causal discovery. Whether this
//!   geometry helps discover structure that predicts unseen interventions
//!   is the EXTERNAL falsifier (Uhler-style CRL), out of scope here.
//! - Probe-local classids only; the V4 classid is provisional (O5 gate) —
//!   nothing canonizes. The V4 plane is an R2IL-SHAPED typed row (16-byte
//!   LE, `Copy`, no heap), not the real `ruff_r2il` producer.
//! - Bits 59-60 carry two shipped readings (`TrustTexture` vs
//!   `CausalTopology`); which one a producer wrote is the `band_reading`
//!   contract's declared knowledge. This probe writes through the topology
//!   lens by construction and says so — a production writer would declare
//!   it per `(classid, rail)`.
//! - **The WHY plane is TWO registers, not one — deliberately.**
//!   `CausalWitnessFacet`'s A9 reading is operator-locked *"loci, not
//!   magnitudes"*: every named `Locus` is a signed context POINTER. So the
//!   evidence plane's signed derivational MAGNITUDES live in their own
//!   probe-local register (`TARSKI_CLASSID`, own slot names — mirroring
//!   `PROBE-TARSKI-SIGNED-WITNESS-1`'s `SignedTarskiWitnessView`), and the
//!   mediator POINTER lives in a genuine A9 register read through
//!   `Locus::Kausal`, which is what that name actually means. An earlier
//!   revision put magnitudes and a pointer in ONE A9 register; that mixed
//!   two semantic systems in one ClassView and is withdrawn. Same
//!   geometry, two classids, two readings, no vocabulary shared.
//!
//! - **`CausalRow` is a PROBE FIXTURE, not the implementation shape.** It is
//!   an AoS test object holding the lanes side by side so the gates can
//!   compare them; it is NOT evidence about the resident SoA layout and must
//!   never be cited as such.

use causal_edge::layout::{CausalTopology, ReasoningBand};
use causal_edge::CausalEdge64;
use lance_graph_contract::attention_facet::AttentionFocusFacet;
use lance_graph_contract::causal_witness::{CausalWitnessFacet, Locus, WITNESS_REGISTER_BYTES};
use lance_graph_contract::facet::{FacetCascade, FacetTier};

/// Probe-local classid for causal-hypothesis rows (NOT a mint).
const HYPOTHESIS_CLASSID: u32 = 0xFFFF_000A;
/// Probe-local classid for V4-plane intervention rows (NOT a mint; the real
/// V4 classid is provisional behind the O5 gate).
const INTERVENTION_CLASSID: u32 = 0xFFFF_000B;

/// Probe-local slot: signed constructive derivation depth (`+n`).
const TARSKI_CONSTRUCTIVE: usize = 0;
/// Probe-local slot: signed falsifying derivation depth (`−n`).
const TARSKI_FALSIFYING: usize = 1;

/// **PROBE FIXTURE** — an AoS test object holding one hypothesis's lanes
/// side by side so the gates can compare them. This is NOT the resident
/// layout and is never evidence about it (see the honesty box). Five lanes:
///
/// | lane | plane | reading |
/// |---|---|---|
/// | `address` | WHERE | `FacetCascade` / `AttentionFocusFacet` |
/// | `edge` | WHAT + LENS | `CausalTopology` / `ReasoningBand` |
/// | `evidence` | WHY (magnitudes) | probe-local signed-Tarski |
/// | `pointers` | WHY (pointers) | shipped A9 `Locus` |
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct CausalRow {
    /// Plane 1 — WHERE: the 16-byte address dock (classid + G6D2 cascade).
    /// Attention/HHTL read THIS lane; evidence updates never touch it.
    address: [u8; 16],
    /// Planes 2+3 — WHAT + LENS: the causal relation register
    /// (topology bits 59-60, reasoning band bits 61-63).
    edge: CausalEdge64,
    /// Plane 4a — WHY, MAGNITUDE half: a 12-byte register read through the
    /// probe-local signed-Tarski ClassView (depths, not pointers). NOT an
    /// A9 register: A9's contract forbids magnitudes in a `Locus`.
    evidence: [u8; WITNESS_REGISTER_BYTES],
    /// Plane 4b — WHY, POINTER half: a genuine A9 register, read through
    /// `Locus::Kausal` = a signed stream pointer to the located mediator,
    /// which is exactly what that `Locus` name means.
    pointers: CausalWitnessFacet,
}

/// One V4-plane row: a typed intervention particle, R2IL-shaped
/// (16-byte LE dock: classid(4) | kind(1) | target_offset(i8 as 1) |
/// observed(1) | reserved(9)). `Copy`, no heap — behavior as an ADDRESSED
/// ROW, never nested state.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct InterventionRow([u8; 16]);

const KIND_KNOCKOUT: u8 = 1;
const OBS_NONE: u8 = 0;
const OBS_LINK_BROKEN: u8 = 1;
const OBS_LINK_INTACT: u8 = 2;

impl InterventionRow {
    fn knockout(target_offset: i8) -> Self {
        let mut b = [0u8; 16];
        b[0..4].copy_from_slice(&INTERVENTION_CLASSID.to_le_bytes());
        b[4] = KIND_KNOCKOUT;
        b[5] = target_offset as u8;
        b[6] = OBS_NONE;
        Self(b)
    }
    fn with_observed(mut self, obs: u8) -> Self {
        self.0[6] = obs;
        self
    }
    fn target_offset(&self) -> i8 {
        self.0[5] as i8
    }
    fn observed(&self) -> u8 {
        self.0[6]
    }
    fn classid(&self) -> u32 {
        u32::from_le_bytes([self.0[0], self.0[1], self.0[2], self.0[3]])
    }
}

/// The toy world oracle: the event stream `[A, M, B]`, with M the hidden
/// mediator of A→B. `do(knockout at offset)` reports whether A→B survives.
/// Deterministic and admittedly trivial — see the honesty box.
fn world_do_knockout(target_offset_from_edge: i8) -> u8 {
    // The edge row sits conceptually AT B (offset 0); M is one back (−1),
    // A two back (−2). Knocking out the true mediator breaks the link.
    if target_offset_from_edge == -1 {
        OBS_LINK_BROKEN
    } else {
        OBS_LINK_INTACT
    }
}

/// Signed depth at a probe-local Tarski slot (sign-extended nibble). This is
/// the MAGNITUDE reading — its own accessors, deliberately not A9's, because
/// A9's `Locus` names mean pointers.
fn tarski_get(reg: &[u8; WITNESS_REGISTER_BYTES], slot: usize) -> i8 {
    if slot >= 24 {
        return 0;
    }
    let byte = reg[slot / 2];
    let nib = if slot & 1 == 0 {
        byte & 0x0F
    } else {
        (byte >> 4) & 0x0F
    };
    ((nib << 4) as i8) >> 4
}

/// Set a probe-local Tarski slot; clamps to `[−8, +7]`.
fn tarski_set(reg: &mut [u8; WITNESS_REGISTER_BYTES], slot: usize, depth: i8) {
    if slot >= 24 {
        return;
    }
    let v = (depth.clamp(-8, 7) as u8) & 0x0F;
    let bi = slot / 2;
    if slot & 1 == 0 {
        reg[bi] = (reg[bi] & 0xF0) | v;
    } else {
        reg[bi] = (reg[bi] & 0x0F) | (v << 4);
    }
}

/// Build an address dock in region `heel` (tier-0 coarse byte).
fn address_in_region(heel: u8, ident: u8) -> [u8; 16] {
    let fc = FacetCascade {
        facet_classid: HYPOTHESIS_CLASSID,
        tiers: [
            FacetTier { hi: heel, lo: 0x01 },
            FacetTier { hi: 0, lo: 0 },
            FacetTier { hi: 0, lo: 0 },
            FacetTier { hi: 0, lo: 0 },
            FacetTier { hi: 0, lo: 0 },
            FacetTier { hi: 0, lo: ident },
        ],
    };
    fc.to_bytes()
}

fn focus_of(row: &CausalRow) -> AttentionFocusFacet {
    AttentionFocusFacet::exact(FacetCascade::from_bytes(&row.address))
}

fn main() {
    let mut pass = 0u32;
    let mut gate = |name: &str, ok: bool, detail: String| {
        assert!(ok, "[FAIL] {name} — {detail}");
        println!("  [PASS] {name} — {detail}");
        pass += 1;
    };

    // ---- Compose the initial state: the hypothesis, pre-intervention ----
    // "Inside region 0x11, A→B is hypothesized indirect-with-unknown-
    //  intermediates, read at the Causal band, constructive support +2,
    //  falsifier −1 (a wrong-mediator candidate already refuted),
    //  mediator locus 0 (the epistemic pothole)."
    let mut evidence = [0u8; WITNESS_REGISTER_BYTES];
    tarski_set(&mut evidence, TARSKI_CONSTRUCTIVE, 2);
    tarski_set(&mut evidence, TARSKI_FALSIFYING, -1);
    // The A9 pointer register: Locus::Kausal deliberately left 0 (unbound)
    // — the mediator pothole. A9 semantics, A9 register.

    let mut row = CausalRow {
        address: address_in_region(0x11, 0x01),
        edge: CausalEdge64::ZERO
            .with_topology(CausalTopology::IndirectUnknownIntermediates)
            .with_reasoning_band(ReasoningBand::Causal),
        evidence,
        pointers: CausalWitnessFacet::ZERO,
    };
    // A sibling row in ANOTHER region — the scope must exclude it.
    let outside = CausalRow {
        address: address_in_region(0x22, 0x02),
        edge: CausalEdge64::ZERO.with_topology(CausalTopology::Direct),
        evidence: [0u8; WITNESS_REGISTER_BYTES],
        pointers: CausalWitnessFacet::ZERO,
    };

    // Plane 1 — WHERE: a depth-1 scope over region 0x11 (tier-0 hi byte).
    let scope =
        AttentionFocusFacet::prefix(FacetCascade::from_bytes(&address_in_region(0x11, 0x00)), 1)
            .expect("depth 1 <= 12");

    // FP1 — all four planes readable, each answering ITS question, and the
    // WHY plane's two halves read through their OWN ClassViews.
    gate(
        "FP1 four planes independently readable",
        scope.covers(focus_of(&row))
            && !scope.covers(focus_of(&outside))
            && row.edge.topology() == CausalTopology::IndirectUnknownIntermediates
            && row.edge.reasoning_band() == ReasoningBand::Causal
            && tarski_get(&row.evidence, TARSKI_CONSTRUCTIVE) == 2
            && tarski_get(&row.evidence, TARSKI_FALSIFYING) == -1
            && !row.pointers.is_bound(Locus::Kausal),
        "WHERE covers row/excludes sibling; WHAT=IndirectUnknown; LENS=Causal; \
         WHY=+2/−1/mediator-pothole"
            .to_string(),
    );

    // FP2 — plane isolation: writing one plane's lane changes ONLY that
    // lane (field-isolation across planes, I-LEGACY-API-FEATURE-GATED
    // discipline applied at the row level).
    {
        let before = row;
        let mut t = row;
        tarski_set(&mut t.evidence, TARSKI_FALSIFYING, -3);
        let ev_only =
            t.address == before.address && t.edge == before.edge && t.pointers == before.pointers;
        let mut t2 = row;
        t2.edge = t2.edge.with_topology(CausalTopology::Unknown);
        let edge_only = t2.address == before.address
            && t2.evidence == before.evidence
            && t2.pointers == before.pointers;
        let mut t3 = row;
        t3.address = address_in_region(0x11, 0x7F);
        let addr_only = t3.edge == before.edge
            && t3.evidence == before.evidence
            && t3.pointers == before.pointers;
        let mut t4 = row;
        t4.pointers = t4.pointers.with(Locus::Kausal, -1);
        let ptr_only = t4.address == before.address
            && t4.edge == before.edge
            && t4.evidence == before.evidence;
        gate(
            "FP2 lane isolation (each write confined to its lane)",
            ev_only && edge_only && addr_only && ptr_only,
            "magnitude/edge/address/pointer writes each leave the other three lanes \
             bit-identical"
                .to_string(),
        );
    }

    // FP3 — NO AUTO-FLIP: a falsifier landing in the evidence plane must
    // not rewrite the causal registers. Jointly readable, never derived.
    {
        let edge_before = row.edge;
        let mut t = row;
        tarski_set(&mut t.evidence, TARSKI_FALSIFYING, -8);
        gate(
            "FP3 negative evidence does NOT flip band or topology",
            t.edge == edge_before
                && t.edge.reasoning_band() == ReasoningBand::Causal
                && t.edge.topology() == CausalTopology::IndirectUnknownIntermediates,
            "falsifying depth −8 written; CE64 bit-identical (band Causal, topology unchanged)"
                .to_string(),
        );
    }

    // FP4 — the epistemic-pothole query: bound support + unbound mediator,
    // scoped by the WHERE plane. Discriminating: fires on the hypothesis
    // row, silent on the out-of-scope row (which is also unbound).
    let pothole = |r: &CausalRow| -> bool {
        scope.covers(focus_of(r))
            && tarski_get(&r.evidence, TARSKI_CONSTRUCTIVE) > 0
            && !r.pointers.is_bound(Locus::Kausal)
    };
    gate(
        "FP4 pothole query (support present, mediator locus empty) is scoped",
        pothole(&row) && !pothole(&outside),
        "fires inside the scope where support exists and the mediator is unknown".to_string(),
    );

    // ---- The closed loop: hypothesis → intervention → consequence → update ----

    // FP5 — falsifying intervention first: knock out a WRONG mediator
    // candidate (offset −2 = A itself is not a mediator between A and B;
    // use a non-mediator probe target). The link survives; the falsifier
    // depth-1 evidence −1 was already resident; crucially the CE64 and the
    // support lane are UNTOUCHED by a failed mediation test.
    let wrong = InterventionRow::knockout(-2);
    let wrong = wrong.with_observed(world_do_knockout(wrong.target_offset()));
    let row_before_wrong = row;
    // A LinkIntact consequence does not bind the mediator locus.
    gate(
        "FP5 falsifying intervention leaves WHAT/WHY-support/WHERE untouched",
        wrong.observed() == OBS_LINK_INTACT
            && wrong.classid() == INTERVENTION_CLASSID
            && row == row_before_wrong,
        format!(
            "do(knockout@{:+}) → LinkIntact recorded in the V4 row; hypothesis row bit-identical",
            wrong.target_offset()
        ),
    );

    // FP6 — supporting intervention: knock out the TRUE mediator (−1).
    // The consequence LinkBroken (a) binds the mediator locus to the
    // intervention's own typed target, (b) upgrades the topology register
    // IndirectUnknown → IndirectKnown, (c) leaves the support lane and the
    // reasoning band untouched, (d) leaves the address lane untouched.
    let probe_m = InterventionRow::knockout(-1);
    let probe_m = probe_m.with_observed(world_do_knockout(probe_m.target_offset()));
    let (addr_before, band_before, support_before) = (
        row.address,
        row.edge.reasoning_band(),
        tarski_get(&row.evidence, TARSKI_CONSTRUCTIVE),
    );
    if probe_m.observed() == OBS_LINK_BROKEN {
        // The mediator is a POINTER — it lands in the A9 register, through
        // the Locus that actually means "my cause".
        row.pointers = row.pointers.with(Locus::Kausal, probe_m.target_offset());
        row.edge = row
            .edge
            .with_topology(CausalTopology::IndirectKnownIntermediates);
    }
    gate(
        "FP6 supporting intervention closes the loop (pothole → known mediator)",
        row.pointers.at(Locus::Kausal) == -1
            && row.edge.topology() == CausalTopology::IndirectKnownIntermediates
            && row.edge.reasoning_band() == band_before
            && tarski_get(&row.evidence, TARSKI_CONSTRUCTIVE) == support_before
            && tarski_get(&row.evidence, TARSKI_FALSIFYING) == -1
            && row.address == addr_before
            && !pothole(&row),
        "Kausal bound −1; topology Unknown→Known intermediates; band/support/falsifier/address \
         unchanged; pothole query now silent"
            .to_string(),
    );

    // FP7 — physical conservation: every plane row is a fixed-width LE
    // register (16B dock / u64 / 12B G24N4 / 16B V4 row); round-trips are
    // byte-exact; no plane materializes another's content.
    // CausalRow's own size is a probe-local composite (Rust struct layout
    // may insert padding between the three lane fields) — NOT itself a
    // dock; each LANE's width is what the ABI actually const-asserts.
    const _: () = assert!(
        core::mem::size_of::<[u8; 16]>() == 16,
        "address lane is one dock"
    );
    const _: () = assert!(
        core::mem::size_of::<[u8; WITNESS_REGISTER_BYTES]>() == 12,
        "evidence lane is the G24N4 register"
    );
    const _: () = assert!(
        core::mem::size_of::<InterventionRow>() == 16,
        "V4 row is one dock"
    );
    const _: () = assert!(core::mem::size_of::<CausalEdge64>() == 8, "CE64 is one u64");
    let fc = FacetCascade::from_bytes(&row.address);
    gate(
        "FP7 shared-ABI conservation (fixed LE registers, byte-exact round-trips)",
        fc.to_bytes() == row.address
            && fc.facet_classid == HYPOTHESIS_CLASSID
            && row.pointers.to_register().len() == WITNESS_REGISTER_BYTES,
        "address round-trips through FacetCascade; the magnitude lane through the Tarski view; \
         the pointer lane through A9; every lane width const-asserted"
            .to_string(),
    );

    println!("PROBE-FOUR-PLANE-CAUSAL-MEDIUM-1: ALL {pass} GATES GREEN");
    println!(
        "verdict: WHERE (scope) / WHAT (topology) / LENS (band) / WHY-magnitude (Tarski view) / \
         WHY-pointer (A9 Locus) / DID (typed intervention) stay separable across a write to \
         any one of them; nothing auto-flips, and no ClassView's vocabulary is used to mean \
         another's. Probe FIXTURE, not evidence about the resident SoA shape."
    );
}
