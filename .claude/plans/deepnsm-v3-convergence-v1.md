# DeepNSM → V3 substrate convergence — v1

> **Status:** PROPOSED (doc-only). Extends `v3-convergence-wiring-v1`
> ("wire, don't invent"). Credits PR #624 (probes P0–P5) as the proven
> static baseline. Does NOT supersede any shipped decision.
>
> **Organizing frame:** DeepNSM is **not migrated onto** the V3 substrate —
> it is the **trained encoder that fills tenants V3 has already reserved.**
> The COCA gridlake `Cell` is a byte-subset of existing value tenants; the
> "migration" is a *recognition* (a `ValueSchema` reading), not a port.

---

## 0. What is already proven (do not re-derive)

PR #624 landed probes P0–P5 (`crates/lance-graph-osint/tests/`), each an
integer-exact `#[test]` that de-blackboxes one convergence claim against
shipped code:

- **P1** — deepnsm `subspace_distance_table` → quantize → palette →
  `SpoDistances::s_dist` (planner) **≡** `MatrixDistance::distance`
  (arm-discovery), byte-exact over 4096 pairs. *(node-code layer)*
- **P2/P3/P3b** — `CausalEdge64::pack_v2` round-trips S/P/O + mask + freq/conf;
  the 8 Pearl masks = 8 questions from 3 cached reads; Association vs
  Intervention rank candidates oppositely. *(edge + truth layers)*
- **P4** — Aerial+ ARM mines a rule, deterministic, `arm_to_truth_u8 →
  CausalEdge64`. *(discovery layer)*
- **P5** — `is_a` transitivity via `syllogize`, exact NAL truth. *(reasoning)*

The reframe is also prior art: `E-V3-TENANTS-ALREADY-EXIST-WIRE-DONT-INVENT`
(#626). This plan is the DeepNSM-specific application of it.

**Consequence:** the "check convergence of CAM-PQ / CausalEdge64 / SPO-NARS-2³ /
arm-discovery" question is **answered — they converge at one integer-exact
256×256 palette metric, proven.** The open work is the *memory* layer and the
*gridlake carrier landing*, below.

## 1. The four-layer stack (not four rival representations)

The representations named as "vs" are one composed stack, bottom to top:

| Layer | Object | V3 home | Status |
|---|---|---|---|
| node-code | CAM-PQ `[u8;6]` (`Heel/Branch/TwigA/TwigB/Leaf/Gamma`) + its `[[f32;256];6]` ADC-table **dual** | `HelixResidue`(6B) + facet re-read; key `HEEL\|HIP\|TWIG` (doc-asserted tile) | code/table dual proven P1; key↔tile identity **doc-only** |
| edge | `CausalEdge64` (u64: S/P/O + freq/conf + causal/dir/infer/W/lens) | `EdgeBlock`(16B, 1-byte refs) → `ValueTenant::MaterializedEdges`(4×u64) | **wired** |
| truth | SPO-NARS 2³ (8 Pearl masks over `SpoDistances` 3×256²) | 3-bit mask in `CausalEdge64` bits 40-42; freq/conf; `Meta` tenant | mask **stored**, projections **computed** (P3) |
| memory | episodic witness + AriGraph basin | `family` field (basin); `episodic_edges.rs` EW64; `witness_tombstone.rs` | **doc-only / scaffold `todo!()`** |

**Collision guards** (from the mapper sweep — pin before any code):
- "6×256" names **three** objects: CAM-PQ per-query ADC table `[[f32;256];6]`;
  `SpoDistances` 3×256² pairwise; OGAR key-tier 256×256 centroid tile. Not
  interchangeable.
- "basin" names **two**: perturbation-sim electrical/Kron basin (Laplacian
  Schur complement) vs canonical `family`. Only the latter maps to the key.
- CAM-PQ 48-bit and 6×palette256² are **not alternatives** — they are the
  *code* and its *distance-table dual*.

## 2. The recognition: the COCA `Cell` IS the value slab

`crates/deepnsm/examples/gridlake_coca_wire.rs` hand-rolls a 20-byte `Cell`
that is a byte-subset of tenants `canonical_node.rs` already carves:

| COCA `Cell` field | V3 tenant (exists today) |
|---|---|
| `helix48: [u8;6]` | `ValueTenant::HelixResidue` — 6 B, "48-bit helix place, 2× Signed360 hemisphere" |
| `campq48: [u8;6]` | canonical 6 B CAM-PQ code (the Phase-2 facet re-reads HelixResidue + 6 B CAM-PQ as one 16 B facet) |
| `count` / `sum_truth` | `ValueTenant::Meta` MetaWord `nars_f(8)+nars_c(8)` |

So "migrate the COCA landing to V3" = land it as a `ValueSchema` reading over
existing tenants, consuming the **328 B reserved headroom, zero new tenant, no
`ENVELOPE_LAYOUT_VERSION` bump**. This is task #17 (gridlake carrier / lane J)
and is the natural first step. It also retires the codec's "deterministic
stand-in" status by pointing it at the real HelixResidue + CAM-PQ contract.

## 3. Staged deliverables (ordered by wired-ness)

- **D-DNV-1 (recognition, = task #17).** Land the COCA/gridlake landing as a
  `ValueSchema` reading over `HelixResidue` + CAM-PQ facet + `Meta`. Zero new
  tenant. jc-pillar certification (ICC/ρ/α, certification-officer pattern)
  before the reading backs any claim. *Buildable now.*
- **D-DNV-2 (SPO → CausalEdge64 + 2³, deepen P1/P3).** Map deepnsm `SpoTriple`
  (36-bit S/P/O) onto `CausalEdge64` S/P/O + freq/conf → `MaterializedEdges`;
  run `nars_engine.all_projections()` over deepnsm's COCA distance matrix. The
  rung decomposition on real COCA data. *Buildable now; extends #624 P3b.*
- **D-DNV-3 (arm-discovery second leg — GATED).** ARM (rule-mine proposer) and
  deepnsm-FSM (grammar-parse proposer) already share the palette256 oracle
  (ρ=0.9973); wire both into one SpoStore. **Blocked on `ARM-JIRAK-FLOOR`**
  (D-ARM-7) — the Jirak noise floor is a hard prereq before touching a live
  `SpoStore`. Do not build the live join before its probe.
- **D-DNV-4 (memory layer — the real new work, own wave).** No episodic-witness
  `ValueTenant` exists; `witness_tombstone.rs` calcify chain is `todo!()`;
  `basin = family` is doc-only (AriGraph runtime references neither `NodeGuid`
  nor `family`). This is a genuinely-new tenant (or the witness_tombstone
  build) + waking the `family` basin field — highest risk, most doc-only.
  Its own probe + jc certification; must NOT ride D-DNV-1..3.

## 4. Explicitly out of scope

- Grammar **templates** as compiled thinking templates (StepMask) — that is
  W3, a separate wave; `StepMask` does not exist in source yet.
- Any new `ValueSchema` enum variant (#496/#500 no-new-variant guardrail).
- Superposing CAM-PQ codes (`I-VSA-IDENTITIES`: bundle identities, not content).

## 5. Cross-refs

PR #624 (P0–P5 probes), `E-V3-TENANTS-ALREADY-EXIST-WIRE-DONT-INVENT`,
`E-V3-JINA-IS-THE-FULCRUM-SUBSTRATE-MEASURED-1` (deepnsm COCA measured vs Jina),
`v3-convergence-wiring-v1`, `canonical_node.rs` (`ValueTenant`, `VALUE_TENANTS`,
`ValueSchema`), `crates/deepnsm/` (encoder/spo/parser + gridlake examples),
`crates/lance-graph-arm-discovery/` (Aerial+ leg), `nars_engine.rs`
(`all_projections`, `SpoHead`), `arigraph/markov_soa.rs` (already V3-native),
`episodic_edges.rs` + `graph/witness_tombstone.rs` (the memory seam), task #17.
