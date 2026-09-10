# Plan: `assertion_wire` — the versioned canonical LE truth DTO (`D-BBB-NARS-2`, `assertion-wire-v1`)

> **Status:** IN PR (#1223, 2026-09-10) — built on the operator's *"CE64 already
> has it globally and we need to wire it, period."*
> **Companions:** `.claude/knowledge/membrane-tiers.md` § "LE is the universal DTO
> layer" + § "coordinates of truth" (the rulings), `dacr7-band-reading-contract-v1.md`
> (the reading contract this composes), `entropy-closure-causal-ground-v1.md` §4b
> (what topology and band MEAN), board entry
> `E-LE-IS-THE-UNIVERSAL-DTO-LAYER-TYPED-SYNTAX-MEANS-A-VERSIONED-LE-SCHEMA-1`.
> **Law under build:** *A field becomes defining when changing or omitting it
> changes the proposition, not merely its presentation. Every defining epistemic
> dimension SHALL participate in the versioned canonical LE DTO; a reader lacking
> its declared lens or provenance must refuse, never project a plausible default.*
> **Boundary:** meaning crosses; machinery does not.

## §1 FROZEN DECISIONS (cite-or-VIOLATES)

| # | Frozen | Source |
|---|---|---|
| F1 | **The DTO is the existing 16-byte edge facet** `classid(4, LE u32) \| CausalEdgeV3 payload(12)`. No new byte, no new bit, no `ENVELOPE_LAYOUT_VERSION` bump, no new address type. | `causal-edge/src/edge_v3.rs` layout; D-ACR-7 F7 |
| F2 | **Schema version rides the envelope + the ABI manifest, never the bytes.** `ASSERTION_WIRE_SCHEMA = 1` pairs with `ENVELOPE_LAYOUT_VERSION = 2`; a G11 host exports it beside its endianness probe (`LgjAbiManifest.endianness`). | LE ruling ("versioned DTO schema"); `soa_envelope.rs:54`; lgj `abi.rs:384` |
| F3 | **Reading is fallible and refusing** — provenance → lens → presence, composing `band_reading::project_truth` / `project_band` unchanged. No plausible default, ever. | D-ACR-7 G3′/G4′/G5b; the law's second sentence |
| F4 | **The contract carries the wire VOCABULARY for the two defining dimensions** — `AssertionTopology` (4) and `AssertionBand` (8) — as `#[repr(u8)]` mirrors of `causal_edge::layout::{CausalTopology, ReasoningBand}`, ordinal- AND name-exact, **fused** by a cross-crate test in the planner (the only crate holding both). No `TrustTexture` mirror (the ×4 homonym debt, `TYPE_DUPLICATION_MAP.md`). | operator: "the 2 dimensions … universal"; zero-dep on both sides (`causal-edge/Cargo.toml`, `lance-graph-contract/Cargo.toml`) |
| F5 | **No arithmetic.** Nothing in the module computes a truth from truths; `rehydrate`/`syllogize`/`nars_engine` stay where they are. A consumer holding the DTO can recognize and preserve, not reason. | `D-BBB-NARS-1`; F-BBB-NARS-1 |
| F6 | **One fence entry.** Everything a G11 reader needs is reachable through `assertion_wire` (re-exports of `band_reading`'s declaration types), so the G11 allowlist grows by ONE module. | `D-BBB-NARS-1` "one scalpel cut, never the cupboard"; lgj `g11_contract_import_fence.rs` `ALLOWED` |
| F7 | **Byte positions are a documented mirror**, exactly as `band_reading.rs` already documents "byte 8 hi-2 / byte 9 lo-3"; the fuse (G3) is what makes the mirror legal (MIRROR-NEEDS-GUARD). | `data-as-config-warden`; `band_reading.rs:26` |
| F8 | **Not in this PR:** the Java-side admission (lgj `ALLOWED` + `CLAUDE.md` + `Cargo.toml` lists move together), the `Truth(…)` plan op (`D-BBB-NARS-3`), any consumer migration. | scope; repo access (lance-graph-java is not in this session's write scope) |

## §2 INPUT INVENTORY (measured 2026-09-10)

- `causal_edge::edge_v3::CausalEdgeV3` — 12-byte LE register, `to_le_bytes`/`from_le_bytes`, size const-assert; positions `[0] f, [1] c, [2] mask|dir, [3] mantissa|plasticity, [4..6] target LE, [6] anaphora, [7] TE, [8] w_slot|topology hi-2, [9] band lo-3, [10..12] reserved`. `from_v1` (tail asserted) vs `from_v1_tail_unstated` (tail zeroed). `rehydrate(s,p,o)` bit-exact back to CE64.
- `causal_edge::layout` — `CausalTopology` (`:239-252`), `ReasoningBand` (`:353-373`), `_LAYOUT_COVERAGE` (`:94`, all 64 bits once); `SPARE_SHIFT` name stale (`TD-SPARE-SHIFT-NAME-IS-STALE-1`).
- `lance_graph_contract::band_reading` — `BandReading {truth_lens, band, witness}`, `EdgeProvenance` (Unknown refuses), `project_truth(requested, raw, prov)`, `project_band(raw, prov)`, `BandReadError`; `ClassView::band_reading(class, rail)` total lookup.
- `lance_graph_contract::nars` — **audited arithmetic-free** (the D-BBB-NARS-2 step-1 audit): `InferenceType`, `QueryStrategy`, `SemiringChoice`, `default_strategy` (enum→enum), `from_mantissa` (decode), one `From` impl. No function computes a truth from truths.
- lgj `g11_contract_import_fence.rs`: `ALLOWED = ["canonical_node", "class_view", "facet", "ontology"]`, must equal the `CLAUDE.md` + `Cargo.toml` lists.
- Planner precedent: `cache::stage26_v3_parity` (cfg(test), holds both crates).

## §3 THE RESOLUTION (built)

`crates/lance-graph-contract/src/assertion_wire.rs`:

- `ASSERTION_WIRE_SCHEMA`, `ASSERTION_WIRE_BYTES = 16`, the byte-offset consts.
- `AssertionTopology` / `AssertionBand` — `from_bits_*`, `to_bits_*`, `label()`, `ALL`.
- `AssertionWire([u8; 16])`, `repr(transparent)`, align 1: `from_le_bytes` / `to_le_bytes` / `as_le_bytes` (identity on the image), `from_parts(classid, payload)`, `classid()`, `payload()`, the coordinate readers (`frequency_u8`, `confidence_u8`, `causal_mask_bits`, `direction_bits`, `inference_mantissa`, `plasticity_bits`, `target`, `w_slot`, `topology_raw`, `band_raw`).
- `read(declared, provenance) -> Result<AssertionView, BandReadError>` — the defining read (Topology lens + band Present); `read_truth_raw(declared, requested, provenance)` for Trust-lensed classes.
- `AssertionView` — classid, target, causal_mask_bits, f, c, topology, band, witness, w_slot; `==` is claim identity.
- Re-exports `band_reading::{BandPresence, BandReadError, BandReading, EdgeProvenance, TruthLens, WitnessKind}`.

`crates/lance-graph-planner/src/cache/assertion_wire_parity.rs` (cfg(test)): the fuse.

## §4 NON-GOALS

- No G11/lgj change here (F8). No `Truth(…)` opcode (D-BBB-NARS-3). No consumer migration. No `TrustTexture` mirror. No permission LOGIC on the band (`ISS-REASONING-BAND-GATES-NOTHING` stays open; the DTO carries the level, the substrate decides what it licenses).
- No rename of `SPARE_SHIFT` (own task, `TD-SPARE-SHIFT-NAME-IS-STALE-1`).

## §5 PRE-REGISTERED GATES

| gate | assertion | disable that must go red |
|---|---|---|
| G1 aliasing pair (contract) | two wires differing ONLY at `[12]` hi-2 and `[13]` lo-3 read to `!=` views with equal `(f,c)`, and stay distinct through `to_le_bytes`/`from_le_bytes`; identical wires read `==` | swap `WSLOT_TOPOLOGY_OFFSET`/`BAND_OFFSET` with a neighbour byte |
| G2 refusal (contract) | `Unknown`/`V1Legacy`/default provenance → `UnknownProvenance`; Trust-declared class → `LensMismatch`; `Absent` band → `BandAbsent`; `ZERO_FALLBACK` refuses | drop the `?` on `project_truth` / `project_band` |
| G3 fuse (planner) | over a 4×8 sweep (every topology × band, varied SPO/f/c/mask/dir/mantissa/plasticity/w_slot): every wire reader equals the `CausalEdgeV3` accessor AND the CE64 source; the defining read's labels equal `format!("{:?}", e.topology()/reasoning_band())`; `CE64 → V3 → wire → V3 → CE64` bit-exact | move any byte-offset const; reorder any vocabulary variant |
| G4 aliasing pair end-to-end (planner) | two CE64 edges equal in S,P,O,(f,c) differing in topology × band stay two claims on the wire and rehydrate to their own CE64 | as G1 |
| G5 unstated lift (planner) | `from_v1_tail_unstated` reads zeros and the contract refuses under `Unknown`; the truthful lift keeps the claim | as G2 |
| G6 width/schema | size 16, align 1, `BAND_OFFSET = 13`, `WSLOT_TOPOLOGY_OFFSET = 12`, schema 1 | — (const-asserted) |

## §6 WHAT THE NEXT BRICKS ARE

1. **lance-graph-java:** add `assertion_wire` to `ALLOWED` (+ `CLAUDE.md`, `Cargo.toml` lists, the fence's doc/code-drift check), export `ASSERTION_WIRE_SCHEMA` in `LgjAbiManifest`, and give Java a reader that consumes `AssertionView` — no arithmetic.
2. **Consumers:** replace any hand-rolled `(f,c)`+predicate sniffing with `AssertionWire::read` (Q6 of `ogar-consumer-preflight.md`).
3. `D-BBB-NARS-3`: `Truth(…)` as a `plan_eval` op returning `TruthLaneId`; the lane's registry binds kind + schema (`ASSERTION_WIRE_SCHEMA`).
4. `TD-SPARE-SHIFT-NAME-IS-STALE-1`.
