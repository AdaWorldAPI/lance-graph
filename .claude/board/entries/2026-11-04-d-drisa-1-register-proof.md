# D-DRISA-1: a sealed DR frame works as a local register file over borrowed SoA, once the SoA states its layout

**Status:** MEASURED-LAB (local, one process). Owner: this session (EXO-153 stage 1). Probe: `crates/drisa-register-probe/` (`src/lib.rs`, `tests/integration/register.rs`, `examples/drisa_bench.rs`), excluded from the workspace. Architecture: #1953 (D-DRISA-0, a plan written by a GPT session, adopted after review). No production change, no transport, no new public type.

## Census: what the register contract can be built from

Read at source on `main` `05f03abb4`.

| Piece | Where | Status |
|---|---|---|
| Sealed frame, compared by issue (not by value) | `dt64::dr::Frame`, `FrameStore::seal/check` (`dr.rs:404`, `:445`) | EXISTING |
| Versioned tile maps, old frames replay their layout | `MapTable::advance/map/map_at` (`dr.rs:871`, `:913`, `:921`) | EXISTING |
| Counted global → local routing; per-tile digest | `LocalMap::local` counts lookups (`:605`), `TileMap::digest` (`:780`) | EXISTING |
| Locate one id across tiles | `MapTable::locate` reads EVERY listed tile (`:955`) | EXISTING; O(tiles) per id by design (ambiguity check) |
| Focus / Scope / Guards | `dr.rs:496`, `:503`, `:509` | EXISTING; `Focus` is a tile revision, not a lease |
| Map vs value lifetime | `dt64::ValueDemand::{MapOnly, Values}` (`lib.rs:189`) | EXISTING, no executor reads it |
| Borrowed operands | `mask-risc` `Planes { masks: &[&[u64]], lanes: &[LaneRef] }`, `Operand::{Plane, Scratch}` (`ir.rs:7`, `:73`) | EXISTING |
| Execution | `mask-risc::execute` / `execute_into` (`exec.rs:763`, `:1343`) over `ndarray::simd` | WIRED |
| Stale scratch refused | `ExecError::ScratchReadBeforeWrite` (`value.rs:128`) | EXISTING; the plan's "StaleSlot" is the R2IL probe's `FoldError::StaleSlot`, not mask-risc |
| A statement of which frame/tile digests a SoA's rows are laid out under | none found in the closed space above | **MISSING** (the lab `Soa` carries it) |
| A global id → tile index | none; `locate` scans tiles | **MISSING**, owner decision (EXO-46) |
| Source-location receipt / lease host | only excluded lab crates (#1951 census) | **BLOCKED** EXO-46/89/151 |
| Cross-call result register lifetime | no owner | **BLOCKED**, owner decision |
| DR → Lance `_rowid` authority | none | **BLOCKED** D4 / S-INGRESS |

## The proof

`RegisterFile` is a descriptor: the frame, the tile digests it was bound under, a scope, the demanded field positions, a value witness when values were demanded, and one map-only register (the identity selection, routed from global ids at bind). It holds no reference into the SoA and no value. At execution it re-checks every operand and hands `mask-risc` planes that borrow the SoA's own slices.

The eight properties of the mission, each with the test that checks it:

| # | Property | Result |
|---|---|---|
| 1 | Registers name coordinates, not values | `execution_borrows_the_soa_and_copies_no_value`: executed lanes and the presence plane have the SoA's own addresses; the file owns only the selection bits (8 KiB for 65,536 rows) |
| 2 | Methods name typed operands, not rows | `MethodOp` names field positions and constants; compiled once to an unchanged `mask-risc` `Program` |
| 3 | No second ALU | every answer comes from `mask-risc::execute` |
| 4 | One binding serves many methods without rerouting | `many_methods_reuse_one_binding_without_rerouting`: 256 rounds × 4 methods, 0 lookups after bind |
| 5 | Lifetimes tied to frame, layout, scope, source, values | each check has its own refusal and its own red disable (below) |
| 6 | A value write invalidates value-scoped registers, not the map | `a_value_write_invalidates_values_but_not_the_map`: COUNT stays valid and correct; value methods and the derived predicate register refuse; re-taking the value witness routes nothing; the stale derived register still refuses until recomputed |
| 7 | An invalid register refuses, never answers wrong or empty | `Refusal` has a variant per cause; the oracle tests show the lax versions answer wrong |
| 8 | No mandatory payload gather | no lane is copied; the only owned buffers are the selection and the optional derived register (one bit per row each) |

All methods equal an independent oracle (global id → value, no map or position) on DR64 (1 tile), DR256 (1 tile) and DR256 × 256 tiles (65,536 identities, local order permuted per tile, 10% selection that includes absent identities). 13/13 tests.

**Disable runs** (commit first, patch with an asserted anchor, `git checkout`): frame check off → 1 red; layout digest compare off → 2 red; value witness off (lanes) → 1 red; value witness off (derived register) → 1 red; scope check off → 1 red; duplicate-identity check off → 1 red. Restored: 13/13.

## Cost (µs per query, round robin over 4 methods, 3 processes)

| fixture | A direct planes | C one binding, revalidated per query | allocations per query (C) |
|---|---|---|---|
| DR64, 1 tile | 0.09–0.20 | 0.14–0.25 | 0 |
| DR256, 1 tile | 0.09–0.27 | 0.13–0.32 | 0 |
| DR256 × 256 tiles, Q ≥ 64 | 6.5–6.7 | 6.8–7.1 | 0 |

Revalidating the register file costs about 0.04–0.6 µs per query over direct planes. Arm B (bind per query) costs 1.2–1.6 µs (DR64) and 3.7–4.3 µs (DR256) per query, 5 allocations and 7 / 26 lookups. At 256 tiles arm B **fails the gate** (above 1,000 µs per query): routing 6,554 ids scans the tile maps one after another, 842,138 lookups per bind, because no global id → tile index exists. That is the cost a binding pays once, and the measured reason the index (or a co-addressed producer) is an owner question.

## Findings

1. **The first missing source-level connection is a layout witness on the canonical SoA**: a statement that its rows are laid out under frame version V with tile digests D. Without it a binding cannot prove that the planes it borrows correspond to the frame; with it (the lab `Soa::layout`), a permuted or foreign layout is refused, and turning that check off gives a wrong answer.
2. **Frame currency is policy; correctness rests on the layout digest.** `MapTable::map` answers old frames by replaying their layout, so an old binding over an unchanged SoA still answers correctly (the lax-frame control). After an extension, the old SoA is refused under the new frame by its digest, not by the frame check.
3. **Map and value lifetimes separate cleanly**, exactly as `ValueDemand` declares; no executor reads `ValueDemand` today.
4. **Routing is the only per-query cost a binding removes.** Execution itself is identical (A vs C differ by the checks).

## Not shown

Remote or asynchronous execution, network or RDMA zero-copy, Lance I/O, source leases, cross-call result registers (the derived register here lives inside one in-process object; giving it a lifetime across calls is an owner decision). Decision memos: `.claude/plans/d-drisa-decision-memos-v1.md`.
