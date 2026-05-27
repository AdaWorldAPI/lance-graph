# RECONCILIATION — `.claude/surreal/` POC ↔ canonical CausalEdge64-Mailbox plan

**Finding (honest):** the `.claude/surreal/` 12-task POC + `cognitive-substrate.md`
is a **parallel, narrower re-derivation** of the already-authored, reviewed
canonical plan **`.claude/plans/causaledge64-mailbox-rename-soa-v1.md`** (which
itself composes 7 prior plans). The canonical plan **subsumes** the surreal POC
and is the driver. This file maps surreal → canonical so we align rather than
compound the duplication.

## 1:1 mapping (surreal POC concept → canonical home)

| surreal POC concept | Canonical plan home |
|---|---|
| SoA container / "no fragmentation" | §5 `MailboxSoA<N>` (compartments = SoA rows) + `bindspace-columns-v1` |
| owned per-thought struct / no singleton | §1 ownership-typed compartments; §9 E-CE64-MB-4 (UB = compile error) |
| Rubikon FLOW / collapse-gate / merge | §3 CausalEdge64 truth-band; CollapseGate Xor/Bundle/**Superposition** |
| attention + goalstate (not request) | §2/§4 `AttentionMask` sparse-rename register file; §6 Σ-tier dispatcher |
| zones (internal / membrane / consumer) | §0 Zone 1 / BindSpace / Zone 2 / Zone 3 (authoritative table) |
| 20-200ns · L1 residency · ~1.5 KB | §5 ~1.2 KB/compartment, ~24K thoughts, L1/L2 |
| SurrealDB-on-Lance persistence | Zone 2 `lance-graph-callcenter` + AriGraph SPO-G quads |
| immutable O(1) OGIT/DOLCE/CAM | §2 universal sparse-rename + `lance-graph-ontology` |
| superposition merge ("Gap #2") | `CollapseGate::MergeMode::Superposition` (PR-CE64-MB-3) |
| SoA speed-lane in contract | §2 8-bit-slot rename (TrustTexture/ThinkingStyle canonical) |
| cognitive-shader shape-shift SPO 2³/NARS | §3 Pearl 2³ rung bits + §9 epiphanies; `cognitive-shader-driver` Columns A-H |

## Course-correction

- **Stop extending the parallel surreal derivation.** Adopt the canonical
  plan's **5-crate inventory (§6)** + **7-PR sequence (§7)**.
- The surreal task-02 **`SoaContainerHeader`** (ndarray, pinned `b5d6b206`) is a
  useful *generic* LE primitive, but it must be **reconciled with the plan's
  `MailboxSoA` / BindSpace-Columns shapes** — it is NOT a new parallel SoA system
  (that would be the duplication the surreal specs themselves forbid). Likely it
  becomes the on-wire descriptor *under* `MailboxSoA`, or folds into
  `bindspace-columns-v1`.
- The surreal `crates/surreal_container/` scaffold overlaps **par-tile** (§6) +
  Zone-2 persistence; fold it into the canonical crates rather than ship separately.

## ndarray is MANDATORY (§8) — the concrete prerequisite

**PR-NDARRAY-MIRI-COMPLETE** (lands BEFORE par-tile):
- Close the `U16x32 / U32x16 / U64x8` (+ i-word) method gaps: `simd_eq/ne/ge/gt/le/lt`,
  `simd_clamp`, `select`, `to_bitmask`, `from_u8x64_lo+hi`, `pack_saturate_u8`,
  `shl/shr`, explicit `zero()` (`simd_nightly/{u,i}_word_types.rs`).
- Route `crate::simd::*` through `simd_nightly` under `cfg(miri)` (~50 LOC, `simd.rs:215+`).
- Delete the dead `simd_nightly/_original_draft.rs` 5-type sketch.

This is the real "ndarray is mandatory" work — the SoA container is a smaller
piece of the same foundation.

## Open decision for the user (§15 has 8 unratified items)

Adopt the canonical plan's 7-PR sequence as the driver (recommended — it's
reviewed and ~80% overlaps what we re-derived), folding the surreal POC into it?
Or keep the surreal POC as a deliberately-simpler standalone slice? They are not
two projects — they are one, and the canonical plan is the better-specified copy.

## Refinements (latest — fold into the canonical plan)

- **Agnostic polyfills.** All SIMD/SoA primitives are **arch-agnostic to the
  consumer (lance-graph)**: lance-graph consumes `ndarray::simd::*` (and the SoA
  speed-lane **trait**) without ever seeing AVX-512 / AVX2 / NEON / scalar — the
  polyfill hides the backend. This is the `simd-savant` invariant ("all SIMD from
  `ndarray::simd` via the polyfill"). The `lance-graph-contract` speed-lane is
  therefore a **trait**, never an arch-specific type — consistent with the
  zero-dep + agnostic requirement.

- **`Vsa16kF32` (16k-dim float VSA) is DEPRECATED.** Cumulative cognitive state
  does NOT live in `Vsa16kF32`. It lives in **CausalEdge64 emissions + AriGraph
  SPO-G quads + the BindSpace SoA columns**. The canonical plan §9 E-CE64-MB-2
  goes part-way ("retire `Vsa16kF32` as universal carrier"); this **deprecates it
  outright**. New work must not reach for `Vsa16kF32` as a carrier.
  - ⚠ **Contradiction to flag:** lance-graph `CLAUDE.md` "The Click" is built on
    `Vsa16kF32` (element-wise multiply+add Markov bundle). Deprecating the carrier
    contradicts that canonical doctrine → needs a `CLAUDE.md` + `EPIPHANIES.md`
    board update before/with the deprecation lands. Do not silently diverge.
