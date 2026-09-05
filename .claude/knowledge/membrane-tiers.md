# Membrane Tiers — the 3+ layer abstraction law, and the entropy ledger

> READ BY: kernel-membrane-warden, bbb-warden, simd-savant,
> preflight-drift-auditor, layer-boundary-warden, integration-lead.
> READ BEFORE: designing any public signature, ABI symbol, mask kernel, or
> consumer-facing surface in any language; briefing a worker whose file sits
> on a membrane.
>
> Status: DOCTRINE (operator-directed 2026-09-04). The tiers formalise the
> lance-graph-java "mask-native invariant" and the ndarray polyfill as ONE
> pattern, repo-wide.

## The principle

**A tier may only know the vocabulary of the membrane directly beneath it.
Nothing crosses a membrane except by NAME.** Three membranes minimum; the raw
substrate beneath them is T0.

| Tier | What it is | May know | Crosses UP as | The membrane (gate) |
|---|---|---|---|---|
| **T0 substrate** | bytes, lanes, Lance columns, SoA v3 rows; `simd_{avx512,avx2,neon,scalar}.rs` | strides, offsets, carvings, intrinsics, alignment | — | none; T0 is where truth lives |
| **T1 primitive** | `ndarray::simd` facade; `lgj-abi/kernels.rs`; `mask_*`, `eq_*_to_mask`, `ternlog` | `&[u64]`, `&[u8]`+`(offset,stride)`, `IMM` | a mask, a count, a lane descriptor | **polyfill rule** (simd-savant): no intrinsic, no `#[cfg(target_arch)]` above this line |
| **T2 selection** | ABI exports; `where`/`hop`/`plan_eval`; `Mask × WideFieldMask → Mask` | handles, `classid`, `FieldMask` (fields by NAME), version | a handle, a count, a status | **no hand-composed T1 op, no computed geometry** (kernel-membrane-warden) |
| **T3 intent** | Java facade; R2IL / OGAR `ActionDef`; low-code | names: class, edge, field, version | an outcome | **no byte position** (bbb-warden + ApiSurfaceTest) |
| **R2IL** | emits T3 artifacts | T3's vocabulary (names, outcomes) | an outcome | its ceiling IS T3's; door-knocker test (layer-boundary-warden) |

`where()` is T2 precisely because it is the alias of a glove: Java owns the
NAME `where`, T2 owns the descriptor, T1 owns the mask op, T0 owns the bytes.
The same word lives at every tier and means one thing at each.

## The polyfill is the worked instance

ndarray's `simd.rs` (T1 membrane) → `simd_ops.rs` (staging) → `simd_{arch}.rs`
(T0) IS this pattern at T0/T1. A consumer that reaches into `simd_int_ops` or
hand-writes a compare-and-pack loop has punched T1 — the violation
`simd-savant` exists to catch. lgj-abi stacks the same shape at T1/T2:
`exports.rs` names `kernels::ternlog::AND3`, never `ndarray::simd` directly.

## The compile-through rule (the Entropy half)

**Old code is not deleted; it is re-admitted only by compiling THROUGH the
membrane beneath it.** A T3 artifact containing T1 vocabulary (a stride, a
`[u8;12]`, a slot index) is a *cast leak* — rewritten as a call through T2, or
it does not compile. The `.claude/v3/ENTROPY-MILESTONES.md` N→1 ledger records
each old path that now compiles through a membrane instead of around it.

Three properties make this enforceable, not aspirational:

1. **Each membrane has a structural gate, not a review note.** T0/T1: the
   simd-savant grep + `ndarray::simd` re-export. T1/T2: the G11 import fence +
   `kernels.rs` as the sole ndarray importer. T2/T3: `ApiSurfaceTest`'s
   forbidden-type list + the array-return naming rule. A membrane without a
   gate is prose.
2. **Leaks are enumerated, dated, closed downward** (the ledger below).
3. **Named breaches, never unnamed ones.** `materializeRows()` / `importRows()`
   are the precedent: a crossing that must exist is allowed only under a name
   that says so at the call site.

## What the gate CANNOT catch (stated honestly)

Reflection cannot distinguish `int classid` (a T2 name, clean) from
`int facet` (a T1 slot index, a leak) — same type. So the T2/T3 gate catches
the *mechanical* subset (raw `byte[]` registers, unnamed array returns, FFM
types) and `bbb-warden` reviews the *semantic* subset (a raw `int` that is
really a slot). Do not claim the gate proves the membrane; it proves the
catchable half. The warden proves the rest.

## Agent → membrane map

| Membrane | Warden | Model | Verdicts |
|---|---|---|---|
| T0/T1 | `simd-savant` | sonnet | POLYFILL-CLEAN / RAW-INTRINSIC / SHADOW-KERNEL |
| T1/T2 | `kernel-membrane-warden` | opus | NAMED / HAND-COMPOSED / GEOMETRY-LEAK |
| T2/T3 | `bbb-warden` | opus | HANDLE-CLEAN / BYTE-POSITION / UNNAMED-BREACH |
| T3/R2IL | `layer-boundary-warden` | opus | COMPILE-TIME-CLEAN / DOOR-KNOCKER / WRONG-SHELF |

All membrane wardens above T0/T1 are Opus: leak detection is accumulation
(read N files, verdict only holds them together). The pipeline: Sonnet writes
the preflight draft → Opus (preflight-drift-auditor) flips it, checking spec
vs main AND spec vs membrane → Sonnet fleet migrates call-sites (shared
checkout, edit-only) → Opus runs the wardens + the gates once. 5+3 rules only
on DOCTRINE changes (a new membrane, a moved line, a new named-breach class),
never on a call-site migration — that would be the recursion the 2026-08-04
ruling stopped.

---

## ENTROPY LEDGER — T2→T3 leaks (append-only; close downward)

Each row: the leak, the T2 name that replaces it, and the gate that will
reject the old spelling once closed. `[OPEN]` until the gate rejects it.

| # | Leak (T1/T0 vocab in a T3 surface) | Replace with (T2 name) | Gate | Status |
|---|---|---|---|---|
| L1 | `WideFieldMask.ofFacets(int... positions)` — slot indices cross | `classid` + `ClassView`-resolved field NAME; or a named `Reading` (RAILS/SPO) the ClassView selects | bbb-warden (semantic; reflection can't) + ApiSurfaceTest name-pin | **CLOSED 2026-09-05** — `WideFieldMask` is a `final class` with a private ctor; `ofFacets` AND `ofMatchBits` package-private (lance-graph-java#75). Demoting one factory was not a fence — a public record's canonical ctor took the raw bits (codex + coderabbit P2, fixed same PR). Zero production callers. Pin is on the SHAPE: no public ctor, not a record, every public factory zero-arg. Name-side replacement needed no ABI: `hop(classid, src)` + native `edge_participation` narrowing already was it. Pinned by name in `ApiSurfaceTest` (added in lance-graph-java#75, the paired code PR). |
| L2 | 97 served `LgjLaneDesc` lanes — offset+stride cross to Java | field NAME; T2/`ClassView` owns geometry, Java never receives it to be "blind" about | ApiSurfaceTest `internal.*` prefix (already) | **CLOSED-BY-EXISTING-GATE 2026-09-05 — regraded; the row overstated it.** `abi.md:312`: "Java's *public* API never sees an address." The carrier (`Engine.LaneWindow`) is `internal.ffm`, fenced from every public signature by the prefix; used only inside `RowStore`/`Mask` (which READ the stride from the served descriptor, `abi.md:367` — NAMED, not GEOMETRY-LEAK) and the sanctioned lab consumers. Structural pin added in lance-graph-java#75, the paired code PR (class exists AND is under a FORBIDDEN prefix) — this ledger row records it, it does not add it. Residual is a design ceiling, not a leak: layout-aware Valhalla views should be OGAR-emitted per ClassView (Tier 3), not hand-carved in the lab — a future wave. |
| L3 | `RowStore.classidAt / payloadLow64At / payloadHi32At` — per-row byte reads | fenced as inspection-only (javadoc line present); execution must not use them | ApiSurfaceTest note + bbb-warden | PARTIAL (fenced, not removed) |
| L4 | `FacetMatchView.matchesOf(row) -> int` — raw facet bitset | `WideFieldMask.ofMatchBits` bridges it; callers take the typed value | bbb-warden | PARTIAL |
| L5 | `Engine.LaneWindow.setU64` — raw word write | `importRows` (named breach) is the only sanctioned writer | ApiSurfaceTest (internal.ffm already fenced from public) | CLOSED |
| L6 | any future `byte[]` / `[u8;12]` rail array in a public signature | a named `Reading` value type OGAR emits per ClassView (Valhalla), read zero-copy | ApiSurfaceTest byte[]-fence (this PR) | CLOSED (forward guard) |
| L7 | any future array return not named `materialize*`/`import*` | a named terminal | ApiSurfaceTest array-return naming rule (this PR) | CLOSED (forward guard) |

Provenance: the two fixes that produced this doctrine — the 7.5→1.1 ms
`lgj_hop` (T1 doing T0's job badly: gathered a contiguous lane; fixed inside
T1, T2/T3 unchanged) and the two-AND→ternlog conjunction (T2 hand-composing a
T1 op; fixed by naming the op at T1) — are the T0/T1 and T1/T2 membranes
working. Both are recorded in ndarray `.claude/blackboard.md` 2026-09-04 and
lance-graph-java `LATEST_STATE.md` 2026-09-04.
