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
| **T1 primitive** | TWO SIBLING ALGEBRAS (2026-09-07, below): **population** — `ndarray::simd` facade, `lgj-abi/kernels.rs`, `mask_*`, `eq_*_to_mask`, `ternlog`, `popcount`; **epistemic** — `TruthU8`, revision, deduction, abduction, … | `&[u64]`, `&[u8]`+`(offset,stride)`, `IMM`, `TruthU8` | a mask, a count, a lane descriptor, **a truth lane** | **polyfill rule** (simd-savant): no intrinsic, no `#[cfg(target_arch)]` above this line |
| **T2 behavior** (was "selection") | ABI exports; `where`/`hop`/`plan_eval`; `Mask × WideFieldMask → Mask`; **and the epistemic siblings, named through the same `plan_eval`** | handles, `classid`, `FieldMask` (fields by NAME), version | a handle, a count, a status | **no hand-composed T1 op, no computed geometry** (kernel-membrane-warden) |
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

## T1 has TWO sibling algebras — the doctrine is a behavior membrane, not a selection pipeline

**Operator ruling, 2026-09-07.** An audit found NARS truth arithmetic nowhere on
the Java side of the membrane — not exported, not imported, not present — and
concluded from that: *"NARS is off the ladder entirely; the ladder has no tier
for scoring."* **The conclusion was wrong and the diagnosis was the wrong axis.**
The distinction that matters is not *selection vs scoring*. It is **syntax vs
execution** — which is the distinction every other tier here is already built on.

The ladder does not need a sixth tier. **T1 was described too narrowly.** It
contains two sibling primitive algebras:

```
T1
├── population algebra          └── epistemic algebra
│     mask                            TruthU8
│     ternlog                         revision
│     eq → mask                       deduction
│     popcount                        abduction
```

Both are primitive behavior. **T2 may name either. T2 may not hand-compose
either. T3 may express intent in either.** Every rule already written applies
unchanged to the second column — `kernel-membrane-warden`'s HAND-COMPOSED
verdict covers a T2 that spells `revision` out of smaller truth ops exactly as
it covers a T2 that spells `AND3` as two `mask_and`s.

### The lowering, end to end

```
T3  Java / R2IL / low-code
      │  NARS operation NAME + opaque handles
      ▼
T2  plan_eval — the behavior membrane
      │  resolved bulk operation
      ▼
T1  epistemic primitives (beside the population primitives)
      │  substrate-native execution
      ▼
T0  TruthU8 lanes / rows / history / state
```

T3 may say `Truth.Revision(lhs_handle, rhs_handle)`. **It may not know how
revision works.** T2 resolves the name; T1 executes the arithmetic; T0 owns
every resulting `TruthU8`.

### Extend the plan language, NOT the ABI surface

The tempting fork — mint `lgj_score_*` beside `lgj_hop` — is rejected. It grows
a second semantic API next to `plan_eval`, and the end state is predictable:
`where()`, `hop()`, `score()`, `nars_revision()`, `nars_deduction()`, … with
Java knowing progressively more about the behavior graph. **The membrane starts
growing little computational fingers.**

`lgj_plan_eval` exists precisely so a whole behavioral expression crosses ONCE.
NARS becomes another named plan operation, not another export:

```
Plan
 ├── Select(…)
 ├── Hop(…)
 ├── Ternlog(…)
 └── Truth(…)
      ├── Revision
      ├── Deduction
      ├── Abduction
      └── …
```

### `TruthU8` is the canonical SUBSTRATE representation — not automatically the wire form

These are two different claims and the workspace had been conflating them.
`TruthU8 { frequency: u8, confidence: u8 }`
(`lance-graph-arm-discovery/src/translator.rs:25-33`) is canonical **at T0**.
What crosses is decided separately, and by shape:

| shape | crosses? | as |
|---|---|---|
| a truth LITERAL, `TruthLiteral(192, 217)` | **yes** — it is meaning supplied by the caller, syntax, T3's to state | itself |
| a truth POPULATION, `[TruthU8; 65536]` | **never** | `TruthLaneId(u64)` — an opaque 8-byte descriptor |

This is the same rule `bbb-warden` already enforces for masks (*"a `long[]` of
selected ids is still a materialised population"*), applied to the epistemic
column — and it lands exactly on the measured Valhalla cliff: **flattening stops
at an 8-byte payload** (VM-confirmed, `valhalla-lab/docs/three-truths.md`), so a
`TruthLaneId(u64)` flattens and a truth array could never. The JVM agrees with
the membrane about where the wall is. **Valhalla carries the noun; Panama
carries the verb; lance-graph owns the reality.**

### The G11 widening rule: one scalpel cut, never the cupboard

Do **not** import `lance_graph_contract::nars` through the G11 fence merely
because it exists. If that module carries arithmetic semantics together with POD
types, **split out a tiny syntax/vocabulary contract first** and admit only that.
The fence widens by one deliberate module, in one commit, in all three places
its allowlist is spelled (`tests/g11_contract_import_fence.rs`'s `ALLOWED`,
lgj `CLAUDE.md § Enforcement`, `Cargo.toml`'s comment) — the shape lgj already
requires, and the reason its own history records the fence being prose until
2026-09-03 (`ISS-LGJ-G11-FENCE-WAS-PROSE`).

### The ruling and its falsifier

> **D-BBB-NARS-1.** NARS truth arithmetic remains substrate-owned. G11/T3 may
> carry only typed NARS **syntax** and **opaque substrate handles**. NARS
> execution is lowered through the existing bulk plan-evaluation membrane; no
> Java-side arithmetic and no materialized truth population crosses Panama.
> `TruthU8` is the canonical substrate representation, while cross-membrane
> results are handles. Any required G11 expansion SHALL expose
> syntax/vocabulary only, never an arithmetic implementation surface.

> **F-BBB-NARS-1.** Fail if Java can implement, inspect, iterate, or reconstruct
> NARS truth arithmetic without invoking the substrate, or if a truth population
> crosses G11/Panama other than as an opaque handle.

The BBB does not move. It stays exactly where it is:

```
                BBB
T3  intent / names        ─────────────
T2  opaque bulk behavior handles
                              ↓
T1  algebra  (population ‖ epistemic)
T0  state
```

No VSA internals. No RoleKey. No NARS arithmetic. No byte positions. No truth
arrays. No Java compute path. **Only names and capabilities.**

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
| L1 | `WideFieldMask.ofFacets(int... positions)` — slot indices cross | `classid` + `ClassView`-resolved field NAME; or a named `Reading` (RAILS/SPO) the ClassView selects | bbb-warden (semantic; reflection can't) | **[OPEN]** |
| L2 | 97 served `LgjLaneDesc` lanes — offset+stride cross to Java | field NAME; T2/`ClassView` owns geometry, Java never receives it to be "blind" about | bbb-warden | **[OPEN]** |
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
