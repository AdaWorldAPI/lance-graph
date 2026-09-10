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
| **T1 primitive** | TWO SIBLING ALGEBRAS (2026-09-07, below): **population** — `ndarray::simd` facade, `lgj-abi/kernels.rs`, `mask_*`, `eq_*_to_mask`, `ternlog`, `popcount`; **epistemic** — `TruthU8`, revision, deduction, abduction, … *(RULED, NOT YET RESIDENT — see § "What is ruled vs what is coded" below: these four are coded only in `lance-graph-planner`'s `nars_engine.rs`, and are ABSENT at the lgj-abi T1 membrane)* | `&[u64]`, `&[u8]`+`(offset,stride)`, `IMM`, `TruthU8` | a mask, a count, a lane descriptor, **a truth lane DESCRIPTOR (the `TruthLaneId` shape D-BBB-NARS-3 would mint — prescriptive, 0 code sites today) — never the population** | **polyfill rule** (simd-savant): no intrinsic, no `#[cfg(target_arch)]` above this line |
| **T2 behavior** (was "selection") | ABI exports; `where`/`hop`/`plan_eval`; `Mask × FieldMask → Mask` *(the shipped type; `WideFieldMask` does not cross the ABI — lgj `fixture.rs:1-6` calls wiring it "a later slice", and `class_view_provider.rs:64,235` wires plain `FieldMask`)*; **and the epistemic siblings, named through the same `plan_eval`** | handles, `classid`, `FieldMask` (fields by NAME), version, **a truth LITERAL (`TruthLiteral`) — never a truth population** | a handle, a count, a status | **no hand-composed T1 op, no computed geometry** (kernel-membrane-warden) |
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

**That claim was FALSE when first written, and is true only because the same
commit made it true.** `kernel-membrane-warden`'s trigger and its method named
masks only; a T2 hand-composing `revision` would have walked straight through
the gate this paragraph invoked. Codex caught it on #1222 (P2), and the card
now carries the epistemic algebra in its trigger and **method step 2b**. Two
sibling gaps of the same shape were closed with it: the T1 row above said a
truth LANE crosses up while the shape table below says a population never does
(now: the DESCRIPTOR, never the population), and `bbb-warden`'s method
classified only signature shapes, so a public helper with a legal
`TruthLiteral` signature computing revision in its BODY passed every step while
doing exactly what `F-BBB-NARS-1` forbids (now: method step 4, an explicit
body-and-import audit). **A doctrine sentence that names a gate must cite the
step that makes it true** — the same defect as the G11 fence being prose until
2026-09-03, found three times in one commit and fixed in the same one.

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

**Target route, not an available one.** T3 will say
`Truth.Revision(lhs_handle, rhs_handle)`, and **will not know how revision
works** — that part is the ruling and is permanent. The rest is not yet built:
once the structural gate exists and `plan_eval` carries a Truth opcode (it
carries `{EQ_U32, GT_I32}` today and no Truth opcode; both gate on
D-BBB-NARS-2/-3), T2 will resolve the name, T1 will execute the arithmetic, and
T0 will own every resulting `TruthU8`. The membrane above is shipped; the
operation that would travel it is not. See § "What is ruled vs what is coded".

### Extend the plan language, NOT the ABI surface

The tempting fork — mint `lgj_score_*` beside `lgj_hop` — is rejected. It grows
a second semantic API next to `plan_eval`, and the end state is predictable:
`where()`, `hop()`, `score()`, `nars_revision()`, `nars_deduction()`, … with
Java knowing progressively more about the behavior graph. **The membrane starts
growing little computational fingers.**

`lgj_plan_eval` exists precisely so a whole behavioral expression crosses ONCE.
(That rule is not new here — it restates lgj `docs/abi.md` §6, which already says the
fused-plan call exists "precisely so that `.where(...).where(...).count()` is **one**
crossing regardless of how many predicates or rows." What is new is extending it to
the epistemic column.)
NARS becomes another named plan operation, not another export:

```
Plan                       ← ILLUSTRATIVE. Not a type that exists today.
 ├── Select(…)
 ├── Hop(…)
 ├── Ternlog(…)
 └── Truth(…)
      ├── Revision
      ├── Deduction
      ├── Abduction
      └── …
```

**Read that as the shape `D-BBB-NARS-3` would mint, not as a description of the
code.** Measured 2026-09-10 across both checkouts: `enum Plan` / `Plan::Select` /
`Plan::Hop` / `Plan::Ternlog` / `Plan::Truth` have **zero hits**; `lgj_hop`
(`exports.rs:1703`) is an ABI function, not a tree variant. `lgj_plan_eval` IS
shipped and tested (`exports.rs:1421,1522,1547`; `abi.rs:246-251,337`), but its
`LgjOpDesc` is a **flat array with a combined AND/OR — not a tree** — and its
opcode set is exactly `{LGJ_OP_EQ_U32 = 1, LGJ_OP_GT_I32 = 2}` (`abi.rs:250,252`),
with **no Truth/Revision opcode**. So "a whole behavioral expression crosses ONCE"
is CODED for population predicates and RULED-BUT-UNBUILT for the epistemic side.

### `TruthU8` is the canonical SUBSTRATE representation — not automatically the wire form

These are two different claims and the workspace had been conflating them.
`TruthU8 { frequency: u8, confidence: u8 }`
(`lance-graph-arm-discovery/src/translator.rs:25-33`) is canonical **at T0**.
What crosses is decided separately, and by shape:

| shape | crosses? | as |
|---|---|---|
| a truth LITERAL, `TruthLiteral(192, 217)` | **yes** — it is meaning supplied by the caller, syntax, T3's to state | itself |
| a truth POPULATION, `[TruthU8; 65536]` | **never** | `TruthLaneId(u64)` — an opaque 8-byte descriptor |

This is the same rule `bbb-warden` already enforces for masks (*"a `long[]` of row ids is a
materialised population"*, `bbb-warden.md:32`), applied to the epistemic
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

### What is ruled vs what is coded (measured 2026-09-10 by the 5+3 council)

The ruling above is binding. Most of what it rules is **not yet resident**, and this
section exists so no future session mistakes a decision for an accomplished fact.
The doctrine's own test is three sections down: *"A membrane without a gate is prose."*

**1. The epistemic column has NO structural gate — yet.** Each membrane is held by a
structural gate: T0/T1 by the simd-savant grep + the `ndarray::simd` re-export, T1/T2
by the G11 import fence + `kernels.rs` as sole ndarray importer, T2/T3 by
`ApiSurfaceTest`'s forbidden-type list + the array-return naming rule. **The epistemic
column adds none of these.** What it adds — `bbb-warden` step 4, `kernel-membrane-warden`
step 2b — are *review notes*, which property 1 below explicitly distinguishes from gates.
Both steps are real and discriminating (each catches a body that every signature-shaped
step passes, and each has a sanctioned silent case), but a review note is not a fence.
`F-BBB-NARS-1` likewise cannot be exercised today: grep across lgj@`8720d1d` `native/`
and `java/` for `TruthU8`/`revision`/`deduction`/`abduction`/`induction` returns **zero
hits**, so there is no Java surface to run it against. **The gate that will hold this
column is `ApiSurfaceTest`'s forbidden-type list plus a G11 allowlist entry, and it is
gated on D-BBB-NARS-2/-3** — which are Queued and marked *do not pre-build*. Until then
this half of the membrane is enforced by review, and saying otherwise would be the exact
defect this arc keeps finding.

**2. The named epistemic primitives are not at T1.** `revision`/`deduction`/`abduction`/
`induction` are CODED, but only inside `crates/lance-graph-planner/src/cache/nars_engine.rs:194-207`
(`Inference::{Deduction,Induction,Abduction,Revision}`) — a planner-internal dispatch,
**not** a T1 primitive callable from T2 — and are ABSENT at the lgj-abi membrane
entirely. `kernel-membrane-warden` step 2b already states the consequence correctly
("if it does not exist at T1, it lands at T1 first"); the T1 row above now carries the
same hedge, which it did not when first written.

**3. `TruthU8` is the RULED TARGET, and it has three incumbents.** The ruling makes
`TruthU8` canonical at T0. Measured, four truth types coexist today, each self-described
as canonical in some register:

| type | shape | site |
|---|---|---|
| `exploration::NarsTruth` | `f32 × 2` | `lance-graph-contract/src/exploration.rs:89` |
| `holograph::width_16k::schema::NarsTruth` | `u16 × 2`, packed | `holograph/src/width_16k/schema.rs:104` |
| `ndarray::hpc::nars::NarsTruth` | — | aliased `Truth` at `lance-graph-planner/src/cache/triple_model.rs:42` |
| `arm-discovery::TruthU8` | `u8 × 2` | `lance-graph-arm-discovery/src/translator.rs:28` |

**The engine that actually executes revision/deduction/abduction uses the third**, via
that alias. `TruthU8` occurs outside its own crate in exactly one file, a test. No
conversion path bridges them. So "T0 owns every resulting `TruthU8`" is the direction of
travel, not the current state — the convergence is tracked as **D-BBB-NARS-4**.

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
| T2/T3 | `bbb-warden` | opus | HANDLE-CLEAN / BYTE-POSITION / UNNAMED-BREACH / ARITHMETIC-SURFACE |
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
| L8 | any future truth POPULATION in a public signature — `TruthU8[]`, a truth lane, a collection of them — or any T3 body that computes a truth FROM truths | the `TruthLaneId(u64)` opaque descriptor for the population; a named `Truth(…)` `plan_eval` operation for the arithmetic | **OPEN — review-note only** (`bbb-warden` step 4 + ARITHMETIC-SURFACE). The structural gate (ApiSurfaceTest forbidden-type entry + G11 allowlist) is gated on D-BBB-NARS-2/-3 | OPEN (forward guard, ungated) |

Provenance: the two fixes that produced this doctrine — the 7.5→1.1 ms
`lgj_hop` (T1 doing T0's job badly: gathered a contiguous lane; fixed inside
T1, T2/T3 unchanged) and the two-AND→ternlog conjunction (T2 hand-composing a
T1 op; fixed by naming the op at T1) — are the T0/T1 and T1/T2 membranes
working. Both are recorded in ndarray `.claude/blackboard.md` 2026-09-04 and
lance-graph-java `LATEST_STATE.md` 2026-09-04.
