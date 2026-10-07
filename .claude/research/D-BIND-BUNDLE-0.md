# D-BIND-BUNDLE-0 — BIND and BUNDLE as the compiler seam

> **READ BY:** anyone adding a planner/fold/physical IR, a prepared-query
> cache, a schema registry, or a second execution strategy.
> **Status:** source-backed, 2026-10-07; the two-bundle probe result is in
> §14 once run. Production code read first. Grades: [G] read/measured,
> [H] declared/literature, [S] reasoned.
> **Bottom line first:** the four verbs PARSE / BIND / BUNDLE / EXECUTE fit the
> live code, but **V4 today is `quack::Query`, not R2IL**, and **BUNDLE is a
> choice of lowering function, not an IR.** No new layer is justified.

## 1. Current bind inventory

| Binder | Lives | Resolves | Status |
|---|---|---|---|
| upstream `SemanticAnalyzer` + `GraphConfig` | `lance-graph/src/semantic.rs`, `config.rs` | labels → id field, rel → src/dst field names, `$params` substituted INTO the AST | live [G] |
| cypher-quack `Binding { nodes: NodeTable{label,id_property,properties:(name,Col,Kind)}, edges: EdgeTable{rel_type,label,src,dst,dst_ordered} }` | `lance-graph-cypher-quack/src/lib.rs` | names → `Col` ordinal + lane kind (I32/U32) + layout fact (`dst_ordered`) | experimental [G] |
| quack `bind::{Binder, Draft, BoundField, Registrar}` | `lance-graph-quack/src/bind.rs` | text field → `{col, kind, validity}`; one table; eq/ne; count/rows | implemented only in tests (report, dir-sim, quack) [G] |
| report `boundary::Catalog` | `lance-graph-report/src/boundary.rs:252` | name → FieldId | live in report [G] |
| `ClassView` / `ViewRegistry` | contract `class_view.rs`, `selection.rs` | classid → ordered `FieldRef{predicate_iri,label}` / `WideFieldMask` views | live, but **no lanes, no Col, no kinds** [G] |

No registry maps classid → lane → Col → kind (closed search over every crate
in `lance-graph/crates/*`). The Python `graph.yaml` carries less than the
cypher-quack `Binding` needs (no types, no endpoint labels, no ordering) — see
`cypher-engine-autopsy.md` §H.

**Proposed source of truth:** extend `graph.yaml` (ordered `properties:
[{name, type}]`, `source_label`, `target_label`, `ordered_by`) and compile it
ONCE into (a) upstream `GraphConfig`, (b) the cypher-quack `Binding`. Parse
with spans (q2 `quarto-yaml` is reusable as that layer); reference checks and
dense-ordinal assignment are ~1 small pass written here. classid/ClassView
registration and relationship-carrier declarations are the same compiled
artefact keyed by classid instead of label — not a second schema language.

## 2. Current physical/lowering inventory

`quack::Query { filter: Filter, agg: Agg }` → one of
`lower` (in-place, conjunct-chained, survivor-gated per 64-row word) ·
`lower_fused` (per-predicate slots, ternlog skeleton, no skipping) ·
`lower_group_by` (two-phase `GroupPlan`) → mask-risc `Program { ops, terminal }`
→ `Scratch::for_program(&program, n_rows)` → `execute_into` /
`execute_extent` over borrowed `Planes`/`Foreign`. [G]

## 3. Existing types that already approximate Bundle

- **mask-risc `Program`** — the executable image: numbered slots, gated ops,
  one terminal.
- **The choice `lower` vs `lower_fused`** — a physical decision over one
  `Query`, documented as such ("a consumer picks by whether it is scratch-bound
  or pass-bound"), proven equal by quack's differential suite. **This is a
  shipped two-bundle witness.** [G]
- **`Scratch::for_program`** — the execution-time resource sizing.
- **cypher-quack `Compiled { program, on, result, demand }`** — program + the
  table it runs over + result shape + demand contract.
- **`ReportPlan::physical_key()`** (report `plan.rs:430`) — a physical cache key
  that ignores axis roles, so pivot/rotate needs no rescan: a bundle key
  precedent.

## 4. BIND (proposed definition)

**BIND = semantic plan × V3 schema → operands that need no names.** It
resolves: label → table / classid; property → `Col` + lane kind; relationship
→ (src Col, dst Col, endpoint table); literal → immediate of the bound kind
(range-checked: `i32::try_from` fails as `Refusal::Value`); `$param` → slot;
**and the layout facts that decide carrier LEGALITY** (`dst_ordered`). It also
consumes the demand decision; it does not make it.

Finding: layout facts (sort order, run structure guarantees, uniqueness) are
**V3 representation, not execution statistics**. `dst_ordered` gates whether
`count(DISTINCT b)` is lowerable at all (`Refusal::Layout`). They belong to
BIND and the bind clock, never to BUNDLE.

## 5. BUNDLE (proposed definition)

**BUNDLE = bound V4 × execution facts → executable image**, choosing only
answer-preserving properties: lowering function (gated vs fused), carrier
(dense today), direction (n/a today), scratch geometry, extent/morsel split,
backend. Today it is **a function choice plus scratch sizing** — the image is
a mask-risc `Program`. That satisfies F7's warning literally: Bundle *is* the
existing `Program`; do not add a wrapper type until a second backend or a
second carrier exists to justify one.

## 6. Ownership matrix

See `D-BIND-BUNDLE-MATRIX.md`. Unambiguous for most rows. The ambiguous rows
are the findings:
- **lane width** appears in BIND (`Kind`) *and* in V4 (`Pred::EqU32` vs `EqI32`
  are distinct ops). Width is part of the operation's meaning (signed vs
  unsigned compare), so it is V4, chosen by BIND.
- **parameter value** lives in the backend `Program` today (`Pred` carries the
  literal). There is no parameter slot in Query, Program or R2IL (R2IL
  immediates are u8 only).
- **sort order** — BIND (legality), not BUNDLE (§4).
- **factorized multiplicity** — semantic demand (§12).

## 7. Two invalidation clocks

- **Bind clock** should tick on: schema, classid map, lane ordinal/kind,
  codebook identity, relation endpoint, layout guarantees. **Today nothing
  carries a generation** — `Binding` has none, `Program` has none. A schema
  change silently leaves a `Compiled` pointing at stale `Col`s. F4 cannot be
  tested yet because there is nothing to invalidate; that is the gap.
- **Bundle clock** should tick on: density/selectivity/run statistics, CPU
  caps, backend availability, parameter stability. Nothing ticks it today
  either; the lowering choice is static.
- They are genuinely different: the same `Query` stays valid while the
  `lower`/`lower_fused` choice flips with selectivity (quack docs: ordering
  under `lower` worth up to 99.61 pp of skipped words, zero under
  `lower_fused`). [G for the independence, S for the policy]

## 8. Prepared-query implications

`PreparedProgram = Binding-generation + Compiled(Query, demand, result) +
chosen Program`. Measured: patching the one `Pred` that carries the parameter
reproduces a fresh lowering exactly for 1004 values, at 10.4 µs vs 1117 µs for
DataFusion re-bind+plan+exec (10K rows). Because Quack's lowering copies the
literal without value-dependent folding (`Cmp::EqU32(v) => Pred::EqU32{lane,v}`),
**parameter change → execute only** holds today, at the Program level. If a
lowering ever folds on value (e.g. `v ≥ 2^31` → empty), the slot must move up
to Query. Schema change → re-bind + re-lower; statistics change → re-lower
only (choose `lower` vs `lower_fused`).

## 9. JITSON implications

JITSON's Cranelift path removes no dynamic work today (`D-V4-FOLD-LAB.md` §6),
so as a bundle specialiser it currently offers dispatch trimming only. Its
useful precedent is the **cache key**: `template_hash` (FNV-1a over the
template) is a BundleKey candidate once it gets field-length separators and
includes the backend key — its current form can collide across field splits.

## 10. Carrier / direction / materialisation

- Carrier: dense packed u64 only; a selection vector is a standing
  ELIMINATE ruling (quack matrix R1). Adaptive carriers are a BUNDLE decision
  in principle and an open measurement in practice (probe §14).
- Direction: no multi-hop executor exists, so push/pull has nothing to choose
  between yet.
- Materialisation: already a bundle-level effect of `lower` (gated `Pred`
  skips a dead word before reading the lane).

## 11. Backend implications

One backend exists (mask-risc over `ndarray::simd`). "CPU → GPU re-bundles
only" (F3) holds vacuously. The real test arrives with CubeCL, where the
device-buffer copy must be named at the membrane.

## 12. #1305 multiplicity boundary

Demand (`TerminalSet` / `EarlierSet` / `TerminalCount` / `EarlierCount` /
`Bindings`) is computed from the bound plan **before** lowering and selects
the semantic carrier: edge-row count (walks) vs ordered-key distinct
(support) vs refusal. Factorized multiplicity (2-hop count = Σ in·out) is the
same kind of choice: it is legal for count/sum and illegal for count-distinct
across factors, so **density or cost may never select it** — it is a demand
decision. Bundle never sees an unresolved bag/set question; that is why two
bundles cannot diverge on it (F5, F8).

## 13. Anti-lasagne analysis

| Proposed layer | Information it would hold that has no other home | Verdict |
|---|---|---|
| PlannerIR between LogicalOperator and Query | none — demand + Binding cover it | KILL |
| FoldIR between Query and Program | none — Query is the fold program | KILL |
| PhysicalPlannerIR | the lowering choice (one enum) | KILL as IR; keep as a parameter |
| VectorIR | lives in `ndarray::simd` | KILL |
| R2IL as live V4 | a second spelling with one test reader | DEFER until a second reader executes it |
| a `Bundle` struct | `Program` + one choice + generations | DEFER until two carriers/backends exist |

## 14. Falsifier results

| | Result |
|---|---|
| F1 BIND needs hardware choices | SURVIVES — Binding holds Col/kind/layout only |
| F2 BUNDLE recovers lost semantics | SURVIVES — demand resolved first |
| F3 CPU/GPU change forces re-bind | SURVIVES vacuously (one backend) |
| F4 schema change does not invalidate bindings | **FAILS today** — there is no generation to invalidate; staleness is undetectable |
| F5 two bundles disagree | SURVIVES — `lower` vs `lower_fused` differential; probe below |
| F6 carrier choice needs V4 opcodes | SURVIVES in the surveyed literature |
| F7 Bundle is a rename of `Program` | **TRUE** — so do not add the abstraction yet |
| F8 V4 cannot carry #1305 after carrier selection | SURVIVES for 1-hop; 2-hop count needs a factorized fold rule (gap, not a boundary error) |

Two-bundle probe: 5 routes × 32 cases, all equal — see "Probe result" at the end.

## 15. Minimal architecture recommendation

PARSE = upstream parser + semantic analyzer. BIND = a compiled `Binding`
(from graph.yaml) **with a generation**. V4 = `quack::Query` + demand + result
shape. BUNDLE = choose the lowering (and later carrier/backend) from
statistics. EXECUTE = mask-risc over borrowed lanes. No fifth verb is forced;
the missing information is the two generations, which are fields, not layers.

- **BIND: KEEP** (add a generation; compile from graph.yaml).
- **BUNDLE: CHANGE** — it is a decision function over existing types, not an
  IR or a struct.
- **V4 boundary: HOLDS** at `quack::Query`; **LEAKS** at R2IL (fold band
  mirrors the backend).
- **New IR required: NO.**
- **Highest-value next experiment:** RAN (probe result below): F5 holds, R1
  holds. Next: `MaskOp::Gather { under }` behind a parity test against the
  ungated gather, then re-run this probe — the emulation predicts ~5× at 10 %
  density and ~25× at 0.01 %.

## Probe result (2026-10-07) — one bound query, five routes

`crates/lance-graph-benches/examples/bundle_probe.rs`, 1M rows, release, debug=0,
median of 15, quiet machine. Query (bound: lanes and planes only):
`sel < t AND dept_fk ∈ allowed` → `count` and `sum(age)`. 8 densities × 2
layouts × 2 terminals; **every route returned the oracle's answer in all 32
cases** (asserted before timing). → **F5 survives: physical route changed,
answer did not.**

Selected rows (µs; full table is the probe's output):

| layout | density | gated | gated-r | fused | ordinal (Vec<u32>) | gated-gather (mask, emulated) | cmp only | via only |
|---|---|---|---|---|---|---|---|---|
| uniform | 100 % | 6164 | 7634 | 6702 | 6983 | 6292 | 203 | 5298 |
| uniform | 10 % | 5765 | 5773 | 5747 | 2320 | 1082 | 197 | 5219 |
| uniform | 1 % | 6375 | 5754 | 5653 | 748 | 477 | 197 | 5217 |
| uniform | 0.01 % | 5446 | 5480 | 5462 | 609 | 209 | 192 | 5157 |
| clustered | 10 % | 5615 | 5594 | 5649 | 1324 | 1041 | 222 | 5306 |
| clustered | 0.01 % | 5471 | 5478 | 5419 | 602 | 198 | 190 | 5160 |

What it shows:

1. **The three mask-risc routes are flat at ~5.5–6 ms at every density and
   layout**, and the semijoin alone costs ~5.2 ms of it (`via only`). The
   survivor law in `quack::lower` gates `Pred`s only: `MaskOp::Gather` has no
   `under`, so the expensive conjunct always runs over the whole population
   and conjunct order changes nothing (gated vs gated-r).
2. **The selection-vector carrier wins below ~25 %** — but only because it
   runs the lookup for survivors alone.
3. **A mask-native gated gather wins more, everywhere below 100 %**, without
   materialising ordinals: it approaches the comparison-only floor (~200 µs) at
   low density, 3× faster than the ordinal vector at 0.01 %.

**Consequence:** quack's R1 ruling ("no selection vector") survives this
measurement. The real gap is an executor one: **`Gather` needs the `under`
gate that `Pred` already has** — a survivor-law extension to the one op that
lacks it, not a new carrier and not a new V4 opcode. Second finding: mask-risc's
ungated gather costs ~5.2 ns/row for a 128-byte foreign plane (the cmp lane runs
at ~0.2 ns/row); it looks unvectorised — worth a separate look.
