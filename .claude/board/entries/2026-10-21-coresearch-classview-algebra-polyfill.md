# Co-research: what classes can do, and an algebra polyfill on ClassView — exploration map (2026-10-21)

**Status:** OPEN — exploration map; ratifies nothing. Question (operator, 2026-10-21):
"Was können Klassen und wo sind Potentiale und Synergien — besonders für Polyfill auf
Algebra." Harness: `.claude/agents/coresearch-council.md`.

**Premise gate (PREMISE-SPLIT).** "Realization selected per class" mixed two lifetimes:
- (a) what a class DECLARES: its law and the properties that law satisfies (per class);
- (b) which realization is CHOSEN for a binding instance (per class × binding state: focus
  size, version, witness availability, answer type).

**Anchors held:**
- A1: ClassView decides, DR maps; no new Association, RelationshipShape, Through,
  MethodDescriptor or planner type without a proven gap.
- A2: D-RA-2 — the contract owns the laws, consumers own execution.
- A3: ADMISSION B.
- A4: no ms; mask-risc is the reference fold.
- A5: SIMD via ndarray.
- A6: G1–G11, one verdict per atom and stratum.

## Council

**Scouts:**
- code cartographer;
- internal prior art;
- literature: Green–Karvounarakis–Tannen 2007, FAQ, GraphBLAS math, TACO format
  abstraction, Halide, tagless-final, egg (all SECTION-READ);
- systems: PostgreSQL operator classes and families, SuiteSparse:GraphBLAS, MLIR linalg,
  Soufflé, DuckDB, Odoo ORM;
- concepts: typeclass coherence, institutions, provenance semirings, Galois connections,
  Kleene K3, OWL 2, FCA, binding-time analysis.

**Co-architects:** bridge, firewall/fit, falsifier. The crosswalk has 33 rows (X1–X22 from
outside, U1–U11 from the operator's reflection); raw output is banked in the session
scratchpad.

## What the outside world agrees on

1. **The law is bound at declaration or bind time, never per row.** Every system and paper
   does this (PG opclass lookup, GraphBLAS factory/JIT kernels, Halide schedules, TACO
   codegen, tagless-final). This matches the IAM measurement: resolving the class once
   costs ±0.4 ns over the fold, while `execute_compute_dag` per row costs ~85 ns
   (entry 2026-10-21-classview-dr-iam).
2. **A cheaper route is licensed by a law PROPERTY:**
   - annihilator 0 → absent elements may be skipped;
   - associativity → reordering or parallelism;
   - semiring homomorphism → a result may be translated (positive RA only);
   - commuting aggregates → variables may be reordered.
3. **Nobody checks that a realization satisfies its law.** PostgreSQL says so explicitly
   for families, and DuckDB's lossy casts change the law silently. The common gap is a
   conformance gate.

## Corrections the co-architects made to the crosswalk

- **M1.** `Activation::register_law_for` returns `register128::RegisterLaw`
  (RelativeOffset / AxisPosition / Support), which is a register byte law, not an algebra
  law. The per-operation algebra carrier is `set_rules::Semantics`. X2 stays PARTIAL on
  that basis alone.
- **M2.** `execute_compute_dag` owns sequencing only (topo order plus abort). Each target's
  value is a consumer closure called per call, so "dependency → recompute" is PARTIAL, not
  ALREADY-HAVE.
- **M3 (soundness gap).** The production `normalize`/`simplify` (`set_algebra.rs:164-215`)
  is two-valued and applies complement rules unconditionally. The Kleene oracle exists only
  under `#[cfg(test)]` (`set_rules.rs:609+`). Any lane that can carry NULL is normalized
  under two-valued laws today.
- **M4.** "compute_dag → executed fold" holds in the IAM probe path only. The contract
  executor is the per-target closure route, so the contract-level grade is PARTIAL.
- **M5.** "why-provenance = union of input masks" holds for monotone queries only; under
  minus/NOT it over-reports.

## The operator's reflection, per atom (no bundle verdict)

| atom | verdict | evidence |
|---|---|---|
| class = signature (T,F,R,M,G) | PARTIAL | T, F, R reachable on ClassView. M (MethodSig / MethodTable) and G (ActionDef guard) sit on other surfaces, which OGAR doctrine wants. |
| A relations → relalg | single table: SUPPORTED · join / composition / closure: OPEN | relalg lacks D-RA-3/4 |
| B computed properties → fold | IAM stratum (Boolean, one relation): SUPPORTED-IN-CODE · other strata: OPEN | `dir-rmap-probe/tests/classview_dr.rs` |
| C polymorphism = masked dispatch by ClassId | OPEN | `ternlog_dispatch` precedent; probe P1 |
| D guards as Boolean masks | PARTIAL | computing a mask never authorizes (`ActionInvocation::commit`) |
| E NARS reasoning | as finisher: SUPPORTED (D-CMH-1 `NarsRevision`) · as semiring: OPEN | two scouts independently say not shown; probe P6 |
| CSE | SUPPORTED-IN-CODE | `SetProgram` lets / share |
| fusion | OPEN | D-CMH-1 fused multi-sink folds |
| incremental | OPEN | `execute_compute_dag` recomputes every target |
| law table: Boolean / bag / topological | SUPPORTED-IN-CODE | `Semantics.truth`, `Semantics.population`, `compute_dag_topo_order` |
| law table: sum | PARTIAL | `Semantics.arith` modes |
| law table: min-plus | OPEN | probe P2 tests whether it lowers onto Boolean frontier folds |
| law table: NARS | finisher only | |
| biggest lever: compute_dag → RISC fold | IAM: SUPPORTED · elsewhere: OPEN | |

## Exploration map

**ADOPT-NOW** (test-only or a defaulted method; nothing changes behaviour):
- **X9 / M3:** extend the `set_rules` rewrite-equality loop to `Truth::ThreeValued` against
  the Kleene oracle (~20 lines). Either `NotTwoValued` refuses every complement rule, or the
  loop prints a counterexample against `normalize`.
- **X10:** a conformance helper generalized from `classview_dr.rs::check()` (~30 lines).
  Every declared realization must match the mask-risc reference fold, per stratum.

**PROBE** (each pre-registered by the falsifier with assertion, disable and silent twin):
- **P1 (X17+U4):** polymorphism as a classid-mask partition plus already compiled folds.
  Two classes, two `compute_dag`s, resolved once. Equals per-row dispatch on every row,
  with timing in ns/row. Green means virtual dispatch is a fold primitive.
- **P2 (X13+U10):** min-plus hop distance from iterated Boolean frontier masks, against
  BFS on the dir-sim nest. Green means `ComputeEdge` needs no aggregate field.
- **P3 (X4):** law-property flags derived from the `Refusal` catalogue
  (`fn licensed(&Semantics)`, ~15 lines). Predicted admit/refuse must equal actual for every
  Rule × Semantics pair.
- **P4 (X6+X8):** `Count > 0 ⇔ Any` on positive programs; at least one counterexample
  under minus/NOT.
- **P5 (X15/U9):** incremental recompute via forward reachability from a dirty
  `WideFieldMask` (`execute_compute_dag_dirty`, ~25 lines). Equals the full recompute.
- **P6 (U6):** NARS revision associativity and distributivity on a quantized grid. The
  semiring claim needs 0 violations.
- **P7 (X18/X19/X20):** a ClassView law suite plus a cross-provider coherence diff. Assert
  that the providers' classid overlap is non-empty first.
- **P8 (X11):** frozen codebook: Eq always holds; Lt only when order-preserving.

**PARK:**
- X3: candidate realizations stay consumer-side (the IAM `DR` array) per A2, until a
  second stratum needs them in the contract.
- X5: blocked on the frozen-codebook witness.
- X12: Soufflé `eqrel` closure (D-RU-6 Lab).
- X13: no aggregate field until P2 decides.
- X22: FCA.
- U8: fused folds, owned by D-CMH-1.

**SKIP:**
- X14: egg on the bind path breaks A1 (cost extraction is a planner). Offline normal-form
  generation only, never per binding.
- Any per-row realization choice.

**Smallest contract bridge, OPEN for the operator:** a defaulted
`ClassView::semantics(&self, ClassId) -> Option<set_rules::Semantics>` returning `None`
(~10 lines), fail-closed like `epistemic_reading`. The class never sets `universe`, which is
bind-time. This makes the law reachable per class from an existing type, with no new type.

## Not searched

- worst-case-optimal joins; FAQ §5–8 (representation and tractability);
- algebraic specification / initial algebras;
- the PG GiST internals beyond the opclass chapter;
- the NarsTruth source;
- whether mask-risc `differential.rs` already covers P4.

## Closeout

`STATUS: open | OUTCOME: exploration map, 2 adopt-now test probes, 8 probes, 1 contract bridge proposal; one soundness gap named (M3) | OPEN: operator picks which to take forward`
