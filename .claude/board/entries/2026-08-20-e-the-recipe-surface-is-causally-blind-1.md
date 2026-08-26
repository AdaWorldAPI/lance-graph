## 2026-08-20 — E-THE-RECIPE-SURFACE-IS-CAUSALLY-BLIND-1

**Status:** FINDING (Stage 2.6b — recorded, deliberately NOT patched; operator-
ruled scope). Falsifies the premise of the original Stage-2.6 brief, which
assumed an `CE64 → recipe/runbook/planner` path.

**That path does not exist.** Three independent checks:

| check | result |
|---|---|
| `lance-graph-contract` (owns `recipe_kernels`, `recipes`, `recipe_dispatch`, `materialize`) depends on `causal-edge`? | **No dependency.** It cannot name the type; the two textual hits are prose inside doc-comments. |
| `style_strategy` — the Stage-2 planner surface — touches a causal edge? | **None.** Its one `causal_*` import is `causal_witness`, an unrelated contract type. |
| What produces the `ThoughtCtx` the 34 recipes reason over? | `thought_ctx_from(&PlanContext)` — exactly two scalars: `free_will_modifier` → temperature, `features.estimated_complexity` → one candidate. `nars_hint` and `witness` are NOT read by it. |

So **recipe/runbook reasoning is causally blind with respect to CE64/V3.**

**Why this is recorded rather than fixed.** A V3 entrance in front of that
surface would rehydrate an edge nothing downstream reads, and the invariance
test would return `discordance = 0` for an entirely trivial reason — a green
result that means nothing. That is the same failure
`E-THE-FILTER-WAS-FILTERING-ON-THE-WRONG-PREDICATE-1` had just corrected, hours
earlier, in the same PR. **This is NOT a V3 migration defect**; it is a Stage-3
substrate-wiring question, and stuffing CE64 fields into `ThoughtCtx` now would
be adding semantic information the recipes do not have today — which Stage 3
explicitly reserves.

**The sharper statement the finding generalises to.** `ThoughtCtx` is not "the
reasoning state"; it is a **lossy projection of the reasoning substrate**, and
two independent measurements this session show the cost of that projection from
opposite sides:

- **Input side (this entry):** the causal/NARS/witness substrate exists and the
  projection does not carry it in.
- **Output side (`E-THE-FILTER-WAS-FILTERING-ON-THE-WRONG-PREDICATE-1`):** 17 of
  34 kernels are `Operational` and confidence-MUTE — they write `candidates` /
  `rung` / `temperature` / `beliefs`, and the dissent consumer observes only
  `confidence`, so their real effects are invisible to it.

**The 17 are not stubs, and must not be "fixed" by making them move
confidence.** Their silence means *the consumer projection is narrower than the
producer effects*, and rewriting them to move confidence would destroy exactly
the distinction this audit discovered. Stage 3 decides whether dissent becomes
(A) a multidimensional comparison over declared `writes()`, (B) per-capability
watchers, (C) a projection into a common epistemic space, or a measured
combination. **Not decided here.**

**Consequence for Stage 3's handoff — two distinct wiring problems, and keeping
them distinct is load-bearing:**

1. **Edge semantics.** `CausalEdgeV3` (addressed, semantic, durable) +
   `CausalEdge64` (hot compact NARS projection); the planner's `nars_engine`
   leg is now proven representation-invariant (Stage 2.6a).
2. **Program/recipe semantics.** 34 recipes × 36 styles operate on a narrower
   `ThoughtCtx` projection; the 17 mute kernels are the measured cost. How
   causal/NARS/V3/witness information enters the program surface **without
   turning every effect into confidence** is the open question.

Cross-ref: `TD-THOUGHTCTX-IS-A-LOSSY-PROJECTION`.

---

