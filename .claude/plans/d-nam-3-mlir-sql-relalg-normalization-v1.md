# D-NAM-3 — MLIR indexing / SQL / RelAlg normalization boundaries

**2026-10-25 • PROPOSED / ENGINEERING SPEC, DOCS ONLY.** Parent [EXO-152](https://linear.app/exo-red/issue/EXO-152), [D-NAM capstone](d-nam-native-algebra-dto-capstone-v1.md). Owners: [EXO-86](https://linear.app/exo-red/issue/EXO-86), [EXO-115](https://linear.app/exo-red/issue/EXO-115), [EXO-126](https://linear.app/exo-red/issue/EXO-126), [EXO-127](https://linear.app/exo-red/issue/EXO-127).

## Normalization without a universal new AST
MLIR `linalg.generic` carries explicit iteration types, indexing maps, and scalar region. A native method consumes the **same meaning** only when legal coordinate mapping and scalar arithmetic are identified. MLIR is an *external dialect*; no MLIR optimizer/interpreter in Quack or RelAlg. SQL has its own K3 and Bag multiplicity; normalization to Boolean `Rel` is legal only for a proven two-valued/Distinct subset (D-RA-2). Do not replace value-bearing algebra with `Rel` set predicates.

## Four comparison cases
| Case | External reading | Native anchor | Non-negotiable guard |
|---|---|---|---|
| Predicate | `arith.cmpi sgt`; SQL WHERE `i32 > t` | `Rel::Atom`/native `gt_i32_to_mask` | signed scalar compare, NULL policy, full row domain |
| Select | `arith.select`, ONNX Where | ndarray `blend_i32` | writes value lane, not `Keep` mask; type/shape |
| Reduction | `linalg.generic` reduction, SQL SUM, ONNX ReduceSum | RelAlg `Measure::Sum` or existing native sum | multiplicity, empty input, i32→i64 vs external dtype, reduction order |
| Numeric contraction | MLIR `linalg.matmul` maps (m,k),(k,n),(m,n) | existing ndarray numerical GEMM, **not** Boolean `relcomp::compose` | numeric sum/product, FP order, full k support |

For the numeric contraction, DR can optionally select a valid support/tile *only after* a proof all required contributions survive. `0*Inf = NaN` defeats naïve zero pruning, signed zero and reassociation affect bit parity. A mask selected by physical focus is not a semantic `WHERE` exclusion.

## Method mapping lifecycle
1. Parse/version-check MLIR/SQL syntax **outside** hot execution; maintain indexing map and parallel/reduction loop distinction.
2. Match a small known form to an existing native method ID and constraint set; refuse unknown effects/scopes or arbitrary region side effects.
3. Normalize only lawful Boolean set expressions in `contract::set_algebra`; route through relalg/Quack's existing preparation, not a new syntax tree.
4. Execute the validated numeric region through ndarray methods; record logical output indices and materialization semantics.
5. Outside these cases return typed refusal; no "expression accepted" inference from parser success.

## Rewrite admission
- Boolean algebra e.g. `A ∧ (B ∨ C)` may admit associative/De Morgan rewrites under the **same** Boolean truth domain and sorted set semantics.
- K3 SQL UNKNOWN requires Kleene-aware rules; Not over NaN comparisons is not automatically a lawful `Rel::Not` transformation.
- SQL Bag multiplication/aggregation is not Boolean reachability; duplicates must survive.
- IEEE FP `(a+b)+c` ≠ `a+(b+c)` bit-for-bit; no Egglog proof of floating-point reassociation inherited from Boolean composition.
- MLIR regions with writes/control-flow effects must preserve alias and order; off-scope operations REFUSE.

## Implementation/test handoff
Audit existing `contract::set_algebra`, `set_rules`, `relalg::normalize`, `quack::Filter`, `lance-graph-r2il`, and [EXO-115](https://linear.app/exo-red/issue/EXO-115) before editing; only the existing Res-expressible subset is eligible. For every admitted canonical method, add an independent scalar/SQL/MLIR reference fixture, one adversarial NaN/NULL/Bag counterexample, and an explicit refusal. No new AST, new JIT, or float math in `Rel`.

**Falsifier:** if mapping MLIR `linalg.generic` needs arbitrary region execution at runtime or SQL NULL collapses to false, the method is **not** admitted. Split into an unadmitted dialect case rather than forcing the shared normalizer.
