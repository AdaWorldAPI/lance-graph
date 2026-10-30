# D-MIST-NATIVE-0 — Mistaber's Kripke holdings as a least fixpoint over DR64 compose

**Date 2026-10-30 · Status LAB PROBE · Crate `crates/mistaber-kripke-probe` (standalone, not a member) · Goldens `AdaWorldAPI/mistaber-rs@f163fa7` (Clingo over `BrainyBlaze/mistaber@b50bf33`) · Plan `.claude/plans/d-mistaber-native-v1.md`**

Upstream `kripke_rules.lp` is stratified: `holds` is recursive through `holds_via(inherited(Parent), …)`, and every negation (`overridden`, `asserts`, `endorses` at the child) is over inputs. So `holds` is the least fixpoint `H = H₀ ∪ ((H ∘ children) ∧ ¬Block)`, one existing `relcomp::compose` plus an AND-NOT per step. Rule and origin lineage are the same fixpoint gated by `H`. No new algebra; relalg (single-table selection, no `μ`) was the wrong host.

All 10 goldens reproduce exactly: F1-modes, F1-reversed, F2-cycle, F5-gates, C-mammal-C2/C4/C6/C6i and E-xlog-plain/override. F3-naf (even negative cycle over `asserts/2`) is refused at export and stays with Clingo. 5 disable runs fail as expected. One guard, the `holds` gate on lineage, is also enforced at render time, so no golden can catch its removal; it has its own unit test. F5-gates was added because no earlier fixture exercised the world masks or that gate.

Holdings are one lens (`proposition, world, mode, immediate_rule, source_rule`), and each predicate is a `WideFieldMask` selection over it. `holds/2` needs a **Set** projection: on the F1 diamond the lens has two rows for `holds(p,d)` and the bag reading fails the golden. The bag/set split was sent to XLOG for the #1866 crosswalk (EXO-50).

**OPEN:** a production home for `compose` (blocks the `lance-graph-mistaber` facade); axes past 64; lineage into `SupportLedger` as `EvidenceSourceId`/`TextAttested`; intake from compiled HLL (expected refusals, `mistaber-rs#3`/`#4`); the upstream licence before fixture text enters a production crate. No timing taken.
