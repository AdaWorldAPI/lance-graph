# D-V4-FOLD-MATRIX — where each optimisation mechanism belongs

> Companion to `D-V4-FOLD-LAB.md`. Status per row: [G] read/measured in code,
> [H] literature or declared code, [S] reasoned. "New opcode?" means a new
> SEMANTIC op in V4; physical kernels below V4 do not count.

| Mechanism | Semantic change? | New opcode? | Candidate layer | In tree today | Grade |
|---|---|---|---|---|---|
| Dense ↔ sparse ↔ run carrier (GraphBLAS / Roaring) | no | no | fold optimizer (BUNDLE) | dense only; sparse forbidden by quack R1 | G: ordinals lose to a gated dense gather (bundle_probe) |
| Push ↔ pull VIA (Beamer / Ligra / GraphIt) | no | no | fold scheduler | no multi-hop executor | H |
| Late materialisation (DuckDB / SLM) | no | no | already in `Pred{under}` | yes, word-level | G |
| Survivor gating | no | no | lowering choice (`lower` vs `lower_fused`) | `Pred` only — **`Gather` is never gated** (the dominant cost) | G |
| Gunrock invalid markers | no (idempotent folds only) | no | scheduler + legality rule | no | H |
| CSR5 / SELL-C-σ / merge-path | no | no | backend kernel | no | H |
| Yannakakis semijoin reduction | no | no | fold optimizer (rewrite order) | semijoin exists | H |
| WCOJ / LFTJ | no | maybe a physical multi-way intersect | backend operator | no | H |
| Factorized multiplicity (Kùzu) | **yes if misapplied** (illegal for count-distinct across factors) | no (fold rule) | **semantic demand** chooses; fold rule executes | probe `TwoFoldsAndDot` only | G (probe) |
| BitWeaving / ByteSlice | no | no | backend lane encoding | no | H |
| Morsel / vector-at-a-time | no (mergeable partials) | no | scheduler | `execute_extent` | G/H |
| JITSON specialisation | no | no | backend (optional) | not wired; reduces no dynamic work | G |
| CubeCL | no | no | backend target | no | H |
| MLIR canonicalisation | no | no | pre-lowering normal form | missing | S |
| R2IL fold band | no | — | V4 spelling | test-only reader; mirrors mask-risc | G |
