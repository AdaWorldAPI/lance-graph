# D-V4-ISA-LAB — decision table

> The full analysis is `D-V4-FOLD-LAB.md` (§2 ISA assessment, §3 lift,
> §4 Quack boundary, §5 coverage, §6 JITSON, §13 CubeCL, §14 MLIR, §17
> falsifiers). This file holds the ISA-lab brief's own decision table.

| Question | Result |
|---|---|
| Is R2IL plausibly the V4 ISA? | **PARTIAL** — ISA-shaped table (no domain names, meaning in operands, IAM lifts with no new op); content is mask-risc's physical vocabulary, one test-only reader, classid ignored |
| Can Fold remain a reading/lens? | **PARTIAL** — one table under two concept ids is declared and tested; the reader must use the classid to pick the stack value type, which it does not |
| Can Quack lower losslessly to it? | **PARTIAL** — 21/41 mask-risc variants exact, 7 by composition, 2 mismatched (Gather, ScatterOr), 11 missing; no Quack→R2IL lowering exists |
| Does JITSON add structural folding? | **NO** — bakes threshold/record size/top-k; prefetch, focus mask, CPU caps never emitted; no dynamic work removed |
| Can CubeCL be only a backend? | **YES** in model; buffer ownership (device copy) is a named membrane cost |
| Is another persistent IR required? | **NO** |

Killer case (#1305): demand is decided before lowering
(`lance-graph-cypher-quack/src/demand.rs`); the 2-hop count is refused, never
answered with the 3-node support. This holds for every backend because the
backend never sees the unresolved question.
