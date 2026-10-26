# D-ONNX-INTAKE-0 — cold frontend bake-off and source-attested native subgraph

**Date 2026-10-26 · Status S0 DOCS ONLY / NOT RUN · Owner D-NAM [EXO-152](https://linear.app/exo-red/issue/EXO-152) · Coordination [EXO-50](https://linear.app/exo-red/issue/EXO-50)**

**Canonical plan:** [`d-onnx-frontend-intake-bakeoff-capstone-v1.md`](../../plans/d-onnx-frontend-intake-bakeoff-capstone-v1.md)  
**Future CC handoff:** [`d-onnx-frontend-intake-bakeoff-cc-handoff.md`](../../prompts/d-onnx-frontend-intake-bakeoff-cc-handoff.md).

## Decision proposal, NOT an implementation decision

Compare pinned Burn `onnx-ir` vs pinned Tract `InferenceModel→TypedModel` as **cold** ONNX semantic importers. Execute an original graph with ORT only as independent oracle; RTen secondary reference and Candle numerical reference optional. Importer coverage is not our native execution coverage. Score both with the *same* actual encoder artifact and independent source/protobuf metadata, not marketing operator counts.

Do not introduce ONNX protobuf, Tract, Burn, any interpreter or model graph into existing hot-path native crates. No ModelProto importer or BERT/Jina full forward pass is presently claimed. S1/S2/S3+ separately gated.

## Relevant existing contracts

- [#1754](https://github.com/AdaWorldAPI/lance-graph/pull/1754)/[#1758](https://github.com/AdaWorldAPI/lance-graph/pull/1758): native ONNX names and bounded aliases, not a graph parser.
- [#1753](https://github.com/AdaWorldAPI/lance-graph/pull/1753): native-first ownership; [#1744](https://github.com/AdaWorldAPI/lance-graph/pull/1744)/[#1750](https://github.com/AdaWorldAPI/lance-graph/pull/1750): source intake/refusal gates.
- [#1779](https://github.com/AdaWorldAPI/lance-graph/pull/1779)–[#1784](https://github.com/AdaWorldAPI/lance-graph/pull/1784): D-NAM typed method, ONNX/TFLite and Palette prepared numerical semantics.
- [#1798](https://github.com/AdaWorldAPI/lance-graph/pull/1798): **primary DT64 numerical-method leaf consumer**, including model digest, opset, tensor type/shape, calibration and computed result kind.
- [#1799](https://github.com/AdaWorldAPI/lance-graph/pull/1799): **D-DT64-Q0 canonical borrowed binding**; do not duplicate its receipt or supersede it with QB.
- [#1794](https://github.com/AdaWorldAPI/lance-graph/pull/1794): predicate slots are not ONNX node ids; [#1795](https://github.com/AdaWorldAPI/lance-graph/pull/1795): source owner/snapshot/ACL; [#1796](https://github.com/AdaWorldAPI/lance-graph/pull/1796): typed directed relation semantics and no MatMul=Boolean DR compose; [#1797](https://github.com/AdaWorldAPI/lance-graph/pull/1797): numeric score is not an observed business fact.
- [#1792](https://github.com/AdaWorldAPI/lance-graph/pull/1792) merged: fresh borrow can execute; escaped stale Keep cannot be trusted from recipe/cycle alone.

## Staged acceptance gates

| Stage | Deliverable | Must fail/refuse |
|---|---|---|
| S0 NOW | plan/handoff/board note, GitHub draft PR | no code/CI/prod changes |
| S1 future | cold dual-parser comparison, original node/metadata inventory, provenance table and dependency graph | unknown opsets, external-data misbinding, optimizer erasing identity silently |
| S2 future | raw model's node-weighted native-method coverage and typed refusal inventory | name-only coverage, missing dtype/axis or fallback-by-accident |
| S3 future | one *complete* static native subgraph with independent ORT/scalar oracle | unsupported node silently evaluated or third-party execution in hot path |
| S4/S5 future | model blocker numerical gates, eventual complete encoder forward parity and total cost | false whole-model claim, unauthorised Palette substitution |

No user source data, DTO writeback, new native IR, training/backward semantics, DT64 physical third axis, or soundness claim from an optimizer alone.

## Required handoff

When PR exists, send an **append-only EXO-50 comment** using the mandatory envelope. Reference PR and plan, #1794–#1799 and D-DT64-Q0, D-NAM/EXO-152, source/oracle boundary, proposed stages and authorization status. Keep issue description and other sessions' messages unchanged. The GitHub capstone, not the Linear comment, owns the detailed spec.

**Evidence status:** S0 plan authored; S1–S5 NOT RUN.
