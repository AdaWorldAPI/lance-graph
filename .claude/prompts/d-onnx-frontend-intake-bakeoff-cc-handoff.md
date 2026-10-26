# CC handoff: D-ONNX-INTAKE-0 — Burn onnx-ir vs Tract cold import bake-off

**Scope right now: read/review and separately gated plan only.** The capstone is [`d-onnx-frontend-intake-bakeoff-capstone-v1.md`](../plans/d-onnx-frontend-intake-bakeoff-capstone-v1.md). This document is a future S1/S2 implementation brief, not an instruction to create code in the S0 documentation PR or to merge it.

## Before touching code
1. Verify current main and PRs #1752, #1753, #1754, #1758, #1744, #1750, #1779–#1784, #1792 and #1794–#1799. Do not assume the six DT64 draft docs are merged.
2. Read the existing ONNX S0/S1 and D-NAM method plans. Locate existing `lance-graph-onnx`, `math-polyfill-probe`, Quack binding and ndarray numeric APIs. Search for overlapping experiments before adding any new crate.
3. Read `#1799` **canonical D-DT64-Q0**. The retired QB note is only a pointer. DT64 owns source-object semantics, not ONNX graph execution. #1798 is the relevant native numerical method leaf.
4. Pin and record candidate versions/commits, licenses, toolchains and transitive dependencies. Check Burn `onnx-ir` independently of the Burn execution backend and Tract `InferenceModel→TypedModel` without assuming that its optimized representation is source-faithful. No fetching live business DTO data.

## Stage S1/S2 (future, requires explicit separate approval)

- Prefer two independent small *cold* probes driven by the **same source model bytes**, not a universal importer adapter. Each outputs a comparable structured inventory of original source node identity, domain/opset/attributes, inputs/outputs, static/dynamic shapes, types, initializers (digest, raw vs external), graph metadata/function/training refusals and import warnings.
- Synthetic tiny model first; then one pinned real encoder artifact. Jina v2 small is a candidate only if model bytes are available and hashed. Otherwise choose a publicly accessible BERT ONNX model and record the change.
- With all optimizations OFF, compare source-node/edge counts to an independent raw ONNX/protobuf/ORT metadata view. With optimizations ON, require a provenance map or explicitly record every lost source identity.
- Derive a native crosswalk against `lance-graph-onnx`, ndarray/Quack/R2IL methods: DIRECT_NATIVE, COMPOSE_NATIVE, BOUNDED_NUMERIC, SOURCE_PARITY, REFUSE or UNVERIFIED. Record exact dtype/rank/axis/opset for each, not name-only matches. Compare distinct operation signatures **and** actual node frequencies. A blocked mandatory node blocks whole-model native execution.
- Test original protobuf metadata, `external_data` safety/path traversal, symbolic dims, sparse/missing weights, unknown opset, optional inputs, bool packing, source `f32` signed zero, int8 sentinel interpretation and optimizer provenance. One intentionally broken admission must turn its named test red.
- S3 only if S1/S2 show a credible complete native slice: compile `Greater→Where→ReduceSum` via existing native methods; compare with ORT and scalar oracle. Third-party runtime must not appear in the native hot path.
- DO NOT integrate Burn/Tract executor, write a second graph AST/IR, add ONNX ops to `Rel`, invent DT64 model slots, consume/emit unauthorized source facts, or automatically quantize everything into Palette256.

## Measurements and exit
Per frontend: dependency and compiler cost, parse/normalize duration, peak memory, raw original node count, preserved metadata, admitted native-node percentage, blocked paths, preparation and execution cost **separately**, independent differential results.

Decision `ADOPT_ONNX_IR` / `ADOPT_TRACT_FRONTEND` / `DUAL_FRONTEND` / `KEEP_EXTERNAL_ORACLE_ONLY` / `DEFER` only after a common-corpus test.

**S0 deliverable only:** reviewed plan, corrections with exact source anchors, DT64/D-NAM crossreferences, EXO-50 handoff. No production changes, code/tests not run, no autonomous merge. Create independent S1/S2 implementation PRs only after further operator approval.

### Linear coordination
Read EXO-50 append-only inbox first. Every message is a *new* comment with `MESSAGE-ID / UTC / FROM / SESSION-ID / TO / TOPIC / TYPE / REPLY-TO / REFS / BODY / ACK`. CC uses its actual full session ID. Never edit another session's comment or claim a proposed handoff as approval.

**Guiding sentence:** Importing ONNX is a cold provenance-preserving act. Native execution remains a separately admitted, typed mathematical recipe.
