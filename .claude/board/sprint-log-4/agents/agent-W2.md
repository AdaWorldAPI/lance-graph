
## W2 Milestone Log — 2026-05-13

### M1: Sprint log read
Read `.claude/board/sprint-log-4/SPRINT_LOG.md`. Confirmed TD-Q2-STUBS-DEDUP-1 is P0.
Deliverable: `.claude/specs/td-q2-stubs-dedup.md`.

### M2: q2 repo structure confirmed via MCP (3 reads used)
- Read 1: `AdaWorldAPI/q2` root directory listing
  - Confirmed: `crates/`, `crates/stubs/`, workspace Cargo.toml, q2 HEAD SHA `dfe3df477282030b48312b74dc573fddd3660e0b`
- Read 2: `crates/` directory listing
  - Confirmed stubs are at `crates/stubs/` (NOT `crates/lance-graph` or `crates/q2-ndarray` as prompt suggested)
  - Stub crates: `q2-ndarray`, `graph-flow`, `notebook-query`
- Read 3: `Cargo.toml` (workspace root)
  - Confirmed canonical deps already declared: lance-graph (path, ndarray-hpc), lance-graph-contract, lance-graph-planner
  - Confirmed stub deps: `q2-ndarray = { path = "./crates/stubs/q2-ndarray" }`, `notebook-query = { path = "./crates/stubs/notebook-query" }`, `graph-flow = { path = "./crates/stubs/graph-flow" }`
  - `graph-flow` explicitly commented "Uses stub when rs-graph-llm repo is not available locally"

### M3: Lance-graph-contract type surface confirmed (local grep)
- `NarsTruth` at `lance_graph_contract::exploration::NarsTruth`
- `CrystalFingerprint::Vsa16kF32(Box<[f32; 16_384]>)` -- VSA carrier
- `OrchestrationBridge` at `lance_graph_contract::orchestration`
- `InferenceType` at `lance_graph_contract::nars`
- `OwlIdentity`, `OgitFamily`, `UnifiedBridge` (struct, not trait) at `lance_graph_callcenter::unified_bridge`
- `SpoQuad` NOT FOUND in any lance-graph crate -- open question OQ-1

### M4: Spec written
Path: `/home/user/lance-graph/.claude/specs/td-q2-stubs-dedup.md`
Size: 16617 bytes (~16 KB, 2x target but dense with diffs and code)
Sections: Inventory / Migration Recipe / Re-export Surface / Compile-Fail Tests / 3-Commit Cutover / Version-Skew Risk / 5 Open Questions

Status: DONE, no commit (main thread aggregates per protocol).
