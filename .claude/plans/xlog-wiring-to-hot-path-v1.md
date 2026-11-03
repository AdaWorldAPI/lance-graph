# XLOG wiring gaps to the hot path — v1 (D-XLOG-WIRE)

**Status:** PLAN. Gating input for D-CODEGRAPH-0-BIN B3 (operator, 2026-11-03: B3 waits until strings are outside the hot path and this list has reached the hot path; any remaining string comes off at the codebook label, the "wardrobe").
**Sources:** `.claude/board/entries/2026-11-02-coresearch-xlog-souffle-crepe-dr3-archaeology.md` §1–§3, re-verified at AdaWorldAPI/xlog `main` `84da1a8` on 2026-11-03 (citations below); today's findings.
**Rule for every row:** a compile result reaches lance-graph as DATA through the existing admission gates. No import of XLOG's runtime, CUDA executor, or a second planner (archaeology §2.7, SKIP list).

## The list

Each row gives the gap, where it is lost, the smallest wiring step, its falsifier, and whether it can run in a CPU-only container.

**W1. Magic-set report discarded.**
- Lost at: `xlog-logic/src/compile.rs:1496` and `:1511`, both `drop(rewritten.report)`.
- Wiring step: carry `MagicSetReport` on the compiled plan; admission refuses a plan whose report is `Declined` when the caller required `on`.
- Falsifier: reach/far (fork PR #7) yields a plan with a `Declined` report, and the no-`far` twin yields `Applied`.
- CPU-only: compile only.

**W2. Edge polarity dropped.**
- Lost at: stratify → lower (archaeology P-A3).
- Wiring step: a polarity record per rule body literal on the plan.
- Falsifier: P-A3, a negated literal whose polarity a consumer must see.
- CPU-only: compile only.

**W3. Join estimates discarded.**
- Lost at: estimates are baked into the tree shape; `RirMeta` is schema-only.
- Wiring step: the optimizer's chosen estimates travel on the plan as data.
- Falsifier: the triangle pairing probe (`xlog-relcomp-probe/examples/triangle_pairing_baselines.rs`) reads the plan's estimates and matches `optimizer.rs:3395-3404`.
- CPU-only: yes, the lab probe.

**W4. Degree statistics have no producer** (found 2026-11-03).
- Lost at: no production caller of `add_prefix_degree`; the only caller is `xlog-integration/src/bin/wcoj_phase_report.rs:347`, with hardcoded values. `PrefixDegreeStats` holds only an average and a maximum.
- Wiring step: a per-key degree statistic maintained on insert, and a producer on the insert path. On the cube that path is `Cube::new` intake and `Cube::add_edge` / `DeltaLog` (`lance-graph-xlog/src/cube.rs:250`, `:742`). Measured cost in the lab: 1.3–2.7 ns per insert, 96 KiB.
- Falsifier: exact-degree pairing regret 1.000 against the estimate argmin, on the probe corpus.
- CPU-only: yes, on the cube. (The first version said the insert path was CUDA-only; that holds for `xlog-runtime`, not for the cube executor.)

**W5. `RirNode::Fixpoint` dormant.**
- Where: production compile never constructs one. The only construction is `optimizer.rs:4672`, inside `#[cfg(test)]`. The runtime handler exists (`xlog-runtime/src/executor/node_dispatch.rs:280` → `execute_fixpoint`).
- Wiring step: decide whether recursive SCCs lower to `Fixpoint` (one owner for semi-naive) or the node is removed (the fork's no-dead-code rule).
- Falsifier: P-A4a.
- CPU-only: compile only.

**W6. Semi-naive recursion has several owners.**
- Where:
  - XLOG runtime: `Executor::execute_recursive_scc` (`xlog-runtime/src/executor/recursive.rs:445`), CUDA (`diff_gpu`, `union_gpu`, `union_many_gpu`).
  - `lance-graph-xlog`, mask-native and CPU: `Cube::closure` (`cube.rs:669`, ≤ 64 positions, `FixpointBound` cap) and `wide::closure_wide` / `wide::refresh` (`wide.rs:395`, `:560`, block presence, retire-aware). Both cover `edge⁺` of one binary relation only.
  - `xlog-cube` (fork): general recursive SCCs with its own host loop over owned rows, `BTreeSet<Vec<u32>>` with a `contains` filter and no cap (`xlog-cube/src/lib.rs:474–500`). D-STAUNEN-5 already names this shortcut.
  - The relcomp gate and others (archaeology §1).
- Wiring step: `lance-graph-xlog` owns the delta. The general SCC round becomes `delta' = new andnot full` on cube masks, generalizing `closure` beyond `edge⁺`. The other copies are compared by folds, then removed.
- Falsifier: P-A4b, all copies agree before any is removed; the existing three-way C0 parity (`xlog-cube/tests/closure_parity.rs`) is the starting point.
- CPU-only: yes.

**W7. `XLOG_LENS` (`0xFFFF::0xF100`) declared, never executed.** This is the polyfill-adjacent gap.
- Where: `lance-graph-xlog/src/lens.rs:33`. `lance-graph-r2il/src/binding.rs:296` refuses every lens except `POLYFILL_LENS` (`F000`) and R2IL-fold concepts.
- What it needs: the words (`functor/3`, `arg/3`, `=../2`) read helper relations the XLOG compiler emits as facts. They are column-reading words, not scalar folds, so r2il needs a relational word execution (a mask fold over the helper relation's lanes).
- Wiring step: an executor decision (operator), then one word end to end.
- Falsifier: P-I1, F100 executes through `BindStore` and `prepare` and agrees with XLOG's helper rows.
- CPU-only: yes once decided.

**W8. Competing plan identities.**
- Where: `MethodSignature`, the SHA-256 lowering digest, and the FNV `plan_id` (`binding.rs`, `plan_id`).
- Wiring step: DT64 is the one identity authority (D-DT64-AUTH); the other two become derived or removed.
- Falsifier: P-I1 / the census ratchet.
- CPU-only: yes.

## Text on the hot path (the precondition named with this list)

| site | state |
|---|---|
| egglog-r2il `Step::PushVar(String)`: a name lookup on every run | FIXED 2026-11-03: names resolve at lowering; `Step` is `Copy` |
| codegraph `Symbols::intern` on open and refresh (2.04 ms of the 2.5 ms changed-crate refresh) | B3; design already in `d-codegraph-0-binary-lanes-v1.md` §2 (`SYM_HASH` u64 + `SYM_OWNER`; labels stay in the KV section, the wardrobe). Deferred behind this list |
| r2il `opcode()` matches a label against `R2ILVocabulary` names | cold (`prepare`, once per method); not hot |

## Order

1. W1 and W5. Both are fork-side and compile-only, and each closes a loss with a known falsifier.
2. W7 once the executor shape is decided.
3. W3, then W4.
4. W2, W6 and W8.

Then B3. Fork changes follow the fork's own rules: an atomic commit, a push, and an integration request to the lead. Local runs are allowed (operator, 2026-11-03).

## Executor: the cube, not CUDA

The route is already decided: `d-xlog-native-v1.md` Amendment E (#1910, operator 2026-11-01). XLOG's RIR kinds are assimilated into our algebra in `lance-graph-xlog`; XLOG's caller API (`pyxlog`, `xlog-cli`, `xlog-integration`) is polyfilled back onto it; the fork depends on `lance-graph-xlog`, never the reverse; XLOG's runtime and its `CudaBuffer` plumbing are not ported. Tiers, in order: (1) assimilate into `lance-graph-xlog`; (2) polyfill or migrate the callers; (3) only for a kernel with no algebra counterpart, port it from CUDA to CubeCL over ndarray. "Replace CUDA" (operator, 2026-11-03) is executed through these tiers, not by a DR³ backing under `CudaKernelProvider`.

The reactor swap is `d-xlog-native-v1.md` Amendment F: the interface callers see (`Executor`, `CudaKernelProvider`, `CudaBuffer`) stays, and the resident cube is the machine underneath. The reactor is built; the device-less provider / `CudaBuffer::View` that connects the callers to it is the open channel.

`xlog-cube::CubeExecutor` is the tier-1/2 executor today: it takes the same `ExecutionPlan` as `xlog_runtime::Executor` and runs it on `lance_graph_xlog::cube::Cube`. Its refusal census is the gap list.

Measured 2026-11-03 (`cargo run --example coverage` over `xlog-integration/tests/*.rs`, fork `main` `84da1a8`): 188 programs, 230 rules, 140 admitted, 90 refused. First refusal reason per rule:

- 33 reads a relation that is not binary
- 17 filter other than `col == const`
- 12 computed or out-of-range output column
- 7 Join body
- 6 join key outside its input
- 5 Project over Project
- 5 XLOG word `0xFFFF :: 0xF100`
- 3 Project over GroupBy
- 2 Project over Join

Admitted is not correct: `coverage` gives every base relation as empty, and a rule counts under its first refusal only (D-STAUNEN-5 plan, "Relation to the xlog-cube coverage work"). That plan also carries the agreed order: projection composition, filter algebra, resident execution, computed columns, F100, N-ary relations.

Amendment E's open item, the caller-API census, is partly answered by a count of `provider.<method>(` calls per crate, multi-line chains included (2026-11-03): pyxlog 50 methods / 234 calls, xlog-integration 24 / 105, xlog-cli 8 / 14. pyxlog calls the provider directly for ILP, DLPack and the semantic engine, so its caller surface is wider than the executor. xlog-prob (426 kernel launch sites) and xlog-solve (27) launch SAT, CNF, circuit, PIR, cache, weights and Monte-Carlo kernels through `provider.device()`; these have no algebra counterpart and are tier-3 candidates.

Limits of the cube that bound this work: `xlog-cube` builds a single-tile cube (`Width::W256`, at most 256 ids; `xlog-cube/src/lib.rs:272`), while `Cube::new_tiled` and `closure_wide` (used by `lance-graph-codegraph`) span many tiles; relations are binary; the cube has no union or difference view.

## Decisions only the operator makes

- **W7:** how a column-reading word executes on DR³. Either a mask fold over the helper relation in r2il, or F100 stays declared-only and the helper relations are plain data reads.
- **W5:** lower recursion to `Fixpoint`, or remove the dormant node.
