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
- Wiring step: a per-key degree statistic maintained on insert, and a producer on the runtime insert path. Measured cost in the lab: 1.3–2.7 ns per insert, 96 KiB.
- Falsifier: exact-degree pairing regret 1.000 against the estimate argmin, on the probe corpus.
- CPU-only: no. The runtime insert path is CUDA; acceptance needs an authorized GPU run.

**W5. `RirNode::Fixpoint` dormant.**
- Where: production compile never constructs one. The only construction is `optimizer.rs:4672`, inside `#[cfg(test)]`. The runtime handler exists (`xlog-runtime/src/executor/node_dispatch.rs:280` → `execute_fixpoint`).
- Wiring step: decide whether recursive SCCs lower to `Fixpoint` (one owner for semi-naive) or the node is removed (the fork's no-dead-code rule).
- Falsifier: P-A4a.
- CPU-only: compile only.

**W6. Semi-naive deltas are runtime-only and CUDA-only.** lance-graph separately re-implements semi-naive recursion 4+ times.
- Where: XLOG runtime; lance-graph `lance-graph-xlog`, relcomp gate, and others (archaeology §1).
- Wiring step: one delta owner on the resident path; the copies compared by folds and then removed.
- Falsifier: P-A4b, all copies agree by folds before any is removed.
- CPU-only: lance-graph half yes.

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
3. W3, then W4 (W4 needs the GPU acceptance run).
4. W2, W6 and W8.

Then B3. Fork changes follow the fork's own rules: no local tests or runs, an atomic commit, a push, and an integration request to the lead. Runtime acceptance is an authorized GPU run, never a local experiment.

## Decisions only the operator makes

- **W7:** how a column-reading word executes on DR³. Either a mask fold over the helper relation in r2il, or F100 stays declared-only and the helper relations are plain data reads.
- **W5:** lower recursion to `Fixpoint`, or remove the dormant node.
