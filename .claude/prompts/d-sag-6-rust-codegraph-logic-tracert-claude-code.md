# Claude Code handoff — D-SAG-6 Rust Codegraph → Mathematics → Logic → Tracert

**Status:** research / implementation handoff only. S0 is DOCUMENTATION, no code authorized. **Canonical plan:** [D-SAG-6](../plans/d-sag-6-rust-codegraph-math-logic-tracert-v1.md). **Coordination:** [Linear EXO-50](https://linear.app/exo-red/issue/EXO-50/agent-message-bus-gpt-xgrok-ccsession-id).

## Mission

Audit whether **Rust source semantics** can be reflected into a *typed*, versioned, read-only Lance-graph Codegraph and fed to the **existing** RelAlg/Quack/R2IL/egglog Math Foundry for:
1. detecting source-level higher-order pattern candidates through Datalog/Ascent;
2. separately verifying a narrowly admitted mathematical rewrite;
3. emitting a reproducible **Mathe → Logik → Tracert** from compiler source facts to a candidate native recipe;
4. optionally comparing a source patch and an MLIR lowering ONLY after review and separate approval.

Do not replace Rust MIR with MLIR, make Ruff a Rust semantics authority, or invent new CodeGraph/ProposedMethod/ProofRef systems. The model must preserve Rust evaluation/borrow/drop/effect semantics; syntax similarity is not equivalence.

## Must read before work

1. Root **CLAUDE.md** P0 architecture, nanosecond hot fold and cold-compiler reporting exception; check newest main SHA and active PR ownership.
2. SAG [capstone alignment](../plans/sag-capstone-alignment-v1.md); D-SAG-1 withdrawal in [software-awareness-v1](../plans/software-awareness-graph-v1.md); [D-SAG-2 Ruff closed loop](../plans/software-awareness-graph-v2-ruff-ona-closed-loop.md).
3. [Ruff ModelGraph plan](../plans/lance-graph-as-the-modelgraph-v1.md) and AdaWorldAPI/ruff crate graph: ruff_spo_triplet, ruff_r2il, ruff_cpp_codegen, ruff_spo_address. **Ruff has no Cargo edge into lance-graph.**
4. [DR Mathematical Compiler](../plans/dr-mathematical-compiler-v1.md), [Ascent/BYODS cold compiler](../plans/ascent-byods-dr-fold-polyfill-v1.md), [D-NFR](../plans/d-native-repr-contract-v1.md) and [D-NFC](../plans/d-native-first-dependency-correction-v1.md).
5. Source: lance-graph-contract ClassView/set_rules/set_algebra, Quack binder, R2IL router, RelAlg, mask-risc, ndarray SIMD; math foundry #1693/#1694, DR certification #1740, binding #1753/#1755; check current contents rather than relying on plans.
6. External actual-source audits, each pinned to a revision/toolchain before claims: rust-analyzer HIR/rowan, rustc MIR or Charon LLBC, Datafrog, Ascent, egglog, eggcc research, MLIR. Do not declare support from a landing page.

## Do Phase 0 now if a coding agent picks this up; STOP after report

**D-SAG-6-1 read-only capability mapping, no code or PR without operator go-ahead:**
- Record SHIPPED / LAB / PROPOSED / MISSING separately for parsing, rust-analyzer source identities, MIR/Charon export (toolchain compatibility), Rust effect/lifetime handling, Ruff crosswalk, DR binding, Datalog planning, egglog proof, independently checked certification, target codegen.
- Pin concrete example Rust sources with closures, trait dispatch and masks; extract only relationships actually witnessed. Name incomplete/unsupported operators. Distinguish callsite, possible dynamic callee and observed call.
- Identify smallest already-owned location for a read-only *projection adapter*; propose typed schema without adding a second store or method registry. Compare existing SAG/Quack/DR types before inventing an alternative.
- Determine which Ascent/Datafrog/Ruff input is **independent** of the candidate under test. No "oracle" based on the same helper.
- Inventory legal source edit path (rust-analyzer text edits or syn/quote only if justified) and optional MLIR path. No source mutation.
- Cross-check the plan's F01–F15 falsifiers. If the correct answer is to PARK (no new math, only aliases), say so.

## After explicit approval: staged probes, one narrow PR per stage

**D-SAG-6-2 graph projection:** read-only Rust source/HIR + MIR evidence into typed DR focused facts; coordinate/source seals, target/compiler revisions, captures, trait dispatch. F01/F04/F06/F07/F11 must fail red on injected defect and pass with correct schema. Do not materialize owned tuples in native hot folds.

**D-SAG-6-3 logic:** one pure predicate-filter-count fixture and one impure captured-closure negative fixture. Datalog/Ascent rules propose candidates; Datafrog or a truly independent Rust oracle checks finite relation answers; include alternative direct SIMD/mask-risc route. F02/F03/F08/F10/F15.

**D-SAG-6-4 math + Tracert:** canonical RelAlg law under exact domain; egglog candidate/replay where supported, effect-safe extraction only for admitted effects; a traceroute with source revision, HIR/MIR witness, rule, law, proof/checker status, lowering, target result and refusals. Reuse #1693/#1694/#1740 without claiming a full independent proof checker. F05/F09/F13/F14.

**D-SAG-6-5 opt-in codegen:** if independently admitted, write one minimal Rust source patch on a dedicated branch with operator approval; run rustfmt, cargo check/test and differential tests. Optional MLIR export is a comparative test, not required runtime. Record bit-exact or declared tolerance behavior; F12.

## Required negative controls

Same symbol label but different crate features; mutated/moved drop; predicate with a mutable captured counter; possible panic; non-monomorphized trait call; incomplete MIR or unsupported unsafe; source SHA change; changed DR focus; fake proof text; K3 UNKNOWN treated as FALSE; SQL BAG mistaken for DISTINCT; wrapping arithmetic treated as checked; FMA signed-zero/NaN divergence; no-approval attempted source write. Correct response is REFUSE or UNKNOWN, never an invented proof.

## Trace and report format

For each derived candidate output: source path + commit + rustc target/features; original span & callsite identity; HIR/MIR body + coverage; typed relation and DR frame; rule ID and premises; admitted law and domain; proof/checker evidence versus replay-only artifact; extraction cost/ABI; native kernel/binding; independent reference answer; measured cold/bind/hot components; refusal cause. Use source links, actual commands and real observations, no speculative speedup. Counterexample tests must be recorded before implementation and disable-red evidence preserved.

**IMPORTANT**: The operator ruled a cold out-of-fold compiler/oracle cost may be reported in microseconds as informational, not gates. Hot folds retain nanosecond/microsecond P0. A millisecond-scale in-fold result fails; never disguise it as useful runtime throughput. Avoid reintroducing tuple materialization, foreign gather or runtime Datalog interpreters.

## Handoff/coordination

- Github PR/board is canonical durable evidence; EXO-50 is **append-only coordination**. Post one new comment using its envelope for each meaningful result or blocking ownership decision. Do not edit earlier session comments or infer ACK from existence.
- D-SAG-6 belongs to the SAG continuation, Math Foundry and D-DRMC owners retain proof/compiler authorities, D-RMAP retains binding and scope, OGAR retains action authorization. No effectful DO method may execute as a learned candidate.
- Check parallel [#1759](https://github.com/AdaWorldAPI/lance-graph/pull/1759), which is only a board catch-up; avoid conflict with its STATUS_BOARD/INTEGRATION_PLANS corrections.
- Report no-code outcome after Phase 0; ask operator before implementing Phase 1+, editing existing source, changing Rust/compiler dependency pins, opening implementation PRs or merging.
- If creating an entry later, use **python3 .claude/tools/entries_index.py --write** then **--check**. Do not hand-edit generated README. Run plan_dids.py, append_only_gate.py, citation_decay.py and project-specific checks where applicable; mark NOT RUN if unavailable.

**Deliverable at this stage:** a falsifiable gap inventory and a recommendation of the smallest independent read-only adapter, NOT a new compiler, codegraph store or automatic Rust rewrite.
