# D-SRAM-v1 — symbolic oracle cases and kill conditions

**Status:** PREREGISTERED ONLY (2026-10-19). These are specifications for future fixtures, **not executed tests, measurements or performance claims**. Companion: [integration plan](../../.claude/plans/souffle-ram-64-thought-stack-v1.md).

Each test must record: source/program revision, base seal/version, source query shape, category (set / bag / truth), expected canonical claim IDs and evidence roots, expected refusal where applicable, active thought high-water mark, number of fixpoint rounds, optimizer plan hash, and the disabled-control outcome. An oracle conflict must be reported, never voted away.

| ID | Fixture | Expected / falsifier |
| --- | --- | --- |
| SR-01 | Pure unary selection on one sealed frame; redundant conjunction, contradiction `p & !p` | Normalization yields exact same mask; contradiction is empty with no payload reads |
| SR-02 | Co-addressed `a(i) & b(i)` across two lanes sharing one sealed allocation | AND mask, exactly survivors read; poison absent payloads never touched |
| SR-03 | Equal ordinal values from *different* bake/version or schema seal | Alignment **refused**; same ordinal alone cannot establish identity |
| SR-04 | A two-hop `path(x,z) :- path(x,y), edge(y,z)` with no co-addressed layout | **Refuse** mask-only lowering; Datafrog/Soufflé may be oracles but not implicit runtime fallback |
| SR-05 | A canonical co-location/projection whose construction is independently verified | Same answers, explicit projection/seal costs; mutation invalidates stale mapping |
| SR-06 | Seeded ancestor chain requiring >64 semi-naive iterations but **one** speculative thought | Fixpoint completes (within explicit iteration budget), active thought high-water mark = 1 |
| SR-07 | Recursive cycle with duplicate derivations | Least fixpoint terminates; disabling dedup must trigger termination limit/falsifier |
| SR-08 | Magic-set specialization for a bound query | Answers AND support roots equal unoptimized source oracle; reduced work is measured, not assumed |
| SR-09 | Magic-set rewrite affecting another output or negation dependency | Either equivalent under proven query scope, or typed refusal; never silently lose proofs |
| SR-10 | MeTTa `BindingsSet` with two identical projected bindings from distinct rule paths | **Bag** has multiplicity 2 when the source semantics does; Datalog **set** is tested separately; no implicit dedup |
| SR-11 | `superpose`, `collapse`, `new-state`, grounded invocation or dependent-type operator | Routed to original Hyperon semantics or typed refusal per D-ASM; never approximated as a Boolean Datalog fact |
| SR-12 | Same evidence copied to two thoughts; NAL revision | Confidence does not increase due to duplicate/overlapping evidential roots |
| SR-13 | Independent evidence supporting same statement across thoughts | NAL revision can combine under existing truth law; source independence not equated with number of lanes |
| SR-14 | Contradictory candidate claims with different `NodeGuid`s | Both survive candidate set; merge does not overwrite one at same coordinate |
| SR-15 | Safe stratified negation with explicit closed-world scope | Order respects negative dependencies; FIFO cannot override prerequisite readiness |
| SR-16 | Cycle through negation across speculative thoughts | Compile/schedule refusal rather than unsound nonmonotone fixpoint |
| SR-17 | 63 live hypotheses + immutable givens, then request one more | Typed `BranchesFull`; T0 unchanged, no W alias and no unauthorized eviction |
| SR-18 | 100 serial hypotheses each closed with a replayable receipt | Slots recycled deterministically; historical logical IDs stable despite reused 6-bit slots |
| SR-19 | Two live hypotheses with unresolved independent consumers | Liveness checker refuses early reuse/squash; no speculative alias |
| SR-20 | Branch/rebase where a premise is removed | DRed or from-scratch oracle retracts dependent conclusion; retained alternative proof survives |
| SR-21 | Same presence but changed typed value across versions | `diff --semantic` detects value change; presence XOR alone must fail control |
| SR-22 | Pickaxe at strict historical `QueryReference::at(v,rung)` | Future-only evidence invisible; `NoDeps` cannot claim full causal closure |
| SR-23 | Change in thought Git ancestry but no change in proof roots | `blame` reports provenance, not causality from commit-parent relation |
| SR-24 | Promote recurring NAL/Datalog subproof as a chunk; later delete a supporting premise | Unfold receipts recover premises, policy and alternatives; stale chunk invalidates without losing independent proofs |
| SR-25 | Compile two equivalent eligible programs through MeTTa and Narsese fronts | Same canonical *semantic* results; R2IL bytes compared only when bag/typing/arguments are genuinely identical |
| SR-26 | Vary density 0.1/1/10/50% and selection aperture with fixed result oracle | Dense AND vs sparse seek vs range-fold crossover measured incl. compilation, masked I/O and terminal reads |
| SR-27 | Repeated recursive program reaching budget without fixpoint | Typed budget refusal, not silently truncated closure; no speculative slot leak |
| SR-28 | 1 million small resident folds versus one eligible durable semantic artifact | No per-fold Lance write; one-writer gate runs on elected nonempty artifact only |
| SR-29 | One-to-many relation that would require foreign fan-out | No mask/GroupReduce trick may undercount bag multiplicity; explicit refusal pending legal tuple carrier |
| SR-30 | The four-sentence rotating-ring ambiguity from #1604 | No invented coreference closure; an explicit identity witness changes the answer; rotation preserves justified IDs |
| SR-31 | Downstream thought reads a negated result from an unlanded branch | Dependency planner orders correctly or refuses; FIFO order alone is insufficient |
| SR-32 | Program compiled once, executed repeatedly on same sealed codebook vs new bake | Warm path does not reparse/recompile; stale compiled binding is rejected when schema/version contract changes |

**Independent oracles:** scalar row-by-row reference for address masks; `AdaWorldAPI/datafrog` for pure monotone Datalog; Soufflé for the supported standard Datalog fragment; Hyperon/PeTTa for their **evaluated** source programs and bag behavior; ONA/OpenNARS for NAL truth arithmetic. Compare compatible categories only; different NAL argument conventions require rule-shape mapping. Soufflé and Datafrog need not reproduce Hyperon's unsupported higher-order semantics.

**Mandatory negative controls:** force wrong bake acceptance (SR-03), replace join with AND (SR-04), drop bag counts (SR-10), double-count roots (SR-12), schedule negative dependency FIFO (SR-16/SR-31), reuse a live slot (SR-17/SR-19), omit DRed (SR-20), use XOR-only semantic diff (SR-21), use unbounded/latest historical horizon (SR-22), and drop chunk receipts (SR-24). Each mutation should make its targeted test fail before reporting a green integration.

**No numbers are benchmark results in this file.** Bounds such as 63 thoughts are architectural constraints, and row counts/iteration counts describe future fixtures. All benchmark results require raw, reproducible output with CPU tier, compiler/dependency revisions, comparison categories, compile/warm split and read-byte accounting.
