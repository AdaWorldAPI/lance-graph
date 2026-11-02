# D-REASONING-CODEGRAPH-0 stage 1: XLOG × egglog over one DR64 relation

**Date 2026-10-31 · Status MEASURED (lab probe) · Crate `crates/xlog-relcomp-probe` (`src/recursive.rs`, `tests/recursive.rs`, `tests/shared_relation.rs`)**

**XLOG recursion on the CPU kernel.** XLOG's compiled transitive-closure SCC is admitted (`admit_closure`) and evaluated semi-naively on the DR64 register with the existing `relcomp::compose`. #1888 noted this was not covered by #1884. It matches BFS and a fixpoint walk of XLOG's own plan, and a 200-case differential. Work is exactly one row OR per derived pair. Non-linear and mutual recursion are refused.

**Shared relation.** One `edge` register under one sealed frame is read by XLOG's closure (directed reachability) and by egglog (equality with `ProofStore` proofs).
- egglog's equality equals the closure of `E ∪ Eᵀ`.
- egglog proof facts are bits of the same register, and lie inside XLOG's `support` mask for a directed answer.
- A removed fact invalidates the XLOG bind (stale frame) and is located in the old egglog proof by address, without re-running egglog.

**Traceroute.** Backward is `support(x, y)` and forward is `affected(u, v)`, both computed from the closure on demand; derivations are not stored. The one conversion is register bits to egglog fact text (114 B for 6 edges).

OPEN:
- timing;
- MLIR/DialEgg leg;
- SQL and Java legs;
- egglog proofs of a directed relation;
- XLOG recursion shapes beyond one linear closure.

## Correction (2026-10-31): the traceroute

The traceroute is a chain of stage identities (derivation, proof, binding, execution, replay), each referenced by an immutable identity, storing no intermediate population (`dr256-where-capstone-v1.md` §C.5). `recursive::trace` rebuilds one node path; it is not that traceroute, and a per-round delta-mask store would violate §C.5. `support` and `affected` stand as mask selections. Graph work is mask selection, not traversal. See D-DT64-7 F8 and O6.

## Erratum (2026-11-02, appended; the text above is unchanged)
**Scope:** the bullet "egglog's equality equals the closure of `E ∪ Eᵀ`" under **Shared relation** (line 8).

**Correction:** that equality follows from the probe's explicitly authored egglog rule `(rule ((edge a b)) ((union a b)))` (`crates/xlog-relcomp-probe/tests/shared_relation.rs:93-96`), which unions along every edge. It is a property of that rule set, not of egglog equality in general. A directed codegraph CALLS edge (or any directed code fact) does not imply equivalence. The test file states the same (`shared_relation.rs:18-19`: XLOG's closure is not an equality).

**Source:** Semantic Grammar Recovery map (`entries/2026-11-02-coresearch-semantic-grammar-recovery.md` §4, §8), operator decision S5.
