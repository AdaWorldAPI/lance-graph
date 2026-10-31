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
