# 2026-10-14 — D-MTP-0 — where Hyperon/MeTTa attaches to the masking-algebra polyfill

**Status:** MEASURED (engine comparison, v4 host, single whole-run timings) + VERIFIED-IN-CODE
(Hyperon fork `1ffdbd6`). Plan `.claude/plans/metta-polyfill-v1.md`. No crate, no code.

**Outcome:**
- MeTTa works as a declaration IR (rules, facts, queries as typed S-expressions, names resolved
  once to addresses), not as the execution IR. Execution: MeTTa Datalog subset → `lance-graph-relalg`
  → R2IL `FoldDialect` → mask-risc. Term construction, bags, grounded ops, types and state stay in
  Hyperon's unchanged interpreter.
- Seam: `hyperon_space::Space::query` (`hyperon-space/src/lib.rs:175`). A space queried directly by
  `match` gets the whole `(, …)` pattern (`stdlib/core.rs:163`); `ModuleSpace` splits it into single
  conjuncts first and panics on a non-`ModuleSpace` dependency (`space/module.rs:33-48`). So the
  adapter is a named space, never `&self` or a module dependency.
- Measured, same generated programs: two-hop join PeTTa 0.29 / 0.41 / 1.47 s vs Hyperon 0.11 / 5.9 s /
  panic at n = 10² / 10³ / 10⁴; recursive ancestor PeTTa 0.29 / 0.33 / 0.42 s vs Hyperon
  0.28 / 7.4 / 417 s at n = 50 / 200 / 800. Answers correct wherever an engine finished.
- New upstream panic: an expression with ≥ 1,024 children passed to a MeTTa function aborts in
  `TrieKey::start_expr` (`hyperon-space/src/index/trie.rs:539`, `TK_MAX_EXPRESSION_SIZE = 1 << 10`);
  1,023 children works. This stopped the n = 10⁴ join.
- Crate name: `lance-graph-metta` (one concern: front end + `Space` adapter); "cog" would mix in
  PLN, the miner, attention and Classic AtomSpace (AGPL).

**Open:** whether `match &self` can reach whole conjunctions without patching Hyperon; D-MTP-1..5
not authorized; the trie panic is not reported upstream.
