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
- New panic, separate from the id-decoding fix already on the fork (`1ffdbd6`, branch
  `fix-trie-key-value`, on which these runs were taken): an expression with ≥ 1,024 children passed to a MeTTa function aborts in
  `TrieKey::start_expr` (`hyperon-space/src/index/trie.rs:539`, `TK_MAX_EXPRESSION_SIZE = 1 << 10`);
  1,023 children works. This stopped the n = 10⁴ join.
- Rust Datalog oracle: datafrog 2.0.1 from the fork `AdaWorldAPI/datafrog` (v4, best of 5) runs join 10⁶ in 65–69 ms (65–69 ns/pair) and the
  seeded ancestor query 10⁶ in 0.29 s; join_10000 in 0.5 ms against PeTTa 1.47 s and Hyperon > 600 s.
  Oracle and comparison bar only, never a production engine (hash/sort joins beside mask-risc).
- Crate name: `lance-graph-metta` (one concern: front end + `Space` adapter); "cog" would mix in
  PLN, the miner, attention and Classic AtomSpace (AGPL).

**Open:** whether `match &self` can reach whole conjunctions without patching Hyperon; D-MTP-1..5
not authorized; the expression-width panic has no fix yet.

Addendum 2026-10-16: "the expression-width panic has no fix yet" was wrong when the next session re-read the fork. ⊘ Corrected 2026-10-16 (the fork was read, not the board): TWO fixes exist on `AdaWorldAPI/hyperon-experimental` — `1ffdbd6` (`fix-trie-key-value`, id decoding above 1023) is upstream PR trueagi-io#1107 (`refs/pull/1107/head` == `1ffdbd6`, open against upstream main `3f76dc4`), and `5c9afd4` on `ccr-9f3b8075-74u47l-trie-expr-width` (2026-10-14; `b5ae248` adds the failing ≥ 1,024-children tests first, `5c9afd4` splits the `TrieKey` value bits so expression width is bounded by trie recursion depth, ~1,100 children debug / 2,048 release on a 2 MB stack, not by `TK_MAX_EXPRESSION_SIZE`). Neither is on the fork's `main`; the lance-graph pins (`rev = 3f76dc4`) therefore still carry the panic. What waits on the operator is merging `5c9afd4` to the fork's main and/or upstreaming it, not writing the fix.
