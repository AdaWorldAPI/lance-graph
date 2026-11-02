# Plan: DT64 as the one method authority — census and ratchet (2026-11-02)

**D-id:** D-DT64-AUTH ([STATUS_BOARD](../board/STATUS_BOARD.md)).
**Source:** architecture review
[entry](../board/entries/2026-11-02-codegraph-dr3-dt64-ruff-architecture-review.md) §2, §7 P3.

DECISION (operator, 2026-11-02): DT64 / `0xFFFF::0xF###` is the canonical
method authority. SCOPE: every executable method call a production surface
exposes. BASIS: the review found four parallel method identities
(`dr_math_binding_probe::MethodSignature`, the tract-harvest `Registry`,
XLOG `enum Word`, ogar-loco `FnIndex`) and dispatch paths DT64 never sees
(`FoldDialect` `match`, r2il `procs`). REVISIT WHEN: a second production
lens executes words outside R2IL.

## 1. The rule
The four may stay as lab, frontend or execution adapters while:
1. none of them is a global method identity or an admission authority;
2. every production-exposed executable method call is bound and admitted
   through DT64 / F### (`BindStore::admit`);
3. internal opcodes stay implementation detail (a `FoldDialect` arm is not a
   method; the word that reaches it is);
4. each deviation is listed in the census with an explicit `LAB` status and a
   removal path;
5. a new undeclared production word, and a stale exception, both fail the
   census test.

No full migration now; nothing here moves an adapter.

## 2. Census (one test, one committed list)
- Inputs, each read from the code that owns it, never restated: the R2IL
  polyfill words (`lance-graph-r2il/src/polyfill.rs` `POLYFILL`), the XLOG
  words (`lance-graph-xlog` `enum Word` and `lens.rs` declarations), the
  `BindStore` declarations each lens makes (`declare_algebra` /
  `polyfill_arities`).
- Check: every production word has a DT64 declaration at its lens position;
  every DT64 declaration names a word that exists.
- Exceptions: `.claude/board/dt64-authority-exceptions.toml` (or a const
  table in the test; decided in the implementing PR), one row per deviation:
  surface, status `LAB`, why, removal path. Exact match in both directions:
  an undeclared word not listed fails; a listed exception that no longer
  deviates fails.
- Home: a test in `lance-graph-r2il` or `lance-graph-dt64`, whichever already
  depends on both inputs without a new edge (decided by reading the
  dependency graph in the implementing PR, not here).

## 3. Falsifiers
- C1: adding a polyfill word without a DT64 declaration turns the census red
  (disable: drop the declaration loop → green with the undeclared word, red
  restored on revert).
- C2: listing an exception that no longer deviates turns it red.
- C3 silence: on the current tree, with the four adapters listed as `LAB`,
  the census is green (the ratchet is not a guard that fires on everything).

## 4. Not in this plan
Migrating `MethodSignature`, the tract `Registry`, `Word` or `FnIndex`;
executing anything through DT64; the `INVOKES_WORD` association (that is
D-CODEGRAPH-0-BIN §8.3).
