# 2026-10-17 — Thinking tables wired to PRs #1447–#1596

**Status:** WORKING-MODEL (documentation only; no code changed).

`.claude/knowledge/thinking-tables-wiring.md` checks a model of thought tables
element by element against the plans, entries and crates of PRs #1447–#1596.
The model: 64 tables over one NodeGuid identity, writes isolated per table,
reads across tables by presence bits at one coordinate, provenance as the
written-by word, set algebra as the AST, a Datalog fixpoint inside a table,
FIFO order as the stratum order, and written-by queries as metacognition.

- **Carried (WIRED):** the 64 tables, write isolation, presence reads,
  written-by provenance (D-HL-5, D-HOLO H14c/H14d); independence by source
  bits (D-HL-3, pre-registered, not run).
- **Partly carried:** the set AST (one algebra in four spellings, hinge D-RA-2),
  the fixpoint (operand reuse and an external d′ driver; residue fixed point
  measured), the wave scheduler (`schedule_depth.md`), uses, metacognition.
- **New:** FIFO order as stratum order, a fixpoint inside one table, the
  written-by queries named as the metacognition surface, the table lifecycle as
  one unit.

OPEN: thoughts vs lanes as the word; session vs storage (D-HL vs #1597);
precedence vs fold on merged reads; derived writes vs split-tunnel rule 7 and
law A1; tables as strata under revision (Simpson); D-RA-2. Coverage gaps are
listed in the document (O8).
