# 2026-10-11 — NAL-1..9 × 34 recipes against OpenNARS and ONA (D-NAL-AUDIT-0)

**Status:** MEASURED where a command is named. Tests only, no production change.
Full audit: `.claude/knowledge/nal-recipe-audit/README.md`.

```text
cargo test -p lance-graph-planner --test mcp_receipt_probe
cargo test -p lance-graph-planner --test nal_revision_oracle
```

## Findings

- **No recipe reads a prior result.** All 34 `Tactic::apply` bodies work on
  scalar `ThoughtCtx` fields. 5 pass their own unit tests; none has parity
  evidence against a NAL reference. VERIFIED-IN-CODE (`recipes.tsv`).
- **MCP (recipe 10) is history-blind** (D-NAL-MCP-0). Its verdict depends on
  confidence and free energy only. A calibrated and an overconfident past
  get the same verdict. A test-local receipt reader over production
  `mul::compass::LearningLoop` separates them, abstains with no receipts, and
  ignores a repeated receipt. `LearningLoop` alone counts the repeat.
  MEASURED; 5 disable runs red, including one that made MCP read `beliefs`.
- **Revision against OpenNARS output** (D-NAL-REV-0). Three points whose
  expected values are OpenNARS for Research's printed output (commit
  `4d1bdb0`): laws A, C, D, E reproduce all three; B16 one (its 0.012 and 0.018 misses exceed
  the one-code u8 allowance); B1 and LIN none.
  B1, the replay path's law, keeps the same confidence whatever the premises.
  LIN can end below the stronger premise. MEASURED; 3 disable runs red,
  including one on production `nars::truth::TruthValue::revise`.
- **Executed steps are dropped.** `wave_dispatch::dispatch_thought` binds
  `run_recipes`' `Vec<ExecutedStep>` to `_steps`. VERIFIED-IN-CODE.
- **Schedule depth is not always 1.** 64 of 256 initial `ThoughtMask`s give
  depth 2 (Candidates known, Sd unknown; LSI writes Sd). The pin at
  `rung_schedule.rs:321` samples 4 masks and says "immer". MEASURED by a
  full scan (temporary test, deleted).
- **The engines are not one reference.** They disagree with each other and
  with Wang's 2009 spec on compound-term confidence; ONA's `{SELF}`
  fixtures are goal-directed behaviour, not self-inspection; OpenNARS for
  Research implements mental operators. OpenNARS's Maven build is BLOCKED
  (retired snapshot repository); a direct `javac` build works.

## Decision

Narsese stays an external test format; both engines run outside the build
at pinned commits. Smallest proposed contract (not implemented): a receipt
a recipe can read. Full basis in the knowledge README.

```
STATUS: measured | OUTCOME: 2 test-only probes (8 tests, 8 disable runs red);
34-recipe and 115-capability tables; revision laws anchored to an outside
engine; MCP shown history-blind | OPEN: D-RPF-CONF-0 G2 (which law the replay
path emits) now has outside evidence but no decision; LIN mislabelled as
standard revision; rung_schedule depth doc overstates; receipt operand not
designed; NAL-2..8 not tested
```
