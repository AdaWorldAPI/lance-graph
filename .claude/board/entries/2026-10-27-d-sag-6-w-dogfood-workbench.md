# D-SAG-6-W — dogfood MCP workbench: read-only project graph over Quack

**Date:** 2026-10-27. **Status:** W0 docs + W1 LAB (`crates/sag6-workbench-probe`, own `[workspace]`, excluded from root); W2–W3 PROPOSED; W4 deployment NOT AUTHORIZED. **Source:** [`d-sag-6-w-dogfood-mcp-workbench-v1.md`](../../plans/d-sag-6-w-dogfood-mcp-workbench-v1.md), [`d-sag-6-w-mcp-board-railway-v1.md`](../../plans/d-sag-6-w-mcp-board-railway-v1.md). Parent: D-SAG-6 (#1760).

**Measured (W1):** `cd crates/sag6-workbench-probe && cargo test` → 16 passed. Eight disable runs (coverage, frame check, symbol revision, inverse column, chain refusal, grade laundering, method binding) each turned the named test red; tree clean after restore. The Quack + mask-risc path equals a `GlobalId`-keyed scalar oracle for every identity × predicate × direction of the fixture; `blocked_by` equals Linear's own `blockedBy` lists, and only the two `blocks` rows are stored.

**Fixture:** 19 real records read 2026-10-27 — Linear EXO-111/112/140/148/50 relations, the source lines the issues name pinned at lance-graph `aa569bc` (`mul.rs:1086`, quack `lib.rs:387`, `rmap.rs:70`), their `ISSUES.md` rows (:74, :176), cited PRs #1640/#1730/#1732, one recorded session id, and one inferred edge (`graph_hop.rs:55`) graded `Hypothesis`.

**Found while building it:**
- One-hop project-graph questions are an edge table under Quack's existing `Filter::and[EqU32, EqU32]` — no DR64², DR64³ or DT64 code is needed for W1, and none exists (DT64: 0 code files).
- Multi-hop has no admitted path on two independent grounds: `Filter::Semijoin` unclassified under ADMISSION B (EXO-140) and `OutgoingHop` quarantined (ISS-GATHER-HOP-1616). The workbench refuses it instead of emulating it.
- OGAR already has `PROJECT_WORK_ITEM` / `PROJECT_RELATION` / `PROJECT_REPOSITORY` / `PROJECT_CHANGESET` (Redmine/OpenProject ports); there is no GitHub or Linear port and no PullRequest or SourceSymbol concept, so nothing is bound.
- `mcp.exo.red` is the live `adarail-production` service (write tools, write-scoped secrets); the workbench must be a separate service on a separate host name.

**OPEN:** OGAR port/concepts (OGAR owner); ADMISSION B (EXO-140, operator); DurableRef home (EXO-148, D-RST-1); domain, Railway ownership, read-only tokens, Redis yes/no (operator).
