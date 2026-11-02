# Semantic Grammar Recovery: correctness gates implemented (operator decisions S1–S6), 2026-11-02

**Status:** IMPLEMENTED in three PRs (not merged), one finding MEASURED, one item BLOCKED on a decision, one item STOPPED on a repository-rule conflict. Follows `2026-11-02-coresearch-semantic-grammar-recovery.md`.

| item | where | outcome |
|---|---|---|
| S5 erratum | `2026-10-31-d-reasoning-codegraph-0-xlog-egglog-shared-relation.md` (appended) | done: equality follows from the probe's authored `edge ⇒ union`; a CALLS edge does not imply equivalence |
| B4 + F8 | lance-graph #1932 `claude/sgr-coverage-gates` | residuals consumed (closed `ResidualKind`, unknown reason refused); `Codegraph::coverage()` typed gaps; `DeadFunction` runs `Partial` on any gap (residual, missing residuals file, dynamic call, partial crate). rev1 re-pinned Complete → Partial (one dynamic call) |
| B7 consumer | lance-graph #1932 | `IntakeRefusal::Placeholder` for any `?` path segment, at open and refresh |
| B7 producer + B1 | ruff #135 `claude/sgr-charon-identity` | unnamed dispatch target → `CALLEE_NOT_NAMED` residual, never `?::?`; `<crate>.effects.tsv`: one row per witnessed effect with Charon's exact assert kind (`Overflow:Mul`, `DivisionByZero`, …) and deref writes. Absence is never purity |
| B9 | ruff #136 `claude/sgr-cpp-call-identity` | `calls` resolved only and scope-qualified (`a::f`); `unresolved_calls` lookup sets with signature candidates; `dynamic_calls` count. Found by the fixture: pointer calls (`fp`) were reported as resolved calls |
| A10 | — | BLOCKED: raw5 = 0 is the legitimate base-fact code, so no reader rule can tell "stamped Direct" from "unstamped" without an encoding decision (new `Epi5Gen` with 0 = unstamped, a spare-bit flag, or a writer-side stamping API). Hazard stays pinned by `epi5_reading.rs` |
| S6 query-relative gate | AdaWorldAPI/xlog, local branch only | MEASURED: with `magic_sets = on` and a second consumer `far(X) :- reach(X, 12)`, the rewrite reports `Applied`, no decline reason, and `far` goes from {10, 11} to {}. Upstream `magic_sets.rs` is unchanged in the fork, so upstream has it too. STOPPED: the fork's `AGENTS.md` / `ENGINEERING.md` current direction forbids adding tests and running local tests; nothing was pushed |

Every fixture was committed red before its fix; every guard has a disable run that goes red (16 disables). Two falsifiers were found vacuous by their disable runs and rewritten (codegraph identity twin queried the wrong node; ruff effects occurrence test could not see a same-kind collapse).

## Closeout
STATUS: done (S1 items 1–3 and 5, S2, S5) / blocked (A10) / stopped (S6 fork) | OUTCOME: lance-graph #1932, ruff #135, #136 open; query-relative Magic Sets hazard measured | OPEN: A10 encoding decision; S6 fork rule conflict and the decline fix; method keys of same-named classes in different namespaces (C++); codegraph does not yet read `effects.tsv` (S3 keeps evidence grades in the lab)
