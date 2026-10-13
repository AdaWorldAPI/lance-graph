# 2026-10-12 — SAG capstone alignment (D-SAG-CAP-1)

**Finding:** six SAG research PRs written in parallel on the #1509 base
named one concept six ways and defined five status enums for "unknown is not
false". `ProposedMethod` (inert, evidence-carrying method candidate; a
rewrite shape and an action shape) is now canonical, and one four-state status maps all five vocabularies. Choosing
what to probe next is a separate concept (`SchedulerDecision`).

Plan: `.claude/plans/sag-capstone-alignment-v1.md`.

PR #1523 | STATUS: done (docs) | OUTCOME: #1510/#1511/#1512/#1513/#1515/#1517
merged and aligned; aliases renamed; the second-hop refusal and SAG defects
A/B confirmed current (#1519, #1522 closed); code anchors removed by #1524 |
OPEN: ProposedMethod and the four-state status are not built (SAG Step 4);
#1517 oracle tests F0/F7/F8 are claimed, unverified.
