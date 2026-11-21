# D-REGC-0: the register contract's contradictions, measured in one consumer (2026-11-21)

**Status:** PROPOSED / DOCS ONLY. Knowledge doc
`.claude/knowledge/register-contract-reconciliation.md`; plan
`.claude/plans/d-regc-register-contract-reconciliation-v1.md`.

What zammad-rs measured, against PostgreSQL 16.15 (zammad-rs #95, #96, #97):

- One `borrow_all_selected_present` read of `knowledge_base_categories` equals
  PostgreSQL: a 684-position bag over 3 rows. Five disable runs go red.
- A deleted parent read through a left join returned 547 stale rows before the
  fix. A missing key now resolves to no position, and the missing side is −0.
- Coordinates are bijective after each of 51 feeds, in both layouts, over 5–6
  tiles. Relations are bags.
- Schema and basis sub-checks are 49,140 of 146,453 per-read check steps.
- Joins resolve one key per kept tuple per dimension: 546 over 57 wide calls.
- 31,432 cells are rendered without a lane check.

What is open: C1 (Decision J vs EXO-199 and the owner's statement), C3
(deletes, EXO-337/294), C4 (outer-join side), C7 (issuer, EXO-329).

**Fruit:** D-REGC-6 first step, counting the predicates zammad-rs's 512 calls
use against the strided-coverage table in C8. Read-only, no decision needed.
