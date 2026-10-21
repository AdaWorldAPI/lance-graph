# 2026-10-20 — D-RELALG-UNIFIED: assessment, decisions, stage order

**Status:** PROPOSAL (documentation only). Plan: `.claude/plans/d-relalg-unified-v1.md`. D-ids D-RU-0..8 (STATUS_BOARD).

## Decided

- ADMISSION = B (operator, 2026-10-20): counted, version-witnessed same-table seeks are admitted in preparation; cross-table positional gathers stay forbidden; frozen codebook lookups are admitted under the defined attestation; D-MPS-8 is not cleared automatically. `ISS-ADMISSION-PREP-BINDING` status updated.
- D-RA-2 = A (operator, 2026-10-20): the single-table set algebra and its rewrite rules belong in `lance-graph-contract`; relalg keeps lowering and relational planning; no general relational AST for JC or Quack.

## Assessment

- The shared RelAlg expression graph does not exist. relalg is a single-table Boolean selection algebra with a terminal measure; Quack has no cost model and does not consume relalg.
- JC's statistics already lower to shared sufficient-statistics terminals (`GroupPowerSumsI32`, `GroupCrossPowerSumsI32`); batching requests over distinct (selection, key, lane pair) triples needs no new IR. That is U1.
- A DAG inside one fold would lose `call_owned`'s constant width; sharing goes through `let`-bound planes instead (U2, on the contract IR).

## OPEN

- Three grep-only negatives are not yet fully verified: no JC expression IR, no D-CMR set IR code, no D-MSUB.
- The 256-rule and the 256-hop benchmarks are separate; rule sharing is no evidence for hop amortization.
- D-MPS-8, D-RPR-1, D-RST-4/6 and `quarantined-gather-hop` keep their own reviews under ADMISSION = B.
