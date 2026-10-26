# 2026-10-26 — coresearch: Lance 13 masked read × DR64 / Quack × DuckDB

**Status:** WORKING-MODEL (exploration map; ratifies nothing). Harness: `.claude/agents/coresearch-council.md`.
Deliverable it fed: `.claude/plans/duckdb-dr-quack-machine-room-reuse-v1.md` § Appendix A (follow-up to #1787).

**Question.** How should a Quack / mask-risc / DR64 mask drive Lance 13's read-side row selection
(`Scanner::with_row_addr_prefilter`) without duplicating work, and which mask shapes reduce bytes read?
**Premise gate (premise-auditor): PREMISE-SPLIT** into Q1 (fold-side handoff: slot coordinates → `_rowid`
`RowAddrTreeMap`, bulk, version-checked) and Q2 (Lance-side bytes read). "The read is not part of the fold" was
added as an allowed answer.

**Scouts.** 1 code cartographer, 2 internal prior art, 3 literature (Roaring 1603.06549, Abadi ICDE'07, Yao/Whang
CACM'83, Moerkotte VLDB'98, X100 CIDR'05, Ngom DaMoN'21, LIP PVLDB'17), 4 systems (Lance 13 source incl.
`lance-encoding`/`lance-io`, arrow-rs `RowSelection`, DuckDB, ClickHouse), 5 set-representation concepts (swap of the
ontology scout, declared). Plus source audits: G1–G8 re-audit, DuckDB @1641475 locations, Quack/mask-risc/DRF map.

## Exploration map

| id | idea | verdict | reason |
|---|---|---|---|
| X1 | chunk-touched estimator (non-zero 64-word groups × bytes/chunk) | **PROBE (P1)** | explains H14c/H14d from Lance source (mini-block ≤ 4,096 values, `miniblock.rs:21`); an estimator, not exact (variable chunk boundaries, lance-io range merge) |
| X2 | `insert_fragment` (Full) for all-ones fragments | PROBE (P2), gated on S-INGRESS | wiring only; valid only when the slot span equals the physical extent at a checked version |
| X3 | `from_lsb0_bytes` words → Roaring | PROBE (P2), gated on S-INGRESS | dense containers avoid per-bit work; sparse ones do not; three copies total (ours + two in Lance) |
| X4 | run → `insert_range` | PARK | exists in lab; blocked on `ISS-WITNESSED-RANGE-DOES-NOT-ATTEST-PLANE-ORDER` |
| X5 | AND masks fold-side, one read | ADOPT-NOW (as written rule; no code) | measured in H14c |
| X6 | placement decides bytes | PARK | write-time question, H14d OPEN |
| X7 | one version-checked coordinate authority | PROBE (P3) | two lab maps with different witnesses; crepe handoff unchecked |
| X8 | DR ordinal → `_rowid` composition | PROBE (P3), gated on S-INGRESS | no code composes them; must look up through both maps, never store a third table; Bag restored at DR boundary |
| X9 | DuckDB late-materialisation re-join | SKIP | owned row-id join (tuple shape); the idea already lives in Lance + DRF |
| X10 | bitmap/list crossover | SKIP | upstream (roaring / Lance) |
| X11 | complement (block list) | SKIP (for now) | K3 (NOT UNKNOWN ≠ NOT 0) and universe hazards |
| X12 | `_under` twins / fused narrowing | PROBE (P4) later | fold speed, not I/O |
| X13 | early-exit emptiness (G8) | PARK | small value; empty mask already reads 0 B |
| X14 | u8 / u64-ordered `Pred` | PARK until a consumer asks | ndarray kernels exist, IR wiring only |
| X15 | KNN/FTS per-row filter branch | SKIP (caveat recorded) | mask cuts rows, not bytes there |
| X16 | resident zone maps | SKIP | would patch storage |

**Firewall rulings (co-architect 2), carried into Appendix A:** the words → `RowAddrTreeMap` step is a copy,
not the `$i` inversion; it needs a named seam (**S-INGRESS**) decided by the operator; storage reads are reported
in bytes/ranges/rows, not wall time (the ms exception covers cold compile and oracle arms only); an allow list
fails safe, a block list does not.

**Recommended single next PR:** P1, offline over the committed H14c/H14d raw CSVs (bytes per cell present) and
masks regenerated from the probes' seeded draws. No runtime implementation recommended.

**Not searched:** Vortex, FeatureBase/Pilosa; Lance `ChunkInstructions::schedule_instructions` in full; the exact
local-store block size; Yao 1977 itself (read through Whang 1983); the H14c/H14d raw CSV rows beyond their headers.

OPEN: S-INGRESS (operator); which single coordinate→`_rowid` authority; stable row ids for lane datasets
(H14c, still open); K-knee claim (`LANE_FOLD_INTEGRATION.md`, section "What this must not integrate") has no code.
