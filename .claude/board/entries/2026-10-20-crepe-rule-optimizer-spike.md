# 2026-10-20 — D-SRAM-2 spike: crepe as a cold rule optimizer in front of relalg

**Status:** MEASURED (lab probe `crates/crepe-plan-probe`, own workspace, excluded). Research only: does not authorize D-SRAM-2 and does not own D-RA-2. Plan: `.claude/plans/crepe-rule-optimizer-spike-v1.md`.

## What landed

- crepe fork (`AdaWorldAPI/crepe` branch `ccr-d3ce1263-ied9yh`, `9fd5785`): `bind_fields` extracted from `make_clause` and shared by `crepe!` and a new `crepe_plan!`, which runs `Context::new` (arity, `@input`, strata, negation cycles) and emits a plan as data, no HashSet/HashMap runtime.
- relalg: `run_with_program` (additive) returns the lowered mask-risc program for inspection.
- Probe: label resolution, binding propagation (Soufflé adornment step, no magic predicates), duplicate-atom and duplicate-rule elimination gated on duplicate-insensitive answers, MORK `query_multi` mapping with D-MPS-9 refusals, star lowering over the D-MPS-8 table with coordinate witness and domain fingerprint.

## Measured

- 20 differential tests (A–N) against `crepe!`, datafrog, the MORK shape and a nested-loop bag reference; 9 disables each red at their named tests.
- `Answer(17, z)`, 50,000 Parent rows, release: terminal 3.6 µs with rewrites vs 11.8 ms without; prepared rows 56,286 vs 118,802; summed cold cost lower with the rewrites at every size (break-even reuse count 0). Parity ok at 100/1k/10k/50k.
- H14d absorption miss holds only when `x` is itself an `And`/`Or` (pinned, test M).

## OPEN

- Dynamic MeTTa rules need crepe's analysis as a library: crepe is a proc-macro crate (follow-up A, `crepe-core` split).
- Semijoin reduction built (`Options::semijoin`): laid-out rows 36 at every size instead of growing with n; preparation 3.47 ms → 0.28 ms at 50,000. The scan stays O(n).
- ndarray for the row filter, measured (`examples/row_filter.rs`): over the row-major store the strided compare is slower than the scalar test at every needle count, so it is not wired; over a contiguous column it is 5–6 µs per needle at 50,000 rows and beats the scalar test up to 64 needles; at 128 needles two runs disagree because of benchmark noise (the scalar time moved 1.8× between runs), so that crossover is not pinned (re-measured after the #1632 Codex fix). Next step: a column-major relation store.
- Stage 3 over Lance 13 / lancedb 0.40 built (`lance_store`): each side read under a relalg address mask with `with_row_addr_prefilter`; every masked read scans exactly its mask (P1–P4 green). The relalg membership mask holds 4 centers at most: `k` centers take `2k - 1` scratch slots (no slot reuse inside a fold), and `MAX_SLOTS = 8` is an r2il setting, not a hard limit (mask-risc addresses 65,536). Above that a scalar test builds it.
- Arm B built (`addressed`): lanes in key order, attested with `SealedFacetLane::attest_sorted`, read only at the `seek_from` ranges of the demanded keys. At 50,000 rows it scans `3k` rows where Arm A scans at least 150,000. It is 2.5–12× faster than Arm A up to 1% of centers and pays back its 7 ms attest after 2–3 queries. From 10% of centers up a full scan wins, and with all centers Arm B never breaks even. Open: the selectivity crossover between 1% and 10%; multi-fragment lanes (an append currently forces a compaction); a second lane for demands on the second column.
