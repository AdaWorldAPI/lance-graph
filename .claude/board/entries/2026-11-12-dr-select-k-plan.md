# 2026-11-12 — D-DR-SELECT-K: plan trimmed to one lab, scenario fixed (docs only)

**Kind:** plan. Ships no code and authorizes none.
**Plan:** `.claude/plans/dr-select-k-v1.md`. **Linear (existing):** EXO-253.

- **Scenario.** Cold path once per frame (Lance snapshot → optional indexed
  WHERE → K survivors → bind DR), hot path many folds over K with no Lance
  access. Resident data never goes back to Lance.
- **Comparison.** `T_full_intake(N) + R × T_fold(N)` against
  `T_indexed_intake(K) + T_bind(K) + R × T_fold(K)`.
- **Ownership.** A K-row batch is counted staging under the source or the
  focus lease, never a second owner of values. The `NodeGuid` → canonical
  global id → DR coordinate → snapshot row chain is not shown yet and
  blocks D-DR-SELECT-K-2.
- **Strategy.** A three-line deterministic rule (resident / cold with a
  measured index win / cold otherwise). The longer input list of PR 2032
  stays as hypothesis text.
- **Scope.** D-DR-SELECT-K-1 is the only candidate: N = 1,000,000, K = 256
  and 65,536, R = 1 and 100, one 75% negative control, independent oracle.
  K-2 to K-7 are contingent on four gates.

**Open:** which Lance 14 call returns a compact exact survivor set and what
it copies; whether mask-RISC takes an implicit `ALL(K)`.
