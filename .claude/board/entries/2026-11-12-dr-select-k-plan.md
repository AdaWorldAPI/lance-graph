# 2026-11-12 — D-DR-SELECT-K: plan trimmed to one lab, scenario fixed (docs only)

**Kind:** plan. Ships no code and authorizes none.
**Plan:** `.claude/plans/dr-select-k-v1.md`. **Linear (existing):** EXO-253.

- **Order of things.** DR → deterministic version → index order and/or
  filter → `DR{dynamic}[K]` → reusable selection → Quack / mask-RISC →
  ndarray. Index order and filter compose over one witnessed version. The
  representation choice is a DR addressing decision in the versioned
  binding, not a storage or residency decision.
- **Ownership.** A K-row batch is counted staging under the source or the
  focus lease, never a second owner of values. The `NodeGuid` → canonical
  global id → DR coordinate → snapshot row chain is not shown yet and
  blocks D-DR-SELECT-K-2.
- **Strategy.** A view is chosen by what already exists and by the next
  operation: IndexVector, SelectionVector, FocusMask or implicit `ALL(K)`.
- **Scope.** D-DR-SELECT-K-1 is the only candidate. Its question: which
  existing view of one versioned selection is cheaper per operation and
  reuse count. N = 1,000,000, K = 256 and 65,536, R = 1 and 100, one 75%
  negative control, independent oracle. K-2 to K-7 are contingent on four
  gates.

**Open:** which Lance 14 call returns a compact exact survivor set and what
it copies; whether mask-RISC takes an implicit `ALL(K)`.
