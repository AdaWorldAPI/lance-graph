# 2026-11-12 — D-DR-SELECT-K-SV-2: a position selection witnessed with its lanes

**Kind:** admission. One socket added to the DT64 lane contract, one boundary function in the report crate. No production caller.
**Grade:** MEASURED (`cargo test -p lance-graph-dt64 -p lance-graph-report`: all
suites pass; new: `tests/bound_sel.rs` 10, `tests/bound_sel_alloc.rs` 1; 8 of 8
binding disable runs red, one equivalent mutation not counted;
`cargo clippy --all-targets -- -D warnings` clean on both crates).
**Plan:** `.claude/plans/dr-select-k-v1.md` §15. **Linear (existing):** EXO-253.
**Follows:** PR 2038 (decision), PR 2039 (`execute_sel`).

- `SelectionSource` / `SelAddr` / `LaneBinding::selected` in dt64: the source
  that issues a lane also issues a position list for the same frame and row
  order; the binding checks frame, issuer, revisions, route, row order and
  scope equality with the existing checks, then lends both.
- `fold_bound_selection` in `lance-graph-report`: calls `execute_sel` once on
  the two borrowed operands. Nothing is copied or allocated.
- Equal length and equal revisions from another `FrameStore` are refused
  (`ForeignFrame`) before any borrow.
- Cost at N = 65,536: witness 139–164 ns, fold 64–139 ns; producing a Top-K
  list is 138–144 µs once per frame (cold, reported, not a gate).

**Open (BLOCKED, owner named in the plan):** no authority issues a reader
`Scope`, so authorization is not established here; no non-lab order witness,
so ordered consumers are not admitted; historical snapshots are refused;
global-to-local identity is untouched; no production `SelectionSource`.
