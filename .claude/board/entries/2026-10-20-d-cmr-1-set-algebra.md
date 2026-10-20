# 2026-10-20 — D-CMR-1 / D-RA-2: the contract owns single-table set algebra

**Status:** TEST-PINNED (in PR, branch `claude/eager-albattani-iufugq`). **Basis:** `ADR-2026-10-20-SIX` §2 (`entries/2026-10-20-six-decisions.md`).

- **Moved:** relalg's rewrite rules to `lance_graph_contract::set_algebra::normalize`, unchanged except absorption. relalg's `normalize` converts `Rel` ↔ `SetExpr` (`Plane(i)` ↔ `Sym(i)`, `Atom` ↔ `Res`) and delegates. `symbolic`, `run`, lowering and the public API are unchanged. The contract stays dependency-free.
- **Absorption** is matched against flattened sibling sets, removing one child at a time against the children still kept. That closes N17 (`ISS-RELALG-ABSORPTION-AFTER-DE-MORGAN`).
- **`Runs`:** sorted, disjoint, half-open intervals with `∩ ∪ ∖`. `complement(universe)` refuses a set that reaches past the universe.
- **Regression:** the 16 pre-existing relalg tests pass unchanged. 3 tests are new: `conversion_round_trips` plus 2 in `tests/absorption_n17.rs`.
- **Disable runs** (commands: `cargo test --manifest-path crates/lance-graph-relalg/Cargo.toml --no-fail-fast` and `cargo test -p lance-graph-contract --lib set_algebra`):

  | probe | what breaks |
  |---|---|
  | contract `normalize` returns its input | 10 of 14 `rewrites.rs` tests, and both N17 tests in relalg |
  | generalized absorption off | only the N17 tests: one in relalg, two in the contract |
  | absorption on a partial cover (`all` → `any`) | the silence twin, and the bitset differential (an unsound rule) |

- **Scope held:** no joins, gathers, positional addressing, recursion, D-MPS-8 or `graph_hop`. Measures (`Count`/`Sum` on `Runs`) are not in this PR.

OPEN: D-CMR-6 catalogue; repointing crepe and Quack at the contract.
