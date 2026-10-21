# 2026-10-20 — D-CMR-1 / D-RA-2: the contract owns single-table set algebra

**Status:** TEST-PINNED, merged in #1644 (`c05a313`). MEASURED after merge: holo-h14d-probe stage 1 widens 0 of 2,000 programs (was 2; `cargo test --manifest-path crates/holo-h14d-probe/Cargo.toml stage1_answers_agree -- --nocapture`). **Basis:** `ADR-2026-10-20-SIX` §2 (`entries/2026-10-20-six-decisions.md`).

- **Moved:** relalg's rewrite rules to `lance_graph_contract::set_algebra::normalize`, unchanged except absorption. relalg's `normalize` converts `Rel` ↔ `SetExpr` (`Plane(i)` ↔ `Sym(i)`, `Atom` ↔ `Res`) and delegates. `symbolic`, `run`, lowering and the public API are unchanged. The contract stays dependency-free.
- **Absorption** is matched against flattened sibling sets, removing one child at a time against the children still kept, in two forms: (1) a member of the dual child is a sibling, or a junction of siblings; (2) a sibling junction's members all occur in the dual child (`x ∨ (x ∧ y)` with `x` itself a conjunction). That closes N17 (`ISS-RELALG-ABSORPTION-AFTER-DE-MORGAN`) for `x` a leaf, an `And` or an `Or`. Form 2 was found by crepe-plan-probe's N17 pin (CI `probe`), which this PR flips to assert the reduction.
- **`Runs`:** sorted, disjoint, half-open intervals with `∩ ∪ ∖`, each one merge pass, O(n + m). `complement(universe)` refuses a set that reaches past the universe.
- **Regression:** the 16 pre-existing relalg tests pass unchanged. 3 tests are new: `conversion_round_trips` plus 2 in `tests/absorption_n17.rs`.
- **Disable runs** (commands: `cargo test --manifest-path crates/lance-graph-relalg/Cargo.toml --no-fail-fast` and `cargo test -p lance-graph-contract --lib set_algebra`):

  | probe | what breaks |
  |---|---|
  | contract `normalize` returns its input | 10 of 14 `rewrites.rs` tests, and both N17 tests in relalg |
  | absorption form 1 off | only the N17 tests: one in relalg, two in the contract |
  | absorption form 2 off | only `absorption_of_a_superset_conjunction` |
  | `max` dropped in the `union` merge | `runs_obey_the_set_laws_against_bitsets` |
  | absorption on a partial cover (`all` → `any`) | the silence twin, and the bitset differential (an unsound rule) |

- **Scope held:** no joins, gathers, positional addressing, recursion, D-MPS-8 or `graph_hop`. Measures (`Count`/`Sum` on `Runs`) are not in this PR.

OPEN: D-CMR-6 catalogue; repointing crepe and Quack at the contract.
