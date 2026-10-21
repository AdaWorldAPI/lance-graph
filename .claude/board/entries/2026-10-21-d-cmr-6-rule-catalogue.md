# 2026-10-21 — D-CMR-6: the guarded rule catalogue and fixpoint driver in the contract

**Status:** TEST-PINNED, in PR (branch `claude/eager-albattani-iufugq`). **Basis:** plan `d-cmr-concept-chain-normalization-v1.md` §3.5, §4, §7 and the D-CMR-6 row; ownership `gpt-ada-01/004`, `/007`; claim EXO-50 `session_013z2e1YXkm6o6HNrDn5BPWA/016`.

- **Where:** `crates/lance-graph-contract/src/set_rules.rs` (`lance_graph_contract::set_rules`), zero dependencies. It consumes `set_algebra::{SetExpr, normalize}` and edits neither `set_algebra.rs` nor `set_program.rs` (U2). crepe, relalg and mask-risc are untouched; no consumer carries a second copy of these rules.
- **Types:** `Measure` = `Const | Count(SetExpr) | Sum{lane,set} | CountRows(Vec<RowRange>) | Opaque{id,pure} | Add | Sub`. `Semantics { universe, arith: Exact|Wrapping|Checked|Ieee, truth: TwoValued|ThreeValued, sums: Plain|Sym, population: Distinct|Bag }`.
- **Catalogue (`CATALOGUE`, data: rule, pattern, requires, refusals):**

  | rule | rewrite | refuses on |
  |---|---|---|
  | `SetNormalize` | the set inside `Count`/`Sum` → `normalize` | three-valued |
  | `Complement` | `Count(¬A) → N − Count(A)` | three-valued; no universe; bag; checked; IEEE |
  | `RelationalDifference` | `Sum(A) − Sum(A∧B) → Sum(A∧¬B)`; `Count` likewise | three-valued; checked; IEEE; `_sym` sums |
  | `AdditiveCancellation` | `(a+b) − a → b` | checked; IEEE; impure `a`; a `_sym` sum in `a` |
  | `RangeIntersect` | `⋂[loᵢ,hiᵢ) → [max lo, min hi)` | basis mismatch; no universe; bound past it |
  | `RangeCount` | `Count([lo,hi)) → max(0, hi−lo)` | bag; IEEE; no universe; bound past it |

- **Driver:** `rewrite` / `rewrite_with(enabled)` try rules at a node before its children, apply only a rewrite that strictly decreases `(set nodes, all nodes)`, stop at the fixpoint or the step budget (`Trace::budget_exhausted` set only if another step would fire). Refused matches are recorded once each in `Trace::refusals`; the original is kept.
- **Tests (21):** 600 random measures over random two-valued tables, under exact (`i128`) and wrapping (`i64`) oracles, with random rule subsets (disabling any optimization keeps the answer): equal answers, size never grows, no refusal on valid input, most inputs rewritten (anti-vacuity). Every non-range rule fires on random input. Experiment A (relational difference vs the `i128` oracle; checked refuses, and the eager `i32` accumulator is the counterexample: the original overflows, the rewritten sum is 5). Experiment D (three ranges → intersect → `Const(20)`; with intersection disabled the count cannot fire; disjoint ranges → 0). Experiment E: NULL (complement, relational difference, and set normalization `A ∨ ¬A → All` each shown wrong on a NULL table and refused), no universe, IEEE (`f32` rounding), checked, impure (`Opaque` evaluated twice), `_sym` sums (the `Count` twin still fires), basis mismatch, bound past the universe, bag. Budget and catalogue-shape tests.
- **Disable runs** (`cargo test -p lance-graph-contract --lib set_rules`): each guard removed in turn turns exactly the named test red — set-normalize three-valued (`experiment_e_null_refuses_set_normalization`), complement three-valued and universe, relational-difference three-valued / checked / IEEE / `_sym`, cancellation checked / IEEE / impure, range basis / past-universe, range-count bag; removing the size check fails the differential test.
- **Codex review of #1651 (four findings, fixed):** complement for bags (multiplicities `[2, 1]`: `Count(¬A) = 1`, `N − Count(A) = 0`) and under checked / IEEE arithmetic; cancellation of a `_sym` sum; range count under IEEE. Each new guard has a test and its own disable run.
- **Gates:** contract lib 1,539 pass; relalg suite (own workspace) 26 pass, unchanged; clippy `-D warnings` and fmt clean.

OPEN: repointing crepe and relalg onto `set_rules` (separate, after this); a prefix-lane `Sum([lo,hi)) → P[hi] − P[lo]` rule (needs a prefix lane, separate candidate); D-CMR-8 only if rules compete.
