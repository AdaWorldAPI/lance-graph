# D-VRL-RECIPE-1: a VRL recipe's aborts as the WHEN guard of a registered method (2026-11-18)

**Status:** MEASURED (test-pinned), lab only. One test file in
`crates/vrl-recipe-probe`; no source of the probe or of any library changed.
No timing taken, so no cost is claimed.

Follows D-VRL-RECIPE-0 and `.claude/knowledge/classid-on-do-call-shape.md` §2
(WHEN is what the `register` call supplies).

## What existed

- `tests/r2il_parity.rs` registers a recipe as DO steps: frozen F000 words
  under `POLYFILL_LENS`, `when: None`, run through r2il's `prepare` / `run`.
- `lance-graph-r2il`'s integration test evaluates a hand-written Kleene guard
  through the binding (`kleene_true_rows`).

Nothing registered a recipe's aborts as a guard.

## What the test does (`tests/registered_guard.rs`)

- Builds the guard from `Recipe::terms()`: a conjunction of `NOT (field <cmp>
  v)` over field positions, registered with the class under `Logic::Kleene`.
- The caller names the method and the route only. The guard and its logic are
  read from the admitted binding; its fields are requested as a field mask;
  lanes and presence planes come from one `borrow_all_present`; the TRUE rows
  from Quack's `SetFilter::where_true`.
- No DO step runs. r2il's `prepare_gated` / `run_gated` are not used.

## Result

`cargo test --release --manifest-path crates/vrl-recipe-probe/Cargo.toml --test registered_guard`: 4 passed.

- For 5 programs at widths 1, 63, 64, 65, 300 and 1000, with about a fifth of
  three lanes absent, the registered guard is TRUE on exactly the rows the
  recipe's own program keeps.
- The same guard registered two-valued gives another count in at least 10 of
  the 30 cases, so the logic the class was bound with is what decides.
- A frame sealed before the source moved is refused before the guard runs.
- A program that was not registered has no method reference.
- Disable run: with the `NOT` removed from the guard, the first test is red.

## OPEN: two presence readings

Pinned as measured in `presence_zero_with_a_nonzero_value_is_read_differently_by_the_two_paths`:

| Row | Recipe (DR512, EXO-294) | Registered guard (Quack `where_true`) |
|---|---|---|
| presence 0, value 0 | absent, UNKNOWN | NULL, not TRUE |
| presence 0, value -4 | a value, decided | NULL, not TRUE |

On two rows the recipe keeps 1 and leaves 1 unknown; the registered guard is
TRUE on 0. `where_true` reads the plane as SQL validity; the recipe reads
absence as presence 0 AND value 0. Which reading a registered guard over a
DR512 lane must use is not decided here.

## Also seen, not changed

`tests/alloc.rs` fails on clean `main` when its two tests run in parallel
(`(55, 1)` allocations counted at n = 64) and passes with `--test-threads=1`:
its allocation counter is shared by both tests.
