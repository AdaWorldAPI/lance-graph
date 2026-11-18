# EXO-326-KLEENE-GATE: a gated r2il plan admits a three-valued guard (2026-11-18)

**Status:** MEASURED for the interpreter path (`prepare_gated` / `run_gated`).
The native path is not done: `prepare_native_present` refuses every gated
plan, two-valued ones included (`binding_native.rs`, "this path freezes
programs and has no gate operand").

What already existed and is reused, nothing new:

- dt64 `Guard::witness`: under `Logic::Kleene` the witness is the TRUE-rows
  form (`normalize_kleene(..).into_where_true()`).
- quack `set::SetFilter::where_true(lanes, nullable)`: the filter that keeps
  the rows where a three-valued set is TRUE, over the validity planes the
  caller names.
- r2il `Gate { plane, guard }`: the seam never evaluates WHEN; the caller
  hands the kept rows over as a resident plane.

Change (`crates/lance-graph-r2il/src/binding.rs`): `prepare_with` no longer
returns `ThreeValuedNotExecutable` for a guard whose logic is not two-valued.
A method whose RESULT is three-valued (`Truth::ThreeValued`) is still refused.

Tests (`tests/integration/binding.rs`, 2 new; integration suite 67 passed):

- `a_kleene_guard_fires_only_on_its_true_rows`: guard `k == 0 AND b < 4`
  over 1,000 rows, `k` absent on every third row and `b` on every fourth
  (decision B: presence bit 0, value 0). The gate is computed by quack
  `where_true` over the two presence planes and equals a scalar Kleene
  reading written in the test. All six combinations occur, among them HIRO's
  KI-1 rows; only (true, true) fires. The count under the gate equals the
  scalar count and is lower than the two-valued reading of the same lanes.
- `a_kleene_guard_still_needs_its_own_gate`: `prepare` refuses it
  (`GuardNotExecutable`), a gate of another guard is `GateMismatch`, a
  three-valued result is still `ThreeValuedNotExecutable`.

Disables, each red: the refusal restored (both tests); the gate evaluated
with no validity planes (the first test).

Open:
- Native gated plans (the literal ask of EXO-326). Needs a gate operand in
  the frozen programs; not started.
- The UNKNOWN rows are the caller's: `normalize_kleene` is valid only where
  TRUE rows are read, so this seam cannot hand them out.
- mailbox-lanes `when` still refuses a Kleene guard as `ValidityUnstated`:
  its socket states no presence for its lanes.
- OGAR #387's red-pinned KI-1 half asserts the old refusal and turns red
  when OGAR moves to this revision.
