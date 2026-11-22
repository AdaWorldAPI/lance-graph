# D-XLOG-W7-2-STEP-4B-CORRECTED (of D-XLOG-W7-2-STEP-4B): the probe's predicate is the method's DO (2026-11-19)

**Status:** MEASURED (test-pinned; timing taken). Supersedes the shape, not
the numbers' meaning, of `2026-11-18-d-xlog-w7-2-method-over-cube-lanes.md`.

What was wrong in #2137: the probe registered a method with no effects, put
the predicate in its WHEN, and counted the rows the guard passed. The owner's
correction on a sibling slice (EXO-333, verbatim): "It assumes a class is
fired without being called" and "the call itself does the inference over the
representation". EXO-333 listed #2137 as using the same shape. On the
question how to restate it, the owner: "As polyfill".

Now (`tests/support/method.rs`, test and example code only):

- The method has no WHEN. Its DO is five frozen words of `0xFFFF :: 0xF000`,
  each resolved by its SQL name through `position_of`: `=`(x, c(src)),
  `=`(y, c(dst)), `NOT`, `AND`, `COUNT`.
- A call prepares once (`lance-graph-r2il::binding::prepare`: admit, basis,
  word positions and arities) and borrows the method's reads from the cube
  through `CubeLanes` with one `borrow_all_present`. `run` executes the bound
  plan over the cube's own words and reads the answer field.
- `lance-graph-r2il` is a dev-dependency of `lance-graph-xlog` for this.

Tests: the same three (whole crate 101 passed). Disables, each red on the
first two tests: the socket hands columns out by field position; the `NOT`
word's result is not used; `AND` is spelled `OR`.

Timing (`cargo run --release -p lance-graph-xlog --example method_cost`,
median of 31 prepares and 201 runs, every run's count asserted against the
edge walk; two invocations; **machine at load average about 2.5**, another
build running, so read these as an upper bound):

| rows | count | prepare | run | run per row |
| --- | --- | --- | --- | --- |
| 1 024 | 5 | 1 056 to 1 062 ns | 581 to 590 ns | 0.57 to 0.58 ns |
| 16 384 | 56 | 1 034 to 1 070 ns | 2 931 to 2 938 ns | 0.18 ns |
| 262 144 | 1 025 | 1 066 to 1 386 ns | 60 866 to 68 220 ns | 0.23 to 0.26 ns |

Not comparable line by line with the 2026-11-18 table: that one prepared a
Quack-lowered guard program and this one binds polyfill words, and the two
were taken at different machine load.

Open:
- The live plane is borrowed and not applied: the plan has no gate and every
  row of the tests is live.
- The constants are tile coordinates of one frame, as before.
- `foundry-parity-lab/src/method.rs` uses the guard-only shape too (EXO-333);
  not touched here.
