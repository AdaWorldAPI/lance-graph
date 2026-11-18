# D-XLOG-W7-2-STEP-4B: a registered method executed over a relation's lent lanes (2026-11-18)

**Status:** MEASURED (test-pinned; timing taken, see below).

Step 4a (#2136) lent a cube relation's two lanes through the lane socket and
had no caller. This is the caller, in the shape of
`foundry-parity-lab::method`: the store holds a `MethodBinding` whose WHEN
guard is a `SetExpr` over the class's field positions; the executor admits
the `MethodRef`, binds the method's `reads` through `CubeLanes`, takes both
lanes and the live plane with one `borrow_all_present`, lowers the guard once
(`quack::set::lower_set`, `two_valued`, `lower` to a `Count` program) and
keeps the program. `run` only executes it over the cube's own words.

It lives in test support, not in the crate: `tests/support/method.rs`
(`Prepared::prepare`, `Prepared::run`), `tests/method.rs` (3 tests),
`examples/method_cost.rs`. No crate source changes.

The method is `from(src) and not to(dst)`: `x == c(src) AND y != c(dst)`.
The lanes hold tile coordinates, so the guard's constants are the
coordinates of the two ids in the table's frame at registration.

Tests (whole crate: 100 passed):

- the count equals an edge walk that shares no code with the cube, for three
  methods (40 rows, repeats in both columns, one pair held twice); at least
  one count is above 1, at least one differs from the count with the two
  fields swapped, and at least one method's second conjunct removes a row;
- a preparation made after an edge is logged counts that edge;
- a reference issued by another store is refused at admit; the store's own
  is admitted.

Disables, each red on the first two tests (`git checkout` after):
the socket hands columns out by field position instead of by predicate; the
guard's second conjunct is made vacuous.

Timing (`cargo run --release -p lance-graph-xlog --example method_cost`,
median of 31 prepares and 201 runs, every run's count asserted against the
edge walk, two invocations on a machine at load average about 1):

| rows | count | prepare | run | run per row |
| --- | --- | --- | --- | --- |
| 1 024 | 5 | 2 600 to 3 415 ns | 314 to 406 ns | 0.31 to 0.40 ns |
| 16 384 | 56 | 2 616 to 3 377 ns | 3 872 to 4 844 ns | 0.24 to 0.30 ns |
| 262 144 | 1 025 | 2 488 to 3 362 ns | 87 919 to 143 826 ns | 0.34 to 0.55 ns |

`prepare` does not grow with the relation: it admits, binds, borrows and
lowers, and reads no row. `run` is one pass over two lanes and the live
plane.

Open:
- The guard's constants are coordinates of one frame. A method registered
  with them is tied to that frame's map; nothing here refuses it after the
  map changes a coordinate. Constants as identities, translated once per
  binding, is the unbuilt half.
- Not a crate API. Whether xlog should own a prepared-method type, or the
  one in `foundry-parity-lab` should move somewhere shared, is undecided.
- One tile only; all rows live in the tests, so the live plane is passed and
  never excludes a row.
