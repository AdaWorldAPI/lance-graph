# D-DRCA-3: DR #4 method routing falsifier

**Status:** TEST-PINNED, lab only. Code: `crates/dr-math-binding-probe/src/dr4.rs`, test `tests/dr4_method_routing.rs`; plan `.claude/plans/d-drca-3-dr4-method-routing-v1.md`. Contract: D-DRDT §5 (#1835).

## Result

- Two methods on one cell with equal read/write sets and effects `[Set(1), Add(1)]` vs `[Add(1), Set(1)]` (oracle 2 vs 1; `check_reorder` refuses the swap) stay two routed entries, each with its own effect-order reference. Two methods whose effects commute also stay two entries: merging is not DR's decision.
- A declared guard with no witness refuses (`GuardDropped`), a different witness refuses (`GuardMismatch`), the right witness is carried; an unguarded method needs none.
- Refused, each with its own variant: stale register frame, resolution under another frame issue (including another store at equal numbers), foreign class, duplicate key, coordinates outside 64 × 64, numeric fold, presence over a three-valued register.
- Cell, row and column values equal a scalar loop; K3 value is masked by validity (the fixture has UNKNOWN cells with a 1 payload).
- DT-owned values (method key, guard witness, effect-order reference) are opaque to DR. The effect-order law is ndarray #395's, ported verbatim into the test as the oracle.
- 12 tests; 7 disable runs, each red on its named test.

## Open

- The method key is test-minted; the real one is DT's (D-CMH §2.1, decision ledger item 2). The DT resolver is D-CMH-1's `PreparedClass`, not built here.
- DT gaps of D-DRDT §4 stay with the contract owner: no MethodId independent of name (`ActionDef.predicate`), single `field == value` guard, no effect order, no single lookup.
- DR #4 over a `LocalSelection` or a `Chain` composition is untested.
