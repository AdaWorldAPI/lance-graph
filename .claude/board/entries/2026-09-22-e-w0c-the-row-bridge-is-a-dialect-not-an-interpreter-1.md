## 2026-09-22 — E-W0C-THE-ROW-BRIDGE-IS-A-DIALECT-NOT-AN-INTERPRETER-1 — one checked R2IL vocabulary and one typed Dialect execute scalar R2IL and population folds on the same stack; the enum explosion is upstream of mask-risc

**STATUS: measured | SCOPE: `crates/r2il-mask-abi-probe/tests/row_bridge.rs` | BASIS: capstone audit "stop hand-rolling semantics", read against HEADs of ogar-loco, ogar-r2il, mask-risc, quack, ndarray `simd_masking_ops` | SHIPPED: #1258 (merged `839de71c`) + the post-review follow-up**

⊘ **This entry was COLLAPSED on 2026-09-22 and now states the current
contract only.** It had accreted three layers: the original W0C finding, a
same-day amendment replacing it wholesale when W1a landed, and a correction
nested inside that amendment. All three described the same object at
different moments, and two of them were already false. Per this repo's own
closeout doctrine — *"the closeout should usually be SMALLER than the
reasoning history it closes"*, and the transient tier is where a superseded
finding dies — the journey stays in git and in #1258's arc rather than here.
Nothing below is a correction of a correction; it is the measured state.

### What it establishes

A merged relational operation carried as loco program DATA reaches the fused
executor with no population crossing, and the seam is a `Dialect`, not an
interpreter. loco's `Interpreter` dispatches; a `Vocabulary` above
`DOMAIN_FLOOR` declares arity/pushes as a table; a `Dialect` whose stack
value is a mask-risc slot NAME turns interpretation into PROGRAM
CONSTRUCTION; `execute_into` runs it tile-fused. loco refuses
`FOR_EACH`/`FOR_RANGE`, so it structurally cannot sweep rows.

Stronger, and the headline: **one checked R2IL vocabulary and one typed
`Dialect` execute ordinary scalar R2IL operations and population folds in
the SAME body, on one stack.** `the_mathcad_case_runs_two_folds_and_subtracts_them`
is the proof — two folds run, each returns a scalar at its fold boundary,
and R2IL arithmetic continues over the results.

### Measured

| quantity | value |
|---|---|
| probe tests | 16, plus 6 in `algebra_differential` |
| dialect-side allocation | 728 B, independent of row count |
| executor scratch | tile-local at 65,536 rows |
| Mathcad folds | `programs_run == 2` |
| quack agreement | bit-identical to `lance_graph_quack::lower` |
| disable arms | 8, each red-then-green |

### Two things it does NOT establish

Both were claimed more strongly than the code supported, and review caught
them rather than a test.

1. **Not that two classids are needed.** The probe never touches
   `VocabularyRegistry`, `CONCEPT_R2IL_MACHINE` or `CONCEPT_R2IL_FOLD`. It
   calls `validate(R2ILVocabulary)` and nothing else. OGAR #306 mints those
   ids and they may be right as an ENTRY-POINT discriminator between the
   machine and folded readings of one table, but scalar R2IL and folds
   coexisting inside one folded body needs no cross-vocabulary call, because
   that is exactly what this file does without one.
2. **Not that three FRONTENDS agree.** Three hand-written byte programs in
   three frontend SHAPES. Only the quack-shaped one is checked against an
   independently executed oracle. `blockly_abi::lower_program_with_pool` is
   never invoked and Mathcad is not a producer. Feeding real blockly-rs
   output through this dialect is the upgrade, blocked only on that repo not
   being present locally.

### The stale-slot gap, and why the first falsifier for it was too weak

`finalize_and_run` takes `ops` and resets `next_slot` to 0, so a `Val::Slot`
held across a fold boundary refers to a discarded ops graph AND its number
can be reissued. Fixed by epoch-tagging: `Val::Slot { slot, epoch }`, an
epoch bumped where `next_slot` resets, and `FoldError::StaleSlot` at every
consuming site.

**Unreachable from a loco body today, for a named reason that is also the
trigger.** All nine value-producing arms in `Dialect::call` push exactly one
result, nothing consumes without pushing, so the stack never shrinks past a
value beneath a fold's result. `Store` (arity 2, pushes nothing) is
`Unimplemented`; implementing it opens the path.
`no_implemented_op_can_expose_a_stale_slot` pins that reason.

**The severity measurement needed a second fixture, and this is the part to
remember.** The first disable run produced mask-risc's own
`ScratchReadBeforeWrite`, suggesting a downstream net already covered it.
That was a fixture artefact: the test's positive control folds, clearing
`ops`, so by the negative case nothing had written slot 0 and read-before-write
was the only possible outcome. With the current epoch's write still live
(`a_stale_slot_that_aliases_a_live_slot_is_refused_not_answered_wrongly`,
which mints the live predicate and deliberately does NOT fold it), the
disabled guard returns **`Ok(())` carrying 1134 where the correct answer is
30** — the sum under the wrong predicate, reported as success. So the
downstream net does not cover the aliasing case and the severity is a silent
wrong number, not a refusal.

### Open, unfixed

- **One trait-level control-flow gap with two instances.** `Dialect::truthy`
  returns `bool` and `Dialect::repeat_count` returns `u32`, both through
  `&self`, so a dialect can detect but not refuse. A population reaching a
  branch is poisoned and reported; a `REPEAT` over a population would
  silently run zero iterations. OGAR's own reference dialect shows the same
  shape, mapping a negative count to 0 via `unwrap_or(0)`. One upstream
  ogar-loco change makes both fallible. `run_frontend` no longer reports a
  poisoned run as success, which contains the first instance at this layer
  without closing the trait gap.
- `GROUP_SUM`'s kind-selection arm is declared in OGAR's arity table and
  exercised by no dialect.
- `quack::lower`'s redundant third facade pass: measured, pinned two-sided,
  not fixed.
- `CONSTANT` (the pool) stays unwired here.
