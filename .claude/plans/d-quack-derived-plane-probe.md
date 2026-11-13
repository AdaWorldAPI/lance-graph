# Derived planes: what an admitted lane's descendants inherit (lab plan)

**Status:** PLAN, falsifiers fixed before any candidate ran (2026-11-13). Lab only.
**D-ids:** D-QUACK-DERIVED-PLANE-PROBE (`STATUS_BOARD.md`).
**Follows:** `quack-boundary-lab-v1.md` (D-QUACK-BOUNDARY-LAB, #2045), whose
unsatisfied contract 1 is "no owner issues a mask plane".
**Crate:** `crates/quack-boundary-lab` (tests added; no production crate changes).

## 1. Question

A `LaneBinding` (`lance-graph-dt64/src/lane.rs`) establishes, at each use,
that a lane belongs to one source, one frame, one row order and one field
mapping. Three labs merged or opened on 2026-11-13 run programs over values
that are *derived* from lanes and carry none of that:

| lab | derived value | how it reaches the executor |
|---|---|---|
| `hidden-state-twins-probe` (`Bound::decide`, #2056) | per-row lanes written by a producer | raw `Vec<u32>`, producer named by a caller-stated digest |
| `dr256-ternlog-lab` (`prepare_edges`, #2055) | fk-visibility planes | `Plane { space, words }`, built from a caller closure |
| `dr256-ternlog-lab` (`BitPlanes::of`, PR #2057, open) | eight bit-planes of a `u8` lane | raw `Planes { masks }`, no binding |

Does a derived value inherit anything from the binding of the lane it came
from, and what is left that the binding cannot establish?

## 2. Three transformations, kept apart

| kind | example | what it depends on |
|---|---|---|
| pure representation | `u8 lane -> BitPlanes` | the input lane and a fixed recipe (`Bit(k)`) |
| derived interpretation | `lemma[head[i]] -> head_frame[i]` | the input lanes, a producer and its revision |
| cross-population derivation | `fk -> foreign visibility` | two populations, two row orders, a visibility decision |

Hypothesis: the three share a dependency rule (a derived value is valid for
exactly the frames of everything it was computed from) without sharing one
witness type. P1 to P3 test that; they do not assume it.

## 3. Scope

- No change to DT64, Quack, mask-RISC, the SoA envelope or any contract type.
- No `Scope` issuer, no field on `Program`, no foreign-plane authority, no
  universal witness type, no multi-terminal execution.
- Anything the existing source binding cannot establish is recorded as an
  unsatisfied contract. No authority is invented to make a case pass.

## 4. P1: source-admitted lanes (starts now)

Fixture: the twins rule shape (at most eight candidate rules; a WHEN per rule
that is a conjunction of `lane in {values}`; a candidate byte per row) over the
`quack-boundary-lab` source. One lane of the fixture (`discountValid`) stands
for a producer-derived lane.

Negative control, kept as it is: `hidden_state_twins_probe::Bound::decide`
over raw vectors.

Pass conditions, fixed before running:

| # | case | bound path must | raw path is expected to |
|---|---|---|---|
| 1 | right source, frame, row order, field mapping | equal a row-loop oracle; the answer selects some rows and not all | equal the oracle |
| 2 | another source with the same row count | refuse with a `LaneRefusal` before execution | accept and answer for the other rows |
| 3 | frame advanced after binding | refuse with a `LaneRefusal` | accept |
| 4 | lanes in another row order than requested | refuse with a `LaneRefusal` | accept and return a wrong answer |
| 5a | same source and frame, the producer-derived lane rewritten by another producer, the source does NOT advance its frame | **recorded, not required to refuse** | accept |
| 5b | as 5a, and the source advances its frame | refuse with a `LaneRefusal` | accept |

Case 5a is the one that matters. It may not be reported as passing because
the caller states the expected producer digest: the digest is an input of the
caller, not a property of the lanes. If the binding admits the rewritten
lane, the result is "producer provenance is not established by the binding",
listed under unsatisfied contracts with the exact line that would have to
know. `LaneSource` is not extended to make it refuse.

Also recorded in P1: the candidate byte per row is caller-supplied in both
paths.

FAIL of P1 as a whole: cases 1 to 4 need a change to `LaneSource` or
`LaneBinding`. Then stop and report the missing contract.

## 5. P2: derived-plane inheritance (after #2057 merges)

From an admitted bound lane, derive a plane inside the lab and record: the
originating field and binding, source frame and row order, the recipe
(`Bit(k)`), the source revision, any further dependency.

The test is not whether that record can be attached. It is whether the plane's
words were produced by the lab's own derivation from the admitted lane.

- A plane the caller built, carrying a matching record, must not be accepted.
- A plane derived under an earlier frame must be refused after the frame moves.
- FAIL: acceptance can only be decided by trusting a caller-supplied record.
  Then the result is "merely asserted provenance", reported as such.

## 6. P3: cross-domain falsification

Apply P2's rule to bit-planes, to a twins guard lane and to an fk-visibility
plane. One witness shape is not assumed. For the fk plane, list what it needs
beyond the local binding (the foreign population's frame and row order, the
visibility decision). If that exceeds the existing contracts, the difference is
the result; no generic authority is added.

## 7. P4: cost and memory

Measured separately, same machine, release, `ndarray::simd` backend named:
cold derivation (time, allocations), warm admission check, warm execution,
resident bytes, scratch bytes, bytes copied or written per population.

Warm execution must add no allocation. The roughly 80 ns per slab binding
check reported in `2026-11-13-weather-dr256-ternlog.md` is a reference, not a
threshold: the comparison is against the binding-check baseline re-measured in
the same run.

## 8. Kill condition

If the derived-plane witness needs an authority the existing source binding
does not supply, stop and report the missing contract.

## 9. Report shape

Every derived value in the final report is filed under exactly one of:
verified inheritance, merely asserted provenance, unauthorized derived data.

## 10. What was read

In full: `crates/hidden-state-twins-probe/src/lib.rs`,
`crates/dr256-ternlog-lab/src/lib.rs`, `crates/quack-boundary-lab/src/lib.rs`
and `tests/field_operand.rs`; the research notes
`docs/research/2026-11-13-{ce64-logic-neutrino-grammar-spine,ce64-logic-neutrino-p0-lab,hidden-state-twins-lab}.md`;
the board entry `2026-11-13-weather-dr256-ternlog.md`; PR #2057's description
and diff at head `5228f0ad`. Not read: the tests and examples of #2055 and
#2056, `dr256-ternlog-lab/src/order_store.rs`, `grammar-constraints-probe`.

## 11. P1 results (run 2026-11-13, after sections 1 to 10 were committed in `8b689d2fa`)

`crates/quack-boundary-lab/tests/twins_lanes.rs`, 6 tests, all pass.
`cargo test -p quack-boundary-lab --test twins_lanes`. No change to DT64,
Quack, mask-RISC or `LaneSource`; the lab source gained two mutators
(`rederive_validity`, `advance_source`).

| # | case | bound path | raw path |
|---|---|---|---|
| 1 | right source | equals the row loop | equals the row loop |
| 2 | another source, same row count | `LaneRefusal::Frame(ForeignFrame)` | accepts; the answer is neither source's |
| 3 | frame advanced after binding | `LaneRefusal::Frame(_)` | accepts |
| 4 | another row order | `CoordinateSpace` at bind; a `LaneRefusal` after a reorder | accepts; wrong for both orders |
| 5a | lane rewritten, frame not moved | **admitted; answers from the rewritten lane** | accepts |
| 5b | lane rewritten, frame moved | `LaneRefusal::Frame(_)` | accepts |

Four disable runs, each red: re-binding at every use (cases 2, 3, 4, 5b),
dropping the description check (5a's description arm), a rewrite that writes
nothing (5a), a revision that does not advance the store (3, 5b).

What P1 establishes, filed as section 9 asks:

- **Verified inheritance:** source identity, frame, row order and field
  mapping of a lane read through `borrow_all`.
- **Merely asserted provenance:** the producer of a lane's contents. The
  digest in the guard (`SetExpr::Sym`) is compared with the caller's
  description of the method, never with the lanes. A caller who states the
  bound digest reads a lane another producer wrote (case 5a).
- **Unauthorized derived data:** the candidate byte per row. Both paths take
  it from the caller.

Unsatisfied contract, stated exactly: a `LaneBinding` witnesses where a lane
is and under which frame, not what wrote it. The only thing that turns a
rewrite into a refusal is the source advancing its own `FrameStore`
(case 5b); `LaneSource::lane_of` and `borrow` have no argument through which
a producer or its revision could be asked for. Nothing was added to make 5a
refuse.

Fixture finding: the first selector used `id % 2` and `id % 3`. Both divide
`ROWS - 1` (40,002), so the selector was the same for a row and its mirror
and the raw path's answer over a reversed source equalled that source's
oracle. Case 2's raw assertion caught it; the selector now uses 2 and 7.
