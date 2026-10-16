# wcoj-closing-atoms-v1 — the exact cyclic existence mask: two closing atoms

> **⛔ THIS IS A VIOLATION. IT READS A VALUE AT A POSITION TAKEN FROM ANOTHER LANE'S VALUES (GATHER / FOREIGN LANE / PREIMAGE HOP). GATHER IS FORBIDDEN (2026-10-16). DO NOT BUILD FROM THIS. THE CORRECT PLANS ARE [`d-holo-h14c-thought-first-v1.md`](d-holo-h14c-thought-first-v1.md) AND [`d-holo-h14d-logic-mask-read-v1.md`](d-holo-h14d-logic-mask-read-v1.md): `Thought(t)[T_a[i] : T_b[i]]` — THE PAIR SITS AT THE SAME COORDINATE; THE JOIN IS THE LAYOUT.**

> Status: RATIFIED SPEC (5+3 council, 2026-10-16); not authorized to build.
> D-ASM-11 is "Queued (not authorized)"; ratifying this spec does not change that.
> Council record: board entry `2026-10-16-wcoj-closing-atoms-council.md`.
> Evidence: entry `2026-10-16-wcoj-probes-1-3.md` (P-7 insufficiency);
> co-research map `2026-10-16-coresearch-wcoj-leapfrog.md` (PR #1586).
> Grades: [G] read in source; [H] design hypothesis awaiting its gate.

## 0. Two questions, two populations

The semijoin chain over-approximates a cyclic body (F4 keeps every row of an
empty query). The exact mask needs a closing atom. The premise audit split the
question in two, by signature:

- **Q-A, comparison family.** Compare a local lane with a foreign lane read
  through an address lane. Three-valued like `EqU32Via`. O(1) work per row.
  Head population: the table holding both local lanes (triangle fixture: node
  rows `a`, lanes `rb`, `tc`).
- **Q-B, EXISTS family.** For each row of a named head population (triangle:
  R-edge rows), do two set-valued addresses (sorted runs) share a value?
  Two-valued NULL rule like `Semijoin`. Data-dependent work per row.

The two populations coincide only when R is functional. Part A's head rows are
node rows `a`; Part B's are R-edge rows; a Part A mask is never a Part B mask
on the same index space.

## 1. Frozen decisions

1. No opcode merely to pass an experiment; each opcode carries its own
   insufficiency proof (decision 14).
2. No second algebra or interpreter beside mask-risc / Quack. An IR that
   describes is fine; a layer that executes beside mask-risc is not.
3. Chained hops fold into one mask expression; nothing is materialized per hop.
4. Nothing is a foreign key in the relational sense; a hop is a coordinate
   range (D-TELEPORT). The IR field name `fk` on `Pred::EqU32Via` is an address
   lane; new fields keep it for continuity.
5. Performance is cited only from the v4 arm, per element, tier printed.
6. All SIMD comes from `ndarray::simd`; W1a contract
   (`ndarray/.claude/knowledge/vertical-simd-consumer-contract.md`); a
   scalar-only kernel states its exemption in its doc comment.
7. The frontier stays inside one mask-risc program and never crosses as a
   `ForeignPlane`; integer semirings only; routing lives in relalg/R2IL, not the
   planner; DataFusion is in grace.
8. No strings in rules, queries or IR.
9. Lance family upstream; AdaWorldAPI forks elsewhere; ndarray `master`.
10. [H] Part B is not expressible with today's ops: re-anchoring on an edge
    table leaves a second fan-out, which the gremlin adapter refuses as
    `Refusal::NonFunctionalChain` (`quack/tests/gremlin_parity.rs:515`, `:33-35`).
11. k-way is not AND of 2-way atoms: `∃c∈A∩B ∧ ∃c∈B∩C ∧ ∃c∈A∩C` is weaker than
    `∃c∈A∩B∩C`, the semijoin-chain over-approximation one level up. k-way needs
    its own fixed arity (`Pred` is `Copy + Eq`, `ir.rs:126`).
12. A new `ExecError` variant is accepted and paired: lgj-abi
    `exports.rs:1737-1769` matches `ExecError` exhaustively, so the Part B
    lance-graph PR carries the lgj-abi status mapping in a paired PR.
    `LaneNotOrdered { lane }` is NOT reused: its `lane` indexes local lanes
    (`value.rs:181`), and foreign/local errors are kept apart
    (`ForeignLaneOutOfRange` vs `LaneOutOfRange`, `value.rs:185-187`).
13. Planner `AdjacencyBatch::intersect` (all-pairs two-pointer merge on u64)
    and the logical-only `WcoJoin` node are not reused.
14. Insufficiency, per part:
    - A: #1587 `p7_disjunction_costs_one_pass_per_domain_value`: the functional
      disjunction `∨_v (EqU32Via(rb, s_dst, v) ∧ tc == v)` grows by about 4 ops
      per value (66 at |D| = 16, 1,026 at 256), so rows · |D|. Measured.
    - B: n:m cannot be represented in functional lanes (P-7), and decision 10.
      Its per-value disjunction `∨_c (Semijoin(x, S∋c) ∧ Semijoin(y, T∋c))`
      needs one resident plane per value. Argued, not measured: G0 measures it.

## 2. Part A — `Pred::EqU32ViaLane { fk, key, lane }`

`EqU32Via` with the constant replaced by this row's own lane value. Row i
matches iff `fk[i] < len(foreign.lanes[key])` and
`foreign.lanes[key][fk[i]] == lanes[lane][i]`.

- **ndarray** `eq_u32_via_lane_to_mask(index, table, local, out_words)` next to
  `eq_u32_via_to_mask` (`simd_masking_ops.rs:3141`); re-export beside
  `simd.rs:844`; parity block, code 0xD41.
  - Scalar, exemption stated. Precedent: `eq_u32_via_to_mask` and
    `mask_gather_u32_under` (scalar because the address comes out of another
    array). The sibling's stated reason ("no vector gather on any backend",
    `:3114-3120`) is corrected in the same PR to: "an `unsafe`, unbounded f32
    gather exists on several backends (`simd_avx512.rs:236-246`, `simd_avx2.rs:703`,
    `simd_scalar.rs:1099`); there is no bounds-masked u32 gather-and-compare
    wrapper, and the out-of-range-is-false contract needs one; whether one pays
    on random access is unmeasured." The gather backend is a TECH_DEBT row.
  - `local.len() != index.len()` panics with the facade's assert message.
  - W1a criterion 7: the PR body names the consumer site (`run_pred`).
- **mask-risc**
  - `run_pred` arm: gate ANDed after, as `EqU32Via` (O(1) per row).
  - `pred_lane_and_kind` (`reference.rs:64-93`, "the one place") returns a small
    fixed list of local lanes, so `fk` and `lane` are both checked through it;
    `check_foreign_lane(key, U32)` beside the `EqU32Via` check (`:386-388`).
  - `eval_pred` oracle arm.
- **quack** `Filter::EqU32ViaLane { fk: Col, key: ForeignLane, lane: Col }`,
  builder, lowering, `gate_walk`, `Fx::oracle`, `sql_where` doc row.
  - Three-valued: UNKNOWN if `fk` OR `lane` is NULL. `leaf_validity`
    (`lib.rs:606-614`) returns up to two masks and `rewrite` (`:580-592`) gates
    on all of them. A NULL in the FOREIGN value lane is not modelled (`:537`);
    Part A inherits this, stated in its doc.
- **Claim [H]:** `R.valid ∧ T.valid ∧ Semijoin(rb, S.valid) ∧
  EqU32ViaLane(rb, s_dst, tc)` is the exact triangle mask over functional R, S,
  T, in one program whose op count does not grow with |D|.

## 3. Part B — `MaskOp::RunsMeet { x, y, s_off, s_val, t_off, t_val, under, dst }`

(v1.1 `Pred::RunsIntersectVia`, renamed and re-placed by the council.)

Row i is set iff `under` keeps it (or there is no gate), `x[i]`, `y[i]` index
runs that exist, and `s_val[s_off[x[i]]..s_off[x[i]+1]]` and
`t_val[t_off[y[i]]..t_off[y[i]+1]]` share a value. A key with no run → clear.

- **Placement: `MaskOp`, beside `Gather`.** `Gather{under}` is the template
  (`exec.rs:1753-1782`, `mask_gather_u32_under`): the kernel visits only rows
  set in its `under` operand; a zero gate word still costs one test, so the
  work is O(words + Σ over gated rows). This matches `D-V4-FOLD-LAB.md:175-177`
  (WCOJ as a physical intersect, not a semantic op).
- **What `under` is under quack lowering.** `gate_walk` (`lib.rs:2079-2097`),
  `emit_gated` (`:2243`, `:2258`) and `assign_slots` (`:2340`, `:2356`) use the
  AND's first resident plane (or the accumulator when no plane exists), not the
  conjunction of all earlier conjuncts. A preceding `Semijoin` or `Cmp` does not
  narrow the rows `RunsMeet` visits. In the triangle body the gate is `R.valid`.
  Narrowing on the accumulator is the cost already filed in `ISSUES.md`.
- **Order invariant: checked completely, at validate, before any tile runs.**
  For each of the two CSRs: `len(off) ≥ 1`, `off` non-decreasing,
  `off[last] ≤ len(val)`, every run non-decreasing. A violation refuses with
  the new `ExecError::ForeignLaneNotOrdered { key }` before any output is
  written. Why at validate: a refusal raised inside the tile loop would leave
  earlier tiles written in a sink terminal (`Keep`, `GroupSum*`, `ScatterOrU32`,
  `BlendI32`; `exec.rs:1319-1337`), and a check on visited elements only would
  make Ok/Err depend on `under`, the data and the gallop schedule, so the oracle
  and exec could disagree. It also closes the `NOT` hazard: on an unsorted run
  the leaf can miss a common value (never invent one), and `NOT` would turn
  that into a wrongly kept row. Cost: O(|S| + |T|) per execution, stated in the
  op's doc. A sealed layout witness (per Lance version) that removes this cost
  is a named follow-up.
- **ndarray** `runs_meet_under_to_mask(x, y, s_off, s_val, t_off, t_val, under,
  out_words)`; re-export; parity block, 0xD6x. Precondition: the order
  invariant holds (checked by the caller at validate; `debug_assert` in the
  kernel). Returns nothing. Out-of-range keys → clear bit.
  - Per row: the shorter run proposes; the longer seeks forward from a cursor;
    exit at the first common value.
  - Seek routing follows `ordered_lane::seek_partition_point` (step if gap ≤ 16;
    whole-range search if gap² ≥ len; else gallop). Both thresholds came from
    v4 timing against the comparison counts (branch prediction; cached top
    midpoints, measured on 2²⁰–2²⁴ lanes). Neither is assumed to transfer to
    per-row runs: re-measure both inside ndarray, and record a regime where
    the whole-range branch stops winning.
  - Scalar, exemption stated. Precedent: `mask_gather_u32_under` and
    `eq_u32_via_to_mask` (the address comes out of another array). Not
    `mask_shift_morton`, whose pass 1 is vectorized.
  - A `#[cfg(test)]` element counter in the kernel feeds G4; never a probe-local
    copy of the kernel.
- **mask-risc**: `MaskOp` variant. Exhaustive matches that grow an arm:
  exec tile loop, `reference.rs` `dst_of` / validate / `run` (×2), `ir.rs` touch
  loop, the two fuse recognizers (explicit `=> return None`), `ir.rs`
  `op_histogram` (`:1279-1341`, a new `runs_meet` field), `program_collapse_probe`
  `op_dst`. Validate checks 2 local + 4 foreign `U32` lanes, the order
  invariant, and `under != Scratch(dst)`.
- **lgj-abi** (paired PR): `exec_error_to_status` arm for
  `ForeignLaneNotOrdered`.
- **quack** `Filter::RunsMeet { .. }` leaf, its own `Node` variant, 2 emitter
  arms, `gate_walk`, `gate_shape`, the test helper at `lib.rs:3148`.
  - Three-valued: Semijoin's row. Gate on `valid(x) ∧ valid(y)` (two masks, the
    same `leaf_validity` change as Part A), then complement the GATED leaf. A
    NULL key is a known FALSE, never UNKNOWN; under `NOT` it is KEPT. Part A's
    NULL row is UNKNOWN and dropped under both polarities.
  - The `if let Filter::Semijoin` at `lib.rs:584` becomes `matches!` over both
    EXISTS leaves.
- **Layout.** An edge table sorted by `(key, val)` plus a run index is the same
  carrier. Probe-local `Csr { off: Vec<u32> }` shapes exist only in probes
  (`holo_coord_probe.rs:329-332`); no contract carrier is added here.
- **Not D-TELEPORT.** Each run is a coordinate range, but the closing step
  intersects run CONTENTS in C's coordinate space (merge / gallop). That is not
  D-TELEPORT's `[max lo, min hi)`; do not cite it as covering this step.
- **Claim [H]:** `R.valid ∧ RunsMeet(x = rb, y = ra, S, T)` is the exact mask
  over R-edge rows; elements touched are Σ_{(a,b)∈R} min(|S_b|, |T_a|) ≤ N^1.5
  by Cauchy–Schwarz, and comparisons add a factor of at most 1 + log(max/min).

## 4. Non-goals

- Multi-table `Draft` binding (`quack-multitable-bind-v1`, D-QMB).
- Counting (P-11). It needs strictly increasing runs or multiplicity lanes;
  its oracle is the X17 bag oracle in `wcoj_probe.rs`. G3 is SET-only by design
  and cannot detect a multiplicity defect.
- `materialize_bindings`; the relalg router (A3); GYO wiring.
- k-way (decision 11); bodies beyond the triangle.
- A gather backend for Part A (TECH_DEBT); the sealed CSR witness.
- Hash tries; any change to `Gather` or `EqU32Via` semantics.

## 5. Gates (pre-registered)

- **G0** Write F2, F3a, F3b (map entry) in `wcoj_probe.rs` first, each with its
  datafrog SET answer. Measure Part B's per-value disjunction (decision 14) on
  one of them.
- **G1** `R.valid ∧ T.valid ∧ Semijoin(rb, S.valid) ∧ EqU32ViaLane(rb, s_dst, tc)`
  == datafrog SET on F4 ×{1, 8, 50} and `mixed` {30, 64, 200}, plus one fixture
  with `fk[i] ≥ len(foreign)`.
- **G2** Part A's program op count == a fixed constant AND < the disjunction
  program's count at |D| = 16 (`p7_disjunction_costs_one_pass_per_domain_value`
  counts `program.ops.len()`).
- **G3** Part B: the R-edge mask == the edge-level oracle
  `{(a,b) ∈ R : S_b ∩ T_a ≠ ∅}`, and its projection onto `a`
  (`Terminal::ScatterOrU32`, `ir.rs:338`) == datafrog SET, on F2, F3a, F3b, F4
  at 10²..10⁵.
- **G4** Elements touched (the kernel counter) on F3b: fitted slope in
  1.5 ± 0.1 over 10³..10⁶. Comparisons are reported, with the log term in the
  prediction, but not gated.
- **G5** disable runs, each red:
  - A: compare with `v = 0` instead of the local lane;
  - A: drop `lane`'s validity in `leaf_validity`, on a NULL-`lane` row whose
    stale payload equals `foreign[key][fk]` (sql_null_3vl fixture) → kept;
  - B: fixed proposer AND no galloping (the map's A1′), F3a oriented hub-run
    first → elements touched ≥ 10× at 10⁶;
  - B: ignore `under` on a 1%-survivor gate → elements touched ≥ 10×;
  - B: restore the Semijoin-only `if let` → the NULL-key row is DROPPED under
    `NOT` (correct: kept); the paired correct run keeps it;
  - B: an unsorted run → `ForeignLaneNotOrdered` before any write, with a sink
    terminal and the bad run in a later tile; a sorted twin stays silent.
- **G6** Seek thresholds: an inertness test per threshold (raising the step
  threshold to ∞ and lowering it to 0 each change the elements-touched count on
  a fixture that exercises it).
- **G7** Portability: the one scalar body per kernel runs identically on each
  target (`simd-masking-parity`, `wasm-simd-parity`, `neon-simd-parity`). This
  catches width/portability defects only, not backend divergence. Clippy
  `-D warnings`; mask-risc / quack / metta / sql_null_3vl / lgj-abi green.
- **G8** v4 ns per row printed with the tier; no v3 numbers.
- **G9** CI does not run `wcoj_probe`; add a CI line or state G1–G5 as
  local-only in the PR body.

## 6. Order of work

1. Authorization of D-ASM-11.
2. Part A: ndarray PR (with its blackboard entry) → lance-graph PR (with its
   STATUS_BOARD row and LATEST_STATE inventory line).
3. Part B: ndarray PR → lance-graph PR + paired lgj-abi PR, each with its board
   rows in the same commit.
