# 2026-11-15 — PmLedger on abakus::pm; #2086 merged without a final-head review

STATUS: done | OUTCOME: `crates/xlog-relcomp-probe` `PmLedger` no longer carries
a copy of the `P × M` arithmetic; it calls `ndarray::hpc::abakus::pm`
(ndarray #433, one source of truth) through `PmSum::into_saturated_parts`.
The i64-model tests (6/6) and the D-XLOG-PM-1 triangle-pairing arm give the
same counts as before (0 of 47,646 batches wrong). Disable: a local ndarray
patch that zeroes `add_pm`'s carry fails 4 of the 6 ledger tests.

## Governance exception: #2086

lance-graph #2086 (cube safety P0) merged on 2026-11-15 10:34:28 UTC at head
`ece628bc4`. Its stated gate was an independent review of the final head. The
only independent review on record is Codex's, of `a551a32`, two pushes
earlier; no review covered `a19545a90` (the pre-check regression test) or the
rebased `ece628bc4`. CI was green on `ece628bc4`. This is recorded as a merge
without its review gate, not reclassified as reviewed. A retrospective review
of `ece628bc4` would close the assurance gap; it has not been done.

OPEN: retrospective review of #2086 at `ece628bc4`; P4 economics (including
checked `i64` arithmetic in the escape); wasm tiers of `tests/add_pm.rs` not in
ndarray CI.

## 2026-11-15 (later): retrospective review of #2086 done; escape checked

**Retrospective review.** An independent reviewer read the merge `98f4e1c18`
(head `ece628bc4`) in full: `cube.rs`, the codegraph `graph.rs` refresh path,
`cube_safety.rs`, and the changed tests. It checked five claims, each against
cited lines:

1. every write to the base lanes bumps the generation;
2. `commit` refuses before it mutates anything;
3. `check_commit` and `commit` agree within the stated condition;
4. the codegraph `unreachable!` holds;
5. each safety test fails without the check it guards.

**No correctness defect.** Three notes, all addressed in the follow-up PR:

- (a) `a_batch_of_another_cube_is_refused` used another table, so its frame
  check refused the batch even without the id check. A twin over the same
  table and frame now isolates the id check. Removing it fails that test.
- (b) A stale `prepare_refresh` doc bullet said "log was empty"; the rule is
  `Cached::Base`.
- (c) The `base_gen` doc said "with different ones"; any non-empty fold
  counts.

The gap recorded above is closed by this review. It does not reclassify the
merge as reviewed at the time it was made.

**Checked escape.** `PmLedger::add` and `absorb` return
`Result<(), PmOverflow>`. A sum past `i64` is refused and changes nothing; for
`absorb`, a prepass over the keys that are wide on either side makes it
atomic. `abs()` on `i64::MIN`, which panicked, is now `unsigned_abs()`. Four
tests were added; three disables each fail their own tests.

**Merge order (DECISION).**
- A stacked or cross-repo PR says "do not merge before #N" in its title.
- A PR is not merged before CI has run on its last push, and not before the
  review it names as its gate has covered that push.

BASIS: #2087 merged before #2086, #432 merged before its last push, and #2086
merged without its review. REVISIT WHEN: a merge queue enforces the order.
First applied: lance-graph #2091, which waits for xlog#8.

OPEN: P4 economics; wasm tiers of `tests/add_pm.rs` not in ndarray CI.
