## 2026-08-23 — E-A-WARRANT-MUST-BE-ABLE-TO-SAY-NO-1 — the trajectory is reconstructible AND grounded; the warrant channel that could not refuse was the defect

**Status:** FINDING (measured — `PROBE-WARRANTED-VIEW-TRACE-1`, 9/9 gates
green, `examples/probe_view_edit_trace.rs`). **Confidence:** High; the ABSENT
at the end bounds it precisely.

**Three vicious invariants, all green.** Over the sealed field (2016 beliefs,
20 rows across rungs {1,2,3,4,6}), trajectory `A(10) -e1-> 4 -e2-> 10 -e3-> 4`:

- **RECONSTRUCTION** — `replay(A, edits) == final` (T1), and stronger,
  `replay(prefix[0..k])` equals the view actually observed at step k for EVERY
  k (T2). An end-to-end match can hide two errors that cancel; a per-prefix
  match cannot. All three edits invert step-by-step back to A exactly (T3).
- **GROUNDING** — every edit NAMES evidence from the sealed state (G1);
  every named count re-verifies against that state (G3, 9 items recounted).
- **ANTI-HINDSIGHT** — `warrant_at(step, initial, prefix, edit, arena, ctx)`
  has no parameter through which a later edit, the final view, or an outcome
  could enter, and all warrants reproduce byte-equal from their prefix alone
  (T4). `witness_fabric`'s `upto` discipline, transplanted.

**THE CORRECTION THAT MATTERS — the first warrant channel could not say no.**
The draft `Warrant` carried `visible_before` / `rungs_before`: a DESCRIPTION
of the situation, which EVERY possible edit satisfies. It would have passed a
reconstruction suite while measuring nothing about justification — the exact
house anti-pattern (*"a guard that fires on everything carries exactly as much
information as one that never fires"*). The fix was not a stronger assertion
but a different object: `Evidence::{BeliefsAtRung{rung,count},
RowsBoundAt{locus,count}}`, drawn from the sealed state, plus **G2, which
proves the channel can REFUSE** — an off-field `RungBand{50,60}` (above the
measured R6 ceiling) and an unbound `Locus::Contradiction` both come back
UNGROUNDED while the real trace stays fully grounded.

**Why that gate is load-bearing for anything downstream.** Without G2, a
later BPE-style learner would compress a recurrent BAD habit exactly as
efficiently as a good one — the same `[e1,e2,e3]` subsequence looks identical
whether real evidence or circular reasoning drove it. G2 is the difference
between acquiring reasoning skill and acquiring superstition, and it belongs
BEFORE the learner, not after.

**The remaining gap is now one arrow.** `Evaluation → Revision → warranted
ViewEdit`. A trace OBJECT exists; a trace PRODUCER does not —
F-REVISION-FOCUS-1 stays ABSENT and every trajectory here is AUTHORED by the
probe, not harvested from a running cognition. `BehaviorTrace` is probe-local
and is NOT proposed as a production type. Nothing is learned, compressed,
promoted, or recurrence-detected.

