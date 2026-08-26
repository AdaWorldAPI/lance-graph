## 2026-08-20 — E-A-WATCHER-THAT-CANNOT-DISSENT-IS-NOT-A-WATCHER-1

**Status:** ⊘ CORRECTED same-day by
`E-THE-FILTER-WAS-FILTERING-ON-THE-WRONG-PREDICATE-1` — the MECHANISM below is
right and is why the arc exists; the "measured null" is an artifact of
filtering on the wrong predicate and is retracted. FINDING for the mechanism;
~~**MEASURED NULL** for its present effect~~. Both halves recorded, because the second is what stops the first
being overclaimed.

`StyleStrategy::{peripheral_dissent, cross_family_dissent}` sample `k`
peripheral tactics as observers and elevate the rung if one of them moves the
score. Their eligibility predicate filtered on `Mechanism` only. But a
`Demonstration` kernel lands **no effect by construction** — enforced, in the
contract crate, by `non_operational_kernels_land_no_effect` — so sampling one
spends a `k` slot on an observer that *structurally cannot dissent*, and its
guaranteed silence is then counted as agreement.

That is `E-ANTI-EIGENVALUE-MACHINERY-CAN-ITSELF-BECOME-THE-EIGENVALUE-1`
inverted. The can-fire / can-stay-silent pair asks whether a guard
discriminates; this asks something prior — **whether the instrument is
connected at all**. A watchdog that can never bark reports the same silence
as one with nothing to bark at, and only the first is a lie.

Measured periphery before the fix: `Surface`/`Shallow` carry 3 silent
watchers of 30 (ARE 19, ZCF 24, HKF 34); `Contextual`/`Analogical` carry 1 of
23 and 1 of 10.

**The null, stated plainly.** Adding the maturity clause visibly changes WHICH
watchers are sampled (the eligible list shrinks, so the stride moves — e.g.
`Surface`/`StructuralDivergence`/same/`k=8` goes `[4,6,9,13,23,28,31,34]` →
`[4,6,9,13,23,28,31]`), but across the full **5,760-cell** sweep of
style × rung × `k` ∈ {1,2,3,4,8} × `tol` ∈ {0, .001, .005, .01, .02, .05, .1,
.2} it changed **no verdict on either channel**. It is a COVERAGE fix, not a
behaviour change, and it is documented at the call site as exactly that.

**Why the null is not a reason to drop it, and how it is kept falsifiable.**
The channel is emphatically not inert — suppressing the watcher run outright
moves **4,830 of those same 5,760 cells** — so the instrument matters; what
does not currently matter is *which* of the surviving instruments is picked.
A guard with no falsifier would be the anti-pattern, so the falsifier was
written at the level the change actually operates: every watcher the shipped
predicate samples can dissent, **plus** the anti-vacuity half proving the
mechanism clause alone would have sampled one that cannot. Both halves are
disable-verified red.

**A structural note worth keeping.** The reason the identity of the watcher
does not move the verdict is that `|tc.confidence − admitted|` is dominated by
a term independent of the watcher: `tc` runs the admitted set *and* the
watcher, while `admitted` comes from `reliability_at`, so any admitted-set
effect cancels and any constant offset between the two paths crosses `tol`
regardless of who observes. That is worth measuring before anyone tunes `tol`
against this channel — recorded here rather than acted on, since it is
outside this PR's scope.

---

