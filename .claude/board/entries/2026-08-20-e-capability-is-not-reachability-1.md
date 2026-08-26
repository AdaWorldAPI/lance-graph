## 2026-08-20 — E-CAPABILITY-IS-NOT-REACHABILITY-1

**Status:** FINDING (measured; codex review on PR #971, third pass). The same
lesson a **third** time, one rung finer each round:

| round | the predicate that looked right | why it was not |
|---|---|---|
| 1 | *(none — no filter at all)* | context-blind watchers counted as agreement |
| 2 | `maturity().is_production()` | `Operational` is a DISJUNCTION; 31 admitted, only 14 can move `delta_conf` |
| 3 | `moves_confidence()` | capability on SOME input ≠ reachability in THIS dispatched context |

**The measurement.** `Mcp` (recipe 10) declares `moves_confidence() == true`
and that declaration is *true* — its branch needs
`confidence > 0.7 && free_energy > 0.5`. But `thought_ctx_from` starts at
`ThoughtCtx::new`'s `free_energy = 0.5`, and **exactly one** of the 34 kernels
writes that field at all (`Rte`, id 1), which only decays it. Swept over all
36 styles × 5 rungs: `free_energy` **never exceeds 0.5**, and `Mcp` moves
confidence in **0 of 180** cells. It is admitted by the filter and is
guaranteed silent.

**Deliberately NOT fixed, and the reason is not laziness.**

1. **A reachability filter is close to circular for the budget argument.**
   Deciding whether watcher `W` can move the answer in context `C` essentially
   requires evaluating `W` in `C` — which is what sampling it already does. The
   slot is not saved. What a reachability notion would genuinely buy is a
   refusal to count a *structural* silence as agreement — which is a change to
   what **dissent means**, not a filter tweak.
2. **That is Stage-3's decision**, alongside the other half of the same
   question (the 17 Operational-but-mute kernels), and making it here would
   move the Stage-2.5 baseline the operator has just frozen as authoritative.

**What generalises past this instance.** Each round the predicate was a true
statement about the *producer* and a wrong statement about what the *consumer*
observes. That is the same shape as `E-THE-RECIPE-SURFACE-IS-CAUSALLY-BLIND-1`
(substrate the projection does not carry in) and the 17 mute kernels (effects
the projection does not carry out). **Three independent measurements, one
underlying fact: `ThoughtCtx` is a lossy projection and every predicate written
against the producer side will keep being wrong about the consumer side until
that projection is made explicit.** Tracked in
`TD-THOUGHTCTX-IS-A-LOSSY-PROJECTION`.

**Two comparator hardenings landed in the same pass** (both codex, both real):
the Stage-2.6a resolver now goes through the V3 edge's OWN `target()` instead
of a side channel — a regressed `from_v1`/`target()` would previously have been
invisible because both arms resolved the expected SPO from the spec; and the
corruption falsifier now requires the **weight-side** leg to fail on a
*composition* invariant, because the input-side assertion alone let the weight
rehydration be replaced with the direct edge and bypass V3 unnoticed.
Disable-verified: that bypass now fails, and previously passed.

---

