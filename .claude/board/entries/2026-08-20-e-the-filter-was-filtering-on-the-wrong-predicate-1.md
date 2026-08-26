## 2026-08-20 — E-THE-FILTER-WAS-FILTERING-ON-THE-WRONG-PREDICATE-1

**Status:** FINDING (measured; corrects `E-A-WATCHER-THAT-CANNOT-DISSENT-IS-NOT-A-WATCHER-1`
and **⊘ STORNOES the headline of** `E-THE-COVERAGE-FIX-IS-REAL-AND-ASYMMETRIC-1`,
both written earlier the same day on this branch). Origin: codex review on
PR #971, against the first version of `StyleStrategy::watcher_can_dissent`.

**The claim that was wrong.** Both entries above reported that the consumer
filter changes which watchers are sampled but changes **no verdict** — 0 of
5,760 paired configurations, on both channels, κ = 1.000000. The measurement
was sound. The predicate it measured was not.

`watcher_can_dissent` filtered on `maturity().is_production()`. But
`Operational` is a **disjunction** — `maturity_operational_implies_an_effect`
requires *mutates some `ThoughtCtx` field* **OR** *moves confidence* — while
both dissent channels compare exactly one quantity, `tc.confidence`. Measured
over the 34 kernels:

| | count |
|---|---|
| `Operational` | **31** |
| can move `delta_conf` | **14** |
| declare `ThoughtField::Confidence` in `writes()` | **0** |

So the filter removed 3 mute watchers and admitted **17 more that were equally
mute** — it *preserved the exact budget loss it was introduced to remove*. The
sharpest cases are `Cas` and `Etd`, both carved to production in this same
arc: both rewrite `candidates`, both return `0.0` forever.

**The fix, and why it needed a new contract method.** Capability is not
derivable from anything the trait previously exposed: `writes()` is the census
of `&mut ThoughtCtx` mutations, and `delta_conf` is applied by `run()`
*afterwards* — deliberately a separate effect, which is why
`no_kernel_writes_outside_its_declared_mask` calls `apply` directly. So
`Tactic::moves_confidence()` was added: non-defaulted like `requires` and
`maturity`, declared per kernel, and pinned **two-sided** against the probe
matrix (`moves_confidence_matches_observation` — over- and under-declaring both
fail) plus a subsumption pin (`moves_confidence` ⇒ `Operational`, checked
rather than commented, so the consumer can filter on capability alone without a
redundant maturity conjunct).

**The corrected result inverts the headline.**

| channel | sample changed | mean Jaccard dist | retention | **verdict change** | κ (fired) |
|---|---|---|---|---|---|
| same-family | 4080/5760 (70.8%) | 0.5306 | 0.4958 | **1098/5760** | 0.6307 |
| cross-family | 4224/5760 (73.3%) | 0.5269 | 0.5681 | **384/5760** | 0.8098 |

**And it has a direction, which is the part worth keeping.** On the same-family
channel `n10 = 0` **exactly**: no configuration that dissented with the filter
OFF goes silent with it ON. That is the coverage argument turned into a
measurement — removing only structurally-mute watchers cannot remove an
objection, because a mute watcher could not have raised one. Dissent rises
2514 → 3612 (+43.7 %).

Cross-family is *almost* one-way: 366 gained against **18 lost**, and the 18
are the SAMPLER, not the filter. `peripheral_sample_where` strides `k` picks
over the eligible list, so shrinking that list changes *which* capable watchers
are picked; a capable dissenter selected under OFF can fall off the stride
under ON. Pinned exactly rather than rounded to "one-way".

**Three process findings, each independent of the numbers.**

1. **A green census is not a correct census.** The Stage-2.5 harness measured
   the filter faithfully and reported zero — because it was pointed at a
   predicate that could not matter. Nothing in the suite could have caught
   that; it took a reader who asked what `Operational` actually guarantees.
   The census *did* earn its keep the moment the predicate changed: it went red
   on the first run and named the number.
2. **The report hardcoded its own conclusion.** `render_report` wrote
   `**Verdict change: 0/{n}.**` as a literal in the format string — so it
   would have printed zero regardless of what was measured, and did, for one
   revision after the correction landed. A report that *states* its result
   instead of deriving it is not a measurement. Now computed from
   `binary_association`, and the test is what caught it.
3. **A coarse label can certify something it never looked at.** The
   cross-family verdict is `(RungLevel, Mechanism)`; the census reduced it to
   the rung, so a swap to a different mechanism at the same rung would have
   counted as agreement while the report claimed the mechanism agreed exactly.
   Both components are now encoded into one nominal label (`cross_label`).
   (Also codex, PR #971.)

**Bound reporting, made reproducible** (CodeRabbit, PR #971). The zero-event
Clopper-Pearson limit `1 − α^(1/n)` is now stated with **α = 0.05 (95 %
one-sided)**, and it is repointed at the surface that IS still zero — the
same-family `n10`. Each rung of the clustering ladder names its independent
unit: one `(style, rung, k, tol)` cell (5760, optimistic); one `(style, rung)`
cell (144, `k`/`tol` as within-cell repeats); one style (36, a 160-cell block
as one observation); one style cluster (6, styles sharing a `Mechanism` — the
most conservative unit the design offers).

---

