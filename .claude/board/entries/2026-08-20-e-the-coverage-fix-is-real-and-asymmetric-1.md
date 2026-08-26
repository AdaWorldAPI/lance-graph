## 2026-08-20 — E-THE-COVERAGE-FIX-IS-REAL-AND-ASYMMETRIC-1

**Status:** ⊘ HEADLINE SUPERSEDED same-day by
`E-THE-FILTER-WAS-FILTERING-ON-THE-WRONG-PREDICATE-1` — the watcher-sample
numbers below stand, the `0/5,760` verdict claim does NOT. FINDING (measured; `strategy/stage25_census.rs`, artifacts at
`docs/probes/stage25-consumer-filter-census.{md,csv}`). Quantifies
`E-A-WATCHER-THAT-CANNOT-DISSENT-IS-NOT-A-WATCHER-1`, which recorded the same
result qualitatively and could not say how large it was.

The Stage-2 consumer filter was documented as "a coverage fix, not a behaviour
change" on the strength of one sentence — *changes which watchers are sampled,
changed no verdict*. Stage 2.5 measured both halves with `jc::stats` over the
same exhaustive 5,760-cell paired design, and **the two channels are not the
same size of effect**:

| channel | configs changed | mean Jaccard distance | mean retention | max sym-diff | verdict change |
|---|---|---|---|---|---|
| same-family (`peripheral_dissent`) | 1440/5760 (25.0%) | 0.1265 | 0.9028 | 6 | **0/5760** |
| cross-family (`cross_family_dissent`) | 2400/5760 (41.7%) | 0.2599 | 0.7993 | 12 | **0/5760** |

**Pooling them was the first thing that had to go.** The pooled mean is 0.1932,
which fell just under a 0.20 cutoff and produced the single word "weakly" for
two channels differing by a factor of two. That is a threshold artifact of the
kind this workspace's own falsifiability rule warns about — the report now
grades per channel against a stated rule (`≥ 0.20` mean Jaccard distance OR
`≥ ⅓` of configurations changed ⇒ *materially*), so the word is checkable
rather than editorial. **Cross-family is material; same-family is weak.**

**Where the effect lives.** Concentrated exactly where the silent watchers do:
`Surface`/`Shallow` carry all three (`ARE` 19, `ZCF` 24, `HKF` 34), and the
peak cell replaces **more than half** the sample (cross-family
`Surface`/`k=3`: Jaccard distance **0.5667**). It falls to exactly 0.0000 at
`Contextual`/`Analogical` for several budgets. Descriptive η² over the design:
same-family is dominated by **style** (0.2761) then `k` (0.1987); cross-family
by **rung** (0.2543) then `k` (0.1795). Style stayed a stratum even though the
Stage-2 verdict surface is inert to it — and it turned out to be the
*largest* factor on one channel, which is the reason to keep an inert-looking
stratum rather than drop it.

**The verdict half, exactly rather than smoothed.** 0 discordant of 5,760 on
BOTH channels, on the FINE label (the elevation `RungLevel` — which watcher
objected — not merely fired/not-fired). Cohen's κ = 1.000000 and, importantly,
**defined**: both outcome categories occur (n11=2514/n00=3246 and
n11=4290/n00=1470), so the perfect agreement is a real measurement and not the
degenerate constant-column case `jc::stats::cohen_kappa` refuses to score.
**McNemar is deliberately NOT reported** — it is a test on the discordant
cells and there are none, so the statistic is degenerate, not significant.

**The bound is a ladder, not a number.** With zero events the exact
one-sided Clopper-Pearson limit is `1 − α^(1/n)`; the 5,760 rows are repeated
measures (one style contributes 160), so quoting `5.2e-4` would assume an
independence the design does not have. Reported at every clustering
assumption — cell `5.2e-4` · style×rung `2.1e-2` · style `8.0e-2` · cluster
`3.9e-1` — with the instruction to quote the one whose independence you are
willing to defend.

**A methodological note that generalises past this measurement.** No
inferential test is run over the census, and the report says so in its own
header: the enumeration is exhaustive and deterministic, so there is no
sampling distribution for a p-value to describe. `jc`'s `t_test_*` /
`anova_one_way` p-values are therefore not reported; what IS used is
cross-tabulation, variance decomposition, and rank association — the things a
census can honestly support. The one exception is the zero-event bound, which
is a statement about an unobserved population and is exactly where inference
belongs.

Also recorded: `multiple_r_squared`'s joint figure is **smaller** than the
largest single η² on one channel (0.0545 vs 0.2761), and that is not a
contradiction — it fits the factors as LINEAR predictors over integer codes
while η² groups them nominally, so a factor whose effect is non-monotone in
its code (style, whose code is an enum position) is largely invisible to the
linear fit. Read as *"how much a linear read of the design explains"*, never
as a ceiling.

---

