## 2026-08-11 — E-MY-OWN-PRE-REGISTRATION-HAD-A-GAP-AND-I-NAMED-IT-1

**Status:** FINDING `[G]` — CT-F14, `comet_tail_f14.py` / `.json`, report
§5.11. Direct follow-up to E-THE-RESCUE-THAT-WEAKENED-ITSELF... below; this
entry is about a defect in *my own pre-registration design*, caught and
corrected by the same discipline it should have applied from the start.

**CT-F14 was pre-registered and committed to git BEFORE it ran** (`4f1a1b4f`)
— fixed dates, fixed bar (n≥20, ≥0.70), and a fixed interpretation table for
a "combined 3-sample" figure, all decided before any output existed. Run: 19
of 85 mechanically-generated candidates qualified (one short of the n=20
floor) → correctly **NO-VERDICT by the pre-registered rule.** The pooled
3-sample figure (n=26, 19/26=0.731) crossed the pre-committed p<0.05
"established" threshold (p=0.0145).

**By the letter of my own pre-registration, this should have been reported
as "established, ready for the audit-gate queue."** It was not, because a
sensitivity check — run precisely because the last entry demands applying
scrutiny to results that help as readily as to ones that hurt — found
something the pre-registration never anticipated: **CT-F14's own qualifying
subset, taken alone, sits at 0.684 (13/19), BELOW the 0.70 bar, p=0.0835 —
not significant.** The single test this whole probe existed to produce does
not independently support the claim it was built to test. The "established"
pooled figure is being carried by two small prior fragments (n=4 at 0.75,
n=3 at 1.00) blended with a properly-powered new sample that came in lower.
Dropping just the smallest fragment (n=3, fully saturated) still barely
clears p<0.05 (0.0466) — so the pooled crossing is not purely an artifact of
one tiny subsample, but the component that mattered most (the large, careful
new test) disagrees with the pooled verdict on its own terms.

**The gap, named plainly: my pre-registration specified thresholds for a
pooled figure without specifying what to do if the new, properly-powered
sample and the pooled figure disagreed.** I did not write a rule for this
exact configuration because I did not anticipate it — I expected CT-F14 to
either clearly pass or clearly fail on its own, not to fall one storm short
of its power floor while *also* landing under the bar. Finding that gap
after the fact and exploiting it silently (reporting only the pooled
"established" number, which the letter of my pre-registration technically
licensed) would have been exactly the failure mode this arc's discipline
exists to prevent — just moved one level up, from cherry-picking a result to
cherry-picking which of two valid readings of a pre-committed rule to report.
**Named instead: graded the verdict down to "still suggestive," and
recorded the pre-registration gap itself as the finding**, alongside a
second, smaller walk-back (§5.10's striking monsoon-band exclusion pattern,
4/5 in the small sample, thinned to 2/21 at 4× the exclusion count in this
larger one — the small-n-looked-like-a-pattern theme recurring one level
down from the main directional claim).

**The reusable lesson, sharper than the previous entry's:** *pre-registration
protects against post-hoc rationalization of the DATA. It does not
automatically protect against post-hoc selection among several VALID readings
of the rule itself, when the rule turns out to admit more than one — that
requires the same discipline applied one level up, at read time, not just at
design time.* After four probes, three independent samples, and 41 total
storms, the honest position is: structural claim solid throughout, directional
claim genuinely undetermined — not because no test was run, but because the
one test built to settle it came back below its own bar.

