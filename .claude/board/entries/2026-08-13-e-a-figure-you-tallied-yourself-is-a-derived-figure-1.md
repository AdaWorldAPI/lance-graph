## 2026-08-13 — E-A-FIGURE-YOU-TALLIED-YOURSELF-IS-A-DERIVED-FIGURE-1

**Status:** FINDING `[G]` — corrects two figures in
`E-THE-REAL-GATE-RAN-AND-QUALIFIED-NOT-RETRACTED-THE-CLAIM-1` (below) and in
PR #950's plan/board/PR-body text. **Confidence:** High. Append-only
correction; the wrong entry is left standing beneath this one, per the
storno rule.

**The wrong figures.** `D-WXS-8`'s bar-B7 tallies were written as
*control 16/16 PASS, primary 10/16 PASS (6 FAIL)*. Counted from the committed
`probes/weather-p1/fixture/fidelity_probe_results.json`:

| | written | **actual** |
|---|---|---|
| B7 control (per-variable floor must LOSE) | 16/16 | **19/19** |
| B7 primary (ρ_shared ≥ 0.9996) | 10/16 (63 %) | **9/19 (47 %)** |
| per season (primary) | — | winter **2/9**, spring **4/5**, summer **3/5** |

There are **19** cross-unit pairs, not 16 (winter 9 + spring 5 + summer 5);
the same-unit pairs (K×K, m/s×m/s) are informational and correctly excluded
from the bar, and I dropped them from the denominator inconsistently.

**Why the direction matters, not just the arithmetic.** 10/16 reads as
*"mostly passing, six close misses."* 9/19 reads as *"the strict bar fails on
a **majority** of cross-unit pairs."* Both describe the same measurements —
the per-pair ρ values were all carried correctly — but only the second is
what the data says. The KILL-gated control is unaffected and in fact better
(19/19), so **no conclusion reverses**; what changes is that the primary bar's
shortfall was under-reported.

**The precise diagnosis, and it is narrower than "check your figures".**
Every number the program *computed and printed* — 56 verdicts, 42 pass /
14 fail, every individual ρ — was carried into the writeup correctly. The
**only** two wrong numbers are the ones I produced by **counting rows in
terminal output by eye**. The pipeline was sound; the tally was not.

> **A figure you tallied yourself is a DERIVED figure, and a derived figure
> needs the artifact exactly as much as a measured one does.** "The audit must
> terminate at an artifact" is usually applied to *citing* a number. This is
> its narrower and easier-to-miss form: **producing** a number by counting,
> summing, or eyeballing a listing is itself an unverified derivation, even
> when every input to it is verified.

**Mechanical fix, now in use:** any count, rate, or ratio that appears in a
writeup is computed from the committed artifact in the same command that
prints it — never transcribed from a previous run's stdout, and never
tallied by reading a list.

**Reach.** The error reached `main` through PR #950 in four places: the plan's
W3 RUN section, the `STATUS_BOARD` `D-WXS-8` row, the epiphany below, and the
PR body. The first three are corrected (in-place for the plan and board, which
are not append-only ledgers; by this prepended entry for `EPIPHANIES`). **The
merged PR body cannot be corrected** — treat #950's body as carrying the
superseded figures and this entry as authoritative.

**Cross-ref:** `E-THE-REAL-GATE-RAN-AND-QUALIFIED-NOT-RETRACTED-THE-CLAIM-1`
(the entry corrected), `E-A-FIGURE-CITED-TWICE-IS-NOT-CONFIRMED-ONCE-1` (the
citing form of the same rule), `E-A-DISABLE-PROBE-CAN-ITSELF-BE-VACUOUS-1`
(same session, the verification layer), `.claude/plans/weather-soa-bake-v1.md`
W3 RUN (the corrected block, carrying the same ⊘ note).

---

