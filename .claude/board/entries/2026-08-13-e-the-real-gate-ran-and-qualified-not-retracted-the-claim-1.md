## 2026-08-13 — E-THE-REAL-GATE-RAN-AND-QUALIFIED-NOT-RETRACTED-THE-CLAIM-1

**Status:** FINDING `[G]` — measured, real live-fetched grid data, 3 seasons,
`jc::reliability::spearman`, disable-sanity-checked. **Confidence:** High.

**What ran.** `D-WXS-7`/`D-WXS-8` — the gate this arc had been treating as
blocked behind the classid mint for four consecutive statements
(`E-...` — see the same-day correction entry above this one) turned out to
need none of it. Once corrected, it ran the same day: real ARCO-ERA5 fetch
(public HTTPS, 404=NaN=valid-missing-chunk semantics), 3 real calendar
seasons found by a live availability sweep (not assumed — a first attempt at
4 fixed 2021 anchors found only 3 variables present at ALL of them),
200,000-pair sampling at the exact `floor.rs` quantisation, scored with
`jc::reliability::spearman` (added to `weather-poc` as a **dev-dependency
only** — the default `[lib]` build stays zero-dep).

**Two results, and they cut in different directions on the SAME underlying
claim — recorded together because reading either alone would mislead.**

1. **`D-WXS-7` (bar B6): the K×K pair does NOT replicate an earlier
   near-miss.** The plan's own §0.3 cites a smaller-scale measurement —
   K×K = 0.999556, below the 0.9996 bar — as the reason the bar "can fire" at
   all. At real grid scale, 3 real seasons, computed with `jc` not `scipy`:
   0.999909 / 0.999895 / 0.999684. Clears the bar with margin, every time.
   The earlier number is not wrong or retracted — it was a real measurement
   on a real, smaller fixture — but it no longer licenses "the substrate's
   fidelity is marginal" as a grid-scale conclusion.

2. **`D-WXS-8` (bar B7): the core claim holds; two literal thresholds do
   not.** The KILL-gated control — per-variable floors must LOSE on every
   cross-unit pair — passed **16/16**, by a wide margin (0.245–0.939 vs
   0.9987–0.9999). The KILL did not fire; the shared-floor design is not
   refuted. But the PRIMARY bar (ρ≥0.9996 exactly) missed on 6/16 pairs
   (close misses, 0.9987–0.9996, skewed toward wind/pressure), and the
   stay-silent twin's **"zero empty buckets"** half — carried verbatim from
   a 1-timestep/3-variable fixture — failed at **all three** real seasons
   (38–45 of 256 empty). Read plainly: a percentile-trimmed window pooled
   across more variables at real grid scale necessarily leaves some slack
   for any one variable's narrower distribution. Unsurprising in direction;
   simply never re-verified at scale before this run.

**The reusable shape: a KILL clause and a PRIMARY/twin bar are not the same
severity, and a report that collapses them loses the finding.** This run had
three tiers of outcome on ONE deliverable — a KILL that did not fire (nothing
retracted), a strict numeric bar that partly missed (a real gap, reported),
and an EARLIER measurement that did not replicate at the new scale (a
correction, not a failure). Flattening any two of those into one sentence —
"D-WXS-8 failed" or "D-WXS-8 passed" — would have been wrong either way. The
discipline that produced a correct report here: apply every pre-registered
verdict literally, print every one including the FAILs, and only THEN decide
what the KILL clause specifically licenses.

**Sanity check before trusting any ρ:** the shuffled-decode arm was verified
to be a genuinely different array from the unshuffled one (range `[0, 4.92]`
vs `[0, 202]`, not a copy or a stub) before its collapsed ρ (0.02–0.024) was
read as a real result rather than a pipeline artifact.

**Cross-ref:** the same-day correction above (`D-WXS-7` was never blocked);
`.claude/plans/weather-soa-bake-v1.md` sec 4 W3 (bars B6/B7, verbatim);
`probes/weather-p1/p2_probe.py` (the fixture-scale predecessor); `E-A-DISABLE-
PROBE-CAN-ITSELF-BE-VACUOUS-1` (the same session's sibling method finding).

---

