## 2026-08-12 — E-A-FIGURE-CITED-TWICE-IS-NOT-CONFIRMED-ONCE-1

**Status:** FINDING `[G]` — measured. Probe
`probes/weather-p1/substrate_comfort_d_cz_0_1.py` + `.json`, one run,
committed with the artifact this entry is about.

**The defect.** `substrate-comfort-zones-v1` §1's nine-row regime preflight
was marked **DONE** on `STATUS_BOARD.md`, quoted in the `PR_ARC_INVENTORY`
entry, quoted again in the `LATEST_STATE` row, and its headline (a 9.3×
dynamic range across four regimes) was carried into three PR bodies. **No
script and no JSON producing any of those figures had ever been committed.**

**The part that makes it worth an entry rather than a fix.** On #945 I ran
an explicit self-audit *specifically because* this arc keeps shipping
summary claims that do not survive checking — and reported the regime-ladder
figures as verified. What that audit actually did was confirm the arc entry
matched the PLAN. Both are prose. Neither is a measurement. **A figure cited
in two documents has been cited twice, not confirmed once** — and an audit
that compares two documents to each other measures their consistency, which
is not the property anyone cares about.

**The rule.** An audit must terminate at an ARTIFACT — a committed script, a
committed JSON, a re-run — or it is a spell-check. Concretely: when a
board row says DONE, the check is *"name the file that produced it"*, not
*"does the summary match the plan"*.

**What the reproduction then found**, none of which was visible from the
prose:

1. **Only 4 of 9 rows are reproducible at all.** R1/R2/R3 centres are in the
   plan and R4's come from committed `comet_tail_f14.json` (the qualifying
   filter — `status == "OK"` AND `displacement_km ≥ 250` — itself had to be
   *recovered* by reproducing the stored `n_qualifying = 19`, and is now
   asserted in code). The five EXCLUDED land candidates carry measured
   figures whose **box centres were never written down anywhere**. They are
   unreproducible as committed. No coordinates were invented to fake them.
2. **The `|∇p|` definition was never committed either — only values.** Four
   candidates were computed and the winner decided from the data: the
   recorded figures are **Pa per grid cell with NO cos(lat) metric**
   (max deviation 0.069; the next-best candidate is off by 0.398). It is a
   plain `np.gradient` over the raw lat/lon array.
3. **Consequence, bounded:** ignoring cos(lat) understates the ZONAL
   gradient by `1/cos(lat)`, so R3 at 60 N is ~40 % low. Metric-corrected
   the ladder reads 10.3 / 15.5 / 61.2 / 100.9 — the **ORDER survives** and
   the dynamic range **widens** 9.3× → ≈9.8×. The regime axis stands; the
   recorded magnitudes do not. Reported as a correction, not as a collapse.
4. **A defect in the reproduction itself, caught by its own guard.** The
   first pass measured all 19 storm boxes at the *preflight* timestep rather
   than at each storm's own `t0` — "the places where storms once were, at an
   unrelated hour". That inverted R3/R4 (220 vs 208) and would have read as
   the regime ladder failing C1. Fetching per storm (19 extra chunks)
   restores 37 / 56 / 220 / 363. Separately, a seam assertion fired on a
   real storm centred at 353.4 E: a plain longitude slice there yields a
   one-column box and a plausible, meaningless number. **Both were caught
   because the probe asserted rather than assumed.**

---

