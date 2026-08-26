## 2026-08-12 — E-THE-DISPLACEMENT-FILTER-ATE-THE-STRANDED-STRATUM-1

**Status:** FINDING `[G]` — W6 RUN (`comet_tail_w6.py`/`.json`), audited
(raw predictors committed per storm after a self-caught gap), both a
sample-composition finding and a genuine model-level negative result.

**The mechanism-level result, stated first because it is the load-bearing
one.** Report §10.2's dipole vector-sum model (`D = c_geo·P_geo +
c_bow·P_bow`, a background-high neighbor predictor plus a relative-motion
bow-wave predictor) was fit on CT-F14's 19 stored storms and **VOIDED by its
own pre-registered anti-vacuity control**: single-geo R²=−0.104 (worse than
predicting the mean); the permuted-P_bow control R²=−0.071 and the
rotated-90° control R²=−0.062 BOTH exceed `single-geo + 0.03 = −0.074` — two
deliberately wrong references score as well as or better than the real
predictor. This is not a marginal miss — it is the anti-vacuity control
doing exactly its job: rejecting a fit that has nothing to identify.

**Checked before concluding it was a clean negative, per the standing
measurement-skeptic discipline: no implementation bug found in the FIT
itself, but a sign convention AND a units error were found in the
NARRATIVE around it (codex + CodeRabbit P2/Major on PR #940, both real,
fixed before merge).** One storm was independently re-fetched and hand-
audited; extended to the full sample by committing the raw
`D`/`P_geo`/`P_bow`/`A_H`/`d_H`/`v_rel_ms` per storm. Every value is
physically sane (`A_H > 0` always, `d_H` inside the 600–2500 km annulus
always, `v_rel` 2.8–27.9 m/s).

**Sign:** `spine()`'s raw fit coefficient points toward the storm's HIGH
side (the gradient of increasing residual pressure), while `P_geo`/`P_bow`
both point toward the LOW side by construction — the exact convention
`low_pole_bearing()` makes explicit via its own `(ph + π) % (2π)` flip. `D`
is correctly `−spine(...)`; the first draft used the unflipped `coef`.
Corrected: **`c_geo = +0.407` (the physically predicted positive sign —
CORRECT)**, `c_bow = −0.0006` km⁻¹ (predicted positive — **wrong sign, but
small**). Verified algebraically and numerically that this flip changes
NOTHING about R² or the B0/B1 VOID verdicts (OLS is odd-symmetric in the
fit target) — only the coefficient signs and the sentence describing them.

**Units:** `D`/`P_geo` are [Pa/km]; `P_bow` is [Pa] — `c_geo` is
dimensionless, `c_bow` carries km⁻¹, and OLS coefficients rescale inversely
under column rescaling while R²/fitted-values stay fixed. **Raw `|c_bow|`
was never valid evidence of "no measurable weight"**, and comparing
`|P_bow|` to `|D|` directly (147×, as first reported) compounded the same
mistake — Pa is not comparable to Pa/km at all. The dimensionally valid
measure is the fitted CONTRIBUTION `|c_bow·P_bow|` against `|D|`, both in
Pa/km: mean `|D|`=0.745, mean `|c_geo·P_geo|`=0.186 (25 % of `|D|`), mean
`|c_bow·P_bow|`=0.068 (9 % of `|D|`) — the geo contribution is ~2.7× the
bow contribution, MODEST rather than "no weight," and both remain
consistent with the R²<0 finding that neither predictor meaningfully
explains `D`'s variance.

**The finding that generalizes past this one probe: an anti-vacuity control
can be voided by SAMPLE COMPOSITION, and the reason is arithmetic, not
physics.** B3's stranded-vs-moving stratification (`|v_storm| < 8 m/s`,
the report's own named test of the "stranded-rescue" reading) came back
**n=0 for the stranded stratum** — every one of the 19 storms has
`|v_storm| ≥ 12.54 m/s`. This is not a null result about storm motion; it is
a DIRECT ARITHMETIC CONSEQUENCE of CT-F14's own qualifying filter
(`displacement_km ≥ 250` over the 6 h window): `250 km / 6 h = 11.574 m/s`,
a hard floor on `|v_storm|` for ANY storm admitted to the sample. **A filter
built to select clearly-moving storms for a displacement-scoring test
silently and permanently excludes the storms a LATER, differently-motivated
test (stranded-rescue) needs to see.** The stranded-rescue claim is
therefore **UNTESTABLE on this sample, not refuted** — the untestability was
knowable from the filter's own arithmetic before a single storm was fetched,
and wasn't checked until B3 came back empty.

**Consequence, stated as a reusable rule:** before scoring ANY new
hypothesis against an EXISTING filtered sample, check whether the sample's
own selection criterion is compatible with the new hypothesis's own
discriminating variable — arithmetically, not by running the probe and
discovering an empty stratum after the fact. A filter selected for one
purpose (fast, clearly-displaced storms, easy to center-find and score
against displacement) is not neutral with respect to every future question;
it is a specific cut through the underlying population, and every later
probe inherits that cut whether or not it is the cut that probe needs.

**Consequence for the report and for CT-F17.** The report's §10.2 vector-sum
model, AS SPECIFIED, is disconfirmed on this sample — not "unproven," not
"needs more data" in the ordinary sense, but VOID by its own control. CT-F17
(the fresh-sample verdict, gated on W6's result + an independent adversarial
audit) is now moot **for this form of the model** — a fresh-sample test of a
model that already fails identifiability on the stored sample is not the
next useful step. Any REVISED form of the vector-sum model (multiple
neighbors, a nonlinear bow term, per-storm coefficients) would need its own
W6-shaped mechanistic test before earning a CT-F17 slot; a genuine
stranded-rescue test needs a sample built without (or explicitly retaining
slow storms despite) a displacement floor.

