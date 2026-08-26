## 2026-08-12 — E-THE-CONTROL-SCORED-THE-HEADLINE-1

**Status:** FINDING `[G]` — `comet_tail_f16.py` / `.json`, bars committed
BEFORE the run (`05f09005`); report §5.12. EXPLORATORY, not an EV.

**The arc's leading rescue was measured and it failed — and the anti-vacuity
control accidentally calibrated the instrument that had been judging it.**

CT-F16 swapped ONE variable: the dipole's motion reference, from 6h surface
displacement to the 500/600/700 hPa steering flow, on CT-F14's OWN 19 storms
(paired; stored centres reused, so selection and disk geometry cannot move).
Report §9.2 had named this *"the single most promising fix"*. Measured:
**sign consistency 0.579 against a 0.70 bar** (worse than surface's 0.684),
**residual sd 68.29° → 87.71°, 28.4 % WIDER** where a ≥10 % tightening was
predicted, and a level sweep improving **monotonically toward the SURFACE** —
best at 850 hPa, outside the 400–650 hPa band the height ladder predicted,
converging exactly on the surface-displacement figure. There is no
mid-tropospheric optimum on this sample.

**The control is the larger finding.** F16c scored two deliberately WRONG
references through the identical pipeline. The **90°-rotated** steering
reference returned **13/19 = 0.684, p=0.0835** — *numerically identical to
CT-F14's headline*, the number this arc has carried as "suggestive" since
§5.11 (a different set of 13 storms, so the count coincides, not the
identity). At n=19 the one-sided ladder is 11→p=0.324, 13→p=0.0835,
**14→p=0.0318**. So **CT-F14 was never one storm short of significance; it was
one storm short of distinguishability from an answer built to be wrong.**

**Rule: an anti-vacuity control does not only guard the test it is attached to
— it measures the RESOLVING POWER of the instrument.** This one was written to
protect CT-F16 and instead retro-calibrated CT-F14. Attach a
deliberately-wrong reference to any claim whose headline is a rate, and read
the control's score as the floor that headline must clear. Had F16c existed at
§5.11, "0.684, suggestive" would have been reported as "0.684, indistinguishable
from a rotated control".

**Second finding, independent of the first: the sign test conflates a
SYSTEMATIC ROTATION with a correct prediction.** Stratified by steering
strength — weak flow (<10 m/s, n=6) **sign 0.833 / median |err| 103°**; strong
flow (≥10 m/s, n=13) **sign 0.462 / median |err| 55°**; `corr(speed,|err|) =
−0.407`. The prediction gets **more accurate in magnitude** as steering
strengthens, exactly as the physics expects, while sign consistency moves the
**opposite** way. High sign consistency in the weak subset is errors clustered
near −103°: a systematic rotation, which a sign test reports as success. **A
one-sided sign test on a distribution not centred at zero measures which SIDE
the bias falls on, not whether the prediction holds** — and this arc has used
it as the primary instrument since §4. The apparatus story told since §5.9
(*slow storms have noisy bearings → filter on displacement*) is not what these
data show: the well-steered storms are the ones whose signs split at chance
while their magnitudes are best.

**What is NOT falsified.** The height ladder (§5.2/5.8) decomposed the FIELD at
each level about that level's own centre; CT-F16 keeps the SURFACE dipole and
swaps the FLOW reference. Different quantities — the ladder stands as a
measurement (n=2, unreplicated), and what died is the operational reading §9.2
built on it. The structural claim (§9.1: ring profile, wn-1 dominance, the
12-byte carrier) is untouched; nothing in CT-F16 touches it.

**Cross-ref:** `E-ZERO-FOR-ELEVEN-THE-AUTHOR-CANNOT-AUDIT-HIS-OWN-FALSIFIERS-1`
(the author is the wrong person to find a spec's vacuous pass routes — here a
control found a *live* one, in a number already published);
`E-THE-HEADLINE-NUMBER-MEASURED-A-MODEL-NOBODY-CLAIMED-1` (the other time this
arc's headline described something other than what was claimed).

