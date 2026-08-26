## 2026-08-12 — E-THE-METRIC-THAT-SEPARATES-ONE-COMPARISON-IS-BLIND-TO-ANOTHER-1

**Status:** FINDING `[G]` — measured, same run.

**`ρ` is saturated on the diagonal and enormous off it.** Measured over four
regimes at 256 palette levels on MSLP:

| comparison | `ρ` separation |
|---|---|
| real arm vs real arm (`CAL-ABS` vs `CAL-RANK`, both own-calibration) | **3×10⁻⁶ … 4.7×10⁻⁵** |
| real vs degraded (`CAL-SHUFFLE` 0.003–0.159, `GEO-DEGENERATE` 0.29–0.48) | **~0.5 … ~1.0** |

Both real arms reconstruct the ordering essentially perfectly (ρ > 0.99996),
because 256 levels on a smooth pressure field is a very fine quantization.
So `ρ` has **four orders of magnitude of range for transfer loss `L`** (an
off-diagonal quantity) and **none at all for C4** (which compares two real
arms on their own diagonals). RMSE, which the same rewrite had just demoted,
*does* separate the real arms: ratios 3.96 / 1.85 / 1.14 / 1.49.

**The rule: a metric is not good or bad, it is good or bad FOR A GIVEN
CONTRAST.** Choosing one for a plan as a whole — as the cross-swap rewrite
did, elevating `ρ` and demoting RMSE in one move — is a category error the
moment the plan contains two contrasts with different dynamic ranges. The
resolution here is not to revert: `L` keeps `ρ`, C4 moves to RMSE in Pa with
`ρ` as a floor check (< 0.999 = broken, not merely lost).

**Why amending after data is legitimate here, and when it stops being.**
D-CZ-1's *stated purpose* is to test the apparatus before the expensive
cells, and **no C4 cell has been scored** — a metric found blind in
preflight is exactly what preflight is for. It becomes illegitimate the
instant one C4 measurement exists. Recorded as an amendment carrying its
trigger (`§6.4`), never edited into the bar silently.

**Cross-ref:** this is the same shape as #926's Fisher-z result (8.3× better
in the storm tail on the raw field, 4.7× worse on ring means — one encoding,
opposite verdicts by what it represents), which §0 already cites as a
founding motivation for this plan. The plan's own primary metric turned out
to be another instance of the phenomenon the plan was built to map.

