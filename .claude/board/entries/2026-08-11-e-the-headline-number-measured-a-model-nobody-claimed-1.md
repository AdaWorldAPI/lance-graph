## 2026-08-11 — E-THE-HEADLINE-NUMBER-MEASURED-A-MODEL-NOBODY-CLAIMED-1

**Status:** FINDING `[G]` — external review of PR #926 (14 CodeRabbit + 2 Codex
findings); re-measured and corrected in `comet_tail_probe.py` and
`COMET_TAIL_REPORT.md` §1 / §8b (commit 5302828e).

**The arc's most-repeated number was measured on a model the arc never
claimed.** The compression headline — "center + ~12 ring means + ONE dipole
(2 values) = 93–97 % of in-disk variance" — came from a `decompose()` that
fits `a1[b]`, `b1[b]` **per ring**: 12 rings × 2 = **24** free dipole
parameters, so the published 0.972/0.926 belongs to a **36-parameter** model,
not the **14-value** one the storage claim describes. Measured properly, the
constrained 2-parameter dipole (one amplitude slope + one bearing — the
linear-background form the report's own §2 derives) gives **0.943 / 0.909**.
Corrected headline: **90.9–94.3 %, not 93–97 %.**

The finding SURVIVES — 14 values still lift a storm from 29–63 % to 91–94 % —
but claim and measurement had drifted apart by ~2.5× in parameter count across
six probes and several report rewrites, and nobody in-session noticed. **The
tell was available the whole time:** the report described the representation
in one place ("2 values") and the code produced another ("per-ring"), and no
test tied the two together.

**Two review findings IMPROVED results rather than damaging them**, which is
the argument for external review as more than ceremony: (a) sunflower E2 was
handing the grid arm up to 25 % more samples than the spiral (every in-disk
lattice point instead of exactly n) — with equal budgets enforced the **spiral
now wins at every N**, where the arc had recorded "parity"; the original result
was PESSIMISTIC. (b) The voxel-chess palette arm compared palette-derived
geostrophic winds against RAW observations, because `geo_corr` closed over
module-level `u`/`v` — a hybrid, not the pre-registered palette result.

**A fourth vacuous assertion, in this arc's own documented house style:** E6's
`rises_then_decays` required only an interior maximum plus a lower final
value, so it accepted a profile that DECREASED before rising to the peak — and
the committed run did exactly that (12.190 → 12.163 m/s before the 525 km
peak) while reporting `true`. Found by a reviewer, not by the author, which is
the same asymmetry `E-ZERO-FOR-ELEVEN-...` already recorded.

**Rule:** *a number that appears in a headline must be produced by code whose
parameter count matches the headline's own description of the object.* Where
prose says "N values", the probe should EMIT N and the report should print it
— which `comet_tail_probe.py` now does
(`n_params_profile_wn1_constrained`), so the two cannot silently drift again.

