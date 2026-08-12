"""EXPLORATORY — the INSTRUMENT comparison: sign test vs circular resultant,
on the SAME 19 stored storms. Post-hoc re-analysis, explicitly NOT a verdict.

Operator (2026-08-12): "die irrationale Aufsummierung hilft, dass [der] Dipol
nicht auf 0.68 kollabiert." This probe measures that claim's core mechanism.

THE ANATOMY OF THE 0.684 COLLAPSE. CT-F14/F16 scored the dipole with a SIGN
test: each storm's signed angular error collapses to one bit (error < 0?), and
19 bits saturate — F16c measured that a 90-deg-ROTATED reference also scores
0.684, i.e. the statistic cannot distinguish the real referent from a wrong
one at this n. Two distinct losses happen at the binarization:

  (1) MAGNITUDE is discarded — a tight cluster at -103 deg and a loose cloud
      straddling zero can produce the same bit count (measured in F16's
      weak/strong stratification).
  (2) The MEAN DIRECTION is discarded — a systematic offset (the arc's -40
      deg) EATS the sign margin instead of being estimated.

THE FIX IS AN AUFSUMMIERUNG: keep the error VECTORS and sum them. The
circular resultant R_bar (mean resultant length) + mean direction mu is the
standard directional statistic (Rayleigh test). It preserves exactly what the
sign test destroys: concentration (R_bar) and offset (mu) come out as two
separate numbers. A rotated control then shows the SAME R_bar with mu shifted
by 90 deg — visibly wrong — where the sign test scored it identically.

Where the IRRATIONALITY enters (the operator's framing, demarcated honestly):
the vector summation itself is what prevents the collapse; the irrational
(golden/low-discrepancy) structure is what keeps the summation UNBIASED —
no resonance between sampling geometry and signal harmonics (within-storm,
load-bearing once the dipole is estimated from SPARSE spiral samples instead
of the dense grid), and natural phase diversity across storms (which is what
makes the CT-W6 multi-component fit identifiable at all).

SCOPE: same 19 storms whose sign test returned NO-VERDICT. This is an
instrument demonstration on stored data — the verdict-grade use of the
resultant is CT-W6 on the same rows with pre-registered bars, and any FRESH
claim needs a fresh sample (CT-F17). Nothing here promotes the directional
claim; it measures what the previous instrument could not see.
"""
import json
import pathlib

import numpy as np

N_BOOT = 20_000
RNG_SEED = 20260812          # fixed before running; bootstrap is deterministic


def wrap_deg(d):
    """Wrap degrees into [-180, 180) — the identical convention CT-F14/F16 use."""
    return (d + 180.0) % 360.0 - 180.0


def circular(errs_deg, n):
    """Resultant length R_bar, mean direction mu (deg), Rayleigh p.

    R_bar in [0,1] measures CONCENTRATION (1 = all vectors identical, 0 =
    uniform); mu is WHERE the cluster sits — the offset the sign test could
    only penalize. Rayleigh Z = n*R_bar^2 with the standard small-n corrected
    p-approximation (Zar/Mardia). Under uniformity E[R_bar] ~ sqrt(pi)/(2*sqrt(n)).
    """
    th = np.deg2rad(np.asarray(errs_deg))
    c, s = np.cos(th).mean(), np.sin(th).mean()
    r = float(np.hypot(c, s))
    mu = float(np.rad2deg(np.arctan2(s, c)))
    z = n * r * r
    p = float(np.exp(-z) * (1 + (2 * z - z * z) / (4 * n)
                            - (24 * z - 132 * z**2 + 76 * z**3 - 9 * z**4)
                            / (288 * n * n)))
    return r, mu, max(min(p, 1.0), 0.0)


def boot_mu_ci(errs_deg, n, rng):
    """Bootstrap 95% CI half-width (deg) for the mean direction mu."""
    th = np.deg2rad(np.asarray(errs_deg))
    idx = rng.integers(0, n, size=(N_BOOT, n))
    c = np.cos(th)[idx].mean(axis=1)
    s = np.sin(th)[idx].mean(axis=1)
    mus = np.rad2deg(np.arctan2(s, c))
    mu0 = np.rad2deg(np.arctan2(np.sin(th).mean(), np.cos(th).mean()))
    dev = wrap_deg(mus - mu0)
    lo, hi = np.percentile(dev, [2.5, 97.5])
    return float(max(abs(lo), abs(hi)))


rows = json.loads(pathlib.Path(__file__).with_name(
    "comet_tail_f16.json").read_text())["rows"]
n = len(rows)
assert n == 19, f"expected CT-F14's 19 qualifying storms, got {n}"
rng = np.random.default_rng(RNG_SEED)

e_surf = np.array([r["err_surface_deg"] for r in rows])
e_steer = np.array([r["err_steering_deg"] for r in rows])
lp = np.array([r["low_pole_rad"] for r in rows])
sb = np.array([r["steer_bearing_rad"] for r in rows])

# the two F16c controls, recomputed per-row with the identical err convention
e_rot = wrap_deg(np.rad2deg(lp - (sb + np.pi / 2 + np.pi / 2)))
perm = sb[(np.arange(n) + 7) % n]
e_perm = wrap_deg(np.rad2deg(lp - (perm + np.pi / 2)))

out = {"n": n, "source": "comet_tail_f16.json rows (paired, stored)",
       "scope": ("POST-HOC instrument comparison on the same storms whose "
                 "sign test returned NO-VERDICT. Not a verdict; CT-W6 is the "
                 "pre-registered use, CT-F17 the fresh-sample one."),
       "referents": {}}

print(f"{'referent':<22} {'sign<0':>7} {'R_bar':>7} {'mu':>9} "
      f"{'mu 95% CI':>10} {'Rayleigh p':>11}")
for name, e in (("surface (CT-F14)", e_surf),
                ("steering (CT-F16)", e_steer),
                ("CONTROL rot+90", e_rot),
                ("CONTROL permuted", e_perm)):
    r, mu, p = circular(e, n)
    ci = boot_mu_ci(e, n, rng)
    frac = float((np.asarray(e) < 0).mean())
    out["referents"][name] = {"sign_neg_frac": frac, "R_bar": r,
                              "mu_deg": mu, "mu_ci95_deg": ci,
                              "rayleigh_p": p}
    print(f"{name:<22} {frac:>7.3f} {r:>7.3f} {mu:>+8.1f}° "
          f"±{ci:>7.1f}° {p:>11.4f}")

su, ro = out["referents"]["surface (CT-F14)"], out["referents"]["CONTROL rot+90"]
sep = abs(wrap_deg(su["mu_deg"] - ro["mu_deg"]))
print(f"\nuniform-expectation R_bar at n={n}: "
      f"{np.sqrt(np.pi) / (2 * np.sqrt(n)):.3f}")
print(f"\nTHE DISCRIMINATION THE SIGN TEST LACKED:")
print(f"  sign test : surface {su['sign_neg_frac']:.3f} vs rotated control "
      f"{ro['sign_neg_frac']:.3f}  -> conflated at n={n}")
print(f"  resultant : mu separated by {sep:.1f}° "
      f"(CIs ±{su['mu_ci95_deg']:.0f}°/±{ro['mu_ci95_deg']:.0f}°) at "
      f"near-identical R_bar -> the wrong referent is VISIBLE")
out["discrimination"] = {"mu_separation_deg": sep}

with open(pathlib.Path(__file__).with_name(
        "comet_tail_resultant_instrument.json"), "w") as fh:
    json.dump(out, fh, indent=2)
print("\nwrote comet_tail_resultant_instrument.json")
