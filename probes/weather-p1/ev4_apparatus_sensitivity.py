"""EV-4 P0 FOLLOW-UP — is the Fisher-Z non-monotone curve a DATA property or an
APPARATUS artifact? (PP-13 P0, wave wf_daae6e63-d39.)

`ev4_window_sweep.py` reported `check2_fisher_z_monotone = False` on all three
variables and the orchestrator elevated that to a headline finding. PP-13
called it a numerical artifact of `np.percentile`'s interpolation against the
`eps` arctanh clip pole. This script settles it by perturbing ONLY the
apparatus knobs and leaving the data untouched: if the curve's SHAPE moves
with `eps` or with the percentile interpolation method, the shape is not a
property of the field.

Verdict is printed, not assumed. Emits ev4_apparatus_sensitivity.json.
"""
import json
import pathlib

import numpy as np

WINDOWS = [(0.4, 99.6), (0.2, 99.8), (0.1, 99.9), (0.02, 99.98)]
EPS_GRID = [1e-6, 1e-9, 1e-12, 1e-15]
METHODS = ["linear", "lower", "higher", "nearest", "midpoint"]
VAR = "2m_temperature"


def fisher_z(s, eps):
    """arctanh, clipped off the +/-1 poles by eps (the shipped formula)."""
    s = np.clip(s, -1 + eps, 1 - eps)
    return 0.5 * (np.log1p(s) - np.log1p(-s))


def ci_med(anom, w, eps, method):
    """Interior-median bucket CI for the Fisher-Z arm at one window."""
    lo_q, hi_q = np.percentile(anom, [w[0], w[1]], method=method)
    scale = max(abs(lo_q), abs(hi_q))
    z = fisher_z(anom / scale, eps)
    zlo, zhi = np.percentile(z, [w[0], w[1]], method=method)
    edges = np.tanh(zlo + np.arange(257) / 256 * (zhi - zlo)) * scale
    ci = np.diff(edges) / 2.0
    return float(np.median(ci[1:255])), float(zlo), float(zhi), float(scale)


def shape(vals):
    """Classify a sequence as increasing / decreasing / non-monotone — the shape verdict EV-4 reports."""
    if all(x < y for x, y in zip(vals, vals[1:])):
        return "increasing"
    if all(x > y for x, y in zip(vals, vals[1:])):
        return "decreasing"
    return "non-monotone"


def main():
    """Run the eps sweep and the method-spread comparison, printing the apparatus verdict."""
    a = np.load(f"fixture/{VAR}.npy").astype(np.float64)
    anom = a - a.mean(axis=1, keepdims=True)
    out = {"variable": VAR, "windows": [list(w) for w in WINDOWS]}

    out["eps_sweep"] = {}
    shapes_eps = []
    for eps in EPS_GRID:
        vals = [ci_med(anom, w, eps, "linear")[0] for w in WINDOWS]
        s = shape(vals)
        shapes_eps.append(s)
        out["eps_sweep"][repr(eps)] = {"ci_interior_med": vals, "shape": s}
        print(f"  eps={eps:<8} {[f'{v:.6f}' for v in vals]}  {s}")

    out["method_sweep"] = {}
    shapes_m = []
    for m in METHODS:
        vals = [ci_med(anom, w, 1e-9, m)[0] for w in WINDOWS]
        s = shape(vals)
        shapes_m.append(s)
        out["method_sweep"][m] = {"ci_interior_med": vals, "shape": s}
        print(f"  method={m:<9} {[f'{v:.6f}' for v in vals]}  {s}")

    # The one structural fact that is NOT apparatus: which tail sets `scale`.
    out["scale_controlling_tail"] = []
    for w in WINDOWS:
        lo_q, hi_q = np.percentile(anom, [w[0], w[1]])
        out["scale_controlling_tail"].append(
            {"window": list(w), "lo_q": float(lo_q), "hi_q": float(hi_q),
             "scale": float(max(abs(lo_q), abs(hi_q))),
             "set_by": "HI" if abs(hi_q) >= abs(lo_q) else "LO"}
        )
    flips = len({r["set_by"] for r in out["scale_controlling_tail"]}) > 1

    # A shape that survives neither knob is not a property of the field.
    eps_unstable = len(set(shapes_eps)) > 1
    method_spread = max(
        max(v["ci_interior_med"]) for v in out["method_sweep"].values()
    ) / max(
        1e-30, min(min(v["ci_interior_med"]) for v in out["method_sweep"].values())
    )
    out["verdict"] = {
        "shape_changes_with_eps": eps_unstable,
        "shapes_across_eps": shapes_eps,
        "method_max_over_min_ratio": method_spread,
        "scale_controlling_tail_flips": flips,
        "conclusion": (
            "APPARATUS-DOMINATED: the Fisher-Z interior-median CI shape is not "
            "stable under eps or percentile-interpolation choice, so the "
            "non-monotone curve must NOT be published as a data finding or fed "
            "to D-2. The scale-controlling-tail flip IS real and independent."
            if eps_unstable or method_spread > 5
            else "STABLE: the shape survives both apparatus knobs."
        ),
    }
    print(f"\n  shape across eps: {shapes_eps}")
    print(f"  method max/min spread: {method_spread:.1f}x")
    print(f"  scale-controlling tail flips across windows: {flips}")
    print(f"\n  {out['verdict']['conclusion']}")
    with open(pathlib.Path(__file__).with_name("ev4_apparatus_sensitivity.json"), "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
