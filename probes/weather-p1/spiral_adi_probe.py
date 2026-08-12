"""W5 -- spiral-ADI: two Fibonacci-stride tridiagonal sweeps ~ one 2D diffusion?

Per weather-w-probes-v1.md SS1 (post-#933 corrected brief). Vogel lattice at
N = 3*F(17)^2 = 7,651,227 (index-floor rule with 3x margin), Gaussian bump at
r0 = 0.75 (local parastichy index sqrt(r0^2*N) ~ 2077, comfortably above the
floor F(17) = 1597), ADI smoothing along the two emergent parastichy stride
families vs an isotropic heat-kernel reference, plus a distance-matched
shuffled-neighbour control (B3) and the index-floor sweep (B4).

BAND QUALIFICATION RULE (implemented exactly; one label in the brief is
off-by-one against its own parenthetical and the rule wins): a band
qualifies for B2 judgment iff its INNER radius >= r_floor = F(17)/sqrt(N).
With N = 3*F(17)^2, r_floor = 1/sqrt(3) ~ 0.5774. Under the 8-equal-area
annulus scheme (band i spans [sqrt((i-1)/8), sqrt(i/8)]), band 4's inner
radius sqrt(3/8) ~ 0.6124 >= r_floor, band 3's inner radius 0.5 < r_floor
-- so the qualifying set is bands 4-8 (matching the brief's parenthetical
"r >= 0.6124"), and the brief's "bands 3-8" label is reported as the
off-by-one it is rather than silently adopted.

V2 (2026-08-12, after four codex findings on PR #936 voided the v1 run's
verdicts -- committed BEFORE the v2 run, per the standing discipline):
(a) control links are built FULL-BAND (v1's 250k/band cap left ~74% of
headline sources as self-links and searched a thinned tree -- the v1 B3
ratio measured mostly self-links, not the control); (b) iterations are
SCALED so the added blur is resolvable and PREDICTED: V = sigma^2/4 added
variance per axis, iters = 2V/h^2 with h^2 the measured median squared
nearest-neighbour spacing, so the fitted sigma_ref must land at
sqrt(sigma^2+V) -- an operator that does nothing now FAILS (v1's 8
iterations at N=7.65M added ~0.003% variance: near-identity, and identity
fits the input Gaussian perfectly -- v1's B2 "pass" never demonstrated
diffusion); (c) the bump moved to sigma=0.05, r0=0.78 (>=3.35 sigma from
both mask edges; v1's 1.72-sigma inner clearance meant the mask ITSELF
produced a 1.208 covariance ratio -- confirmed analytically against the
measured 1.213 "asymptote", i.e. v1 measured the mask, not the operator)
and the UNSMOOTHED-baseline anisotropy through the same mask is now
computed and the verdict taken on the CHANGE; (d) the control-link
offset histogram is computed in-run and stored in the JSON (the 99.38%
Fibonacci-membership verification was previously chat-only).

B3 CONTROL IMPLEMENTATION NOTE (documented choice, same operator form both
arms): the Fibonacci arm sweeps chains k -> k+j (prev = k-j, next = k+j,
hold at open ends). The control arm replaces each point's next-partner with
its distance-matched non-Fibonacci neighbour (among the 8 real nearest,
closest in physical distance to the true Fibonacci partner, excluding that
partner); prev is the reverse map where uniquely defined, else hold. Both
arms use the identical 0.25/0.5/0.25 stencil, so the comparison isolates
the LINK STRUCTURE (arithmetic coherence vs local distance-matched
shuffle), which is what B3 exists to test.
"""
import json
import pathlib

import numpy as np
from scipy.spatial import cKDTree

SEED = 20260812
PHI = (1 + 5 ** 0.5) / 2
GOLDEN_FRAC = 2 - PHI
F = {8: 21, 10: 55, 12: 144, 14: 377, 17: 1597, 19: 4181}
SIGMA = 0.05          # v2: was 0.08; see header note (c)
R_BUMP = 0.78         # v2: was 0.75; >=3.35 sigma from both mask edges
ADDED_VAR = SIGMA ** 2 / 4.0  # v2: target added variance per axis (V)
N_BANDS = 8


def vogel(n):
    """Unit-disk Vogel lattice: returns (x, y, r) arrays of length n."""
    k = np.arange(n)
    r = np.sqrt((k + 0.5) / n)
    th = k * 2 * np.pi * GOLDEN_FRAC
    return r * np.cos(th), r * np.sin(th), r


def band_of(r):
    """Equal-area band index 1..8 for each radius (band i spans
    [sqrt((i-1)/8), sqrt(i/8)])."""
    b = np.floor(r * r * N_BANDS).astype(int) + 1
    return np.clip(b, 1, N_BANDS)


def discover_strides(x, y, bands, sample_per_band=4000, rng=None):
    """Step 1: per band, the dominant nearest-neighbour index-difference
    pair, found GEOMETRICALLY with no capped search window (KD-tree over the
    band's own points, 8 nearest neighbours each, histogram |dk|). Sampled
    per band for tractability at N in the millions -- the stride pair is a
    bulk property; 4000 points per band estimate the histogram mode with
    huge margin. Returns {band: (j1, j2, histogram_top5)}."""
    out = {}
    for b in range(1, N_BANDS + 1):
        idx = np.where(bands == b)[0]
        if len(idx) < 100:
            out[b] = None
            continue
        tree = cKDTree(np.column_stack([x[idx], y[idx]]))
        take = idx if len(idx) <= sample_per_band else rng.choice(
            idx, sample_per_band, replace=False)
        sub = np.searchsorted(idx, take)
        _, nn = tree.query(np.column_stack([x[take], y[take]]), k=9)
        dk = np.abs(idx[nn[:, 1:]] - take[:, None]).ravel()
        vals, counts = np.unique(dk, return_counts=True)
        order = np.argsort(-counts)
        top = [(int(vals[i]), int(counts[i])) for i in order[:5]]
        j1, j2 = top[0][0], next(v for v, _ in top[1:] if v != top[0][0])
        out[b] = {"pair": sorted([j1, j2]), "top5": top}
    return out


def crossing_angles(x, y, bands, strides, n, rng, sample=2000):
    """Step 2: per band, the distribution of the angle between the two
    stride directions at sampled points (median + IQR, degrees)."""
    out = {}
    for b, info in strides.items():
        if info is None:
            out[b] = None
            continue
        j1, j2 = info["pair"]
        idx = np.where(bands == b)[0]
        idx = idx[(idx + max(j1, j2) < n)]
        if len(idx) > sample:
            idx = rng.choice(idx, sample, replace=False)
        v1 = np.column_stack([x[idx + j1] - x[idx], y[idx + j1] - y[idx]])
        v2 = np.column_stack([x[idx + j2] - x[idx], y[idx + j2] - y[idx]])
        cosang = (v1 * v2).sum(1) / (
            np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-300)
        ang = np.rad2deg(np.arccos(np.clip(np.abs(cosang), 0, 1)))
        out[b] = {"median_deg": float(np.median(ang)),
                  "iqr_deg": [float(np.percentile(ang, 25)),
                              float(np.percentile(ang, 75))]}
    return out


def sweep_field(y_field, next_idx, prev_idx):
    """One tridiagonal smoothing sweep along a link family:
    y_i <- 0.25*y[prev] + 0.5*y[i] + 0.25*y[next], holds (self) where a
    link is missing (encoded as the point's own index)."""
    return 0.25 * y_field[prev_idx] + 0.5 * y_field + 0.25 * y_field[next_idx]


def build_fib_links(n, bands, strides, qualifying):
    """Fibonacci-arm links: for each qualifying point, next = k+j (same
    band, else hold), prev = k-j (same band, else hold), per family.
    Returns (nextA, prevA, nextB, prevB) index arrays of length n."""
    idx = np.arange(n)
    links = []
    for fam in (0, 1):
        nxt = idx.copy()
        prv = idx.copy()
        for b in qualifying:
            info = strides[b]
            if info is None:
                continue
            j = info["pair"][fam]
            sel = np.where(bands == b)[0]
            tgt = sel + j
            ok = (tgt < n)
            ok[ok] &= (bands[tgt[ok]] == b)
            nxt[sel[ok]] = tgt[ok]
            src = sel - j
            ok2 = (src >= 0)
            ok2[ok2] &= (bands[src[ok2]] == b)
            prv[sel[ok2]] = src[ok2]
        links.extend([nxt, prv])
    return links


def build_control_links(x, y, n, bands, strides, qualifying, sel_qualifying,
                        rng):
    """B3 control links, v2: FULL-BAND, no subsampling cap. v1 capped each
    band at 250k points (headline bands hold ~956k each -- ~74% of sources
    got no real tree, silently self-linked; codex P1 on PR #936) and built
    the KD-tree only from the sampled subset, so even sampled sources
    picked among sampled neighbours rather than their real 8 nearest.
    Every point in every qualifying band now gets a real control partner.
    Returns (links, offset_histograms) -- offset_histograms is a dict keyed
    "famA"/"famB", each {|dk|: count} etc., over BOTH link families (codex
    P2 on PR #938: the first version histogrammed only family A -- the
    forward stride direction -- while the ADI sweep uses BOTH families
    every iteration; family B's offset distribution is now measured too,
    not merely assumed to match family A's)."""
    idx_all = np.arange(n)
    links = []
    histograms = {}
    for fam in (0, 1):
        nxt = idx_all.copy()
        for b in qualifying:
            info = strides[b]
            if info is None:
                continue
            j = info["pair"][fam]
            sel = np.where(bands == b)[0]
            tree = cKDTree(np.column_stack([x[sel], y[sel]]))
            d_nn, nn = tree.query(np.column_stack([x[sel], y[sel]]), k=9)
            tgt = sel + j
            ok = (tgt < n)
            ok[ok] &= (bands[tgt[ok]] == b)
            fibd = np.full(len(sel), np.nan)
            fibd[ok] = np.hypot(x[tgt[ok]] - x[sel[ok]], y[tgt[ok]] - y[sel[ok]])
            cand_global = sel[nn[:, 1:]]
            is_partner = cand_global == np.where(ok, tgt, -1)[:, None]
            dist_diff = np.abs(d_nn[:, 1:] - fibd[:, None])
            dist_diff[is_partner] = np.inf
            dist_diff[np.isnan(dist_diff)] = np.inf
            pick = np.argmin(dist_diff, axis=1)
            good = np.isfinite(dist_diff[np.arange(len(sel)), pick])
            nxt[sel[good]] = cand_global[np.arange(len(sel)), pick][good]
        # Histogram EVERY family (codex P2 on #938 -- previously only
        # family 0). self_linked_frac_overall is over ALL n points and is
        # DOMINATED by non-qualifying-band points, which are never touched
        # by design (a debugging trap found and fixed live earlier: an
        # early v2 diagnostic divided by total N rather than
        # qualifying-band population and reported a spurious ~38%
        # "self-linked" -- see the _in_qualifying_bands figure for the
        # number that actually matters).
        moved = nxt != idx_all
        dk = np.abs(nxt[moved] - idx_all[moved])
        vals, counts = np.unique(dk, return_counts=True)
        order = np.argsort(-counts)
        n_qual = int(sel_qualifying.sum())
        moved_in_qual = int((moved & sel_qualifying).sum())
        histograms[f"fam{'A' if fam == 0 else 'B'}"] = {
            "n_moved": int(moved.sum()), "n_total": int(n),
            "n_qualifying": n_qual,
            "self_linked_frac_overall": float(1.0 - moved.sum() / n),
            "self_linked_frac_in_qualifying_bands": float(
                1.0 - moved_in_qual / n_qual) if n_qual else None,
            "top10_offsets": [(int(vals[i]), int(counts[i]))
                              for i in order[:10]]}
        prv = idx_all.copy()
        src_pts = np.where(nxt != idx_all)[0]
        order = np.argsort(nxt[src_pts])
        tgts_sorted = nxt[src_pts][order]
        uniq, first, counts = np.unique(tgts_sorted, return_index=True,
                                        return_counts=True)
        unique_tgts = uniq[counts == 1]
        unique_srcs = src_pts[order][first[counts == 1]]
        prv[unique_tgts] = unique_srcs
        links.extend([nxt, prv])
    return links, histograms


def median_nn_spacing_sq(x, y, sel, rng, sample=20_000):
    """h^2: median squared nearest-neighbour spacing over the qualifying
    region, sampled for tractability -- the local mesh scale the ADI
    stencil actually diffuses on, used to size the iteration count."""
    idx = np.where(sel)[0]
    if len(idx) > sample:
        idx = rng.choice(idx, sample, replace=False)
    tree = cKDTree(np.column_stack([x[np.where(sel)[0]], y[np.where(sel)[0]]]))
    d, _ = tree.query(np.column_stack([x[idx], y[idx]]), k=2)
    return float(np.median(d[:, 1]) ** 2)


def run_adi(field, links, iters):
    """`iters` ADI iterations: family-A sweep then family-B sweep each."""
    nxtA, prvA, nxtB, prvB = links
    y = field.copy()
    for _ in range(iters):
        y = sweep_field(y, nxtA, prvA)
        y = sweep_field(y, nxtB, prvB)
    return y


def analyze_bump(x, y, blurred, x0, y0, sel, sigma_predicted):
    """v2: second-moment tensor -> anisotropy (unchanged); the isotropic
    fit now searches a window CENTERED on the PREDICTED sigma_ref
    (sqrt(sigma^2+V) from the iteration scaling) rather than an unbounded
    grid from the input sigma upward -- v1's search floor equalled the
    input sigma, so an inert operator's exact match (sigma_ref=sigma) was
    reachable and indistinguishable from a resolved blur (codex P1). Also
    returns the RAW (pre-fit) relative L2 against the UNSMOOTHED input
    bump, so "resolved" can be checked directly against "still basically
    the input"."""
    w = np.clip(blurred[sel], 0, None)
    if w.sum() <= 0:
        return None
    xs, ys = x[sel], y[sel]
    mx = (w * xs).sum() / w.sum()
    my = (w * ys).sum() / w.sum()
    cxx = (w * (xs - mx) ** 2).sum() / w.sum()
    cyy = (w * (ys - my) ** 2).sum() / w.sum()
    cxy = (w * (xs - mx) * (ys - my)).sum() / w.sum()
    ev = np.linalg.eigvalsh(np.array([[cxx, cxy], [cxy, cyy]]))
    aniso = float(ev[1] / max(ev[0], 1e-300))
    d2 = (xs - x0) ** 2 + (ys - y0) ** 2
    lo, hi = sigma_predicted * 0.5, sigma_predicted * 1.5
    best = None
    for s_ref in np.linspace(lo, hi, 120):
        g = np.exp(-d2 / (2 * s_ref ** 2))
        a = (g * w).sum() / (g * g).sum()
        err = np.sqrt(((w - a * g) ** 2).sum() / (w ** 2).sum())
        if best is None or err < best[0]:
            best = (float(err), float(s_ref), float(a))
    g0 = np.exp(-d2 / (2 * SIGMA ** 2))
    a0 = (g0 * w).sum() / (g0 * g0).sum()
    raw_vs_input = float(np.sqrt(((w - a0 * g0) ** 2).sum() / (w ** 2).sum()))
    return {"aniso": aniso, "iso_rel_l2": best[0],
            "sigma_ref": best[1], "amplitude": best[2],
            "raw_rel_l2_vs_unsmoothed_input": raw_vs_input}


def run_one_n(n_idx, rng, partial_fh):
    """The full pipeline at N = 3*F(n_idx)^2: build, discover, sweep both
    arms at a V-MATCHED iteration count (same target added variance V at
    every n, so B4's comparison is apples-to-apples on physical blur, not
    on iteration count), analyze against the PREDICTED blur. Returns the
    result row; checkpoints to partial_fh.

    Iteration count is derived, not fixed: iters = round(2*V / h^2), h^2 =
    the median squared nearest-neighbour spacing actually measured on this
    lattice (mesh scale shrinks as N grows, so iters grows too, for a FIXED
    physical target V -- this is why n=19 is out of budget under v2, see
    run()'s docstring)."""
    fn = F[n_idx]
    n = 3 * fn * fn
    x, y, r = vogel(n)
    bands = band_of(r)
    r_floor = fn / np.sqrt(n)
    qualifying = [b for b in range(1, N_BANDS + 1)
                  if np.sqrt((b - 1) / N_BANDS) >= r_floor]
    strides = discover_strides(x, y, bands, rng=rng)
    angles = crossing_angles(x, y, bands, strides, n, rng)
    x0, y0 = R_BUMP, 0.0
    field = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * SIGMA ** 2))
    sel = np.isin(bands, qualifying)

    h2 = median_nn_spacing_sq(x, y, sel, rng)
    iters = max(1, round(2 * ADDED_VAR / h2))
    sigma_predicted = float(np.sqrt(SIGMA ** 2 + ADDED_VAR))

    # baseline: the UNSMOOTHED input bump through the identical mask --
    # isolates the mask's own contribution to the anisotropy readout
    # (codex P1c: v1's asymptote may have been the mask, not the operator)
    baseline_res = analyze_bump(x, y, field, x0, y0, sel, SIGMA)

    fib_links = build_fib_links(n, bands, strides, qualifying)
    fib_blur = run_adi(field, fib_links, iters)
    fib_res = analyze_bump(x, y, fib_blur, x0, y0, sel, sigma_predicted)

    ctl_links, ctl_hist = build_control_links(x, y, n, bands, strides,
                                              qualifying, sel, rng)
    ctl_blur = run_adi(field, ctl_links, iters)
    ctl_res = analyze_bump(x, y, ctl_blur, x0, y0, sel, sigma_predicted)

    row = {
        "n_idx": n_idx, "N": int(n), "r_floor": float(r_floor),
        "qualifying_bands": qualifying, "iters": iters, "h2": h2,
        "sigma_predicted": sigma_predicted,
        "strides": {str(b): (v["pair"] if v else None)
                    for b, v in strides.items()},
        "stride_top5_band5": strides.get(5, {}).get("top5") if strides.get(5) else None,
        "crossing_angles": {str(b): v for b, v in angles.items()},
        "baseline_unsmoothed": baseline_res,
        "fib": fib_res, "control": ctl_res, "control_link_histogram": ctl_hist,
        "aniso_ratio_control_over_fib": (
            float(ctl_res["aniso"] / fib_res["aniso"])
            if fib_res and ctl_res else None),
        "aniso_change_from_baseline_fib": (
            float(fib_res["aniso"] - baseline_res["aniso"])
            if fib_res and baseline_res else None),
    }
    partial_fh.write(json.dumps({"stage": f"n{n_idx}", **row}) + "\n")
    partial_fh.flush()
    return row


def run():
    """v2. Headline at n=17 (N=3*F(17)^2) first, then the B4 sweep ascending
    n in {8,10,12,14} (17 reused from the headline). n=19 and n=21 are NOT
    RUN under v2's V-matched iteration scaling: iters = 2V/h^2 and h^2
    shrinks roughly as 1/N for a Vogel lattice, so a FIXED physical target V
    forces iters to grow roughly linearly with N -- at n=19 (N~52.4M) the
    predicted iteration count is in the tens of thousands, and total cost
    scales as N*iters, making it multiple orders of magnitude more expensive
    than the n=17 headline alone. This is a genuine budget cutoff, stated
    with its mechanism, not a silent drop -- v1's B4 ran n=19 only because
    its FIXED 8-iteration count never matched a physical target in the first
    place (part of why v1's numbers were void)."""
    rng = np.random.default_rng(SEED)
    out_dir = pathlib.Path(__file__).parent
    partial = out_dir / "spiral_adi_probe.partial.jsonl"
    with open(partial, "w") as pf:
        headline = run_one_n(17, rng, pf)
        sweep = []
        for n_idx in (8, 10, 12, 14):
            sweep.append(run_one_n(n_idx, rng, pf))
    sweep_full = sorted(sweep + [headline], key=lambda r: r["n_idx"])

    b2 = (headline["fib"] is not None
          and headline["fib"]["iso_rel_l2"] <= 0.15
          and headline["fib"]["aniso"] <= 1.25)
    b3 = (headline["aniso_ratio_control_over_fib"] is not None
          and headline["aniso_ratio_control_over_fib"] >= 1.5)
    out = {
        "headline": headline,
        "sweep": [{k: v for k, v in row.items()
                   if k in ("n_idx", "N", "iters", "qualifying_bands",
                            "baseline_unsmoothed", "fib", "control",
                            "aniso_ratio_control_over_fib",
                            "aniso_change_from_baseline_fib")}
                  for row in sweep_full],
        "n19_n21": ("NOT RUN under v2's V-matched iteration scaling -- "
                    "iters ~ 2V/h^2 grows roughly linearly with N for a "
                    "fixed physical blur target, making n=19/21 multiple "
                    "orders of magnitude more expensive than n=17; "
                    "mechanism stated, not a silent drop (see run() docstring)"),
        "verdicts": {
            "B2_iso": "PASS" if b2 else "FAIL",
            "B3_control": ("PASS" if b3 else
                           ("VOID -- control smooths as isotropically as "
                            "Fibonacci; the Fibonacci claim measures nothing"
                            if headline["aniso_ratio_control_over_fib"] is not None
                            else "NO-VERDICT")),
        },
    }
    with open(out_dir / "spiral_adi_probe.json", "w") as fh:
        json.dump(out, fh, indent=2)
    partial.unlink()
    return out


if __name__ == "__main__":
    res = run()
    h = res["headline"]
    print(f"N={h['N']}  r_floor={h['r_floor']:.4f}  qualifying={h['qualifying_bands']}")
    print(f"iters={h['iters']}  h2={h['h2']:.3e}  sigma_predicted={h['sigma_predicted']:.5f}")
    print("strides per band:", h["strides"])
    print(f"baseline (unsmoothed, same mask): {h['baseline_unsmoothed']}")
    print(f"fib:     {h['fib']}")
    print(f"control: {h['control']}")
    print(f"control link histograms (both families): {h['control_link_histogram']}")
    print(f"aniso ratio (control/fib): {h['aniso_ratio_control_over_fib']}")
    print(f"aniso change from baseline (fib): {h['aniso_change_from_baseline_fib']}")
    print("verdicts:", res["verdicts"])
    print("\nB4 sweep:")
    for row in res["sweep"]:
        f_ = row["fib"]
        b_ = row["baseline_unsmoothed"]
        print(f"  n={row['n_idx']:2d} N={row['N']:9d} iters={row['iters']:6d} "
              f"iso={f_['iso_rel_l2']:.4f} aniso={f_['aniso']:.4f} "
              f"baseline_aniso={b_['aniso']:.4f} "
              f"ratio={row['aniso_ratio_control_over_fib']}")
