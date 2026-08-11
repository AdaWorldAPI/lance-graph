"""EV-8 (v2) -- stability of the P2 estimator via seed-resampling.

Per `.claude/plans/weather-substrate-evaluation-v1.md` Section 3, EV-8 (v2,
post-audit): the v1 spec was a rubber-stamp ("state effective-n and re-grade"
can never fail) and used the WRONG n (the 1,038,240 gridpoints instead of the
200,000 INDEPENDENT UNIFORM INDEX PAIRS p2_probe.py actually draws -- see
p2_probe.py:42,48).

This probe re-runs p2_probe.py's estimator K>=20 times, varying ONLY the RNG
seed that drives the index-pair draw (nothing else -- the z-values, the
shared/per-variable quantization floors, and the bucket boundaries are all
deterministic and computed once, exactly as p2_probe.py computes them), and
reports per-correlation mean / sd / min / max / CI width across seeds.

PRE-REGISTERED (before interpreting any number below): for two independent
uniform index draws (ia, ib each drawn fresh per seed via rng.integers with
replace=True -- see p2_probe.py:48), n_eff ~= n=200,000 is the EXPECTED
finding. That outcome is a FINDING (the estimator behaves as a genuinely IID
sampling design), NOT a wiring alarm. n_eff << n would indict the ESTIMATOR
(a hidden dependence smuggled into the pairing/index-draw), not the
underlying field.

rho(d_shared, truth) and rho(d_per_variable, truth) are CODEC-CONSISTENCY
statistics: d_shared / d_per_variable are DETERMINISTIC MONOTONE
QUANTIZATIONS of the SAME z-values `truth` is computed from (p2_probe.py
:29-33,49-51) -- the correlation measures self-consistency of a lossy
quantizer against its own lossless input, not a relationship between two
independently-measured quantities. Accordingly the deliverable of this probe
is CI WIDTH and CROSS-SEED STABILITY of that self-consistency statistic --
NOT a p-value dressed up as inference. No p-value is computed anywhere in
this file.

Jirak 2016 weak-dependence rates (I-NOISE-FLOOR-JIRAK) are the mandatory
citation for POOLED/SPATIAL statistics on this workspace's fields, because
the underlying grid has weakly-dependent (spatially correlated) values. That
does NOT apply to the quantity resampled here: p2_probe.py's index pairs
(ia, ib) are drawn independently and uniformly WITH replacement per resample
(rng.integers, default replace=True), so each of the N=200,000 samples per
seed is an IID draw (with replacement) from a fixed bivariate population --
the spatial correlation of the underlying field is irrelevant to the
SAMPLING design of THIS estimator (it would matter for a statistic computed
directly over spatially-adjacent points, which this is not). Per the plan's
own escape clause for EV-8 ("Jirak's `p` pinned in the artifact or the Jirak
citation dropped for the resampling CI"): the Jirak citation is DROPPED here
and the classical IID Fisher-z delta-method SE is used instead, strictly as
a REFERENCE SCALE to express n_eff -- never as a significance/p-value claim.
"""
import numpy as np
import json
from scipy import stats as st

K_SEEDS = list(range(1, 26))   # K=25 independent seed-resamples (>= 20 required by spec)
N = 200_000                    # the P2 estimator's own n -- p2_probe.py:42
GRID_N = 721 * 1440            # 1,038,240 -- the WRONG n the v1 spec used; recorded for contrast, never used as n below

VARS = ['2m_temperature', '2m_dewpoint_temperature', '10m_u_component_of_wind']
UNITS = {'2m_temperature': 'K', '2m_dewpoint_temperature': 'K', '10m_u_component_of_wind': 'm/s'}


def load(v):
    """Identical to p2_probe.py:load -- zonal-mean anomaly of the raw field."""
    a = np.load(f'fixture/{v}.npy').astype(np.float64)
    assert np.isfinite(a).all()
    return a - a.mean(axis=1, keepdims=True)


def quant(x, lo, hi):
    """Identical to p2_probe.py:quant -- 256 uniform buckets over [lo, hi];
    only the bucket INDEX is needed here (p2_probe.py's `cen` is unused by
    the correlation computation, so it is dropped)."""
    idx = np.clip(np.floor((x - lo) / (hi - lo) * 256), 0, 255).astype(np.uint8)
    return idx


anom = {v: load(v) for v in VARS}
# Identical to p2_probe.py: standardize (identity after scaling; §12.1
# measured arctanh as actively harmful here -- not re-litigated by this probe).
z = {v: (anom[v].ravel() - anom[v].mean()) / anom[v].std() for v in VARS}

# Deterministic (no randomness, computed ONCE): the shared canonical floor
# and the per-variable floors, exactly as p2_probe.py computes them.
pool = np.concatenate([z[v] for v in VARS])
s_lo, s_hi = np.percentile(pool, [0.4, 99.6])
shared_idx = {v: quant(z[v], s_lo, s_hi) for v in VARS}
per_idx = {v: quant(z[v], *np.percentile(z[v], [0.4, 99.6])) for v in VARS}

pairs = [(VARS[i], VARS[j]) for i in range(len(VARS)) for j in range(i + 1, len(VARS))]

# key -> floor_type -> list of rho values, one entry per seed
series = {}
for va, vb in pairs:
    series.setdefault(f"{va} x {vb}", {})['shared'] = []
    series[f"{va} x {vb}"]['per_variable'] = []
for v in VARS:
    series.setdefault(f"{v} (within)", {})['shared'] = []
    series[f"{v} (within)"]['per_variable'] = []

for seed in K_SEEDS:
    rng = np.random.default_rng(seed)
    # cross-variable pairs (in the SAME order p2_probe.py draws them, so a
    # given seed draws the same index streams for the same key across runs)
    for va, vb in pairs:
        ia = rng.integers(0, z[va].size, N)
        ib = rng.integers(0, z[vb].size, N)
        truth = np.abs(z[va][ia] - z[vb][ib])
        d_sh = np.abs(shared_idx[va][ia].astype(np.int32) - shared_idx[vb][ib].astype(np.int32))
        d_pv = np.abs(per_idx[va][ia].astype(np.int32) - per_idx[vb][ib].astype(np.int32))
        r_sh = st.spearmanr(d_sh, truth).statistic
        r_pv = st.spearmanr(d_pv, truth).statistic
        series[f"{va} x {vb}"]['shared'].append(float(r_sh))
        series[f"{va} x {vb}"]['per_variable'].append(float(r_pv))
    # within-variable controls
    for v in VARS:
        ia = rng.integers(0, z[v].size, N)
        ib = rng.integers(0, z[v].size, N)
        truth = np.abs(z[v][ia] - z[v][ib])
        r_sh = st.spearmanr(
            np.abs(shared_idx[v][ia].astype(np.int32) - shared_idx[v][ib].astype(np.int32)), truth
        ).statistic
        r_pv = st.spearmanr(
            np.abs(per_idx[v][ia].astype(np.int32) - per_idx[v][ib].astype(np.int32)), truth
        ).statistic
        series[f"{v} (within)"]['shared'].append(float(r_sh))
        series[f"{v} (within)"]['per_variable'].append(float(r_pv))


def classical_se(rho, n):
    """Fisher-z delta-method SE of an IID-sampled correlation coefficient:
    Var(rho) ~= (1-rho^2)^2 / (n-3)  =>  SE(rho) ~= (1-rho^2)/sqrt(n-3).
    Used strictly as a REFERENCE SCALE for n_eff -- see module docstring
    re: dropping the Jirak weak-dependence citation for THIS probe."""
    return (1.0 - rho ** 2) / np.sqrt(max(n - 3, 1))


results = {}
for key, by_floor in series.items():
    results[key] = {}
    for floor_type, vals in by_floor.items():
        arr = np.array(vals)
        mean = float(arr.mean())
        sd = float(arr.std(ddof=1))  # sample sd across the K seed-resamples
        se_theory = classical_se(mean, N)
        # Solve se_theory(n_eff) = observed sd for n_eff at the SAME mean rho:
        #   sd = (1-rho^2)/sqrt(n_eff-3)  =>  n_eff = ((1-rho^2)/sd)^2 + 3
        n_eff = float(((1.0 - mean ** 2) / sd) ** 2 + 3.0) if sd > 0 else None
        results[key][floor_type] = {
            'seed_count': len(vals),
            'values_by_seed': vals,
            'mean': mean,
            'sd': sd,
            'min': float(arr.min()),
            'max': float(arr.max()),
            'range': float(arr.max() - arr.min()),
            'ci_95_width_normal_approx': float(2 * 1.959964 * sd),
            'ci_95_lo': float(mean - 1.959964 * sd),
            'ci_95_hi': float(mean + 1.959964 * sd),
            'classical_se_theory_at_n_pairs': float(se_theory),
            'n_eff_estimate': n_eff,
            'n_eff_over_n_pairs': (n_eff / N) if n_eff is not None else None,
        }

out = {
    'n_pairs_per_resample': N,
    'grid_n_gridpoints_NOT_USED_recorded_for_contrast': GRID_N,
    'k_seed_resamples': len(K_SEEDS),
    'seeds': K_SEEDS,
    'vars': VARS,
    'units': UNITS,
    'shared_window_z': [float(s_lo), float(s_hi)],
    'jirak': {
        'dropped_for_this_probe': True,
        'reason': (
            'index pairs are drawn independently and uniformly WITH replacement '
            'per resample (rng.integers, default replace=True); each of the '
            'N=200000 samples is therefore an IID draw from a fixed bivariate '
            'population regardless of the spatial correlation structure of the '
            'underlying field. Classical Fisher-z delta-method SE is used as a '
            'reference scale for n_eff instead of a Jirak weak-dependence rate.'
        ),
    },
    'preregistration': {
        'expected_finding': (
            'n_eff ~= n_pairs_per_resample (200000) for these IID index-pair '
            'draws. This is the EXPECTED outcome and is a FINDING, not a wiring '
            'alarm. n_eff << n_pairs_per_resample would indict the ESTIMATOR '
            '(a hidden dependence in the index-pair draw), not the underlying '
            'field.'
        ),
        'correct_n_for_this_estimator': N,
        'wrong_n_used_by_v1_spec': GRID_N,
        'wrong_n_reason': (
            'v1 cited the grid size 721*1440=1038240 gridpoints as n; the P2 '
            'estimator (p2_probe.py:42,48) draws N=200000 INDEPENDENT UNIFORM '
            'INDEX PAIRS, not the grid -- that is the correct n for any CI '
            'statement about this estimator.'
        ),
    },
    'statistic_class': {
        'is_codec_consistency_statistic': True,
        'note': (
            'rho(d_shared, truth) and rho(d_per_variable, truth) measure '
            'self-consistency of a DETERMINISTIC MONOTONE QUANTIZER against the '
            'SAME z-values it quantizes -- not a relationship between two '
            'independently-measured quantities. The deliverable of this probe is '
            'CI WIDTH and CROSS-SEED STABILITY of that statistic, not a p-value '
            'dressed up as inference.'
        ),
    },
    'no_pvalue_computed': True,
    'results': results,
}

# Honest, PROGRAMMATICALLY-derived summary of what the pre-registered
# "n_eff ~= n_pairs" expectation actually found. This is a description of
# the measured pattern (bucketed by ratio band), NOT a p-value and NOT a
# verdict -- see 'interpretation_caveat' for why a single verdict would
# overreach what this probe alone can establish.
n_eff_ratios = {
    f"{key} / {floor_type}": r['n_eff_over_n_pairs']
    for key, by_floor in results.items()
    for floor_type, r in by_floor.items()
    if r['n_eff_over_n_pairs'] is not None
}
near_expected = {k: v for k, v in n_eff_ratios.items() if 0.3 <= v <= 3.0}
far_below = {k: v for k, v in n_eff_ratios.items() if v < 0.3}
far_above = {k: v for k, v in n_eff_ratios.items() if v > 3.0}

out['finding'] = {
    'measured_pattern': (
        'MIXED, not uniformly n_eff ~= n_pairs as pre-registered. '
        f'{len(near_expected)}/{len(n_eff_ratios)} series land within 0.3x-3x '
        f'of n_eff/n_pairs == 1 (the pre-registered expectation); '
        f'{len(far_below)}/{len(n_eff_ratios)} land BELOW 0.3x (n_eff far '
        f'smaller than n_pairs); {len(far_above)}/{len(n_eff_ratios)} land '
        f'ABOVE 3x.'
    ),
    'series_near_expected_0p3_to_3x': near_expected,
    'series_far_below_0p3x': far_below,
    'series_far_above_3x': far_above,
    'interpretation_caveat': (
        'The near-expected cases are the two cross-variable per_variable-floor '
        'pairs whose mean rho sits well below unity (~0.86-0.87: T x u and '
        'Td x u, per-variable floor). The far-below cases are every series '
        'whose mean rho sits at or above ~0.9994 (all shared-floor series, '
        'both within-variable controls under both floors, and the '
        'shared-floor cross-variable pairs). This split tracks correlation '
        'LEVEL, not pair identity, which is consistent with a KNOWN '
        'limitation of the classical Fisher-z delta-method reference formula '
        'rather than necessarily a hidden dependence in the index-pair draw: '
        'd_shared/d_per_variable are INTEGER-valued (256-bucket '
        'quantizations), so Spearman rank correlation on them carries heavy '
        'TIES, and the untied delta-method variance formula '
        '(1-rho^2)^2/(n-3) is known to understate variance near rho~1 under '
        'tie-heavy discretization. This probe does NOT adjudicate that '
        'hypothesis (a ties-corrected Spearman variance estimator would be '
        'needed, out of scope here) -- it reports the measured n_eff/n_pairs '
        'split honestly and flags the caveat rather than asserting either '
        '"the estimator is indicted" or "n_eff ~= n confirmed" uniformly.'
    ),
}

with open('ev8_estimator_stability.json', 'w') as f:
    json.dump(out, f, indent=2)

print(f"K={len(K_SEEDS)} seed-resamples, N={N} pairs/resample "
      f"(NOT the grid size {GRID_N})\n")
print(f"{'key':<40s} {'floor':<13s} {'mean':>10s} {'sd':>10s} "
      f"{'min':>10s} {'max':>10s} {'ci95_width':>12s} {'n_eff/n':>10s}")
for key, by_floor in results.items():
    for floor_type, r in by_floor.items():
        n_eff_ratio = r['n_eff_over_n_pairs']
        ratio_str = f"{n_eff_ratio:10.4f}" if n_eff_ratio is not None else f"{'None':>10s}"
        print(f"{key:<40s} {floor_type:<13s} {r['mean']:10.6f} {r['sd']:10.2e} "
              f"{r['min']:10.6f} {r['max']:10.6f} {r['ci_95_width_normal_approx']:12.2e} {ratio_str}")
