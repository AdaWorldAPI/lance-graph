"""P2 (re-scoped per §12.1) — does cross-variable comparability survive
WITHOUT a shared Fisher-Z? Shared canonical floor vs per-variable floors."""
import numpy as np, json
from scipy import stats as st
rng = np.random.default_rng(7)

def load(v):
    a = np.load(f'fixture/{v}.npy').astype(np.float64)
    assert np.isfinite(a).all()
    return a - a.mean(axis=1, keepdims=True)      # zonal-mean climatology proxy

VARS = ['2m_temperature', '2m_dewpoint_temperature', '10m_u_component_of_wind']
UNITS = {'2m_temperature':'K','2m_dewpoint_temperature':'K','10m_u_component_of_wind':'m/s'}
anom = {v: load(v) for v in VARS}
for v in VARS:
    a = anom[v].ravel()
    print(f"{v:30s} unit={UNITS[v]:4s} std={a.std():8.4f} kurt={st.kurtosis(a):+7.3f} "
          f"skew={st.skew(a):+7.3f} range=[{a.min():9.3f},{a.max():9.3f}]")

# ---- the shape-appropriate transform for bell-shaped anomalies = standardize
# (identity after scaling; §12.1 measured arctanh as actively harmful here)
z = {v: (anom[v].ravel() - anom[v].mean()) / anom[v].std() for v in VARS}

def quant(x, lo, hi):
    idx = np.clip(np.floor((x - lo)/(hi - lo)*256), 0, 255).astype(np.uint8)
    cen = lo + ((np.arange(256)+0.5)/256)*(hi - lo)
    return idx, cen

# SHARED canonical floor: one window over the pooled z of all variables
pool = np.concatenate([z[v] for v in VARS])
s_lo, s_hi = np.percentile(pool, [0.4, 99.6])
shared = {v: quant(z[v], s_lo, s_hi) for v in VARS}
# PER-VARIABLE floors: each its own window
per = {v: quant(z[v], *np.percentile(z[v], [0.4, 99.6])) for v in VARS}

N = 200_000
res = {'shared_window': [float(s_lo), float(s_hi)], 'n_pairs': N, 'pairs': {}}
print(f"\nshared z-window = [{s_lo:.4f}, {s_hi:.4f}]")
print(f"\n{'pair':<52s} {'rho_shared':>11s} {'rho_pervar':>11s} {'verdict':>9s}")
for i, va in enumerate(VARS):
    for vb in VARS[i+1:]:
        ia = rng.integers(0, z[va].size, N); ib = rng.integers(0, z[vb].size, N)
        truth = np.abs(z[va][ia] - z[vb][ib])                       # float z-domain truth
        d_sh  = np.abs(shared[va][0][ia].astype(np.int32) - shared[vb][0][ib].astype(np.int32))
        d_pv  = np.abs(per[va][0][ia].astype(np.int32)   - per[vb][0][ib].astype(np.int32))
        r_sh = st.spearmanr(d_sh, truth).statistic
        r_pv = st.spearmanr(d_pv, truth).statistic
        key = f"{va} x {vb}"
        res['pairs'][key] = {'rho_shared_floor': float(r_sh), 'rho_per_variable_floor': float(r_pv),
                             'units': [UNITS[va], UNITS[vb]]}
        print(f"{key:<52s} {r_sh:11.4f} {r_pv:11.4f} {'SHARED' if r_sh>r_pv else 'PER-VAR':>9s}")

# within-variable control: shared floor must not wreck same-variable distance
print(f"\n{'within-variable control':<52s} {'rho_shared':>11s} {'rho_pervar':>11s}")
for v in VARS:
    ia = rng.integers(0, z[v].size, N); ib = rng.integers(0, z[v].size, N)
    truth = np.abs(z[v][ia] - z[v][ib])
    r_sh = st.spearmanr(np.abs(shared[v][0][ia].astype(np.int32)-shared[v][0][ib].astype(np.int32)), truth).statistic
    r_pv = st.spearmanr(np.abs(per[v][0][ia].astype(np.int32)-per[v][0][ib].astype(np.int32)), truth).statistic
    occ = np.bincount(shared[v][0], minlength=256); q=occ/occ.sum(); nz=q[q>0]
    res['pairs'][f"{v} (within)"] = {'rho_shared_floor': float(r_sh), 'rho_per_variable_floor': float(r_pv),
                                     'shared_effective_buckets': float(np.exp(-(nz*np.log(nz)).sum())),
                                     'shared_empty_buckets': int((occ==0).sum())}
    print(f"{v:<52s} {r_sh:11.4f} {r_pv:11.4f}   shared_eff_buckets={np.exp(-(nz*np.log(nz)).sum()):5.1f} empty={int((occ==0).sum())}")
json.dump(res, open('p2_results.json','w'), indent=2)
