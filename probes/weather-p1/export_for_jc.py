"""Export (truth, reconstructed) pairs as f64 LE binary for the jc reliability battery."""
import numpy as np
rng=np.random.default_rng(11)
VARS=['2m_temperature','2m_dewpoint_temperature','10m_u_component_of_wind']
def load(v):
    """Load an ERA5 fixture variable and return its zonal-mean anomaly (K)."""
    a=np.load(f'fixture/{v}.npy').astype(np.float64); return a-a.mean(axis=1,keepdims=True)
anom={v:load(v) for v in VARS}
z={v:(anom[v].ravel()-anom[v].mean())/anom[v].std() for v in VARS}
pool=np.concatenate([z[v] for v in VARS]); lo,hi=np.percentile(pool,[0.4,99.6])
cen=lo+((np.arange(256)+0.5)/256)*(hi-lo)
N=50_000
cols=[]   # each var contributes truth + reconstructed, subsampled on the SAME indices
for v in VARS:
    idx=rng.integers(0,z[v].size,N)
    t=z[v][idx]
    q=np.clip(np.floor((t-lo)/(hi-lo)*256),0,255).astype(np.uint8)
    cols.append((v,t,cen[q]))
with open('jc_input.bin','wb') as f:
    np.array([len(cols),N],dtype='<i8').tofile(f)
    for _,t,r in cols:
        t.astype('<f8').tofile(f); r.astype('<f8').tofile(f)
for v,t,r in cols:
    print(f"{v:30s} n={N} truth[std={t.std():.4f}] recon[std={r.std():.4f}] max|err|={np.abs(r-t).max():.5f}")
print("wrote jc_input.bin")
