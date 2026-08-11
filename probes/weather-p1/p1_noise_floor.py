"""SUPERSEDED by p1_ci_vs_floor.py -- kept for provenance only.

This script's `dev` is |bucket_center - original|, i.e. a DECODED RECONSTRUCTION
ERROR, which is the round-trip metric its own docstring disclaims (codex P1 on
PR #920). Its exceedance fractions fed doc section 12.10 and are superseded by
12.11. The Fisher-Z widths below were also adjacent-CENTRE spacings, not bin
widths, and produced 255 spans for 256 buckets. Both are fixed properly in
p1_ci_vs_floor.py; this file is not the measurement of record.

Original header follows.

P1 re-evaluated in the CORRECT frame (operator correction 2026-08-11):
no roundtrip -- compare the code's confidence-interval thresholds against the
original's noise floor. Deviation only matters where it EXCEEDS what the
original itself can resolve."""
import numpy as np, json
a=np.load('fixture/2m_temperature.npy').astype(np.float64)
clim=a.mean(axis=1,keepdims=True); anom=a-clim
def fz(s,eps=1e-9):
    """Fisher-Z: arctanh(s), clipped off the +/-1 poles by eps."""
    s=np.clip(s,-1+eps,1-eps); return 0.5*(np.log1p(s)-np.log1p(-s))

paths={}
# LINEAR path
lo,hi=np.percentile(anom,[0.4,99.6])
idx=np.clip(np.floor((anom-lo)/(hi-lo)*256),0,255).astype(np.uint8)
cen=lo+((np.arange(256)+0.5)/256)*(hi-lo)
paths['linear']=(np.abs(cen[idx]-anom), (hi-lo)/256/2)  # dev, half-bucket CI
# FISHER-Z path
lo_q,hi_q=np.percentile(anom,[0.4,99.6]); scale=max(abs(lo_q),abs(hi_q))
z=fz(anom/scale); zlo,zhi=np.percentile(z,[0.4,99.6])
idxz=np.clip(np.floor((z-zlo)/(zhi-zlo)*256),0,255).astype(np.uint8)
cenz=zlo+((np.arange(256)+0.5)/256)*(zhi-zlo)
devz=np.abs(np.tanh(cenz)[idxz]*scale-anom)
# CI per bucket from EDGES (257 of them -> 256 spans), not centre spacing.
_edges = zlo + np.arange(257)*(zhi-zlo)/256
wid = np.abs(np.diff(np.tanh(_edges))*scale)      # 256 full widths, in K
paths['fisher_z'] = (devz, None)

FLOORS=[0.25,0.5,1.0,2.0]   # K -- reference class; exact citable floor to pin
print(f"{'path':10s} {'CI_mid(K)':>10s} " + " ".join(f">{f}K".rjust(9) for f in FLOORS) + f" {'max(K)':>8s}")
out={}
for name,(dev,ci) in paths.items():
    ci_mid = ci if ci is not None else float(wid[len(wid)//2]/2)
    row=[float((dev>f).mean()) for f in FLOORS]
    print(f"{name:10s} {ci_mid:10.4f} " + " ".join(f"{v:9.2e}" for v in row) + f" {dev.max():8.2f}")
    out[name]={'ci_mid_K':ci_mid,'exceed_frac':dict(zip(map(str,FLOORS),row)),'max_K':float(dev.max())}
# fisher-z CI across the range (the CI is position-dependent -- that IS the shape finding)
q=[0,64,128,192,255]
print("\nfisher-z per-position CI (half bucket width in K): ",
      {f"idx{k}": round(float(wid[k]/2),4) for k in q})
out['fisher_z']['ci_by_position_K']={str(k): float(wid[k]/2) for k in q}
json.dump(out,open('p1_noise_floor.json','w'),indent=2)
