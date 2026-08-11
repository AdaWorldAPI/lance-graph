"""P1, CORRECTED metric (codex P1 on #920): compare each point's BUCKET
CONFIDENCE INTERVAL to the floor. Never the decoded error.

CI(b) = half-width of bucket b expressed in ORIGINAL units (K).
Saturated edge buckets (0, 255) have UNBOUNDED CI and are reported separately,
never folded into an 'exceeds' fraction as if they had a finite width."""
import numpy as np, json
a=np.load('fixture/2m_temperature.npy').astype(np.float64)
clim=a.mean(axis=1,keepdims=True); anom=a-clim
def fz(s,eps=1e-9):
    s=np.clip(s,-1+eps,1-eps); return 0.5*(np.log1p(s)-np.log1p(-s))
FLOORS=[0.25,0.5,1.0,2.0]
out={}

def report(name, idx, edges_K):
    """edges_K: 257 bucket edges in ORIGINAL units. CI(b)=half the bucket width."""
    ci = np.diff(edges_K)/2.0                      # 256 half-widths, in K
    interior = (idx>0)&(idx<255)
    sat = ~interior
    occ = np.bincount(idx.ravel(), minlength=256)
    print(f"\n{name}")
    print(f"  saturated (buckets 0/255, CI UNBOUNDED): {sat.mean()*100:.3f}% of points")
    print(f"  interior CI (K): min={ci[1:255].min():.5f} med={np.median(ci[1:255]):.5f} max={ci[1:255].max():.5f}")
    rows={}
    for f in FLOORS:
        # fraction of ALL points whose own bucket's CI exceeds the floor
        exceed = (ci[idx][interior] > f)
        frac_int = exceed.sum()/idx.size
        rows[str(f)]={'interior_CI_exceeds':float(frac_int),
                      'saturated_unbounded':float(sat.mean()),
                      'total_at_or_above_floor':float(frac_int+sat.mean())}
        print(f"  floor {f:>4}K : interior CI>floor {frac_int*100:8.4f}%  "
              f"+ saturated {sat.mean()*100:.3f}%  = {(frac_int+sat.mean())*100:8.4f}%")
    out[name]={'sat_frac':float(sat.mean()),
               'interior_ci_min':float(ci[1:255].min()),'interior_ci_med':float(np.median(ci[1:255])),
               'interior_ci_max':float(ci[1:255].max()),'floors':rows}

# LINEAR: uniform buckets in K
lo,hi=np.percentile(anom,[0.4,99.6])
idx=np.clip(np.floor((anom-lo)/(hi-lo)*256),0,255).astype(np.uint8)
report('linear', idx, lo+np.arange(257)/256*(hi-lo))

# FISHER-Z: uniform in z, NON-uniform in K after tanh
lo_q,hi_q=np.percentile(anom,[0.4,99.6]); scale=max(abs(lo_q),abs(hi_q))
z=fz(anom/scale); zlo,zhi=np.percentile(z,[0.4,99.6])
idxz=np.clip(np.floor((z-zlo)/(zhi-zlo)*256),0,255).astype(np.uint8)
report('fisher_z', idxz, np.tanh(zlo+np.arange(257)/256*(zhi-zlo))*scale)

json.dump(out,open('p1_ci_vs_floor.json','w'),indent=2)
