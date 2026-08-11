"""Verify the P1 apparatus BEFORE claiming fisher-z is falsified."""
import sys
import numpy as np
from scipy import stats as st
a=np.load('fixture/2m_temperature.npy').astype(np.float64)
clim=a.mean(axis=1,keepdims=True); anom=a-clim

# 1) sanity: bf16 implementation must have ~8 bits mantissa -> rel err <= 2^-9
def bf16(x):
    """Round f64 to bfloat16 precision (round-to-nearest-even) and back to f64."""
    f=np.asarray(x,dtype=np.float32).view(np.uint32)
    r=((f>>16)&1).astype(np.uint32)
    return (((f+0x7FFF+r)&0xFFFF0000).astype(np.uint32)).view(np.float32).astype(np.float64)
v=np.array([1.0,300.0,-6.5,1e-3]); rel=np.abs(bf16(v)-v)/np.abs(v)
ok = bool((rel <= 2**-9 + 1e-12).all())
print("bf16 rel err (must be <= 2^-9 = 0.00195):", rel, "OK" if ok else "BROKEN")
if not ok:
    # A self-check that prints BROKEN and keeps going is not a self-check.
    # Every BF16 number downstream would be unvalidated. Fail loudly instead.
    sys.exit("APPARATUS FAILURE: bf16 helper exceeds 2^-9; refusing to emit measurements")

# 2) THE MECHANISM: what shape is the input distribution?
lo_q,hi_q=np.percentile(anom,[0.4,99.6]); scale=max(abs(lo_q),abs(hi_q)); s=(anom/scale).ravel()
print(f"\nanomaly/scale  -> |s| distribution: mean|s|={np.abs(s).mean():.4f} "
      f"frac|s|<0.25={np.mean(np.abs(s)<0.25):.4f} frac|s|>0.9={np.mean(np.abs(s)>0.9):.4f}")
print(f"  kurtosis(s)={st.kurtosis(s):.3f} (0=gaussian, <0=flat, >0=peaked)  skew={st.skew(s):.3f}")

# a correlation-like variable for CONTRAST: piles at the bounds
r_like=np.tanh(np.random.default_rng(0).normal(0,1.5,200000))
print(f"  CONTRAST corr-like: mean|r|={np.abs(r_like).mean():.4f} "
      f"frac|r|<0.25={np.mean(np.abs(r_like)<0.25):.4f} frac|r|>0.9={np.mean(np.abs(r_like)>0.9):.4f}")

# 3) where do the 256 buckets actually land, both ways?
def quant(x):
    """Quantize x into 256 uniform buckets over its range; return (index, centre)."""
    lo,hi=np.percentile(x,[0.4,99.6])
    idx=np.clip(np.floor((x-lo)/(hi-lo)*256),0,255).astype(np.uint8)
    cen=lo+((np.arange(256)+0.5)/256)*(hi-lo)
    return idx,cen
def fz(s,eps=1e-9):
    """Fisher-Z: arctanh(s), clipped off the +/-1 poles by eps."""
    s=np.clip(s,-1+eps,1-eps); return 0.5*(np.log1p(s)-np.log1p(-s))

for name,x,inv in [("LINEAR(anom)",anom.ravel(),lambda c:c),
                   ("FISHERZ(s)",fz(s),lambda c:np.tanh(c)*scale)]:
    idx,cen=quant(x); occ=np.bincount(idx,minlength=256)
    rec=inv(cen)[idx]
    truth=anom.ravel() if name.startswith("LINEAR") else anom.ravel()
    err=np.abs(rec-truth)
    n=idx.size;p=1/256;mu=n*p;sd=(n*p*(1-p))**.5
    # effective buckets = exp(entropy) : how many buckets are REALLY used
    q=occ/occ.sum(); nz=q[q>0]; eff=float(np.exp(-(nz*np.log(nz)).sum()))
    print(f"\n{name}: mae={err.mean():.5f}K p99={np.percentile(err,99):.5f}K "
          f"empty={int((occ==0).sum())} effective_buckets={eff:.1f}/256 drift={np.abs(occ-mu).max()/sd:.0f}")
