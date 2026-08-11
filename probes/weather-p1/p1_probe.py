"""P1 — re-runnable measurement of the doc's §6 claims on REAL ERA5 2m_temperature."""
import numpy as np, json
a = np.load('fixture/2m_temperature.npy').astype(np.float64)
ny, nx = a.shape
assert np.isfinite(a).all(), "unexpected non-finite"
print(f"field: {a.shape} = {a.size:,} gridpoints, K range [{a.min():.3f}, {a.max():.3f}]")

def bf16(x):
    """Round-to-nearest-even f32->bf16 (truncate mantissa to 7 bits), back to f64."""
    f = np.asarray(x, dtype=np.float32).view(np.uint32)
    r = ((f >> 16) & 1).astype(np.uint32)
    f2 = ((f + 0x7FFF + r) & 0xFFFF0000).astype(np.uint32)
    return f2.view(np.float32).astype(np.float64)

# ---- §6.3 : BF16 on raw Kelvin vs BF16 on anomaly -------------------------
# climatology proxy = zonal mean (per-latitude), the cheapest honest PLACE term.
clim = a.mean(axis=1, keepdims=True)          # (721,1)
anom = a - clim

err_raw  = np.abs(bf16(a) - a)
err_anom = np.abs((bf16(anom) + clim) - a)
r = {}
r['bf16_raw_kelvin_mae']   = float(err_raw.mean())
r['bf16_raw_kelvin_p99']   = float(np.percentile(err_raw, 99))
r['bf16_raw_kelvin_max']   = float(err_raw.max())
r['bf16_anomaly_mae']      = float(err_anom.mean())
r['bf16_anomaly_p99']      = float(np.percentile(err_anom, 99))
r['bf16_anomaly_max']      = float(err_anom.max())
r['bf16_improvement_x']    = float(err_raw.mean()/err_anom.mean())
r['anomaly_std_K']         = float(anom.std())
r['raw_std_K']             = float(a.std())

# ---- palette256: fisher-z normalize the anomaly, then 256-bucket ----------
def fisher_z(s, eps=1e-9):
    s = np.clip(s, -1+eps, 1-eps)
    return 0.5*(np.log1p(s) - np.log1p(-s))

# map anomaly onto [-1,1] by a robust scale (0.4-99.6 pct, mirroring RollingFloor.roll)
lo_q, hi_q = np.percentile(anom, [0.4, 99.6])
scale = max(abs(lo_q), abs(hi_q))
s = anom/scale
z = fisher_z(s)
zlo, zhi = np.percentile(z, [0.4, 99.6])
idx = np.clip(np.floor((z - zlo)/(zhi - zlo)*256), 0, 255).astype(np.uint8)   # RollingFloor.quantize
centers = zlo + ((np.arange(256)+0.5)/256)*(zhi - zlo)                        # bucket_center
z_hat = centers[idx]
anom_hat = np.tanh(z_hat)*scale                                               # inverse fisher-z
err_pal = np.abs((anom_hat + clim) - a)
r['palette256_mae_K']   = float(err_pal.mean())
r['palette256_p99_K']   = float(np.percentile(err_pal, 99))
r['palette256_max_K']   = float(err_pal.max())
def effective_buckets(counts):
    """exp(Shannon entropy) of the occupancy histogram: how many of the 256
    addresses are REALLY carrying data. Zero-count buckets contribute nothing."""
    q = counts / counts.sum()
    nz = q[q > 0]
    return float(np.exp(-(nz * np.log(nz)).sum()))

occ = np.bincount(idx.ravel(), minlength=256)
r['palette_occupancy_min'] = int(occ.min()); r['palette_occupancy_max'] = int(occ.max())
r['palette256_effective_buckets'] = effective_buckets(occ)
r['palette_empty_buckets'] = int((occ==0).sum())
r['palette_saturated_frac'] = float((occ[0]+occ[255])/idx.size)
# drift_score exactly as quantize.rs: max |occ_i - mu|/sd, multinomial
n = idx.size; p = 1/256; mu = n*p; sd = (n*p*(1-p))**0.5
r['drift_score_uniform_sd'] = float(np.abs(occ-mu).max()/sd)

# LINEAR (no fisher-z) control — does the normalization actually buy anything?
alo, ahi = np.percentile(anom, [0.4, 99.6])
idx_lin = np.clip(np.floor((anom-alo)/(ahi-alo)*256), 0, 255).astype(np.uint8)
cen_lin = alo + ((np.arange(256)+0.5)/256)*(ahi-alo)
err_lin = np.abs((cen_lin[idx_lin] + clim) - a)
occ_lin = np.bincount(idx_lin.ravel(), minlength=256)
r['palette256_LINEAR_mae_K'] = float(err_lin.mean())
r['palette256_LINEAR_p99_K'] = float(np.percentile(err_lin,99))
r['drift_score_LINEAR']      = float(np.abs(occ_lin-mu).max()/sd)
r['palette256_LINEAR_effective_buckets'] = effective_buckets(occ_lin)
r['palette256_LINEAR_empty_buckets']     = int((occ_lin==0).sum())
r['bytes_per_point_f32'] = 4; r['bytes_per_point_palette'] = 1

for k,v in r.items(): print(f"  {k:32s} {v}")
json.dump(r, open('p1_results_temperature.json','w'), indent=2)
