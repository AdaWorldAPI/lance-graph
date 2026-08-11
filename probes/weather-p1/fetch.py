"""Fetch one ERA5 timestep from the public ARCO-ERA5 Zarr v2 store.

Reads chunk (t, 0, 0) for each variable at 2021-06-15 12:00 UTC and writes
`fixture/<var>.npy` (721x1440 f32). Requires `zmeta.json` (the store's
consolidated .zmetadata) alongside this script; see README.md for the fetch.

A 404 is VALID Zarr v2 semantics -- a missing chunk is entirely `fill_value`
(NaN here), not a fetch failure -- so `get()` returns the fill array and the
run continues. Re-runnable and idempotent; no credentials needed (public
bucket, anonymous HTTPS)."""
import json, urllib.request, numpy as np, numcodecs, datetime as dt, os
B="https://storage.googleapis.com/gcp-public-data-arco-era5/ar/1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr"
meta=json.load(open('zmeta.json'))['metadata']
# time units
print("time .zattrs:", json.dumps(meta.get('time/.zattrs',{}))[:200])
epoch=dt.datetime(1959,1,1)
target=dt.datetime(2021,6,15,12)
t=int((target-epoch).total_seconds()//3600)
print(f"target={target} -> time index t={t} (shape0=561264, in range={t<561264})")

def get(var, t):
    """Fetch one chunk. A 404 is VALID Zarr v2 semantics -- a missing chunk means
    the chunk is entirely `fill_value` (NaN here), NOT a fetch failure. Several
    ARCO-ERA5 variables are legitimately absent at a given timestep; returning
    the fill array is what the format specifies (codex P2 on #920)."""
    za=meta[f'{var}/.zarray']; comp=za['compressor']
    key=f'{var}/{t}.0.0'
    req=urllib.request.Request(f'{B}/{key}')
    try:
        raw=urllib.request.urlopen(req, timeout=300).read()
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
        fill=za.get('fill_value')
        fill=np.nan if fill is None else fill
        a=np.full(za['chunks'][1:], fill, dtype=za['dtype'])
        return a, 0, a.nbytes          # 0 compressed bytes: nothing is stored
    dec=numcodecs.get_codec(comp).decode(raw)
    a=np.frombuffer(dec,dtype=za['dtype']).reshape(za['chunks'])[0]
    return a, len(raw), a.nbytes

os.makedirs('fixture',exist_ok=True)
rows=[]
for var in ['2m_temperature','total_column_water_vapour','surface_pressure',
            '10m_u_component_of_wind','10m_v_component_of_wind']:
    a,ncomp,nraw=get(var,t)
    if ncomp==0:
        print(f"{var:34s} MISSING CHUNK -> all-fill (valid Zarr semantics); not saved")
        continue
    np.save(f'fixture/{var}.npy', a)
    finite=np.isfinite(a)
    rows.append((var,a.shape,a.dtype.str,ncomp,a.nbytes,a.nbytes/ncomp,
                 float(a[finite].min()),float(a[finite].max()),int((~finite).sum())))
    print(f"{var:34s} shape={a.shape} dtype={a.dtype} comp={ncomp:>9,}B raw={a.nbytes:>9,}B "
          f"ratio={a.nbytes/ncomp:5.3f}x min={a[finite].min():12.4f} max={a[finite].max():12.4f} nonfinite={(~finite).sum()}")
json.dump({'time_index':t,'target_utc':target.isoformat(),
           'store':B,
           'vars':{r[0]:{'shape':list(r[1]),'dtype':r[2],'compressed_bytes':r[3],
                         'raw_bytes':r[4],'blosc_ratio':r[5],'min':r[6],'max':r[7],'nonfinite':r[8]} for r in rows}},
          open('fixture/manifest.json','w'), indent=2)
tot_c=sum(r[3] for r in rows); tot_r=sum(r[4] for r in rows)
print(f"\nTOTAL compressed={tot_c:,}B raw={tot_r:,}B  blosc mean ratio={tot_r/tot_c:.4f}x")
