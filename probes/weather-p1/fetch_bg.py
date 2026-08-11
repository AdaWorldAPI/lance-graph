import json, urllib.request, numpy as np, numcodecs, os, time
B="https://storage.googleapis.com/gcp-public-data-arco-era5/ar/1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr"
meta=json.load(open('zmeta.json'))['metadata']; t=547476
os.makedirs('fixture',exist_ok=True)
out={}
for var in ['2m_dewpoint_temperature','10m_u_component_of_wind','10m_v_component_of_wind',
            'surface_pressure','total_column_water_vapour','total_cloud_cover']:
    if os.path.exists(f'fixture/{var}.npy'): print('skip',var,flush=True); continue
    za=meta[f'{var}/.zarray']
    for attempt in range(3):
        try:
            t0=time.time()
            raw=urllib.request.urlopen(f'{B}/{var}/{t}.0.0', timeout=600).read()
            dec=numcodecs.get_codec(za['compressor']).decode(raw)
            a=np.frombuffer(dec,dtype=za['dtype']).reshape(za['chunks'])[0]
            np.save(f'fixture/{var}.npy', a)
            out[var]={'compressed_bytes':len(raw),'raw_bytes':int(a.nbytes),
                      'blosc_ratio':a.nbytes/len(raw),'secs':round(time.time()-t0,1),
                      'nonfinite':int((~np.isfinite(a)).sum())}
            print(f"OK {var} comp={len(raw):,} ratio={a.nbytes/len(raw):.4f} {time.time()-t0:.0f}s",flush=True)
            break
        except Exception as e:
            print(f"try{attempt} {var}: {type(e).__name__} {e}",flush=True)
            time.sleep(5)
    else:
        out[var]={'error':'unavailable after 3 tries'}
json.dump(out, open('fixture/fetch_log.json','w'), indent=2)
print("DONE",flush=True)
