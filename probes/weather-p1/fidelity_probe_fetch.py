"""D-WXS-7/D-WXS-8 fetch stage. Extends `fetch.py`'s proven pattern (real
ARCO-ERA5 Zarr v2 HTTP fetch, 404=NaN=valid-missing-chunk semantics) to
grid-scale scope: >=4 variables spanning >=2 distinct units, at each of >=3
distinct seasons, as bar B7 (`.claude/plans/weather-soa-bake-v1.md` W3)
requires.

The three (season, timestep, variable-set) triples below are NOT assumed --
they were found by a live HEAD-request sweep of this script's own store
(probes/weather-p1/README.md section 1's own finding: this store is sparse
by design, and NOT just at one timestep -- an initial attempt at 4 fixed
calendar-season anchors in 2021 found only the SAME 3 variables present at
EVERY one of them; total_column_water_vapour / total_cloud_cover /
sea_surface_temperature / mean_sea_level_pressure / surface_pressure /
10m_v_component_of_wind were absent at all 4. `10m_wind_speed` does not
exist in this store at all -- confirms the pre-existing FINDING in
`.claude/knowledge/weather-normalized-substrate.md` section 2. A follow-up
sweep of 24 candidate timesteps spread across the whole 1959-2023 archive
found these three, each independently satisfying >=4 variables / >=2 units,
in three genuinely different calendar seasons."""
import json, urllib.request, numpy as np, numcodecs, datetime as dt, os

B = "https://storage.googleapis.com/gcp-public-data-arco-era5/ar/1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr"
meta = json.load(open('zmeta.json'))['metadata']
EPOCH = dt.datetime(1959, 1, 1)

# (season_label, time_index, {variable: unit}) -- each triple confirmed
# present by a real HEAD sweep before this script existed (see module doc).
SEASONS = {
    'winter': (96000, {
        '2m_temperature': 'K', '2m_dewpoint_temperature': 'K',
        '10m_u_component_of_wind': 'm/s',
        'mean_sea_level_pressure': 'Pa', 'total_cloud_cover': '1',
    }),
    'spring': (98000, {
        '2m_temperature': 'K', '2m_dewpoint_temperature': 'K',
        '10m_u_component_of_wind': 'm/s',
        'total_column_water_vapour': 'kg/m2', 'sea_surface_temperature': 'K',
    }),
    'summer': (102000, {
        '2m_temperature': 'K',
        '10m_u_component_of_wind': 'm/s', '10m_v_component_of_wind': 'm/s',
        'total_cloud_cover': '1', 'sea_surface_temperature': 'K',
    }),
}


def get(var, t):
    """One (var, t) chunk. 404 -> all-fill (NaN here); valid Zarr v2
    semantics for a missing chunk, not a fetch failure (codex P2 on #920,
    documented in probes/weather-p1/README.md section 1)."""
    za = meta[f'{var}/.zarray']
    comp = za['compressor']
    key = f'{var}/{t}.0.0'
    req = urllib.request.Request(f'{B}/{key}')
    try:
        raw = urllib.request.urlopen(req, timeout=300).read()
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
        fill = za.get('fill_value')
        fill = np.nan if fill is None else fill
        a = np.full(za['chunks'][1:], fill, dtype=za['dtype'])
        return a, 0
    dec = numcodecs.get_codec(comp).decode(raw)
    a = np.frombuffer(dec, dtype=za['dtype']).reshape(za['chunks'])[0]
    return a, len(raw)


def main():
    os.makedirs('fixture', exist_ok=True)
    report = {}
    for season, (t, varmap) in SEASONS.items():
        target = (EPOCH + dt.timedelta(hours=t)).isoformat()
        os.makedirs(f'fixture/season_{season}', exist_ok=True)
        report[season] = {'target_utc': target, 'time_index': t, 'vars': {}}
        n_finite_units = set()
        for var, unit in varmap.items():
            a, ncomp = get(var, t)
            finite = np.isfinite(a)
            n_finite = int(finite.sum())
            usable = n_finite == a.size
            status = 'usable' if usable else ('partial' if n_finite > 0 else 'absent')
            report[season]['vars'][var] = {
                'unit': unit, 'compressed_bytes': ncomp, 'n_finite': n_finite,
                'total': int(a.size), 'status': status,
            }
            if usable:
                np.save(f'fixture/season_{season}/{var}.npy', a.astype(np.float32))
                n_finite_units.add(unit)
            print(f"{season:8s} {var:28s} unit={unit:6s} status={status:8s} "
                  f"n_finite={n_finite:>8,}/{a.size:>8,} compressed={ncomp:>9,}B")
        n_usable = sum(1 for v in report[season]['vars'].values() if v['status'] == 'usable')
        print(f"  -> {season}: {n_usable} usable vars, {len(n_finite_units)} distinct units "
              f"(bar B7 needs >=4 vars, >=2 units) "
              f"{'OK' if n_usable >= 4 and len(n_finite_units) >= 2 else 'SHORT'}\n")
    json.dump(report, open('fixture/fidelity_probe_availability.json', 'w'), indent=2)
    print("wrote fixture/fidelity_probe_availability.json")


if __name__ == '__main__':
    main()
