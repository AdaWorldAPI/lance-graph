# P1 — the weather-substrate measurements, made re-runnable

> **Status: RUN, 2026-08-11.** Results in `p1_results_temperature.json`.
> **Two doc claims were FALSIFIED by this run** — see §3 and
> `.claude/knowledge/weather-normalized-substrate.md` §12.
>
> Probe P1 of the queue in that doc §10. Python by design: the plan's D-WXA-1
> rules Stage-A ingest **disposable** (`Zarr → numpy → f32 slab`, no eccodes,
> no C deps). These scripts are the ingest half; nothing here is a Rust
> deliverable.

## Why this exists

The knowledge doc's §6 evidence ledger carried numbers that existed **only in a
session transcript**. Under the falsifiability rule (`CLAUDE.md` P0) that is a
claim, not a finding. This directory makes them re-derivable by command — and
the first honest run promptly contradicted two of them.

## Reproduce

```bash
pip install numpy zarr numcodecs scipy
cd probes/weather-p1
python3 fetch.py        # consolidated metadata + 2m_temperature chunk
python3 fetch_bg.py     # remaining variables, with retries
python3 p1_probe.py     # the measurements
python3 verify_apparatus.py   # apparatus self-check + the mechanism diagnosis
```

`fetch.py` needs `zmeta.json` first:

```bash
curl -s --noproxy '*' -o zmeta.json \
  "https://storage.googleapis.com/gcp-public-data-arco-era5/ar/1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr/.zmetadata"
```

`--noproxy '*'` is required in this environment (the proxy 403s/stalls on GCS;
plain `curl` through it does not reach the store).

## 1. Source, verified rather than assumed

- **Store** `gs://gcp-public-data-arco-era5/ar/1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr`
  — HTTP 200, 52 arrays, consolidated `.zmetadata` 46,842 B. **This confirms the
  plan's ⊘ C3 correction**: the previously-cited
  `era5/…-1h-1440x721.zarr` does not exist.
- **Grid** `[561264, 721, 1440]`, dtype `<f4`, chunks `[1, 721, 1440]`,
  compressor blosc, `fill_value: NaN`, order C. One chunk = one global field =
  1,038,240 points = 4,152,960 B raw.
- **Time** `hours since 1959-01-01 00:00:00`, proleptic_gregorian. Fixture uses
  `t = 547476` = **2021-06-15 12:00 UTC**.
- **Sparse by design.** Several variables 404 at this timestep
  (`10m_v_component_of_wind`, `surface_pressure`, `total_column_water_vapour`,
  `total_cloud_cover`, `mean_sea_level_pressure`). In Zarr v2 a missing chunk
  means *all `fill_value`* — here all-NaN — so a 404 is valid store semantics,
  **not** a fetch bug. Any ingest must treat 404 as data, not as failure.

## 2. FINDING — ARCO-ERA5 has no `wind_speed` variable at all

Exhaustive: all 52 arrays enumerated; wind-related are exactly
`10m_u_component_of_wind`, `10m_v_component_of_wind`, `u_component_of_wind`,
`v_component_of_wind`. There is **no** `10m_wind_speed`.

Consequence: the doc's §6.5 measurement ("stored speed ≥ `hypot(ū,v̄)` at 100 %
of samples, mean ratio 1.115, max gap 14.37 m/s") **cannot have come from this
store**, and §5's "wind speed — NOT derivable" row is wrong *for the Phase-A
source the plan names*. The Jensen-gap physics (mean of magnitudes ≥ magnitude
of means) is still sound; the *measurement's provenance* is not. Re-attribute to
WeatherBench2 (which does publish derived surface fields) or drop it.

## 3. FINDING — Fisher-Z **hurts** on weather anomalies (the headline)

Real ERA5 `2m_temperature`, 1,038,240 gridpoints, climatology proxy = zonal
(per-latitude) mean, both paths quantized to the same 256 buckets over the same
0.4–99.6 percentile window (mirroring `RollingFloor::roll`):

| path | MAE (K) | p99 (K) | empty buckets | effective buckets | drift score |
|---|---|---|---|---|---|
| **linear** (no transform) | **0.0684** | 0.0939 | **0** | **115.7 / 256** | 607 |
| **Fisher-Z** then linear | 0.2168 | 0.4253 | **76** | **28.1 / 256** | 2240 |

Fisher-Z is **3.2× worse on error** and uses **28 of 256 buckets** instead of
116. (`effective_buckets` = `exp(entropy)` of the occupancy histogram; `drift
score` = `max|occ−μ|/σ` exactly as `quantize.rs:119-146`.)

**Mechanism — measured, not assumed.** `arctanh` is ≈identity near 0 and
explodes near ±1, so it redistributes resolution *toward the bounds*. That is
correct when mass piles at the bounds and wrong when mass sits in the middle:

| variable | mean\|x\| | frac \|x\|<0.25 | frac \|x\|>0.9 | kurtosis |
|---|---|---|---|---|
| ERA5 T anomaly / robust scale | 0.172 | **0.769** | 0.012 | +3.30 (peaked) |
| correlation-like control `tanh(N(0,1.5))` | 0.672 | 0.135 | **0.327** | — |

So the transform must **match the input distribution's shape**. Fisher-Z earns
its keep on correlation-like inputs (mass at ±1) and on helix's own rim radius
`r = √u`, which equal-area placement concentrates toward `r → 1` by construction
(`placement.rs:147-151`). A bell-shaped geophysical anomaly is the opposite case.

**This falsifies the doc's §1.2 claim** that variance stabilization is what makes
8 bits sufficient — for weather scalars, plain linear quantization over a robust
window is what makes 8 bits sufficient, and Fisher-Z actively degrades it.

**Consequence for the product claim.** "Fisher-Z everything into one comparable
substrate" does not survive. Cross-variable comparability must be re-derived
(probe P2) on a transform chosen per distribution shape — the shared *palette
and LUT* may still hold; the shared *transform* does not.

## 4. Other measured deltas from the doc's §6

- **§6.1 blosc ratio.** Measured per variable at this timestep:
  `2m_temperature` **1.794×**, `2m_dewpoint_temperature` **1.771×**,
  `10m_u_component_of_wind` **1.248×**. The doc's "1.27× mean" is within range
  for wind but not representative of temperature — the ratio is
  strongly variable-dependent, so a single scalar is the wrong summary.
- **§6.3 BF16 raw-vs-anomaly.** Reproduced in *direction*, not in magnitude:
  raw-Kelvin MAE **0.456 K** → anomaly MAE **0.00609 K** = **74.9×**
  (doc: 1.069 K → 0.0110 K = 97×). Different climatology (zonal-mean proxy here)
  makes this a *different* measurement, not a contradiction — but the doc's
  numbers remain unreproduced, so they stay `[H]` with their conditions
  unpinned.
- **Apparatus self-check passed**: the BF16 round-to-nearest-even helper is
  verified to ≤ 2⁻⁹ relative error before any BF16 claim is made
  (`verify_apparatus.py`).

## 5. Honest boundaries

- **One timestep, one season, one variable** for the headline. 2021-06-15 12:00
  UTC only. The linear-beats-Fisher-Z result should be re-run across seasons and
  on a second variable before it is graded `[G]` rather than `[H]`.
- **Climatology is a zonal-mean proxy**, not a real ERA5 climatology. It is the
  cheapest honest PLACE term and it is stated as such; a real climatology would
  shrink the anomaly further and likely *improve* both paths.
- **Saturation is real and unhidden**: 0.82 % of points land in the two rim
  buckets, giving a palette max error of 17.5 K against a 0.068 K mean. That is
  `quantize.rs`'s documented controlled-saturation tail, not a surprise — but any
  product claim must quote the tail, not just the mean.
- **Fixtures are not committed** (4 MB each, re-fetchable). Scripts + the exact
  store object + time index are, which is what makes the run reproducible.
- Nothing here touches Rust, the SoA lanes, or Lance. P1 is the ingest and
  measurement half only.
