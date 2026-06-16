# Harvesting grid topology & state into perturbation-sim

How to feed real grid data into `perturbation-sim`, for any source or country.
The loader (`ingest::from_pypsa_csv`) is **schema-flexible** (column-alias based)
and **zero-dep**, so adapting a new source is usually just "name the columns it
already understands, or add an alias."

---

## Tier 1 — Topology + electrical parameters (what the simulator eats)

### Primary: PyPSA-Eur / OSM prebuilt network (recommended)

- **Source:** Zenodo record [`10.5281/zenodo.13358976`](https://zenodo.org/records/13358976), v0.3 (Aug 2024).
- **License:** ODbL v1.0 (attribute OpenStreetMap contributors; share-alike on redistributed derivatives).
- **Coverage:** 35 European countries incl. ES, FR, PT, DE, IT… AC 220–750 kV + all DC.
- **Format:** plain CSV (`buses.csv`, `lines.csv`, `transformers.csv`, `converters.csv`, `links.csv`).

```sh
mkdir -p /tmp/pypsa && cd /tmp/pypsa
# Either the canonical Zenodo source …
curl -L -o buses.csv 'https://zenodo.org/records/13358976/files/buses.csv?download=1'
curl -L -o lines.csv 'https://zenodo.org/records/13358976/files/lines.csv?download=1'
# … or our ODbL release mirror (same files, kept out of git):
#   R=https://github.com/AdaWorldAPI/lance-graph/releases/download/perturbation-sim-data-v0.1
#   curl -L -o buses.csv $R/buses.csv ; curl -L -o lines.csv $R/lines.csv
cargo run --manifest-path crates/perturbation-sim/Cargo.toml --example iberian -- \
    /tmp/pypsa/buses.csv /tmp/pypsa/lines.csv ES        # ← any ISO-2 code: FR, PT, DE, IT…
```

**Verified real schema (v0.3):**
- `buses.csv`: `bus_id, voltage, dc, symbol, under_construction, x, y, country, geometry`
  → `bus_id` = id, `x`/`y` = lon/lat, `country` = ISO-2 filter.
- `lines.csv`: `line_id, bus0, bus1, voltage, circuits, length, underground, under_construction, geometry`
  → **no reactance, no `s_nom`** — the base network is topology-only; the loader
  estimates both (see Parameter Derivation). `voltage` is in **kV**; `length` is
  in **metres**.

**Two real-data gotchas the loader/examples already handle:**
1. **`length` is in metres**, not km. This is a *uniform* scale on every line, so
   `b = 1/x` scales uniformly and the DC flow distribution — hence the
   perturbation *shape* — is invariant. (It would only matter for absolute MW,
   which needs per-unit + real injections anyway.)
2. **A country filter fragments the grid** (cross-border ties dropped, OSM gaps).
   The raw ES extract is 705 buses / 903 lines but in **25+ disjoint islands**
   with `λ₂ ≈ 0` before any trip. Always call
   `PypsaImport::largest_component()` (the `iberian` example does) — for ES that
   yields a 261-bus / 348-line connected core where a single trip cascades to
   ~20% of lines. For a *cross-border* study, don't filter by country: import all
   buses (`country = None`) and keep the interconnector lines.

### Alternatives

| Source | What you get | Notes |
|---|---|---|
| **Raw OSM via Overpass** (`power=line/cable/substation`) | Geometry + voltage + (sometimes) circuits | ODbL. You estimate `x`/`s_nom` yourself — same derivation the loader uses. Aggregate substations within ~5 km to single buses. |
| **GridKit / SciGRID_power** | Buses + links + transformers + geo + electrical metadata, digitized from the ENTSO-E map | The classic pre-PyPSA route; richer metadata than raw OSM. |
| **PyPSA-Eur full workflow** | The same network *after* `add_electrical_parameters` → real `x`, `r`, `s_nom` per standard line type | Run the upstream snakemake workflow if you want PyPSA's own estimated parameters instead of ours. The loader will use the `x`/`s_nom` columns automatically if present. |

---

## Tier 2 — Live state (injections + the observed footprint to validate against)

The topology CSV has **no generation/load** — injections `p` come from here.

| Source | Use | Access |
|---|---|---|
| **ENTSO-E Transparency Platform** ([API](https://transparency.entsoe.eu/)) | Per-zone load, generation, cross-border flows, **outages** (the observed footprint to correlate against) | Free RESTful API, register for a token (EU reg 543/2013). |
| **REE REData / ESIOS** (Spain) ([apidata](https://www.ree.es/en/datos/apidata)) | Real-time Spain demand/generation/exchange/transmission | Free token by email. The Apr-2025 blackout ground truth lives here. |
| **Electricity Maps** ([docs](https://app.electricitymaps.com/docs)) | Flow-traced cross-border flows | Commercial API; the [`electricitymaps-contrib`](https://github.com/electricitymaps/electricitymaps-contrib) parsers (open) point at the same official TSO sources. |

**Wiring injections:** zonal generation−load is published per bidding zone, not per
bus. Distribute a zone's net injection across its buses (e.g. proportional to
local installed capacity / population, or uniformly) to get a per-bus balanced
`p` (∑p = 0). This is the single biggest modelling choice; document it.

---

## The column-alias contract

`from_pypsa_csv` resolves columns case-insensitively by these aliases. To adapt a
new source, rename its columns to any alias — or add one to `ingest.rs`:

| Field | Aliases recognized | Meaning |
|---|---|---|
| bus id | `bus_id`, `name`, `station_id`, `bus` | node key |
| bus lon | `x`, `lon`, `longitude` | for the shape map |
| bus lat | `y`, `lat`, `latitude` | |
| country | `country`, `country_code` | ISO-2 filter |
| line ends | `bus0`/`bus_0`/`from`, `bus1`/`bus_1`/`to` | endpoints (must match a bus id) |
| reactance | `x`, `reactance`, `x_pu` | `b = 1/x`; absent → estimated |
| thermal limit | `s_nom`, `s_nom_mva`, `rating`, `thermal_limit` | line `limit`; absent → estimated |
| voltage | `v_nom`, `voltage`, `vn` | kV; for limit estimate |
| length | `length`, `length_km` | for reactance estimate |
| circuits | `circuits`, `num_parallel` | parallel circuits |

Note the deliberate per-file meaning of `x`: in **buses** it's longitude; in
**lines** it's reactance. The loader keys off the file, not the column name.

---

## Parameter derivation (when the source lacks them) — and the honesty line

- **Reactance absent** → `x = X_PER_KM · length / circuits` (`X_PER_KM = 0.33 Ω/km`,
  the PyPSA-Eur standard-line regime), then `b = 1/x`.
- **`s_nom` absent** → coarse thermal rating by voltage class
  (`estimate_snom_mva`) × circuits.
- `PypsaImport` reports `n_estimated_reactance` / `n_estimated_limit` so you can
  **disclose** how much is estimated. OSM carries no measured reactance or
  as-built thermal/protection settings — estimated values are an engineering
  proxy fit for **DC contingency screening**, not utility protection data.

---

## The validation loop (predicted shape vs observed footprint)

1. Import topology → `largest_component()` → balanced injections from
   ENTSO-E/ESIOS.
2. `simulate_outage(seed = the line that actually tripped)` → `shape.node_field`.
3. Pull the **observed** outage footprint (which buses lost supply) from the
   ENTSO-E outage feed / news reconstruction.
4. Correlate predicted `node_field` vs observed footprint with
   `ndarray::hpc::reliability` — **Pearson / Spearman / ICC(2,1)**.
5. **Significance:** grid telemetry is autocorrelated (weakly dependent), so use
   the **Jirak 2016** rate `n^(p/2−1)` (arXiv:1606.01617), *not* classical IID
   Berry–Esseen. See `ada-docs/research/JIRAK_MATH_THEOREMS_HARVEST.md` §3.

---

## Per-country quick recipe

```sh
# Same two CSVs, just change the ISO-2 code. Larger countries → larger core.
for CC in ES FR PT DE IT; do
  cargo run --manifest-path crates/perturbation-sim/Cargo.toml --example iberian -- \
      /tmp/pypsa/buses.csv /tmp/pypsa/lines.csv "$CC"
done
# Cross-border (Iberian peninsula incl. interconnectors): import with country=None
# in your own driver (call from_pypsa_csv(.., None)) and keep ES+PT+FR buses.
```

*Optional follow-ups (not yet built): a GeoJSON/Overpass adapter, an
ENTSO-E/ESIOS injection + observed-footprint fetcher, and a per-unit
normalization pass so flows are in MW against the real `s_nom` limits.*
