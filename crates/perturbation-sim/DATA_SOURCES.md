# DATA_SOURCES — provenance, by model layer

Every external source for `perturbation-sim`, organized by the layer it feeds,
with format / openness / extraction reality. Keeps provenance in one place
(anti-dilution: one ledger, not URLs scattered across commits). Honesty column
flags what is actually usable today vs needs extraction.

## 1. Topology + electrical parameters (→ `ingest.rs`, the `Grid`)

| Source | Feeds | Format | Open | Note |
|---|---|---|---|---|
| [Zenodo 13358976 — PyPSA-Eur/OSM prebuilt network](https://zenodo.org/records/13358976) | buses+lines, 35 ctry incl. ES | CSV | ODbL | **primary.** v0.3 is topology + voltage + circuits + length only → `r`/`x`/`s_nom` estimated (see `HARVESTING.md`) |
| [Nature SciData 2025](https://www.nature.com/articles/s41597-025-04550-7) · [arXiv 2408.17178](https://arxiv.org/pdf/2408.17178) | the method | paper | open | how the OSM grid was built (5 km substation aggregation) |
| [PyPSA-Eur](https://github.com/PyPSA/pypsa-eur/) | full workflow → real `x,r,s_nom` | code | open | run upstream if you want PyPSA's own line parameters |
| [OSM Power networks](https://wiki.openstreetmap.org/wiki/Power_networks) (Overpass) | raw lines/substations | XML/JSON | ODbL | full control; you estimate electrical params |
| [openmod datasets (GridKit/SciGRID)](https://wiki.openmod-initiative.org/wiki/Transmission_network_datasets) | ENTSO-E-map-derived grid | CSV | open | richer electrical metadata than raw OSM |

## 2. Live state — injections `p` + the observed footprint (→ validation, §5)

| Source | Feeds | Format | Open | Note |
|---|---|---|---|---|
| [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/) | load, generation, flows, **outages** | REST API | free token | the observed-footprint feed to correlate against |
| [REE apidata / ESIOS](https://www.ree.es/en/datos/apidata) | Spain demand/gen/exchange | REST API | free token (email) | the Apr-2025 blackout ground truth |
| [Electricity Maps](https://app.electricitymaps.com/docs) · [contrib](https://github.com/electricitymaps/electricitymaps-contrib) | flow-traced flows | API | commercial / parsers open | underlying sources are the official TSOs |

## 3. Modernization / spend (→ `model.rs` `AgeModel::ModernizationSpend`)

| Source | Feeds | Format | Open | Note |
|---|---|---|---|---|
| [planificacionelectrica.es — current planning](https://www.planificacionelectrica.es/en/current-planning) | ~260 transmission projects (2021-26; 2025-30 in progress): codes, voltage 66-400 kV, geo connecting points | **PDF** | public | no GIS/Excel → parse + geo-match projects to PyPSA buses to build per-bus newness |
| [MITECO — electricidad](https://www.miteco.gob.es/es/energia/energia-electrica/electricidad.html) | ministry electricity portal | HTML/PDF | public | policy/spend context |
| [MITECO — planificación electricidad/gas](https://www.miteco.gob.es/es/energia/estrategia-normativa/planificacion/planificacion-electricidad-gas.html) | national planning | HTML/PDF | public | the planning-process root |
| [REE — Informe del Sistema Eléctrico 2024 (ISE_2024.pdf)](https://www.sistemaelectrico-ree.es/sites/default/files/2025-03/ISE_2024.pdf) | realized capacity, grid additions, regional stats | PDF | public | the realized-state companion to the forward plan |

## 4. Renewable + storage context (→ the solar/wind layer; future storage hook)

| Source | Feeds | Note |
|---|---|---|
| [Climate17 — Spain grid infrastructure](https://www.climate17.com/blog/spain-renewable-power-puzzle-strengthening-grid-infrastructure) | renewable-integration / grid-strengthening narrative | context for the solar/wind feasibility doc (`ada-docs/research/SOLAR_WIND_PEAK_PREDICTION_FEASIBILITY.md`) |
| [Energy-Storage.news — NECP 22.5 GW by 2030](https://www.energy-storage.news/spain-increases-energy-storage-target-in-necp-to-22-5gw-by-2030/) | storage targets per the NECP | implies the **storage hook** below |

### The storage hook (genuine new modeling axis — not yet built)
Storage (NECP target 22.5 GW by 2030) is a **controllable injection**: a battery
at bus *i* is a dispatchable `p_i` the operator sets to relieve a contingency.
In `perturbation-sim` terms this is a **Pearl rung-2 intervention** on the
injection vector — `do(p_i += discharge)` — and the counterfactual question
becomes "with storage at bus *X*, does the cascade still propagate?" It slots
onto the existing cascade with no field-tier change (it only reshapes `p`, the
same way the solar/wind net-load does), and ties the renewable-ramp decision
(solar/wind doc) to the grid-risk check (this crate). A future `intervention.rs`
would expose `simulate_with_storage(grid, p, storage_buses, dispatch)` returning
the contingency regime *with vs without* the dispatch — the value of storage as
collapse-prevention, quantified.

---

*Provenance discipline: new sources append here, tagged by layer. Extraction
status (raw-usable vs PDF-needs-parsing) is stated so no layer over-claims data
it cannot yet read — same honesty as `ingest.rs`'s `n_estimated_*` counters.*
