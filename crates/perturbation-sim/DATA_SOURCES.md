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
| [Awesome-Electrical-Grid-Mapping (open-energy-transition)](https://github.com/open-energy-transition/Awesome-Electrical-Grid-Mapping) | curated index of grid datasets/tools | links | open | meta-source — start here when adding a new country/region feed |
| [i-DE (Iberdrola DSO) hosting-capacity map, 11 Apr 2025](https://www.i-de.es/en/grid-connection/energy-generation/capacity-map) · [PDF](https://www.i-de.es/documents/2803098/3162312/MapaDeCapacidad_iDE_11_Abril_2025.pdf/d07461a2-ee12-f2e2-e56b-55a777c031b8) | **distribution layer**: per-substation (*Umspannwerk*) available generation-connection capacity | PDF/map | public | the **DSO complement** to the REE/PyPSA transmission layer — finally per-substation granularity. Saturated substations (≈0 headroom) = candidate stressed nodes; ties to `AgeModel`. **Dated 17 days pre-blackout** → near-contemporaneous grid-state snapshot for an "as-of 28 Apr 2025" study. Map-style PDF → OCR/parse to a substation→capacity table |

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
| [REE — Auditoría Calidad de Servicio RdT 2024](https://www.ree.es/sites/default/files/datos/transporte/Informe_Auditoria_Calidad_de_Servicio_RdT_SEE_2024.pdf) | **transmission quality-of-service**: availability, energy-not-supplied (ENS), interruption indices per element/region | PDF | public | **the empirical condition/reliability layer.** Lifts the tech-debt modifier from a topology *proxy* (`AgeModel::DensityProxy`) to *measured* per-region condition → legitimate `DataLevel::WithHeterogeneousAssets`. Low availability / high ENS = older/weaker asset → `newness ↓`. Parse PDF → per-region condition index → `ModernizationSpend`-style `newness` vector |

## 4. Renewable + storage context (→ the solar/wind layer; future storage hook)

| Source | Feeds | Note |
|---|---|---|
| [Climate17 — Spain grid infrastructure](https://www.climate17.com/blog/spain-renewable-power-puzzle-strengthening-grid-infrastructure) | renewable-integration / grid-strengthening narrative | context for the solar/wind feasibility doc (`ada-docs/research/SOLAR_WIND_PEAK_PREDICTION_FEASIBILITY.md`) |
| [Energy-Storage.news — NECP 22.5 GW by 2030](https://www.energy-storage.news/spain-increases-energy-storage-target-in-necp-to-22-5gw-by-2030/) | storage targets per the NECP | implies the **storage hook** below |

## 5. Ground truth — the observed footprints (→ validation, §5 of METHODS)

| Source | Footprint | Note |
|---|---|---|
| [ENTSO-E expert-panel final report (2026-03-20)](https://www.entsoe.eu/news/2026/03/20/entso-e-publishes-expert-panel-final-report-on-28-april-2025-blackout-in-spain-and-portugal/) · [publications](https://www.entsoe.eu/publications/blackout/28-april-2025-iberian-blackout/) · [full PDF (50 MB)](https://eepublicdownloads.blob.core.windows.net/public-cdn-container/clean-documents/Publications/2025/iberian-blackout/Final%20Report%20on%20the%20Grid%20Incident%20in%20Spain%20and%20Portugal%20on%2028%20April%202025.pdf) | **electrical mechanism** | the authoritative reconstruction: **overvoltage-driven cascading *generation* disconnection** + oscillations + reactive-power/voltage-control gaps + uneven stabilisation — a **voltage-collapse** event, *not* a line-overload cascade. Reconstructed sequence below |
| [Eurosurveillance 30/26 (2500405)](https://www.eurosurveillance.org/content/10.2807/1560-7917.ES.2025.30.26.2500405) · [PMC12231376](https://pmc.ncbi.nlm.nih.gov/articles/PMC12231376/) | **human footprint** | MoMo excess-mortality surveillance: **147 excess deaths over 3 days** (95% CI −35..330, +4.2%); 65–84 yr **+7.9% = 94 deaths** (CI 63..125); ~10 directly attributed. Per-region severity target, not mechanism |

### ⚠ Validation caveat — read before claiming the model "explains" the blackout
The expert panel is explicit: the 28 Apr 2025 event was **voltage/reactive driven**
(AC mode). `perturbation-sim`'s DC cascade is the **line-overload** mechanism —
**not** what triggered this blackout. What legitimately transfers:

- **The structural / field tier is mechanism-agnostic and stays relevant:**
  Weyl/Davis–Kahan (`perturbation.rs`), Cheeger + Fiedler + Kron (`basin.rs`)
  measure *where the grid is structurally weak* and *how any perturbation would
  propagate through its connectivity* — a vulnerability screen independent of
  whether the trigger is overload or overvoltage. The Go-meta Raumgewinn side
  (global connectivity collapse) is the right lens for a *system-wide* event
  like this one.
- **The DC overload cascade does NOT reproduce the Iberian sequence.** Do not
  claim it does. The voltage/reactive trigger needs the **AC fork** (METHODS §8:
  full π-model `R+jX+jB/2`, voltages, reactive Q) — the rung that unlocks the
  voltage-collapse mode. This event is the concrete justification for climbing it.
- **The human footprint validates *severity*, any mechanism:** correlate a
  model's predicted regional impact against the per-autonomous-community excess
  mortality (the §5 ICC/Pearson/Spearman battery, Jirak-significant). Ties to the
  workspace's public-health surface (medcare-rs / MoMo-style surveillance).

So: use the field tier as an honest **structural-vulnerability screen**; treat
the DC cascade as one mechanism among several; reach for the AC fork to model
the actual voltage-collapse trigger; validate severity against the mortality
footprint. Over-claiming "we model the Iberian blackout" with the DC path alone
would be exactly the dilution this crate's METHODS doc guards against.

#### Reconstructed sequence (from the full report — the concrete validation target)
All times 28 Apr 2025 CEST; numbers quoted from the report:
- **Pre-conditions:** high RES, exports ≈ **5 GW**; two damped oscillation events
  earlier — a local mode at **0.63 Hz** (~12:03) and an East-Centre inter-area
  mode at **0.2 Hz** (12:19–12:22). Operators damped both.
- **12:32:00:** system voltage begins rising across many nodes (PMU data). From
  12:32:00–12:32:48, distribution loss ≈ **317 MW**, >5 MW generators down
  ≈ **500 MW**, ≈ **208 MW** distributed wind lost.
- **12:33:16–18:** the overvoltage-protection cascade — chunks of ≈ **525**,
  ≈ **727**, ≈ **928**, **355 MW** trip (e.g. a 220 kV overvoltage protection on
  a transformer injecting 355 MW; 727 MW PV+thermosolar lost at 12:33:16.820 to
  overvoltage protection).
- **12:33:20.473:** AC link to Morocco trips (underfrequency).
- **12:33:21.535:** France–Spain AC lines disconnect by protection (loss of
  synchronism) — **this is the connectivity cut that isolates Iberia from
  Continental Europe**.
- **12:33:23.960:** HVDC Spain→France trips → full electrical separation; the
  Iberian system collapses. Frequency falls toward ~**48 Hz**; RoCoF within
  ±1 Hz/s until 12:33:20.56.

**Mechanism → model mapping (the honest split):**
- The **trigger** (overvoltage → cascading generation trips, ~24 s electro-
  mechanical) is **AC/reactive** → only the **AC fork** (METHODS §8) can
  reproduce it; the DC overload cascade cannot.
- The **separation** (France–Spain AC + HVDC disconnect isolating Iberia) **is a
  connectivity cut / islanding** → exactly what the **field tier** describes:
  Cheeger min-cut, Fiedler `λ₂` collapse, Kron cross-border reduction. So the
  field tier legitimately models the *separation geometry* even though it did
  not cause the *trigger*. Use it to screen "which cut isolates Iberia"; use the
  AC fork for "why the voltage ran away."

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
