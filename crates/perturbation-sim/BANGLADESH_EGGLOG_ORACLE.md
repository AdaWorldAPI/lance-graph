# Bangladesh outage: dual-mechanism reference oracle and cold mathematical compiler

**Status:** bounded offline scalar oracle plus compile-contract proposal. No egglog runtime integration, R2IL executable, measured accuracy, or Bangladesh operational claim is implied by this note. It is deliberately not a second physical engine.

## Source and provenance

- [Bangladesh-outage](https://github.com/AdaWorldAPI/Bangladesh-outage) is a historical telecom tower power-outage predictor, with synthetic A10H outage-generation / telemetry scripts, data preparation, FNN/LSTM/TCN training, and a Word project plan. The executable Python generators and pipeline are **reference behaviour for what the legacy system actually did**, not an independent causal ground-truth dataset. The Word project plan was not machine-inspected in this change.
- [`OutageOracle/MVP1/A10H/010_Generate_Raw_Outage_Logs.py`](https://github.com/AdaWorldAPI/Bangladesh-outage/blob/main/OutageOracle/MVP1/A10H/010_Generate_Raw_Outage_Logs.py): synthetic events `grid_failure | voltage_drop | maintenance`; tower `power_source` metadata and grid groups; shared weather and synthetic grid cascades. No independent telecom-path failure mechanism is generated.
- [`020_Generate_Raw_Telemetry.py`](https://github.com/AdaWorldAPI/Bangladesh-outage/blob/main/OutageOracle/MVP1/A10H/020_Generate_Raw_Telemetry.py): sets `grid_status = 'off' if is_off else 'on'` where `is_off` means a tower was inside a *synthetic outage interval*. **It is not an independent measurement of the electricity grid.** Likewise 0 V/0 A are generated from that same label. Never train/validate a causal classifier on these as independent evidence of the root cause.
- [`040_Merge_And_Aggregate.py`](https://github.com/AdaWorldAPI/Bangladesh-outage/blob/main/OutageOracle/MVP1/A10H/040_Merge_And_Aggregate.py): aggregates tower-level rows into region/day outage counts and durations, losing individual site/incident/root-cause identity.
- [`050_ExtendFeatures.py`](https://github.com/AdaWorldAPI/Bangladesh-outage/blob/main/OutageOracle/MVP1/A10H/050_ExtendFeatures.py): creates `target_outage_15d` before training; [`070_Train.py`](https://github.com/AdaWorldAPI/Bangladesh-outage/blob/main/OutageOracle/MVP1/A10H/070_Train.py) selects all numeric features except four basic columns, **not** `target_outage_15d`. That permits target leakage into its historical sequences. [`060_PreProcessScaling.py`](https://github.com/AdaWorldAPI/Bangladesh-outage/blob/main/OutageOracle/MVP1/A10H/060_PreProcessScaling.py) fits the scaler before the chronological train/validation split; region-level historical averages also use the full series. `070_Train.py` has sigmoid occurrence heads passed to `BCEWithLogitsLoss`, and fits its logistic calibrator on the same validation set later scored. These all need an independently verified walk-forward baseline before reported prediction accuracy has evidentiary force.
- [PR #1689](https://github.com/AdaWorldAPI/lance-graph/pull/1689): egglog cold functional database/semi-naive/equality saturation, eggcc effect-safe extraction, 3LA source-to-target validation. The PR is research-only, not an integrated compiler.
- [`perturbation-sim`](README.md): `simulate_outage` computes DC power-flow line-trip cascades and spectral perturbation **on a bus/line electrical network**. It does not model electrical distribution to a tower, telecom backhaul failures, battery runtime, or measured tower availability. Its `node_field`, `tripped` and `islanded` outputs are **grid risk witnesses**, not telecom labels.

## Two axes, not one Boolean

1. **Grid failure:** physical grid supply is absent at the tower connection. Independent recorded grid supply is distinct from generator/battery supply. A grid line trip or high Laplacian perturbation alone does not establish the site's grid outage.
2. **Telecom-path failure:** radio hardware, tower subsystem, upstream backhaul or other explicitly observed non-power path is unavailable. Even perfect grid power need not restore service.
3. **Coupling:** under the expressly simplified Boolean availability contract, `site_power = grid_supply OR backup_supply`, `tower_service = site_power AND telecom_path`. An upstream backhaul supply dependency, delayed battery depletion, maintenance, multi-feed switching and uncertain observations require more state or refusal, **not** a Boolean reinterpretation of the historical `grid_status` column.

A grid incident may leave the tower online (backup works). An independent telecom-path failure may leave the electricity grid fully healthy. Both can occur at once. Neither can be established from a tower-down observation alone.

The executable independent **scalar reference** is [`src/outage_oracle.rs`](src/outage_oracle.rs): `Observed::{Up,Down,Unknown}` with strong-Kleene `and/or/not`, `assess_site` with separate `grid_supply_outage`, `power_unavailable`, `telecom_path_failure`, predicted service, non-exclusive mechanism and contradiction/refusal. `assess_bound` checks globally stable tower GUID, common window and independently pinned source generations. It deliberately has no simulation dependency so compiler and reference cannot share an implementation bug.

## Compiler split (proposal, NOT shipped)

```text
historical Bangladesh data (synthetic labels stay marked synthetic)
            + separately measured grid/tower/backup observations
       -> time align, resolve tower↔electrical bus mapping and version-pin
       -> physical perturbation-sim grid risk / line-trip evidence [separate]
       -> typed site-availability equation + witness provenance
       -> egglog: COLD pure guarded algebra / shared-DAG search
       -> eggcc-style extractor refuses cross-version or effect reordering
       -> typed R2IL FoldDialect procedure [when actually admitted]
       -> Quack -> mask-risc execute_into [existing hot executor]
       -> **independent** outage_oracle::assess_bound differential
       -> source-age, authorization and actual write ack [separate effects]
```

**Possible legal pure rewrites:** `x AND x → x`; `x OR x → x`; De Morgan; guarded distributivity; common-subexpression reuse for the *same* focus, generation, truth domain and source read. Scalar `outage_oracle` tests enumerate all 27 K3 triples. A rewrite certificate only proves an expression identity in its declared truth domain, not a root-cause claim.

**Forbidden simplifications:** `grid_down == tower_down`; `tower_down ⇒ grid_failure`; `line_tripped ⇒ site_unpowered`; `Unknown == Down`; `grid_synthetic_status == independently_observed_supply`; `bag_COUNT == boolean_EXISTS`; `cross-row ordinal match == tower identity`; `eq_saturation == source freshness/committed write`. No new hot executor or positional cross-table gathers.

For each admitted extraction pin `source_revision`, `tower_guid`, `time_window`, source-generation tuple, input truth domain, output shape, multiplicity, effect-order witness and reference-case ID. `BIND`, external `READ`, `EMIT`, and `COMMIT` are effect-bearing. Numeric/spectral outputs must keep units and floating-point tolerance; K3 Boolean truth-tables do not justify numeric reassociation. Cost measurements must include compilation, binding, reads, execution, replay/validation and dirty/commit, amortized over repeated frames.

## Minimum acceptance battery

| ID | Input | Mandatory invariant |
|---|---|---|
| BD1 | grid down; backup up; telecom up | grid outage is observed, tower remains up; **not a telecom outage** |
| BD2 | grid down; backup down; telecom healthy | `GridPower` mechanism, tower down |
| BD3 | grid up; telecom down | `TelecomPath` mechanism; **not grid outage** |
| BD4 | grid and telecom down | `Both`; do not merge causes or erase either lineage |
| BD5 | grid down; backup unknown; telecom healthy | tower status and mechanism `Unresolved`, never default to DOWN |
| BD6 | healthy inputs; separately observed tower down | independent oracle returns `Contradiction`, not a fitted explanation |
| BD7 | same source-local ordinal, different tower GUID; reused stale generation | bind refuses `CrossFocus` or `StaleSource`, even if expression is algebraically legal |
| BD8 | grid `simulate_outage` changes spectral/line-trip fields only | cannot assert tower service, and may not change a direct observed telecom cause |
| BD9 | legacy A10H label reused as grid evidence | reject as circular; oracle requires independent input or an explicitly marked synthetic scenario |
| BD10 | compare optimized R2IL on separate held-out windows | final per-tower K3 tuple, mechanism, duration and site/day multiplicity agree with independent scalar; no source leakage |

`BD1`–`BD7` and K3 algebra have Rust unit tests in `outage_oracle.rs`. `BD8`–`BD10` are **unimplemented integration gates**, not claimed passing tests. No physical-mechanism learning, site↔bus registry, egglog extraction, observed Bangladesh calibration or R2IL differential is present in this patch.

## Sequenced next changes

1. **Fix the original data boundary**, preferably in a separate Bangladesh-outage PR: add independent `grid_supply_status`, `backup_status`, `telecom_path_status`, `observed_tower_service`, cause evidence and source timestamps. Preserve legacy generated `grid_status` under an unmistakably synthetic label. Use stable tower IDs and event IDs. Maintain grid-group to feeder/bus mappings separately; do not infer them from region adjacency.
2. Separate grid-event/tower-event simulation and apply generator/battery behaviour before generating service outages. Enumerate mutually independent faults and a shared-weather counterfactual. Remove `random.choices(..., k=...)` duplicate tower incidents unless explicitly modeled as separate overlapping events, and handle empty same-grid cascade candidate sets.
3. Rebuild label-safe, time-safe training: *no* `target_outage_15d` or other future-derived columns in features; historical rolling stats/encoders/scaling fitted on training windows only; walk-forward temporal evaluation and separated calibration. Use raw logits with `BCEWithLogitsLoss` or sigmoid with `BCELoss`, not both; enforce event and tower-group holdouts and class-specific diagnostics.
4. Only then attempt one bounded egglog 3LA-style offline extraction with independent scalar oracle on the first pure K3 site equation. Profile end-to-end against unoptimized Quack/mask-risc. Keep grid spectral numerics as separate operands with their own bounds and units, not interchangeable Boolean evidence.