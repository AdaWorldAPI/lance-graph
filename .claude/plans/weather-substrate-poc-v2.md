# Weather Substrate POC — v2 (jc-gated: representation → hardware → prediction)

> **Status:** PLAN (2026-08-10). **Supersedes `weather-substrate-poc-v1.md`** (PR #914,
> merged) — v1's structure was "encoder bake-off"; v2 restructures around the
> operator's three-phase gate and corrects two factual errors in v1 (below).
> Doc-only. No code shipped. The POC runs in a separate session.
>
> **The operator's framing, verbatim in intent:** *find the reliability of the
> representation of the weather substrate using the lance-graph JC crate, then go
> into the prediction correctness using JC crate, and somewhere in between ndarray
> to make sure the hardware acceleration is doing it.*
>
> That is three phases with three different instruments, in a fixed order:
>
> | phase | question | instrument |
> |---|---|---|
> | **A** | is the *address* faithful to the field? | `jc` battery |
> | **B** | is the hardware actually doing the work? | `ndarray` + parity/throughput |
> | **C** | is the *forecast* faithful to what happened? | `jc` battery |
>
> **A and C are the same instrument on different pairs** — `corr(code_distance,
> field_distance)` vs `corr(predicted, observed)`. C therefore costs no new
> statistical machinery once A is wired. B sits between them because a Phase-C
> number measured on a silently-scalar path would be honest about forecasting and
> dishonest about the substrate.

---

## 0. Two corrections to v1 (stated before anything is built on them)

**C1 — GRIB2 is gone.** v1's `D-WX-0` specified `GRIB2 → flat f32 slab`, and v1 §6.3
argued at length that GRIB2 must never enter Rust. WeatherBench2 publishes ERA5 as
`era5/1959-2023_01_10-full_37-1h-1440x721.zarr` on public GCS — **1440 × 721 =
1,038,240 points**, exactly the target grid, hourly, 1959–2023, 37 levels, already
cloud-optimized. Ingest becomes `Zarr → xarray → numpy → f32 slab`: no `eccodes`, no
`gribberish`, no C dependency. **v1 §6.3 is moot rather than solved.** Consequence for
Phase C: history sizing goes from ~58k states (40 yr, 6-hourly) to **~570k** (65 yr,
hourly).

**C2 — `ecmwf-opendata` is Phase-C, not Phase-A.** It is the client for *live
operational* IFS/AIFS. Phase A runs on ERA5 **reanalysis**, which comes from
WeatherBench2. v1 put it on the critical path; it is not.

> **⊘ C3 — CORRECTION (2026-08-11, append-only): C1's Zarr object name is
> wrong.** `era5/1959-2023_01_10-full_37-1h-1440x721.zarr` does not exist; the
> session-verified object is
> **`1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr`** under the public
> ARCO-ERA5 bucket (`gcp-public-data-arco-era5`, `ar/` prefix). C1's grid math
> is unchanged (0.25° ⇒ 1440 × 721 = 1,038,240 points, hourly, 37 levels) —
> only the object name was invented. Apparatus lesson: the wrong name had been
> "confirmed" from a GitHub issue *title*; a title match is not an existence
> check — list the bucket. Full ingest spec now lives in
> `.claude/knowledge/weather-normalized-substrate.md` §8.

---

## 1. Ingest — disposable vs permanent (operator-directed split)

Ingest is **two different artifacts**, because the recurring-job requirement changes
its properties. Building the permanent one first is premature; discovering the
distinction later is expensive. The split:

| | **Stage-A ingest (disposable)** | **Stage-C ingest (permanent)** |
|---|---|---|
| corpus | ERA5 reanalysis — **static history, never re-ingested** | new cycles, continuous |
| trigger | manual, once | per init cycle |
| idempotency | irrelevant | **mandatory** |
| granularity | one slab | **per-cycle slab, append-only** |
| provenance | none | model · init time · revision |
| partial/late data | rerun | a *normal state* needing policy |
| lifetime | **thrown away after Phase A** | maintained |

**What is shared, and is the permanent decision made once:** the **512-byte row
stride** and the `soa:*` schema-metadata block (`envelope_layout_version`,
`row_stride`, `row_carving`, `endianness`, `classid`, `slab_digest`, `source`). Both
ingests emit byte-identical layout, so Stage-A slabs stay readable by the Stage-C
pipeline and only the driver script is discarded. The 512 B stride is *why* the rows
land verbatim — PR #907 measured that the stride clears Lance's 256 B mini-block
cutoff and takes the full-zip path, which ignores compression metadata entirely.

**Dataset versioning (operator ruling, 2026-08-10): ONE dataset, versions are
cycles.** Forecast and analysis lanes live in the **same** dataset — not separate
datasets joined on valid-time. One init cycle = one Lance version.

### ⚠ This is CONSUMED, not built — PRs #900–913 shipped it (operator, 2026-08-10)

The v1 plan and this section's first draft both read as if per-cycle versioning were
new machinery to design. **It is shipped and tested.** The weather POC writes *no*
versioning code; it calls the following:

| need | shipped surface |
|---|---|
| open a versioned graph (incl. remote) | `graph::versioned::VersionedGraph::{local, s3, azure, gcs}` |
| **time-travel read** | `VersionedGraph::at_version(v) -> Result<Dataset>` (`versioned.rs:428`) |
| current head | `VersionedGraph::current_version()` (`:435`) |
| commit a round | `VersionedGraph::commit_encounter_round(…)` (`:184`) |
| **diff two versions** | `GraphDiff` (`:70`), `GraphSealStatus` (`:54`) |
| one cycle = one version **writer** | `graph::cycle_sink::LanceCycleWriter` (#913, +1030 lines): `open`, `bootstrap`, `head() -> DatasetVersion`, `scan_image`, `reconcile_scans`; schema via `cycle_store_schema()` |
| **version-range read** (the ±window generalization) | `planner::temporal::QueryReference::at(v, rung)` + `deinterlace(rows, v_ref, deps)`; `LanceVersion = u64`, `EpistemicMode`, `TemporalStatus`, `classify` |
| per-mailbox trajectory across versions | `temporal::{local_trajectories, local_trajectory_of}` |

Two shipped tests already pin the exact invariant this plan depends on —
`a_whole_cycle_of_casts_is_one_wal_write_one_version` (`persist_sink.rs:961`) and
`p4a_drains_casts_and_seals_one_wal_write_one_version` (`cycle_driver.rs:1258`).
**Any weather deliverable that re-implements a version writer is the defect**, not
the feature.

Counterfactual forecasting and ensembles therefore need *no new primitive*:
fork-from-version-*k* is `at_version` + a divergent commit; ensemble spread is
branch divergence read through `deinterlace`.

### ⚠ S3 is the hydration path, NEVER the store (`.claude/knowledge/s3-hydration-lifecycle.md`, #901)

This plan's earlier framing — "slabs land on S3, read them back" — **collapses two
layers the doctrine separates.** The binding statement:

> *The object store hydrates; the local filesystem stores; the volume only decides
> whether hydration repeats. Three layers, one job each.*

For weather that means: **S3 is where a cycle's dataset comes FROM, and the process
opens it from a local mmap-capable directory.** Reading `s3://` as the runtime store
makes every read a network fetch into a fresh buffer — no mmap, no page cache — which
silently forfeits the zero-copy property the whole 512 B slab design exists to
deliver. `RAILWAY_VOL=/volume01` is therefore an **optimization on hydration
frequency**, never a correctness requirement; an ephemeral local path is correct and
merely re-hydrates.

The trap named in that doc applies directly to a Railway deployment: a network mount
(NFS/EFS/FUSE) *looks* like a local directory while reintroducing the network into
the read path. **Correctness axis = mmap-capable local filesystem; persistence axis =
hydration frequency. Never conflate them.** Also from §3a: this repo's `s3://` path
goes through `lance` (default features, `aws` ON), **not** `lancedb` — the `lancedb`
feature gate is a consumer-side trap, not this crate's.

---

## 2. Phase A — representation reliability (`jc`)

**The question.** Does the encoding rank-preserve the field? This is `F-1` in weather
clothing: the substrate's oldest un-run fidelity gate, now against ground truth this
workspace did not author.

### Deliverables

- **D-WXA-1 — Stage-A ingest (disposable).** WB2 Zarr → flat `f32` slab + JSON
  sidecar + manifest. Python stays in the `weatherbench2` fork's `tools/`. Scope: **Z500,
  one year, 6-hourly** (~1,460 states). Emits the shared `soa:*` metadata block.
- **D-WXA-2 — `crates/weather-poc` scaffold.** Workspace-**EXCLUDED** (the
  `perturbation-sim` template). Zero-dep default build; every encoder arm behind an
  off-by-default feature. **No new repository.**
- **D-WXA-3 — encoder arms.** A `helix48` · B `helix + residue` · C `bgz17` flat
  palette · D `bgz17` **hierarchical** palette · E `cascade_key` V3 `(part_of:is_a)`.
  Arms C/D are **zero-dep and build first**; A/B pull `ndarray` *by git URL* (not path
  — codex P2 #460), so they are gated and sequenced second. This ordering means the
  first real number needs no network-resolved dependency.
- **D-WXA-4 — the `jc` battery.** Primary: `jc::reliability::spearman(code_dist,
  field_dist)` — rank fidelity. Secondary: `icc(ratings, IccForm)` for cross-arm
  agreement; `cronbach_alpha` for lane-internal consistency. Escalation available in
  `jc::stats`: `kr20`, `phi`, `binary_association`, `omega_total`, `eta_squared`,
  `t_test_*`, `anova_one_way`.
- **D-WXA-5 — the gate.**

> **PASS:** ρ ≥ 0.98 for at least one arm **AND** the shuffled-codebook control
> **FAILS**. A bake-off in which every arm passes has measured nothing.

**Anti-vacuity is mandatory, not decorative** (`E-VACUOUS-ASSERTION-IS-THE-HOUSE-STYLE-1`):
the control must be a *destroyed* codebook on otherwise identical input, and it must
be shown to fail. Additionally, `jc`'s degenerate-input contract returns `None` — a
window with zero variance must be reported, never silently folded into an aggregate
as 0.0 (see §3, D-WXB-4).

**KILL:** every arm below ρ ≈ 0.9 ⇒ the hierarchical-prefix-is-ancestry assumption
does not survive contact with a real physical field, and a large fraction of the
substrate's `[H]`/`[S]` map downgrades from "faithful code" to "useful router." That
is a thesis-level result, not a weather result — which is exactly why Phase A runs
before anything is built on top of it.

### Prior art — shape from these, do not reinvent

`jc` already ships structurally-adjacent probes: `ontology_locality_probe`,
`substrate_compare`, `style_table_agreement`, `splat_perturbationslernen`, and the
`run_all_pillars()` harness pattern. The weather probe is a sibling of these.

---

## 3. Phase B — hardware acceleration (`ndarray`), in between

**This is not a statistics question.** It is *parity + throughput*: is the SIMD path
actually engaged, and does it compute the same answer?

- **D-WXB-1 — scalar ↔ SIMD parity.** Bit-identical, or a tolerance documented with
  its reason. A divergence here silently corrupts every Phase-A and Phase-C number.
- **D-WXB-2 — no silent scalar fallback.** `simd_caps()` must report the path
  actually taken, and the probe must record it. A scalar fallback that reports success
  is the failure mode this deliverable exists to catch.
- **D-WXB-3 — throughput against a measured anchor.** The workspace has two
  independent measurements in the same envelope — **1.86 ns/op** (3DGS top-k, 268 M
  comparisons in ~500 ms) and **1.8 ns/lookup** (611 M SPO lookups/sec). A Phase-A
  encode/compare sweep materially off that envelope means the SIMD path is not doing
  what it claims.
- **D-WXB-4 — `jc` ↔ `ndarray` reliability agreement (a real divergence, found
  2026-08-10).** `pearson` / `spearman` / `cronbach_alpha` / `icc` exist in **both**
  `jc::reliability` and `ndarray::hpc::reliability`, with **different degenerate-input
  contracts**:

  | | `jc` | `ndarray::hpc` |
  |---|---|---|
  | signature | `-> Option<f64>` | `-> f64` |
  | degenerate input | `None` | `0.0` |
  | `icc` | `icc(ratings, IccForm)` | `icc_a1(ratings)` |

  ρ = 0.0 is *also a legitimate measured value*, so the ndarray form cannot
  distinguish "no correlation" from "undefined." **`jc` is the authority** (operator-
  named); ndarray's is the SIMD-side mirror. The probe: run both over identical
  non-degenerate inputs and assert agreement, then assert the degenerate case is
  reported rather than folded. **Every Phase-A/C number is computed with `jc`.**

---

## 4. Phase C — prediction correctness (`jc`)

**The theory of victory is retrieval, not learned dynamics.** Analogue forecasting
(Lorenz 1969) failed for exactly one reason: state-space search was too slow. A
compact code over 65 years of reanalysis makes the corpus content-addressable. This
lane is open because nobody has rebuilt it on modern content-addressing — **not**
because learned dynamics can be out-forecast. Competing with a trained model on RMSE
is explicitly **out of scope**.

- **D-WXC-1 — Stage-C ingest (permanent) — WIRING ONLY.** Recurring, append-only,
  one version per cycle, provenance in table metadata, idempotent re-run — all of it
  through `LanceCycleWriter` + `VersionedGraph`, per §1. The deliverable is the
  *weather driver*: cycle → slab → `commit_encounter_round`. **Zero versioning code.**
  Hydration follows the #901 lifecycle (S3 → local mmap dir), and its
  `absent → hydrated` transition is explicitly *not* unconditionally idempotent
  (§4a of that doc) — the driver must respect that boundary rather than re-derive it.
- **D-WXC-2 — two-stage retrieval.** Coarse tiers first (~4 KB synoptic signature ×
  ~570k states ≈ **RAM-resident**, the compute-bound regime the 1.86 ns/op figure
  actually measured), then hydrate top-*k* full skeletons from disk. **Full-skeleton
  scan is the wrong regime** — 570k × 6.2 MB ≈ 3.5 TB is bandwidth-bound, not
  compute-bound. The coarse→fine cascade is *required here*, not merely nice.
- **D-WXC-3 — the `jc` battery on predicted vs observed.** Same instrument as
  Phase A. The forecast/analysis join is a **version-range read** —
  `QueryReference::at(v, rung)` + `deinterlace` — not a bespoke temporal join.
- **D-WXC-4 — the external score.** WeatherBench2 metrics (RMSE/ACC at standard lead
  times) against **persistence and climatology**. This is the anchor: the first number
  in this program whose scoring rules the workspace did not choose.
- **D-WXC-5 — the three comparison lanes** (operator: *all 3*):
  1. **forecast vs analysis** — verification; needs both lanes joined on valid-time
     *within the one dataset*.
  2. **model vs model** — IFS / AIFS / WeatherNext on the same init.
  3. **encoder drift** — does the codebook still rank well as seasons turn? This is
     the one that guards the *substrate* claim over time; (1) and (2) are meteorology.

Pillar 11 (`hambly_lyons`, sigker-gated) certifies signature uniqueness on rough
paths. A weather trajectory through code-space **is** such a path — so the analogue
lane has an existing certification pillar rather than needing an invented one.
`[S]` until exercised.

---

## 5. Repos

**New repositories required: ZERO.**

| repo | role | state |
|---|---|---|
| **lance-graph** | hosts `crates/weather-poc`; supplies `jc`, `helix`, `bgz17`, `perturbation-sim`, `lance-graph-contract`, `dev_s3_env` | ✅ cloned, in scope |
| **ndarray** | Phase B; helix's **git-URL** dep | ✅ cloned, in scope |
| **weatherbench2** (fork) | **Phase A data + the external scoring harness** | ✅ cloned `95c36d5` |
| **arco-era5** (fork) | alternative ERA5 path | ✅ cloned `8fb5e9b` |
| **ecmwf-opendata** (fork) | **Phase C only** — live IFS/AIFS | ✅ cloned `b7ff73d` |
| `graphcast` (GraphCast + GenCast + WeatherNext 2, one repo) | Phase C baseline | **zipball on demand** — under the 3-reads-per-repo bar |
| `ai-models-graphcast` | only if baselines are *run* | **zipball on demand** |
| **OGAR** | only if a weather domain is minted (0x03–0x06 free; 0x0F = `Geo`) | ✅ cloned; **not needed for A or B** |

`weatherbench2` and `arco-era5` are **not** in the session's MCP repo scope — the POC
session needs them added.

---

## 6. Pins (verified against the tree, 2026-08-10)

```
rust        1.97.1        (rust-toolchain.toml)
lance       =9.0.0        lance-encoding 9.0.0 · lance-linalg =9.0.0 · lance-index =9.0.0
lancedb     =0.33.0
arrow       58.3.0
datafusion  54            ← NOT 53; see the correction below
```

The lance family moves in **exact lockstep** (`=X.Y.Z` in every manifest, verified);
a bump is one deliberate PR, never a drift.

> **⊘ CORRECTION (2026-08-10, same day, before any POC work started; corrected
> twice).** This section's first version said **`datafusion 53`** and filed the
> presence of 54.1.0 in `Cargo.lock` as suspicious drift. **That was backwards.**
> `datafusion = "54"` is our direct pin in every crate manifest, and `lance` /
> `lancedb` / `lance-index` / `lance-datafusion` all require 54; the move is
> recorded and **MEASURED** (`lance9-datafusion54-upgrade-probe-v1.md`, 2026-08-05).
>
> **Second correction (operator):** the first fix then called 53 a *"residual
> transitive"* — also wrong, and more dangerous, because it invites someone to
> collapse the lock to one version. **Both majors are REQUIRED.**
> `deltalake-core 0.32.4` pins `datafusion 53.1.0` (+ `-datasource`,
> `-physical-expr-adapter`) upstream, backing the optional `delta` feature. Two
> semver majors coexisting is the *correct, documented* state — not a defect to
> tidy. It lifts only when deltalake moves to DF 54.
>
> **Root cause, worth more than the fact:** `CLAUDE.md`'s Key Dependencies block —
> the mandatory first read for every session — was itself stale (`lance = "=7.0.0"`,
> `lancedb = "=0.30.0"`, `datafusion = "53"`, dated 2026-06-14, pre-dating the
> lance-9 sweep). The wrong pin propagated *from* that block *into* this plan. A POC
> session that trusted either would have pinned lance 7 against a lance-9 tree and
> failed to build. Both are corrected in the same PR as this note.

The `rust-toolchain.toml` comment drift noted in the first version is also fixed
there — **structurally**: the comment no longer restates the channel value (which is
what made it go stale twice), and carries an append-only bump log instead.

---

## 7. Credentials

`AWS_ENDPOINT_URL` · `AWS_DEFAULT_REGION` · `AWS_S3_BUCKET_NAME` ·
`AWS_ACCESS_KEY_ID` · `AWS_SECRET_ACCESS_KEY` — **referenced by name, expanded at
runtime, never printed, captured, or written to any file.** `dev_s3_env::s3_options()`
already reads exactly this set and returns `None` when any is missing; on a remote
path that must be a **hard error**, never a silent local fallback. `RAILWAY_VOL` is
the Stage-C staging mount.

---

## 8. Sequencing, and what each phase can kill

```
D-WXA-1 ingest ─→ D-WXA-2 scaffold ─→ D-WXA-3 arms (C/D first, zero-dep)
                                          │
                                          ▼
                                   D-WXA-4/5  jc battery + gate   ← KILL: thesis-level
                                          │
                                          ▼
                                   D-WXB-1..4  ndarray + jc/ndarray agreement
                                          │
                                          ▼
                        D-WXC-1..5  permanent ingest → retrieval → external score
```

**Nothing in Phase C is built before Phase A's gate reports.** A negative Phase A is
not a wasted POC — it is the most informative result available, and it is
thesis-relevant rather than weather-relevant.

---

## 9. Honest grading

| claim | grade |
|---|---|
| 1.04 M grid at 6 B/point ≈ 6.2 MB; 13.5 M full-3D inside a demonstrated 20 M budget | `[G]` arithmetic on shipped formats |
| the retrieval kernel runs at ~1.86 ns/op | `[G]` measured, two independent workloads |
| 512 B stride ⇒ verbatim on disk, incl. S3 | `[G]` PR #907, with a compression-detection falsifier |
| helix48 rank-preserves a real atmospheric field | **`[H]` — this is D-WXA-5** |
| analogue retrieval forecasts usefully | `[S]` — the bet, gated on Phase A |
| Pillar 11 certifies the code-space trajectory | `[S]` — unexercised |

---

## Cross-references

- Supersedes `.claude/plans/weather-substrate-poc-v1.md` (PR #914).
- `crates/jc/` — `reliability.rs` (pearson/spearman/cronbach_alpha/icc), `stats.rs`,
  `run_all_pillars()`, `hambly_lyons.rs` (Pillar 11).
- `ndarray/src/hpc/reliability.rs` — the mirrored battery (D-WXB-4).
- PR #879 (`seal_cycle` → one version per cycle) · PR #907 (verbatim slab + S3) ·
  PR #902 (`identity_quad`, pre-bake resolution).
- `substrate-unification-thesis.md` §4 (F-1, F-collapse), §8.
- Iron rules: `I-VSA-IDENTITIES` (bundle identities, table-compose content),
  `I-NOISE-FLOOR-JIRAK` (weak dependence — cite Jirak's rate, not classical
  Berry-Esseen), `I-LEGACY-API-FEATURE-GATED`.
