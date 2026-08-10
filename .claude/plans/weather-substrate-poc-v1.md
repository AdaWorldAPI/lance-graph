# Weather Substrate POC — Plan v1 (encoder bake-off first, forecast second)

> **Status:** PROPOSED (doc-only). No code in this PR. The POC runs in a
> follow-on session.
> **Type:** discovery/ablation plan — the operator's ask was *"find what works vs
> the best combination"*, so the first deliverable is a **bake-off**, not a build.
> **Grounded on:** `origin/main` @ `f675a0ff`, via a 5-agent parallel recon
> (2026-08-08). Every "already exists" below carries a file:line. Everything
> unverified is marked.
> **Cross-refs:** `.claude/knowledge/stockfish-nnue-as-perturbation-cascade.md`
> (the NNUE ledger — *read before claiming NNUE proves anything*),
> `.claude/knowledge/neurosymbolic-rlvr-causal-curriculum-v1.md` §3.8 (GRPO),
> `substrate-unification-thesis.md` §4 (F-1, F-collapse), PR #879 (cycle driver),
> PR #907 (verbatim Lance/S3 sink), `crates/perturbation-sim` (the template).

---

## 0. Corrections this plan must carry (read first)

Recon contradicted several things asserted earlier in the originating session.
Recording them so the POC session does not inherit them:

| Claim in circulation | What the tree actually says |
|---|---|
| "helix is ~1.7° at equator, 2.3–3.7° at poles" | **Not in the repo.** Exhaustive grep across `crates/`, `docs/`, `.claude/`, and `/home/user/ndarray/src` finds no such figure. Documented resolution is **sub-degree**: polar 7-bit ≈ 0.45°, azimuth ≈ 0.35° (`.claude/knowledge/helix-cartesian-vs-fisher2z.md:96-97`); `helix_orient` cap-cascade ≤ 0.3° (:57). `Signed360`'s azimuth is a `u16` over 360° ⇒ 0.0055°/step. The operator figure may come from an out-of-tree measurement or from golden-spiral *placement* spacing at some N; **treat it as unverified**. Either value clears the meteorological bar (§5.4), so nothing downstream depends on resolving it — but do not cite 1.7° as in-repo fact. |
| "we measured 64k weather at ~125 ms compute / 233 ms disk"; "3DGS top-k 16k×16k ≈ 500 ms" | **Operator-reported, out-of-tree.** Recon found **zero** weather/NWP work anywhere in lance-graph. The board's `measure-64k-axes` arm is **65,536 mailbox owners** with seal/WAL timings — a different thing. Do **not** conflate; a "64k weather test" would be terminologically ambiguous on the board. These numbers are usable as priors, not as citations. |
| "879 dispatches via `kanban_actor.rs`" | **Stale.** `kanban_actor`'s actor/ack/tick surface was **deleted 2026-08-05**. The complete production phase-progression path is `lance-graph-supervisor::cycle_driver` behind feature `cycle-driver` (no message bus; applies inline via `MailboxSoaOwner::try_advance_phase`). What remains in `kanban_actor` is `mul_target` / `parse_kanban_step` / `PhaseCensus`. |
| "#907 gives us a verbatim sink to call" | Half true. The sink is an **example binary** (`crates/lance-graph/examples/soa_to_lance.rs`), not a library function — a consumer replicates ~80 lines. The one *pub lib* surface is **`lance_graph::dev_s3_env`** (`src/dev_s3_env.rs:23,42`), which is exactly what the POC needs for S3. |
| "NNUE/stockfish acceleration is a new idea to introduce" | **Already documented prior art** with a graded mechanism-vs-rhyme ledger: `.claude/knowledge/stockfish-nnue-as-perturbation-cascade.md`. Reuse its grades; do not re-derive, and do not exceed them (§5.5). |
| "0x0F is free for a new domain" | **No — 0x0F is `Geo` (OSM)** (`ogar_codebook.rs:83-88`). Free domain bytes: **0x03–0x06**, 0x10+. Whether weather mints its own domain or lives under `Geo` is an **operator/OGAR mint decision**, not this plan's to make (§6.4). |

---

## 1. What already exists (do NOT rebuild)

The single most important recon result: **almost every part is already in-tree and
unwired.** The POC is composition + measurement, not construction.

### 1.1 Encoders / codecs

- **`crates/helix`** (standalone, workspace-excluded). Four-stage Place/Residue
  codec: `HemispherePoint::lift` equal-area `r=√u`, `azimuth=n·φ`
  (`placement.rs:112-119`) → `CurveRuler` stride-4-over-17 (`curve_ruler.rs:31-56`)
  → Fisher-Z arctanh (`fisher_z.rs:55-58`) → `RollingFloor` 256-palette quantise
  (`quantize.rs:99-108`).
  - **`ResidueEdge` = 24-bit / 3 B** (`residue.rs:20-27`).
  - **`Signed360` = 48-bit / 6 B EXISTS** (`residue.rs:63-87`), doc'd verbatim as
    *"the 24-bit hemisphere ResidueEdge doubled to 48 bit"*. Wire LE
    `[rim.start, rim.end, rim.floor_version, polar, azimuth_lo, azimuth_hi]`;
    **polar = |y| in 7 bits with the hemisphere sign carried in the partition**
    (≥128 upper / <128 lower) — *this is the operator's "signed in/out bit",
    already shipped*; azimuth = `u16` over the full 360°. Size pinned by test
    (`residue.rs:306-315`).
  - `DistanceLut` 256×256 `u16` L1, metric-safe (`distance.rs:25-56`).
  - ⚠ **`ndarray` is a MANDATORY dep, sourced by GIT** (branch `master`), not
    path (`Cargo.toml:8-30`, per codex P2 #460). Constrains the POC crate's dep
    story (§6.2).
- **`crates/bgz17`** palette family — the hierarchical-vs-flat arm:
  - `Palette` + `build_distance_table()` → `PaletteDistanceTable`, 256×256 `u16`
    L1, O(1), 128 KB at k=256 (`palette.rs:77,506-532`).
  - **`HierarchicalPalette`** — 16 coarse × 16 leaves, `code>>4 == coarse`,
    `coarse_is_ancestor_of` (`palette.rs:214,451-493`). **This is the F-1
    structure** (hierarchical ancestry vs flat k-means).
  - `PaletteResolution{Full256 ρ=0.992, Half128 ρ=0.965, Quarter64 ρ=0.738}`
    (`palette.rs:543-582`) — **measured fidelity anchors already exist** on other
    data; weather gives them an external re-test.
  - `PaletteSemiring` compose table k×k `u8` (`palette_semiring.rs:18-135`).
- **Fisher-Z exists twice — do not write a third.** `helix/src/fisher_z.rs`
  (f64 analytic, `Similarity`, `hyperbolic_depth`) and
  `bgz-tensor/src/fisher_z.rs` (`FamilyGamma` i8-quantised + `FisherZTable` k×k
  i8, **certified ρ≥0.999**).
- **`FacetCascade`** (`lance-graph-contract/src/facet.rs`) — 16 B
  `facet_classid(4) | 6×(8:8)`; zero-copy `ref_from_bytes` (:183), `as_u128`,
  `tier_bytes()` 12-unit ladder (:302), `cascade_group_shared` per-group LCP
  (:326). Its doc **already names the lane this POC wants**, verbatim
  (`facet.rs:388-393`): *"A separate `G2×48bit` lane reads the same 12 tier-bytes
  as the two 48-bit chains (`hi_chain`/`lo_chain`) … for **helix** (location) and
  **CAM-PQ** (centroid) encoding."* ⇒ the weather facet is a **sanctioned
  existing reading**, not a new carving.

### 1.2 Storage (both legs the operator named)

- **`lance_graph::dev_s3_env`** (`src/dev_s3_env.rs:23,42`, exported `lib.rs:43`)
  — `env()` (strips sandbox quotes, empty⇒absent) and `s3_options()` reading
  **exactly** `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_ENDPOINT_URL`
  (→ `aws_endpoint`), `AWS_DEFAULT_REGION` (default `auto`),
  `aws_virtual_hosted_style_request=false`. Returns `None` if any required var is
  missing — **a committed-remote path must treat `None` as a hard error, never a
  silent local fallback.**
- **Verbatim guarantee is structural, not metadata**: `NODE_ROW_STRIDE = 512 ≥
  MINIBLOCK_MAX_BYTE_LENGTH_PER_VALUE = 256` ⇒ full-zip path ignores compression
  metadata. `lance-encoding:compression="none"` is a *defensive pin only*
  (PR #907's own correction). `tests/soa_verbatim.rs` pins contiguity,
  512-alignment, <64 KiB overhead, an anti-vacuity absent-needle search, a
  narrow-column falsifier proving zstd *is* honoured below 256 B, and an S3 arm
  against this exact env-var set.
- **Out-of-line bulk doctrine** (`canonical_node.rs:706-730`): bulk raw data lives
  in a **separate Lance table referenced by key/classid**; the node stays 512 B.
  *This is the mechanism weather field data needs* — the skeleton is addressable,
  the 227-variable column is out-of-line.

### 1.3 The loop

- **`cycle_driver`** (`lance-graph-supervisor/src/cycle_driver.rs`, feature
  `cycle-driver`): `collect_casts` :357 → `seal_cycle` :434 (exactly one WAL
  write / `DatasetVersion`) → `apply_sealed_transitions` :509 (sparse set only;
  unrepresented owners byte-identical) → `run_cognitive_work_gated*` :858/:876
  (`gate_decision_i4` + `advance_on_gate`) → `recover_fleet` :969.
  `MailboxFleet` trait :316 with blanket impl for `HashMap`.
  ⚠ **WAL is still a contract-probe fake** (`WalSink`); concrete `LanceShardSink`
  has not landed. Any durability claim in the POC must say so.

### 1.4 NNUE ("stockfish acceleration") — already graded

`.claude/knowledge/stockfish-nnue-as-perturbation-cascade.md`. Reuse its grades:

- **`[G]` transferable:** deterministic-address phase + **stored-magnitude-only**;
  incremental base+delta update with **coarse-tier escalation** (king move ⇒ full
  refresh ≡ `RouteAction::Escalate`); SoA + `matmul_i8_to_i32` (literally shared
  code with ndarray).
- **Fenced rhymes — do not ship as proven:** Walsh-Hadamard / bipolar sign pyramid
  has **no NNUE analog** (NNUE is magnitude-only; no XOR sign side); Morton
  2bit×2bit tiling is **not** NNUE's addressing (king-bucket × piece);
  palette256² is not how NNUE stores weights.

### 1.5 "DeepSeek" in this workspace = GRPO/RLVR, not the LLM

`neurosymbolic-rlvr-causal-curriculum-v1.md` §3.8 — **GRPO / DeepSeekMath**
(arXiv 2402.03300): group-relative advantages over multiple rollouts, no learned
critic, **works with deterministic verifiers**. Weather is an unusually good fit
(§5.6) because verification is deterministic and arrives on a schedule — but it
is **Stage 3**, gated (§4).

---

## 2. The claim ladder — four separable claims, never blurred

Recon's clearest warning. These have wildly different costs and evidence bars:

| # | Claim | Cost | Status |
|---|---|---|---|
| C1 | The substrate can **store** weather verbatim + addressably | ~free (PR #907 did the hard part) | near-`[G]` |
| C2 | A 48-bit code **preserves synoptic distance** | days | **`[H]` — the gate** |
| C3 | **Retrieval over history forecasts** better than trivial baselines | ~2 weeks | `[S]` |
| C4 | It **beats a learned model** (GraphCast/AIFS) | quarters | `[S]`, out of scope |

**This plan targets C2 first and C3 second. C1 is a side effect. C4 is explicitly
not attempted** — §5.7 explains why the winnable lane is different.

---

## 3. The POC choice (the operator asked me to choose)

**The POC is the Stage-0 encoder bake-off.** Not the loop, not the forecast.

Rationale:

1. **It is the only stage whose failure kills everything downstream.** If a 48-bit
   code does not rank like the physical field, the analogue lane, the gating, and
   the addressable-skeleton story all collapse together.
2. **It is the cheapest decisive experiment** — needs ~10³ states, one variable,
   no forecast loop, no cycle driver, no scale.
3. **It doubles as F-1**, the thesis's #1 unrun gate: `HierarchicalPalette`
   (16×16 ancestry) vs flat `Palette` (k-means 256) is *literally*
   hierarchical-vs-flat fidelity — now on **external data the workspace did not
   choose**. Two gates, one probe.
4. **`crates/helix` already owes exactly this probe.** `KNOWLEDGE.md:338-343`:
   fidelity vs certified `Base17Fz` is **CONJECTURE, probe NOT RUN**, gate
   ≥0.9980 Pearson. The POC discharges an existing debt rather than inventing one.
5. **It produces an externally-refereed number.** Correlation against physical
   RMSE is not a metric this workspace defined.

**Explicitly NOT first:** re-running the loop at weather scale. PR #879 already
proved the control-loop contract and the operator has out-of-tree scale numbers —
running it again on weather is *confirmation*, not information.

---

## 4. Stage ladder (each gated on the prior)

```
Stage 0  encoder bake-off        C2   ← THE POC. days.        gate: ρ ≥ 0.98 (§5.4)
Stage 1  analogue retrieval      C3   ← weeks.                gate: beats persistence+climatology
Stage 1.5 GRPO analogue weighting     ← only if Stage 1 green
Stage 2  loop at scale + sink    C1   ← engineering, no new science
Stage 3  RLVR closed loop             ← only if Stage 1 green; operator-gated
Stage 4  (NOT PLANNED) learned-model comparison — see §5.7
```

---

## 5. Stage 0 in detail — the bake-off

### 5.1 Dataset (fixed, small, external)

- **Source:** ECMWF open data (CC-BY-4.0) via the operator's `ecmwf-opendata`
  fork; ERA5 acceptable if easier to obtain in bulk.
- **Field:** **Z500** (500 hPa geopotential) — the canonical synoptic field and
  the headline WeatherBench2 variable. One variable, one level. Deliberately not
  the full 227.
- **Extent:** ~1,000 states (e.g. 1 yr, 6-hourly) at 0.25° = 1440×721 =
  **1,038,240 points** (the operator's "1 mio"). Coarsened copies at 1° and 2.5°
  as ablation rungs.
- **Format:** **GRIB2 never enters Rust** (§6.3).

### 5.2 The arms (this is the "best combination" question)

Each arm encodes one Z500 field → a code; we then rank code-distance against
physical distance.

| arm | encoder | bits | why it's in |
|---|---|---|---|
| **A** | `helix::Signed360` | 48 | the named candidate; signed polar + u16 azimuth |
| **B** | `helix::ResidueEdge` | 24 | is half the budget enough? |
| **C** | `bgz17::Palette` (flat k-means 256) | 8/cell | the flat control |
| **D** | `bgz17::HierarchicalPalette` (16×16) | 8/cell | **the F-1 arm** — ancestry vs flat |
| **E** | `CascadeKeyV3` 8:8 `(part_of:is_a)` tiles | 48 | the address-native reading |
| **F** | raw f32 / PCA-k | — | **honest control**; if F ≈ A the codes buy nothing |
| **G** | golden-disc PIT normalisation × A | 48 | the operator's equiprobable-code idea (§5.3) |

Arms are **independent**; run them all, report the matrix. Do not pre-commit to a
winner.

### 5.3 One design note carried from the session (arm G)

Wind speed is Weibull(≈2)/Rayleigh ⇒ probability-integral-transforming (u,v) makes
the velocity plane ~uniform on the disc, which is *exactly* the distribution the
Vogel spiral quantises optimally. So "normalise to golden 2D" is
**maximum-entropy vector quantisation**, not numerology. **Iron constraint:** the
normaliser must be a **frozen climatological CDF** (the ruler), never fitted to
the live field — otherwise we rebuild the ketchup conflation `place_buffer.rs`
was written to fix (`place_buffer.rs:1-34`, ICC 0.14 → 1.00).

### 5.4 Metric + gate

- **Primary:** Spearman ρ(code-distance, physical RMSE) over deterministically
  sampled state pairs. **Gate: ρ ≥ 0.98** for a headline arm; helix's own
  `Base17Fz` gate is ≥0.9980 Pearson — report both, don't silently soften.
- **Secondary:** ICC (absolute agreement), Cronbach α across arms.
- **Significance:** Jirak 2016 `n^(p/2−1)` rates per `I-NOISE-FLOOR-JIRAK` —
  **never classical Berry-Esseen** (grid fields are weakly dependent by
  construction).
- **Anti-vacuity (mandatory, per `E-VACUOUS-ASSERTION-IS-THE-HOUSE-STYLE-1`):**
  arm F must be *able to lose*; include a deliberately-broken encoder (shuffled
  codebook) that **must** fail the gate. A bake-off where every arm passes has
  measured nothing.
- **Meteorological sanity:** 1.7° (unverified) or 0.45° (documented) both sit far
  below operational wind-direction reporting (10° increments) and day-3 direction
  error (20–30°). Resolution is **not** expected to be the binding constraint;
  magnitude ranking is. Test the **full** code (place ⊕ residue), not the angular
  part alone.

### 5.5 What Stage 0 must NOT claim

- Not that NNUE proves the cascade (§1.4 fences).
- Not that a passing ρ implies forecast skill — that is C3, Stage 1.
- Not durability (`WalSink` is a fake).

### 5.6 Stage 1 sketch (for sequencing only)

Analogue forecasting: k-NN over historical codes → forecast = weighted composite
of their T+N successors. Score **externally on WeatherBench2 metrics** (RMSE, ACC)
at T+24/72/120 h against **persistence + climatology** (not GraphCast). Two-stage
retrieval is *required*, not optional: coarse-tier signature (~4 KB/state ⇒
~234 MB for 58 k states, RAM-resident, compute-bound) → hydrate top-k full
skeletons from S3. A naive full-skeleton scan is 58 k × 6.2 MB = **362 GB** and is
bandwidth-bound — a different regime from any measured number.

**Stage 1.5 (GRPO):** retrieved analogues form a natural *group*; group-relative
advantage is the principled way to weight them, and their spread is an ensemble
uncertainty for free. This is the earliest honest use of the DeepSeek/GRPO
primitive — after retrieval works, not before.

### 5.7 On "beating Google" — the winnable lane

Do **not** contest learned dynamics at WeatherBench2 RMSE; that needs training
runs we do not have. GraphCast (2212.12794) is also *evidence for our side of the
split*: its multi-mesh is **hand-engineered hierarchical structure** (icosahedron
refined 7×, 4-ary) with learned propagation on top — engineered address ×
learned dynamics. The open lane is what GraphCast lacks: **episodic memory**
(versioned history to reason over), **native uncertainty** (forks, not a second
diffusion model), **counterfactuals** (edit at version *k*), and **cost** (CPU
SIMD random access vs TPU dense). Retrieval-as-forecasting is the research bet;
memory/uncertainty/counterfactual are structural wins that need no luck.

---

## 6. Repos — what is needed, and what is not

### 6.1 The answer: **no new repository.**

Anti-scope-explosion. The template already exists.

| repo | role | change |
|---|---|---|
| **lance-graph** | host of the POC crate | **NEW crate `crates/weather-poc`**, workspace-**excluded**, exactly the `perturbation-sim` template |
| **ndarray** | helix's mandatory git dep + SIMD | **unchanged**, consumed |
| **`crates/helix`** | arms A/B | consumed (path dep) |
| **`crates/bgz17`** | arms C/D | consumed (path dep) |
| **`lance-graph-contract`** | `FacetCascade`, `NodeGuid`, tenants | consumed |
| **`ecmwf-opendata`** (operator fork) | data acquisition | **small `tools/` addition**: download → flat slab. Python stays here. |
| **`weathernext`** (operator fork) | baseline comparison | **Stage 2+ only.** Not needed for Stage 0/1. |
| **OGAR** | classid/domain mint | **only if** a weather domain byte is minted (§6.4) — cross-repo arc, operator-gated |

### 6.2 Crate shape (`crates/weather-poc`)

Follow `perturbation-sim` precisely: standalone, workspace-**excluded**,
**zero-dep by default**, heavy deps behind off-by-default features, probes as
`examples/`, `#[test]` density in-module.

```toml
[features]
default = []
helix-codec = ["dep:helix"]     # pulls ndarray via helix's MANDATORY git dep
palette     = ["dep:bgz17"]     # bgz17 is 0-dep
contract    = ["dep:lance-graph-contract"]
lance-sink  = ["dep:lance-graph"]  # Stage 2 only — heavy (lance/datafusion/arrow)
```

⚠ **`helix` pulls `ndarray` by git, not path** (`helix/Cargo.toml:8-30`) — a clean
checkout must resolve it. Keep `helix-codec` **off by default** so the bake-off's
pure-palette arms build with zero network.

### 6.3 GRIB2 stays out of Rust

Ingest is Python-side in the `ecmwf-opendata` fork: download GRIB2 → decode →
emit a **flat little-endian `f32` slab** (+ a JSON sidecar: shape, variable,
level, valid-time, units, CRS). Rust reads slabs only. No `eccodes` C dep, no
`gribberish` risk in the POC's critical path. This also makes every arm read
**byte-identical input**, which the bake-off requires.

### 6.4 Classid / domain — flagged, not decided

`0x0F` is **`Geo` (OSM)**, taken. Free: **0x03–0x06**, 0x10+. Whether weather
mints its own `ConceptDomain` or rides under `Geo` is an **OGAR mint decision**
(cross-repo arc per `E-CODEBOOK-MINT-IS-A-CROSS-REPO-ARC`: `ogar-vocab` +
contract `CODEBOOK` mirror + `parity::domains_agree` move **together**).
**Stage 0 needs no classid at all** — it is pure encoder measurement. Do not mint
to unblock a probe that does not need it.

---

## 7. Storage plan (operator-provided)

| tier | target | contents |
|---|---|---|
| **hot** | `RAILWAY_VOL=/volume01` | coarse-signature index (~234 MB/58 k states), the working slab, probe outputs |
| **bulk** | S3 `AWS_ENDPOINT_URL=https://t3.storageapi.dev`, bucket `AWS_S3_BUCKET_NAME=stashed-pannier-sko9w7jc`, `AWS_DEFAULT_REGION=auto` | GRIB2 archives, full skeleton history, Lance datasets |

- Credentials `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` are **environment-only**:
  never printed, captured, written to a file, embedded in a URL, or committed.
  Reference by variable name; let the runtime expand.
- **Reuse `lance_graph::dev_s3_env::s3_options()`** — it already reads this exact
  var set and is the surface PR #907's S3 arm proved against this endpoint shape.
  `None` ⇒ hard error on a remote path, never a silent local fallback.
- Scratch writes under a `_tests/`-style prefix with cleanup, mirroring
  `soa_verbatim.rs`'s S3 arm.

---

## 8. `perturbation-sim` — the comparison the operator asked for

**It is the template, and it is a genuinely different physics.**

| axis | `perturbation-sim` (shipped) | `weather-poc` (proposed) |
|---|---|---|
| domain | power grid, ~10³ buses | atmosphere, ~10⁶ points |
| topology | irregular graph (lines) | regular lattice |
| "stencil" | **electrical** 3×3 Moore = 8 nearest by effective resistance (`place_buffer.rs:82`) | **geometric** PDE 9-point |
| place | derived from spectrum → **conflated**, then fixed by `helix_place` (ICC 0.14→1.00) | **given** (lat/lon) — conflation structurally impossible |
| dynamics | DC flow + LODF + dense `laplacian_pinv` — **O(n³), compute-bound** | explicit timestep — **memory-bound**, AI≈2.5 FLOP/byte |
| sparsity | naturally sparse cascade (17-of-64k regime) | **dense** unless gated ⇒ the adversarial load |
| AMX/GEMM | **yes** — `laplacian_pinv` is dense O(n³) | **no** for the stencil (40× below tile break-even); yes for *ensembles* (batch dim) and ML emulators |
| deps | zero-dep + optional `ndarray` behind `ndarray-simd` (default OFF) | same pattern |

**Reuse, don't reinvent:** `columns.rs` `SoaMemberSpec` + `GuardrailVerdict`
(spec-before-encoder, §0 anti-invention gate); `buffer.rs::ketchup_yield`
(threshold semantics — convective triggering is the same shape);
`chaoda.rs::anomaly_ranking` (storm/front detection *is* anomaly ranking);
the 28-examples-as-probes pattern; the zero-dep + optional-feature Cargo shape.

**The two crates are complements, not competitors:** perturbation-sim is the
sparse/compute-bound endpoint, weather-poc the dense/memory-bound one. Running
both **brackets** the substrate. Perturbation-sim also has **no** weather
references anywhere — no collision.

---

## 9. Deliverables

| id | deliverable | gate |
|---|---|---|
| **D-WX-0** | `ecmwf-opendata` fork: GRIB2 → flat f32 slab + JSON sidecar; ~1 yr Z500 to S3 | — |
| **D-WX-1** | `crates/weather-poc` scaffold (excluded, zero-dep default, perturbation-sim template) | — |
| **D-WX-2** | Arms C/D (bgz17 flat vs hierarchical) + arm F control + broken-encoder anti-vacuity | zero-dep build |
| **D-WX-3** | Arms A/B (helix `Signed360`/`ResidueEdge`) behind `helix-codec` | git-dep resolves |
| **D-WX-4** | Arms E/G (`CascadeKeyV3`; golden-disc PIT with **frozen** climatological CDF) | D-WX-2 |
| **D-WX-5** | **The bake-off report** — ρ/ICC/α matrix, Jirak-bounded, per arm × resolution | **ρ ≥ 0.98 for ≥1 arm, and the broken encoder fails** |
| **D-WX-6** | Board hygiene: EPIPHANIES entry recording the result **whichever way it goes** | D-WX-5 |
| — | *Stage 1+ deliverables are not enumerated here — they are gated on D-WX-5.* | |

---

## 10. Gates, risks, what kills this

- **Kill (C2):** no arm clears ρ ≥ 0.98 ⇒ the 48-bit skeleton is a *router, not a
  code*; the analogue lane and the addressable-skeleton story both die. **Report
  it and stop** — that is a successful probe.
- **Kill (F-1):** if D ≉ C (hierarchical no better than flat), the
  prefix-is-ancestry assumption weakens **workspace-wide**, not just for weather.
  This POC is the first external test of it — treat a negative as
  thesis-relevant, not weather-relevant.
- **Risk — the normaliser drifts:** any live-fitted CDF re-creates the ketchup
  conflation. Frozen climatology only.
- **Risk — bandwidth, not compute:** Stage 1's 362 GB full scan is a different
  regime from every number quoted so far. The two-stage index is mandatory.
- **Risk — BF16 conservation:** BF16 is fine for read-only coefficients
  (`BufferResidue`), **hazardous for accumulated state**. Any BF16 in a
  prognostic path needs an fp32 accumulator and its own conservation falsifier.
- **Risk — scope blur:** C1/C2/C3/C4 must stay separate in every report.
- **Coordination:** the board's `measure-64k-axes` arm and the Stage-A0
  persistence/WAL harness are active lanes; D-BLW-5 is paused and D-HWV-1 gated.
  A weather stress test must not be read as unblocking any of them.
- **Honesty:** `WalSink` is a fake; no durability claim until `LanceShardSink`.

---

## 11. One-paragraph summary

Almost every piece already exists and is unwired: `helix::Signed360` is a shipped
48-bit signed place⊕residue code, `bgz17` ships both a flat and a **hierarchical**
256-palette with measured ρ anchors, `FacetCascade`'s own docs already sanction a
`G2×48bit` helix lane, `dev_s3_env` already reads the operator's exact S3 var set,
`cycle_driver` already closes a sparse cycle, and NNUE is already a graded
knowledge doc rather than a new idea. What does **not** exist is a single number
saying whether a 48-bit code preserves *synoptic* distance — and that number
happens to also be F-1, the thesis's oldest unrun gate, now answerable against
data and metrics this workspace did not choose. So the POC is a **bake-off**:
seven encoder arms plus a deliberately-broken control over one year of Z500,
scored by Spearman/ICC under Jirak bounds, in a new workspace-excluded
`crates/weather-poc` built on the `perturbation-sim` template — **no new repo**,
no GRIB2 in Rust, no classid mint, and no forecast claim until the encoder earns
one.
