# splat-native-ultrasound-v1 — integration plan (debt + future)

> **Status:** PROPOSAL / integration plan. Design-spec only; **no code in this plan**.
> **Authored:** 2026-06-05 (session `claude/lance-graph-ontology-review-Pyry3`).
> **Trigger:** user-supplied architecture diagrams (English 6-stage + German business-facing) describing a CPU-only Gaussian-splat ultrasound SaMD pipeline, with the FMA atlas as the registration target.
>
> **Cross-workspace plan.** Lance-graph is canonical; companion docs live at:
>
> - `ndarray/.claude/plans/splat-native-ultrasound-simd-substrate-v1.md` — SIMD math owed (D-SPLAT-2)
> - `OGAR/docs/SPLAT-NATIVE-CUSTOMER.md` — §6 FMA-litmus customer narrative
> - `MedCare-rs/.claude/handovers/2026-06-05-splat-native-medcare-hipaa-wire.md` — HIPAA path (D-SPLAT-10/11)
>
> **Anchored to (FINDING-grade):** `E-SOA-IS-THE-ONLY`, `E-BATON-1`, `E-CE64-MB-4`, `E-MAILBOX-IS-BINDSPACE`, `E-RUBICON-RACTOR`, `I-VSA-IDENTITIES`, `I-LEGACY-API-FEATURE-GATED`, `I-SUBSTRATE-MARKOV`, `I-NOISE-FLOOR-JIRAK`.

---

## 0. Executive summary (one screen)

A SaMD layer that consumes raw outputs (RF / IQ / channel-RF / Doppler / IMU) from existing certified probes (Clarius / Telemed ArtUs / us4us) and emits a **Gaussian Splat Volume** — thousands-to-millions of anisotropic 3D Gaussians carrying:

- **Amplitude → opacity**
- **Ultrasound PSF → anisotropic Σ** (covariance matrix)
- **Frequency content / Doppler → SH coefficients** (spherical harmonics, ℓ≤3)

Multi-frame 4D accumulation via IMU/POSE fusion lands every new frame in the same splat volume. Splat-to-splat registration against the FMA anatomical atlas closes the loop ("Σ-sandwich" Mahalanobis ICP). **CPU-only by design** (no GPU vendor lock; no GPU cart at bedside). The same splat math serves acquisition, registration, and rendering — one carrier, three operations.

This is the medical-imaging incarnation of `E-SOA-IS-THE-ONLY`: one substrate, never re-encoded, three legitimate operations (fit / accumulate / render). The clinical AR surface is the visual twin of `cognitive-shader-driver`; the FMA atlas is the explicit customer for OGAR PR #30 §6 ("FMA bones-rendering as the architectural litmus", ~75K classes, ~2.1M relationships, sub-millisecond HHTL traversal). The splat-native architecture is what makes that litmus actually pass.

**Total new deliverables:** D-SPLAT-1 … D-SPLAT-14 (14 deliverables across ndarray, contract, ontology, callcenter, MedCare, AR rendering, SaMD docs). Five open questions (`OQ-SPLAT-1` … `OQ-SPLAT-5`).

---

## 1. The architectural mapping

### 1.1 What the diagrams show

**English diagram (6 stages, technical):**

1. **Existing probe** — generic wireless ultrasound probe with raw outputs RF / IQ / channel-RF / Doppler / IMU (the certified hardware, untouched).
2. **Pipelines** — classical lossy (RF → beamform → envelope → log-compress → scan-convert → 2D B-mode pixels, *"information destroyed at each step"*) vs **splat-native** (RF → beamform → splat-fit → Gaussian3D batch, information-preserving).
3. **Splat representation** — Gaussian Splat Volume: thousands-to-millions of anisotropic Gaussians carrying amplitude, shape, frequency.
4. **Multi-frame integration** — real-time 4D accumulation (frames t₁…t_N, IMU/POSE fusion, pose-aligned frame fusion).
5. **AR rendering** — patient-aligned (HoloLens-class HMD on clinician; rendered volume overlaid on patient).
6. **Anatomical atlas** — FMA Atlas (splat volume) ↔ Live Patient (splat volume) via splat-to-splat registration / pose optimizer / **Σ-sandwich** (covariance-aware Mahalanobis distance).

The bottom card spells the engine: **Splat-native Converter Engine (SaMD)** — "Turn raw ultrasound data into rich 3D information—without discarding it." Four properties: existing probe as data source; CPU-friendly (modern multi-core); no GPU cart (lightweight, portable, cost-effective); **same math for acquisition, registration, rendering** (unified splat representation → simpler, more accurate, efficient).

**German diagram (business-facing):**

- Bestehende Hardware: Clarius / Telemed ArtUs / us4us
- Klassische Pipeline → Neue Pipeline (Splat-Fit + Gaussian3D Batch)
- Splat-native Konvertierungs-Engine — *Optimiert für CPU • Echtzeit • Deterministisch • Kliniktauglich*
- Outputs: Echo-Amplitude → Opacity, PSF → anisotrope Σ, Doppler → view-dependent Farbe, IMU → Multi-Frame-Integration, Splat-zu-Splat Registrierung
- **Warum das wichtig ist:** Direkt 3D statt 2D-B-Mode, AR-ready Output, CPU-only kliniktauglich, Kein GPU-Vendor-Lock
- **SaMD statt neuer Scanner:** Forschungstool → klinische Studie → Class IIa

### 1.2 Why this maps cleanly onto the existing substrate

- **"One splat substrate, never re-encoded, three operations"** ≡ `E-SOA-IS-THE-ONLY` R1 (cognitive-shader thinking / cold-path read/write / chain context). Direct analog.
- **"Same math for acquisition, registration, rendering"** ≡ "the object speaks for itself" / "thinking is a struct" (lance-graph CLAUDE.md "The Click").
- **"CPU-only, no GPU vendor lock"** ≡ ndarray SIMD substrate (AVX-512 / NEON / scalar dispatch, all consumer-runtime-selected). No new runtime infrastructure.
- **"Real-time 4D accumulation"** ≡ mailbox-owned SoA mutation as the only hot-path activity. Each frame is one ractor mailbox handoff (E-BATON-1: `(u16 target, CausalEdge64)`), accumulating into the per-mailbox `MailboxSoA<Gaussian3D>`.
- **"Splat-to-splat registration"** ≡ NARS revision: prior (atlas) + evidence (live frame) → revised pose. The Σ-sandwich Mahalanobis IS the confidence-weighted truth-revision math.
- **"SaMD Class IIa"** ≡ ADR-022 (The Firewall) discipline at the regulatory layer. The audit-controls + access-controls evidence base is already built; the regulatory wrapper is documentation, not architecture.

---

## 2. Workspace cross-cuts (where this lands, repo by repo)

| Repo | Surface that splat-native consumes | New surface owed |
|---|---|---|
| `ndarray` | `simd::*` AVX-512/NEON/scalar dispatch, `simd_exp_f32`, `permute_bytes`, BLAS L1/L2/L3 backend | **Anisotropic Gaussian batch ops** (3D Σ Cholesky, batched Mahalanobis, opacity blend, SH eval up to ℓ=3) — D-SPLAT-2 |
| `lance-graph-contract` | `MailboxSoA<N>` (per #434 unified-SoA), `CausalEdge64`, `KnowableFromStore` (OGAR #25/#31) | **`Gaussian3D` carrier** + `SplatBatch` SoA + `SplatPose` (SE(3) with IMU covariance) + `SplatRegistration` (pose optimizer state) — D-SPLAT-1/3 |
| `lance-graph` | SPO + `palette`/`bgz17` codec stack, `OntologyRegistry`, `LanceMembrane::commit_event` (callcenter PR #467) | **`SplatPalette`** codec extension + **`fma_atlas.lance`** dataset + **`splat::registration`** module — D-SPLAT-4/5 |
| `lance-graph-ontology` | OGAR FMA hydrator (PR #30 Phase 8), `style_recipe` pattern (PR #433), `odoo_blueprint::extracted` pattern (PR #426) | **`FmaEntity` typed SoA** + `fma_blueprint::style_recipe` DAtom catalogue — D-SPLAT-8/9 |
| `crates/splat-fit` (new, standalone) | ndarray SIMD, `Gaussian3D` carrier | **Splat-fit engine** (RF → beamform → splat-fit → Gaussian3D batch) — D-SPLAT-6 |
| `MedCare-rs` | `column_mask_bridge`, `soa_mapping.rs`, `commit_event` audit | **`memory.ultrasound_frame.lance`** dataset; `SensitivityReason::UltrasoundRawPHI` variant — D-SPLAT-10/11 |
| `cesium` / `blender` | rendering surfaces | **AR splat renderer** (HoloLens OpenXR target; Cesium ion + browser fallback) — D-SPLAT-12 |
| `ractor` / `ractor_actors` | mailbox + lifecycle hooks | **`SplatFitActor`** + **`PoseAccumulatorActor`** + **`RegistrationActor`** — D-SPLAT-7 |
| `bardioc` | Rubicon 4-phase kanban (PR #17) | **frame ratification gate** (Planning=pose-accumulate, Cognitive=splat-fit, Evaluation=ICP residual, Commit·Plan·Prune) — consumes, no new code |
| `OGAR` | §6 FMA litmus (PR #30), ADR-022 firewall | **OGAR-side FMA `Class` walk** (PR #30 Phase 8) — splat-native is the explicit customer that proves the litmus |
| `surrealdb` (fork) | `kv-lance` backend (BLOCKED OQ-11.6) | unaffected; splat volume stays in Lance directly |
| `q2` / `quarto` | docs publication | **SaMD documentation track** (research-tool → study → Class IIa) — D-SPLAT-14 |

---

## 3. Per-deliverable specifications

Each deliverable is one PR (or a small PR series). All include: tests (per-deliverable spec), board hygiene (LATEST_STATE / PR_ARC / STATUS_BOARD updates in the same commit), and the `I-LEGACY-API-FEATURE-GATED` discipline where applicable.

### D-SPLAT-1 — `Gaussian3D` carrier in `lance-graph-contract`

**Owner:** `lance-graph-contract`. **~120 LOC + tests.** **Risk: LOW.**

```rust
#[repr(C)]
pub struct Gaussian3D {
    pub mu: [f32; 3],          // 12 B — centroid in scanner-frame coords
    pub sigma: [f32; 6],       // 24 B — packed lower-tri Σ (Cholesky-ready)
    pub amplitude: f32,        //  4 B — peak echo amplitude
    pub opacity: u8,           //  1 B — amplitude-derived (per stage-3 mapping)
    pub _pad: [u8; 3],         //  3 B — alignment to 64 B with sh
    pub sh: [f16; 16],         // 32 B — degree-3 SH (4² coefficients) for color/Doppler
    pub frame_idx: u32,        //  4 B — provenance pointer into FrameRing
    pub class_id: u16,         //  2 B — FMA class via OntologyRegistry (0 = unclassified)
    pub _pad2: [u8; 2],        //  2 B
}
// Total: 80 B per Gaussian. 1 M Gaussians = 80 MB per mailbox (hot ceiling).
```

**Tests:** byte-layout golden test; round-trip via `MailboxSoAHeader` v(N); LE byte ordering on x86 + ARM.

**Gates on:** D-MBX-A2 (`MailboxSoAHeader` version gate from unified-soa-convergence-v1 §3.1) MUST land first OR D-SPLAT-1 lives behind its own feature flag.

### D-SPLAT-2 — `ndarray::simd::splat` batch ops

**Owner:** `ndarray`. **~600 LOC + tests + bench.** **Risk: MED.**

**Adds:**

- `batched_cholesky_3x3` over `[f32; 6]` packed lower-tri Σ — Cholesky factor for Mahalanobis evaluation; SIMD-batched 16-at-a-time (AVX-512) or 4-at-a-time (NEON).
- `batched_mahalanobis` — `(x - μ)ᵀ Σ⁻¹ (x - μ)` over N queries × M Gaussians; key kernel for splat-to-splat registration.
- `batched_opacity_blend` — front-to-back alpha composite over a sorted Gaussian sequence (CPU-side rasterization for the AR renderer).
- `batched_sh_eval_l3` — degree-3 spherical-harmonic evaluation over 16 coefficients per Gaussian; view direction in `[f32; 3]`; output `f32` luminance per view.
- `batched_se3_transform` — SE(3) rigid transform over N Gaussians (μ rotates + translates; Σ becomes `R Σ Rᵀ`).

**Tests:** correctness vs reference (Eigen / scipy); SIMD parity across dispatch backends; bench gates (≥ 1.4× SIMD payoff over scalar at N=1 M Gaussians).

**Gates on:** none (foundation work); can land in parallel with D-SPLAT-1.

### D-SPLAT-3 — `SplatBatch<N>` SoA carrier

**Owner:** `lance-graph-contract`. **~150 LOC + tests.** **Risk: LOW.**

`SplatBatch<N>` is the struct-of-arrays SoA view over `[Gaussian3D; N]` — for SIMD sweep. Per-column arrays: `mu_x: [f32; N]`, `mu_y: [f32; N]`, `mu_z: [f32; N]`, `sigma_xx/yy/zz/xy/xz/yz: [f32; N]`, `amplitude: [f32; N]`, `opacity: [u8; N]`, `sh_l0/l1m-1/...: [f16; N]` (16 columns), `frame_idx: [u32; N]`, `class_id: [u16; N]`.

Inherits `MailboxSoAHeader` versioning gate from unified-soa-convergence-v1 D-MBX-10. Same field-isolation-matrix discipline (`I-LEGACY-API-FEATURE-GATED`).

**Tests:** layout equivalence with `[Gaussian3D; N]`; SoA-AoS round-trip; field isolation matrix.

**Gates on:** D-SPLAT-1.

### D-SPLAT-4 — SH-aware palette extension in `crates/bgz17`

**Owner:** `bgz17`. **~250 LOC + tests.** **Risk: MED.**

The palette codec is currently `(centroid_palette: u8, edge_weight: u8, scent: u8)` (3-byte palette edge). For splat-native, the centroid carries Σ-eigenbasis + SH-basis-id alongside the existing mu:

- `centroid_mu: [f32; 3]` (centroid position)
- `centroid_sigma_eigenbasis: [f32; 3]` (eigenvalues; eigenvectors implicit from a shared 256-entry rotation table)
- `sh_basis_id: u8` (index into 256 SH basis fingerprints — typically: planar / spherical / Doppler-flow signatures)

**Compose-table size doubles** (from 256×256×1 B to 256×256×2 B) — still fits in 128 KB, still L1-resident.

**ADR-024 adoption (palette256 + HHTL codec, OGAR #39).** D-SPLAT-4 is named in `OGAR/docs/ARCHITECTURAL-DECISIONS-2026-06-04.md § ADR-024 Consequences` as one of the two queued ADR-024 adopters (paired with D-OSM-2 — see `cesium-osm-substrate-v1.md §11` for the geographic-side adoption note). Mapping ADR-024's four-step adoption checklist onto D-SPLAT-4:

1. **Prefix.** FMA NiblePath + SH basis-id — the per-region spatial frame (FMA class identity within the atlas + ℓ=3 SH basis index).
2. **Palette domain.** Quantized SH coefficients (16 coeffs at ℓ=3) clustered per FMA region. Per-region palette captures dominant tissue echo signatures (planar fascia / spherical organ surfaces / Doppler-flow vascular).
3. **ρ-vs-reference target ≥ 0.99**, matching the `lance-graph-arm-discovery` aerial-codebook empirical anchor (ρ = 0.9973 vs cosine). Reference: analytic SH evaluation at the same view direction (the same reference used in the D-SPLAT-4 test plan below). Report empirical ρ on the **first FMA-region palette build** (default region: femur cortical bone — smallest tractable corpus; cf. OGAR PR #30 §6 FMA-bones litmus).
4. **Decode = const-table lookup.** Per-region SH palette is runtime const-table; decode path is zero-allocation. Compile-time HHTL where the palette is shared across regions (e.g. global basis-ID lookup for the 256 SH signatures).

The 256-ceiling escape hatches from ADR-024 apply directly: per-region palettes are the cheapest answer (different FMA region, different 256 SH-signature entries); hierarchical palettes (coarser at OrganSystem level, finer per Bone leaf) mirror the SH ℓ=0/ℓ=1 vs ℓ=2/ℓ=3 split. Palette-64K is reserved for measured-cardinality escalation, not speculation.

**Tests:** palette compose against ground-truth Gaussians; SH-basis discrimination (cosine ≥ 0.95 against analytic SH on phantom data); **NEW: empirical ρ-vs-reference reported on first FMA-region palette build (target ≥ 0.99 per ADR-024 adoption contract)**; per-region palette cardinality distribution (mean / p95 / p99) at FMA region granularity.

**Gates on:** D-SPLAT-1.

### D-SPLAT-5 — splat-to-splat registration math (`lance-graph::splat::registration`)

**Owner:** `lance-graph`. **~400 LOC + tests + bench.** **Risk: HIGH.**

The "Σ-sandwich" pose optimizer from stage 6. Two pieces:

1. **Σ-sandwich Mahalanobis ICP** — for each live-volume Gaussian, find nearest atlas Gaussian by Mahalanobis distance using the *sum* covariance `Σ_atlas + Σ_live` (the "sandwich"). This is the covariance-aware analog of point-to-plane ICP.
2. **SE(3) pose optimizer** — Levenberg-Marquardt over the rigid body pose. Jacobians provided by `ndarray::simd::splat::batched_se3_transform`. NARS-style confidence weighting on per-correspondence Mahalanobis residuals.

**Tests:** convergence on synthetic phantom (ground-truth pose perturbed by [σ_translation = 5mm, σ_rotation = 5°] → recover within [1mm, 0.5°]); bench gate (< 100ms for 1M Gaussians × 1M atlas on a 16-core CPU).

**Gates on:** D-SPLAT-2 + D-SPLAT-3.

### D-SPLAT-6 — `crates/splat-fit` engine

**Owner:** new standalone crate `crates/splat-fit`. **~1500 LOC + tests + Field-II validation.** **Risk: HIGH.**

The splat-fit step in the pipeline diagram. Inputs: beamformed RF or IQ (depending on probe — see OQ-SPLAT-3); per-scanline channel data optional. Outputs: `SplatBatch<N>` of fitted Gaussians.

Algorithm (sketch — pinned in implementation review):

1. **Local-maxima detection** in the beamformed envelope (scale-space peak finder; ~10⁴–10⁶ peaks per frame).
2. **PSF estimation** at each peak — locally fit anisotropic Gaussian to the 3D point spread function (axial = pulse-length-dominated; lateral = aperture-dominated). Yields Σ.
3. **Amplitude** from peak envelope value → maps to opacity per stage-3 curve.
4. **Frequency / Doppler content** from local FFT or autocorrelation → projects onto SH ℓ=3 basis (16 coefficients).
5. **Emit** as `Gaussian3D` in scanner-frame coordinates (pose transform applied later by `PoseAccumulatorActor`).

**Zero-dep crate** (mirror `deepnsm` / `bgz17` pattern: stays excluded from workspace until matured). Reuses ndarray SIMD via the `ndarray-hpc` feature flag.

**Tests:** validation against Field II reference data (industry-standard ultrasound simulator); ground-truth phantom recovery (synthetic Σ → fit → recovered Σ within 5% Frobenius); bench gate (≥ 30 frames/sec at typical clinical resolution on 16-core CPU).

**Gates on:** D-SPLAT-1 + D-SPLAT-2; **OQ-SPLAT-3** (do we ship our own beamformer or consume probe's?).

### D-SPLAT-7 — Splat actors in `ractor_actors`

**Owner:** new actors in `ractor_actors` (or a new `splat-actors` crate). **~500 LOC + tests.** **Risk: MED.**

Three actors, each owns one `MailboxSoA<Gaussian3D>`:

- **`SplatFitActor`** — consumes RF/IQ frames; emits `SplatBatch` to its mailbox; baton (E-BATON-1) handoff to `PoseAccumulatorActor`.
- **`PoseAccumulatorActor`** — consumes IMU + splat-batch; integrates pose; emits pose-aligned `SplatBatch` to its mailbox; baton handoff to `RegistrationActor`.
- **`RegistrationActor`** — runs D-SPLAT-5 against `fma_atlas.lance`; emits registration result + pose; baton handoff to renderer.

**Inherits the 4-phase Rubicon kanban** (bardioc PR #17):

| Kanban column | Splat actor activity | Libet anchor |
|---|---|---|
| Planning | pose accumulator integrates IMU (counterfactual: where would we be after this frame?) | t < −550 ms |
| Cognitive work | splat-fit + Σ-sandwich registration (the splat volume mutates) | t ≥ −550 ms |
| Evaluation | ICP residual + truth-revised confidence | t > 0 |
| Commit · Plan · Prune | commit (calcify to `ultrasound_frame.lance`) / plan (re-acquire) / prune (drop low-quality frame) | terminal |

**Tests:** end-to-end frame flow; baton round-trip; kanban transitions; mailbox SoA invariants.

**Gates on:** D-SPLAT-3 + D-SPLAT-6 + bardioc PR #17 already shipped.

### D-SPLAT-8 — FMA atlas hydrator

**Owner:** `lance-graph-ontology` + new `crates/fma-hydrator`. **~800 LOC + tests.** **Risk: HIGH.**

Accelerates **OGAR PR #30 Phase 8** (FMA hydrate). Loads FMA TTL (75K classes, 2.1M relationships) into a `fma_atlas.lance` dataset; emits typed `FmaEntity` SoA (mirrors `OdooEntity` pattern from PR #426/#432); pre-computes the atlas splat volume per anatomical region.

**Datasets emitted:**

- `fma_class.lance` — one row per FMA class with NiblePath identity (OGIT-prefixed: `ogit-fma/Bone#7474`, etc.)
- `fma_relation.lance` — class-class relations (`isA`, `partOf`, `innervates`, `supplies`, etc.)
- `fma_atlas_splat.lance` — pre-computed atlas splat volume; one row per Gaussian (~150M rows full body, ~5 GB compressed via D-SPLAT-4 palette).

**Tests:** round-trip FMA TTL → Lance → query (`ogit-fma/Muscle#9663 partOf?`); sub-millisecond NiblePath traversal (the §6 litmus); atlas splat volume rendering against a known anatomical region.

**Gates on:** OGAR PR #30 Phase 8 (FMA TTL prepared upstream) + D-SPLAT-3 + ndarray PR #189 (`OntologySchema::is_ancestor`, already shipped).

### D-SPLAT-9 — `fma_blueprint::style_recipe` D-Atom catalogue

**Owner:** `lance-graph-ontology`. **~400 LOC + tests.** **Risk: LOW.**

Mirrors the `odoo_blueprint::style_recipe` pattern from PR #433. FMA-specific D-Atoms:

- `AnatomicalRegion` (head / thorax / abdomen / limb)
- `OrganSystem` (cardiovascular / nervous / digestive / etc.)
- `Innervation` (motor / sensory / autonomic)
- `Vasculature` (artery / vein / capillary)
- `Joint` (synovial / cartilaginous / fibrous)
- `Muscle` (skeletal / cardiac / smooth)
- `Bone` (cortical / cancellous)
- `OrganParenchyma` (the soft tissue inside an organ)
- `Tract` (white-matter or fluid pathway)

Each FMA class gets a cognitive fingerprint = sparse weighted vector over these DAtoms + regulatory anchors (PROV-O, BFO upper-ontology axes per OGAR ADR alignment).

**Tests:** mirrors `style_recipe.rs` test suite — every-DAtom-anchor case + recipe_id determinism + atom collapse + corpus sort.

**Gates on:** D-SPLAT-8.

### D-SPLAT-10 — `memory.ultrasound_frame.lance` dataset

**Owner:** `MedCare-rs` (`crates/medcare-analytics`). **~250 LOC + tests.** **Risk: MED.**

New SoA mapping entry per the existing pattern in `soa_mapping.rs`. Columns:

- `patient_ref: FixedSizeBinary(8)` — anonymized patient ID (per `column_mask_bridge` Hash mode for unauthorized roles)
- `frame_idx: u32` — monotonic frame counter
- `acquisition_ts: TimestampMicros` — frame acquisition time
- `pose_se3: FixedSizeBinary(24)` — packed SE(3) pose (3 translation + 9 rotation matrix, both `f16`; 12 × 2 = 24 B)
- `splat_batch_handle: Binary` — pointer into the splat-volume storage (NOT the splats themselves; those live in `ultrasound_splat.lance` to keep this table queryable)
- `splat_count: u32`
- `mean_amplitude: f32`
- `quality_score: f32` — ICP residual after registration; below threshold ⇒ frame is pruned (kanban col 4)

**Raw RF/IQ stays out of MedCare-rs entirely** — only fitted splats land. Per **NR-SPLAT-PHI** (normative rule, see below): scanner-frame splat geometry is non-identifying on its own; the link `patient_ref ↔ splat_volume` in `memory.ultrasound_frame.lance` IS PHI; atlas-aligned anatomical annotations carrying patient-specific landmarks ARE PHI. The `column_mask_bridge` extension adds:

```rust
pub enum SensitivityReason {
    // ... existing variants ...
    UltrasoundRawPHI,      // → RedactionMode::Hash for non-clinical roles
    UltrasoundAnonymized,  // → RedactionMode::Constant("[REDACTED]")
}
```

**Tests:** SoA round-trip; column-mask enforcement per `Physician/Nurse/Cashier/Researcher/HipaaAudit/Admin` role; ICD code cross-reference with `memory.diagnosis.lance` via `patient_ref`.

**Gates on:** D-SPLAT-3 + MedCare PR #162 (handover with `column_mask_bridge` patterns) already shipped.

### D-SPLAT-11 — `commit_event` audit for splat ingest

**Owner:** `MedCare-rs` (`crates/medcare-analytics`). **~100 LOC + tests.** **Risk: LOW.**

**NR-SPLAT-PHI (normative rule, single source of truth):** *Scanner-frame splat geometry (the `mu`/`sigma_packed` columns; `pose_se3` in scanner frame) is non-identifying on its own. The link between `patient_ref` and a splat volume — i.e. the row in `memory.ultrasound_frame.lance` — IS PHI. Atlas-aligned annotations that name patient-specific landmarks (e.g. `class_id` resolving to "this patient's left biceps brachii at scanner-pose P") are PHI. Raw RF/IQ is PHI by default and is never persisted (it does not enter the MedCare wire). All §3.10 (`column_mask_bridge`) and §7 (risk-matrix HIPAA row) policy decisions cite this rule.*

Every splat ingest writes one `CognitiveEventRow` via `LanceMembrane::commit_event` (callcenter PR #467, the sole-writer membrane). Carries: `actor` (the `SplatFitActor`), `action: "ultrasound_ingest"`, `target_class: "memory.ultrasound_frame"`, `target_row_id`, `version` (the Lance row-version from `KnowableFromStore::register`), `subject` (clinician identity), `consent_evidence` (the patient consent chain). Inner-side; no JSONL audit sink (per MedCare CLAUDE.md Iron Rule 7).

**Tests:** every ingest produces exactly one `CognitiveEventRow`; row-version monotonic; consent-chain non-empty.

**Gates on:** D-SPLAT-10 + callcenter PR #467 already shipped.

### D-SPLAT-12 — AR splat renderer

**Owner:** new `crates/splat-render` (or `cesium`/`blender` integration). **~1200 LOC + tests + bench.** **Risk: HIGH.**

The renderer is **stupid** by design — it reads the splat volume zero-copy via `mmap` over `fma_atlas.lance` + live `ultrasound_frame.lance`, sorts front-to-back by view direction, calls `ndarray::simd::splat::batched_opacity_blend`, paints. All math is done by the time data reaches it; the renderer's only job is rasterization.

Three render targets:

1. **HoloLens OpenXR** — the clinical AR target. WebXR-compat render loop; CPU-only; runs in the device's standard SaMD compute budget.
2. **Cesium ion + Three.js fallback** — browser-based prototype for demos and the clinical-study phase.
3. **Headless PNG** — for regression testing + clinical-study screenshot evidence.

**Tests:** golden-image regression on phantom data; pose-tracking accuracy (atlas-aligned overlay drift < 1mm at 30fps); render latency (< 33ms per frame at 1024×1024).

**Gates on:** D-SPLAT-2 + D-SPLAT-3 + D-SPLAT-5.

### D-SPLAT-13 — IMU/POSE 4D accumulator

**Owner:** `PoseAccumulatorActor` (D-SPLAT-7 sub-deliverable). **~200 LOC + tests.** **Risk: MED.**

Tight-coupled IMU + visual-inertial odometry against the splat volume itself (the splats serve as the visual-feature substrate; no separate VSLAM stack). Pose updates at IMU rate (~200 Hz); splat-fit ratchets at frame rate (~30 Hz). Pose-aligned frame fusion uses the most-recent pose estimate at the splat-fit deadline.

**Tests:** IMU integration drift bound (< 1 cm over 10s without splat correction); splat-corrected pose stability (< 1 mm over 60s static probe); kanban Planning-column readiness (pose state available at t = −550ms).

**Gates on:** D-SPLAT-7.

### D-SPLAT-14 — SaMD documentation track

**Owner:** `q2` / `quarto` (or repo docs). **~600 LOC across multiple docs.** **Risk: LOW (regulatory).**

Three docs:

- **`docs/SAMD-INTENDED-USE.md`** — v1 intended use: research tool only (no CE/FDA scope). v2: clinical-study scope (IEC 62366 usability + IEC 80001 risk). v3: Class IIa target (Rule 11 IVD-MDR / FDA 510(k)).
- **`docs/SAMD-RISK-CLASSIFICATION.md`** — ISO 14971 risk file; maps each hazard to mitigation evidence. The ADR-022 firewall discipline + `commit_event` audit + `KnowableFromStore` registry IS the audit-controls + access-controls evidence base — cited directly, not re-derived.
- **`docs/SAMD-AUDIT-TRAIL.md`** — how a regulator queries the system: `commit_event` row chain + `KnowableFromStore` version trace + Lance dataset version history. Worked example: "show me every access to patient X's ultrasound frames in the last 30 days" → one Lance query against `CognitiveEventRow` filtered by `subject` + `target_class`.

**Tests:** none (regulatory documentation); cross-reference link integrity; ADR cross-ref completeness.

**Gates on:** none architecturally; v1 ships with the research-tool path; v2/v3 phased with clinical-study and certification milestones.

---

## 4. Migration phases — sequenced gating

### Phase P1 — Substrate (sprint 1-2)

- **D-SPLAT-1** (`Gaussian3D` carrier in contract)
- **D-SPLAT-2** (ndarray SIMD splat ops)
- **D-SPLAT-3** (`SplatBatch<N>` SoA)

**Output:** the carrier + math primitives. Nothing renders yet; nothing fits yet. But every downstream phase has its hardware floor.

**Acceptance:** `cargo bench --bench splat_simd` shows ≥ 1.4× speedup over scalar at N=1M Gaussians.

### Phase P2 — Splat-fit engine (sprint 3)

- **D-SPLAT-6** (`crates/splat-fit` engine)

**Output:** RF/IQ in → `SplatBatch` out. Validated against Field II phantom data.

**Acceptance:** ≥ 30 fps single-frame fit on 16-core CPU; phantom recovery within 5% Frobenius.

### Phase P3 — Actors + multi-frame (sprint 4-5)

- **D-SPLAT-7** (splat actors)
- **D-SPLAT-13** (IMU/POSE 4D accumulator)
- **D-SPLAT-4** (SH palette codec extension)

**Output:** end-to-end frame → fitted splats → pose-accumulated → mailbox. No registration yet; no atlas yet.

**Acceptance:** kanban round-trip on synthetic phantom sequence; pose drift bound.

### Phase P4 — FMA atlas + registration (sprint 6-8)

- **D-SPLAT-8** (FMA atlas hydrator)
- **D-SPLAT-9** (FMA `style_recipe`)
- **D-SPLAT-5** (splat-to-splat registration)

**Output:** the §6 FMA litmus passes. Live frame registers against atlas in real-time.

**Acceptance:** sub-millisecond NiblePath traversal on FMA TTL; < 100ms registration on 1M × 1M Gaussian volumes; ICP convergence on synthetic phantom.

### Phase P5 — HIPAA wire (sprint 9-10)

- **D-SPLAT-10** (`memory.ultrasound_frame.lance` dataset)
- **D-SPLAT-11** (`commit_event` audit)

**Output:** MedCare-rs consumes the splat volume; PHI-safe persistence + audit chain.

**Acceptance:** every role / table combination enforced by `column_mask_bridge`; every ingest writes exactly one `CognitiveEventRow`.

### Phase P6 — AR surface (sprint 11-13)

- **D-SPLAT-12** (AR splat renderer)

**Output:** the clinical demo. Patient-aligned overlay of atlas + live volume.

**Acceptance:** < 33ms frame latency at 1024×1024; HoloLens OpenXR render; Cesium browser fallback.

### Phase P7 — SaMD certification (sprint 14+; parallel through P4-P6)

- **D-SPLAT-14** (SaMD documentation)

**Output:** regulatory wrapper. v1 research-tool intended-use; v2 clinical-study; v3 Class IIa.

**Acceptance:** doc completeness for the v1 milestone (research tool); IEC 62366 usability file for v2; Class IIa technical file for v3.

---

## 5. Dependencies graph (textual)

```text
OGAR #30 Phase 8 (FMA hydrate prep) ─────────┐
                                             ▼
ndarray D-SPLAT-2 (SIMD splat ops) ──► D-SPLAT-8 (FMA atlas) ──┐
       │                                                       │
       ▼                                                       ▼
D-SPLAT-1/3 (carriers) ──► D-SPLAT-6 (splat-fit) ──► D-SPLAT-7 (actors)
       │                                                       │
       ▼                                                       ▼
D-SPLAT-4 (SH palette) ────────────────────────────► D-SPLAT-5 (registration)
                                                               │
                                  ┌────────────────────────────┼────────────────────────────┐
                                  ▼                            ▼                            ▼
                          D-SPLAT-10/11                  D-SPLAT-12                   D-SPLAT-14
                       (MedCare HIPAA)                  (AR renderer)             (SaMD regulatory)
```

---

## 6. Open questions

| OQ | Question | Blocks | Default proposal |
|---|---|---|---|
| **OQ-SPLAT-1** | Which probe SDK first? Clarius (cloud-tethered REST) / Telemed ArtUs (USB raw-data driver) / us4us (open Verasonics-compatible)? | D-SPLAT-6 | **Telemed ArtUs first** — gives RF directly, USB-attached, no cloud round-trip, lowest regulatory friction for research-tool phase. |
| **OQ-SPLAT-2** | SH degree budget — ℓ=3 (16 coefficients) vs ℓ=2 (9)? | D-SPLAT-1 / D-SPLAT-4 | **ℓ=3 default; revisit at D-SPLAT-12 bench** — gives realistic Doppler/color rendering but doubles per-Gaussian storage vs ℓ=2. |
| **OQ-SPLAT-3** | Ship our own beamformer or consume probe's beamformed output? | D-SPLAT-6 | **Consume probe's beamformed RF when available (Telemed); ship a delay-and-sum beamformer as fallback for Clarius (B-mode only).** |
| **OQ-SPLAT-4** | Does the AR-rendered splat volume need to leave the device? | D-SPLAT-12 | **No** — everything stays on the HoloLens; renderer reads `mmap` directly; audit is local. If yes later, that's a separate `ExternalMembrane` extension. |
| **OQ-SPLAT-5** | Which canonical home for the plan? | (this plan) | **lance-graph canonical; companion docs in ndarray + OGAR + MedCare-rs** (this plan). |

---

## 7. Risk matrix

| Risk | Severity | Mitigation |
|---|---|---|
| Telemed ArtUs SDK access (research-tool phase) | HIGH | Pre-engage Telemed for SDK access during P1 substrate work; fall back to Clarius (B-mode only, accepts D-SPLAT-6 fallback beamformer) if Telemed slips. |
| Splat-fit doesn't converge on real clinical data (vs phantom) | HIGH | Field II validation gates P2; if phantom-to-clinical gap is large, fall back to D-SPLAT-12 rendering of raw envelope (degraded but functional) while iterating splat-fit. |
| FMA atlas splat pre-computation too large (~5 GB) | MED | Region-on-demand loading (per OQ-SPLAT-X); only the regions matching `class_id` of fitted live splats need to be paged in. Lance versioning + `KnowableFromStore` registry handles the on-demand path. |
| Registration convergence basin too narrow (live ≠ atlas anatomy) | HIGH | Coarse-to-fine multi-resolution ICP (lower SH degree first); patient-specific atlas pre-registration during clinical-study phase. |
| HoloLens OpenXR compute budget exhausted | MED | Render-at-IMU-rate (200 Hz) is not required; render-at-frame-rate (30 Hz) is. Backpressure via the renderer mailbox. |
| HIPAA PHI leak via splat coordinates | MEDIUM | Per **NR-SPLAT-PHI** (§3.10): scanner-frame splat coordinates are non-identifying on their own; PHI lives in the `patient_ref ↔ splat_volume` link and in atlas-aligned annotations carrying patient-specific landmarks. Default `column_mask_bridge` mode for `patient_ref` is `Hash` for non-clinical roles; splat coordinate columns require no masking. |
| SaMD Class IIa technical-file gap | MED | Build the audit-controls evidence chain during P5 (HIPAA wire); the chain IS the certification evidence base. |

---

## 8. Success criteria

### Per-phase acceptance

- **P1 (substrate):** ≥ 1.4× SIMD speedup on splat ops; LE byte-layout golden tests green.
- **P2 (splat-fit):** ≥ 30 fps fit; phantom recovery within 5% Frobenius.
- **P3 (actors):** kanban round-trip on phantom sequence; pose drift < 1cm/10s.
- **P4 (atlas + registration):** sub-millisecond NiblePath traversal; ICP convergence on synthetic phantom; < 100ms registration on 1M × 1M.
- **P5 (HIPAA):** every role × table combination column-mask-enforced; one `CognitiveEventRow` per ingest.
- **P6 (AR):** < 33ms frame latency; HoloLens OpenXR demo; Cesium browser fallback.
- **P7 (regulatory):** v1 SaMD intended-use doc; v2 clinical-study technical file; v3 Class IIa submission.

### Workspace-level acceptance

- `cargo check --workspace` clean across the splat-affecting crates (ndarray + lance-graph-contract + lance-graph + lance-graph-ontology + splat-fit + splat-actors + medcare-analytics).
- `cargo test --workspace` green.
- **Same math signature** (D-SPLAT-2 ndarray ops) is the only floating-point math in the splat-fit, registration, and renderer paths — verified by `grep` audit.
- **No raw RF/IQ in MedCare-rs storage** — verified by `grep` audit against `crates/medcare-analytics/`.
- **AR-rendered splat volume zero-copy from Lance** — verified by `mmap` trace in the renderer.

---

## 9. Cross-references

- **Diagrams** (this plan's trigger):
  - English 6-stage technical pipeline
  - German business-facing variant with SaMD Class IIa pathway

- **Plans:**
  - `lance-graph/.claude/plans/unified-soa-convergence-v1.md` — the SoA carrier doctrine splat-native inherits
  - `lance-graph/.claude/plans/unified-soa-convergence-v1-addendum-2026-05-29-review.md` — post-merge review
  - `lance-graph/.claude/plans/bindspace-singleton-to-mailbox-soa-v1.md` — the §11 layered rulings
  - `lance-graph/.claude/plans/causaledge64-mailbox-rename-soa-v1.md` — Baton + Mailbox-as-owner

- **Epiphanies:**
  - `E-SOA-IS-THE-ONLY` — one substrate, three operations
  - `E-BATON-1` — discrete owned handoff carrier
  - `E-MAILBOX-IS-BINDSPACE` — mailbox = full BindSpace as LE
  - `E-RUBICON-RACTOR` — Heckhausen + Libet grounding of the 4-phase kanban
  - `E-CE64-MB-4` — mailbox-as-owner compile-time UB impossibility
  - `E-CONTRACT-NO-SERIALIZE` — the audit witness stays inner

- **Iron rules:**
  - `I-SUBSTRATE-MARKOV` — Chapman-Kolmogorov (splat-fit + pose accumulation is a Markov trajectory)
  - `I-VSA-IDENTITIES` — bundle identities not content (splat class_id points to FMA URI; never bundle Gaussians themselves)
  - `I-LEGACY-API-FEATURE-GATED` — governs the `MailboxSoAHeader` version gate on `Gaussian3D` schema upgrades
  - `I-NOISE-FLOOR-JIRAK` — for any threshold (Mahalanobis significance, ICP convergence) cite Jirak under weak dependence

- **PRs (recent context):**
  - lance-graph PR #434 — unified-soa-convergence-v1 (E-SOA-IS-THE-ONLY)
  - lance-graph PR #470 (open) — BindSpace dissolution architectural delta
  - OGAR PR #30 — RDF/OWL alignment + §6 FMA bones-rendering litmus (the explicit customer)
  - OGAR PR #25 / #31 — `KnowableFromStore` trait (splat ingest registers as a knowable schema source)
  - MedCare PR #162 — HIPAA architectural-delta handover (D-SPLAT-10/11 inherits the firewall)
  - bardioc PR #17 — Rubicon kanban impl (splat actors consume)
  - callcenter PR #467 — `LanceMembrane::commit_event` sole-writer (audit trail home)
  - ndarray PR #189 — `OntologySchema::is_ancestor` (FMA atlas traversal uses this)

---

## 10. Repository perspectives — work division + interconnect map

This plan touches **seven repositories**. Each owns a piece of the splat-native arc; **none can ship in isolation**; the interconnect points are the integration discipline. From each repo's vantage:

### 10.1 `adaworldapi/ndarray` — the SIMD substrate

**What we own:** D-SPLAT-2 (`ndarray::simd::splat` batch ops). Five primitives: `batched_cholesky_3x3`, `batched_mahalanobis`, `batched_opacity_blend`, `batched_sh_eval_l3`, `batched_se3_transform`. All AVX-512 / NEON / scalar dispatch via existing `simd_caps()` singleton.

**What we consume from upstream:** nothing (we are the substrate).

**What we feed downstream:**
- `lance-graph-contract` D-SPLAT-1/3 (`Gaussian3D` carrier + `SplatBatch<N>` SoA layout aligned to our SIMD load width).
- `crates/splat-fit` D-SPLAT-6 (every floating-point op in the fit path).
- `lance-graph::splat::registration` D-SPLAT-5 (Σ-sandwich Mahalanobis + LM optimizer).
- `crates/splat-render` D-SPLAT-12 (opacity blend + SH eval).

**Critical interconnect:**
- D-SPLAT-2 lands **before** D-SPLAT-1/3 — the carrier layout (Σ packed lower-tri vs SoA columns) must respect our SIMD load width (16-lane f32 on AVX-512, 4-lane on NEON). If the layout doesn't match, we lose 2-4× perf. This is a contract negotiation at design time, not a runtime dispatch decision.

**Sprint commitment:** Sprint 1-2 (P1 substrate). Single PR.

**Definition of done:** `cargo bench --bench splat_simd` on Skylake-X (AVX-512) shows ≥ 2× scalar; on Apple M2 (NEON) shows ≥ 1.4× scalar; Cholesky numerical equivalence to Eigen reference within 1 ulp; SH eval matches scipy `sph_harm` within 1e-5.

### 10.2 `adaworldapi/lance-graph` — the spine

**What we own:**
- D-SPLAT-1 (`Gaussian3D` carrier in `lance-graph-contract`)
- D-SPLAT-3 (`SplatBatch<N>` SoA carrier in `lance-graph-contract`)
- D-SPLAT-4 (SH-aware palette extension in `crates/bgz17`)
- D-SPLAT-5 (splat-to-splat registration in `lance-graph::splat::registration`)
- D-SPLAT-8 (FMA atlas hydrator emitting `fma_class.lance`/`fma_relation.lance`/`fma_atlas_splat.lance`)
- D-SPLAT-9 (`fma_blueprint::style_recipe` DAtom catalogue)

**What we consume from upstream:**
- `ndarray` D-SPLAT-2 (all the SIMD math)
- `OGAR` PR #30 Phase 8 (FMA TTL prep + `ogar-fma` crate scaffold)
- `OGAR` PR #25/#31 (`KnowableFromStore` for the splat-volume registry)

**What we feed downstream:**
- `crates/splat-fit` D-SPLAT-6 (carrier + SH palette + ontology lookup)
- `ractor_actors` D-SPLAT-7 (`MailboxSoA<Gaussian3D>` discipline)
- `MedCare-rs` D-SPLAT-10/11 (carrier + `LanceMembrane::commit_event`)
- `crates/splat-render` D-SPLAT-12 (atlas + live splat volumes)

**Critical interconnect:**
- D-SPLAT-1 layout MUST be ratified with ndarray D-SPLAT-2 author before either commits — single design review, two-repo handshake.
- D-SPLAT-5 registration math sits on top of D-SPLAT-2 Mahalanobis; correctness is validated by a joint test (synthetic Σ + perturbed pose → recover within tolerance) that lives in lance-graph but uses the ndarray ground-truth math as oracle.
- D-SPLAT-8 FMA atlas dataset is emitted in lance-graph but the FMA TTL source comes from OGAR; the cross-repo data handshake is `ogar-fma` → `fma_class.lance` (one-way producer; no back-edge).

**Sprint commitment:** D-SPLAT-1/3 sprint 1-2 (parallel with ndarray); D-SPLAT-4 sprint 3; D-SPLAT-5 sprint 6-7; D-SPLAT-8/9 sprint 6-8 (parallel after OGAR Phase 8 unblocks).

### 10.3 `adaworldapi/OGAR` — the FMA atlas customer

**What we own:**
- The §6 FMA bones-rendering litmus narrative (already in PR #30; this plan is its customer-side proof).
- `crates/ogar-fma` (Phase 8 of OGAR PR #30 sequencing) — FMA TTL → `Class` walk; emits typed FMA classes into a Lance-compatible producer surface.
- ADR-022 + `KnowableFromStore` registry are the architectural foundation; the splat ingest is one new registered schema source.

**What we consume from upstream:**
- W3C FMA OWL/TTL release (external; one-time download + vendor under `ogar-fma/vocab/`)
- `lance-graph-contract::KnowableFromStore` (already shipped in OGAR PR #25/#31 — we register-and-discover, the registry assigns the monotonic version)

**What we feed downstream:**
- `lance-graph-ontology` D-SPLAT-8 (FMA typed entities → `fma_class.lance` hydration)
- The §6 litmus IS the splat-native arc's keystone — proving sub-millisecond HHTL traversal on real biomedical-scale ontology data.

**Critical interconnect:**
- OGAR Phase 8 (FMA hydrator) and lance-graph D-SPLAT-8 (FMA atlas Lance dataset) are tightly coupled — same upstream FMA data, different downstream representations. **They should ship as a coordinated wave** (OGAR Phase 8 → lance-graph D-SPLAT-8 within one sprint cadence, not interleaved with other work).
- The "splat-to-splat registration" closes the loop with the existing OGAR `commit_event` / `KnowableFromStore` pattern — every successful registration writes one `CognitiveEventRow` per ADR-008 receipts. **No new audit surface** — reuse the firewall.

**Sprint commitment:** OGAR Phase 8 = sprint 6 (own work); lance-graph D-SPLAT-8 = sprint 7-8 (consumes our output). v1 customer demo = sprint 8.

### 10.4 `adaworldapi/MedCare-rs` — the HIPAA consumer

**What we own:**
- D-SPLAT-10 (`memory.ultrasound_frame.lance` dataset in `crates/medcare-analytics::soa_mapping`)
- D-SPLAT-11 (`commit_event` audit chain via `LanceMembrane`)
- `SensitivityReason::UltrasoundRawPHI` + `SensitivityReason::UltrasoundAnonymized` variants in `column_mask_bridge`
- Per-role enforcement matrix (Physician/Nurse/Cashier/Researcher/HipaaAudit/Admin × the new ultrasound table)
- The HIPAA worked example for the §6 SaMD audit-trail evidence base

**What we consume from upstream:**
- `lance-graph-contract` D-SPLAT-1/3 (the carrier + SoA layout)
- `lance-graph` D-SPLAT-8/9 (FMA classification for `class_id` resolution)
- `lance-graph-callcenter` PR #467 (`LanceMembrane::commit_event` sole-writer membrane) — **already shipped**
- `crates/splat-fit` D-SPLAT-6 (the splat-fit engine itself runs upstream of MedCare; we only persist the fitted output)
- `ractor_actors` D-SPLAT-7 (`SplatFitActor` lives upstream too; MedCare is the post-fit cold persistence + access-control boundary)

**What we feed downstream:**
- `crates/splat-render` D-SPLAT-12 (anonymized splat volume reads for the AR overlay)
- SaMD evidence base D-SPLAT-14 (the audit chain IS the regulatory evidence — no separate compliance docs needed)

**Critical interconnect:**
- **Raw RF/IQ NEVER enters MedCare storage** — verified by `grep` audit. This is a hard invariant (Iron Rule 7 + Iron Rule from MedCare CLAUDE.md): audit witness stays inner; raw signal stays at the splat-fit boundary; only fitted Gaussians cross into MedCare.
- The MedCare ultrasound dataset is **append-only from the SplatFitActor's perspective** — the actor is the sole writer; MedCare only reads + redacts on query. This preserves the sole-writer membrane discipline from PR #467.
- Per **NR-SPLAT-PHI** (§3.10): the patient-identity boundary is the `patient_ref ↔ splat_volume` LINK (Hashed for non-clinical roles); splat coordinates are in *scanner frame* and are not patient-identifying on their own; atlas-aligned annotations carrying patient-specific landmarks ARE PHI and inherit `patient_ref`'s sensitivity class via foreign-key join.

**Sprint commitment:** sprint 9-10 (P5, after FMA atlas + registration land upstream). 1 PR (D-SPLAT-10 + D-SPLAT-11 bundled).

### 10.5 `adaworldapi/bardioc` — the Rubicon kanban

**What we own:**
- Rubicon 4-phase kanban (PR #17, already shipped)
- The Σ10 commit timing anchor at t = −550 ms wall-clock (per the Libet anchor in lance-graph #434 R3)

**What we consume from upstream:** nothing (we are the kanban substrate).

**What we feed downstream:**
- D-SPLAT-7 (splat actors map their lifecycle onto Planning → Cognitive work → Evaluation → Commit·Plan·Prune)

**Critical interconnect:**
- The splat-fit kanban grammar is **inherited verbatim** from bardioc PR #17. We add zero new kanban surface; we just route splat domain semantics through the existing columns.
- The "frame ratification gate" (D-SPLAT-7 column 4: Commit/Plan/Prune) is the splat-domain instance of the Rubicon decision — same code path, different domain payload.

**Sprint commitment:** none new (already shipped). One handover note when D-SPLAT-7 actor lifecycle pattern lands.

### 10.6 `adaworldapi/cesium` (or new `splat-render` repo) — the AR renderer

**What we own:**
- D-SPLAT-12 (AR splat renderer with three targets: HoloLens OpenXR / Cesium ion + Three.js / headless PNG)
- The "renderer is stupid" discipline — no math, just rasterize

**What we consume from upstream:**
- ndarray D-SPLAT-2 (`batched_opacity_blend` + `batched_sh_eval_l3`)
- lance-graph D-SPLAT-3 (`SplatBatch` mmap-readable carrier)
- lance-graph D-SPLAT-8 (`fma_atlas_splat.lance` for the atlas overlay)
- MedCare D-SPLAT-10 (`memory.ultrasound_frame.lance` for the live splat volume, with column-mask enforcement applied by the read path)

**What we feed downstream:** the clinical AR experience itself.

**Critical interconnect:**
- The renderer reads **both** atlas and live datasets via zero-copy `mmap`; it never decodes or transforms; all math is upstream. Verified by `mmap`-trace in the test suite.
- HoloLens render budget = 11ms (90 Hz target); we target 33ms (30 Hz frame fit rate); 22ms slack for the OpenXR runtime + display pipeline.
- WebGPU/wgpu is OK for the GPU painting step **only**; the CPU still does Σ-eval, opacity blend, sorting. No GPU vendor lock at the math layer.

**Sprint commitment:** sprint 11-13 (P6). 1 PR per target (HoloLens / Cesium / headless).

### 10.7 `adaworldapi/q2` or `quarto` — the docs publication

**What we own:** D-SPLAT-14 (SaMD documentation track; research-tool → study → Class IIa).

**What we consume from upstream:** the audit-controls + access-controls evidence base materialized by the other repos (it's already built; the docs cite, not re-derive).

**What we feed downstream:** the regulatory submission package (CE / FDA, jurisdiction-specific, out of this plan's scope).

**Sprint commitment:** sprint 14+ (parallel through P4-P6 as the evidence base materializes).

---

## 10.8 The interconnect map (visual)

```text
                    ┌──────────────────────────────────────────┐
                    │  ndarray (D-SPLAT-2)                     │
                    │  SIMD substrate: Cholesky, Mahalanobis,  │
                    │  opacity blend, SH eval, SE(3) xform     │
                    └──────────────────────────────────────────┘
                       │                  │                  │
                       │ (SIMD)           │ (SIMD)           │ (SIMD)
                       ▼                  ▼                  ▼
        ┌────────────────────────┐  ┌──────────────┐  ┌──────────────┐
        │  lance-graph-contract  │  │  splat-fit   │  │  splat-render│
        │  D-SPLAT-1/3           │  │  D-SPLAT-6   │  │  D-SPLAT-12  │
        │  Gaussian3D + SoA      │  │  RF→splats   │  │  AR renderer │
        └────────────────────────┘  └──────────────┘  └──────────────┘
            │           │                │                  ▲
            │           │ (carrier)      │ (carrier)        │ (mmap)
            ▼           ▼                ▼                  │
  ┌────────────────┐  ┌──────────────────┐                  │
  │  bgz17         │  │  ractor_actors   │                  │
  │  D-SPLAT-4     │  │  D-SPLAT-7/13    │                  │
  │  SH palette    │  │  Splat actors    │                  │
  └────────────────┘  └──────────────────┘                  │
            │                  │                            │
            │                  │ (baton)                    │
            │                  ▼                            │
            │       ┌──────────────────┐                    │
            │       │  bardioc (PR#17) │                    │
            │       │  Rubicon kanban  │                    │
            │       │  (already ship)  │                    │
            │       └──────────────────┘                    │
            │                  │                            │
            │                  │ (commit)                   │
            │                  ▼                            │
            │       ┌──────────────────────────┐            │
            └──────►│  lance-graph             │            │
                    │  D-SPLAT-5 (registration)│            │
                    │  D-SPLAT-8 (FMA atlas)   │◄───────────┤
                    │  D-SPLAT-9 (FMA styles)  │            │
                    └──────────────────────────┘            │
                                  ▲                         │
                                  │ (FMA Class walk)        │
                                  │                         │
                    ┌──────────────────┐                    │
                    │  OGAR (#30 Ph 8) │                    │
                    │  ogar-fma crate  │                    │
                    │  KnowableFrom-   │                    │
                    │  Store registry  │                    │
                    └──────────────────┘                    │
                                                            │
                                  ┌─────────────────────────┘
                                  │
                                  ▼
                    ┌──────────────────────────┐
                    │  MedCare-rs              │
                    │  D-SPLAT-10/11           │
                    │  ultrasound_frame.lance  │◄─── q2/quarto D-SPLAT-14
                    │  commit_event audit      │     (SaMD docs cite this)
                    │  column_mask_bridge      │
                    └──────────────────────────┘
```

**Reading the map:**
- Arrows are **producer → consumer** edges.
- `ndarray` is the only source (no upstream); `MedCare-rs` is the only sink (no downstream, except the SaMD docs that read its audit).
- `bardioc` is already shipped and is consumed by `ractor_actors`; no new bardioc work is owed.
- The two cross-repo handshakes that need explicit coordination: **(ndarray D-SPLAT-2 ↔ lance-graph-contract D-SPLAT-1)** carrier-layout-meets-SIMD-load-width, and **(OGAR Phase 8 ↔ lance-graph D-SPLAT-8)** FMA TTL-to-Lance handoff.

---

## 10.9 Work-division schedule (sprint cadence)

| Sprint | ndarray | lance-graph-contract | lance-graph | OGAR | splat-fit | ractor_actors | MedCare-rs | splat-render | q2/docs |
|---|---|---|---|---|---|---|---|---|---|
| 1-2 (P1) | **D-SPLAT-2** | **D-SPLAT-1/3** (coord with ndarray) | — | — | — | — | — | — | — |
| 3 (P2) | bench follow-up | — | — | — | **D-SPLAT-6** | — | — | — | — |
| 4-5 (P3) | — | — | **D-SPLAT-4** | — | bench follow-up | **D-SPLAT-7/13** | — | — | — |
| 6 (P4 a) | — | — | (D-SPLAT-5 review) | **Phase 8 (FMA hydrate)** | — | — | — | — | start SAMD-INTENDED-USE |
| 7-8 (P4 b) | — | — | **D-SPLAT-5 + D-SPLAT-8/9** | (FMA review) | — | — | — | — | — |
| 9-10 (P5) | — | — | — | — | — | — | **D-SPLAT-10/11** | — | SAMD-AUDIT-TRAIL draft |
| 11-13 (P6) | — | — | — | — | — | — | bench follow-up | **D-SPLAT-12** | SAMD-RISK-CLASSIFICATION |
| 14+ (P7) | — | — | — | — | — | — | — | clinical-study iterations | **D-SPLAT-14** v2/v3 |

**Reading the schedule:**
- Each cell is "this repo's primary commitment this sprint."
- Empty cells = no required work this sprint (but maintenance / review may happen).
- The schedule has **slack** by design — most sprints have only 1-2 active repos, allowing review + integration testing time.
- The two parallel waves: sprint 6-8 (OGAR Phase 8 + lance-graph D-SPLAT-5/8/9 together) and sprint 11-13 (renderer + docs in parallel).

---

## 10.10 Cross-repo coordination protocol

Per the workspace's standing autoattend pattern + lance-graph's two-layer A2A discipline:

**Layer-1 (runtime / code-level):** the `OrchestrationBridge` + `StepDomain` taxonomy already handles cross-domain dispatch. The shipped contract (`crates/lance-graph-contract/src/orchestration.rs:37`) defines exactly eight variants: `Crew, Ladybug, N8n, LanceGraph, Ndarray, Smb, Medcare, Kanban`. Splat-native lands as **two new `StepDomain` variants ADDED to that shipped enum** (extending the single-source-of-truth taxonomy, not parallel to it):
- `StepDomain::SplatFit` (the engine consumes RF/IQ → emits SoA)
- `StepDomain::SplatRender` (the renderer consumes SoA → emits pixels)

These compose with the existing variants per the `from_step_type` dispatch — splat ingest routes `SplatFit → Ndarray → LanceGraph → Medcare` (fit → SIMD → registration → HIPAA wire); splat render routes `SplatRender ← LanceGraph` (read-only consumer). The two new variants land in the same PR as D-SPLAT-1 (`Gaussian3D` carrier) so the contract stays single source of truth. **Step-type prefixes** for `from_step_type`: `splat_fit.*` → `SplatFit`, `splat_render.*` → `SplatRender`.

**Layer-2 (session / Claude-code-level):** each per-repo doc has a `READ BY:` header naming which session-tier agents bootload it:

| Per-repo doc | READ BY |
|---|---|
| `lance-graph/.claude/plans/splat-native-ultrasound-v1.md` (this) | all subagents touching any splat-native deliverable; canonical |
| `ndarray/.claude/plans/splat-native-ultrasound-simd-substrate-v1.md` | `simd-savant`; any agent touching `ndarray::simd::splat` |
| `OGAR/docs/SPLAT-NATIVE-CUSTOMER.md` | any OGAR agent touching FMA hydrator or §6 litmus closure |
| `MedCare-rs/.claude/handovers/2026-06-05-splat-native-medcare-hipaa-wire.md` | any MedCare agent touching `soa_mapping.rs`, `column_mask_bridge.rs`, or audit chain |

Cross-session handoffs (when one repo's session blocks on another's) use the `.claude/handovers/YYYY-MM-DD-HHMM-from-to-topic.md` pattern. Append-only; carry FINDING/CONJECTURE/Blocker/OQ shape.

---

## 11. What this plan does NOT cover

- **Regulatory submission filings** — v3 Class IIa goes through CE / FDA processes that are jurisdiction-specific; this plan is the *evidence base*, not the submission.
- **Probe firmware modifications** — probes stay certified-as-is; if a probe needs firmware changes to expose raw RF, that's a separate engagement with the probe vendor.
- **Patient-specific atlas calibration** — v3 clinical workflow needs per-patient atlas pre-registration; v1 uses the standard FMA atlas only.
- **Multi-modal fusion** (CT/MRI atlas alignment) — out of scope for v1-v3; would be its own arc.
- **Federated learning across deployments** — out of scope; each deployment has its own splat volume and atlas registration.

---

_End of plan v1._
