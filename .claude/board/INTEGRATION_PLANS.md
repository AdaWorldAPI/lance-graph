## 2026-06-09 — polyglot-container-query-membrane-v1 (Node Container answers Cypher + SurrealQL AST + DataFusion UDF over one HHTL address space; mailbox = a normal cold path)

**Status:** RESEARCH MAP + PLAN. Grounded by two parallel sweeps (lance-graph + surrealdb fork) with main-thread spot-verification; one agent claim caught false (SoaEnvelope has ZERO real impls — identity N3 stands live). **Plan file:** `.claude/plans/polyglot-container-query-membrane-v1.md`.
**Owns:** 6 deliverables D-PG-1..6.
- D-PG-1: `addr64 = path << 4·(16−depth)` codec + order-preservation property test (subtree ⇔ contiguous range ⇔ `is_ancestor_of`) — the falsifiable first brick
- D-PG-2: `SoaEnvelope` impl for `MailboxSoA<N>` (= identity-plan N3) + LE parity test
- D-PG-3: read-only mailbox `Transactable` adapter (get/keys/keysr/scan/scanr over phase-pinned snapshot) + hot==cold differential test
- D-PG-4: `SurrealqlParse` strategy → ArenaIR (frontend #5, same slot as sparql_parse)
- D-PG-5: DDL ⇄ registry bridge (DEFINE TABLE/FIELD ⇄ dedup-by-URI mint + MappingRow/FieldMask; C16b builders as exchange format)
- D-PG-6 (optional): `surreal_container` unblock → Rubicon kanban VIEW over leading LanceDB (ruling-compliant; off critical path)
**Key findings:** surrealdb fork keys are storekey-encoded (order-preserving, arrays included) and record-ranges lower to native KV byte-range scans (`stream_keys_vals`, pipeline.rs:223); kv-lance is FULLY implemented in-tree (18/19 Transactable methods + MVCC/timeline) — `surreal_container` BLOCKED(C/D) is stale; the fork's typed `surrealdb-ast` crate + C16b DDL builders (`new_for_ddl` → `ToSql`, database-free) are the AST-adapter surface.
**Ruling respected:** LanceDB leads; SurrealDB is a view/dialect (handover 2026-05-28 §2, E-RUBICON-RACTOR).
**Companion plans:** `identity-architecture-exists-vs-needs-v1` (addresses), `bindspace-singleton-to-mailbox-soa-v1` (mailbox, #418).

---

## 2026-06-07 — singleton-to-snapshot-nudge-v1 (workspace-wide audit: every shared-mutable singleton → per-owner MailboxSoA + Arc-swap COW snapshot; read-only codebooks explicitly left as-is)

**Status:** PROPOSAL. Design-spec + audit only, no code beyond the AttentionMatrix correctness fix. **Plan file:** `.claude/plans/singleton-to-snapshot-nudge-v1.md`.
**Owns:** 7 deliverables D-SNGL-1..7.
- D-SNGL-1: workspace-wide singleton census (codebook vs shared-mutable classification)
- D-SNGL-2: classification gate — the mechanical "mutated-after-init?" decision procedure
- D-SNGL-3: `AttentionMatrix.gestalt` correctness (TD-UNBUNDLE-FROM-1) → raw-sum+count or rebuild-on-read
- D-SNGL-4: `ndarray/crates/burn` ATTENTION_CACHE / LINEAR_CACHE audit (JIT-kernel cache = keep; runtime belief = nudge)
- D-SNGL-5: `SnapshotProvider` adoption checklist per nudged crate (reuses cycle-coherent-soa-snapshot-v1 trait)
- D-SNGL-6: no-cross-cycle-lag falsification per nudged crate (reuses D-SOA-SNAP-5 test shape)
- D-SNGL-7: board hygiene + candidate epiphany E-SINGLETON-IS-CODEBOOK-OR-SOA
**Key distinction:** read-only `LazyLock` codebooks (role keys, UDFs, simd_caps) are NOT targets — only shared-mutable runtime state is. The test is mechanical: mutated-after-init → nudge; never-mutated → codebook, keep.
**Companion plans:** `bindspace-singleton-to-mailbox-soa-v1` (BindSpace singleton dissolution — referenced, not duplicated), `cycle-coherent-soa-snapshot-v1` (the snapshot mechanism reused here).
**Anchored epiphanies:** E-MAILBOX-IS-BINDSPACE, E-BATON-1, E-DEINTERLACE-TWO-SCALES. **Anchored debt:** TD-UNBUNDLE-FROM-1.

---

## 2026-06-06 — cycle-coherent-soa-snapshot-v1 (Arc-swap COW at column granularity; byte-scale deinterlace; no-cross-cycle-lag guarantee)

**Status:** QUEUED. Design-spec only, no code. **Plan file:** `.claude/plans/cycle-coherent-soa-snapshot-v1.md`.
**Owns:** 6 deliverables D-SOA-SNAP-1..6.
- D-SOA-SNAP-1: `MailboxSoaSnapshot` type in lance-graph-contract
- D-SOA-SNAP-2: `SnapshotProvider` trait in lance-graph-contract
- D-SOA-SNAP-3: Arc-swap write path in `MailboxSoa::advance_phase`
- D-SOA-SNAP-4: `snapshot()` impl on `MailboxSoa`
- D-SOA-SNAP-5: No-cross-cycle-lag falsification test (writer thread + 8 reader threads)
- D-SOA-SNAP-6: Wire `snapshot.cycle` into `QueryReference` (close row-scale / byte-scale clock loop)
**Epiphany:** E-DEINTERLACE-TWO-SCALES (prepended 2026-06-06).
**Companion:** PR #468 (`temporal.rs`, row-scale, SHIPPED); PR #477 (`soa_envelope.rs`, IN REVIEW).
**Boundary:** ndarray stays layout-only (`MultiLaneColumn`); Arc-swap policy in lance-graph only.

---

## 2026-06-05 — cesium-osm-substrate-v1 (OpenStreetMap as 6th source class for the 3DGS-ArcGIS-Cesium ingestion plan; OSM PBF → Arrow → Lance → SPO → cesium tileset → splat renderer; substrate-reuse with splat-native-ultrasound-v1)

**Status:** PROPOSAL. Design-spec only, no code. **Plan file:** `.claude/plans/cesium-osm-substrate-v1.md` (~430 LOC). **Trigger:** user feasibility question on OSM × Cesium × Gaussian-splat coupling; cross-session coordination with OGAR.
**Owns:** 7 deliverables D-OSM-1..7. ndarray D-OSM-1 (cesium stub), D-OSM-4 (SIMD `batched_sample_height`), D-OSM-6 (`cesium-3dtiles-writer` — the genuine Rust gap, no existing crate produces b3dm/i3dm/cmpt); lance-graph D-OSM-2 (osmpbf → Arrow → Lance), D-OSM-7 (Nominatim sidecar; optional); lance-graph-ontology D-OSM-3 (OSM tag → SPO triple lift; **OGAR-crossing contract**); new `crates/splat-fit-geo` or `splat-fit` `geo` feature D-OSM-5 (OSM footprint × DEM → extruded Gaussian3D).
**OGAR coordination (locked 2026-06-05):** Q1 ruling = Tag-as-Class (c) end-state with `tags: List<Struct<key,value>>` Arrow column (b) as v1 fallback. Q2 ruling = Cesium TMS quadkey NiblePath `osm/qk:<level>/<x>/<y>/<type>/<id>`. Q3 ruling = OSM-XYZ → TMS Y-flip at ingest boundary (per `I-LEGACY-API-FEATURE-GATED`). OGAR-side docs PR (`DOMAIN-INSTANCES.md §2.6` + `RDF-OWL-ALIGNMENT.md §10 Phase 2c`) queued behind this PR to cite D-OSM-* by ID.
**Anchored:** E-SOA-IS-THE-ONLY (one substrate, three operations — fit/accumulate/render — shared across ultrasound + geospatial; D-SPLAT-1/2/3/12 reused verbatim in D-OSM-5/6), I-VSA-IDENTITIES (Tag identity fingerprints not content bundling), I-LEGACY-API-FEATURE-GATED (Y-axis Q3 boundary), I-NOISE-FLOOR-JIRAK (geocode significance under weak dependence).
**Companion plans:** `3DGS-ArcGIS-Cesium-ingestion-plan.md` (parent / structural; 5 source classes — this adds 6th), `splat-native-ultrasound-v1.md` (substrate-sibling; Gaussian3D carrier reuse anchor).
**5 open questions:** OQ-OSM-1 Tag canonicalization rules · OQ-OSM-2 DEM source (default: SRTM 1-arcsec) · OQ-OSM-3 writer scope (MVP: b3dm + cmpt + tileset.json; full: + i3dm/pnts/subtree) · OQ-OSM-4 ingest discipline (per-country PBF v1; planet-scale sharded) · OQ-OSM-5 coordinate-policy compliance for DEM CRS declaration.

---

## 2026-06-05 — splat-native-ultrasound-v1 (CPU-only Gaussian-splat ultrasound SaMD: probe RF/IQ → splat-fit → Mailbox<Gaussian3D> → 4-phase Rubicon kanban → FMA-atlas Σ-sandwich registration → AR rendering; the explicit customer of OGAR PR #30 §6 FMA litmus + ADR-022 firewall as SaMD Class IIa evidence base)

**Status:** PROPOSAL. Design-spec only, no code. **Plan file:** `.claude/plans/splat-native-ultrasound-v1.md` (~930 LOC, 12 sections incl. §10 per-repo work-division + interconnect map). **Trigger:** user-supplied architecture diagrams (English 6-stage + German business-facing).
**Owns:** 14 deliverables D-SPLAT-1..14 across 4 repos. ndarray D-SPLAT-2 (SIMD splat ops); lance-graph D-SPLAT-1/3/4/5/8/9 (carriers + SH palette + registration math + FMA atlas + style_recipe); new standalone `crates/splat-fit` D-SPLAT-6; `crates/splat-actors` D-SPLAT-7/13; `crates/splat-render` D-SPLAT-12; MedCare-rs D-SPLAT-10/11 (`memory.ultrasound_frame.lance` + `commit_event` audit); OGAR Phase 8 customer of #30; q2/quarto D-SPLAT-14 (SaMD documentation track Forschungstool→Studie→Class IIa).
**Anchored:** E-SOA-IS-THE-ONLY (one substrate, three operations: fit/accumulate/render), E-BATON-1 (frame baton handoff), E-RUBICON-RACTOR (Libet −550 ms 4-phase kanban), I-VSA-IDENTITIES (Gaussian class_id points to FMA URI), I-NOISE-FLOOR-JIRAK (Mahalanobis significance + ICP convergence thresholds).
**Companion docs:** `ndarray/.claude/plans/splat-native-ultrasound-simd-substrate-v1.md` (SIMD-substrate perspective), `OGAR/docs/SPLAT-NATIVE-CUSTOMER.md` (§6 FMA-litmus customer narrative), `MedCare-rs/.claude/handovers/2026-06-05-splat-native-medcare-hipaa-wire.md` (HIPAA wire perspective).
**5 open questions:** OQ-SPLAT-1 first probe SDK (default: Telemed ArtUs — RF directly, USB, lowest regulatory friction); OQ-SPLAT-2 SH degree budget (default: ℓ=3, 16 coeffs); OQ-SPLAT-3 beamformer scope (default: consume probe's BF where available, fall back DAS); OQ-SPLAT-4 AR splat egress (default: no — stays on HoloLens); OQ-SPLAT-5 plan home (resolved: lance-graph canonical + per-repo companions).

---
## 2026-06-02 — le-domino-cognition-v1 (the converged cognitive-substrate north-star + the shared kanban-door contract: one mailbox substrate, SPO-2³ `syllogize` on the full edge, (f,c)-diff driver, LE-domino cascade, escalation tiers; DESIGN over scaffold)

**Status:** DESIGN (north-star + coordination contract). **Plan file:** `.claude/plans/le-domino-cognition-v1.md`. Capstone of the 2026-06-02 co-design session (jan + Opus), grounded by two 3-way fan-outs. Thesis: most-expressive substrate × most-efficient ops, grounded (never wishful) by construction. Also the **shared contract** where this work meets a parallel session's SurrealDB DDL-AST `actions`→kanban push — both produce `KanbanMove`/`ConsumerEnvelope::Plan` through `Scheduler::on_version`, meeting at the contract not a private path. Honest seam-map (everything/nothing): the operators (`syllogize`/`CausalMask`/`NarsTables`) exist as types+tests with 0 live callers; the wiring is the build. Ordered seams: 1 backward domino (collision-free) → 2 escalation emit (shared door) → 3 forward link → 4 cold/calcify (blocked on D-PERSONA-5).

---

## 2026-06-02 — baton-collapse-dereification-v1 (de-reify the "baton"/"collapse gate emission" doctrine: a folk metaphor that ossified into canonical types/doctrine; 2 doc edits, ZERO code, no new ticket)

**Status:** APPLIED. **Plan file:** `.claude/plans/baton-collapse-dereification-v1.md`. Corrects a doctrine reification jan flagged ("there is no collapse; baton was a figure of speech; the write is the witness; mailboxes own their SoA and write safe in the hot path"). 5-research + 3-brutal council (all SHIP-WITH-FIXES). Scope: CLAUDE.md P-1 `2026-05-26` blockquote rewritten (kills "inter-mailbox state IS the Baton"; states live-vs-scaffold honestly) + north-star WD-5 terminology (L27 direct, L53 ratified-cell annotated). The type rename (`CollapseGateEmission`→`WireWrite` etc.) is DEFERRED with NO ticket minted (B3: don't mint ceremony for a rename that may never run). One ISSUE filed: `MergeMode::Bundle` doc-vs-code split-brain.

---

## 2026-06-01 — a4-resolver-v1 (the resolver: texture → thinking style in ~4-cycle integer CAM addressing; firewall newtypes AtomLane(0..64)/LaneMask + AtomGroup::is_signed; A4a offline-contract, A4b cross-crate gated)

**Status:** DRAFT / pre-savant-review. **Plan file:** `.claude/plans/a4-resolver-v1.md`. Follows A3 (PR #451 merged). NO f32, NO vector search — deterministic CAM addressing; the only fuzzy step is the coarse upstream "smells like odoo → financial OGIT" route. Doctrine: plan → 5 savant-dev review → split → per-file sprint (commented) → 3 brutal-tester → fix → uncomment → PR.

---


## 2026-06-01 — north-star-integration-v1 (the two-ViewAngle destination: business "Odoo-in-Rust × OGIT+DOLCE+GoBD" ∥ semantic cognitive-substrate §9–14; maps current→target + the 9 wiring decisions WD-1..WD-9 for the 5-dev council)

**Status:** PROPOSAL / pre-council. **Plan file:** `.claude/plans/north-star-integration-v1.md`. **North star:** `.claude/north-star/README.md` (the 2 reference diagrams, textual capture; PNG slots open). Shipped: §3 capstone + A1 rule + A2 rung (#450), A6 emission surface (branch). The council irons WD-1 (I4x32D 2-halves vs 4-view OGIT O/G/I/T) · WD-2 (OGIT resolver) · WD-3 (vart key + indexes) · WD-4 (version subscription — **no Delta Lake**: vart+Lance-native+SurrealDB-LIVE) · WD-5 (ractor + Belief/Goal BDI) · WD-6 (**Rust core vs Elixir OTP/BEAM split** — the big one) · WD-7 (Figure 4→64 moods) · WD-8 (GoBD audit) · WD-9 (DeepNSM dual-view head2head). The determinism→GoBD thesis: the firewall IS the compliance guarantee.

---


## 2026-05-31 — episodic-risc-spine-v1 (the converged episodic addressing spine: 3 lifecycle-separated structures — CAM/OGIT identity, Lance-version pseudo-radix index, CLAM ephemeral KV; EW64 episodic edges with a 4-bit inherited palette; bounded-horizon compression)

**Status:** ACTIVE — contract slices shipped (D-EW64-1, D-VIEW-1; + VersionScheduler D-MBX-9-IN, head2head D-H2H-1); core/cold CI-gated. **Plan file:** `.claude/plans/episodic-risc-spine-v1.md`. **Finding:** EPIPHANIES E-EPISODIC-CLOSURE.

---


## 2026-05-31 — odoo-classes-bitmask-render-v1 (the bounded-weekend classes.md fix: `ClassId` discriminator + per-class `FieldPositionTable` + `FieldMask(u64)` + per-class askama templates + Aerial+ shape-family discovery over the 66 OdooEntities)

**Status:** PROPOSAL / integration plan, pre-council. Design-spec only, no code. **Plan file:** `.claude/plans/odoo-classes-bitmask-render-v1.md`. **Anchored spec:** `cognitive-risc-classes.md` v0.2 §"Jinja = classes + presence bitmask" + §"NON-DEFERRABLE freeze-time moves" (N1, N3) — bounded-weekend doctrine (line 56-57): *"discriminator + parent-pointer + parent-walking resolution against the existing cache. Full machinery (shape-compiler-to-grid, behavior/traits, SIMD kernels) is explicitly DEFERRED."*
**Owns:** the structural fix for the union-disease classes.md known-debt names (line 55: "single-SoA is a union pretending to be uniform"). Adds 9 deliverables D-CLS-1…D-CLS-9 across 5 waves (10 agent runs: 2 Opus + 9 Sonnet, ~1,800 working LOC + ~2,400 LOC of generated snapshots/templates). One new excluded crate `lance-graph-ontology-render` (askama, F3 lean). Resolves the 4-way `DolceCategory` duplication on-disk via additive re-export + deprecation (zero rename per hard constraint C6).
**Anchored iron rules:** none new — this plan respects all existing iron rules. **Anchored epiphanies:** `E-DISCOVERY-ORIGIN-HOME-IS-ARIGRAPH-BRIDGE` (the structurally adjacent insight from yesterday's council), `E-TEMPLATE-IS-CHECKLIST-IS-DATOMS` + `E-RELIABILITY-IS-CHECKLIST-COVERAGE` (the parallel-session #439 ThoughtMask shape this plan generalises to FieldMask).
**Predecessors:** the 4 canonical cognitive-risc specs (now on main via commits `d1635dbe`, `93ac0463`, `a16d0f41`, `45276eb3`); PR #435 (the 66-entity survival dossier this plan operates on); PR #436 (the Aerial+ Rust crate D-CLS-3 calls); PR #439 (the `ThoughtMask` whose shape this plan generalises). **Spec-owner gates open:** OD-DOLCE-CANONICAL, OD-CLASSID-WIDTH, OD-CLASSID-VS-ENTITYKIND, OD-TEMPLATE-ENGINE — all 4 ratified before any agent starts (plan §2).

---

## 2026-05-29 — streaming-arm-nars-discovery-v1 (upstream proposer leg: 20K-200K rows/window pair-stats + optional Aerial+ neural fan-in → NARS-truth translator → SpoStore hypothesis test → epiphany-council ratification → op_emitter codegen)

**Status:** PROPOSAL / integration plan. Design-spec only, no code. **Plan file:** `.claude/plans/streaming-arm-nars-discovery-v1.md`. **Handover:** `.claude/handovers/2026-05-29-2030-arm-discovery-author-to-impl.md`. **Candidate epiphany:** `E-DISCOVERY-CODEGEN-BRACKET-1` (council-pending).
**Owns:** the missing upstream discovery leg into the SPO substrate. Today's proposers (`D-ODOO-BP-1b` curated, `D-ODOO-EXT-2` AST-extracted) are bounded by literal artifacts; this plan adds runtime-data discovery via streaming pair-stats (deterministic trunk) + optional Aerial+ neural-symbolic fan-in. The two papers (Karabulut 2025 arxiv 2504.19354v1; Abreu 2025 arxiv 2511.13661v1) ratify the architecture: discovery upstream, codegen downstream, SPO+NARS middle. Adds 12 deliverables D-ARM-1…D-ARM-12; one new crate `lance-graph-arm-discovery`; 10 OQs; two corrections proposed to PR #434 (`discovery_arc: [u32; D]` column, `discovery_origin: u8` byte).
**Anchored iron rules:** I-NOISE-FLOOR-JIRAK (mandatory Stage-A threshold), I-SUBSTRATE-MARKOV (NARS revision IS the Markov trajectory), I-VSA-IDENTITIES (operates on typed triples, never content). **Anchored epiphanies:** E-SOA-IS-THE-ONLY (writes via SpoBuilder only), E-BATON-1 (Stage-C emissions are batons), E-INTERPRET-NOT-STORE-1 (ARM is one interpretation projection).
**Predecessors:** PR #433 (style_recipe.rs + epiphany-brainstorm-council + 5 savant cards = the ratification gate), PR #434 (unified-soa-convergence-v1.md = the SoA contract this writes against), op_emitter.rs (this branch = Phase 2 codegen target).

---

## 2026-05-29 — unified-soa-convergence-v1 (THE single LE SoA end-to-end across the workspace: 9 half-baked consumers + SoA version gate + Lance 6.0.1/LanceDB 0.29/DataFusion 53 alignment + 4-phase Rubicon kanban + lance-graph-planner DTO overhaul + Libet −550 ms anchor + Staunen×Wisdom plasticity spreader + SPO-W pointer via AriGraph episodic Markov chain)

**Status:** SHIPPED (PR #434 merged 2026-05-29, `1186dfd3`). Design-spec only, no code. **Plan file:** `.claude/plans/unified-soa-convergence-v1.md`. **Handover:** `.claude/handovers/2026-05-29-1825-soa-convergence-author-to-impl.md`. **Review addendum:** `.claude/plans/unified-soa-convergence-v1-addendum-2026-05-29-review.md` (post-merge review by `claude/lance-graph-ontology-review-Pyry3`). **Epiphany:** `E-SOA-IS-THE-ONLY` (+ refinements §11.3/4/6).
**Owns:** the meticulous integration sequence for the 5 layered user rulings ratified this session — one SoA never transformed; mailbox = full BindSpace as LE; Libet −550 ms ratchets the Rubicon kanban; SPO-W = pointer via AriGraph episodic Markov chain; counterfactual Staunen×Wisdom = plasticity spreaders. Adds D-MBX-A2/A3/A4/A5/A6, D-MBX-7/8/9/10/11/12 (12 sub-PRs); 8 open questions OQ-11.1–11.8; risk matrix; phased dependency graph (P0–P7).
**Anchored iron rules:** I-VSA-IDENTITIES, I-LEGACY-API-FEATURE-GATED (governs the SoA version gate), I-SUBSTRATE-MARKOV (untouched), I-NOISE-FLOOR-JIRAK.
**Predecessors:** `bindspace-singleton-to-mailbox-soa-v1` (§11 rulings); `causaledge64-mailbox-rename-soa-v1` (canonical 5-crate + 7-PR plan); `cognitive-substrate-convergence-v1/v2/v3` (Σ10 Rubicon shipped #388).
**Council bypass note:** the underlying §11 epiphanies are author-stated (user) — `epiphany-brainstorm-council` (shipped #433) is bypassed for them; the plan's *spec content* IS open to council review via PR.
**Workspace stack verified 2026-05-29:** arrow 58 ✓ · datafusion 53 ✓ · lancedb =0.29.0 ✓ · lance =6.0.0 → **=6.0.1** (one patch bump pending, D-MBX-11).

---
## 2026-05-28 — normalized-entity-holy-grail-v1 (typed unified normalization + Op chain over OGIT/OWL/DOLCE/Odoo with three-context execution — the trunk that unifies BP-1 + EXT-1..6 + jit + MailboxSoA into one consumer surface)

**Status:** PROPOSAL. The trunk plan that ties together every prior architectural deliverable into ONE typed consumer pipeline grammar. Closes the structural gap that lets consumers re-implement business logic in regex / hand-rolled pseudo-code by making such interpretation a MISSING FUNCTION (per `E-CONSUMER-CANNOT-INTERPRET-1`) rather than a code-review concern. Re-encodes Odoo's three-regime decomposition (interactive / bulk / periodisch) as compile-time typestate, not stringly-typed `env.context` flags.
**Confidence:** HIGH on carrier + algebra shape (one struct, 5 verbs — direct analogue of the SoA-as-AGI unification). HIGH on the three-context split (genuinely different SLAs; Odoo prior art proves the decomposition is real). MED on the Op-trait three-call-site specialisation (jit substrate exists but has never been wired against a typed Op grammar). LOW on the macro-DSL layer (consumer ergonomics; needs per-repo iteration).
**Plan file:** `.claude/plans/normalized-entity-holy-grail-v1.md`
**Predecessors:** `D-ODOO-BP-1a..g` typed surface (Wave 1-3 lane modules); `D-ODOO-EXT-1..6` source-extracted backing (PR #426 merged); PR #411 (`lance-graph-contract::jit::{JitCompiler, StyleRegistry, KernelHandle}`); PR #427 (`MailboxSoA` thoughtspace columns + WitnessTable + §10 refinements); `lance-graph-contract::callcenter::ogit_uris`; `lance-graph-ontology::dolce_odoo`; `lance-graph-contract::orchestration::{OrchestrationBridge, UnifiedStep}`; `lance-graph-rbac::{SuperDomain, smb_policy}`.
**Anchored epiphanies (all 8 prepended to EPIPHANIES.md in the same commit):** `E-NORMALIZED-ENTITY-1` (single typed carrier); `E-OP-FIVE-VERBS-1` (resolve/hydrate/classify/align/think closure); `E-OP-THREE-CALLSITES-1` (cold/warm/hot, shared const data); `E-TRANSACTION-CONTEXT-1` (interactive/bulk/periodisch own commit+epoch+version policy); `E-CASCADE-AS-EDGECOLUMN-1` (Odoo's 6 mechanisms collapse into 1 typed graph); `E-ODOO-AS-PRIOR-ART-1` (decomposition borrowed, encoding improved); `E-CONSUMER-CANNOT-INTERPRET-1` (regex structurally banned via missing-function); `E-NO-AUTOMATIC-REGIME-PICK-1` (consumer-typed context picks mode, not shader).

### Stage-1 deliverables (this plan ships ~2 500 LOC; mostly typed signatures + compile-fail tests)

| D-id | Description | Site | LOC |
|---|---|---|---:|
| **D-NEH-1a** | `lance-graph-contract::cognition::{NormalizedEntity, stages, Op, OpKind, MailboxRow, Output}` typed surface — zero-dep, `todo!()` bodies, compile-fail tests | `lance-graph-contract/src/cognition/` | 600 |
| **D-NEH-1b** | `lance-graph-contract::transaction::{Interactive, Bulk, Periodisch, Context, OgitCtx/OwlCtx/DolceCtx/FibuCtx}` — context shapes + commit-policy traits | `lance-graph-contract/src/transaction/` | 400 |
| **D-NEH-1c** | 5-verb advancement methods on `NormalizedEntity<S>` (`resolve_ogit` / `hydrate_owl` / `classify_dolce` / `align_fibu` / chain methods) + provenance | `lance-graph-contract/src/cognition/advance.rs` | 300 |
| **D-NEH-1d** | `CascadeKind` + cascade-graph traversal trait on `EdgeColumn` + per-context traversal mode (`Sync` / `Batched` / `JitFixedPoint`) | `lance-graph-contract/src/cognition/cascade.rs` | 350 |
| **D-NEH-1e** | Compile-fail tests proving the typestate gate (out-of-order chain calls fail to compile) | `lance-graph-contract/tests/cognition_typestate.rs` | 250 |
| **D-NEH-1f** | Doc-level example consumer chain (woa-rs invoice flow as `cargo doc` example) + crate-level doc landing page | `lance-graph-contract/src/cognition/mod.rs` + `docs/COGNITION_HOLY_GRAIL.md` | 400 |
| **D-NEH-1g** | Board hygiene (this entry + EPIPHANIES prepend + STATUS_BOARD rows) | `.claude/board/` | 200 |

### Subsequent waves (sketched in the plan; separate v2..v7 plans)
- **v2** kernel bodies for ~50 typed Ops (port to shader dispatch table + JIT-compile hot path)
- **v3** consumer DSL macros (`medcare_think!` / `woa_think!` / `smb_think!`) — pipe-style ergonomics on top of the typestate chain
- **v4** Stream + ractor-supervised actor wiring (the Elixir-OTP analogue)
- **v5** Jahresabrechnung JIT kernel (target ≥100× throughput vs Odoo's `account.fiscal.year.close` wizard)
- **v6** palantir-foundry parity audit (ontology objects / branches / lineage / action types / functions-on-objects → workspace primitives matrix)
- **v7** elixir-OTP parity audit (processes / supervision / streams / behaviours → workspace primitives matrix)

When v6 and v7 are ≥80% present, the workspace has earned the "better palantir foundry / better elixir" framing the plan was named for.

---

## 2026-05-28 — odoo-source-extraction-v1 (TIER-1 Odoo source extraction → `OdooConfidence::Extracted` backing for `D-ODOO-BP-1b`; sub-plan unfolding `D-ODOO-BP-1f`)

**Status:** SHIPPED (Stage 1 — EXT-1..6 complete); per-lane gate test in `extracted::coverage`; Stage 2 addresses TIER-2 addons (POS, HR, website, fleet, maintenance, non-DE l10n, payment providers); the 5 known L13/L14 gaps (4 hr.* + stock.valuation.layer) close via hr + stock_account extraction. Unfolds `D-ODOO-BP-1f` of `odoo-business-logic-blueprint-v1` into a tractable Stage 1 over 12 TIER-1 addons (`account`, `account_payment`, `l10n_de`, `product`, `stock`, `uom`, `base`, `analytic`, `purchase`, `sale`, `account_peppol`, `account_edi_ubl_cii`) of the 622 in `/home/user/odoo/addons/`. Validates and adds source-extracted backing to the L-doc-curated `OdooEntity` consts that Wave 1-3 just shipped (commits `9507b36`..`2aca3e3`).
**Confidence:** HIGH on the substrate decision (extraction is purely additive — cannot regress curated set). HIGH on the tooling substitution (Python stdlib `ast` replaces the absent `tree-sitter` — handles every ORM shape observed in inventory). MED on per-addon yield (~5–8K typed const declarations across TIER-1, ~2–3K condensed Rust LOC). LOW on regulation-IRI density per entity.
**Plan file:** `.claude/plans/odoo-source-extraction-v1.md`
**Predecessors:** `D-ODOO-BP-1a` (typed surface, pre-`9507b36`); `D-ODOO-BP-1b` Wave 1–3 (L1–L15 L-doc projection, shipped `9507b36`..`2aca3e3`); 2026-05-28 Odoo source inventory (622 addons, 3 141 ORM classes, 989 K Python LOC; 8/9 German concept anchors located; ELSTER absent — Enterprise-only).
**Anchored iron rules:** `I-VSA-IDENTITIES` (identity in const data, content in tables — regulation IRIs are codebook handles, not content), `E-CODEBOOK-INHERITS-FROM-OGIT` (regulation rules inherit from OGIT codebook — drives `OdooProvenance.regulation_iri`), "audit-by-construction" (every extracted entity carries `(path, line_range)` + UStG/HGB/GoBD anchor when applicable).

### Scope
**TIER-1 addons (12)**: `account`, `account_payment`, `l10n_de`, `product`, `stock`, `uom`, `base`, `analytic`, `purchase`, `sale`, `account_peppol`, `account_edi_ubl_cii` — full German accounting flow + ORM roots + value chain + e-invoice. **TIER-2 (POS, HR, website, payment providers, non-DE l10n) explicitly deferred** until Stage 1 EXT-6 reports < 5 % parser-fallback rate. **Tooling**: Python stdlib `ast` module (stale `tree-sitter` reference in `odoo_blueprint/mod.rs:321` corrected by EXT-3). **Output**: `crates/lance-graph-ontology/src/odoo_blueprint/extracted/<addon>.rs`, additive to lane modules `l{1..15}`; consts prefixed `EXT_` to avoid symbol collisions with curated lane consts. **Provenance enhancement**: `OdooProvenance` gains `regulation_iri: &'static [&'static str]` (UStG §15 / HGB §238 / GoBD / AO §146a / EN 16931 IRIs into the OGIT codebook) + `OdooEntityKind::{Model,Transient,Abstract}` variant; back-filled `&[]` / `Model` across the existing Wave 1-3 const set.

### Deliverables

| D-id | Description | Site | LOC | Conf | Status |
|---|---|---|---:|:--:|:--:|
| **D-ODOO-EXT-1** | Python `ast` extractor scaffold + smoke test on `uom` (1 model) | `tools/odoo-blueprint-extractor/` | 800 | HIGH | Queued |
| **D-ODOO-EXT-2** | TIER-1 extraction in 3 waves (foundation / value-flow / DE+e-invoice) → `extracted::*` module tree; `<5%` `OdooFieldKind::Other` fallback rate gate | `odoo_blueprint/extracted/` | 5000–8000 | HIGH | Queued |
| **D-ODOO-EXT-3** | `OdooProvenance.regulation_iri` slot + `OdooEntityKind` variant + back-fill of Wave 1-3 const set + tree-sitter→`ast` doc-comment correction | `odoo_blueprint/mod.rs` + lane modules | 200 | HIGH | Queued |
| **D-ODOO-EXT-4** | `l10n_de` specifics: SKR03/04 chart consts (~2 466 accounts) + 17 UStVA Kennzahl consts (Kz.81..95) + GoBD audit-trail flag wiring | `odoo_blueprint/extracted/l10n_de.rs` | 1500 | HIGH | Queued |
| **D-ODOO-EXT-5** | Curated-vs-extracted pairing table + mismatch audit (`pairing.rs` + `audit/pairings.json`); curated stays canonical on conflict | `odoo_blueprint/extracted/pairing.rs` | 300 | MED | Queued |
| **D-ODOO-EXT-6** | Honest-coverage report (`COVERAGE.md`) + lane test that fails if any L1..L15 lane drops below 80 % extracted-backing | `odoo_blueprint/extracted/COVERAGE.md` + test | 200 | MED | Queued |

### Execution ordering
1. EXT-3 first (provenance shape stabilizes before any extracted const lands).
2. EXT-1 in parallel with EXT-3 (extractor scaffold has no shape dep).
3. EXT-2 in 3 waves (A: base/uom/product/analytic; B: account/account_payment/purchase/sale/stock; C: l10n_de/account_peppol/account_edi_ubl_cii).
4. EXT-4 right after Wave C (heaviest extraction — CSV + XML + Python in one addon).
5. EXT-5 after EXT-2 lands (pairing needs both sides).
6. EXT-6 closes Stage 1.

---

## 2026-05-28 — odoo-business-logic-blueprint-v1 (typed Odoo entity DTOs as the substrate for the OGIT → OWL → DOLCE → FIBU/FIBO normalization + JITson / recipe codegen)

**Status:** PROPOSAL. **PREREQUISITE for `odoo-savant-reasoners-v2` Group F** (per `E-SAVANT-COMPOSITION-1`): v2's `SavantPattern` consts compose *over normalized typed DTOs*; without this blueprint they would be ad-hoc interpretations of L-doc prose. Establishes the missing layer between the Odoo prose curation (L-docs + savant docs) and the agnostic shader substrate.
**Confidence:** HIGH on the structure (typed DTOs are the missing inheritance-chain input — today every OGIT/OWL/DOLCE/FIBU layer string-keys against `model_name`). MED on per-lane density (15 lanes × ~5-30 entities ≈ ~150-450 const declarations is an estimate; actual count depends on lane density). HIGH on the JITson wiring path (infrastructure exists at `lance-graph-contract::jit` + `thinking-engine` Cranelift integration, just not yet pointed at recipes/savants).
**Plan file:** `.claude/plans/odoo-business-logic-blueprint-v1.md`
**Predecessors:** PR #411 (33-TSV atoms + 34 `Tactic` + `jit` infrastructure), PR #412 (DOLCE classifier + `dolce_odoo`), PR #414 (D-ODOO-SAV-1/2/3 OGIT families + Layer-2 axioms), PR #416 (FIBU/FIBO + `contract::savants` roster), PR #407/408 (bO-* OWL hydrators), PR #418 (mailbox-owned SoA), PR #420 (v1 reasoners — deprecated in v2). Driver epiphany: `E-SAVANT-COMPOSITION-1` (2026-05-28).
**Anchored iron rules:** `I-VSA-IDENTITIES` (typed Layer-2 catalogues — identity in const data, content in tables), AGI-as-glove (typed DTOs ARE the column the inheritance chain reads from), "consult before guess" (L-doc curation IS the curated surface; project into typed structure, don't re-derive).

### Scope
**Source**: Both passes — L-docs first as the savant-relevant curated filter; Odoo source extraction (`/home/user/odoo`) follows as the exhaustive backing. **Lane coverage**: All 15 lanes (L1–L15) in v1 — complete typed blueprint of the Odoo domain, not a savant-rich subset. **Typed surface**: `lance-graph-ontology::odoo_blueprint` module with `OdooEntity` + `OdooField` + `OdooMethod` + `OdooDecorator` + `OdooStateMachine` + `OdooConstraint` + `OdooProvenance` (zero-dep, const-only, no serde).

### Deliverables
D-ODOO-BP-1a: `OdooEntity` + sub-types typed surface in `lance-graph-ontology` · D-ODOO-BP-1b: L-doc projection (one `OdooEntity` const per entity × 15 lanes, provenance=Curated) · D-ODOO-BP-1c: OGIT classifier wired to `&OdooEntity` (replaces string-keyed `resolve_odoo`) · D-ODOO-BP-1d: OWL hydrator wired to `&OdooEntity` (relational → edges, computed → SHACL constraints, decorators → axioms) · D-ODOO-BP-1e: DOLCE + FIBU/FIBO wired to `&OdooEntity` (closes D-ODOO-SAV-2's `None`-class alignment over typed input) · D-ODOO-BP-1f: Odoo source extraction tool (tree-sitter Python AST → candidate consts with Confidence=Extracted; validates 1b) · D-ODOO-BP-1g: JITson → recipes wiring (proof-of-concept: `jit::JitCompiler` compiles a `Tactic` kernel parameterized by `(&OdooEntity, AtomTouchMask)` → DTO-ish NARS in shader-driver).

### Execution
1a first (typed surface, ships with this plan + board hygiene) → 1b in Waves (per-lane subagent, Sonnet, mechanical projection) → 1c/d/e parallel after 1b (three inheritance hops, independent) → 1f after 1b/c/d/e (extraction validates curated) → 1g closes the loop (JIT compiles `Tactic` over normalized DTO). THEN `odoo-savant-reasoners-v2` Group F unblocks. v2 Groups D/E/G remain unblocked (ship independently; F is the only group blocked on the blueprint).

### Invariants
Zero-dep const-only no-serde on DTO surface (AGI-as-glove + contract discipline) · provenance on every entity (`I-VSA-IDENTITIES` — content in typed registries, never bundled) · inheritance chain operates on typed DTOs never on string model names · L-doc projection is canonical v1; Odoo source extraction is validation/extension · **PREREQUISITE for v2 Group F**; v2's composition layer composes ON TOP of the normalized DTOs.
## 2026-05-28 — odoo-savant-reasoners-v2 (reshape: `Reasoner` trait → typed composition over `CausalEdge64` + `Tactic` + `callcenter/role_keys`)

**Status:** PROPOSAL (architectural reshape of v1, shipped PR #420 with a "MED on dispatch shape" caveat). v1 review resolved: the `Reasoner` trait surface is the wrong shape per CLAUDE.md "P-1 The Click" + "P0 AGI-as-glove" litmus tests. v2 deprecates the v1 surface (feature-gated + `#[deprecated]`) and routes the canonical path through the agnostic substrate that already exists (`CausalEdge64` + `Tactic` + 33-TSV atoms + role-key catalogues — all shipped via PR #411 + #418 + #419 + #420).
**Confidence:** HIGH on the diagnosis (litmus tests in CLAUDE.md name v1's anti-patterns verbatim). HIGH on the right-shape vocabulary (PR #411 ratified the 34-tactic `Tactic` trait as "the Elixir-like recipe layer that later fronts the real fingerprint substrate via cognitive-shader-driver with no change to the 34 call sites"). MED on per-savant typed composition consts (substantial translation pass from savant-doc + L-doc curation).
**Plan file:** `.claude/plans/odoo-savant-reasoners-v2.md`
**Predecessors:** PR #420 (v1 ship — the surface to deprecate), PR #411 (33-TSV atoms + 34-tactic `Tactic` + `recipe_kernels`), PR #418 (BindSpace → mailbox-owned SoA; `E-BATON-1`), PR #419 (25 AXIS-B evidence contracts), PR #416 (`contract::savants` roster), PR #414 (D-ODOO-SAV-1/2/3). v1 plan: `odoo-savant-reasoners-v1.md`.
**Anchored iron rules:** AGI-as-glove (P0), The Click P-1 ("free function on carrier = reject"), `I-VSA-IDENTITIES` (`callcenter/role_keys.rs` named home), `E-BATON-1` (mailbox-owned SoA, CE64 baton as cross-boundary state), `I-LEGACY-API-FEATURE-GATED` (v1 deprecation under feature + migration pointer).

### Scope
Three deliverable groups: **Group D** — agnostic composition primitives (`SavantPattern` + `TacticInvocation` + `EdgeEmissionSpec` + `AtomTouchMask`) in `lance-graph-contract`, zero-dep, sits next to `atoms`/`recipe_kernels`/`nars`. **Group E** — `crates/lance-graph-callcenter/src/role_keys.rs` with 25 disjoint Vsa16kF32 slices per `I-VSA-IDENTITIES` Layer-2 catalogue (slice manifest coordinated with `grammar/role_keys.rs`). **Group F** — 25 typed `SavantPattern` consts in callcenter drawn from `.claude/odoo/savants/<N>.md` slot 1/4 + `.claude/odoo/L*.md` business semantics. **Group G** — deprecate v1 surface (`Reasoner` trait + 4 impls + `SavantConclusion` + `SavantSuggestion` + `build_conclusion`) under `legacy-reasoner` feature with `#[deprecated]` migration pointers; removal in a follow-up after woa-rs migrates.

### Deliverables
D-ODOO-SAV-5a (Group D): composition primitives in contract · D-ODOO-SAV-5b (Group E): `callcenter/role_keys.rs` 25 disjoint slices + lookup · D-ODOO-SAV-5c (Group F): 25 `SavantPattern` consts · D-ODOO-SAV-5d (Group G): `#[deprecated]` + `legacy-reasoner` feature gate on v1 surface · D-ODOO-SAV-5e: end-to-end test (FiscalPositionResolver → expected `CausalEdge64` row, SPO + NARS + v2 signed mantissa).

### Execution
5a + 5b parallel (additive, zero churn) → 5c after both → 5d after 5c (deprecation pointers name a real target) → 5e throughout (per-D-id tests + e2e proof in 5c completion). woa-rs consumer migration OUT OF SCOPE but UNBLOCKED by 5d. Board hygiene: this plan + INTEGRATION_PLANS PREPEND + STATUS_BOARD rows + EPIPHANIES `E-SAVANT-COMPOSITION-1` land in the same commit as 5a.

### Invariants
AGI-as-glove (new capability = column population, not new trait) · `I-VSA-IDENTITIES` (identity in `role_keys`, content in atoms/tactic/emission/AriGraph, never bundled) · The Click P-1 (no free function on carrier's state) · `I-LEGACY-API-FEATURE-GATED` (v1 deprecation under feature + migration pointer) · suggestion-only never un-guarded write (woa-rs Iron Rule 7).
## 2026-05-27 — multi-server-cognition-expansion-v1 (legacy-stack displacement + Raft-log / SoA-state-machine)

**Status:** PROPOSAL — §2 displacement is current-state; §3 multi-server expansion is unbuilt, gated on the §4 determinism probe
**Confidence:** HIGH vs JanusGraph + Cassandra; Zitadel displaced by Ory (Kratos+Hydra) for authN; HIGH the log-replicated-SoA shape is correct; LOW it is proven (determinism unproven)
**Plan file:** `.claude/plans/multi-server-cognition-expansion-v1.md`
**Predecessors:** `cognitive-substrate-convergence-v1` (belief-delta log = the episodic arc); `surrealdb/core/src/kvs/lance` (the LSM-on-lance engine the WAL-as-Raft-log rides)

### Scope

Kills the "for multiple servers we need JanusGraph / Cassandra / Zitadel" argument. (1) Component displacement: distributed graph + CP via SurrealDB-on-TiKV + lance-graph; Cassandra's AP is the wrong consistency model for a belief substrate; **Ory** (Kratos+Hydra) fills the one real gap (authN/IdP), binary stays verification-only. (2) Multi-server path: Raft replicates the **belief-delta / episodic LOG**, the zero-copy SoA is the per-node **state machine**, the Rubikon `commit_gate` is the Raft **append point**; TiKV-the-log sits *under* lance-the-state (not "instead of"). Hard prerequisite: byte-deterministic NARS apply (`reencode_safety` / D-SDR-26) — **unproven**; next deliverable is the determinism probe, not more synthesis. authN (Ory) is orthogonal to the consensus layer.
## 2026-05-27 — bindspace-singleton-to-mailbox-soa-v1 (dissolve the shared `Arc<BindSpace>` into per-mailbox `MailboxSoA<N>` ephemeral thoughtspace)

**Status:** PROPOSAL / design (migration spec; NOT yet implemented). **Plan file:** `.claude/plans/bindspace-singleton-to-mailbox-soa-v1.md`. **Epiphany:** `E-MAILBOX-IS-BINDSPACE`.
**Scope:** `MailboxSoA<N>` *becomes* the per-mailbox, mailbox-owned, ephemeral "thoughtspace" — the BindSpace surrogate. The singleton `ShaderDriver.bindspace: Arc<BindSpace>` (`cognitive-shader-driver/src/driver.rs:56`; one `BindSpace::zeros(4096)` in `bin/serve.rs:29`) is **dissolved**, not copied. Column map: drop the 64 KB `Vsa16kF32` `cycle` plane; own `edges`/`qualia`/`meta`/`entity_type` in the mailbox; reference content via CAM-PQ; keep `ontology` a shared `Arc`.
**Deliverables:** D-MBX-1 add migrated columns to `MailboxSoA<N>` (feature-gated) · D-MBX-2 move `engine_bridge` per-row surface onto mailbox rows · D-MBX-3 driver holds sea-star of mailboxes (kill the singleton) · D-MBX-4 death→SPO+Lance tombstone-witness · D-MBX-5 delete `BindSpace`+`cycle` plane.
**Gates:** `D-CE64-MB-1-impl` (par-tile) + `PR-NDARRAY-MIRI-COMPLETE` first; D-MBX-5 blocked on the CLAUDE.md "The Click"/`Vsa16kF32` doctrinal update (OQ-4).
**Invariants:** `E-CE64-MB-4` (ownership = compile-time no-alias) · `E-BATON-1` (LE baton is the only cross-boundary state) · `I-VSA-IDENTITIES` (reference content, don't copy planes) · `I-LEGACY-API-FEATURE-GATED` (feature-gate v1 `BindSpace` accessors during S1–S4) · no double-mailbox (sync inner / ractor outer).

---
## 2026-05-27 — odoo-savant-reasoners-v1 (lance-graph side of the Odoo richness harvest: 2 new OGIT families + Layer-2 axioms + StyleCluster wiring + 5 Reasoner impls)

**Status:** PROPOSAL (picks up the cross-repo handover boundary in `.claude/odoo/SAVANTS.md` §"lance-graph handover boundary"). woa-rs defined the 25-Savant roster + delegation tuples; lance-graph implements (a) Reasoner impls, (b) 2 new families + Layer-2 alignment axioms for the `None` classes, (c) StyleCluster wiring.
**Confidence:** HIGH on (b)/(c) — additive extensions of `odoo_alignment.rs` seed + alignment TTLs. MED on (a) — Reasoner dispatch shape (one impl per ReasoningKind vs savant-config registry) pinned but needs a review pass.
**Plan file:** `.claude/plans/odoo-savant-reasoners-v1.md`
**Predecessors:** PR #412 (odoo hydrator + dolce_odoo classifier + ODOO slot 50), PR #413 (briefing pack).
**Anchored iron rules:** I-VSA-IDENTITIES (savant = Layer-2 role catalogue), AGI-as-glove, board-hygiene, Iron Rule 1 (no brain-crate in customer binary), Iron Rule 7 (verhaltens-bewahrend — reasoner output is suggestion-only).

### Scope
Group B — `0x63 ProductCatalog` (Analytical) + `0x90 HRFoundation` (Empathic) families + Layer-2 alignment axioms for `stock.*` / `account.analytic.distribution.model` / `account.account.tag` (land on existing pivot where honest, else documented `None`). Group C — `StyleCluster` per family (field-or-sidecar). Group A — `SavantConclusion` + 5 `Reasoner` impls (one per `ReasoningKind`) in lance-graph-callcenter, dispatching on evidence + family style, `InferenceType::default_strategy()` → QueryStrategy, NarsTruth evidence fusion.

### Deliverables
D-ODOO-SAV-1 two new families + seed rows + family_registry.ttl · D-ODOO-SAV-2 Layer-2 alignment axioms TTL · D-ODOO-SAV-3 StyleCluster per family · D-ODOO-SAV-4 5 Reasoner impls (gated on dispatch-shape review, own PR).

### Execution
D-ODOO-SAV-1/2/3 additive + low-risk → first PR (this session). D-ODOO-SAV-4 → follow-up PR after `/code-review` on dispatch shape. Plan + INTEGRATION_PLANS prepend land with D-ODOO-SAV-1.

### Invariants
Option B (inherit existing slots; new families are genuine basins not per-class mints; `None` stays `None` w/o honest pivot) · public OWL pristine (axioms are NEW TTL) · savant = Layer-2 catalogue · reasoner output = suggestion (guard stays in woa-rs) · impls in callcenter behind contract `Reasoner` trait.
---
## 2026-05-27 — odoo-savant-roster-v1 (the lance-graph side of the woa-rs Odoo savant delegation: 25 delegated reasoners)

**Status:** PROPOSAL. **Plan file:** `.claude/plans/odoo-savant-roster-v1.md`. **Source:** `.claude/odoo/SAVANTS.md` + L1–L15 (PR #413). **Predecessor:** PR #412 (odoo→FIBO/SKR alignment + DOLCE classifier).
**Scope:** 25 Odoo savants = delegated reasoners (woa-rs keeps the AXIS-A deterministic guard; the ambiguous AXIS-B core delegates to lance-graph via `reasoning::Reasoner`). Each savant = a dispatch tuple (OGIT family · `ReasoningKind` · `InferenceType` · `SemiringChoice` · `StyleCluster`).
**Deliverables:** D-ODOO-1 roster-as-data (`contract::savants`, ✅ DONE) · D-ODOO-2 `Reasoner` impls per `ReasoningKind` · D-ODOO-3 new OGIT families `0x63 ProductCatalog` + `0x90 HRFoundation` + style wiring · D-ODOO-4 Layer-2 alignment axioms for the `None` classes (stock.*, analytic.distribution.model, account.account.tag) · D-ODOO-5 delegation call-site conformance (ReasoningContext + Arrow EvidenceRef schemas).
**Invariants:** suggestion-only never un-guarded write (Iron Rule 7) · deterministic guard stays woa-rs · BBB-allowed crates only · tuple fully determines dispatch · business = OGIT-inherited sidecar (odoo inherits FIBO/SKR slots; 0x63/0x90 are the only new families, need ratification).

---

## 2026-05-27 — atom-mailbox-substrate-v1 (ladder-serves-mailbox: atoms→styles→personas, quorum projection, counterfactual mantissa, AriGraph hot/cold/tombstone)

**Status:** PROPOSAL (implements `EPIPHANIES.md` E-LADDER-SERVES-MAILBOX; extends `rung-persona-orchestration-v1` D-PERSONA-1 downward into the atom layer and outward into the mailbox lifecycle).
**Confidence:** HIGH on the mechanism→shipped-type mapping; **CONJECTURE on the atom basis (D-ATOM-0, the load-bearing unsolved decision)** and the I4-32D SIMD layout until probed.
**Plan file:** `.claude/plans/atom-mailbox-substrate-v1.md`
**Predecessors:** `rung-persona-orchestration-v1`, `rung-mul-grounding-v1`, `cognitive-substrate-convergence-v1`.
**Anchored iron rules:** `I-VSA-IDENTITIES`, `E-BATON-1`, `I-LEGACY-API-FEATURE-GATED`, The Click.

### Scope
Three-layer cognitive basis under the mailbox-served ladder: **atoms (bipolar I4-32D, 32 dims / 64 poles) → thinking styles (compositions) → persona recipes (compositions + thresholds + β)**. Each atom *measured by a quorum* (`(position, confidence)` = NARS truth per axis; splits = Contradiction never averaged); split-poles *preserved as a `CausalEdge64` −6 counterfactual mantissa* (ghost-tier test → `awareness.revise`); memory *ephemeral-hot in mailbox → calcified-cold SPO + Lance tombstone-witness* (GoBD audit by construction). wisdom↔Staunen = sampling temperature (self-regulated by free energy; the `WisdomMarker` 0.1 floor = min temperature).

### Decision gates (block scaffolding)
D-ATOM-0 atom-basis route (ICA/PCA over 36 / theory-driven from 6 clusters / hybrid) · D-ATOM-0b NARS as categorical register (Test 0, recommended) vs bipolar atoms.

### Deliverables
D-ATOM-1 atom catalogue + `I4x32` type + pack/SIMD (`contract::atoms`, blocked on D-ATOM-0) · D-ATOM-2 style/persona Cranelift recipe templates (`contract::jit`/`thinking`) · D-ATOM-3 quorum-projection per axis (`contract::escalation`/`a2a_blackboard`) · D-ATOM-4 counterfactual mantissa v2 deposit / v3 mailbox+revision (basis-independent) · D-ATOM-5 AriGraph hot/calcify/tombstone (basis-independent).

### Execution
Sonnet `///`-scaffold wave (disjoint file scopes, BLOCKED-not-guess) → P2 review (`/code-review` high, ultra for D-ATOM-1/2; no literal codex binary) → implement+remove stubs → per-deliverable PR into the working branch → subscribe+autofix CI → merge → repeat. Parallel-now: D-ATOM-4 v2, D-ATOM-5 (basis-independent); D-ATOM-1/2 spawn after D-ATOM-0.

### Invariants
persona=Layer-2 (no container) · NARS type in register (Test 0) · markers ≤32 (I-VSA-IDENTITIES) · splits=Contradiction never averaged · counterfactual in separate lane · one graph · no persisted singleton (E-BATON-1) · ractor async only at swarm boundary · bounded respawn · `latency_budget` arbiter, no hot-Pod wall-clock · SIMD gated on `ndarray-vertical-simd-alien-magic.md`.

---

## 2026-05-26 — rung-persona-orchestration-v1 (time-bound persona orchestration: boring checklist → meta-recipe → hot/cold/feedback anneal)

**Status:** PROPOSAL (sibling to `rung-mul-grounding-v1`; the time-bound + composition layer)
**Confidence:** HIGH on structure (hot/cold/feedback = the original ladybug architecture + OpenAI macro-evals + ADK Memory Bank, converged); MED on ractor adoption + macro-eval harness (net-new).
**Plan file:** `.claude/plans/rung-persona-orchestration-v1.md`
**Predecessors:** `rung-mul-grounding-v1` (b4efb55), `rung-ladder-grounding-v1` (b0ef6fa), `cognitive-substrate-convergence-v1`.
**Design refs (read-only general-web; ladybug-rs is outside GitHub-MCP scope):** ladybug-rs `INTEGRATION_PLAN.md` @177a321 §"BF16 Superposition (Hot/Cold/Feedback)" L542+ + 4-phase `[DONE]/[TODO]` gate + 3 composition modes + BindSpace-blackboard; `src/spectroscopy/detector.rs` @177a321; Claude chief-of-staff; OpenAI macro-evals; Google ADK Memory Bank.

### Scope

Ground (restore-on-SoA, NOT port) ladybug's hot/cold/feedback loop + phase-gate checklist + blackboard composition onto our contract/SoA floor, as the time-bound layer over the `rung-mul` experience curve.

- **Two orthogonal orderings × time budget:** epistemic experience-curve (Axis A) × social etiquette arc (Axis B), arbitrated by `latency_budget` (`elevation/mod.rs:131`). 2D menu phase×DK-position; etiquette = soft prior + anytime graceful-degradation, not rigid FSM.
- **Boring checklist (verify, temp≈0):** hard gates (contracts/SoA/store/NARS/thresholds/FreeEnergy) vs soft (capabilities/wisdom-store/eval — degrade). Continuous health invariant: red-at-runtime → let-it-crash → supervisor restart = rung-shift + NaN→Lab.
- **Meta-recipe (compose, cold):** declarative child-spec manifest (data not code, macro-evaluable); blackboard composition on `a2a_blackboard`/SoA (per ladybug BindSpace), ractor supervises + carries Batons.
- **Hot/cold/feedback (L542 grounding):** hot = annealed cognitive cycle; cold = macro-eval = **wisdom-marker factory** (ladybug `CrystalCodebook` "lived history" → our *distilled calibrated* marker); feedback = hydrate-before-the-fact (ADK Preload).
- **Temperature anneal:** explore hot → exploit cold, **evidence-gated** (Boole-bound caps cooling — no premature Mount Stupid). Grounded: detector `noise_tolerance=base·(1+(1−conf)·0.5)`, `fanout=base·(1+bridgeness·0.5)` (bridgeness=macro-eval suspect-bridge=our work-metric, triple convergence).
- **Substrate:** ractor YES (outer swarm under `OrchestrationBridge`, async only at boundary, SoA inner sync); surrealdb NO for cognitive (redundant w/ lance-graph/AriGraph, not boring; prefer SQLite/Lance operational); AriGraph = the one graph.

### Deliverables

D-PERSONA-1 hard/soft checklist verifier (~180) · D-PERSONA-2 meta-recipe manifest (recipe-as-data, ~150) · D-PERSONA-3 hot/cold/feedback wiring + CrystalCodebook→wisdom-marker + Preload hydrate (~240) · D-PERSONA-4 macro-eval harness (scenario→trace→discover→diagnose, suspect-bridge=blasgraph betweenness, ~280, HIGH) · D-PERSONA-5 ractor outer-swarm runtime (~200).

### Honest gaps vs original

ladybug `detector.rs` has NO null/dead-end/escalation ("all inputs produce valid output") → our NaN→cautious-exploration→Lab + dead-end-as-work is net-new. `CrystalCodebook` dumps lived history → we distill it into a calibrated marker (Boole-bound, ≤32 identities). ractor + etiquette arc not in original.

### Invariants

restore-on-SoA not port · hard/soft graceful-degradation · recipe-as-data (macro-evaluable) · evidence-gated anneal (Boole-bound cooling cap) · blackboard composition not direct calls · ractor async only at swarm boundary · no second graph · I-VSA-IDENTITIES (markers ≤32) · `latency_budget` time arbiter.

---

## 2026-05-26 — rung-mul-grounding-v1 (the MUL fine-tuned into the ladder as an experience curve over the SPO 2³ NARS decomposition)

**Status:** PROPOSAL (follow-on to `rung-ladder-grounding-v1`)
**Confidence:** HIGH on structure (it is the Dunning-Kruger curve mechanized); MED on the per-projection `SpoHead` refactor; CONJECTURE on the wisdom-marker calibration readout until D-RUNG-MUL-4 tests it.
**Plan file:** `.claude/plans/rung-mul-grounding-v1.md`
**Predecessors:** `rung-ladder-grounding-v1` (b0ef6fa), `cognitive-substrate-convergence-v1` (CausalEdge64 v2 §6 — causal mask = Pearl 2³ IS the rung axis), `E-AGICHAT-DIMENSION-CONTRACT` (afabefd), `E-I4-META-1`.

### Scope

Grade the coarse integer rung ladder with the MUL, organized as an **experience curve**: every strategy ordered by the evidence level at which it becomes *necessary* — which collapses into the Dunning-Kruger curve with a mechanical trigger at each point.

- **SPO 2³ corrected:** it is the **powerset of {S,P,O}** (8 evidential projections `___,S__,_P_,__O,SP_,S_O,_PO,SPO`) for causality testing through NARS **decomposition** — NOT a distance-cube/popcount. `nars_engine.rs` today computes `all_projections() -> [u32;8]` as *distances* and `SpoHead` carries *one* truth; de-grounding that to per-projection truth is D-RUNG-MUL-1.
- **Causation = screening-off:** `S_O` strong but screened off by P (`SP_`∧`_PO`) ⇒ spurious/mediated; all projections compared to `___` for lift over base rate.
- **Work (exploit):** decomposition + screening-off coverage, **confidence/expectation-gated (never frequency)**, AIKR-gated by `budget.quality`. Two curves over one axis: work climbs monotone, confidence is DK-shaped; **wisdom = calibration gap `|conf−competence|→0`**.
- **Two sparse-data routes:** NaN sentinel ("no field") → cautious-exploration (Exploratory, high exploration_rate) + `ElevationLevel`↑ + **Lab request**; sparse field → gaussian splat → `FreeEnergy::compose` as the *sole* confidence source (F caps confidence ⇒ **no data ⇏ overconfidence**). Explore drive = `wonder` × `free_will_modifier` × trust.
- **Wisdom markers:** long-term VSA-**identity** bundle (≤32 per I-VSA-IDENTITIES; truths in content store) hydrated *before the fact* as the KL prior — the curve becomes a spiral.

### Deliverables

D-RUNG-MUL-1 per-projection NARS truth (`SpoHead` 8 `(f,c)`, planner ~220) · D-RUNG-MUL-2 NaN→cautious-exploration+Lab gate, distinct from `c=0` (~160) · D-RUNG-MUL-3 wisdom marker (identity bundle + hydrate-as-KL-prior, contract+planner ~180) · D-RUNG-MUL-4 screening-off work + Boole/Fréchet bound + calibration-gap readout (~150) · D-RUNG-MUL-5 splat→`FreeEnergy::compose` as sole sparse-data confidence (~120).

### Invariants

Confidence-gated never frequency-gated (frequency alone = Mount Stupid) · Boole/Fréchet bound on conjunction confidence · no data ⇏ overconfidence (only FreeEnergy or floored-NaN may signal) · I-VSA-IDENTITIES (markers ≤32 identities, content in store) · AIKR `budget.quality` fanout cap · AGI-as-SoA (markers = column ops, not a new service) · decomposition not distance-cube. Folds into `elevation/homeostasis.rs` (MUL-L6) beside `evaluate_rung_shift`; does not fork.

---

## 2026-05-26 — rung-ladder-grounding-v1 (the most-obvious first grounding of the agichat gestell)

**Status:** PROPOSAL
**Confidence:** HIGH — deterministic integer/threshold logic, zero VSA in the decision path; cleanest possible first restore.
**Plan file:** `.claude/plans/rung-ladder-grounding-v1.md`
**Predecessors:** `E-AGICHAT-DIMENSION-CONTRACT` (afabefd), `E-I4-META-1`, `E-BATON-1`; shipped floor ndarray `SoaColumns` (42cb7123) + i4-32 unpack (8de1dcf8).
**Follow-on (planned, user-flagged):** `rung-mul-grounding-v1` — the **MUL fine-tuned into the ladder**: ladybug's 10-layer MUL (`MulSnapshot`) becomes the *trigger source* refining the ladder's coarse binary triggers into graded escalation (DK MountStupid → escalate; homeostasis Anxiety + allostatic-load → escalate; false-flow → escalate; gate-block reason → escalate). `elevation/homeostasis.rs` is already MUL-L6 — ladder + MUL co-finetune there.

### Scope

Ground agichat's **RungShift ladder** + **CollapseGate SD** as LE-contract types/logic on the SoA floor. The ladder was never inflated (ladybug-rs `rung.rs` is a faithful port) — the work is to express it as a bit-exact Pod and wire its triggers to grounded signals.

- **CollapseGate:** SD over candidate scores → `FLOW(<0.15)/HOLD/BLOCK(>0.35)`; SD = dispersion, not confidence.
- **RungShift:** rung 0-9, bands 0-2/3-5/6-9; triggers sustained-block(≥3) / predictive-failure(avg P<0.3 / window 5) / structural-mismatch → +1 (cap 9); tick-based cooldown.
- **Grounding:** `RungState` = 16-byte `#[repr(C)]` Pod (no `Vec` — fixed `[u8;5]` P-ring; tick cooldown; u8/i4-quantized scores) in a `SoaColumns` column; `evaluate_rung_shift` PURE (no `&mut` during compute) folded into `lance-graph-planner/src/elevation/` beside `homeostasis.rs`; SD via ndarray SIMD; `GateState` into `collapse_gate.rs`.
- **Hook:** RungLevel = the **R1-R9 dim-group** of the 33-TSV (`ThinkingStyleI4_32D`).

### Deliverables

D-RUNG-1 contract types (lance-graph-contract, ~150) · D-RUNG-2 pure ladder logic in `elevation/` (planner, ~200) · D-RUNG-3 `RungState` SoA column + tick update (~100) · D-RUNG-4 SD→GateState in `collapse_gate.rs` + rung→TSV-R1-R9 map (~120). Parity tests vs verbatim agichat semantics.

### Invariants

No `Vec`/alloc in hot Pod · no `&mut` during compute (pure evaluate, builder apply) · tick-based not wall-clock · integer rung (no float-resonance carrier — the de-grounding ladybug-rs did) · SD = dispersion not confidence · RungShift separate from SD.

---

## 2026-05-15 — cognitive-substrate-convergence-v1 (CSV — i4 mantissa + gapless baton + active inference)

**Status:** Active (PROPOSAL — awaits OQ-CSV-1..6 ratification before sprint-11 D-CSV-* spawn)
**Confidence:** HIGH on architecture; MED on i4-16D qualia per-dim assignment (OQ-CSV-1); HIGH on i4-mantissa NARS (PR-LL-1 already shipped equivalent in `nars_dispatch.rs`); HIGH on gapless-baton model
**Plan file:** `.claude/plans/cognitive-substrate-convergence-v1.md` (~46 KB, 18 sections)
**Predecessors:** `causaledge64-mailbox-rename-soa-v1` (PR #372), `neurosymbolic-rlvr-causal-curriculum-v1` (PR #373), PR-LL-1 (PR #375)

### Scope

Locks the architectural decisions made during sprint-10 + the post-sprint-10 cross-session A2A discussion (2026-05-15) before context dilution. Consolidates 7 design questions that converge into ONE substrate: CausalEdge64 v2 layout + QualiaColumn quantization + CollapseGate wire format + WitnessCorpus pointer + MUL evaluation + Σ-tier Rubicon orchestration + thinking-engine ↔ SoA reunification.

### The five compressions

1. **Encoding** — signed i4 mantissa family across NARS / Qualia / ThinkingAtom / direction
2. **Wire format** — discrete `Vec<(u16, CausalEdge64)>` baton tuples between mailboxes (no analog `Vsa16kF32` envelope)
3. **Addressing** — i4 payload IS its own CAM key (content = address; 16¹⁶ ≈ 1.8×10¹⁹ unique states)
4. **Temporal axis** — structural (chain-position + AriGraph anchor), not stored in edge
5. **Cycle driver** — entropy-driven (free-energy gradient), not request-driven

### Final CausalEdge64 v2 bit layout (plan §6)

```text
[0:23]  S/P/O palette indices  (3 × u8)
[24:39] NARS frequency + confidence  (2 × u8)
[40:42] Causal mask  (3b — Pearl 2³, IS the rung axis; counterfactual at 0b111 SPO)
[43:45] Direction triad  (3b)
[46:49] Inference mantissa  (4b SIGNED — direction × rule)
[50:52] Plasticity flags  (3b)
[53:58] W slot  (6b — discourse corpus root handle)  ← NEW
[59:60] Truth-band lens  (2b — 4 lens states incl. "13% ambiguous direction")  ← NEW
[61:63] Spare  (3b — sprint-12+ probe headroom)
Total   64b zero unused
```

### 12 deliverables (D-CSV-1..D-CSV-12) across sprints 11-13

| Phase | D-id | Title | Sprint | LOC | Risk |
|---|---|---|---|---|---|
| A | D-CSV-1 | `causal-edge` v2 layout per §6 | 11 | ~250 | LOW |
| A | D-CSV-2 | `QualiaI4_16D` type + f32 ↔ i4 migration helpers | 11 | ~180 | LOW |
| A | D-CSV-3 | InferenceType signed-mantissa expansion + absorb PR-LL-1 variants into canonical edge enum | 11 | ~120 | MED |
| A | D-CSV-4 | `CollapseGateEmission` wire format spec + impl | 11 | ~150 | LOW |
| B | D-CSV-5 | QualiaColumn `[f32; 18]` → `QualiaI4_16D` (5a sibling-column / 5b cutover) | 11 | ~400 | HIGH |
| B | D-CSV-6 | `WitnessCorpus` (CAM-PQ-indexed) replaces `SpoWitnessChain<32>` | 11 | ~600 | HIGH |
| B | D-CSV-7 | MailboxSoA integration: W-slot + plasticity accumulator + apply_edges | 11 | ~350 | MED |
| C | D-CSV-8 | MUL evaluation in integer SIMD (i4 × i4 → i8 products) | 12 | ~500 | MED |
| C | D-CSV-9 | 8-channel ↔ SPO-palette transcoder (Option R-3) at L3 commit | 12 | ~180 | LOW |
| C | D-CSV-10 | Σ-tier Rubicon-resonance dispatch in SigmaTierRouter | 12 | ~250 | MED |
| D | D-CSV-11 | Vertical streaming structs in ndarray (qualia.history, inference.trajectory, splat.evolve) | 13+ | ~700 | HIGH |
| D | D-CSV-12 | Splat shader op fleet on i4 substrate (splat_gaussian, score_*, emit_if_epiphany) | 13+ | ~800 | MED |

Total: ~4,480 LOC across 12 PRs. Cross-spec patches to sprint-10 W2/W3/W4/W5/W6/W7/W10/W11 specs estimated ~870 LOC (bundled into one sprint-11 prep PR).

### What this locks (vs prior plan)

- **Resolves sprint-10 meta-review CSI-1** (`.claude/board/sprint-log-10/meta-review.md`) with the definitive v2 bit layout per §6
- **Resolves dual-CausalEdge64 finding (E-META-7)** via Option R-3 transcoder at L3 commit boundary
- **Locks the i4 substrate family** unifying qualia + NARS mantissa + ThinkingAtom + direction into one quantization vocabulary
- **Locks gapless-baton wire format** — no analog envelope between mailboxes; `Vsa16kF32` narrows to intra-tier Markov + crystal carrier + grammar bind/unbind testing
- **Locks MailboxSoA semantics** as spatial-temporal meaning accumulators, not channels
- **Locks Σ-tier Rubicon-resonance orchestration** — F-gradient driver, not request-response
- **Locks active-inference cycle driver** — "can't stop thinking" per CLAUDE.md doctrine; entropy-of-state IS the dispatch trigger

### Open questions (6) requiring user ratification

1. **OQ-CSV-1** — Qualia 16D per-dim assignment (BLOCKS D-CSV-2): proposed layout in plan §7.2 needs `qualia-engineer` agent cross-check
2. **OQ-CSV-2** — W-slot width 6 vs 8 bits (BLOCKS D-CSV-1): 6=64 corpora generous, 8=256 corpora for multi-tenant SaaS
3. **OQ-CSV-3** — Spare bits reserved vs pre-allocated (non-blocking): default reserved
4. **OQ-CSV-4** — QualiaColumn migration phasing (BLOCKS D-CSV-5): default sibling-then-cutover (lower risk)
5. **OQ-CSV-5** — Pre-computed Magnitude (Staunen×Wisdom) column vs on-demand (non-blocking): default on-demand
6. **OQ-CSV-6** — Σ10 Rubicon threshold Jirak-derived vs hand-tuned (BLOCKS D-CSV-10 sprint-12): hand-tuned acceptable per `I-NOISE-FLOOR-JIRAK` if TECH_DEBT documented

### Cross-references

- 12 sprint-10 specs at `.claude/specs/` (W1-W12); 11 require small spec patches per plan §12
- 8 sprint-10 knowledge docs at `.claude/knowledge/causal-edge-64-*.md`, `spo-*.md`, `ogit-*.md`, `cognitive-shader-driver-*.md`, `splat-*.md`
- Meta-review at `.claude/board/sprint-log-10/meta-review.md` — CSI-1 resolved by this plan's §6
- PR #375 PR-LL-1 — Intervention/Counterfactual variants already in `nars_dispatch.rs`, this plan absorbs them into canonical edge enum
- PR #379 — 4-branch retirement (orphan sweep) cleared the pre-condition

---

## 2026-05-14 — neurosymbolic-rlvr-causal-curriculum-v1 (LL-CURRICULUM)

**Status:** Active (PROPOSAL — curriculum landed, 5-PR roadmap ratification pending §7 OQs)
**Confidence:** High (composition of published methods + existing substrate; no novel research)
**Plan file:** `.claude/knowledge/neurosymbolic-rlvr-causal-curriculum-v1.md`
**Predecessor:** `causaledge64-mailbox-rename-soa-v1` (PR #372 — landed)

### Scope

8-paper curriculum + 5-PR implementation roadmap for the stack's learning layer. Composes Schölkopf-style structural causal models, MIT-style Bayesian program learning, Solar-Lezama × Tenenbaum neurosymbolic dispatch (LINC), and DeepSeekMath-style RLVR into one substrate that turns the existing `Think` struct (post-PR #372) into a self-improving system.

### What this composes

- **Causal de Finetti** (Guo+Schölkopf 2022, arXiv:2203.15756) → AriGraph SPO-G grouping doctrine
- **LPN** (Bonnet 2024, arXiv:2411.08706) → `StyleVectors` test-time gradient adaptation
- **LINC** (Olausson+Solar-Lezama+Tenenbaum 2023, arXiv:2310.15164) → Σ9-Σ10 EPIPHANY classical-prover dispatch
- **Executable Counterfactuals** (Vashishtha 2025, arXiv:2510.01539) → Pearl 2³ trainable verbs + RL>SFT for OOD
- **Conformal CFG** (Farzaneh 2026, arXiv:2601.20090) → calibrated counterfactual sets for MedCare-rs / q2 safety
- **TextGrad** (Yuksekgonul 2024, arXiv:~2406.07496) → closed-loop style optimizer (textual gradient)
- **Opt-Sym** (Yeo+Solar-Lezama 2026) → symbolic-space adaptive data generation
- **GRPO/DeepSeekMath** (Shao 2024, arXiv:2402.03300) → RLVR trainer algorithm

### 5-PR sequencing (this curriculum doc is governance only; the 5 implementation PRs follow)

| # | Scope | LOC | Risk |
|---|---|---|---|
| PR-LL-1 | NARS Intervention/Counterfactual InferenceType variants + AriGraph::intervene_on | ~200 | Low (additive to enum) |
| PR-LL-2 | ICM-invariance BindSpace column + `lance-graph-planner::data_gen` (Opt-Sym generator) | ~800 | Med (new SoA column + new module) |
| PR-LL-3 | Hybrid TextGrad/LPN `style_synthesize` (numerical + textual gradient on StyleVector) | ~400 | Med (closes Gap 1) |
| PR-LL-4 | `crates/lance-graph-trainer/` (GRPO loop, candle/burn-backed) | ~800 | High (new training crate, ~2 weeks prep work) |
| PR-LL-5 | `crates/linc-bridge/` (Z3 prover + conformal CFG wrap) | ~600 | Med (new crate, external dep on z3-rs) |

Sequential: each PR is a precondition for the next. PR-LL-4 requires ~2 weeks of separate Qwen3-head-via-candle prep work before fan-out.

### Closes / unblocks

- `THINKING_ORCHESTRATION_WIRING.md` **Gap 1** (Contract Not Consumed — 12 vs 36 ThinkingStyle) → PR-LL-3 learns the missing 24 from runtime trajectories
- `THINKING_ORCHESTRATION_WIRING.md` **Gap 4** (Elevation not connected) → SigmaTierRouter consumes PR-LL-3's free-energy gradient as elevation signal
- **Pearl 2³ named-but-not-dispatched** → PR-LL-1 makes intervene/counterfactual first-class verbs
- **L4 planner shell empty** → PR-LL-5 fills with LINC dispatch + conformal calibration
- **TD-LEARNING-LOOP-MISSING** (implicit; no doc exists for the unwired GRPO trainer) → PR-LL-4

### Blast radius

- **New crates:** `lance-graph-trainer` + `linc-bridge` (~1400 LOC together)
- **Crates modified:** `lance-graph-planner` (data_gen + style_synthesize modules), `causal-edge` (Intervention/Counterfactual variants), `lance-graph-contract` (StylePoolProvider trait per OQ-LL-4)
- **Zone 3 surface UNCHANGED**
- **External deps added:** `z3-rs` (PR-LL-5), `candle` or `burn` (PR-LL-4) — both gated behind feature flags
- **ndarray side:** UNCHANGED (the curriculum stays on the thinking-side of the doctrinal split)

### Open Questions (6 — ratify before sprint fan-out)

OQ-LL-1 reward shape (graded NARS confidence vs binary) · OQ-LL-2 TextGrad optimizer location (local Qwen3 vs frontier API) · OQ-LL-3 prover choice (Z3 vs Prover9 vs HOL Light) · OQ-LL-4 style-pool location (contract vs separate) · OQ-LL-5 ICM-invariance update protocol · OQ-LL-6 Σ-tier-as-difficulty probe (hot-path latency)

### Iron rule compliance

| Rule | Status |
|---|---|
| I-SUBSTRATE-MARKOV | All synthesized trajectories pass Chapman-Kolmogorov test in PR-LL-2 verify step |
| I-NOISE-FLOOR-JIRAK | PR-LL-5 conformal calibration uses Jirak-derived bounds, not classical Berry-Esseen |
| I-VSA-IDENTITIES | `style_synthesize` produces identity fingerprints; content stays in YAML registries |
| I1 BindSpace read-only | `IcmInvarianceColumn` writes go through `CollapseGate::bundle` |
| Method-on-carrier | All 4 new capabilities are methods on existing carriers |
| AGI-as-glove SoA | Synthesized styles land in `StyleColumn` extension; no new layer |

---

# Integration Plans — Versioned Index

> **APPEND-ONLY.** Every integration plan ever authored for this
> workspace, versioned, with status. New plans append to the top
> as new entries. Superseded plans stay — they are the design arc.
>
> Governance: same rule as `PR_ARC_INVENTORY.md`. The **Status**
> field is the only mutable field per entry. Supersedure is marked
> by adding a new top entry that references the prior version; the
> prior entry is NOT deleted.

---

## APPEND-ONLY RULE

1. **New plans PREPEND** a new section at the top.
2. **Old plan entries are IMMUTABLE** except the **Status** line.
3. **Supersedure:** if plan vN is replaced by vN+1, prepend a new
   entry for vN+1 that cites vN; update vN's **Status** to
   "Superseded by vN+1".
4. **Corrections** to plan scope during its lifetime: append a
   `**Correction (YYYY-MM-DD):**` line to the entry; do not edit
   the original scope line.
5. **Retire but never delete.** When a plan is complete or
   abandoned, update Status and move on. The entry stays.

**Per-entry format:**

- **Plan name + version**
- **Author + date**
- **Scope** — one-sentence goal (immutable)
- **Path** — workspace file location
- **Deliverables** — D-id list (immutable)
- **Status** — **mutable**: Active / Shipped / Superseded / Deferred / Abandoned
- **Confidence** — **mutable**: Working / Partial / Broken — see PR #N


---

## causaledge64-mailbox-rename-soa-v1 — CausalEdge64 mailbox + sparse-rename SoA composition (authored 2026-05-14)

- **Plan:** `.claude/plans/causaledge64-mailbox-rename-soa-v1.md`
- **Author + date:** main thread (Opus 4.7 1M), 2026-05-14, branch `claude/resolve-pr-369-conflicts-ozMXd`.
- **Status:** Active (draft, pre-execution).
- **Scope:** Compose 5 already-authored plans + Σ10 Rubicon doctrine + 1 genuine ndarray-side gap into a single substrate where ractor mailboxes carry CausalEdge64 emissions, share BindSpace via zero-copy views, communicate cross-compartment via SPOW witnesses persisted to AriGraph SPO-G quads, and rename architectural identities (OGIT domain / witness palette / thinking style / truth) into 5-8 bit physical slots via session-ephemeral `AttentionMask` rename SoA. Result: ownership-typed reasoning compartments with compile-time UB-impossibility, ~1.5 KB per compartment, supporting ~24K parallel thoughts at 200ns cycle speed across ≤32 active OGIT domains.
- **Deliverables (immutable, 9 D-ids across 7 PRs):**
  - D-CE64-MB-1 — CausalEdge64 v2 layout extension (G:5, W:6, truth:2 reclaim from reserved 13 bits) in `crates/causal-edge/`
  - D-CE64-MB-2 — PAL8 round-trip regression test (v1 ↔ v2 binary compat)
  - D-CE64-MB-3 — NarsTables LUT regression test (key-bearing bits 0-50 unchanged)
  - D-CE64-MB-4 — `AttentionMask` SoA + LRU eviction + broadcast notifications in `crates/par-tile/`
  - D-CE64-MB-5 — `AttentionMaskActor` ractor singleton wrapping rename tables
  - D-CE64-MB-6 — Property tests for rename round-trip + LRU + eviction broadcast
  - D-CE64-MB-7 — `MailboxSoA<N>` + lifecycle methods (`push_row`, `dispatch_cycle`, `drop_row`) in `crates/par-tile/`
  - D-CE64-MB-8 — `BindSpaceView<'_>` with row-range + column-mask filter (zero-copy borrow into shared `Arc<BindSpace>`)
  - D-CE64-MB-9 — Property tests for spawn-dispatch-prune + XOR-cancel + plasticity counter + intent gate strictness
- **PR sequencing (7 PRs):**
  - PR-CE64-MB-1: `par-tile` crate apex (~1500 LOC, new crate, no consumers yet)
  - PR-CE64-MB-2: CausalEdge64 v2 layout (~400 LOC, binary compat critical)
  - PR-CE64-MB-3: BindSpace Columns E/F/G/H per `bindspace-columns-v1` Phase 2 (~800 LOC)
  - PR-CE64-MB-4: SPO-G quad mode + ghost-edge persistence in AriGraph per `ogit-g-context-bundle-v1` D-OGIT-G-1 (~600 LOC)
  - PR-CE64-MB-5: MailboxSoA + AttentionMaskActor cross-crate wiring (~1200 LOC)
  - PR-CE64-MB-6: SigmaTierRouter + cycle-speed InMemoryMailbox backing + Σ10 Rubicon dispatcher (~1500 LOC, new dispatcher, replaces ad-hoc paths)
  - PR-CE64-MB-7: Bevy `NdarrayCullPlugin` proof plugin (~500 LOC bevy side)
- **Composes (deps):** `bindspace-columns-v1` (Phase 2 implementation) · `oxigraph-arigraph-cognitive-shader-soa-merge-v1` (SPOW §8 + Gaussian splat §9 + 64²/256²/4096² planes) · `ogit-g-context-bundle-v1` (D-OGIT-G-1 SPO-G u32 slot) · `pr-g2-ractor-supervisor` (Tokio shape shipped #366 S7-W3) · `pr-j-1-int4-32d-atoms` (cold-start K-NN fallback) · `thought-cycle-soa-awareness-integration-v1` (AwarenessPlane16K / GrammarMarkovLens64 / ReasoningWitness64 / ThoughtCycleSoA factoring) · `tetrahedral-epiphany-splat-integration-v1` (Gaussian splat integration) · `jc-pillars-runtime-wiring-v1` (JC Pillar 10/11 math kernels).
- **Doctrine anchors:** Σ10 Rubicon Tier Architecture (`linguistic-epiphanies-2026-04-19.md` E21 — 10 tiers × edge-type × Pearl rung × theta mode); VSA switchboard three-layer architecture (`.claude/knowledge/vsa-switchboard-architecture.md`); lab-vs-canonical-surface (`.claude/knowledge/lab-vs-canonical-surface.md` — Wire DTO Zone 3 only); encoding-ecosystem (`.claude/knowledge/encoding-ecosystem.md` — palette/CAM-PQ/HHTL cascade roles).
- **Closes / unblocks:**
  - PR #355 deferred Tier B: FIX-4 (codebook_index bit-collision via 256² PaletteSemiring binning), FIX-5 (`trust_below_floor` wiring test via Column H landing), per-row `BindSpace.context_ids` for `driver.rs:311` (Column H = `TypeColumn: EntityTypeId u16`).
  - `THINKING_ORCHESTRATION_WIRING.md` Gap 1 (Contract Not Consumed — 12 vs 36 ThinkingStyle) via AttentionMask 8-bit-slot rename collapse.
  - `THINKING_ORCHESTRATION_WIRING.md` Gap 3 (JIT pipeline never executed end-to-end) — compartment-spawn consumes `KernelHandle` from `lance-graph-planner::strategy::jit_compile`.
  - `THINKING_ORCHESTRATION_WIRING.md` Gap 4 (Elevation not connected to execution) — SigmaTierRouter IS the runtime elevation policy.
  - TD-INT4-32D-ATOMS-6 (cold-start proximity) — OQ-4 K-NN fallback path.
  - TD-THINKING-ENGINE-UNWIRED-1 (582 KB cognitive substrate dormant) — `BindSpaceView` references resolve thinking-engine encode/decode + lens stack on demand.
  - Type-duplication debt (TrustTexture 4 copies, ThinkingStyle 4 copies) — via lens-collapse + 8-bit-slot rename.
- **Iron rule compliance:** I-SUBSTRATE-MARKOV preserved (Vsa16kF32 retreats to single-cycle Markov bundle; cumulative state moves to AriGraph SPO-G + CausalEdge64); I-NOISE-FLOOR-JIRAK noted as σ-threshold OQ at sign-off; I-VSA-IDENTITIES strengthened (Vsa16kF32 no longer universal-carrier means content-bundling temptation is removed); I1 preserved (CollapseGate single point of mutation); method-on-carrier discipline preserved (no new free functions).
- **Open design questions (OQ-1 through OQ-8 in §11):** Σ-tier banding policy (OQ-1), ghost-edge NARS decay vs fixed-rung (OQ-2), plasticity granularity (OQ-3), INT4-32D cold-start wiring (OQ-4), rayon vendor decision (OQ-5), Vsa16kF32 final residence (OQ-6), AwarenessColumn sizing (OQ-7), SpoWitness shape variants (OQ-8).
- **Confidence (2026-05-14):** Pre-execution. Architecture is composition of named-and-reviewed pieces, not new invention. Risk concentrates in §3 CausalEdge64 bit-layout reclaim (must not break PAL8 serialization or NarsTables LUT layout — explicit regression tests D-CE64-MB-2 + D-CE64-MB-3 gate the merge).
- **Cross-PR refs:** PR #355 (Pillar 0 + cascade columns SHIPPED), PR #366 (sprint-7 + Tokio ractor shape SHIPPED), PR #369 (Tier-A close + lance_cache schema bump SHIPPED), PR #370 (schema versioning + cfg(miri) bypasses + Miri sweep — in-flight on `claude/resolve-pr-369-conflicts-ozMXd` branch).
- **Blast radius (§10):** New code in 5 crates (par-tile NEW + causal-edge / cognitive-shader-driver / lance-graph-supervisor / lance-graph::arigraph extensions). Zone 3 surface (postgrest / drain / grpc / supabase-realtime) completely unchanged. `lance-graph-callcenter` Zone-2 surface unchanged except `CallcenterSupervisor` gains `SigmaTierRouter` sub-actor. **Supabase realtime transcode logic complemented, not retired.**
- **Recursive-eyes acknowledgments:** 3rd pair (bevy session — diamond dep graph + Slice↔Plane bridge + NdarrayCullPlugin proof-first); 4th pair (semantic naming over shape, `MultiLaneColumn` already named, 5-Layer Stack already named); 5th pair (Vsa16kF32 single-purpose correction, two-shape ractor framing, INT4-32D as North Star); 6th pair (ephemeral BindSpace + role-as-mailbox + space-time-collapse + external-intent gate + Ractor-SoA + Think-as-reference); 7th pair (CausalEdge64-as-emission-carrier + truth-collapse + 24K parallel thoughts via 32-slot session-ephemeral sparse rename + 12/34/144 hot-context pattern + zone-naming clarification).

---

## 2026-05-13 — Status correction: `super-domain-rbac-tenancy-v1` Tier A nearly complete; follow-up PR + Tier B+ harvest

- **Plan:** `.claude/plans/super-domain-rbac-tenancy-v1.md` (§1-§19, ~1387 lines)
- **Source PR (merged):** [`AdaWorldAPI/lance-graph#363`](https://github.com/AdaWorldAPI/lance-graph/pull/363) — D-SDR-1 + D-SDR-2 + spec + Codex P2 canonical-name fix; merged at sha `421e71e`, 2026-05-13 07:24Z.
- **Working branch:** `claude/lance-datafusion-integration-gv0BF` (5 commits ahead of `main` post-#363; **follow-up PR not yet opened**).
- **Status:** Active.
- **Tier A (D-SDR-1..5):** D-SDR-1+2 SHIPPED via #363; D-SDR-3 (`2c3e87d`, family codebook) + D-SDR-4 (`1d0157f`, merkle audit) + D-SDR-5 (`dc9e081`, wired authorize_*) committed but unmerged. 96/96 lib tests green; clippy `-D warnings` clean.
- **Consumer wirings:** `medcare-rs` commit `31e999b` + `smb-office-rs` commit `342f601` local, both unpushed.
- **Tier B onward:** NOT STARTED. D-SDR-6 + D-SDR-7 blocked on `AdaWorldAPI/OGIT` MCP scope. D-SDR-27 column inventory blocked on `AdaWorldAPI/MedCare` + `MedCareV2` MCP scope. D-SDR-35..39 unblock LanceProbe M2-M6 in medcare-rs.
- **Spec refinements absorbed (§13-§19):** D-SDR-5 composes onto shipped `PolicyRewriter` chain (~30% LOC reduction lever); §18 collapsed Tier F from ~12 nominal items to **5 endpoints + 1 reduced import tool** (~700 LOC). The "3DES" is broken-single-DES (128-bit truncated, ECB-equivalent, zero IV) → Argon2 backfill on login replaces AES-GCM rewrap. MedCareV2 is overlay-only; LanceProbe IS the drift bridge.
- **Build invariants (§19):** rust 1.94.1 stable; `lance =4.0.0`; `lancedb 0.27.2`; `ndarray::simd` canonical SIMD path; `cargo clippy -- -D warnings` merge gate.
- **Companion docs:** `.claude/handovers/2026-05-13-0852-d-sdr-tier-a-complete-tier-b-and-beyond-pending.md` (formal status), `.claude/handovers/2026-05-13-0855-brainstorm-arc-synthesis.md` (the brainstorming arc + verdicts + outlook + priority-ordered next steps).
- **Scrubbed transcripts:** `.claude/transcript/` (79 jsonl main-window-only, 15 MB raw + 4.5 MB zip, 2026-05-01 → 2026-05-13).
- **Resolves ledger rows:** D-SDR addressing layer adds 4 new contract modules (`unified_bridge`, `super_domain`, `family_table`, `unified_audit`) to `lance-graph-callcenter`; updates `CONTRACT-INV-1` (stale Contract Inventory in `LATEST_STATE.md`) pending the follow-up PR's hygiene update.
- **Next deliverables (priority ordered):** (1) follow-up PR for D-SDR-3..5 + governance hygiene + consumer-side push; (2) self-contained D-SDR-13 (HKDF per super-domain), D-SDR-17 (hard-lock matrix), D-SDR-10 (`JsonLinesAuditSink`), D-SDR-14 (audit replay schema); (3) Tier F harvest D-SDR-18+19 (`MetaBridge` extraction); (4) LanceProbe wiring D-SDR-35..39 once MCP scope expands.

---

## palantir-parity-cascade-v2 — Foundry/Gotham parity capstone + DTO ladder (authored 2026-05-07)

- **Plan:** `.claude/plans/palantir-parity-cascade-v2.md`
- **Companion knowledge:** `.claude/knowledge/soa-dto-dependency-ledger.md` (the SoA DTO entropy ledger; ships with this plan).
- **Author + date:** main thread (Opus 4.7 1M), 2026-05-07 (immediately after PR #352 merge).
- **Status:** Active.
- **Scope:** Integration capstone over 4 prior Foundry parity docs (`q2-foundry-integration-v1`, `lf-integration-mapping-v1`, `foundry-consumer-parity-v1`, `medcare-foundry-vision`) and v1 cascade Pillar 0. **Pillar 0 carry-forward**: Foundry parity IS SoA-as-canon parity — Column H (`EntityTypeId: u16`, PR #272 SHIPPED) is already the Foundry Object Type bridge; v2 just makes the SoA carry the Foundry-equivalent shape, NOT duplicate the table set. **DTO ladder finding (2026-05-07 audit)**: `StreamDto`, `ResonanceDto`, `BusDto` all live in `thinking-engine::dto.rs` (Tiers 0/1/2), upstream of contract. 22 DTOs classified across 3 buckets: 9 bare-metal, 7 SoA-glue, 6 bridge-projection (3 OPEN re-classifications). **Business Logic ↔ Thinking-style ↔ OGIT triangle**: each business operation has 3 faces (`thinking_style: ThinkingStyle` dispatch, `ogit_verb: TTL`, `ogit_entities[]: TTL`); v2 D-PARITY-V2-2 ships the routing table.
- **Originating context:** main-thread requests 2026-05-07: (a) updated roadmap with Foundry/Gotham parity; (b) SoA DTO dependency-graph / entropy ledger to classify bare-metal vs SoA-glue (StreamDto, BusDto, ResonanceDto); (c) cognitive-shader-driver internal vs lance-graph-callcenter external O(1) mapping; (d) Business Logic Thinking-style OGIT mapping later.
- **Resolves ledger rows:** none directly. **Hardens** v1 D-CASCADE-V1-7 (codec cascade column population) via the explicit ledger entry tracking the "OPEN" status of each cascade column.
- **Branch:** `claude/create-graph-ontology-crate-gkuJG`. PR target: `AdaWorldAPI/lance-graph` base=`main`.
- **Confidence (2026-05-07):** Pre-execution. Pillar 0 carry-forward is right per existing PR #272 (Column H is the bridge already). Top-3 ranked: D-PARITY-V2-1 (DTO ledger — ships with this plan), D-PARITY-V2-2 (triangle ledger — ships with this plan), D-PARITY-V2-3 (BusDto bridge into engine_bridge.rs).
- **Cross-plan deps:** v1 D-CASCADE-V1-2 (`SchemaPtr.context_id`) → v2 D-PARITY-V2-4 (`Schema::ObjectView`); v1 D-CASCADE-V1-7 (codec cascade columns) → v2 D-PARITY-V2-12 (`SchemaPtr.thinking_style`); v5 D-9 (`MulThresholdProfile`) → v2 D-PARITY-V2-12 (column extension).
- **Foundry parity status snapshot:** SHIPPED — Column H (PR #272), audit trail, RBAC/RLS, PostgREST. IN PROGRESS — Q2 cockpit. QUEUED — LF-12 Pipeline DAG, LF-20 FunctionSpec, LF-22/23 ObjectView/Notification, LF-50 ModelRegistry.
- **Out of v2 scope:** CRDT scenario branching (Column F already exists; UI affordance is v3), Foundry Marketplace/Compass, Foundry Code Repositories, Vertex/Workshop UX (covered by `q2-foundry-integration-v1`), Foundry-export-format ingest.

---

## ogit-cascade-supabase-callcenter-v1 — OGIT SPO-G + Supabase realtime + Zone 1/2/3 (authored 2026-05-07)

- **Plan:** `.claude/plans/ogit-cascade-supabase-callcenter-v1.md`
- **Author + date:** main thread (Opus 4.7 1M), 2026-05-07.
- **Status:** Active.
- **Scope:** 15 deliverables across `lance-graph-callcenter`, `lance-graph-ontology`, AdaWorldAPI/OGIT (extension fork), and a future `lance-graph-rdf` consumer. Pillar 0 (the holy-grail click): `OntologyRegistry` IS the SoA; per-domain schema (Healthcare, WorkOrder, SMB, CallCenter, Medical) IS the DTO + name→row index. Codec cascade per row: identity Vsa16kF32 → CAM-PQ 6 B → Base17 34 B → palette key 4 B → Scent 1 B → qualia/meta/edge columns. Every step O(1). Pillar 1: OGIT as universal SPO-G lingua franca with `ontology_context_id: u32` per named graph. Pillar 2: Zone 1 (BindSpace, no Serialize) / Zone 2 (Arrow scalar membrane, BBB invariant) / Zone 3 (Supabase RPC, REST, transcode — the only emission point). Pillar 3: smb-bridge + medcare-bridge collapse to 2-line projections over `OntologyRegistry::enumerate(ns)`. Pillar 4: BioPortal arsenal — 10 namespace stubs under `OGIT/NTO/Medical/{ICD10CM,RxNorm,LOINC,FMA,RadLex,SNOMED,MONDO,HPO,DRON,CHEBI}/` carrying provenance + license + size, with full ingestion gated on `lance-graph-rdf-fma-snomed-v1`.
- **Originating context:** main-thread question 2026-05-07: *"should the lance-graph-ontology be the SoA and the schema the DTO + index?"* — answered YES, with the codec cascade chain making it content-addressable through every encoding tier (the holy grail). User-supplied references: `MedCare-rs/.MYSQL/Struktur.sql` (104 tables, 5 dominant prefixes) and `MedCare-rs/releases/tag/bioportal-ontologies-2026-05-05` (25 bundles, ~2.4 GB).
- **Resolves ledger rows:** none directly. **Hardens** v5's D-9 (`MulThresholdProfile` becomes `ontology_context_id`-aware, so medical thresholds are stricter than callcenter thresholds). **Locks down** the BBB membrane doctrine from `callcenter-membrane-v1.md` § 10.9 with a `cert-officer` static check (D-CASCADE-V1-1).
- **Branch:** `claude/create-graph-ontology-crate-gkuJG` (continues the v4/v5 thread). PR target: `AdaWorldAPI/lance-graph` base=`main`. OGIT-fork PRs land under the same branch on the OGIT-fork side.
- **Confidence (2026-05-07):** Pre-execution. Pillar 0 is the only architectural commitment that admits no rollback — and it is right per the existing `LazyLock<&OntologyRegistry>` pattern in `lance-graph-ontology/src/bridges/`. Top-3 ranked: D-CASCADE-V1-1, D-CASCADE-V1-2, D-CASCADE-V1-3 (no upstream blockers).
- **Cross-plan deps:** v5 D-9 (`MulThresholdProfile`), `lance-graph-rdf-fma-snomed-v1` (`SemanticQuad`), `supabase-subscriber-v1` (DM-4 watcher / DM-6 drain), `callcenter-membrane-v1` § 10.9 (BBB iron rule).
- **Out of v1 scope (deferrals):** full SNOMED CT import (license-gated; BioPortal release ships only 666 KB partial), full DRON / CHEBI import (size unclear-payoff; revisit after D-CASCADE-V1-11 measures cascade), n8n-rs / crewai-rust consumption of new SoA columns (separate plan), bgz-tensor attention layer integration (orthogonal).

---

## lance-graph-ontology-v5 — post-merge follow-ons (authored 2026-05-07)

- **Plan:** `.claude/plans/lance-graph-ontology-v5.md`
- **Author + date:** integration-lead (Opus 4.7 1M), 2026-05-07
- **Status:** Active
- **Scope:** Picks up where v4 (`claude/create-graph-ontology-crate-gkuJG`, OGIT#1 merged) left off. 15 deliverables ranked by leverage / cost: D-ONTO-V5-1 (dcterms:source provenance, closes TTL-PROBE-5), D-ONTO-V5-2 (`arigraph::SpoBridge::promote_to_spo`, closes SPO-1), D-ONTO-V5-3 (Healthcare TTL transcode), D-ONTO-V5-4 (smb-ontology export-only, NOT migration — brutal-honest reversal, ratified by main thread 2026-05-07), D-ONTO-V5-5 (q2 TTL transcode), D-ONTO-V5-6/7 (MySQL/MSSQL `SchemaSource` impls), D-ONTO-V5-8 (customer admin form, owned by woa-rs surface), D-ONTO-V5-9 (ontology-aware MUL trust thresholds — registry as namespace-keyed lookup), D-ONTO-V5-10 (callcenter-bridge, deferred until SUBJECT-DTO-1 lands), D-ONTO-V5-11 (woa-rs 80/20 binary cut), D-ONTO-V5-12 (MUL publishers — Brier/damage/sandbox), D-ONTO-V5-13 (hydration parallelism), D-ONTO-V5-14 (Lance dictionary load probe), D-ONTO-V5-15 (in-memory → Lance-backed cutover).
- **Originating context:** v4 OGIT#1 merge (15 entities + 12 verbs in `NTO/WorkOrder/`, master); 36 ontology tests pass; cognitive-shader-driver wired (read-only registry attachment).
- **Resolves ledger rows:** TTL-PROBE-5 (D-ONTO-V5-1), SPO-1 (D-ONTO-V5-2 70+245). Partial leverage on MUL-ASSESS-1 (registry as namespace-keyed threshold table). No leverage on TRUST-1 / FLOW-1 / COMPASS-1 / PARSER-1 (out of scope; the ontology crate has no influence on enum consolidation or the cypher cold/hot split).
- **Branch:** `claude/onto-v5-<D-id>` per deliverable; OGIT-fork PRs per namespace transcode. Upstream `almatoai/OGIT` is never PR'd (ratified 2026-05-07).
- **Confidence (2026-05-07):** Pre-execution. Plan reviews v4's outputs as FINDING-grade and v5's deferrals as honestly-deferred (not punted). Next-3 ranked: D-ONTO-V5-1, D-ONTO-V5-9, D-ONTO-V5-2.
- **Cross-ref:** `.claude/RECON_ONTOLOGY_CRATE.md`, `.claude/DECISION_SPO_ARIGRAPH.md`, `.claude/knowledge/ontology-registry.md`, `sql-spo-ontology-bridge-v1.md` (partially superseded), `foundry-roadmap-unified-smb-medcare-v1.md` (adjacent).
- **Ratifications (main-thread, 2026-05-07):** Q1 smb-ontology export-only — RATIFIED (consistent with v4 "preserved as native fallback"; not a contradiction). Q2 D-9 above D-2 ordering — RATIFIED (registry has zero behavioral consumer until V5-9 lands; SPO L1/L2 cache works without the bridge fn today). Q3 `MulThresholdProfile` location — RATIFIED in `lance-graph-contract` (zero-dep canonical home; co-located with `MulAssessment`). Q4 OGIT-fork upstream PR rule — RATIFIED (AdaWorldAPI/OGIT extension fork only; never PR back to almatoai/OGIT).

---

## splat-osint-ingestion-v1 — Splat contract + EWA OSINT bridge (authored 2026-05-06)

- **Plan:** `.claude/plans/2026-05-06-splat-osint-ingestion-v1.md`
- **Author + date:** Claude (for Jan), 2026-05-06
- **Status:** Active (PR 1+2 of 6 in flight)
- **Scope:** SPLAT-1 ledger row Aspirational -> Wired (x1). Materialise SplatChannel/CamPlaneSplat/SplatPlaneSet/CamSplatCertificate in lance-graph-contract; demonstrate EWA-sandwich Sigma-push-forward as neo4j-edge-traversal substitute via crates/jc/examples/osint_edge_traversal.rs.
- **Originating question:** q2 PR #35 review
- **Resolves ledger rows:** SPLAT-1 (entropy 4 -> 2, Aspirational -> Wired stage 1).
- **Branch:** claude/splat-osint-ingestion
- **Confidence (2026-05-06):** Working (math certified by Pillar 6 PR #289).

---

## v1 — Grammar + Foundry Follow-up (authored 2026-04-29)

**Author:** main thread (Opus 4.7), session 2026-04-29
**Status:** Active
**Scope:** Wire the stubs and scaffolds shipped in PRs #275-#283 to existing tissue. Six explicit `stub`/`skeleton`/`placeholder`/`unimplemented!` markers in the merged code (verified by grep) name what remains. 13 PRs across two parallel tracks (6 Foundry + 6 Grammar) sharing one keystone (LF-12 Pipeline DAG). All deliverables target `main` directly; no stacking PRs (avoids the merge-order orphaning that bit #281/#283 → #284/#285).
**Path:** `.claude/plans/grammar-foundry-followup-v1.md`
**Deliverables:** PR-S1 (Pipeline DAG keystone), PR-F1..F6 (Foundry: PolicyRewriter UDF wrap, Encrypt+DP, Lance audit, PostgREST dispatch, audit_from_plan, dn_path scent), PR-G1..G6 (Grammar: Triangle causality, Disambiguator wiring, ContextChain fp, verb_table seed, AriGraph unbundle, Animal Farm real run).
**Cross-refs:**
- `lf-integration-mapping-v1.md` — LF-12 keystone rationale (PR-S1)
- `foundry-roadmap.md` — original PR-1..PR-5 (PR-1/PR-2 shipped as #278/#280; PR-3..PR-5 ship as PR-F1..F4 here)
- `integration-plan-grammar-crystal-arigraph.md` — original AriGraph follow-up (now ships as PR-G5)
- `grammar-landscape.md` — case inventories that PR-G4 consumes
**Open decisions:** (1) PR-F2 encryption key management (KMS? in-process? user-supplied?); (2) PR-G6 Animal Farm text licensing; (3) PR-F6 bgz-tensor → callcenter dep; (4) PR-G4 ownership.

---

## v1 — Super-Domain RBAC + Multi-Tenancy (authored 2026-05-13)

**Author:** main thread (Opus 4.7 1M), session 2026-05-13 (branch `claude/lance-datafusion-integration-gv0BF`)
**Status:** Active
**Scope:** 4-level addressing hierarchy (meta-anchors → super domain → OGIT basin → within-basin slot) with explicit byte-sized DTOs, RBAC + multi-tenant Chinese walls wired onto the super-domain boundary. 6 bytes per row (4-byte `TenantId` + 2-byte `OwlIdentity`), inline per-family codebook with label+schema+verbs, single masked DataFusion predicate enforces tenant + super-domain + role + slot in one vector pass. Foundry-parity selling point at the enforcement surface, sub-microsecond hot path. Locks the 2-consumer ticket-system constraint (`hiro-rs` absorbs OSLC-* off-label, `hubspot-rs` is fresh basin) and collapses 4 OSLC-* namespaces into a single Hiro basin with provenance lineage.
**Path:** `.claude/plans/super-domain-rbac-tenancy-v1.md`
**Deliverables:** D-SDR-1..D-SDR-12 (Tier A DTOs / Tier B TTL namespaces / Tier C consumer crates / Tier D compliance + audit / Tier E cross-tenant federation Phase 2)
**Substrate:** Builds on shipped `lance-graph-ontology::namespace::SchemaPtr`, `bridges::OgitBridge` + `BridgeFromRegistry`, `holograph::dntree::WellKnown` (promoted to `SuperDomain` enum), `lance-graph-callcenter::dn_path::DnPath` compression chain, `bgz-tensor::HhtlDEntry` bit-packed-hierarchy pattern, `lance-graph-contract::cam` CAM-PQ codec contract.
**Cross-ref:** `palantir-parity-cascade-v2.md` (this spec adds the enforcement surface), `lance-graph-ontology-v5.md` (this spec sits above v5; v5 unchanged), `GLUE_LAYER_OGIT_TO_OWL_SPEC.md` (source for OWL property characteristics bitfield).
**Open questions:** Foundry ObjectType cross-walk targets, Wikidata QID mappings, audit format choice (JSON Lines / CloudEvents / OTel), DEK rotation cadence, escalation UX, HPO/MONDO multi-member confirmation, slot 0xFF schema-only convention.

**Correction (2026-05-13):** §13 refinements added (same session). (a) Enforcement composes onto shipped `lance-graph-callcenter::policy::PolicyRewriter` chain + `PolicyKind` taxonomy (RowFilter/ColumnMask/RowEncryption/DifferentialPrivacy/Audit) rather than introducing parallel path — ~30% Tier A LOC reduction. (b) Cross-tenant federation upgraded to A+B+C all accepted; Option C (`EncryptedViewAggregate`) viable now via LanceDB transparent encrypted views, not 2027+ R&D. (c) Audit chain integrity built-in via `MerkleRoot::from_fingerprint` + `ClamPath` from `graph/spo/merkle.rs` (the merkle/DN-path mixing already shipped). (d) Hard-lock requirement formalized: Healthcare ↔ OSINT (and 3 other pairs) get 3 layers of defense — predicate + per-super-domain merkle salt + super-domain-scoped HKDF key derivation. (e) `researcher` role hardened to anonymized-projection-only with k-anonymity floor + DP noise injection on aggregates. New deliverables D-SDR-13..17 added. Open questions on audit format + cross-tenant federation RESOLVED; new open questions on hard-lock partner matrix + per-super-domain DP epsilon + merkle salt rotation cadence.

**Correction (2026-05-13, fourth commit):** §19 build invariants + SIMD strategy added. Pins: rust 1.94.1, lance =4.0.0, lancedb 0.27.2 (per PR #275). All vectorized ops across D-SDR-1..39 use `ndarray::simd` from the workspace's vendored ndarray fork — single SIMD path, single test surface, single cross-platform behavior contract. Hot-path ops mapped: OwlIdentity bitmask scans, batch MerkleRoot computation, BitSet256 bitwise ops, per-family codebook PQ centroid distance, canonicalization rule application, DataFusion predicate vector composition, ArrowBatchDriftSignal MerkleRoot-of-batch. Tier A LOC drops ~15-25% (scalar fallback paths collapse to ndarray::simd one-liners). Mandatory-ndarray-as-dep promotion (retire `ndarray-hpc` feature flag) is a separate concurrent workstream, NOT in this spec's scope but assumed baseline; Tier A may temporarily ship behind `#[cfg(feature = "ndarray-hpc")]` until the promotion lands.

**Correction (2026-05-13, third commit):** §18 empirical reality check added after pygithub REST inspection of `AdaWorldAPI/MedCareV2` + `AdaWorldAPI/MedCare-rs@claude/csharp-handoff-docs-L3DF0`. Major findings: (a) The §15-§17 drift bridge concept is already designed and partially scaffolded as `MedCareV2/MedCare_2.0/LanceProbe/` (M1 complete; M2-M6 pending Rust-side endpoints). 8 LanceProbe components (ParityClient/ParityWitness/DriftSink/etc.) map nearly 1:1 to the spec's DTOs. (b) MedCareV2 is overlay-only (copy of MedCare + LanceProbe additions) — cannot be reshaped freely as I assumed; "do NOT refactor" is the explicit constraint. (c) CRITICAL crypto correction: the "3DES" in MedCare's `Crypt.cs:438-451` uses 128-bit truncated key + zero IV + ECB-equivalent + non-standard MD5+RC2 KDF + 62-entry hardcoded password array — cryptographically equivalent to single DES (broken). The migration is NOT 3DES→AES-GCM rewrap; it's Argon2-backfill-on-login per existing `MedCare-rs/docs/AUTH_LEGACY_TRIPLEDES_MIGRATION.md` plan. (d) Only the `u_pwd` column on `praxis_mitarbeiter` uses the 3DES path; rest of the schema is plaintext. D-SDR-27 scope reduces from "decrypt-rewrap pipeline" to "carry ciphertext forward, Argon2-backfill on first login." (e) §15.2 abstract 12-rule determinism table replaced by 6 concrete canonicalization rules from `CSHARP_HANDOFF_PROMPT.md` lines 93-104 (date / decimal / bool / soft-delete / pwd / timestamp). (f) §17.3 Arrow Flight SQL convergence is aspirational end-state; immediate path is HTTP+JSON over JWT (what LanceProbe already targets); Flight SQL is Phase 5+ migration. (g) New deliverables D-SDR-35..39 for medcare-rs side: parity ingest endpoint, dashboard, DTO contracts doc, TripleDES fallback feature flag, telemetry endpoint. M5 is blocked until these land. Resolved 7 prior open questions (audit format, federation, DEK rotation, hard-lock matrix scope, DP epsilon, MedCareV2 reshape, 3DES inventory). 3 new open questions: other columns calling EncryptMessage in MySQL_Connect.cs, DTO contracts for 40+ planned routes, AUTH_LEGACY_TRIPLEDES_MIGRATION.md DRAFT-to-Active blockers.

**Correction (2026-05-13, second commit):** §14-§17 refinements added (same session). (§14) Meta-bridge extracted from shipped medcare_bridge.rs + sharepoint_bridge.rs harvest, not designed clean-room. New bridges hubspot_bridge.rs + hiro_bridge.rs added as templates; woa_bridge.rs retrofit. Tier F (D-SDR-18..20, 23) + Tier G (D-SDR-21..22) deliverables. (§15) Drift detection initially framed as production parallelbetrieb infrastructure with 12 cross-language determinism rules — substantially refined by §16+§17. (§16) Pre-prod posture corrected per user clarification: nothing in production yet, single 3DES cipher (not 3-cipher chain), one-shot import tool not persistent infrastructure. Zone 3 boundary placement collapses determinism rules from 12 to ~3 (decimal + timestamp + FP aggregate). MerkleRoot-cleartext-beside-ciphertext insight: drift bridge compares without ever decrypting in steady-state production, so encryption uses random nonces (no need for AES-GCM-SIV). MedCare MySQL Struktur reality check (104 tables, all VARCHAR/TEXT/DATETIME, app-layer 3DES not at-rest, schema is purely clinical with billing/tickets in separate WoA/Hiro databases). New deliverables D-SDR-27..30. (§17) Convergence on LanceDB+DataFusion SQL as unified persistence; both Rust (in-process) and C# (Arrow Flight SQL gRPC) clients hit the same DataFusion logical plan layer. Custom Protobuf IDL (D-SDR-20) SUPERSEDED by Arrow Flight SQL — Substrait extension types for OwlIdentity/MerkleRoot/SuperDomain. Drift bridge bounded to Phase 2-3 cutover window, then retires to CI gate. New deliverables D-SDR-31..34. Dropped scope: MySQLAdapterBridge (D-SDR-24), persistent production drift infra, multi-trustee key escrow, C-ABI FFI option, custom Protobuf IDL. §18 deferred pending MCP scope expansion to AdaWorldAPI/MedCare + AdaWorldAPI/MedCareV2 for 3DES column inventory + transcoded shape grep.

---

## v1 — LF Integration Mapping (authored 2026-04-25)

**Author:** main thread (Opus 4.7 1M), session 2026-04-25 (branch claude/scenario-world-facade)
**Status:** Active
**Scope:** Comprehensive mapping of all 41 LF + 4 W chunks shipped or queued across the lance-graph workspace. Mirrors the SMB-side foundry-parity-checklist; producer-side companion. Documents Tier 1 (8/8 LF + 4/4 W shipped) + Tier 2 (28 chunks across 8 stages, ~38% shipped, sequencing for next 10 chunks). Includes Stage 7 redesign notes (LF-71 column rejected; LF-73/74/75 added wiring NARS counterfactual / Chronos-method palette forecast / Apache-Temporal-method deterministic replay).
**Path:** `.claude/plans/lf-integration-mapping-v1.md`
**Companions:** `.claude/agents/scenario-world.md`, `docs/ScenarioWorldCounterfactual.md`
**Cross-ref:** `smb-office-rs/docs/foundry-parity-checklist.md` (consumer mirror)

---

## v1 — Q2 Foundry-Equivalent Integration (authored 2026-04-24)

**Author:** main thread (Opus 4.7 1M), session 2026-04-24
**Status:** Proposed
**Scope:** Q2 = user interface (Gotham/Workshop/Vertex equivalent) + SMB = first tenant testbed. 4 phases: demo-able → operational → intelligent → fly. Firefly stack: Ballista + Dragonfly + GEL.
**Path:** `.claude/plans/q2-foundry-integration-v1.md`
**Deliverables:** Q2-1.1..1.7 (Phase 1 MVP), Q2-2.1..2.7 (Phase 2 workflows), Q2-3.1..3.7 (Phase 3 reasoning), Q2-4.1..4.7 (Phase 4 scale). 28 deliverables total.
**Foundation:** 8 PRs merged this session (#253-#260) provide substrate.
**Key differentiators vs Palantir Foundry:** active inference as dispatch, NARS truth as primary data, CausalEdge64 Pearl 2³ masks, JIT-compiled lenses, zero-dep contract crate.

---

## v1 — Supabase Subscriber Wire-up (authored 2026-04-24)

**Author:** sonnet agent, session 2026-04-24 (branch claude/supabase-subscriber-wire-up)
**Scope:** Flip `LanceMembrane::subscribe()` from Phase-A stub to a live `tokio::sync::watch::Receiver<CognitiveEventRow>` wired to `LanceVersionWatcher`; ship `DrainTask` scaffold.
**Path:** `.claude/plans/supabase-subscriber-v1.md`
**Deliverables:** DM-4a swap Subscription type, DM-4b `version_watcher.rs`, DM-4c uncomment `pub mod version_watcher`, DM-6a `drain.rs` scaffold, DM-6b uncomment `pub mod drain`.

**Status (2026-04-24):** In PR. All deliverables in branch `claude/supabase-subscriber-wire-up`.

**Confidence (2026-04-24):** FINDING — 17 tests pass (13 without realtime, 17 with; 4 new tests in `version_watcher.rs`, 1 new `subscribe_receives_on_project` in `lance_membrane.rs`). Zero regressions.

**Correction (2026-05-06):** The `tokio::sync::watch::Receiver` choice violates I-2 (tokio outbound only) per `SINGLE_BINARY_TOPOLOGY.md`. Sync substitute is `ArcSwap<u64>` + `event_listener::Event`, polled on a `std::thread`. WATCHER-1 entropy-ledger row carries the corrected spec.

---

## v1 — Unified Integration: PersonaHub × ONNX × Archetype × MM-CoT × RoleDB (authored 2026-04-23)

**Author:** main-thread session 2026-04-23
**Scope:** Integrate four upstream systems (PersonaHub compression, ONNX persona classifier @ L4/L5, Archetype ECS adapter, MM-CoT stage split) into the lance-graph cognitive substrate without adding new architectural layers — each maps onto existing contract types.
**Path:** `.claude/plans/unified-integration-v1.md`
**Deliverables:** DU-0 PersonaHub 56-bit compression, DU-1 ONNX persona classifier (replaces Chronos proposal), DU-2 Archetype ECS bridge crate, DU-3 RoleDB DataFusion VSA UDFs, DU-4 MM-CoT `rationale_phase: bool` in `CognitiveEventRow`, DU-5 board hygiene.

**Status (2026-04-23):** Active. No deliverables shipped yet. Plan written and committed (commit `468357d`). Architectural ground truth in `callcenter-membrane-v1.md` §§ 15–17.

**Confidence (2026-04-23):** CONJECTURE — all integration mappings grounded in repo evidence and upstream docs; no code shipped beyond plan.

**Correction (2026-04-23):** Chronos (Amazon) proposal superseded by ONNX classifier for DU-1. Chronos predicts 1D style scalar; ONNX classifier predicts full 288-class `(ExternalRole × ThinkingStyle)` product. ONNX infra already justified by Jina v5 ONNX on disk.

---

## v1 — Callcenter Membrane: Supabase-shape over Lance + DataFusion (authored 2026-04-22)

**Author:** main-thread session 2026-04-22
**Scope:** Assimilate the design and ergonomics of the Supabase callcenter surface into a new crate (`lance-graph-callcenter`) that sits entirely outside the canonical cognitive substrate, backed by Lance + DataFusion, enforcing the BBB (blood-brain barrier) at compile time via the Arrow type system — Phoenix channel realtime + PostgREST query surface without PostgreSQL.
**Path:** `.claude/plans/callcenter-membrane-v1.md` (254 lines)
**Deliverables:** DM-0 `ExternalMembrane` + `CommitFilter` in contract, DM-1 callcenter crate skeleton, DM-2 `LanceMembrane::project()` + compile-time leak test, DM-3 `CommitFilter → DataFusion Expr`, DM-4 `LanceVersionWatcher`, DM-5 `PhoenixServer`, DM-6 `DrainTask`, DM-7 `JwtMiddleware + RLS rewriter`, DM-8 `PostgRestHandler`, DM-9 end-to-end test.

**Status (2026-04-22):** Active. DM-0 and DM-1 shipped in this session. DM-2 through DM-9 queued.

**Confidence (2026-04-22):** CONJECTURE on the full architecture (grounded in Arrow BBB analysis + repo evidence; no DM-2+ implementation shipped). DM-0/DM-1 are working stubs; Arrow compile-time BBB enforcement verified structurally, awaiting DM-2 compile-time leak test.

**Correction (2026-05-06):** The framing "callcenter sits *outside* the canonical cognitive substrate" was read by some sessions as "separate process". Per `SINGLE_BINARY_TOPOLOGY.md`, callcenter is in-process Layer 2, sync, zero-copy over Layer 1 BindSpace. DM-5 / DM-8 are the only L3 (post-tokio) components in this plan.

---

## v1 — Categorical-Algebraic Inference (authored 2026-04-21)

**Author:** main-thread session 2026-04-21
**Scope:** Meta-architecture document proving that parsing (Kan extension), disambiguation (free-energy minimization), learning (NARS revision), memory (AriGraph commit), and awareness (method-call history) are one algebraic operation — element-wise XOR on role-indexed slices of a 10K binary VSA vector — viewed through five lenses. Grounded in Shaw 2501.05368 (category theory) + 13 supporting papers. Does not replace elegant-herding-rocket — extends it with the categorical foundation.
**Path:** `.claude/plans/categorical-algebraic-inference-v1.md` (496 lines)
**Deliverables:** This plan produces no NEW D-ids. It grounds the existing D2/D3/D5/D7/D8/D10 deliverables from elegant-herding-rocket in the categorical-algebraic framework and establishes the five-lens litmus + object-does-the-work test as architectural invariants.

**Status (2026-04-21):** Active. Companion to elegant-herding-rocket-v1, not a replacement.

**Confidence (2026-04-21):** CONJECTURE on the Kan-extension-IS-free-energy equivalence. FINDING on all other claims (grounded in shipped code + paper proofs).

---

## v1 — Codec Sweep via Lab Infra, JIT-first (authored 2026-04-20)

**Author:** main-thread session 2026-04-20
**Scope:** Operationalise PR #220's "What's Needed to Fix" list (wider codebook / residual PQ / Hadamard pre-rotation / OPQ) as a parameter sweep through the lab endpoint, with every codec candidate difference expressed as a JIT-compiled kernel rather than a cargo rebuild — one upfront API hardening rebuild, unlimited candidates afterwards.
**Path:** `.claude/plans/codec-sweep-via-lab-infra-v1.md` (396 lines)
**Deliverables:** D0.1 `CodecParams` in `WireCalibrate`, D0.2 `WireTokenAgreement` endpoint (I11 cert gate), D0.3 `WireSweep` streaming endpoint + Lance append, D0.4 surface freeze. D1.1 `CodecKernelCache` via `JitCompiler`, D1.2 rotation primitives (Identity / Hadamard / OPQ) as JIT kernels, D1.3 residual PQ via JIT composition. D2.1 reference-model loader, D2.2 decode-and-compare loop, D2.3 handler wiring. D3.1 server-side sweep handler, D3.2 curl-driven client. D4.1 DataFusion over Lance log, D4.2 Pareto frontier notebook. D5 graduation bridge (fires only on candidate passing all gates).

**Status (2026-04-20):** Active. Plan authored; no deliverables shipped yet. Depends on merge of PR #224 (three-part lab-surface framing + I11 measurability invariant) for the architectural grounding.

**Confidence (2026-04-20):** Pre-execution. Risk hot-spots: (a) JIT compile cost for residual PQ composition — needs measurement; (b) token-agreement harness load time on ref model — may dominate latency for small sweeps; (c) Lance append concurrency under streaming writes. Plan assumes these are tractable; D0 surface freeze is deliberate to prevent iterating on the DTO shape mid-sweep.

---

## v1 — Elegant Herding Rocket (authored 2026-04-19)

**Author:** main-thread session 2026-04-19
**Scope:** DeepNSM as full parser via Grammar Triangle wiring + Markov ±5 SPO+TEKAMOLO bundling + NARS-tested grammar thinking styles + coreference resolution + story-context bridge + ONNX arc emergence.
**Path:** `.claude/plans/elegant-herding-rocket-v1.md` (2,085 lines)
**Deliverables:** D0 landscape doc, D2 FailureTicket emission, D3 Triangle bridge, D4 ContextChain reasoning, D5 Markov ±5 bundler, D6 role keys, D7 grammar thinking styles, D8 story context + contradictions, D9 ONNX arc export, D10 Animal Farm validation harness, D11 bundle-perturb emergence.

**Status (2026-04-19):** Active. Phase 1 (D0 + D4 + D6) shipped in PR #210.

**Confidence (2026-04-19):** Phase 1 working (125 tests passing).
Phases 2–4 queued.

**Phases:**

- **Phase 1 — SHIPPED** (PR #210, merged): D0 landscape doc + D4
  ContextChain reasoning ops + D6 role keys. 125 tests passing.
- **Phase 2 — QUEUED:** D2 FailureTicket emission + D3 Triangle
  bridge + D5 Markov bundler + D7 grammar thinking styles.
  Estimate ~930 LOC, one PR.
- **Phase 3 — QUEUED:** D8 story-context/contradictions + D10
  Animal Farm validation harness.
- **Phase 4 — FUTURE:** D9 ONNX arc export + D11 bundle-perturb
  emergence interface.

---

## How to use this file

1. **Starting a new session:** check top entry. If Status is Active,
   that's the current plan. Read it at
   `.claude/plans/<plan-file>.md`.
2. **Proposing a new plan:** prepend a new v entry; move prior
   plan's Status to Superseded.
3. **Tracking deliverable progress:** use
   `.claude/board/STATUS_BOARD.md` for the cross-deliverable
   view (which D-ids are in which phase / PR).
4. **User requests / open threads** that aren't yet a plan: capture
   in `.claude/knowledge/OPEN_PROMPTS.md`.

## Cross-references

- **`SINGLE_BINARY_TOPOLOGY.md`** — canonical architecture reference
  (three layers, four invariants: single-binary, tokio-outbound-only,
  BBB compile-time-enforced, per-row vs per-cadence gates distinct).
  **READ FIRST** before proposing any new "membrane" / "transcode" /
  "subscriber" / "external surface" plan.
- **`STATUS_BOARD.md`** — deliverable-level status (D0 / D2 / D3 / …
  across all plans).
- **`OPEN_PROMPTS.md`** — outstanding user questions / threads that
  aren't yet scoped into a plan.
- **`PR_ARC_INVENTORY.md`** — shipped-PR decision history.
- **`LATEST_STATE.md`** — current-state snapshot.

## 2026-04-20 — cam-pq-production-wiring-v1
**Status:** DRAFT
**Plan:** `.claude/plans/cam-pq-production-wiring-v1.md`
**Scope:** Wire CAM-PQ as default codec for argmax-regime tensors.
**Deliverables:** D1-D7 (classifier, calibration, storage, decode, validation, E2E, fallback).
**Driver:** ICC 0.9999 at 6 B/row on Qwen3-8B (PR #218 bench).
**Effort:** ~8 person-days.
**Confidence:** HIGH.

---

## v1 — BindSpace Columns E/F/G/H (authored 2026-04-26)

**Author:** main thread (Opus 4.7 1M), session 2026-04-26
**Status:** Active
**Scope:** Extend BindSpace SoA from 4 → 8 column families. Column H (EntityTypeId, Foundry Object Type). Column E (OntologyDelta, per-cycle structural learning). Column F (AwarenessColumn, BF16-mantissa-inline per-word epistemic annotation). Column G (ModelRef, ONNX style_oracle binding). Total overhead +5.9% per row (6212→6578 bytes), still fits L3 cache.
**Path:** `.claude/plans/bindspace-columns-v1.md`
**Companions:** EPIPHANIES.md 2026-04-26 (4 entries), TD-AWARENESS-INLINE-1, TD-PALETTE-SENTINEL
**Scientific review:** 7 SOUND, 7 CAUTION, 0 WRONG (Jirak/Pearl/NARS/Kleyko/Shaw cross-check)
**Deliverables:** D-H1..4 (Phase 1), D-E1..6 (Phase 2), D-F1..9 (Phase 3), D-G1..5 (Phase 4). 24 total.
**Cross-ref:** LF integration mapping v1 (Stages 2/5/6), Q2 Foundry plan (Vertex parity), soa-review.md §semantic kernel

---

## v1 — Foundry Consumer Parity: Shared Ontology for SMB + MedCare (authored 2026-04-26)

**Author:** main thread (Opus 4.7 1M), session 2026-04-26
**Status:** Active
**Scope:** Map the shared Foundry parity surface consumed by both smb-office-rs and medcare-rs. Resolve 5 callcenter UNKNOWNs (consumer-validated). Document the DataFusion/SQL groundtruth pattern. Identify shared build priorities (DM-8 PostgREST is P-0 for both). Ontology unification: one contract shape, two domain-specific instances.
**Path:** `.claude/plans/foundry-consumer-parity-v1.md`
**Cross-ref:** `smb-office-rs/docs/foundry-parity-checklist.md` (45 LF chunks); `medcare-rs` callcenter-as-owner architecture; `q2-foundry-integration-v1.md`; `lf-integration-mapping-v1.md`; `callcenter-membrane-v1.md` (UNKNOWNs resolved)

## 2026-05-07 — Status annotation: `sql-spo-ontology-bridge-v1` partially superseded

**Status:** Active (partially superseded by `lance-graph-ontology` crate, 2026-05-07)
**Note:** The `SchemaExpander` proposed in `sql-spo-ontology-bridge-v1` already shipped in earlier work, and the new `lance-graph-ontology` crate (commit `4cf9a26`, branch `claude/create-graph-ontology-crate-gkuJG`) consumes it as its sole bridge surface. The plan's Phase 4 (NARS cold sink) and `promote_to_spo` writer bridge remain owned by the original plan. Recon + decision for the new crate: `.claude/RECON_ONTOLOGY_CRATE.md` + `.claude/DECISION_SPO_ARIGRAPH.md` (prior commit `edef321`). Federated two-layer cache (Option B): SPO + ARiGraph triplet_graph are not duplicates by design; entropy-ledger rows 70 + 245 cite the L1/L2 cache pair. APPEND-ONLY annotation; original plan entry not edited.

---

## 2026-05-07 — Unified OGIT Architecture plans (sprint-2)

Sprint-2 (12-agent + meta) synthesized 15 architectural patterns (A-O) into a layered plan-doc structure. ~80% of the architecture is already shipped in workspace; the plan-docs name and expose what exists + the ~20% remaining wiring work.

### Master plan-doc

- **`unified-ogit-architecture-v1.md`** (Active) — master synthesis covering all 15 patterns A-O, Tier 0-4 structure. The single document future sessions read first to understand the unified architecture and its current state. Cross-references the 3 sub-plans below and the proof-of-vision.

### Tier 1 — G-overlay wiring (Patterns A+B+C+E)

- **`ogit-g-context-bundle-v1.md`** (Active) — concrete plan for Patterns A (SPO-G u32 slot), B (ContextBundle typed surface), C (GenericBridge dispatching per-G ConsumerPointer). Threads G through existing primitives. Closes TD-OGIT-G-SLOT-1, TD-CONTEXT-BUNDLE-2, TD-GENERIC-BRIDGE-3.

### Tier 2 — Supervised consumer mesh (Patterns E+F)

- **`compile-time-consumer-binding-v1.md`** (Active) — concrete plan for `/modules/<name>/manifest.yaml` build-script glue (Pattern E) + ractor supervisor port from gRPC service trait shape (Pattern F). Closes TD-MANIFEST-MODULES-4, TD-RACTOR-SUPERVISOR-5.

### Proof of vision

- **`anatomy-realtime-v1.md`** (Active) — end-to-end demo: hydrate FMA (75K-class anatomy ontology) via OWL hydrator + ingest medical scan (DICOM) + render in Q2 cockpit with realtime anatomy-graph overlay. Exercises every pillar (Splat, EWA-Sandwich, α-saturation, OGIT-G, Generic Bridge, medcare-rs RBAC, ractor supervisor). Multi-PR; ~5-7 PRs spread over weeks. Closes TD-ANATOMY-DEMO-8.

### Pre-existing plans reframed by sprint-2

These existing plans absorb cleanly into the new architecture and remain in scope:
- `lance-graph-ontology-v5.md` — Pillar 0 work (already merged via PR #355); the OGIT registry is the Pattern B carrier.
- `palantir-parity-cascade-v2.md` — Foundry-equivalent surface; ConsumerPointer + actor shape lands its deliverables.
- `ogit-cascade-supabase-callcenter-v1.md` — already merged via PR #355; GenericBridge replaces the per-callcenter scaffolding.
- `callcenter-membrane-v1.md` — DM-2/DM-3 still in flight; supervisor shape (Pattern F) defines how they compose.

### Plans deferred / aspirational

- Tier 4 (Pattern K: JIT circular compilation via cranelift) — captured as TD-CIRCULAR-COMPILATION-7; aspirational only.

### Cross-references

- `.claude/plans/unified-ogit-architecture-v1.md` (W1 — master synthesis)
- `.claude/knowledge/tier-0-pattern-recognition.md` (W2 — code → pattern map)
- `.claude/patterns.md` (W3 — appended Pattern Recognition Framework section)
- `.claude/board/EPIPHANIES.md` (W4 — 17 architectural epiphanies appended)
- `.claude/board/TECH_DEBT.md` (W5 — 11 TD entries appended)
- `.claude/board/ARCHITECTURE_ENTROPY_LEDGER.md` (W6 — 5 reframes + 15-pattern absorption table)
- `.claude/board/ARCHITECTURE_ENTROPY_LEDGER_RESOLVED.md` (W7 — RECOGNITION-1 row)
- `.claude/board/LATEST_STATE.md` (W9 — sprint-2 deliverables added to Recently Shipped)

### Sprint-2 governance

This sprint was orchestrated as 12 worker agents + 1 meta agent on branch `claude/unified-ogit-architecture-synthesis`. CCA2A pattern: per-agent append-only logs in `.claude/board/sprint-log-2/agents/agent-W*.md`; meta review in `.claude/board/sprint-log-2/meta-1-review.md`; sprint summary in `.claude/board/sprint-log-2/sprint-summary.md`.

## 2026-05-12 — Sprint-3: Tier-1 Implementation Specs (PR #360 + #361 + substrate-recognition sweep)

11 PR-X-1 implementation specs landed via PR #360 + 2 architectural corrections via PR #361 + 3 spec re-scopes via post-#360 substrate-sweep PR. After this sprint sequence, an engineer picks any PR-X-Y spec and starts coding without re-design.

### New specs (`.claude/specs/`)

| Spec | Pattern | Status | Effort (post-substrate-sweep) |
|---|---|---|---|
| `sprint-3-execution-plan.md` | (master) | Active | n/a |
| `sprint-3-pr-graph.md` | (sequencing; compressed timeline) | Active | n/a |
| `pr-a-1-spo-g-u32-slot.md` | A | Active (re-scoped) | ~150 LOC / 1 day |
| `pr-b-1-context-bundle.md` | B | Active | ~200 LOC / 1 day |
| `pr-c-1-generic-bridge.md` | C | Active (re-scoped) | ~80 LOC / ½ day |
| `pr-d-1-fma-owl-hydrator.md` | D (PARTIALLY SHIPPED) | Active (re-scoped) | ~250 LOC / 1-2 days |
| `pr-e-1-manifest-modules.md` | E | Active (post-#361 cycle fix) | ~330 LOC / 2 days |
| `pr-f-1-ractor-supervisor.md` | F | Active (post-#361 inert-bundle skip) | ~400 LOC / 3 days |
| `pr-j-1-int4-32d-atoms.md` | J | Active | ~120 LOC / 1 day |
| `consumer-crate-template.md` | (Pattern C dry-run; re-targeted to woa-rs + medcare-rs precedents) | Active | n/a |
| `ogit-g-smoke-test.md` | (validation; PR-A-1+B-1+C-1+E-1+F-1) | Active | ~200 LOC / 1 day |
| `trivia-prs-bundle.md` | (3 quick wins: TD-CAM-DIST + TD-ADJ-THINK + TD-DEEPNSM-NSM) | Active | ~60 LOC total / <1 day |

### New knowledge docs (`.claude/knowledge/`)

- `pattern-recognition-cross-source.md` — 4-taxonomy matrix (A-O ↔ Pillars 0-4 ↔ `.grok/` ↔ shipped substrate)
- `cca2a-sprint-prompt-template.md` — substrate-grep checklist + wrong-repo guardrail + pattern-letter discipline

### Pre-existing plans absorbed by sprint-3

- `lance-graph-ontology-v5.md` — Pillar 0 (already shipped via PR #355)
- `palantir-parity-cascade-v2.md` — Pillars 0-4 architecture (parallel taxonomy; cross-referenced via new matrix)
- `ogit-cascade-supabase-callcenter-v1.md` — Pillars 0-4 cascade execution (parallel taxonomy)
- `unified-ogit-architecture-v1.md` (W1 master) — sprint-2 north star
- `ogit-g-context-bundle-v1.md` (sprint-2 Tier-1 sub-plan)
- `compile-time-consumer-binding-v1.md` (sprint-2 Tier-2 sub-plan)
- `anatomy-realtime-v1.md` (sprint-2 proof-of-vision)

### Cross-references

- `.claude/board/LATEST_STATE.md` — sprint-3 entry (paired with this index update)
- `.claude/board/TECH_DEBT.md` — TD-OGIT-G-SLOT-1 through TD-DEEPNSM-NSM-COLLAPSE-11 (the 11 entries spec'd by sprint-3)
- `.claude/board/sprint-log-3/` — coordination directory (12 agent logs + meta-1 + sprint-summary)

## 2026-05-21 — unified-bridge-consumer-migration-v1 (UnifiedBridge wiring across woa-rs / smb-office-rs / MedCare-rs)

**Status:** Active (DRAFT — sister to super-domain-rbac-tenancy-v1 §3.9)
**Confidence:** HIGH on architecture (UnifiedBridge already shipped in lance-graph-callcenter; smb_unified_bridge is the working reference); MED on the SmbBridge promotion path (depends on `OGIT/NTO/SMB/` namespace landing upstream); HIGH on the medcare critical-path (D-UB-7 / D-UB-8 are concrete and bounded).
**Plan file:** `.claude/plans/unified-bridge-consumer-migration-v1.md`
**Predecessors:** `super-domain-rbac-tenancy-v1` (Tier A D-SDR-1..5 + §13.1 PolicyRewriter compositor)

### Scope

Three consumers (woa-rs, smb-office-rs, MedCare-rs) converge to one constructor pattern: `<repo>_unified_bridge(registry, actor_role, tenant) -> Result<UnifiedBridge<NamespaceBridge>>`. CAM-codebook + schema + label + verbs from OGIT + OWL/DOLCE materialised as `OgitFamilyTable` and persisted as a LanceDB column under the `lance-cache` feature. D-UB-1..11 across three tiers; D-UB-7 (fix `ontology_dto.rs:85`) is the critical path; D-UB-8 (RLS coverage for Treatment / Visit / VitalSign) is safety-critical.

---

## 2026-05-21 — lance-graph-in-woa-rs-v1 (greenfield consumer integration)

**Status:** Active (DRAFT — six phases, additive)
**Confidence:** HIGH on architecture (every step mirrors MedCare-rs or smb-office-rs reference); MED on Phase-4 ceiling (Cypher/SPARQL surface may or may not be wanted in production).
**Plan file:** `.claude/plans/lance-graph-in-woa-rs-v1.md`
**Predecessors:** `super-domain-rbac-tenancy-v1` (substrate); `unified-bridge-consumer-migration-v1` (Tier B D-UB-4)

### Scope

Lift woa-rs from zero-baseline (today: OGIT TTL vendored, sea-orm + MySQL writer-parity, no lance-graph dep) to "ontology + RBAC + Lance-third-writer" via six additive phases. Phase 0 mechanical (vendor + exclude). Phase 1 lands `woa-bridge` + `woa-ontology` crates. Phases 2-3 wire route handlers + Lance projection. Phases 4-5 opt-in Cypher + CAM-PQ. Respects 2026-05-15 DualSink-Pivot (MySQL stays authoritative; Lance is a third witness).

---

## 2026-05-21 — lance-graph-in-smb-office-rs-v1 (consumer integration completion)

**Status:** Active (DRAFT — five phases, finish what unified_bridge_wiring's doc-comments already promise)
**Confidence:** HIGH (most substrate ships — smb-bridge + smb-ontology + auth + rls features wired; smb_unified_bridge is the working reference for the cross-consumer migration plan).
**Plan file:** `.claude/plans/lance-graph-in-smb-office-rs-v1.md`
**Predecessors:** `super-domain-rbac-tenancy-v1` (Tier A); `unified-bridge-consumer-migration-v1` (Tier A D-UB-2 + Tier B D-UB-5)

### Scope

Phase A ships dedicated `SmbBridge` upstream (~50 LOC + 2 tests). Phase B authors `OGIT/NTO/SMB/` TTL + SMB-shaped role groups (`tax_clerk` / `partner` / `client_user` / `audit_observer`) per D-SDR-2 + swaps `smb_unified_bridge` from `UnifiedBridge<OgitBridge>` to `UnifiedBridge<SmbBridge>` (15-LOC type-parameter change). Phase C consolidates rich `auth::TenantId` ↔ transparent `callcenter::TenantId`. Phases D-E opt-in Cypher / CAM-PQ. Smallest delta of the three consumer plans.

---

## 2026-05-21 — lance-graph-in-medcare-rs-v1 (consumer integration in flight)

**Status:** Active (DRAFT — seven phases, critical path on Phase 1 + 2)
**Confidence:** HIGH on architecture; CRITICAL on Phase 1 (lance-phase2 build is broken at `ontology_dto.rs:85` — blocks everything); SAFETY-CRITICAL on Phase 2 (3 newly-OGIT-surfaced Healthcare entities have no RLS policy → fail-OPEN bypass risk); HIGH on Phase 5 (LanceProbe-side endpoints D-LGMC-7..11 are concrete per super-domain plan §18.7).
**Plan file:** `.claude/plans/lance-graph-in-medcare-rs-v1.md`
**Predecessors:** `super-domain-rbac-tenancy-v1` (Tier H D-SDR-35..39); `unified-bridge-consumer-migration-v1` (Tier C D-UB-6..10); lance-graph#355 (post-PR-355 migration arc)

### Scope

Phase 1 fix the lance-phase2 build (`MedcareOntology::default()` calls broken no-arg form). Phase 2 close RLS fail-OPEN for Treatment / Visit / VitalSign. Phase 3 ship `medcare_unified_bridge` constructor. Phase 4 wire `MulThresholdProfile::MEDICAL` + `ontology_context_id` third axis (§73 SGB V Überweisung). Phase 5 unblock LanceProbe M2..M6 with the 5 medcare-rs endpoints (D-LGMC-7..11). Phases 6-7 opt-in Cypher / CAM-PQ. Two-branch reality: `main` (full lance-phase2) vs `claude/scaffold-medcare-rs-rZD5A` (lean fallback) — most deliverables land on `main` only.

## 2026-05-21 — lance-graph-business-logic-poc-via-woa-rs-v1 (consolidating POC roadmap across the 4 consumer plans)

**Status:** Active (Draft)
**Confidence:** HIGH on the POC framing (per 4 predecessor plans' session-appended §§4.5-4.7 + §§8-13 refinements + 3 attached distillation docs); MED on RFC v02-006 codegen-pipeline readiness (DRAFT — emitter side may need build).
**Plan file:** `.claude/plans/lance-graph-business-logic-poc-via-woa-rs-v1.md`
**Predecessors:** unified-bridge-consumer-migration-v1; lance-graph-in-{woa-rs,smb-office-rs,medcare-rs}-v1; super-domain-rbac-tenancy-v1.

### Scope

Consolidates the 4 consumer-integration plans into a P0/P1/P2-prioritised POC roadmap. **First POC slice = woa-rs PR-5 (XRechnung visible reward)** — the moment 6 months of lance-graph substrate work produces its first customer-deliverable artefact (EN16931-conformant ZUGFeRD/Factur-X invoice via `hydrate_zugferd` + `SchematronHydrator` + `XsdHydrator`). Maps 1:1 to "First Foundry-style projection: fibo:Transaction" Phase 9 from `erp_foundry_hhtl_ontology_distillation.md`. P0 effort ~7-8 days. P1 closes parity dashboard + RLS-via-codegen-bucket + MedCare/SMB cross-consumer harvest. P2 = opt-in Cypher/similarity/MongoDB alt cold path. No new D-ids; this plan re-indexes existing D-UB-/D-WLG-/D-LGMC-/D-LGSMB- IDs by priority.
