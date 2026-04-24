# Agent Log — Append-Only Activity Record

> **APPEND-ONLY.** Every agent run gets one entry. Newest first.
> Never edit past entries. This is the durable record of what each
> agent did — future sessions read this instead of replaying
> conversations.
>
> **Format:** `## YYYY-MM-DDTHH:MM — <description> (<model>, <branch>)`
> followed by D-ids, commit, test counts, verdict/outcome, and any
> findings worth preserving.
>
> **Chunking purpose:** An agent's entry here REPLACES its full
> transcript in the knowledge graph. If you need to know what an
> agent did, read this file — don't search for task transcripts.
>
> **Who writes:** The main thread appends after each agent completes.
> Agents themselves should also append if they run long enough to
> risk context compaction (write progress incrementally, not just
> at the end).

---

## Entries (reverse chronological)


## 2026-04-24T15:20 — RBAC crate scaffold (sonnet, claude/smb-contract-traits)

**D-ids:** lance-graph-rbac (permission, role, policy, access)
**Commit:** `0df8780`
**Tests:** 14 pass (14 new: 1 access + 3 permission + 4 role + 6 policy)
**Outcome:** New workspace crate `lance-graph-rbac`. PermissionSpec ties RBAC to ontology via PrefetchDepth gates + action whitelists. Example roles: accountant (Detail on Customer, Full+write on Invoice), auditor (Full read-only everywhere), admin (Full+write+act everywhere). `smb_policy()` composes all three. `Policy.evaluate()` returns `AccessDecision { Allow, Deny, Escalate }`.


## 2026-04-24T15:05 — Foundry ontology layer (main thread, claude/smb-contract-traits)

**D-ids:** LinkSpec, PrefetchDepth, ActionSpec (property.rs) + ModelBinding, ModelHealth, SimulationSpec, Ontology builder (ontology.rs)
**Commit:** `574a93d`
**Tests:** 209 pass (19 new: 10 property + 9 ontology)
**Outcome:** Fills all 5 Palantir Foundry gaps. LinkSpec = typed edges (Cardinality). PrefetchDepth = L0-L3 progressive property loading (Identity → Detail → Similar → Full). ActionSpec = Manual/Auto/Suggested triggers. ModelBinding = external model I/O → ontology property. ModelHealth = NARS-based prediction quality tracking. SimulationSpec = World::fork() what-if parameters. Ontology builder composes schemas + links + actions.


## 2026-04-24T14:55 — Schema builder + board hygiene (main thread, claude/smb-contract-traits)

**D-ids:** Schema, SchemaBuilder
**Commit:** `cb8fb37`
**Tests:** 190 pass (6 new Schema builder tests)
**Outcome:** Declarative API: `Schema::builder("Customer").required("tax_id").searchable("industry").free("note").build()`. `.validate()` returns missing Required predicates. `.searchable()` = Optional + CamPq shorthand. Board-hygiene: LATEST_STATE + EPIPHANIES updated for full SMB surface.


## 2026-04-24T14:45 — PropertySpec + CAM-PQ routing (sonnet, claude/smb-contract-traits)

**D-ids:** PropertyKind, PropertySpec, PropertySchema, CUSTOMER_SCHEMA, INVOICE_SCHEMA
**Commit:** `b1ff05e`
**Tests:** 184 pass (10 new property tests)
**Outcome:** bardioc Required/Optional/Free maps to I1 Codec Regime Split: Required = Passthrough (Index), Optional = configurable, Free = CamPq (Argmax). PropertySpec carries predicate + kind + codec_route + nars_floor. CUSTOMER_SCHEMA (10 props) + INVOICE_SCHEMA (10 props).


## 2026-04-24T14:30 — SMB contract traits (sonnet, claude/smb-contract-traits)

**D-ids:** repository.rs, mail.rs, ocr.rs, tax.rs, reasoning.rs
**Commit:** `3ab8a52`
**Tests:** 174 pass (0 new — trait-shape only, no executable logic)
**Outcome:** 5 new zero-dep trait files per smb-office-rs proposal. EntityStore + EntityWriter + Batch (repository). MailParser + ThreadLinker (mail). OcrProvider + PageImage + Bbox + LayoutBlock (ocr). TaxEngine + TaxPeriod + Jurisdiction + RuleBundle (tax). Reasoner + ReasoningKind + Budget (reasoning). Additive-only: 5 `pub mod` appends to lib.rs.


## 2026-04-24T14:15 — FingerprintColumns.cycle f32 migration (sonnet, claude/teleport-session-setup-wMZfb)

**D-ids:** PR B (SoAReview expansion item #1, bindspace substrate)
**Commit:** `121acc1`
**Tests:** 42 pass in cognitive-shader-driver (40 unit + 2 e2e), 174 contract — 0 regressions
**Outcome:** `FingerprintColumns.cycle` migrated from `Box<[u64]>` (256 × u64, Binary16K) to `Box<[f32]>` (16,384 × f32, Vsa16kF32 carrier). New constant `FLOATS_PER_VSA = 16_384`. `set_cycle(&[f32])` for direct VSA write, `set_cycle_from_bits(&[u64; 256])` adapter with `binary16k_to_vsa16k_bipolar` projection. `write_cycle_fingerprint()` API unchanged (takes u64, converts internally). `byte_footprint()` for 1 row = 71,774 bytes. Module doc updated.


## 2026-04-24T13:45 — Vsa16kF32 switchboard carrier (main thread, claude/vsa16k-f32-carrier-type → PR #253 merged)

**D-ids:** PR #253, expansion-list item #1 from SoAReview sweep
**Commit:** `dc56586` (merged to main as `ddb3017`)
**Tests:** 174 contract, 11 callcenter — 0 regressions. 7 new fingerprint tests.
**Outcome:** `CrystalFingerprint::Vsa16kF32(Box<[f32; 16_384]>)` shipped as first-class variant. 6 algebra primitives: vsa16k_zero, binary16k_to_vsa16k_bipolar, vsa16k_to_binary16k_threshold, vsa16k_bind, vsa16k_bundle, vsa16k_cosine. Inside-BBB only. to_vsa10k_f32() downcast wired.


## 2026-04-24T13:00 — SoAReview multi-angle sweep (opus, two parallel agents)

**D-ids:** Supabase-shape subscriber (verdict: GHOST), Archetype transcode (verdict: LOCKED-MAPPING-INCOMPLETE)
**Commits:** none (review-only agents)
**Tests:** n/a
**Outcome — Supabase:** `subscribe()` = disconnected mpsc stub (lance_membrane.rs:186-189). DM-4 LanceVersionWatcher + DM-6 DrainTask modules commented out (lib.rs:71-79). CognitiveEventRow BBB-clean (11 LIVE, 2 ghost fields). 7-item expansion path identified.
**Outcome — Archetype:** `lance-graph-archetype/` crate does not exist. Contract-layer mappings (PersonaCard/Blackboard/CollapseGate) LIVE. 0 archetype-specific types exist. ADR-0001 Decision 1 deblocks scaffold (Rust interface defined BY new crate, not mirrored from Python). 8-item scaffold path identified.


## 2026-04-24T12:30 — Supabase subscriber wire-up (opus, claude/supabase-subscriber-wire-up) [STILL RUNNING]

**D-ids:** DM-4a/b/c, DM-5a, DM-6a/b, DM-7
**Plan:** `.claude/plans/supabase-subscriber-v1.md`
**Status:** In flight. tokio::sync::watch swap, version_watcher.rs, drain.rs scaffold, test flip.
**Target verdict:** GHOST → PARTIAL


## 2026-04-24T12:30 — Archetype crate scaffold (opus, claude/archetype-crate-scaffold) [STILL RUNNING]

**D-ids:** DU-2.1 through DU-2.6
**Plan:** `.claude/plans/archetype-scaffold-v1.md`
**Status:** In flight. New crate + Component/Processor traits + World/CommandBroker stubs.
**Target verdict:** LOCKED-MAPPING-INCOMPLETE → LOCKED-AND-SCAFFOLDED
