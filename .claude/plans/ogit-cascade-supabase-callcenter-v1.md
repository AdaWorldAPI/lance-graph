# OGIT-Cascade · Supabase Realtime · Callcenter Membrane — v1

> **Status:** plan, not implementation.
> **Authored:** 2026-05-07 (immediately after lance-graph-ontology v5 PR #352).
> **Owner crates:** `lance-graph-callcenter`, `lance-graph-ontology`, `lance-graph-rdf` (planned), AdaWorldAPI/OGIT (extension fork).
> **Depends on:** lance-graph-ontology-v5 (D-9 thresholds, D-2 SpoBridge), lance-graph-rdf-fma-snomed-v1 (SemanticQuad importer), supabase-subscriber-v1 (DM-4 watcher, DM-6 drain), callcenter-membrane-v1 (parent membrane doctrine).
> **Carry-over:** prior plans are NOT superseded. This plan references their D-ids and defines net-new ones (D-CASCADE-V1-*).

## Pillar 0 — The holy-grail click (answers main-thread question, 2026-05-07)

**`lance-graph-ontology::OntologyRegistry` IS the SoA. Schema is the DTO + index.**

Restated: every per-domain schema (Healthcare, WorkOrder, SMB, CallCenter, Medical-BioPortal) is a thin **name→row map + column projection** over a single canonical struct-of-arrays held by `OntologyRegistry`. Bridges already hold `LazyLock<&OntologyRegistry>` — they do not own columns; they own a scoped view. This is the right pattern; v1 makes it consequential.

**The codec cascade per row** (one entry per ontology concept):

| Column | Type | Source | Cost |
|---|---|---|---|
| `identity_fp` | `Vsa16kF32` | hashed IRI + role-key bind | 64 KB |
| `cam_pq_code` | `[u8; 6]` | quantized projection of `identity_fp` against the 4096-centroid codebook | 6 B |
| `base17` | `[u8; 34]` | bgz17 encoder over `identity_fp` | 34 B |
| `palette_key` | `u32` | PaletteSemiring keyed by `base17` archetype | 4 B |
| `scent` | `u8` | final cascade tier (per `docs/CODEC_COMPRESSION_ATLAS.md`) | 1 B |
| `qualia` | `[f32; 18]` | NARS truth + DK + flow + compass | 72 B |
| `meta` | `MetaWord` | dispatch bits + dcterms:source pointer | 8 B |
| `edge` | `CausalEdge64` | predicate-relations to other rows | 8 B |

Every step in `name → row → fingerprint → CAM-PQ → palette key → Scent` is **O(1)**. A schema (TTL, SQL DDL, JSON Schema, Cypher node label) is *literally* a list of (column-projection + name-resolution) declarations. **The "encode literally everything with indices" outcome is content-addressable memory**: same row addressable through name (HashMap), through similarity (CAM-PQ), through composition (PaletteSemiring), through cosine (Vsa16kF32).

**Consequences for v1**:

- The OGIT TTL files in `AdaWorldAPI/OGIT/NTO/<Namespace>/` are **the seed**. They populate the SoA on hydrate; they are not the runtime store.
- `MedCare-rs/.MYSQL/Struktur.sql` (104 tables) and the BioPortal ontology bundles (25 ontologies, ~2.4 GB) are **DTO declarations**: each becomes a column-projection + provenance pointer over the SoA, not a parallel copy.
- The smb-bridge / medcare-bridge / callcenter-bridge collapse is **mechanical**: they're already `LazyLock<&OntologyRegistry>` views; v1 just gives the registry the columns they need to project.

This pillar is non-negotiable. Every subsequent deliverable serves it.

## Pillar 1 — OGIT as the universal SPO-G lingua franca

OGIT (the AdaWorldAPI fork of `almatoai/OGIT`) is the **single source of truth for ontology namespaces**. v4 shipped `NTO/WorkOrder/`. v1 expands the fork (never PR'd back to upstream — ratified 2026-05-07) to host:

- `NTO/Medical/` — BioPortal-derived medical extensions (the user's "Medical arsenal").
- `NTO/SMB/` — small-business namespace (already export-only per v5 ratification).
- `NTO/CallCenter/` — callcenter wire shapes (FacultyDescriptor, CommitFilter, CognitiveEventRow column families).
- `NTO/Healthcare/` — **delegated** to `lance-graph-rdf-fma-snomed-v1` (FMA + RadLex + SNOMED CT named-graph importer); v1 does not duplicate its work.

The SPO-G shape: `(subject, predicate, object, ontology_context_id)` per `lance-graph-rdf-fma-snomed-v1` §Core types. v1 extends `OntologyRegistry::SchemaPtr` to carry `ontology_context_id: u32` so the same row in the SoA can resolve in multiple named-graph contexts without semantic mud.

## Pillar 2 — Zone 1 / Zone 2 / Zone 3 (BBB membrane refinement)

The user's "outbound serialization only in Zone 3 outside callcenter" is a **tightening** of the existing BBB membrane doctrine (`callcenter-membrane-v1.md` § 10.9). The concrete map:

| Zone | Substrate | What may exist | What may NOT exit |
|---|---|---|---|
| **Zone 1 — Inner BindSpace** | `cognitive-shader-driver` SoA, `BindSpace` columns | Vsa16kF32, palette codes, MetaWord, CausalEdge64 | anything across the BBB; nothing here is `serde::Serialize` |
| **Zone 2 — Membrane** | `lance-graph-callcenter::lance_membrane`, `ExternalMembrane` trait | Arrow `RecordBatch`, scalar-only columns (`bbb_scalar_only_compile_check`) | VSA / palette / NARS truth — already enforced |
| **Zone 3 — Outbound transcode** | `lance-graph-callcenter::transcode`, `phoenix` (planned), `postgrest`, `drain` | Supabase realtime payloads, REST JSON, gRPC, JSON Schema responses | direct reads from `BindSpace` (must traverse Zones 1→2→3) |

**The hard rule**: `serde::Serialize` may only be derived on types that live under `crates/lance-graph-callcenter/src/transcode/` or downstream of it. v1 ships a `cert-officer` static check that fails CI if a Zone 1 / Zone 2 type acquires `Serialize`.

## Pillar 3 — Bridge collapse (smb / medcare → 20-LOC scoped views)

Per the user: *"when callcenter speaks OGIT i understand we can DTO all existing smb-bridge and medcare-bridge and just expand OGIT in our fork to include everything"*. RATIFIED. The mechanism:

- **Today**: `lance-graph-callcenter::ontology_dto` exports `medcare_ontology()` + `smb_ontology()` factory fns (lib.rs:51). Each is a hand-rolled DTO mirror.
- **v1**: both factories become **2-line projections** over `OntologyRegistry::enumerate(namespace)` filtered through `MedcareBridge::filter` / `SmbBridge::filter`. The bridges stay 15–20 LOC; the heavy lifting moves into the registry's column projection.
- **Schema vs OGIT comparison**: `MedCare-rs/.MYSQL/Struktur.sql` (104 tables grouped: combo_* 24, pf_* 30, praxis_* 14, pat_* 4, glob_* 7, file_* 4, misc 21) becomes a per-table **DTO declaration** under `OGIT/NTO/Medical/sql_mirror/` — one TTL per logical table, with `dcterms:source = "MedCare-rs/.MYSQL/Struktur.sql:<table>"` and column-mapping triples. The MySQL schema does not move; OGIT carries its **shape**.

## Pillar 4 — BioPortal arsenal under `OGIT/NTO/Medical/`

The release `bioportal-ontologies-2026-05-05` (private mirror, ~2.4 GB) bundles 25 ontologies. v1 does **not** load all of them — that's the remit of `lance-graph-rdf-fma-snomed-v1`. v1 emits **OGIT TTL stubs** (one per ontology) under `AdaWorldAPI/OGIT/NTO/Medical/<ONTOLOGY>/` declaring:

```turtle
ogit.Medical:ICD10CM
    a ogit:Namespace ;
    rdfs:label "ICD-10 Clinical Modification" ;
    ogit:contextIri <http://purl.bioontology.org/ontology/ICD10CM/> ;
    ogit:contextId 10 ;
    dcterms:source "bioportal-ontologies-2026-05-05/ICD10CM.ttl" ;
    dcterms:license "UMLS-Metathesaurus" ;
    ogit:fileSize "51.6 MB" ;
    ogit:tripleCount "~1.8M (estimate)" ;
    ogit:loaderCrate "lance-graph-rdf" .
```

These stubs make the registry **aware** of every BioPortal ontology without loading it. The actual triple ingestion is gated on the importer in `lance-graph-rdf-fma-snomed-v1`. Priority for stubbing: `ICD10CM` (51.6 MB), `RxNorm` (218.8 MB), `LOINC` (739.2 MB), `FMA` (266.2 MB), `RadLex` (64.9 MB), `SNOMED-stub` (666 KB partial), `MONDO` (215.5 MB), `HPO` (10.7 MB), `DRON` (701.7 MB), `CHEBI` (259.6 MB) — top 10 by clinical leverage.

## The cascade (end-to-end)

```
INBOUND                                  OUTBOUND
                                         (Zone 3 only)
Supabase Postgres CDC                                      ▲
   │ realtime change-feed (websocket)                      │
   ▼                                                       │
Zone 3: drain.rs ingest                Zone 3: transcode emit
   │  parses change row                    │  Cypher → SPARQL CONSTRUCT
   ▼                                       │  → JSON-LD → Supabase RPC payload
oxigraph store (oxttl/oxrdf parser)        │
   │  triples land in named graph         OR
   ▼                                       │  CommitFilter → Expr → DataFusion
Zone 2: ExternalMembrane.ingest            │  → Arrow RecordBatch → Phoenix WS push
   │  Arrow scalar projection              ▲
   ▼                                       │
Zone 2: LanceMembrane.project              │
   │  versioned Lance write                │
   ▼                                       │
Zone 1: BindSpace SoA append               │
   │  CollapseGate bundles                 │
   ▼                                       │
Cognitive cycle (CognitiveShader)          │
   │  resolve(F < 0.2) → commit            │
   ▼                                       │
TripletGraph (AriGraph) write              │
   │  promote to SPO via SpoBridge (v5 D-2)│
   ▼                                       │
LanceVersionWatcher.bump                   │
   │  watch::Sender<CognitiveEventRow>     │
   ▼                                       │
Zone 2 → Zone 3 fan-out  ──────────────────┘
```

The path is symmetric across the membrane. **No row crosses Zone 1 → Zone 3 without a Zone 2 RecordBatch projection.**

## Deliverables (15 total, ranked by leverage / cost)

| Rank | D-id | Scope | LOC | Owner crate |
|---|---|---|---|---|
| 1 | **D-CASCADE-V1-1** | `cert-officer` static check: deny `serde::Serialize` on Zone 1 / Zone 2 types | ~120 | `lance-graph-callcenter` (build script) |
| 2 | **D-CASCADE-V1-2** | Extend `OntologyRegistry::SchemaPtr` to carry `ontology_context_id: u32` (per `lance-graph-rdf` §Core) | ~60 | `lance-graph-ontology` |
| 3 | **D-CASCADE-V1-3** | Collapse `medcare_ontology()` + `smb_ontology()` to 2-line projections over `OntologyRegistry::enumerate(ns)` | ~40 (delete >100) | `lance-graph-callcenter::ontology_dto` |
| 4 | **D-CASCADE-V1-4** | Emit BioPortal namespace stubs under `AdaWorldAPI/OGIT/NTO/Medical/{ICD10CM,RxNorm,LOINC,FMA,RadLex,SNOMED,MONDO,HPO,DRON,CHEBI}/` | ~10 TTL files × ~20 lines | OGIT fork |
| 5 | **D-CASCADE-V1-5** | Transcode `MedCare-rs/.MYSQL/Struktur.sql` → 104 TTL files under `OGIT/NTO/Medical/sql_mirror/` (one per table) | ~104 × ~25 lines (mostly mechanical) | OGIT fork |
| 6 | **D-CASCADE-V1-6** | `OGIT/NTO/CallCenter/` namespace: 6 entities (FacultyDescriptor, CommitFilter, CognitiveEventRow, ExternalIntent, DnPath, ActorContext) | ~6 × ~30 lines | OGIT fork |
| 7 | **D-CASCADE-V1-7** | Add codec-cascade columns to `OntologyRegistry` SoA (`cam_pq_code`, `base17`, `palette_key`, `scent`, `qualia`, `meta`, `edge`) | ~250 | `lance-graph-ontology` |
| 8 | **D-CASCADE-V1-8** | Wire `lance-graph-rdf::SemanticQuad` consumer into `OntologyRegistry::ingest_quads(quads, context_id)` | ~150 | `lance-graph-ontology` (depends on v1 of `lance-graph-rdf`) |
| 9 | **D-CASCADE-V1-9** | Supabase realtime inbound: `drain.rs` change-feed parser → SemanticQuad → `OntologyRegistry::ingest_quads` | ~200 | `lance-graph-callcenter::drain` (extends DM-6 from supabase-subscriber-v1) |
| 10 | **D-CASCADE-V1-10** | Supabase realtime outbound: `transcode/supabase.rs` Cypher → SPARQL CONSTRUCT → JSON-LD → Supabase RPC | ~250 | `lance-graph-callcenter::transcode` |
| 11 | **D-CASCADE-V1-11** | O(1) probe: measure `name → cam_pq_code` lookup p99 latency vs raw oxigraph SPARQL p99; target ≥ 100× speedup | ~80 (bench harness) | `lance-graph-ontology-benches` (new) |
| 12 | **D-CASCADE-V1-12** | `MulThresholdProfile` (v5 D-9) consults `ontology_context_id` so `medical/clinical` thresholds are stricter than `callcenter/conversational` | ~80 | `lance-graph-contract::mul` (extends v5) |
| 13 | **D-CASCADE-V1-13** | End-to-end integration test: Supabase webhook → OGIT triple → cognitive cycle → outbound RPC, asserting Zone 3 is the only emission point | ~300 | `lance-graph-callcenter/tests` |
| 14 | **D-CASCADE-V1-14** | `OGIT/NTO/Medical/sql_mirror/` round-trip: emit MySQL DDL from TTL projection, diff against `Struktur.sql` (must round-trip identity for column names + types) | ~150 | `lance-graph-callcenter::transcode` |
| 15 | **D-CASCADE-V1-15** | BioPortal ICD-10 actual import (smallest of the BIG ontologies at 51.6 MB), populate `OntologyRegistry` with codec-cascade columns | ~200 | `lance-graph-rdf::importers::icd10cm` (new) |

## Acceptance criteria

- [ ] `cargo test -p lance-graph-callcenter --features full` passes; `bbb_scalar_only_compile_check` still compiles.
- [ ] D-CASCADE-V1-1 fails the build if a Zone 1 type acquires `Serialize` (test with a deliberate poison-pill type in `tests/`).
- [ ] D-CASCADE-V1-3: `git diff` on `ontology_dto.rs` is net negative LOC (collapse, not addition).
- [ ] D-CASCADE-V1-5: at least 90/104 tables round-trip through D-CASCADE-V1-14 (10/104 may legitimately differ due to MySQL-specific types like `MEDIUMTEXT`).
- [ ] D-CASCADE-V1-11: O(1) probe shows ≥ 100× p99 speedup for `name → cam_pq_code` over raw SPARQL `SELECT ?o WHERE { :name ogit:hasCamPqCode ?o }`. Lower bound on the holy grail.
- [ ] D-CASCADE-V1-13: webhook → cycle → RPC round-trip passes; the test asserts (via `cert-officer`) that no Zone 1 / Zone 2 type is reachable through `Serialize`.
- [ ] All BioPortal stubs (D-CASCADE-V1-4) carry `dcterms:source` (per v5 D-1) and `dcterms:license`.
- [ ] No upstream PRs to `almatoai/OGIT` (per v5 ratification Q4).

## Out of v1 scope (explicit deferrals, not punts)

- **Full SNOMED CT** import: license-gated per `lance-graph-rdf-fma-snomed-v1` §SNOMED, also the BioPortal release ships only a 666 KB partial. v1 stubs the namespace; full import waits on affiliate attestation.
- **DRON / CHEBI** import: 700 MB + 260 MB respectively, large; benefit is unclear before D-CASCADE-V1-11 measures the cascade payoff. Stub now, import in v2 if probe motivates it.
- **bgz-tensor attention layer integration** with the codec-cascade columns: orthogonal to this plan; the AttentionSemiring composes over PaletteSemiring already.
- **n8n-rs / crewai-rust consumption** of the new SoA columns: those repos consume `lance-graph-contract`; v1 does not change the contract surface beyond D-CASCADE-V1-12.

## Open questions

1. **Codec-cascade column population trigger**: synchronous on `ingest_quads()` or lazy (compute on first read)? Lazy adds complexity; sync makes hydrate slower. **Recommend sync** — hydrate is once-per-process, the cascade computation is bounded.
2. **`ontology_context_id` allocation policy**: dense (FMA=1, RadLex=2, SNOMED=3, ...) per `lance-graph-rdf-fma-snomed-v1` §Core, or sparse hash-derived? Dense is simpler and traceable; sparse avoids cross-repo coordination. **Recommend dense + a `NamespaceRegistry` allocation table sidecar**.
3. **Supabase realtime auth**: JWT verification at Zone 3 entry (per `auth.rs`) or row-level via `rls.rs`? **Recommend both** — JWT is the gate, RLS is the post-gate filter, neither replaces the other.
4. **Schema-as-DTO inheritance pattern**: derive macro (`#[derive(OntologyDto)]`), declarative builder, or hand-rolled per bridge? **Recommend declarative builder** for the first three bridges, escalate to a derive macro only if the pattern repeats six+ times.
5. **OGIT verb expansion for medical**: BioPortal ontologies use OWL object properties (e.g. `regional_part_of` from FMA). Do we map these to `ogit:Verb` shape (one TTL per verb) or carry them through as named-graph triples without verb registration? **Recommend named-graph triples** — we don't ontologize the ontology's predicates, that's infinite recursion.

## Self-bootstrapping prompt for next session

```
Read .claude/plans/ogit-cascade-supabase-callcenter-v1.md cover-to-cover before
proposing any change. The Pillar 0 click — OntologyRegistry IS the SoA, schema
IS the DTO + index — is the architectural anchor; if you find yourself proposing
a parallel store, a copy of columns, or a non-projection bridge, stop and re-read
Pillar 0.

Top-3 deliverables to start: D-CASCADE-V1-1 (cert-officer static check),
D-CASCADE-V1-2 (SchemaPtr.ontology_context_id), D-CASCADE-V1-3 (collapse
medcare_ontology + smb_ontology to 2-line projections). All three are bounded,
testable, and serve Pillar 0 directly.

Do NOT start D-CASCADE-V1-{4,5,15} (the BioPortal / SQL transcode work) until
D-CASCADE-V1-{2,7,8} are merged — those are the registry surfaces those
deliverables write into.

Cross-plan deps: lance-graph-ontology v5 D-9 (MulThresholdProfile lands in
lance-graph-contract::mul), lance-graph-rdf-fma-snomed-v1 (SemanticQuad type
+ NamedGraphRegistry), supabase-subscriber-v1 (DM-4 watcher, DM-6 drain).
Confirm those are merged before merging this plan's PRs that depend on them.

Branch: claude/create-graph-ontology-crate-gkuJG (per workspace policy).
PR target: AdaWorldAPI/lance-graph base=main.
```

## Cross-references

- `.claude/plans/lance-graph-ontology-v5.md` — D-9 (MulThresholdProfile), D-2 (SpoBridge::promote_to_spo).
- `.claude/plans/lance-graph-rdf-fma-snomed-v1.md` — SemanticQuad row type, NamedGraphRegistry, OntologyContextId.
- `.claude/plans/supabase-subscriber-v1.md` — DM-4 LanceVersionWatcher, DM-6 DrainTask scaffold.
- `.claude/plans/callcenter-membrane-v1.md` — § 10.9 Membrane Role Place Translation iron rule (parent doctrine).
- `docs/CODEC_COMPRESSION_ATLAS.md` — full cascade chain (Vsa16kF32 → ZeckBF17 → Base17 → CAM-PQ → Scent).
- `docs/ORCHESTRATION_IS_GRAPH.md` — orchestration-as-graph capstone (Zone 3 routing maps onto graph traversal).
- `MedCare-rs/.MYSQL/Struktur.sql` — 104-table source.
- `MedCare-rs/releases/tag/bioportal-ontologies-2026-05-05` — 25 ontology bundles, ~2.4 GB.
- `AdaWorldAPI/OGIT` (extension fork) — never PR'd to upstream.

## Confidence (2026-05-07)

Pre-execution. Pillar 0 is the only architectural commitment that admits no rollback — if it's wrong, the entire plan is wrong. It is right (per PR #223 AGI-as-SoA invariant + the existing `LazyLock<&OntologyRegistry>` pattern in the bridges). Pillars 1-4 are mechanical consequences. The 15 deliverables are bounded; D-CASCADE-V1-1 / 2 / 3 land first because they have no upstream blockers.
