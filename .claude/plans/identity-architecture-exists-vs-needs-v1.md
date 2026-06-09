# Identity Architecture — What Exists vs What Needs Building (v1)

> **Status:** INTEGRATION MAP + PLAN. Grounded by first-hand reads + two parallel
> cross-repo sweeps (2026-06-09). Companion to
> `cognitive-write-roundtrip-substrate-v1.md` (the round-trip mechanism).
> **Branch:** `claude/nice-edison-g4rhhl`.

## Thesis

The hot path should carry a lean **128-bit structured immutable identity** (a
UUIDv8 = the HHTL nibble-address *formalized + namespaced*); heavy content stays
in consumer stores keyed by it. The identity does five jobs as register reads of
one object: **resolve** (class-from-address), **route** (delegate switch),
**witness** (immutable audit + merkle), **ground-truth** (shape_hash drift), and
**dispatch-to-store** (EntityKey → consumer). This doc maps what already exists
against what must be built, and phases the integration.

## Four headline findings (grounded)

1. **The 128-bit identity space is empty** — no committed `u128`/`Uuid`(binary)/
   `[u8;16]`-as-id exists (the single `[u8;16]`, `atoms.rs:74 I4x32`, is a
   thinking-style vector, doc-confirmed *not* an identity). A new GUID won't
   byte-collide. *(Agent A sweep, lance-graph + ndarray.)*

2. **But every GUID FIELD already exists as a committed scalar** → the iron
   mandate is **compose existing fields, do NOT re-invent**: `namespace` =
   `NamespaceId(u8)` inside `SchemaPtr.packed:u32 = [ns:8|entity_type:16|kind:8]`;
   `class/address` = `NiblePath` + `ClassId(u16)` + `EdgeRef{family:u8,local:u16}`;
   `shape_hash` = `StructuralSignature`; `local` = `EdgeRef.local`. A parallel
   re-pack duplicates ratified discriminators (`OD-CLASSID-WIDTH`,
   `I-VSA-IDENTITIES`). *(Agent A finding #2.)*

3. **The cross-store transport is ALREADY solved** — `EntityKey<'a>(pub &'a [u8])`
   (repository.rs:12) is an opaque length-agnostic key both consumer repos use;
   `smb-bridge::key_to_filter` already branches on length (12→ObjectId, else→
   `Bson::Binary`) on Mongo *and* Lance. A 16-byte GUID is "just another length"
   the tested plumbing handles. *(Agent B sweep.)*

4. **The cold path has NO stable structured identity today** — it keys nodes by
   bare `node_id:u32` (no edge id; `String` label + `HashMap<String,String>`
   props), the SPO hot path keys by a `u64` *content* `dn_hash` (not stable),
   `CogRecord` carries no id ("id is the external dn_hash"), and durable identity
   is ad-hoc `Uuid→String` (learning crate) + `OgitUri(String)`. **The structured
   identity fills a real gap** — provided it *subsumes* `SchemaPtr` + `EdgeRef`,
   never parallels them. *(Agent A finding #3.)*

## WHAT EXISTS — grounded inventory (6 layers, file:line)

### Layer 0 — address / discriminator scalars (the GUID's fields)

| Type | Width | Role | Status | Evidence |
|---|---|---|---|---|
| `NiblePath{path:u64,depth:u8}` | 72 | HHTL tree address (basin/child/is_ancestor_of, 16ⁿ) | **[G]** | hhtl.rs |
| `SchemaPtr{packed:u32=[ns:8\|entity_type:16\|kind:8], ctx:u32}` | 64 | schema/type pointer | **[G]** | namespace.rs:119 |
| `NamespaceId(u8)` | 8 | OGIT namespace ordinal | **[G]** | namespace.rs:24 |
| `ClassId = u16` | 16 | per-row shape discriminator ("never a content hash") | **[G]** | class_view.rs:53 |
| `EntityTypeId = u16` | 16 | per-row object-type (Palantir) | **[G]** | ontology.rs:81 |
| `FieldMask(u64)` + `inherit` | 64 | presence bitmask, parent-OR-delta | **[G]** | class_view.rs:69,136 |
| `StructuralSignature` (shape_hash) | hash | "deterministic hash over property-id set" | **[G] type / [H] live-wire** | odoo_blueprint::class_signature |
| `EdgeRef{family:u8,local:u16}` | 24 | episodic HHTL family+local address | **[G]** | episodic_edges.rs:34 |

### Layer 1 — edge / handoff carriers (the LE "sound members")

| Type | Width | Role | Status |
|---|---|---|---|
| `EpisodicEdges64(u64)` = 4×EdgeRef, MRU promote/evict, `to_le_bytes` | 64 | AriGraph episodic edges | **[G]** episodic_edges.rs |
| `CausalEdge64(u64)` (NARS 10+10 ×1023) | 64 | baton/causal edge payload | **[G]** ndarray causal_diff.rs:153 |
| Baton `(target:u16, edge:u64)` | 80 | inter-mailbox handoff | **[G]** collapse_gate.rs:235 |
| `MailboxId=u32`, `MailboxRow{mailbox_ref:u32,row_idx:u32}` | 32/64 | mailbox + row address | **[G]** |

### Layer 2 — cold-path stores (TODAY: thin + inconsistent)

| Store | Key | Status |
|---|---|---|
| `MetadataStore`: `NodeRecord{node_id:u32, label:String, properties:HashMap<String,String>}`, `EdgeRecord{source:u32,target:u32,edge_type:String}` | u32 + **STRING label/props (legacy Cypher)** | **[G]** metadata.rs:60,86 |
| `SpoStore`: `HashMap<u64 dn_hash, SpoRecord>` | u64 **content-hash** (not stable id) | **[G]** spo/store.rs:38 |
| ndarray `CogRecord{meta,cam,btree,embed}` | **no id** ("id is external dn_hash") | **[G]** cogrecord.rs:56 |
| `WitnessId(u64)` (arigraph witness) | 64 opaque handle | **[G]** witness_corpus.rs:63 |

### Layer 3 — resolution (class-from-address)

| Surface | Status |
|---|---|
| `RegistryClassView: ClassView` (fields/template/dolce_category_id) | **[G] resolve / [H] field-enum deferred** class_resolver.rs |
| `OntologyRegistry`: `resolve_uri`, `enumerate_first_with_entity_type_id(u16)`, `resolve_iri_in` | **[G]** registry.rs |

### Layer 4 — commit + witness (the membrane)

| Surface | Status |
|---|---|
| `SoaEnvelope` trait + `ColumnDescriptor` (container-LE geometry) | **[G] trait / [H] ZERO impls** soa_envelope.rs |
| `MailboxSoaView`/`MailboxSoaOwner` (read airgap + Rubicon `try_advance_phase`) | **[G]** soa_view.rs |
| `commit_event` sole-writer + `ExternalMembrane::project` + `CommitFilter`/`MembraneGate` | **[G]** lance_membrane.rs:315 |
| `CognitiveEventRow` (scalar audit event — VSA stripped) | **[G]** external_intent.rs:113 |
| `MerkleRoot(u64)` ×3 (audit/SPO/unified) + `AuditSink` (jsonl/lance) | **[G]** audit_sink/, merkle.rs |
| `SlaPolicy`, `TenantScope` | **[G] types** sla.rs |

### Layer 5 — cross-store transport (the consumer boundary)

| Surface | Status |
|---|---|
| `EntityKey<'a>(pub &'a [u8])` — opaque length-agnostic key | **[G]** repository.rs:12 |
| `EntityStore`/`EntityWriter`/`Batch` traits | **[G]** repository.rs |
| `smb-bridge`: implements both for Mongo+Lance, `key_to_filter` length-branch | **[G]** smb-bridge/mongo.rs:79, lance.rs:92 |
| MedCare-rs: MySQL i64 PKs; DMS `sha256`(NOT NULL)+`storage_key`; imports EntityKey | **[G]** dms.rs:14, graph_contract.rs:31 |
| smb-office-rs: Mongo `ObjectId`(12B) + `String` refs; actively impls repository | **[G]** base.rs:92 |

### Layer 6 — round-trip / substrate-hardening

| Surface | Status |
|---|---|
| `TripletProjection` trait + `roundtrip_eq` → `RoundTripFailure` | **[G]** codegen_spine.rs:107 |
| cognitive-write projection (mailbox SoA → SPO+edges) | **[H] does not exist** |

## WHAT NEEDS BUILDING — 7 gaps (each: what it REUSES [G] + what it ADDS [H])

| # | Gap | Reuses (exists [G]) | Adds [H] | Blocked? |
|---|---|---|---|---|
| **N1** | **`NodeGuid`/`EdgeGuid`** 128-bit identity type | `SchemaPtr` ⊕ `NiblePath` ⊕ `StructuralSignature` ⊕ `EdgeRef.local` | the UUIDv8 composition + layout version + the 5 readings | no |
| **N2** | wire `StructuralSignature` into live `RegistryClassView` | `StructuralSignature` type, `ClassView` | the field-enum from `MappingRow` (the deferred D-CLS audit) | no |
| **N3** | `SoaEnvelope` **implementor** for `MailboxSoA<N>` | `SoaEnvelope` trait, `MailboxSoaView` | the zero-copy impl (mailbox bytes == cold bytes) | no |
| **N4** | cognitive-write `TripletProjection` + `roundtrip_eq` | `TripletProjection`, `EpisodicEdges64`/`CausalEdge64` `to_le_bytes` | the project/decompile over the identity graph | no |
| **N5** | `project_graph` emitter through the gate | `commit_event`, `CommitFilter`/`MembraneGate`, `ExternalMembrane` | the node/edge projection (today emits scalar `CognitiveEventRow`) | no |
| **N6** | **`MetadataStore` string→identity migration** | `MetadataStore`, `EntityKey` | `NodeRecord`/`EdgeRecord` keyed by `NodeGuid` not `String` label/props | no (I-LEGACY-API gated) |
| **N7** | GUID-as-`EntityKey` wiring + MedCare `external_ref` | `EntityKey`, `EntityStore`/`EntityWriter`, smb `key_to_filter` | pass 16-byte key + **one** MedCare column (or reuse `sha256`) | no |
| **N8** | surreal_container SurrealQL read glove | `surreal_container` skeleton | the kv-lance read path | **BLOCKED(C)** fork coords |

**Only N8 is blocked.** N1-N7 need no surrealdb coords.

## N1 — the identity type as a COMPOSITION (the iron mandate from Agent A #2)

```rust
// crates/lance-graph-contract/src/identity.rs  (NEW, zero-dep)
// EVERY field is an existing committed type. No re-invention.

/// 128-bit immutable structured node identity (UUIDv8, RFC 9562).
/// Frozen at write; the class is RE-RESOLVED from the address (never stored mutable).
#[repr(C, align(16))]
pub struct NodeGuid([u8; 16]);
//   bits  0..32  : SchemaPtr.packed   [ns:8 | entity_type:16 | kind:8]  ← REUSE namespace.rs:119
//   bits 32..74  : NiblePath prefix   (path bits + small depth; ver nibble carved at 48..52)
//   bits 74..98  : StructuralSignature (shape_hash, truncated)          ← REUSE odoo_blueprint
//   bits 98..122 : local instance     (EdgeRef.local widened)           ← REUSE episodic_edges
//   bits 48..52  : version = 8  ·  bits 64..66 : variant = 10            ← RFC 9562 reserved (6 b)

/// 128-bit edge identity: source address ⊕ the episodic EdgeRef.
#[repr(C, align(16))]
pub struct EdgeGuid([u8; 16]);
//   = [ source SchemaPtr/NiblePath | EdgeRef{family:u8, local:u16} | shape_hash ]  ← REUSE EpisodicEdges64
```

**The five readings (register reads of one key):**
- **resolve** `guid.schema_ptr() → entity_type → ClassView` (class-from-address, O(1) bit-shift + cache)
- **route** `guid.niblepath().is_ancestor_of(...)` → delegate switch (HHTL bit-shift, through `OrchestrationBridge`)
- **witness** frozen `[u8;16]` + `MerkleRoot` chain (immutable, examined-in-place)
- **ground-truth** `guid.shape_hash() != resolve(addr).shape_hash_now` → drift (read-time diff)
- **dispatch-to-store** `EntityKey(guid.as_bytes())` → consumer (Layer-5 transport, already [G])

**Immutability law (ratified this session):** `class_id` never updates — it's the
lineage id, re-resolved from the address for free; the GUID is write-once; drift
*repair* is a **new immutable version** (Lance is versioned), never an in-place
mutation. `I-VSA-IDENTITIES` Test 0: the GUID is a register key (points to
content), never VSA-bundled.

### ✅ DECISION — RESOLVED (eineindeutig / bijective, ratified 2026-06-09; Phase A landed)
**Carry BOTH, bound by an enforced bijection** (`entity_type ↔ NiblePath`), pre-production so it's baked in with zero migration debt:
- **Canonical, exact, in the GUID:** `entity_type:u16` — fixed-width, no truncation. A *truncated* NiblePath prefix CANNOT be bijective (two distinct deep paths collide past the prefix, `hhtl.rs`), so the exact identity MUST be the dense `entity_type`.
- **Derived view:** `NiblePath` — the bijective radix encoding of the SAME class; full depth resolves from the registry; the GUID carries a coarse prefix (`PREFIX_NIBBLES = 4`) as a *derived routing cache* = `niblepath_of(entity_type)` truncated (the prefix `is_ancestor_of` the full path — tested).
- **identity-IS-address holds:** `entity_type` IS the dense encoding of the address; bijective with `NiblePath` (ADR-1374 satisfied).
- **Eineindeutigkeit enforced 3 ways:** (a) the registry mints `(entity_type, NiblePath)` as a unique pair [Phase B]; (b) a build-time bijection round-trip test proves it [Phase B]; (c) the GUID prefix-consistency invariant. (c) + the byte-layout field-isolation matrix landed in Phase A (`identity.rs`, 15 tests); (a)/(b) are Phase B (ontology, needs the registry).
- **Landed (Phase A, D-IDENTITY-1):** `lance_graph_contract::identity::NodeGuid` + `NiblePath::from_packed`. Byte layout, UUIDv8 version/variant gates, field-isolation matrix, `prefix is_ancestor_of full` invariant — all green (599 contract tests, +15; clippy -D clean).

## Phased integration plan (A→H; each phase = one landable PR)

| Phase | Gap | Crate | Deliverable | DoD | Dep |
|---|---|---|---|---|---|
| **A** | N1 | contract | `NodeGuid`/`EdgeGuid` as composition of existing fields + layout version | byte-decompose round-trips to `SchemaPtr`/`NiblePath`/`StructuralSignature`/`local`; UUIDv8 validates; zero-dep; clippy/fmt | — |
| **B** | N2 | ontology (OGAR) | OGAR = one-way OGIT mirror; mint **immutable append-only ClassIds** over a shared **north-star template spine** (DOLCE-rooted shapes reused across domains via `namespace` + `FieldMask` inherit/delta); seed `entity_type ↔ NiblePath` from it; wire `StructuralSignature` → `RegistryClassView` | bijection round-trips at build time; `shape_hash(class_id)` stable; ClassId never renumbers (protobuf-field-number discipline); D-CLS field-enum closed | A |
| **C** | N3 | shader-driver | `impl SoaEnvelope for MailboxSoA<N>` (zero-copy) | `as_le_bytes().as_ptr()==backing`; `verify_layout()` green | — |
| **D** | N4 | lance-graph | cognitive-write `TripletProjection` + `roundtrip_eq` over the identity graph | passes the `account.move` fixture; corrupt-pack fails; (f,c) within 1/1023 | A, C |
| **E** | N5 | callcenter | `project_graph` (node/edge emitter) through `commit_event`+gate | committed cycle queryable as `NodeGuid` nodes + `EdgeGuid` edges; version ticks; RBAC applies | A, D |
| **F** | N6 | lance-graph core | `MetadataStore` string→identity: `NodeRecord`/`EdgeRecord` keyed by `NodeGuid` (label/props → resolved-from-identity) | old string path feature-gated/migrated; field-isolation tests (I-LEGACY-API); query parity | A, B, E |
| **G** | N7 | consumers | GUID-as-`EntityKey`(16B) + MedCare `external_ref` (or `sha256` reuse) | smb: 16-byte key resolves via existing `key_to_filter`; MedCare: GUID→row reverse lookup | A |
| **H** | N8 | surreal_container | SurrealQL read glove | DEFERRED — **BLOCKED(C)** fork coords | E |

**Critical path:** A → (B, C) → D → E → F. G hangs off A (parallel). H is gated.
**Smallest unblocked first brick:** Phase A (the `NodeGuid` composition, zero-dep contract) OR Phase C (the `SoaEnvelope` impl) — both leaf, both needed by D.

## Honest ledger

- **[G] (exists, reuse):** all 6 layers above — `NiblePath`, `SchemaPtr`, `ClassId`,
  `StructuralSignature` (type), `EdgeRef`, `EpisodicEdges64`/`CausalEdge64` LE,
  `commit_event`+gate, `MerkleRoot`+`AuditSink`, `SlaPolicy`/`TenantScope`,
  `EntityKey`+`EntityStore`/`EntityWriter`, `TripletProjection`. **The substrate is
  ~80% present.**
- **[H] (build):** N1-N7 — but each is a *composition/wiring* of [G] parts, not a
  green-field invention. The largest is N6 (cold-path string→identity migration).
- **[BLOCKED(C)]:** N8 only (surrealdb fork coords — human gate; lance-graph P0
  "STOP and ask").
- **[DECISION] RESOLVED (Phase A):** carry BOTH — `entity_type:u16` is the exact
  canonical class; the `NiblePath` prefix is the bijective *derived* routing view
  (full statement in the "DECISION — RESOLVED" block above). Landed in `NodeGuid`.
- **[DECISION-2] RATIFIED (2026-06-09):** the ontology cache's home is **OGAR**,
  a *one-way mirror of OGIT* (+ OWL / Wikidata class-backbone / HHTL) with an
  **append-only immutable ClassId** space — chosen for **ownership + dissolving
  the upstream dependency**, NOT as a drift fix (drift is already contained by
  map-as-source; see the guard below). The ClassId space is organized as a
  **shared north-star template spine**: `entity_type`/`NiblePath` is the DOLCE-
  rooted *shape* (small, reused across every domain); `namespace:u8` selects the
  domain (healthcare / Odoo / WoA-rs / OpenProject-nexgen-rs / OWL / Wikidata).
  Domains REUSE a template by default (switch `namespace`, inherit the field-set);
  specialize only via `NiblePath` descent + `FieldMask` delta; mint a new ClassId
  only for a genuinely novel shape. The surrealdb-coords blocker (N8/Phase H) is
  unrelated and remains.

## Guards (iron rules this plan must not violate)

- **I-VSA-IDENTITIES:** the GUID is a register key that POINTS TO content; never
  VSA-bundle it, never intern open content (only the closed vocabulary). Identities
  intern; scanned papers / free text stay in consumer stores (Layer 5).
- **No content-drift for existing entities — ontology-cache provenance is the ONLY
  drift surface.** An existing entity's identity is a *derived function* of its
  ontology-mapped class (`entity_type` / `NiblePath` / `shape_hash` all fall out of
  the registry mapping), so a mapped entity **cannot** drift from its content. The
  one residual drift surface is an **ontology cache that was not mapped from the
  authoritative source** (hand-filled / stale / regenerated out-of-band) — that,
  not per-instance divergence, is exactly what the `shape_hash` witness reading of
  `NodeGuid` guards. ⇒ the registry MUST treat the authoritative ontology as its
  mapped source (the Phase B bijection is *seeded from* that source, never from a
  hand-filled cache).
- **North-star templates — reuse is the default, mint-new is the exception.**
  `entity_type`/`NiblePath` is a *shared* DOLCE-rooted shape spine: the same
  template ClassId is reused across domains, disambiguated by `namespace:u8`.
  Reuse via `FieldMask` inherit (parent-OR-delta) where a domain shape aligns;
  `NiblePath`-descent + delta where it specializes; a new template ClassId ONLY
  for a genuinely novel shape. DRY-frugal (the shape codebook / `shape_hash` is
  encoded once, reused 256 ways; cross-domain alignment is free — same
  `entity_type` ⇒ same shape) AND composes with immutability — reusable templates
  ARE the immutable spine, not a relaxation of it. Reuse ≠ drift: sharing a
  template across domains is intended, not the cache-provenance drift surface.
  NB: the *mechanism* (octet split + inherit/delta + ancestry + `dolce_category_id`)
  exists today; the *content* (the curated template ontology + domain→template
  mappings) is the OGAR / Phase B build.
- **Compose, don't parallel (Agent A #2):** N1 MUST subsume `SchemaPtr` +
  `EdgeRef`, not re-pack ns/class/family beside them.
- **I-LEGACY-API-FEATURE-GATED:** N6's string→identity layout reclaim needs a
  version gate + field-isolation matrix tests.
- **Sole-writer / no-&mut-during-compute:** N5 reads SoA (`&self`), builds owned
  identity rows, `commit_event` is the gated write-back; drift *repair* is a new
  version, never in-place mutation (the immutability law).
- **AGI-as-SoA:** the GUID is per-NODE at the membrane, NOT a 16-byte-per-row SoA
  column (the hot SoA keeps its lean `u16 class_id`).

## Provenance

First-hand reads (2026-06-09): hhtl.rs · soa_envelope.rs · soa_view.rs ·
class_resolver.rs · class_view.rs · episodic_edges.rs · metadata.rs:60-94 ·
registry.rs · namespace.rs · wikidata_hhtl.rs · lance_membrane.rs:315-429 ·
external_intent.rs:113 · sla.rs · codegen_spine.rs · atoms.rs:74 · audit_sink/.
Cross-repo sweeps: Agent A (lance-graph + ndarray identity-type inventory) ·
Agent B (MedCare-rs + smb-office-rs store keys — `EntityKey`, MySQL i64 / Mongo
ObjectId, DMS `sha256`/`storage_key`). Companion:
`cognitive-write-roundtrip-substrate-v1.md`.
