# KNOWLEDGE: Ontology Registry — `lance-graph-ontology` Crate Map

## READ BY: `workspace-primer`, `host-glove-designer`, `bus-compiler`,
##          `tenant-bridge-author` (future). MANDATORY before any work
##          touching tenant bridges, OGIT TTL ingest, MappingProposal
##          producers, or schema-source wiring.

## Status: FINDING / SHIPPED. Crate scaffolded and tested in PR commits
## `4cf9a26` (initial structure + bridge trait) and `edef321` (TTL hydrator,
## scope-lock tests, three default bridges). 28 tests pass across
## `tests/bridge_scope_lock.rs`, `tests/hydrate_real_ogit.rs`, and
## `tests/round_trip_ttl.rs`. Phases 1–5 of the unified-ontology plan are
## done; Phases 6–7 (WorkOrder TTL emission, woa-rs/MedCare-rs/q2 binary
## wiring, BindSpace consumer) are queued for follow-up sessions.

---

## Thesis

`lance-graph-ontology` is the OGIT-canonical ontology spine for every
lance-graph tenant. It hydrates TTL files (today, the AdaWorldAPI/OGIT
fork; tomorrow, MySQL/MSSQL schema scanners and a customer admin form)
into `MappingProposal` DTOs, accumulates them into a single
`OntologyRegistry` keyed by `(bridge_id, public_name)` and by raw OGIT
URI, optionally persists rows append-only to a Lance dictionary table,
and exposes the registry to consumers as thin scoped views called
*namespace bridges*. The crate invents no new storage layer — it is a
parser plus a cache plus a scoping facade over the existing
`lance-graph-contract::ontology` (`Ontology` / `Schema` / `LinkSpec` /
`SchemaExpander`) surface. Tenant bridges are ~15-20 lines each, lock
every operation to one OGIT namespace at construction time, and route
resolution through the shared registry. The registry never decides which
SPO substrate a triple ends up in; that remains the consumer's choice
through the existing `SchemaExpander` paths.

## Producer → Consumer map

The crate is the narrow waist between four producers (one shipped, three
future) and three current consumers (plus tenant binaries downstream).
Everything funnels through `MappingProposal` on the way in and through
the `NamespaceBridge` trait on the way out.

| Layer | Component | File / Surface | Status |
|---|---|---|---|
| Producer | OGIT TTL hydrator | `src/ttl_parse.rs`, `src/schema_source.rs` walking `AdaWorldAPI/OGIT/NTO/<Namespace>/` | SHIPPED |
| Producer | MySQL `SchemaSource` | `src/schema_source.rs` (trait); impl pending | FUTURE |
| Producer | MSSQL `SchemaSource` | `src/schema_source.rs` (trait); impl pending | FUTURE |
| Producer | Customer admin form | future UX layer emitting `MappingProposal` | FUTURE |
| Waist | `MappingProposal` + `MappingRow` | `src/proposal.rs` | SHIPPED |
| Spine | `OntologyRegistry` | `src/registry.rs` (in-memory dictionary) | SHIPPED |
| Spine | Lance dictionary cache | `src/lance_cache.rs` (feature-gated `lance-cache`) | SHIPPED |
| Spine | `NamespaceBridge` trait + `BridgeFromRegistry` | `src/bridge.rs` | SHIPPED |
| Default bridge | `OgitBridge` (raw-URI pass-through, per-namespace) | `src/bridges/ogit_bridge.rs` | SHIPPED |
| Default bridge | `WoaBridge` (`WorkOrder` namespace) | `src/bridges/woa_bridge.rs` | SHIPPED |
| Default bridge | `MedcareBridge` (`Healthcare` namespace) | `src/bridges/medcare_bridge.rs` | SHIPPED |
| Consumer | `lance-graph-callcenter::ontology_dto` | callcenter DTO surface (existing) | INTEGRATES NEXT |
| Consumer | `lance-graph::graph::spo::ontology_bridge` | existing `SchemaExpander` writer into SPO | UNCHANGED |
| Consumer | `cognitive-shader-driver::BindSpace` (Phase 7) | future MetaWord / MetaColumn emission | QUEUED |
| Consumer | Tenant binaries (`woa-rs`, `MedCare-rs`, `q2`) | construct one bridge per tenant | QUEUED |

The waist is intentional: every producer becomes a `SchemaSource` impl
and emits `MappingProposal { kind, schemas, links, rows }`; every
consumer holds an `Arc<OntologyRegistry>` and constructs its bridge once
at startup. Adding a new producer or a new tenant never requires
touching the spine.

## The five-step "I want to add a tenant bridge" recipe

A new tenant bridge (say a `qualicare` bridge over `Healthcare` or
`q2` over `WorkOrder`) is mechanical. The default methods on
`NamespaceBridge` carry resolution and scope-lock; the new struct
supplies four constants and a constructor.

1. **Define the struct** holding `Arc<OntologyRegistry>` and a cached
   `NamespaceId` (the `g_lock`). Mirror `bridges/medcare_bridge.rs` —
   it is the smallest, ~45 LOC end-to-end including imports.
2. **Implement `NamespaceBridge`** — supply `bridge_id() -> &'static
   str`, `registry() -> &OntologyRegistry`, `g_lock() -> NamespaceId`.
   The trait's default `entity()`, `edge()`, `entity_by_uri()`, `row()`
   handle resolution and the cross-namespace leak check.
3. **Implement `BridgeFromRegistry`** so callers can use the generic
   `bridge::make_bridge::<MyBridge>(registry)?` constructor. One line:
   delegate to `Self::new(registry)`.
4. **Write a scope-lock test** in `tests/bridge_scope_lock.rs` (extend,
   do not duplicate the file) verifying that `entity("ForeignName")`
   returns `BridgeError::NotInScope` or `BridgeError::CrossNamespaceLeak`.
   That single test is what ratifies the bridge as scoped.
5. **Re-export from `bridges/mod.rs`** and ship. The tenant binary
   constructs one instance at startup and uses `bridge.entity("...")`
   throughout — no other plumbing required.

LOC budget for steps 1–3: ~15-20 lines. The recipe is the same whether
the namespace already exists in TTL (`WorkOrder`, `Healthcare`,
`Network`, etc.) or is reserved for future hydration; an unknown
namespace fails at construction with `Error::UnknownNamespace`, which
is the right time to fail.

## The three-step "I want to extend the ontology" recipe

Three producer pathways exist; each ships a `MappingProposal` and the
spine integrates the rest.

1. **TTL fork PR.** Add or edit a TTL file under
   `AdaWorldAPI/OGIT/NTO/<Namespace>/entities/<EntityName>.ttl`
   following the convention in `NTO/Network/entities/IPAddress.ttl`
   (prefixes, `rdfs:Class`, `rdfs:subClassOf ogit:Entity`,
   mandatory/optional/indexed attribute lists, `ogit:allowed` block).
   The hydrator picks it up next time the registry is rebuilt; no Rust
   code changes.
2. **Schema scanner.** Implement `SchemaSource` against a database
   driver (MySQL, MSSQL, Postgres). Map the source schema's tables /
   columns / FKs to `MappingProposal::Schemas` rows. Future work; the
   trait shape exists today and the OGIT hydrator is the reference impl.
3. **Customer admin form.** A future UX layer where a customer's
   administrator paints entities and edges; the form emits
   `MappingProposal` directly. Same append path as the other two; no
   spine change required.

The three producers are not in tension. They can run side-by-side
against the same registry; collisions on `(bridge_id, public_name)` or
on raw OGIT URI surface as `HydrationReport` warnings.

## SPO-1 disposition

This crate adopts **Option B (federated)** from
`.claude/DECISION_SPO_ARIGRAPH.md`. The two SPO stores
(`lance-graph::graph::spo::*`, fingerprint-keyed cold; and
`lance-graph::graph::arigraph::triplet_graph`, string-keyed warm) are
not duplicates by design — they are an L1/L2 cache pair serving
fundamentally different operations. The ontology registry has zero
opinion about which one a downstream consumer chooses. Hydration
produces `Ontology` values; the existing `SchemaExpander` path produces
`ExpandedTriple`s; whichever store the consumer holds is where the
triples land. Closing SPO-1 (the `arigraph::SpoBridge::promote_to_spo`
writer) is unblocked by this decision but remains owned by SPO-1's own
entropy-ledger row; it is explicitly out of scope here.

## What's NOT here (non-goals)

The crate is deliberately small. None of the following lives in
`lance-graph-ontology` and proposals to add any of them should be
redirected to their owners.

- No new SPO store, no new SPOG quad store, no janus-driver — the
  existing `lance-graph::graph::spo::*` and `arigraph::triplet_graph`
  remain canonical.
- No new SPARQL / Gremlin / GQL / Cypher parsers — those live in
  `lance-graph-planner::strategy::*`. Phase 7's woa-rs Cypher
  integration test routes through `lance_graph::parser::
  parse_cypher_query` directly (PARSER-1 stub gap is documented).
- No MUL changes, no new `CausalEdge64` variants, no new `BindSpace`
  columns. The shader-driver MUL veto at `driver.rs:271-320` is
  unchanged.
- No `smb-bridge` or `callcenter-bridge` ship in this session. SMB
  stays on its native `smb-ontology` declarative Rust fallback;
  callcenter has auth + per-customer scoping concerns that warrant
  their own design pass.
- No `promote_to_spo` writer — that closes SPO-1, not this crate's row.

## Pointers

- `.claude/RECON_ONTOLOGY_CRATE.md` — Phase 1 reconnaissance:
  verified-shipped table, OGIT NTO/ inventory, woa-rs / WoA state,
  PARSER-1 caveat.
- `.claude/DECISION_SPO_ARIGRAPH.md` — binding ruling for SPO-1 within
  this crate's scope (Option B, federated).
- `crates/lance-graph-ontology/src/lib.rs` — module surface,
  re-exports, "What this crate is NOT" doc-comment.
- `crates/lance-graph-ontology/src/bridge.rs` — `NamespaceBridge` +
  `BridgeFromRegistry` + `BridgeError` definitions.
- `crates/lance-graph-ontology/src/registry.rs` — the
  `OntologyRegistry` itself.
- `crates/lance-graph-ontology/src/bridges/{medcare,ogit,woa}_bridge.rs`
  — the three default tenant bridges; copy `medcare_bridge.rs` as a
  template.
- `crates/lance-graph-ontology/tests/bridge_scope_lock.rs` — the
  scope-lock test pattern every new bridge must extend.
- `crates/lance-graph-ontology/tests/hydrate_real_ogit.rs` — end-to-end
  TTL → registry hydration smoke test against the real OGIT fork.
- `crates/lance-graph-ontology/tests/round_trip_ttl.rs` — proposal →
  registry → row → resolved-URI round trip.
- `crates/lance-graph-contract/src/ontology.rs` and `property.rs` —
  upstream `Ontology` / `Schema` / `PropertySpec` / `Marking` /
  `SemanticType` / `SchemaExpander` definitions the crate is a facade
  over.
- `crates/lance-graph/src/graph/spo/ontology_bridge.rs` — existing
  `SchemaExpander` integration that consumes the crate's output.
