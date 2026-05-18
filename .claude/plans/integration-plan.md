# Integration Plan: lance-graph ↔ surrealdb ↔ sea-orm ↔ ndarray

**This repo**: `AdaWorldAPI/lance-graph` — graph engine + ontology + cognitive shaders + Arrow producer.

**Status**: planning document. Companion plans live at the same path in the other repos:
- `AdaWorldAPI/surrealdb:.claude/plans/integration-plan.md`
- `AdaWorldAPI/sea-orm:.claude/plans/integration-plan.md`
- `AdaWorldAPI/ndarray:.claude/plans/integration-plan.md`

---

## 1. Why this plan exists

Four repos already point at one architecture, but the connective tissue is missing. The target is:

> *Foundry-style ontology + BEAM-style supervision + ClickHouse-style analytic + Postgres-style ACID + cognitive primitives — all on one Arrow substrate, surfaced to consumers as a typed sea-orm API.*

No existing stack has that combination:
- Foundry has the ontology + actions but no cognitive layer co-located
- BEAM/OTP has the supervision but no analytical substrate
- ClickHouse has the analytics but no awareness/reactivity
- Postgres has the durability but no graph + cognitive
- JanusGraph+Cassandra has graph + scale but eventual consistency and no cognition

What each repo provides today:

| Repo | Provides | Key crates / paths |
|---|---|---|
| **lance-graph** | Cypher on Arrow + ontology + cognitive shaders | `lance-graph-contract`, `-catalog`, `-ontology`, `-rbac`, `cognitive-shader-driver`, `thinking-engine`, `causal-edge`, `deepnsm`, `holograph`, `bgz-tensor` |
| **surrealdb** | Multi-model DB + change feeds + KV abstraction | `core/src/cf/`, `core/src/kvs/lance/`, `core/src/kvs/tikv/`, `core/src/kvs/key.rs` |
| **sea-orm** | Typed entity ORM + Arrow surface | `sea-orm-arrow` (2.0.0-rc.4), `sea-orm-macros`, `sea-orm-codegen` |
| **ndarray** | SIMD distance kernels + tensor primitives | `hpc-extras` feature, `heel_f64x8::cosine_f64_simd` etc. |

What is missing — the **four glue crates**:

| # | Glue crate | Owner repo | Bridges |
|---|---|---|---|
| 1 | `surrealdb-ractor` | surrealdb | `cf` / live queries → ractor mailboxes |
| 2 | `lance-graph-tikv-provider` | **this repo** | TiKV ranges → Arrow `TableProvider` |
| 3 | `sea-orm-ractor` | sea-orm | `Entity::PK` → ractor process registry |
| 4 | `cognitive-shader-actor` | **this repo** | cognitive shaders → `ractor::Actor` adapter |

This repo owns glues **#2** and **#4**.

### Integration principle: additive contract shape

**All work in this plan is additive.** No existing trait signature changes. No existing module moves. No existing file deletes. New capabilities ship as **new traits in new modules** that consumers opt into by importing. The existing surface that downstream consumers already depend on stays exactly as-is. Any deprecation runway is signposted but out of scope here — five+ versions before any old surface is touched.

**Contract crates are the integration surface.** Cross-engine vocabulary lives in zero-dep trait crates any consumer can pin without bringing heavy implementations. New capability = new trait. New trait = optional dep. No load-bearing dependency added to an existing consumer.

### Contracts (existing + new)

| Contract | Owner repo | Status today | This plan adds |
|---|---|---|---|
| `lance-graph-contract` (graph + IR vocabulary) | lance-graph | 0.1.x, agnostic build wired into surrealdb-core (`surrealdb/core/Cargo.toml:69`) | **new submodules** `ir`, `provider`, `actor` — 0.2.0, **additive only** (no existing item moves or changes signature) |
| `KVKey` / `KVValue` / `Datastore` / `Transaction` | surrealdb | stable (`surrealdb/core/src/kvs/key.rs`, `kvs/api.rs`) | unchanged — **new traits** `CfStream` and `MvccSource` added alongside |
| `EntityTrait` / `ColumnTrait` / `Select<E>` | sea-orm | 2.0 (strongly-typed `COLUMN`, entity loader) | unchanged — **new trait** `EntityActor` + **new extension trait** `SelectArrowExt` |
| `ndarray::hpc::F64x8` + `heel_f64x8::*` | ndarray | 0.17 fork, stable | unchanged — **only new kernels** added |

**New submodules this repo adds to `lance-graph-contract`** (all 0.2.0, additive):

```rust
// crates/lance-graph-contract/src/ir.rs       — NEW
pub trait Operator { fn cardinality_estimate(&self) -> Option<u64>; /* ... */ }
pub trait OperatorTree { /* compose-able plan IR */ }

// crates/lance-graph-contract/src/provider.rs — NEW
/// Marker trait for Arrow providers backed by a specific storage.
/// Extends datafusion::catalog::TableProvider — doesn't replace it.
pub trait TikvBackedProvider: datafusion::catalog::TableProvider {
    fn snapshot_ts(&self) -> Option<u64>;
}

// crates/lance-graph-contract/src/actor.rs    — NEW
/// Marker trait for shaders that can be wrapped as ractor::Actor.
/// Implementors live in lance-graph-cognitive crates; the actor
/// wrapper lives in cognitive-shader-actor.
pub trait CognitiveShader: Send + Sync + 'static { /* ... */ }
```

**Per-repo enforcement**: every Sprint item below is read as "add this; don't change what's there."

---

## 2. Architecture diagram

```
                ┌──────────────────────────────────────────┐
                │              consumer crate              │
                └──────────────────┬───────────────────────┘
                                   │ typed entities
                                   ▼
                ┌──────────────────────────────────────────┐
                │            sea-orm-arrow 2.0             │  (planner-aware ORM)
                └────┬─────────────────┬───────────────┬───┘
                     │                 │               │
                     ▼                 ▼               ▼
              ┌───────────┐     ┌───────────┐    ┌───────────┐
              │  ractor   │◄────│ surrealdb │    │ THIS REPO │
              │ (actors,  │ #1  │  (cf +    │    │lance-graph│
              │ mailboxes,│     │   live    │    │ (Cypher,  │
              │ supervis.)│     │  queries) │    │ ontology, │
              └─────┬─────┘     └─────┬─────┘    │cognitive) │
                    │ #3              │          └─────┬─────┘
                    ▼                 │ #4             │ #2
              ┌───────────┐           │                │
              │ sea-orm   │           ▼                ▼
              │ Entity    │     ┌──────────────────────────┐
              │ Registry  │     │       TiKV substrate     │
              └───────────┘     │ (Raft + Percolator MVCC) │
                                └──────────────────────────┘
                                          │
                                          ▼
                                ┌─────────────────────────┐
                                │   ndarray fork (SIMD)   │
                                │   shared across stack   │
                                └─────────────────────────┘
```

---

## 3. Role of lance-graph in the integration

### What this repo provides upstream

- **Cypher → Arrow execution** via `crates/lance-graph` against any Arrow `TableProvider`
- **Ontology types** (`crates/lance-graph-ontology`) — typed node / edge / property shapes
- **Catalog** (`crates/lance-graph-catalog`) — schema registry, target for cross-repo codegen
- **RBAC** (`crates/lance-graph-rbac`) — permission enforcement
- **Contract crate** (`crates/lance-graph-contract`) — zero-dep traits already imported by `surrealdb-core` when the `lance-graph` feature is on (`surrealdb/core/Cargo.toml:69`)
- **Cognitive operators** — `cognitive-shader-driver`, `thinking-engine`, `causal-edge`, `deepnsm`, `holograph`

### What lance-graph consumes downstream

- Arrow `RecordBatch` inputs (from TiKV via glue #2, or from Lance projections fed by surrealdb's CDC pipeline — see surrealdb plan §6)
- ndarray SIMD distance kernels via `hpc-extras`
- `bgz-tensor` for shared tensor representation

---

## 4. Current state — file-by-file

### `crates/lance-graph-contract`
Zero-dep trait crate. Today this is the **only** lance-graph crate that `surrealdb-core` pulls in. **Sprint 0 adds new submodules** (`ir`, `provider`, `actor`) without touching the existing surface so today's consumers compile unchanged.

### `crates/lance-graph-catalog`
Schema registry. Internal to lance-graph today. **Becomes the single source of truth** consumed by:
- `surrealdb` (via `DEFINE` codegen — new method added to `Catalog`)
- `sea-orm` (via Entity codegen — new method)
- `lance-graph` itself (existing `NodeShape` / `EdgeShape` API unchanged)

Additive: new `to_sea_orm_entity()` and `to_surrealql_define()` methods; nothing existing changes.

### `crates/lance-graph-ontology`
Typed shapes. This is the Foundry-equivalent **Object Model**.

### `crates/lance-graph-rbac`
Maps to Foundry's permissions tier.

### Cognitive crates
- `cognitive-shader-driver` — shader execution engine
- `thinking-engine` — reasoning operators
- `causal-edge` — causal graph primitives
- `deepnsm` — neural state machines
- `holograph` — holographic embeddings

These run today as in-process invocations and **continue to do so**. The actor wrapper is a NEW crate (`cognitive-shader-actor`); the cognitive crates themselves stay unchanged. Consumers that want supervised behaviour opt into the wrapper; consumers that want in-process invocation keep it.

### `crates/bgz-tensor`
Shared tensor type. Used by surrealdb's vector path and lance-graph cognitive crates.

---

## 5. Glue #2 — `lance-graph-tikv-provider`

**Goal**: lance-graph reads typed columnar projections from TiKV ranges as Arrow.

**Why**: replaces JanusGraph-over-Cassandra. lance-graph today reads from in-process Arrow tables; for distributed graph workloads it needs a `TableProvider` that scans TiKV ranges using the schema declared in `lance-graph-catalog`.

**Additive shape**: this is a **new crate** that implements DataFusion's existing `TableProvider` trait and lance-graph-contract's new `TikvBackedProvider` marker. No existing lance-graph crate changes.

**Crate location**: `crates/lance-graph-tikv-provider/`

### API sketch

```rust
// crates/lance-graph-tikv-provider/src/lib.rs
use std::sync::Arc;
use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use datafusion::catalog::TableProvider;
use datafusion::physical_plan::ExecutionPlan;
use lance_graph_catalog::{NodeShape, EdgeShape};
use lance_graph_contract::provider::TikvBackedProvider;

pub struct TikvNodeTableProvider {
    /// TiKV transactional client.
    client: Arc<tikv_client::TransactionClient>,
    /// Shape from lance-graph-catalog. Drives schema + range encoding.
    shape: NodeShape,
    /// Optional MVCC snapshot. None = read latest.
    snapshot_ts: Option<u64>,
}

impl TikvNodeTableProvider {
    pub async fn new(
        client: Arc<tikv_client::TransactionClient>,
        shape: NodeShape,
    ) -> anyhow::Result<Self> {
        Ok(Self { client, shape, snapshot_ts: None })
    }

    pub fn with_snapshot(mut self, ts: u64) -> Self {
        self.snapshot_ts = Some(ts);
        self
    }
}

#[async_trait::async_trait]
impl TableProvider for TikvNodeTableProvider {
    fn schema(&self) -> SchemaRef {
        self.shape.arrow_schema()
    }

    async fn scan(
        &self,
        _state: &dyn datafusion::execution::context::SessionState,
        projection: Option<&Vec<usize>>,
        filters: &[datafusion::logical_expr::Expr],
        limit: Option<usize>,
    ) -> datafusion::error::Result<Arc<dyn ExecutionPlan>> {
        // 1. Translate filters into a TiKV key range via the shape.
        let key_range = self.shape.range_from_filters(filters)?;
        // 2. Snapshot read at the configured MVCC timestamp.
        let snapshot_ts = match self.snapshot_ts {
            Some(v) => v.into(),
            None => self.client.current_timestamp().await?,
        };
        let snapshot = self.client.snapshot(snapshot_ts, Default::default());
        // 3. Stream rows back as Arrow RecordBatch via batch_scan + decode.
        let exec = TikvScanExec::new(
            snapshot, key_range, self.shape.clone(),
            projection.cloned(), limit,
        );
        Ok(Arc::new(exec))
    }
}

/// Marker impl from lance-graph-contract::provider.
impl TikvBackedProvider for TikvNodeTableProvider {
    fn snapshot_ts(&self) -> Option<u64> { self.snapshot_ts }
}
```

### Why this is the JanusGraph replacement

- JanusGraph's Cassandra adapter does the same thing — encode element into a key, scan ranges, decode rows.
- This version inherits TiKV's Percolator ACID + native MVCC — Cassandra is eventually consistent.
- The Arrow output drops directly into lance-graph's Cypher executor with zero copy.

### Snapshot integration with the rest of the stack

The `snapshot_ts: u64` field is the same number as:
- `surrealdb-core`'s `version` column in `kv-lance` and `lance-projection` schemas
- TiKV's native HLC timestamp (when surrealdb runs in `kv-tikv-native-mvcc` mode — surrealdb plan §5b)
- The version a Lance projection refresh commits at

**One clock, all storage targets.** Enables cross-engine snapshot-consistent reads.

### Edge tables

A sibling `TikvEdgeTableProvider` handles edge shapes. Edges stored as `(src_node_id, edge_type, dst_node_id) → edge_props` so the encoded range starts with `src_node_id` — outbound expansion is a single range scan.

---

## 6. Glue #4 — `cognitive-shader-actor`

**Goal**: cognitive shaders become first-class supervisable actors.

**Why**: today the cognitive crates are in-process. For an AGI-style architecture with let-it-crash recovery and back-pressure, they need to live in ractor's supervision tree.

**Additive shape**: NEW crate. The cognitive crates themselves are unchanged — the wrapper is a *separate* dep that consumers opt into. Existing in-process call sites keep working.

**Crate location**: `crates/cognitive-shader-actor/`

### API sketch

```rust
// crates/cognitive-shader-actor/src/lib.rs
use ractor::{Actor, ActorRef, ActorProcessingErr};
use lance_graph_contract::actor::CognitiveShader;  // new trait, see §1 Contracts
use arrow_array::RecordBatch;

/// A cognitive shader wrapped as a ractor actor. One actor per shader instance;
/// the supervisor restarts it if the shader panics. Mailbox-bounded → back-pressure.
pub struct CognitiveShaderActor;

#[derive(Debug)]
pub enum ShaderMessage {
    Apply {
        input: RecordBatch,
        reply: ractor::RpcReplyPort<anyhow::Result<RecordBatch>>,
    },
    ApplyDelta { delta: RecordBatch },
    Drain { reply: ractor::RpcReplyPort<()> },
}

pub struct ShaderState {
    shader: Box<dyn CognitiveShader>,
    inflight: usize,
}

impl Actor for CognitiveShaderActor {
    type Msg = ShaderMessage;
    type State = ShaderState;
    type Arguments = Box<dyn CognitiveShader>;

    async fn pre_start(
        &self, _myself: ActorRef<Self::Msg>, shader: Self::Arguments,
    ) -> Result<Self::State, ActorProcessingErr> {
        Ok(ShaderState { shader, inflight: 0 })
    }

    async fn handle(
        &self, myself: ActorRef<Self::Msg>, msg: Self::Msg, state: &mut Self::State,
    ) -> Result<(), ActorProcessingErr> {
        match msg {
            ShaderMessage::Apply { input, reply } => {
                state.inflight += 1;
                let result = state.shader.apply(&input).await;
                state.inflight -= 1;
                let _ = reply.send(result);
            }
            ShaderMessage::ApplyDelta { delta } => {
                state.shader.apply_delta(&delta).await?;
            }
            ShaderMessage::Drain { reply } => {
                while state.inflight > 0 { tokio::task::yield_now().await; }
                let _ = reply.send(());
                myself.stop(Some("drained".into()));
            }
        }
        Ok(())
    }
}
```

### Supervisor topology

```
ShaderSupervisor (one-for-one, max 5 restarts in 60s)
├── ThinkingEngineActor
├── CausalEdgeActor
├── DeepNSMActor
├── HolographActor
└── CognitiveShaderDriverActor (orchestrates the others)
```

A misbehaving shader (panic, OOM, timeout) is restarted by its supervisor without affecting peers. **In-process call sites of the same shader continue to work** — the actor wrapper is opt-in.

---

## 7. Sprint sequence (this repo)

All sprints are **additive** — no existing crate changes signature, no existing module moves, no existing file deletes.

### Sprint 0 — `lance-graph-contract` 0.2.0 (additive) (1 week)
- **Add** new submodules `ir`, `provider`, `actor`
- Existing surface stays exactly as-is; SemVer-minor bump because new public items
- Cut 0.2.0 so `surrealdb-core` can pin it
- Existing surrealdb consumers compile without code change at 0.1.x → 0.2.0

### Sprint 1 — `lance-graph-tikv-provider` MVP (2 weeks)
- New crate; depends on `tikv-client` (workspace), `arrow`, `datafusion`, `lance-graph-contract = 0.2`
- Implement `TikvNodeTableProvider::scan` for the read path
- Implement `TikvEdgeTableProvider::scan` for outbound expansion
- Implement `TikvBackedProvider` marker from the new contract submodule
- Integration test against local TiKV via `make tikv-up`
- Honor `snapshot_ts` for MVCC

### Sprint 2 — `cognitive-shader-actor` MVP (1 week)
- New crate; depends on `ractor` + `lance-graph-cognitive` + `lance-graph-contract = 0.2`
- Wrap one shader (e.g. `thinking-engine`) end-to-end
- Demonstrate supervisor restart on shader panic
- Bench mailbox overhead vs direct invocation
- **In-process direct invocation continues to work** — the wrapper is additive

### Sprint 3 — catalog → codegen (2 weeks)
- `lance-graph-catalog` gets **new methods** `to_sea_orm_entity()` and `to_surrealql_define()`
- No existing Catalog methods change signature
- Codegen sea-orm entity files (see sea-orm plan §4)
- Codegen SurrealDB `DEFINE` statements (see surrealdb plan §6)
- Round-trip property tests both directions

### Sprint 4 — planner IR cost model (3 weeks)
- Operators in `lance-graph-contract::ir` carry cardinality estimates (additive within the new submodule)
- surrealdb's planner consumes the IR to route operators per-engine
- E2E test: a Cypher query touching surrealdb-kv + lance-graph + cognitive shader, results consumed via sea-orm-arrow

---

## 8. Examples

### Example 1 — Cypher over TiKV via the new provider

```rust
use std::sync::Arc;
use lance_graph::CypherQuery;
use lance_graph_tikv_provider::{TikvNodeTableProvider, TikvEdgeTableProvider};
use lance_graph_catalog::Catalog;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let tikv = Arc::new(
        tikv_client::TransactionClient::new(vec!["pd:2379"]).await?,
    );
    let catalog = Catalog::load("./schema.yml")?;

    let person_provider = TikvNodeTableProvider::new(
        tikv.clone(), catalog.node_shape("Person")?,
    ).await?;
    let knows_provider = TikvEdgeTableProvider::new(
        tikv.clone(), catalog.edge_shape("KNOWS")?,
    ).await?;

    let result = CypherQuery::new(
        "MATCH (a:Person)-[:KNOWS]->(b:Person)
         WHERE a.age > 30 RETURN b.name, b.age",
    )
    .with_provider("Person", Arc::new(person_provider))
    .with_provider("KNOWS",  Arc::new(knows_provider))
    .execute().await?;

    println!("{}", result.to_pretty_string());
    Ok(())
}
```

### Example 2 — Cognitive shader in a supervision tree, fed by a live query

```rust
use ractor::Actor;
use cognitive_shader_actor::{CognitiveShaderActor, ShaderMessage};
use lance_graph_cognitive::ThinkingEngineShader;
use futures::StreamExt;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let (actor, _h) = Actor::spawn(
        Some("thinking-1".into()), CognitiveShaderActor,
        Box::new(ThinkingEngineShader::new()),
    ).await?;

    let mut deltas = surrealdb_ractor::live_stream(
        "SELECT * FROM events WHERE category = 'plan'",
    ).await?;

    while let Some(delta) = deltas.next().await {
        actor.send_message(ShaderMessage::ApplyDelta {
            delta: delta?.into_record_batch(),
        })?;
    }
    Ok(())
}
```

### Example 3 — Catalog-driven entity codegen (additive `to_*` methods)

Single source of truth (`schema.yml`):

```yaml
nodes:
  Person:
    pk: id
    columns:
      id: UInt64
      name: String
      age: UInt32
      email: { type: String, unique: true }
edges:
  KNOWS:
    src: Person
    dst: Person
    properties: { since: Date }
```

Generates (via `catalog.to_sea_orm_entity()` — new method, no existing API touched):

```rust
#[sea_orm::model]
#[derive(Clone, Debug, PartialEq, Eq, DeriveEntityModel)]
#[sea_orm(table_name = "person")]
pub struct Model {
    #[sea_orm(primary_key)]
    pub id: i64,
    pub name: String,
    pub age: u32,
    #[sea_orm(unique)]
    pub email: String,
    #[sea_orm(has_many, via = "knows")]
    pub knows: HasMany<super::person::Entity>,
}
```

And (via `catalog.to_surrealql_define()`):

```sql
DEFINE TABLE person SCHEMAFULL;
DEFINE FIELD id    ON person TYPE int;
DEFINE FIELD name  ON person TYPE string;
DEFINE FIELD age   ON person TYPE int ASSERT $value >= 0;
DEFINE FIELD email ON person TYPE string;
DEFINE INDEX person_email ON person FIELDS email UNIQUE;
DEFINE TABLE knows SCHEMAFULL TYPE RELATION FROM person TO person;
DEFINE FIELD since ON knows TYPE datetime;
```

---

## 9. Open questions

1. **Snapshot delivery** — should `lance-graph-tikv-provider` carry `snapshot_ts` via the provider, or via a DataFusion session setting? Sketch picks provider; alternative lets the planner change snapshot mid-query.
2. **Cognitive shader hot reload** — BEAM has it; Rust doesn't. Mitigation: stop + restart with new config via supervisor.
3. **Catalog format** — YAML for cross-tool consumption, Rust DSL for type safety, or both? Probably YAML canonical + Rust types generated.
4. **bgz-tensor placement** — should this crate move to the workspace root or even to `AdaWorldAPI/ndarray`? Currently at `crates/bgz-tensor` but consumed cross-repo.
5. **`lance-graph-contract` 0.2.0 SemVer** — 0.2.0 confirms additive-only intent (new public items, no breaking change). 0.1.x consumers compile unchanged.

---

## 10. Cross-references

- **Glue #1** (surrealdb-ractor): `AdaWorldAPI/surrealdb:.claude/plans/integration-plan.md` §5
- **Glue #3** (sea-orm-ractor): `AdaWorldAPI/sea-orm:.claude/plans/integration-plan.md` §5
- **SIMD kernels**: `AdaWorldAPI/ndarray:.claude/plans/integration-plan.md`
- **lance-projection (additive sibling to kv-lance)**: `AdaWorldAPI/surrealdb:.claude/plans/integration-plan.md` §6
