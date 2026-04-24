# ADR 0001 — Archetype Transcode + Stack Lock

> **Status:** **Accepted** (locked 2026-04-24)
> **Supersedes:** None
> **Superseded by:** None — unlocking requires a new ADR that cites this one by number
>
> **Scope:** Two coupled decisions that bind all future Archetype-related
> deliverables, grammar work, callcenter/persona work, and any code that
> stores, queries, schedules, or superposes over tick-structured data in
> the lance-graph workspace.

---

## Decision 1 — Archetype is TRANSCODED, not bridged

The `VangelisTech/archetype` upstream Python package is **assimilated into
a native Rust crate** in the lance-graph workspace. The Archetype MODEL
(append-only columnar tick snapshots, Component / Archetype-table /
AsyncProcessor / CommandBroker / World contracts, DataFrame-first
processors, append-only writes, world forking via version branch) is
replicated in Rust against the lance-graph substrate.

### What transcode means (and what it doesn't)

| | Transcode (THIS decision) | Bridge (REJECTED) |
|---|---|---|
| Python Archetype runtime | Not used | Runs alongside, receives/sends events |
| Call direction | Rust owns the contract; no FFI | Rust calls Python via FFI or HTTP |
| Operational model | One process, one deployment | Two processes, two deployments |
| Arrow wire | Native (no marshal) | Arrow IPC over stdio/socket |
| Failure modes | Rust compilation + test | Python availability + FFI crashes + wire serialization drift |

**DU-2 in `unified-integration-v1.md`** currently uses bridge wording
("`ArchetypeWorld → Blackboard` adapter") — that framing is updated by
this ADR to transcode. The external Python Archetype is a DESIGN
SPEC, not a runtime dependency. Adapters in DU-2 become native Rust
impls of the transcoded contracts, not translation layers to a running
Python service.

### Crate name

`lance-graph-archetype` — not `lance-graph-archetype-bridge`. The
"bridge" suffix implies an external runtime; the transcode has none.

### Contracts assimilated (minimum viable surface)

```rust
// ─── Core contracts (from Archetype src/archetype/core) ────────────

trait Component: Send + Sync + 'static {
    fn arrow_field() -> Field;  // maps to Arrow schema
}

trait Processor: Send + Sync {
    fn matches(&self, schema: &Schema) -> bool;
    fn process(&self, batch: RecordBatch) -> Result<RecordBatch>;
}

// ─── World / tick / dataset ────────────────────────────────────────

struct World {
    dataset: Arc<LanceDataset>,     // versioned Lance = append-only tick history
    tick: u64,                      // monotonic tick counter
    command_queue: CommandBroker,
}

impl World {
    fn tick(&mut self, processors: &[Box<dyn Processor>]) -> Result<()>;
    fn fork(&self) -> Result<World>;          // Lance version branch
    fn at_tick(&self, t: u64) -> Result<Snapshot>;  // Lance time-travel
}

// ─── Service layer (from src/archetype/app) ───────────────────────

struct CommandBroker { /* channel-based enqueue + per-tick drain */ }
struct WorldService  { /* wraps World with service-level semantics */ }
struct QueryService  { /* DataFusion SQL over Lance, not FastAPI */ }
struct SimulationService { /* tick-loop scheduler */ }
```

The Rust crate exposes the SAME conceptual contracts as the Python
upstream. It is NOT a wrapper of Polars DataFrames — `RecordBatch` is
the DataFrame; DataFusion is the query engine; Lance is the storage.

### Rationale

- **Rust performance where simulation scales.** Archetype is designed
  for LLM-powered processors running over many entities in parallel.
  Rust native + DataFusion parallel execution is the right runtime
  for that scale; Python has async but the GIL caps processor
  parallelism.
- **Lance version model matches Archetype's storage model by
  construction.** Both are append-only, both do free time-travel,
  both fork cheaply. The storage layer already implements Archetype's
  persistence contract. Transcoding makes this alignment explicit.
- **Zero-copy Arrow throughout.** DataFusion ↔ Lance is zero-copy;
  Polars-in-Python would require serialization at every boundary.
- **Python Archetype remains the upstream reference.** When the
  upstream API evolves, the Rust crate re-transcodes the surface that
  matters. The design spec stays upstream; the runtime stays in Rust.


---

## Decision 2 — Stack lock: Lance + DataFusion + Supabase-shape scheduler + Arrow temporal

The substrate stack for the Archetype transcode (and, by extension, all
future tick-structured / role-indexed / windowed work in lance-graph) is
locked:

| Layer | Choice | Why |
|---|---|---|
| **Storage** | Lance (versioned append-only) | Already in stack. Native Archetype-model match (append-only snapshots, version branch = world fork, time-travel free). |
| **Query** | DataFusion | Already in stack via `lance-graph-planner` (16 strategies). Native `lance-datafusion` zero-copy Arrow. UDF surface for VSA operations (DU-3). Window functions for ±5 Markov frame are SQL-native. |
| **Scheduler** | Supabase-shape tick loop transcoded to Rust channels (DM-4 `LanceVersionWatcher` + DM-6 `DrainTask`) | Supabase's scheduler SHAPE (enqueue → tick → drain → process → append → notify) maps to channel-based Rust tick loops. Supabase's BACKING (PostgreSQL + pg_cron) is rejected — PG conflicts with Lance. |
| **Temporal** | Arrow temporal types + DataFusion window functions | For multi-rate tick joins: `ASOF JOIN` on `Timestamp` columns. For ±5 Markov frame: `ROWS BETWEEN 5 PRECEDING AND 5 FOLLOWING OVER (ORDER BY tick)`. No dedicated temporal engine needed until scale demands. |
| **Distribution** | Ballista — DEFERRED until trigger | Free upgrade from DataFusion (same plans, same UDFs, add Arrow Flight). Premature until P99 latency demands it. |
| **Polars** | **REJECTED** in production | Dual-engine maintenance outweighs single-node speed advantage. Benchmark-as-experiment is OK; Polars in crate dependencies is not. |

### Ballista activation trigger (immutable once locked)

Ballista is added to the stack WHEN:

> **Single-node query P99 latency on the Animal Farm benchmark OR the
> callcenter hot path exceeds 1 second after reasonable DataFusion
> optimization has been applied.**

"Reasonable optimization" means: Lance file compaction tuned, DataFusion
window function plans inspected, UDF hot paths profiled, obvious
bottlenecks addressed. The trigger is the threshold, not the date.
Before the threshold, Ballista is deferred. After, it is mandatory.

### Polars rejection scope

| | Polars permitted | Polars forbidden |
|---|---|---|
| Crate dependencies | No | No — zero `polars = *` lines in any `Cargo.toml` |
| UDF implementations | No | No — UDFs target DataFusion |
| Benchmark / experiment repos | YES | — (benchmarks are orthogonal) |
| External tooling calling lance-graph | YES | — (upstream Python Archetype uses Polars; that's upstream's business) |

### Rationale

- **DataFusion already in stack.** Adding Polars creates two DataFrame
  engines. Two engines = two code paths, two UDF surfaces, two type
  systems. One is enough.
- **Lance integration is native to DataFusion.** Zero-copy Arrow. Polars
  ↔ Lance would require bridge code that re-marshals at every boundary.
- **DU-3 VSA UDFs target DataFusion.** Porting to Polars expressions is
  possible but doubles the surface. The UDFs (`role_bind`, `role_unbind`,
  `vsa_bundle`, `vsa_cosine`, `markov_window`) are DataFusion-first.
- **Ballista upgrade path requires DataFusion.** If we commit to Polars,
  distributed execution means rewriting query orchestration. Building on
  DataFusion keeps the upgrade free.
- **Supabase-shape without PostgreSQL is cheaper than it sounds.** DM-4
  `LanceVersionWatcher` (detects new Lance versions → notify) + DM-6
  `DrainTask` (tick-based command processing) replicate pg_cron +
  Realtime without the PG dependency. Three Rust tasks + channels; not
  a full database extension.


---

## Consequences

### What this ADR locks (immutable)

1. **`lance-graph-archetype` crate name** — not `-bridge`, not `-adapter`.
2. **Rust native transcode of Archetype contracts** — no Python runtime
   dependency.
3. **Storage = Lance** — versioned append-only. The Archetype tick history
   is Lance versions.
4. **Query = DataFusion** — UDFs, window functions, SQL. No Polars.
5. **Scheduler = Rust channel-based tick loop** — transcoded from
   Supabase's `pg_cron + Realtime` shape; no PostgreSQL.
6. **Temporal = Arrow types + DataFusion window functions** — no dedicated
   temporal engine below the Ballista trigger.
7. **Polars = rejected in production code** — no crate deps, no UDFs.
   Benchmarks outside production are fine.

### What this ADR does NOT lock (mutable)

1. **Ballista trigger threshold (1s P99)** — tune empirically after
   first benchmark runs. If 500ms is the right number, amend this ADR.
2. **Arrow temporal APIs** — whether to use `Timestamp(Microsecond, None)`
   vs `Timestamp(Nanosecond, UTC)` etc. is implementation choice.
3. **Crate partitioning** — whether `lance-graph-archetype` becomes one
   crate or splits into core/service/adapter sub-crates is tactical.
4. **DU-2 surface details** — re-clarify on a follow-up PR now that the
   ADR is in place.

### What it implies for other plans

- **`unified-integration-v1.md` DU-2** — needs a clarification commit:
  rename "bridge" to "transcode", rename target crate to
  `lance-graph-archetype`, reframe adapters as native Rust impls of
  transcoded contracts.
- **`callcenter-membrane-v1.md` DM-4/DM-6** — already aligned. The
  `LanceVersionWatcher` + `DrainTask` pattern IS the Supabase-shape
  transcode; this ADR validates that design.
- **`categorical-algebraic-inference-v1.md`** — unchanged. The Five
  Lenses meta-architecture sits above the storage/query layer; choice
  of Lance + DataFusion doesn't alter the Kan extension / free-energy
  / NARS revision / AriGraph commit / awareness loop.
- **`elegant-herding-rocket-v1.md`** — unchanged. D5 (Markov ±5 bundler)
  becomes a DataFusion window UDF rather than a ring buffer when
  Apache temporal becomes necessary. Before that, ring buffer is fine.

### What it implies for future ADRs

- **Unlocking any Decision-1 or Decision-2 item requires a new ADR** that
  cites this one by number (`ADR 0001`) and explicitly supersedes the
  reversed item.
- **Ballista activation is an AMEND** to this ADR (update the "Mutable"
  section with the measured threshold), not a new ADR.
- **Additional storage/query engines for NEW workloads** (e.g. a
  specialized vector index) can be added without unlocking this ADR, as
  long as they don't REPLACE Lance or DataFusion for the locked
  workloads.

---

## Cross-references

- `.claude/plans/unified-integration-v1.md` DU-2 (needs update per
  "what it implies for other plans")
- `.claude/plans/callcenter-membrane-v1.md` DM-4, DM-6 (aligned)
- `.claude/board/STATUS_BOARD.md` — add row: `ADR 0001 — Archetype
  transcode + stack lock (Accepted 2026-04-24)`
- `.claude/board/TECH_DEBT.md` — add row: `Ballista trigger threshold
  tuning` (P3, mutable per this ADR)
- `.claude/knowledge/vsa-switchboard-architecture.md` — VSA discussion
  unaffected; the stack lock is orthogonal to substrate choice
- `VangelisTech/archetype` upstream Python reference
- Lance: https://github.com/lancedb/lance
- DataFusion: https://github.com/apache/arrow-datafusion
- Ballista: https://github.com/apache/arrow-ballista

---

## Lock statement

> Decisions 1 and 2 of this ADR are **locked as of 2026-04-24**. Any
> future session that proposes to use Polars in production code, to
> add Ballista before the latency trigger fires, to bridge-to-Python
> instead of transcode, or to switch storage/query engine for a locked
> workload — MUST author a new ADR that cites this one and justifies
> the reversal. Individual sessions cannot unlock by reinterpretation;
> the lock survives handoff.


---

## Decision 3 — Persona 16^32 atom-space is THE identity layer

### The single identity space

Persona identity lives in **one** coordinate system, workspace-wide:

```
Persona identity space = 32 cognitive atoms × 16 weightings per atom
                       = 16^32 addressable coordinates
                       ↓ deterministic compression
                       = 56-bit PersonaSignature
```

This is the ONLY persona identity representation that crosses any
internal interface. YAML runbooks, JIT `KernelHandle`s, dispatch
tables, and training corpora all reference personas by signature,
never by filename or ad-hoc string ID.

### The three contracts share substrate, differ by lifecycle position

The user observation this ADR codifies:

> The Blackboard (A2A), the current contracts, and the Grammar
> Markov ±5 / ±500 semantic kernel are reusable — or even belong
> into the same DTO in different places.

**Accepted.** These three objects share a common algebraic
substrate (role-indexed VSA identity superposition in the
switchboard carrier). They differ by WHERE in the lifecycle they
appear and WHICH fields the BBB allows through:

| Contract | Lifecycle stage | Role-indexed fields | BBB side |
|---|---|---|---|
| **A2A Blackboard entry** (`a2a_blackboard::BlackboardEntry`) | Per-round, per-expert deposit | `expert_id`, `capability`, `result`, `confidence`, `support[4]`, `dissonance`, `cost_us` | **Internal-only.** Never crosses BBB; the `ExternalMembrane` strips them before projection to `CognitiveEventRow`. |
| **PersonaSignature** (56-bit) | Addressing identity at dispatch time | 56 bits of atom-space coordinate | **BBB-bidirectional** — identity is public; internals are private. The SIGNATURE crosses; the 32×16 atom-weighting vector does NOT. |
| **Grammar Markov ±5 / ±500 kernel** | Per-cycle trajectory formation + coherence replay | `trajectory`, `context_chain` (±5 ring buffer), `episodic ±500` (retrieval) | **Internal-only.** Lives inside `Think` struct. The SCALAR projection (coherence score, F-value) crosses via `CognitiveEventRow`; the bundle does not. |

**The shared substrate** (all three):
- Role-indexed slice addressing over `Vsa10kF32` (or post-rename `Vsa16kF32`)
- Element-wise multiply/add algebra (per `I-VSA-IDENTITIES` + `I-SUBSTRATE-MARKOV`)
- NARS truth with φ-1 confidence ceiling
- Permutation-braided temporal ordering where applicable

**The shared DTO shape (conjecture, not yet formalized):**

The Blackboard entry, PersonaSignature, and Markov trajectory could
share a "role-indexed identity payload" DTO at the algebraic level —
with role assignment determining which semantics apply:

```rust
// CONJECTURE — not a commit, a sketch for future ADRs to consider
struct RoleIndexedIdentity {
    signature: PersonaSignature,       // 56-bit coordinate (if persona)
    trajectory: Option<Vsa10kF32>,     // bundle (if trajectory-typed)
    role_fillers: Vec<(RoleKey, Vsa10kF32)>,  // bound identities
    truth: TruthValue,                 // NARS truth with φ-1 ceiling
    lifecycle_stage: LifecycleStage,   // Blackboard | Dispatch | Markov | Episodic
}
```

If this unification lands in a future ADR, the three contracts
become views on one DTO. If it doesn't, they stay separate types
but all respect the same algebraic substrate. **Either way, Decision
3 locks the identity space itself.**

### BBB enforcement

The `ExternalMembrane` type in `lance-graph-contract` already bans
`Vsa10k`, `RoleKey`, `SemiringChoice`, `NarsTruth` from crossing the
gate (per `external_membrane.rs:10`). This ADR extends that ban:

- **Banned from crossing BBB:** Blackboard entries (internal A2A),
  raw atom-weighting vectors (the 32×16 unfolding of the signature),
  Markov trajectory bundles (`Vsa10kF32`), NARS per-role truth tables.
- **Permitted across BBB:** `PersonaSignature` (56 bits, opaque),
  `CognitiveEventRow` scalar projection (coherence / F-value /
  commit status), `Fingerprint<256>` as hashed identity (not
  unbindable, hence safe).

The type system enforces this at compile time via `ExternalMembrane`;
no runtime check needed.

### What this locks

1. **Persona identity = 56-bit PersonaSignature.** No ad-hoc string
   IDs, no raw YAML paths, no UUIDs that bypass the signature.
   Comparison is a 56-bit equality / Hamming distance, not a string
   diff.
2. **Atom-weighting vector (32×16) stays internal.** BBB-banned. The
   signature crosses; the unfolded vector does not.
3. **Blackboard / Persona / Markov contracts share substrate.** They
   MAY share DTO shape (future ADR), but they share role-indexed VSA
   algebra NOW. Any code that treats one as algebraically distinct
   from the others is wrong.
4. **BBB enforcement extension to Decision 3 objects.** The ban list
   grows to include atom-weighting vectors and Markov trajectory
   bundles; the permit list grows to include PersonaSignature +
   scalar projections.

### Rationale

- **One identity space means one comparison semantics.** If persona
  identity lived in multiple spaces (string ID, UUID, YAML path, atom
  vector, signature), comparing two personas would have four possible
  answers. 56-bit PersonaSignature is the ONE answer.
- **BBB extension mirrors I-VSA-IDENTITIES.** The iron rule already
  says "VSA operates on identities, not content." Decision 3 applies
  this rule specifically to persona: signature = identity (crosses),
  atom-weighting vector = content (does not).
- **Shared-substrate observation enables future unification.** If
  Blackboard + Persona + Markov share a DTO in a future ADR, the
  payoff is one codec, one serde surface, one CollapseGate predicate,
  one replay mechanism. Decision 3 doesn't make that commitment now
  — it preserves the OPTIONALITY.

---

## Lock statement (updated to cover all three decisions)

> **Decisions 1, 2, and 3 of this ADR are locked as of 2026-04-24.**
>
> Any future session that proposes to:
>
> - Use Polars in production code
> - Add Ballista before the latency trigger fires
> - Bridge-to-Python instead of transcode Archetype
> - Switch storage/query engine for a locked workload
> - Introduce a persona identity representation other than 56-bit
>   PersonaSignature
> - Let atom-weighting vectors or Markov trajectories cross the BBB
>
> MUST author a new ADR that cites this one (`ADR 0001`) and justifies
> the reversal. Individual sessions cannot unlock by reinterpretation;
> the lock survives handoff. The Blackboard + Persona + Markov
> shared-DTO unification is an OPEN QUESTION for future ADRs, not a
> lock here — but all three contracts share algebraic substrate and
> this ADR makes that sharing explicit.


---

## Addendum to Decision 2 — Ballista path is shorter than it looks (the lab gRPC is already there)

The `cognitive-shader-driver` crate already ships a gRPC surface
behind the `grpc` feature gate (see `crates/cognitive-shader-driver/
src/grpc.rs` and `lab-vs-canonical-surface.md`). Ballista's
distribution protocol is **Arrow Flight over gRPC** — the transport
we already serve on the lab surface.

### What this means for the Ballista trigger

When the 1s-P99 latency threshold fires, the upgrade path is:

| Step | Work | Cost |
|---|---|---|
| 1 | Keep lab `grpc` feature endpoints as-is (Wire DTOs) | Zero — already shipped |
| 2 | Add an Arrow Flight endpoint (`DoGet` / `DoPut` / `ListActions`) alongside the existing lab gRPC handlers | ~150 LOC, tonic + arrow-flight crates |
| 3 | Package the DataFusion query plans as Ballista tasks (same `LogicalPlan`; different executor) | ~80 LOC, ballista-core |
| 4 | Deploy scheduler + N executors; route lab clients through scheduler | Operational, not code |

The code surface for Ballista activation is ~230 LOC of adapter plus
deployment config. It is not "add gRPC" (already there) nor "rewrite
query plans" (same DataFusion plans ship to Ballista unchanged).

### The implied invariant

**The `grpc` feature gate on `cognitive-shader-driver` becomes load-
bearing for Ballista readiness.** Future sessions must not remove or
rename it without amending this ADR — doing so breaks the zero-cost
upgrade path.

### Cross-reference

- `crates/cognitive-shader-driver/src/grpc.rs` — current lab gRPC
  handlers (token-agreement, sweep, calibrate, etc.)
- `.claude/knowledge/lab-vs-canonical-surface.md` — lab = API +
  Planner + JIT, I11 measurability invariant
- Ballista Arrow Flight docs: https://arrow.apache.org/docs/format/Flight.html
- Tonic (Rust gRPC): https://github.com/hyperium/tonic

---

## Summary — three locks, three decisions, one ADR

| Decision | Lock | Mutability |
|---|---|---|
| **1 — Archetype transcode** | Native Rust crate `lance-graph-archetype`, no Python runtime dependency, contracts assimilated | Immutable (supersede via new ADR) |
| **2 — Stack lock** | Lance + DataFusion + Supabase-shape scheduler + Arrow temporal; Polars rejected in production; Ballista deferred to 1s P99 trigger | Ballista threshold mutable; everything else immutable |
| **3 — Persona 16^32** | 56-bit PersonaSignature is THE identity space; atom-weighting vector stays internal (BBB-banned); Blackboard / Persona / Markov share algebraic substrate | Shared-DTO unification is OPEN (future ADR); identity space itself is immutable |


---

## Addendum — Grok gRPC as first external-LLM integration lane

xAI Grok exposes a gRPC API (Arrow Flight compatible transport). The
lab `grpc` feature gate that this ADR makes load-bearing for Ballista
readiness is ALSO the natural integration lane for Grok as an
external A2A expert:

```
Grok gRPC response  →  lab grpc handler  →  Blackboard expert entry
                                              (expert_id = "grok",
                                               capability = ...,
                                               result = ...,
                                               confidence, support, etc.)
```

**This is usable NOW, pre-Ballista.** The lab gRPC endpoints can
accept external LLM responses encoded as `BlackboardEntry`-shaped
payloads. The BBB rule from Decision 3 holds: the external LLM
receives scalar projections (`CognitiveEventRow`) on the outbound
path; the `BlackboardEntry` it fills is internal-only, transported
through the gRPC lane but never exposed as external content. Grok
writes to the internal blackboard via a constrained outbound call;
Grok receives only projection rows.

**Not a lock, an observation.** This ADR does not mandate Grok
integration; it records that the lab gRPC surface is already shaped
for it. If Grok-as-expert becomes a deliverable, no new transport
work is needed.

## Addendum — Context enrichment for external consumers (DESIGN QUESTION, not locked)

`CognitiveEventRow` is the minimum scalar projection that crosses BBB.
Some external consumers (dashboards, LLM routers, simulation monitors)
need MORE context than bare scalars — but **less than** full internal
state (which is BBB-banned).

The design question (OPEN, tracked as tech debt P2):

> What is the ENRICHED projection shape for external consumers that
> exceeds `CognitiveEventRow` but respects `I-VSA-IDENTITIES` and the
> `ExternalMembrane` ban list?

**Candidates (not decided):**
- `EnrichedCognitiveRow` with additional scalar fields (e.g. Staunen
  magnitude, arc pressure, recent commit digest as `Fingerprint<256>`)
- `TrajectorySummary` with hashed identities (no unbindable VSA) and
  scalar coherence metrics
- `BlackboardRoundDigest` — round ID + expert count + aggregate
  confidence, no per-expert detail

**Governance:** ANY enrichment must pass the BBB type-system gate in
`external_membrane.rs`. The `deny` list stays authoritative.
Additions to the `permit` list are ADR-worthy changes, not ad-hoc
edits — the BBB is the most load-bearing invariant in the workspace.

This ADR does not decide the enrichment shape. It records the
question so a future ADR can answer it with the right evidence
(which consumers need what).


---

## Closing finding — AriGraph alone isn't enough, Markov kernel alone isn't enough, Blackboard alone isn't enough

The shared-substrate observation in Decision 3 isn't just an
optimization opportunity — it's a **load-bearing architectural claim
about what cognition requires**:

| Alone | What it gives | What it lacks |
|---|---|---|
| **AriGraph** (triplet-graph episodic memory) | Persistent epistemic state; SPO facts with NARS truth + Pearl 2³ + Contradiction markers | No per-cycle cognitive trajectory; no active-inference loop; no multi-expert round composition |
| **Markov ±5 / ±500 semantic kernel** | Per-cycle role-indexed trajectory; contextual coherence via windowed superposition; counterfactual replay within a cycle | No persistence beyond the window; no cross-cycle memory; no multi-expert coordination |
| **A2A Blackboard** | Per-round multi-expert deposit + composition; capability-tagged entries with confidence/dissonance | No memory depth; no temporal coherence; no long-range context |

None of the three is sufficient. Each is NECESSARY. The **Think**
struct binds all three as tissue (per `CLAUDE.md § The Click`):

```
Think {
    trajectory:     Vsa10kF32 Markov bundle,   ← Markov kernel  (cycle-local context)
    episodic:       &EpisodicMemory ±500,      ← AriGraph       (persistent memory)
    graph:          &TripletGraph,             ← AriGraph       (committed beliefs)
    global_context: &Vsa10kF32,                ← AriGraph       (ambient prior)
    awareness:      &GrammarStyleAwareness,    ← Blackboard-ish (self-tracking NARS)
    free_energy:    FreeEnergy,                ← derived from all three
    resolution:     Resolution,                ← derived from all three
}
```

The lock implication: **no ADR may remove or disable any one of
{AriGraph, Markov kernel, Blackboard/A2A} from the Think substrate
without re-proving cognition can emerge from the remaining two.**
Empirical evidence (the 2026-04-21 categorical-algebraic session's
trace analysis + Shaw's categorical foundation) shows all three are
necessary. The shared-DTO unification question (Decision 3) is about
HOW they compose into one DTO at different lifecycle stages, not
whether any of them can be dropped.

**This is why the Archetype transcode matters.** Archetype (ECS
tick + DataFrame processor) is the execution substrate that drives
all three simultaneously per tick:

- The tick advances the Markov kernel (±5 window slides)
- The tick commits winning hypotheses to AriGraph (`World.tick()` →
  append to Lance dataset)
- The tick drains a Blackboard round (CommandBroker → processors →
  per-expert deposits)

Without Archetype as the orchestrator, the three components lack a
single clock. With Archetype, all three advance in lockstep. **The
transcode is not a convenience — it is the missing execution frame
that makes the three-component minimum operable.**

