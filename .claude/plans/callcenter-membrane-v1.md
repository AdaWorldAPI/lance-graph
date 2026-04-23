# Plan — External Callcenter Membrane (Supabase-shape over Lance + DataFusion)

> **Version:** v1
> **Author:** main-thread session 2026-04-22
> **Status:** Active
> **Supersedes:** None
> **Confidence:** CONJECTURE — design grounded in repo evidence + conversation arc;
> no implementation shipped yet beyond DM-0 / DM-1 skeleton.
>
> **READ BY:** Every session touching REST/gRPC external API, realtime
> subscriptions, agent handover, n8n/crewai-rust/openclaw integration,
> auth, or any "external consumer sees cognitive state" work.

---

## § 0 — The One-Sentence Goal

Assimilate the *design and ergonomics* of the Supabase callcenter surface
into a new crate (`lance-graph-callcenter`) that sits entirely outside
the canonical cognitive substrate, backed by Lance + DataFusion instead of
PostgreSQL, and enforces the BBB (blood-brain barrier) at compile time via
the Arrow type system.

---

## § 1 — What This Is NOT

- NOT a PostgreSQL-compatible server.
- NOT a full Supabase clone.
- NOT a new canonical reasoning substrate.
- NOT a replacement for `OrchestrationBridge` (the inbound steering
  path still goes through `OrchestrationBridge::route(UnifiedStep)`;
  the callcenter translates external intent *into* `UnifiedStep`).
- NOT a new home for semiring types, Vsa10k, RoleKey, or FreeEnergy.

---

## § 2 — Architecture (four layers, clean separation)

```
╔═══ A: Canonical internal substrate (untouched) ══════════════════════╗
║  Vsa10k · BindSpace SoA · CognitiveShader · CollapseGate · AriGraph  ║
║  a2a_blackboard · NARS revision · FreeEnergy · Awareness              ║
╚══════════════════════════════╤═══════════════════════════════════════╝
                               │ CollapseGate fire (EmitMode::Persist)
                               │ ShaderBus + MetaWord cross here
╔═══ B: ExternalMembrane trait (in contract crate, zero-dep) ══════════╗
║  project(&ShaderBus, MetaWord) → Self::Commit                         ║
║    — strips VSA fields, produces Arrow scalars                        ║
║  ingest(Self::Intent) → UnifiedStep                                   ║
║    — translates external intent to canonical dispatch                 ║
║  subscribe(CommitFilter) → Self::Subscription                         ║
║    — typed subscription handle                                        ║
╚══════════════════════════════╤═══════════════════════════════════════╝
                               │ LanceMembrane impl (in callcenter crate)
                               │ Arrow RecordBatch is the Commit type
╔═══ C: Dual ledger (Lance datasets) ══════════════════════════════════╗
║  External ledger (append-only, versioned):                            ║
║    cognitive_event   — every CollapseGate fire, Arrow scalars only    ║
║    steering_intent   — inbound, consumed by drain task                ║
║    memory / episode  — rolled-up facts                                ║
║    actor / session   — identity + context accumulation                ║
║                                                                       ║
║  Internal ledger (mutable, NARS-revised):                             ║
║    AriGraph + SPO store  — inside lance-graph, never touches callcenter║
║                                                                       ║
║  Lance version counter = CDC stream (replaces PG WAL)                 ║
║  tokio::sync::watch on version → subscribers notified                 ║
╚══════════════════════════════╤═══════════════════════════════════════╝
                               │ DataFusion SQL + Lance reads
╔═══ D: lance-graph-callcenter crate (external, feature-gated) ════════╗
║  LanceMembrane   — ExternalMembrane impl, Arrow RecordBatch commit    ║
║  CommitRecord    — concrete Arrow scalar row shape                    ║
║  CommitFilter→Expr — PostgREST filter params → DataFusion Expr        ║
║  PhoenixServer   — Phoenix channel WS server (scaraude protocol shapes)║
║  DrainTask       — steering_intent → UnifiedStep → OrchestrationBridge ║
║  JwtMiddleware   — verify only; actor_id → LogicalPlan rewriter       ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## § 3 — Blood-Brain Barrier Invariant (the thing that makes it safe)

**The Arrow type system enforces the BBB at compile time.**

`LanceMembrane::project()` returns `arrow::array::RecordBatch`. Arrow
columns are typed: `Float32`, `UInt32`, `Utf8`, `Timestamp`, `Binary`.
Semiring types (`SemiringChoice`, `NarsTruth`, `HammingMin`, etc.) and
VSA types (`Vsa10k = [u64; 157]`, `RoleKey`, `FreeEnergy`) do NOT implement
Arrow's `Array` trait. They cannot be placed in a `RecordBatch` column.

**No runtime check needed. The compiler rejects the violation.**

The only Arrow column that carries any VSA-adjacent data is
`bundle_bytes: Option<Binary>` — an opaque pre-computed fingerprint
for fast ANN queries over the external ledger. This is a black-box byte
string to the callcenter; it cannot be unbound or used as a Vsa10k.

---

## § 4 — Cognitive event schema (the only Arrow schema crossing the BBB)

```
cognitive_event:
  timestamp:        Timestamp(Microsecond, UTC)
  actor_id:         UInt64
  session_id:       UInt64
  external_role:    UInt8             ← ExternalRole (0..7: User..Agent)
  faculty_role:     UInt8             ← FacultyRole (0..15: ReadingComp..Empathy..)
  expert_id:        UInt16            ← ExpertId / card hash (stable hash of card YAML)
  dialect:          UInt8             ← query dialect tag (Cypher/GQL/NARS/Redis/Spark/SQL)
  scent:            UInt8             ← 1-byte compressed address scent (see § 10.13)
  free_energy:      Float32           ← MetaWord::free_e() as f32
  resonance:        Float32           ← ShaderResonance::entropy
  recovery_margin:  Float32           ← top-1 ShaderHit::resonance
  thinking_style:   UInt8             ← MetaWord::thinking()
  awareness:        UInt8             ← MetaWord::awareness()
  nars_f:           UInt8             ← MetaWord::nars_f()
  nars_c:           UInt8             ← MetaWord::nars_c()
  gate_decision:    UInt8             ← GateDecision packed
  hit_count:        UInt16            ← ShaderResonance::hit_count
  cycles_used:      UInt16            ← ShaderResonance::cycles_used
  bundle_bytes:     Binary (optional) ← opaque pre-computed fingerprint
  is_epiphany:      Boolean
  is_failure:       Boolean
```

No Vsa10k. No semiring. No RoleKey. These are numbers and bytes. The five
new identity/scent columns (`external_role`, `faculty_role`, `expert_id`,
`dialect`, `scent`) are the metadata address bus coordinates (§ 10.11); they
are queryable via SQL / Cypher / GQL / NARS / qualia without ever exposing
the internal RoleKey slot mapping.

---

## § 5 — What we take from upstream Supabase Rust crates

### From `scaraude/supabase-realtime-rs` (v0.1.0, Phoenix channel client)

Take:
- Phoenix channel message type shapes (`phx_join`, `phx_leave`,
  `broadcast`, `postgres_changes`, `PresenceState` shape)
- Topic routing enum structure
- Heartbeat lifecycle pattern

Do NOT take:
- JWT refresh reconnect machinery (we handle auth differently)
- The full reqwest-based connection manager (we are the server)

Vendor strategy: manually extract the 6–8 Phoenix message types into
`lance-graph-callcenter/src/phoenix/types.rs`. ~150 lines. No dep on
the upstream crate.

### From `xylex-group/supabase_rs` (v0.7.0, PostgREST client)

Take:
- Filter operator enum (`eq`, `neq`, `gt`, `lt`, `gte`, `lte`, `like`,
  `ilike`, `in`, `is`, `not`)
- Query-string parameter parsing patterns

Do NOT take:
- `reqwest` (client-side HTTP; we are the server)
- Storage module (S3 gateway; not needed)
- GraphQL experimental module

Vendor strategy: define `FilterOp` + `QueryParam` locally in
`lance-graph-callcenter/src/query/filter.rs`. ~200 lines.

### Explicitly NOT ported

- GoTrue (Elixir Auth) → JWT verify via `jsonwebtoken` only
- Supabase Storage → not needed
- Edge Functions (Deno) → shader IS the compute
- Kong gateway → axum routing
- Dashboard UI → cockpit is the UI
- PostgreSQL wire protocol → omit; add `pgwire` compat only if n8n demands it

---

## § 6 — Dependency strategy

```
# Always compiled (zero-dep contract boundary):
lance-graph-contract  (path dep, no extra cost)

# Behind feature flags (never in default build):
[persist]   arrow = "57", lance = "2"
[query]     datafusion = "51", arrow = "57"
[realtime]  tokio = "1" (sync only), tokio-tungstenite = "0.24", serde, serde_json
[serve]     axum = "0.7", tower-http = "0.5"  (implies realtime + query)
[auth]      jsonwebtoken = "9", argon2 = "0.5"
[full]      all of the above

# NEVER add:
reqwest     — client HTTP; we are the server
postgres    — no Postgres in architecture
sqlx        — same
pg-embed    — same
```

---

## § 7 — Deliverables

| D-id | Title | Status |
|---|---|---|
| DM-0 | `ExternalMembrane` trait + `CommitFilter` in `lance-graph-contract` | Shipped |
| DM-1 | `lance-graph-callcenter` crate skeleton (Cargo.toml + feature gates + stub lib.rs) | Shipped |
| DM-2 | `LanceMembrane: ExternalMembrane` impl with `project()` + compile-time leak test | Queued |
| DM-3 | `CommitFilter` → DataFusion `Expr` translator (`[query]` feature) | Queued |
| DM-4 | `LanceVersionWatcher` — tails Lance version counter, emits Phoenix `postgres_changes` events (`[realtime]`) | Queued |
| DM-5 | `PhoenixServer` — minimal WS server, Phoenix channel subset, `postgres_changes` + `broadcast` (`[realtime]`) | Queued |
| DM-6 | `DrainTask` — `steering_intent` Lance read → `UnifiedStep` → `OrchestrationBridge::route()` | Queued |
| DM-7 | `JwtMiddleware` + `ActorContext` → `LogicalPlan` RLS rewriter (`[auth]`) | Queued |
| DM-8 | `PostgRestHandler` — query-string → DataFusion SQL → Lance scan → Arrow response (`[serve]`) | Queued |
| DM-9 | End-to-end test: shader fires → LanceMembrane::project() → Lance append → Phoenix subscriber receives event | Queued |

---

## § 8 — Stop-and-re-evaluate points

1. **After DM-2**: verify compile-time leak test passes. A `RecordBatch`
   column containing a `SemiringChoice` should not compile. If the test
   reveals a gap in the type system (e.g., newtype tricks), close the gap
   before proceeding to DM-3.

2. **After DM-5**: check shader throughput under realtime fanout load.
   Confirm `project()` on every commit does not affect hot-path latency
   (should be < 1 μs given it's just field extraction + Arrow row builder).

3. **Before DM-7**: confirm actual auth requirement from consumers
   (n8n, crewai-rust, openclaw). Do they need per-actor RLS, or is
   process-level auth sufficient for v1? Do not build RLS rewriter if
   single-actor usage is the real deployment model.

4. **Before DM-8**: confirm PostgREST compat is needed vs cockpit using
   Phoenix channel subscriptions only. The `serve` feature is the heaviest;
   defer if direct Lance reads via Rust API satisfy all consumers.

---

## § 9 — Unknowns (explicitly open)

- **UNKNOWN-1**: Does `cognitive-shader-driver`'s `ShaderSink` trait
  already serve as an `ExternalMembrane` façade, or is there an overlap?
  Inspect `crates/cognitive-shader-driver/src/` before wiring DM-2.

- **UNKNOWN-2**: Which consumers (n8n-rs / crewai-rust / openclaw) actually
  need a Phoenix/Supabase wire protocol vs direct Rust API calls?
  Verify before building the realtime server (DM-4/DM-5).

- **UNKNOWN-3**: Does n8n-rs need a Postgres wire (pgwire) connection or
  is the existing `OrchestrationBridge` path sufficient?

- **UNKNOWN-4**: What is the right `actor_id` type? The contract crate
  does not yet have an Actor concept. A plain `u64` hash suffices for v1
  but may conflict with future identity semantics. Mark explicitly.

- **UNKNOWN-5**: Lance dataset path / object-store URL configuration.
  Single env var `LANCE_URI` is proposed but not yet defined anywhere in
  the repo.

---

## § 10 — Markov XOR Gate + Blackboard Mediation Refinement (2026-04-22)

> **Status:** FINDING — replaces § 2's "dual Lance datasets" as the live mechanism.
> Lance persistence is still correct for durability; but the LIVE mediator is the
> existing A2A `Blackboard`, not two row-appended datasets.

### 10.1 — The blackboard is the explicit firewall

External events do NOT go directly into the BindSpace SoA or the Markov trajectory.
They land on `a2a_blackboard::Blackboard` first:

```
Consumer sends seed
  → BlackboardEntry { capability: ExpertCapability::ExternalSeed, ... }
  → A2A experts run N rounds (30–300 ns/round)
  → CollapseGate fires, project() → scalar commit
  → External channel receives event (30–300 ms later)
```

This is explicit, not conflated. The blackboard round boundary IS the anti-corruption
boundary. Internal experts never see raw external payloads — they see blackboard entries
with confidence scores, dissonance fields, and capability tags, same as any other expert.

### 10.2 — The Markov ±5 reuse (no new research)

The grammar parser uses Markov ±5 XOR braiding:

```
token at position d  →  XOR-superpose with ρ^d braiding into trajectory
```

The callcenter membrane uses the SAME mechanism across blackboard rounds:

```
Blackboard.round  =  trajectory position
ExternalSeed entry at round N  →  XOR-braided at ρ^d into the next context bundle
```

The `Blackboard.round: u32` counter already exists. The ±5 window means: when
computing the next context bundle, look at entries from `[round - 5, round]`.
No new data structure. No new research. Same ρ^d decay. Same XOR-bundle accumulation.

For fast agents (LLM round-trips ~10ms): `context_band = (-5, 0)` — 5 rounds back.
For human-in-the-loop: `context_band = (-500, 0)` — 500 rounds back via episodic.
The seed carries this as an explicit parameter; the blackboard honours it.

### 10.3 — Role taxonomy (`ExternalRole`, now in contract crate)

Every event crossing the gate is role-tagged before XOR-braiding:

| Direction | Role variant | Meaning |
|---|---|---|
| Inbound | `User` | direct human input |
| Inbound | `Consumer` | generic API caller |
| Inbound | `N8n` | n8n-rs workflow step |
| Inbound | `OpenClaw` | openclaw agent |
| Inbound | `CrewaiUser` | crewai-rust user role |
| Inbound | `CrewaiAgent` | crewai-rust agent role |
| Outbound | `Rag` | shader presenting as RAG retriever |
| Outbound | `Agent` | specific cognitive agent result |

Role binding at the gate: `RoleKey::bind(payload, role as u16)` — same slot
addressing as SUBJECT/PREDICATE/OBJECT in the grammar. "Who said this" becomes
a readable coordinate in the trajectory; unbinding recovers it.

### 10.4 — `ExternalEventKind` (now in contract crate)

| Kind | Behaviour |
|---|---|
| `Seed` | Deposits `BlackboardEntry(ExternalSeed)`. DrainTask picks it up, routes to `OrchestrationBridge`. |
| `Context` | Deposits `BlackboardEntry(ExternalContext)`. XOR'd into trajectory on next bundle pass; no active cycle triggered. |
| `Commit` | Projected scalar leaving the substrate toward an external subscriber. Never enters the blackboard. |

### 10.5 — Speed gap absorbed here, not elsewhere

The substrate runs continuously at 30–300 ns/op. The external consumer pulls at
30–300 ms. The substrate does NOT wait. CollapseGate fires on its own Markov schedule;
the current bundle is whatever is latest. External channels receive the projection
when the consumer is ready — pull-whenever, push-when-committed. The 10⁵–10⁷× gap
is structural, not a problem to solve. The gate just keeps bundling.

### 10.6 — Agent cards as A2A experts (one identity space)

The three identity spaces we had —
  (a) internal A2A experts (`ExpertId` + `ExpertCapability`),
  (b) external roles (`ExternalRole`),
  (c) YAML agent cards (`crewai-rust/*`, `.claude/agents/*.md`) —
collapse into one system by convention:

| Layer | Carries | Granularity |
|---|---|---|
| `ExternalRole` (at the gate) | Family — who invoked this | 8 variants |
| `ExpertId` (on the entry) | Specific card / expert | u16 (65k) |

**Convention:** for agent cards, `ExpertId = stable_hash_u16(card_yaml)`.
A `crewai-rust` agent or a `.claude/agents/family-codec-smith.md` card
both produce the same kind of `ExpertEntry` and post to the same blackboard
as any internal A2A expert. No distinction in the bus.

**Addressability** (corrected 2026-04-22 — see erratum below):

Role and card are SEPARATE typed metadata columns on `cognitive_event` rows
(`external_role: UInt8`, `expert_id: UInt16`). They are addressable independently
via the five query dialects on the metadata bus:

- *"All `Rag`-family cards"* — `WHERE external_role = 6`
- *"Just card `0x7F3A`"* — `WHERE expert_id = 0x7F3A`
- *"family-codec-smith as CrewaiAgent at round N"* — `WHERE external_role = 5 AND expert_id = 0x7F3A AND round = N`

Stack-side VSA binding of these identities happens through a deterministic
metadata→RoleKey slot mapping; the mapping never crosses the BBB.

> **Erratum (2026-04-22):** earlier drafts of this section proposed a packed
> 32-bit braid key `(role << 16) | expert_id` as a VSA slot address. That was
> wrong — it would have carved a parallel address space incompatible with the
> existing 4096 COCA / CAM-PQ / NARS-head vocabulary. Identity lives in
> metadata; VSA binding is a stack-side internal concern. See § 10.11 for the
> metadata address bus doctrine.

**Meta-awareness consequence:** `QualiaClassification` and `StyleModulation`
experts can fire on features like *"current context is RAG-heavy but
card-diverse"* or *"family-codec-smith resonance falling while palette-engineer
rising"*. The texture and resonance across both family AND card coordinates
become observables in the same SoA sweep — no new column, no new lookup,
just unbind at different mask depths.

**Registration:** an agent card YAML gets hashed at load time into an
`ExpertEntry`. The `capability` field is read from the card's declared
primary capability; `base_confidence` from its trust prior. After registration
the card participates in blackboard routing (`next_round_experts`) exactly like
any hand-coded expert.

### 10.7 — Consumers address roles: explicit OR implicit

External consumers (`crewai-rust`, `n8n-rs`, `openclaw`, …) don't just deposit
seeds — they can *address* specific roles through the same blackboard, two ways:

**Explicit routing** via `RoutingHint` on the seed:

| `target_role` | `target_card` | Router behaviour |
|---|---|---|
| `Some(r)` | `Some(c)` | Activate exactly card `c` in family `r` (full address). |
| `Some(r)` | `None` | Activate the best card in family `r` (family route; AriGraph-resonance tiebreak). |
| `None` | `Some(c)` | Activate card `c` regardless of family. |
| `None` | `None` | Implicit — AriGraph-resonance matching against seed payload. |

**Implicit routing** (no hint): the router matches the seed's context fingerprint
against each registered persona's AriGraph subgraph resonance. Top-k personas
activate for the next round. This is the same `next_round_experts` mechanism the
blackboard already runs — the new input is "persona resonance score" alongside
the existing `base_confidence × weight` term.

**Consequence:** a crewai-rust agent can say *"route this to the palette-engineer
card explicitly"* OR *"find whoever resonates with this codec question"* through
the exact same seed shape. The two modes are not separate APIs — they're the
presence/absence of the hint on the same entry.

### 10.8 — AriGraph integration: the persona IS its subgraph

Per CLAUDE.md's AGI-as-glove doctrine, AriGraph is thinking tissue — not a
service the struct queries, but part of its reasoning surface. Extending this:
**each persona's AriGraph subgraph IS its memory.**

A persona's full reasoning surface:

```
PersonaCard (in contract, identity only)
    ├── role: ExternalRole              — family at the gate
    └── entry: ExpertEntry              — id, capability, trust, weight
         │
         │  resolved at lance-graph boundary:
         ▼
    AriGraph::subgraph_for(entry.id)    — in crates/lance-graph/src/graph/arigraph/
         ├── committed SPO triples      — what this persona has "said" or "believed"
         ├── reversal history           — NARS-revised truth values
         ├── episodic trace             — ±5..±500 round window of participation
         └── resonance index            — for implicit routing matching
```

**AriGraph module layout (in lance-graph core, not contract):**
- `graph/arigraph/triplet_graph.rs` — already the SPO backbone
- `graph/arigraph/episodic.rs` — already the ±5..±500 window
- `graph/arigraph/retrieval.rs` — the resonance-match surface the router reads
- `graph/arigraph/orchestrator.rs` — already coordinates multi-expert cycles

**New wiring (future DM):** `AriGraph::subgraph_for(ExpertId) → PersonaSubgraph`,
a filtered view over the shared graph keyed by the persona's id. When the router
does implicit resonance matching, it iterates registered personas and calls
`persona_subgraph.resonance_against(seed_fingerprint)`. No new data structure,
no new research — just a view projection over the existing AriGraph.

**What this unlocks:** personas that LEARN. When card X handles a seed and the
resulting cycle commits, the SPO triple lands in X's subgraph. Next time a
semantically similar seed arrives, X's resonance is higher — it has memory.
This is the same Commit/Epiphany/FailureTicket loop from CLAUDE.md's P-1 Click,
now applied per-persona. Each agent card is its own cognitive loop on a
subgraph; the blackboard composes across them.

### 10.9 — The "Membrane · Role · Place · Translation" contract (iron rule from the prior stack)

Every piece of data crossing the gate MUST satisfy FOUR conditions. This is
older doctrine from the prior stack, now explicit and enforced here:

**1. Pass the membrane.** No direct writes to BindSpace SoA. The only
BindSpace-adjacent write site is `CollapseGate::merge()` with
`MergeMode::Bundle` (Markov-respecting) or `MergeMode::Xor` (single-writer
delta). A payload that bypasses the gate is rejected at the type system —
Arrow's column types cannot hold `Vsa10k`, `RoleKey`, `SemiringChoice`, etc.

**2. Get a role.** The payload is bound with a `RoleKey` keyed on
`ExternalRole × ExpertId`. No role = no binding = the payload cannot enter.
*Who said this* is a required field of entry, not metadata.

**3. Get a place.** The payload lands at a specific trajectory position
(blackboard `round`) and a specific VSA slot (role-indexed slice within
the `[u64; 157]` substrate). Every byte has coordinates. No roving.

**4. Translate into internal reasoning.** The external payload is not
*stored* as bytes — it is *transcribed* into the substrate's grammar:

| Event kind | Translation at the gate |
|---|---|
| `Seed` | `Vsa10k` trajectory token (RoleKey::bind + ρ^d braid into next round's bundle) |
| `Context` | XOR-superposed into the current context bundle (no new cycle) |
| `Commit` | Already internal; project-only path, no translation needed (it's leaving) |

The translation is **not reversible**. The substrate does not keep external
bytes "just in case". It only speaks its own language. If a consumer needs
their original bytes back, that's their persistence concern — the gate does
not cache pre-translation payloads.

**Compile-time enforcement points:**

| Contract rule | Where it's enforced |
|---|---|
| (1) Membrane | Arrow column types reject VSA/semiring (§ 3 BBB invariant) |
| (2) Role | `ExternalMembrane::ingest(Self::Intent) → UnifiedStep` — the impl must role-bind |
| (3) Place | `BlackboardEntry` always carries `expert_id`; `Blackboard.round: u32` |
| (4) Translation | `ingest()` return type is `UnifiedStep`, not a raw payload — translation is forced at the trait boundary |

**Consequence:** there is no "untagged external state" anywhere in the
system. Every piece of memory the substrate holds was, at some point in
its history, role-bound, place-addressed, and translated. If you cannot
answer *who said it, when, and in what VSA grammar form* for a piece of
data, it does not belong in the substrate.

This is the test to apply before merging any new code path that touches
the gate: can I name the role, the place, and the translation? If not,
the code is leaking external ontology inward — reject.

### 10.10 — VSA 10000-D: lossless role bind/unbind for the internal face

Roles, faculties, dialects, scents — every internal identity — are bound
into the `Vsa10k = [u64; 157]` substrate via `RoleKey::bind` using
role-indexed slices in the 10000-dim space. This is NOT compression — it
is *lossless*:

- `RoleKey::bind(payload, role_slot)` places the payload at a specific
  role-indexed slice; the other 10000 − slice_width bits are untouched.
- `RoleKey::unbind(trajectory, role_slot)` recovers the payload exactly
  from that slice. Bind / unbind are inverse operations at the bit level.
- XOR-superposition across roles remains lossless by construction: the
  Johnson-Lindenstrauss + concentration-of-measure bounds (I-SUBSTRATE-MARKOV)
  guarantee associativity in expectation; d = 10000 suppresses deviation
  at rate ~e⁻ᵈ. See EPIPHANIES.md E-SUBSTRATE-1.

**Which identities live here (internal face):**

| Identity | Slot region | Semantics |
|---|---|---|
| `ExternalRole` | reserved family-slot window | "who crossed the gate" |
| `FacultyRole` | reserved faculty-slot window | "which cognitive function is active" |
| `ExpertId` (card hash) | card-slot window | "which specific persona" |
| `dialect` | dialect-slot window | "which query language" |
| `scent` | scent-slot window | "which address neighborhood" |

Each gets its own reserved slot window in the 10k-dim space. No collisions
with the 4096 COCA vocabulary / CAM-PQ codebook / NARS-head region because
those live in their own allocated ranges; the identity slots are
non-overlapping by partition (zone map belongs in `encoding-ecosystem.md`
when we codify the full layout).

**Why lossless matters.** A cycle that commits with `FacultyRole::Reasoning`
+ `ExternalRole::CrewaiAgent` + `ExpertId::family-codec-smith` must be
exactly recoverable at any later round via unbind. If the role encoding were
lossy, replay / audit / reversal would drift. Lossless bind/unbind is what
makes the Markov ±5 trajectory a *substrate*, not a buffer.

**BBB role.** These bindings live exclusively on the internal face.
`RoleKey`, `Vsa10k`, and the slot assignments never cross the gate. The
external face sees the SAME identities as metadata columns (§ 10.11) —
deterministically mapped stack-side. Two faces, one identity, zero leak.

### 10.11 — The metadata address bus (queries ARE dispatch, external face)

The `cognitive_event` Arrow table (§ 4) is not a log. It is the **uniform
address bus** of the entire callcenter. Every row carries a typed identity
tuple `(external_role, faculty_role, expert_id, dialect, scent, …)`;
every query over this table returns a row set; every returned row's tuple
IS an execution address. Dispatch = (predicate, action); there is no
separate router.

**Five dialects, one bus:**

| Dialect | Strength | Example addressing |
|---|---|---|
| SQL (DataFusion) | tabular / range / aggregation | `WHERE external_role = 3 AND free_energy < 0.2` |
| Cypher / GQL | graph-path, persona neighborhoods | `MATCH (c:Card)-[:HANDLED]->(e:Event {faculty_role: 2})` |
| NARS | truth-weighted selection | `f > 0.7 AND c > 0.6` — route only to confident personas |
| Qualia | fuzzy family match | route by qualia signature, not exact ids |
| OrchestrationBridge | domain-coarse pre-scope | `StepDomain::Thinking` narrows the bus range |

Composes with shipped machinery:
- `RoutingHint` (persona.rs) — explicit case = degenerate query
  (`WHERE role = target_role AND card = target_card`).
- `Blackboard::by_capability()` — already a bus primitive.
- `CommitFilter` (external_membrane.rs) — a narrow predicate on the bus.
- `PlannerContract` — every dialect compiles through it to DataFusion.

**BBB role.** The bus is the EXTERNAL face of identity. Row values are
`UInt8` / `UInt16` / `Float32` — Arrow scalars that cross the gate safely.
The internal 10k-dim slot mapping (§ 10.10) of these same identities never
appears in any row, any query, any API.

### 10.12 — DN-addressed REST + polyglot dialects (ladybug-rs pattern)

The external wire surface is a single endpoint shape:

```
POST /tree/{ns}/heel/{h}/hip/{h}/branch/{b}/twig/{t}/leaf/{l}
Content-Type: application/{dialect}+text
Body:   <dialect expression + inline seed content>
```

**URL path = metadata predicate.** Deterministically parsed to:
```sql
WHERE tree = $ns AND heel = $h AND hip = $h
  AND branch = $b AND twig = $t AND leaf = $l
```
No per-endpoint routing logic — the parser is regular, the predicate is
regular, the dispatch is bus § 10.11.

**Starter dialects** (four already shipped, three added):

| Dialect | Status | How it lands in the stack |
|---|---|---|
| Cypher | shipped (`parser.rs`, 44 tests) | full parser → logical plan |
| GQL (ISO 39075) | shipped | full parser → logical plan |
| Gremlin | shipped | full parser → logical plan |
| SPARQL | shipped | full parser → logical plan |
| **NARS** | planned | full parser (`?x <--> animal. %{f>0.7;c>0.6}%`) — typed cognitive query |
| **Redis** | planned | **thin shape-adapter over DataFusion — NOT a new parser.** Redis commands map to DataFusion ops on a slightly-shaped row view. `GET tree:foo:leaf:bar` → `SELECT content WHERE tree=foo AND leaf=bar`. `HGET k f` → single-column `SELECT`. `ZRANGE k 0 n` → `ORDER BY score LIMIT`. No separate execution engine. |
| **Spark SQL / DataFrame** | planned | DataFusion handles most Spark SQL directly; DataFrame API maps to DataFusion's DataFrame API; structured streaming maps onto realtime channels |
| DataFusion SQL | shipped (baseline) | canonical power-user path |
| PostgREST filter params | planned (§ 5) | ergonomic REST query strings |

**Shared compile stack.** Every full-parser dialect → `LogicalPlan` IR →
DataFusion physical plan → metadata bus dispatch. Redis is the exception:
it is an ergonomic view over DataFusion, not a distinct IR branch.

**Dialect-as-signal.** The dialect itself is metadata: `dialect: UInt8`
on the cognitive_event row. Tells the router which cognitive faculty the
consumer is most likely exercising (Cypher → graph-path reasoning;
NARS → truth-aware inference; Redis → DN-KV pattern; Spark → bulk
analytic). Routing can pre-bias persona activation on dialect before the
body is parsed.

**BBB role.** The REST surface only ever serves / accepts Arrow-scalar
JSON and parsed query IR. It never sees `Vsa10k`, `RoleKey`, or any
internal type. Responses are projected commits (§ 4 schema rows).

### 10.13 — Address = scent = context-pull key

The DN path is not just a routing predicate. It compresses through the
existing codec stack into a **scent** — a 1-byte fingerprint that serves
four uses with one representation:

```
DN path (16Kbit)  →  ZeckBF17 (48B)  →  Base17 (34B)  →  CAM-PQ (6B)  →  Scent (1B, ρ=0.937)
```

(Same chain as `docs/CODEC_COMPRESSION_ATLAS.md`.)

**Four uses, one scent:**

| Use | Traditional key | Now |
|---|---|---|
| Route | routing key | scent |
| Retrieve | retrieval vector | scent |
| Similar | nearest-neighbor fingerprint | scent |
| Frame | cognitive prior | scent |

**Context pull.** On seed ingress, scent is computed first. It fires four
parallel pulls before the shader reads the body:
- AriGraph subgraph retrieval — triples near scent neighborhood
- Episodic memory ±5..±500 — past cycles with similar scent
- Persona trust lookup — cards whose AriGraph resonates with this scent
- Qualia classification — codebook cell signature for fuzzy family

The shader's F-descent cycle runs with the pulled context as prior; the
body content is interpreted AGAINST that context, not in isolation.

**Result.** Supabase-shape RAG returns document chunks matched by vector.
This returns reasoned intelligence contextualized by the full cognitive
substrate state at this scent. Same REST envelope, same JSON response
shape — response grounded in accumulated reasoning, not lexical similarity.
Next query at nearby scent benefits from this commit's SPO write-back
(AriGraph keyed on scent accumulates). Training-without-labels loop
(E-DEPLOY-1) deepens the cascade with use.

**BBB role.** `scent: UInt8` is a metadata column (safe to cross). Its
computation uses internal codec primitives (Base17, CAM-PQ, Hamming) but
those never leave the stack side. The scent byte is the surface artifact;
the internal compressive path is invisible to the consumer.

---

## § 11 — Sequencing (post-epiphany)

Given E-DEPLOY-1 and §§ 10.10–10.13, the DM sequence gains three phases
inside the existing plan:

| Phase | Deliverables |
|---|---|
| **A — BBB spine** | DM-2 (LanceMembrane impl) + compile-time leak test; § 4 schema with five identity/scent columns; metadata address bus wired; stack-side metadata ⇄ 10k slot mapping |
| **B — Polyglot front end** | extend `PolyglotDetect` with NARS + Spark full parsers; Redis shape-adapter over DataFusion (not a parser); DN-path URL parser → DataFusion `Expr` |
| **C — Scent cascade** | DN → ZeckBF17 → Base17 → CAM-PQ → Scent wiring; context-pull on seed ingress; persona AriGraph subgraph view keyed on scent |
| **D — Realtime + training** | DM-4 + DM-5 (Phoenix channels); commit-writeback trains personas; A2A consumer integration (crewai-rust, n8n-rs first) |

Each phase is independently testable and reversible. Litmus test
(§ 10.9 + E-DEPLOY-1 footer) applies at every merge.

---

## § 13 — "Git for Cognition" — the unified mental model

**Brainstorm origin:** 2026-04-23 — the observation that callcenter needs
git-like access to internal state. Not a metaphor: the shared algebraic
foundation is a commutative monoid on blobs. Git approximates it (and
patches around conflicts). VSA d=10000 saturating bundle IS it —
lossless, by CK construction (E-SUBSTRATE-1). Jirak bounds exactly
where the approximation becomes noise.

### Primitive mapping (one-to-one)

| Git primitive | Callcenter primitive | Existing machinery |
|---|---|---|
| `commit` | CollapseGate fire → Lance version N | `GateDecision::COMMIT` + `project()` |
| `branch` | Speculative blackboard round (not yet fired) | `Blackboard.round` before CollapseGate |
| `merge` | Bundle two trajectories | `MergeMode::Bundle` (CK-safe) |
| `rebase` | Replay trajectory against different NARS prior | Markov ±5 replay + `awareness.revise()` |
| `checkout HEAD~5` | Markov position −5 in the braid window | Braid offset, already exists |
| `checkout <version>` | Lance time-travel | `dataset.checkout(version=N)` |
| `diff V1..V2` | Projected RecordBatch row comparison | Two Lance versions, subtract |
| `blame` | `unbind(role)` at trajectory position → metadata column | VSA unbind, map slot → scalar |
| `cherry-pick` | Inject one BlackboardEntry into another round | Router decision |
| `stash` | Speculative bundle without Persist emit | `EmitMode::Bundle` (not `Persist`) |
| `pull` | `subscribe(filter)` | `ExternalMembrane::subscribe` |
| `push` | `ingest(ExternalIntent)` | `ExternalMembrane::ingest` |
| `log --max-count=500` | Markov ±500 episodic window query | Existing episodic memory |
| `tag` | Named persona checkpoint | `PersonaCard` + Lance tag |
| `.gitignore` | `CommitFilter` | Already in contract |
| `pre-commit hook` | NARS check before `EmitMode::Persist` | CollapseGate predicate |

### Why the callcenter is stronger than git

Git merge commutativity is approximate — conflicts require human
resolution. VSA d=10000 bundle commutativity is exact (concentration-of-
measure at rate ~e^(−d)). Git blobs are opaque bytes; blame is textual
search. Callcenter blobs are VSA bundles with semantic coordinates;
`blame` is `unbind(role)` — a structured algebraic query. Git rebase can
corrupt history; Markov replay is provably lossless.

**The scalar-vs-VSA distinction is the blob boundary:**
- Git: blob = file bytes (opaque to git itself)
- Callcenter: blob = VSA bundle (unbind-able by role, Cartan-character-
  indexed slots — see [FORMAL-SCAFFOLD])

### Chat rounds as commits (§§ 10–13 synthesis)

A chat turn is a commit. The blackboard round between turns is the
staging area. CollapseGate fire = `git commit`. Lance version bump =
the append to the object store. The ±5/±500 Markov window is `HEAD~5`
and `HEAD~500`. The 10⁵–10⁷× speed gap is absorbed exactly at the
turn boundary — substrate runs at 30 ns/bind internally; external
subscribers see one tick per committed turn.

**Jirak** (I-NOISE-FLOOR-JIRAK) bounds the turn-update density: too
sparse → consumer context diverges; too dense → weak-dependence breaks.
The ±5 window is the implicit rate limit.

**Cartan-Kuranishi** governs which columns get projected outward — not
arbitrary, but the intrinsic fiber geometry of the external signal.
`dialect: u8` and `scent: u8` are the Cartan-intrinsic columns;
their slot widths in the 10k substrate should not be chosen by
convention. [FORMAL-SCAFFOLD] revival candidate 3 (learned attention
masks) applies for empirical confirmation.

### Porcelain vs plumbing (BBB as the split)

| | Git | Callcenter |
|---|---|---|
| **Porcelain** | `git add`, `git commit`, `git log` | Supabase-shape REST, DN paths, JSON |
| **Plumbing** | `git hash-object`, `git cat-file` | VSA bind/unbind, Markov braid, AriGraph |
| **The line** | `git` CLI | `ExternalMembrane` trait |

Consumers use porcelain only. Internal faculties use plumbing.
`ExternalMembrane::ingest()` and `project()` are the translators.

### Consumer mental model (adoption surface)

**The pitch:** "git for thoughts."

- n8n-rs / crewai-rust author who knows git → zero onboarding friction
- Human via q2 → Neo4j browser ≈ `gitk` / `git log --graph`
- Curl consumer → `git show <refspec>` shape over REST

DN URL path (`/tree/ns/heel/h/hip/x/branch/b/twig/t/leaf/l`) IS a
refspec. PersonaCard is the author. FacultyDescriptor is the committer.
RoutingHint is the refspec target.

### Three primitives to name explicitly

1. **`Speculative`** — blackboard round that exists but has not yet
   fired CollapseGate. Equivalent to git's staged-but-not-committed.
   Already implicit; needs a first-class name in the callcenter API.
2. **`Rebase` verb** — replay Markov trajectory against a different
   NARS prior. Mechanics exist; needs an `ExternalMembrane` method or
   REST endpoint in Phase B.
3. **`Blame` projection** — `unbind(role)` internally, project to
   metadata-column answer externally. VSA slots never cross the BBB;
   the response is always `external_role: u8`, `faculty_role: u8`,
   `expert_id: u16`.

### BBB invariant holds through the git metaphor

None of the git-shaped primitives requires VSA types to cross the gate:
- `commit` result → Arrow scalars only
- `blame` result → metadata columns only
- `checkout` result → projected RecordBatch
- `branch` / `merge` → internal only (MergeMode::Bundle)

§ 10.9 iron rule (membrane → role → place → translate) holds at every
git verb.

---

## § 14 — Cold Storage = Git Cold Storage (Two Dataset Classes)

**Epiphany trigger:** "lance-graph/lancedb + S3 cold storage becomes a git cold storage."

### Two dataset classes

| Dataset class | Content | Crossing BBB? | Queryable? |
|---|---|---|---|
| **External / scalar** | Arrow RecordBatch rows from `project()` | Yes (IS the BBB output) | Yes — DataFusion, Supabase FDW, n8n subscribe |
| **Internal / VSA** | `Fingerprint<256>` = `[u64;256]` cycle fingerprints (L4/L5 speed tier, 2 KB/row); NARS truth vectors, braid offsets. L3 cold tier can promote to Vsa10k BF16 (20 KB, lossless) or RaBitQ-quantized Lance columns (zero-copy ANN). | No — never crosses BBB | Yes — DataFusion + VSA UDFs (see § 15) |

### Parallels with git cold storage

```
git object store  ≈ Lance + S3 (append-only, content-addressed by version)
git blob          ≈ internal VSA fingerprint bundle (opaque to external consumers)
git tree          ≈ Blackboard round snapshot (expert entries + round number)
git commit object ≈ external scalar RecordBatch row (the `project()` output)
git pack file     ≈ Lance fragment files (batched, compressed, indexed)
git remote        ≈ S3 bucket (cold tier, queryable in-place via DataFusion S3 scan)
```

### Why Lance is stronger than git cold storage

- Git blobs are opaque bytes, indexed only by SHA1. Lance fragments carry Arrow
  schema; every column is queryable in-place without extracting.
- Git time-travel is commit-hash lookup. Lance time-travel is `dataset.checkout(version=N)`,
  which DataFusion can scan directly across versions (temporal join).
- Git has no query engine. Lance has DataFusion — the internal VSA dataset IS
  a queryable database via the VSA UDFs in § 15.

### Training corpus path (E-DEPLOY-1)

Internal VSA dataset is pre-labeled by F outcome per epiphany E-DEPLOY-1:
- Each row: `{ fingerprint: [u64; 256], meta: MetaWord, f_outcome: f32, role: u8, style: u8 }`
- Rows where F < 0.2 are positive training examples (committed cycle = "good commit")
- Rows where F > 0.8 are negative examples (failed cycle = "bad commit")
- This dataset trains the ONNX persona classifier in § 17 with zero labeling effort

### Two-tier storage topology

```
Hot tier  (Lance in-process):
  external_dataset  — last N committed RecordBatch rows (Supabase-facing)
  internal_dataset  — last M cycle fingerprints (VSA UDF-facing)

Cold tier (S3):
  external_cold/   — full projection history (audit log, training data)
  internal_cold/   — full fingerprint history (ONNX training corpus)
```

`ExternalMembrane::subscribe()` wires to a `tokio::sync::watch` on the
Lance version counter of `external_dataset`. External consumers never
touch `internal_dataset`.

---

## § 15 — VSA Dispatch: role × thinking = persona (RoleDB)

**Core insight:** routing IS a VSA query. `unbind(target_role, trajectory)` returns
the overlap between the trajectory's role-indexed region and the query role —
locality-sensitive, not exact hash. This means dispatch is approximate-nearest-neighbor
over the role-key space, not an if/else over role enums.

### The product identity

```
persona ≡ ExternalRole × ThinkingStyle
```

- `ExternalRole` (8 variants) — the "who" coordinate
- `ThinkingStyle` (36 variants) — the "how" coordinate
- Product space: 8 × 36 = **288 canonical personas**
- `PersonaCard` IS this product pair: `(role: ExternalRole, style: ThinkingStyle)`
- AriGraph subgraph keyed on `(ExternalRole, ThinkingStyle)` — one subgraph per persona

### VSA dispatch mechanics

```
route(step):
  role_hint   = step.thinking.map(|ctx| ctx.style) or RoutingHint.target_role
  trajectory  = current ShaderBus.cycle_fingerprint
  overlap     = unbind(role_key[role_hint], trajectory)   // VSA inner product
  best_match  = argmax(overlap over all registered personas)
  → dispatch to best_match's FacultyDescriptor
```

This replaces exact-match routing in `OrchestrationBridge::route()` with VSA
locality-sensitive routing. A `CrewaiUser` step with `Reasoning` style routes to
the persona `(CrewaiUser, Reasoning)` — the subgraph whose VSA fingerprint has the
highest overlap with the current trajectory state.

### RoleDB — DataFusion + VSA UDFs

Five UDFs registered in DataFusion that make the internal VSA dataset queryable
as a "DuckDB over roles":

| UDF | Signature | Purpose |
|---|---|---|
| `unbind(role, trajectory)` | `(u8, [u64;256]) → f32` | Role overlap (dispatch score) |
| `bundle(cols...)` | `([u64;256]...) → [u64;256]` | Bundle multiple fingerprints |
| `hamming_dist(a, b)` | `([u64;256], [u64;256]) → u32` | Fingerprint distance |
| `braid_at(pos, traj)` | `(i32, [u64;256]) → [u64;256]` | Markov position lookup |
| `top_k(bundle, k)` | `([u64;256], u32) → [u16]` | Top-K persona candidates |

SQL example:
```sql
SELECT expert_id, unbind(2, fingerprint) AS n8n_overlap
FROM internal_dataset
WHERE round = (SELECT max(round) FROM internal_dataset)
ORDER BY n8n_overlap DESC
LIMIT 5;
```

This is "DuckDB emulation based on roles" — DataFusion with VSA semantics
replacing exact predicate matching.

### n8n-rs + Supabase Realtime wiring

```
Lance external_dataset
  → PostgreSQL FDW (read-only view over Lance S3 parquet)
  → Supabase Realtime
  → n8n-rs WebSocket subscription (filter: external_role = N8n)
```

No polling. No webhook. Pure subscribe-on-row-insert. Each `project()` call that
produces a row with `external_role = N8n` notifies n8n-rs WebSocket subscriber
within one Lance version tick.

---

## § 16 — Persona as Function: 32 Atoms × 16 Weightings + YAML Runbooks

**Three layers of persona representation (not three different definitions — one identity, three representations):**

### Layer 1 — Identity (product type, 9 bits)
```rust
struct PersonaId {
    role:  ExternalRole,   // 3 bits (8 variants)
    style: ThinkingStyle,  // 6 bits (36 variants → 64 slots)
}
// 288 canonical personas; AriGraph keyed on this pair
```

### Layer 2 — Signature (56 bits, compressed PersonaHub)
```rust
struct PersonaSignature {
    atom_bitset:   u32,  // which of 32 named cognitive atoms are active
    palette_weight: u8,  // 16 weight levels packed as 4-bit × 2 atoms (or 8-bit coarse)
    template_id:   u16,  // which YAML runbook template handles this signature
}
// Total: 7 bytes = 56 bits per persona signature
```

The 32 named cognitive atoms (semantic operations, not styles):
```
deduction, induction, abduction, analogy, counterfactual,
causal, temporal, spatial, modal, deontic,
metaphor, narrative, hypothesis, contradiction, revision,
retrieval, synthesis, compression, expansion, clarification,
empathy, perspective, intention, belief, desire,
uncertainty, confidence, negation, quantification, comparison,
classification, decomposition
```

**Addressing space:**
- Each of 32 atoms takes one of 16 weight levels (0–15)
- Configurations: 16^32 ≈ 3.4×10^38
- PersonaHub's 370M personas are samples in this space
- 56-bit signature reduces 370M rows to a lookup table + 370M×7 bytes ≈ 2.6 GB flat file

### Layer 3 — Runbook (YAML template, macro scaffolding)

YAML runbooks are NOT persona identity. They are behavioral scripts for
specific context-loop shapes — multi-turn recovery, escalation, handoff:

```yaml
# template_0042.yaml — deduction + counterfactual heavy, reasoning style
name: deep_deduction_loop
atoms_required: [deduction, counterfactual, hypothesis, contradiction]
min_weight: 8
loop:
  - step: establish_premises     # braid_at(-1, trajectory) → premise bundle
  - step: generate_counterfactual
  - step: test_contradiction
  - step: revise_if_dissonance   # awareness.revise() if F > 0.4
  - step: commit_if_stable       # CollapseGate if F < 0.2
fallback: escalate_to_llm
```

**Why YAML and not inline prompts:**
- Deterministic, versioned, Git-trackable (can `git diff` across persona generations)
- Composable — templates can `include` sub-templates (context-loop macros)
- More precise than prompt templates for multi-step reasoning scaffolding
- Separate from the persona's identity and separate from per-cycle VSA reasoning
- JIT-compilable: `JitCompiler::compile(template_id)` → `KernelHandle` at dispatch time

### Storage arithmetic
```
PersonaHub 370M rows:
  Original:    370M × ~2KB/row = ~740 GB
  Signatures:  370M × 7 bytes  =  2.6 GB  (285× reduction)
  + template library: 10K × ~200 bytes YAML = 2 MB
  Total: ~2.6 GB vs ~740 GB  (46,000× total reduction)
```

---

## § 17 — The Four-Way Multiply + Style Oracle (ONNX@L4/L5)

### The four multiplicands

```
Persona × Style × Stage × Learned-dynamics = cognitive configuration space
```

| Axis | Cardinality | Representation | Implementation |
|---|---|---|---|
| **Persona** | 288 (8 × 36) | `PersonaId: (ExternalRole, ThinkingStyle)` | Identity type, AriGraph keyed |
| **Style** | 36 ThinkingStyles | `ThinkingStyle` enum + `FieldModulation(7D)` | Already in contract |
| **Stage** | 2 (rationale/answer) | `rationale_phase: bool` in `CognitiveEventRow` | MM-CoT stage split = FacultyDescriptor.inbound_style/outbound_style asymmetry |
| **Learned-dynamics** | continuous | `style_oracle: &OnnxPersonaClassifier` | ONNX classifier at L4/L5 |

Total configurations: 288 × 36 × 2 × (oracle prediction space) ≈ **20,736 × oracle**

F-descent = automatic architecture search over this space. Misaligned configurations
are dropped by the CollapseGate predicate; no explicit selection needed.

### ONNX persona classifier (replaces Chronos proposal)

**Why ONNX over Chronos:**

| Criterion | Chronos | ONNX classifier |
|---|---|---|
| Output | 1D scalar (style ordinal) | 288 logits (full persona product) |
| Task type | Time-series forecasting | Classification |
| Training | Pre-trained (zero shots) | Trains from Lance E-DEPLOY-1 corpus |
| Precision | Style axis only | Full `(role, style)` product axis |
| Infra | Separate model download | `ort` crate — already justified by Jina v5 |
| Fit | Partial fit (style only) | Full fit (role × thinking = persona) |

**Architecture:**
```
Input:
  recent_fingerprints: Tensor[N, 16384]   // N cycle fingerprints [u64;256] as f32 bits (L4/L5 speed tier)
  current_meta: Tensor[4]                  // MetaWord fields unpacked

Hidden:
  Linear(N×16384 → 512) + ReLU
  Linear(512 → 288)

Output:
  logits: Tensor[288]  // log-softmax over all (ExternalRole, ThinkingStyle) pairs
  → argmax → PersonaId { role, style }
```

**Integration into Think struct:**
```rust
struct Think {
    trajectory:     Vsa10k,
    awareness:      ParamTruths,
    free_energy:    FreeEnergy,
    resolution:     Resolution,
    episodic:       &EpisodicMemory,
    graph:          &TripletGraph,
    global_context: &Vsa10k,
    codec:          &CamPqCodec,
    // ── § 17 addition ──
    style_oracle:   Option<&OnnxPersonaClassifier>,  // None = StyleSelector::Auto fallback
}
```

`StyleSelector::Auto` remains the fallback when no oracle is loaded (cold start,
no training corpus yet). ONNX oracle augments, does not replace, the static
qualia→style rule.

**Training pipeline:**
1. Lance internal_cold dataset accumulates: `{ fingerprint[u64;256], meta: MetaWord, f_outcome: f32, role: u8, style: u8 }`
2. Rows labeled by F: `f_outcome < 0.2` → committed persona, `f_outcome > 0.8` → failed persona
3. Export to ONNX-format Parquet tensors
4. Train small classifier (< 2MB ONNX, fits in process memory)
5. Hot-reload via `ort::Session::new()` at version boundary

**Layer placement:** L4/L5 (internal, before CollapseGate). Never exposed externally.
ONNX model file is internal asset, not an external API concern.

### MM-CoT stage split (zero new architecture)

The two-stage CoT structure (rationale → answer) from MM-CoT maps exactly to
`FacultyDescriptor.inbound_style` / `outbound_style` asymmetry:

```
inbound_style  = ThinkingStyle for rationale generation (Stage 1, "thinking")
outbound_style = ThinkingStyle for answer emission     (Stage 2, "answer")
is_asymmetric() returns true iff these differ — exactly the MM-CoT condition
```

`CognitiveEventRow` gains one column:
```rust
pub rationale_phase: bool,  // true = Stage 1 (rationale), false = Stage 2 (answer)
```

No new trait, no new struct. The stage is already intrinsic to `FacultyDescriptor`;
`rationale_phase` surfaces it in the projected scalar row for external subscribers.

### PersonaHub compression summary

```
370M personas → extract:
  atom_bitset:   u32   (which of 32 named atoms are active)
  palette_weight: u8   (16 weight levels, coarse encoding)
  template_id:   u16   (which YAML runbook handles this signature)
               = 56 bits = 7 bytes per persona

Offline extraction:
  For each PersonaHub row → parse YAML → map to atom presence → pack 56-bit signature
  Output: 2.6 GB flat binary + 10K YAML templates (~2 MB)
  Hash-deduplicate → ~1–5M unique signatures in practice (PersonaHub has high redundancy)

Runtime lookup:
  PersonaId → signature (56-bit lookup table, negligible memory)
  signature.template_id → JIT compile YAML runbook → KernelHandle
  PersonaId + KernelHandle → cognitive configuration for this turn
```

---

## § 18 — VSA Precision Tiers + Generational Compression (Father-Grandfather)

**Trigger:** 2026-04-23 — clarification that L4/L5 is the speed lane and must stay
at fingerprint resolution; L3 is where precision lives; VSA is the wire format.

### Three precision tiers

| Tier | Format | Size | Properties | Where |
|---|---|---|---|---|
| **Fingerprint** | `[u64;256]` = `Fingerprint<256>` | 2 KB (16 Kbit) | Hash-quantized. One-way (no unbind). SIMD-fast Hamming. | L4/L5 substrate (30 ns/bind) |
| **Vsa10k BF16** | `[bf16; 10000]` | 20 KB | Effectively lossless (Jirak noise floor not crossed). Supports `unbind(role)`. | L3 memory / cold storage |
| **Vsa10k f32** | `[f32; 10000]` | 40 KB | Fully lossless. Full bind/unbind algebra. | Offline training / precision UDFs |

**L4/L5 is the speed lane (motion, learning, fast dispatch).** The 16Kbit
fingerprint is the right format there — shrinking it would leave the speed
lane; inflating it to Vsa10k would blow the L3 memory budget at 30 ns/bind.
This boundary is a hardware-budget invariant, not a design choice.

**L3 is where precision is affordable.** The callcenter operates at human-turn
rate (seconds between commits), not substrate rate. Full Vsa10k BF16 at L3
costs 20 KB/row — trivial at that cadence. Alternatively: **RaBitQ**-quantized
columns in Lance provide zero-copy ANN search at < Vsa10k RAM cost while
preserving approximate unbind accuracy.

### VSA as the wire format

VSA IS the wire format — the medium through which cognitive content travels
between layers:

```
Wire format:   bundle(SPO triples) → trajectory [Vsa10k or Fingerprint<256>]
               Markov ±5 window  → bundle last 5 trajectories → episodic context
               Markov ±500 window → bundle last 500 cycles → long-range context
               SPO role bind     → bind(role_key, payload) → superposed into trajectory
```

`CognitiveEventRow` is NOT the wire format — it is the BBB-safe SCALAR
projection of the wire state. The wire (VSA trajectory) stays inside. The
projection (Arrow scalars) crosses the gate.

### Father-Grandfather generational compression

**Motivation:** Per-cycle fingerprints accumulate at 2 KB/cycle. 10K cycles
= 20 MB (manageable in hot tier). 10M cycles = 20 GB (cold storage, still
feasible). But for full-precision Vsa10k cold storage at 20 KB/cycle,
generational compression prevents unbounded RAM growth during replay.

**Hierarchy:**

```
"Son" (hot tier, L4/L5):
  Per-cycle Fingerprint<256> rows — 2 KB each — fast, hash-quantized
  Lance version-tagged; last M rows in hot dataset

"Father" (~100-cycle bundle, L3):
  MergeMode::Bundle over last ~100 per-cycle fingerprints
  → single Vsa10k BF16 (20 KB) representing 100 cycles
  → CK-safe (I-SUBSTRATE-MARKOV); saturating bundle preserves Markov property
  → 100 × 2 KB → 20 KB: 10:1 compression; unbind survives at full L3 precision

"Grandfather" (~1000-cycle bundle, cold tier):
  MergeMode::Bundle over last ~1000 cycles (or over 10 Father vectors)
  → single Vsa10k BF16 (20 KB) representing 1000 cycles
  → 1000 × 2 KB → 20 KB: 100:1 compression; Markov property preserved
  → unbind(role, grandfather) = approximate role trajectory over 1000 turns
```

**Relationship to existing Markov windows:**
- ±5 window → per-cycle fingerprint braid (existing, L4/L5)
- ±500 window → episodic memory (existing, L3)
- ±1000 window → grandfather bundle (new, L3/cold storage)

**CK safety proof (informal):** `MergeMode::Bundle` is commutative and
associative in expectation (Johnson-Lindenstrauss + concentration-of-measure
at rate ~e^(−d), I-SUBSTRATE-MARKOV). Bundling 100 fingerprints into one
Vsa10k is a saturating bundle; the Chapman-Kolmogorov property holds for the
bundle as a whole — the state transition from "Son" to "Father" is a valid
Markov step.

**Implementation note:** deferred. Current hot dataset stores fingerprints
only. Father/Grandfather columns are Phase B additions when cold dataset
accumulates > 10K cycles. No schema change to `CognitiveEventRow` required —
generational bundles live in a separate Lance dataset (compression tier).
