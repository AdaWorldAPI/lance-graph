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
