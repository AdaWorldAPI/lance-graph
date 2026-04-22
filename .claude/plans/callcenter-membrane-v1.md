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

No Vsa10k. No semiring. No RoleKey. These are numbers and bytes.

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

**Combined braid key at the gate:** `(role as u16) << 16 | expert_id` — 32 bits.
The shader can unbind along either coordinate:

- *"All `Rag`-family cards collectively"* — unbind with role mask, drop expert_id
- *"Just card `0x7F3A`"* — unbind with expert_id, drop role
- *"family-codec-smith speaking as CrewaiAgent at round N"* — full 32-bit key

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
