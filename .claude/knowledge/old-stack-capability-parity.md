# substrate-b ↔ OLD HIRO/Bardioc stack capability correspondence

**READ BY:** integration-lead, truth-architect, anyone wiring a lance-graph + ractor + surrealdb stack as a replacement for a HIRO/Bardioc-shaped OLD stack (BEAM/OTP + JVM + Cassandra + Titan/JanusGraph + Gremlin + Elasticsearch + ClickHouse + Kafka).

**Status:** preventive documentation + capability ground-truth. Captures the **confirmed** correspondences (from Almato's own published OSS manifest at `bitbucket.org/almatoag/opensource` + the OGIT PR harvest + the Almato r7.1 Product Description) between lance-graph-substrate-b capabilities and the OLD HIRO/Bardioc stack they replace.

**Companion to:** `.claude/knowledge/lab-vs-canonical-surface.md` (the canonical surface rule), `.claude/knowledge/hollow-wire-failure-modes.md` (the failure mode when the canonical wire is incomplete), `.claude/knowledge/encoding-ecosystem.md` (the full codec ledger).

---

## 1. Why this doc exists

Anyone integrating lance-graph + ractor + surrealdb (the substrate-b shape, per `lance-graph-callcenter`'s ExternalMembrane + LanceVersionWatcher + Timeline view) as a replacement for a HIRO-like OLD stack hits the same question: *which lance-graph primitive replaces which OLD component, at what fidelity, with what version-substitution rationale?*

This doc answers it once, with **confirmed sources only** (Almato's own manifest pins the OLD versions; the OGIT PR harvest pins the OLD data model; the r7.1 Product Description pins the OLD architectural shape). No conjecture — every mapping is grounded.

## 2. The OLD-stack shape — Almato r7.1 9-component "OS for data"

Per the Almato r7.1 Bardioc Product Description (Oct 2025), Bardioc IS an *"operating system (OS) for data"* with nine functional components (§2.3-§2.4):

1. **Access Manager** — IAM (OIDC via Zitadel/Go) + Security Mesh (4-axis bit-op per-element auth: Subject(n) × Object(n) × Implicit(4) × Explicit(x), uncached) + AES256-at-rest + signed immutable audit
2. **Data Manager** — Scheduler + Memory Manager (tiered, COW, dedup, speculative pre-compute)
3. **Message Bus** — Apache Kafka 0.8.2.2 + ZooKeeper 3.4.6 (Producer/Consumer/Streams/Connect/Admin; Chain-of-Responsibility per-app-version topics)
4. **Knowledge Core** — six storage backends: Graph DB (Titan 0.4.4 → JanusGraph) + NoSQL (Cassandra 2.1.8) + TSDB (InfluxDB via exometer) + BLOB (S3) + KV (Cassandra) + Indexing (Elasticsearch 1.7 + Lucene 4.10)
5. **Knowledge APIs** — REST + WebSocket (Jetty 9.2) + Program API (.jar to JG node) + Streaming + Connector SDK; query langs OGP + Gremlin 2.4 + GraphQL
6. **Desktop** — Frontend SDK + Profile Manager + Graph Explorer + Ontology Manager
7. **Graph Applications** — Gremlin algorithm library (centrality/PageRank/shortest-path/tree + Spark-on-YARN OLAP)
8. **OGIT Ontology** — 5-ring meta-model (SGO → NTO → SNRA → SNBA → SNFA)
9. **OGIT Ontology Extensions** — domain-specific extensions (BGFS, Auth, Documents, Automation, Knowledge, Tickets, OSINT, Forms, Rating, Security, MARS-survives-within-OGIT)

## 3. The substrate-b correspondence

| # | OLD component (confirmed version) | substrate-b replacement | Fidelity |
|---|---|---|---|
| 1 | Zitadel (Go) OIDC/OAuth2 | **retained** + `auth-plug` in-proc JWT validation (substrate-b never replaces the IdP) | shape-exact |
| 2 | Security Mesh 4-axis bit-ops | per-vertex `_effectiveReaders` palette256/Binary16K bitmap + Hamming popcount | shape-exact |
| 3 | AES256-at-rest + signed audit log | Lance append-only version log (immutable by construction) | shape-exact |
| 4 | Data Manager scheduler + memory tiers | Lance version log + lance-graph-planner `AutocompleteCache` (cache/p64) | functional |
| 5 | **Apache Kafka 0.8.2.2 + ZooKeeper 3.4.6** | **`LanceVersionWatcher`** (in-proc, std::sync Condvar, `CognitiveEventRow` payload, BBB-invariant) + cluster bus for cross-instance | shape-exact in-proc |
| 6 | Titan 0.4.4 (Cassandra/ES/HBase) → JanusGraph | **lance-graph** (Lance columnar storage + indices) | functional |
| 7 | Gremlin 2.4 (Groovy) | **`lance-graph-planner`** (16 strategies) + future SurrealQL → lance-graph compiler | functional |
| 8 | Cassandra 2.1.8 (KV + NoSQL) | TiKV / surrealdb kv-lance (PR #35/#36 surrealdb merged) | functional |
| 9 | Elasticsearch 1.7 + Lucene 4.10 | Tantivy (Rust-native full-text) | functional |
| 10 | ClickHouse + Hadoop + Spark/YARN (OLAP) | DataFusion over Lance versions | functional |
| 11 | InfluxDB (TSDB via exometer_influxdb) | **Lance versions as time-series** (no separate TSDB needed) | shape-exact |
| 12 | BLOB store (S3 client) | Lance fragment storage OR external object store + Lance pointers | functional |
| 13 | swarm + libcluster + libring + locker (distributed actors) | ractor + mailbox-as-owner rotating topology + optional gRPC transport | functional |
| 14 | sbroker + pobox (backpressure) | ractor bounded mailbox + `MessagingErr::Saturated` (PR `AdaWorldAPI/ractor#1` merged) | shape-exact |
| 15 | lru_cache + con_cache (ETS hot cache) | `dn_redis` (Redis-protocol over Lance via DataFusion + OGIT class views; FalkorDB/KuzuDB shape — talk Redis without being Redis) | functional |
| 16 | rafted_value (gen_statem Raft) | openraft (Rust) | functional |
| 17 | **gen_statem** (the OTP state-machine substrate) | **ractor Actor + Lance-version-as-state-machine** (the Rubicon phase machine; see §4) | shape-exact |
| 18 | expr 0.1.0 (Elixir math rule evaluator) | `lance-graph-planner::thinking` (12 styles + NARS dispatch + sigma chain) | functional |
| 19 | Jetty WebSocket + websocket_client | substrate-b WebSocket endpoint (Layer-3 outbound, tokio per the I-2 invariant) | functional |
| 20 | Jena (Java OGIT runtime validation) | OGIT compile-time check (Rust types encode SNRA required-attrs) — **stronger than OLD** | improvement |

## 4. The three load-bearing structural findings

### 4.1 Three OLD components collapse to one NEW primitive

OLD **Historisation** (JG versioning + temporal query) + OLD **TSDB** (InfluxDB via exometer_influxdb) + OLD **signed audit log** all become a **single substrate-b primitive**: Lance versions.

- `dataset.checkout_version(V_ref)` = point-in-time query (Historisation)
- The version log itself = time-series (TSDB)
- Append-only immutability = signed audit log

This is the substrate-b efficiency win — three OLD components, one NEW primitive. It is NOT a slogan; it is the architectural consequence of choosing an immutable append-only storage as the foundation.

### 4.2 Security Mesh bit-ops = palette256 + Hamming popcount (shape-exact)

The OLD-stack Security Mesh Layer (r7.1 §3.1.3) uses *"extremely efficient bit operations to verify authorisation for individual data access ... always executed for each data element without impacting access performance. As authorisation changes are not cached, they take effect immediately in the graph."*

The substrate-b hot-path primitive is bit-op-per-element via palette256 + Hamming popcount on `Binary16K = [u64; 256]`. Same algorithmic shape; same per-element guarantee; same uncached / immediate-effect property. The materialised `_effectiveReaders` / `_effectiveDevices` (per BGFS PR `almatoai/OGIT#773` platform-reserved attributes) are computed via these primitives.

### 4.3 gen_statem is the confirmed OLD-stack precedent for the Rubicon model

`ericentin/gen_state_machine 2.0.1` is in the HIRO manifest. It is the OTP state-machine substrate; `rafted_value`'s Raft consensus runs on it (follower / candidate / leader = `:gen_statem` states). Its feature set IS the Rubicon model:

| gen_statem feature | Rubicon analogue (substrate-b) |
|---|---|
| `state_functions` callback mode | ractor ClassActor: one `handle` arm per epistemic state (Contemporary / Anachronistic / Spoiler — see `STANDING_WAVE_ARCHITECTURE.md` §1.5 in any substrate-b consumer) |
| `handle_event_function` mode | typed-enum match in `Actor::handle` |
| `postpone` (defer until next state change) | Rubicon: hold events arriving before the Decision/Rubicon commit; replay after the phase transition |
| `state_enter` callbacks | Rubicon: fire the Lance commit on entering the Decision state |
| per-state / per-event / generic timeouts | SLA-bounded cold-path waits (ractor + `MessagingErr::Saturated`) |
| internal (self-generated) events | cognitive cycle self-dispatch (encode → decode → F-check → persist) |

**The Rubicon model IS a gen_statem.** It is NOT an invention; it is the faithful translation of an OTP behavior HIRO already runs in production (rafted_value + the lifecycle machines). This matters for any consumer mapping the OLD-stack thinking layer onto substrate-b: the Rubicon shape is the confirmed precedent, not a hypothesis.

## 5. The OLD data model — exact OGIT shapes (PR harvest)

The OLD stack carries 10 production workloads (confirmed by walking ~493 OGIT closed PRs, 2017-2026). Any substrate-b consumer replacing HIRO needs to handle all of them:

- **BGFS** (Bardioc Graph File System, PR #773 merged 2026-06-01) — 5 entities (File, Folder, Symlink, ShareLink, AppHandler) + 2 verbs (`refersTo`, `sharedVia`) + 17 attributes incl. capability fixed-set `{read, write, share, admin, delete}`; platform-reserved `_effectiveReaders` / `_effectiveDevices`; 16-hop symlink resolution
- **Auth/Device** (PR #775 open) — `Device` (deviceType ∈ {mobile, desktop, web, hsm, hardware-wallet}; securityLevel ∈ {0, 50, 80, 100}) + `usesDevice` + `hasVault` (Person-anchored, survives Account deactivation) + `isWorkspace` (Team marker)
- **Three-layer identity** (established 2018 in PRs #362/#376/#403) — Person (persistent) / Account (ephemeral, job-scoped) / DataScope (access perimeter; **mandatory-named** since 2018-11)
- **Documents** — `DocumentInfoRecord` with blob/status/creationTime/typ
- **Automation** — `AutomationIssue` with `savedTimeSeconds` (RPA ROI metric)
- **Knowledge Items** — with `manualProcessingTimeSeconds` (human cost metric)
- **Tickets** — `Ticket`/`ChangeRequest`/`ConfigurationItem` with `assignedTo`/`isPartOf`/`precedes`/`affects`/`subType`
- **OSINT** — Person/Organization/Address/Position/SecurityIncident relational graph + `locatedIn`
- **Org lifecycle** — Organization with `endedAt`/`precedes`
- **Trust** — Rating
- **Forms** — Form → Attachment connection
- **MARS** — survives within OGIT (`ogit.Automation:MARSNode` + `ogit/MARS/Resource` + `mars2ogit`); NOT deprecated, refactored

## 6. Boundary collapse (the migration value)

The OLD-stack request path traverses **8 boundaries**:

1. Operator Query → BEAM (HTTP)
2. BEAM → JVM (HTTP, graph_conn → Java OGIT)
3. JVM → C++ (JNI, Java → Cassandra/Lucene)
4. Gremlin → Storage (TinkerPop → Titan backend)
5. App → DB (CQL serialize)
6. DB → Analytics (Cassandra → ClickHouse/Spark)
7. **Auth boundary** (Zitadel JWT validation)
8. **Message Bus produce/consume** (Kafka — every event)

Plus **4 concurrency models**: BEAM (Phoenix + swarm + sbroker), JVM (Titan + Gremlin + ES), Go (Zitadel — retained), C++ (Cassandra + Lucene).

The substrate-b shape collapses to: **0 in-binary application boundaries** + 1 retained external Go IAM (Zitadel) + 1 concurrency model (ractor) + the irreducible Raft consensus tax (honest — see `ROADMAP_RUST_PRIMARY_HEADSTONE.md` in any substrate-b consumer for the per-workload tax accounting).

The 8 → 0 boundary collapse is the migration's measured value, not a slogan; the §14 acceptance gate proves it per-endpoint via dual-stack workload replay.

## 7. Process rule for substrate-b consumers

Before proposing a new lance-graph integration trait / contract / coordination surface to replace an OLD-stack capability:

1. Check this doc — does the §3 correspondence already name the substrate-b primitive?
2. Check `.claude/knowledge/encoding-ecosystem.md` — does an existing codec serve the capability?
3. Check `.claude/knowledge/lab-vs-canonical-surface.md` — is the canonical bridge the right place?
4. Check `.claude/knowledge/hollow-wire-failure-modes.md` — am I about to wire the lab surface without plugging the canonical bridge?

If steps 1-4 don't dissolve the proposal, then a new structure is warranted. Otherwise, the OLD-stack correspondence here already names what to wire against.

---

## Cross-references

### Source documents (confirmed ground truth)
- **Almato r7.1 Product Description** (Oct 2025, 33 pages) — the 9-component OS architecture (`BARDIOC_PRODUCT_CANONICAL.md` in substrate-b consumers)
- **Almato OSS dependency manifest** (`bitbucket.org/almatoag/opensource`, project KG) — the confirmed component versions (`BARDIOC_DEPENDENCY_MANIFEST_CONFIRMED.md`)
- **OGIT closed PR harvest** (~493 PRs 2017-2026 from `github.com/almatoai/OGIT`) — the exact data model shapes (`BARDIOC_HIRO_CONTEXT_HARVEST.md` + `TARGET_STACK_REFERENCE.md`)
- **substrate-b architecture doc** — `STANDING_WAVE_ARCHITECTURE.md` (awareness-in-Lance-versioning + §1.6 OGAR Sprint 7 muscle memory)

### Companion lance-graph knowledge docs
- `.claude/knowledge/encoding-ecosystem.md` — the full codec ledger (MANDATORY read before any codec work)
- `.claude/knowledge/lab-vs-canonical-surface.md` — the canonical surface rule (MANDATORY read before any REST/gRPC/Wire DTO work)
- `.claude/knowledge/hollow-wire-failure-modes.md` — DRAFT-INERT / sealed-but-shadowed / feature-flag mismatch (companion to lab-vs-canonical-surface)
- `.claude/knowledge/vsa-switchboard-architecture.md` — the three-layer carrier (Switchboard / Domain role catalogues / Content stores)

### Upstream PRs from this correspondence
- `AdaWorldAPI/lance-graph#452` (Lance-append-Raft dovetail), `#453` (cluster asymmetry), `#454` (post-merge corrections), `#455` (dn_redis as key-shape protocol), `#456`/`#457`/`#458` (Pearl junctions classifier + post-merge corrections) — all MERGED
- `AdaWorldAPI/lance-graph#464` (hollow-wire failure modes catalogue) — OPEN
- `AdaWorldAPI/surrealdb#35` (kv-lance SDK feature), `#36` (Lance backend struct + endpoint helper) — both MERGED
- `AdaWorldAPI/ractor#1` (`MessagingErr::Saturated` variant + `From<TrySendError>` split) — MERGED
