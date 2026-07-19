# Graph Desktop Platform — Synergy Map v1

> Phase-0 deliverable of the operator's graph-desktop-platform master prompt
> (2026-07-19): application store → auth → semantic remote desktop → graph
> execution → reasoning → DTO/IR memory, Fable-5-orchestrated, multi-session.
> **MAP BEFORE BUILDING**: no new production type lands anywhere in this
> program until this map names its nearest existing equivalent and states why
> an adapter/extension is insufficient. This file is the gate ledger.
>
> **Inherits, does not duplicate:** `SYNERGY-MAP-S00-S07.md` (same directory,
> dated 2026-07-16, pinned-tip verified) already mapped the retrieval/reasoning
> axis (GraphRAG operators, Stockfish, tenants, six of the eight external
> `automataIA/*` repos) under an operator-ratified governing rule this map
> adopts verbatim: **"Reuse canonical owners, transcode useful algorithms onto
> existing representations, and add new structures only where a concrete
> missing capability is demonstrated."** Where this map's findings overlap that
> one's, it cites rather than re-derives. Where this program's axis differs
> (application store, session auth, DTO/IR lazy-loading, browser cache — none
> of which S00-S07 covers), this map is the primary source.
>
> Status vocabulary (honesty rule, this map's own bar): **SOLID** = has a real
> consumer on a dependency graph, not just island tests (the a2ui-server
> lesson from the P-REHOST arc: its whole gap was one `[patch]` seam, found
> only when a real consumer tried to build against it). **PROBE-GREEN** =
> passing tests/examples only, no external consumer yet. **SHAPED** = types
> exist, unwired. **PROPOSED** = docs/plans only. **MISSING** = verified
> absent. **DEFERRED** = out of reach this session (proxy scope), not a
> statement about whether it exists.

---

## 0. Session-scope facts (2026-07-19)

- **Repos locally mapped this session:** OGAR (`84369e5`, newer than the
  `OGAR-fresh` clone), lance-graph (`916d5422`, in sync with origin/main),
  a2ui-rs (origin/main `362284e`; local HEAD was a stale side-branch tip at
  `beb53f0` — corrected mid-session), MedCare-rs (origin/main `9e81cb6`, PR
  #217 merged — the P-REHOST-full Citrix loop with paint tier is now live).
- **rs-graph-llm**: cloned fresh into this session via the local git proxy
  (`AdaWorldAPI/rs-graph-llm`, HEAD `59f9315`). **This corrects
  SYNERGY-MAP-S00-S07.md §4.F**, which recorded it as "not an AdaWorldAPI
  repo (org search 0 hits), absent locally, out of scope." It exists, it is
  real, it is tested (47 tests across 4 crates), and it already consumes
  `lance_graph_contract::kanban` directly (§6 below). Treat the S00-S07
  "external, design-ref only" verdict for execution orchestration as
  **superseded** by this map's §6.
- **External reference repos — proxy-blocked this session** (403, uniform,
  session GitHub allowlist — `"Use add_repo to request access"`; the outer
  TLS `CONNECT` succeeds, the block is the proxy's own JSON, not a
  GitHub-level 404): `AdaWorldAPI/stockfish-rs`,
  `automataIA/{graphrag-rs, graph-librarian-rs, wasm-typst-studio-rs,
  lodviz-rs, dashboard-studio-rs, agentic-graphrag-rl-trainer}`.
  **All eight already have prior receipts in SYNERGY-MAP-S00-S07.md §4.C/G**
  (stockfish-rs graded SOLID at pinned tip `f3f728a`; the six automataIA repos
  profiled with explicit REUSE/REJECT verdicts, self-flagged as
  "README-level, not proven"). This map carries those forward rather than
  re-deriving; **re-verify against current tips once `add_repo` unblocks them**
  — do not treat 2026-07-16 receipts as permanently current.
- **Push scope this session:** lance-graph only. OGAR and a2ui-rs are
  sibling-session arcs — findings below are surfaced, not acted on.

---

## 1. Ownership confirmations (master prompt §1, checked against repo evidence)

The master prompt's assignment holds with **one correction** and **one
addition**, both evidence-backed:

| Layer | Master-prompt owner | Repo evidence | Verdict |
|---|---|---|---|
| Semantic ABI (classes, actions, guards, ClassView adapters, codebook, codegen) | OGAR | Confirmed — `ogar-vocab` (ActionDef/EnterEffect/KausalSpec), `ogar-class-view` (OgarClassView over ~79 concepts), `ogar-emitter`/`ogar-from-ruff`/`ogar-render-askama` (codegen), `ogar-a2ui-frame` (wire frames) | **CONFIRMED** |
| Canonical state, transactions, memory | lance-graph | Confirmed for nodes/edges/SoA/AriGraph/NARS/CAM-PQ/RBAC/orchestration-contract. **Correction:** "transactions" is weaker than the master prompt implies — see §6.5, this is a SHAPED gap, not SOLID | **CONFIRMED w/ gap flagged** |
| Remote semantic projection | a2ui-rs | Confirmed — DesktopSession/SealedTransport/FieldviewClient/NestedSurface/a2ui-paint, all SOLID with real tests. Session/browser persistence and reconnection are MISSING (§6.2) | **CONFIRMED w/ gaps flagged** |
| Execution orchestration | rs-graph-llm | **Addition, not correction**: this is not a design reference to build toward later — it is a real, tested, already-partially-wired crate (`graph-flow-kanban` consumes `lance_graph_contract::kanban::{ExecTarget, KanbanColumn, KanbanMove}` directly; `graph-flow-action-ogar` already gates through `commit_via` = RBAC ∧ state-guard ∧ MUL). Promote from S00-S07's "external, out of scope" to **in-scope, partially wired** | **CONFIRMED + PROMOTED** |
| AriGraph/NARS/policy | lance-graph | Confirmed, SOLID (§4.A/B of S00-S07, re-confirmed by this session's lance-graph mapper) | **CONFIRMED** |

---

## 2. Capability map

Table columns per master-prompt spec: Capability · Existing implementation ·
Status · Canonical owner · Reuse decision · Missing seam · Duplication risk ·
Proof · Consumer · Next gate.

### 2.1 Projection & addressing core

| Capability | Existing impl | Status | Owner | Reuse decision | Missing seam | Duplication risk | Proof | Next gate |
|---|---|---|---|---|---|---|---|---|
| ClassView / FieldRef / ObjectView / DisplayTemplate | `lance-graph-contract/src/{class_view.rs:903, ontology.rs:457-486}` | SOLID | lance-graph-contract | REUSE | — | Any new "projection trait" | 2352-LOC file, consumed by OGAR (`OgarClassView`), a2ui-server (`project_node`), MedCare-rs (`MedcareClassView::for_patient`, 5 view consumers) | — |
| FieldMask (64-bit) / WideFieldMask (Small/Wide) | `class_view.rs` (`FieldMask::{EMPTY,FULL,from_positions,has,intersect}`; `WideFieldMask::{full_for,from_positions,has,from_universe_present}`) | SOLID | lance-graph-contract | REUSE | Permit-all identity (`ALL` const vs `full_for`) still **OPEN** — deferred to the WideFieldMask retype PR per both a2ui-rs and S00-S07 | `ClassId` is **duplicated**: `u16` in `class_view.rs:54` vs `u32` in `rbac.rs:103` — live wart, not yet a platform risk but flag before a package-manifest classid field is minted | a2ui-server `project.rs:70` fail-closed test, past-64-position test | Resolve ClassId u16/u32 split before §2.5 package manifests reference it |
| NodeDelta / ActionInvoke / Frame / FRAME_VERSION | `OGAR/crates/ogar-a2ui-frame/src/lib.rs:40,124,137,150,170,207` | SOLID | OGAR | REUSE | — | A parallel frame type for any new wire need — extend `FrameKind`, never add a second frame enum | a2ui-core re-export, a2ui-server 6 transport tests, this session's own P-REHOST-full (MedCare-rs #216/#217) drove real frames over a real screen | — |
| ActionDef / ActionInvocation / EnterEffect / KausalSpec / GuardFailurePolicy | `OGAR/ogar-vocab/src/lib.rs:389,466,508,547,566,585,603,489` | SOLID (types); PROBE-GREEN (guard runtime) | OGAR | REUSE | Guard→ractor-state-machine lowering is documented (`OGAR-AST-CONTRACT.md §6`), not shipped — execution deferred to rs-graph-llm `graph-flow-action` | Any "Action"/"Effect"/"Transition" type in a platform crate | `ogar-action-handler` `parse_applicabilities`, medcare-bridge codegen consumer | Land the guard→state-machine lowering when rs-graph-llm execution (§6.4) is wired |
| ActionInvocation (session-side lowering) | a2ui-rs `crates/a2ui-server/src/lowering.rs` (PR #9, NEW since the earlier `beb53f0` snapshot): `lower_action_fire`, `lower_screen_jump → NavWitness` | PROBE-GREEN | a2ui-rs (compile-time value construction only; vocabulary stays OGAR's) | REUSE | `nav_witnessed` runtime SPO predicate is OGAR issue #210, OPEN — a2ui deliberately stops at a plain `NavWitness` value and does not mint a predicate | Minting `nav_witnessed` outside OGAR | 3 golden tests, corpus-free | OGAR #210 |
| Projection dependency index (changed predicate → affected view) | **Nearest brick, not the thing itself**: `ClassView::compute_dag`/`ComputeEdge`/`compute_dag_topo_order`/`screens_reachable_from` (contract `class_view.rs`, execute_compute_dag l.816) + `KausalSpec::{Depends,Constrains,Onchange}` (OGAR) + `NodeDelta.mask_words` (transport) | **MISSING** — verified by two independent mappers (OGAR + lance-graph), zero hits for `projection.*invalid\|affected_view\|dirty.*view` | Would be lance-graph-contract (the DAG lives there) fed by OGAR's Depends declarations | **NEW_REQUIRED** — this is the one clean genuinely-missing brick both mappers converged on independently | Composing `(classid, changed field position) → {views to reproject}` from existing Depends paths + N3 field order — the inputs all exist, the inversion does not | — | — | **Gate 1** (§4) |
| Evidence address / projection address | `reasoning.rs:39 EvidenceRef` (wrong granularity — batch, not span) + scattered projection components (NodeDelta.key + classid→ClassView + FieldView.position + template slot) | **MISSING** (S00-S07 §6.2 already names this as one of only 3 genuinely-new structures needed for that arc too — same finding, independently reached) | OGAR (evidence is doc-ir/meaning-owned) | **NEW_REQUIRED**, but already scoped in S00-S07 as S03 — do not re-scope here, inherit | Binding doc-ir `content_sha256` + page + (missing) region-id + BBoxRail into one addressable handle | A second evidence type | — | S00-S07's own S03 |

### 2.2 Desktop & client tier

| Capability | Existing impl | Status | Owner | Reuse decision | Missing seam | Proof | Next gate |
|---|---|---|---|---|---|---|---|
| DesktopSession (render/receive/sync loop) | `a2ui-server/src/desktop.rs`: `establish`, `render_node`, `receive_action`, `sync_codebook`, `KlickwegEdge`, `take_klickwege` | SOLID | a2ui-rs | REUSE | Klickwege→AriGraph SPO drain is an **OGAR-side hop**, not built | 6 tests incl. tamper-refused-no-edge | Confirm OGAR landing API before wiring the drain |
| FieldviewClient + NestedSurface | `a2ui-wasm/src/lib.rs`: `apply_node_delta`, `resolved_fields`/`resolved_actions`, `link_child`/`child_at`/`resolve_nested`, `NestedSurface.depth()` | SOLID | a2ui-rs | REUSE | Browser-native, not wasm-only, e2e harness missing | 10 tests incl. G5 reusability falsifier, G6 data-identity | Browser e2e (a2ui-rs's own queue item 3) |
| PaintLayout / Form+Flow skins / wgpu / PNG raster | `a2ui-paint/src/{lib.rs, raster.rs, gpu_lut_probe.rs}` | SOLID (layout+raster core); SHAPED (gpu — headless rect fills only, no glyphs, no windowed present) | a2ui-rs | REUSE | **⚠ Regression found by the a2ui-rs mapper**: commit `9d9505b` dropped `#[cfg(test)] mod gpu_lut_probe;` from lib.rs while adding `pub mod raster;` — the palette-distance-LUT GPU probe (128 KiB, 65536-entry bit-exact parity) silently stopped compiling on origin/main. This is the gate the x265/H.268 sprite-replay plan depends on | 12 wired tests + 2 dead (the regression) | **Surface to a2ui-rs session**: re-wire `mod gpu_lut_probe;` |
| KlickwegEdge / Klickweg telemetry | `desktop.rs:56` | SOLID | a2ui-rs (emit) / OGAR (predicate) | REUSE | Same as row 1 — the drain target | MedCare-rs `klickwege-digest-v2.txt` golden (106 screens / 154 edges per prior session count; 840 lines / re-confirmed 1 SCREEN header this session — recount before citing exact numbers downstream) | — |
| Browser persistent cache (IndexedDB or equiv) | — | **MISSING** — zero hits, codebook + node state live in in-RAM `HashMap`s scoped to the wasm instance lifetime | a2ui-rs (browser cache §5.10 of master prompt) | **NEW_REQUIRED** when productized | The "amortize once like a font" claim is currently per-page-load only, not per-session/persistent | — | Master prompt §10 — not yet even PROPOSED anywhere |
| Reconnect / session resumption | — | **MISSING** — `establish_random_salt` mints a fresh key every session by design; no resumption token, no server-side session registry | a2ui-rs | **NEW_REQUIRED** | — | — | Not in any existing plan |
| Codebook sync (delta push) | `desktop.rs::sync_codebook → CodebookSnapshot` | SHAPED — versioned snapshot exists server-side; **no delta protocol, no client ingest API**; snapshot carries action labels only, not field labels/predicates the client's `ClientClass` needs | a2ui-rs | ADAPT | The "SyncCodebook amortization" loop is not end-to-end closable today | — | Widen `ClientClass`/snapshot shape together |

### 2.3 Session security

| Capability | Existing impl | Status | Owner | Reuse decision | Missing seam | Proof |
|---|---|---|---|---|---|---|
| Argon2id session derivation | `OGAR/ogar-auth::password.rs` (hash/verify) re-exported via `ogar-encryption::kdf`; consumed in `a2ui-server/session.rs::Session::establish` | SOLID | OGAR (primitive) / a2ui-rs (session use) | REUSE | Permit-all mask identity (same as §2.1) | 5 session tests incl. peer re-derivation |
| XChaCha20-Poly1305 sealed transport | `a2ui-server/transport.rs::SealedTransport` — per-direction counters, strict-monotonic replay/reorder rejection | SOLID | a2ui-rs (wiring) / OGAR (`ogar-encryption`, re-exports ndarray's `encryption` crate) | REUSE | None found — explicitly rests on the fresh-salt-per-session invariant, no rekey/ratchet | 6 tests: roundtrip, wrong-key, tamper, replay, reorder, truncation. **This session's own P-REHOST-full test drove this over real MedCare screens both directions** |
| Replay/reorder protection | Same file — checked before AEAD open | SOLID | a2ui-rs | REUSE | — | Same 6 tests |
| Tenant identity | `lance-graph-callcenter::unified_bridge::TenantId(u32)` | SOLID (type); SHAPED (no platform-wide tenant registry/capability composition yet) | lance-graph | REUSE the type, **NEW_REQUIRED** the composition (§2.5) | Session establishment doesn't yet carry tenant+application+capability as one composed grant — master prompt §4's capability model doesn't exist as a type anywhere | — |
| OIDC / WebAuthn / device identity / forward secrecy | — | **MISSING**, entirely — nothing in OGAR/a2ui-rs implements identity-provider login, step-up auth, or ephemeral key agreement; current auth is a shared-secret Argon2id KDF, which the master prompt itself flags as insufficient alone (§5) | none yet | **NEW_REQUIRED** | The whole master-prompt §5 "separate responsibilities" section describes work that hasn't started | — |

### 2.4 State & transactions

| Capability | Existing impl | Status | Owner | Reuse decision | Missing seam | Proof |
|---|---|---|---|---|---|---|
| Canonical node identity (NodeGuid/EdgeBlock/NodeRow, 16\|16\|480) | `lance-graph-contract/src/canonical_node.rs` (3094 LOC), V3 4+12 facet, `mint_for`/`TailVariant`/`classid_read_mode` | SOLID | lance-graph-contract | REUSE | — | Const size asserts, `classid_scan.rs` adoption monitor |
| SoaEnvelope / LE contract / MailboxSoA | `soa_envelope.rs` (498 LOC), `ENVELOPE_LAYOUT_VERSION=2`, `verify_layout` | SOLID | lance-graph-contract | REUSE | — | `.claude/v3/soa_layout/le-contract.md` (operator-locked) |
| Lance versions as temporal stream | `lance-graph-planner/src/temporal.rs` (`QueryReference::at`, `deinterlace`); contract mirror `temporal_pov.rs` | SOLID (stream-side migration itself is probe-gated: D-MTS-1..3, per lance-graph CLAUDE.md ruling `E-MARKOV-TEMPORAL-STREAM-1`) | lance-graph-planner | REUSE | The VSA-to-stream cutover (D-MTS-1) is Queued, not this program's concern directly | 663-LOC file |
| **Graph transaction / commit** | `graph/versioned.rs::VersionedGraph` (`commit_encounter_round`, `at_version`, `diff`, `checkout_version`) — this **is** real Lance-version wiring, not a stub. Separately: `contract/transaction/{Interactive,Bulk,Periodisch}` are typed transaction **contexts**, not a commit protocol | **SHAPED — precise gap, not absent**: no `expected_version`/CAS optimistic-concurrency API on graph writes anywhere; nodes and edges are **separate Lance datasets** checked out pairwise in `diff` — no multi-dataset atomic commit | lance-graph | EXTEND | **This is the master prompt §6 "Transaction requirements" gap** (optimistic version checks, idempotency keys, outbox pattern) — none of that exists at the graph-commit layer today, only per-action idempotency (`action.rs:204 idempotency_key`) | `VersionedGraph` tests exist; no CAS test exists because there's no CAS |
| Idempotency (per-action) | `contract/action.rs:204` | SOLID at the action level | lance-graph-contract | REUSE | Doesn't compose into a graph-commit-level idempotency guarantee | — |
| Policy checkpoints / RBAC | `lance-graph-rbac`: `Policy`, `smb_policy()` factory, `authorize_scoped`, `ClassGrants` | SOLID | lance-graph-rbac | REUSE | **`medcare_healthcare_policy` does NOT live in lance-graph** (verified: zero hits) — it's in MedCare-rs's own `medcare-rbac::policy` crate, consumed by `medcare-realtime::gate`. The shipped upstream precedent factory is `smb_policy()` only | `medcare-realtime/src/stack.rs:48,149` | When platform-izing RBAC, decide: does every domain repo keep its own policy factory (current pattern) or does a platform capability composer (§2.5) subsume them? |
| DTO persistence | `lance-graph-callcenter::{ontology_dto::OntologyDto, unified_bridge::UnifiedBridge}` | SOLID | lance-graph-callcenter | REUSE | — | — |
| Audit / witness | `callcenter::audit_sink::{AuditSink trait, JsonlSink, LanceSink}` + `unified_audit::UnifiedAuditEvent` + merkle chain | SOLID, **but not contract-side** — MedCare-rs's own CLAUDE.md commitment 7 explicitly forbids `JsonlAuditSink`/`with_jsonl_audit` as sanctioned surfaces ("legacy ship-logs-to-Splunk pattern"), yet both exist in callcenter today | lance-graph-callcenter | **CONFLICT TO RESOLVE, not just adapt** — the contract crate has `WitnessEntry`/`WitnessTable<64>` (zero-dep) but the typed `AuditSink`/`UnifiedAuditEvent` pair a consumer would want without pulling callcenter has no contract-side home | — | **Also found: two separate `AuditSink` trait definitions coexist** (`audit_sink/mod.rs:46` and `audit.rs:70`) despite PR #366 claiming unification — verify before consuming either |
| IR bundles / lazy loading / content-addressed artifacts | Nearest: `ogar-from-ruff/examples/compile_corpus.rs` (gz bundles, example-tier only); the **lossless-DO rule** in `ogar-from-schema/src/do_arm.rs` *mandates* content-addressed action bodies but ships no hashing code — `body_source` stays `None` | **PROPOSED** (doctrine, no code) | OGAR (doctrine) / lance-graph (storage, if built) | **NEW_REQUIRED** | The content-hash store + lazy body resolution doesn't exist anywhere | — |
| Hot/warm/cold placement | `canonical_node.rs::ReadMode` variants (OSINT=hot, FMA=cold-compressed) — placement is encoded as **read-mode semantics**, not a storage-tier engine | SHAPED (semantics) / **MISSING** (engine) | lance-graph | EXTEND | No `StorageTier`/eviction/migration API exists; the three-tier model is a doc (`docs/architecture/soa-three-tier-model.md`), never reified | — |

### 2.5 Application store

| Capability | Existing impl | Status | Owner | Reuse decision | Missing seam |
|---|---|---|---|---|---|
| Application manifest / codebook digest | `OGAR/ogar-vocab/src/app.rs::render_classid` (canon-high `(concept<<16)\|prefix`), `PortSpec` trait (`APP_PREFIX`, `classview()`, `aliases()`), `docs/APP-CLASS-CODEBOOK-LAYOUT.md` (SPEC status) | SOLID (registry) / **MISSING (signed manifest)** | OGAR | REUSE registry, **NEW_REQUIRED** signing | Zero hits for sha256/digest/signature anywhere in OGAR crate code — the "signed package" the master prompt wants doesn't exist as an artifact, only as codebook-registration doctrine |
| Capability registration | `ogar-vocab/src/capability_registry.rs::{CapabilityRegistration, verify_registration, RegistrationDrift, resolve_hotplug}` | SOLID (coverage/registration) / **PROPOSED (grants)** | OGAR (registration) / lance-graph (grants, per RBAC retype pending) | REUSE registration | This is coverage tracking (declared capability → consumer arm → classid activated), not per-tenant capability **grants** — the master prompt's `GrantApplicationCapability`/`RevokeApplicationCapability` operations don't exist |
| Package catalogue / install / migration ops | — | **MISSING**, entirely — no `PublishApplication`/`InstallApplication`/`ApplicationMigrationTenant` anywhere | none | **NEW_REQUIRED** | Whole §3 of the master prompt is greenfield |
| First golden application (Patient projection) | MedCare-rs: `MedcareClassView::for_patient` (concept 0x0901) consumed by 5 view modules (`views/{wartezimmer,vital,lab,sono,anamnese}.rs`); `patient_projection.generated.rs` (live codegen'd `From<PatientRow>`); **`tests/p_rehost_full.rs` — merged PR #217** — the real a2ui-server + a2ui-wasm FieldviewClient + a2ui-paint driven against a real harvested screen (`uc_patfile_sub_diagnosis`), sealed both ways, painted to real PNG pixels, pixel-click resolved by address | **SOLID** — this is the furthest-along concrete slice of the whole master prompt, already merged | MedCare-rs | REUSE as the reference vertical | Feature-gated `p-rehost`, off by default — not yet a "real installed application" in the package-store sense (no manifest, no install step, it's a dev-dependency test) | Use this exact slice as the S07-style golden vertical for the platform's first `InstallApplication` proof |

### 2.6 Memory & reasoning

Inherited wholesale from `SYNERGY-MAP-S00-S07.md` §4.A/B/C — re-confirmed by
this session's independent lance-graph mapper with zero contradiction:

| Capability | Status | Notes |
|---|---|---|
| AriGraph episodes / SPO / TripletGraph | SOLID | RRF, PPR/HippoRAG, Leiden, BM25, chained episodic search all landed reading the same `TripletGraph` — no fork, no relationship table |
| NARS truth + revision | SOLID, **but triplicated**: canonical `contract::crystal::TruthValue`, plus duplicates in `holograph/src/width_16k/schema.rs:104` and `lance-graph-arm-discovery/src/translator.rs:61` — REUSE canonical, do not consume the other two |
| Palette256 / CAM-PQ | SOLID — `bgz17` + `ndarray` codec + `cognitive_palette` 226-atom codebook |
| Retrieval trace / explainability | **MISSING** — S00-S07 names this exact gap (`RetrievalHit` explanation record) independently; do not re-scope, inherit their S03 |
| Evidence address | MISSING — see §2.1 above, same finding as S00-S07 §6.2, inherited not re-derived |
| Episodic-witness tenant | SHAPED/reserved — 292 B headroom in the 480-B slab, `EpisodicWitness64` deferred accessor named but not a live column. **Note:** S00-S07 flags the doc `tenants.md:41` ("328 B reserved") as stale against live code (292 B) — re-check against `canonical_node.rs`, not the doc, before citing a number |

---

## 3. Golden-application slice — status against the master prompt's §3 example flow

```
package Patient application     → MISSING (no manifest/package artifact exists)
  → install into test tenant    → MISSING (no InstallApplication op)
  → launch                      → SOLID (Session::establish)
  → render Patient list/detail  → SOLID (MedcareClassView::for_patient, PR #217)
  → invoke one real action      → SOLID (P-REHOST-full: real ActionDef, real predicate resolve)
  → commit state                → SHAPED (Lance version bump exists; no CAS/expected_version)
  → receive one minimal delta   → SOLID (NodeDelta, RBAC-projected, sealed, painted to pixels)
```

**Reading:** five of seven steps are already SOLID, on a merged PR, in this
exact repo, as of today. The two gaps — a real package/install step, and a
CAS-checked commit — are also the two gaps independently surfaced in §2.4/§2.5
above. This narrows the whole platform program's Phase-1 work to those two
seams plus the projection-dependency index (§2.1), not a from-scratch build.

---

## 4. Gates (next-proof-before-implementation queue)

Ordered; each is a falsifiable probe, not a synthesis exercise.

1. **Projection-dependency index.** Build `(classid, changed field position) →
   {views to reproject}` from existing `KausalSpec::Depends` + N3 field order.
   Falsifier: one real MedCare write (e.g. diagnosis edit) produces the correct
   minimal reproject set without a full-screen resend. This is the cleanest
   net-new brick two independent mappers converged on without prompting.
2. **Graph-commit CAS.** Add `expected_version` + idempotency-key checking to
   `VersionedGraph::commit_encounter_round` (or a wrapper). Falsifier: two
   concurrent writers to the same node, one must lose deterministically.
3. **Signed package manifest (minimal).** One `ApplicationManifest` +
   `package_digest` + `InstallApplication` op, exercised against the P-REHOST
   Patient slice as the first golden package. Falsifier: install → launch →
   render, with the codebook/asset set the session receives provably scoped to
   what the manifest declared (not "everything the server has").
4. **gpu_lut_probe re-wiring** (a2ui-rs, surfaced not owned) — mechanical, low
   risk, but currently silently dead on main; the x265/H.268 sprite-replay gate
   depends on it.
5. **AuditSink contract-side home** — resolve the two-trait-definition
   duplication and decide whether the typed-event vow (MedCare-rs commitment
   7) gets a zero-dep contract surface, before any platform-wide audit wiring
   references it.
6. **ClassId u16/u32 unification** — before a package manifest or capability
   grant type references `ClassId`, pick one width; both exist today.

---

## 5. Duplication risks (do not re-mint, across the whole program)

Action/Effect/Guard/Transition types (OGAR `ogar-vocab`) · classid bit math
(`app.rs::render_classid`, never a local `(c<<16)|p` literal) · `PortSpec`
allocations (prefixes 0x0001–0x0007 already assigned) · ClassView/ObjectView/
FieldMask (lance-graph-contract; OGAR only adapts) · `SoaMemberSpec`/
`UnifiedStep`/`StepDomain`/`ClassRbac`/`LanceVersionWatcher` (lance-graph
shelf) · crypto primitives (`ogar-auth`/`ogar-encryption` are the mandated
single surfaces — hand-rolling Argon2/ChaCha anywhere else is a violation) ·
wire frames (`FrameKind` is closed-vocabulary; extend it, never add a parallel
frame type) · `NarsTruth` (canonical is `contract::crystal::TruthValue`; two
other definitions exist and must not be consumed) · `Context` in rs-graph-llm
is untyped `Arc<DashMap<String, Value>>` JSON storage today — **the master
prompt's "pass handles, not graph-sized JSON payloads" principle is NOT yet
true of graph-flow's own Context type**; a platform-level typed-handle layer
(§2.4 DTO/IR handles) would sit above/replace this, not extend it as-is.

---

## 6. Execution orchestration — rs-graph-llm detail (new this session)

Corrects and extends S00-S07 §4.F/§3 (which had this "external, out of scope,
design-ref only"). Now cloned, real, mapped directly:

| Capability | Path | Status | Notes |
|---|---|---|---|
| Task trait / NextAction / pause-resume | `graph-flow/src/task.rs`: `Task::run`, `NextAction::{Continue, ContinueAndExecute, GoTo, GoBack, End, WaitForInput}` | SOLID | `WaitForInput` is the human-approval primitive; consumed by `medical-document-service::human_review.rs` and insurance-claims-service tasks |
| Graph / GraphBuilder | `graph-flow/src/graph.rs` | SOLID | conditional edges, `find_next_task` |
| Context (state carrier) | `graph-flow/src/context.rs::Context{data: Arc<DashMap<String,Value>>, chat_history}` | SOLID as a JSON store; **NOT the typed-handle pattern** the master prompt wants | This is the gap: a platform DTOHandle/IRBundle layer would need to sit above this, since Context today is untyped-by-design |
| FlowRunner | `graph-flow/src/runner.rs::FlowRunner::run` | SOLID | 5 tests |
| kgV ActionHandler (minimal executor plug) | `graph-flow-action/src/lib.rs::{ActionHandler, dispatch, dispatch_via}` | SOLID | 11 tests; explicitly documents itself as "the smallest common multiple" — hot path only runs on `Committed` |
| OGAR uplink daemon | `graph-flow-action-ogar/src/daemon.rs::{Daemon, ResolvingDaemon, OgarResolver, ClassRbac, CapabilityExecutor}` | SOLID (18 tests) | `run_gated: commit_via (RBAC ∧ state-guard ∧ MUL) → executor` — this is the master prompt's §6 execution-mode dispatch, already built for at least one mode (native command / SSH executors) |
| Kanban outer envelope | `graph-flow-kanban/src/{lib.rs, orchestrate.rs}::{KanbanPlanEnvelope, run_cycle, CycleOutcome}` | SOLID (12 tests) | **Directly imports `lance_graph_contract::kanban::{ExecTarget, KanbanColumn, KanbanMove, RubiconTransitionError}`** — this crate IS the `OrchestrationBridge`/`UnifiedStep` adapter the master prompt's rs-graph-llm section describes wanting; it already exists, wired to the real contract type |
| Template-task / episodic-arc-task | `crates/template-task` (0 tests — check before relying on it), `crates/episodic-arc-task` (1 test) | SHAPED / PROBE-GREEN | Cognitive-Compilation Elixir-template loop + AriGraph/OSINT episodic arc — additive, cherry-pickable per the workspace Cargo.toml comments |

**Reading:** the master prompt's execution-orchestration section describes
work that is **materially further along than either the master prompt or
S00-S07 assumed** — real dispatch, real RBAC-gated commit, real kanban
integration with lance-graph-contract. The actual gap is narrower than
"build orchestration": it's (a) the untyped-Context-vs-typed-handle mismatch,
and (b) wiring the already-built `Daemon`/`ResolvingDaemon` execution modes to
the projection-dependency index (§4 gate 1) once that exists.

---

## 7. Failed-agent note (process, not findings)

Two of five originally-spawned Phase-0 mapping agents failed on a session API
limit (not a content failure) and were completed directly with tools instead
of re-spawning: the rs-graph-llm mapping (§6 above) and part of the golden-app
slice (§2.5/§3 above). Recorded here per board-hygiene discipline — no content
was lost, the direct-tool-call substitution is receipted inline above.
