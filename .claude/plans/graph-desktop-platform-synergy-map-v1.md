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
- **The `automataIA/*` repos are scoped to ONE thing (operator ruling,
  2026-07-19): a learning reference for the browser client hardware-acceleration
  layer — NOT an IR/backend source.** They were a ChatGPT-master-prompt artifact
  and are **not** dependencies; the IR/retrieval/package substrate is **our own
  OGAR transpiler sink-in** (scope box below), never those repos — this narrows
  SYNERGY-MAP-S00-S07 §4.G's broad REUSE treatment. **What they ARE good for:**
  their "army of wasm + WebGL hardware acceleration" is the reference to *learn
  the client acceleration layer from* — the wgpu / WebGL2 / wasm rendering that
  drives `a2ui-paint` (the GPU tier over `PaintLayout`) and the `FieldviewClient`
  browser tier, i.e. how to hardware-accelerate the ClassView / FieldMask /
  WideFieldMask-addressed, askama/ERB-templated fieldview surface locally. Most
  relevant for that lens: `lodviz-rs` (native/wasm algo-core split + LOD),
  `wasm-typst-studio-rs` (persistent wasm render session), `dashboard-studio-rs`
  (wasm dashboard rendering). Pull the *pattern*, not the code; re-clone on demand
  when the a2ui-paint client-acceleration gate (§4) is worked, not before.
  (`stockfish-rs` remains a real AdaWorldAPI repo with a prior S00-S07 receipt,
  not part of *this* PoC.)
- **Push scope this session:** lance-graph only. OGAR and a2ui-rs are
  sibling-session arcs — findings below are surfaced, not acted on.

> ### Scope box — what this program actually is (operator, 2026-07-19)
>
> **The IR substrate is our own: OGAR's transpiler sink-in substrate.** Not
> external references. Applications become semantic graph packages by being
> **transpiled into the OGAR IR** (`ogar-from-{ruff,schema,rails,docv1}` →
> `ogar-vocab` Class/ActionDef + `ogar-doc-ir` + the codebook + classid-keyed
> adapters), then projected through the graph-desktop loop. The IR-bundle /
> lazy-loading / application-package rows in §2.4/§2.5 are all this one
> substrate, not a thing to source elsewhere.
>
> **The concrete target: "Odoo-rs at the cost of a ~2 MB import."** Import an
> Odoo-shaped app's model definitions (a small import, not a port), transpile
> them into the OGAR IR, and get a running Rust ERP *projected as a graph
> desktop* — the application-store vision (§3/§2.5) realized by transcode, not
> reimplementation. Prior art already in-tree: `ogar-from-rails`,
> `lance-graph-ontology/src/odoo_blueprint/`, the Odoo `PortSpec` (`0x0002`),
> `ogar-adapter-*`.
>
> **The main focus is the proof-of-concept of the architecture** — not
> exhaustive capability mapping. The map below exists to show the PoC is a
> *wiring* of things that already exist (the P-REHOST Citrix loop + the OGAR
> transpiler + the projection loop), and to name the few genuinely-missing
> seams that the PoC must cross. Read the gates (§4) as "what the PoC needs,"
> and everything marked MISSING/PROPOSED as "the PoC's actual build surface."
>
> #### The organizing thesis: the pattern IS the platform (operator, 2026-07-19)
>
> MedCare-rs, Odoo-rs, OpenProject-nexgen-rs, and A2UI-RS are **not four
> projects — they are four instances of ONE pipeline**:
>
> ```text
> legacy app → OGAR transpiler sink-in  (harvest → Class / ActionDef / ClassView
>                                        / classid IR + codebook + classid-keyed adapters)
>            → a2ui graph-desktop projection  (project_node → sealed NodeDelta →
>                                              FieldviewClient → a2ui-paint pixels)
> ```
>
> **This is already parameterized, not aspirational.** OGAR's `PortSpec`
> registry (`ogar-vocab/src/ports.rs`) already assigns the per-app slots:
> OpenProject `0x0001`, Odoo `0x0002`, WoA `0x0003`, SMB `0x0004`, MedCare
> `0x0005`, q2 `0x0006`, Redmine `0x0007` (+ Healthcare). An "app" = **a
> `PortSpec` + a harvest**, nothing bespoke per app. **MedCare already proved
> the entire loop end-to-end** (`p_rehost_full.rs`, PR #216/#217, merged): a
> real harvested screen, transpiled, projected, sealed both ways, painted to
> pixels, action resolved by address. So "why not reuse the pattern" is the
> answer, not the question — **Odoo-rs is `OdooPort` + an Odoo model import
> through the same pipeline; OpenProject-nexgen-rs is `OpenProjectPort` + a
> harvest through the same pipeline.** The application store is the set of
> registered transcodes; the platform is the pipeline. Consequently every
> MISSING/PROPOSED row below is a *pipeline* gap (paid once, reused N apps),
> never per-app work — which is exactly why the PoC on one app (MedCare) is a
> proof for all of them.
>
> #### The technology statement the PoC makes (operator, 2026-07-19)
>
> **Moving parts from Citrix to graph rendering:** replace pixel-remoting with
> **address-remoting + local hardware-accelerated render**. The wire carries
> addressed state (`NodeDelta`) + semantic actions (`ActionInvoke` / `SetField`),
> **never pixels**; the client decodes, unseals, and paints on its own silicon.
> What makes it *serious* — not a demo — is that every layer is a proven,
> already-in-use pattern, composed in the browser:
>
> - **Sealed channel** — Argon2id KDF + XChaCha20-Poly1305 AEAD, the *same*
>   `encryption` crate native servers use ("one codebase for native servers **and
>   wasm32 browsers**", `ndarray/crates/encryption`; re-exported by
>   `ogar-encryption`; wired in `a2ui-server::SealedTransport`). Proven both
>   directions in the merged P-REHOST loop.
> - **Hardware acceleration** — ndarray SIMD (native **and** wasm, via the
>   `wasm-simd-parity` crate) for decode, and **wgpu (WebGPU) / WebGL2** for the
>   GPU raster of the addressed fieldview (`a2ui-paint` `wgpu` feature — "one
>   crate covers both browser targets").
> - **Graph rendering** — the ClassView / FieldMask / WideFieldMask-addressed,
>   askama/ERB-templated surface (`a2ui-paint` + `FieldviewClient`), projected
>   from the canonical graph, not a server-side framebuffer.
>
> The PoC is the *composition* of these three already-shipped patterns into one
> browser thin client. No new cryptography, no new SIMD, no new GPU code — the
> statement is that the combination **replaces Citrix**. §4 gate 7 (the
> wgpu/WebGL2 client-acceleration last-mile) is therefore the PoC's headline
> build: the crypto and the addressing are done; the browser-native paint is the
> seam that turns the proof into the statement.

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

**Zero-copy invariant (whole substrate, read every row below through it):** every
`SoaEnvelope` is zero-copy from creation to Lance tombstone — Lance's columnar I/O
writes the in-place SoA LE bytes; **nothing is serialized between mailboxes**
(`docs/architecture/soa-three-tier-model.md`). So "transaction/commit" here is
*not* a serialize-then-CAS protocol; it is an in-place batch writer + Lance
version + temporal deinterlace.

| Capability | Existing impl | Status | Owner | Reuse decision | Missing seam | Proof |
|---|---|---|---|---|---|---|
| Canonical node identity (NodeGuid/EdgeBlock/NodeRow, 16\|16\|480) | `lance-graph-contract/src/canonical_node.rs` (3094 LOC), V3 4+12 facet, `mint_for`/`TailVariant`/`classid_read_mode` | SOLID | lance-graph-contract | REUSE | — | Const size asserts, `classid_scan.rs` adoption monitor |
| SoaEnvelope / LE contract / MailboxSoA | `soa_envelope.rs` (498 LOC), `ENVELOPE_LAYOUT_VERSION=2`, `verify_layout` | SOLID | lance-graph-contract | REUSE | — | `.claude/v3/soa_layout/le-contract.md` (operator-locked) |
| Lance versions as temporal stream | `lance-graph-planner/src/temporal.rs` (`QueryReference::at`, `deinterlace`); contract mirror `temporal_pov.rs` | SOLID (stream-side migration itself is probe-gated: D-MTS-1..3, per lance-graph CLAUDE.md ruling `E-MARKOV-TEMPORAL-STREAM-1`) | lance-graph-planner | REUSE | The VSA-to-stream cutover (D-MTS-1) is Queued, not this program's concern directly | 663-LOC file |
| **Graph transaction / commit** | The commit path is the **zero-copy ahead-firing batch writer** `lance-graph-planner/src/batch_writer.rs::BatchWriter<P>` (payload-generic — never inspects `P`, DTO purity; `cast(on_behalf: MailboxId, moves, payload)` write-on-behalf + W1c `resolve_owner` delegation cache), which reads via `temporal.rs` `QueryReference::at` + `deinterlace` and persists through lance `=7.0.0` / lancedb `=0.30.0` columnar I/O, in place. Plus `graph/versioned.rs::VersionedGraph` (`commit_encounter_round`, `at_version`, `diff`) for the Lance-version layer; `contract/transaction/{Interactive,Bulk,Periodisch}` are the typed tx contexts | **PROBE-GREEN** (batch_writer 4 tests + W1 probes; VersionedGraph tests). The earlier "missing `expected_version`/CAS" read is **largely moot by design**: writes are **single-owner** (mailbox-as-owner, supervisor sole mutator), so there are no concurrent writers to the same node to reconcile | lance-graph | REUSE | To *verify*, not build: confirm CAS is unnecessary under single-owner ownership (add optimistic-concurrency only if multi-owner writes are ever introduced). The one genuine open atomicity question is cross-dataset (nodes+edges checked out pairwise in `diff`) | `batch_writer.rs` 4 tests; `VersionedGraph` tests; zero-copy per `soa-three-tier-model.md` |
| Idempotency (per-action) | `contract/action.rs:204` | SOLID at the action level | lance-graph-contract | REUSE | Doesn't compose into a graph-commit-level idempotency guarantee | — |
| Policy checkpoints / RBAC | `lance-graph-rbac`: `Policy`, `smb_policy()` factory, `authorize_scoped`, `ClassGrants` | SOLID | lance-graph-rbac | REUSE | **`medcare_healthcare_policy` does NOT live in lance-graph** (verified: zero hits) — it's in MedCare-rs's own `medcare-rbac::policy` crate, consumed by `medcare-realtime::gate`. The shipped upstream precedent factory is `smb_policy()` only | `medcare-realtime/src/stack.rs:48,149` — open Q when platform-izing: keep per-domain policy factories (current) vs a platform capability composer (§2.5) |
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
| First golden application (Patient projection) | MedCare-rs: `MedcareClassView::for_patient` (concept 0x0901) consumed by 5 view modules (`views/{wartezimmer,vital,lab,sono,anamnese}.rs`); `patient_projection.generated.rs` (live codegen'd `From<PatientRow>`); **`tests/p_rehost_full.rs` — merged PR #217** — the real a2ui-server + a2ui-wasm FieldviewClient + a2ui-paint driven against a real harvested screen (`uc_patfile_sub_diagnosis`), sealed both ways, painted to real PNG pixels, pixel-click resolved by address | **SOLID** — this is the furthest-along concrete slice of the whole master prompt, already merged | MedCare-rs | REUSE as the reference vertical | Feature-gated `p-rehost`, off by default — not yet a "real installed application" in the package-store sense (no manifest, no install step, it's a dev-dependency test). Use this exact slice as the S07-style golden vertical for the platform's first `InstallApplication` proof |

### 2.6 Memory & reasoning

Inherited wholesale from `SYNERGY-MAP-S00-S07.md` §4.A/B/C — re-confirmed by
this session's independent lance-graph mapper with zero contradiction:

| Capability | Status | Notes |
|---|---|---|
| AriGraph episodes / SPO / TripletGraph | SOLID | RRF, PPR/HippoRAG, Leiden, BM25, chained episodic search all landed reading the same `TripletGraph` — no fork, no relationship table |
| NARS truth + revision | SOLID (triplicated) | canonical `contract::crystal::TruthValue`; duplicates in `holograph/src/width_16k/schema.rs:104` + `lance-graph-arm-discovery/src/translator.rs:61` — REUSE canonical, do not consume the other two |
| Palette256 / CAM-PQ | SOLID | `bgz17` + `ndarray` codec + `cognitive_palette` 226-atom codebook |
| Retrieval trace / explainability | MISSING | S00-S07 names this exact gap (`RetrievalHit` explanation record) independently; do not re-scope, inherit their S03 |
| Evidence address | MISSING | see §2.1 above, same finding as S00-S07 §6.2, inherited not re-derived |
| Episodic-witness tenant | SHAPED/reserved | 292 B headroom in the 480-B slab, `EpisodicWitness64` deferred accessor named but not a live column. S00-S07 flags `tenants.md:41` ("328 B reserved") as stale vs live code (292 B) — re-check `canonical_node.rs`, not the doc, before citing a number |

---

## 3. Golden-application slice — status against the master prompt's §3 example flow

```text
package Patient application     → MISSING (no manifest/package artifact exists)
  → install into test tenant    → MISSING (no InstallApplication op)
  → launch                      → SOLID (Session::establish)
  → render Patient list/detail  → SOLID (MedcareClassView::for_patient, PR #217)
  → invoke one real action      → SOLID (P-REHOST-full: real ActionDef, real predicate resolve)
  → commit state                → PROBE-GREEN (zero-copy BatchWriter + Lance version +
                                   temporal deinterlace; CAS moot under single-owner writes — §2.4)
  → receive one minimal delta   → SOLID (NodeDelta, RBAC-projected, sealed, painted to pixels)
```

**Reading:** **four of seven steps are SOLID** (launch/render/invoke/delta), on a
merged PR, in this exact repo, as of today; `commit state` is PROBE-GREEN (the
zero-copy batch writer exists and is tested, just not exercised end-to-end in the
golden slice). The two genuinely-absent steps — a real package/install step
(MISSING) — are also the gaps independently surfaced in §2.5. This narrows the
platform program's Phase-1 work to the package/install seam plus the
projection-dependency index (§2.1), not a from-scratch build. (Corrected from an
earlier "five of seven SOLID" miscount — `commit` is PROBE-GREEN, not SOLID —
flagged independently by Codex and CodeRabbit on PR #763.)

---

## 4. Gates (next-proof-before-implementation queue)

Ordered; each is a falsifiable probe, not a synthesis exercise.

1. **Projection-dependency index.** Build `(classid, changed field position) →
   {views to reproject}` from existing `KausalSpec::Depends` + N3 field order.
   Falsifier: one real MedCare write (e.g. diagnosis edit) produces the correct
   minimal reproject set without a full-screen resend. This is the cleanest
   net-new brick two independent mappers converged on without prompting.
2. **Verify commit CAS is unnecessary (do NOT build it by default).** The commit
   path is already the zero-copy single-owner `BatchWriter` + Lance version +
   temporal deinterlace (§2.4). Falsifier: exhibit any code path where two owners
   can write the same node concurrently. If none exists (expected, given
   mailbox-as-owner + supervisor sole-mutator), CAS/`expected_version` is moot and
   this gate closes GREEN with no code. Only if a multi-owner write path is found
   does optimistic-concurrency become real work. The one residual open question is
   cross-dataset (nodes+edges) atomicity across the pairwise `diff`.
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
7. **Client hardware-acceleration layer (the browser last-mile).** `a2ui-paint`
   today is layout + headless raster + a headless wgpu rect-fill (no glyphs, no
   windowed present); the PoC needs the real browser tier — **wgpu / WebGL2 +
   wasm** rendering the ClassView/FieldMask/WideFieldMask-addressed,
   askama-templated fieldview surface on the client's own silicon (the "pixels
   on the client, address on the wire" half of the Citrix-via-graph thesis).
   Falsifier: one real harvested screen painted to a live browser canvas via
   wgpu, with a pixel click resolving to an ordinal address (the browser-native
   extension of the merged `p_rehost_full.rs` loop). **Learning reference (not a
   dependency):** the `automataIA/*` wasm+WebGL acceleration patterns — pull the
   technique, keep the substrate ours. This is the gate the operator elevated as
   a first-class PoC concern (2026-07-19).

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

---

## 8. Near-term PoC instance — the document app (operator, 2026-07-19)

The concrete next instance of the §0 organizing thesis. A **document is not
special** — it is a PortSpec instance whose projection skin is a document
editor and whose body lives in KV. **Most** pieces already exist as named seams
(the KV substrate, the sealed transport, the ClassView addressing) — this is
**mostly reused wiring** across three repos, plus **two genuinely-new
implementation seams**: the `SetField` write-frame and the PDF renderer (both
PROPOSED in the table below). Not greenfield, but not pure wiring either.

| Piece | Shape | Existing seam it wires | Owner repo | Status |
|---|---|---|---|---|
| Document = KV | `DocumentPath` / `DocumentID` **as key**, raw data **as value** — the doc lives dynamic in the graph; **pointers in the hot path** (16-byte key), raw body pulled on demand (zero-copy value-slab-skip, L4 lazy bundle) | the "DocumentID-KV / witness-handle seam" (`graphrag-doc-retrieval-soa-integration-v1.md §4a`) + `ogar-doc-ir::{DocIr, DocPage, content_sha256}` + lance-KV | lance-graph (contract type) / OGAR (doc-ir) | SHAPED — doc-ir exists; the `DocumentID` KV key type is the small NEW contract brick |
| Word WYSIWYG editor | edits emit `SetField{key, field_position, value}` **write-frames** — the write-mirror of `ActionInvoke`, never a blob/char-range mutation; ProseMirror/Tiptap *concepts* over a canonical GUID-keyed tree | `a2ui-rs/.claude/plans/projectional-knowledge-editor-v1.md` (ratified direction) | a2ui-rs | PROPOSED — the `SetField` frame is the missing brick (3rd `FrameKind`, a2ui queue item 4) |
| PDF on demand | `tesseract-rs` renders the graph-resident document → PDF when asked; the PDF is a **projection**, not the stored form | `pdf-to-text-ocr-v1.md` P4 PDF renderer | tesseract-rs | PROPOSED — P4 renderer not built |
| Local hardware accel + seal | wgpu/WebGL2 paint of the ClassView/FieldMask/WideFieldMask-addressed, askama/ERB-templated surface; XChaCha20-Poly1305 + Argon2id transport | a2ui-paint (§2.2) + SealedTransport (§2.3) | a2ui-rs | SOLID transport; SHAPED paint last-mile (§4 gate 7) |

**Reuse, immediately:** a **WoA-rs work order** (`PortSpec 0x0003`) is the same
document-app pattern — a work order is a document. A **mail program** is another
(`stalwart`, already in the workspace). Enough of these — documents, work
orders, mail — over the same ClassView/FieldMask/WideFieldMask + askama/ERB +
local-hardware-accel + ChaCha20/Argon2 loop **is** the "serious thin client over
graph-execution projection" the master prompt names. Nothing here re-proves the
architecture (MedCare already did, §3); each is a PortSpec + a harvest + this
document skin.

**New pipeline bricks this instance needs** (each paid once, reused by every
document/work-order/mail app): the `DocumentID` KV key contract (lance-graph),
the `SetField` write-frame (a2ui-rs), the P4 PDF renderer (tesseract-rs). This
session can push only the lance-graph brick; the other two are sibling-arc
surfaces to hand off.

---

## 9. The FIRST PoC — bulletproof-access monitoring + BI reporting, "for $0" (operator, 2026-07-19)

Start read-only, below the document editor: **bulletproof access** to a
**Grafana-class monitoring + Power-BI / Databricks-class reporting & data-search**
surface that showcases the one property no pixel-remoting or cloud-BI stack can
match — **computationally "for free" / "$0."** Two headlines:

1. **Bulletproof access** is the security statement, not an afterthought: RBAC
   fail-closed (`WideFieldMask ∩ role`, proven past bit-64) + Argon2id +
   XChaCha20-Poly1305, banking-grade, the same crypto native *and* wasm use.
   The client only ever receives the fields its **RBAC role mask** permits;
   masked columns never cross the wire (proven negatively in P-REHOST's
   fail-closed test). This is the *implemented* guarantee today — the role-mask.
   Per-tenant **capability composition** (§2.5) and stronger identity (OIDC /
   WebAuthn / forward secrecy, §2.3) are still **future work**, not part of the
   current guarantee.
2. **"$0" / for free** is literal, and matches the external signal the operator
   cited (a Reddit r/databricks post, "data search engine for $0 using Rust +
   Hugging Face" — could not be fetched this session, reasoned from the title:
   local embeddings + Rust search = no cloud cost). The workspace has a *more
   complete* version of that idea, all local Rust, no cloud, no per-query cost:
   - **$0 embeddings** — DeepNSM (local semantic engine, 680 GB → 16.5 MB; no
     cloud embedding API).
   - **$0 search** — CAM-PQ compressed ANN + AriGraph RRF / BM25 / PPR (SOLID,
     §2.6) over the same zero-copy store.
   - **$0 analytics** — datafusion analytical SQL over the zero-copy Lance
     columnar store (no Databricks cluster, no ETL, no serialization boundary).
   - **$0 render** — the client's own wgpu/WebGL2 silicon (§ technology
     statement), not a server framebuffer or a BI-cloud render tier.

The rest of this section is the read-only monitoring/timeline half, which is
almost entirely SOLID today.

**Why "for free" is literal (scoped to the server-side data path):** there is
**no server-side serialization / ETL boundary**. The SoA is zero-copy from
creation to Lance tombstone (§2.4); a *metric* is already a ClassView field over
an existing SoA column. So a dashboard read is a `WideFieldMask` projection over
columns **that already exist** — no query engine, no ETL, no metrics store, no
server-side serialize/transform step. (The `NodeDelta` frame is still LE-encoded,
sealed, transferred, and **decoded on the client** — but those LE bytes *are* the
in-place SoA bytes, not a re-serialized copy; what's eliminated is the server-side
`query → serialize → deserialize → render` pipeline and its copies, **not** the
network itself.) Grafana does `query → serialize → transfer → deserialize →
render`; here it is `mask-existing-columns → NodeDelta (in-place LE bytes) → seal
→ transfer → client decodes + paints`. The server does ≈nothing beyond mask +
frame + seal; the client's own silicon renders. A
**timeline chart is free the same way**: a metric's history *already exists* as
Lance versions — a line chart is a `temporal.rs` `QueryReference::at` +
`deinterlace` version-range read, zero copies.

| Piece | Status | Note |
|---|---|---|
| Zero-copy RBAC read projection | SOLID | `a2ui-server::project_node` (RBAC-masked `NodeDelta`) read-only over existing rows — no harvest, the data is already in the graph |
| Time-series / timeline | SOLID | `temporal.rs` `QueryReference::at` + `deinterlace` — a metric's history already exists as Lance versions; a chart is a version-range read |
| RBAC fail-closed (banking-grade access) | SOLID | `WideFieldMask ∩ role` (`a2ui-server::project.rs`), proven past bit-64 |
| Sealed transport (banking-grade safety) | SOLID | Argon2id + XChaCha20-Poly1305 (`SealedTransport`), native+wasm |
| Analytical reporting (Power-BI/Databricks-class) | SOLID | datafusion analytical SQL over the zero-copy Lance columnar store — heavy aggregations without a cluster or ETL (`lance-graph` `datafusion_planner`) |
| Semantic data-search ("$0" search engine) | SOLID | DeepNSM local embeddings + CAM-PQ compressed ANN + AriGraph RRF/BM25/PPR — no cloud embedding API, no per-query cost (§2.6) |
| Grid / Timeline / chart skins | PROPOSED | `a2ui-paint` has Form/Flow only; a dashboard needs Grid + Timeline — the **one new client brick** (a2ui-rs, sibling-arc). **Learning reference (client-accel lens only, §0):** automataIA data-viz — `lodviz-rs` (LTTB/M4 LOD downsampling + native/wasm algo-core split + linked selection) and `dashboard-studio-rs` (dashboard layout/interaction ergonomics; *reject* its ECharts/JSON renderer — we paint via wgpu/WebGL2 over the addressed surface). **Hard caveat (S00-S07 needle-pinning):** LOD must never drop rare events — an alert / anomaly / threshold breach is a pinned point, never a downsampled-away sample |
| PDF report on demand | PROPOSED | tesseract-rs P4 renderer — optional, for the "reporting" half |

**Why this first:** read-only (no `SetField` write-frame — simpler than §8's
editor), no harvest/transpile for the demo (zero-copy over existing columns), and
it demonstrates the "for free" differentiator **+ RBAC + banking safety** in the
smallest credible surface. It is the cleanest first "moving parts from Citrix to
graph rendering" statement: a monitoring wall that costs the server nothing to
serve, because there is no copy.

**Ideal first dashboard — a GraphSentinel-style access/audit monitor.** The
operator's `GraphSentinel` reference (a lightweight Python graph sentinel — a
watchdog over graph access/security; not fetched, understood conceptually) is
already how bulletproof access works natively here, so the cleanest first §9
dashboard **is a security-access / audit monitor over the graph's own witness**:
- **Access gate** = RBAC fail-closed (`WideFieldMask ∩ role`), SOLID.
- **The witness** = the merkle-chained `UnifiedAuditEvent` → SPO + Lance
  tombstone + `WitnessTable`, **examined in place as a query, never egressed to
  a SIEM** (medcare-rs commitment 7 — audit review *is* a query against the
  witness).

> **Readiness (honest):** the **access-gate half is SOLID** (RBAC fail-closed).
> The **audit-witness half is GATED on §4 Gate 5**, not ready today: §2.4 flagged
> **two conflicting `AuditSink` trait definitions** and **no zero-dep
> contract-side home** — the witness currently lives in `lance-graph-callcenter`
> (`audit_sink/`), not the contract crate. So the GraphSentinel *access* monitor
> is buildable now; the *audit* dashboard is **PROPOSED / gated** until the
> AuditSink canonicalization lands.

This unifies both §9 headlines in one demo: a live wall watching access + audit
events, read-only (no `SetField`), zero-copy over a witness that already
exists (the "$0"), RBAC-masked + sealed (the "bulletproof"), and audit-review-
as-a-query rather than a log export. "GraphSentinel" is thus not a thing to port
— it is the *first concrete ClassView* the monitoring PoC renders (the access
half now; the audit half after Gate 5).

The single new build on the critical path is the
Grid/Timeline skin in `a2ui-paint` (over §4 gate 7's wgpu/WebGL2 last-mile); the
read, RBAC, seal, and time-series are all SOLID today.
