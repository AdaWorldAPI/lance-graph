# Component Map — every subsystem onto the V3 substrate

> Produced by the 2026-07-02 mapping fleet (7 subsystem mappers, file:line
> evidence) + main-thread synthesis. Verdicts: REUSE (as-is) / REPURPOSE
> (changed role) / EXTEND (defined addition) / RETIRE (superseded, successor
> named) / NEW (missing piece) / BLOCKED (external). Per-file granularity:
> `MODULE-TABLE.md`. Collapse opportunities: `ENTROPY-MILESTONES.md`.

## Status: FINDING (consumers section lands with the consumer audit)

---

## 1. lance-graph-contract (the wire truth)

| Item | Verdict | V3 mapping |
|---|---|---|
| `MailboxId` (collapse_gate.rs:121) | REUSE | the mailbox addressing handle; survives the emission tombstone |
| `mul::GateDecision` {Flow/Hold/Block} (mul.rs:138) | REUSE | **the LIVE kanban gate** — `KanbanColumn::advance_on_gate` consumes THIS one |
| `collapse_gate::GateDecision` + `MergeMode` | REPURPOSE | per-row write-merge gate, per-mailbox-reachable via `mailbox_owner()`; doc vocab still says "blackboard/BindSpace" (stale prose, live mechanism) |
| `CollapseGateEmission` | RETIRE | already tombstoned (PR #477); comment-only remains |
| `NodeGuid/EdgeBlock/NodeRow` 16\|16\|480 (canonical_node.rs) | REUSE | CANON, const-asserted |
| `ValueTenant::Kanban` 8 B @[144,152) | REUSE | the coded per-row half of "one mailbox = one board as tenant" |
| `BUILTIN_READ_MODES` + `_LEGACY` aliases | REUSE | textbook I-LEGACY discipline; retirement = corpus proof (W6) |
| `TailVariant::{V1,V2,V3}` + read-mode registry | REUSE/EXTEND | classid → {tail, value_schema, edge_codec}; the maturity step toward the full ClassView focus lens |
| `FacetCascade`/`FacetTier`/`CascadeShape`/`hi_chain/lo_chain` (facet.rs) | REUSE | **the coded 4+12 atom** — see le-contract.md §5 |
| `hhtl.rs` NiblePath + `from_guid_prefix{,_v2,_v3}` | REUSE | dual-form fold + V3 both-bytes routing, falsifier-tested |
| `ogar_codebook.rs` CLASSID_ORDER=CanonHigh + compat | REUSE | the shipped flip; involution proven |
| `rbac.rs` ClassGrant via `classid_canon_compat` | REUSE | canon-keyed grants; no mailbox coupling (correct) |
| `orchestration.rs` StepDomain::Kanban | REUSE | the D-MBX seam's step-routing domain |
| StepDomain::{Crew,N8n} | REPURPOSE | reserved-dormant post-eviction — needs an in-file doc note (doc-only fix) |
| `thinking.rs` 36 styles/τ/FieldModulation | REUSE | canonical style taxonomy (the 36-lens catalogue post-P4) |
| `jit.rs` StyleRegistry/JitTemplate | BLOCKED | n8n-era orphan, zero implementors; successor = the template stack (W3); fold or retire when ExecTarget::Jit is deprecated |
| `a2a_blackboard.rs` | REUSE | Layer-1 per-round expert bus — ephemeral, NOT a singleton sink |
| `cam.rs`, `nars.rs`, `mul.rs` i4_eval, `plan.rs` | REUSE | orthogonal to the ruling; clean |
| **0x1000 adoption monitor (counter/scanner)** | NEW | constants exist; no counting code anywhere in the contract — W6a builds it |

**Key risks:** GATE-1 name collision (two `GateDecision` types, documented
unresolved); "BindSpace" vocabulary in 4+ doc comments describing
per-mailbox-reachable mechanisms.

## 2. SoA ground truth (envelope + tenants)

See `soa_layout/le-contract.md` §5 and `soa_layout/tenants.md` — headline:
the 4+12 facet is CODED; `ENVELOPE_LAYOUT_VERSION = 2`; Full schema = 152 B
of 480 (RESERVE-DON'T-RECLAIM headroom); **`SoaEnvelope` trait has ZERO
production implementors** (MailboxSoA implements the sibling
`MailboxSoaView/Owner` pair; NodeRow reads via the VALUE_TENANTS table);
Meta/Plasticity widths differ persisted-vs-hot with no parity test;
**MailboxId ≠ NiblePath in code** (doc-only claim — ruling needed).

## 3. thinking-engine (the DTO ladder producer)

| Item | Verdict | V3 mapping |
|---|---|---|
| `StreamDto` (Φ) | REUSE | perturbation ingress; standing-async-plan ancestor |
| `dto.rs::ResonanceDto` (Ψ) | REPURPOSE | → `PerturbationDto` (D-PERT-1); blast radius confirmed SMALL: 7 in-crate files + 1 comment-only cross-crate (engine_bridge.rs imports only BusDto) |
| `BusDto` (B) | REUSE | `converged` + `cycle_count` = **the D-MBX-A6 Outcome signal**; never grows ownership fields |
| `ThoughtStruct` (Γ) | EXTEND | persists via owner-stamped envelope (stamping is the boundary's job, not a field) |
| `awareness_dto.rs::ResonanceDto` | REUSE | perspectival (Three-Mountains); KEEPS name; = the Gestalt-resonance MATCHING surface for resonance-based thinking dispatch |
| 4 engines (u8/BF16/i8/f32) | REUSE | near-identical APIs — N→1 collapse candidate (M8) |
| `LayeredEngine::process()` | REUSE | the most complete BusDto producer (3-tier fuse) |
| `DominoCascade` + `CognitiveMarkers` | REPURPOSE | `epiphany` marker = a natural Evaluation→Commit trigger; needs an Outcome adapter |
| `cognitive_stack.rs::GateState` ("collapse gate" doc) | REPURPOSE | intra-cascade SD gate — fine IF intra-mailbox; warden sign-off queued, not assumed |
| `persona.rs::A2AMessage` | BLOCKED | cross-agent handoff shape; warden must rule intra-mailbox (OK) vs cross-mailbox (RESURRECTION) |
| `l4_bridge.rs::commit_to_l4` | BLOCKED | `&mut L4Experience` write with no visible owner stamp — possible ORPHAN-WRITE; needs l4.rs read |
| `cognitive_stack.rs::ThinkingStyle` (12) | RETIRE-toward-contract | a 5th ThinkingStyle copy NOT in the duplication ledger — reconcile before StepMask work (M9) |

## 4. lance-graph-planner + the executor arms

| Item | Verdict | V3 mapping |
|---|---|---|
| `style_strategy.rs` (Strategy #18) | EXTEND | the D-MBX-A6 seam is real and honestly deferred ("faking a KanbanMove would be theatre"); `plan()` proven pass-through; the emit edge is the next slice |
| 18-strategy registry | REUSE | **two natures** (operator ruling): DataFusion-routed query strategies + resonance-based thinking (style template × object Gestalt resonance × rung) — never force the latter through DataFusion |
| `mul/` | REUSE | OUTER per-query f64 gate; distinct from the supervisor's i4 S2 gate; composes cleanly |
| `elevation/` + PatienceBudget | REUSE/EXTEND | per-strategy latency budget ≠ the 550 ms Libet anchor — two budget concepts, unification candidate (M12); the load-balancing wiring (W2d) does not exist yet |
| `physical/collapse.rs::CollapseOp` + Strategy #10 | REPURPOSE (rename) | per-query resonance-dispersion gate — NOT the retired singleton; rename candidate (`ResonanceDispersionGate`) to end the vocabulary collision |
| `cache/` + `serve.rs` | REUSE | autocomplete/thinking cache; serve.rs is LAB surface |
| symbiont `SymbiontBoard` (kanban_loop.rs) | EXTEND | POC proves the loop shape + the −550_000 µs Libet anchor **in code**; gap: `phase` is a bare struct field, not tenant-shaped (W2a); Domino work is called, not free-running — the standing-async-plan is NOT yet realized |
| supervisor `KanbanActor<O>` | EXTEND | the most complete arm: S2 atomic MUL-gate-advance, S3 tick, S4 registry delivery, codex #578/#579 fixes tested; gap: unit-proven on TestBoard only, never integration-driven over a real MailboxSoA |
| **ahead-firing batch writer** | NEW | zero code fires a kanban update at write CAST anywhere (grep-confirmed); lands as a new module wrapping BusDto commits, reading `envelope.mailbox_owner()` |
| **delegation cache** | NEW | no delegation concept exists anywhere; small keyed cache inside the batch writer |
| surreal_container | BLOCKED→flip-on | **coordinates RESOLVED 2026-06-16**; remaining block is the deliberate cold-build gate (`BlockedColdBuild`) — arm #2 is one dependency-uncomment away, at a ~10 min cold-build cost |
| `SurrealMailboxView` (read glove) | EXTEND | compiles today, contract-only; read-side of arm #2 is ready |

## 5. Templates + oracle (W3 stack)

| Item | Verdict | V3 mapping |
|---|---|---|
| elixir-template DSL (7 OgarAction variants + Custom) | REUSE | deterministic parser; `source_ranking_v1` is the first vertical slice |
| template-runtime ReflexExecutor | REUSE | real dispatch, linear-only — **no control-flow type exists** |
| template-equivalence (Exact/RankOrder live; Semantic fail-closed) | REUSE | the W3 merge gate; honest deferred grading |
| cognitive-compiler ScaffoldCompiler | REUSE | "no trace → no template" enforced in types; synthesis = first probe |
| `FieldMask` (class_view.rs:69) | REUSE | the sibling StepMask mints next to (u64 position bitmask) |
| `StepMask` | BLOCKED/NEW | zero .rs matches — docs-only; D-V3-W3a |
| graph-flow NextAction | REUSE | **6 variants incl. no-op GoBack** (docs said 5 — corrected) |
| **NextAction ↔ OgarAction "1:1"** | CORRECTED | the honest 1:1 is `Step ↔ Task` and `ogar_name() ↔ Task::id()`; Continue/WaitForInput/GoTo/End have NO template-side counterparts — closing this IS the StepMask/adapter work, see compiled-templates.md correction |
| rs-graph-llm `template-task` crate | EXTEND | the adapter's natural home — Task shims exist with literal placeholders; ZERO Cargo dep on lance-graph yet (episodic-arc-task proves the git-dep pattern) |
| `Session` (graph-flow) | EXTEND | 5 fields, no ownership — `mailbox_owner` field = the D-V3-W3b site; SurrealSessionStorage = natural persistence (same kv-lance fork) |
| Rig oracle artifacts | EXTEND | PromptResponse + ToolCall + Extractor exist; SourceSpan-shaped provenance from live runs is the unbuilt half of D-V3-W3c |
| graph-flow-action-ogar `GatedOgarHandler` | REUSE (disambiguate) | a SECOND "OGAR" (contract ActionDef + RBAC/MUL gate) — never conflate with elixir OgarAction (M13) |

## 6. Shader / convergence / foundation

| Item | Verdict | V3 mapping |
|---|---|---|
| `engine_bridge.rs::dispatch_busdto` (:239) + `persist_cycle` (:739) | BLOCKED→W4a | THE cast-pairing call sites; write the legacy singleton today, correctly grandfathered until the batch writer (W1) exists |
| `unbind_busdto` (:341) | EXTEND | already feature-gated for the mailbox cutover — the read side is V3-shaped |
| `bindspace.rs::BindSpace` + its call sites (driver.rs, serve.rs, bins, lib.rs re-exports) | RETIRE (W7) | the primer §6 singleton; parity test (mailbox_soa.rs:1145) is the deletion gate |
| `mailbox_soa.rs::MailboxSoA<N>` | REUSE | the shipped successor: soa_view traits, phase field, WriteCell/write_row; needs only the caller-side owner stamp |
| p64-bridge | REUSE | pure stateless mapping, zero BindSpace/ownership deps — clean by design |
| ndarray `MultiLaneColumn` | REUSE (HW) | the LE lane carrier; repurpose opportunity: back MailboxSoA's `Box<[u64]>` identity planes (64 B-aligned) |
| ndarray hpc (fingerprint/cascade/blackboard/read_bgz7) | REUSE (HW) | hardware-tier; causal_diff.rs "self-reinforcement LoRA" flagged for a leak-check read |
| ladybug-rs bind_space + CogRedis | RETIRE (their repo) | pre-V3; zero SoaEnvelope adoption confirmed; contract pulls only (W5e) |

## 7. Consumers (q2 / MedCare-rs / woa-rs / openproject-nexgen-rs / OGAR / smb-office-rs)

Full audit: `soa_layout/consumer-map.md`. Digest:

| Consumer | Classid | Writes | Verdict |
|---|---|---|---|
| openproject-nexgen-rs | T1 reference quality (pure re-exports; bit math only in round-trip tests) | none (pre-persistence) | REUSE |
| MedCare-rs | T1 (auth via rbac membrane, never hardcodes) | none yet — `medcare-soa` writer forthcoming | REUSE + **born-stamped gate (W5h)** |
| OGAR | T2 (the canonical composer) | codegen text only | REUSE + **emit.rs 3× `as u16` post-flip mislabel (W5g)** |
| q2 | REUSE (osint-bake, BodyV3 dual-alias = fleet-best patterns) + T3 interim (cpic, fma) | bakes only — BOOTSTRAP-OK; **`data/osint-v3` codebook is stale pre-flip (latent, W5i/M22)** | REUSE/REPURPOSE |
| woa-rs | T3 unwired module (zero callers; Phase-3 doc points at render_classid) | none (MySQL writer-parity world) | EXTEND |
| smb-office-rs | T4 — never entered the classid ladder (EntityKey strings) | **ORPHAN-WRITE: live `LanceConnector::upsert`, no stamp/classid/envelope — W5f, the first live migration** | BLOCKED |

Cross-cutting: no write-on-behalf adoption anywhere (expected — the batch
writer doesn't exist); the "all consumer writes are bakes" assumption was
FALSIFIED by smb-office-rs (write-on-behalf.md corrected accordingly).

---

## Cross-cutting verdict summary

- **REUSE dominates** — the ruled model is mostly wired-not-invented, as
  the convergence doctrine predicted. The load-bearing NEW pieces are
  exactly three: batch writer + delegation cache (W1), board-as-tenant
  type (W2a), StepMask + control-flow closure (W3a/b).
- **The three honest gaps** between doc and code: SoaEnvelope trait
  unimplemented in production; MailboxId≠NiblePath; NextAction↔OgarAction
  not 1:1. All three are now recorded where the next session will look.
- **Vocabulary debt is the top footgun class**: CollapseGate×3,
  GateDecision×2, BindSpace×3, OGAR×2, Blackboard×2, ThinkingStyle
  12-vs-36 — see ENTROPY-MILESTONES for the collapse ledger.
