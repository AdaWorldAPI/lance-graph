# Integrated Cognitive Planner — meticulous reference map (v1, 2026-06-15)

## READ BY / AGENT TARGET (mandatory)

This is THE reference doc for the integrated-cognitive-planner arc. Any savant
(expansion) or brutally-honest (hardening) agent **MUST cite this map by
`file:line`** and may not assert architecture not grounded here — if a claim
isn't in §1/§7, it's a hypothesis, not a fact. Grounded by a 5-agent research
sweep (2026-06-15, integrated-planner axes) + a 3-agent external-pattern sweep
(Google Pinpoint / TiKV / addressing). Capture-before-dilution of a long design
session (post-#495 merge; #495's `ValueSchema`/`EdgeCodecFlavor` ride #496).

**Verdict of the research:** ~90% of the substrate EXISTS across #437–#492 +
the unmerged `jolly-cori-clnf9`. The remaining work is **6 small additive
seams + the addressing/ScopedReference + the cognitive-cycle sequencer** — NOT
a new build. The recurring failure mode this doc prevents: re-deriving what
already ships.

**Expansion pass (2026-06-15):** 5 savants deepened §1–§7 (each `file:line`-grounded)
and CORRECTED three doc errors — see §8 (cross-savant synthesis + the 2 tensions
T1/T2 + the 7-item additive new-code ledger), §2.1 (ExecTarget routing), §3.1
(causal-arc persistence), §4.1 (0-friction strategy↔step table). Corrections folded
inline: `cache/convergence.rs` is wired+tested NOT stubbed (seam #5); emit cutover is
5 sites incl. the contract twin (seam #1); the dep-wall fold is A-then-B NOT either/or
(seam #2); `documentid`=`dn_hash` NOT `local_key` (§3); the `NiblePath`↔prefix bijection
overflows 16-nibble `MAX_DEPTH` and must be tier-structured (§3).

**Hardening pass (2026-06-15):** 3 brutally-honest agents (PP-13 HOLD / PP-15 CATCH-LATENT
/ PP-16 READY-TO-DISPATCH) — verdicts + 5 LOCKED decisions + the latent boundary fixes are
in **§9**, and the operator's anti-invention guardrail is now **§0 (READ FIRST)**. Net: the
plan is dispatch-ready once §9's spec-text fixes land in the §8 ledger; no architectural
rewrite. Biggest catches: emit channels are separate-not-derived; `cycle()` must stay inherent
(object-safety); seam #2's read is closure-injected (planner can't reach async `at_version`);
the DUAL `RungLevel` must not be re-derived; "#495 rides #496" is mis-attributed (branch-only).

---

## §0 — ANTI-INVENTION GUARDRAIL (operator-locked 2026-06-15, READ FIRST)

**No agent invents new skewed SoA properties. We already have good, well-tested ideas; transcodes map ONTO them, never extend them.** For any work targeting this plan:
- The value slab is the **9 `ValueTenant`s** (`canonical_node.rs`); the node is **`key|edges|value`** (512 B, locked CANON); the **four BindSpace columns** (FingerprintColumns / QualiaColumn / MetaColumn / EdgeColumn) are **closed**. New capability lands as a **new column / new class**, never a new layer / struct / skewed field (AGI-as-SoA PR #223; RESERVE-don't-reclaim CANON).
- Before reaching for any new carrier, apply I-VSA-IDENTITIES **Test-0 (register laziness):** if the thing has a name / enum / id, use the register (HashMap / enum / class) — NOT a fancy property.
- The **§8 7-item additive ledger is the MAX scope.** Anything beyond it — a new tenant, a new column, a new struct, a "skewed" field — requires explicit operator sign-off. An agent that wants one **STOPS and surfaces it**, it does not implement it.
- Specialisation is **opt-IN** (`TD-VALUESCHEMA-FULL-POC-DEFAULT`: the POC FULL default): a consumer mints a class to go smaller/denser; it never adds a property to the shared slab.
- Enforced by `dto-soa-savant` + `iron-rule-savant` (epiphany council) and the 3 pre-merge hardeners (§9). PP-16's top catch — "don't duplicate `should_elevate`, it already ships in thinking-engine" — is this guardrail in action.

---

## P0 — The architecture (4 layers). FORGET LADYBUG.

```
 SurrealDB  (BUILDING BLOCK — orchestrates the META AST/Elixir level; GOOD at it)
   │  returns the AST via the contract → ExecTarget::{SurrealQl, Elixir}
   │  NEVER thinks, NEVER writes. PROJECTS a read-view (kanban time-series).
   ▼
 lance-graph-planner :: PlannerAwareness   = the SHARED COORDINATION contract
   │  orchestrate → ThinkingContext (style / rung / ExecTarget)
   │  emits Vec<KanbanMove>(ExecTarget); rubicon-DAG lifecycle
   │  DELEGATES the thinking ▼            (it COORDINATES; it does not think)
   ▼
 thinking-engine → P64 → cognitive-shader-driver   = THE COGNITION
   │  Φ StreamDto → Ψ ResonanceDto → B BusDto  (thinking-engine/src/dto.rs)
   │  via p64-bridge convergence; driver.rs:581 dispatch + engine_bridge.rs:29
   │  ⚠ NOT `StepDomain::Ladybug` — that is legacy ladybug-rs (forget it;
   │    only residue is in lance-graph-cognitive, minimal impact).
   ▼
 ractor (lance-graph-supervisor) DRIVES the KanbanMove lifecycle
   │  MailboxSoaOwner::try_advance_phase (Planning→CognitiveWork→Evaluation→…)
   ▼  on Commit
 lance-graph-callcenter WRITES = the OUTER BOUNDARY
      lance_membrane.rs:315 commit_event → audit_sink/lance_sink.rs:292 (RocksDB/kvs-lance/ORM)
 SurrealDB PROJECTS the kanban columns read-only.

 temporal-aware throughout: Lance-version read-as-of + temporal.rs deinterlace (HLC).
```

---

## §1 — GROUNDED CURRENT STATE (what EXISTS, `file:line`)

**Planner core**
- `lance-graph-planner/src/lib.rs:99` `PlannerAwareness { strategies, selector }`; entry points `plan_full` (:171), `plan_auto` (:264); impls `OrchestrationBridge` (`orchestration_impl.rs`).
- 16 strategies (lib.rs:21 table), `selector.rs`, `compose.rs`, `pipeline.rs` (DAG), `cache/` (autocomplete), `ir/` (Polars-arena `LogicalPlan`, `ir/mod.rs:92`).
- `strategy/style_strategy.rs` (Strategy #18, **D-MBX-A6-P3a**): `reliability_of` (:168) over recipe kernels; `plan()` (:138) is a deliberate **pure pass-through — emits NO KanbanMove** (:144-151, "faking one here would be theatre").
- `PlanResult.emitted_edges: Vec<u64>` (lib.rs:123) — **always `Vec::new()`** at every construction site (:216/254/312 + api.rs).

**Temporal (the deinterlace = "klares zeitliches Bewusstsein")**
- `lance-graph-planner/src/temporal.rs` — 4-frame deinterlace (lance-version / surrealql `knowable_from` / ractor `V_ref` / cognitive trajectory) via HLC: `EpistemicMode{Strict,Aware,Retro}`+`for_rung` (:63-87), `TemporalStatus` (:90), `QueryReference{server_id,ref_version,hlc_tick,mode,rung}`+`::at` (:107-138), `classify` (:147), `DependsClosure`/`classify_ready`/`NoDeps` (:218-269), `deinterlace` (:301). Tested; **ZERO non-test callers** (only `temporal.rs` + lib.rs doc reference it).
- Lance read-as-of lives on the OTHER side of a dep wall: `lance-graph/src/graph/versioned.rs:419` `VersionedGraph::at_version → checkout_version` (real, tested). Planner deps `contract` NOT `lance-graph` core (anti-circular) → cannot reach it.
- Granger is a SEPARATE module: `lance-graph-cognitive/src/search/temporal.rs::granger_effect`.
- `prediction/{mod,scenario,temporal,ingestion}.rs` = simulated-time NARS rounds (abstract, not Lance versions).

**Kanban / ExecTarget (contract, #437 A6-P1 + #439 A6-P2, both merged)**
- `lance-graph-contract/src/kanban.rs:32` `KanbanColumn` Rubicon DAG (`next_phases`/`can_transition_to`/`is_absorbing`); `:112` `KanbanMove{mailbox,from,to,witness_chain_position,libet_offset_us,exec}` (`Copy`, ≤16 B); `:136` `ExecTarget{Native,Jit,SurrealQl,Elixir}`.
- `soa_view.rs:112` `MailboxSoaOwner::try_advance_phase` → `KanbanMove` | `RubiconTransitionError`.
- `scheduler.rs:46` `NextPhaseScheduler::on_version`; concrete `LanceVersionScheduler` deferred (scheduler.rs:20).

**Escalation / NARS / MUL (the cycle primitives)**
- `escalation.rs`: `CollapseHint{Flow,Fanout,RungElevate}` (:45), `fanout_width` (:59), `rung_delta(emergence,coherence)` (:75), `InnerCouncil::deliberate` (:144, 3 archetypes + split-amplify), `EpiphanyDetector::observe` (:241, surprise>baseline×1.5 ∧ window≥4), `GhostEcho{…,Staunen,Wisdom,…}` (:289), `WisdomMarker` (:312, decays to FLOOR 0.1). **Consumed** by `planner::mul::escalation`.
- `nars::InferenceType{Deduction,Induction,Abduction,Revision,Synthesis}`; carried on `ThinkingContext.inference_type` (plan.rs:19).
- `mul/` `MulAssessment` (Dunning-Kruger / trust / compass / homeostasis).
- `cognitive_shader.rs:157` `RungLevel(0..9)`; `ThinkingContext.rung` (plan.rs:22).

**Cognition pipeline (thinking-engine > P64 > shader-driver)**
- `thinking-engine/src/dto.rs`: Φ `StreamDto` (:40), Ψ `ResonanceDto` (:59), B `BusDto` (:120). (ResonanceDto rename in progress — `TD-RESONANCEDTO-DUP-1`.)
- `p64-bridge/src/lib.rs` convergence (CausalEdge64/palette).
- `cognitive-shader-driver/src/driver.rs:581` `CognitiveShaderDriver::dispatch`; `engine_bridge.rs:29` wires thinking-engine ↔ ShaderBus; deps planner (lab) + p64-bridge + (opt) thinking-engine.

**The loop (jolly-cori-clnf9 ONLY — unmerged)**
- `cognitive-shader-driver/src/mailbox_soa.rs:349/397` `impl MailboxSoaView + MailboxSoaOwner for MailboxSoA<N>` + driving test `:648` (`463d71bd`, +149 LOC). On main/this branch `MailboxSoA` has **no owner impl** → loop can't run in-tree.

**Surreal + callcenter write**
- Contract correctly declares surreal=project-read-only, callcenter=commit (kanban.rs:1-21).
- `lance-graph-callcenter/src/lance_membrane.rs:315` `commit_event` (sole writer, version tick) → `audit_sink/lance_sink.rs:292` real Lance `InsertBuilder…Append`.
- `surreal_container` ~all stub, BLOCKED on lance-7 fork (`TD-SURREALDB-KVLANCE-LANCE7`).

**Canonical node + addressing (contract; #489/#490 + #495-mine)**
- `canonical_node.rs:35` `NodeGuid` = `classid(u32)|HEEL(u16)|HIP(u16)|TWIG(u16)|family(u24)|identity(u24)`; `local_key()` (:106) = trailing 6 B; zero-fallback ladder. `EdgeBlock` (:181) 12+4. `EdgeCodecFlavor` (:207, #495-mine). `ValueSchema`/`ValueTenant`/`VALUE_TENANTS` (#495-mine — unmerged, rides #496).
- `hhtl.rs` `NiblePath` radix walk (`child`, `is_ancestor_of`).

---

## §2 — THE 6 SEAMS (gap → tracking → FOLD `file:line`)

1. **Planner emits `KanbanMove`** — gap: `emitted_edges` always empty; planner imports none of `kanban`/`soa_view`. Tracking: **D-MBX-A6-P3**. FOLD (S2): the inputs ALREADY exist — `StyleStrategy::reliability_of` (style_strategy.rs:168) already computes the settled `ThoughtCtx` but discards all but `confidence` (:178); refactor to `settle() → ThoughtCtx` keeping `dissonance` (recipe_kernels.rs:31) + `rung` (:37). Adapter `thoughtctx_to_kanban_move` (planner-side, pure): `exec` per §2.1; `to` from `tc.gate_state()` (recipe_kernels.rs:58, expose it) vs `KanbanColumn::can_transition_to`; `libet_offset_us=-550_000` iff `to==CognitiveWork`; `witness_chain_position=0` (no arc yet). **Cutover = 5 sites, NOT 4: lib.rs:123/216/254/312 + api.rs:190 + the CONTRACT TWIN plan.rs:44** (skipping it diverges the `PlannerContract` trait). Add `emitted_moves: Vec<KanbanMove>` beside `emitted_edges` as a **SEPARATE channel, NOT derived** (PP-13/PP-15 §9: `emitted_edges` keeps carrying CausalEdge64/EpisodicEdges64 words from the collapse-gate path; there is NO `KanbanMove→u64` cast — inventing one is the I-LEGACY trap) → resolves OQ-11.7. **Missing input: `KanbanMove.mailbox` (kanban.rs:114) — the planner has NO mailbox; add `mailbox: Option<MailboxId>` to `PlanContext` (traits.rs:69), caller-supplied, sentinel default.**
2. **temporal.rs unconsumed + dep-wall** — gap: `deinterlace`/`QueryReference` 0 callers; planner ⊥ lance-core. FOLD (S4 — two layers, BOTH required, ordered, NOT either/or): **(A)** relocate the *policy* `temporal.rs` → zero-dep `contract::temporal` (`classify`/`deinterlace`/`EpistemicMode` name no lance/arrow/async; `DeinterlaceRow` is the airgap, mirroring scheduler.rs:31's `MailboxSoaView`) so BOTH planner+core `use contract::temporal::*`; **(B)** add `lance-graph/src/graph/temporal_read.rs` — the ONLY site joining `VersionedGraph::at_version(T)` (versioned.rs:419) → `deinterlace` (temporal.rs:301), in core where both are nameable. A-alone never feeds real rows to `deinterlace`; B-alone can't share `classify` with planner. Add ONE field `as_of: QueryReference` to `PlanContext` (traits.rs:69) — NOT two loose `ref_version`/`rung` (rung lives inside `QueryReference`, temporal.rs:121, drives `mode` via `for_rung`).
3. **Loop only on jolly** — gap: `MailboxSoaOwner for MailboxSoA` is +149 LOC on `463d71bd`, unmerged. FOLD: cherry-pick `mailbox_soa.rs:349-460` + test `:648` (purely additive, traits already on main).
4. **Rung inert** — gap: `RungLevel::Surface` hardcoded (orchestration_impl.rs:151, api.rs:178, pipeline.rs:593(test fixture — leave), thinking/mod.rs:86); `rung_delta` imported never called; Staunen/Wisdom orphaned. **Root cause (S3): `RungLevel` has NO constructor** — no `from_ordinal`/`shift`/`from_entropy` (cognitive_shader.rs:157 is a bare enum), so every site CAN'T do anything but default `Surface`. FOLD (S3): (1) add `RungLevel::{from_ordinal, shift(i8), from_entropy(f32)}` after cognitive_shader.rs:169 + a `PlannerContract::rung_for(&self,&SituationInput)->RungLevel` trait method (default Surface, mirrors `gate_check`, so the contract itself KNOWS the rung). (2) bridge the 3-level `EntropyRung` (ndarray entropy_ladder.rs:64) ↔ 10-level `RungLevel` via `from_entropy(h)=from_ordinal(round((1−h)·9))` — bare `f32`, NO ndarray type crosses the wall (scalar `h` is the wire, like seam #2's `QueryReference`). (3) drive the 2 PRODUCTION sites (thinking/mod.rs:86 has `mul` in scope; orchestration_impl.rs:151): `rung = from_entropy(nars_entropy(f,c)).shift(rung_delta(emergence,coherence))`; rises on sustained `mul::GateDecision::Block`; floored by a live `WisdomMarker` (fresh Epiphany→low rung, decayed→high). (4) bind `ShaderDispatch.rung = think_ctx.rung` (was Surface default, cognitive_shader.rs:219). NB the two `GateDecision` types: planner-contract driver = `mul::GateDecision::Block` (mul.rs:150); `collapse_gate::BLOCK` (collapse_gate.rs:79) is the shader-side echo.
5. **Think-delegation (thinking-engine>P64>shader-driver, NOT Ladybug)** — gap: planner doesn't call shader-driver; driver→planner is lab-only (`planner_bridge.rs`). **Correction (S2): `cache/convergence.rs` is NOT stubbed — it is wired + tested (8 tests, :363-436); only the `#[allow(unused_imports)]` for CausalEdge64/SpoBase17/DistanceMatrix (:22-27) is unwired** (the triplet→palette path works). Carrier ALREADY aligned 1:1: `ThinkingContext.rung` (plan.rs:22) ↔ `ShaderDispatch.rung` (cognitive_shader.rs:190) same `RungLevel`; only new code = a field-copy builder mirroring `dispatch_from_top_k` (engine_bridge.rs:93). FOLD (S2): bridge home = a NEW third crate **`lance-graph-cognitive-cycle`** {planner, shader-driver, p64-bridge} — confirmed by BOTH Cargo.tomls that planner ⊥ shader-driver (planner has no driver dep; driver deps planner only as optional `with-planner`). Route `ThinkingContext → ShaderDispatch(rung) → CognitiveShaderDriver::dispatch (driver.rs:581) → ShaderBus.emitted_edges[..count] (cognitive_shader.rs:353) → PlanResult.emitted_moves`. **Never `StepDomain::Ladybug`.** (TENSION with S1's "cycle is a method ON PlannerAwareness" — see §8.)
6. **Write mis-framing (doc)** — gap: `plan.rs:42-44` "the vart/surreal seam persists". FOLD (zero-code): → "callcenter (`commit_event` → `LanceAuditSink`) calcifies; surreal projects read-only." Hardenable further (S2 OQ-11.8): a `const fn ExecTarget::can_drive(self, to:KanbanColumn)->bool` makes "Surreal never writes/thinks" a COMPILE-checked invariant (bars `SurrealQl`+`Commit`/`CognitiveWork`), extending the `MailboxSoaView`/`Owner` split from SoA-access up to plan-routing.

---

## §2.1 — ExecTarget routing (AST ↔ Elixir ↔ SurrealQl ↔ Native) — S2

The operator's law: **Surreal RETURNS the AST via the contract; the planner ROUTES `ExecTarget` (kanban.rs:136); callcenter `commit_event` (lance_membrane.rs:315, sole version-ticking writer, the Rubicon `CommitHook` seam :301) WRITES via Lance. Thinking + writing NEVER through Surreal.** Decision rules, in the §2-seam-1 adapter reading the settled `ThoughtCtx`:

- **`Jit`** — style `tau()` resolvable (style_strategy.rs:15: `τ → JitTemplate → KernelHandle`) + high `tc.confidence` + low `tc.dissonance`. The kernel hot path.
- **`Elixir`** — the interpreted `recipe_kernels` layer (style_strategy.rs:21); the deliberative DEFAULT for a thinking step not JIT-warmed (cold, or high `tc.dissonance` needing interpreted tactics).
- **`SurrealQl`** — a pure read-projection over the kanban SoA columns ONLY (kanban.rs:143 "lowered to SurrealQL and run in the substrate"). **Structurally barred from mutation:** `surreal_container` implements `MailboxSoaView` (soa_view.rs:11) NOT `MailboxSoaOwner` (:107-112) — it can never drive a `Commit`. A thinking/writing step that picked `SurrealQl` would be a category error the trait system already forbids — this is seam #6 made concrete.
- **`Native`** (default, kanban.rs:138) — in-process graph traversal.

**The AST flow mapped:** Surreal RETURNS the AST → through the contract surface `OrchestrationBridge::route` (orchestration_impl.rs:47) consuming a `UnifiedStep`; planner ROUTES → `ExecTarget` selection (the four rules); callcenter WRITES → `LanceMembrane::commit_event` (lance_membrane.rs:315) on `KanbanColumn::Commit` (the absorbing terminal). Lance writes its own bytes (audit_sink/lance_sink.rs:292); Surreal is nowhere in this write path.

**`BusDto` is the accountable delegation packet** (thinking-engine/src/dto.rs:120, "the first accountable structured thought"): peak `energy` (support) + `top_k` (spread/dark-horse) + `ShaderBus.gate` (cognitive_shader.rs:356) + `tc.dissonance` are SEPARATE channels — support ≠ contradiction in one scalar. Indices, never strings (text is LAZY at Γ `ThoughtStruct`, dto.rs:132). Round-trip proven by `dispatch_busdto`/`unbind_busdto` (engine_bridge.rs:231/310). The planner→driver delegation returns the `BusDto` + `emitted_edges[..count]`; no prose crosses the seam.

---

## §3 — ADDRESSING: `identity / ScopedReference / (hhtl-guid):path:documentid`

Resolves left→right, mirroring `NodeGuid`:
```
(hhtl-guid)                 : path                       : documentid
classid|HEEL|HIP|TWIG       : VALUE_TENANTS offset        : dn_hash (content) ⇒ leaf local_key
= routing prefix (radix)    : intra-row value-slab path   : dedup key, resolves the entity-MID
↔ TiKV Region-prefix / Pinpoint collection              ↔ Pinpoint document-id (content-addressed)
```

**Three DISTINCT hashes — the doc previously conflated two (S5 correction):**
- **identity / local_key** = the *arc anchor*: `local_key()` (canonical_node.rs:106) = trailing 6 bytes (family ++ identity), a MINTED stable address (uniqueness-guarded, canonical_node.rs:139), **never a content hash**. Survives every revision — Pinpoint's alias→Knowledge-Graph-MID collapse (one stable id per entity; surface forms resolve to it). This is the entity-MID.
- **documentid** = the *dedup key*, a DISTINCT object from the leaf: the live `dn_hash(canonical_dn(triplet)) → u64` (fingerprint.rs:65, used at spo_bridge.rs:127). Same content → same key = Pinpoint dedup, **ALREADY shipped** (not an unbuilt ADOPT). The address resolves a row by `local_key`; dedup happens via `dn_hash` at ingest. Complementary, not the same field. (If you content-hashed the identity, every revision would mint a new address and shatter the arc — the anti-pattern.)
- **shape_hash** = a THIRD hash, `StructuralSignature(u32)` (class_signature.rs:31, FNV-1a over field/method structure) — the shape-family / ground-truth-drift key. Canon dropped it from the GUID (identity errata 2026-06-13) but drift-detection carries it ALONGSIDE the address, never inside it.

**Bijection width constraint (the real P-SCOPE-CLASSIFY blocker, S5):** the prefix `classid|HEEL|HIP|TWIG` is bytes 0..10 of `NodeGuid` = 10 bytes = **20 nibbles**, but `NiblePath` (hhtl.rs:55) holds only `MAX_DEPTH = 16` nibbles in its `u64` (hhtl.rs:45). A flat byte→2-nibble lowering OVERFLOWS the path word. The bijection MUST be tier-structured: the 3 canon cascade tiers (HEEL/HIP/TWIG, canonical_node.rs:28) each contribute a *bounded, cache-allocated* nibble run, basin nibble resolved from the DOLCE-from-cache binding (hhtl.rs:18-23), never a raw classid byte. Proposed `NiblePath::from_guid_prefix(&NodeGuid, TierWidths) -> Option<NiblePath>` next to `from_packed` (hhtl.rs:199), `None` when depth exceeds `MAX_DEPTH`. `is_ancestor_of` (hhtl.rs:176) is a base-16 prefix-equality test = TiKV `[start,end)` range-containment (reflexive ⇒ closed-start; empty path = "no region assigned" sentinel). Proven by `is_ancestor_of_is_cheap_prefix_reachability` (hhtl.rs:387).

- **ScopedReference** (the genuinely NEW piece; "ticket" — but **NOT** named `ticket`: collides with `grammar::ticket::FailureTicket`) = `(NiblePath scope, QueryReference as-of)` = a TiKV-TSO snapshot-handle scoped to a key-range = "this subtree, as-of Lance version T". `QueryReference::at(ref_version,rung)` is already the as-of half. **Retrieval shape (additive, S5):** `admits(row,path,knowable,deps) = scope.is_ancestor_of(path) (hhtl.rs:176, range-containment) && classify_ready(...).dispatchable(mode) (temporal.rs:258/249)`; `retrieve_arc(rows,deps) = deinterlace (temporal.rs:301, TIME∧DATA filter + HLC sort) with the is_ancestor_of pre-filter`. A METHOD on the carrier (Click litmus: accept). **Snapshot-isolation (P-TICKET-SNAPSHOT) is FREE not built (S4):** two reads at the same `Copy` handle call `at_version(ref_version)` (versioned.rs:419); Lance versions are immutable → byte-identical batches regardless of concurrent head ticks. The handle pins the version; the version pins the bytes; Lance's versioning IS the MVCC.
- **Bardioc's** "which export-restriction at data-window T" = `deinterlace(rows, QueryReference::at(T,rung), deps)` (temporal.rs:301) → rows Contemporary-at-T. **But the marking carrier is MISSING (S4, highest-value gap):** `Marking{Public,Internal,Pii,Financial,Restricted}` exists (property.rs:771) but is schema-level/static (`&'static`, `const fn with_marking`) — NO version axis, so it cannot answer "which held at T"; `CognitiveEventRow` (external_intent.rs:113) carries no marking. **Minimal additive seam:** emit a versioned `MarkingRow{predicate, Marking, knowable_from: LanceVersion}` into the SPO store on each classification change; `impl DeinterlaceRow for MarkingRow` → `deinterlace` returns the marking `Contemporary` at T. `Marking` + `DeinterlaceRow`, both exist, joined with zero new machinery. Route: **read-as-of, NEVER counterfactual** (Bardioc asks "what *was*", not "what *would be*" — the latter = `counterfactual.rs`, the wrong surface for audit).

**ADOPT** — TiKV: prefix-routes-to-placement (`NiblePath::is_ancestor_of` = range-containment); snapshot-as-handle (never implicit "latest"); coprocessor pushdown ↔ `DependsClosure`/`deinterlace` filter-at-source; a **batched monotonic ticket-oracle = TSO over Lance versions** (but keep our decentralized HLC `Option<u64>`, no central oracle). Pinpoint: entity-MID = `identity`; cross-doc entities = `EdgeBlock`/`MaterializedEdges` (entities are EDGES between leaves, not a sidecar index); content-addressed `documentid` (free dedup) — **already implemented as `dn_hash` (spo_bridge.rs:127); the seam is wiring it into the `NodeGuid` address, not building it**; as-of-T first-class from day one (their post-2023-08-01 date-epoch trap = our `I-LEGACY-API` violation to avoid).
**DON'T** — Pinpoint filename-as-id / no-dedup / no-versioning; TiKV central TSO bottleneck.

### §3.1 — Causal-arc persistence (the revision-aware memory mechanism, S5)

A causal arc = stable anchor + append-only adjacency. **Anchor:** `local_key` (canonical_node.rs:106), bytes 13..16 never move (RESERVE-don't-reclaim, canonical_node.rs:16). **Adjacency:** `EpisodicEdges64` (episodic_edges.rs:103), 4 MRU slots, slot 0 = hottest. A revision is `promote(e)` (episodic_edges.rs:168): re-rank, never overwrite; the coldest (`coldest()`, :227) demotes via `DemotionSink::demote` (:311) to the cold connectome — NOT deleted. **Recency = slot index; causality = demoted history + co-addressed `CausalEdge64` plasticity** (:163) — separate channels (resolves the recency≠causality anti-pattern). **Stabilization:** an `Episode` (episodic.rs:43) unbundles into individually-addressable facts only at `truth.confidence ≥ UNBUNDLE_HARDNESS_THRESHOLD = 0.8` (episodic.rs:166); `rebundle_cold` (:273) retires per-triplet addressability for aged low-confidence episodes but keeps them fingerprint-searchable. Visible-after-revision: the fingerprint bundle always; per-triplet facts only while hot + confident ≥ 0.8. **As-of replay makes revision non-destructive:** versions are append-only; `ScopedReference(scope, QueryReference::at(T))` reconstructs the arc as knowable at T (`classify → Contemporary`, temporal.rs:150), excluding later revisions (`Anachronistic`, dropped under Strict, :153). The arc at any past version is permanently reconstructable.

---

## §4 — THE COGNITIVE CYCLE (8 steps → primitives → Rubicon phases)

| Step | Primitive (`file:line`) | Rubicon phase |
|---|---|---|
| fanout | `escalation::CollapseHint::Fanout` + `fanout_width` (:45/:59) | Planning |
| consolidate | `escalation::InnerCouncil::deliberate` (:144) | Planning |
| induction | `nars::InferenceType::Induction` | Planning→Σ |
| synthesize insights | `escalation::EpiphanyDetector::observe` (:241) + `InferenceType::Synthesis` | CognitiveWork |
| think | `CognitiveShaderDriver::dispatch` (driver.rs:581; thinking-engine>P64) | CognitiveWork ← seam #5 |
| deduction | `nars::InferenceType::Deduction` | CognitiveWork |
| meta awareness | `mul::MulAssessment` (DK/trust/compass) | Evaluation |
| abduction | `nars::InferenceType::Abduction` | Evaluation→{Commit\|Plan\|Prune} |

Spiral: abduction → `KanbanColumn::Plan` → re-deliberate → next fanout (Peirce abductive-inductive-deductive loop, gated by EpiphanyDetector synthesis + MUL anti-Mount-Stupid). **Add = a `CognitiveCycle` sequencer** (method on the integrated Planner) that drives the 8 steps through the kanban phases, setting `ThinkingContext.inference_type` per step. Everything it calls EXISTS; the sequencer is the only new code — and it is the consumer that closes seams #1 (emit per phase), #2 (as-of read per step — but the read is CLOSURE-INJECTED, see §9: the planner can't reach the async `at_version` in core), #4 (rung drive), #5 (think delegation). **The spiral re-entry is a GROUNDED legal Rubicon edge (S1):** `KanbanColumn::Plan.next_phases() == &[Planning]` (kanban.rs:95) and `Plan.is_absorbing() == false` (kanban.rs:71) — `Plan → Planning` already compiles; P-CYCLE-SPIRAL is its only falsifier.

**Fields vs methods (the load-bearing call, S1):** the sequencer adds exactly THREE state fields to `PlannerAwareness` (lib.rs:99) — `phase: KanbanColumn` (default Planning), `checklist: Checklist` (built by the existing `mul::escalation::boot_checklist()`, mul/escalation.rs:36), `epiphany: EpiphanyDetector` (escalation.rs:213). `mul`/`thinking`/`elevation`/`cache` stay STATELESS module calls (lib.rs:177/183 construct-and-drop) — they do NOT become fields. The 8 steps are `&mut self` methods; the public entry is one new `cycle(&mut self, query, situation) -> Result<CycleOutcome, PlanError>` ALONGSIDE `plan_full` (lib.rs:171) / `plan_auto` (lib.rs:264), both UNCHANGED.

**Subsumption (S1):** the integrated planner is `PlannerAwareness` GAINING the cycle (+ a zero-cost `pub type Planner = PlannerAwareness` alias for the operator's literal name) — NOT a rename (would force touching the `OrchestrationBridge` impl orchestration_impl.rs:46 + every `Box<dyn OrchestrationBridge>` consumer) and NOT a wrapper (re-introduces the delegation boilerplate the contract dedup killed). (NB: S2 argues the *think-step* must call into a third crate because planner ⊥ shader-driver — the reconciliation is closure-injection à la `run_convergence`; see §8.)

**Carrier (S1):** the cycle threads the CONTRACT `plan::ThinkingContext` (plan.rs:16), NOT the planner `thinking::ThinkingContext` (thinking/mod.rs:31) — the `inference_type: InferenceType` field the table sets exists only on the contract type (plan.rs:19; the planner type has `nars_type` instead). The `thinking_to_contract` converter (orchestration_impl.rs:163) is the existing bridge.

**Rung threads every step (S3):** each step's `ThinkingContext.rung` (plan.rs:22) is `from_entropy(nars_entropy(f,c)).shift(rung_delta)` — it RISES on sustained `mul::GateDecision::Block` (can't settle → deepen) and is FLOORED by any live `WisdomMarker` (settled insight → dispatch deep). The "think" step carries it into cognition via `ShaderDispatch.rung` (seam #5). Rung is the depth axis of the spiral; entropy (Staunen→Wisdom) is its source coordinate — the same one entropy-ladder-spo-rung-v1 persists as the 2-bit edge class.

### §4.1 — 0-friction strategy↔step collapses (S1)

| Boundary | Verdict | Collapse |
|---|---|---|
| `PlanResult.emitted_edges` (lib.rs:123) ≡ cycle step-5/8 output | **OPPORTUNITY** | same `Vec<u64>`, same doc — the carrier was provisioned before the cycle existed (seam #1) |
| CollapseGate strategy #13 ≡ step-8 abduction terminal | **OPPORTUNITY** | both the Commit/Plan/Prune 3-way (kanban.rs:94); `mul::GateDecision` is a third encoding of the same trichotomy |
| TruthPropagation strategy #12 ≡ steps 3/6/8 NARS setters | **OPPORTUNITY** | same NARS→semiring map (nars.rs:45) |
| StyleStrategy::reliability_of (style_strategy.rs:168) ≡ step-7 council confidence | **WORTH-EXPLORING** | probe OQ-CSV-CYCLE-1 (extends P-RUNG-VARIES) |
| JitCompile strategy #14 ≡ KanbanMove.exec=Jit (kanban.rs:128) | **WORTH-EXPLORING** | probe OQ-CSV-CYCLE-2 (τ-address → ExecTarget) |
| Parse strategies #1-4 ≡ step-1 fanout | **DROP** | syntactic-parse vs search-breadth, no shared algebra |
| WorkflowDAG strategy #15 ≡ Rubicon kanban DAG | **DROP** | dynamic workflow topology vs fixed 6-state automaton |

---

## §5 — PROBES (measure-first; falsifiable, declared before running)

- **P-DEDUP-ASOF**: ingest the same doc at versions T₁,T₂ → assert one `identity` and as-of-T₁ excludes the T₂ copy (collapses Pinpoint's no-dedup + no-versioning into one invariant).
- **P-TICKET-SNAPSHOT**: one `ScopedReference` per session → two reads at the same ticket see byte-identical as-of-T snapshots across split basins (TiKV snapshot-isolation, mapped).
- **P-SCOPE-CLASSIFY**: `ScopedReference` admits a row iff `scope.is_ancestor_of(row.niblepath) && classify(...)==Contemporary` (needs the `NiblePath`↔`classid|HEEL|HIP|TWIG` byte↔nibble bijection — currently unwritten, `hhtl.rs:48`).
- **P-RUNG-VARIES** (exists, `style_strategy.rs:264`): reliability varies by style → a rung/style gate is non-cosmetic.
- **P-CYCLE-SPIRAL**: abduction → Plan → fanout actually changes the next cycle's candidate set (else the spiral is decorative).
- **P-RUNG-FROM-ENTROPY-VARIES** (S3, additive sibling of P-RUNG-VARIES): `RungLevel::from_entropy(1.0 − reliability_of(style, ctx))` (style_strategy.rs:168) must produce DIFFERENT rungs for different styles over the same ctx. If Analytical and Creative land on the same rung, the entropy→rung map is cosmetic and must NOT be wired. Inherits the ρ=−0.78 reliability-proxy grounding (entropy_ladder.rs:281).
- **P-RUNG-ROUNDTRIP** (S3, gated on entropy-ladder R2 / D-EL-2): `RungLevel::from_entropy_class(entropy_class(h))` must land in the same 3-level `EntropyRung` band as `from_entropy(h)` for all h — else the 2-bit `CausalEdge64`-spare-bits persist is lossier than the band semantics allow and needs 3 bits. This is what makes the rung persist across the Lance tombstone WITHOUT a `RungState` singleton.
- **P-ARC-PROMOTE-IS-REVISION** (S5): a sequence of `EpisodicEdges64::promote` (episodic_edges.rs:168) on one `local_key` preserves slot-0 = most-recent and routes exactly `coldest()` to the `DemotionSink` (assert `evicted == coldest()`, episodic_edges.rs:593); then assert the demoted edge is still reachable via the cold connectome AND an as-of-T₁ `ScopedReference` read excludes a T₂-promoted edge. Falsifies "revision destroys the path" — proves recency (slots) and history (sink) are separate channels.
- **P-GRANGER-VERSION-LAG** (S4, HYPOTHESIS): with Granger's `lag` = Lance-version-delta, `granger_effect(series_a, series_b, max_lag)` over `[for V in T-k..T: fingerprint(deinterlace(at_version(V)))]` returns a signal that subtree A's as-of-V state predicts subtree B's as-of-(V+lag) state beyond autocorrelation. Constraint (search/temporal.rs:35/39): ≥3 versions, `max_lag ≤ window/2`.

---

## §6 — OPEN QUESTIONS / DEFERRALS

- OQ-11.6 surreal external trigger (`Notification→KanbanMove` `LanceVersionScheduler`) — fork-blocked (`TD-SURREALDB-KVLANCE-LANCE7`). NB Frame 2 (`knowable_from`) IS Surreal's `DEFINE TABLE` timeline (temporal.rs:280) → the `Unknowable` axis stays dark until the fork lands.
- OQ-11.7 planner DTO cutover scope — S2 makes it concrete: `emitted_moves` is the new truth, `emitted_edges` the derived legacy view (keep both per I-LEGACY-API-FEATURE-GATED).
- The jolly→main merge of the loop (`463d71bd`).
- `temporal.rs` dep-wall — **DECIDED (S4): A-then-B, both required** (policy→contract + a lance-core `temporal_read` join). Residual: where the `at_version→deinterlace` consumer lives (core `temporal_read` vs S5's "callcenter `commit_event` returns a stamped `ScopedReference`").
- ResonanceDto rename (`TD-RESONANCEDTO-DUP-1`).
- `StepDomain::Ladybug` → mark deprecated (forget Ladybug).
- OQ-CYCLE-MUT (S1): `EpiphanyDetector::observe` is `&mut self` (escalation.rs:241) but `route`/`plan_full` are `&self` → interior mutability on the `epiphany` field (RwLock/LazyLock per data-flow.md) OR `cycle()` is the sole `&mut` entry kept OFF the contract trait.
- OQ-11.8 (S2): bind `ExecTarget` legality to `KanbanColumn` — `const fn ExecTarget::can_drive(self, to)->bool` beside `KanbanColumn::can_transition_to` (kanban.rs:102); bars `SurrealQl` from `CognitiveWork`/`Commit` → "Surreal never writes/thinks" becomes a compile invariant (folds seam #6 from doc-fix to type-guarantee). Falsifier: assert `KanbanMove{exec:SurrealQl, to:Commit}` rejected.
- OQ-rung-state (S3): the sustained-BLOCK counter + WisdomMarker age need cycle-to-cycle persistence, but `RungState` (named escalation.rs:39) does NOT exist. Candidate: ride D-EL-2's 2-bit entropy-class on the per-mailbox SoA edge (co-located with `last_active_cycle`), read back via `from_entropy_class` — NOT a new singleton. Also confirm `MulAssessment` exposes the NARS `(f,c)`/emergence/coherence scalars feeding `nars_entropy`/`rung_delta` (mul/escalation.rs:22-27 shows only trust/humility/flow/load — entropy inputs may need deriving).
- OQ-11.9 (S4, the strong second-order): does the cycle build `Think.global_context` via `deinterlace(at_version(T))` — think *as-of-T* (historical cognition gated by `EpistemicMode`), not merely read-as-of-T? If yes → defensible-audit re-thinking (auditor proves a decision used only T-contemporaneous knowledge via `EpistemicMode::Strict`). HYPOTHESIS, grounded in `deinterlace` output "IS the standing wave" (temporal.rs:20) = `Think.global_context` shape.
- OQ-CSV-CYCLE-1/2 (S1, from §4.1): does `StyleStrategy::reliability_of` subsume the step-7 council confidence; does τ-address resolution determine `ExecTarget` (making JitCompile-as-strategy redundant with exec-on-move)?

---

## §7 — REFERENCE INDEX (the file:line grounding agents MUST target)

- Planner: `lance-graph-planner/src/{lib.rs:99/123/171, strategy/style_strategy.rs:138/168, temporal.rs:107/147/301, orchestration_impl.rs:151, traits.rs:69, ir/mod.rs:92}`
- Contract: `lance-graph-contract/src/{plan.rs:16/42/144, orchestration.rs:37/56/390, kanban.rs:32/112/136, jit.rs:48, escalation.rs:45/144/241/289/312, scheduler.rs:46, soa_view.rs:112, canonical_node.rs:35/106/181/207, hhtl.rs, cognitive_shader.rs:157}`
- Cognition: `thinking-engine/src/dto.rs:40/59/120`, `p64-bridge/src/lib.rs`, `cognitive-shader-driver/src/{driver.rs:581, engine_bridge.rs:29, mailbox_soa.rs:349(jolly)}`
- Write boundary: `lance-graph-callcenter/src/lance_membrane.rs:315`, `audit_sink/lance_sink.rs:292`
- Lance versioning: `lance-graph/src/graph/versioned.rs:419`
- External patterns: Google Pinpoint (entity-MID, collection-ACL, no-dedup/no-version traps); TiKV (key→region→store via PD, MVCC/TSO snapshot reads, coprocessor pushdown).
- Emit/cycle (S1/S2): `lance-graph-planner/src/{lib.rs:107/216/254/312, api.rs:190, compose.rs, thinking/mod.rs:31/86, mul/escalation.rs:21/36}`, `lance-graph-contract/src/{recipe_kernels.rs:25/31/37/58/79, plan.rs:44, mul.rs:144/150, collapse_gate.rs:59/79, cognitive_shader.rs:178/190/219/349/353/356/427}`, `cognitive-shader-driver/src/{driver.rs:582, engine_bridge.rs:93/143/231/310, planner_bridge.rs:1(LAB), Cargo.toml:54(with-planner)}`, `lance-graph-planner/Cargo.toml`.
- Rung/entropy (S3): `lance-graph-contract/src/cognitive_shader.rs:155-169`, `ndarray/src/hpc/entropy_ladder.rs:55/64/99/208/281`, `.claude/plans/entropy-ladder-spo-rung-v1.md`.
- Temporal (S4): `lance-graph-planner/src/temporal.rs:67/107/121/135/147/249/258/301`, `lance-graph/src/graph/versioned.rs:419/450/461`, `lance-graph-cognitive/src/search/temporal.rs:30`, `lance-graph-callcenter/src/{version_watcher.rs:57, external_intent.rs:113}`, `lance-graph-contract/src/{property.rs:771, scheduler.rs:46}`.
- Addressing/episodic (S5): `lance-graph-contract/src/{hhtl.rs:45/55/72/93/176/199/235/387, canonical_node.rs:16/106/139/329/351, episodic_edges.rs:103/168/227/311/593, soa_envelope.rs:87/107}`, `lance-graph/src/graph/{fingerprint.rs:65, arigraph/episodic.rs:43/161/166/273, arigraph/spo_bridge.rs:127}`, `lance-graph-ontology/src/odoo_blueprint/class_signature.rs:31`.

---

## §8 — CROSS-SAVANT SYNTHESIS (2026-06-15, 5-savant expansion pass)

Five expansion savants (S1 convergence-architect / S2 bus-compiler / S3 truth-architect / S4 scenario-world / S5 trajectory-cartographer) deepened §1–§7, each citing `file:line`. Net: the integrated planner is confirmed **~90% existing**; the new code is small + additive. This section holds what only emerges from holding all five together — the hardeners (3 brutal agents) bite HERE first.

**TENSION T1 — where the cycle lives (S1 ⊥ S2, must be resolved before D-MBX-A6-P3 impl).**
S1: the `CognitiveCycle` is a method ON `PlannerAwareness` (+ `pub type Planner` alias) — no new crate. S2: planner ⊥ shader-driver (proven by BOTH Cargo.tomls) → the *think-step* (seam #5) can't call `CognitiveShaderDriver::dispatch` from inside the planner crate, so it needs a third crate `lance-graph-cognitive-cycle`. **Reconciliation (recommended): the cycle COORDINATION is a method on `PlannerAwareness` (S1), but the think-step delegates via CLOSURE-INJECTION — `cycle(&mut self, …, think: impl Fn(&ThinkingContext)->Vec<u64>)` — exactly the `run_convergence(triplets, apply)` precedent (cache/convergence.rs:223) where the planner never names the driver type.** The third crate then becomes thin: it owns only the `|ctx| driver.dispatch(to_dispatch(ctx))` closure + the `main`-side wiring, not the sequencer. Falsifier for the reconciliation: does a closure-typed think-step keep `cycle()` object-safe / `&self`-compatible? (interacts with T2.)

**TENSION T2 — `&mut self` vs the `&self` contract surface (S1, latent baton-handoff issue).**
`EpiphanyDetector::observe` is `&mut self` (escalation.rs:241); `OrchestrationBridge::route` (orchestration_impl.rs:47) + `PlannerContract::plan_full` (plan.rs:148) are `&self`. So `cycle()` cannot both mutate `epiphany` AND ride the existing contract trait unchanged. Two ways out: (a) interior mutability on the `epiphany` field (RwLock/LazyLock per data-flow.md "no &mut self during computation"); (b) `cycle()` is the sole `&mut` entry, deliberately kept OFF the `PlannerContract` trait. Decide before impl — see OQ-CYCLE-MUT.

**CONVERGENCE C1 — three savants independently hit the `Think`-carrier (S1 §SECOND-ORDER + S4 §SECOND-ORDER).**
S1: the 8-step order ≅ the `Think` struct's TEKAMOLO field-DAG (CLAUDE.md §The-Click); `CycleOutcome` should BE a `Think`-shaped carrier, and `cycle()` ≅ `Think::resolve()`. S4: `deinterlace(at_version(T))` builds `Think.global_context` *as-of-T* → the system can **think-as-of-T**, not just read-as-of-T (defensible-audit primitive). **Joint implication:** the cycle + the temporal as-of read are two faces of one object — a `Think` whose `global_context` is the deinterlaced as-of-T frame and whose `resolve()` is the 8-step cycle. **GATED:** `Think` carrier unification is already DEFERRED (LATEST_STATE PR #372) — so this is WORTH-EXPLORING, NOT do-now. Do not let a hardener treat it as in-scope for #496.

**CONVERGENCE C2 — the rung carrier is already aligned 1:1 (S2 + S3).**
S2: `ThinkingContext.rung` (plan.rs:22) ↔ `ShaderDispatch.rung` (cognitive_shader.rs:190) are the SAME `RungLevel`, matching default — the only missing code is a field-copy builder. S3: `RungLevel` is inert ONLY because it has no constructor; add `from_ordinal`/`shift`/`from_entropy` and the 4 hardcode sites have something to call. **Joint:** seam #4 (drive) + seam #5 (carry) close together — give `RungLevel` constructors (S3), drive it from entropy+delta at the 2 production sites (S3), copy it into `ShaderDispatch` at the seam-#5 lowering (S2). One `RungLevel` flows planner→cognition.

**CONVERGENCE C3 — carriers provisioned before their consumers (S1 + S2 + S5).**
`PlanResult.emitted_edges` (S1: same field/type/doc in both `PlanResult` twins, awaiting the cycle); `dn_hash` dedup (S5: Pinpoint's lesson already shipped at spo_bridge.rs:127); `QueryReference`'s `Copy` as-of half (S4: snapshot-isolation free). The recurring shape: the substrate was built anticipating these seams; the work is WIRING existing carriers, not building new ones. This is the doc's thesis (~90% exists) confirmed three independent ways.

**The additive new-code ledger (everything else is wiring):** (1) the `CognitiveCycle` sequencer (T1/T2); (2) `RungLevel::{from_ordinal,shift,from_entropy}` + `PlannerContract::rung_for` (C2); (3) the `temporal.rs` A→contract relocation + B core `temporal_read` join (seam #2); (4) `ScopedReference` + its `admits`/`retrieve_arc` methods (§3); (5) `MarkingRow: DeinterlaceRow` for Bardioc (§3); (6) the tier-structured `NiblePath::from_guid_prefix` bijection (§3, the width-overflow fix); (7) optional `const fn ExecTarget::can_drive` (seam #6 hardening, OQ-11.8). Each is small, additive, and has a falsifying probe in §5.

---

## §9 — HARDENING VERDICTS (3 brutally-honest agents, 2026-06-15)

PP-13 brutally-honest-tester → **HOLD** (3 P1, all spec-text fixes). PP-15 baton-handoff-auditor → **CATCH-LATENT** (4 latent boundary drops). PP-16 preflight-drift-auditor → **READY-TO-DISPATCH** (2 spawn-caution addenda). All three confirmed the load-bearing `file:line` grounding is real, the dependency-wall claims hold, and the measure-first ratio (9 probes / 7 ledger items) is honest. The fixes are spec-text, NOT architectural rewrites.

**LOCKED decisions (fold into the §8 ledger before any worker dispatch):**
1. **Emit channels are SEPARATE, neither derived (PP-13 P1-1 + PP-15 B2).** `emitted_edges: Vec<u64>` (CausalEdge64/EpisodicEdges64 words, collapse-gate path) and `emitted_moves: Vec<KanbanMove>` (16-B lifecycle records) are independent — NO `KanbanMove→u64` cast exists; inventing one is the I-LEGACY trap. Also note `ShaderBus.emitted_edges` is `[u64;8]` not `Vec` (the seam-#5 lowering crosses array→Vec).
2. **`cycle()` is an INHERENT method on `PlannerAwareness`, NEVER a trait method (PP-13 P1-3).** The `impl Fn` closure-injection arg is not object-safe → must stay OFF `PlannerContract`/`OrchestrationBridge` (the `Box<dyn>` consumers: n8n-rs, crewai-rust). Only `rung_for` goes on the trait. The `run_convergence` precedent (convergence.rs:223) is a free function — its object-safety doesn't transfer to a trait method.
3. **Seam #2's as-of read is CLOSURE-INJECTED too (PP-15 B8).** The planner cannot reach `at_version` (async, in lance-graph core, behind the anti-circular wall) — so the sequencer does NOT close seam #2 by itself; the temporal read is closure-injected exactly like the think-step (T1), OR core pre-deinterlaces and hands rows over.
4. **DUAL `RungLevel` — do NOT duplicate (PP-16 CoC-1, SPAWN-CAUTION — paste into the rung-seam worker prompt):** the contract `cognitive_shader::RungLevel` (cognitive_shader.rs:157) is the bare-enum target for the new constructors; `thinking_engine::cognitive_stack::RungLevel` (cognitive_stack.rs:264) ALREADY has `from_u8` + `should_elevate(consecutive_blocks, free_energy, cascade_depth)` (:278-314) — MIRROR its shape, never re-derive. (The plan disambiguates the dual `GateDecision` + dual `ThinkingContext` but was silent on the dual `RungLevel` — and seam #4 sits exactly on it.)
5. **P-RUNG-ROUNDTRIP is ill-posed as stated (PP-13 P1-2).** `entropy_class` quantizes at QUARTERS, `EntropyRung` bands at THIRDS — boundaries don't nest (counterexample: `class(0.3)==class(0.4)==1` straddle two bands). Restate as a one-band-tolerance probe OR require a thirds-banded 2-bit quantizer. Also `ThoughtCtx.rung` is `1..=9` while `RungLevel` is `0..9` → `from_ordinal` must clamp the off-by-one.

**Latent boundary fixes (PP-15, pin before D-MBX-A6-P3 impl):**
- **THREE `PlanResult`** (B2): inherent lib.rs:107, contract plan.rs:30 (NO live producer today — only nominally twinned), AND `arigraph/language.rs:34`. Name the 3-way collision; specify the inherent→contract adapter.
- **`MailboxId` sentinel (B4):** the type is `collapse_gate.rs:121` (NOT kanban.rs:114 — that's the field; citation fix). `0` is a real mailbox, `u32::MAX` is already overloaded (#386) → do NOT collapse `Option<MailboxId>` to a magic value; emit NO move without a real mailbox (matches StyleStrategy's no-theatre stance).
- **Newtype the address u64s (B7):** `dn_hash` + `local_key` are both bare `u64` → wrap `DnHash(u64)`/`LocalKey(u64)` at the `MarkingRow`/addressing boundary; cite the pre-existing `StructuralSignature` ~50% collision (`TECH_DEBT.md:134`) in §3 (it's not a clean third hash).

**P2 / nits:** `ExecTarget::can_drive` must enumerate all 24 (exec×column) cells (default-allow + SurrealQl deny-list); `MarkingRow` predicate should be an interned id, not `String` (the `&str` carrier forces a heap alloc per row); split P-ARC-PROMOTE-IS-REVISION into the in-tree half (eviction identity) + the DemotionSink-gated half.

**Sub-line drift to fix (PP-16, low-impact):** `ThinkingContext.rung` is plan.rs:**23** (not :22, cited ~5×); `try_advance_phase` soa_view.rs:**128**; `NextPhaseScheduler::on_version` scheduler.rs:**76**; `QueryReference::at` temporal.rs:**135**; `convergence.rs` has **9 tests, mod at :246, span :249-437** (not "8, :363-436"). And **"#495 rides #496" is mis-attributed (DRIFT-6, SPAWN-CAUTION):** `ValueSchema`/`ValueTenant`/`EdgeCodecFlavor` are POST-#495 BRANCH commits (`4e3496ab` + `920671d2`), present on this branch but NOT on origin/main, and NOT part of the already-merged #495 — **target the branch tree; do not verify these against origin/main** (a worker who does gets a false negative and thrashes).
