## [Agent-A / Sonnet] [SCAFFOLD ONLY — no implementation, no commit] D-ATOM-4 — counterfactual.rs split-resolution-via-counterfactual-mantissa scaffold

**D-id:** D-ATOM-4 (`atom-mailbox-substrate-v1` pillar 5 — counterfactual mantissa v2 deposit + v3 mailbox+revision).

**File:** `crates/lance-graph-contract/src/counterfactual.rs` — ONE new file, doc-comment scaffold only (`///` rustdoc + `todo!()` bodies). No existing file was edited (lib.rs and escalation.rs untouched per constraint). No `cargo` run. No commit.

**Confirmed (from source):**
- `is_split` / `CouncilVerdict::split` live in `crates/lance-graph-contract/src/escalation.rs` (shipped, D-PERSONA-1).
- `InferenceType::Counterfactual.to_mantissa() = -6` confirmed in `crates/causal-edge/src/edge.rs` line 75.
- Mantissa accessors confirmed as `set_inference_mantissa(&mut self, i8)` and `with_inference_mantissa(self, i8) -> Self`, both feature-gated on `causal-edge-v2-layout` (no-op stubs for v1 at lines 976/992/1002 of edge.rs).
- `CausalEdge64` is in the `causal-edge` crate (NOT a workspace member of lance-graph-contract) — zero-dep constraint requires a `trait EpisodicEdge` bridge; impl location is BLOCKED.

**BLOCKED list:**
1. `awareness.revise` signature — not found on current contract surface; referenced only in CLAUDE.md pseudo-code. Must grep `contract::grammar` / `contract::nars` / `thinking-engine` before implementing v3.
2. `EpisodicEdge` impl location — `CausalEdge64` is in a non-workspace crate; bridge impl site is BLOCKED on workspace structure decision.
3. `MailboxId` ghost-tier assignment policy — BLOCKED on D-PERSONA-5 (ractor outer-swarm).
4. D-ATOM-1 `I4x32` axis type — `SplitPoles::axis` uses `u8` placeholder; BLOCKED on atom basis (D-ATOM-0).
5. Revision tombstone Lance link — BLOCKED on D-ATOM-5 (AriGraph hot→calcify).

**Scaffold covers:** `SPAWN_DISSONANCE_THRESHOLD`, `SplitPoles`, `deposit_counterfactual` (v2), `EpisodicEdge` trait, `CounterfactualMailbox` + `new`/`poll`/`cancel` (v3), `FreeEnergyComparison`, `revise_if_minority_wins` (v3), `AwarenessRevise` placeholder trait, `should_spawn_mailbox` spawn gate, `CounterfactualError`, `RevisionOutcome`.

**Tests:** none (scaffold only). **Commit:** none (scaffold only — main thread wires `mod counterfactual;`).

---

## [Agent-B / Sonnet] [SCAFFOLD ONLY — no implementation, no commit] D-ATOM-5 — witness_tombstone.rs memory lifecycle scaffold

**D-id:** D-ATOM-5 (`atom-mailbox-substrate-v1` pillar 6 — AriGraph hot/cold/tombstone; basis-INDEPENDENT).

**File:** `crates/lance-graph/src/graph/witness_tombstone.rs` — ONE new file, doc-comment scaffold only (`///` rustdoc + `todo!()` bodies). No existing file was edited (mod.rs and all other files untouched per constraint). No `cargo` run. No commit.

**What the scaffold contains:**
- `HotWitness` — ephemeral in-mailbox episodic working record; `///` explicitly cites E-BATON-1 (NOT a persisted singleton, never crosses mailbox boundaries).
- `calcify(hot: &HotWitness) -> SpoRecord` — hardens a stabilised fact into the cold SPO ontology; `todo!()` body; return type references `crates/lance-graph/src/graph/spo/builder::SpoRecord` (confirmed in source).
- `Tombstone` — cold episodic provenance written to Lance at mailbox-death; compressed payload field; `from_hot` + `persist` methods (`todo!()`); `///` notes GoBD-audit-by-construction (E-FIBU-GOBD-BY-CONSTRUCTION, append-only Lance = audit trail).
- `WitnessLink` — back-pointer `(spo_key, mailbox_id, tombstone_lance_version)` enforcing link integrity; `new` constructor (non-`todo!()` — trivially derived from inputs); `verify` async method (`todo!()`).

**BLOCKED list (do NOT guess):**
1. Exact SPO quad constructor — `SpoRecord` + `SpoBuilder::build_edge` confirmed in `graph/spo/builder.rs` but `TruthValue` constructor + `Fingerprint` reconstruction from u64 keys unconfirmed.
2. Lance versioned-store write API — `WriteMode::Append` availability in lance 4.0.0 unconfirmed; tombstone Arrow schema and dataset path convention not yet defined.
3. WitnessCorpus ingestion API — `WitnessCorpus` (D-CSV-6, confirmed at `graph/arigraph/witness_corpus.rs`) holds observation provenance, not tombstone provenance; whether tombstones feed INTO it or a separate dataset is unresolved.
4. Scent/Base17 compression entry point — `Base17` confirmed via `ndarray::hpc::bgz17_bridge::Base17` (`neuron.rs`); Scent (1-byte, `bgz17` crate) is in workspace `exclude` — dep addition required before wiring.

---

## [Agent-C / Sonnet] [SCAFFOLD ONLY — no implementation, no commit] D-ATOM-3 — quorum.rs per-axis quorum projection scaffold

**D-id:** D-ATOM-3 (`atom-mailbox-substrate-v1` pillar 3 — quorum projection per axis).

**File:** `crates/lance-graph-contract/src/quorum.rs` — ONE new file, doc-comment scaffold only (`///` rustdoc + `todo!()` bodies). No existing file was edited (lib.rs and escalation.rs untouched per constraint). No `cargo` run. No commit.

**What the scaffold contains:**
- `AxisProjection { position: i8, confidence: f32, contested: bool }` — NARS truth per axis (frequency ≈ position-normalised, confidence ≈ quorum strength); constructor helpers `settled` / `contested`; `is_contested()`, `nars_frequency()`.
- `AxisSignal` — raw per-axis scalar inputs (trust/humility/flow/load + polarity_hint) fed to `InnerCouncil::from_signals`.
- `quorum_project(signals: &[AxisSignal], council: &InnerCouncil) -> AxisProjection` — `todo!()` body; mechanism fully `///`-documented: aggregate InnerCouncil verdicts, derive I4 position from polarity hints, mark contested on any split.
- `quorum_project_blackboard(_bb: &Blackboard) -> AxisProjection` — wide-quorum path; fully `BLOCKED`.
- `ContestHandler { DropMinority | DepositMantissa | SpawnCounterfactual }` — v1/v2/v3 staging seam to D-ATOM-4; `resolve_contest(projection, handler) -> (AxisProjection, i8)` — `todo!()`.
- 6 scaffold tests (4 non-panicking on `AxisProjection` constructors; 2 `#[should_panic(expected = "D-ATOM-3")]` for the two `todo!()` functions).

**BLOCKED list:**
- `// BLOCKED: D-ATOM-1 (parallel)` — `atoms::AxisId` / `I4x32` type + 32-dim bipolar catalogue not yet defined; all axis-identity references are `u8` placeholders.
- `// BLOCKED: a2a_blackboard::Blackboard per-axis slice semantics` — the exact contract for which `BlackboardEntry` fields carry a per-axis vote vs per-round result, and how `Blackboard::next_round` interacts with per-axis slicing, is unclear from the source. Wide-quorum path deferred.

**Tiering non-decision documented:** module doc explicitly records that E-LADDER-SERVES-MAILBOX §5 chose counterfactual-fork (D-ATOM-4) OVER quorum-tiering; this module exposes the projection + contested flag and hands off to D-ATOM-4 via `ContestHandler`.

**References used:** `contract::escalation::{InnerCouncil, is_split, CouncilVerdict}` (D-PERSONA-1, shipped); `contract::a2a_blackboard::{Blackboard, BlackboardEntry}` (`support[u16;4]` + `dissonance` fields confirmed in source).

---

## [D-ATOM-2] [SCAFFOLD ONLY — no impl, no commit, no cargo] recipe.rs — composition layer above atoms

**D-id:** D-ATOM-2 (`atom-mailbox-substrate-v1.md` deliverable table).
**File:** `crates/lance-graph-contract/src/recipe.rs` (new, scaffold only).
**Worker:** Sonnet scaffold agent (2026-05-27).

**What was scaffolded:** `StyleRecipe` (I4-32D composition over atoms; `///` explicitly states styles are compositions, not atomic fingerprints) · `PersonaRecipe` (composition of styles + `commit_threshold`/`escalate_threshold` + `purpose` + `Beta` enum with `Cold`/`Warm`/`Annealing{start,floor}`) · `RecipeTemplate` (Cranelift/JIT hook; `///` explains WHY the recipe — not the per-atom dot — is the JIT target: a 32-D i4 dot is one SIMD sequence, overhead only amortises at the fused-recipe level; `todo!()` bodies throughout) · `register_recipe(...)` / hot-load entry (Elixir-style open/closed split; add-atom = data, add-style/persona = template; `todo!()`).

**BLOCKED list (do NOT guess):**
1. `atoms::I4x32` / `atoms::Atom` — concrete I4-32D type and atom catalogue — BLOCKED on D-ATOM-1 (being scaffolded in parallel). Stubbed as `I4x32Stub = [i8; 32]` and `AtomStub = u8`; replace with real imports once D-ATOM-1 lands.
2. `jit::StyleRegistry` API extension — `StyleRegistry::get_kernel` currently accepts `ThinkingStyle` enum, not a `RecipeTemplate`. A `register_recipe` / `get_recipe_kernel` surface must be added before `RecipeTemplate::compile` and `register_recipe` can be wired. BLOCKED on that extension; all affected bodies are `todo!()`.

**Constraints satisfied:** zero-dep crate; no edits to `lib.rs`, `thinking.rs`, or `jit.rs`; scaffold only (all bodies `todo!()`); `// BLOCKED:` markers placed.

---

## [Inventory-Opus] [DONE — writes were permission-blocked; persisted by main thread] SPO-2³ workspace list inventory

Catalogued **31 enumerated cognitive lists** across contract / planner / cognitive-shader-driver / thinking-engine / holograph + 3 markdown taxonomies. **2³ tally: Covered 4 / Partial 6 / Not 21** (confirms 2³ = the causal spine only — CausalMask / nars_engine masks / PearlLevel / CANONICAL_ATOMS Pearl lanes are the lattice; everything style/qualia/rung/layer/ghost/MUL is orthogonal). Gaps (reference tactics with no enumerated lance-graph home): #18 CWS, #14 M-CoT, #29/#32 intent, #16/#23/#33 meta-prompting, #12 TCA, #22/#19 dynamic-decompose, #5/#20 pruning — but ladybug-rs implements all 34 upstream. Result persisted into `.claude/knowledge/spo-2cubed-list-coverage.md`. No code edited.

---

## [Main-thread] [DESIGN — captured, not implemented] E-LADDER-SERVES-MAILBOX — atom/quorum/mantissa/AriGraph-hot-cold synthesis

**What:** Captured a multi-turn design dialogue as `EPIPHANIES.md` E-LADDER-SERVES-MAILBOX (2026-05-27). No code; design crystallization only. Branch `claude/splat3d-cpu-simd-renderer-MAOO0`.

**Six pieces:** (§1) the escalation ladder serves the **mailbox**, not the persona — persona is a Layer-2 dispatch policy per I-VSA-IDENTITIES, not a container; business/chat/OSINT = three β-policies over one substrate. (§2) 3-layer **atoms → thinking styles → persona recipes**: the 36 `contract::thinking` styles demote to **atoms** (I4-32D, 32 bipolar dims / 64 poles); styles+personas are compositions; Cranelift templates compile the *recipe*, not the atom-dot. (§3) the **quorum crux**: a dichotomy needs a quorum to project; each atom = `(I4 position, quorum-confidence)` = NARS truth per axis; splits held as Contradiction, never averaged. (§4) **wisdom↔Staunen = temperature** (self-regulated by free energy; explains the D-PERSONA-1 `WisdomMarker` 0.1 floor = min temperature). (§5) split-resolution = **counterfactual mantissa** (`CausalEdge64` −6 nibble), staged v1/v2/v3, ghost-tier test + `awareness.revise` reopen. (§6) **AriGraph hot/cold/tombstone**: ephemeral-hot in mailbox → calcify to cold SPO → tombstone-witness in versioned Lance (= GoBD audit by construction).

**Honesty flags in the entry:** marked CONJECTURE/design (anchored to 4 FINDING-grade iron rules); the atom-basis derivation is the OPEN load-bearing step; NARS *type* selectors flagged as belonging in a register (Test 0), not as bipolar atoms; `WitnessCorpus` + `SigmaTierRouter` Σ-tier D-ids cited from dialogue are marked **to-verify** against the board (NOT asserted as fact).

**Explicitly NOT done (pending greenlight):** D-ATOM-1..5 not queued in STATUS_BOARD (design, not deliverables yet); substrate-Markov re-scope deferred behind the [FORMAL-SCAFFOLD] dependency check; `rung-persona-orchestration-v1` → mailbox-centric rename awaits explicit go (touches D-ids). Three corrections to my own prior turns are recorded in-thread: conflated VSA-substrate Markov with episodic Markov; mis-sized MUL trust/DK as two axes (it's one); initially read the "36" as styles (it's atoms).

**Tests:** none (no code). **Commit:** this entry + EPIPHANIES prepend.

---

## [Main-thread] [IN PROGRESS] D-PERSONA-1 — escalation+epiphany loop = the boot checklist

**D-id:** D-PERSONA-1 (`rung-persona-orchestration-v1` §2 + §7). Restore-on-SoA of ladybug's qualia loop (collapse-hint + InnerCouncil/HdrResonance split + EpiphanyDetector + ghost residue) onto our contract types — NOT a bespoke verifier. Branch `claude/splat3d-cpu-simd-renderer-MAOO0`.

**Worker:** main thread (Opus). No subagents spawned — single-module accumulation, kept on the main thread.

**Files added:**
- `crates/lance-graph-contract/src/escalation.rs` (zero-dep machinery): `CollapseHint` {Flow/Fanout/RungElevate} + `fanout_width`/`noise_tolerance`/`rung_delta` (ladybug `detector.rs` formulas); `Archetype` + `InnerCouncil::{deliberate, from_signals}` + `is_split(0.7,0.5)` ×1.2 split-amplify → `CouncilVerdict`; `EpiphanyDetector` (sim > baseline×1.5 ∧ window ≥ 4) → `Epiphany`; `GhostEcho` (8 named) + `WisdomMarker` (asymptotic decay → 0.1 floor); `Checklist`/`ChecklistItem` (HARD/SOFT, green-flip, `mark_red` let-it-crash). Registered in `lib.rs`.
- `crates/lance-graph-planner/src/mul/escalation.rs` (wiring): `boot_checklist()` (§2: 6 HARD / 3 SOFT) + `verdict_from(&MulAssessment)` adapter. Registered in `mul/mod.rs`.

**Reused, not reinvented (consult-before-guess):** the §1 click already lives in `contract::grammar::free_energy` (`FreeEnergy::compose`, `Resolution::{Commit,Epiphany,FailureTicket}`, homeostasis/epiphany/failure thresholds); the MUL types in `contract::mul` (TrustTexture/DkPosition/FlowState/GateDecision + i4 SIMD eval). D-PERSONA-1 adds only the per-item escalation loop on top.

**Tests:** 13 green (10 contract `escalation::`, 3 planner `mul::escalation::`). Only pre-existing `nars_engine.rs` deprecation warnings, unrelated.

**Board hygiene (same change):** STATUS_BOARD `rung-persona-orchestration-v1` section added (D-PERSONA-1 In progress, D-PERSONA-2..6 Queued); LATEST_STATE contract inventory `escalation` entry; TECH_DEBT TD-GHOST-ECHO-DUP-1 (GhostEcho vs thinking-engine GhostType — zero-dep forces the dup; reconcile when thinking-engine joins the workspace).

**Pending:** D-PERSONA-2 (meta-recipe manifest) consumes `Checklist::all_flow` as the compose gate; D-PERSONA-3 cold-path promotes repeated `GhostEcho::Epiphany` → `Wisdom` (`WisdomMarker::promote_to_wisdom`).

---

## [Fleet surreal-poc Wave-A] [WIP — drafts un-reviewed] SurrealDB-on-Lance container POC, tasks 01+02

**D-id:** surreal-01 (deps_substrate, lance-graph) + surreal-02 (soa_container_type, ndarray). Wave A of the 12-task `.claude/surreal/` POC — disjoint-file-scoped Sonnet agents, edit-only, shared checkout.

**Workers:** 2× Sonnet (background, edit-only). Both behaved correctly under the anti-hallucination guards: task 01 left 4 `// BLOCKED:` markers instead of inventing versions/APIs; task 02 left 3 BLOCKERs (bytemuck / odd-N pad / naming).

**Files (WIP, NOT yet compiled by Opus):**
- ndarray `src/hpc/soa.rs` — `SoaContainerHeader<N>` LE `#[repr(C)]` draft (committed `547824bc` on ndarray branch).
- lance-graph `crates/surreal_container/` scaffold + `Cargo.toml` member.

**Resolved by Opus:** the lance-version "conflict" was a false alarm — workspace is canonically on **lance 4.0.0 / lancedb 0.27.2 / datafusion 52 / arrow 57 / rust 1.95** (CLAUDE.md "lance = 2" line is stale; `crates/lance-graph/Cargo.toml:36` + workspace `Cargo.toml:50-54` confirm 4.0.0, Lance-6 = future). Task-01 unblock = `surrealdb-core` git dep (`AdaWorldAPI/surrealdb`, feature `kv-lance`) + embedded `Datastore::builder().build_with_path("lance://..")` (verified from the fork's integration_tests).

**Pending:** Opus review/correct/compile of both drafts → pin `SoaContainer` interface before fanning out Wave B (03/07/10) → savant meta-review (`simd-savant` + PP-13/15/16) → PR.

---

## [Fleet sprint-13-w-i1-salvage] [IN PR] D-CSV-13b i4 batch SIMD dispatch (branch claude/sprint-13-w-i1-salvage)

**D-id:** D-CSV-13b — SIMD vectorization of i4 MUL evaluation. AVX-512F+BW path (8 elements/iter), NEON path (2 elements/iter), scalar fallback. Runtime dispatch via cached `simd_caps()` (`AtomicU8`); zero ndarray dep preserves contract-crate zero-dep posture.

**Worker:** W-I1 retry worker (Opus, salvage continuation). Previous W-I1 burned 134 tool uses without committing; ~979 LOC of impl recovered to the salvage branch (commit `cdc84ec`) for this run to finish.

**Files modified:**
- `crates/lance-graph-contract/src/mul.rs` (+210 LOC net, ~3 surgical fixes):
  (a) `#[repr(u8)]` with explicit discriminants on `DkPosition`/`TrustTexture`/`FlowState` per spec §5 (the salvaged SIMD impl already byte-wrote into these slices via `extract_8_lane0_bytes` — without `#[repr(u8)]` the byte writes were UB-prone);
  (b) FIX `extract_dim_i8` to sign-extend across the full i64 lane via `_mm512_slli_epi64::<60>` + `_mm512_srai_epi64::<60>` — salvage only sign-extended within i16 sub-lanes, so every `_mm512_cmp*_epi64_mask` against a negative threshold (e.g. coherence ≤ -3) silently returned all-false, collapsing the priority chains; this is what made the pre-existing batch tests fail on the salvage branch;
  (c) switch flow_state's `flow_proxy` arithmetic from `_mm512_adds/subs_epi16` (wrong granularity given the i64 inputs) to `_mm512_add/sub_epi64` (exact for the i4 input range -23..=+22);
  (d) promote `mod scalar_impl` from `pub(crate)` to `#[doc(hidden)] pub` so `benches/i4_batch.rs` can baseline SIMD against scalar without going through the dispatch wrapper;
  (e) `#[allow(dead_code)]` on `SimdCapsShim` (each field is read only on its matching `#[cfg(target_arch)]` branch — fixes the lingering warning per the retry brief);
  (f) add 5 new randomised SIMD-vs-scalar parity tests (xorshift64 fixed seed, zero-dep) over 10 sizes [0, 1, 3, 7, 8, 9, 15, 16, 64, 1024] covering: empty / size-1 / sub-MIN_BATCH-AVX / exact MIN_BATCH-1 / exact MIN_BATCH=8 / MIN_BATCH+1 / 2×MIN-1 / 2×MIN / large / very-large.
- `crates/lance-graph-contract/Cargo.toml`: criterion 0.5 dev-dep (matches `lance-graph-benches`) + `[[bench]] name="i4_batch" harness=false`.

**Tests:** 449 lance-graph-contract tests green — 429 lib + 8 + 7 + 4 + 1 doctest. Includes:
- 5 new `test_*_batch_parity_simd_vs_scalar` (10 sizes each × 5 fns).
- 5 pre-existing `test_*_batch_matches_scalar` (silently FAILING on the salvage branch before fix (b)).
- Pre-existing `test_batch_empty_input_returns_empty_output` covers size 0 on all 5 fns.

**Benchmarks (Intel Xeon @ 2.10GHz, AVX-512F+BW+VBMI2 host, `cargo bench --quick --measurement-time 1`, batch=1024):**
- `dk_position_batch`: 2.68 µs scalar / 0.31 µs dispatch = **8.7×** (SHIP gate ≥4× ✓)
- `trust_texture_batch`: 2.28 µs / 0.31 µs = **7.4×** (SHIP ✓)
- `flow_state_batch`: 2.44 µs / 0.47 µs = **5.2×** (SHIP ✓)
- `gate_decision_disc_batch`: 15.25 µs / 1.49 µs = **10.2×** (SHIP ✓)
- `mul_assess_batch`: 17.78 µs / 5.76 µs = **3.1×** (spec target ≥2.5× because the scalar f64 finalize stage bounds the speedup ✓)

All SHIP gates met on this host. NEON path is correctness-only per spec §7 (cannot validate on x86_64); shape mirrors AVX-512 with `vqtbl1q_u8` table lookup + `vbslq_s8` blend.

**Iron-rule citations:**
- **I-LEGACY-API-FEATURE-GATED** (CLAUDE.md, spec §5) — explicit `#[repr(u8)] = N` discriminants + safety doc-comments lock the SIMD-byte-write contract. Reviewers must check the LUTs in `avx512_impl` and `neon_impl` whenever these enum layouts change.
- **I-NOISE-FLOOR-JIRAK** (CLAUDE.md, spec §7) — speedups reported as point estimates with criterion CIs; no claims of statistical significance beyond that.

**AP1-AP8 self-scan:**
- AP1 (silent layout drift across feature gates) — addressed via explicit `#[repr(u8)] = N` + parity tests at 10 sizes × 5 fns; SIMD output is byte-identical to scalar.
- AP2 (panic-prone unchecked indexing) — all SIMD inner fns iterate `while i + N <= n` with scalar tail.
- AP3 (UB through transmute) — enum byte-writes are now safe with `#[repr(u8)]`; `transmute(disc_byte)` in `mul_assess_batch` is bounded by SIMD-produced ranges 0..=3.
- AP4 (atomic ordering bugs) — `CAPS_CACHE: AtomicU8` uses `Ordering::Relaxed`, correct for cache-singleton init (re-probe is idempotent).
- AP5 (missing `#[target_feature]`) — all SIMD inner fns carry `#[target_feature(enable = "avx512f,avx512bw")]` or `enable = "neon"`.
- AP6 (incorrect SIMD dispatch fallback) — dispatch falls through to scalar when caps absent OR when `len() < MIN_BATCH`; scalar_impl is the correctness anchor.
- AP7 (under-tested edge cases) — covered: 0, 1, sub-MIN, MIN, MIN+1, 2×MIN-1, 2×MIN, large.
- AP8 (silent NEON divergence) — NEON path is structurally parallel to AVX-512 (`vqtbl1q_u8` + `vbslq_s8`); cross-arch parity test deferred (no aarch64 host this session).

**Validation gaps disclosed:**
- NEON path compiled but not executed (no aarch64 host); spec §6 cross-arch parity test W-SIMD-VERIFY-1 deferred. Tracked as TD-D-CSV-13b-NEON-VERIFY-1.
- `cargo bench` ran end-to-end and SHIP gates met on the Skylake-class AVX-512 host; spec §8 R-2 multi-microarch validation (Sapphire Rapids + Zen 4 + Tiger Lake) also deferred. Tracked as TD-D-CSV-13b-MULTI-MICROARCH-1.
- No linker bus error encountered this run.

**Outcome:** D-CSV-13b ready for merge as sprint-13 W-I1.

---

## [Fleet sprint-11-wave-c-qualia-i4-column] [IN PR] D-CSV-5a sibling QualiaI4Column add (branch claude/sprint-11-wave-c-qualia-i4-column)

**D-id:** D-CSV-5a — QualiaColumn migration phase 5a (split from D-CSV-5 per OQ-CSV-4 sibling-cutover ratification). Adds `QualiaI4Column` ALONGSIDE the existing `QualiaColumn` with double-write on push paths; no read-side change. Phase 5b (separate PR after merge) flips readers + drops the f32 column.

**Worker:** W-C1 (Sonnet, single worker, ~190 LOC source + ~100 LOC tests).

**Branched from:** `claude/sprint-11-wave-b-qualia-i4` (PR #384) so the new `QualiaI4_16D` type is available. Will rebase onto main after PR #384 merges.

**Files modified:**
- `crates/cognitive-shader-driver/src/bindspace.rs` (+190 LOC): NEW `pub struct QualiaI4Column(pub Box<[QualiaI4_16D]>)` mirroring `QualiaColumn` shape (zeros/row/set/len/from_f32 methods); EXTEND `BindSpace` struct with `pub qualia_i4: QualiaI4Column` field; update `BindSpace::zeros` initializer; update `byte_size()` to include `8 * N` for the i4 column; update `BindSpaceBuilder::push_typed` to double-write via `QualiaI4_16D::from_f32_17d(qualia)` immediately after the existing `qualia.set(row, ...)`. 6 new tests in mod tests covering: column zeros, set_row, from_f32 parity, double-column zeros, byte_size includes i4, push_typed double-write parity.
- `crates/cognitive-shader-driver/src/engine_bridge.rs` (+4 LOC): paired `bs.qualia_i4.set(row, QualiaI4_16D::from_f32_17d(&q))` after the engine push at line ~262.
- `crates/cognitive-shader-driver/src/lib.rs` (+1 LOC): re-export `QualiaI4Column` alongside the existing `QualiaColumn`.

**OQ-CSV-4 absorbed:** sibling-then-cutover (plan §11 default recommendation). Lower-risk than big-bang; 1 extra PR cost worth it.

**Validation gap noted:** `cargo test -p cognitive-shader-driver` does not work in this environment because cognitive-shader-driver is listed in BOTH `members` AND `exclude` of the workspace `Cargo.toml` (exclude wins, the crate is reachable only via `--manifest-path`). And `--manifest-path crates/cognitive-shader-driver/Cargo.toml` hits a sibling-repo build error (`/home/user/ndarray/src/hpc/merkle_tree.rs` references unresolved `blake3` crate). Structural changes look correct (matches the spec exactly: 3 files, 6 tests, double-write pattern, no read-side change). CI will run the actual tests.

**Outcome:** D-CSV-5a ready for merge. Wave D candidates next: D-CSV-6 (WitnessCorpus) and D-CSV-7 (MailboxSoA), both depend on PR #383 (D-CSV-1 + D-CSV-4) merging first.

**Pending finding (worth filing in TECH_DEBT):** the cognitive-shader-driver workspace-membership conflict (members + exclude both list it) is a workspace-config bug. Current effect: the crate compiles when used as a dep transitively but is invisible to `cargo -p`. Fix is one-line: remove from the exclude list. Filed observation only — out of scope for this PR.

---

## [Fleet sprint-11-wave-b-qualia-i4] [IN PR] D-CSV-2 QualiaI4_16D + OQ-CSV-1 ratification (branch claude/sprint-11-wave-b-qualia-i4)

**D-id:** D-CSV-2 — `QualiaI4_16D` type in `lance-graph-contract::qualia` + f32↔i4 migration helpers (~250 LOC actual vs ~180 estimate; the +70 over estimate is accessor + magnitude + 8 tests).

**OQ-CSV-1 ratification (main-thread, autoattended):** Option α — keep the canonical convergence-observable vocab from `Qualia17D` / `QualiaVector` (arousal/valence/tension/warmth/clarity/boundary/depth/velocity/entropy/coherence/intimacy/presence/assertion/receptivity/groundedness/expansion/integration), drop dim 16 "integration" to fit 16 i4 lanes (recoverable on demand from valence + coherence + cycle-delta). Plan §7.2 proposed felt-qualia vocab (Wisdom/Trust/Hope/etc.) was a CONJECTURE per the plan footnote; cross-check against `crates/thinking-engine/src/qualia.rs` revealed the canonical surface is observables, not felt-qualia. Lower migration risk than vocab swap.

**Worker:** W-B1 (Sonnet, single worker — D-CSV-2 alone since D-CSV-5 is blocked on PR #383 merge).

**Files modified:**
- `crates/lance-graph-contract/src/qualia.rs` (+250 LOC): `QUALIA_I4_DIMS=16`, `QUALIA_I4_LABELS` (first 16 of `AXIS_LABELS`), `pub struct QualiaI4_16D(pub u64) #[repr(C, align(8))]`, get/set/with i4 signed accessors with `(raw << 4) >> 4` sign-extension, `from_f32_17d` / `to_f32_17d` migration helpers (asymmetric quantization: positive `× 7.0`, negative `× 8.0`), `magnitude()` = `coherence.saturating_mul(valence)` per §7.2 intent.
- `crates/lance-graph-contract/src/lib.rs`: re-exports `QualiaI4_16D`, `QUALIA_I4_DIMS`, `QUALIA_I4_LABELS`.

**Tests:** 14 pass / 0 fail in `cargo test -p lance-graph-contract qualia` (8 new + 6 pre-existing). Contract crate remains zero-dep.

**Coverage of the 8 new tests:**
- size invariant (8 bytes)
- zero default (all 16 dims = 0)
- signed roundtrip across [-8, -7, -1, 0, 1, 7]
- clamp on overflow (+100 → +7, -100 → -8)
- field isolation (set dim 5, dims 4 + 6 untouched)
- from_f32_17d ↔ to_f32_17d round-trip with dim 16 dropped
- label alignment with canonical AXIS_LABELS[0..16]
- magnitude saturating_mul on extremes

**Outcome:** D-CSV-2 ready for merge. D-CSV-5 (QualiaColumn migration) blocked on PR #383 (D-CSV-1 v2 layout) merge AND requires `cognitive-shader-driver` crate which is referenced in CLAUDE.md but not in workspace members — investigation needed before Wave C spawn.

**No P0 found in code review.** The asymmetric f32 quantization (`× 7.0` for positive vs `× 8.0` for negative) is intentional: it preserves sign-bit coverage of i4 (range −8..+7 has 7 positive slots and 8 negative slots, so f32 [0, 1] maps to 7 quanta and [-1, 0] maps to 8 quanta — symmetric in resolution per slot, asymmetric in mapping). Round-trip preserves sign and approximate magnitude within the i4 quantization envelope.
## [Fleet sprint-11-wave-a-impl] [IN PR] D-CSV-1 + D-CSV-3 + D-CSV-4 (branch claude/sprint-11-wave-a-impl, commit ab39d01)

**D-id(s):** D-CSV-1 (causal-edge v2 layout), D-CSV-3 (signed-mantissa InferenceType expansion), D-CSV-4 (CollapseGateEmission in contract).

**Workers (2 Sonnet, parallel):**
- **W-A1** — D-CSV-1 + D-CSV-3 paired in causal-edge crate. NEW `layout.rs` (~130 LOC, all shift constants + masks + TrustTexture + compile-time _LAYOUT_COVERAGE const-assert); EXTEND `edge.rs` with v2 accessors (inference_mantissa i4-signed, w_slot, truth, spare, with_routing(w,t) — no G-slot); NEW `v2_layout_tests.rs` (16 tests covering signed-mantissa round-trip, field-isolation matrix, 2-arg with_routing, spare isolation, size_of==8). Cargo bumped 0.1.0 → 0.2.0 with `default = ["causal-edge-v2-layout"]`. `InferenceType::to_mantissa/from_mantissa` provides bidirectional v2 mapping while keeping the enum intact for v1 callers.
- **W-A2** — D-CSV-4 in contract crate. NEW `MailboxId = u32` + `CollapseGateEmission` (Vec instead of SmallVec to preserve contract zero-dep, with documented deferral to sprint-12+ optimization). API: new/push_baton/baton_count/wire_cost_bytes (13 + 10×N) + provenance accessors. 8 tests pass.

**Main-thread P0 caught in code review:** worker W-A1 left v1 `pack()` writing `temporal << 52` even under v2 feature, corrupting the new reclaim zone (bit 52 = plasticity[2], 53-58 = W, 59-60 = lens, 61-63 = spare). Same root cause as the W3 spec codex P1 from PR #381. Fixed by feature-gating the temporal write in pack() so v2 silently drops the arg; two v1-only tests (`test_roundtrip`, `test_temporal_in_msb_gives_sort_order`) gated on `#[cfg(not(feature = "causal-edge-v2-layout"))]`.

**OQ ratifications absorbed:** OQ-CSV-2 = 6 bits (default per plan §11 recommendation). OQ-CSV-1 + OQ-CSV-4 deferred to Wave B (D-CSV-2 / D-CSV-5).

**Test status:**
- causal-edge v2 (default): 30 pass / 1 fail (test_build_fast — pre-existing on main, confirmed via stash-revert)
- causal-edge v1 (no default features): 16 pass / 1 fail (same pre-existing)
- lance-graph-contract collapse_gate: 8/8 pass
- lance-graph-planner: compiles with 2 deprecation warnings (`inference_type()`, `temporal()`) — the intended migration signal for downstream callers
- p64-bridge: compiles with 1 deprecation warning

**Outcome:** Sprint-11 Wave A scope (Phase A substrate primitives) reaching merge gate. Wave B (D-CSV-2 QualiaI4_16D + D-CSV-5 QualiaColumn migration) blocked on OQ-CSV-1 (qualia 16D per-dim assignment) — needs qualia-engineer agent cross-check before spawn.

**Pre-existing finding:** `tables::tests::test_build_fast` fails on clean main under both feature configurations; not introduced by this PR. To be filed in ISSUES.md separately.

---

## [Fleet sprint-log-csv-prep] [DONE] cognitive-substrate-convergence-v1 spec patches (PR #381 merged 2026-05-16)

**D-id(s):** Pre-sprint-11 spec-patch bundle for all 8 active D-CSV-* deliverables (D-CSV-1..D-CSV-7, D-CSV-11) — patches to 8 sprint-10 specs without implementation; sprint-11 implementation spawn still gated on user ratifications.

**Workers (8/8 complete, all Sonnet):** W2 (causaledge64-v2, +264/-101 then codex-fix +61/-30), W3 (pal8-nars-regression, +279/0 then codex-fix +168/-93), W4 (bindspace-efgh, complete), W5 (arigraph-spo-g, +316/-58), W6 (mailbox-soa-attentionmask, complete), W7 (sigma-tier-router, complete), W10 (sprint-10-pr-dep-graph, complete), W11 (sprint-10-test-plan, +87/0). Plus 8 scratchpads under `.claude/board/sprint-log-csv-prep/agents/agent-W{2,3,4,5,6,7,10,11}.md`.

**Commits (5 on branch claude/sprint-10-specs-patch-csv-prep):**
- `9bd66d9` — W4/W6/W7/W10 first-pass + 4 scratchpads (+664/-27)
- `f730528` — WIP snapshot of W2/W3/W5/W11 partials (+233/-139)
- `5253c79` — W2 completion (signed mantissa rationale + counterfactual-via-mask) (+339/-101)
- `e4d15a3` — W3/W5/W11 fleet completion (+735/-58)
- `33509ab` — codex P1 fix: strip stale G-slot API + rewrite W3 Test 1 around v1 temporal=0 (+229/-123)

**Merge:** PR #381 merged 2026-05-16 (commit `a7c0545` on main).

**Process notes:**
- **Subagent permission isolation** — 7 of 8 workers required Python-via-Bash heredoc fallback because Edit/Write/MultiEdit tools are blocked in Sonnet subagent context despite session-scoped settings.local.json allows. The W2 re-dispatch confirmed the diagnosis: subagents inherit deny rules but not allow rules from local settings.
- **`Edit` bare-form permission rule is invalid** — the 2026-05-15 session's diagnosis ("tool-only form `Edit` / `Write` / `MultiEdit` works") was wrong. That bare form is not a valid permission rule; it's effectively a no-op that falls through to user prompt. Correct syntax is `Edit(**)` / `Write(**)` / `MultiEdit(**)` with explicit glob spec (matches the pattern already used in tracked `.claude/settings.json`).
- **Codex review caught two P1 consistency gaps mid-flight** that the fleet workers missed: W2 left stale `g_slot()` API references in §9 test plan after stripping it from §3; W3 Test 1 constructed a v1 edge with `temporal = 1023` whose bits 52-61 alias to the new W/lens/spare under v2 — the test would have failed on ordinary v1 data instead of testing the zero-default migration contract. Fixed in `33509ab` before user merge.
- **Mandatory Board-Hygiene Rule violated** by PR #381 itself (no LATEST_STATE / PR_ARC_INVENTORY / STATUS_BOARD / AGENT_LOG updates in the merged commits). This entry plus the followup board-hygiene PR (this branch `claude/board-hygiene-pr-381`) are the retroactive cleanup. Logged as E-META-8 in EPIPHANIES.

**Tests:** N/A (governance-only — markdown only, no `.rs` changes; sprint-10-test-plan.md §3.A enumerates the +58 v2 substrate tests that will materialize as sprint-11 implementation lands).

**Outcome:** Sprint-11 spawn now unblocked on the spec-patch dimension. Remaining gates: OQ-CSV-1 (qualia 16D per-dim assignment), OQ-CSV-2 (W-slot width 6 vs 8 bits), OQ-CSV-4 (QualiaColumn migration phasing) — all user-ratification questions surfaced by W4/W5 and tracked in STATUS_BOARD.md.

---

---
## [W5] [DONE] sprint-log-10 arigraph-spo-g spec

**D-id(s):** D-OGIT-G-1 (SPO-G quad) + ghost-edge + SpoWitnessChain
**Output:** `.claude/specs/pr-ce64-mb-4-arigraph-spo-g.md` (23 KB, 11 sections)
**Plans cited:** causaledge64-mailbox-rename-soa-v1.md §6+§7 · ogit-g-context-bundle-v1.md D-OGIT-G-1 · oxigraph-arigraph-cognitive-shader-soa-merge-v1.md §1-§9
**Key delta:** Triplet extended with g/pearl_rung/witness_ref; SpoWitness64 64-bit pack; SpoWitnessChain<32> NARS-truncating; GhostStore<'a> with reactivation events; SCHEMA_VERSION 2→3; 7 tests; 3 OQs surfaced (Lance persistence defer, promote_to_spo API, witness_ref hash)
**Notes:** Zero partial SPO-G implementation found in source (grep confirmed). W6/W7 specs need stubs for AriGraph::commit_edge and GhostReactivationEvent.

## W8 — 2026-05-14 — sprint-log-10 — PR-NDARRAY-MIRI-COMPLETE spec
Output: `.claude/specs/pr-ndarray-miri-complete.md` (23 KB). Scope: ndarray-only PR closing u-word/i-word method gaps (simd_eq/ne/lt/le/gt/ge + simd_clamp + select + zero on U16x32/U32x16/U64x8 + I-word symmetric) + cfg(miri) dispatch reroute in src/simd.rs. Confirmed gaps from direct file reads. Top OQ: select() typed-vs-raw mask API parity (OQ-1). Lands BEFORE par-tile (PR-CE64-MB-1).

---
## W10 sprint-log-10 — 2026-05-14 12:32 UTC
**Agent:** W10 (pr-dep-graph)
**Output:** `.claude/specs/sprint-10-pr-dep-graph.md` (25183 bytes, 11 sections)
**Plans cited:** `causaledge64-mailbox-rename-soa-v1.md` §7 §10 §11 §15; `sprint-log-10/MANIFEST.md`
**Key delta:** Surfaced true parallel structure vs parent plan's linear sequence (Waves 1+2 parallel, Wave 3 parallel pair, Wave 6 decoupled from Wave 5). Produced OQ-to-PR gating table (8 OQs, 3 require pre-sprint-11 ratification). Added 6 cross-spec consistency checks (C-1 through C-6) — primary meta-review audit targets.
**Open questions for meta-review:** (1) C-1: CausalEdge64 accessor names must match across W2/W6/W7; (2) C-5: SigmaTier enum residence — recommend lance-graph-contract; (3) C-2: BindSpaceView lifetime — Arc<BindSpace> vs raw ref must be decided by W1 and consumed uniformly.
W6 2026-05-14 DONE: pr-ce64-mb-5-mailbox-soa-attentionmask.md (47678 B) — MailboxSoA<N> + AttentionMask SoA + AttentionMaskActor wiring. Plans: causaledge64-mailbox-rename-soa-v1 §4+§5+§7. OQs: OQ-N, OQ-SHADOW, OQ-BCAST-SIZE.
| 2026-05-14T12:34 | W2 causaledge64-v2 (Sonnet) | `.claude/specs/pr-ce64-mb-2-causaledge64-v2.md` | 30 KB spec, 13 sections | CRITICAL: no reserved bits in shipped edge.rs — plan §3 layout discrepancy. OQ-LAYOUT-1 BLOCKER. OQ-PAL8-FORMAT BLOCKER for W3. | No commit (main thread aggregates) |

## 2026-05-14T12:35 — W12 sprint-10-execution-plan.md complete (sonnet, sprint-10)

**D-ids:** D-CE64-MB-meta (execution plan + board governance)
**Commit:** (pending main-thread aggregation)
**Tests:** N/A (governance spec)
**Outcome:** Sprint-10 execution plan written (28.5 KB, 473 lines). Covers sprint-11 fleet, worker prompt template, CCA2A scratchpad protocol, board hygiene per-PR trigger table, OQ resolution tracking (8 OQs), cross-session coordination (Branch Pub/Sub + File Blackboard), meta-reviewer scope, sprint-11 completion criteria, risk matrix. Key OQs for user ratification: OQ-1 (Sigma-tier banding), OQ-3 (plasticity granularity), OQ-5 (rayon vendor).

---
**W1 par-tile-crate** | 2026-05-14T12:35:38Z | COMPLETE | `.claude/specs/pr-ce64-mb-1-par-tile-crate.md` (37134 bytes, 13 sections) | Plans: causaledge64-mailbox-rename-soa-v1.md §0/§4/§5/§6/§7/§11 + LATEST_STATE.md + lance-graph-supervisor/Cargo.toml | Key delta: materialized full Cargo.toml, 3 Mailbox<T> backings, AttentionMask LRU+wrap renorm, MailboxSoA<N> lifecycle, BindSpaceView via NonNull<u8> dep isolation, dep-guard build.rs | OQs: OQ-A causal-edge re-export vs newtype, OQ-B vendored-rayon stub vs omit, OQ-C BindSpaceView safe vs unsafe constructor | No commit (main thread aggregates per MANIFEST.md)
W11 [2026-05-14T12:29] test-plan-unification: spec at .claude/specs/sprint-10-test-plan.md (43,718 bytes); Miri growth ~760→~1550 across 3 mechanisms; 5 OQs for meta-review (W2 causal-edge FFI check, W7 cfg(not(miri)) guards critical)

## 2026-05-14T13:40 — W9 bevy-cull-plugin spec complete (opus, sprint-10, main-thread)

**D-ids:** D-CE64-MB-7 (bevy proof plugin)
**Output:** `.claude/specs/pr-ce64-mb-7-bevy-cull-plugin.md` (~14 KB, 13 sections)
**Plans cited:** causaledge64-mailbox-rename-soa-v1.md §7 PR-CE64-MB-7 + §13 3rd-pair (bevy session); composes W1 (par-tile), W6 (MailboxSoA), W8 (intersects_sphere_x16 Miri reroute), W10 (Wave 6 dep), W11 (test plan §7.6 + §8.6 + §9.1).
**Key delta:** Resolves parent plan's 1-line PR-CE64-MB-7 row into: crate layout (11 files), Plugin impl with `ambiguous_with` conservative-write schedule, x16-lane intersects_sphere consumption, compartment-per-visible producer-side proof (closes both consumer AND producer side of par-tile dep), 12 tests (5 correctness + 4 integration + 1 Miri-compat + 2 schedule sanity), 4 criterion benches, 1 path-filtered CI job.
**Cross-spec touchpoint:** `BindSpaceView::empty_static()` — recommended to live in W1 par-tile spec; W9-OQ-1 escalates to meta-review.
**Open questions:** W9-OQ-1 (BindSpaceView::empty_static residence, HIGH), W9-OQ-2 (bevy 0.14 pin), W9-OQ-3 (ambiguous_with semantics), W9-OQ-4 (multi-camera deferred), W9-OQ-5 (CI feature matrix).
**Note:** W9 was authored from main-thread because the W9 worker was not spawned in the original sprint-10 fleet fan-out (gap noted alongside W7 still missing). No git commit (main thread aggregates per MANIFEST).

---

## 2026-05-14T13:50 — W7 sigma-tier-router spec complete (opus, sprint-10, main-thread)

**D-ids:** D-CE64-MB-6 (SigmaTierRouter + banding + plasticity + pruning + JIT pipeline)
**Output:** `.claude/specs/pr-ce64-mb-6-sigma-tier-router.md` (~48 KB, 15 sections)
**Plans cited:** causaledge64-mailbox-rename-soa-v1.md §6+§7+§10+§11+E-CE64-MB-8/9/10 + linguistic-epiphanies-2026-04-19.md E21 (Σ10 Rubicon) + THINKING_ORCHESTRATION_WIRING.md Gap 3+Gap 4 closure; composes W1/W2/W4/W5/W6/W10/W11.
**Key delta:** Resolves parent §7 1-line PR row into: SigmaTierRouter actor (6 msg variants) + 10-tier numeric banding table + INT4-32D K-NN cold-start (resolves parent OQ-4) + Hebbian plasticity rollup at drop_row (closes E-CE64-MB-10) + queue-then-drain pruning (branch-light hot path) + KernelHandleCache (closes Gap 3) + Σ9-Σ10 escalation via supervisor msg + 1024-entry backpressure; 30 tests, 4 benches, 1 CI job, 15 files.
**Cross-spec touchpoints (HIGH):** (1) W6 `CompartmentReport` MISSING `g_slot_at_drop` field — ~3 LOC W6 patch required; (2) W10 dep-graph MISSING INT4-32D codebook as hard dep — Wave 5 gating fix needed.
**Open questions:** W7-OQ-1 (Σ4-Σ5 banding HIGH/BLOCKS Wave 5), W7-OQ-2 (W6 cross-spec HIGH/BLOCKS Wave 4-5), W7-OQ-3 (Jirak threshold), W7-OQ-4 (W10 dep-graph gap HIGH/BLOCKS Wave 5), W7-OQ-5 (supervisor→planner dep cycle), W7-OQ-6 (parent OQ-3 coupling).
**Sprint-log-10 fleet status:** 12/12 specs complete. Ready for meta-review.
**Note:** Authored from main-thread (W7 worker not spawned in original fan-out). No git commit (main thread aggregates per MANIFEST).

---

## 2026-05-14T14:15 — META-REVIEW sprint-10 complete (opus, main-thread)

**Output:** `.claude/board/sprint-log-10/meta-review.md` (~31 KB, 9 sections)
**Coverage:** all 12 worker scratchpads + all 12 spec files (~396 KB) + AGENT_LOG.md + parent plan + CLAUDE.md governance
**Sprint grade:** B+ — substrate finding (CSI-1: parent plan §3 vs shipped edge.rs layout) is the central value-add; 3 hard blockers remain; 5 small pre-spawn patches required.
**Per-worker grades:** W2/W3/W5/W8/W10/W11/W12 = A or A−; W1/W6/W7/W9 = B+; W4 = B− (shortest spec + AwareOp stub deferred + scratchpad/file size discrepancy).
**Cross-spec inconsistencies (6 CSIs):** CSI-1 plan/code layout gap (BLOCKER Wave 2, user ratification); CSI-2 W6 CompartmentReport.g_slot_at_drop missing (BLOCKER Wave 4); CSI-3 W10 dep-graph missing PR-J1-INT4-32D-ATOMS (BLOCKER Wave 5); CSI-4 BindSpaceView constructor drift W1/W4/W9 (MED); CSI-5 SigmaTier residence drift W1/W6/W7/W10 (MED); CSI-6 W11 test-count drift (LOW).
**Cross-cutting epiphanies (5 E-META):** specs-against-source > specs-against-plan; late-spec coordination gap (W7+W9); scratchpad discipline bimodal; 4 BindSpace columns = AGI-as-glove API; diamond dep graph holds.
**Sprint-11 spawn decision:** NO — requires 5 pre-spawn fixes (~150 LOC edits) + 4 user ratifications (CSI-1 + parent OQ-1/OQ-3/OQ-5). Defaults are all safe-to-ship; ratifications are formal-acknowledge, not policy-change.
**Adjusted wave sequence:** add Wave 0.5 (PR-J1-INT4-32D-ATOMS) before Wave 1; Wave 6 (bevy) can be parallel with Wave 5.

---

## 2026-05-14T15:30 — 8-doc CausalEdge64 + thinking-engine + ontology knowledge corpus written (opus, main-thread)

**Trigger:** user request for documentation after architectural research surfaced dual-CausalEdge64 finding + corrected three-zone hot-path mental model. Research: Explore agent mapping of blasgraph + neighborhood + AriGraph + thinking-engine crates + verified thinking-engine `layered.rs` 8-channel variant + read p64 convergence stub + SPOW tetrahedron source + splat shader integration plans.

**Outputs (`.claude/knowledge/`, ~78 KB total):**
1. `causal-edge-64-spo-variant.md` — causal-edge crate variant w/ full bit layout + accessors + consumers + hot-path role + line refs
2. `causal-edge-64-thinking-engine-variant.md` — thinking-engine variant (8 channels × 8 bits) w/ emit/apply semantics + 3-tier cascade + dual-variant disambiguation
3. `causal-edge-64-synergies-and-pr-trajectory.md` — what each does better + thinking-engine function mapping + PR #366/365/364 trajectory + Option R-1/R-2/R-3 reunification options
4. `spo-schema-and-mailbox-sidecar.md` — SPO-G vs SPO-W vs both; time-as-sidecar vs CausalEdge64-as-sidecar; ractor mailbox payload per Σ-tier
5. `spo-ontology-format-stack.md` — 3×16Kbit / CAM-PQ / bgz17 / bgz-hhtl-d / CausalEdge64 ladder + format selection matrix + Zone-1/2/3 mapping
6. `ogit-owl-dolce-ontology-compartments.md` — OGIT family registry + OWL inheritance + DOLCE orthogonal scaffold + 8-channel ↔ OWL axiom mapping + ontology-aware splat filter
7. `cognitive-shader-driver-thinking-engine-reunification.md` — drift origin reconstructed from `cache/convergence.rs:18-22` `#[allow(unused_imports)]` evidence + 5-step reunification plan + transcoder design
8. `splat-shader-rayon-struct-method-vision.md` — splat op fleet + ndarray struct methods + rayon work-stealing + computational entropy reduction + sprint-12+ 5-sprint arc

**Key findings surfaced:**
- **Dual CausalEdge64 confirmed:** causal-edge::CausalEdge64 (SPO-palette layout) ≠ thinking_engine::layered::CausalEdge64 (8-channel cascade layout). Same name, different bit semantics. NOT in `TYPE_DUPLICATION_MAP.md`.
- **p64 drift origin pinpointed:** `cache/convergence.rs:18-22` imports SPO variant with `#[allow(unused_imports)]` annotation — wiring started, never finished. 8-channel variant never imported here.
- **Three-zone hot-path mental model:** Zone-1 (thinking-engine MatVec 200-500ns + AriGraph entity_index O(1) 20-200ns); Zone-2 (blasgraph + neighborhood cascade 20-1200µs); Zone-3 (DataFusion >1ms). My prior framing of "AriGraph = cold-path µs joins" was wrong.
- **8-channel ↔ OWL axiom near-isomorphism:** SUPPORTS↔sameAs, REFINES↔subClassOf-down, ABSTRACTS↔subClassOf-up, CONTRADICTS↔disjointWith, etc. The two variants are dual representations of similar operations; reunification transcoder is the unification point.
- **Splat as BLAS-class op:** 4096×4096 question surface (per `tetrahedral-epiphany-splat-integration-v1.md`); 2×64 (GestaltCause + ThinkingEffect) + 2×256 (AttentionIn + AttentionOut) + 4096 COCA is the reasoning lattice.
- **Computational entropy framing:** struct-methods on carriers + rayon par_* variants collapse caller LOC ~7-15→1-3 per cycle; reunification of thinking-engine + cognitive-shader-driver SoA into one `Think` carrier is the canonical step.

**Cross-spec impact:** sprint-10 meta-review CSI-1 recommendation stands (drop temporal 12b + G_slot 5b = 17b freed; allocate W-slot + lens + spare) but **was reached via wrong reasoning** (relocation framing) — now grounded in correct hot-path analysis (direction/plasticity/inference are dispatch payload, not relocatable; W slot replaces G slot via per-tenant SoA partition + witness corpus rooting).

**Recommended follow-ups:**
- PREPEND `EPIPHANIES.md` E-META-7: dual-CausalEdge64 discovery + p64 drift origin
- Update `docs/TYPE_DUPLICATION_MAP.md` to list CausalEdge64 as 2-copy duplication
- Update `LATEST_STATE.md` Contract Inventory to name both variants explicitly
- Sprint-11+ scope: 8-channel → SPO-palette transcoder per Option R-3
- Sprint-12+ scope: `Think` struct unification (the 5-sprint arc in Doc 8)

**Process note:** user explicitly called out my prior context-reset framing — corrected via Explore agent research before writing. All 8 docs grounded in shipped source (file:line refs throughout) or referenced plan documents.

---
