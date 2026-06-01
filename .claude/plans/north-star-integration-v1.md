# North-Star Integration — current state → the two-ViewAngle destination (v1)

**Status:** RATIFIED (council resolved + gates ratified 2026-06-01; the open-WD framing below is kept for traceability — final calls are in § COUNCIL RESOLUTION). North star: `.claude/north-star/README.md` (the two reference diagrams). Design: `.claude/specs/atoms-styles-nal-planner-dto-unification-v1.md` §0–14. **This plan enumerates the open WIRING DECISIONS for the 5-developer council to iron out** before the A3→C7 + business-layer run.

## Current state (shipped + in-flight)

| Picture-2 band | Status | Code |
|---|---|---|
| §3 Reasoning (capstone) | ✅ merged #450 | `causal-edge::syllogism` — `Figure`/`figure()`/`syllogize()` (4 SPO figures) |
| §3 `rule` | ✅ merged #450 | `contract::nars::InferenceType::{to,from}_mantissa` + `From<grammar::NarsInference>` (A1) |
| §6 Meta-aware handle | ✅ merged #450 | `rung: RungLevel` on both `ThinkingContext` (A2) |
| §4 entry arrow | 🔵 branch | `PlanResult.emitted_edges: Vec<u64>` (A6) |
| §2 Resolver | ⬜ A3+A4 | `atoms::I4x32`→`I4x32D` carrier (pack `todo!()`); `ThinkingStyle→I4x32D` resolver; OGIT classifier |
| §3 moods | ⬜ A5 | style-biased figure try-order; the 4→64 mood expansion |
| §4 store / §5 kanban | ⬜ C7 | vart-backed `DemotionSink` + `VersionScheduler`; `surreal_container` (BLOCKED(C)) |
| §5 actors | ⬜ C6 | `lance-graph-supervisor` (ractor); `ConsumerEnvelope::Plan` + the supervisor→child forward |
| Business layer (P1) | ⬜ | OGIT inherited classes + bitmask; GoBD audit; the Rust/Elixir runtime; `/home/user/odoo` reference |

Reusable assets already present: `vart` (vendored `/home/user/vart`), `LanceVersionWatcher` (working LIVE primitive, callcenter), the merkle-chained `UnifiedAuditEvent` + `AuditSink`/`JsonlAuditSink`/`LanceAuditSink` (PR #364/#366), `head2head` (#446), `episodic_edges`/`DemotionSink` + `scheduler::VersionScheduler` (#446/#448), `OGIT` + `odoo` repos cloned.

## The open WIRING DECISIONS (council deliverable)

- **WD-1 — I4x32D layout: 2 halves vs 4-view OGIT.** jan decided "ride two i4×32 halves (64 lanes)"; Picture-2 §2 says *"integrated 4-view 32D latent"*. Is the `D` *dual* (2×32) or *4-view* (4×32, one plane per OGIT **O/G/I/T** stake — Objectives/Gaps/Impacts/Tradeoffs)? The 4-view reading grounds the carrier in the OGIT stakes lens. **Decide the carrier shape + the per-view/half semantics** (gates A3/A4).
- **WD-2 — The OGIT resolver (§2).** How does `I4x32D → OGIT class (multi-label) → best-practice template + attention bitmask` wire? Which OGIT class structure (the inherited classes from Picture-1)? Where do "best-practice templates" (playbooks/patterns/anti-patterns) live — `recipe.rs::PersonaRecipe` as the OGIT-inherited codebook? The bitmask = `ViewAngle` presence (D-VIEW-1, §10). Concrete adjacency, not a flat map.
- **WD-3 — vart key projection + connectome store (§4).** The `α.β.γ.δ.ε.ζ.η.θ` 8-position Base-16 address = the 8-byte `CausalEdge64`/`EpisodicEdges64` u64 → `FixedSizeKey::<8>::from(edge)`. Pin the BE key projection (S→P→O / family→local) for prefix-shared basins; LE `to_le_bytes` stays the value. Define the 4 indexes (by concept / rung / vart-version / style·rule) over vart.
- **WD-4 — version-update subscription + active Rubicon kanban (§5).** Confirmed **NO Delta Lake**: vart (hot MVCC clock) + Lance native versions (durable) + **SurrealDB LIVE** (the push trigger / "Live Actions"). Wire `LanceVersionWatcher`→`DatasetVersion`→`VersionScheduler::on_version`→`KanbanMove`→ractor (the `WatchReceiver→on_version→try_advance_phase` adapter). Unblock `surreal_container` BLOCKED(C) (the kv-lance fork dep).
- **WD-5 — ractor seam + Belief/Goal state (§5 / C6).** `ConsumerEnvelope::Plan` arm + the one `supervisor.rs:198` supervisor→child forward; emissions ride as `CollapseGateEmission` batons. Map the **Belief State (Weltmodell) + Goal State (Ziele)** BDI framing onto the substrate (belief = the connectome/NARS truth; goal = the kanban target).
- **WD-6 — Rust core vs Elixir (OTP/BEAM) runtime split.** Picture-1's title is "Business-Logik **in Elixir**", with a **Rust core** (WASM-ready, compiled rules) + Elixir (supervision trees, hot-code-upgrade, fault tolerance). Is `ractor` (Rust) the in-process actor mesh and **Elixir the outer OTP supervision + hot-upgrade shell**? How does the §13 dual-compile (cranelift template **vs** Elixir clause from one `FIGURE_RULES` table) realize this? **The biggest architectural decision** — the runtime boundary.
- **WD-7 — Figure 4→64 moods (§3).** Expand the 4 SPO-term-sharing figures to the "temporal & eternal syllogistic forms (64 moods)" — what is the increment, and does it stay firewall-clean (integer figure detection)? Or keep 4 + treat moods as the (figure × temporal-tag × copula) product?
- **WD-8 — GoBD audit & compliance wiring.** Map the existing merkle-chained `UnifiedAuditEvent` + `AuditSink` + vart-immutability + Lance time-travel onto the **6 GoBD criteria**. Determinism = the replay guarantee. Define the audit-log → export/archive (GoBD-konform) surface.
- **WD-9 — DeepNSM facts-proposer + dual-view head2head (§1).** Wire grammar → DeepNSM (meaning) ∥ OGIT (stakes) → `head2head` resolution. How the `grammar::inference` "grammar resolution IS reasoning" intent (now bridged by A1's `From`) triggers NARS→syllogism→reasoning.

## Council assignment (5 developers, one cluster each)

- **R1 — Carrier & Resolver:** WD-1, WD-2.
- **R2 — Representation & vart:** WD-3, WD-7.
- **R3 — Orchestration (surreal/Lance/vart subscription) & GoBD:** WD-4, WD-8.
- **R4 — ractor seam & the Rust/Elixir runtime split:** WD-5, WD-6 (the big one).
- **R5 — Reasoning & grammar facts-proposer:** WD-9, plus the §3 mood/firewall sanity for WD-7.

Each returns: per-WD **decision + rationale + concrete wiring (files/types) + risk/firewall note + offline-buildability**. Iron rules apply: read full files (`E-READ-NOT-GREP`), firewall (similarity proposes / CAM addresses), zero-dep contract, no Delta Lake on the subscription path.

---

## COUNCIL RESOLUTION (5-dev, 2026-06-01) — all 9 WD ironed out

| WD | DECISION | The one new wire | Offline? |
|---|---|---|---|
| **WD-1** (R1) | **[SUPERSEDED 2026-06-01: dual REJECTED — "64" = poles not lanes; shipped carrier is the single-vector `I4x32`/`I4x64`, #451]** ~~`I4x32D` = DUAL `{instance, reference}` (64 lanes)~~. 4-view **REJECTED** — O/G/I/T is the *business ViewAngle's* stakes question, never a carrier axis (firewall: "business is not an atom"). Confirms jan's §8. | `I4x32::pack/unpack` (un-`todo!()`) + `I4x32D`; unblocks `recipe.rs`. | ✅ |
| **WD-2** (R1) | OGIT resolver = 2-stage CAM: i4 distance **PROPOSES** → OGIT `ClassView` **ADDRESSES**. `recipe.rs::PersonaRecipe` = the inherited template codebook (fills plane 1). **Attention bitmask = `class_view::FieldMask`, NOT `ViewAngle`** (ViewAngle is the 4-bit *selector*; `attention = class.view_schema(angle) & row.presence_bitmask`). | `ThinkingStyle→I4x32` bridge (A4). Gated tail: `StyleRegistry::register_recipe` (A3.5, reopens Cranelift). | ✅ (attention path shipped) |
| **WD-3** (R2) | **4 vart trees** keyed by BE projection `[S,P,O,…tail]` (NOT `from(edge.0)`), value = frozen `to_le_bytes()`. Indexes: by-concept (primary) · by-rung · by-style·rule; **by-time is intrinsic MVCC**. Snapshot = **`Tree::clone()`** (no `Snapshot` type in the fork). | `project_be(edge)→FixedSizeKey<8>` + the 4-tree wrapper (vendored vart). | ✅ |
| **WD-4** (R3) | The IN-loop adapter `drive_in_loop` owned by the supervisor: `WatchReceiver→DatasetVersion→on_version→KanbanMove→try_advance_phase`. **Delta Lake redundant** (vart clock + Lance `versions()` + SurrealDB LIVE). surreal BLOCKED(C) = fork-coords only; loop ships **now** against `LanceVersionWatcher`. | **`pub observed_version(&self)->u64`** on `WatchReceiver` — the single load-bearing edit. | ✅ (gated: real surreal LIVE) |
| **WD-5** (R4) | ractor seam = one `ConsumerEnvelope::Plan(PlanStep)` arm + the `supervisor.rs:191` forward. **Belief = connectome/NARS-truth + EW64 column; Goal = `KanbanColumn` target** — BDI is a *reading* of the substrate, no new ractor state; batons = `CollapseGateEmission`. | the `Plan` arm + the forward (replace the `DispatchNotImplemented` stub). | ✅ |
| **WD-6** (R4) | **`ractor` (Rust) IS the runtime** — mesh **and** OTP supervision/fault-tolerance/hot-load. **No BEAM/NIF/port/gRPC.** "Elixir" = the gen_server *idiom* (shipped: `ExecTarget::Elixir=3` tag + `recipe.rs` open/closed) + an **optional build-time `elixir_clause()` source-emitter** (emit, never execute, off the hot path). §13 dual-compile = one `FIGURE_RULES` table → 2 pure lowerings. A live BEAM would break replay/GoBD + offline. | (decision only — no runtime change). `elixir_clause()` rides A3.5. | ✅ |
| **WD-7** (R2+R5) | **Keep the 4 `Figure` variants.** 64 moods = `figure(2b) × copula(2b) × temporal(2b)` **derived `u8` tag**, never truth math. **Literal 64-branch enum = firewall breach (R5).** Temporal source = structural chain-position. | `mood_tag(figure, copula, temporal)` (post-`syllogize`, reads existing fields). | ✅ |
| **WD-8** (R3) | GoBD **4-of-6 already shipped** (`UnifiedAuditEvent` merkle chain + `verify_chain`/`audit-verify` + `LanceAuditSink` + vart/Lance immutability). **Determinism = replay = the firewall = the compliance moat** (validated; LLM tools structurally can't). | Missing *Aufbewahrung*: **G1** retention WORM-seal · **G2** `audit-export` GoBD-Z3 bundle · **G3** Verfahrens-hook. All additive. | ✅ |
| **WD-9** (R5) | Wire the 3-stage relay (3-of-4 hops shipped via A1). Dual-view selector = `head2head` + a **new `WinnerCriterion::Repulsion`** (SemDiD, additive zero-dep); `margin` = the §14 tension → escalate. | **the 4096→256 palette projection** (driver fn: SpoTriple 12-bit rank → CausalEdge64 8-bit palette idx). | ✅ |

### CROSS-CUTTING INVARIANT (R5 ↔ R1) — ratify
**The 256-entry palette codebook is ONE.** The 4096→256 projection the DeepNSM proposer uses (WD-9) MUST be the *same* codebook the `I4x32D` OGIT resolver addresses (WD-2) — else the proposer and resolver fork. The WD-9 palette projection is lossy (4096→256): collisions are semantically-near (CAM nearest-centroid) and `syllogize` output is a *proposal* confirmed downstream (small `margin` → escalate), so the hazard is bounded — but the shared-codebook invariant is the guard.

### RATIFICATION GATES (jan)
- **G-CODEBOOK** — confirm the single shared 256-entry palette codebook (proposer projection == resolver codebook).
- **A3.5 deferred** — `StyleRegistry::register_recipe` + `elixir_clause()` emitter (reopens the Cranelift JIT surface) is its own later slice; confirm.
- **surreal BLOCKED(C)** — needs a human with fork access to supply the `surrealdb { git, branch, features=["kv-lance"] }` dep; not a design block.
- **GoBD hash** — FNV-1a is tamper-*evident*, cross-platform-deterministic (by design); if a regulator demands crypto non-repudiation, swap `AuditMerkleRoot::chain` → BLAKE3/SHA-256 behind a feature flag (`canonical_bytes()` stays frozen).

### UNBLOCKED RUN ORDER (all offline unless noted)
**A3** (I4x32D carrier) → **A4** (resolver bridge + FieldMask attention) → **A5** (figure-bias + mood_tag) → **WD-9** (palette projection + `WinnerCriterion::Repulsion`) → **C6** (`ConsumerEnvelope::Plan` + forward) → **WD-4** (`observed_version()` + `drive_in_loop`) → **WD-3** (vart 4-tree store) → **WD-8** (G1/G2/G3 audit surfaces). Gated tail: **A3.5** (JIT codebook + elixir emitter), **surreal LIVE** (BLOCKED(C) fork-coords).

### WD-6 refinement (jan) — the optional cold codegen is JITSon (cranelift fork), in lance-graph-planner

The §13 dual-compile's *compilation* lowering is **JITSon** — the JIT template format (`contract::jit::JitTemplate`, "JITSON"), compiled by the **cranelift fork** (AdaWorldAPI, same fork-everything pattern as ndarray/vart/elixir/surrealdb/ruff), living in **`lance-graph-planner`** (the `JitCompile` strategy + `jitson_kernel`; `StyleRegistry::warm_cache`/`register_recipe`). This is the **"optional cold codegen"**: style/figure kernels are compiled **COLD** (at style-registration / warm-cache time) into a cached `KernelHandle`, then the hot path calls the compiled fn-pointer — `ExecTarget::Jit`.

It is DISTINCT from `ExecTarget::Elixir` (the even-colder external `.ex` source emitter for a customer's *external* BEAM). So one `FIGURE_RULES` table lowers three ways:
- **`Native`** — interpreted, in-process (no compile).
- **`Jit`** — **JITSon → cranelift-fork** cold-compile → `KernelHandle`, hot-exec, in-process. **The primary optimization** ("optional cold codegen").
- **`Elixir`** — emit `.ex` source, external cold-path, optional (never executes in the north-star runtime).

The hot reasoning path stays integer/deterministic regardless of which lowering backs it — the cold compile is a *performance* substitution, not a semantic one (replay-safe: same table → same kernel → same `CausalEdge64`). This refines, not changes, WD-6: ractor is still the only runtime; JITSon/cranelift is the cold-compile *of* the in-process kernels, gated with the rest of the JIT codebook in **A3.5**.

---

## GATE RATIFICATION (jan, 2026-06-01) + the process doctrine

**Gates confirmed:**
- **G-CODEBOOK ✅** — the single shared 256-entry palette codebook is ratified: the WD-9 proposer projection (4096→256) and the WD-2 OGIT resolver address into the **same** codebook. The proposer and resolver must not fork.
- **A3.5 ✅ deferred** — `StyleRegistry::register_recipe` + the JITSon/`elixir_clause` emitters are their own later slice (reopens the Cranelift fork surface). The A3→A4 attention/resolution path does NOT wait on it.
- **surreal BLOCKED(C) — coordinate supplied:** fork = **https://github.com/AdaWorldAPI/surrealdb** (`features=["kv-lance"]`). Records the BLOCKED(C) coordinate; full offline-build unblock still needs the branch/rev (+ likely vendoring like vart). Per R3, the WD-4 loop ships **now** against `LanceVersionWatcher` regardless — surreal LIVE is the later swap.
- **GoBD hash — deferred (future idea):** keep FNV-1a (tamper-evident, cross-platform deterministic). Revisit BLAKE3/SHA-256-behind-a-flag only if a regulator demands crypto non-repudiation.

**PROCESS DOCTRINE (standing, per jan):** every plan is sandwiched —
1. **5-agent RESEARCH council BEFORE planning** (fan-out exploration → informs the plan; the R1–R5 council served this).
2. **3× BRUTALLY-HONEST council AFTER planning** (adversarial red-team → must clear before execution).
This is now the default rigor for any non-trivial slice. "Keeps it airtight."
