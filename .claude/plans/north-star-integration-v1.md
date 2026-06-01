# North-Star Integration — current state → the two-ViewAngle destination (v1)

**Status:** PROPOSAL / pre-council. North star: `.claude/north-star/README.md` (the two reference diagrams). Design: `.claude/specs/atoms-styles-nal-planner-dto-unification-v1.md` §0–14. **This plan enumerates the open WIRING DECISIONS for the 5-developer council to iron out** before the A3→C7 + business-layer run.

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
