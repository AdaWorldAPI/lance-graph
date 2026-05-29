# Handover — soa-convergence-author → soa-convergence-impl

**Date:** 2026-05-29 18:25 UTC
**From session:** `017GFLBn` (branch `claude/splat3d-cpu-simd-renderer-MAOO0`)
**To:** the implementation session that picks up `unified-soa-convergence-v1.md`.
**Plan ratified in this session (PRs to read first):**
  - PR #418 (merged) — `bindspace-singleton-to-mailbox-soa-v1.md` + 5-ruling epiphanies (E-MAILBOX-IS-BINDSPACE / E-RUBICON-RACTOR / E-SOA-IS-THE-ONLY).
  - PR (this handover) — `unified-soa-convergence-v1.md` integration plan + handover.

---

## What this session did (chronological)

1. **2026-05-27** — opened PR #416 (recipes + atoms + savants + FIBU re-parent), addressed codex/CodeRabbit review on `b291ac5`, watched it merge.
2. **Investigated** SurrealDB / ractor / owned BindSpace status across the workspace; recorded the surreal POC reconciliation.
3. **2026-05-27** — drafted `bindspace-singleton-to-mailbox-soa-v1.md` (§0–§10) capturing the singleton-dissolution doctrine, hot/cold model, DTO vertical audit (p64-bridge conforms, engine_bridge re-encodes), and the LanceDB-leading / SurrealDB-view correction. Shipped as PR #418.
4. **2026-05-29** — captured five layered architectural rulings from the user (§11.1–§11.5):
   - One SoA, never transformed; mailbox SoA mutation IS the hot path.
   - Mailbox = full BindSpace as LE; witness = belief-state arc (no separate revision log).
   - Libet −550 ms anchors the Rubicon kanban in `surrealkv`-on-lance.
   - SPO-W witness is a *pointer* via AriGraph episodic Markov chain.
   - Counterfactual Staunen × Wisdom = plasticity spreaders.
5. **2026-05-29** — refined §11.3 (4-phase kanban: Planning / Cognitive work / Evaluation of goalstate / Commit·Plan·Prune) and §11.4 (AriGraph episodic Markov chain as the index space).
6. **2026-05-29** — added §11.6: the nine half-baked consumers (AriGraph / Vsa16k substrate / BindSpace / lance-graph cold / planner / shader-driver / callcenter / ontology AS IS / thinking-styles), SoA version byte at layout root, Lance 6.0.1 / LanceDB 0.29 / DataFusion 53 stack alignment, planner DTO surface overhaul.
7. **2026-05-29** — drafted `unified-soa-convergence-v1.md` (this plan): integration sequence across all 9 consumers, per-deliverable specs for D-MBX-A2/A3/A4/A5/A6/7/8/9/10/11/12, OQ catalogue (11.1–11.8), risk matrix, dependency graph, success criteria.

---

## FINDING (high-confidence facts the next session inherits)

- **D-MBX-A1 columns landed** between PRs #418 and #433. `mailbox_soa.rs` now carries `edges: [CausalEdge64; N]`, `qualia: [QualiaI4_16D; N]`, `meta: [MetaWord; N]`, `entity_type: [u16; N]`. Verified in the current `mailbox_soa.rs` (lines 67–83).
- **Workspace stack pins** (verified `Cargo.toml` 2026-05-29): `arrow = "58"` ✓, `datafusion = "53"` ✓, `lance = "=6.0.0"` (target =6.0.1), `lancedb = "=0.29.0"` ✓. Only one bump pending (D-MBX-11), files identified.
- **The `ResonanceDto` dup** (`TD-RESONANCEDTO-DUP-1`) is P3 confirmed (not a compile error — distinct modules `dto::ResonanceDto` and `awareness_dto::ResonanceDto`); deferred to D-MBX-2 per user.
- **`epiphany-brainstorm-council`** is a pre-merge gate for `EPIPHANIES.md` additions (shipped in PR #433). This session bypassed the council for the §11 epiphanies because they are *author-stated* by the user, not derived. The plan IS open to council review on spec content.
- **`p64-bridge`** is the conformance template — already LE-types-to-palette with no re-encode. Code-anchor for D-MBX-7.
- **`surreal_container`** is BLOCKED(A/B/C/D); D-MBX-11 removes BLOCKED(A); the rest still need a fork-access human (OQ-11.6) — long-standing.
- **`SigmaTierRouter`** (D-CSV-10) shipped in PR #388; D-MBX-8 adds the −550 ms wall-clock stamp on top of it.
- **`E-NORMALIZED-ENTITY-1`** (2026-05-28) informs D-MBX-A6: typestate carrier (`NormalizedEntity<Stage>`) is the pattern for re-expressing planner DTOs as SoA-row-lenses.

---

## CONJECTURE (load-bearing, ratify before acting on it)

- **OQ-11.1** — Staunen × Wisdom plasticity spread radius / decay. Plan default: radius = 3, decay = bump / (1 + |offset|), column-local in v1. Needs user ratification before D-MBX-A4 lands.
- **OQ-11.2** — Witness arc width W. Plan default: W = 16 (~64 B/row at u32 handles). Needs user ratification before D-MBX-A3 lands.
- **OQ-11.5** — SoA version field width. Plan default: `version: u16` at layout root; no per-column version stamps in v1. Needs user ratification before D-MBX-10 lands.
- **OQ-11.7** — `lance-graph-planner` DTO overhaul: clean break vs feature-gated coexistence. Plan default: feature-gated per `I-LEGACY-API-FEATURE-GATED`. Needs user ratification before D-MBX-A6 cuts over.
- **The kanban "Plan" loop closes the active-inference cycle.** If you observe a session where the system stops thinking before reaching homeostasis floor, suspect the Plan-branch in column 4 is not re-entering Planning correctly.
- **`witness_arc` rotation only after Commit/Prune.** If the arc rotates mid-cycle, witnesses are lost before SPO-G calcification — that's a P0 bug.

---

## Blockers

- **`PR-NDARRAY-MIRI-COMPLETE`** — close `U16x32 / U32x16 / U64x8` SIMD method gaps. Cross-repo work in `AdaWorldAPI/ndarray`. Blocks D-CE64-MB-1-impl and therefore the whole P3+ chain.
- **`D-CE64-MB-1-impl`** (par-tile crate apex) — already specced (Sprint-11 W1) but not yet implemented. Blocks D-MBX-A2/A3/A4/A5.
- **`surreal_container` BLOCKED(B/C/D)** — surrealdb fork URL + branch + `kv-lance` feature flag. OQ-11.6. Needs a fork-access human. Blocks D-MBX-9 (kanban view). D-MBX-11 removes BLOCKED(A).
- **CLAUDE.md "The Click" / `Vsa16kF32` doctrinal update** (OQ-11.4) — must precede D-MBX-5 (`BindSpace` singleton + `Vsa16kF32` plane deletion). Separate doc-PR.
- **Cargo prohibited in this session** (user-stated 2026-05-29 over stability concern). Next session should verify cargo prohibition is lifted before running tests; if still in effect, defer cargo and continue spec work.

---

## Open questions for the user

| # | Question | Default proposal | Blocks |
|---|---|---|---|
| OQ-11.1 | Staunen × Wisdom spread radius/decay/scope? | r=3, decay 1/(1+|offset|), column-local v1 | D-MBX-A4 |
| OQ-11.2 | Witness arc width `W`? | W = 16 (~64 B/row) | D-MBX-A3 |
| OQ-11.3 | Need separate "vetoed"/"ghosted" kanban columns? | No — Prune is terminal-veto; ghost preempt drops pre-column-2. | D-MBX-9 |
| OQ-11.4 | When does CLAUDE.md "The Click" doctrinal update land? | Before D-MBX-5. Separate doc-PR. | D-MBX-5 |
| OQ-11.5 | SoA version field width? | u16 at layout root; no per-column stamps in v1 | D-MBX-10 |
| OQ-11.6 | surrealdb fork URL + branch + feature flag? | unknown — needs fork-access human | D-MBX-9 |
| OQ-11.7 | Planner DTO overhaul: clean break or feature-gated? | feature-gated per I-LEGACY-API-FEATURE-GATED | D-MBX-A6 |
| OQ-11.8 | D-MBX-12 sub-PR sequencing? | 12.4 → 12.5 → 12.6 → 12.7 → 12.1 → 12.9 → 12.2 → 12.8 | D-MBX-12 |

---

## Recommended next-session entry sequence

1. **Read Tier-0 + this handover + `unified-soa-convergence-v1.md`.** Do NOT re-derive — the plan is meticulous.
2. **Confirm cargo prohibition status** with the user.
3. **Ratify the four blocking OQs** (11.1 / 11.2 / 11.5 / 11.7) via AskUserQuestion or direct ask. The plan defaults are sensible but they ARE defaults.
4. **Pick the next deliverable:**
   - **If cargo allowed:** start D-MBX-11 (mechanical Lance 6.0.0 → 6.0.1 bump). Verify with `cargo check`. Push.
   - **If cargo prohibited:** start D-MBX-10 spec work (version gate design + test plan), or the CLAUDE.md doctrinal update (OQ-11.4) as a docs-only PR.
   - **In parallel:** push for `PR-NDARRAY-MIRI-COMPLETE` resolution (cross-repo, blocking P3+).
5. **Do NOT bypass `epiphany-brainstorm-council`** for any new derived epiphany. User-stated rulings remain author-stated; derived insights from spec review go through the council.
6. **Board hygiene every commit.** LATEST_STATE + PR_ARC + STATUS_BOARD + EPIPHANIES (if applicable) update in the SAME commit as the change.

---

## Code anchors (don't re-grep these)

- `crates/cognitive-shader-driver/src/mailbox_soa.rs` (D-MBX-A1 columns at lines 67–83; A2/A3/A4/A5 extend here)
- `crates/cognitive-shader-driver/src/bindspace.rs` (singleton to dissolve in D-MBX-3/5)
- `crates/cognitive-shader-driver/src/driver.rs:56` (`Arc<BindSpace>` holder)
- `crates/cognitive-shader-driver/src/bin/serve.rs:29` (`BindSpace::zeros(4096)` to remove)
- `crates/cognitive-shader-driver/src/engine_bridge.rs:199` `busdto_to_binary16k` / `:310` `unbind_busdto` (re-encode seam to collapse in D-MBX-2)
- `crates/lance-graph-contract/src/cognitive_shader.rs:382` (`ShaderCrystal.persisted_row`)
- `crates/lance-graph-ontology/src/registry.rs:39` (`LazyLock<NamespaceRegistry>` — AS IS)
- `crates/lance-graph-ontology/src/lance_cache.rs` (ontology cache — AS IS)
- `crates/surreal_container/src/lib.rs` (BLOCKED view layer)
- `crates/p64-bridge/src/lib.rs` (conformance template)
- `crates/thinking-engine/src/dto.rs:40,59,120` (`StreamDto`/`ResonanceDto`/`BusDto` to collapse in D-MBX-2)
- `crates/thinking-engine/src/awareness_dto.rs:21` (the second `ResonanceDto` — TD-RESONANCEDTO-DUP-1, Deferred)

---

## Provenance

User-stated rulings recorded verbatim in the plan:
- §11.1: *"the same SoA is the one and only SoA consumed and transmitted everywhere, never transformed"*
- §11.2: *"the mailbox needs to have everything that BindSpace had reinvented as little endian contract"*
- §11.3: *"planning > ractor mailbox owned SoA > cognitive work > evaluation of goalstate > commit vs plan vs prune"*
- §11.4: *"the SPO-W witness is the pointer via AriGraph episodic/belief state arc array … [pointer to] other mailboxes in the AriGraph episodic Markov chain"*
- §11.5: *"counterfactual Staunen and wisdom should become helpers of spreading plasticity"*
- §11.6: *"all have to consume the same SoA from A-Z … the SoA can be versioned … for surrealdb the versioning gets aligned with lance 6.0.1 / lancedb 0.29 / datafusion 53"*

These rulings are **load-bearing** for the entire plan. Do not paraphrase or "fix" them without consulting the user.
