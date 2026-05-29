# Handover — arm-discovery-author → arm-discovery-impl

**Date:** 2026-05-29 20:30 UTC
**From session:** continuation on `claude/activate-lance-graph-att-k2pHI` (post PR #433 + PR #434 merges).
**To:** the implementation session that picks up `streaming-arm-nars-discovery-v1.md`.

**Plans + PRs to read first (in this order):**
  1. PR #433 (merged) — `style_recipe.rs` + epiphany-brainstorm-council + 5 savant cards. The interpretation layer + the ratification gate.
  2. PR #434 (merged) — `unified-soa-convergence-v1.md` + the 5 layered rulings (`E-SOA-IS-THE-ONLY`, `E-BATON-1`, `E-MAILBOX-IS-BINDSPACE`, `E-RUBICON-RACTOR`, the witness-arc handover).
  3. This handover.
  4. `.claude/plans/streaming-arm-nars-discovery-v1.md` — the plan this handover ships with. **Do NOT re-derive** — the plan is meticulous, 766 lines, every section spec-ratified.
  5. PR (this handover's PR) — `op_emitter.rs` (Phase 2 bucket-dispatch codegen) + this plan + this handover.

---

## What this session did (chronological)

1. **Pre-context** (summarized) — shipped Phase 1 `style_recipe.rs` (PR #433: 13 tests, DAtom catalogue, 7-rule cascade) + epiphany-brainstorm-council orchestrator + 5 savant cards. Council ran on `E-INTERPRET-NOT-STORE-1` and produced LAND verdict with corrections (`StyleRecipe` → `OdooStyleRecipe` rename, FNV exemption documented, P-1 litmus respected). All applied, merged.

2. **This session entry** — user re-shared two arxiv papers (Aerial+ 2504.19354v1, ontology M2M 2511.13661v1) and a paste of full paper text. Synthesized the two papers against the existing `op_emitter.rs` pipeline:
   - Paper 1 (M2M) is independent confirmation of the externalize-interpretation doctrine; its failure mode (5.81% — runtime behavior absent from static JSON) IS our Stage-2 dark-atom gap.
   - Paper 2 (Aerial+) supplies the missing upstream discovery leg. Its `(support, confidence)` mapping to NARS `(c, f)` is verbatim — `SpoBuilder::build_edge` consumes it natively.
   - The two papers bracket the architecture: discovery upstream, codegen downstream, SPO+NARS middle.

3. **Phase 2 op_emitter shipped** — `op_emitter.rs` (400 LOC, 12 tests) committed to this branch (`63f3e2ca`). Bucket-dispatch codegen: groups `OdooStyleRecipe` corpus by `OdooMethodKind`, emits deterministic Rust (RECIPE_* consts + per-kind Op struct + static slice). All 230 lance-graph-ontology tests green; zero warnings. Board hygiene done (`e7ee368f`).

4. **Rebased branch onto PR #434 merge.** Two doc commits inherited (`7c289678` unified-SoA + `eb5c4a58` Lance 6.0.1 stack pin); clean rebase.

5. **Reviewed PR #434's plan + handover.** Identified two corrections to fold in:
   - **OQ-11.2 W=16 witness-arc width** is too narrow for tracking multiple in-flight candidate rules per row. Proposed: separate `discovery_arc: [u32; D]` column, D=8.
   - **OQ-11.5 SoA-root version u16** doesn't disambiguate proposer provenance. Proposed: per-row `discovery_origin: u8` byte (2 bits tier + 2 bits proposer-id + 4 reserved).

6. **Authored `streaming-arm-nars-discovery-v1.md`** — 766 lines, 18 sections, 12 deliverables. Sectioned the upstream discovery leg into 5 stages (proposers → translator → hypothesis test → ratification → codegen), each with concrete code shape and threshold semantics. Plan written via `tee -a` chunking per user instruction (12 chunks).

---

## FINDING (high-confidence facts the next session inherits)

- **The five-stage pipeline shape is ratified:**
  ```
  parquet stream → Stage A (proposer) → Stage B (translator) → Stage C (hypothesis test)
  → Stage D (ratification — epiphany-brainstorm-council) → Stage E (op_emitter codegen)
  ```
  Each stage has a contract surface; the boundaries between Stages C and D form the determinism firewall (Stage D ratification gate is the only nondeterministic-to-deterministic transition).

- **Pair-stats is the default trunk; Aerial+ is fan-in.** Per the determinism boundary; Aerial+'s autoencoder is nondeterministic and must NEVER cross the ratification gate. The trunk is fully deterministic over windowed sufficient statistics.

- **ARM truth → NARS truth mapping is verbatim:**
  - `frequency` ← ARM `confidence` (= P(Y|X))
  - `confidence` ← `(support × n) / (support × n + k)` (NAL-9 default `k=1.0`)

- **I-NOISE-FLOOR-JIRAK is mandatory at Stage A.** Without Jirak-bound thresholds, the proposer's false-positive rate exceeds the substrate's noise floor and the SpoStore calcifies on noise. D-ARM-7 is non-skippable.

- **The discovery leg is strictly additive to PR #434.** No SoA-contract changes required for Waves 1-4. Wave 5a's `discovery_arc` column is a v1.1 follow-up; v1 lives with `edges` arc contention.

- **The new crate is `lance-graph-arm-discovery`.** Sits next to `lance-graph-ontology`. Depends only on `lance-graph-contract` + arrow/parquet. Zero deps beyond that for the default trunk; `tokio` + `serde_json` behind `arm-aerial` feature flag.

- **The papers ARE the support.** Karabulut 2025 §2 + §3.3 ratify the truth mapping; Abreu 2025 §4 ratifies the externalize-interpretation doctrine. No new conjectures — both are in print.

---

## CONJECTURE (load-bearing, ratify before acting on it)

- **OQ-ARM-2 — Jirak `p_moment` for Odoo data.** Plan defaults to `p = 3.0` (giving `n^{-1}` decay). This is conservative; actual measurement of Odoo `account.move` weak-dependence index is needed. The default is safe (over-strict), but D-ARM-12 bench should empirically pin `p` for typical Odoo feeds.

- **OQ-ARM-3 — NARS personality constant `k`.** Plan defaults to `k = 1.0` (NAL-9 standard). Different feeds may justify different `k` — higher `k` means more evidence needed for high confidence. Per-feed override is available; default is safe.

- **OQ-ARM-6 — Contradiction commit shape.** Plan proposes symmetric pair (one `CausalEdge` per side, back-pointer between). Verify this matches the existing `lance_graph::graph::spo::truth::Contradiction` primitive at D-ARM-5 time; if the primitive doesn't exist yet, surface a follow-up to `lance-graph-contract`.

- **The §7 corrections to PR #434.** The `discovery_arc` column and `discovery_origin` byte are author-stated by the planner (me), not council-ratified. They should pass through the council before D-ARM-6 lands. Cross-ref: `epiphany-brainstorm-council` invocation.

- **Aerial+ IPC overhead.** Plan assumes NDJSON-over-Unix-socket is cheap enough; at 100 K candidates/window from a subprocess that's 10-100 MB/window of NDJSON. Bench (D-ARM-12) should measure; may justify a binary IPC protocol if line-overhead dominates.

- **Reverse-fingerprint contradiction detection (OQ-ARM-8).** Plan defers concrete cutoff to D-ARM-5; the cutoff must be Jirak-derived, not hand-tuned. This is downstream of D-ARM-7's threshold helper.

---

## Blockers

- **PR #434 D-MBX-A3 landing.** Wave 5a (`discovery_arc` + `discovery_origin` corrections) depends on D-MBX-A3 having added the witness-arc handle column to the mailbox SoA. If D-MBX-A3 is still in flight when Wave 4 completes, Wave 5a can wait without blocking the discovery leg's operational utility.

- **`lance_graph::graph::spo::truth::Contradiction` primitive verification.** D-ARM-5's contradiction-commit path assumes a primitive that may not exist yet. Verification pass needed at Wave 4 entry. If missing, add to `lance-graph-contract` first.

- **The cargo prohibition for agents.** Per the session-stability rule (no cargo invocations from spawned agents — disk pressure constraint), all `cargo check / cargo test` runs are main-thread orchestrator only. Subagents do code review + write only; main thread verifies. This constraint is documented in CLAUDE.md / AGENT_LOG.md.

- **`tools/odoo-blueprint-extractor` Stage-2 enrichment.** The dark atoms (Money/Quantity/ApplyRate/EmitAmount/Event/FiscalCtx) don't fire today because `return_kind`/`semantic_role` aren't populated. ARM discovery can light them via runtime data, but the static extractor should ALSO fix this — D-ODOO-EXT enrichment is parallel work.

- **Aerial+ upstream access.** `DiTEC-project/aerial-rule-mining` and `AdaWorldAPI/aerial-rule-mining` are both outside the workspace MCP allowlist as of this session. Wave 7 (D-ARM-9, optional) is blocked on either (a) allowlist update, (b) user-pasted reference, or (c) re-implementation from paper Algorithm 1. (c) is feasible — the algorithm is in print.

---

## Open questions for the user

| # | Question | Default proposal | Blocks |
|---|---|---|---|
| OQ-ARM-1 | Default window size n? | 100K, per-Feed configurable | D-ARM-3, D-ARM-8 |
| OQ-ARM-2 | Jirak `p_moment` for Odoo? | p = 3.0 conservative | D-ARM-7 |
| OQ-ARM-3 | NARS personality `k`? | k = 1.0 NAL-9 standard | D-ARM-4 |
| OQ-ARM-4 | `RatificationQueue` persistence? | In-memory v1; persist v2 | D-ARM-6 |
| OQ-ARM-5 | Antecedent bound `a`? | Hard-cap 2 in pair-stats; Aerial+ for higher | D-ARM-3 |
| OQ-ARM-6 | Contradiction commit shape? | Symmetric pair w/ back-pointer | D-ARM-5 |
| OQ-ARM-7 | `discovery_arc` D=8 column day-one or v1.1? | Defer to v1.1; bench first | D-ARM-6 |
| OQ-ARM-8 | Inverse-fingerprint contradiction policy? | Cite Jirak; concrete cutoff at D-ARM-5 | D-ARM-5 |
| OQ-ARM-9 | Council ratification trigger? | Session-trigger; no webhook v1 | D-ARM-6 |
| OQ-ARM-10 | Aerial+ as separate crate or feature? | Feature inside; promote later | D-ARM-9 |

Plus the corrections-to-#434 ratification ask: do §7's `discovery_arc` D=8 and `discovery_origin: u8` byte get folded into D-MBX-A3 (the SoA owner's PR), or stay in D-ARM-6 (the discovery proposer's PR)?

---

## Recommended next-session entry sequence

1. **Read Tier-0** (LATEST_STATE.md, PR_ARC_INVENTORY.md, agents/BOOT.md).
2. **Read this handover + the plan** (`.claude/plans/streaming-arm-nars-discovery-v1.md`). Do not re-derive.
3. **Council ratification of corrections** (§7 of plan). Spawn the epiphany-brainstorm-council with `E-DISCOVERY-CODEGEN-BRACKET-1` candidate epiphany + the two §7 corrections. Expect LAND or LAND-with-revision; act on verdict.
4. **Wave 1 (D-ARM-1 + D-ARM-2)** — contract additions. One PR. Sonnet agent. Main thread runs cargo verify.
5. **Wave 2 (D-ARM-7)** — Jirak helpers. Pure math. One PR. Sonnet agent.
6. **Waves 3a + 3b in parallel** (D-ARM-3 pair-stats + D-ARM-4 translator). Two PRs. Two Sonnet agents in one main-thread turn.
7. **Wave 4 (D-ARM-5)** — hypothesis test. Opus agent (multi-source). One PR.
8. **Wave 5a** — corrections to #434 + queue impl. Two PRs. Coordinate with D-MBX-A3 author.
9. **Wave 5b (D-ARM-8)** — feed + projector. Sonnet agent.
10. **Wave 6 (D-ARM-10 + D-ARM-11)** — op_emitter filter + style_recipe rule. Sonnet agent. Trivial.
11. **(Optional) Wave 7 (D-ARM-9)** — Aerial+ IPC. Only if user signals demand.
12. **Wave 8 (D-ARM-12)** — end-to-end test + bench. Opus agent. Bench numbers inform OQ-ARM-2 and OQ-ARM-7.

---

## Risk to flag explicitly

The discovery leg ships an **upstream proposer node that didn't exist before in this architecture.** Three doctrinal risks to keep in mind throughout implementation:

1. **Don't promote Stage A → Stage E without Stage D.** The council ratification gate is the only nondeterministic-to-deterministic transition. Skipping it (e.g. "auto-ratify if confidence > 0.95") sounds appealing and is the Kahneman-Tversky System-1 trap. Reject.

2. **Don't conflate the witness-arc with a discovery-arc.** PR #434's witness arc is for *ratified* belief-state revisions; ARM-discovery's `discovery_arc` (proposed §7) is for *in-flight* candidate evidence. Cohabiting them in the same `edges` column pollutes the audit trail.

3. **Don't optimize before benchmarking.** D-ARM-12 bench establishes the actual throughput envelope and the actual `p_moment`. Premature SIMD/GPU work on Stage A is a distraction; the pair-stats inner loop is simple and the optimization target should be the FeedProjector's row-decode cost, not the counters.

---

## Cross-refs

- `streaming-arm-nars-discovery-v1.md` — the plan this handover hands off.
- `unified-soa-convergence-v1.md` (PR #434) — the SoA contract this plan writes against.
- `style_recipe.rs` (PR #433 / `OdooStyleRecipe`) — the interpretation layer that consumes ratified triples.
- `op_emitter.rs` (this branch) — the codegen layer that consumes ratified `OdooEntity` SoA.
- `epiphany-brainstorm-council` (PR #433) — the ratification gate.
- CLAUDE.md `I-NOISE-FLOOR-JIRAK`, `I-SUBSTRATE-MARKOV`, `I-VSA-IDENTITIES`, "The Click" — the doctrinal anchors.
- Papers: Karabulut 2025 (arxiv 2504.19354v1), Abreu 2025 (arxiv 2511.13661v1).

End of handover.
