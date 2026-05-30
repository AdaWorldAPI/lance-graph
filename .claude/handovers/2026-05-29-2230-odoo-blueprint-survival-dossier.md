# Handover — Odoo blueprint survival dossier

**Date:** 2026-05-29 22:30 UTC
**From session:** continuation on `claude/activate-lance-graph-att-k2pHI` (post PR #433 + PR #434 merges; PR #435 open).
**To:** any future session that needs to find the 66 curated Odoo entities, the three extraction strategies, the tools that back each, or the survival index of all of the above.

---

## Purpose of this handover

User-stated need (2026-05-29 21:30 UTC): "create a PR with all 70, strategies for extraction and tools for extraction so it doesn't get lost and if the session dies the other session has EVERYTHING."

This dossier exists so that **a fresh session reading only this file + the three companion knowledge docs has full context** to:
1. Find any of the 66 curated `OdooEntity` consts by name, model, or lane.
2. Understand which extraction strategy each existed entity came from.
3. Run any of the three extraction tool stacks.
4. Identify gaps (Stage-2 dark atoms, TIER-2 deferrals, L12-L15 post-EXT-6 pairing).
5. Pick up the work on PR #435 (ARM-discovery plan + op_emitter) without re-deriving.

---

## The four artifacts (all in `.claude/`)

| Artifact | Path | Purpose |
|---|---|---|
| **Inventory** | `.claude/knowledge/odoo-blueprint-inventory-v1.md` | Index of all 66 entities (alphabetical), per-lane summary, wave provenance, field/method density audit |
| **Strategies** | `.claude/knowledge/odoo-extraction-strategies-v1.md` | The three proposer legs (Curated/Extracted/ArmDiscovered): what each sees, what each emits, throughput, confidence, council posture |
| **Tools** | `.claude/knowledge/odoo-extraction-tools-v1.md` | Where to find each tool, how to run it, current status, gaps |
| **This handover** | `.claude/handovers/2026-05-29-2230-odoo-blueprint-survival-dossier.md` | Pointer + survival summary |

---

## The numbers (verified on disk 2026-05-29)

- **66 `pub const OdooEntity` declarations** across 15 lane files.
  - Wave 1 (commit `f5702675`): 21 entities, L1-L5, **5 Sonnet agents** (the question's "5 agents").
  - Wave 2 dedicated (`d30186e5`): 6 entities, L9 canonical FiscalPositionResolver.
  - Wave 2 + Wave-3 trims (`333a1ff2`): 39 entities, L6-L10 (incremental) + L11-L15 (5 Sonnet agents again).
  - EXT-3 follow-up (`c04adf10`): 0 entities added; back-fill of `OdooEntityKind` + `regulation_iri` across all 66.
- **11,563 LOC** across `crates/lance-graph-ontology/src/odoo_blueprint/l{1..15}.rs`.
- **130 lane tests** (Wave-1 left L1 with 0 tests at parent-module level; Wave 2+3 propagated per-lane test convention).
- **99,209 LOC of EXT-2 extracted backing** in `crates/lance-graph-ontology/src/odoo_blueprint/extracted/{account,…}.rs` (11 addon files).
- **48/53 = 90.6% TIER-1 coverage** per EXT-6 report (covered: L1-L11 to 100%; 5 TIER-2 deferrals on L11; L12-L15 are post-EXT-6 and need a fresh pass).
- **0 entities at risk of loss** — all 66 are in `main` as of `c04adf10`.

---

## The three extraction strategies (one-paragraph each)

1. **Curated (Leg 1, D-ODOO-BP-1b):** Sonnet agents project L-doc prose → typed `OdooEntity` consts. 5 agents per Wave, 1 per lane. Highest-trust tier. No council ratification needed (PR #433 council-bypass for human-authored content).
2. **Extracted (Leg 2, D-ODOO-EXT-2):** `tools/odoo-blueprint-extractor/` Python package walks Odoo source ASTs with stdlib `ast` module, emits `EXT_<NAME>` consts with `OdooConfidence::Extracted`. 12 TIER-1 addons covered; TIER-2 deferred. `pairing.rs::CURATED_EXTRACTED_PAIRS` surfaces conflicts for human review (canonical-on-conflict rule: curated wins).
3. **ArmDiscovered (Leg 3, this PR's plan):** Streaming pair-stats over parquet windows (deterministic trunk) + optional Aerial+ neural-symbolic fan-in. Outputs candidate `(s,p,o,f,c)` triples; routes through NARS revision (Stage C); novel candidates queue for council ratification (Stage D, MANDATORY). The council gate IS the determinism firewall before codegen.

---

## The three tool stacks (one-paragraph each)

1. **Sonnet-agent fan-out (Leg 1).** Not a code tool — a prompt + spawn pattern. Main thread spawns N `Agent(general-purpose, sonnet)` calls in parallel; each agent reads one L-doc + the typed `OdooEntity` surface + (optionally) an exemplar lane; emits one `l<N>.rs` file. Main thread reviews, trims, commits.
2. **`tools/odoo-blueprint-extractor/` (Leg 2).** Python 3 package, stdlib-only. 654 LOC entry + pairing + 950 LOC parsers (classes/fields/methods/state_machine/constraints/decorators/regulation). Run via `python -m odoo_blueprint_extractor --addons /home/user/odoo/addons --addon <name> --out <path>`. Stage-2 enrichment targets: richer `return_kind` inference in `parsers/methods.py`; lexical `semantic_role` mapper in `parsers/fields.py`.
3. **`lance-graph-arm-discovery` (Leg 3, queued).** Rust crate, specified in `.claude/plans/streaming-arm-nars-discovery-v1.md`. 2,400 LOC across 12 D-ARM-* deliverables. Default trunk is pair-stats (deterministic); `arm-aerial` feature gates Aerial+ Python subprocess IPC. Cross-cutting: `ruff_spo_triplet` in `AdaWorldAPI/ruff` provides the language-agnostic SPO IR.

---

## Five fastest paths into the corpus (if you have a specific need)

| If you need to … | Read this first |
|---|---|
| Find a specific entity by name | `odoo-blueprint-inventory-v1.md` §2 (alphabetical index) |
| Find all entities in a lane | `odoo-blueprint-inventory-v1.md` §1 (per-lane summary, opens lane file + line numbers) |
| Run the extractor on a new addon | `odoo-extraction-tools-v1.md` §2 (run procedure) |
| Light a Stage-2 dark D-Atom | `odoo-extraction-tools-v1.md` §2 (known gaps subsection — extractor enrichment targets) |
| Add a new ProvenanceTier (4th leg) | `odoo-extraction-strategies-v1.md` §6 (what this doctrine forbids — the rules for any new leg) |
| Implement ARM-discovery Wave 1 | `.claude/plans/streaming-arm-nars-discovery-v1.md` §9 (execution order) + `.claude/handovers/2026-05-29-2030-arm-discovery-author-to-impl.md` |

---

## What's at risk of getting lost across sessions (and isn't anymore)

**Pre-this-dossier risks:**
1. The 5-agent Wave pattern was tacit (no agent card formalized it).
2. The "Curated vs Extracted vs ArmDiscovered" doctrine was implicit across three different plans.
3. The 5 TIER-2 deferrals were buried in `extracted/COVERAGE.md` without cross-reference.
4. The L12-L15 post-EXT-6 pairing gap was unflagged.
5. The Stage-2 dark-atom enrichment targets were spread across `style_recipe.rs` comments, the extractor source, and the user's chat history.
6. The `ruff_spo_triplet` cross-language relevance was discoverable only via session-history pattern-matching.

**Post-this-dossier:** all six are explicitly captured. A session that loads only `odoo-blueprint-inventory-v1.md` + `odoo-extraction-strategies-v1.md` + `odoo-extraction-tools-v1.md` + this handover has the complete map.

---

## Concrete next moves (in priority order)

1. **Council ratification of `E-DISCOVERY-CODEGEN-BRACKET-1`** + the §7 corrections to PR #434. Spawn the 5-savant council. Verdict: LAND / REVISE-to-LAND / REJECT.
2. **Apply the Jirak math fix** to `streaming-arm-nars-discovery-v1.md` §4 (formula `n^{-(p/2-1)}` not `n^{-1/(p/2-1)}`, default `p ≈ 2.5` not `3.0`) — see cross-session review from #434 author (in session record).
3. **Coordinate with #434 author on unified-soa-convergence-v1.1 amendment** for the `discovery_arc` D=8 column + `discovery_origin: u8` byte (preferred path per cross-session review: amend v1 plan rather than carve into D-ARM-6).
4. **Stage-2 extractor enrichment**: `parsers/methods.py` `return_kind` inference + `parsers/fields.py` `semantic_role` lexical mapper. Lights 6 dark D-Atoms (Money/Quantity/ApplyRate/EmitAmount/Event/FiscalCtx) — independent of ARM-discovery.
5. **Post-EXT-6 pairing pass** for L12-L15 to bring the Wave-3 curated entities into `CURATED_EXTRACTED_PAIRS`.
6. **OQ-ARM-11 batch**: workspace MCP allowlist update for `AdaWorldAPI/aerial-rule-mining` + `surreal_container` (cross-session review identified these as shared blocker class).
7. **ARM-discovery Wave 1**: D-ARM-1 (`ProvenanceTier` enum) + D-ARM-2 (`Proposer` trait). Contract additions; one PR, one Sonnet agent, main thread verifies cargo.

---

## Risk to flag

The discovery leg (Leg 3) ships a new upstream proposer node. If it's implemented without observing these three invariants the substrate gets polluted:

1. **Don't promote ArmDiscovered → Ratified without the council** (the gate is non-negotiable; auto-ratification destroys the determinism firewall).
2. **Don't conflate the witness arc with a discovery arc** (committed revisions vs in-flight candidates are different audit trails; PR #435 §7 proposes the separation).
3. **Don't optimize before benchmarking** (D-ARM-12 bench establishes throughput envelope + `p_moment` empirically; premature SIMD/GPU work is a distraction).

---

## Cross-refs

- `odoo-blueprint-inventory-v1.md` — the 66-entity index.
- `odoo-extraction-strategies-v1.md` — the three legs.
- `odoo-extraction-tools-v1.md` — the tool stacks.
- `.claude/plans/odoo-business-logic-blueprint-v1.md` — Leg 1's plan.
- `.claude/plans/odoo-source-extraction-v1.md` — Leg 2's plan + EXT-6 report.
- `.claude/plans/streaming-arm-nars-discovery-v1.md` — Leg 3's plan (this branch, PR #435).
- `.claude/handovers/2026-05-29-2030-arm-discovery-author-to-impl.md` — Leg 3's implementation handover.
- `.claude/handovers/2026-05-29-1825-soa-convergence-author-to-impl.md` — PR #434's unified-SoA handover (the substrate that all three legs write into).
- `crates/lance-graph-ontology/src/odoo_blueprint/extracted/COVERAGE.md` — Leg 2's coverage report.
- `crates/lance-graph-ontology/src/odoo_blueprint/extracted/pairing.rs` — Leg 1 ↔ Leg 2 audit table.
- `CLAUDE.md` `E-SOA-IS-THE-ONLY`, `I-NOISE-FLOOR-JIRAK`, "The Click" — doctrinal anchors.
- Papers: Karabulut 2025 (arxiv 2504.19354v1) — Leg 3 anchor; Abreu 2025 (arxiv 2511.13661v1) — externalize-interpretation doctrine.

End of survival dossier.
