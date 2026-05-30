# post-438-integration-options-v1 — what to do next (council recalibration input)

> **READ BY:** council reviewers, integration-lead, the next session that lands on this branch.
> **Status:** OPTIONS LIST (pre-council). Composed 2026-05-30 after rebase onto post-#438 main.
> **Purpose:** the user asked for an integration plan AS A LIST OF POSSIBILITIES so the council can recalibrate via brutally honest review before autonomous auto-resolve. This is that list — not a single path, not a recommendation that bypasses review.

---

## 0. State snapshot (verified on-disk this session)

- **Branch:** `claude/activate-lance-graph-att-k2pHI` HEAD `8d75294b`, rebased onto `origin/main` HEAD `4b00d049`.
- **My 4 commits on this branch (all docs, no code):** survival dossier (66-entity inventory + strategies + tools), reconciliation doc (discovery_origin + Jirak), Jirak §7 verification + EPIPHANIES board prepend, CLAUDE.md dependency pin fix.
- **Merged since I started:** #436 (Aerial+ Rust crate, +2,693 LOC standalone), #437 (kanban + soa_view in contract, +4 files), #438 (DOLCE projector). #439 open (kanban Phase 2: lifecycle DAG + ExecTarget).
- **What is verified UNCHANGED in code post-rebase:**
  - `mod.rs:450 OdooConfidence = {Curated, Extracted, Conjecture}` — same 3 variants
  - 66 `pub const OdooEntity` declarations across 15 lanes — same count
  - `ProvenanceTier` / `discovery_origin` — still in ZERO `.rs` files
- **What is NEW on this branch via the rebase:** the 4 canonical `cognitive-risc-*` specs now live at `.claude/specs/` (they came in via main commits `d1635dbe`/`93ac0463`/`a16d0f41`/`45276eb3`).
- **Stale points in my own committed docs (NOT yet fixed):**
  - Reconciliation §8 says specs live only on `origin/claude/cognitive-risc-core-9PMW8` — WRONG since the rebase. They are on main and on this branch.
  - Survival dossier "Concrete next moves" §7 ("ARM-discovery Wave 1 = D-ARM-1 + D-ARM-2") — STALE. Wave 1 went D-ARM-13/14 (Aerial+ port + DOLCE projector) per #436/#438.

---

## 1. The 8 options (for council to rank)

Each option lists: **What** · **Why** · **Cost** · **Risk** · **Reversibility**.

### Option A — Retire the branch as-is

- **What:** Do nothing further. PR #435 is already merged; the 4 doc commits are on origin. Close out.
- **Why:** Sometimes the right move is to ship the body of work and walk away. Anything else risks scope creep and another round of "you fixed something I didn't ask for."
- **Cost:** Zero.
- **Risk:** Stale docs stay stale; the reconciliation §8 + dossier §7 wrongness rots in the knowledge base. Future sessions read it as ground truth.
- **Reversibility:** Trivial — a follow-up session can fix the rot later, but it'll cost a re-read of the canonical specs.

### Option B — Fix the 2 stale citations in my own docs, push, retire branch

- **What:** One commit: correct reconciliation §8 (specs are on main, list paths under `.claude/specs/`) and dossier §7 (Wave 1 went D-ARM-13/14, not D-ARM-1/2; restate the actual queue).
- **Why:** Cheap, honest, fixes lies-on-disk that I committed before the rebase.
- **Cost:** ~30 min, 1 commit, no code touched.
- **Risk:** None.
- **Reversibility:** Trivial.

### Option C — Lift ProvenanceTier + DiscoveryOrigin into `lance-graph-contract` (close OD-1/2/3)

- **What:** Implement the council-blessed canonical-set decision: add `lance_graph_contract::provenance::{ProvenanceTier, DiscoveryOrigin}` with the canonical `{Curated, Extracted, ArmDiscovered, Ratified}` per core spec + widen proposer-id to 6 bits (or move whole byte to u16) per core spec recommendation. Add a `Conjecture` aux variant or treat `Derived` as a separate orthogonal field per the wikidata spec's "reasoning store is orthogonal" stance. Wire the Aerial crate's `CandidateRule` to it.
- **Why:** #436 PR note explicitly flags this as the missing piece ("no lance-graph-contract carriers were added yet — `rule`/`translator` are the local seam until D-ARM-1/2 land the shared `CandidateRule`/`Proposer`/`ProvenanceTier` types"). My reconciliation doc identified this as the highest-leverage decision. The core spec is now on-branch so the citation is direct.
- **Cost:** 1 new contract module (~100 LOC), ~5 unit tests, wire the Aerial crate (1 small change in `lance-graph-arm-discovery`), update mod.rs:450 OdooConfidence to either alias or wrap the new ProvenanceTier (decision: which way the migration runs). Maybe 2-3 hours.
- **Risk:** Touches the contract crate (cross-repo blast radius — n8n-rs and crewai-rust depend on it). Decision OD-2 (Conjecture vs Derived as orthogonal axis) is non-trivial and load-bearing.
- **Reversibility:** Contract additions are additive; reversal is a follow-up PR. The byte-layout choice is harder to reverse once the WAL hardens, but the WAL is NOT YET hardened (verified).

### Option D — Help land #439

- **What:** Pull #439 into context. Review CodeRabbit comments if any, see whether the lifecycle DAG / ExecTarget design needs another savant pass. If clean, push toward merge. If not, raise the concerns.
- **Why:** It's the open scaffold for the Rubicon kanban machinery the rest of the architecture rests on. Held-up PRs are technical debt that compounds.
- **Cost:** Variable. If clean: ~30 min review pass. If issues: hours.
- **Risk:** Low. Review effort is bounded.
- **Reversibility:** N/A — review is read-only until I push code.

### Option E — D-ARM-1 + D-ARM-2 only (subset of C, no widening)

- **What:** Just lift `ProvenanceTier` + `Proposer` + `CandidateRule` into `lance-graph-contract` with the existing committed-plan value-set (2-bit tier, 3-bit proposer-id per #436's local types or per my plan §7.2). Do NOT make the widening decision; do NOT touch OD-2. Just unblock D-ARM-3+ which were waiting on the carriers.
- **Why:** Smallest unblocker; defers the contested decisions to a separate council convening.
- **Cost:** Smaller than C, ~1 hour.
- **Risk:** Bakes in the over-subscribed widths my reconciliation doc flagged as wrong. The reviewer who reads the §7 corrections will (correctly) ask why D-ARM-1 landed at the known-wrong width.
- **Reversibility:** Painful once shipped — the whole point of the reconciliation doc is that widening *after* the byte hardens is expensive.

### Option F — Wikidata loader (the big integration the wikidata-hhtl-load spec describes)

- **What:** Begin implementing the 115M-entity HHTL/CAM loader. 4 reduction levers + 2-pass streaming. The spec says ~38GB landing target.
- **Why:** Wikidata is the second domain the cognitive-risc N4 freeze test demands ("don't freeze SoA schema until ≥2 genuinely different domains have run through it"). Right now it's "Odoo's schema cosplaying as universal."
- **Cost:** Substantial. Multi-PR. Needs splat top-k codebook from jc (per #438), needs the Wikidata dump, needs basin sharding.
- **Risk:** Largest scope. Many open measurements (mask density, dedup rate, P279 fan-out — all explicitly flagged "untested" in the spec).
- **Reversibility:** Each PR can be reverted but the early architectural choices set the trajectory.

### Option G — Chess bring-up test

- **What:** Encode chess openings/methods/verbs as OWL/ttl, run proposers over it, see if GM-flavored legal candidates fall out of the same pipeline. Stockfish for ground truth.
- **Why:** The cognitive-risc-core spec calls this **the** falsifiable slice. It's tiny and decisive — does the universal SoA actually serve a second domain, or does chess need a column Odoo's SoA didn't anticipate?
- **Cost:** Bounded — OWL/ttl encoding is small, proposers exist (or scaffolded in #436), Stockfish is a subprocess.
- **Risk:** It might FALSIFY a load-bearing claim. That is actually the point of the test, but emotionally distinct from "another build-out PR."
- **Reversibility:** A failed test is a finding, not a regression.

### Option H — Cleanup target/ (3.3G of 3.5G repo)

- **What:** `cargo clean` in the workspace and the standalone crates.
- **Why:** Pure disk hygiene. The user just asked about disk space.
- **Cost:** Trivial.
- **Risk:** Next build is from scratch (slow).
- **Reversibility:** Trivial.

---

## 2. Combinations the council should consider

| Combo | Rationale |
|---|---|
| B + H | Smallest safe move: fix the rot, free 3.3G, walk away. |
| B + C | Fix the rot, then ship the byte the reconciliation doc was about — close the loop in one branch. |
| B + D + H | Fix rot, push #439 toward merge, free disk. Mostly review-only work. |
| B + C + D | Whole-branch closure: fix rot, ship the contract carrier, push #439. |
| B + C + G | Fix rot, ship the contract carrier, then run the chess bring-up to falsify or confirm the choices that just hardened. |
| C alone | The risky single-bet: change the byte grammar without first correcting the docs that name the wrong tier sets. Council should reject if so. |

---

## 3. My pre-council bias (state it explicitly for the council to attack)

I'm leaning **B + C** with **OD-2 = Derived as a separate orthogonal "reasoning provenance" field, NOT a tier value** (per wikidata spec's "orthogonal=beside not mixed in" stance), tier set = the canonical core-spec stable-4 `{Curated, Extracted, ArmDiscovered, Ratified}`, proposer-id = **6 bits (64 slots)** keeping the byte at u8 (cheaper, fits, matches the spec's stated alternative).

**Why I'm biased here:** it's the path that takes my own analysis seriously. That is exactly the kind of bias the council exists to attack — "of course you want to ship YOUR analysis."

**What I might be missing (where the council should poke hardest):**
- Is the canonical-stable-4 the RIGHT set, or is `Conjecture` (which is in code today) load-bearing in a way I haven't grokked?
- Is widening the byte right now actually safe given #437 just added `KanbanMove` (which is `Copy` and ≤16B and has a `const _` size assertion) — would the `discovery_origin` byte go INSIDE `KanbanMove` or alongside, and does either fit the assertion?
- Have I confused "what the spec says" with "what the team has converged on"? Specs are aspirational; main is reality.
- Is the chess bring-up (G) actually more urgent than the contract carrier (C)? The spec is emphatic that schema falsification is cheap on a board and expensive at domain 3-4.

---

## 4. Council brief (the prompt the reviewers receive)

The four reviewers are spawned in parallel with this prompt verbatim:

> Read this options doc (`.claude/plans/post-438-integration-options-v1.md`) + my reconciliation doc (`.claude/knowledge/discovery-origin-provenance-reconciliation-v1.md`) + the four canonical specs (`.claude/specs/cognitive-risc-*.md`, `.claude/specs/faiss-homology-cam-pq.md`, `.claude/specs/wikidata-hhtl-load.md`). Also read PR #436/#437/#438 PR bodies via `mcp__github__pull_request_read`.
>
> Rank the 8 options + 6 combinations. **State the one you would execute if you had the keys.** Be brutally honest — call out anything where the author's stated bias (§3) is wrong, where they're cherry-picking the spec, where the easy path is the wrong path. If you would execute something NOT on the list, propose it.
>
> Hard constraints: 1) do not propose work that needs the WAL to harden; the WAL is OPEN. 2) do not propose work that requires the canonical specs to be reconciled with themselves first (e.g. the core/wikidata `Derived` vs `ArmDiscovered/Ratified` split) — name it as a blocker if you think it is one. 3) you may propose retiring the branch (Option A) if you think the right move is "stop doing things."
>
> Output: one section with your verdict, one with what you'd execute first this session, one with what you'd flag as the biggest unstated risk in the options as written. Under 400 words total.

---

## 5. After the council — auto-resolve protocol

1. Wait for all four reviewers to return.
2. If 3 of 4 agree on a single option or combo → execute it.
3. If 2/2 split → tiebreak by: highest reversibility, lowest cross-repo blast radius, smallest cost. Pick.
4. If 4 different verdicts → take the union of "fix the obvious stale citations" (Option B) and stop. Surface the 4 dissenting verdicts to the user.
5. If any reviewer rejects everything → that reviewer's reasoning is treated as a veto on the contested options; pick the union of the non-vetoed set.

Execute. Commit. Push. Report.

End of options doc.
