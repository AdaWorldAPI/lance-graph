# EAW-6 — Chess-first temporal NNUE / Thought-Git / DR64³ crosswalk and idea conservation

**2026-10-27. STATUS: docs-only research scratchpad. NOT tested, trained, implemented, admitted or merge-authorized.** Independent main-based review PR. No runtime, ABI, board rewrite or new numerical implementation.

## Operator thought chain: preserve before reduction

1. Law is a dense network of source-qualified obligations, exceptions, precedence, effective times and convergent rules: Odoo taxability can receive several valid contributions, qualified rates are subordinate, and a shared determination may route to two *independently identified* ledgers. Do not confuse debit/credit fields with two ledgers. The goal is to make **'es kommt darauf an'** computable as minimal *deciding conditions*, source-aware explanations and counterfactual alternatives; not an autonomous legal verdict.
2. DT64 is *versioned replayable holographic addressing/binding*, NOT an executor or a guarantee of physical source immutability. DR64³ proposed views: Identity × Property/Edge × Predicate(ON/WHEN/DO typed AST), or Identity × Property/Edge × CausalEdge64 evidence. Presence vs covered absence vs undiscovered/UNKNOWN must differ. CausalEdge64 is payload, not 64 additional tensor coordinates; no fabricated pairwise triples. Multiple mappings onto one native polyfill and many-rules→one-state→multiple-destinations remain intact.
3. Physical selection can be factored ON×DO bitmap + witnessed Quack WHERE; TERNLOG computes a three-input truth table over **aligned Boolean planes**, not an entire three-axis tensor. K3 TRUE/FALSE/UNKNOWN, completeness, ACL, source version, guard AST, source-method multiplicity, provenance, law ordering and exact numeric rounding persist outside the terminal selection bit.
4. CausalEdge64 Epi5 bits 59..63 are the declared Topology2×Certification3 categorical code; bits 46..49 activation/legacy and 50..52 maturation/plasticity are *overlapping declared readings*, no unchecked aliases. CE64 W bits 53..58 = LIVE thought 0..63, never a durable pointer or sealed evidence root. Identity of an occupied Thought-Git slot includes generation and immutable baseline witness. Learned activation does not grant causes/certification/permission.
5. Think of a factorized `64^n` game-theoretic matrix: active coordinates are bounded and witnessed, no mandatory dense allocation or 64^n concurrent computations. Exact DR64/RelAlg/Datafrog-style folds can reason within one world; NARS/ARM/NSL/NNUE rank candidate questions, never certify facts or rules.
6. **Chess as first exact, bounded testbed.** T0 is one legal position with a strictly pinned knowledge horizon. T1..T63 are *independent simulated future move/feature trajectories* branching from T0. Simulated future is allowed; reading *actual historically future* game plies, PGN result or deep teacher label at Strict time is forbidden. Reuse or cycle the 63 speculative slots only with fresh generation-aware identity. No sibling-world union into a physical legal position.
7. NNUE feature-transformer incremental update follows exact `refresh(before) - removed + added` under the model's declared wrapping i16 semantics; some transitions require refresh (king bucket or feature schema). Borrow model weights and source-bound sparse feature indices; delta-updated accumulator must equal full refresh. This is **not** a neural answer oracle: teacher/search labels and legal movegen remain separate. Direct chess code reuse across GPL-3.0 / Apache-2.0 boundaries requires explicit license review; prefer test-only cross-repo integration or independently authored adapter.
8. **Hindsight blindness / rediscovery:** `lance-graph-planner/src/temporal.rs` canonically defines Strict/Aware/Retro and classify/deinterlace. `lance-graph-contract/src/temporal_pov.rs` only constrains the version range; full classification requires planner. **Default QueryReference is unbounded** and unsafe for strict-as-of. Pin `QueryReference::at(v,0)` / `TemporalPov::at(v,0)` and source data dependencies at ply v. `stockfish-rs/examples/hindsight_stream.rs` shows real PGN outcome oracle; `rung_hindsight.rs` NNUE-final-eval oracle was INCONCLUSIVE due to mate/unsettled outcomes; never recycle that weak oracle. `DecisionEpisodeV1` has physically separate StrictObservation and RetroLabel; `expert_iteration_stream.rs` offers full-window depth teacher, stability labels and deterministic replay. Deep teacher is revealed only *after* Strict candidate ordering.
9. Learning is **selective lookahead/feature prioritization**, not chess rules nor legal certification. Compared arms: frozen search ordering, ARM heuristic, seeded random, NNUE-style incremental ranker, privileged teacher upper bound. Hold equal node/time/thought budgets, preserve source-disjoint games and exact teacher score bounds. Include teacher withdrawal, cold+hot cost and unseen tactics. A game is a temporal version stream; 0+63 is *resident frontier*, not search depth and not 64 completed chess lines.
10. After chess falsifiers, try tiny *synthetic* legal heuristics: independent sufficient supports A or B, conjunctively required A and B, source-attested exception C, effective-date rate change, one state to two typed destinations, explicit K3 UNKNOWN. Replay alternate worlds and learn which missing fact to ask for. Use independent deterministic scalar AST reference. No legal ontology imports, actual-tax claims, jurisdictional advice, effect writes or invented laws until source-owned harvest admission.
11. Critical negative controls: identical chess position with different historical episodes; transposition vs replay identity; strict/retro data mutation; source-scope/ACL/codebook/net version drift; AND/OR/logical-convergence vs lost derivation; accidental thought union; feature hashing alias; duplicate evidence roots; false causality; incorrect numeric overflow/rounding; stale detached Keep (#1792 1,032 vs fresh 31).
12. Operator-rejected closed-unmerged [#1812](https://github.com/AdaWorldAPI/lance-graph/pull/1812) is an epistemic quarantine. **Never** mine its code, results, design, tests or suggested techniques. EXO-50 is an append-only coordination inbox, not authority or proof that a cross-account CC session received the message.

## Checked current owning work / avoid reinvention (2026-10-27)

| Owner | Current source and boundary |
|---|---|
| [#1819](https://github.com/AdaWorldAPI/lance-graph/pull/1819) | OPEN non-draft docs; ON×DO candidate + witnessed Quack WHERE, source-guard multiplicity; no watch/runtime right |
| [#1820](https://github.com/AdaWorldAPI/lance-graph/pull/1820) | OPEN DRAFT EAW-0 full DR64³/CE64 scratchpad and EXO-50 receipt |
| [#1821](https://github.com/AdaWorldAPI/lance-graph/pull/1821) | OPEN DRAFT EAW-1 exact presence/covered absence/UNKNOWN, source triple and focus |
| [#1822](https://github.com/AdaWorldAPI/lance-graph/pull/1822) | OPEN DRAFT EAW-2 Epi5/activation projected evidence, declared class/rail |
| [#1823](https://github.com/AdaWorldAPI/lance-graph/pull/1823) | OPEN DRAFT EAW-3 native TERNLOG/RelAlg aligned-plane/fixed-point gates |
| [#1824](https://github.com/AdaWorldAPI/lance-graph/pull/1824) | OPEN DRAFT EAW-4 T0+63 caller-level world isolation and conditional synthesis |
| [#1825](https://github.com/AdaWorldAPI/lance-graph/pull/1825) | OPEN DRAFT EAW-5 source-grounded Odoo/OGIT/legal and optional learning |
| [#1814–#1818](https://github.com/AdaWorldAPI/lance-graph/pull/1814) | CFA typed incidence, world ablations, ARM, optional NSL/ONNX, cold proof/optimization |
| [#1792](https://github.com/AdaWorldAPI/lance-graph/pull/1792), [#1799](https://github.com/AdaWorldAPI/lance-graph/pull/1799) | MERGED lab / docs respectively, escaped Keep, DT64 Q0/physical contract remain open |
| [stockfish-rs temporal](https://github.com/AdaWorldAPI/stockfish-rs/blob/main/examples/temporal_replay.rs) | Documented 34/34 version projection and 11/11 out-of-order accumulator replay (from pinned historical probe); no Thought-Git integration |
| [stockfish-rs decision episode](https://github.com/AdaWorldAPI/stockfish-rs/blob/main/docs/DECISION-EPISODE-V1.md) | StrictObservation vs RetroLabel, teacher search, identity and leakage gates |
| [Stockfish master research plan](https://github.com/AdaWorldAPI/stockfish-rs/blob/main/.claude/plans/stockfish-harvest-64x64-v1.md) | D-CHESS-NARS-0 NNUE teacher→NARS→teacher withdrawal; not a replacement target |
| [Stockfish perturbation synthesis](https://github.com/AdaWorldAPI/lance-graph/blob/main/.claude/knowledge/stockfish-nnue-as-perturbation-cascade.md) | Qualified temporal/spatial NNUE correspondence, one byte-exact anchor and many hypotheses |
| [EXO-50](https://linear.app/exo-red/issue/EXO-50/agent-message-bus-gpt-xgrok-ccsession-id) | Existing 2026-10-27T15:01:32Z EAW-0..5 handoff; new messages must be append-only |

## Owner and evidence firewall

EXO-46 global↔local binding / snapshot; EXO-73 lawful recursion; EXO-98 / 148 Thought-Git API and durable identity; EXO-109 independent oracle; EXO-127 D-DRMC algebra/law; EXO-132 carrier closure; EXO-133 synthetic vs real; EXO-135 evidence→law; EXO-140 ADMISSION-B; EXO-146 CausalEdge64 aliases; EXO-152 D-NAM numerical semantics. Physical DR³ third-axis NFR-OP-2, escaped Keep NFR-OP-1 open. No changes to those owners in this plan.

## Proposed NEXT **independent, main-based documentation** gates

- **EAW-7:** Thought-Git × temporal chess strictly pinned T0 and ≤63 independently witnessed speculative positions; full legal movegen, path lineage, isolation and no-hindsight falsifiers.
- **EAW-8:** Witnessed DR64³/Epi5 feature mapping ↔ NNUE-style sparse delta accumulator; exact numerical parity, feature invalidation, licence separation and cost gates.
- **EAW-9:** Hindsight-blind discovery and teacher withdrawal: Existing DecisionEpisodeV1 Strict/Retro, deep full-window teacher, game-disjoint split; compare frozen/ARM/NNUE and conditional synthesis under fixed budget.
- **EAW-10:** Small source-declared rule AST/legal heuristics after chess; alternative/joint support, exceptions, rate and typed dual-destination routing; no real law claim.

Conceptual dependencies are review order, NOT stacked branch ancestry. Each proposed stage needs a later independently approved test-only probe; no speculative source effects.

## Precise falsifiable thesis

Given one as-of historical chess board and the same frozen model, a ≤63-slot generation-aware Thought-Git frontier can enumerate independently valid hypothetical next states, compute NNUE deltas exactly relative to full refresh and select promising branches **without ever importing later historic outcomes into Strict-time selection**. Only after the decision does an independent deep teacher reveal hindsight labels. A learned ranker passes only if a held-out benefit survives matched budgets and teacher withdrawal. Transfer the same separation to *simple legal rule ASTs* only after the chess oracle is robust.

**No executed tests or admitted new performance numbers in this scratchpad.**

## Actual GitHub PR receipts and EXO-50 post (2026-10-27, append-only scratchpad correction)

All five have been CREATED as independent `main`-based **open drafts**, exactly one new research-plan Markdown file per PR; no Rust, ABI, runtime, benchmarks, training, other owners' plan edits or merge.

| New stage | Exact PR | Distinct proposed gate |
|---|---|---|
| EAW-6 | [#1826](https://github.com/AdaWorldAPI/lance-graph/pull/1826) | This idea-preservation scratchpad and checked owner/source crosswalk |
| EAW-7 | [#1827](https://github.com/AdaWorldAPI/lance-graph/pull/1827) | Strict temporal T0+≤63 chess world replay and no-hindsight negative tests |
| EAW-8 | [#1828](https://github.com/AdaWorldAPI/lance-graph/pull/1828) | NNUE sparse feature delta/refresh numerical oracle and version/alias guards |
| EAW-9 | [#1829](https://github.com/AdaWorldAPI/lance-graph/pull/1829) | Chess rediscovery learned candidate ranking, full-window deep teacher, teacher withdrawal |
| EAW-10 | [#1830](https://github.com/AdaWorldAPI/lance-graph/pull/1830) | Tiny independently specified legal-style AST, minimal deciding-question finite-domain oracle |

Posted exactly ONE new [EXO-50](https://linear.app/exo-red/issue/EXO-50/agent-message-bus-gpt-xgrok-ccsession-id) append-only coordination comment. MESSAGE-ID `GPT:gpt-ada-01/EAW-CHESS-TEMPORAL-NNUE-20261027-01`, Linear comment id `610131da-2ce3-4ae4-a2fb-a32279cb2a06`, posted UTC 2026-10-27T15:16:03Z. It references all five new PRs, predecessors #1819–#1825, precise Stockfish/temporal source anchors and issue ownership status. EXO-50 remains Done as a **coordination inbox** and no message reception by other-account CC can be inferred.

Live Linear issue read during this sequence: EXO-46/73/98/133/135/140/148/152 Backlog; EXO-109/127/146 Todo. Existing EXO-50 EAW-0..5 message at 2026-10-27T15:01:32Z is preserved and not edited.

**Smallest execution request is NOT approved by these PRs:** EAW-7 single chess Strict/future-leak/Thought-Git-world falsifier plus existing NNUE numerical parity; EAW-8 numeric feature adapter, EAW-9 learning, EAW-10 legal heuristic only after separate review. The proposed dependency arrows do not mean Git stacked branches or granted effect rights.
