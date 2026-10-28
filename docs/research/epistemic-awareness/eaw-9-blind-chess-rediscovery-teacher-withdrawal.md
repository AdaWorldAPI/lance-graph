# EAW-9 — Hindsight-blind chess rediscovery: 0+63 Thought-Git × teacher × NNUE/NARS

**2026-10-27. Research plan only, no learning run, implementation, promotion or merge authorization.** Independent main-based branch. Idea provenance [EAW-6 #1826](https://github.com/AdaWorldAPI/lance-graph/pull/1826); read EAW-7 strict world frontier, EAW-8 NNUE feature adapter, EAW-4 [#1824](https://github.com/AdaWorldAPI/lance-graph/pull/1824) and CFA-4 [#1817](https://github.com/AdaWorldAPI/lance-graph/pull/1817). This stage **does not** replace Stockfish-rs D-CHESS-NARS-0.

## Central research hypothesis
Can a chess learner, limited to the exact position and historically available evidence at ply v, use at most 63 active speculative Thought-Git lookahead worlds and sparse NNUE feature deltas to rediscover conditional tactical dependencies or better candidate ordering over time? *Only after deciding* may a deeper independent teacher reveal its evaluation and actual game result. The learner's advantage must survive source-disjoint held-out games and explicit teacher withdrawal. If it fails, do not claim rediscovery or emerging causal understanding.

## Existing shipped / probed spine, avoid duplication
- [D-CHESS-NARS-0](https://github.com/AdaWorldAPI/stockfish-rs/blob/main/.claude/plans/stockfish-harvest-64x64-v1.md): learning flow NNUE teacher → NARS recipe revision → persistent strategy → teacher-withdrawal, with D-CHESS-NARS-0a recipe census; current L6 search shell / policy seam, not complete Stockfish tournament strength.
- [DecisionEpisodeV1](https://github.com/AdaWorldAPI/stockfish-rs/blob/main/docs/DECISION-EPISODE-V1.md): physically separate StrictObservation and RetroLabel, game_id/ply vs exact PositionKey vs approximate recall vs palette; stable deep `TeacherLabel` and rare search events. No Retro data in `CandidatePolicy::score`.
- [expert_iteration_stream.rs](https://github.com/AdaWorldAPI/stockfish-rs/blob/main/examples/expert_iteration_stream.rs): full-window exact per-root candidate teacher, frozen/privileged baselines, deterministic episode encoding, candidate refutation/recovery/PV changes, game-level partition; deep *play-path alpha-beta* values may be bounds, not exact rankings.
- [hindsight_stream.rs](https://github.com/AdaWorldAPI/stockfish-rs/blob/main/examples/hindsight_stream.rs): real Lichess PGN results and version read fence; [rung_hindsight.rs](https://github.com/AdaWorldAPI/stockfish-rs/blob/main/examples/rung_hindsight.rs) was INCONCLUSIVE under final NNUE-only score oracle (mate blindness).
- [temporal_replay.rs](https://github.com/AdaWorldAPI/stockfish-rs/blob/main/examples/temporal_replay.rs) / [planner temporal.rs](https://github.com/AdaWorldAPI/lance-graph/blob/main/crates/lance-graph-planner/src/temporal.rs): independently replayable as-of histories and strictly pinned reading horizon. `TemporalPov::at(v,0)` enforces version range, but full `classify/deinterlace` and data-causal readiness belong to planner.
- [CFA-3 #1816](https://github.com/AdaWorldAPI/lance-graph/pull/1816): ARM co-activation is not causal proof; [CFA-4 #1817](https://github.com/AdaWorldAPI/lance-graph/pull/1817) optional NSL/ONNX ranking, training split and evidence restrictions.
- [EAW-5 #1825](https://github.com/AdaWorldAPI/lance-graph/pull/1825): legal / Odoo to follow *after* chess; no real domain causal claims from synthetic worlds.

## Study design: distinguish THREE time domains
- **Observed past** `Past≤v`: PGN and source records available to the actor at ply v, source witness pinned; Strict reader `QueryReference::at(v,0)`. Historical future positions/results are invisible.
- **Simulated future** `Hypothetical(T_i,depth)`: legal alternative moves generated from current board/learned heuristic within one independent Thought-Git world. Its successor board isn't a source observation and carries no empirical independent evidence root.
- **Hindsight/teacher** `Retro(v)`: full-window independent deeper evaluation, PV/refutations/search labels and actual PGN result. This exists as a separate later evidence lane only. Evaluating a hypothetical continuation for scoring is allowed Strict-time search; reading *actual* continuation/teacher labels is not.
- Multiple candidate rules/heuristics can coexist in one world. A Thought-Git sibling world with a different choice cannot be semantically unioned into one chess position. T0 immutable, T1..T63 live slots with external generation and source seals. All training labels reference correct baseline and source split.

## Stage ladder (future test-only PRs require owner approval)
**R0: Frozen baseline and leak gate.** Pin game-level train/validation/test splits, board position, legal moves, frozen NNUE model, strict observation and policy; exact teacher score bound and source revision. Copy existing Golden episode record, do not invent a new one.
**R1: Bounded lookahead oracle parity.** EAW-7 compare legal successors and protected path-world receipts to independently replayed shakmaty positions; EAW-8 exact NNUE delta vs refresh (king special cases). Count world slots and budget; do not claim parallel execution.
**R2: Simple *interpretable* rediscovery features.** Without teacher knowledge at Strict-time, expose capture, check, threats, material, king safety, mobility and typed relational support as deliberately declared chess features. Lookahead can learn patterns of `requires`, `prevents`, conditional forks, forced mates, transpositions, tactic refutations. Do not label chess heuristic scores as certified causality.
**R3: Train ranking (optional).** Compare seeded-random, frozen MVV-LVA/butterfly ordering, ARM support/association, NNUE-style incremental feature ranking, candidate rules and privileged deep teacher upper bound. Same CPU, nodes, total work, branch/world budget, move legality, teacher depth, position corpus and per-game split. Train on Retro labels after Strict decisions, freeze weights and feature schema for held-out measurements. Teacher labels with unstable depth/margins remain withheld or separately scored; `NarsTruth` synthetic branches aren't independent empirical observations.
**R4: Teacher withdrawal.** Freeze learned ranker, remove future teacher feedback on held-out matches and examine whether improvement persists over frozen/reset baselines. Compare training-only vs inference-with-teacher. Retrospective label stream cannot alter earlier Strict ranking, no leakage from adjacent plies of same test game into training.
**R5: Conditional explanation not just rank.** For a given scenario, ask which guard/fact/move causes the *tested* deeper outcome to differ. Use protected intermediate traces, independent first and joint ablations, preserve derivation roots and multi-feature alternatives. Emit candidate explanation such as `if king exposed ∧ defender pinned, check sequence wins`, with explicit scope and counterexample; never promote to universal theorem without separate independent proof.

## Falsifiers and negative controls (NOT RUN)
| ID | Perturbation | GREEN only if |
|---|---|---|
| BL-01 | Replace all future game plies/Retro labels for fixed Strict observation | Strict feature IDs, branch selection and ranking byte-identical |
| BL-02 | Select candidates with privileged deep teacher inside policy callback | Hard reject; teacher only in Retro/training |
| BL-03 | Final NNUE score sign disagrees with mate-aware game outcome | PGN result/teacher retains authority, NNUE-only proxy invalid |
| BL-04 | Same PositionKey in distinct episodes across train/test | Game_id-disjoint split rejects historical leakage |
| BL-05 | Identical root appears in many Thought-Git branches | No artificial NARS confidence or ARM independence inflation |
| BL-06 | Equal final eval but differing PV refutation, checkmate trace or path | Protected differences remain, no strong equivalence |
| BL-07 | Learned high score selects illegal move | shakmaty rejects; no execution right granted |
| BL-08 | Schoolbook heuristic coactivation says causal influence despite no ablation effect | Labeled association only, not causal law |
| BL-09 | NNUE feature update stale on king move, book model/feature hash changes | Fresh refresh / refusal; exact parity |
| BL-10 | All 63 slots live / ABA stale recycled W | Explicit bounded queue/refusal; no slot-as-evidence-pointer |
| BL-11 | Training uses same game, future plies, or transposed episode across split | Leak detected; held-out invalidated |
| BL-12 | Deep teacher scores are narrow-window bounds but used as total exact ranking | Refuse; full-window teacher or typed bound treatment |
| BL-13 | NNUE learned arm loses to frozen/ARM when full cost charged | PARK optional learning, keep deterministic path |
| BL-14 | Teacher withdrawal destroys measured gain | Report dependence; no autonomous rediscovery claim |
| BL-15 | Learned chess conditional transplanted as legal norm | Reject cross-domain promotion and any source effect |

## Measurements / reproducibility
Repeat across unlike sources: short forced-mate/pin/tactic corpus, broad game-disjoint real PGN, positions with severe class imbalance/quiet positional play and deceptive sacrifices. Seed splits and world choices. Report teacher instability and precision/recall@k, top-1/top-3, node count to tactical discovery, search wall-clock and cold cost, model training/inference, abstentions, book leak checks, exact delta-vs-refresh rate, memory/slot pressure, branch roots, protected trace bytes and outcome variance. Compare new learned heuristics with carefully matched baseline. Do not declare a single test-game success generalization.

## Ownership and explicit red boundaries
Stockfish-rs owns chess search, `shakmaty` legal semantics, licensed NNUE weights/feature index and teacher-policy seam; lance-graph owns temporal contracts, Thought-Git world receipts, DR64³ research and exact Quack/RelAlg. Existing D-CHESS-NARS-0 is the learning arc, do not launch a rival scheduler or learned truth store. EXO-98/148 durable world refs OPEN; EXO-46 physical source binding OPEN; EXO-109 independent oracle, EXO-133 real data, EXO-135 evidence law and EXO-152 numeric owner. No commit from rejected [#1812](https://github.com/AdaWorldAPI/lance-graph/pull/1812), no production APIs or cross-license code transplant. EXO-50 coordination message ≠ delivery/ACK.

## Handoff to EAW-10
Only after objective chess temporal blind gates, challenge the same framework with a *tiny, fully specified rule AST*: alternative support, conjunction, one qualified exception, effective dates and typed routing. Its law source/authority is separate; the chess experience supplies a **testing methodology**, not transferred legal correctness.
