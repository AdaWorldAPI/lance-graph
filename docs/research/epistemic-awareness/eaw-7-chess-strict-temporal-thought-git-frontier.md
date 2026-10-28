# EAW-7 — Chess Thought-Git T0+63 lookahead: temporally strict worlds and independent oracles

**Status 2026-10-27:** docs-only independent main-based research gate. No test executed, no new executor, source mutation, abi/runtime/merge permission. Companion idea ledger [EAW-6 #1826](https://github.com/AdaWorldAPI/lance-graph/pull/1826); existing owner [EAW-4 #1824](https://github.com/AdaWorldAPI/lance-graph/pull/1824). Do not duplicate Thought-Git, temporal.rs or Stockfish search.

## Question
Can a borrowed, as-of chess position spawn at most 63 resident legal *hypothetical futures* through existing Thought-Git while all real historical future observations and teacher labels stay unreadable to the Strict selector? Can independent oracles establish exact world contents and distinguish in-world move composition from incompatible sibling landing?

## Exact owner interfaces audited
- `lance-graph-planner/src/temporal.rs`: canonical temporal epistemology, `QueryReference::at(v,0)`, `EpistemicMode::{Strict,Aware,Retro}`, `classify`/`deinterlace`. Default QueryReference uses unbounded head, NOT safe for hindsight-blind evaluation. Temporal classification is not chess legality or quality.
- `lance-graph-contract/src/temporal_pov.rs`: independent range-only filter `TemporalPov::at(v,0).admits(u)`; not a replacement for planner epistemic source/data readiness.
- `stockfish-rs/examples/temporal_replay.rs`: existing NNUE HalfKA per-ply version stream with historically documented 34/34 baseline projections, 11/11 out-of-order accumulator replays; model-gated, not Thought-Git worlds.
- `stockfish-rs/examples/hindsight_stream.rs`: current-ply read fence and independent PGN result; `rung_hindsight.rs` previous final-NNUE-sign oracle INCONCLUSIVE for sacrificial mates/unsettled random games.
- `stockfish-rs/docs/DECISION-EPISODE-V1.md`: distinct PositionKey, GameEpisodeKey, evidence address, StrictObservation, RetroLabel; exact per-candidate teacher bounds only from `trace::teacher_search` full-window evaluator.
- `stockfish-rs/.claude/plans/stockfish-harvest-64x64-v1.md`: D-CHESS-NARS-0 actual NARS learning / teacher-withdrawal owner, no second chess search or policy.
- `crates/thought-git-probe/src/oracle.rs`, CFA-2 [#1815](https://github.com/AdaWorldAPI/lance-graph/pull/1815), EAW-4 [#1824](https://github.com/AdaWorldAPI/lance-graph/pull/1824): T0 sealed, T1..T63 independent speculative slots; `next_version` sibling union is a comparison instrument, not one world. CE64 W 53..58 is *live* slot ordinal, generation and durable replay witness external (EXO-98/148).
- `shakmaty`: chess legal movegen, FEN, apply/unapply and terminal condition oracle. Never let the candidate model hallucinate legal moves. Real Stockfish C++ deep search optional independent privileged teacher; pure position semantics remain shakmaty.

## Semantic experiment grammar (proposal only)
T0 = { game_id, ply=v, exact board/FEN + side/castling/ep, QueryReference::at(v,0), source/game/engine/net/feature-codebook versions, completeness and permissions }. T0 is immutable.

Each active T_i, 1≤i≤63 = {slot, generation, baseline_ref, parent-world/path, selected legal UCI, local simulated position, NNUE delta receipt, consumed-node budget, protected intermediate move/guard/effect trace, world identity}. Simulated descendants may be at hypothetical ply v+k, but `historical_read_allowed(u)` stays bounded by `u≤v`, including PGN and teacher labels. Logical search depth is NOT 63; recycle slots with generation checks between frontiers. Alternative siblings never become a physically committed combined board; multiple legal move steps within one single path compose normally.

Represent chess legal constraints with source evidence and simulated legal successor transitions. Do not treat speculative moves as real future events in `TemporalPov` or as empirical evidence roots. PositionKey can be shared for exact transpositions but distinct episode/path provenance must be retained; hashing collisions and equivalence conditions need independent checks.

## Smallest proposed test-only probe (separate owner authorization)
Fixture A: one deterministic midgame FEN, 3–8 legal candidates including a tactical trap and two distinct routes to a transposition; fixture B: a capture, castle/king-bucket refresh, promotion or en-passant special move, and a terminal mate. Pin two independent game histories with same exact board but separate EpisodeKeys.
1. Seal T0 and enumerated legal move set from shakmaty, independent of Thought-Git.
2. Branch legal selected candidate moves T1..Tmin(63,budget); capture generation-aware IDs, paths and source versions.
3. Compute board and legal transitions for each branch using shakmaty; compare every occupied world to an independent from-start replay and hash/full board equality, not score only.
4. Enforce Strict present read only on real history through explicit `QueryReference::at(v,0)` / `TemporalPov::at(v,0)`; the simulated successor is separately typed `Hypothetical`, not a spoiler read.
5. Run the same candidate selection after replacing all future PGN plies, result and deep teacher labels; identical Strict result must follow. Deeper teacher revealed only after.
6. Compare bounded breadth/reuse under T1..T63 occupancy pressure, ABA generation reuse and exact explicit independent world records. Cold budget includes source opening, masks, fork, movegen, replay and numerical refresh.
7. Test sibling union by passing incompatible world results to general `next_version`: the union may be a valid comparison receipt but MUST NOT be exposed as a legal chess state or new historical episode.

## Pre-registered red/green falsifiers (NOT RUN)
| ID | Falsifier | Correct behavior |
|---|---|---|
| CHT-01 | Future historical ply v+1 or result injected into Strict input | Reject; same selected candidates after every Retro mutation |
| CHT-02 | Use default unbounded QueryReference instead of explicit as-of | Gate detects historical future admission |
| CHT-03 | Hypothetical T1 move becomes a real source observation | Reject cross-namespace contamination |
| CHT-04 | Two sibling moves both committed into one board | Reject world interpretation; only a comparison receipt allowed |
| CHT-05 | Recycle W=slot without external generation seal | Stale token refuse, T0 unchanged |
| CHT-06 | Two different game episodes share exact transposition board | Share position representation only, retain distinct historical/derivation roots |
| CHT-07 | Pawn promotion, en passant, castle rights or 50-move clock lost by projection | Independent game-state oracle mismatch |
| CHT-08 | Mate tactic labelled by final NNUE sign | Reject oracle; use mate-aware teacher / actual PGN result later |
| CHT-09 | No slots remain or path horizon >63 | Capacity management explicit; never overclaim 63 parallel ply |
| CHT-10 | Teacher/real future alters STRICT trace, scores or candidate ranks | Zero tolerated contamination |
| CHT-11 | Feature/evidence values from different source revisions share local ordinal | Rebind or refuse under DT64 Q0 |
| CHT-12 | Different branch enumeration order changes any deterministic complete-world outcome | Must remain equivalent after normalizing candidate IDs, except declared budget-limited search-policy order effect |

## Benchmarks / exit decisions
Compare (A) vanilla legal shallow search / frozen ordering, (B) identical search and borrowed Thought-Git receipt overlay, (C) eventual NNUE delta heuristic from EAW-8; no neural claim in this stage. Report slots resident, batches, candidate/legal count, exact replay matches, world contamination count, future admissions (must be zero), cold/warm setup, heap, branch and generation overhead. Prefer a tiny falsifiable proof to speculative throughput.

Exit: REVIEW for a separately approved test-only PR only when owner confirms world/temporal membrane. Otherwise DEFER. No production DR³ third axis (NFR-OP-2), escaped Keep (NFR-OP-1), new time store, new engine, teacher selection, source action, or law admission. [#1812](https://github.com/AdaWorldAPI/lance-graph/pull/1812) excluded epistemically. EXO-46/98/109/133/148/50 relevant; EXO-50 messages are not recipient ACKs.
