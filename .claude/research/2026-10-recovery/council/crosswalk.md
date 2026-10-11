# CROSSWALK — coresearch "wiring thinking onto CE64 + mask algebra" (2026-10-08)

Questions (re-asked after premise audit): Q0 carriers per state kind (E per-edge
belief, T per-thought scalars, P populations); Q1a candidate identity; Q1b who
owns ranking; Q2 home of per-thought scalars; Q3a closed CE64 op set + which
ops must preserve W 53..58 and bits 59..63; Q3b does any CE64 field encode an
opcode; Q4a contradicts(a,b); Q4b contradiction storage; Q4c population scan;
Q4d demotion; Q5 multi-hop trust (NARS chains vs semirings vs EWA).

Anchors (frozen): CE64 v2 layout; I-VSA-IDENTITIES; masks ≠ magnitude
(mask-risc A2, "no ORDER BY", A1 no selection vectors); EWA ranks WHERE never
WHAT; lane-fold capstone (u16 address, permutation is the only index,
128 KB cap); D-PEARL-IO-0 (only 40..42 selects the Pearl operator; raise-only;
F6/F7 forbid demotion); receipts-over-mask ruling for withdrawal.

Grades: code = VERIFIED-IN-CODE / CLAIMED / ABSENT; outside = READ-IN-FULL /
SECTION-READ / ABSTRACT-ONLY / SECONDHAND. Relation = ALREADY-HAVE / PARTIAL /
NEW / CONFLICTS-ANCHOR.

| # | idea | source (grade) | our surface | relation | Q |
|---|---|---|---|---|---|
| C1 | candidate = ordinal id + separate integer score; sort permutes pairs | Stockfish ExtMove (READ-IN-FULL); AlphaZero 4672 action space (SECTION); DuckDB TopNEntry (SECTION) | stockfish-rs `policy.rs:37-205` MoveCode u16 + frozen_score i32 (VERIFIED); `ThoughtCtx.candidates: Vec<f32>` drops the id (VERIFIED) | PARTIAL (chess side has it; recipe side loses identity) | Q1a, Q0 |
| C2 | selection vector of row indices | DuckDB (SECTION) | mask-risc A1 forbids | CONFLICTS-ANCHOR | Q1a |
| C3 | top-k as MEMBERSHIP mask via bit-sliced magnitude + COUNT (BSI Alg 4.1); ties trimmed arbitrarily | Rinfret/O'Neil 2001 (SECTION, two scouts) | mask-risc integer lanes + popcount; no BSI type (ABSENT) | NEW (needs magnitude slices classed as a magnitude column, not topology masks — A2 fence) | Q1b |
| C4 | masked argmax set = MaskedMaxI32 then Pred::EqI32 then Keep | — | mask-risc `ir.rs:303,137` primitives VERIFIED; composition ABSENT; no position terminal; ndarray trailing_zeros walk exists | PARTIAL | Q1b |
| C5 | ranking as a lazy kernel/cursor outside the membership algebra (staged partial sort; bounded heap + boundary prefilter) | Stockfish movepick (SECTION); DuckDB PhysicalTopN (SECTION) | `HdrCascade::query` RankedHit sorted top_k (VERIFIED); planner `LogicalOp::TopK` (VERIFIED) | ALREADY-HAVE (pattern exists elsewhere in-tree) | Q1b |
| C6 | order carried as a u16 permutation | lane-fold capstone (RULING) | no Permutation type in mask-risc (ABSENT) | PARTIAL (doctrine without type) | Q1b |
| C7 | K-relations: Boolean semiring = set semantics, cannot rank; valued semirings carry magnitude, no witness | Green et al. PODS'07 (SECTION) | blasgraph `HdrSemiring` 7 variants, monoids keep no witness index (VERIFIED) | PARTIAL | Q1b, Q5 |
| C8 | normalization: E keyed by edge, T by thought/cycle, P by (thought, ordinal); ONA keeps truth in concepts, attention in queues | Codd (SECONDHAND); ONA (SECTION, two scouts) | `ThoughtCtx` per-thought vs `MailboxSoA.meta[MetaWord;N]` per-row are already separate carriers (VERIFIED); OGAR D-META64 "CE64 is the edge, not meta" (RULING) | ALREADY-HAVE (carriers); wiring between them ABSENT | Q0, Q2 |
| C9 | per-basin side table beside one CE64 (evidence 4×2 bits, attention, contradictions) | — | staunen probe `Basin` (VERIFIED, example only); D-CE64-CYCLE-0 measured evidence table is load-bearing (0.955→0.000 when forgotten) | PARTIAL (measured, home OPEN) | Q2 |
| C10 | closed op set where each op preserves a reading or visibly strips its validity tag (CHERI monotone derivation, tag-strip on write-back) | CHERI ISA v9 (SECTION) | pack/forward/syllogize silently zero W + 59..63; learn + single-field setters preserve (VERIFIED) | NEW (framing); defect evidence ALREADY-HAVE | Q3a |
| C11 | opcode belongs in the instruction stream, never in the operand | tagged architectures (SECONDHAND) | `forward` dispatches on the WEIGHT edge's mantissa (VERIFIED, edge.rs ~686); D-PEARL-IO-0 forbids mantissa as Pearl opcode (TEST-PINNED); mantissa has 4 readings (D-CE64-LOOP-0) | CONFLICTS-ANCHOR (in-tree code vs the spirit of D-PEARL-IO-0) | Q3b |
| C12 | contradiction absorbed into F/C by revision; no contradiction store | Wang UAI'93 (READ-IN-FULL); ONA (SECTION) | `NarsTables::revise` (VERIFIED) | ALREADY-HAVE | Q4a, Q4d |
| C13 | evidential-base (stamp) disjointness gates revision; overlap → choice | Wang; ONA; OCI covariance twin (SECTION) | `BeliefArena` stamp guard (VERIFIED, belief.rs:180-196); ABSENT in `replay_step`, `forward`, `TruthPropagatingSemiring` | PARTIAL | Q4b, Q5, Q3a |
| C14 | contradiction as a pair/set relation stored apart from beliefs (nogoods) | de Kleer ATMS (SECONDHAND); paraconsistent semi-revision (SECTION) | `CausalWitnessFacet` `Locus::Contradiction` i8 peer offset (VERIFIED); `unresolved_tension` set (VERIFIED); receipts-over-mask RULING | ALREADY-HAVE (two carriers, unjoined) | Q4b |
| C15 | glut = AND of a "supports" plane and a "rejects" plane; gap = NOR | Belnap twist structures 2608.20228 (SECTION); FCA (SECONDHAND) | mask-risc And/Not over integer lanes (VERIFIED); no supports/rejects planes exist | NEW | Q4c, Q4a |
| C16 | per-topic min/max of F via GroupReduce; max−min = spread, not pair identity | — | mask-risc `Terminal::GroupReduce` (VERIFIED) | PARTIAL | Q4c |
| C17 | demotion needs an explicit entrenchment order; irrevocable = top of order | AGM (SECTION); Coniglio et al. 2412.06117 (SECTION) | `pearl::revise` raise-only, F6/F7 forbid demotion (TEST-PINNED) | CONFLICTS-ANCHOR for any demotion; NEW as formalization of raise-only | Q4d |
| C18 | multi-hop trust as a bounded monotone discount chain (uncertainty grows along the path); path problem on ordered semiring | Jøsang/PaTAS 2511.20586 (SECTION); Theodorakopoulos & Baras (ABSTRACT) | `chain_replay::replay_step` REVISES each hop so confidence RISES (VERIFIED); `NarsTables::deduce` stores c_out = f_out, ignores c1,c2 (VERIFIED); `TruthPropagatingSemiring` non-associative multiply, raw-c add, no prod caller (VERIFIED) | NEW + defect evidence | Q5 |
| C19 | trust → additive noise Q(t) (Kalman predict) instead of M=√t·I | Kalman (SECONDHAND) | plan F-MEP-0b already names the inversion + bounded inflate (PROPOSED) | ALREADY-HAVE (named, unwired) | Q5 |
| C20 | covariance intersection when fusing correlated trust paths | Julier-Uhlmann / OCI 2603.16768 (SECTION) | no EWA fusion exists | NEW | Q5 |
| C21 | learned ordering stored in a table keyed by situation features, bounded saturating update | Stockfish history gravity (SECTION) | D-MOORE-NARS-0 situation→recipe learner (MEASURED) | PARTIAL | Q2, Q1b |

Cross-cutting (scouts 4 and 5 independently): every outside system keeps the
score beside an identity, never inside it; ranking is always a separate
operator; only BSI stays inside bitmap+popcount, and it yields a set, not an
order.
