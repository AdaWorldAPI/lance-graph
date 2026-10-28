# EAW-8 — NNUE-style incremental evaluation attached to DR64³ epistemic projections

**2026-10-27. DOCS-ONLY, UNRUN, NOT AN IMPLEMENTATION/ABI AUTHORIZATION.** Independent main-based plan. Research chain [EAW-6 #1826](https://github.com/AdaWorldAPI/lance-graph/pull/1826) and temporal probe EAW-7; upstream exact evidence [#1821](https://github.com/AdaWorldAPI/lance-graph/pull/1821), CE64 projection [#1822](https://github.com/AdaWorldAPI/lance-graph/pull/1822), native fold [#1823](https://github.com/AdaWorldAPI/lance-graph/pull/1823), optional learned ranker [CFA-4 #1817](https://github.com/AdaWorldAPI/lance-graph/pull/1817).

## Problem (precisely)
Can **borrowed, source-qualified** DR64³ relation/epistemic deltas select a small changed set of *model features*, permitting an NNUE-style incremental numeric accumulator to predict valuable Thought-Git lookahead branches, with exact accumulator-refresh parity and zero influence on factual presence, K3, Epi5 certification, authorization or legal move validity? NNUE ranks *questions* and search candidates, it does not create facts.

## Repository evidence already exists; reuse and do not rebuild
- [stockfish-rs/src/eval/incremental.rs](https://github.com/AdaWorldAPI/stockfish-rs/blob/main/src/eval/incremental.rs) `HalfKaAccumulator`: `refresh(before)-removed+added` under **wrapping i16** and PSQT semantics; refresh king-relative invalidations. `FullThreats` delta is not proven fully incremental.
- [stockfish-rs/src/feature/half_ka.rs](https://github.com/AdaWorldAPI/stockfish-rs/blob/main/src/feature/half_ka.rs) type-dependent sparse feature `make_index`; NNUE numerical semantics are not the chess QR64 Boolean move matrix.
- [stockfish-rs/src/eval/network_eval.rs](https://github.com/AdaWorldAPI/stockfish-rs/blob/main/src/eval/network_eval.rs) quantized network through existing `ndarray::simd::matmul_i8_to_i32`, oracle-pinned quantization/shifts and layer stack. Reuse its independent value/structure tests, no separate new GEMM/TERNLOG ISA.
- [stockfish-rs D-QR64 plan](https://github.com/AdaWorldAPI/stockfish-rs/blob/main/docs/plans/d-qr64-stockfish-nnue-feature-mask-v1.md) proposes boolean from×to 64² 512B selector and separate 2×4096 i16 ButterflyHistory, never casting i16 numeric history to Boolean.
- [lance-graph NNUE perturbation cascade](https://github.com/AdaWorldAPI/lance-graph/blob/main/.claude/knowledge/stockfish-nnue-as-perturbation-cascade.md): conjectural domain-independent perturbation accelerator with one independently tested NNUE anchor. Do not promote analogies (Morton, phase, HHTL) to code contracts without independent tests.
- `lance_graph_contract::epistemic_state5`: 32-code raw domain, 24 meaningful codes, derived u32 predicate selector distinct from DR64 row mask. CausalEdge64 46..49 and 50..52 overlap and require correct declared class/rail/generation; W 53..58 is ephemeral live thought index. [#1822](https://github.com/AdaWorldAPI/lance-graph/pull/1822) owns projection semantics.
- `lance-graph-mask-risc::fuse/ternlog_dispatch` and `ndarray::simd::mask_ternlog` already own Boolean three-input masks; NNUE is **numeric**, downstream of witnessed active-feature bitmask. No new hardware feature assumption.

## Typed semantic pipeline (research-only; no new concrete Rust ABI)
1. DT64 source pin and actual snapshot/borrow witness {source object, schema, codebook generation, class/rail, scope/ACL/completeness, focus, recipe, read epoch}. NO escaped invalid Keep.
2. Quack/RelAlg aligns relationship/guard/epistemic populations over the same complete coordinate domain; EAW-3 can AND3 presence, applicable TRUE, epi-eligible only as a *terminal selector*. K3 knownness and support roots stay separately addressable.
3. Derive model-specific **sparse feature IDs** from the typed source coordinate plus declared categorical observation (present / covered-absent / UNKNOWN, K3 guard verdict, Epi5 topology/certification, declared activation bin, source-witnessed support class, world-relative intervention). Not raw CE64 bits; alias risk assessed; source evidence roots never flattened to one numerical confidence.
4. Compute a changed-feature multiset `adds/removes/requalifications` between two independently witnessed states (within one world or comparison only). Count multiplicities if the feature mapping is not set-valued. A qualification change is remove old feature and add new feature, even when the final Boolean Eligible bit does not change.
5. Native numerical adapter: `h(S)=bias+Σ_weight_column(feature_i)`; `h(S')=h(S)-Σ_removed+Σ_added` under declared accumulator arithmetic. For chess, exact wrapping i16 parity with Stockfish-rs full refresh on every checked state. For an epistemic *new* model, pick numeric semantics explicitly, prove parity against independent full-reference, do NOT simply assume Stockfish overflow/bucket layout transfers.
6. Model output is inert score for which bounded Thought-Git hypothesis/branch to investigate. Scores/calibrations/model weights/version are not witnesses, causal certificates, permissions, physical DR64³ flags, legal norms or source writes. Keep existing CFA-4 optional NSL/ONNX graph training as independent method.

## Three distinct axes that must not collapse
- Feature-index geometry `coord × categorical state × model feature namespace` != DR64³ logical three axes.
- A boolean bitmap of active features != NNUE integer accumulator and weight layout. NNUE weight table may be far larger than a 32 KiB dense Boolean DR64³ cube.
- A model score `predicted usefulness` != `EpistemicState5::Causes` and != law admission.
- Model weight updates during **training** vs accumulator updates during **inference** are separate processes; source schema changes rebind/rebuild features regardless of learned score.

## Exact model and provenance witnesses
Pin before any comparable result: model SHA, network config, quantization/overflow order, feature schema hash, codebook version, source snapshot id, operator-declared class/rail/activation reading, authorization/completeness, borrowed membrane, thought slot and generation, training corpus split, game_id/episode_key and actual legal state. A model trained with future data can legitimately score *a separate evaluation game* only under an explicitly disclosed training cutoff; it must not receive future *features/teacher labels of that same decision* at Strict-time.

License boundary: Stockfish-rs plan explicitly GPL-3.0; lance-graph Apache-2.0. **Do not copy NNUE code** into the Apache repository unreviewed. Cross-repo test-only interface or independently written numeric adapter after license review.

## Single minimal two-part gate (all TEST-ONLY after owner permission)
**Part A exact feature/accumulator semantics**: Pin two or more chess positions, including capture, king move, promotion or en passant and two hypothetical alternate continuations from same T0; extract changed HalfKA features with existing Stockfish-rs, compare full refresh with delta update byte-for-byte. Mutate model hash, feature generation, Epi5 categorical reading and source focus; invalidate stale accumulator and rebind. A second synthetic DR64³ 2×2×2 even-parity fixture verifies no 3D triple hallucination and features change on requalification without changing presence.
**Part B usefulness**: With equal node/branch/thought budgets, compare frozen handcrafted ranking, seeded random, ARM co-activation and minimal fixed-weight (later trained) NNUE-style score on hidden tactical positions; measure recall@k of full-window teacher's top candidates, how quickly actual misses are discovered, cold and warm overhead. No training requirement in Part A.

## Mandatory negative controls
| ID | Mutation | Must detect |
|---|---|---|
| NND-01 | Drop one removed feature or duplicate one add | Incremental accumulator != full refresh; RED |
| NND-02 | Wrong i16 overflow or layer quantization order | Byte-exact numeric disagreement; RED |
| NND-03 | King-relative index/bucket changes with stale cache | Forced refresh rather than stale score |
| NND-04 | Presence same, Epi5 certification changed | Different typed feature or clear abstention; cannot fake causal promotion |
| NND-05 | UNKNOWN and known-absent share feature ID | Refuse lossy feature map unless explicitly typed as lossy ranker, never source identity |
| NND-06 | Activation aliases opcode under undeclared 46..49 reading | Refuse feature read (EXO-146) |
| NND-07 | Sealed W=6-bit thought id used as provenance | Reject; external generation/world witness required |
| NND-08 | T1's accumulator overwrites T0 or T2 cache | World isolation / alias mismatch |
| NND-09 | Different source/schema/codebook same local index | Stale binding/feature fingerprint refused |
| NND-10 | Duplicate evidence across 63 hypothetical worlds | No confidence inflation or fake independent support |
| NND-11 | Learned high-confidence score recommends impossible chess move / effect | Legal movegen/authority rejects, ranker never dispatches action |
| NND-12 | Learned ranker loses to frozen / ARM after cold+hot accounting | PARK neural component, keep deterministic work |

## Admission
One independently authored scalar numeric oracle, one existing stockfish-rs refresh oracle and disjoint chess games; report matched full traces, cost of source pinning/rebinding and feature work, CPU feature sets including AVX2 fallback. EAW-8 cannot authorize an NNUE hot kernel **inside** DR64³, new physical third axis NFR-OP-2, new global learned-feature catalogue, D-NAM/EXO-152 changes or direct license-crossing code copies. Proceed only with a scoped read-only adapter after proof. #1812 quarantined. EXO-46/98/109/132/133/135/146/148/152/50 remain owner references.
