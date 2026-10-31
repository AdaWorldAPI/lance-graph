# D-STAUNEN-3 | Symbolic structure and neural learning | draft
**Docs only; no second Datalog, provenance store, optimizer, palette registry.**
## First pass
- Reuse #1884 `xlog-relcomp-probe` and #1869 DT bind catalogue; keep XLOG RIR/PIR/EIR/NN behavior distinct. Join support/multiplicity is not confidence or causality; Epi5 `IndirectKnown×Open` stays open.
- Reuse #1846 `lance-graph-nsl` frozen Google NSL packing/loss oracles and adjacency. Stage signed/order-dependent graph losses. Do not substitute HDC Hamming distance for NSL L2/KL/JS gradients.
- U9 policy: DR512 signed epistemic code + never-touched != TurboVec9 512 Lloyd-Max centroids; #1839 class codec contract; #1847 residuals. Quantized neighbors for candidate discovery, exact/declared tolerance for ranking, loss, gradients.
- Structural reinforcement hydration: incoming neural traces + HDC novelty -> evidence-source guarded candidate relations in THOUGHT, not WORLD; XLOG generates/verifies bounded inferred candidates; NSL computes local graph regularization; independent observation permits promotion.
- Preserve bags and overflow refusal on counts; K3 unknown, signed zero, NaN, evidence duplicates, unique roots, slot-version.

## Research gates
X0 owner/capability inventory; X1 native signature and source plan binding; X2 bag/overflow golden; X3 NSL neighborhood oracle; X4 loss/gradient parity; X5 9-bit/residual ablations; X6 ontology/hydration provenance; X7 no self-certifying joins; X8 benchmark with no learning/model comparison.
**Prereq:** #1846, #1847, #1884; cross-reference D-STAUNEN-1.

## Round 2 | CC probe matrix, typed adaptation, reference constraints

### Do not collapse distinct logic and learning paths
XLOG upstream's typed deterministic **RIR**, probabilistic **PIR**, epistemic **EIR**, and neural-symbolic training surfaces have different query meanings; there is no universal RIR lowering for every program. Our current XLOG lab #1884 only admits the bounded compiled count rule and masked relational composition on `ndarray::simd`. It is **not** admission for XLOG recursion, probabilistic exact inference, differentiable ILP, general aggregates or whole-language portability.
- XLOG https://github.com/BrainyBlaze/xlog
- Native probe https://github.com/AdaWorldAPI/lance-graph/pull/1884
- DT64 bind catalogue https://github.com/AdaWorldAPI/lance-graph/pull/1869
- NSL parity and exact neighbor semantics https://github.com/AdaWorldAPI/lance-graph/pull/1846
- Codec and cross-thought candidate research https://github.com/AdaWorldAPI/lance-graph/pull/1847

### Hydration action, as existing algebra (not another store)
```text
observed_neural_event + HDC novelty + sealed source identity
  -> DR bound local focus
  -> DT64 admitted method (source lens, K3 guards, declared effects)
  -> XLOG compiled rule body / NSL graph loss under exact owned source
  -> candidate neural correlation / relational inference in THOUGHT
  -> independent evidence + causal ablation -> promotion candidate
```
- `XLOG->CE64` projection must preserve `Direct×Open` base facts, `IndirectKnown×Open` composed derivations; neither is `Causes`, and no `CausalCandidate` without its certifying test.
- Bag multiplication `count_R(y)*count_S(y)` uses checked i64. Negative, zero, NaN and `-0.0` semantics must remain source-faithful; don't assign counts to F/C without calibrated legal conversion.
- Use immutable borrowed source + versioned `FocusLease`/frame. Duplicate relation keys preserve NSL's documented stable tie, first equal-weight bits and f64 ordering at freeze; signature/state lives in DT64; no duplicate adjacency tables.
- NSL loss is defined on neighbor **logits** / features and weights; contrastive local HDC Hamming distance is *not* a drop-in for the NSL TensorFlow graph loss.
- An HDC/VSA candidate can rank what evidence to ask for, but its similarity is not proof, causality, calibrated Bayesian uncertainty or neural physiology.

### Codec/admissibility matrix
| Task | Source/format | Gate |
|---|---|---|
| Node identity | exact rank/codebook once | #1846 label oracle, reused |
| Sorted selected neighbors | f64 order in compat path | no tied-quantization top-k reversal |
| Exploratory edge weight encoding | Fisher-z i8 / z-linear U9 | not production without fixed tolerance and real-data gate |
| Graph-loss numerics | NSL f32, L2/L1/cosine/KL/JS declared | TensorFlow frozen oracle and gradient checks |
| Very close neighbor features | source-code + scale-separated residual | compare with non-quantized convergence; no false zero |
| Epistemic U9 `EpiLane9` | 511 numeric + never touched | class declaration and signed-zero / absent rules |
| TurboVec9 | 512 numeric centroids | dedicated retrieval model, not epistemic U9 |
| Pair LUT | prepared 256² or candidate 512² | check numerical cost, aliasing and native parity |

### Execute or refuse gates
X0 independent source inventory: current `main` #1884 tests, XLOG source version, NSL TensorFlow oracle, U9 codec and exact semantics.
X1 same typed compiled RIR operation -> same `ndarray::simd` entry for ONNX/XLOG when signature matches; wrong signatures stop.
X2 join/aggregate has correct multiplicity, overflow refusal and no materialized owned result rows; null/unknown guard refuses.
X3 use NSL G01–G12 plus supplements and source-sealed graph neighbors; disable stable tie or source seal -> red.
X4 graph-loss vs TensorFlow and finite differences or analytical gradient reference; include cancellation/softmax corners. Tolerance source-specific, never assert exact gradients from an exact residual address.
X5 compare exact f64 top-k, Fisher-z i8, z-linear U9, TurboVec9 and residual-coded arms on held-out graph sets; report precision, memory and mistakes *before* speed.
X6 source-rooted inference: every candidate carries original observed event root, XLOG program hash and CE64 rail. If two thoughts read one root, evidence count remains one.
X7 source-off/novelty-shuffle/quantized-nearest-collision negatives. Require generalization to unseen structure.
X8 resource gate: native resident fold ns/batch µs per `CLAUDE.md`. Do not conflate cold parse/plan with hot execution.

### Scientific context
- Bui et al. Neural Graph Machines, 2017: https://arxiv.org/abs/1703.04818
- SpikeHD SNN/HDC, 2022: https://doi.org/10.1038/s41598-022-11073-3
- SynapseHD joint SNN/HDC, 2026: https://doi.org/10.1016/j.neucom.2025.131757
- Kinavuidi et al. continual spike-time HDC decoding, 2026: https://doi.org/10.1088/2634-4386/ae4f1e
- Denève, Bayesian Spiking Neurons I/II, 2008: https://doi.org/10.1162/neco.2008.20.1.91 and https://doi.org/10.1162/neco.2008.20.1.118

**CC handoff:** initial implementation belongs in existing `xlog-relcomp-probe` or `lance-graph-nsl` after owners sync; no new `lance-graph-xlog` crate and no new renderer, classifier or external graph engine until a measured source gap exists.
