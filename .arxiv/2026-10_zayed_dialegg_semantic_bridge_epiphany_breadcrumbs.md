# DialEgg: dialect-agnostic equality saturation as a bridge over the existing substrate

**Research breadcrumb, 2026-10-25.** **Status:** PRIMARY-SOURCE READING + REPOSITORY-CONTEXT HYPOTHESES. This note does **not** claim that ONNX/TFLite graph import, a shared relational ISA, DialEgg, or an egglog optimizer is wired end-to-end in lance-graph. No new kernel, compiler or runtime is authorized by this note.

**Primary paper:** Abd-El-Aziz Zayed and Christophe Dubach, *DialEgg: Dialect-Agnostic MLIR Optimizer using Equality Saturation with Egglog*, CGO 2025, pp. 271–283, DOI [10.1145/3696443.3708957](https://doi.org/10.1145/3696443.3708957). [Author PDF (13 pp.)](https://www.azizzayed.com/publications/dialegg/zayedcgo24.pdf) · [CGO programme](https://2025.cgo.org/details/cgo-2025-papers/44/DialEgg-Dialect-Agnostic-MLIR-Optimizer-using-Equality-Saturation-with-Egglog) · [reference code](https://github.com/AzizZayed/DialEgg) · [paper index](https://pldb.kirancodes.me/p/conf/cgo/ZayedD25).

**Shelf warning:** `.arxiv/` is this repository's research-note directory. DialEgg is a **CGO/ACM paper**, not hereby assigned an arXiv identifier. Source claims, merged code, and architectural aspirations are distinguished below.

## The one-sentence epiphany

**Foreign tensor/query/compiler vocabularies can be mapped to a typed graph of mathematical and relational operations, explored using guarded equality saturation, and extracted into existing execution primitives; the optimizer need not become a second tensor engine or the owner of storage, proof, or effects.**

More precisely: this is a **proposed bridge**. The repository currently contains several distinct IR and execution paths; one shared production IR cannot be assumed simply because the underlying arithmetic, masking and addressing ideas resonate.

## Breadcrumb trail: how the insights connect

| ID | Spark / previous question | What clicked | Boundary / next evidence |
|---|---|---|---|
| B0 | ONNX ModelProto and TFLite FlatBuffer seemed to demand new tensor engines | Treat external representations first as **graph and metadata DTOs**: node identity, dependencies, axes/shapes, operators, constants, quantization, control flow. | No claim that a complete importer exists. Validate each schema/opset/version and unsupported-op refusal. |
| B1 | "The tensor should disappear at the adapter boundary" | The internal candidate is **typed addressing + SoA value planes + DR relationship/mask tiles + relational/arithmetical ops**, not a clone of ONNX's runtime tensor object. | This is not an assertion that every dense/ragged/dynamic tensor is representable by DR64/DR256. Preserve shape, strides, broadcasting, aliasing, dtype and layout semantics; prove any tile decomposition. |
| B2 | Cartesian coordinates, ClassId/ClassView, HHTL, DR64/DR256 and SoA already describe much of our substrate | Separate **logical axes/coordinates** from **physical storage and execution**. Source/target/feature axes, local palette indices and generation witnesses must remain typed. | Identity, coordinate provenance, focus and source generations cannot be inferred from coincidental ordinal equality. Historical notes say QR; later discussions use DR terminology. |
| B3 | ONNX names mapped to native functions | A name adapter is valuable without a second interpreter. [#1754](https://github.com/AdaWorldAPI/lance-graph/pull/1754) provides selected native ONNX names; [#1758](https://github.com/AdaWorldAPI/lance-graph/pull/1758) keeps bounded tanh/exp in an explicit module. | Those merged PRs are **operator API work**, not model-graph ingestion, generic RelAlg lowering, or proof of arbitrary ONNX parity. Exact and ULP-bounded families must remain distinct. |
| B4 | Egglog/Datalog asks how to rewrite across different vocabularies | DialEgg shows that **operations, typed values, attributes, blocks and regions** can be translated to Egglog, optimized, then translated back (paper ``3–5, Fig. 2). | Copy the *principle*, not automatically its MLIR dependency or implementation. Current LG paths are separate; a canonical normalized graph is work to design and test. |
| B5 | What if the optimizer doesn't know a new operation? | DialEgg preserves unknown constructs as **opaque** so optimization can happen around them (paper ``3–4). This suggests staged operator coverage instead of all-or-nothing ingestion. | Opaque must mean preserve identity, type, dependencies, results and effects; if those contracts cannot be preserved, **refuse**. "Unknown but pure" is never the default. |
| B6 | Why Egglog/Ruler/Enumo rather than more local rewrites? | Equality saturation explores equivalent **candidate programs** together, avoiding a single greedy phase-order decision; guarded rule inference could populate a versioned catalogue (paper `7). | Candidate discovery is **not proof of rule soundness**. Domain-bound oracles and independently verified laws precede admission. |
| B7 | Why would a shorter program be slower? | Costed extraction can depend on input dimensions and operations (DialEgg `6, `8). Our cost should include **source reads, binding, focus preparation, DR density, masks, SIMD path, cache churn, materialization and validation**, not only AST node count. | [#1763](https://github.com/AdaWorldAPI/lance-graph/pull/1763) measures focus/direct break-even; [#1772](https://github.com/AdaWorldAPI/lance-graph/pull/1772) shows MRU vs LRU changes winner with workload. No universal winning policy. |
| B8 | "Mathe-Logik-Tracert" and Thought-Git replay | Distinguish an **equality witness**, an **extracted plan**, an **independent execution oracle**, and an **authorized durable mutation**. Proof is not an effect acknowledgement. | Merged lab [#1693](https://github.com/AdaWorldAPI/lance-graph/pull/1693) has a cold E1 proof/replay capsule; [#1694](https://github.com/AdaWorldAPI/lance-graph/pull/1694) adds a guarded in-memory catalogue/fallback. Neither is a general formal verifier or production compiler. |
| B9 | "Codegraph → MIR/MLIR → Egglog → codegen?" | A **codegraph** could make call/data dependencies and legal transformations explicit, while MLIR/Egglog are *optional interchange / exploration tools*. Extract back toward our typed procedures rather than automatically replacing native Rust execution. | Need a separately specified front/back translator and a sound mapping of SSA, regions, control flow and side effects. Existing MIR/MLIR codegen is not established by this analogy. |
| B10 | Neural Structured Learning: structure plus differentiable learning | Structural relations may provide adjacency, examples and constraints to a model; learnable scores can propose operator orderings or candidate rules. | **A learned score is not an equality proof**, and neural/continuous approximate contracts cannot silently rewrite exact Boolean, SQL K3, NARS evidence, or CE64 state. |
| B11 | DR64³/DR256³, a 1/16 focus window and tiled "holographic" addressing | Think of *compilation and focus* as selecting a resident view of a larger typed relation, not a license to materialize every Cartesian cube. | Explicitly **speculative** here: validate dimensional capacity, address mapping, storage and semantics before claiming 3D support or performance. |

The surprising convergence is not "everything becomes MLIR" or "everything is a tensor." It is that **multiple source languages could share verified laws and costed executable procedures after their separate semantics are preserved**.

## What DialEgg actually establishes (paper vs transfer)

1. **Representational bridge (paper ``3–5):** predefined/user-defined Egglog forms capture MLIR types, attributes, operations, results and region/block structure. Unrecognized operations can be round-tripped without optimizing their internals. **Transfer:** build a small, typed mathematical/relational normalization seam, if needed, between frontends and existing lowering paths.
2. **Cost-aware extraction (paper `6):** cost functions can exploit type and tensor dimensions; shape-driven matrix association changes operation counts. **Transfer:** an LG cost model must measure addressing/preparation and executor-specific costs, never assume equal cost per algebraic node.
3. **Nonlocal wins (paper ``7–8):** the 3MM result is driven by changing matrix multiplication association, i.e. algorithmic work, not magic SIMD acceleration. Figure 3 reports up to **12.88×** for the studied 3MM setting; polynomial + canonicalization obtains **1.12×** via Horner rewriting. These are **DialEgg benchmark results**, not LG predictions.
4. **Cold-stage expense (paper Table 2, p. 281):** on its 80MM case, Egglog takes **4939.3 ms**, including **3732 ms** saturation; this is not suitable as an unbudgeted nanosecond/microsecond hot-path dependency. The paper uses limited workloads and the authors note that rewrite correctness is the user's responsibility (`9).
5. **Rule unsoundness is real (paper `9 and examples):** illustrative `x/x → 1` fails for zero; signed division by `2^k` does not always equal arithmetic shift for negative dividends; unrestricted FP reassociation can alter rounding/NaNs/signed zero. A performance result is not a semantic proof.

## Current boundary: what is shipped versus what remains hypothetical

The repository [README](../README.md) explicitly documents **distinct paths**:

    Cypher AST -> logical plan -> DataFusion -> batches
    SQL -> DataFusion SQL/plan -> batches
    Rust Quack builders -> Quack query -> mask-risc Program -> execute_into
    ReportPlan -> Quack -> mask-risc

It says no production text frontend reaches Quack and no shared IR currently joins these paths. **Do not erase that architectural fact in diagrams.**

Current pieces relevant to the idea:
- [lance-graph-onnx #1754](https://github.com/AdaWorldAPI/lance-graph/pull/1754) and [#1758](https://github.com/AdaWorldAPI/lance-graph/pull/1758): selected native operator-name mappings and explicit numeric-contract split.
- [E1 proof/replay #1693](https://github.com/AdaWorldAPI/lance-graph/pull/1693) and [catalogue/fallback #1694](https://github.com/AdaWorldAPI/lance-graph/pull/1694): narrow laboratory proof/reuse mechanisms.
- [DR focus/direct break-even #1763](https://github.com/AdaWorldAPI/lance-graph/pull/1763) and [focus churn #1772](https://github.com/AdaWorldAPI/lance-graph/pull/1772): evidence that locality, setup and workload affect extracted-plan economics.
- [RelAlg unified plan](../.claude/plans/d-relalg-unified-v1.md) and [ONNX native plan](../.claude/plans/d-onnx-native-v1.md): design context, **not** evidence that every target path is shipped.

**Potential architecture, not implemented pipeline:**

    ONNX ModelProto | TFLite FlatBuffer | SQL | Datalog | codegraph
           -> per-source parser + versioned semantic DTO
           -> typed operator/dependency DAG + shapes + effects + lineage
           -> ADMIT or REFUSE each operator / law domain
           -> [bounded, cold equality exploration + costed extraction]
           -> [versioned, checkable proof / execution-plan capsule]
           -> existing lawful procedures / Quack / mask-risc / ndarray SIMD
           -> independent result oracle
           -> separate authenticated commit path, only if requested

The square-bracketed stages are **research targets**. Actual ONNX and TFLite parsers are not established by this diagram. DR is a representation/execution option for proven tile/mask cases, **not** a blanket encoding of every model graph. MLIR may be an *optional import/export bridge*; it is not the mandatory internal ISA.

## Semantic domains that must not be unified accidentally

| Law family | Needed guard | Counterexample / refusal |
|---|---|---|
| Bool2 / set | typed co-addressed focus, valid two-valued inputs | XOR parity is not Boolean OR; source ordinals are not identity |
| SQL strong Kleene K3 | validity/UNKNOWN encoded separately | UNKNOWN is not valid FALSE; excluded-middle rewrites can fail |
| Bag / Distinct | multiplicity retained if observable | `UNION ALL(R,R)` does not equal `R` |
| Signed integer / quantized i8 | width, overflow, rounding, saturating/truncating mode | divide by power of two vs shift for negative values |
| IEEE FP / approximate | dtype, NaN, signed zero, tolerance, fast-math flags | reassociation/FMA alters results; ULP-bounded does not imply bit-identical |
| Relational addressing | axis types, Codebook/DR view, source generation and witness | same ordinal in two sources ≠ same entity |
| Effects / persistence | BIND/READ/EMIT/WRITE ordering and authority | equating return values does not prove safe write reordering |
| Reasoning / evidence | separate epistemic update law and provenance | neural likelihood, NARS revision and e-class equality are different objects |

A rewrite-catalogue key therefore needs: law ID/version, input/output types and shapes, exact/approx status, semantics (set/bag/K3), target ABI and capabilities, effect class, and necessary binding/source witness. Cache hits with stale generations must refuse or execute the unchanged trusted baseline.

## A minimal falsifiable capstone, not an implementation request

| Gate | Smallest test | Independent oracle / kill condition |
|---|---|---|
| DIAL-0 | Two toy frontend DAGs encode identical `int32 Add→Mul` semantics and one unknown operation | Canonical typed graph matches only where metadata matches; opaque op round-trips or refuses |
| DIAL-1 | Two different matrix-chain parenthesizations with fixed legal dimensions | Scalar/matrix reference; compare arithmetic work, cold planning cost and output, with FP policy explicit |
| DIAL-2 | 256 repeated typed predicate rules on a fixed 4096-slot bound frame | Compare baseline normalization vs bounded e-classes; record sharing and nonsharing controls |
| DIAL-3 | Adversarial Bool2/K3/Bag/Distinct and `x=0` / negative divide / FP NaN fixtures | Every unsound equivalence must be rejected, never merely penalized |
| DIAL-4 | Extract DR-friendly mask plan versus direct SoA read plan | Compare bind + focus gather + cache + execute + validate at 1/8/64/256 uses; no time-free setup |
| DIAL-5 | Change source generation, axis mapping or target ABI after capsule creation | No stale cache execution; deterministic refusal or trusted fallback |
| DIAL-6 | Effectful program with two aliasing writes and reordered reads | Reject changed observable order; no "proof of equality" counts as commit |
| DIAL-7 | Unknown ONNX/TFLite op with preserved graph dependencies | Opaque boundary preserves shape/version/effects or typed refusal; no silent numerical fallback |
| DIAL-8 | Bounded saturation with cyclic rewrites and a large nonsharing graph | Node/fuel/time ceilings enforced; no e-graph explosion in hot path |
| DIAL-9 | Repeat unchanged computation, then one genuine dirty mutation | Silent no-op stays silent; CandidateDirty, ActualDirty and CommittedDirty kept distinct |

**Report:** baseline, candidate and oracle outputs; schema/opset pin; semantic law IDs; extracted DAG and capsule bytes; compile/planning time; cold binding/source reads; warm/hot execution; cache churn; proof replay cost; failures/refusals; hardware and input shape. A shorter expression without end-to-end improvement is *not* a win.

## Decisions and open questions left as breadcrumbs

1. **No second tensor engine.** Prefer native arithmetic/mask/SoA/DR paths when lawful; preserve an explicit fallback/refusal surface for unsupported tensor semantics.
2. **No "universal relational ISA" claim yet.** First specify a minimal typed DAG and test a round trip, preserving existing separate query paths.
3. **No speculative rewrite as certified truth.** Ruler/Enumo/learning may generate *candidate* laws. Only exact declared domains, independent oracles and replayable witnesses can admit a rule.
4. **No optimizer in the hot loop by default.** Compile cold, bind with generation witnesses, execute existing primitives; measure amortization and miss cost.
5. **No hidden side effects.** Extraction and replay are not durable commit authorization.
6. **Keep the direction of translation reversible wherever supported.** Source AST/DTO → normal form → extracted procedure should retain operator provenance, shapes and a typed reason for any refusal.
7. **Question to answer before a new compiler crate:** what is the smallest cross-dialect example impossible to optimize soundly with current relalg normalization plus explicit hand-coded laws? Demonstrate that counterexample before adding a general engine.

## Nearby research notes / continuity links

- [Egglog unifies Datalog and equality saturation](2304.04332_egglog_equality_saturation_datalog_excerpt.md)
- [Eggcc: effect-safe extraction and ordering](2026-10_eggcc_effect_safe_extraction_statewalk_r2il_map.md)
- [Relational e-matching, Ruler and Enumo](2026-10_tatlock_relational_ematching_ruler_enumo_harvest.md)
- [3LA: formal interface, flexible matching, application validation](2026-10_3la_formal_interface_flexible_matching_application_validation.md)
- [Sutra: symbolic → tensor and exact/approx boundaries](2605.02337_sutra_tensor_normal_form_exact_approximate_boundary.md)
- [DR/QR256, Gaussian, egglog: independent falsifiers](2026-10_qr256_gaussian_egglog_rotation_falsifier_paths.md)
- [Kleene algebra / relational law boundaries](2026-10_zucker_kleene_algebra_kat_derivatives_relalg_excerpt.md)

**Historical breadcrumb (conversation, 2026-10-23–25):** the path of insight was *ONNX/TFLite graph untangling → internal typed Cartesian/DR/SoA addressing → RelAlg/mask-risc as execution mathematics → Egglog/Ruler for law exploration → DialEgg for a dialect-independent translation-and-extraction pattern → proof capsule, cost/focus and effect gates*. This is a **research lineage**, not a claim that the chain has been implemented.

**Source hygiene:** paper data come from DialEgg CGO 2025 ``3–9/Fig. 2–3/Table 2; repository statements from the paths and PRs linked above, checked 2026-10-25. No DialEgg binary or LG integration was executed for this note.
