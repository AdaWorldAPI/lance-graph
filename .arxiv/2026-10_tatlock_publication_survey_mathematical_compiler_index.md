# Tatlock publications: prioritized research map for the LG mathematical compiler

**Checked:** 2026-10-22 against Zachary Tatlock's [official publications catalog](https://ztatlock.net/pubs/) (conference/journal papers and workshop papers). **Scope:** bibliographic survey of the wider catalog, with closer inspection of the listed primary sources. It is not a claim to have read every PDF in the publication list.

**Relationship to existing work:** This is an index for follow-up after [egglog PLDI 2023](2304.04332_egglog_equality_saturation_datalog_excerpt.md), [eggcc OOPSLA 2026](2026-10_eggcc_effect_safe_extraction_statewalk_r2il_map.md) and [3LA 2024](2026-10_3la_formal_interface_flexible_matching_application_validation.md), which are covered separately. Companion deeper harvests: [matching/rule discovery](2026-10_tatlock_relational_ematching_ruler_enumo_harvest.md) and [abstraction/proofs/target-aware](2026-10_tatlock_babble_smallproofs_chassis_lakeroad_glenside.md).

## Most directly transferable publications

| Rank | Paper and primary link | Verified main contribution | LG hypothesis (NOT a shipped feature) | Evidence depth |
|---|---|---|---|---|
| A1 | [Relational E-matching (POPL 2022)](https://ztatlock.net/pubs/2022-popl-rematch/2022-popl-rematch.pdf) | E-matching lowered to conjunctive queries and worst-case-optimal generic joins for fixed-pattern data complexity; includes scan fast path, variable-order and batching considerations | Match many rule templates over term relations through a cold, typed join/mapping plane | PDF + project page |
| A2 | [babble (POPL 2023)](https://ztatlock.net/pubs/2023-popl-babble/2023-popl-babble.pdf) | Library learning modulo equational theory; e-graph anti-unification discovers reusable abstractions across syntactic variations | Compress 256 related R2IL recipes into *shared templates* with independently costed physical reuse | PDF + project page |
| A3 | [Small Proofs from Congruence Closure (FMCAD 2022)](https://ztatlock.net/pubs/2022-fmcad-smallproofs/2022-fmcad-smallproofs.pdf) | Compact and checkable equality certificates; distinguishes proof DAG size from proof tree size | Share certified law derivations without conflating logical proof with committed physical writes | PDF + project page |
| A4 | [Chassis / Target-Aware Implementation of Real Expressions (ASPLOS 2025)](https://arxiv.org/pdf/2410.14025) | Target-aware numerical instruction selection modulo a real-expression desugaring, balancing accuracy and cost | Explicit approximate numeric subroute and target descriptor, **not** a license to approximate SQL/Bool/CE64 logic | PDF + author abstract |
| A5 | [Enumo (OOPSLA 2023)](https://ztatlock.net/pubs/2023-oopsla-enumo/2023-oopsla-enumo.pdf) | Composable, programmable exploration strategies for candidate rewrite theories | Bounded rule-synthesis research without hard-coding one monolithic search | PDF + project page |
| A6 | [Ruler (OOPSLA 2021)](https://ztatlock.net/pubs/2021-oopsla-ruler/2021-oopsla-ruler.pdf) | Infer smaller rewrite theories using equality saturation during enumeration/filtering | Propose new typed laws with external confirmation gates | PDF + project page |
| A7 | [Glenside (MAPS 2021)](https://ztatlock.net/pubs/2021-maps-glenside/) | Pure tensor IR with layout-aware access patterns and emergent data-layout rewrites | Versioned coordinate/view semantics for global-to-local binding | Author abstract |
| A8 | [Lakeroad (ASPLOS 2024)](https://ztatlock.net/pubs/2024-asplos-lakeroad/2024-asplos-lakeroad.pdf) | Generate verified FPGA mappings from sketch-guided synthesis and HDL model semantics | Pattern-to-primitive correctness checker over *existing* R2IL/mask-risc ops | PDF + project page |

## Secondary ideas worth preserving, distinct from the compiler's exact-algebra core

| Paper | Source-level observation | Possible scoped follow-up | Evidence depth |
|---|---|---|---|
| [Combining Precision Tuning and Rewriting / Pherbie (ARITH 2021)](https://ztatlock.net/pubs/2021-arith-herbie/) | Interleaving precision changes and rewrites improves Pareto trade-offs on tested FPBench suite | Experimental f64/f32 accuracy/cost frontiers; separate from exact set arithmetic | Author abstract |
| [Odyssey (UIST 2023)](https://ztatlock.net/pubs/2023-uist-odyssey/) | Numerical analysts need editable expression alternatives and local error visualization | A human-review surface for unsafe/approximate optimization candidates; no autonomous semantic approval | Author abstract |
| [Dynamic Tensor Rematerialization (ICLR 2021)](https://ztatlock.net/pubs/2021-iclr-dtr/) | Online eviction and recomputation with a memory-budget objective | Experiment with compiler scratch-plane retention versus recompute, after profiling | Author abstract |
| [Sinking Point (CoNGA 2019)](https://ztatlock.net/pubs/2019-conga-sinkingpoint/) | Precision metadata follows intermediate floating-point values | Numeric error telemetry for separately admitted approximate lanes | Author abstract |
| [egg: Fast and Extensible Equality Saturation (POPL 2021; CACM highlight 2026)](https://ztatlock.net/pubs/) | Shared basic e-graph saturation concepts; CACM revisit is a research highlight of the earlier egg work | Historic baseline, not evidence of new egglog 3.0 engine behavior | Official catalog |
| [Equality Saturation: A New Approach to Optimization (POPL 2009 / LMCS 2011)](https://lmcs.episciences.org/1016) | Foundational equality saturation and rewrite-phase ordering | Cite historical origin only; current extraction/verification needs later papers | Journal landing page |
| [Icing: Supporting Fast-Math Style Optimizations in a Verified Compiler (CAV 2019)](https://ztatlock.net/pubs/) | Source catalog places the work in verified numerical compilation | Investigate exactly which fast-math preconditions can be admitted; no blanket f64 algebraic associativity | Index/title only |
| [Proving Optimizations Correct using Parameterized Program Equivalence (PLDI 2009)](https://ztatlock.net/pubs/2009-pldi-pec/2009-pldi-pec.pdf) | Parameterized translation-validation approach seeks reusable proofs for transformations | Candidate proof-policy gate for typed rewrite templates; separate full-paper audit needed | Abstract/primary search text |
| [Magic Markup (Programming 2024)](https://ztatlock.net/pubs/2024-programming-magicmarkup/2024-programming-magicmarkup.pdf) | LLM-supported external markup reattachment as source documents change | Potential annotation/provenance relocation for evolving Wordcraft/docs; heuristic matching is *not* stable identity or authoritative source location | Primary abstract only |
| [Formal Semantics and Automated Verification for BGP (NetPL 2016)](https://conferences.sigcomm.org/sigcomm/2016/netpl.php) | Coq BGP semantics, network-policy checkers and concrete anomaly detection | A **network-policy oracle** for outage hypotheses, not evidence identifying tower power failure versus grid failure | Conference abstract |
| [Scalable Verification of BGP Configurations with SMT (OOPSLA 2016)](https://ztatlock.net/pubs/) | Config-level BGP policy verification topic in catalog | Additional network-routing source to review separately for Bangladesh-outage; do not conflate observed outages with configuration causes | Index/title only |

The site also includes CAD/fabrication/knitting, page-layout verification, robotics, distributed systems, low-level language/proof toolchains, and workshop papers. These are real publications, but no direct compiler-specific recommendation is supported by merely seeing their titles. Do not claim this index is a PDF-level assessment of every item.

## Revised tentative compiler decomposition

```text
COLD, auditable
    domain-specific typed grammar / source IR
       -> Ruler/Enumo-style law candidate generator
       -> independent oracle + guard admission
       -> egglog saturation with semi-naive deltas
       -> relational e-matching (where measured worthwhile)
       -> babble-style reusable abstraction templates
       -> eggcc-style effect-safe extraction
       -> Chassis-style target-aware accuracy/cost (approximate numeric ONLY)
       -> compact equality proof DAG / fingerprint / cache

BOUNDARY
       R2IL: canonical SIMD polyfill + FoldDialect procedures
       Quack: plan and physical execution selection
       mask-risc: execute_into on pinned, typed coordinate planes

INDEPENDENT VERIFICATION
       scalar/postgresql/other domain-specific oracle
       validation of source generations, multiplicity, K3/validity
       authorization + distinct actual writeback acknowledgement
```

The diagram shows responsibilities one could combine, **not** a proposal to import every upstream tool. Begin with already-shipped relalg normalization/CSE, measure its gaps, and only then test external design techniques on one bounded offline fixture. Several paper approaches would compete or overlap in code and planning costs if imported literally.

## Explicit refusal fences

1. Source fact equality/e-classes must not become identity equality of external AD/Entra/EXO records, even if local ordinals match.
2. SQL valid FALSE, SQL invalid/NULL and separately modeled logical UNKNOWN do not collapse.
3. Bag versus Distinct cannot be erased by Boolean idempotence; an equality in a set algebra cannot be promoted across semirings.
4. Exact integer GF(2)/Bool and f64 approximate arithmetic use distinct verified law sets; do not use Chassis-style mathematical desugaring to promise float-bit identity.
5. Compiling an effect-safe R2IL procedure does not authorize an out-of-policy cross-table gather, change durable storage or assert `CommittedDirty`.
6. Proof size, AST compression, match throughput and end-to-end runtime are separate metrics. No speedup claims for LG without measurements.

## Minimal next falsifier (no implementation in this docs harvest)

**256×4096 benchmark:** generate one shared-core 256-rule suite over 4,096 pinned contexts, plus a nonsharing control. Compare baseline direct relalg/R2IL compilation to bounded law search, matching, reusable templates and extracted procedures. Independent scalar/PostgreSQL oracles pin truth, bag counts, source generation and adversarial duplicates. Record planning, e-graph size, matches, proof/certificate bytes, bind/index time, source reads, mask executions, scratch/output bytes, verification time and total wall time. A narrower but reliable pipeline is preferred over a larger unverified compiler.

**Context:** [D-RELALG-UNIFIED](../.claude/plans/d-relalg-unified-v1.md) documents the intended owner boundary and ADMISSION=B; **do not silently upgrade plans or lab probes to shipped production functionality**. Review merged main again before coding.
