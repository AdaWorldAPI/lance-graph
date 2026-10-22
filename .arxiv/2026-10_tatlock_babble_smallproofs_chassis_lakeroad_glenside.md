# Tatlock / PLSE: reusable abstractions, compact proofs, target-aware numerical lowering

**Date:** 2026-10-22. **Status:** research-only, original paraphrase, falsifier design; not deployed in lance-graph. **Sources read:** author-hosted PDFs for babble, Small Proofs, Lakeroad and arXiv Chassis. Abstract-level cross-check for Glenside, Dynamic Tensor Rematerialization and Pherbie. Distinguish claims in a source from hypothetical LG transfer.

## A. babble: library learning modulo an explicitly chosen theory

**Source:** Cao et al., [*babble: Learning Better Abstractions with E-Graphs and Anti-Unification*](https://ztatlock.net/pubs/2023-popl-babble/2023-popl-babble.pdf), POPL 2023, DOI [10.1145/3571207](https://doi.org/10.1145/3571207), §§2–5 and limitations.

**Paper-derived:** Syntactically different programs can contain identical *behavioral* subpatterns that ordinary library learning misses. babble performs library learning modulo theory (LLMT): e-graphs represent equivalence classes under a domain-specific set of equational laws; anti-unification finds common patterns across e-classes rather than comparing only original AST shapes. The system then selects library definitions according to a compression objective. The paper's Figure 3 compares syntactic learning against LLMT on a geometric example, illustrating that compression can remain stable despite syntactic variation. The paper explicitly notes that equational theories are provided by domain experts and that the candidate generator is not exhaustive.

**LG transfer hypothesis:** Treat a learned abstraction as a **cold template library** that compiles many logical requests into a small catalog of typed, reusable R2IL procedures. For 256 related rules, e-class anti-unification could find a common predicate core even when syntactic forms differ. This is *not* evidence that one physical fold will execute all 256 requests at no additional cost. Every template still requires focus bindings, valid source versions, output materialization and identity/lineage guards.

**BAB1 falsifier:** Two deliberately syntax-varied rule corpora, one with genuine shared structure and one with no cross-rule sharing. Compare direct CSE/normalize against anti-unification templates on (1) total source reads, (2) unique compiled procedures, (3) binding/projection time, (4) extracted instruction count and (5) end-to-end time with trace verification. If savings exist only in AST size, do not claim runtime improvement. If learned abstractions merge Bag and Set or invalid NULL and FALSE, reject.

## B. Small Proofs: proof minimization is separate from program minimization

**Source:** Flatt et al., [*Small Proofs from Congruence Closure*](https://ztatlock.net/pubs/2022-fmcad-smallproofs/2022-fmcad-smallproofs.pdf), FMCAD 2022, 9 pages. [Project page](https://ztatlock.net/pubs/2022-fmcad-smallproofs/).

**Paper-derived:** E-class congruence proofs can contain overlapping subproofs. The paper distinguishes *proof tree size* from *DAG size of unique equalities*; minimizing DAG size is NP-complete, while an O(n^5) algorithm optimizes a relaxed tree-size objective. A practical greedy shortest-path-inspired method runs in O(n log n) and returns small (not necessarily minimal) proofs. The reported corpus compares against existing proof-generation approaches, not a general reversible state-history mechanism.

**LG transfer hypothesis:** For cached mathematical lowering, store a DAG of the applied **typed rewrite laws** and replay-check it against the canonical source term. Reuse shared subproofs rather than copying one proof per focus context. The proof certifies logical equality of terms under the declared domain, **not** physical pointer validity, source versions, exact output bytes without execution, or a durable write acknowledgment.

**SP1 falsifier:** Chain/fan-out algebra witnesses with deliberate repeated subproofs; compare naive proof tree and shared DAG bytes, independent verification time and failed-premise refusal. A tampered law ID, revoked premise, inconsistent source-version guard or swapped coordinate type must invalidate the claimed equivalence/compilation. Compare certificate size with the number of **independent output facts**, which cannot be compressed by merely sharing proof structure.

## C. Chassis: desugaring-preserving instruction selection, not fast-math without bounds

**Source:** Saiki et al., [*Target-Aware Implementation of Real Expressions*](https://arxiv.org/pdf/2410.14025), ASPLOS 2025, arXiv:2410.14025; DOI [10.1145/3669940.3707277](https://doi.org/10.1145/3669940.3707277). Author PDF §§3–5 checked.

**Paper-derived:** Chassis represents each target instruction in terms of the *real-valued expression* it approximates (a desugaring) plus cost/accuracy behavior. Different floating-point implementations may share a mathematical desugaring while yielding distinct floating behavior, including special cases. Instruction selection modulo equivalence searches a target-aware implementation space; an iterative heuristic concentrates work on expensive or inaccurate subexpressions. The evaluated systems cover nine targets and report improvements on the authors' selected metrics/workloads. This is **not** a theorem that every mathematically equivalent float reassociation is bitwise valid.

**LG transfer hypothesis:** A target description should distinguish the *mathematical task* from exact machine semantics and approximation policy. In the **exact** lane, R2IL execution respects fixed integer, Boolean, SQL validity/K3 and Bag contracts; there is no numerical budget license to change those. In an independently opted-in **approximate numeric** lane, allow an accuracy objective together with target-specific operators, e.g. ndarray/SIMD/fp64, using a specified distribution, domain, overflow behavior and measured max/ULP error. Never reinterpret CE64 evidential certainty, K3 UNKNOWN or SQL NULL as continuous float without a distinct proven contract.

**CH1 falsifier:** Compare stable and cancellation-prone expressions on exact and approximate routes (including near-zero denominators, nonfinite floats and large-magnitude cancellation). Pin bit-exact baseline where required and a declared error metric where approximate. Log compile time, selected target op, materializations and end-to-end execution, not just simulated cost. Refuse approximate lowering into access-policy or predicate semantics.

## D. Two additional design analogues, distinct scope

### Lakeroad (ASPLOS 2024): synthesize mapping to implementation primitives

**Source:** Smith et al., [*FPGA Technology Mapping Using Sketch-Guided Program Synthesis*](https://ztatlock.net/pubs/2024-asplos-lakeroad/2024-asplos-lakeroad.pdf), ASPLOS 2024; DOI [10.1145/3620665.3640387](https://doi.org/10.1145/3620665.3640387). Author PDF inspected.

**Paper-derived:** Architecture-independent sketches plus HDL-derived primitive semantics allow synthesis of hardware technology mappings, with correctness arguments relative to those models. The reported 2–3.5x/6–44x mapping improvements are based on selected FPGA microbenchmarks, not general CPUs.

**LG transfer:** Analogize the sketch to a typed R2IL procedure template and the implementation library to existing mask-risc fused operators. A *model-based* checker can reject unavailable/incorrect opcodes before physical selection. No hardware/RTL equivalence is established for LG by borrowing this idea, and no FPGA dependency is proposed.

**LK1:** Enumerate tiny permitted operator families with an independently interpreted reference and one intentionally incorrect primitive specification. The selector must reject the invalid mapping and preserve the original compiled fold.

### Glenside (MAPS 2021): make access/layout explicit in a pure IR

**Source:** Smith et al., [*Pure Tensor Program Rewriting via Access Patterns*](https://ztatlock.net/pubs/2021-maps-glenside/), MAPS 2021; DOI [10.1145/3460945.3464953](https://doi.org/10.1145/3460945.3464953). Source status: author abstract, not a line-by-line full-paper audit.

**Paper-derived:** Glenside uses a pure access-pattern IR to express layout-aware tensor transformations and expose accelerator mappings, including im2col-like patterns.

**LG transfer:** Carry explicit **coordinate domain and pinned local-index mapping** into law guards. A change of view is not a change of entity identity. Reordering a layout can be legal as a pure view when the inverse mapping and axis typing are proved; value-directed cross-table positional gathers remain prohibited under ADMISSION=B.

**GL1:** Permute a purely presentational QR view and verify inverse lookup yields the identical typed relation. Swapping method/entity axes or directed-edge endpoints without reinterpreting types must refuse.

## Decision gates

Priority 1: relational e-matching plus babble patterns for **measured cold compilation reuse**, but only after simpler CSE baseline. Priority 2: proof DAG for *checked* extraction and a replayable cached law derivation. Priority 3: Chassis-type target-aware math optimization confined to opted-in numeric domains. Priority 4: Lakeroad/Glenside as specification/design references. No evidence here warrants importing a second hot runtime, weakening SQL NULL semantics, or changing R2IL's role as the SIMD polyfill.

**Adjacent sources:** [rule generation and matching](2026-10_tatlock_relational_ematching_ruler_enumo_harvest.md); [egglog](2304.04332_egglog_equality_saturation_datalog_excerpt.md); [eggcc](2026-10_eggcc_effect_safe_extraction_statewalk_r2il_map.md); [3LA](2026-10_3la_formal_interface_flexible_matching_application_validation.md).
