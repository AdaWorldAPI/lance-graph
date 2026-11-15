# D-XLOG-KNOWLEDGE-0: XLOG as witnessed reasoning compiler for DT64 and AriGraph

**Status:** research and staged integration charter only; NOT implementation approval, not an admitted execution route.  
**Date:** 2026-11-15. **Owner:** existing EXO-50 / D-XLOG / D-CGMM / domain owners, decision required per gate.  
**Intent:** preserve the useful semantics of XLOG, Soufflé, Crepe, Scallop, MeTTa, MORK, Datafrog, Mistaber, Legalis and regelrecht in the existing Codegraph → RelAlg/R2IL/Quack → DT64/DR Cube → witnessed AriGraph architecture, not create a second ontology runtime.

## 0. Operator thesis and non-goals

The goal is **not** to make Datafrog faster or to invent a universal graph interpreter. XLOG is the candidate *central reasoning compiler and caller contract*. MeTTa/Hyperon supplies higher-order language declarations, MORK supplies structural matching/rewrite behavior where lawful, Crepe/Soufflé provide cold compiler/demand/stratification donor evidence, Datafrog supplies an independent monotonic Datalog oracle, and Scallop supplies provenance-semiring/probabilistic comparison obligations. All actual admitted hot relational work is a borrowed ABI operation over resident DT64/DR Cube via existing RelAlg/R2IL/Quack/mask-risc. XLOG RIR/PIR/EIR remain **separate** reasoning routes; a probabilistic or modal fact must never be flattened into an unqualified Boolean Datalog tuple.

Mistaber's authority/world/override/source semantics, Legalis's distinction between deterministic decision and judicial discretion, regelrecht's statute/scenario/explanation contract and AriGraph's semantic→episodic memory are *consumers and falsifiers*. No automatic legal advice or normative judgement is certified by a green algebra test.

No new persistent ontology graph, ClassId or MethodRef authority, new F### word, arbitrary interpreter, generic `CudaBuffer` port, memory copy membrane, hash table keyed by DN as join engine, hot JSON/Arrow IPC/owned tuple sets, unbounded value-derived cross-lane Gather, unauthorized effects, or production admission. Internal evidence may be readable through a cold reporting projection, not carried as serializations through the execution path.

## 1. Source-backed decisions to preserve

| Existing authority | What it means for this charter |
| --- | --- |
| [lance-graph #1910, Amendment E](https://github.com/AdaWorldAPI/lance-graph/pull/1910) | Assimilate XLOG's RIR node semantics into OUR existing algebra; retain XLOG caller API via polyfill; do not port CUDA runtime. Unsupported shapes refuse. |
| [lance-graph #1911](https://github.com/AdaWorldAPI/lance-graph/pull/1911), [xlog #2](https://github.com/AdaWorldAPI/xlog/pull/2) | One resident Cube, view handles and explicit terminals; XLOG `CubeExecutor` is an excluded/lab integration, not main runtime. Host-side adjacency masks/diff and row construction remain to be eliminated or typed as not-native. |
| [lance-graph #1913](https://github.com/AdaWorldAPI/lance-graph/pull/1913) | Shared F000 algebra and dedicated F100 XLOG behavior lens for `functor/3`, `arg/3`, `=../2`. A term builtin is not a data relation. W7 behavior authority remains a separate admission decision. |
| [lance-graph #2084](https://github.com/AdaWorldAPI/lance-graph/pull/2084), [#2091](https://github.com/AdaWorldAPI/lance-graph/pull/2091), [xlog #8](https://github.com/AdaWorldAPI/xlog/pull/8) | Existing independent parsed-AST naive Datalog oracle and coverage census, with seeded derived facts and refusal-propagation regressions. Last reported 230 rules: 87 oracle agreements (43 native-tested), 94 refused from empty bases, 49 refused on data, 0 disagreements and 0 unverified; **reproduce at pinned commits**. These are verdict categories, not production admission. W5 recursion/fixpoint and W7 F100 remain undecided. |
| [lance-graph #1873](https://github.com/AdaWorldAPI/lance-graph/pull/1873), [#1874](https://github.com/AdaWorldAPI/lance-graph/pull/1874), [#1880](https://github.com/AdaWorldAPI/lance-graph/pull/1880) | Mistaber Kripke `holds` least-fixpoint has 11 Clingo goldens; three-axis DR64³ world/origin/rule contraction; 141 `makor` sources addressed via deepnsm-v2. Non-stratified negative cycle is explicitly outside this native subset. |
| [lance-graph #1921](https://github.com/AdaWorldAPI/lance-graph/pull/1921), [#1958](https://github.com/AdaWorldAPI/lance-graph/pull/1958) | Codegraph already uses a resident XLOG Cube for graded `CALLS/NODE/MAY/IMPL/RCONF/RPOSS`. Kythe/Ruff/Charon are source facts, not algebraic law. MeTTa-shaped `FieldViewDef` remains a **proposal**, not an executable new schema owner. |
| [lance-graph #1555](https://github.com/AdaWorldAPI/lance-graph/pull/1555) | `LanceSpace` is the proposed named Hyperon Space boundary. Keep term construction, grounded effects, types and non-admitted behavior in its source owner. |
| [lance-graph #725](https://github.com/AdaWorldAPI/lance-graph/pull/725), [#1005](https://github.com/AdaWorldAPI/lance-graph/pull/1005) | AriGraph-like semantic→episodic recall is distinct from proof authority; memory should be an ABI-resident view over existing geometry, not a second heap-valued BeliefArena/TripletGraph. |
| [lance-graph #1381](https://github.com/AdaWorldAPI/lance-graph/pull/1381), [#449](https://github.com/AdaWorldAPI/lance-graph/pull/449) | CE64 Epi5 and existing EW64/episodic witness tenancy must be inventoried first. The user calls the intended bridge `EpistemicWitness64`; **do not assume this literal Rust type exists** or mint bits without residue/ownership audit. |

## 2. Target flow and exact authority cuts

```text
Ruff/Charon/Kythe source receipts + Legalis/regelrecht/Mistaber domain declarations
        |   MeTTa declarations / MORK pattern candidates
        v
Typed Codegraph facts + source identity + scope + coverage grade
        |           [cold: Soufflé demand/magic + Crepe binding patterns + XLOG plans]
        v
XLOG AST -> (RIR | PIR | EIR)        [never EIR/PIR as plain RIR by fiat]
        |   semantic witnesses, source spans, owner, rule/MethodRef, world
        v
Existing RelAlg / R2IL / Quack / mask-risc admission
        v
DT64 / DR64 / Cube borrowed masks and views; fold or retained facet
        v
Existing CE64/Epi5 + proven witness owner + DerivationLedger/Thought-Git
        v
AriGraph semantic+episodic projection / supervised decisions
```

MORK's PathMap and MeTTa's executable language remain independent external oracle/behavior surfaces for non-lowered forms. Datafrog is a Datalog fixpoint comparator, **not** the production substrate. The source of truth stays in original facts/code; Codegraph is a versioned, coverage-aware projection; XLOG plans do not issue MethodIds or execution permission.

### Required logical receipt (conceptual fields, not new Rust structs)

`proposition_id, predicate_signature, truth_mode, world_id, authority_id, rule_ref, source_roots, provenance/independence, source_generation, tenant/scope, revision/thought, evidence_grade, derivation_kind, refusal_or_uncertainty, semantic_route, compiled_plan_version`.

Resolve any named label to existing IDs **once cold**. A witness address may point to existing facts, masks, or compact proof facets; do not cram full derivations into 64 bits. Preserve independent source-root counting. Distinguish:
- **fact** vs inferred relation, **confirmed** vs **possible** vs **unknown** vs **refused**;
- **set** support vs **bag** multiplicity and multiple derivations;
- **direct** vs **indirect** topology, **epistemic status** vs **causal certification**, **authorization** vs mathematical entailment;
- **world** vs source authority vs tenant boundary vs thought-git revision;
- **founded least fixpoint (FAEEL)** vs **G91 greatest compatible tuple fixpoint** vs **WFS three-valued** vs **Clingo ASP stable models**;
- **probability** vs NARS frequency/confidence vs source reliability. No automatic numeric mapping without proved semantics.

Never turn missing Charon coverage, untracked dispatch, unsupported modal key or rejected SCC into a false closed-world negative.

## 3. Source harvest: XLOG and donors

Harvest the deployed [xlog.md](https://xlog.md/) **and** the pinned [AdaWorldAPI/xlog](https://github.com/AdaWorldAPI/xlog) and [BrainyBlaze/xlog](https://github.com/BrainyBlaze/xlog) source trees, comparison tests, CHANGELOG/release tags, site docs source and build status. Do not trust `/llms.txt` as a complete index until verified; as of this intake it did not resolve through the public fetch. Enumerate source documentation files and reachable site navigation before declaring page count or completeness. Preserve URLs, source commit/path/line, retrieval date, release status and page-to-code symbol maps.

Required domains: frontend/parser/typecheck/AST normalization; modules and imports; magic-sets, stratification, aggregation, negation; 13 RIR node kinds and optimizer dispatch; RIR/PIR/EIR ownership; FAEEL/G91/WFS contracts; provenance semiring / exact and MC inference; WCOJ/Free Join/factorized delta; CudaBuffer and residency boundary; GpuCnf/CDCL; diagnostics, declines, counters, route kill switches; packaging and public caller API; Mistaber pinned acceptance corpus; benchmark and claim provenance; licenses.

Donor census with anchored code facts and semantics, not name-only analogies:
- [Soufflé](https://github.com/souffle-lang/souffle): magic-demand, stratification, index planning, incremental recursion; distinguish optimizations from semantics.
- [Crepe](https://github.com/ekzhang/crepe): typed Rust macro grammar and compile-time relation binding; verify actual fork/API before use.
- [Scallop](https://github.com/scallop-lang/scallop): annotated/provenance semirings, multiplicity, differentiable inference; compare to PIR independently.
- [Datafrog](https://github.com/AdaWorldAPI/datafrog): minimal `Iteration` and join/fixpoint comparator.
- [MORK](https://github.com/AdaWorldAPI/MORK), [metta-on-mork](https://github.com/AdaWorldAPI/metta-on-mork), [PeTTa](https://github.com/AdaWorldAPI/PeTTa), [Hyperon](https://github.com/AdaWorldAPI/hyperon-experimental): pattern, normalization, unification, execution, Set/Bag semantics; don't assume equivalence.
- [Mistaber](https://github.com/AdaWorldAPI/mistaber), [mistaber-rs](https://github.com/AdaWorldAPI/mistaber-rs): Clingo goldens, authority/world/override/makor provenance.
- [Legalis](https://github.com/AdaWorldAPI/legalis), [regelrecht](https://github.com/AdaWorldAPI/regelrecht), [regelrecht-corpus](https://github.com/AdaWorldAPI/regelrecht-corpus), [AriGraph](https://github.com/AdaWorldAPI/AriGraph): legal determinism/discretion, law model + BDD, episodic memory; licenses and corpus rights reviewed before fixture copying.

Output a **capability and claim matrix**: source URL + commit/line, declared claim, exact semantics, implementation proof, oracle/fixture, DT64 operation mapping, external consumer, runtime vs cold only, current status (implemented / lab / proposal / refused / unknown), owner, smallest falsifier and next decision. See companion research seed and Claude Code prompt in this PR.

## 4. Stage gates: separate PRs, not one big implementation

| Gate | Minimal proof and independent falsifier | Refusal / dependency |
| --- | --- | --- |
| G0 XLOG harvest and loss census | Pinned page/source coverage matrix; three-IR fanout; 13 RIR kinds; no silent source-vs-site drift. Corrupt one cited source anchor -> validation red. | Documentation evidence is not executor admission. |
| G1 deterministic coverage | Re-run existing 230-rule census; nonempty seeds; query heads, source-compiled plans, refusal propagation, derived facts, diagonal, projection, n-ary, duplicates, negation; parsed-AST naive oracle and Datafrog as separate comparators. Corrupt a rule -> mismatch or typed refusal, never empty success. | Do not reset #2091 baseline or count host enumeration as native. |
| G2 native delta and factorized algebra | Frame-stamped resident views; recursive SCC, multi-way fallback and support/bag terminals; same results with route kill switch; counters prove which path executed. Break frame/source generation -> fail closed. | W5/EXO-73 recursion, EXO-108 Bag, Gather prohibition, no host set diff as hot implementation. |
| G3 MeTTa/MORK structural subset | One bounded typed common relational subset across Hyperon, PeTTa, metta-on-mork, Datafrog and XLOG; normalized binding semantics including repeated variables, nested term refusal and duplicate multiplicity. Mutate a match or Set/Bag witness -> comparator red. | No universal MeTTa interpreter or unbounded grounded callbacks; EXO-79/84/85/143. |
| G4 Mistaber Kripke provenance | Reproduce 11 Clingo goldens, three-axis world/origin/rule contraction and 141 makor intake; withdrawal of an override's supporting root affects only the right world while preserving lineage. | Non-stratified ASP negative cycles stay Clingo-owned; no EIR=ASP claim. |
| G5 Legalis + regelrecht | One bounded, synthetic legal scenario per owner and an override case; compare Legalis deterministic/judicial-discretion/void with regelrecht YAML→decision/explanation and independent Gherkin BDD. Change one statute/source revision -> expected explanation and witness invalidation. | Different jurisdictions/formalisms are not asserted equivalent; human judgment stays explicit; licensed corpus boundaries. |
| G6 epistemic/provenance bridge | Audit `EpistemicWitness64` name against existing EW64/CE64/Epi5; preserve world, rule, evidence root, source-independent count and thought revision; EIR/PIR source spans retained. Disable one root -> certification drops or abstains, never rises. | No invented bit assignment, no causal certification from modal truth, no unauthorized F### word; EIR/PIR gate separate from RIR. |
| G7 AriGraph projection | Semantic result seeds episodic search; retrieval of historical observation must not overwrite revised belief. Change source generation -> new derived view; historical episode remains attributable. | AriGraph memory is not another truth authority; no `Vec`/hash-based population transfer. |
| G8 performance and admission | Measure per-operation ns, batched µs, allocations, copied canonical bytes, native counts, declines and hot-path materializations; preserve source/feature/CPU hardware pins. Disable optimized route -> observable counter and equivalent answer. | Any hot-path row reconstruction or serialization is a failed gate. No generalized transfer-free claim or production admission. |

G0+G1 evidence first. G2+G3 are independent; G4 reuses existing probes; G5-G7 require owners. Each follow-up PR must carry a fixture, independent oracle, deliberate disable/red test, exact source provenance, implementation/admission classification and a refusal boundary. Benchmark comparisons are reported only under matched semantics, hardware and scope.

## 5. Immediate deliverables of *this* PR

1. This cross-repository charter, keeping prior decisions rather than duplicating old plans.
2. [xlog.md source-backed preliminary harvest](xlog-knowledge-spine-harvest-20261115.md), which explicitly marks partial coverage and documentation drift.
3. [Claude Code execution prompt](xlog-knowledge-spine-claude-code-prompt.md) to complete the harvest, build the claim/IR/ABI crosswalk, and propose the **smallest next lab PR** without auto-merging.

**Nothing is built, run, refactored, or authorized by merging this documentation PR.** Do not backfill measured status, use unchecked references or rewrite historical board records. Use existing board/EXO supersession procedure in the next owner-approved work.
