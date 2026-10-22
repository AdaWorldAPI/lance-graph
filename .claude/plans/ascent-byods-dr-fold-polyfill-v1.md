# Ascent/BYODS → DR-native fold polyfill: staged falsification and cold-compiler plan

**Status:** PROPOSAL / documentation only, 2026-10-22. No implementation, dependency, P0 policy, or production admission change authorized by this PR.
**Scope:** harvest Ascent's *logic compiler* capabilities and BYODS formalism to produce existing RelAlg/R2IL/DR procedures; do not create an Ascent-backed runtime, another database, or a competing MeTTa AST.
**Sources:** [BYODS primary-source excerpt](../../.arxiv/2023_oopsla_byods_ascent_custom_relations_dr_excerpt.md), [author PDF](https://kmicinski.com/assets/byods.pdf), [AdaWorldAPI/ascent](https://github.com/AdaWorldAPI/ascent), [2026-10-21 prior evaluation](../board/entries/2026-10-21-eval-egglog-datafrog-ascent.md).
**Ownership:** [EXO-87](https://linear.app/exo-red/issue/EXO-87) D-SRAM; [EXO-84](https://linear.app/exo-red/issue/EXO-84) D-MTP; [EXO-85](https://linear.app/exo-red/issue/EXO-85) D-ASM; [EXO-79](https://linear.app/exo-red/issue/EXO-79) D-MPS-9; [EXO-86](https://linear.app/exo-red/issue/EXO-86) D-RA/D-CMR; [EXO-102](https://linear.app/exo-red/issue/EXO-102) D-RU; [EXO-46](https://linear.app/exo-red/issue/EXO-46) BIND admission; [EXO-73](https://linear.app/exo-red/issue/EXO-73) recursion driver; [EXO-50](https://linear.app/exo-red/issue/EXO-50) coordination.

## 0. Hypothesis, counter-hypothesis, and existing ruling

**Question:** can Ascent's compile-time rule dependency, SCC/stratification, bound/free matching and abstract-relation semantics be compiled into **already available**, certified RelAlg polyfill templates that produce R2IL or direct prebound DR SIMD? Then MeTTa and MORK are front-end aliases, not owners of a second execution model.

**Counter-hypothesis:** Ascent BYODS only substitutes its *relation provider*; its generated join loops and head writes still iterate over individual tuple bindings. A DR-backed `RelIndexRead` that yields bit positions one at a time is not a DR-native mask fold.

**Binding ruling:** [2026-10-21 egglog/datafrog/Ascent evaluation](../board/entries/2026-10-21-eval-egglog-datafrog-ascent.md) already ruled **SKIP** for Ascent as a hot runtime and withdrew tuple-native oracle roles. This research does not quietly reverse it. The difference is **cold compiler reuse**, not runtime delegation.

```text
COLD [candidate only]:
  static Ascent rule analysis / BYODS (D,inj,γ) semantic contract
    -> find genuinely missing schedule or relation law beyond existing crepe_plan!
    -> canonical, typed contract/relalg template
    -> independent proof/guards, bounded compiler work
    -> compiled R2IL FoldDialect recipe or direct prebound DR SIMD recipe
HOT [unchanged]:
  MeTTa/MORK/Datalog aliases -> Quack binding -> admitted compiled recipe
    -> R2IL/mask-risc/ndarray OR direct prebound DR bit folds
    -> Count/Any/Keep/etc.; no tuple expansion
OFFLINE PROOF [separate owners]:
  egglog congruence, proof; Ruler/Enumo law candidates; eggcc effect extraction
```

Do not manufacture an extra compiler representation when `crepe_plan!`, contract set_algebra, relalg and Quack already cover the operation. No hidden MORK/PathMap execution dependency.

## 1. Stages

| Stage | Narrow task | Falsifier / result gate | Status |
|---|---|---|---|
| **A0 source/ownership audit** | Extract BYODS formal model and compare Ascent source to D-SRAM, D-MTP, existing crepe-plan-probe and D-RU. Record conflict with old hot-path SKIP. | If no unique cold compiler feature is missing, PARK. | This docs PR |
| **A1 generated-code microscope** | Inspect fork `#[ds]` expansion, `rel_ind!`/`rel_ind_common!`, `RelIndexRead::index_get`, `RelIndexWrite` and one `R(x,y), S(y,z)` rule. Count per-binding iteration, owned tuples, head writes and indices. Source/codegen inspection before building anything. | If generated loops enumerate each match, **BYODS-only is NOT native fold execution**. Do not retrofit a new Ascent runtime in response. | Queued, lab-only authorization required |
| **A2 mathematical relation contract** | Specify finite-focus `D, inj, γ` (abstract lattice/domain, inject tuple logically, concretize without actually enumerating it) over sealed DR coordinates. Validate monotone `γ`, bottom, join laws and delta/total/new fixed-point semantics. | 257th out-of-focus slot, stale seal/ABA, non-monotone delta, changed coordinate identity or missing completeness must refuse. | Queued, docs/math first |
| **A3 *cold* static rule planning** | Compare SCC, negation strata, lattice preconditions, binding propagation, join-order metadata with existing `crepe_plan!`, `set_rules` and Quack on identical fixed rules. No second AST or executor. | If metadata gives no new lawful plan shape or only a second spelling, PARK. Test negative cycle, repeated variable, disconnected rule and nested expression refusal. | Queued, requires EXO-87 coordination |
| **A4 DR-native lowering** | On presealed co-addressed source/target frame, translate one positive 2-hop rule into native `AND` + fold; compare existing direct bitmask/RelAlg execution. Preserve `ADMISSION=B`. | Any per-result `next()`/write, foreign value-driven gather, owned tuple, stale source, wrong bag fan-out, NULL/FALSE conflation, unadmitted bind or silent result truncation kills the path. | Gated on A1/A2/A3 and EXO-46 |
| **A5 equality and recursion boundary** | Counterexample: `a ≡ b` should imply `f(a) ≡ f(b)` under congruence; an `eqrel` provider alone cannot supply rebuilding/proofs. Separately check a ≤256-slot cyclic/diamond closure by *independent mask algorithm* against frontier folds. | No claim Ascent = egglog e-graph; no claim recursion is production (EXO-73). Refuse cross-version and out-of-scope closures. | Optional follow-up |
| **A6 reuse economics** | Compare direct existing relalg/DR compile+execute to offline Ascent-derived schedule plus same hot recipe. Test shared-core 256 templates ×4096 focus frames, nonsharing control, cold cache hit/miss/staleness and negative syntax. | Net win only with *same workload and semantics*, accounting separately for cold compile, proof/verify, bind/attest, hot fold, outputs and memory. Avoid counting reduced tuple volume as a fold speedup. | Gated on real A3/A4 benefit |

## 2. Cold versus hot accounting: operator clarification requested

**New operator observation, 2026-10-22:** a previously rejected **400–600 ms** overhead must be evaluated as possible *one-time cold compilation*, not automatically against the nanosecond *hot* target; the previous MeTTa/MORK workloads can run for many seconds. These measurements are **not paired on identical inputs** and must not be treated as a measured acceleration.

Mathematical break-even, for the **same program and output semantics**, with one compile reused `N` times:

`T_compiled_total(N) = T_cold + N × (T_bind + T_execute + T_verify)`
`T_baseline_total(N) = N × T_interpreted`

Admit an **economic hypothesis** only when `T_baseline_total(N) > T_compiled_total(N)` for measured matched workloads. Also show cold-only, warm-only, first-call and nonsharing arms; cache invalidation by proof-law version, source generation, coordinate basis and compiler version.

**Policy conflict requiring an explicit owner amendment before publishing any timings as PASS:** `CLAUDE.md` P0 currently says “no millisecond results anywhere, ever,” including labs and PRs. This PR **does not edit or silently override** that ruling. Request operator decision to distinguish offline compiler economics (which may be measured with appropriate time units) from immutable hot-path fold constraints, while retaining P0 no owned tuple/materialized row responses. Until then, keep reported timings descriptive of independent past measurements, never as an admitted result or benchmark PASS.

## 3. Required independent semantics/refusals

- **Set/Distinct versus Bag:** count duplicate derivations and foreign fan-out separately. Boolean OR path existence is not a bag count and XOR is not OR. Repeatedly derived value does not imply new NARS evidence.
- **Bool2 versus SQL K3:** valid FALSE is not invalid NULL or an independent UNKNOWN. No unsafe negation or complement rewrite.
- **E-class versus entity identity:** an egglog union is mathematical congruence, never an authority to merge external identities or GUIDs.
- **Snapshots and local coordinate binding:** the same ordinal in two contexts need not denote the same object. Check axis/source/codebook/permission generation before reuse.
- **Pure versus effectful extraction:** proof of equality does not commit a Lance write, authorize a read, or allow moving effects across version barriers.
- **No independent tuple oracle:** compare direct fold against independently implemented mask/reference folds; no owned-pair vector, BTreeSet answer, or hidden row materialization.
- **No runtime Ascent, MORK or datafrog fallback:** if template unsupported, typed refusal or separately authorized frontend behavior, never silently invoke another evaluator.

## 4. Concrete first lab question

Generate a *static*, rule-bounded reachability fixture with a shared intermediate coordinate:

`Reach(x,z) :- Left(x,y), Right(y,z).`

- Baseline A: existing Quack/RelAlg co-addressed direct bitmask fold.
- Candidate B: Ascent-generated **plan metadata only**, translated into the same mask fold, without executing Ascent runtime.
- Diagnostic C: source-audited `#[ds]` generated code illustrating whether BYODS retains tuple enumeration. Do not use C as a runtime contender if it violates P0.
- Negative: same `y` ordinal from another seal; 257th focus; two distinct intermediate paths (OR true/XOR false); duplicate bag answers; cyclic recursion candidate not admitted; no-sharing workload.
- If B is byte-identical and no cheaper or more broadly legal than A, **stop**, keep Ascent as literature/reference rather than introducing optimizer machinery.
- If B exposes a genuinely missing legal transformation, propose **one** owner-scoped lab code PR, with a separate build/test plan, not an architecture rewrite.

## 5. Decision requested in EXO-50

1. Confirm cold-compiler amortization accounting as a *separate* explicit P0 exception/rule revision, rather than weakening the hot path. Until decided, no timing result promoted.
2. Authorize **A1 source/codegen audit only**, not a BYODS runtime adapter or a new Ascent/RelAlg executor.
3. Assign any A3 translator work to existing EXO-87 / D-SRAM and EXO-86 / D-RA owners, after their acknowledgement. No new plan family issue or D-id unless a unique deficiency survives A1–A3.
4. Keep egglog proof ownership and #1690–#1704 DR Math Foundry independent; no duplicate e-graph or proof verifier.

**Outcome sought:** either precise evidence that Ascent contributes a reusable cold optimization to native DR folds, or a documented reason to leave the existing RelAlg/DR compiler alone.
