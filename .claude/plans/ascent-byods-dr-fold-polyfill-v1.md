# Ascent/BYODS → DR-native fold polyfill: staged falsification and cold-compiler plan

**Status:** PROPOSAL / documentation only, 2026-10-22. No implementation, dependency, P0 policy, or production admission change authorized by this PR.
**Scope:** harvest Ascent's *logic compiler* capabilities and BYODS formalism to produce existing RelAlg/R2IL/DR procedures. A logical tuple is an **addressed projection of original lanes**. When the index itself witnesses the join, no mask intersection is required; when searching/filtering, intersect masks to obtain eligible addresses. Neither case constructs an owned pair/row. Do not create an Ascent-backed runtime, another database, or a competing MeTTa AST.
**Sources:** [BYODS primary-source excerpt](../../.arxiv/2023_oopsla_byods_ascent_custom_relations_dr_excerpt.md), [author PDF](https://kmicinski.com/assets/byods.pdf), [AdaWorldAPI/ascent](https://github.com/AdaWorldAPI/ascent), [2026-10-21 prior evaluation](../board/entries/2026-10-21-eval-egglog-datafrog-ascent.md).
**Ownership:** [EXO-87](https://linear.app/exo-red/issue/EXO-87) D-SRAM; [EXO-84](https://linear.app/exo-red/issue/EXO-84) D-MTP; [EXO-85](https://linear.app/exo-red/issue/EXO-85) D-ASM; [EXO-79](https://linear.app/exo-red/issue/EXO-79) D-MPS-9; [EXO-86](https://linear.app/exo-red/issue/EXO-86) D-RA/D-CMR; [EXO-102](https://linear.app/exo-red/issue/EXO-102) D-RU; [EXO-46](https://linear.app/exo-red/issue/EXO-46) BIND admission; [EXO-73](https://linear.app/exo-red/issue/EXO-73) recursion driver; [EXO-50](https://linear.app/exo-red/issue/EXO-50) coordination.

## 0. Hypothesis, counter-hypothesis, and existing ruling

**Question:** can Ascent's compile-time rule dependency, SCC/stratification, bound/free matching and abstract-relation semantics be compiled into **already available**, certified RelAlg polyfill templates that produce R2IL or direct prebound DR SIMD? Then MeTTa and MORK are front-end aliases, not owners of a second execution model.

**Counter-hypothesis (about unmodified Ascent codegen only):** BYODS substitutes a *relation provider*, but Ascent's current generated join loops and head writes still iterate over individual bindings. A DR-backed `RelIndexRead` that yields positions one at a time is not itself a DR-native mask fold. **This does not reject a compiler rewrite** from Ascent's *logical* tuple operations to shared index/address projections, using mask intersections only for search/filtering.

**Prior ruling and explicit correction:** [2026-10-21 evaluation](../board/entries/2026-10-21-eval-egglog-datafrog-ascent.md) ruled **SKIP for Ascent's unmodified tuple-oriented runtime** and withdrew tuple-native oracle roles. The operator's counterproposal distinguishes **tuple by index** from **tuple search**: values `A[i]` and `B[i]` form a virtual tuple at known `i`; mask intersections identify candidate `i` when searching. If a witnessed relationship index *is the join*, lookup needs neither a separate relational join nor unconditional intersection. Rejection of owned tuple materialization must **not** be extrapolated to rejection of any of these virtual forms. This remains a cold compiler/lowering research question, not a claim that BYODS already generates DR-native code.

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
    -> Count/Any/Keep/etc.; logical tuple = witnessed index + borrowed lane template; intersect selection masks only when needed; no owned row
OFFLINE PROOF [separate owners]:
  egglog congruence, proof; Ruler/Enumo law candidates; eggcc effect extraction
```

Do not manufacture an extra compiler representation when `crepe_plan!`, contract set_algebra, relalg and Quack already cover the operation. No hidden MORK/PathMap execution dependency.

## 1. Stages

| Stage | Narrow task | Falsifier / result gate | Status |
|---|---|---|---|
| **A0 source/ownership audit** | Extract BYODS formal model and compare Ascent source to D-SRAM, D-MTP, existing crepe-plan-probe and D-RU. Record conflict with old hot-path SKIP. | If no unique cold compiler feature is missing, PARK. | This docs PR |
| **A1 generated-code microscope** | Inspect fork `#[ds]` expansion and one `R(x,y), S(y,z)` rule. Classify (a) *logical* tuple bindings, (b) per-binding loop and head-emission code in current Ascent, (c) which logical joins could lower to (i) indexed borrowed reads where the index is the join, or (ii) mask-intersection search over a shared coordinate. Count index and tuple allocations in existing code only. | If default generated loops enumerate each match, **BYODS-only is NOT a native fold**. That does **not** falsify an independent cold rewrite into DR masks. No runtime adapter unless separately authorized. | Queued, lab-only authorization required |
| **A2 mathematical relation contract** | Specify finite-focus `D, inj, γ` (abstract lattice/domain, inject tuple logically, concretize without actually enumerating it) over sealed DR coordinates. Validate monotone `γ`, bottom, join laws and delta/total/new fixed-point semantics. | 257th out-of-focus slot, stale seal/ABA, non-monotone delta, changed coordinate identity or missing completeness must refuse. | Queued, docs/math first |
| **A3 *cold* static rule planning** | Compare SCC, negation strata, lattice preconditions, binding propagation, join-order metadata with existing `crepe_plan!`, `set_rules` and Quack on identical fixed rules. No second AST or executor. | If metadata gives no new lawful plan shape or only a second spelling, PARK. Test negative cycle, repeated variable, disconnected rule and nested expression refusal. | Queued, requires EXO-87 coordination |
| **A4 DR-native *virtual tuple* lowering** | Three distinct lowering shapes: (A) given `i`, read `LeftLane[i]` / `RightLane[i]` as an addressed virtual tuple; (B) unknown `i`, derive `eligible = mask_left AND mask_right`, then fold/projection; (C) a sealed global-to-local relationship index provides witnessed left/right offsets, so the **index is the join** and no second join is run. Compare direct RelAlg/DR, test both functional and one-to-many facts. Preserve `ADMISSION=B`. | Wrong support coordinates, stale/mismatched basis, forbidden gather, owned-row allocations or wrong observable answers kill the rewrite. Multi-valued `y` requires factorized combinations or counted fan-out; an `AND` plane alone cannot emit the bag or `DISTINCT(x,z)`. An Ascent default per-result `next()` does not invalidate this alternate lowering. | Gated on A1/A2/A3 and EXO-46 |
| **A5 equality and recursion boundary** | Counterexample: `a ≡ b` should imply `f(a) ≡ f(b)` under congruence; an `eqrel` provider alone cannot supply rebuilding/proofs. Separately check a ≤256-slot cyclic/diamond closure by *independent mask algorithm* against frontier folds. | No claim Ascent = egglog e-graph; no claim recursion is production (EXO-73). Refuse cross-version and out-of-scope closures. | Optional follow-up |
| **A6 reuse economics** | Compare direct existing relalg/DR compile+execute to offline Ascent-derived schedule plus same hot recipe. Test shared-core 256 templates ×4096 focus frames, nonsharing control, cold cache hit/miss/staleness and negative syntax. | Net win only with *same workload and semantics*, accounting separately for cold compile, proof/verify, bind/attest, hot fold, outputs and memory. Avoid counting reduced tuple volume as a fold speedup. | Gated on real A3/A4 benefit |

## 2. Physical interpretation: tuple by construction, tuple by search, join as index

The operator's PowerShell shorthand `$hashA[$i].Value:$hashB[$i].Value` means the **logical composite view at a given address**, not allocation of an object. Three separate execution situations must not be conflated:

```text
A. KNOWN ADDRESS (no search)
  i is given and both lanes are complete at i:
  logical_tuple(i) := LeftLane[i] : RightLane[i]       // borrowed read only

B. SEARCH (index not known)
  selected_i := PredicateMaskLeft & PredicateMaskRight // same witnessed axis
  fold selected_i over the borrowed original lanes
  Any(selected_i)   // existence, not tuple emission

C. JOIN ALREADY ENCODED IN THE INDEX
  binding[i] := { left_position, right_position }     // compact address map, not value tuple
  virtual_tuple(i) := LeftLane[binding[i].left] : RightLane[binding[i].right]
  // No additional join operation or AND is inherent to the read.
  // Only apply validity/presence/filter masks not guaranteed by the binding witness.
```

Case C can collapse to case A when both sides are co-addressed at the same slot. A binding is sufficient only if it attests which keys were equated, completeness, source versions, coordinate bases, authorization and the claimed cardinality. A pointer map that encodes a value-derived foreign-row read not permitted by `ADMISSION=B` must refuse; an index is *not* a loophole for prohibited gathers. Counting preparation and attestation costs is mandatory.

No `Vec<(x,z)>`, `BTreeSet`, result-object materialization or per-result sorting is required. The logical tuple is a **template over borrowed values**, and the relationship may be precompiled into its index.

For case B, when `L[y]` and `R[y]` are presence masks on a common attested coordinate, `L & R` identifies eligible `y` slots. `PopCount(L & R)` counts matching **slots**, not necessarily joined logical result tuples or bag multiplicity.

The equation by itself is **only a conjunction/existence mask**. If `Left(x,y)` or `Right(y,z)` has multiple values per `y`, the resulting logical bag contains a product of per-`y` fan-outs: `Σ_y left_count[y] × right_count[y]` for bag counting. If projected `DISTINCT(x,z)` is requested, de-duplicate pairs across `y` using an explicitly costed/falsified factorized representation or refuse the unsupported shape. Do not promise an `AND` mask is a complete general join.

**Specific compiler hypothesis:** convert Ascent's logical tuple-at-a-time *rule IR* into a predicate/mask-template graph **before** its current iterator-and-head-emitter lowering. The present BYODS runtime remains ineligible, but it does not prove this compiler route impossible. Test a functionally unique `y` binding first, then non-functional and duplicate paths as mandatory counterexamples.

### DR/Inception is a logical perspective, not a new tuple type

The repository already has the required **conceptual separation**: canonical SoA values, an independently witnessed RelationshipMap / focus-index binding, and a thought-specific mask or delta. The expression `Thought(t)[T_a[i] : T_b[i]]` is the existing zero-copy logical-tuple idiom, not a request to allocate a tuple object. `WORLD × ANGLE × THOUGHT` is **navigation terminology**, not a concrete runtime `Inception` struct, tensor or new register (D-RPR §11.1 explicitly warns against that reading).

**Implementation grade matters:** `Focus`, `CsrPred`, DR tiles and `ThoughtDelta` exist as measured/probe shapes in `mork-pathmap-probe`. The canonical RelationshipMap/MapPalette/ThoughtMapPaletteBinding store, lawful production global→local binding, and recursive child-angle driver remain proposed/gated. Do not cite a general nested Inception mechanism as already shipped.

The prospective **superpower** is the ability to encode *reasoning behavior* as a reusable, typed program over those already-owned carriers, with separate semantic proof, source witness and commit authority:
- **KNOWN i:** `A[i]:B[i]` is a virtual tuple/address projection; no search or mask intersection.
- **SEARCH:** intersect predicate/presence masks over a single attested domain to select qualifying `i`; the result remains a mask.
- **INDEX-AS-JOIN:** a legally prepared relationship map is itself the factored join mapping. Read the underlying lanes through that map; do not repeat an equijoin or add redundant intersections.
- **INFERENCE TRANSITION:** a rule-licensed, typed fold maps bound input masks to derived masks/frontiers; a thought delta records hypotheses while a proof/receipt records *why* the transformation is justified. For recursion, iterate the frontier only under a lawful driver and explicit convergence/complete-scope contract.

This is **not** an automatic theorem or proof: a mask encodes selected coordinates, and a register encodes a bounded relation, but a semantic inference needs its original premises, law/plan identity, snapshot scope, truth algebra and independent proof. Repeated proofs do not create independent CE64 evidence. A one-to-many relationship still needs factorized bag semantics or an explicit refusal.

**Ascent integration test:** preserve the Ascent rule's logic while replacing its physical tuple enumeration with one of the already permitted `KNOWN i`, `SEARCH` or `INDEX-AS-JOIN` shapes, plus an external/frontier schedule where needed. Do not build a separate Ascent tuple type, pretend BYODS auto-vectorizes, or use a precomputed pointer map to bypass `ADMISSION=B`.

## 3. Cold versus hot accounting: now recorded as scoped P0 exception

**Operator observation, 2026-10-22:** an offline stage in the hundreds of thousands of microseconds is not comparable to a single nanosecond-scale fold; the MeTTa/MORK reference workloads can take many seconds. These are **not paired identical workloads**, so no acceleration is claimed. The same-day `CLAUDE.md` P0 amendment now explicitly permits reporting cold compiler/oracle costs in **µs, labelled 'reported, not gates'**.

Mathematical break-even, for the **same program and output semantics**, with one compile reused `N` times:

`T_compiled_total(N) = T_cold + N × (T_bind + T_execute + T_verify)`
`T_baseline_total(N) = N × T_interpreted`

Admit an **economic hypothesis** only when `T_baseline_total(N) > T_compiled_total(N)` for measured matched workloads. Also show cold-only, warm-only, first-call and nonsharing arms; cache invalidation by proof-law version, source generation, coordinate basis and compiler version.

**Already resolved P0 timing scope:** the existing 2026-10-22 `CLAUDE.md` **SCOPED EXCEPTION** allows cold compilation/proof and differential-oracle costs to be reported in **µs as informational, not gates**. Hot encode/lower/execute/terminals remain strictly constrained; no millisecond in-fold result is licensed. This PR does not edit P0 or claim a new exception. Only matched end-to-end economics can establish an amortization win, separately from hot-path acceptance.

## 4. Required independent semantics/refusals

- **Set/Distinct versus Bag:** count duplicate derivations and foreign fan-out separately. Boolean OR path existence is not a bag count and XOR is not OR. Repeatedly derived value does not imply new NARS evidence.
- **Bool2 versus SQL K3:** valid FALSE is not invalid NULL or an independent UNKNOWN. No unsafe negation or complement rewrite.
- **E-class versus entity identity:** an egglog union is mathematical congruence, never an authority to merge external identities or GUIDs.
- **Snapshots and local coordinate binding:** the same ordinal in two contexts need not denote the same object. Check axis/source/codebook/permission generation before reuse.
- **Pure versus effectful extraction:** proof of equality does not commit a Lance write, authorize a read, or allow moving effects across version barriers.
- **Logical tuple is not an owned tuple:** a logical pair can be a *direct borrowed read at a known index*, an *AND-mask search* selecting indices, or a *witnessed relationship index that already is the join*. Compare results using independent mask/reference folds; no owned-pair vector, BTreeSet answer, or hidden row materialization.
- **No runtime Ascent, MORK or datafrog fallback:** if template unsupported, typed refusal or separately authorized frontend behavior, never silently invoke another evaluator.

## 5. Concrete first lab question

Generate a *static*, rule-bounded reachability fixture with a shared intermediate coordinate:

`Reach(x,z) :- Left(x,y), Right(y,z).`

- Baseline A: existing Quack/RelAlg co-addressed direct bitmask fold.
- Candidate B: Ascent-generated **plan metadata only**, translated into the same mask fold, without executing Ascent runtime.
- Diagnostic C: source-audited `#[ds]` generated code illustrating whether BYODS retains tuple enumeration. Do not use C as a runtime contender if it violates P0.
- Negative: same `y` ordinal from another seal; 257th focus; two distinct intermediate paths (OR true/XOR false); duplicate bag answers; cyclic recursion candidate not admitted; no-sharing workload.
- If B is byte-identical and no cheaper or more broadly legal than A, **stop**, keep Ascent as literature/reference rather than introducing optimizer machinery.
- If B exposes a genuinely missing legal transformation, propose **one** owner-scoped lab code PR, with a separate build/test plan, not an architecture rewrite.

## 6. Decision requested in EXO-50

1. **Acknowledge the existing 2026-10-22 P0 cold-cost exception**: cold/opaque oracle costs reported in µs, not gate results. Preserve separate hot-fold admission; no further P0 amendment proposed.
2. Authorize **A1 source/codegen audit plus a docs-level three-case lowering sketch (direct index / AND-mask search / index-as-join)**, not a BYODS runtime adapter or a new Ascent/RelAlg executor.
3. Assign any A3 translator work to existing EXO-87 / D-SRAM and EXO-86 / D-RA owners, after their acknowledgement. No new plan family issue or D-id unless a unique deficiency survives A1–A3.
4. Keep egglog proof ownership and #1690–#1704 DR Math Foundry independent; no duplicate e-graph or proof verifier.

**Outcome sought:** either precise evidence that Ascent contributes a reusable cold optimization to native DR folds, or a documented reason to leave the existing RelAlg/DR compiler alone.
