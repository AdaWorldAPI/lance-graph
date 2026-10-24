# D-SAG-6 — Rust Codegraph → Mathematics → Logic → Tracert

**Date:** 2026-10-24. **Status:** PROPOSAL / documentation and falsifier preregistration only. **Implementation:** NOT STARTED. No Rust, Cargo, ABI, policy, storage, proof checker, source rewrite, CI or performance changes are authorized by this PR.
**Coordination:** [Linear EXO-50](https://linear.app/exo-red/issue/EXO-50/agent-message-bus-gpt-xgrok-ccsession-id). **Execution brief:** [D-SAG-6 prompt](../prompts/d-sag-6-rust-codegraph-logic-tracert-claude-code.md).
**Parent contracts:** [SAG capstone alignment](sag-capstone-alignment-v1.md), [D-SAG-2 Ruff/ONA](software-awareness-graph-v2-ruff-ona-closed-loop.md), [DR mathematical compiler](dr-mathematical-compiler-v1.md), [Ascent/BYODS cold fold](ascent-byods-dr-fold-polyfill-v1.md), [D-NFR native representation](d-native-repr-contract-v1.md), [D-NFC native-first correction](d-native-first-dependency-correction-v1.md).

## 0. Thesis and explicit authority boundaries

The goal is **Codegraph → Mathematics → Logic → Tracert**, not a second Rust compiler, AST interpreter, query executor, method registry or rewrite authority.

- **Codegraph** is a source/revision-aware, typed, directed relational *view* of Rust items and operations. Source symbols, MIR places, effects, dataflow edges and callsite identities are distinct.
- **Mathematics** expresses admitted equivalence laws or candidate rewrites in the existing ClassView/RelAlg/set_rules/R2IL vocabulary. A discovered pattern is not a verified law.
- **Logic** derives reachability, dependency, alias/capture eligibility, preconditions and closed-world findings only under declared coverage. Ascent/Datafrog/Prolog are comparative **cold analysis or oracle** arms; not additional production runtimes. Egglog performs guarded equality exploration, not source authority.
- **Tracert** is a typed provenance path linking source → semantic witness → logic rule → law admission → proof/extraction → native lowering → differential/target result. A rendered egglog proof or replay capsule is not an independent proof checker.
- **Extraction** may use eggcc-style effect-ordered candidate selection offline. It must not reorder Rust drops, mutations, panics, volatile/unsafe behavior or externally visible effects without proved legality.
- **Output** is a *proposed*, reviewable Rust source patch or existing native recipe, never an automatic hot write. MLIR is an optional export/lowering arm, not the semantic source of truth.

**SAG ownership:** D-SAG-1 software-awareness code was removed/reset; D-SAG-2 is a research plan, not a shipped generic graph. The SAG capstone reserves one inert **ProposedMethod** shape and separate proposal/verification/approval/authorization/application/re-observation stages. Do not revive removed glove-parity code or create a second ProposedMethod type.

**Ruff disambiguation:** the relevant [AdaWorldAPI/ruff](https://github.com/AdaWorldAPI/ruff) is the fork with polyglot SPO harvest/ModelGraph and codegen arms. It may independently corroborate supported **source structure**, not Rust MIR semantics, executable equivalence or machine-code performance. Its existing ModelGraph → addressed SoA plan is [lance-graph as ModelGraph](lance-graph-as-the-modelgraph-v1.md). Preserve the stated dependency direction: Ruff crates do **not** import lance-graph; an external projection/byte boundary connects them. Do not mistake this fork for a complete Rust compiler frontend merely because upstream Ruff is written in Rust.

## 1. Orthogonal views, not a linear tower

~~~text
Rust source/crate/feature/target revision
    |                     |
    v                     v
rust-analyzer HIR      rustc MIR / Charon LLBC (candidate; pin and test)
    \                    /
     \-- typed read-only source/semantic binding --/
                         |
                   Codegraph (DR focus / ClassView / SoA)
                   {defines, calls, captures, trait_impl,
                    def_use, cfg, borrow, drop, effect, reads}
                    /             |             \
                   /              |              \
    Ascent/Datafrog/Datalog    RelAlg/egglog     Trace receipts
       rule queries           guarded rewrites    evidence lineage
                   \              |              /
                    \----- cold admission -------/
                         |
              eggcc-style lawful extraction
                         |
            inert ProposedMethod / candidate recipe
                      /         \
               Rust AST edits   optional MLIR dialect export
               + rustc/tests    + target-specific differential
                         |
                  independent replay
~~~

MIR is an **input semantic view**; MLIR may be an optional **downstream representation**. Datalog and egglog are parallel reasoning views over shared native facts. Tracert follows all of them; it is not a single later pass. None is a mandatory hot dependency.

## 2. Minimal graph contract (conceptual, not new ABI)

**Node identities:** RepoIdentity, CargoCrateInstance (revision, features, target triple, rustc build), SourceDefId, SourceCallsiteId, MIRBodyId, BasicBlockId, PlaceId, ValueVersionId, TraitImplId, Method/ClassId, RuleId, ProofId and RecipeId. An identical label or source line number across builds does not establish identity.

**Edge families:**
- DEFINES / CALLS / DISPATCH_POSSIBILITY / IMPLEMENTS / CAPTURES / CONSTRAINS_TRAIT
- CFG_NEXT / DEF_USE / BORROWS / REBORROWS / MOVES / DROPS / ALIASES_POSSIBLY
- DEPENDS_ON / READS / WRITES / MAY_PANIC / HAS_EFFECT / LOWERS_TO
- DERIVED_BY / USES_LAW / PROOF_FOR / VALIDATED_BY / REFUSED_FOR / REPLAYED_AT

**Evidence grades:** observed source, compiler-derived, logically derived, hypothesis, tested, independently verified; use SAG aligned Witnessed/Refuted/Undetermined/Conflicted plus separate stale/unsupported markers. These are not interchangeable. A call edge (possibly dynamic) does not imply an observed call. An unexamined region is UNKNOWN, not FALSE.

**Physical interpretation:** DR64² / DR256² are bounded local relationship views; proposed DR64³ and any DRⁿ are typed, factorized logical axes, never compulsory dense allocations. The global identity and original source remain in their owners; DR ordinals are local witnessed coordinates. SoA holds canonical values. Do not create owned pairs/row lists for hot folds or resurrect withdrawn method tables. Keep masks, bag multiplicity, SQL K3 validity and epistemic evidence distinct.

**Higher order:** closure/function values require signature, captures, trait-bound dispatch and monomorphization/dynamic resolution status. Trait method bodies, callsites, and possible callees must not collapse into one identity. MIR lifetimes and Rust borrow checking take precedence over labels or LLM-inferred aliasing.

## 3. Rule engines, extraction and proof separation

| Tool | Candidate responsibility | NOT claimed |
|---|---|---|
| rust-analyzer | source-anchored syntax/HIR, symbol/type/reference maps and source edits | target-accurate MIR execution proof |
| rustc MIR / Charon | CFG, places, borrows, drops, operations and semantic witness under pinned toolchain | lossless reproduction of all Rust source formatting, comprehensive unsafe semantics |
| AdaWorldAPI/ruff | independent supported SPO structure/codegen source crosswalk | Rust binary equivalence, mandatory Cargo dependency of lance-graph |
| Datalog / Ascent | cold declarative relations, SCC, reachability, guarded selection, rule metadata | automatic DR-native folds from stock BYODS or arbitrary negation |
| Datafrog | independent fixed-point differential/recursion oracle | production graph executor or semantic proof |
| Prolog | optional goal-directed exploration / diagnostics | independent oracle when derived from same evidence |
| egglog | typed congruence and equational candidate exploration | independent validation of underlying axioms |
| eggcc-style | choose candidate extraction with explicit effect constraints | proof that a transform preserves all Rust semantics |
| MLIR | optional vector/LLVM/GPU lowering experiment | replacement for MIR ownership/borrow truth |

Direct native execution remains Quack-bound R2IL router → supported direct/composed ndarray SIMD or mask-risc; ClassId route chosen once, not per fold. Egglog and Datalog do not execute in the hot path.

**Three independent warrants:** (1) sourced method/graph facts, (2) admissible mathematical law with independently checked proof where required, (3) current binding, authority and executable target parity. A proof of algebraic equality cannot authorize a source edit or an external effect.

## 4. Concrete first fixture (read-only)

Source shape: a Rust iterator count of values satisfying a pure unsigned scalar predicate, and a second fixture that counts a set of Boolean DR mask positions.

Candidate (not asserted as a universally valid source rewrite):

~~~text
count(filter(P, X))  <candidate>  popcount(mask(P, X))
~~~

For the narrow pure/read-only, exactly-once and compatible-integer case, compare a Rust scalar oracle, an explicitly prepared ndarray mask+popcount path, and an independently evaluated relational query. Keep two source identities: original source and proposed rewrite; neither replaces the other during the proof.

Capture and disposition: exact type/trait implementation, closure capture, predicate evaluation count, panic behavior, overflow/range, provenance of the mask, snapshot and target. The rule engine may find this candidate even when the legal response is REFUSE. Negative examples are essential.

## 5. Falsification matrix, preregistered

| Gate | Adversarial case | Required behavior |
|---|---|---|
| F01 | same-named symbol across crates/feature sets/revisions | distinct identities; no cross-build evidence merge |
| F02 | generic Fn closure captures mutable state | no pure-count fusion and no predicate-call elision |
| F03 | predicate can panic / integer overflow / eager evaluation differs | refuse unproven reordering, preserve observable evaluation |
| F04 | trait dispatch unresolved or changed impl | possible-call edge not treated as confirmed callee |
| F05 | borrow/move/drop/unsafe/volatile/atomic ordering | effects remain ordered; unsupported semantics typed refusal |
| F06 | a missing edge in incomplete MIR/Ruff coverage | UNKNOWN not FALSE; no closed-world claim |
| F07 | DR 64/256 same ordinal in different frames/scopes | stale/foreign coordinates refused |
| F08 | Boolean DISTINCT count vs SQL K3 NULL / BAG duplicates | distinct typed contracts; no false equivalence |
| F09 | egglog equality derived under wrong law assumptions, forged proof text, wrong ABI | refused; external independent checker required before certified status |
| F10 | cold optimized path compared against unfair baseline / identical source oracle | no performance win or correctness claim |
| F11 | changed source SHA while retaining old recipe/callsite ids | invalidation or independently established rebinding |
| F12 | MLIR lower is successful but changes signed zero, NaN, FMA or rounding | parity failure; no source rewrite |
| F13 | graph candidate is applied without operator/source approval | forbidden; ProposedMethod inert |
| F14 | arbitrary generic method invokes dynamic external side effect | no mathematical reduction and no OGAR bypass |
| F15 | fixture passes only with synthetic labels or fabricated AST semantics | evidence grade = unsupported/undetermined; no admission |

For each implemented gate: preregister counterexample, independent oracle, disable-red-run, fixture/compiler revision and reproducible result. Zero discovered candidates is a permissible result; report it honestly.

## 6. Staged handoff (this PR is S0 only)

| D-id | Deliverable | Gate | Owner/status |
|---|---|---|---|
| **D-SAG-6-0** | this additive architecture + prompt + board reference | all scope/authority statuses explicit | docs PR only |
| **D-SAG-6-1** | pinned Rust HIR/MIR/Charon and Ruff capability audit | real source fixtures, unsupported inventory, no new runtime | future read-only audit |
| **D-SAG-6-2** | minimal read-only native Codegraph projection in lab | exact source identity; F01/F04/F06/F07/F11 | separate review/approval |
| **D-SAG-6-3** | logic candidate + independent Datafrog/Ascent oracle | F02/F03/F08/F10/F15 | separate review/approval |
| **D-SAG-6-4** | egglog law, effect-aware extraction, complete traceroute | F05/F09/F13/F14, external proof gap named | separate review/approval |
| **D-SAG-6-5** | opt-in Rust patch/codegen vs optional MLIR cross-check | F03/F05/F12/F13, compilation + target+oracle parity | separate review/approval |

**Ownership conflict stop:** Check D-SAG capstone, D-RFT/Math Foundry and D-DRMC owners before introducing types, changing proof references or producing source patches. EXO-50 is the communication bus, not a merge/approval authority. No automatic code changes, authorization decisions or writes to source/SoA in S0.

## 7. Reference ledger (entry points to inspect; status is not proof)

- [lance-graph #1523](https://github.com/AdaWorldAPI/lance-graph/pull/1523): SAG capstone; D-SAG-1 original method graph reset/withdrawn.
- [AdaWorldAPI/ruff](https://github.com/AdaWorldAPI/ruff): polyglot graph extraction + Rust codegen arms, structural observer only.
- [lance-graph #1693](https://github.com/AdaWorldAPI/lance-graph/pull/1693), [#1694](https://github.com/AdaWorldAPI/lance-graph/pull/1694): replay capsule and guarded math catalogue; lab only, not independent theorem prover.
- [lance-graph #1740](https://github.com/AdaWorldAPI/lance-graph/pull/1740): frame-reused DR recipe lab, independent checker still missing.
- [lance-graph #1752](https://github.com/AdaWorldAPI/lance-graph/pull/1752), [#1753](https://github.com/AdaWorldAPI/lance-graph/pull/1753), [#1755](https://github.com/AdaWorldAPI/lance-graph/pull/1755): native router, binding status and borrowed SoA.
- [lance-graph #1759](https://github.com/AdaWorldAPI/lance-graph/pull/1759): parallel board catch-up; do not amend/reinterpret its status changes.
- External subjects for pinned capability audit: rust-analyzer HIR, rustc MIR, [Charon](https://github.com/AeneasVerif/charon), [Ascent](https://github.com/AdaWorldAPI/ascent), Datafrog, egglog, [eggcc paper](https://rtjoa.com/papers/oopsla26-eggcc.pdf), [MLIR](https://mlir.llvm.org/).

**D-SAG-6-0 is a design/coordination artifact, NOT a functional codegraph, MIR/MLIR bridge, verified rewrite, executed benchmark, or authorization to merge.**
