# OpenCog / MeTTa / Hyperon -> lance-graph: semantic projection and falsification map

**Date:** 2026-10-12
**Status:** SOURCE-INVENTORY / WORKING-MODEL / DOCUMENTATION ONLY. No evaluator built; no oracle run; no equivalence established.
**Sibling of:** [draft PR #1497, PLN x Shannon attention](https://github.com/AdaWorldAPI/lance-graph/pull/1497). Independent branch and scope, based on `main`. Either sibling may merge first; re-run the board entries index generator on the remaining branch.
**Read by:** sessions evaluating metagraph representations, MeTTa syntax/semantics, typed pattern matching, OGAR-loco/R2IL folding, metagraph morphisms, AtomSpace frames, pattern mining, self-modifying inference control.
**Existing LG anchors:** [frontend parity witness](../.claude/plans/frontend-parity-witness-v1.md) (SQL/Gremlin/Quack, including its correction notes), [SQL/R2IL polyfill](../.claude/plans/sql-polyfill-r2il-v1.md), [NAL/R2IL polyfill](../.claude/plans/nal-polyfill-v1.md), [CE64 plateau](../.claude/knowledge/ce64-reasoning-substrate-plateau.md), [D-HL](../.claude/plans/ce64-hypothesis-layers-v1.md), [V3 entry point](../.claude/v3/README.md).

## Precise question

> Which **specified, executable subset** of Hyperon MeTTa atom/pattern/rewrite semantics and Goertzel's typed-metagraph fold/unfold laws can be represented with the current classid/NodeGuid + V3 ClassView/hotplug, Quack query, OGAR loco/R2IL procedures and mask-risc executor, preserving query answers, multiplicity, rewrite state, binding scope and refusal behavior? Where do existing primitives or the one-population execution model demonstrably fail?

Do NOT conflate three different subjects:
1. **OpenCog Classic AtomSpace**: its C++ metagraph representation, mutable Values and Frames, separate from new Hyperon.
2. **Hyperon/MeTTa runtime**: actual Rust `Atom::{Symbol,Variable,Grounded,Expression}`, matcher, spaces, eval and executable fixtures.
3. **Metagraph morphism theory**: catamorphisms/anamorphisms/histomorphisms/etc. on typed metagraphs, a formal proposal, not evidence that existing `mask-risc::Terminal` implements them.

A fourth related subject, **PLN/attention/miner**, is owned by sibling #1497. Shared fixtures are allowed; this PR asks about *semantic mapping and lowering*, not scheduler superiority or entropy as a performance claim.

## Source inventory (verified 2026-10-12; head SHAs are **snapshots** only)

| Origin | Revision | Direct source / oracle material | Reading / qualification |
|---|---|---|---|
| [Goertzel et al., OpenCog Hyperon (2023)](https://arxiv.org/abs/2310.18318) | arXiv:2310.18318 | Framework overview: metagraphs, MeTTa, cognitive synergy, inference and possible fold/unfold algorithms | Architectural proposal, not a validated LG lowering |
| [Goertzel, Folding and Unfolding on Metagraphs (2020)](https://arxiv.org/abs/2012.01759) | arXiv:2012.01759 | Directed typed metagraphs, constructors, metapaths, morphism zoo | Formal laws / definitions to extract into a bounded test case, not "fold = sum" |
| [trueagi-io/hyperon-experimental](https://github.com/trueagi-io/hyperon-experimental) | `3f76dc460da6961f57f69f6c3e550c59c74ada83` | [four atom kinds](https://github.com/trueagi-io/hyperon-experimental/blob/main/hyperon-atom/src/lib.rs); [matcher/bindings](https://github.com/trueagi-io/hyperon-experimental/blob/main/hyperon-atom/src/matcher.rs); [unit fixtures](https://github.com/trueagi-io/hyperon-experimental/tree/main/python/tests/scripts); [README/runtime](https://github.com/trueagi-io/hyperon-experimental/blob/main/README.md); [LICENSE](https://github.com/trueagi-io/hyperon-experimental/blob/main/LICENSE) | Executable Hyperon Rust/MeTTa **oracle candidate**. MIT according to LICENSE. Pin interpreter mode and dependencies before running. |
| [MeTTa specification](https://wiki.opencog.org/w/File:MeTTa_Specification.pdf) + [Hyperon documentation](https://trueagi-io.github.io/hyperon-experimental/) | version not pinned | Language and runtime semantics | Spec-vs-implementation differences explicitly recorded as findings |
| [trueagi-io/PeTTa](https://github.com/trueagi-io/PeTTa) | `1d19a1da1ad8ba6327d660abd65433a0e341c23f` | Alternative MeTTa runtime; primary execution dependency for modern [PLN](https://github.com/trueagi-io/PLN) and miner | Do NOT swap PeTTa and native Hyperon and call it one oracle. Diff them where relevant. |
| [opencog/atomspace](https://github.com/opencog/atomspace) | `c8d633bf272b838c2132c21b7c8ddf169a18dd22` | [README](https://github.com/opencog/atomspace/blob/master/README.md): typed hypergraph, executable graphs, pattern engine, Values, queries-as-graphs, Frames | Classic C++ behavior, separate from Hyperon; AGPL with OpenCog linking exception in LICENSE, check file headers |
| [trueagi-io/hyperon-miner](https://github.com/trueagi-io/hyperon-miner) | `e955479bca1603597529d175cc22ac1583c67e15` | [README](https://github.com/trueagi-io/hyperon-miner/blob/master/README.md): frequent nested patterns, variable instantiation, conjunction expansion, optional surprisingness score | Concrete workload and oracle for subgraph mining; not proof of Shannon EIG |
| [opencog/miner](https://github.com/opencog/miner) | `444eb6474c304775c457410592fa47a51455e36c` | Classic pattern miner + URE | Historical comparative semantics; dependency/runtime/license audit first |
| [trueagi-io/metta-examples](https://github.com/trueagi-io/metta-examples) | `799ad9dbf92987cabeede6d1eca02ac7f8abb0ad` | Additional MeTTa usage examples | README warns examples may be unmaintained; use pinned core regression scripts first |

**Good initial oracle fixtures, from the upstream Hyperon repo at the pinned revision:**
- [`b4_nondeterm.metta`](https://github.com/trueagi-io/hyperon-experimental/blob/main/python/tests/scripts/b4_nondeterm.metta): multiple matches, `superpose`/`collapse`, unreduced terms and a 2x2 Cartesian branch yielding four answers.
- [`d3_deptypes.metta`](https://github.com/trueagi-io/hyperon-experimental/blob/main/python/tests/scripts/d3_deptypes.metta): typed vectors, dependent lengths, accepted/rejected typing and subtle invalid-length behavior.
- [`e1_kb_write.metta`](https://github.com/trueagi-io/hyperon-experimental/blob/main/python/tests/scripts/e1_kb_write.metta): `new-space`, `add-atom`, `match`, state isolation.
- [`c1_grounded_basic.metta`](https://github.com/trueagi-io/hyperon-experimental/blob/main/python/tests/scripts/c1_grounded_basic.metta): grounded arithmetic, recursively defined functions, type errors, unreduced symbolic expressions.

## Translation matrix (hypotheses, NOT implementations)

| Hyperon/AtomSpace construct | Candidate LG surface | Category | What must be proven / refuse |
|---|---|---|---|
| Symbol atom / name equality | OGAR vocabulary label -> immutable `classid x NodeGuid`, binder with declared context | **CONDITIONAL** | Source-level symbol equality, namespaces, aliases, collisions and renames must not silently change identity. Labels never hot addresses. |
| Variable atom, repeated bindings | bound structural plan + Quack `Filter` and joins | **SUBSET** | Repeat-variable unification, variable alpha-renaming and scope; unsupported occurs-check or recursive unification **refuses**. |
| Ordered nested expression atom | typed expression rows/child ordinals + structural relation or edge population | **OPEN** | Preserve order, arity, nested references, sharing, equality, roundtrip, cycle handling; CE64 8-bit S/P/O cannot itself hold arbitrary expression trees. |
| Typed hyperedge, edge targeting an edge | canonical edge-as-entity with relation-to-edge target and per-class ClassView | **OPEN** | Reify higher-order targets without substituting ordinary binary SPO or losing target type. Hashes, identity and reverse mapping must roundtrip. |
| Pattern matching and substitution | Quack predicate/semijoin + planned multi-hop lowering | **SUBSET** | Set versus bag multiplicity, substitution, binding scope, a match with zero results, and pattern output values. Existing multi-hop/fan-in limits from D-FPW remain. |
| MeTTa equation / rewrite | procedure over bound representation, with explicit pure rewrite result | **GAP** | `(= ...)` is not reducible to a mere mask fold; match, substitute, evaluate and generate NEW terms. No silent scalar fallback claiming bulk parity. |
| Multiple results / `superpose` / `collapse` | result populations; explicit bag/set and unordered-result semantics | **GAP** | `(pair (bin) (bin))` must yield four bindings; `collapse` order treated per source; empty result distinct from unreduced symbolic term. |
| `Grounded` atoms and executable functions | typed host-bound functions with an explicit effect contract | **BOUNDARY** | Pure math can use existing lane ops when present; callbacks, state, I/O and arbitrary Rust values are **not** silently translated into R2IL. |
| MeTTa dependent types | declared ClassView and typed bind/refusal | **GAP** | Dependent type reduction and source error outcomes; field masks and class declarations alone do not imply dependent typing. |
| Named Spaces, `add-atom` and mutation | scoped V3 mailbox/Alpha state, append and replay after validation | **OPEN** | Isolation, query visibility, rollback, transactional and error semantics; no write bypasses owner/gating contract. |
| Classic AtomSpace mutable Values / Frames | immutable identities plus versioned Value readings and hypothesis deltas | **ANALOGY** | Frames and Alpha layers are NOT synonymous; check branch/revert/merge, tombstones and shadowing individually. |
| Metagraph catamorphism (finite acyclic case) | typed postorder traversal + R2IL fold procedure | **CANDIDATE** | Same *result and traversal law* for a bounded acyclic typed expression forest. Prove associativity/order assumptions; recursion scheme not equal to scalar SUM. |
| Histomorphism, futumorphism, metamorphism, cycles | composed procedures, possible replayed intermediate context | **OPEN / LIKELY GAP** | Need access to intermediate histories, generation and higher-order context; explicitly identify missing primitive or cost, not invent new ISA. |
| Query-as-data / metaprogramming / reflective rewrite | OGAR loco `PROC_DEF` / `PROC_CALL` only after executable semantics are verified | **BOUNDARY** | Storing a rule graph and executing it are separate facts; `PROC_*` bodies and generator remain gated in D-SQLP. |
| Hyperon miner support counts, conjunction, surprisingness | Quack group counts, intersections, masked folds; scoring outside hot loop | **CANDIDATE** | Exact repeated-variable and nested-pattern match counts first; information-theoretic measures belong to #1497. |

**Vocabulary:** VERIFIED-IN-CODE applies to cited existing LG primitives only. SUBSET/CANDIDATE/GAP describe translation *proposals*; no source target parity has been run. Never infer equivalence from a matching word like "fold", "graph", "attention" or "atom".

## Open questions to answer (HY-Q01 .. HY-Q15)

| ID | Question | Minimum evidence and falsifier |
|---|---|---|
| HY-Q01 | What is the exact source atom equality/identity contract for Symbol, Expression, Variable and Grounded in the selected engine? | Pinned source + runnable equal/not-equal examples, collisions and variable renaming; roundtrip through bound ids. |
| HY-Q02 | How do arbitrary nested/typed hyperedges, edges-as-targets and ordered argument slots map without inventing new permanent bytes? | Structural source->LG->source roundtrip; depth and arity limits explicit; preserve shared subexpressions and typed edge references. |
| HY-Q03 | Can repeated-variable matches, substitutions, pattern joins and empty matches lower to existing Quack predicates? | `matcher.rs` oracle fixtures; two-variable and repeated-variable cases; typed refusal for unsupported recursion/joins. |
| HY-Q04 | How are result multiplicity, nondeterministic order, `collapse/superpose`, empty, unreduced and Error distinguished? | Run `b4_nondeterm`, `c1_grounded_basic`; compare normalized multisets *only* where source says order unspecified; do not flatten Error or unreduced into empty. |
| HY-Q05 | How are rewrite rules `=`, evaluation, recursion and normal forms executed without introducing a second engine? | A pure terminating rewrite/substitution fixture first; compare exact outputs and termination/fuel rules; return refusal on missing production primitive. |
| HY-Q06 | What can ClassView type checking represent, and where do MeTTa dependent types require a dedicated proof? | `d3_deptypes` green in source; selected good/bad cases ported; wrong type must not be accepted silently. |
| HY-Q07 | Can a result be a NEW expression rather than only a bitmap/aggregate? Where is result ownership allocated? | Query that outputs a nested term with no preexisting row; track allocations and persistence; no row-sized allocations hidden inside the bulk loop. |
| HY-Q08 | What exactly are the semantics of `new-space`, mutation and multi-space scope versus mailbox/Alpha and Lance versions? | `e1_kb_write` source parity for isolated spaces, read-your-writes, rollback and replay; test classic Frames separately. |
| HY-Q09 | Which typed-metagraph recursion scheme is the smallest genuinely representative case? | Quote precise constructor/law from arXiv:2012.01759; a rooted acyclic graph evaluation with a hand-computed oracle; distinguish catamorphism vs ordinary column SUM. |
| HY-Q10 | What extra state do histomorphism/futumorphism and graph rewriting need that `FoldDialect` cannot express? | Execution-level insufficiency proof and one failed case, plus a lower-cost composition attempt. No new opcode/IR by resemblance. |
| HY-Q11 | Which query-as-graph and reflective/self-modifying operations are data versus execution? | Explicit compile/bind/run lifecycle; after binding no strings/hot labels; mutation requires committed provenance and ownership. |
| HY-Q12 | Which execution semantics are pinned: native Hyperon Rust, `metta-py`, `metta-repl`, or PeTTa? | Pin exact runtime mode, flags and dependencies; record observed divergences; no floating toolchain. |
| HY-Q13 | Does the optimized LG execution preserve outputs before it wins on speed, including output cardinality? | Identical answers, failures and state snapshots before wall time; stage-by-stage adapter/binder/kernel/alloc measurements; no 0.6ns word-fold as total query time. |
| HY-Q14 | What may be used from upstream under what license? | Per-file license and tests/data provenance; Hyperon MIT per root license, AtomSpace classic AGPL with linking exception; no copied source without audit. |
| HY-Q15 | How does Miner or PLN plug into a **shared** source fixture without merging #1497's scheduler question into this PR? | Same structural atoms/labels/IDs and provenance, independent inferencer/scheduler selection; metrics, evidence and returned patterns compared without mixing attention and translation. |

## Falsification-first probe sequence (proposed, not authorized)

**P0: Pin and grade.** Pick one Hyperon runtime and upstream commit, plus a small fixture. Record which behavior is spec, executed, or merely architecture prose. No LGPL/AGPL dependencies pulled into LG as a shortcut.

**P1: Canonical atom projection.** Construct four atom kinds and nested expressions, including repeated variables, two equal symbols, distinct contexts with identical labels, ordered children and a link targeting another link. Translate into a *probe-local* V3-compatible table/address map. Reverse-translate; exact structural equality and stable identity, or a named refusal.

**P2: Queries and multiplicity.** Run `match`, substitution, semijoin and nested patterns against native Hyperon as oracle. `(pair (bin) (bin))` must yield 4 results, an absent match 0, an unresolved symbol remains unreduced where source does. Preserve ordering only when source specifies it. A bitmap giving 2 survivors where 4 substitutions are required is a failure, not an optimization.

**P3: Pure rewrite versus normal-form interpreter.** One typed finite rewrite system, no grounded host side effects; measure both values and termination/fuel. Scope `Grounded` functions and dependent types as typed refusals until justified. If existing procedures cannot construct output expressions, stop, record the insufficiency and do **not** introduce an interpreter ad hoc.

**P4: Mathematical fold witness.** Extract ONE explicit typed-metagraph constructor and catamorphism law from the 2020 paper; implement a scalar reference and test-only postorder/SoA composition on a bounded DAG. Check shared subgraphs, cycles, traversal order and histories separately. A `MaskedSumI32` equality alone is **not** proof of a general metagraph fold.

**P5: Scoped hypothesis / classic Frame witness.** Probe `new-space/add-atom/match` in Hyperon and classic Frame conflict/revert as a **separate** source. Only then compare with Alpha and validated sparse append. Never let an unvalidated hypothesis mutate the base.

**P6: Miner/PLN workload crossover.** Only after P1-P4 preserve semantics, run one miner subpattern-support fixture through exact joins/counts; pass the same bound graph to #1497 attention arms. Keep the scheduler constant in all translation performance runs.

**P7: Cost + refusal ledger.** Compare native runtime vs scalar LG oracle vs bulk mask-risc for *the same expressed subset*: setup/bind/compile, row decode, term constructions, candidate expansion, memory, replay, and wall time. Publish explicit negative cases.

### Kill conditions

- Structural reverse-projection is not injective for the declared supported subset -> **no "lossless translation" claim**.
- Source yields multiple substitutions/new terms while LG returns only a row mask or wrong cardinality -> **no semantics parity**.
- A grounded function needs arbitrary host effects / dynamic callbacks -> **typed boundary**, not "handled by R2IL".
- A metagraph morphism needs intermediate history or recursive expansion and does not compose from current fold calls -> **gap**, not a silent new opcode.
- Empty, unreduced, Error or type-refusal differ -> **failure**, even if counts look close.
- An attention/surprisingness uplift appears only when scheduling changes -> **belongs to #1497**, not evidence for projection.
- Performance win disappears when binder/translation/output materialization is counted -> **no end-to-end win**.

## Architectural guardrails

- The byte layout and ALU/ISA register contracts are **fixed**. MeTTa is an optional front end and source oracle, not a CE64 semantic recoding. Fields and meanings are declared via hotplug/ClassView.
- **One execution spine**: OGAR loco/R2IL procedures -> `FoldDialect` -> mask-risc -> ndarray for the supported subset. No second MeTTa VM inside LG, no second IR, no serialized MeTTa text on hot wires.
- No new OGAR byte without an execution-level insufficiency proof; no `PROC_DEF/PROC_CALL` execution assumption ahead of D-SQLP's real gate.
- Identity is immutable; labels stay at the binder/display edge; graph/hypergraph translation must retain argument order and nested references if claiming parity.
- Preserve per-evidence provenance and separate hypothesis layer ID from evidence source (D-HL-3).
- Every unsupported shape has a named refusal and an upstream counterexample; do not call an approximation "equivalent".
- No production code, new crate or runtime integration authorized. A later implementation needs its own plan, D-ids, benchmark thresholds, permission and board hygiene.

## What this is NOT

Not "MeTTa is just Cypher", not "Metta expressions are CE64 edges", not "hypergraph fold = SIMD reduce", not "AtomSpace Frames are Alpha", not "OpenCog Miner is Shannon entropy", not a claim of AGI or consciousness, and not a mandate to reproduce Hyperon's VM. The research outcome may validly be a small composable subset with carefully scoped refusals.
