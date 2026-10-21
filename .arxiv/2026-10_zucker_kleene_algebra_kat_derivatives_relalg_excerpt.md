# Kleene algebra: prove the path expression before lowering its masks

**Primary source:** Philip Zucker, [*Knuckledraggin' Kleene Algebra*](https://www.philipzucker.com/bryzzowski_kat/), blog post, 2024-06-10. **Not an arXiv paper.**
**Related theory, distinct source:** Steffen Smolka et al., [*Guarded Kleene Algebra with Tests: Verification of Uninterpreted Programs in Nearly Linear Time*](https://arxiv.org/abs/1907.05920), arXiv:1907.05920v4 (2019), POPL 2020.
**Existing experimental reference:** [Zucker's egg-smol Kleene/KAT rules](https://github.com/philzook58/egg-smol/blob/scratchpad2/tests/kleene.egg).
**Research status (2026-10-21):** Source-grounded excerpt and proposed Lance-graph falsifiers. No integration, optimizer speedup, or correctness proof for Lance-graph has been demonstrated.

## Source-grounded excerpt

Zucker encodes the Kozen-style Kleene algebra operations in a small Z3/Knuckledragger proof experiment: alternative `+`, sequential composition `*`, iteration `star`, empty `0`, and identity `1`. His axioms include associative composition, commutative/idempotent choice, distributivity, star unfolding `star(e) = 1 + e*star(e)`, and a least-fixpoint/induction principle. The blog walks through proofs of `star(x)*star(x) = star(x)` and `star(star(x)) = star(x)` with a calculational proof helper.

Zucker also sketches Brzozowski derivatives: `E(expr)` checks whether the empty word is accepted, while `D(symbol, expr)` describes the residual expression after consuming that symbol. This is an interesting *potential* compilation surface for labeled graph paths, not a demonstrated graph-query engine. The later sections explore egglog-style rewriting and Twee as alternative proof tactics; they report no measured comparisons against Lance-graph or a production KAT verifier.

**Prototype caveat:** The pasted derivative sketch mixes a Boolean `E(b)` with a regex constructor `seq`, and a later Z3 datatype sketch references undefined `REsyn` (the page displays a `NameError`). Treat these as exploratory notebook fragments, not a drop-in, tested oracle. In particular, no complete KAT-with-tests decision procedure is established by the blog.

The *separate* GKAT paper restricts choice and iteration to predicate-guarded forms and proves that equivalence in this **restricted algebra** can be decided in almost linear time, whereas unrestricted KAT equivalence is PSPACE-complete. That is a claim about expression equivalence in their formal setting, **not** about the runtime of joins, arbitrary Datalog programs, or graph transitive closure.

## Hypothetical transfer to lance-graph-relalg / Quack

Under **set** relational semantics, represent `0` as empty relation, `1` as identity relation, `+` as union, `;` as relational composition, and `R*` as reflexive transitive closure. A Boolean test `b` is a *subidentity relation* `{(x,x) | b(x)}`, rather than an untyped free-floating mask. Thus `b;R` filters the source domain of `R`, while `R;b` filters its target domain. Their masks can be folded only with correct coordinate and mapping witnesses.

Example law: `(R + S);T = (R;T) + (S;T)` is a legal algebraic equivalence for set relations. It is **not** a blanket license to turn a foreign-key join into `AND`: `AND` only covers correctly co-addressed row populations or precomputed separator projections with the relevant binding witnesses and a pinned generation. Equal integer offsets in unrelated relations are not identity.

One candidate architecture is a **cold, typed, version-pinned proof/optimizer surface**, rather than a second hot query engine:

```text
relalg/R2IL typed plan -> KAT/GKAT-eligible subset + proof or refusal
                     -> normalize / egglog extract with cost constraints
                     -> optional derivative automaton for labeled paths
                     -> D-RPF / global-to-local pinned bindings
                     -> Quack / mask-risc positional folds or fixpoint execution
```

The derivative automaton could name residual path states and guarded transitions. The execution kernel would still need to resolve edges, carry endpoint bindings, respect duplicate/path semantics, and converge correctly for recursion. A proof about symbolic path expressions does **not** prove a physical index mapping, stale-version read, aggregate terminal, or durable write.

**Semantics fence:** Ordinary KAT uses idempotent `+` and therefore models set-style union. SQL bag union, duplicate-sensitive `COUNT(*)`, multiplicity of paths, provenance semirings, weighted/costed paths, nullable comparisons, side effects, and unbounded general Datalog recursion require explicitly different contracts. Do not silently normalize them under Boolean KAT laws.

## Falsifiers / test path K

- **K1, identities:** Enumerate tiny finite relations and compare `R;1 = R`, `R;0 = 0`, `(R+S);T = R;T + S;T`, `R*;R* = R*`, and `(R*)* = R*` against a standalone scalar set-relation oracle.
- **K2, guard placement:** Verify that `b;R` and `R;b` yield different answers for an asymmetric graph when `b` filters source versus target; reject rewrites that float a test across an edge without proof.
- **K3, address witness:** Two foreign row universes intentionally reuse numeric indices. A direct `AND` must refuse until an explicit correct mapping/coordinate witness is supplied; stale mapping or generation must also refuse.
- **K4, set versus bag:** Duplicate edges and two independent paths to one endpoint expose illegal idempotence and duplicate-sensitive `COUNT(*)`; the set-only optimizer must not claim SQL bag equivalence.
- **K5, derivatives:** Compare residual-automaton acceptance and bounded path answers against a naive interpreter for 0, 1, alternation, sequence, nested stars, empty-word paths, and guarded paths. Detect nullable-derivative and termination errors independently of the blog's notebook code.
- **K6, optimizer regret:** Measure proof time, certificate/replay bytes, automaton states, mapping/preparation cost, payload reads, output materialization and end-to-end runtime against unchanged relalg/Quack, Datafrog fixpoint and simple traversal baselines. Include cases where a mathematically smaller expression runs slower.
- **Silence twin:** Algebraic no-op rewrites produce identical answers, no extra source reads when avoidable, and no false committed dirty bit.

## Decision boundary

Use KAT/GKAT initially as a **semantic checker and bounded planning research experiment**. Do not assert the GKAT almost-linear equivalence result for arbitrary KAT, SQL, or physical graph operations. Require typed laws, bounded extraction/fuel and independent scalar oracles before considering lowering. No new production engine is authorized by this note.

**Adjacent local research:** [egglog equality saturation / Datalog](2304.04332_egglog_equality_saturation_datalog_excerpt.md); [Yannakakis / SMASH, mask-first shared coordinates](2502.20233_selective_yannakakis_smash_mask_first_thought_coordinates.md); [D-RELALG-UNIFIED](../.claude/plans/d-relalg-unified-v1.md).
