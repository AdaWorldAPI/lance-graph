# D-DRMC-3 — law, proof, compiled recipe, runtime bind and effect are distinct

This experimental metadata layer enforces the **shape of reuse**, not its scientific truth or physical execution. A `LawReference` keys a mathematical statement by primitive, row/column *kinds*, truth family, population semantics, law revision and the statement a proof must prove. It intentionally does not key on `ClassId`, method name, frame or source revision. A theorem about a genuinely common law could therefore be used by multiple ClassViews, after each class method is resolved to a matching mathematical shape. The fully typed method and witnesses remain on each `BoundDr64` and are never stored as law identity.

Proof and target identity are Stage B's, not re-spelled here. A `ProofReference` has private fields and one constructor, `ProofReference::rechecked`, which requires the `ReplayCapsule` to prove the law's `statement` (Stage B's `SemanticKey`) and re-runs `ReplayCapsule::recheck` (a fresh egglog proof session, byte-equal proof). There is no caller-set "checked" flag and no caller-supplied proof digest, so the earlier hole (a forged nonzero digest passed) cannot be written. `CompiledReference` identifies the compiler revision, code artifact, backend and Stage B's `TargetAbi`. `RecipeOffer` remains **metadata**, not bytes to execute.

**Limit, stated plainly:** Stage B's checker proves exact-integer substitution statements. No statement language and no checker exist yet for a Boolean-DR64 law, so the tests pair a DR64 law shape with Stage B's E1 statement to exercise the admission plumbing only; that pairing is not a claim that E1 says anything about DR64. Adding the dependency also brings egglog and ogar-loco (git) and the ndarray sibling patch into this probe's build.

Stage 2's finite-focus `PopulationReport` CANNOT produce a `ProofReference`. Even a complete 64x64 population pass is not a theorem outside its declared scope.

Relevant existing foundation: merged #1690–#1696 `egglog-r2il-probe` E1, with #1698 handling contradictory assumptions and exact-vs-checked overflow in a *different owned branch*. This PR does not reimplement or change egglog, Ruler, Enumo, eggcc or their proof format. Their admission must happen COLD, with source/outcome provenance, scalar/domain oracles, deterministic replay, effect-order constraints, measured costs and a target ABI gate.

The next integration stage must prove an actual **ClassView→relation frame→method law→RelAlg AST→compiled recipe→mask-risc/native SIMD** round trip, including refusals, independent oracles and full cold/amortized economics. Until then this is an isolated falsifiable seam.

ACT-R/Rubicon/kanban/cognitive-shader-driver are downstream consumers and are not allowed to use a mathematical proof as effect authorization. Thought-Git may link immutable law/proof/recipe/binding/replay/commit references when its own write and provenance contracts are ready.
