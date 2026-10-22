# D-DRMC-3 — law, proof, compiled recipe, runtime bind and effect are distinct

This experimental metadata layer enforces the **shape of reuse**, not its scientific truth or physical execution. A `LawReference` keys a mathematical statement by primitive, row/column *kinds*, truth family, population semantics and law revision. It intentionally does not key on `ClassId`, method name, frame or source revision. A theorem about a genuinely common law could therefore be used by multiple ClassViews, after each class method is resolved to a matching mathematical shape. The fully typed method and witnesses remain on each `BoundDr64` and are never stored as law identity.

A `ProofReference` identifies external checker revision and proof digest. `CompiledReference` identifies separate compiler revision, code artifact, target ABI and backend (existing R2IL/mask-risc or direct resident SIMD). `RecipeOffer` remains **metadata**, not bytes to execute. The `CheckedByExternalAuthority` enum is **caller asserted** in this probe. It does not run or independently authenticate a proof; a production owner must require actual independent checker receipts, pinned law/ISA versions and source metadata. Zero digest refuses only a missing identity, not a forged nonzero digest.

Stage 2's finite-focus `PopulationReport` CANNOT produce a `ProofReference`. Even a complete 64x64 population pass is not a theorem outside its declared scope.

Relevant existing foundation: merged #1690–#1696 `egglog-r2il-probe` E1, with #1698 handling contradictory assumptions and exact-vs-checked overflow in a *different owned branch*. This PR does not reimplement or change egglog, Ruler, Enumo, eggcc or their proof format. Their admission must happen COLD, with source/outcome provenance, scalar/domain oracles, deterministic replay, effect-order constraints, measured costs and a target ABI gate.

The next integration stage must prove an actual **ClassView→relation frame→method law→RelAlg AST→compiled recipe→mask-risc/native SIMD** round trip, including refusals, independent oracles and full cold/amortized economics. Until then this is an isolated falsifiable seam.

ACT-R/Rubicon/kanban/cognitive-shader-driver are downstream consumers and are not allowed to use a mathematical proof as effect authorization. Thought-Git may link immutable law/proof/recipe/binding/replay/commit references when its own write and provenance contracts are ready.
