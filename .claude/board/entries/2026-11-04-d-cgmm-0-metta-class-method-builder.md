# D-CGMM-0: Kythe-shaped facts, lazy Codegraph, MeTTa schema scratch and DT64 method building

**2026-11-04. STATUS: DOCS / PRE-REGISTRATION ONLY.** This board entry is a proposed architecture, not an AST intake implementation, executable MeTTa/Datalog runtime, new ClassId/FieldRef/ActionDef API, source identity owner, vendor integration, NNUE certification or speedup claim.

**Core operator insight:** Use MeTTa-shaped **logical class-method schema constraints**, not a second MeTTa interpreter. Codegraph should become **demand-scoped and lazy**, backed by compiler-evidence facts with Kythe-like stable source identity, with Crepe/Soufflé/XLOG compilation **metadata** guiding required relation/field views. A witnessed `ClassView` provides the stable field basis; proposed `FieldViewDef` offers are matched in temporary logical constraint scratch; OGIT/OGAR/DT64 admit ontology/ActionDef/method semantics; existing RelAlg/R2IL/mask-risc/ndarray executes eligible operations on borrowed SoA.

**Architecture:** [D-CGMM master](../../plans/d-cgmm-codegraph-metta-method-builder-v1.md) · [Kythe/Datalog/lazy Codegraph](../../plans/d-cgmm-lazy-codegraph-fact-demand-v1.md) · [OGIT/OGAR ActionDef and M:N](../../plans/d-cgmm-action-relation-schema-v1.md) · [staged PR charter](../../plans/d-cgmm-staged-pr-charter-v1.md) · [46 falsifiers](../../plans/d-cgmm-falsifier-matrix-v1.md).
**Tracking:** [EXO-155](https://linear.app/exo-red/issue/EXO-155), [EXO-50](https://linear.app/exo-red/issue/EXO-50) message bus; [EXO-153](https://linear.app/exo-red/issue/EXO-153)/[EXO-154](https://linear.app/exo-red/issue/EXO-154) related virtual driver/zero-copy plans.

## Existing proofs vs new proposal

- Codegraph `graph.rs` currently interns symbols into `Pos`, builds a `MapTable` plus six XLOG `Cube` relations and masks; the reverse derived structures have `OnceCell`. `query.rs` uses bound joins and graded confirmed/possible/unknown masks. **Partially lazy today, not full demand-compiled Codegraph.**
- Ruff/Charon `.rows.bin`, FUNDECL and PROV are existing source-evidence intake; source effect/dispatch/receiver gaps remain **UNKNOWN**. Codegraph interned `Pos` is NOT Kythe durable source identity, ClassId, MethodId or DR ordinal.
- The [Crepe/Soufflé/XLOG archaeology](2026-11-02-coresearch-xlog-souffle-crepe-dr3-archaeology.md) documents binding/demand/index, magic-set/stratification and compile metadata losses. It expressly warns not to import a second runtime, and records `crepe_plan!` drift D1/D2. **Compiler metadata can be cold admission evidence only after falsification.**
- `ClassView` is already the meta lookup above SoA; `FieldMask`/`WideFieldMask` signify instance presence, NOT speculative rule truth. `lens_fields(full_classid)` is mandatory for behavior lens; unknown lens refuses.
- `lance-graph-ontology::MappingProposalKind::Edge{LinkSpec}` and `Cardinality::ManyToMany` already model typed ontology edges, **not** edge instances or Bag derivation counts. OGAR `ActionDef` owns richer behavior; `ActionSpec` is not equivalent.
- `lance-graph-relalg::Rel` supports one-population All/Empty/Plane/I32/U32 atom/And/Or/Not + Count/Sum/Min/Max/Keep. It does **NOT** execute an arbitrary M:N join, effect, guarded DT64 WHEN or MeTTa rewrite.
- `lance-graph-r2il::binding(_native)` admits only a restricted executable subset and explicitly refuses guarded WHEN/three-valued truth; `ExecPermit` is lab only. No automatic schema-to-code promotion.
- NNUE may rank candidate equivalence **offline only**. Source evidence, tenant permission and semantic effect order decide admission, not probability.

## Independent future PR stages

| ID | Purpose | State / owner |
| --- | --- | --- |
| D-CGMM-0 | docs, owner gates, 46 independent tests, board/Linear handoff | THIS DOCS PR |
| D-CGMM-1 | Ruff/Charon binary/Kythe fact identity and effect-loss census | FUTURE LAB; producer/Codegraph |
| D-CGMM-2 | compiled binding/demand metadata; lazy Codegraph against eager baseline | FUTURE LAB; XLOG/Codegraph/EXO-73 |
| D-CGMM-3 | ClassView-based FieldViewDef, disposable MeTTa constraint scratch | FUTURE LAB; EXO-121/152 |
| D-CGMM-4 | OGIT M:N edge tenant + OGAR ActionDef schema offers | FUTURE LAB, semantic/tenant owner gates |
| D-CGMM-5 | pure one-row RelAlg polyfill of eligible FieldViewDef | FUTURE LAB, Quack/R2IL owner |
| D-CGMM-6 | admitted DT64 method → native binding proof | GATED EXO-121/152, execution owner |
| D-CGMM-7 | memoized recipe/binding economics, no source-value cache | FUTURE LAB after 5/6 |
| D-CGMM-8 | independent datafrog/MORK oracle, NNUE-only candidate ranking and reachability | GATED EXO-79/143/73/154 |

## Mandatory refusals

Unknown effects ≠ no effects; possible dispatch ≠ confirmed; missing source producer ≠ known absence; ClassId concept ≠ app method equivalence; many-to-many ≠ Bag; ontology context ≠ tenant authorization; Codegraph `Pos` ≠ durable source ID; RelAlg GYO acyclicity ≠ a join executor; no recursive/cross-tenant/guarded action under a fallback interpreter. **Scratch is disposable and cannot issue ClassId/MethodId or mutate registered ontology.**

**Related owner issues:** [EXO-84](https://linear.app/exo-red/issue/EXO-84), [EXO-85](https://linear.app/exo-red/issue/EXO-85), [EXO-79](https://linear.app/exo-red/issue/EXO-79), [EXO-143](https://linear.app/exo-red/issue/EXO-143), [EXO-101](https://linear.app/exo-red/issue/EXO-101), [EXO-73](https://linear.app/exo-red/issue/EXO-73), [EXO-108](https://linear.app/exo-red/issue/EXO-108), [EXO-95](https://linear.app/exo-red/issue/EXO-95), [EXO-121](https://linear.app/exo-red/issue/EXO-121), [EXO-152](https://linear.app/exo-red/issue/EXO-152), [EXO-153](https://linear.app/exo-red/issue/EXO-153), [EXO-154](https://linear.app/exo-red/issue/EXO-154).

**Deferred:** any inspection or consolidation of cognitive-shader-driver / migrated Ladybug-rs / P64/StreamDTO/BusDTO/PerturbationDTO; no speculative production endpoint or new schema store.

## Implementation packet / explicit EXO decisions (2026-11-04)

New source-linked documents in the same **docs-only draft PR #1958**: [implementation blueprint](../../plans/d-cgmm-implementation-blueprint-v1.md), [EXO owner gate ledger](../../plans/d-cgmm-exo-gates-v1.md), [concrete independent fixture corpus](../../plans/d-cgmm-executable-fixtures-v1.md). The fixtures fix call-grade expectations, pure selection COUNT=2, M:N tenant-A Set=3/Bag=4, contradictory application-action effects and invalidation under source/scope change. **All NOT_RUN**. Explicit critical path: EXO-155 coordination; EXO-84/85 demand and MeTTa rules; EXO-121 FieldViewDef/ClassView; EXO-152 normalized native method; EXO-46/140 foreign binding; EXO-95 OGIT/OGAR/SAP/Odoo; EXO-108 Bag; EXO-73 recursion; EXO-79/143 MORK oracle; EXO-153/154 borrowed execution; EXO-104/105/109 oracle/copy firewall; EXO-50 append-only messaging. No production admission, new schema authority or automatic merge.
