# D-SAG-2 evidence / parity contract: the observations are not the authority

**Type:** cross-repo method and validation checklist. **READ BY:** anyone changing SoftwareGraph/MethodTable, Ruff SPO/AST frontends, SAG prediction loops, OGAR ActionDef, Quack/R2IL or NARS evidence.
**Status:** proposed guardrail, with source-grounded possible defects to *test*, not confirmed bugs.

> ⊘ 2026-10-13: `SoftwareGraph` and `MethodTable` (`lance-graph-glove-parity`'s `method.rs` / `awareness.rs`) were removed from `main` by #1524 (D-CMR-RESET); the code is on `backup/session-011BMJ-2026-10-12`. Wherever this doc names them, read them as the withdrawn first build. The separation-of-concerns rules still apply to the D-CMR rebuild, whose binding replaces them.

## Separation of concerns

| Question | Evidence source | Permitted conclusion |
|---|---|---|
| What did the author declare? | MethodSpec, Ruff AST, source digest | Declared syntax with span and revision |
| What did the compiler accept? | MethodTable::prepare, Program, refusal | Compiled or typed refusal, not runtime proof |
| What actually ran? | independently executed oracle / instrumentation | Observed result under bound environment |
| Was it independently confirmed? | signed/registered or replayable oracle receipt | Verified only when oracle identity and output validated |
| Which hypothesis deserves testing? | observations + revision policy | Hypothesis; no method authority |
| May an external effect occur? | OGAR action capability + current policy/roles | Authorization decision only at execution/commit |

**"Unknown" is not "NotHeld" and "not emitted" is not "false."** Dir-sim #1503 explicitly distinguishes Held, NotHeld, Undetermined and NotDeclared. Ruff may omit a triple for unsupported semantic reasons. Never infer a negative fact from absent output without known complete extraction.

**No self-certifying loop:** a SoftwareGraph claim() or a user-supplied string never produces Verified. Compiler-produced edges are not sufficient to prove behavior; tests are not a valid oracle unless actually run. A test identifier and matching body digest prove only identity compatibility, not that execution occurred.

## Minimal source→graph crosswalk

Require:
- repository identity and commit SHA;
- language and compiler/harvester version;
- source span and symbol qualified name where available;
- stable concept identity, schema seal and reading;
- method key with explicit mapping provenance, never derived from display label alone;
- edge predicate with source declaration, derived inference, or unavailable-status;
- extraction closure/completeness per predicate on the specific source domain.

Do not promise that all Ruff languages lower directly to one lossless ModelGraph round-trip. As of the verified 2026-10-12 Ruff source, reassemble() reverses C++ machine-plane projected form only; core-7 and OpenProject IR content are not reconstructed. The Ruff RFC is a desired destination, not proof of full coverage.

Ruff #117 preserved ordered Ruby MethodOre (set reductions are lossy). Ruff #118 deliberately did not overload C++ IsStatic to represent C file-scope linkage and did not emit a free-function call graph into triples. Ruff #120 resolves a subset of Odoo extension-view locators; unsupported XPath remains unresolved, not guessed.

Therefore distinguish:
- *source structure parity* (supported SPOs, anchors, signatures);
- *behavioral equivalence* (tests/runtime oracle);
- *temporal counterfactual prediction* (precommitted A→B with blind independent observation);
- *optimization* (semantic parity + measured cost);
- *authority* (action gate plus current permission witness).

## Provenance and snapshots

A logical fact and its evidence witnesses need separate identities. If (src,rel,dst,tier) is logically deduplicated, multiple ctx/prov owners still need independently removable ownership. Test current awareness.rs link() and project() behavior, including classes and shared fields. When a source disappears in B, remove current witness without erasing the other source or rewriting the archived A receipt.

A compiler's current refusal can change at revision B; no old ComputeCycle remains active merely because its node exists in the interning table. For source/version-sensitive hypothesis revision, replay a pinned read at k during writes to k+1. Immutable method/node identity is not equivalent to immutable snapshot semantics.

Evidence stamps represent *originating observations*, not each derivative inference or each layer containing the same observation. Retraction needs explicit original contributor; where provenance is insufficient, return typed insufficiency. Keep Jøsang belief fission distinct from NARS evidence revision.

## Controlled attention and intervention

Run five attention policies under fixed candidate pool, oracle, inference rules, source revision, seed and total work budget:
1. FIFO;
2. priority by declared confidence;
3. naive high Shannon entropy;
4. expected information gain, only after a normalized and calibrated predictive distribution is defined;
5. Moore/Staunen policy.

Include irrelevant high-entropy distractors, correlated evidence, a delayed useful discovery, negative tests and bad predictions. Report oracle-supported findings / total work, decoded bytes, allocations, and time. Do not treat NARS f/c or PLN strength as a probability distribution without calibration.

For Rubicon-style agent control, distinguish: predecisional evidence deliberation → revocable implementation intention → action *proposal* → OGAR commit gate → witnessed outcome → adaptation/disengagement. A predicted benefit never grants action capability. A committed intent must re-check policy and role evidence at commit and cannot override Unread/Undetermined.

## Measurement and failure semantics

Hard gates: correctness, independent provenance, schema/reading version, graph/row population, evidence overlap, permission, and replay. Soft measurements: discovery quality, hypothesis coverage, entropy reduction, MDL, latency, allocations, decoded bytes and energy if available.

H1 reported a gap (current Quack masked sum slower than a fused scalar reference) and an absent fused terminal; it did not report a speedup. Don't put that gap into a success numerator. Use native and x86-64-v3 controls from #1505.

If a new result cannot be expressed through existing Quack/OGAR/R2IL, return an explicit typed refusal and file the exact minimal insufficiency. Do not work around it by injecting a graph evaluator.

## Source map

- [lance-graph #1503 role evidence](https://github.com/AdaWorldAPI/lance-graph/pull/1503)
- [#1504 method parity](https://github.com/AdaWorldAPI/lance-graph/pull/1504)
- [#1506 DAG range](https://github.com/AdaWorldAPI/lance-graph/pull/1506)
- [#1508 predicate reuse](https://github.com/AdaWorldAPI/lance-graph/pull/1508)
- [#1509 SoftwareGraph](https://github.com/AdaWorldAPI/lance-graph/pull/1509) (withdrawn by #1524; backup branch above)
- [Ruff #117 ordered behavioral ore](https://github.com/AdaWorldAPI/ruff/pull/117)
- [Ruff #118 C linkage semantics](https://github.com/AdaWorldAPI/ruff/pull/118)
- [Ruff #120 Odoo inherit_id](https://github.com/AdaWorldAPI/ruff/pull/120)
- [Ruff ModelGraph](https://github.com/AdaWorldAPI/ruff/blob/main/crates/ruff_spo_triplet/src/ir.rs)
- [Ruff reassemble](https://github.com/AdaWorldAPI/ruff/blob/main/crates/ruff_spo_triplet/src/reassemble.rs)
