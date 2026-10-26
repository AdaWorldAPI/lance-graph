# D-DT64-3: Directed inverse projections, DR focus and measurable holographic access

> **DT64↔Quack coordination Q0 (2026-10-26):** [Minimal source-attested borrowed binding contract, capstone #1799](https://github.com/AdaWorldAPI/lance-graph/blob/research/dt64-predicate-holographic-capstone-20261026/.claude/plans/d-dt64-q0-borrowed-quack-binding-seam-v1.md). Directed (s,p)→o, (p,o)→s and (s,o)→p are existing Quack/relalg algebra queries over source-borrowed bindings. Quack owns evaluation, DR64² is an optional measured physical view; no required DR64³ allocation. All first-landing tests require #1792's detached-\`Keep\` 1,032-vs-31 refusal/recompute. Coordination note only; not new implementation authorization.

**Date:** 2026-10-26. **Status:** GATED PLAN, docs only. **Parent:** DT64-0–2 independent drafts.
**Predecessors:** D-QR-CAP/EXO-121, EXO-120, D-RMAP/EXO-89, D-DRF/EXO-151, D-DRMC/EXO-127, D-QSP/EXO-126, [#1792](https://github.com/AdaWorldAPI/lance-graph/pull/1792). Do not duplicate existing executors.

## 1. What 'graph turned inside out' means operationally

Given source-authorized typed relation R(subject,predicate,object), expose three reversible *queries*, not necessarily three copies:
- `(s,p) -> {o}` object projection;
- `(p,o) -> {s}` subject/inverse projection;
- `(s,o) -> {p}` predicate projection.
The **same directed fact** produces all three; `(o,p,s)` is not asserted. Predicate inverses only when separately named and typed. A given local focus can invert relation *access*, but this is not automatically a lossless holographic encoding of the global graph.

Define `R_f[s,p,o]` for a sealed focus `f`, as a **logical** typed sparse relation. The 64×64×64 dense Boolean comparison shape is 32 KiB *before* sidecars, but true physical DR64³ admission is an open NFR-OP-2 decision. Axis 3 might mean properties only after DR owner approval; do not conflate with DR64 64-thought occupancy, a palette value axis or universal identity. Per-predicate domain/range and local↔global tables need independent witnesses. DR64² = 512 B, DR256² = 8 KiB. Type conversion and valid-time/time-series are not hidden in these 1-bit cells.

## 2. Relation mechanics: how to avoid a fake hologram

```text
F = source-identity + immutable snapshot + schema/codebook generation
A = authorized scope and negative-observation completeness
C = focused global→local coordinate witnesses (subject and object)
P = versioned typed predicate ordinal map
R[s,p,o] = TRUE only where source evidence, C, P, F, A all admit it
```

For every `R[s,p,o]`, maintain source provenance separately and distinguish **unobserved/unread/NULL/forbidden** from known FALSE. The logical relation `R` is a TRUE-only sparse set, but a complete answer requires an additional *knownness/coverage* relation. `NOT R` is valid only inside a complete authorized domain. Any answer outside focus must be classified EXACT_LOCAL, EXACT_EXPANDABLE, APPROXIMATE (declared) or INVALID/UNKNOWN by existing D-DR3 criteria. Boolean folds don't create unknown truth.

### Directed projection oracle

Synthetic typed identities `u:User,g:Group,r:Role,m:Mailbox`:
```
(u0,MEMBER_OF_DIRECT,g0) true
(g0,GROUP_GRANTS_ROLE,r0) true
(u0,HAS_EXO_MAILBOX,m0) true
(g0,MEMBER_OF_NESTED,g1) true
(g1,GROUP_GRANTS_ROLE,r0) true
```
Queries `(u0,MEMBER_OF_DIRECT)->{g0}`, `(MEMBER_OF_DIRECT,g0)->{u0}` and `(u0,g0)->{MEMBER_OF_DIRECT}` return different typed result sets from the SAME observed facts. Inverting to `(g0,MEMBER_OF_DIRECT,u0)` must be FALSE only with evidence of closed-world coverage, otherwise UNKNOWN. Transitive role inference returns r0 with **two supports**; drop one path, r0 remains. Drop both under complete witness, it ceases to be true.

## 3. Boolean / Bag / epistemic composition

- Relational Boolean composition: `C[i,j] = OR_k(A[i,k] AND B[k,j])`. Distinct TRUE (multiple paths still one).
- Bag multiplicity: `count_k(A[i,k] AND B[k,j])` requires a counted measure, not XOR and not OR. Do not collapse duplicate paths when support is needed for retraction.
- Missing data (SQL K3): UNKNOWN persists on incomplete observations. `FALSE` is not `no bit set` absent an explicit completeness witness.
- Unary `IS_ADMIN` is scoped derived predicate: a policy role plus target resource. Predicate means nothing without policy source, applicable time and scope.
- Edge×edge relation composition is an optional typed operator over *edge identities*, not the same as user→group edges; defer to DT64-4.

Leverage existing `lance-graph-relalg` `Rel/Measure`, Quack `Keep`, mask-risc terminals, `dir-rmap-probe` `Compiled::roles_of` and the #1707 Boolean associative lab. Do not introduce a new Tensor op or require a new data structure just to rotate an existing mask.

## 4. Candidate physical layout routes (compare, do not assume)

| Route | When legal | Bytes counted | Candidate outcome |
|---|---|---|---|
| direct borrow + `FieldMask` | only structural field presence, 8/23 fields | scalar mask + source lanes | baseline, probably KEEP_DIRECT |
| borrowed Quack single-plane | typed predicate WHERE and Count/Exists | lane + select mask | likely primary |
| DR64² predicate adjacency | repeated 64×64 typed same focus | 512 B + index/seal and coverage | candidate |
| DR256² relation block | repeated 256×256 focus | 8 KiB + index/seal and coverage | candidate, measure |
| logical DR64³ | repeated joins across chosen third axis, only if typed and admitted | 32 KiB dense hypothetical + sidecars | not implementation authorization |
| sparse CSR / direct walk | low selectivity / irregular identities | pointers and provenance | often cheaper |
| materialized detached Keep | only under valid cross-borrow source witness | extra bytes + guards | BLOCKED by #1792 |

The first experiment should **not** build the cube. Compare direct borrowed lanes, DR64² one-predicate tiles and scalar oracle first; expand to cubic logical oracle only if repeated questions demand it.

## 5. Falsifier ladder

- **H0:** 8 named predicates over 64 nodes and exact scalar directed triples, all three projection orientations; toggle each direction and verify a different expected answer.
- **H1:** Distinct identities with same local ordinal in two foreign source maps: refuse, no alias.
- **H2:** Missing `MEMBER_OF` data vs explicit complete negative: UNKNOWN vs FALSE.
- **H3:** Two-role-support diamond: retraction of one support must not remove effective role.
- **H4:** Predicate endpoint reversal and symmetric misreading: fail type contract.
- **H5:** OR-AND vs XOR, bag counts and K3 independent oracles; two true paths must not cancel.
- **H6:** Focus omits an intermediate contributor: exact local result must refuse or legally expand; compare before vs after projection.
- **H7:** Stale source, focus map, schema/codebook, changed scope, same numeric revision from different sources: refuse detached reuse; borrowed execution allowed under same read scope.
- **H8:** Exceed data size / sparse selection: direct and CSR controls, full prepare/prefetch/accounting; no uncharged masks.
- **H9:** Attempt typed predicate as `ClassView FieldMask` or predicate ordinal as a palette code: negative mismatch must bind.
- **H10:** Attempt 64×64×64 dense allocation without independent performance win: negative cost result must count as failure.

All functional oracles must be independent of Quack and other candidate kernels. Disable each falsifier by a named mutation that really reddens the test; record inert controls. Add windowed warm reuse 1, 8, 64, 256 queries; report cold admission, identity seal, tile/CSR prep, SIMD work and output emission separately.

## 6. Holographic claim, precisely gated

Label result `relational projection` until:
(a) a family of local projections reconstructs the relevant directed fact set without missing/invented triples **under a declared complete focus and witness**, and
(b) total coordinate, index and provenance cost is measured; 
(c) corrupted/incomplete lenses refuse instead of hallucinating global reconstruction.

If it is only reversible query access over a complete tensor, say exactly that. Optical holography and VSA/HRR reconstruction are different claims. No 'learning' implied by edge inversion.

## 7. Hard boundaries and handoff

NFR-OP-2 third-axis physical semantics remains DR-owner decision, EXO-46 prepared global-to-local map requires explicit production admission; #1792 escaped result reuse remains blocked. ClassView governs field presence, not permission. No value-derived cross-table gather, no tuple materialization in hot path, no proof capsule claiming independent checking. Defer time/hypotheses to DT64-4 and ONNX numerics to DT64-5.

**Exit:** first 8 groundable predicates pass forward/inverse/orientation and role diamond differential tests; per-route cost and coverage measured; decision ADOPT / KEEP_DIRECT / DEFER / REJECT for each physical route. Research only until separately authorized.
