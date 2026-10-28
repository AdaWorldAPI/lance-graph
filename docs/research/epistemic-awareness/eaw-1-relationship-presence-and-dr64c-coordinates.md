# EAW-1 — Exact relationship-presence and DR64³ holographic coordinate plan

**2026-10-27 — Docs-only research stage, proposed, not implemented or admitted.** Independent main-based PR; no physical DR³ ABI choice, materialized tensor, Quack ownership change, source mutation, test pass or performance claim.

**Canonical breadcrumb:** [EAW-0 scratchpad #1820](https://github.com/AdaWorldAPI/lance-graph/pull/1820), held as a separate reviewable PR. **Nearby open owners:** [CFA-1 #1814](https://github.com/AdaWorldAPI/lance-graph/pull/1814), [factored ON×DO #1819](https://github.com/AdaWorldAPI/lance-graph/pull/1819), [DT64-Q0 #1799 MERGED](https://github.com/AdaWorldAPI/lance-graph/pull/1799), [D-DR256-WHERE #1792 MERGED](https://github.com/AdaWorldAPI/lance-graph/pull/1792). [EXO-46](https://linear.app/exo-red/issue/EXO-46), [EXO-140](https://linear.app/exo-red/issue/EXO-140), [EXO-151](https://linear.app/exo-red/issue/EXO-151) own unadmitted physical/addressing seams.

## Hypothesis, explicitly distinct from authority

A source-qualified DR64³ binding can address **three existing typed dimensions** (e.g. Subject×Predicate×Object or Identity×Property/Edge×Predicate/operation). It does not invent subjects, methods, object identities or relationship assertions. An exact presence plane over bound coordinates tracks *source-attested occurrence*, distinct from the ability to formulate a candidate address and separately calculated applicability. The third axis is LOGICAL until NFR-OP-2 is resolved.

A 64×64×64 Boolean cube has 262,144 potential coordinate positions and occupies 32 KiB only when dense; 64 64² slices each occupy 512 B. A DR256 selection spans 65,536 distinct row addresses, also 8 KiB when dense. These are DIFFERENT address universes, never interchangeable by equal byte size. 64 denotes a local block width, not the total extent of Odoo axes (#44 ON 194 / WHEN 5 provisional / DO 219).

For any attested focus/scope/version v, separate:
- `P_v(a,b,c)`: explicit positive source-attested relationship occurrence.
- `K_v(a,b,c)`: source-issued *coverage/completeness for that specific relationship predicate and address*, permitting a closed-world absence claim.
- `Unknown_v(a,b,c)`: not discovered or insufficiently covered; not a negative.
- `A_v(a,b,c)`: independently evaluated applicability of a source-attested predicate expression, with K3 TRUE/FALSE/UNKNOWN.
- `M_v(a,b,c)`: mapping occurrence IDs, source MethodIds, source kinds, recipe refs, priority and scope held separately from one Boolean bit. Several valid mappings to one recipe and several mappings contributing to one outcome are normal.

A presence=0 bit **alone** establishes neither a negative fact nor a missing identity. Even K=1 must be scoped: complete *for which predicate, keys, source window, class and snapshot?* Positive and negative evidence may require different source proofs. A hypothetical valid combination with P=0 is a research discrepancy/candidate, never a minted triple or permission to write.

## Two overlapping address views

1. **Rule binding:** Identity×Property/Edge×typed Predicate, which can interpret ON/WHEN/DO as semantic roles of *registered dimensional mappings*. A typed WHEN refers to a guard AST (not an atomic Boolean slot). Source-qualified map entries may point to the same already admitted polyfill recipe.
2. **Evidence binding:** Identity×Property/Edge with *associated* CausalEdge64 evidence records. Those 64-bit records are edge payloads; do NOT turn every payload value into one of 64 axis positions or assume the evidence axes and the rule axes share raw ordinals.

The coordinate alignment operation is separately attested by a source owner and DT64-Q0: source object/predicate/class identity, schema/catalogue version, actual pinned physical snapshot/borrow window, source-issued immutable lease, ACL/coverage, global→local focus seal, codebook and generation, distinct recipe identity. A local 6-bit ordinal is NOT a global ID; W on CausalEdge64 is a live thought lane only, never a source identity. Any source-changing replay must bind a new frame or refuse.

## Co-addressed execution and lawful preparation

Existing [D-RMAP](https://github.com/AdaWorldAPI/lance-graph/blob/main/.claude/plans/d-rmap-implicit-relationship-register-v1.md), DRF focus labs, [D-DR256-WHERE capstone](https://github.com/AdaWorldAPI/lance-graph/blob/main/.claude/plans/dr256-where-capstone-v1.md), Quack, RelAlg, and ndarray own the representation mechanisms. No new register type or third-axis ABI in this stage. A true ternary relationship may use source-provided exact presence or a *proved lossless factorization*. Pairwise relations SP, PO, SO are insufficient in general to reconstruct SPO.

ADMISSION B (operator decision, #1640, [EXO-140](https://linear.app/exo-red/issue/EXO-140)) permits certain counted, version-witnessed same-table preparation and co-addressed positional execution; it does NOT admit a value-addressed read into another table simply by calling it 'lifting'. Report `attest_cost`, `projection_build_cost`, `query_read_cost` separately. The per-callsite classification of Quack Semijoin/EqU32Via/Gather remains OPEN, so no unreviewed promise of in-fold joins or invisible gather.

## Pre-registered falsifier matrix (NOT RUN)

| ID | Mechanism counterexample / required oracle | Expected |
|---|---|---|
| E1-PRES-01 | Synthetic four attested triples on 2×2×2 even-parity relation: SP, PO, SO projections are all FULL. Independent literal triple list says 4, naive three-way pairwise join says 8. | Exact DR64³ presence returns 4, never invents the other 4; an unproven pairwise-only representation explicitly refuses exactness. |
| E1-PRES-02 | Same coordinate present in snapshot v1 and absent in pinned v2, with full and incomplete coverage alternatives. | `Present/KnownAbsent/Unknown` vary by proof; a copied stale bitmap is refused. |
| E1-PRES-03 | Two sources map the same local tuple (a,b,c) to different global identities, plus same local bits with rotated focus map. | No cross-source equality without explicit binding; no stale focus alias. |
| E1-PRES-04 | Two coexisting mappings to the same compiled polyfill at one coordinate; one has K3-UNKNOWN guard. | Both mapping identities preserved; presence cannot promote UNKNOWN to TRUE; no dedup to one MethodId. |
| E1-PRES-05 | Candidate SPO exists in ontology but source has not discovered the *relation*, and separate source explicitly claims negative with certified completeness. | Candidate identity is not invented; UNKNOWN and justified negative remain distinct. |
| E1-PRES-06 | Permute source row enumeration, direction, source owner and generation. | Rebinding or typed refusal, never an accidental parity match by ordinal. |
| E1-PRES-07 | More than 64 ON and DO entries from #44 194/219 fixture. | Blocked logical addressing works or declared limit refuses; no wrap/alias. |
| E1-PRES-08 | A pretend inner-loop gather makes a performance path look fast without its counted preparation. | Gate rejects hidden gather, separate costs reveal the discrepancy. |

**Oracles:** small independent raw tuple sets / source-declared completeness boundaries, separate scalar K3 reference; never validate a pairwise reconstruction with another pairwise reconstruction (EXO-109). Run mechanics on at least two unlike synthetic shapes; wait for pinned real data before any distribution-dependent performance claim (EXO-133). Each disabling condition must go red then green on an independently specified failure.

## Output requested from a future authorized lab

A classification report, NOT an implementation: `source identity / v / scope / focus seal / logical shape / 0,1,unknown coverage / mapping cardinality / orphan candidates / claimed gather cost / refusal reason`. Demonstrate that exact presence is not equivalent to a complete materialized third physical axis.

## Out-of-scope and promotion refusals

No DR³ physical decision (NFR-OP-2), global/local production admission (EXO-46), escaped Keep reuse, persistent semantic binding store, source write, law synthesis, new graph engine, learned reconstruction or new non-attested relationship identity. Operator-rejected #1812 is quarantined as an epistemic failure; not a source of evidence or implementation patterns.

**Next-stage logical prerequisite:** EAW-2 epistemic record projection depends on a valid EAW-1 *binding contract hypothesis*, not on a merged implementation or new ABI.
