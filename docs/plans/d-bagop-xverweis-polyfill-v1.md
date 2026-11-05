# D-BAGOP-XVERWEIS-1: Ordered XVERWEIS as a holographic Bag fold

**Status:** PROPOSED / admission unresolved. **Date:** 2026-11-05. **Owner:** Bagop, Quack, DT64 and mask-risc in lance-graph. **Dependency:** P0 + ndarray P1.

**Operator invariant:** Every XVERWEIS is a FACET STRIDE-shaped ndarray-backed Bagop folding polyfill on the DR64 holographic Bag. PSCustomObject is a logical property projection, not a per-row heap object graph. This is a requested law, not proof of existing admission.

## Semantic contract

Input is a source-qualified, authorized Bag population with identities, occurrences, typed properties, values, frame/value/layout witnesses and an explicitly witnessed **search order**. XVERWEIS adds a search key, lookup-property lens, return-property lens, search/match mode and not-found behavior.

For exact equality, the abstract (oracle-only) meaning is:

    matches = occurrences whose lookup property equals search key
    selected = first_or_last(matches, witnessed_search_order)
    output = property(selected, return_lens) or not_found

Do not implement this oracle list as a production Vec<Row>. Selecting one occurrence does not erase the Bag's multiplicities. Raw DR local order is not automatically Excel row order. The scalar oracle must define string case/collation, number precision, error propagation and blank/absent distinctions explicitly rather than guess.

## Mode admission matrix

| Operation | Obligation | Status pending source census |
|---|---|---|
| Exact match | Typed equality and state masking | OPEN: locate admitted predicate path |
| First/last | Stable witnessed order and occurrence selection | OPEN: find lawful ordered fold |
| Duplicate keys | Preserve Bag count; select specified occurrence | OPEN: prove no implicit DISTINCT |
| Not found | #N/A versus caller-supplied default | OPEN: explicit result-state law |
| Wildcard | Excel wildcard and escaping rules | NOT ADMITTED until text predicate proof |
| Approximate | Comparator, monotonic bound, tie behavior | NOT ADMITTED until independent oracle |
| Binary search | Sortedness/direction precondition and stale-order refusal | NOT ADMITTED until witness |
| Return array/spill | Shape + ordered children by address | OPEN: P3 consumer contract |
| Cross-table lookup | Requires lawful alignment/co-addressed producer | BLOCKED for banned Gather or ADMISSION B |
| Reused prepared selection | Zero additional global→local lookups | DRISA lab #1955; M1/M2 production gates open |

A missing operation must produce an explicit refusal; no hidden HashMap, CAM, or row scan fallback disguised as Bagop.

## Source to audit before implementation

- [ndarray simd_stride.rs](https://github.com/AdaWorldAPI/ndarray/blob/master/src/simd_stride.rs): position discipline only, no complete XVERWEIS.
- [D-DYNBAG-0 #1906](https://github.com/AdaWorldAPI/lance-graph/pull/1906): Quack→mask-risc Bag count/support lab.
- [DRISA #1955](https://github.com/AdaWorldAPI/lance-graph/pull/1955), [M1–M7](https://github.com/AdaWorldAPI/lance-graph/blob/main/.claude/plans/d-drisa-decision-memos-v1.md): borrowed SoA and layout/routing lifetime limits.
- [DR source](https://github.com/AdaWorldAPI/lance-graph/blob/main/crates/lance-graph-dt64/src/dr.rs): frame-local positions, duplicate/ambiguity refusal.
- XLOG NotASide, unadmitted derived-mask reuse, OP-14 composition and ADMISSION B: never enable by analogy.

Perform exact symbol census: Bagop API, Quack lowering, mask-risc predicates/folds, order-bearing selection, DT64 field basis, identity source, cold text comparison, scope attestation and result states. Label each IMPLEMENTED, LAB, PROPOSED, BLOCKED or NOT EXPRESSIBLE with a source URL.

## Falsifiers

Two tenants with colliding labels, duplicate lookup keys, 256 permuted DR tiles, count/first/last, absent vs empty text vs blank, large exact decimals, value revision and frame change. Oracle is independent scalar logic over canonical IDs, values, multiplicities and **source order**, not a shared query plan.

Disable checks: remove order witness (first/last differs), collapse duplicates (counts differ), ignore scope (cross-tenant leak), ignore layout/value version (wrong or stale value), enable forbidden gather (admission violation). Compare repeated 256 evaluations after prepared bind to lookup counter = 0. Measure cold bind, warm fold and terminal serialization separately.

## PSCustomObject polyfill and ownership

A named path like Customer.Services.ExchangeOnline binds DT64 property references and yields a Bag projection with the same scope, frame and value witnesses. No nested object allocation until an external PowerShell consumer explicitly needs serialized values. It may not invent its own identity dictionary.

| Downstream | Contract |
|---|---|
| P3 GridCraft | Formula grammar, viewport and spill presentation only |
| P4 lance-graph-java | Cheap typed membrane descriptors, never alternate engine |
| P5 report-rs | Graph workbook source and scoped rendering without second codebook |
| P6 GridCraft | Independent XLSX / scalar / Bagop parity and disable runs |

**Decision log 2026-11-05:** operator intent recorded; ordering, exact match, approximate/wildcard/binary and cross-table admissions remain OPEN/BLOCKED until evidence. **Next implementation PR:** only a lawfully admitted exact-match ordered fold with tests; no implementation in this document PR.

## P7 formula F### method-binding amendment (2026-11-05)

[GridCraft P7 #3](https://github.com/AdaWorldAPI/gridcraft/pull/3) requires formula functions to resolve to canonically owned **F###-identified typed polyfill methods** where admitted. This document's XVERWEIS must be one such method contract, not an independent GridCraft evaluator, a number assigned by a local lookup table, or a second Bagop engine. **The current source audit has not verified the existing F### registry or numeric namespace**; locate the actual authority and MethodRef/fingerprint rules before minting identifiers. A name-only function table cannot confer canonical method identity.

Formula parsing and source spans belong to GridCraft; accepted operators and Bag semantics belong here; Kythe/Codegraph may describe derived source/provenance graphs but cannot create a second F### registry; R2IL machine opcodes retain their distinct execution semantics and must not be aliased to Excel function names. Unsupported methods refuse explicitly, without materializing `Vec<Value>` as a stealth fallback. This amendment adds an admission question, not an owner-approved F### numbering scheme.
