# D-QSP — ClassView × Quack: PostgreSQL semantics, shape-polymorphic execution and DuckDB benchmark (v1)

**Plan ID:** D-QSP (Quack SQL/PostgreSQL semantic parity). Dependencies: D-RFT (relalg × thought-git), D-LFA (optional flow driver); existing D-SQLP, D-RA, D-CMR retained. No duplicate ownership.\n\n**Status:** PROPOSAL / DOCS ONLY, 2026-10-21. No runtime API, SQL parser, code, benchmark result, storage migration, or implementation ownership is granted by this document.

**Priority order:** (1) lance-graph-relalg semantic depth, (2) lance-graph-thought-git replay/frontier, (3) ClassView method binding and SQL parity, (4) parallel, *bounded* lance-graph-flow assimilation census/probe. Odoo deferred. This plan supersedes no established law and does not change #1666's owner.

**Purpose:** Treat PostgreSQL SQL behavior as an independently checkable set of class/method laws executable by Quack over multiple admitted physical shapes. PostgreSQL is the **correctness oracle**, DuckDB the **analytical performance comparator**. Do not fork PostgreSQL, implement an FDW, adopt an ORM, or build a second SQL evaluator as the goal. A SQL-shaped front end may be a test harness only.

## Existing evidence / boundaries

- 'crates/lance-graph-quack/src/lib.rs' already describes scan/filter/project/group/aggregate and COUNT, EXISTS, ALL, SUM, MIN, MAX, grouped key operations including GroupSumI32 and GroupSumViaI32. It **lowers** programs. Execution stays in 'lance-graph-mask-risc'; not a new interpreter.
- 'crates/lance-graph-relalg' currently centers on single-population set expressions, normalization and R2IL lowering. Its 'join_body' only diagnoses alpha-acyclic query hypergraphs; it is not a general join executor or proven PostgreSQL semantics.
- 'crates/lance-graph-contract/src/set_rules.rs' carries guarded semantics: Arith, Truth, SumFamily, Population, universe. 'set_algebra.rs' complement normalization must be tested with three-valued Kleene NULL inputs BEFORE semantic equivalence can be claimed.
- #1659 shared-domain lanes; #1666 DR256 nested IAM group closure and lazy per-256-user pages/population chains are **lab evidence only**. #1666 reports ~124 microseconds for 16 license counts at 65k users, with ~3 microseconds spent on AND/count *after population construction*. Earlier CSR prepare still has a millisecond gate. Not a general SQL/PG or DuckDB win.
- SAP CATS already uses direct field/BAPI ordinals; KEEP_DIRECT remains a legal route, not forced into DR64/DR256.
- Existing plans: '.claude/plans/relalg-v1.md', '.claude/plans/thought-git-basics-v1.md', '.claude/plans/thought-git-api-v1.md', '.claude/plans/d-cmr-concept-chain-normalization-v1.md'. Coordinate with EXO-83, EXO-86, EXO-98, EXO-101, EXO-102, EXO-121 and EXO-50.

## Contract, not a new engine

Conceptual call: ClassView resolves (ClassId, MethodOrdinal, declared operation law); Quack Binding resolves snapshot/version, typed variable domains, stable codebook, focus, validity and owned/borrowed lanes; optional DR maps numeric coordinates to a physical route. 'relalg/R2IL' composes and lowers only lawful rewrites into the ONE mask-RISC evaluator. Do not put a behavior choice into the binding store or use strings as hot-path identity. 'ClassView::actions' remains a desired *defaulted accessor over existing action metadata*, never a new behavior owner, action registry or commit executor.

The declaration fixes semantic laws **per method/operation**, not one 'Semantics(ClassId)' for all methods. Source universe size and focus/source generation are BIND-time values; method semantics are not cost-selected. Existing 'set_rules::Semantics' applies where its vector is sufficient. Do not conflate 'hotplug::register_law_for' byte-register laws with this semantic vector.

Choice among routes is allowed only if an equivalence fixture and law witness admit it; e.g. direct SAP ordinals, ordered runs, sparse positions, DR64/DR256, dense masked populations. Neither 32×128 nor 64×64 is universal or a persisted ABI decision. A 32×128 Boolean tile is 512 bytes (32 subject rows, 128 target slots); a 64×64 tile is also 512 bytes (64-bit rows). A tile does NOT represent bag multiplicities, NULL validity, Wikidata qualifiers, or evidence without additional lawful carriers. Tiles are shared per focus; count cold binding, per-tile witness, auxiliary metadata and scratch.

## D-QSP work items: staged PostgreSQL parity, always with counterexamples

| Gate | Fixture | Oracle and refusal |
| --- | --- | --- |
| D-QSP-0 semantic normalization | three-valued Kleene rows including NULL; predicate complement, absorption, NOT IN | compare both as-written and normalized against PostgreSQL and independent row oracle. Refuse unsound rewrite selectively, not all nullable algebra |
| D-QSP-1 single-table | WHERE, CASE, GROUP BY, COUNT(*), COUNT(expr), COUNT DISTINCT, SUM empty/all-NULL, MIN/MAX, HAVING | PostgreSQL exact result plus error/type behavior; do not replace empty SUM NULL with zero or lose validity |
| D-QSP-2 cross-table | INNER, LEFT, SEMI/EXISTS, anti/NOT EXISTS, duplicates, key fan-out, absent FK, NULL join key | every output checked with bags/multiplicity, NULL extension and SQL K3; 'Boolean membership' ≠ general SQL JOIN |
| D-QSP-3 scalar laws | integer overflow/casts, numeric scale/rounding, float association, comparison, collation/order | declare what is supported exactly; typed refusal outside admitted scope; no false PG parity claim |
| D-QSP-4 recursion/updates | recursive reachability with cycles, negative dependency stratification, evolving snapshot | datafrog oracle; snapshot invalidation, multi-support deletion, independent references; no implicit MVCC parity |
| D-QSP-5 broader conformance | sqllogictest-style differential cases, randomized counterexamples, representative TPC-H analytical subset | both exact PostgreSQL-parity and fair performance vs DuckDB; scope exclusions published |

A1 from the ClassView discussion maps to D-QSP-0. SetAlgebra's unconditional two-valued rules must not slip into a SQL K3 path. PostgreSQL correctness is per *declared scope*; transaction isolation/MVCC, DDL, time zones, collation, error behavior and client protocol are separate future strata, never silently covered.

## Fair physical performance gate

Run an identical query and same data/correctness against PostgreSQL, DuckDB and Quack; explicitly separate: ingest, source prepare, codebook bind, tile/run building, first-touch page, steady-state fold, and result export. Compare cold and warm caches, repeated questions per focus, key skew, varying density, 1/8/32/64/256 selected tiles, changing relation edges, and output-size dominance. No per-user nanosecond headline may hide millisecond preparation. The win condition is a stated query corpus with median and tails and preserved equality; speed on IAM alone is not maturity.

For one source/focus, contrast:
- KEEP_DIRECT projected ordinals (especially SAP).
- DR32×128 Boolean membership (first low-commitment fixture, 512 B/tile), DR64×64 (512 B), DR256×256 (8 KiB).
- Borrowed ordered runs and sparse positions with lazy tile pointer tables, against full mask rebuild.
- Whole-focus versus OU/team focus. A physical proxy chain stays inside its Dn128 OU locality; typed membership may cross OUs with witnessed hop. Proxy metadata does not steal slot 255 or stand in for closure.

Preserve ADMISSION B: no value-directed gather across foreign tables during PREPARE. Zero-copy means buffers can be borrowed or shared, **not** that PostgreSQL heap tuples magically become Lance buffers. Frozen codebook witness needs a contract owner and version-attested round-trip, Eq semantics and order preservation proof separately for Lt. Exhaustive 256-local-slot validity, 257th explicit escape/refusal.

## Admission milestones (canonical D-QSP-* IDs)

The **D-QSP-0..5 rows in the staged parity table above** are the canonical milestone IDs: D-QSP-0 K3 normalization safety; D-QSP-1 single-table semantic parity; D-QSP-2 cross-table join/bag/NULL semantics; D-QSP-3 scalar types and overflow/collation; D-QSP-4 bounded recursive/mutation semantics; D-QSP-5 broader SQL conformance and fair DuckDB performance.

Shape conformance across direct, sparse, DR32×128 and DR64/256 is a **separate binding/cost stratum** within D-QSP-1/2, never an alternative semantic law. The replay and frontier dependency is D-RFT-5 (with existing D-TGB work), not an untracked additional D-QSP milestone. The datafrog recursion proof remains D-RFT-3 and a prerequisite to D-QSP-4. Avoid duplicate D-* IDs for related checks.

Every milestone has independent verdicts for semantic correctness, binding legality/provenance, performance, memory and cost of mutation. A PASS in one does not waive a FAIL in another. No production write path, new ABI, PostgreSQL compatibility or 'faster than DuckDB' claim is authorized by this plan.
