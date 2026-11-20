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

> **Note (2026-11-11, operator rulings, `dr-widefieldmask-dt-classview-v1.md` §1, §1c).** "Quack Binding resolves snapshot/version" reads: it resolves the version of the *mapping* (identities, properties, source positions, `MapTable` / D-DR-VMAP, #1923); contents stay with their source owner and are not versioned by the binding. A PostgreSQL tuple is a reading over one shared index (`Tuple(i) = (A[i], B[i], C[i])`), never a stored row; PostgreSQL is the compatibility surface and correctness oracle above the shared coordinate substrate, not an engine rebuilt beneath it (EXO-174 states the same for the workbook). D-QSP-2 joins follow D-DYNBAG's bag algebra (ℕ-compose for multiplicity, Boolean only for existence, chosen by demand). D-QSP-4 "snapshot invalidation" follows each artifact's real dependencies: a pure address binding stays valid across a content change while its mapping and interpretation stay valid; a stored recursion result, a precomputed WHERE mask, a join assignment from mutable keys or an order index depends on content and needs a matching change witness or a recomputation, never reuse because the coordinates are unchanged. Recursion itself is the semi-naive fold already in xlog (`Cube::closure`). None of this requires storing historical contents.

## The virtual row: a PostgreSQL row as a register binding (2026-11-20)

**Status:** WORKING-MODEL plus one queued probe (D-QSP-VROW). It grants no production read path and no PostgreSQL compatibility claim; serving reads to a PostgreSQL client still needs a `DECISION:` issue (P-EXO-11).

A row that a PostgreSQL client sees is never built. It is a reading over source-owned lanes at one position, through one DR frame:

- **The row's identity** is a position in the frame. `identity[index]` resolves through the DR map (`MapTable::{map, map_at, locate}` → `LocalMap`, `Frame`), inbound with `local(global)` and outbound with `global(local)`, in the same frame.
- **A column** is a field of the class's ClassView. `identity[index].key` is the field's predicate identity, which `LaneBinding` records per bound field at bind. `identity[index].value` is the lane value at the position, read in place. Text is a code through `CodeTable` and is decoded only at the wire.
- **NULL** is the field's presence plane, not a value in the lane.
- **A result set** is a frame version plus a selection of positions. Duplicates in the selection are bag multiplicity, as in D-VAS R12 (#2069: selection `[1, 1, 3]` gives bag count 3 and distinct count 2).

This is the Untersetzer form, not a glove. No cube terminal that builds rows (`Cube::materialize_rows`) is on this path. The only materialization is the wire encoding of the returned cells, after the fold.

### One check, every lane: the `borrow_all` pattern

The read side uses the existing DT64 socket, not a new one:

1. `LaneBinding::borrow_all_present` runs `LaneBinding::check` once (frame, route, basis, mapping, scope, row order) and lends every bound lane with its presence plane, in field order, with no allocation. Reading N fields through `LaneBinding::lane` instead re-runs the check per field (N² source lookups); `borrow_all` does N. Precedents: `xlog-coverage-probe/tests/group_by.rs`, `quack-boundary-lab`, `vrl-recipe-probe`.
2. `LaneBinding::selected` lends the positions of a selection that the source issued under the same frame, row order and scope, refusing `SelectionSpace` / `SelectionScope` before anything is borrowed.
3. The fold runs over the lent lanes at the lent positions (mask-risc `execute_sel`); the wire encoder reads the same lanes at the same positions.

Today `act_over` (`lance-graph-mailbox-lanes/src/act.rs`) takes its N operand lanes through `LaneBinding::lane`, one call per read field, and the positions through `LaneBinding::selected` when a selection is bound: N checks per call, N + 1 with a selection, each walking every bound lane. It also builds a row mask from the positions on every call. The probe measures both; a single call that lends lanes, presence and positions under one check is a candidate for DT64's owner, not part of this plan.

### Ownership stays where it is

The virtual row adds no owner. The lanes are lent by the source that owns them, under that owner's publication: for a mailbox that is `MailboxLanes` over one `Published` borrow of a `MailboxSoA` (D-VAS-A1..A3), whose frame ends with the borrow. The adapter is a reader under the owner's scope. It holds no `mailbox_owner`, no second mailbox or copy of one, no ownership field on any DTO, and no positions of its own: selections are issued by the socket (`issue_where`, `issue_order`, `issue_concat`), never handed in. A write, when one is ever admitted, goes the existing way: an `Intent` applied through the owner's `MailboxSoA::write_row` on behalf of `SoaEnvelope::mailbox_owner()`, after the publication has ended. A second delegation path beside that one is out of scope for this plan.

### What #1943 already proves, and what is new

D-GLOBAL-WHERE P0 (#1943) establishes the two-way contract this needs: DR routing is exact in both directions (F1–F5: joining by local index gives 32, the oracle and the DR-aligned path give 14); a stale, foreign or future frame is refused (F10, F11); a routed selection reused within its frame costs 5 µs where routing per call costs 650–970 µs (W7); folds allocate nothing (F14, F15); multi-tile 16 × DR256 agrees with DuckDB on 336/336 answers. The open owner points stand as #1943 lists them: `MapTable::locate` scans every listed tile per id (all of its cold losses), one selection for several methods under one scope (EXO-46), and scale ≥ 10M rows.

New here is only the chunk size. A selection is read in chunks, as DuckDB reads a vector through its selection vector (2048 rows by default). A chunk's positions, the mask words it touches and the encoder's scratch must stay in L1; nothing else is sized per chunk, because the mapping and the binding are reused for the frame version. A content-dependent selection (a WHERE mask or an order index) is reused only with a matching change witness or after recomputation, as the 2026-11-11 note above requires.

### D-QSP-VROW probe

- **Fixture:** one Zammad table read by the zammad-rs shadow (`tickets`), its lanes and presence planes lent through `borrow_all_present`, one issued selection; PostgreSQL holds the same rows.
- **Equality:** the wire cells equal PostgreSQL's for the same query, as a multiset without ORDER BY and as a sequence with it (D-QSP-H rules); NULL from the presence plane, never from a sentinel value.
- **Disables, each red:** lanes read by local index instead of the DR route; a selection from another frame admitted; presence ignored; the round trip `global(local(g)) == g` broken by one swapped coordinate.
- **Measured:** checks per request (one is the target), allocations per chunk, bytes copied from source lanes (zero is the target), and fold time per chunk at 256, 1024, 2048 and 4096 positions, against the L1 size of the machine. Per-operation results are in nanoseconds and batches in microseconds; a millisecond result is a failed gate.
- **Not measured here:** client protocol, transactions, DDL and writes; they stay separate strata.

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
