# D-DRMC-1: DR64 typed borrow admission probe

A **lab-only** carrier/binder seam next to Quack's existing developer-name-to-numeric resolution, not a replacement for `lance-graph-quack::bind::Binder`. Uses `lance_graph_contract::class_view::ClassId` and `morton8x8` directly; does not add types to the zero-dependency contract, modify the shared Quack binding code or implement another bulk executor.

Run from the crate's directory (own `[workspace]`): `cargo test`.

## Proved by these tests (once executed)

- A DR64 register is borrowed as exactly `&[u64; 64]` (512 bytes), not copied or hydrated.
- `Identity×Identity` and `Identity×Property` are different **semantic method signatures** even with identical physical bits.
- Method ordinal matters within the same ClassView class.
- A three-valued method binds only with its validity bits, which travel WITH the relation bits (`Operand`); a two-valued method refuses validity; BAG is refused. No FALSE/UNKNOWN collapse and no lost multiplicity.
- Three guards, three types: the **frame** (source, relationship map, coordinates) is a store-issued seal compared by identity, so two stores at equal revisions do not alias and a re-seal is a new frame; the **focus** (motion) and the **scope** (policy) are separate guards that leave the frame valid when they move. Each moved revision is its own refusal.
- Morton 8×8 arithmetic addresses `(x,y)` precisely but does not establish relation truth.

**Explicit non-proofs:** `FrameStore` is a lab stand-in for the source owner (the seal proves issue identity, not that the revisions describe real data); focus and scope are caller-supplied values, `MethodSignature` is a fixture rather than registry-resolved ClassView method metadata, no optimized fold or real foreign relation binder executes, no ACL policy is evaluated. Production admission requires the existing source owner, D-RMAP-1A table bindings, D-RFT/D-QSP method law, index seal and permission checks. A bitmap is not a universal row binding and cannot express all graph path multiplicities.

Next stack: finite population claim/violation ledger, then compiler recipe eligibility.

## D-DRMC-5: associativity of Boolean relation composition (`relcomp`)

The first law the certified-compiler boundary can certify (charter, "Experimental success criterion"). `(R ∘ S) ∘ T = R ∘ (S ∘ T)` over OR/AND is established algebra; the probe tests whether the pieces around it preserve it.

- **Typing.** A `Chain` composes only where each operand's column domain is the next one's row domain, under one frame issue (still current), one focus and one scope, and only for two-valued Distinct relations.
- **Law admission.** `admit_composition_assoc` admits associativity only for the `BooleanCompose` shape over two-valued Distinct relations; Bag (counting) and three-valued composition are other semirings and are refused. An `AdmittedLaw` has no other constructor.
- **Proof.** egglog, given only the admitted rule (both directions), proves every reassociation of a 5-chain and checks its own proofs. Commutation, which associativity does not imply, gets no proof even with both terms in the e-graph.
- **Lowering.** An order lowers to push/compose steps that raise back to the same order. This is a structural check; the compose kernel is checked against an independent pair enumeration.
- **Certificate.** `certify_reassociation` issues a `RecipeCertificate` (carrying egglog's proof text) only with all of the above, and refuses a law reference for another shape, a non-reassociation, a missing code identity, and a compiled digest that is not `lowering_digest` of the checked lowering. A `Chain` borrows its frame store, so the frame cannot advance while the chain is alive.

`cargo run --release --example assoc_orders` reports one dense-to-sparse 5-chain on a 4-vCPU shared Xeon @ 2.10 GHz: all 14 orders give the same relation; work runs from 6 414 to 15 310 row-ORs and the time per evaluation from about 4.4 to 10.8 µs. The cold egglog proof between the two extreme orders is about 12.5 ms (a cold, out-of-fold cost: reported, not a gate).

**Not done:** choosing an order (the specialization-selection gate; the cost above is measured by evaluating every order, which a selector must not do), rebinding a certificate to a second frame (X8), any GF(2)/M4RI reduction, and wiring into R2IL/Quack.

## D-DRCA-1: HHTL16 Cartesian direct routing (`hhtl16`)

Lab test of the 2026-10-28 DR/DT contract: DR maps coordinates and projects by folds; it carries no class meaning and no behaviour. Module `src/hhtl16.rs`, test `tests/hhtl16_projection.rs`, costs `examples/hhtl16_stages.rs`.

- **Address.** A cell `(x, y)` of a bound 64 × 64 register is the existing `morton8x8::Morton8x8` code (`x` row, `y` column), the HHTL16 Cartesian address the contract already has. No second address system.
- **One storage, several orientations.** The relation is the borrowed `BoundDr64` rows. The YZ rotation swaps the code's axes and reads the same storage through the swapped address, the rule of `lance-graph-mask-risc/tests/rotation.rs` (rotation is a coordinate system, not a mask op). The property orientation (`column`) returns one `u64` of row positions; no transposed plane is built.
- **Projections.** Identity × Property (primary), Identity × Identity (optional; the same reads over a binding whose two axes are `Identity`), Property × Property (`cooccurrence`, Boolean `Rᵀ ∘ R` from the canonical rows in outer-product form; `cooccurrence_count`, one pair), cross-table Identity × Property (`FK ∘ R_B` through `relcomp::Chain`).
- **Global → local.** `LocalMap` is the counted preparation (EXO-46): built once per frame, it holds only the pointer table and counts every lookup.

The fixture has two tables (6 employees with 5 skills; 3 departments with 4 sites), a same-table `reports_to` relation and a `works_in` foreign key. Its facts are global-id arrays; the oracle answers each question by scanning them and shares no code with the module. The tables share local ordinals on purpose.

Twelve tests, each against that oracle: Identity × Property cell for cell; YZ rotation over all 4,096 addresses (the fixture is asymmetric, so an identity "rotation" fails); property column; Identity × Identity forward and through the rotation, which reads the inverse without asserting a symmetric relation; Property × Property Boolean and counts; cross-table projection through the FK; equal local ordinals of two tables refused by `Chain` domain typing, and a foreign frame owner refused at equal revision numbers; every set bit decodes back to an existing source fact and none is lost; coordinate and local maps invert exactly; three-valued reads keep TRUE, FALSE and UNKNOWN apart in the property orientation; a changed scope or an advanced coordinate revision refuses; projections make no global→local lookup and no heap allocation.

Disable runs, each turning its named test red and restored from the commit: identity "rotation" (2 tests), wrong column bit (3), co-occurrence overwrite instead of OR (1), an allocation inside `cooccurrence` (1), no range check in `address` (1), no axis check in `cooccurrence` (1).

`cargo run --release --example hhtl16_stages` on a 4-vCPU shared Xeon @ 2.10 GHz, dense register (1,998 set cells), median of 21 runs:

| stage | ns/op | allocs |
|---|---|---|
| prepare `LocalMap` (64 ids, cold, once per frame) | 1,050 | 3 |
| global→local lookup | 7.7 | 0 |
| `bind_dr64` (frame, focus, scope) | 117 | 0 |
| address + inverse | 0.11 | 0 |
| cell read, stored orientation | 0.11 | 0 |
| cell read through YZ rotation | 0.19 | 0 |
| identity row | 0.73 | 0 |
| property column (64 rows) | 10.0 | 0 |
| Property × Property Boolean (64 × 64) | 818 | 0 |
| Property × Property count, one pair | 48 | 0 |
| cross-table `FK ∘ R` through `Chain::eval` | 881 | 4 |

Warm reads made 0 global→local lookups. Single host, single run; reported, not gates.

**Not done:** DR256::WHERE selection over 65,536 addresses and the stay / rebind / masked-read choice; focus larger than 64; any SIMD kernel (the column read is a scalar per-row gather, the largest warm cost after the folds); a real source owner, ACL or ClassView registry; DT method semantics (DR #4 handoff, amendment plan).

## D-DRCA-2: DR256::WHERE × DR#3 (`where_dr3`)

Lab test of the composition `global selection → optional amortized global→local mapping → HHTL16 Cartesian projection → native fold`. Module `src/where_dr3.rs`, test `tests/dr256_where_dr3.rs`, costs `examples/dr256_where_stages.rs`.

- **WHERE is Quack's, once.** `table("node").where_eq("class", "employee").where_eq("sel", 1).bind(..)` (`Draft::bind`, three-valued through `Filter::sql_where`), lowered and executed by mask-risc's `Keep` terminal over 65,536 addresses. Its output, the rows where the criteria are TRUE, is the only selection input. The module evaluates no predicate and depends on neither engine.
- **DT is the existing ClassId-bound schema.** The register's `MethodSignature` (owner `ClassId`, axes, truth, population) is what a selection is projected for and re-admitted against.
- **Global → local.** `project` tests bit `global(l)` of the WHERE mask for each local slot of the table's `LocalMap`: `map.len()` probes, never a pass over the universe, no map search. A selected identity outside the frame is not projected. The frame must carry `LocalMap::digest()` as its `coordinates` revision, so a selection cannot be projected through one coordinate assignment and read under another.
- **Admission.** Every read goes through `LocalSelection::admit`, which re-checks the global source seal (owner, source revision, codebook, address space), the local frame, register frame identity, identity axis, DR method, focus, scope and criteria. The only readable form is the `Admitted` it returns. A `LocalSelection` is a caller-owned value; there is no store or cache.
- **Folds.** The selection is a `u64` over the identity axis, ANDed into reads of the borrowed rows (`selected_count`, `selected_column`, `selected_column_k3`, `selected_cell`). No Cartesian product, no copied row.

**Epistemic absence.** `$null`, NaN, SQL NULL, an unset Arrow validity bit and K3 UNKNOWN are one logical value: no TRUE/FALSE judgment has been established. It does not say the entity or relationship does not exist, and no finer epistemic state (CausalEdge64 Epi5, under its own evidence contract) is derived from it. Consequences here:

- WHERE selects TRUE. A slot that is not selected is FALSE **or** UNKNOWN, and `Admitted` offers no complement, because the complement of a selection would read UNKNOWN as FALSE.
- `NOT UNKNOWN` is UNKNOWN. A negated selection is a second WHERE over the negated criteria through the same `sql_where`, with its own criteria identity; reusing one selection as the other is refused.
- A three-valued property keeps the register's own value + validity form (`BoundDr64::validity`). `selected_column_k3` returns `(value, validity)` over the selection, with the value masked by validity, so a stale payload under an UNKNOWN cell is never TRUE. The Boolean folds read payload bits as presence and refuse a three-valued or bag register (the rule `hhtl16::cooccurrence` and `relcomp::Chain::new` apply).
- Physical encodings (Quack validity planes, Arrow bitmaps, NaN payloads) are adapter concerns; none adds a semantic category.

Oracle: the source is plain Rust values per global address (`class`, `sel: Option<i32>`, `live`). Verdicts are a `match` on `Option`, local positions are `position()` in the frame's source-order id list, and registers are built from those positions, not through the map. It shares no code with Quack, mask-risc, `hhtl16` or `where_dr3`. NULL rows carry the payload the predicate tests for, so an engine that read NULL payloads would keep them.

Thirteen tests against that oracle:

- the global WHERE, whole mask over all 65,536 addresses, and its in-frame selection (TRUE, FALSE and UNKNOWN all occur in the frame; neither FALSE nor UNKNOWN is selected; most selected identities lie outside the frame);
- equal local ordinals of two tables: the employee selection projected through the department map selects no department, and reading it against a department-row register is refused (`AxisMismatch`, `DifferentFrame`);
- selected Identity × Property cell by cell, by column and by count, and the same relationships through the YZ rotation;
- UNKNOWN: the negated WHERE keeps exactly the oracle's FALSE rows, a NULL row is kept by neither WHERE (globally and in the frame), the complement of a selection differs from its negation, one selection is refused as the other, the three-valued column matches the oracle in value + validity form including with a stale payload under an UNKNOWN cell, and the Boolean folds refuse that register;
- a source change after WHERE (refused, then rebound to the oracle);
- a scope or focus change with identical coordinates (refused; the unchanged register admits);
- selected employees' sites through a witnessed FK (`Chain`), then a changed referenced property (refused, rebound);
- a reseal at equal numbers and a new coordinate assignment (refused; the old map is refused under the new frame, and the new map projects to the new ordinals);
- a foreign owner at equal numbers, other criteria, another DR method, and a new codebook (refused; the stale binding run silently keeps nobody);
- empty and full selections, plus a malformed width and an id outside the universe;
- the #1792 shape: a detached `Keep` reused after its criteria input changed answers silently with the old population; projected under the old seal it is refused;
- projection under a stale global seal, a foreign owner's seal or a resealed local frame;
- projection, admission and folds allocate nothing.

Sixteen disable runs, each restored from the commit and each turning at least one test red: the two projection-time seal checks, `MapNotFrame`, the complement projected as the selection, the eight admission guards (global, local, frame issue, axis, method, focus, scope, criteria), the three-valued column reporting UNKNOWN as known, its value not masked by validity, `selected_count` ignoring the selection, and the Boolean folds accepting a three-valued register.

The original #1792 control is unchanged and still reproduces: `cargo run --release --example p2_quack_where -- --controls` in `crates/drf-focus-probe` prints `reused R2 1032 vs fresh 31`.

`cargo run --release --example dr256_where_stages` on a 4-vCPU shared Xeon @ 2.10 GHz (avx512f), median of 21 runs. The single-WHERE path was measured back to back against the previous two-program path (TRUE and FALSE `Keep`, commit `cbc7930`) on the same fixture, alternating old/new three times; the two quiet pairs are shown (the third pair was disturbed by host load in both arms).

| stage | previous (two WHERE programs) | single WHERE | allocs |
|---|---|---|---|
| WHERE `Keep`, 65,536 rows | 41.7 / 40.2 µs | 20.1 / 20.0 µs | 0 |
| end-to-end fresh read | 43.2 / 41.5 µs | 20.6 / 20.6 µs | 0 |
| warm total, N = 8 | 42.9 / 42.4 µs | 21.4 / 21.0 µs | 0 |

Other stages of the single-WHERE path (two runs): `Draft::bind` 465–513 ns (10 allocs), `lower` 472–482 ns (7 allocs), `project` 912–926 ns, `admit` 76–82 ns, warm read 84–91 ns, `selected_count` 14–16 ns, all 0 allocs. Cold: `LocalMap::prepare` 891–963 ns (3 allocs); encoding the 65,536-row fixture into planes and lanes ~600 µs (fixture, once).

Totals for N compatible reads, single WHERE (µs, quiet pairs; fresh = WHERE + project + admit + read each time; warm = WHERE + project once, then admit + read; rebind = the owner advances before every read, the old selection is refused, then WHERE + project + admit + read):

| N | fresh | warm | rebind | fresh / warm |
|---|---|---|---|---|
| 1 | 20.7 / 21.1 | 20.6 / 20.7 | 20.8 / 21.1 | 1.00 / 1.02 |
| 2 | 42.4 / 42.4 | 21.1 / 20.6 | 41.3 / 41.2 | 2.01 / 2.05 |
| 4 | 83.2 / 80.1 | 21.0 / 20.1 | 85.7 / 81.7 | 3.96 / 3.99 |
| 8 | 170.8 / 165.1 | 21.4 / 21.0 | 169.4 / 168.6 | 7.96 / 7.86 |
| 64 | exceeds 1,000 µs (failed gate) | 25.2 / 24.1 | exceeds 1,000 µs (failed gate) | — |

One WHERE halves the fresh read. Warm reuse breaks even at N = 1 and pays from N = 2; at N = 64 only warm reuse stays inside the fold budget. Rebind costs the same as fresh. Microbenchmarks, reported, not gates.

**Open limitations.**

- **The seal attests the caller's claim, not the mask's provenance** (independent gate, D-NFR S5). A detached `Keep` projected under a new seal after its source changed is accepted (pinned by the #1792-shape test, so a fix turns it red). Closing it needs the producer to bind a mask to the seal it was computed under; Quack's bound `Query` has no frame today.
- `Criteria` is a caller-supplied id; Quack exposes no digest of a bound query.
- The global codebook revision rides in `FrameRevisions::relation_map`; the owner, ACL and scope are the lab `FrameStore`, `Scope { id }` equality.
- A negated selection costs its own WHERE, which is the price of `NOT UNKNOWN` staying UNKNOWN.
- One 64-slot frame per table; no larger focus, no DR256 resident register, no reuse of a scattered (`ScatterOrU32`) mask.
- `project` is 64 scalar probes (~1 µs); admission (~76–82 ns) is not broken down.

## D-DRCA-3: DR #4 method routing (`dr4`)

The first DR #4 falsifier of D-DRDT §5 (#1835): two methods on one cell with the same read/write sets and a different effect order must stay distinct through routing, and a dropped guard must refuse. Module `src/dr4.rs`, test `tests/dr4_method_routing.rs`, plan `.claude/plans/d-drca-3-dr4-method-routing-v1.md`.

- **DR carries DT's values opaque.** `MethodKey`, `GuardWitness` and `EffectOrderRef` are 16-byte references that `route` compares for equality and never interprets. The method key is DT's to mint (D-CMH §2.1); the test mints its own, named as such. `Resolution` (class, frame, methods) is DT's answer handed in by the caller; this crate has no DT resolver (that is D-CMH-1's `PreparedClass`).
- **One entry per input method, in input order.** Key, signature revision, read and write field sets, guard witness and effect-order reference pass through unchanged, with the register's frame on the result. Nothing is merged: two methods whose effects commute are still two entries, because merging is the law owner's decision.
- **Refusals, in order:** a stale register frame (`Frame(..)`); a resolution made under another frame issue, including another store at equal numbers (`ResolvedUnderOtherFrame`); the resolution or a method for another class (`ForeignClass`); a duplicate key (`DuplicateMethod`; equal read/write sets are not a duplicate); coordinates outside 64 × 64 (`OutOfRegister`); a numeric fold (`UnsupportedAlgebra`); a presence read over a three-valued register (`NotBoolean`); a declared guard with no witness (`GuardDropped`) or another one (`GuardMismatch`). DR checks witnesses; it evaluates no guard.
- **Values** come from the existing register reads: cell, row and column presence, and K3 value + validity with the value masked by validity.

Oracle: the DT stub in the test holds each method's real effects and guard. The effect-order law is ndarray #395's `Effect` / `run_effects` / `commutes` / `check_reorder` (`examples/pof_antennas/order_law.rs`), ported verbatim. In the core fixture `[Set(1), Add(1)]` and `[Add(1), Set(1)]` give 2 and 1 and the swap is refused, so the two references DR keeps apart are provably different methods; the commuting control `[Add(2), Add(3)]` and its reverse give equal results and the swap is legal, and routing still keeps both. Values are compared against plain scalar loops over the register words.

Twelve tests; seven disable runs, each restored from the commit and each turning its named test red: entries collapsed by equal read/write sets (3 tests red), a dropped guard accepted, a mismatched witness accepted, the resolution frame not checked, a method's class not checked, a duplicate key accepted, the K3 value not masked by validity.

**Not done.** The DT resolver (D-CMH-1), a minted `MethodKey`, guard evaluation, effect execution, DR #4 over a `LocalSelection` (D-DRCA-2) or a `Chain` composition, production placement.
