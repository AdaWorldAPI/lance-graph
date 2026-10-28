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

- **WHERE is Quack's.** `table("node").where_eq("class", "employee").where_eq("sel", 1).bind(..)` (`Draft::bind`, three-valued through `Filter::sql_where`), lowered and executed by mask-risc's `Keep` terminal over 65,536 addresses. The FALSE set is the same leaves under `NOT`, also through `sql_where`. A row in neither is UNKNOWN. The module evaluates no predicate and depends on neither engine: it takes the two `Keep` masks as borrowed words.
- **Global → local.** `project` tests bit `global(l)` of both masks for each local slot of the table's `LocalMap`: `map.len()` probes, never a pass over the universe, no map search. A selected identity outside the frame is not projected. The frame must carry `LocalMap::digest()` as its `coordinates` revision, so a selection cannot be projected through one coordinate assignment and read under another.
- **Admission.** Every read goes through `LocalSelection::admit`, which re-checks the global source seal (owner, source revision, codebook, address space), the local frame, register frame identity, identity axis, DR method, focus, scope and criteria. The only readable form is the `Admitted` it returns. A `LocalSelection` is a caller-owned value; there is no store or cache.
- **Folds.** The selection is a `u64` over the identity axis, ANDed into reads of the borrowed rows (`selected_count`, `selected_column`, `selected_column_k3`, `selected_cell`). No Cartesian product, no copied row.
- **K3.** WHERE keeps TRUE. UNKNOWN is carried separately and never folded into FALSE; the negation of a selection is its FALSE set. The Boolean folds read payload bits as presence, so they refuse a selection admitted against a three-valued or bag register (the rule `hhtl16::cooccurrence` and `relcomp::Chain::new` apply); `selected_column_k3` reads validity and serves both.

Oracle: the source is plain Rust values per global address (`class`, `sel: Option<i32>`, `live`). Verdicts are a `match` on `Option`, local positions are `position()` in the frame's source-order id list, and registers are built from those positions, not through the map. It shares no code with Quack, mask-risc, `hhtl16` or `where_dr3`. NULL rows carry the payload the predicate tests for, so an engine that read NULL payloads would keep them.

Thirteen tests against that oracle:

- the global selection, whole mask over all 65,536 addresses, and its in-frame TRUE / FALSE / UNKNOWN (all three non-empty; most selected identities lie outside the frame);
- equal local ordinals of two tables: the employee selection projected through the department map selects no department, and reading it against a department-row register is refused (`AxisMismatch`, `DifferentFrame`);
- selected Identity × Property cell by cell, by column and by count, and the same relationships through the YZ rotation;
- UNKNOWN: kept apart from FALSE in the selection, in its negation and in a three-valued property column, and the Boolean folds refuse that three-valued register;
- a source change after WHERE (refused, then rebound to the oracle);
- a scope or focus change with identical coordinates (refused; the unchanged register admits);
- selected employees' sites through a witnessed FK (`Chain`), then a changed referenced property (refused, rebound);
- a reseal at equal numbers and a new coordinate assignment (refused; the old map is refused under the new frame, and the new map projects to the new ordinals);
- a foreign owner at equal numbers, other criteria, another DR method, and a new codebook (refused; the stale binding run silently keeps nobody);
- empty and full selections, plus malformed widths, TRUE ∧ FALSE overlap and an id outside the universe;
- the #1792 shape: a detached `Keep` reused after its criteria input changed answers silently with the old population; projected under the old seal it is refused;
- projection under a stale global seal, a foreign owner's seal or a resealed local frame;
- projection, admission and folds allocate nothing.

Disable runs, each restored from the commit and each turning at least one test red: the two projection-time seal checks, `MapNotFrame`, the TRUE ∧ FALSE check, TRUE/FALSE swapped, the eight admission guards (global, local, frame issue, axis, method, focus, scope, criteria), UNKNOWN computed as "not TRUE", the three-valued column reporting UNKNOWN as FALSE, `selected_count` ignoring the selection, and the Boolean folds accepting a three-valued register. The first pass found the two projection-time seal checks covered by no test; the projection test above was added for them and both now go red.

The original #1792 control is unchanged and still reproduces: `cargo run --release --example p2_quack_where -- --controls` in `crates/drf-focus-probe` prints `reused R2 1032 vs fresh 31`.

`cargo run --release --example dr256_where_stages` on a 4-vCPU shared Xeon @ 2.10 GHz (avx512f), median of 21 runs, two runs. Per-op rows below are batched over 40–256 calls where a single call is below timer resolution.

| stage | run 1 | run 2 | allocs |
|---|---|---|---|
| cold: encode 65,536 source rows into planes and lanes (fixture) | 409 µs | — | 5 |
| cold: `LocalMap::prepare` (64 ids) | 642 ns | — | 3 |
| cold: `FrameStore::new` + seal | 786 ns | — | 2 |
| `Draft::bind` | 308 ns | — | 10 |
| `lower` | 318 ns | — | 7 |
| WHERE TRUE + FALSE `Keep`, 65,536 rows (lowered once) | 42.2 µs | 70.5 µs | 0 |
| `project` (64 probes) | 1,023 ns | 1,028 ns | 0 |
| `admit` | 65.8 ns | 80.1 ns | 0 |
| warm read (`admit` + `selected_count`) | 70.7 ns | 89.9 ns | 0 |
| rotated cell read, selection applied | 10.7 ns | 9.6 ns | 0 |
| `selected_count` | 13.2 ns | 30.3 ns | 0 |
| `selected_column` | 15.0 ns | 16.6 ns | 0 |
| end-to-end fresh read | 44.3 µs | 73.3 µs | 0 |

Totals for N compatible reads (µs; fresh = WHERE + project + admit + read each time; warm = WHERE + project once, then admit + read; rebind = the owner advances before every read, the old selection is refused, then WHERE + project + admit + read):

| N | fresh | warm | rebind | fresh / warm |
|---|---|---|---|---|
| 1 | 41.6 / 72.3 | 42.9 / 70.2 | 42.1 / 70.3 | 0.97 / 1.03 |
| 2 | 85.9 / 142.4 | 43.8 / 70.0 | 86.5 / 140.8 | 1.96 / 2.03 |
| 4 | 173.0 / 281.9 | 41.9 / 69.8 | 168.9 / 289.2 | 4.13 / 4.04 |
| 8 | 347.9 / 586.3 | 43.0 / 69.7 | 340.5 / 589.6 | 8.10 / 8.41 |
| 64 | exceeds 1,000 µs (failed gate) | 45.9 / 45.6 | exceeds 1,000 µs (failed gate) | — |

Warm reuse breaks even at N = 1 and pays from N = 2. At N = 64 only warm reuse stays inside the fold budget. Rebind costs the same as fresh: invalidation is a cheap refusal followed by the full work. Absolute numbers move by up to 1.7× between runs on this shared host; only ratios within a run are used. These are microbenchmarks, reported, not gates.

**Open limitations.**

- **The seal attests the caller's claim, not the mask's provenance.** A detached `Keep` projected under a new seal after its source changed is accepted (pinned by the #1792-shape test, so a fix turns it red). Closing this needs the producer to bind a mask to the seal it was computed under; Quack's bound `Query` has no frame today (D-NFR S5).
- `Criteria` is a caller-supplied id; Quack exposes no digest of a bound query.
- The global codebook revision rides in `FrameRevisions::relation_map`; the owner, ACL and scope are the lab `FrameStore`, `Scope { id }` equality.
- The FALSE set costs a second WHERE program. A terminal producing both K3 masks in one pass is not built.
- One 64-slot frame per table; no larger focus, no DR256 resident register, no reuse of a scattered (`ScatterOrU32`) mask.
- `project` is 64 scalar probes (~1 µs); admission (~65–80 ns) is not broken down.
