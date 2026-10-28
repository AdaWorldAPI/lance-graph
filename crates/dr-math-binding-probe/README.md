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
