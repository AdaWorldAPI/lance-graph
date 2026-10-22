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
