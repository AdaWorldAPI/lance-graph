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
