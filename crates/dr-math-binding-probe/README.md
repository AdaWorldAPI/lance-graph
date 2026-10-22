# D-DRMC-1: DR64 typed borrow admission probe

A **lab-only** carrier/binder seam next to Quack's existing developer-name-to-numeric resolution, not a replacement for `lance-graph-quack::bind::Binder`. Uses `lance_graph_contract::class_view::ClassId` and `morton8x8` directly; does not add types to the zero-dependency contract, modify the shared Quack binding code or implement another bulk executor.

Run from the crate's directory (own `[workspace]`): `cargo test`.

## Proved by these tests (once executed)

- A DR64 register is borrowed as exactly `&[u64; 64]` (512 bytes), not copied or hydrated.
- `Identity×Identity` and `Identity×Property` are different **semantic method signatures** even with identical physical bits.
- Method ordinal matters within the same ClassView class.
- K3/NULL and BAG are refused for a bare Boolean bitmap; no FALSE/UNKNOWN collapse and no lost multiplicity.
- A stale source, relationship map, coordinate frame, focus or scope is refused separately.
- Morton 8×8 arithmetic addresses `(x,y)` precisely but does not establish relation truth.

**Explicit non-proofs:** `BindingWitness` values are caller-supplied (not signed or independently attested), `MethodSignature` is a fixture rather than registry-resolved ClassView method metadata, no optimized fold or real foreign relation binder executes, no ACL policy is evaluated. Production admission requires the existing source owner, D-RMAP-1A table bindings, D-RFT/D-QSP method law, index seal and permission checks. A bitmap is not a universal row binding and cannot express all graph path multiplicities.

Next stack: finite population claim/violation ledger, then compiler recipe eligibility.
