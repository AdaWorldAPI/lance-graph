# 2026-10-10 — D-XGP-1: CATS behind Quack's Binder, addressed in canonical names

**Status:** TEST-PINNED on branch `ccr-f5674497-orfkox` (unmerged). Plan: `.claude/plans/cross-glove-business-parity-v1.md` §C.8.

- `CatsBinder` resolves by name the same lanes `CatsQuery` hard-codes; a fold over the bound lanes equals `CatsQuery`'s sums across word tails (`lance-graph-sap/tests/binder.rs`).
- Through `OntologyRegistry` + `CanonicalBinder`, a `Draft` in canonical URIs binds to the identical `Query` and `Program` (`lance-graph-glove-parity/tests/canonical_front.rs`).
- The registry key is `(bridge_id, public_name)` and a re-proposal with the same checksum is idempotent; the checksum is the caller's content hash and must cover the URI.
- `MappingRow::active` has no setter in the registry, so no guard on it can be tested yet; the front does not consult it.
- `0x000C` is now claimed by `HiroPort` (OGAR #333, unmerged); the next free app prefix is `0x000D`.

OPEN: canonical attribute URI minting for `0x0103`; cross-glove algebra parity needs an Odoo population binder.
