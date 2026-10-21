# 2026-10-21 — D-CMR-5 re-scoped: preimage via frozen coordinates, still BLOCKED

**Status:** DECISION (operator, 2026-10-21, in session `session_013z2e1YXkm6o6HNrDn5BPWA`; also relayed on EXO-50 as `gpt-ada-01/003`). **Basis:** read-only check of D-CMR-5 against the global→local registers (this session; `crates/mork-pathmap-probe/src/rmap.rs`, `crates/crepe-plan-probe/examples/walk_tile.rs`), the ADMISSION ruling (`2026-10-20-six-decisions.md` §1).

Operator's words: "Der entscheidende Unterschied ist, ob ein Wert nur in eine bereits festgelegte Koordinate übersetzt wird oder darüber bestimmt, welche Zeile einer Relation gelesen wird."

| question | ruling |
|---|---|
| Q1: seek `f`'s attested sorted key with range bounds taken from `S` (or a previous fold) | **C, not admitted.** Foreign values steer the reads of `f`; a sorted index and doing it in preparation change nothing. Independent literal / intrinsic same-table key-range seeks on `f` stay B. |
| Q2: translate `S`'s global ids into tile coordinates through a frozen codebook | **B, conditionally.** Pure representation translation, provided it only produces an `S` membership plane, triggers no relational access to `f`, and the mapping is attested, version-bound and fully accounted. |

- **Design direction (not an admission):** a once-built, version-bound projection of `f` (sequential same-table scan, frozen-codebook endpoint translation) may be reused across queries; `S` is translated into the same coordinates; `f⁻¹(S)` is a co-addressed transposed fold. Reading `f` selectively by the values found in `S` stays forbidden.
- **Two existing registers:** R1 (`rmap` `Focus` + `Csr` + `rebind`) seeks with escapes / values from another column, so it stays option C and is not legalized. R2 (`walk_tile`, D-RU-6) has the admissible shape, but its witness is the probe-local `rmap::Witness`.
- **Many-to-one:** not an error for the set-valued preimage; `Count(f⁻¹(S))` over distinct source coordinates is valid. Refuse a bag-sensitive aggregation that set deduplication would destroy, unless a separately proven multiplicity-preserving aggregation exists.
- **Still BLOCKED:** no contract-level attestation for the codebook; no preimage operator exists (`rmap::hop` computes the image). The plan row's former "CORRECT: H14c/H14d" pointed at the co-addressing principle, not a hop implementation.

OPEN: the contract-level codebook witness (version, map generation / seal, coordinate-domain identity, scope / completeness, invalidation).
