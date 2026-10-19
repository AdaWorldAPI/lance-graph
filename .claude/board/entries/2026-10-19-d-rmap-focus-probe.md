# D-RMAP focus probe: a focus-bound RelationshipMap over existing indexes (2026-10-19)

**Status:** MEASURED (D-RMAP-1A-P), probe only. Record: `.claude/plans/d-rmap-separate-tables-thought-palettes-v1.md` §8. Code: `crates/mork-pathmap-probe/src/rmap.rs`, `tests/rmap.rs`, `examples/rmap_cost.rs`.

## Measured

- A 256-slot focus over 65 536 identities, prepared into a slot-space CSR (and a 256×256 tile per predicate), answers 1–3-hop scoped reach through the existing mask-risc executor with no gather and no copied fact. 12 tests equal a full-slice scan; 11 guards disable-verified.
- Break-even reuse count (arithmetic from measured medians): 76 (sparse, degree 2), 11 (degree 8), 1.8 (dense). The tile beats CSR only on the dense relation.
- Reporting escapes as source slots instead of identities cut the degree-8 CSR eval from 50.4 to 5.6 µs.
- Incremental rebind beats full re-preparation for small k and loses at k = 256 (dense: from k ≥ 16).
- 1/8/64 thoughts over one shared tile: flat per-eval cost; 32 B per thought as sparse edits, 16 KiB as executor planes.
- PathMap: prefix seek beats the sorted run on sparse relations; trie build is 4–6× slower and keys are ≥ 3× the bytes; `restrict` is exact but slower than CSR preparation. MORK matching not exercised.
- Place focus: zero-copy counting over canonical lanes is 20–24 µs (word-granular gate); a 3 072 B register copy counts in ~1–2 µs.

## Open

- Lance-backed `DurableRef` and stable-row-id seals; cold-cache measurement; bit-granular `under` gating; any gather admission.
