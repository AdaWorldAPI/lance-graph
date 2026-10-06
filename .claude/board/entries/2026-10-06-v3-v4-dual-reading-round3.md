# 2026-10-06 — V3 data and V4 IR as two readings of one resident buffer (Round 3)

## MEASURED

`crates/r2il-mask-abi-probe/tests/v3_v4_dual_reading.rs`, 8 tests, 6 disable
runs red.

There is one owner: a 64-aligned `[u8; 2 × 512]`. The intake step is
`FunctionBody::write_into_value_slab`. Two shipped readings run over it with
no translation between them:

- **V3**: `node_rows_from_le_bytes` → `NodeRow::value` →
  `FacetCascade::ref_from_bytes` per 16-byte slot.
- **V4**: `ogar_loco::call_in_slab` under a `LaneShape`, plus
  `ogar_r2il::{r2il_mask, project}`.

`G6D2` (`6 × FacetTier{lo,hi}`) and `LaneShape::Pairs`
(`6 × (function:value)`) are the same carving. For all 180 calls, tier `k`
of slot `l` is call `6l+k`, at the same address (pointer identity).
`Triples`/`Quads` also read the slab in place.

- An owner write is seen at once by the V3 facet, the V4 call and the R2IL
  mask.
- Taking every reading leaves the bytes unchanged.
- The readings allocate 0 heap bytes. The meter can fire: it sees
  `materialize_indices`.
- Per-slot `facet_classid` bytes are V3 data. Rewriting all 30 changes no V4
  call.

## FINDING

- **Reading is shared; loco execution is not.** `Interpreter` runs a
  `Program { functions: Vec<FunctionBody> }`. The only path from resident
  bytes is `read_from_value_slab`, which gathers the 360 payload bytes into a
  second representation. That is a heap-owned copy at a different address.
  Reading a stored V4 body is zero-copy; executing it through loco today is
  not.
- **The selector is not in the bytes.** The same slab under `Pairs` and
  `Triples` yields different call streams. No shipped function maps a classid
  to a `LaneShape`; every caller passes it in. `LocoConcept::FunctionBody`
  (`0x1701`) is one concept for all three shapes.
- **Semantic-only difference:** V4 recovers `len` from the last non-zero
  call (`read_from_value_slab`). A V3 `0:0` tier is valid data, but as V4 it
  is a NOP and padding. An interior all-zero call cannot be represented.

## OPEN

- No mutable reinterpret (`mut_from_bytes`) exists on `FacetCascade`.
  Mutation goes through the owner's bytes, which matches "bytes are stored,
  integers are projected".
- The classid → reading resolver (classid → `LaneShape`, R2IL vs core
  dialect) is not shipped; plan W1 / `D-R2IL-5` is still "Not started".
- The documented `ruff_r2il` defect (`VarnodeFacet` classid lo-u16 = space
  rank, W0) is not re-tested here; ruff is not in this checkout.
