# Step 2: Wire FisherZTable into HhtlDTensor

> Agent: family-codec-smith
> READ FIRST: encoding-ecosystem.md, crates/bgz-tensor/src/fisher_z.rs
> Depends on: Step 1 (fisher_z.rs — DONE)

## What to Do

Add `fisher_z_table: Option<FisherZTable>` to `HhtlDTensor` (hhtl_d.rs).

The existing `AttentionTable` (u16 L1 distances) stays — it's used for HHTL
cascade routing (Skip/Attend decisions). The FisherZTable is the pairwise
cosine lookup that the cascade routes INTO.

```rust
// In hhtl_d.rs, add to HhtlDTensor:
pub struct HhtlDTensor {
    pub role: String,
    pub basin: HeelBasin,
    pub cache: HhtlCache,          // existing: routing decisions
    pub entries: Vec<HhtlDEntry>,  // existing: per-row Slot D + Slot V
    pub original_shape: [usize; 2],
    pub gamma_meta: [f32; 4],
    pub fisher_z: Option<FisherZTable>,  // NEW: pairwise cosine as i8
}
```

## How to Build It

In `HhtlDTensor::encode()`, after building the palette and entries,
compute the FisherZTable from the palette's representative rows:

```rust
// After cache + entries are built:
let fisher_z = FisherZTable::build_from_palette(&cache.palette, n_cols);
```

## Serialization

Add Fisher z bytes to `entries_to_bytes()` / `entries_from_bytes()`:
- Append gamma (8 bytes) + k×k i8 entries after the HHTL-D entries
- Use a magic byte `0xFZ` prefix so legacy files without Fisher z
  still deserialize (backwards compatible)

## Tests

- `hhtld_with_fisher_z_roundtrip`: encode a tensor, check fisher_z is present
- `hhtld_lookup_cosine`: fisher_z.lookup_f32(a, b) returns plausible cosine

## Pass Criteria

- cargo test --manifest-path crates/bgz-tensor/Cargo.toml passes
- HhtlDTensor with fisher_z serializes/deserializes correctly
