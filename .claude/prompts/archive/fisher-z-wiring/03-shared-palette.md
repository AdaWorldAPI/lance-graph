# Step 3: Compute Fisher z Tables per Shared Palette Group

> Agent: family-codec-smith
> READ FIRST: crates/bgz-tensor/src/shared_palette.rs, fisher_z.rs
> Depends on: Step 2

## What to Do

In `shared_palette.rs`, add FisherZTable computation to `SharedPaletteGroup`.

Each of the 26 palette groups gets one FisherZTable (k=256 → 64 KB).
Total: 26 × 64 KB = 1.6 MB.

```rust
pub struct SharedPaletteGroup {
    pub key: PaletteGroupKey,
    pub tensor_names: Vec<String>,
    pub cache: HhtlCache,
    pub hip_families: Vec<u8>,
    pub tensor_entries: Vec<(String, Vec<HhtlDEntry>)>,
    pub fisher_z: FisherZTable,  // NEW
}
```

## How to Build

When building the shared palette group, collect the representative f32 rows
(one per centroid — the row closest to each CLAM centroid in the original data).
Pass these to `FisherZTable::build()`.

The representatives come from the FIRST tensor in the group during encoding.
All tensors in the group share the same palette → same table.

## Storage Budget

```
26 groups × (256×256 i8 + 8 bytes gamma) = 26 × 65,544 = 1.7 MB
Previous palette overhead: 5.4 MB
Fisher z tables: 1.7 MB
Total: 7.1 MB (was 5.4 MB) — still fits in 11.2 MB budget
```

## Tests

- `shared_palette_has_fisher_z`: build a group, check table.k == 256
- `shared_palette_table_symmetric`: table[a][b] == table[b][a]

## Pass Criteria

- All existing shared_palette tests still pass
- New tests pass
- Total model size with Fisher z ≤ 13 MB
