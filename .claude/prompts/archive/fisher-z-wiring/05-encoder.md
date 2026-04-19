# Step 5: Update Encoder to Emit Fisher z Tables

> Agent: family-codec-smith
> READ FIRST: tts_17b_hhtld_encode.rs, shared_palette.rs, fisher_z.rs
> Depends on: Steps 2, 3

## What to Do

Update `tts_17b_hhtld_encode.rs` to include Fisher z tables in the
output safetensors.

### Per shared palette group, emit:

1. `{group}_fisher_z_table` — k×k i8 values as u8 tensor [k, k]
2. `{group}_fisher_z_gamma` — 8 bytes as u8 tensor [8]
3. `{group}_fisher_z_meta` — JSON string in safetensors metadata

### The encode pipeline becomes:

```
safetensors (BF16)
  → SpiralEncoding → γ+φ rehydrate → Base17
    → WeightPalette (256 CLAM centroids)
      → HhtlDTensor (Slot D + Slot V, 4 bytes/row)
      → FisherZTable (k×k i8, from centroid representative cosines)  ← NEW
        → output safetensors
```

### Key: Representative Rows

During encoding, when building the palette from rows, save the f32 row
closest to each centroid. These representatives are the source for the
Fisher z table's pairwise cosines.

```rust
// After CLAM palette build:
let representatives: Vec<Vec<f32>> = palette.entries.iter()
    .enumerate()
    .map(|(i, _centroid)| {
        // Find the row assigned to centroid i with smallest L1 distance
        let best_row = rows_assigned_to[i]
            .iter()
            .min_by_key(|&&r| base17_rows[r].l1(&palette.entries[i]))
            .copied()
            .unwrap_or(0);
        f32_rows[best_row].clone()
    })
    .collect();

let fisher_z = FisherZTable::build(&representatives, palette.len());
```

## Output Format

The output safetensors adds per-group:
- `talker_gate_fz_table`: u8 tensor [256, 256] (k×k i8 stored as u8)
- `talker_gate_fz_gamma`: u8 tensor [8] (z_min f32 LE + z_range f32 LE)

## Pass Criteria

- Encoder runs on 1.7B model without error
- Output safetensors contains Fisher z tensors
- Fisher z table values are non-uniform (not all zeros)
- Total output size ≤ 13 MB
