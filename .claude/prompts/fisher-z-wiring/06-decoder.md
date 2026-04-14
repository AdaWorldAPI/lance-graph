# Step 6: Update Decoder with Fisher z Restore

> Agent: family-codec-smith
> READ FIRST: tts_17b_hhtld_decode.rs, fisher_z.rs
> Depends on: Step 5 (encoder emits tables)

## What to Do

Update `tts_17b_hhtld_decode.rs` to load Fisher z tables and use
them for cosine restore.

### Load path:

```rust
// For each group in the encoded safetensors:
let fz_table_bytes = read_tensor("{group}_fz_table");  // [k, k] u8
let fz_gamma_bytes = read_tensor("{group}_fz_gamma");  // [8] u8

let gamma = FamilyGamma::from_le_bytes(&fz_gamma_bytes);
let table = FisherZTable::from_bytes(&fz_table_bytes, k);
```

### Cosine restore:

```rust
// For any pair (row_a, row_b):
let centroid_a = entries[row_a].twig_centroid();
let centroid_b = entries[row_b].twig_centroid();
let cosine = table.lookup_f32(centroid_a, centroid_b);
// cosine is the Fisher z restored value: tanh((i8+127)/254 * z_range + z_min)
```

### Route statistics should now include:

- Number of pairs looked up via Fisher z table
- Mean/max restore error vs raw f32 cosine (spot check on 100 pairs)
- Distribution of i8 values (should use full range if gamma is correct)

## Pass Criteria

- Decoder loads Fisher z tables from encoded safetensors
- Spot-check: |restored_cosine - true_cosine| < 0.002 (mean)
- Route statistics printed
