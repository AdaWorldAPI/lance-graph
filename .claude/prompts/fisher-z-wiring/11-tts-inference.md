# Step 11: Wire Fisher z into TTS Inference

> Agent: family-codec-smith + resonance-cartographer
> READ FIRST: tts_full_inference.rs, tts_stream_hhtld.rs, fisher_z.rs
> Depends on: Steps 2, 3, 10 (tables available)

## What to Do

Replace raw matmul attention in the Rust TTS inference with
Fisher z table lookup. This is the core optimization:

```
BEFORE: attention_score = dot(q_row, k_row) / sqrt(d)  ← O(d) per pair
AFTER:  attention_score = fisher_z.lookup_f32(q_centroid, k_centroid)  ← O(1) per pair
```

### Where attention happens in Qwen3-TTS:

Each of the 33 transformer layers computes:
```
Q = X @ W_q        → assign to centroid → q_idx
K = X @ W_k        → assign to centroid → k_idx
score = Q @ K.T     → REPLACE with table[q_idx][k_idx]
V = X @ W_v
output = softmax(score) @ V
```

The Q/K matmul (X @ W_q, X @ W_k) still runs — it produces the hidden states
that get assigned to centroids. The centroid assignment is the HHTL-D entry's
`twig_centroid` field. The score computation is what gets replaced.

### Implementation:

```rust
// In the inference loop, per layer:
let q_centroids: Vec<u8> = q_hidden.iter()
    .map(|row| palette.nearest(&Base17::from_f32(row)).0)
    .collect();
let k_centroids: Vec<u8> = k_hidden.iter()
    .map(|row| palette.nearest(&Base17::from_f32(row)).0)
    .collect();

// Attention scores: O(1) per pair via Fisher z table
let mut scores = vec![vec![0.0f32; seq_len]; seq_len];
for i in 0..seq_len {
    for j in 0..=i {  // causal mask
        scores[i][j] = fisher_z.lookup_f32(q_centroids[i], k_centroids[j]);
    }
}
// softmax + V projection unchanged
```

### HHTL cascade skip:

Before the table lookup, check the route table:
```rust
match cache.route(q_centroids[i], k_centroids[j]) {
    RouteAction::Skip => scores[i][j] = f32::NEG_INFINITY,  // masked out by softmax
    _ => scores[i][j] = fisher_z.lookup_f32(q_centroids[i], k_centroids[j]),
}
```

This gives the 60% skip rate that makes it fast.

### What stays as raw matmul:

- Q/K/V projections: X @ W_q, X @ W_k, X @ W_v (hidden state computation)
- Output projection: attn_out @ W_o
- MLP: gate, up, down projections
- The conv decoder (52M params, not attention-based)

Only the Q·K^T attention score computation is replaced.

### Stream path (tts_stream_hhtld.rs):

Same change. Load Fisher z tables from the downloaded codebook archive.
The streaming encoder already downloads palettes — add Fisher z table
download alongside.

## Performance Budget

```
Per layer, per token pair:
  BEFORE: 128 multiply-adds (head_dim=128) = 128 FMAs
  AFTER:  1 table lookup (1 byte read) + route check

33 layers × seq_len² pairs:
  seq=50: 33 × 2500 = 82,500 lookups
  At 1.8 ns/lookup (measured): 148 µs total
  vs 33 × 2500 × 128 FMAs = 10.6M FMAs at ~0.5 ns each = 5.3 ms
  Speedup: ~36×
```

## Pass Criteria

- TTS inference produces WAV using Fisher z table lookup
- WAV has comparable spectral characteristics to raw inference WAV
- Inference time decreases (attention phase only)
