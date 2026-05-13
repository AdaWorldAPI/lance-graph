# SESSION: Jina v5 ONNX Calibration — Ground Truth Without API Keys

## THE GOAL

Replace synthetic ground truth in `calibrate_lenses.rs` with REAL embeddings.
Jina v5 ONNX + rten = f32 ground truth on CPU. No API key. No network at runtime.

Then: run the 5-path calibration matrix from PR #113 handover.

---

## JINA v5 MODEL CARD

```
Model:     jinaai/jina-embeddings-v5-text-small-text-matching
Base:      Qwen3-0.6B-Base
Params:    677M (0.6B)
Emb dim:   1024
Matryoshka: 32, 64, 128, 256, 512, 768, 1024
Max seq:   32,768
Pooling:   last-token
Tensor:    BF16
Vocab:     Qwen3 BPE (tokenizer.json = 11.4 MB)

Files:
  model.safetensors              1.19 GB   (safetensors weights)
  v5-small-text-matching-F16.gguf  1.2 GB   (F16 GGUF, streamable)
  onnx/model.onnx                1.27 MB   (graph only)
  onnx/model.onnx_data           2.38 GB   (weights, external data)
  tokenizer.json                 11.4 MB   (Qwen3 BPE)

v5 Task Variants (all have GGUF + ONNX):
  text-small:          general, retrieval, text-matching, clustering, classification
  text-nano (0.2B):    same 5 tasks (smaller, faster, less accurate)
```

## WHY v5 REPLACES v3 AS TRUTH ANCHOR

```
v3:  API-only ground truth (needs JINA_API_KEY, network, rate limits)
v5:  ONNX available (rten loads it, forward pass on CPU, no API)
     GGUF available (stream F16 → CLAM → bake, same pipeline as v3)
     SAME repo has both → calibration in one session, no external deps

v5 is also newer, better MTEB scores, Matryoshka dims, and based on Qwen3.
```

---

## IMPLEMENTATION PLAN

### Phase 1: Add rten dependency (10 min)

```toml
# crates/thinking-engine/Cargo.toml
[dependencies]
rten = { version = "0.16", optional = true }
rten-tensor = { version = "0.16", optional = true }

[features]
default = ["tokenizer"]
tokenizer = ["dep:tokenizers"]
onnx-calibration = ["dep:rten", "dep:rten-tensor"]
```

Feature-gated. Default builds don't pull 2+ GB of ONNX weights.
Only the calibration example enables it.

### Phase 2: Download Jina v5 ONNX + GGUF (15 min, one-time)

```bash
# Create data directory
mkdir -p crates/thinking-engine/data/jina-v5-text-matching

# Download ONNX (2.39 GB total — model graph + external weights)
# NOTE: model.onnx_data is 2.38 GB. Do NOT commit to git.
cd crates/thinking-engine/data/jina-v5-text-matching
wget https://huggingface.co/jinaai/jina-embeddings-v5-text-small-text-matching/resolve/main/onnx/model.onnx
wget https://huggingface.co/jinaai/jina-embeddings-v5-text-small-text-matching/resolve/main/onnx/model.onnx_data

# Download tokenizer (11.4 MB)
wget https://huggingface.co/jinaai/jina-embeddings-v5-text-small-text-matching/resolve/main/tokenizer.json

# GGUF is STREAMED, not downloaded (existing pipeline via HTTP range requests)
# Source: v5-small-text-matching-F16.gguf (1.2 GB)
```

Add to `.gitignore`:
```
crates/thinking-engine/data/jina-v5-text-matching/model.onnx
crates/thinking-engine/data/jina-v5-text-matching/model.onnx_data
```

### Phase 3: ONNX inference module (jina_v5_onnx.rs, ~150 lines)

```rust
//! Jina v5 ONNX ground truth via rten.
//!
//! Loads the ONNX model, tokenizes input, runs forward pass,
//! returns 1024D f32 embedding. This IS the ground truth.

use rten::Model;
use rten_tensor::NdTensor;

pub struct JinaV5Onnx {
    model: Model,
}

impl JinaV5Onnx {
    /// Load from ONNX file path.
    pub fn load(onnx_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let model = Model::load_file(onnx_path)?;
        Ok(Self { model })
    }

    /// Run inference: token_ids → 1024D f32 embedding.
    /// Uses last-token pooling (Jina v5 convention).
    pub fn embed(&self, token_ids: &[i64]) -> Vec<f32> {
        let seq_len = token_ids.len();
        let input_ids = NdTensor::from_data(
            [1, seq_len],
            token_ids.to_vec(),
        );
        let attention_mask = NdTensor::from_data(
            [1, seq_len],
            vec![1i64; seq_len],
        );

        let result = self.model.run(
            vec![
                ("input_ids", input_ids.into()),
                ("attention_mask", attention_mask.into()),
            ],
            &["last_hidden_state"],  // or "sentence_embedding"
        ).expect("ONNX forward pass failed");

        // Extract last-token embedding (1024D)
        let output = result[0].as_float().unwrap();
        // Last token pooling: take embedding at position seq_len-1
        let embedding: Vec<f32> = (0..1024)
            .map(|d| output[[0, seq_len - 1, d]])
            .collect();

        // L2 normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            embedding.iter().map(|x| x / norm).collect()
        } else {
            embedding
        }
    }

    /// Cosine similarity between two texts (ground truth).
    pub fn cosine(&self, ids_a: &[i64], ids_b: &[i64]) -> f32 {
        let emb_a = self.embed(ids_a);
        let emb_b = self.embed(ids_b);
        emb_a.iter().zip(&emb_b).map(|(a, b)| a * b).sum()
    }
}
```

### Phase 4: Build Jina v5 lens (stream_jina_v5.rs example)

```
Stream v5-small-text-matching-F16.gguf via HTTP range requests
→ Extract token_embedding rows (vocab × 1024 × BF16)
→ CLAM 256 centroids
→ Build 256×256 distance table
→ CDF-percentile HDR encoding → u8 table
→ Also build i8 signed table (subtract 128)
→ Bake via include_bytes! → jina_v5_lens.rs
```

### Phase 5: Calibration matrix (calibrate_v5.rs example)

```
For each of 16 sentence pairs:
  1. Tokenize with Jina v5 tokenizer (tokenizers crate)
  2. ONNX inference → f32 cosine = GROUND TRUTH
  3. Baked lens distance (u8 HDR) = PATH 1
  4. Baked lens signed (i8) = PATH 2
  5. γ+φ encoded distance = PATH 3
  6. highheelbgz spiral distance = PATH 4

Compare each path vs ground truth:
  Spearman ρ (rank preservation)
  Linear ICC (transfer curve)
  RMSE (absolute error)
  Effective bits (information preserved)

Result: per-path calibration scores.
Winner per model × role.
```

### Phase 6: Wire v5 as default truth anchor

```
Update calibrate_lenses.rs:
  Replace synthetic ground truth with JinaV5Onnx::cosine()
  Keep the Jina v3 and Reranker baked lens comparisons
  Add Jina v5 baked lens when available
  Report: v3 vs v5 correlation (should be high)
```

---

## 5-PATH CALIBRATION MATRIX (from PR #113 handover)

```
For EACH model (6 models × 6 roles = 36 cells):

  Path 1: ONNX (rten)         f32 ground truth        ← REFERENCE
  Path 2: GGUF raw u8 CDF     existing HDR pipeline
  Path 3: GGUF γ+φ            golden ratio redistribution
  Path 4: GGUF i8 signed      preserves gate sign      ← OUR NEW PATH
  Path 5: GGUF highheelbgz    spiral + golden stride

ICC profile per path:
  transfer_curve: f(baked_distance) → ground_truth_similarity
  noise_floor: minimum detectable difference
  effective_bits: Shannon entropy of the encoding
  spearman_rho: rank correlation with ground truth

Per model × role winner:
  i8 expected to win for: reranker (symmetric), gate roles (sign matters)
  γ+φ expected to win for: narrow-range roles (gate, up)
  raw CDF expected to win for: positive-skewed (embedding models)
  hhbgz expected to win for: ??? (that's why we test)
```

## MODELS IN THE MATRIX

```
Model                    ONNX available?    GGUF available?    Notes
─────                    ───────────────    ───────────────    ─────
Jina v5 text-matching    ✓ (2.39 GB)       ✓ (1.2 GB F16)    NEW truth anchor
BGE-M3                   ✓ (via ONNX repo)  ✓ (baked)          multilingual
Jina Reranker v3         ? (check)          ✓ (baked)          widest cos range
Reader-LM 1.5B           ? (check)          ✓ (baked)          HTML→text
Qwopus 27B               ✗ (too large)      ✓ (streamed)       dense LLM
Maverick 128E            ✗ (800 GB)         ✓ (18 shards)      real MoE
```

For models without ONNX: use Jina v5 embeddings of the TEXT as cross-model truth.
The thinking-engine distance should correlate with Jina v5 text similarity.

## SIGNED EXPERIMENT INTEGRATION

```
The dual signed experiment (just pushed) showed:
  Jina v3:     88% agreement (narrow cos, weak inhibition)
  Jina Reranker: 50% agreement (wide cos, strong inhibition)
  BGE-M3:      62% agreement (moderate)

The calibration matrix will answer:
  Does the 50% DISAGREEMENT on the reranker mean
    (a) signed is MORE accurate (finds real structure unsigned misses), or
    (b) signed is LESS accurate (inhibition kills valid peaks)?

  Compare both paths against ONNX ground truth.
  If signed ρ > unsigned ρ on reranker: signed wins, drop SiLU-ONNX.
  If unsigned ρ > signed ρ: keep both, SiLU-ONNX still needed.
```

---

## DISK BUDGET

```
ONNX model:    2.39 GB  (downloaded, NOT committed, .gitignore'd)
GGUF F16:      1.2 GB   (streamed via HTTP, never on disk)
tokenizer:     11.4 MB  (downloaded, could commit or gitignore)
Baked table:   64 KB    (committed, include_bytes!)
Baked index:   ~300 KB  (committed, include_bytes!)

Total new disk: ~2.4 GB temporary for calibration, 364 KB permanent.
```

## WHAT NOT TO DO

- Do NOT commit the ONNX model to git (2.4 GB). gitignore it.
- Do NOT stream ONNX via HTTP range requests. rten needs the full file.
- Do NOT use Q8_0 GGUF. cos[0,0]. F16 required. (Doctrine #9)
- Do NOT run calibration in CI. It needs the ONNX file on disk.
- Do NOT assume output tensor name. Check model.onnx graph for actual names.
- Do NOT skip L2 normalization. Jina v5 uses last-token pooling, needs normalize.
