# Model Integration Plans: BERT + GPT-2 + ComfyUI + Stable Diffusion

## Universal Recipe (applies to ALL models)

```
1. Download smallest GGUF/safetensors quantization
2. hpc::gguf::read_gguf_header() → tensor directory
3. Find embedding matrix (token_embd.weight or equivalent)
4. read_tensor_f32() → dequantize to f32
5. Base17Token::from_f32() per token → 34 bytes each
6. JinaPalette::build() → 256 centroids + assignments  
7. cache::save_base17_cache() + save_palette_cache()
8. Commit 1-2MB cache to ndarray/src/hpc/jina/weights/
9. Delete original model. Never needed at runtime.

The cache IS the runtime. Load via LazyLock at startup.
All models compress to the same Base17 + palette format.
CausalEdge64 S/P/O fields accept any model's palette indices.
```

---

## Plan A: BERT-base

### Model
```
Architecture:   Transformer encoder (bidirectional)
Parameters:     110M
Vocab:          30,522 tokens (WordPiece)
Embedding dim:  768
Layers:         12
Heads:          12
Context:        512 tokens
```

### Weight Sources
```
HuggingFace:  google-bert/bert-base-uncased (safetensors, ~440MB)
GGUF:         may not exist natively — convert via llama.cpp or extract safetensors
Alternative:  onnx format → extract embedding matrix directly
```

### Cache Sizes
```
Embedding matrix:    30,522 × 768 × 4B = 89.4 MB (f32)
Base17 cache:        30,522 × 34B = 1.01 MB
Palette cache:       256 × 34B + 30,522 × 1B = 38.2 KB
COCA vocabulary:     997 KB (already committed, shared with Jina)
TOTAL runtime:       ~2.1 MB
```

### What BERT Gives Us
```
BERT's embedding matrix captures BIDIRECTIONAL context:
  "bank" near "river" → different embedding than "bank" near "money"
  
Our DeepNSM uses COCA distributional vectors (unidirectional, frequency-based).
BERT embeddings are CONTEXTUAL — they encode meaning-in-context.

Cross-check:
  DeepNSM dist(bank_river, bank_money) should be LARGE (different meanings)
  BERT dist(bank_river, bank_money) should be LARGE (contextual separation)
  If both agree → polysemy correctly detected
  If DeepNSM misses but BERT catches → Jina/BERT OOV path needed
  
Integration:
  Known words: DeepNSM (10μs, deterministic)
  Ambiguous words: BERT palette lookup (0.01μs) for disambiguation
  Full BERT inference (via burn backend): only when both DeepNSM + palette disagree
```

### Attention Head Extraction
```
12 layers × 12 heads = 144 attention perspectives
Each: Q[768,64] × K[768,64] → Base17 → AttentionTable[256][256]
144 tables × 128KB = 18MB (fits L3 cache)

Each table IS a different "how words relate" function:
  Head 0: syntactic structure (subject-verb agreement)
  Head 5: semantic similarity (synonym detection)
  Head 11: positional bias (nearby words relate more)
  
These become 144 CausalEdge64 networks — one per attention head.
```

### Implementation Steps
```
1. pip install transformers && download bert-base-uncased
2. Extract embedding matrix: model.embeddings.word_embeddings.weight
3. Save as binary: [vocab_size:u32][embed_dim:u32][f32 × vocab × dim]
4. Rust: load binary → Base17Token::from_f32() per token → save cache
5. Build palette → save cache
6. Commit to ndarray/src/hpc/jina/weights/bert_base17_30k.bin
7. Test: BERT palette distance vs DeepNSM distance → compute ρ
```

---

## Plan B: GPT-2 (124M)

### Model
```
Architecture:   Transformer decoder (autoregressive)
Parameters:     124M (GPT-2 small) / 355M (medium) / 774M (large)
Vocab:          50,257 tokens (BPE, same tokenizer as Jina v4!)
Embedding dim:  768 (small) / 1024 (medium) / 1280 (large)
Layers:         12 / 24 / 36
Heads:          12 / 16 / 20
Context:        1024 tokens
GGUF:           Available on HuggingFace (multiple quantizations)
```

### Why GPT-2 Matters
```
GPT-2 uses the SAME BPE tokenizer as Jina v4 (both are GPT-2 BPE).
This means: GPT-2 token indices DIRECTLY MAP to Jina token indices.
The palette assignments can be SHARED between models.

GPT-2 small (124M) is tiny enough to run full inference on CPU via burn.
This gives us a GENERATION model (predict next token) in addition to
the EMBEDDING model (Jina) and SEMANTIC model (DeepNSM).

The trifecta:
  DeepNSM:  WHAT does it mean? (semantic decomposition)
  Jina:     HOW similar? (contextual embedding distance)  
  GPT-2:    WHAT comes next? (autoregressive prediction)
```

### Cache Sizes
```
GPT-2 small embedding:  50,257 × 768 × 4B = 147 MB (f32)
Base17 cache:           50,257 × 34B = 1.67 MB
Palette cache:          256 × 34B + 50,257 × 1B = 57.4 KB
TOTAL runtime:          ~1.8 MB
```

### Synergy with Existing Stack
```
GPT-2 BPE tokens = Jina BPE tokens → SAME palette!
  Already have: jina_palette_20k.bin (20K tokens)
  Extend to: gpt2_palette_50k.bin (50K tokens, same codebook style)
  
  Shared palette → CausalEdge64 edges from GPT-2 AND Jina use the SAME
  S/P/O palette indices → directly comparable in the SAME causal network.

GPT-2 attention heads → AttentionTable per head
  12 layers × 12 heads = 144 tables
  Each: ComposeTable for multi-hop prediction
  
  "Given token A, what's the next most likely token?"
  = AttentionTable[A_palette][?] → sort by distance → top-K predictions
  
  This IS autocompletion via table lookup. No matmul.
```

### Implementation Steps
```
1. Download GPT-2 GGUF from HuggingFace
2. hpc::gguf::read_gguf_header() → find wte.weight (embedding matrix)
3. read_tensor_f32() → 50,257 × 768 f32 vectors
4. Base17Token::from_f32() → save gpt2_base17_50k.bin (1.67 MB)
5. JinaPalette::build() → save gpt2_palette_50k.bin
6. Verify: GPT-2 palette indices align with Jina palette (shared BPE)
7. Build AttentionTable per head for O(1) next-token prediction
8. Commit caches to ndarray/src/hpc/jina/weights/
```

---

## Plan C: Stable Diffusion (for ComfyUI)

### Model
```
Architecture:   Latent Diffusion Model
Components:     
  - CLIP text encoder (77 tokens, 768D embeddings)
  - UNet denoiser (860M params, NOT token-level)
  - VAE decoder (latent → pixel)
Parameters:     ~1B total
Vocab:          49,408 tokens (CLIP BPE)
GGUF:           SD 1.5 available as GGUF via stable-diffusion.cpp
```

### What We Can Extract
```
CLIP text encoder embedding:  49,408 × 768 × 4B = 145 MB (f32)
Base17 cache:                 49,408 × 34B = 1.64 MB
Palette cache:                256 × 34B + 49,408 × 1B = 56.5 KB
TOTAL runtime:                ~1.8 MB

The CLIP embedding IS the "text → image" bridge.
Every text prompt → CLIP tokens → CLIP embeddings → guide image generation.

Our palette-compressed CLIP embeddings enable:
  "how similar is this prompt to that prompt?" → palette distance → O(1)
  "what images would this text generate?" → nearest CLIP archetypes → O(1)
  "is this image matching the prompt?" → CLIP similarity → O(1)
```

### What We CANNOT Extract (UNet)
```
The UNet denoiser is NOT token-level — it operates on latent images.
It cannot be compressed to token-level Base17 palettes.

However: UNet attention heads CAN be compiled into AttentionTables.
Each UNet attention layer: Q×K^T → 256-archetype distance table.
This replaces UNet attention matmul with table lookup.

UNet attention compilation:
  UNet has ~16 attention layers × ~8 heads = ~128 tables
  Each: AttentionTable[256][256] = 128KB
  Total: ~16MB of compiled attention (fits L3 cache)
  
  This IS the bgz-tensor thesis applied to image generation:
  Every diffusion step's attention = 128 table lookups instead of matmul.
```

### ComfyUI Integration
```
ComfyUI uses Stable Diffusion via Python + PyTorch.
Our integration path:
  1. CLIP text encoder: replace with palette-compressed version
     → text prompt → BPE → palette lookup → guide vector → O(1)
  2. UNet attention: compile to AttentionTable per head
     → each diffusion step: table lookup instead of matmul
  3. VAE decoder: keep as-is (it's the pixel output stage)
  
  The speedup: attention is ~60% of UNet compute time.
  Replacing it with table lookup → ~2.5× faster diffusion.
  
  ComfyUI node: "AdaWorld Attention" → uses compiled tables.
```

### Implementation Steps
```
1. Download SD 1.5 GGUF (stable-diffusion.cpp format)
2. Extract CLIP text encoder embedding matrix
3. Base17 + palette compress → commit 1.8MB cache
4. Extract UNet attention heads (optional, for full compilation)
5. Build AttentionTable per UNet attention layer
6. Create ComfyUI custom node that loads our compiled tables
```

---

## Plan D: Cross-Model Synergies

### Shared BPE Tokenizer Family
```
GPT-2 BPE:    50,257 tokens  (GPT-2, Jina v4)
CLIP BPE:     49,408 tokens  (Stable Diffusion, DALL-E)
WordPiece:    30,522 tokens  (BERT)

GPT-2 and Jina share the SAME tokenizer → SAME palette → interoperable.
CLIP uses a DIFFERENT BPE → needs its own palette → but similar subwords.

Cross-model token mapping:
  GPT-2 token "hello" = Jina token "hello" → SAME palette index
  CLIP token "hello" → DIFFERENT index → needs mapping table
  BERT token "hello" → DIFFERENT index (WordPiece) → needs mapping table

The mapping tables are small (50K entries × 1B = 50KB each).
After mapping: ALL models use the SAME CausalEdge64 palette space.
```

### Unified Embedding Space
```
Model         Embed dim   Base17   Palette   What it captures
─────         ─────────   ──────   ───────   ────────────────
DeepNSM       96D         34B      1B        Distributional (COCA frequency)
BERT          768D        34B      1B        Bidirectional context
GPT-2         768D        34B      1B        Autoregressive prediction
Jina v4       2048D       34B      1B        Retrieval similarity
CLIP          768D        34B      1B        Text-image alignment
SD UNet       varies      34B      1B        Image latent structure

ALL compress to the same 34-byte Base17 format.
ALL map to the same 256-entry palette.
ALL pack into CausalEdge64 S/P/O fields (8 bits each).

The palette IS the universal embedding space:
  256 archetypes × 5 models = 5 different "views" of each archetype.
  Cross-model similarity: do BERT and GPT-2 agree about this word?
  If yes → high NARS confidence. If no → interesting disagreement.
```

### CausalEdge64 Multi-Model Network
```
One CausalEdge64 network, multiple evidence sources:

  Frame 1 (text):   DeepNSM → SPO(bird, perch, fence) → edge with NARS truth
  Frame 2 (image):  Centroid focus → SPO(bird, perch, fence) → revision ↑
  Frame 3 (BERT):   Contextual embedding confirms "perch" not "fly" → revision ↑
  Frame 4 (GPT-2):  Next-token prediction: "bird perch fence" is likely → revision ↑
  Frame 5 (CLIP):   Text-image similarity confirms match → revision ↑
  
  Each model is an INDEPENDENT evidence source.
  NARS revision combines them.
  5 models agreeing → very high confidence.
  Models disagreeing → interesting → exploration target.
  
  Cronbach's α across 5 models = measurement reliability.
  High α → robust finding. Low α → ambiguous, needs more evidence.
```

---

## Priority Ordering

```
P0 (immediate, this session had everything ready):
  GPT-2 small GGUF → Base17 + palette cache (shared BPE with Jina)

P1 (next session, straightforward):
  BERT-base safetensors → Base17 + palette cache

P2 (medium, needs SD.cpp GGUF):
  SD 1.5 CLIP encoder → Base17 + palette cache

P3 (complex, needs UNet attention compilation):
  SD 1.5 UNet attention → AttentionTable per head → ComfyUI node

All share the same codec:
  hpc::gguf (loader) + hpc::jina::codec (Base17 + palette) + hpc::jina::causal (CausalEdge64)
  One codebase. Multiple models. Same 34-byte format.
```
