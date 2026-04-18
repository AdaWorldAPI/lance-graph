# Cognitive Shader Architecture — Session 2026-04-18

> READ BY: all agents working on inference, codec, thinking-engine, learning, holograph

## Status: FINDING (measured, not conjecture)

### Cascade Inference
- 11-13x speedup over brute-force cosine on real Qwen3-TTS weights
- 100% argmax match (zero quality loss)
- Sign-bit fingerprint + Hamming popcount pre-filter → exact cosine on 3%
- TurboQuant KV cache: 3.2x memory reduction, 13x attention speedup, 100% argmax

### Codec Findings
- 67 codecs tested: Hadamard > SVD, full-rank > narrow, i4+i2 cascade
- ICC 0.999 on pairwise cosine, but argmax fails at k=64 on hard tensors
- XOR-adaptive: sign-flip per-dimension precision → 81% argmax on hardest tensor
- CLAM-adaptive: LFD-driven precision → 97% on KV projections
- Holographic residual: sign-only gets cos 0.6-0.75, needs magnitude slots

### Architecture Decision
- **Don't compress weights lossy for inference** (breaks argmax)
- **Accelerate search instead** (cascade gives speed, original weights give quality)
- **TurboQuant on KV cache** (gain-shape split, cascade-compatible fingerprints)
- **Holographic memory for codebook** (slot-encoded phase+magnitude, future work)

### Key Types
- `Fingerprint<256>` — canonical 16K bit vector (ndarray, const-generic)
- `CausalEdge64` — u64 packed SPO+NARS+Pearl+plasticity
- `CognitiveShader` — 8 predicate planes × 64×64 topology × bgz17 metric
- `TurboQuantEntry` — gain(BF16) + shape(i4) + fingerprint(sign bits)
- `HadCascadeTensor` — WHT + i4 + i2 cascade codec
- `VectorWidth` — LazyLock W8K(deprecated)/W16K(production)

### Crate Layout (post-session)
```
ndarray         — Fingerprint<256>, WHT, i2/i4 quant, kmeans, cascade, CLAM
holograph       — BitpackedVector (→ migrate to Fingerprint<256>), slot encoding, resonance
learning        — 16 modules from ladybug-rs (wip-gated), 300K+ LOC
lance-graph-cognitive — grammar + world (compiling), spo (wip-gated)
bgz-tensor      — HadCascade, TurboQuant KV, adaptive/xor/holographic codecs
causal-edge     — CausalEdge64, NarsTables, CausalNetwork
p64-bridge      — CognitiveShader, style params, palette addressing
thinking-engine — unified surface (to absorb learning + cognitive)
```

### Endgame: GGUF → Holographic Memory → Cognitive Shader Inference
```
GGUF weights → hydrate into palette + fingerprints + holographic memory
  → CognitiveShader cascade per layer (no matmul, no FP)
  → CausalEdge64 output (SPO + NARS)
  → 4096 COCA codebook → output token
```

### Pending Debt
1. Unify Fingerprint types (holograph BitpackedVector → ndarray Fingerprint<256>)
2. Enable learning crate (rustynum→ndarray migration, 124 errors)
3. Container/CogRecord port to lance-graph-contract
4. GPTQ Hessian compensation for argmax stability
5. Holographic magnitude slot encoding
6. CognitiveShader → thinking-engine end-to-end wiring
7. VectorWidth LazyLock not yet consumed by any module
