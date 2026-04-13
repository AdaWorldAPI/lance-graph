# Role Enum Complementarity — Decision Record

> **Date**: 2026-04-13
> **Status**: OPEN — treat as complementary until impact proven
> **READ BY**: any agent touching TensorRole, LayerTables, NeuronPrint, or role mapping

---

## The Two Enums

### `TensorRole` (lance-graph/src/graph/hydrate.rs:20) — 10 variants
```
QProj=0, KProj=1, VProj=2, OProj=3, GateProj=4, UpProj=5, DownProj=6,
Embedding=7, Norm=8, Other=9
```
Purpose: classify GGUF/HF tensor names into roles for Arrow partitioning.
Scope: all tensors in a model, including non-per-layer ones.

### `LayerTables` (thinking-engine/src/role_tables.rs:89) — 6 fields
```
attn_q, attn_k, attn_v, ffn_gate, ffn_up, ffn_down
```
Purpose: per-role BF16 distance tables from ClamCodebook centroids.
Scope: per-layer, only roles that produce meaningful cosine distance tables.

### `NeuronPrint` (lance-graph/src/graph/neuron.rs:26) — 6 Base17 fields
```
q, k, v, gate, up, down
```
Purpose: 6D holographic representation of a single neuron (204 bytes).
Scope: per-(layer, feature) pair.

---

## Alignment Map

| TensorRole | LayerTables field | NeuronPrint field | Distance table? |
|------------|-------------------|-------------------|-----------------|
| QProj      | attn_q            | q                 | Yes (raw cos)   |
| KProj      | attn_k            | k                 | Yes (raw cos)   |
| VProj      | attn_v            | v                 | Yes (raw cos)   |
| OProj      | —                 | —                 | **UNKNOWN**     |
| GateProj   | ffn_gate          | gate              | Yes (raw cos)   |
| UpProj     | ffn_up            | up                | Yes (gate-mod)  |
| DownProj   | ffn_down          | down              | Yes (raw cos)   |
| Embedding  | —                 | —                 | No (layer 0)    |
| Norm       | —                 | —                 | No (scale/bias) |
| Other      | —                 | —                 | No (catch-all)  |

---

## Decision: Complementary, Not Merged

**Do NOT align these enums by force.** Keep them as separate, complementary views
of the same underlying data until we have empirical evidence for the cost/benefit
of merging.

### What we'd need to prove before merging:

1. **OProj distance table**: Does `o_proj` have meaningful cosine distance
   structure? It's the attention output projection (mixes V back into residual).
   If its distance table is ~identity or ~random, there's no signal to extract
   and adding it to LayerTables wastes 256×256×2 = 128KB per layer.

2. **OProj in NeuronPrint**: Would a 7th field (238 bytes) improve bundle/gestalt
   quality? The current 6D bundle averages Q/K/V/Gate/Up/Down. Adding O might
   dilute attention signal (O is downstream of V) or might capture output mixing
   that the other 6 miss. Need to measure: does 7D gestalt L1 distance
   separate neuron functions better than 6D?

3. **Embedding distance table**: Embedding is one matrix (vocab_size × hidden_dim).
   A palette over its rows would be a vocabulary clustering — potentially useful
   for DeepNSM but architecturally different from per-layer role tables.

4. **Norm as signal**: LayerNorm/RMSNorm weights are 1D (hidden_dim). Their
   variance across layers might carry signal (which dimensions the model
   considers important), but cosine distance over scale vectors is not the
   right metric.

### Risks of premature merging:

- Losing the OProj signal by averaging it into existing fields
- Inflating NeuronPrint size (204 → 238 bytes, 17% overhead) without proven benefit
- Forcing LayerTables to build tables for roles with no distance structure
- Breaking the clean 6-role symmetry (3 attention + 3 FFN) that maps to
  NeuronPrint's attention()/retrieval()/mlp() decomposition

### Data requirement: safetensors, not GGUF

OProj and Embedding are only cleanly separable in **safetensors** format.
GGUF quantizes all tensors into block format (Q4_K_M, Q8_0, etc.) and
flattens role metadata — you can recover role from tensor name but the
values are quantization-degraded. The probes below require safetensors
variants (BF16 or F16) to measure actual role distance structure, not
quantization artifacts. This is why the project obtained safetensors
variants (Jina v5 safetensors on disk at `jina-v5-onnx/model.safetensors`).

**Not all models ship safetensors.** Some only distribute GGUF (community
quants) or gated weights (Llama, Gemma). The role separation pipeline must
degrade gracefully: when safetensors are available, build all 10 role tables
including OProj/Embedding/Norm at native precision. When only GGUF is
available, build the 6 core tables (Q/K/V/Gate/Up/Down) from dequantized
blocks and mark OProj/Embedding/Norm as UNAVAILABLE rather than guessing.
The `TensorRole` enum already handles this — `from_name()` parses both
HF and GGUF naming conventions, and roles that can't be extracted cleanly
fall through to `Other`.

### Probes to run (in priority order):

1. **PROBE: OProj distance structure** — Build `build_raw_table(o_codebook)`,
   compare entropy to Q/K/V tables. If entropy ≈ max (uniform), OProj has no
   structure. If entropy < Q, it's worth including.

2. **PROBE: 7D vs 6D gestalt** — Build NeuronPrint with and without OProj,
   compare L1 separation on a known-function neuron set (e.g., layer 0 vs
   layer 31 in Qwen3.5-9B).

3. **PROBE: Gate modulation on OProj** — Does `silu(gate) × o_proj` differ
   from raw `o_proj`? If yes, OProj is FFN-entangled and belongs in the FFN
   group. If no, it's purely attention-side.

---

## Cross-references

- `TensorRole::from_name()` — hydrate.rs:36 (GGUF + HF name parsing)
- `LayerTables::build()` — role_tables.rs:109 (takes 6 codebooks)
- `NeuronPrint::bundle()` — neuron.rs:48 (6-role average)
- `NeuronPrint::attention()` — neuron.rs:66 (Q ⊕ K)
- `NeuronPrint::mlp()` — neuron.rs:76 (Gate ⊕ Up ⊕ Down)
- AGI_P64_INTEGRATION_PLAN.md — verified type map (PR #161)
