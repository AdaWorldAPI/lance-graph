# NeuronPrint Rosetta Stone

> **Date**: 2026-03-31
> **Status**: Exploration — we built the instrument, now we learn to read it

---

## What We Built

Every neuron (layer `i`, feature `j`) in a transformer has 6 functional roles,
each compressed to 34 bytes (Base17, ρ=0.993 vs BF16). Together: **204 bytes
per neuron** — a complete holographic fingerprint of what that neuron does.

```
NeuronPrint {
    q:    Base17,  // 34B — how this neuron queries (attention Q projection)
    k:    Base17,  // 34B — what this neuron matches (attention K projection)
    v:    Base17,  // 34B — what this neuron retrieves (attention V projection)
    gate: Base17,  // 34B — whether this neuron fires (SwiGLU/MLP gate)
    up:   Base17,  // 34B — how this neuron amplifies (MLP up projection)
    down: Base17,  // 34B — how this neuron compresses (MLP down projection)
}
```

Three operations on it:

| Struct | Purpose | Metaphor |
|--------|---------|----------|
| `NeuronPrint` | What a neuron IS | The object — its complete behavior in 204 bytes |
| `NeuronQuery` | How you ASK it | The query — selective role probing (6-bit mask) |
| `NeuronTrace` | How it REASONS | The thinking — NARS truth derived from role ratios |

---

## The Epiphany: 6D SPO

The 6 roles map to an extended SPO decomposition. Classical SPO has 3 planes
(Subject, Predicate, Object). NeuronPrint has 6 — which factor into two triads:

```
Attention Triad (how the neuron communicates):
  Q = Subject    "who is asking?"
  K = Predicate  "what is the relationship?"
  V = Object     "what is the answer?"

MLP Triad (how the neuron transforms):
  Gate = Subject    "what input feature is this about?"
  Up   = Predicate  "how does it transform?"
  Down = Object     "what does it produce?"
```

The two triads are linked by the residual stream — attention writes to it,
MLP reads from it. The NeuronPrint captures BOTH sides: the communication
(Q/K/V) and the computation (Gate/Up/Down) in a single 204-byte struct.

### Why This Is a Rosetta Stone

The same neuron appears in all 6 tables, aligned by row index. This means:

1. **Q tells you what the neuron looks for** — its query pattern
2. **K tells you when the neuron responds** — its matching criteria
3. **V tells you what the neuron says** — its contribution
4. **Gate tells you IF the neuron speaks** — its activation threshold
5. **Up tells you HOW MUCH it speaks** — its amplification factor
6. **Down tells you how it's COMPRESSED afterward** — the information bottleneck

Reading all 6 together is like having the Rosetta Stone for that neuron —
the same information expressed in 6 different "languages" (projection spaces).

---

## Retrieval vs Reasoning

The 6 roles split cleanly into two uses:

### Retrieval (Key-Value Store)
```
Q probes against K → finds matching neurons
V at those positions → the retrieved information
```
This IS attention, reconstructed from palette indices. It's a key-value cache
where K is the key and V is the value, and Q is the lookup query.

### Reasoning (NARS Hydration)
```
Gate magnitude → NARS frequency (how often does this fire?)
Up/Down ratio  → NARS confidence (how strong is the evidence?)
Q·K alignment  → attention strength (how relevant is this?)
K·V alignment  → retrieval coherence (how consistent is the stored info?)
```
The MLP roles encode causal structure. A neuron with high Gate, high Up,
low Down is a "confident amplifier" — it fires often and boosts its signal.
A neuron with low Gate, low Up, high Down is a "skeptical compressor" —
it rarely fires and attenuates when it does.

---

## The LLM Architecture Zoo

Different LLM architectures use different naming conventions but the same
6 functional roles. Here's the mapping:

### Llama / Qwen / Mistral (GQA attention + SwiGLU MLP)
```
model.layers.{L}.self_attn.q_proj.weight  → Q
model.layers.{L}.self_attn.k_proj.weight  → K (grouped, fewer heads)
model.layers.{L}.self_attn.v_proj.weight  → V (grouped, same as K)
model.layers.{L}.self_attn.o_proj.weight  → O (output projection, maps back)
model.layers.{L}.mlp.gate_proj.weight     → Gate (SwiGLU σ(x) branch)
model.layers.{L}.mlp.up_proj.weight       → Up (SwiGLU linear branch)
model.layers.{L}.mlp.down_proj.weight     → Down (back to hidden dim)
```

### GPT-2 / GPT-J (MHA attention + GELU MLP)
```
transformer.h.{L}.attn.c_attn.weight      → Q+K+V fused (split by dim)
transformer.h.{L}.attn.c_proj.weight       → O
transformer.h.{L}.mlp.c_fc.weight          → Up (no gate in GELU MLP)
transformer.h.{L}.mlp.c_proj.weight        → Down
```
Note: GPT-2 has no separate Gate — GELU activation is implicit. The Gate
role is absent; use Up magnitude as a proxy for both gating and amplification.

### GGUF (llama.cpp naming)
```
blk.{L}.attn_q.weight    → Q
blk.{L}.attn_k.weight    → K
blk.{L}.attn_v.weight    → V
blk.{L}.attn_output.weight → O
blk.{L}.ffn_gate.weight  → Gate
blk.{L}.ffn_up.weight    → Up
blk.{L}.ffn_down.weight  → Down
```

### What Varies Between Architectures
- **GQA vs MHA**: K and V may have fewer heads than Q (grouped query attention).
  Row count differs: Q has `n_heads × d_head` rows, K/V have `n_kv_heads × d_head`.
- **SwiGLU vs GELU**: SwiGLU has explicit Gate; GELU doesn't. For GELU models,
  the Gate NeuronPrint role is empty or derived from Up.
- **Fused QKV**: Some models fuse Q/K/V into one weight matrix. Need to split
  by dimension when extracting.

---

## What We Don't Know Yet (Rosetta Exploration)

### Unanswered Questions
1. **Do Q archetypes cluster by semantic role?** If palette entry 42 in the
   Q palette consistently corresponds to "entity lookup" across layers, that's
   a universal attention primitive. If it doesn't, the palette is just compression.

2. **Does Gate magnitude correlate with neuron importance?** Literature suggests
   yes (see: SwiGLU analysis papers), but we haven't verified on our Base17
   projections. The ρ=0.993 preservation should keep this relationship intact.

3. **Are cross-role distances meaningful?** Does `L1(Q[i][j], K[i][j])` (the
   Q-K alignment for one neuron) predict attention entropy? Theory says yes:
   a neuron whose Q and K are similar attends broadly; one whose Q and K
   differ attends sharply.

4. **Does the Up/Down ratio track with polysemanticity?** A neuron with many
   features (polysemantic) should have high Up magnitude (many activations)
   but also high Down magnitude (aggressive compression). The ratio might
   identify monosemantic vs polysemantic neurons.

5. **Layer-wise structure**: Do early layers (feature detection) have different
   Gate/Up/Down distributions than late layers (concept composition)?
   The Hyperprobe paper suggests probing only the second half of layers.

### What the Literature Tells Us
- **Anthropic's "Scaling Monosemanticity"** (2024): Individual neurons often
  represent single concepts. The NeuronPrint should capture this — a monosemantic
  neuron has a tight, unique fingerprint across all 6 roles.
- **"Attention Head Superposition"** (2024): Attention heads can represent multiple
  features simultaneously. The Q/K alignment in NeuronPrint detects this —
  broad alignment = superposed, tight alignment = specialized.
- **SwiGLU analysis** (Shazeer 2020, PaLM): Gate projection acts as a learned
  binary mask over features. High Gate magnitude = important feature.
- **Residual stream as communication bus** (Elhage et al. 2021): All layers
  read from and write to the same residual stream. NeuronPrint captures both
  the read (Q/K) and write (V/Down) sides.

---

## Next Steps

1. **Hydrate a real model** with partition columns and build per-role palettes.
   Compare archetype distributions across Q/K/V/Gate/Up/Down.

2. **Cross-role distance analysis**: For each neuron, compute Q·K, K·V,
   Gate magnitude, Up/Down ratio. Correlate with known interpretability results.

3. **Layer progression**: Plot NeuronTrace (frequency, confidence, attention,
   coherence) across layers. Does it match the feature→concept gradient?

4. **Diff between models**: Compare Opus 4.5 vs 4.6 NeuronPrints.
   Which roles diverge? Which layers? This tells you WHERE the behavioral
   difference lives in the architecture.

5. **Wire NeuronQuery into serve.rs**: Replace flat SPO extraction with
   role-aware probing. "What does this model know about X?" becomes
   `NeuronQuery::attention(encode(X)).at_layer(15)` → searches Q partition
   at layer 15 only.

---

## Memory Budget

For Qwen3.5-27B (28 layers, ~5M weight rows across all tensors):

```
Per neuron:        204 bytes (6 × 34)
Per layer:         ~180K neurons × 204 bytes = ~36 MB
Full model:        28 layers × 36 MB ≈ 1 GB (NeuronPrint for every feature)
Bundled per layer: 28 × 34 bytes = 952 bytes (one HEEL per layer)
Bundled per role:  6 × 34 bytes = 204 bytes (one HEEL per role type)
Full model HEEL:   34 bytes
```

Compare: original BF16 weights = ~54 GB. NeuronPrint = ~1 GB. 54× compression
while adding structural metadata (role, layer) that the raw weights don't have.

The 5M × 34 bytes (170 MB) we already have in bgz7 = the LEAF level.
NeuronPrint organizes the same data into 6 aligned tables with semantic meaning.
No additional extraction needed — just grouping by tensor role.
