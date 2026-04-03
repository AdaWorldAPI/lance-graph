# Architecture: 16M RISC Thought Engine

> Updated: 2026-04-03
> Status: ThinkingEngine + branching + DTOs built, 27 tests passing

---

## Core Principle

One MatVec per cycle. 10 cycles per thought. The distance table IS the brain.

```
Perturbation → energy[4096] → distance_table × energy → convergence → thought
```

Models are sensors. They produce codebook indices, not inference.
The engine does NOT run forward passes. It runs MatVec.

---

## Three Layers of Branching (NOT filtering)

```
Layer 1:   64 ×  64 =   4 KB    L1-resident    64-wide vector unit
Layer 2:  256 × 256 =  64 KB    L1/L2          4× branching from L1
Layer 3: 4096 × 4096 = 16 MB    L3 (or GPU)    16× branching from L2
```

Each L1 lane SPAWNS 4 parallel L2 lanes. Each L2 spawns 16 L3 lanes.
4×4 = 16 parallel paths per L1 activation. RISC parallel vector thinking.

NOT cascade filtering. NOT survival. BRANCHING.
The energy flows DOWN from coarse to fine, spawning children.

### Synergy with p64

- L1 64×64 = same scale as Palette64 / CausalEdge64 addressing
- 8 predicate layers in Blumenstrauß = 8 parallel L1 tables
- Subject(8) + Object(8) of CausalEdge64 = L1 row × column

### Synergy with NARS

The distance table encodes NARS truth values:

```
table[i][j] = cosine(i,j) × frequency(i→j) × confidence(i→j)
```

Each MatVec cycle IS a NARS revision step.
Energy convergence IS reasoning. Entropy reduction IS conclusion.

```
Cycle 0:  H = 8.3 bits  (uniform — everything possible)
Cycle 5:  H = 2.1 bits  (3-5 peaks — candidates crystallizing)
Cycle 10: H = 0.3 bits  (converged — this IS the conclusion)
```

---

## What Replaces What

```
Transformer (sequential):       Branching Engine (parallel):
─────────────────────           ─────────────────────────
Q×K^T attention                 distance_table[i][j] encodes it
softmax normalization           energy normalization after MatVec
V retrieval                     codebook[argmax(energy)]
Gate (SwiGLU)                   destructive interference = gate closed
Up projection (FFN expand)      L1→L2 branching (4× expansion)
Down projection (FFN compress)  energy concentration (256 → ~20 peaks)
96 sequential layers            10 convergence cycles

6 ops × 96 layers × N tokens   1 MatVec × 10 cycles × 1 energy vector
```

---

## Memory Budget (CPU)

```
Component                     Size         Cache Level
──────────                    ────         ───────────
Energy L1 (f64[64])           512 B        L1 (register-adjacent)
Energy L2 (f64[256])          2 KB         L1
Energy L3 (f64[4096])         32 KB        L1
L1 distance table (u8[64²])  4 KB         L1 (always hot)
L2 distance table (u8[256²]) 64 KB        L1/L2
L3 distance table (u8[4096²]) 16 MB       L3 (or GPU)
Codebook (4096 × 1 KB)       4 MB         L2

Total CPU (L1+L2 only):      ~70 KB       runs in register file
Total CPU (all layers):       ~20 MB       fits L3
```

---

## DTOs (Cognitive Laws as Bus Adapters)

```
Φ Dispersion:   StreamDto       sensor → codebook indices, no meaning yet
Ψ Interference: ResonanceDto    f64[4096] energy = the ripple field
B Consequence:  BusDto          argmax + top-k + provenance
Γ Collapse:     ThoughtStruct   stabilized, text is lazy display
```

ResonanceDto IS f64[4096]. Not a struct with candidate lists.
ThoughtStruct carries the BusDto + provenance + tension history.
Text is generated LAZILY — the thought exists before the words.

---

## Sensors (ALL models = codebook lookup, no forward pass)

```
Sensor      Tokenizer          Codebook   Status
──────      ─────────          ────────   ──────
Jina v3     from GGUF / API    4096       GGUF downloaded, sensor agent building
BGE-M3      SentencePiece      4096       tokenizer wired, needs codebook
reader-LM   Qwen2 BPE          4096       tokenizer wired, needs codebook
DeepNSM     N/A (co-occur)     256        DONE (standalone crate)
Wikidata    N/A (entity IDs)   1024       next
AriGraph    N/A (graph nodes)  256        family basins ready
```

GGUF weights streamed ONCE to build codebook + indices. Then DISCARDED.
"Inference" = tokenize → token_id → indices[token_id] → codebook entry.
Total stored for ALL models: ~60 MB. Raw weights: ~118 GB.

Jina API = only LIVE sensor for fresh text (10ms exception).
Everything else = HHTL codebook lookup (ns).

---

## Persona Integration (separate repository, future wire)

```
Distance table:    static topology (the roads)
Persona layer:     which roads this agent PREFERS (frozen/crystallized)
Texture resonance: which roads match THIS thought's character
Mirror modeling:   which roads the USER would take
Autopoiesis:       persona updating ITSELF through superposition
```

Persona does NOT modify the distance table. It modulates the energy
vector BEFORE and AFTER the MatVec:
- Before: frozen traits get +baseline energy at perturb time
- After: texture resonance amplifies matching peaks, dampens others

AriGraph family basins = which codebook regions this agent has LIVED in.
Basin membership = frozen traits. Basin boundaries = crystallization.
Reaching outside basins = exploration.

Gating is NOT Gate-proj. It's persona texture resonance selecting
which MatVec results to amplify. The thought object's texture matches
against the agent's texture. High resonance = amplify.

Mirror neuron modeling = parallel MatVec walk with user's persona weights.
Difference between agent walk and user walk = empathy signal.

---

## Calibration (burn, offline)

```
Phase          Tool              Speed        When
─────          ────              ─────        ────
Stream weights burn-ndarray      min/model    once
Build codebook CLAM sampling     seconds      once
Distance table burn matmul       seconds      once (4096² cosine in one MatMul)
Token embed    bert/llama-burn   min/batch    once per corpus
RL training    burn-rl + autodiff hours       chess calibration
```

burn is NOT runtime. burn builds the thinking engine.
Runtime = handrolled F64x8 / VNNI u8 MatVec.

---

## Validation Path

```
Phase 5: Schach
  Position → codebook indices → perturb → 10 cycles → energy peaks
  Peaks = candidate moves. Stockfish eval = ground truth.
  RL adjusts distance_table entries: good moves → increase, blunders → decrease.
  64 squares = Palette64. Alpha-Beta = Belichtungsmesser cascade.
  Perfect feedback → transfers to NLP.
```

---

## Crate Map

```
thinking-engine/
├── src/
│   ├── lib.rs          module declarations
│   ├── engine.rs       flat 4096² MatVec (reference implementation)
│   ├── layered.rs      three-layer cascade (L1→L2→L3 with survivors)
│   ├── branching.rs    4×4 branching parallel thinking (CORRECT architecture)
│   ├── dto.rs          StreamDto / ResonanceDto / BusDto / ThoughtStruct / ThoughtIndex
│   └── sensor.rs       (building) Jina GGUF → codebook → perturb pipeline
├── Cargo.toml          deps: ndarray + bgz-tensor
└── tests: 27 passing
```

---

## Key Numbers

```
i16 Base17 (old):       0.458 Pearson, OpenChat = ALL ZEROS     DEAD
StackedBF16 SPD=32:     0.996 Pearson                           production
φ-sampling stride=8:    +10% Pearson at same bytes              proven
Three-finger HEEL:      72% reject, zero data access            proven
Euler-gamma fold n=6:   0.958 on similar vectors                proven
BF16→f32:               zero loss (truncation invisible)        proven
SIMD rewire:            372/372 intrinsics → crate::simd        done
Distance table:         4096² × u8 = 16 MB                     fits L3
MatVec cycle:           ~175μs (VNNI) / ~700μs (F64x8)          per cycle
10 cycles:              1.75ms (VNNI) / 7ms (F64x8)             per thought
```

---

## Doctrine

1. The distance table IS the brain. One MatVec IS one reasoning step.
2. Models are SENSORS. Codebook + indices. No GGUF on disk. EVER.
3. "Inference" = tokenize → indices → codebook. Not a forward pass.
4. Branching, not filtering. 4×4 parallel lanes, not survival cascade.
5. Entropy reduction IS reasoning. NARS provides direction.
6. Persona modulates energy, does NOT modify the distance table.
7. Text is lazy display. ThoughtStruct is the thought.
8. ResonanceDto IS f64[4096]. Not a struct with candidate lists.
9. 7+1: contradiction acts ON the energy field, not IN it.
10. Services are LAWS (Φ/Ψ/B/Γ), not endpoints.
