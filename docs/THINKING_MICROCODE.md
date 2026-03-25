# Thinking Style Microcode: YAML Templates + JIT + LazyLock + NARS RL

## The Idea

The 36 thinking styles are not Rust enums. They are YAML microcode templates
that define execution behavior. The JIT engine compiles them. LazyLock freezes
the hot paths. NARS revision learns which combinations work. The topology
self-organizes from bruteforce reinforcement.

```
YAML template (thinking_style_analytical.yaml)
     │
     ▼ jitson::from_json() (parse + validate)
JitsonTemplate (typed struct)
     │
     ▼ JitEngine::compile() (Cranelift → native)
ScanKernel (function pointer)
     │
     ▼ LazyLock freeze (BUILD phase → RUN phase)
Frozen code page (1 pointer deref at runtime)
     │
     ▼ Execute with FieldModulation parameters
Result + quality measurement
     │
     ▼ NARS revision on topology edges
Topology learns which styles work together
```

## Why YAML, Not Hardcoded

Current state (style.rs line 14):
```rust
/// Runtime YAML templates can extend to 36+ via StyleOverride.
```

This comment exists but StyleOverride is never implemented. Every style's
FieldModulation is a hardcoded match arm:

```rust
// What exists: frozen, unchangeable
Self::Analytical => FieldModulation {
    resonance_threshold: 0.85,
    fan_out: 4,
    depth_bias: 0.9,
    // ... hardcoded values that can never evolve
}
```

What should exist:

```yaml
# thinking_styles/analytical.yaml
style: Analytical
cluster: Convergent
tau: 0x41

modulation:
  resonance_threshold: 0.85
  fan_out: 4
  depth_bias: 0.9
  breadth_bias: 0.2
  noise_tolerance: 0.1
  speed_bias: 0.3
  exploration: 0.1

# Which cognitive verbs this style prefers
verb_weights:
  explore: 0.3     # light exploration
  deepen: 0.9      # heavy deepening
  hypothesis: 0.7  # moderate hypothesis formation
  review: 0.8      # strong review orientation
  synthesis: 0.6   # moderate synthesis

# Which other styles to co-activate (initial priors, NARS-revisable)
activates:
  - style: Systematic
    frequency: 0.7
    confidence: 0.3   # low — will be revised by experience
  - style: Convergent
    frequency: 0.7
    confidence: 0.3

contradicts:
  - style: Creative
    frequency: 0.6
    confidence: 0.3
  - style: Exploratory
    frequency: 0.5
    confidence: 0.3

# JIT scan parameters (compiled to native code)
scan:
  kernel: "thinking_scan"
  threshold: 1800      # from resonance_threshold * 2000 + 100
  top_k: 4             # from fan_out
  prefetch_ahead: 5    # from (1 - speed_bias) * 7 + 1
  filter_mask: 0xFFFFFFFF  # from noise_tolerance < 0.3

# NSM semantic affinity (which word domains resonate)
semantic_affinity:
  mental_predicates: 0.8     # think, know, feel → strong resonance
  evaluators: 0.3            # good, bad → weak resonance
  action_predicates: 0.2     # do, move → weak resonance
  descriptors: 0.5           # big, small → moderate resonance
```

36 such files. One per style. Hot-reloadable. Editable without recompilation.
The YAML IS the microcode. The JIT compiles it to native.

## JIT Compilation Path

```
thinking_styles/analytical.yaml
     │
     ▼ serde_yaml::from_str() → ThinkingTemplate (Rust struct)
     │
     ▼ template_to_jitson() → converts to JitsonTemplate
     │  The scan parameters become a JITSON pipeline:
     │  {
     │    "kernel": "thinking_scan",
     │    "scan": { "threshold": 1800, "top_k": 4, ... },
     │    "pipeline": [
     │      { "stage": "load_distances",   "backend": "distance_matrix" },
     │      { "stage": "filter_threshold", "avx512": "VCMPPS+KMOV" },
     │      { "stage": "gather_neighbors", "avx512": "VPGATHERDD" },
     │      { "stage": "accumulate_truth", "avx512": "VFMADD231PS" },
     │      { "stage": "top_k_heap",       "avx512": "VPERMT2D" }
     │    ]
     │  }
     │
     ▼ JitEngine::compile(&mut self) → ScanKernel { code_addr }
     │  Cranelift compiles to native x86_64 with AVX-512
     │  The compiled kernel does:
     │    1. Load distance row from matrix (VMOVDQU8)
     │    2. Compare against threshold (VCMPPS)
     │    3. Gather neighbor activations (VPGATHERDD)
     │    4. Propagate NARS truth values (VFMADD231PS)
     │    5. Maintain top-K heap (VPERMT2D)
     │  16 styles evaluated per AVX-512 instruction
     │
     ▼ PrecompileQueue::enqueue() → hash → deduplicated entry
```

## LazyLock Freeze Pattern

From ndarray's JitEngine (already implemented):

```rust
/// Phase 1 — BUILD (&mut self, single-threaded):
///   compile_all_styles() loads 36 YAML files, compiles each to ScanKernel
///   LazyLock::get_mut → &mut KernelCache (no lock, no contention)
///
/// Phase 2 — RUN (&self via Arc, zero-cost reads):
///   get_style_kernel(style_id) → Option<ScanKernel>
///   LazyLock deref → &KernelCache (HashMap::get)
///   No lock. No atomic. No compilation.
///   ~5ns per lookup (plain HashMap::get, no synchronization)

static THINKING_KERNELS: LazyLock<ThinkingKernelCache> = LazyLock::new(|| {
    let mut cache = ThinkingKernelCache::new();

    // Load all 36 YAML templates
    for path in glob("thinking_styles/*.yaml") {
        let template = load_thinking_template(path);
        let jitson = template_to_jitson(&template);
        let kernel = compile_thinking_kernel(&jitson);
        cache.insert(template.style_id, CompiledStyle {
            kernel,
            modulation: template.modulation,
            verb_weights: template.verb_weights,
            activates: template.activates,
            contradicts: template.contradicts,
        });
    }

    cache
});

/// Get a compiled thinking style. One pointer deref after first access.
#[inline(always)]
pub fn thinking_kernel(style_id: u8) -> &'static CompiledStyle {
    &THINKING_KERNELS[style_id as usize]
}
```

LazyLock slicing: the 36 compiled styles are contiguous in memory.
Accessing style N is `base_ptr + N * sizeof(CompiledStyle)`.
CPU prefetches the next style while executing the current one
(same pattern as jitson's `prefetch_next()`).

## Precomputed Slices via LazyLock

Not just the kernels — precompute everything that's static:

```rust
/// Precomputed thinking data, frozen at first access.
static THINKING_PRECOMPUTED: LazyLock<ThinkingPrecomputed> = LazyLock::new(|| {
    let styles = load_all_yaml_styles();

    ThinkingPrecomputed {
        // 36 × 7D modulation fingerprints (for Hamming matching)
        fingerprints: styles.iter()
            .map(|s| s.modulation.to_fingerprint())
            .collect(),

        // 36 × 36 style distance matrix (from fingerprint Hamming)
        style_distances: precompute_style_distances(&styles),

        // 36 verb weight vectors (which verbs each style prefers)
        verb_weights: styles.iter()
            .map(|s| s.verb_weights.clone())
            .collect(),

        // 36 semantic affinity vectors (which NSM fields resonate)
        semantic_affinities: styles.iter()
            .map(|s| s.semantic_affinity.clone())
            .collect(),

        // 36 JIT-compiled scan kernels
        kernels: styles.iter()
            .map(|s| compile_thinking_kernel(&template_to_jitson(s)))
            .collect(),

        // 36 × 36 initial topology edges (from YAML activates/contradicts)
        initial_topology: build_topology_from_yaml(&styles),
    }
});
```

First access compiles everything. All subsequent accesses are pointer derefs.
The thinking pipeline never does runtime compilation, YAML parsing, or
fingerprint computation. It's all frozen in `LazyLock`.

## NARS Bruteforce Reinforcement Learning

The topology starts from YAML priors (low confidence: c=0.3).
Then bruteforce RL discovers which combinations actually work.

```
Step 1: Bruteforce all pairs
  For each of 36² = 1,296 style pairs:
    Run both styles on 100 representative queries
    Measure plan quality (cost, accuracy, coverage)
    Compute quality_delta = quality_A_with_B - quality_A_alone

Step 2: NARS revision on topology edges
  For each pair (A, B):
    If quality_delta > 0.1: observe_coactivation(A, B, +delta, Activates)
    If quality_delta < -0.1: observe_coactivation(A, B, -delta, Contradicts)
    If |quality_delta| < 0.1: observe_coactivation(A, B, 0, Activates)
    → NARS revision updates edge truth value (f, c)

Step 3: After N rounds, topology converges
  High-confidence edges (c > 0.7): stable relationships
  Low-confidence edges (c < 0.3): insufficient evidence, keep exploring
  Flipped edges: styles that were thought to contradict actually activate

Step 4: Persist learned topology
  topology.save() → 1,440 bytes (12×12) or ~10KB (36×36)
  Next startup: load learned topology, skip bruteforce for known pairs
  Still bruteforce NEW style templates (when YAML files change)
```

The key insight: you don't need gradient descent. You don't need a loss function.
You don't need backpropagation. NARS revision IS the learning algorithm:
- Positive evidence → frequency increases, confidence increases
- Negative evidence → frequency decreases, confidence increases
- No evidence → confidence stays low, keep exploring
- Contradictory evidence → frequency oscillates, confidence drops → COMPASS engages

This is computationally cheap: 1,296 pairs × 100 queries × ~10μs per query
= ~1.3 seconds for a full bruteforce round. Run it nightly.

## How Texture/Gestalt Matching Works

The "resonance" between a thinking style and a query's "texture" is computed
from the semantic affinity field in the YAML template:

```yaml
# In analytical.yaml:
semantic_affinity:
  mental_predicates: 0.8
  evaluators: 0.3
  action_predicates: 0.2
  descriptors: 0.5
```

When a query arrives:
1. DeepNSM tokenizes → PoS tags reveal the query's "texture"
2. Count: 3 mental verbs, 1 evaluator, 0 actions → texture = [0.6, 0.2, 0.0, 0.2]
3. Dot product: texture · analytical.semantic_affinity = 0.58
4. Dot product: texture · metacognitive.semantic_affinity = 0.82
5. Metacognitive resonates stronger → activate Metacognitive

This is the "felt resonance" — the query's semantic content matches the style's
affinity profile. Not string matching. Not rule-based. Distributional overlap
between the query's word types and the style's semantic strengths.

```
"I think I know what you feel"
  → tokens: [i, think, i, know, what, you, feel]
  → PoS texture: 60% mental predicates (think, know, feel)
  → Highest resonance: Metacognitive (0.82), Reflective (0.78)
  → Lowest resonance: Pragmatic (0.15), Direct (0.22)
  → Topology activates: Metacognitive cluster + Reflective deepening

"the big dog bit the old man"
  → tokens: [the, big, dog, bite, the, old, man]
  → PoS texture: 43% nouns, 29% descriptors, 14% action verb
  → Highest resonance: Direct (0.71), Analytical (0.65)
  → Topology activates: Direct + Analytical co-fire

The CONTENT drives the THINKING MODE.
The YAML defines the mapping.
The JIT compiles the scan.
NARS learns whether the mapping is correct.
```

## Complexity/Difficulty Matching

Beyond semantic texture, task difficulty also selects styles:

```yaml
# In each YAML template:
difficulty_affinity:
  simple:    0.8    # Analytical is great for simple queries
  moderate:  0.6    # still good for moderate
  complex:   0.3    # not great for complex
  unknown:   0.1    # worst for unknown territory

# vs creative.yaml:
difficulty_affinity:
  simple:    0.2    # Creative is overkill for simple
  moderate:  0.5
  complex:   0.8    # shines on complex
  unknown:   0.9    # best for unknown territory
```

Difficulty is estimated from:
- Query complexity (DeepNSM: VLP, multi-MATCH, UNION → complexity score)
- Topology entropy (high entropy = uncertain, low = well-understood)
- MUL assessment (felt vs demonstrated competence)

Combined selection:
```
style_score(S, query) =
    semantic_resonance(S.affinity, query.texture) × 0.4    // what's it about?
  + difficulty_match(S.difficulty, query.complexity) × 0.3   // how hard is it?
  + topology_activation(S, current_cluster) × 0.3            // what does experience say?

Pick top-scoring style. If top-2 differ by < 0.1 → fork both.
```

## Hot Reload

YAML files can change without recompilation:

```rust
/// Watch thinking_styles/ directory for changes.
/// On change: recompile affected style, update LazyLock via get_mut.
///
/// NOTE: This only works during BUILD phase (single-threaded startup).
/// During RUN phase, LazyLock is frozen. Hot reload requires restart
/// OR a separate mutable style registry behind RwLock (cold path).
pub fn hot_reload_styles(dir: &Path) {
    // Cold path: used for development/tuning
    // Production uses LazyLock frozen at startup
    let templates = load_all_yaml_styles(dir);
    for template in templates {
        let jitson = template_to_jitson(&template);
        let kernel = compile_thinking_kernel(&jitson);
        STYLE_REGISTRY.write().insert(template.style_id, kernel);
    }
}
```

Development cycle:
1. Edit `thinking_styles/creative.yaml` — change `exploration: 0.8` to `0.95`
2. Hot reload → JIT recompiles just that one style
3. Run queries → observe behavior change
4. NARS RL runs overnight → topology adjusts to the new Creative behavior
5. Tomorrow: topology has learned whether the change was good or bad

No Rust recompilation. No binary rebuild. Edit YAML, reload, observe, learn.

## The 36 YAML Files

```
thinking_styles/
├── convergent/
│   ├── logical.yaml        (τ 0x40)
│   ├── analytical.yaml     (τ 0x41)
│   ├── critical.yaml       (τ 0x42)
│   ├── systematic.yaml     (τ 0x43)
│   ├── methodical.yaml     (τ 0x44)
│   └── precise.yaml        (τ 0x45)
├── creative/
│   ├── creative.yaml       (τ 0xA0)
│   ├── imaginative.yaml    (τ 0xA1)
│   ├── innovative.yaml     (τ 0xA2)
│   ├── artistic.yaml       (τ 0xA3)
│   ├── poetic.yaml         (τ 0xA4)
│   └── playful.yaml        (τ 0xA5)
├── empathic/
│   ├── empathetic.yaml     (τ 0x80)
│   ├── compassionate.yaml  (τ 0x81)
│   ├── supportive.yaml     (τ 0x82)
│   ├── nurturing.yaml      (τ 0x83)
│   ├── gentle.yaml         (τ 0x84)
│   └── warm.yaml           (τ 0x85)
├── direct/
│   ├── direct.yaml         (τ 0x60)
│   ├── concise.yaml        (τ 0x61)
│   ├── efficient.yaml      (τ 0x62)
│   ├── pragmatic.yaml      (τ 0x63)
│   ├── blunt.yaml          (τ 0x64)
│   └── frank.yaml          (τ 0x65)
├── exploratory/
│   ├── curious.yaml        (τ 0x20)
│   ├── exploratory.yaml    (τ 0x21)
│   ├── questioning.yaml    (τ 0x22)
│   ├── investigative.yaml  (τ 0x23)
│   ├── speculative.yaml    (τ 0x24)
│   └── philosophical.yaml  (τ 0x25)
└── meta/
    ├── reflective.yaml     (τ 0xC0)
    ├── contemplative.yaml  (τ 0xC1)
    ├── metacognitive.yaml  (τ 0xC2)
    ├── wise.yaml           (τ 0xC3)
    ├── transcendent.yaml   (τ 0xC4)
    └── sovereign.yaml      (τ 0xC5)
```

Each file: ~40 lines YAML. Total: ~1,440 lines across 36 files.
The entire thinking microcode layer in editable text.
