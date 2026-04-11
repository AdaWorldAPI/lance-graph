# KNOWLEDGE: Frankenstein Composition Checklist

## READ BY: ALL AGENTS. truth-architect enforces.
## SOURCE: Xu et al. 2026, "VibeTensor" (arXiv:2601.16238), §7
## APPLIES TO: any multi-encoding stack, any agent-generated architecture

---

## The Frankenstein Effect (definition)

> Individually reasonable components compose into a globally suboptimal design.
> Locally correct subsystems interact to yield emergent failure modes that
> no single subsystem's tests can catch.

VibeTensor demonstrated this with a correctness-first autograd gate that
serialized concurrent backward passes — each subsystem was correct in
isolation, but the composition starved efficient GPU kernels.

**This is the exact risk for the lance-graph codec stack.** Each encoding
passes its own lane certification (v2.4/v2.5), but the full pipeline
(Base17 → BGZ17 palette → HighHeelBGZ → ZeckF64 → CLAM → pairwise cosine)
has never been tested end-to-end for composed fidelity.

---

## Meta-Checklist: Before Composing Subsystems

### 1. Single-shot correctness ≠ composed stability

```
RULE: A subsystem that passes unit tests in isolation may fail
      under repeated composition (multi-step, multi-layer, multi-query).

TEST: Run the full pipeline (encode → store → retrieve → decode → compare)
      at least 100× in a loop. Check for:
      - Drift (does ρ degrade over iterations?)
      - State leaks (does one query's state affect the next?)
      - Accumulation (do rounding errors compound?)

LANCE-GRAPH INSTANCE:
  Base17 encode → palette lookup → HEEL scan → HIP refine → LEAF decode
  → pairwise cosine → compare to f32 reference.
  Has this chain been run 100× on held-out data? NO.
```

### 2. Serialization bottlenecks hide in correctness gates

```
RULE: Safety mechanisms (mutexes, global locks, sequential validation)
      that make a subsystem correct can serialize the pipeline and
      starve downstream components.

TEST: Profile the full pipeline for serialization points.
      Measure wall-clock time per stage. If any stage is >10× slower
      than the theoretical minimum, check for correctness gates.

LANCE-GRAPH INSTANCE:
  Does the HHTL cascade have a sequential gate between levels?
  Does the CLAM tree lock during descent?
  Does palette lookup serialize on the codebook?
  UNKNOWN — not profiled.
```

### 3. Validation gaps multiply at boundaries

```
RULE: Agent-generated code passes LOCAL tests while failing at
      BOUNDARIES between subsystems. The gap is at the handoff:
      - type mismatches (i16 vs i8 vs BF16 vs f32 at interfaces)
      - convention mismatches (row-major vs column-major, endianness)
      - precision mismatches (one stage rounds, the next assumes exact)
      - semantic mismatches (one stage uses rank, the next uses value)

TEST: At every boundary between two encodings, insert a round-trip
      assertion: encode in system A → decode → re-encode in system B
      → compare to original. If |Δ| > expected precision, the boundary
      is lossy.

LANCE-GRAPH INSTANCE:
  Base17 i16 → ZeckF64 u64 boundary: zeckf64_from_base() — tested?
  StackedN BF16 → Base17 i16 boundary: precision loss? — tested?
  NeuronPrint palette index → Base17 atom: round-trip fidelity? — CONJECTURE
```

### 4. "Works once" ≠ "works under load"

```
RULE: A pipeline that produces correct output for one query may fail
      under concurrent or batched load due to:
      - shared mutable state (global codebooks, cached palettes)
      - memory allocation patterns (fragmentation under batch)
      - SIMD alignment assumptions (broken by odd batch sizes)

TEST: Run the pipeline with batch sizes 1, 17, 256, 1024, 10000.
      Check that output is identical regardless of batch size.
      Check that throughput scales linearly (or explain why not).

LANCE-GRAPH INSTANCE:
  Does Base17 golden-step traversal depend on batch alignment?
  Does the palette semiring cache invalidate correctly under batch?
  UNKNOWN.
```

### 5. Correctness-first generation misses performance objectives

```
RULE: When agents build subsystems one at a time (correctness-first),
      global performance objectives are not encoded early. The result
      is a system that is correct but slow because performance-critical
      paths were never designed — they emerged from the composition.

TEST: Define the end-to-end performance budget BEFORE building subsystems.
      Each subsystem gets a time/space budget. If it exceeds budget,
      it must be redesigned, not worked around.

LANCE-GRAPH INSTANCE:
  What is the target latency for a single pairwise cosine lookup
  through the full HHTL cascade?
  What is the target throughput (pairs/second)?
  UNDEFINED — no performance budget exists.
```

### 6. Redundant abstractions accumulate silently

```
RULE: Agent-generated code creates new abstractions for each problem
      rather than reusing existing ones. Over time, this produces
      multiple overlapping representations of the same concept.

TEST: Search for type duplication. If two structs represent the same
      concept with different field names, one must be eliminated.
      See: docs/TYPE_DUPLICATION_MAP.md

LANCE-GRAPH INSTANCE:
  Base17 (bgz17) vs BasePattern (codec-research) — same concept?
  SpoBase17 (contract) vs ZeckBF17Edge (codec-research) — overlap?
  HighHeelBGZ (contract) vs HHTL cascade (bgz-tensor) — boundary?
  ThinkingStyleFingerprint (neuron_hetero) vs ThinkingStyleVector (planner)?
```

### 7. Test what you compose, not just what you build

```
RULE: Integration tests must exercise the ACTUAL composition path,
      not a simplified version. If the production path is
      A → B → C → D, the integration test must run A → B → C → D,
      not A → D with B and C mocked.

TEST: Write at least one integration test per critical path:
  - Encode path: raw weights → Base17 → palette → ZeckF64
  - Search path: query → HEEL → HIP → TWIG → LEAF → pairwise cosine
  - Causal path: SPO triple → CausalEdge64 → NARS truth propagation
  - Full loop: encode → store → search → decode → compare to reference

LANCE-GRAPH INSTANCE:
  Does any test in the repo exercise the full encode→search→decode loop?
  UNKNOWN — check with: grep -rn "end_to_end\|integration\|full_pipeline"
```

---

## The Composition Test Matrix

For any two subsystems A and B that connect in the pipeline, verify:

```
[ ] A's output type matches B's input type exactly (not "close enough")
[ ] A's output precision is sufficient for B's input requirements
[ ] A's output semantics (rank vs value vs bucket) match B's expectation
[ ] Round-trip A→B→A preserves the invariant A is supposed to protect
[ ] The A→B boundary has a test that runs on real data (not synthetic)
[ ] The A→B boundary has been profiled for serialization
[ ] The A→B boundary works at batch sizes 1, 256, and 10000
```

### Critical boundaries in lance-graph:

```
Boundary                          Tested?  Profiled?  Batch-tested?
─────────────────────────────────  ───────  ─────────  ─────────────
StackedN BF16 → Base17 i16         ?        ?          ?
Base17 i16 → BGZ17 palette         ?        ?          ?
BGZ17 palette → HighHeelBGZ        ?        ?          ?
HighHeelBGZ → ZeckF64 u64          partial  ?          ?
ZeckF64 → CLAM tree                v2.5     ?          ?
CLAM → pairwise cosine             ?        ?          ?
Base17 → NeuronPrint 6D            ?        ?          ?
CausalEdge64 → SPO graph           ?        ?          ?
Thinking engine → codec selection  ?        ?          ?
```

Fill these in as probes run. Every "?" is a potential Frankenstein boundary.

---

## Process Integration

### When the truth-architect reviews a proposal:

Add this question to the review checklist:

```
## Frankenstein check
- Does this proposal compose with its upstream producer?
- Does this proposal compose with its downstream consumer?
- Has the composition been tested (not just the component)?
- Is there a performance budget for this component?
- Does this introduce a new abstraction that overlaps an existing one?
```

### When any agent proposes a new encoding or layer:

```
BEFORE implementing:
1. Name the upstream producer and downstream consumer
2. Define the boundary contract (type, precision, semantics)
3. Write the boundary round-trip test FIRST
4. Set a performance budget (time, space)
5. Only THEN implement the component
```

### Citation

```
Xu et al., "VibeTensor: System Software for Deep Learning, Fully Generated
by AI Agents," arXiv:2601.16238, Jan 2026. §7: "The Frankenstein composition
effect — locally correct subsystems interact to yield globally suboptimal
performance."
```
