# arxiv.md — survey breadcrumbs

**Purpose**: running index of arxiv papers that inform the codec/cache/certification
architecture. One paragraph per paper, one section at the end that distills the
cross-paper meta-epiphany. Not a literature review — a memory trail so a fresh
session can see which external reference points the design is anchored to.

**Scope**: only papers whose findings are structurally relevant to the Ada workspace
design (codec stack, HHTL cascade, BGZ palette, γ+φ calibration, thinking engine,
permanent cache, streaming inference, forecast prefetch). Not all papers we read —
just the ones that changed or confirmed a design choice.

---

## Breadcrumbs

### fastsafetensors (IBM Research, May 2025)
- **arxiv**: `2505.23072` · Yoshimura, Chiba, Sethi, Waddington, Sundararaman
- **claim**: safetensors loaders underutilize storage bandwidth because they
  serialize tensor instantiation into host memory before GPU transfer. Batching
  the I/O + offloading preprocessing to GPU via DLPack + GPUDirect Storage gives
  4.8–7.5× speedup on Llama-7B / Bloom-176B model load times.
- **numbers**: 26.4 GB/s storage throughput (vs 5 GB/s baseline), Llama-7B load
  11 s → 2.3 s, Bloom-176B 120 s → 18 s across 8 GPUs.
- **role in our design**: the **GPU-side ad-hoc baseline**. This is "load
  everything fast and hope the working set fits" — no precast, no forecast, no
  permanent cache. Useful reference point for what "not-our-architecture" looks
  like when the constraint is GPU bandwidth instead of battery / RAM / disk.

### NANOMIND (UW Madison et al., Sep 2025 → Mar 2026)
- **arxiv**: `2510.05109` · Li, Zhang, Zeng, Zhang, Xiong, Liu, Hu, Banerjee
- **title**: *Tiny but Mighty: A Software-Hardware Co-Design Approach for
  Efficient Multimodal Inference on Battery-Powered Small Devices*
- **claim**: multimodal inference (LLaVA-OneVision with Qwen2-VL + SigLIP
  encoders) can run 20.8 hours on a battery-powered edge device by decomposing
  the model into modules and dynamically scheduling them across NPU / GPU / DSP
  on a unified-memory SoC.
- **numbers**: –42.3 % energy, –11.2 % GPU memory, 20.8 h runtime with camera
  on, LLaVA-OneVision variants.
- **role in our design**: **hardware-level validation of codebook-cache-only**.
  Under hard power constraints you don't store embeddings, you store codebook
  indices and reconstruct from cache-resident palettes. Per sibling workspace
  commit `1e40514` (user report): our Base17 palette = 256 atoms × 34 bytes =
  8.5 KB fits in L1, NeuronPrint carries 6 bytes per neuron instead of 204.
  NANOMIND proves this pattern already works under real battery constraints
  at the multimodal-inference layer. Our 7-lane encoder + bgz-hhtl-d cascade
  is the same pattern applied one layer deeper — at the codec layer. The
  architecture is not speculative; the hardware evidence for codebook-cache-
  only as a viable substrate exists and ships.

### VibeTensor (NVIDIA + collaborators, Jan 2026)
- **arxiv**: `2601.16238` · Xu, T. Chen, Zhou, Tianqi Chen, Jia, Grover, +9
- **title**: *VibeTensor: System Software for Deep Learning, Fully Generated
  by AI Agents*
- **claim**: LLM coding agents can generate a complete PyTorch-style eager
  tensor runtime (C++20 core, nanobind Python, experimental TS) across
  language bindings + CUDA memory management, validated through automated
  builds + differential tests + microbenchmarks rather than per-change human
  review.
- **key insight we import**: the paper names the **"Frankenstein" composition
  effect** — *"locally correct subsystems interact to yield globally suboptimal
  performance"* (quoted from abstract). The remedy they ship is a
  *stream-ordered caching allocator with diagnostics* that makes composition
  ordering a first-class, inspectable property.
- **role in our design**: names the failure mode we are trying to avoid by
  construction, and points at the remedy shape (stream-ordered composition
  discipline, not per-subsystem correctness). See meta-epiphany below.

---

## Meta-epiphany: ad-hoc loading vs forecast prefetch vs codebook-cache-only

**Verification provenance (2026-04-12):**
All three arxiv IDs WebFetch-verified against arxiv.org. Titles, authors,
dates, and core claims confirmed. Two minor fabrications corrected in this
commit: commit hash `1e405148` → `1e40514` (extra digit), co-author count
+8 → +9 on VibeTensor. The specific numbers from each paper (26.4 GB/s,
42.3% energy, 20.8 h) are transcribed from abstracts, not independently
measured. The **architectural mapping** (axis framing, Frankenstein
inoculation claims, permanent-cache-as-forecast hypothesis) is CONJECTURE
— it is our interpretation of the papers, not what the papers claim about
themselves.

These three papers sit on an axis. Reading them together is the high-level
framing we've been circling around in the certification work without quite
naming.

### The axis — CONJECTURE (our framing, not from the papers)

```
 ad-hoc fast load         forecast prefetch         codebook-cache-only
    (fastsafetensors)      (our target)              (NANOMIND, 1e40514)
    ───────────────── ─────────────────────── ──────────────────────────
    load everything        stream only the          never load at all;
    as fast as possible    lane we predict          cache-resident palette
    when needed            will win this query      + small-index payload
    26.4 GB/s GPU-side     ~84 MB/s CPU-side        L1-resident (8.5 KB)
    GB-scale working set   100s of MB working set   KB-scale working set
```

Each position is the right answer for its constraint. GPU with unlimited VRAM
wants ad-hoc fast load. CPU with a working-set budget wants forecast prefetch.
Battery-powered SoC with no room at all wants codebook-cache-only with O(1)
reconstruction. We live in the middle position and we're *building toward* the
right position as the endgame.

### The Frankenstein trap and where it hides

VibeTensor's warning is that composing locally-correct subsystems doesn't
automatically give a globally-correct system. **EXTERNAL FINDING** (VibeTensor §7).

We've already solved Frankenstein at the **statistical layer** via v2.5
(**FINDING** — each bullet below is measured, see v2.4/v2.5 certification commits):

- Each lane is locally certified (Fisher z 3σ CI, BCa 2σ cross-check).
- CHAODA outlier filter proves the distribution is globally clean, so the
  locally-correct per-lane metrics aggregate honestly.
- The naive-u8 BGZ floor proves every lane locally beats the trivial baseline,
  so composing lanes into a cascade can't land below the floor.
- Reality-anchor pairs at p0/p1/p25/.../p99/p100 ground the 4-decimal
  population metrics into interpretable per-pair errors — no "locally 0.9998
  but globally wrong at the tails" drift.

That is Frankenstein-proofing at the statistics layer. v2.5 is done. **FINDING.**

The **I/O layer** is where the next Frankenstein trap lives. **CONJECTURE — not measured.** User flagged it
directly: if we try to stream all 7 lanes concurrently into one substrate at
runtime, we need 7 × 128 MB = **896 MB** of HttpRangeReader cache. That's a
locally-innocent choice (each lane uses a normal 128 MB chunk) that is
globally catastrophic (blows the working-set budget on exactly the edge
device NANOMIND shows we should be targeting).

The solution is not "tune the chunk size per lane". The solution is
**don't stream 7 lanes concurrently**. The permanent cache (bgz-hhtl-d) IS
the Frankenstein inoculation at the I/O layer: the lanes are composed once
at bake time, the composed result is stored permanently, and at runtime the
substrate streams through exactly one path. The forecast picks which path.
**CONJECTURE — bgz-hhtl-d is not implemented; this is design intent, not measurement.**

### "Permanent cache learns from forecast prefetch" — CONJECTURE

This is the part that wasn't obvious before reading these papers side by side.
**This entire subsection is CONJECTURE — a design hypothesis, not a measured
property.** The permanent cache is not storage — it's a **standing prediction**
about which lane will answer which query cheapest. Every query that early-exits at Lane 1
confirms the forecast. Every query that drops through to Lane 6 is a **forecast
error**, and the residual is training signal.

Two concrete consequences for the design:

1. **The cascade order is not static.** If residual errors cluster around a
   specific subregion of centroid space, the cascade can demote those centroids
   to a "hot" path that starts higher in the lane stack (Lane 3 first instead
   of Lane 1). The `early_exit_cascade` block in the v2.5 JSON already reports
   the cumulative-per-lane fraction. Recording *which* centroids miss at which
   lane would turn that block into a routing table that specializes over time.

2. **The cache eviction policy is inverted.** In a normal cache, cold entries
   get evicted. In a permanent codec cache, *cold entries are the point* —
   they're the ones that survived without ever needing Lane 6, so they're the
   cheapest to serve. Hot entries (the ones that force expensive paths) are
   the eviction candidates because they're paying the full cascade tax anyway
   and we'd rather stream them ad-hoc than keep them warm.

Combine both: the permanent cache is a **residual-aware predictive routing
table**, not a storage tier. Forecast errors are the training signal that
reshapes it. This is what "permanent cache learns from forecast prefetch"
means operationally. **CONJECTURE — none of this routing behavior is implemented or measured.**

### Where we sit after v2.5

- **fastsafetensors** shows what the GPU-side maximum ad-hoc rate looks like
  (26.4 GB/s) and is not what we're building. Reference point only. **FINDING** (paper's own numbers).
- **NANOMIND** hardware-validates the codebook-cache-only destination (8.5 KB
  L1-resident palette + 6-byte index payload). We've been building toward this
  without citation; the citation now exists. **FINDING** (paper's own numbers);
  **CONJECTURE** (that our Base17 palette is architecturally equivalent to their approach).
- **VibeTensor** names the Frankenstein failure mode and points at
  stream-ordered composition as the remedy shape. **EXTERNAL FINDING** (paper §7).
  v2.5 solved Frankenstein at the statistical layer (**FINDING** — measured);
  bgz-hhtl-d solves it at the I/O layer (**CONJECTURE** — not implemented).

The remaining work on the cascade is not *proving each lane* — v2.5 finished
that (**FINDING**). The remaining work is **proving the composition**: demonstrating that
the forecast-driven permanent cache avoids the 896 MB Frankenstein at runtime,
and that residual errors measurably reshape the routing table over time.
**CONJECTURE — this is the next probe, not a result.** That
probe goes into the HHTL queue alongside M1 / I / M2 / M3 / M4.

---

## Citation format for future additions

When adding a new paper, use this shape so the breadcrumbs stay greppable:

```
### <short-name> (<venue/group>, <month year>)
- **arxiv**: `<id>` · <authors short list>
- **title**: *<exact title if not obvious from short-name>*
- **claim**: one-sentence core technical claim
- **numbers**: the specific figures that justify the claim (throughput, energy, accuracy)
- **role in our design**: one paragraph on how this paper changes / confirms
  a design choice in the Ada workspace. If it doesn't change anything, don't
  add it — this file is not a reading list.
```

And if the paper is on the axis (ad-hoc ↔ forecast ↔ codebook-cache-only),
name the axis position explicitly in the role paragraph.

---

## Crossword cluster (2026-07-29) — the SEMANTIC teacher rung

> Operator-supplied, five papers at once, with the framing:
> *"sudoku > numbers; crossword > semantic + lewensteyn."*
> Read against `.claude/plans/epistemic-quadrant-materialization-v1.md` §4d
> (PROBE-SUDOKU-TEACHER + the teacher ladder). These five insert a rung
> **between** Sudoku and stockfish-rs, and the rung is not optional enrichment —
> see the meta-epiphany at the end of this section.

### Down and Across (Kulshreshtha, Kovaleva, Shivagunde, Rumshisky · ACL 2022)
- **arxiv**: `2205.10442`
- **claim**: crossword solving is a distinct NLU benchmark — it demands
  knowledge retrieval AND global constraint satisfaction simultaneously.
- **numbers**: ~9,000 NYT puzzles over 25 years; **500,000+ unique clue-answer
  pairs**; two baseline families (QA seq2seq / retrieval, plus a *non-parametric
  constraint-satisfaction* full-puzzle solver).
- **role in our design**: the **corpus + the two-tier architecture receipt**.
  Their split (per-clue answering ⊕ whole-grid constraint satisfaction) is the
  same two tiers as §4d's Sudoku design (lane-local election ⊕ predicate sweep),
  arrived at independently in a semantic domain. The 500k clue-answer pairs are
  a candidate teacher corpus of the right order for a 16 MB-class substrate.

### Decrypting Cryptic Crosswords (Rozner, Potts, Mahowald · Apr 2021)
- **arxiv**: `2104.08620`
- **claim**: cryptic clues are "semantically complex, highly compositional" —
  each clue carries BOTH a semantic definition and a character-level wordplay
  cipher, and models fail to generalize the way humans do.
- **numbers**: three non-neural baselines + T5 all fail; curriculum pre-training
  (word unscrambling etc.) "considerably improves" T5 but does not close the
  human generalization gap. Includes a deliberately hard split + perturbation
  studies.
- **role in our design**: **the falsifiable Doppelspalt instance, and the
  Levenshtein-is-literal receipt.** A cryptic clue is definition ⊕ wordplay —
  two slits over one answer, where *neither half alone determines it and both
  together do*, and the answer is UNIQUE AND KNOWN. That makes it a checkable
  cross-term test for the §4c cross-term rule, which is what that rule has been
  missing. Second: cryptic wordplay operations ARE the edit operations —
  anagram = permutation, insertion/deletion/substitution = the literal
  Levenshtein primitives. So here edit distance is not a *metric over* the
  content, it is the *content*.

### Language Models are Crossword Solvers (Saha, Chakraborty, Saha, Garain · NAACL 2025)
- **arxiv**: `2406.09043`
- **claim**: off-the-shelf LLM clue-solving plus a search algorithm over the
  grid solves full puzzles at high accuracy.
- **numbers**: **2–3× prior SOTA** on cryptic benchmarks; **93 % on NYT** grids.
- **role in our design**: **BASELINE ONLY — explicitly not a method to adopt.**
  This stack is no-LLM in the hot path by construction (DeepNSM replaces
  transformer inference: 680 GB → 16.5 MB, 50 ms/token → <10 µs/sentence). The
  transferable part is the *architecture* (per-clue solve + search over the
  grid), which is the same two tiers as `2205.10442`; the 93 % is the number a
  no-LLM substrate would be measured against, not a technique to import.
  Recording the distinction because "LLM gets 93 %" is exactly the kind of
  result that quietly becomes a design.

### CrossWordBench (Leng, Huang, Huang, Lin, Cohen, Wang, Huang · Mar 2025)
- **arxiv**: `2504.00043`
- **claim**: controllable crossword generation as a multimodal reasoning
  benchmark — semantic constraints from clues AND structural constraints from
  the grid, in both text and image form.
- **numbers**: 20+ models; reasoning-equipped LLMs substantially beat
  non-reasoning variants *by exploiting crossing-letter constraints*; LVLM
  performance correlates strongly with **grid-parsing accuracy**.
- **role in our design**: **the difficulty knob = a ready-made inertness-test
  axis.** Their *prefill-ratio* control is exactly the calibrated parameter the
  P0 falsifiability rule demands ("raising it must silence something, lowering
  it must admit something") — the `heel_threshold` lesson, solved upstream. The
  LVLM finding is the sharper one for us: performance gated on *grid-parsing*
  means the bottleneck was ADDRESSING the structure, not reasoning over it —
  which is the whole claim of a key-addressable substrate (the key prerenders
  nodes with zero value decode).

### Crossword: Semantic Compression via Masking (Li, Jin, Xiang, Shen, Cui · Apr 2023)
- **arxiv**: `2304.01106`
- **claim**: mask semantically-minor words at the encoder, reconstruct them from
  context with a Transformer decoder; beats symbol-level compression (Huffman,
  UTF-8) because it stops treating text as i.i.d. symbols.
- **numbers**: reports "much higher compression efficiency" than Huffman/UTF-8;
  **no specific ratio given in the abstract — treat as unquantified.**
- **role in our design**: **an independent instance of the refined zero-copy
  falsifier.** Their masking criterion — *drop what context can regenerate* — is
  the same rule as `zero-copy-lens-law.md` § "The one apparent exception" as
  REFINED 2026-07-29: what is reproducible from the lens must not be stored;
  only what is not reproducible earns storage. Two domains (text compression /
  SoA lane eligibility), one criterion. This is the strongest cross-domain
  hit in the cluster because it is a *mechanism* match, not a rhyme: both
  compute "is this recoverable from its context?" and both act on the answer.

---

## Meta-epiphany — Sudoku proves the LOOP; crossword proves the SUBSTRATE

The operator's two-line framing (*sudoku > numbers; crossword > semantic +
lewensteyn*) names a gap in the §4d probe that is easy to miss:

**A Sudoku digit has no semantics, so Sudoku exercises none of this repo.**
The codebook, the palette, DeepNSM's 4096-word COCA vocabulary, CLAM,
Hamming-over-fingerprints, the Levenshtein/CER surface — all bypassed. A
Sudoku teacher validates the *promotion loop* (explore → learned → frozen,
held-out gating, fork-return, quadrant census) on an oracle that is free and
exact. That is real and it is the right first step, but it is a test of the
**harness**, not of the substrate the harness drives.

A crossword answer is a WORD. It routes through the actual encoding stack, and
its constraints are of both kinds this workspace separates:

| | Sudoku | Crossword |
|---|---|---|
| alphabet | closed (1–9) | **open (natural language)** |
| cell constraint | exact equality | crossing LETTER = character-level |
| answer length | fixed 1 | **variable, must fit a fixed slot** |
| distance metric | Hamming (no indel possible) | **Hamming across the grid + Levenshtein within the answer** |
| what it tests | the promotion loop | **the codebook / DeepNSM / CLAM substrate** |
| clue structure | none | **definition ⊕ wordplay = two slits, unique answer** |

So the teacher ladder gains a rung, and the rung is load-bearing:

**T0 Sudoku** (numbers; exact; closed alphabet; no adversary; binary outcome)
→ **T0.5 Crossword** (semantic; exact; OPEN alphabet; no adversary; and the
first rung where edit distance is the content rather than the metric)
→ **T1 stockfish-rs** (graded centipawns; adversarial; deep counterfactuals;
GPL data-only seam — oracle, never linked).

Each rung adds exactly one new capability to the *same* teacher-agnostic
promotion record `(position_key, elections[], outcome_grade, teacher_path)`:
T0 adds nothing (it establishes the record), T0.5 adds an open alphabet and a
real encoder, T1 adds grading and an adversary. **Status: the ladder is design;
T0 is in build (§4d), T0.5 and T1 are unbuilt.** Nothing here is a measured
result on our substrate — the papers are external anchors, not our numbers.
