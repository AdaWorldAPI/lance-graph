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
  commit `1e405148` (user report): our Base17 palette = 256 atoms × 34 bytes =
  8.5 KB fits in L1, NeuronPrint carries 6 bytes per neuron instead of 204.
  NANOMIND proves this pattern already works under real battery constraints
  at the multimodal-inference layer. Our 7-lane encoder + bgz-hhtl-d cascade
  is the same pattern applied one layer deeper — at the codec layer. The
  architecture is not speculative; the hardware evidence for codebook-cache-
  only as a viable substrate exists and ships.

### VibeTensor (NVIDIA + collaborators, Jan 2026)
- **arxiv**: `2601.16238` · Xu, T. Chen, Zhou, Tianqi Chen, Jia, Grover, +8
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

These three papers sit on an axis. Reading them together is the high-level
framing we've been circling around in the certification work without quite
naming.

### The axis

```
 ad-hoc fast load         forecast prefetch         codebook-cache-only
    (fastsafetensors)      (our target)              (NANOMIND, 1e405148)
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
automatically give a globally-correct system. We've already solved Frankenstein
at the **statistical layer** via v2.5:

- Each lane is locally certified (Fisher z 3σ CI, BCa 2σ cross-check).
- CHAODA outlier filter proves the distribution is globally clean, so the
  locally-correct per-lane metrics aggregate honestly.
- The naive-u8 BGZ floor proves every lane locally beats the trivial baseline,
  so composing lanes into a cascade can't land below the floor.
- Reality-anchor pairs at p0/p1/p25/.../p99/p100 ground the 4-decimal
  population metrics into interpretable per-pair errors — no "locally 0.9998
  but globally wrong at the tails" drift.

That is Frankenstein-proofing at the statistics layer. v2.5 is done.

The **I/O layer** is where the next Frankenstein trap lives. User flagged it
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

### "Permanent cache learns from forecast prefetch"

This is the part that wasn't obvious before reading these papers side by side.
The permanent cache is not storage — it's a **standing prediction** about which
lane will answer which query cheapest. Every query that early-exits at Lane 1
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
means operationally.

### Where we sit after v2.5

- **fastsafetensors** shows what the GPU-side maximum ad-hoc rate looks like
  (26.4 GB/s) and is not what we're building. Reference point only.
- **NANOMIND** hardware-validates the codebook-cache-only destination (8.5 KB
  L1-resident palette + 6-byte index payload). We've been building toward this
  without citation; the citation now exists.
- **VibeTensor** names the Frankenstein failure mode and points at
  stream-ordered composition as the remedy shape. v2.5 solved Frankenstein at
  the statistical layer; bgz-hhtl-d solves it at the I/O layer.

The remaining work on the cascade is not *proving each lane* — v2.5 finished
that. The remaining work is **proving the composition**: demonstrating that
the forecast-driven permanent cache avoids the 896 MB Frankenstein at runtime,
and that residual errors measurably reshape the routing table over time. That
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
