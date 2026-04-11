# Combined Research: BF16-HHTL Stack — Session Notes × Canonical Knowledge

> **Status:** NON-CANONICAL. This file integrates the 4 session notes with the
> canonical knowledge files landed on main. It is a research index, not
> doctrine. Canonical claims live in `.claude/knowledge/`.
>
> **Date**: 2026-04-11
> **Read by**: any agent resuming the BF16-HHTL thread — check the probe queue
> in `bf16-hhtl-terrain.md` before proposing changes.
>
> **Hard rule from `CLAUDE.md § Knowledge Activation`**: If the probe relevant
> to a proposed change is NOT RUN, the next deliverable is the probe, not more
> synthesis. This file is an index; the action item at the bottom is Probe M1.

---

## Purpose

During this session I wrote 4 research notes that iterated on a BF16-HHTL-D
proposal across 5 corrections. In parallel (via PR, outside this session), the
canonical knowledge files `bf16-hhtl-terrain.md`, `two-basin-routing.md`, and
an updated `zeckendorf-spiral-proof.md` landed on main. After rebasing, I now
have both sets of content.

This file:

1. **Indexes** both sets (canonical + session) so future agents can find the derivation.
2. **Labels each claim** as CANONICAL / CONJECTURE-SESSION / SOURCE-GROUNDED / FRAMING.
3. **Notes where session notes add to or contradict canonical** knowledge.
4. **Identifies the pairwise rule gap** (session notes did not derive it; canonical does).
5. **Proposes the one combined probe** (M1) that tests the stack as a whole.

## The two sets of files

### Canonical (landed on main, authoritative)

| File | Role | Status |
|---|---|---|
| `.claude/knowledge/phi-spiral-reconstruction.md` | φ-spiral theory, family zipper, stride/offset, collision proof | CANONICAL (pre-existing) |
| `.claude/knowledge/primzahl-encoding-research.md` | Prime fingerprint, Zeckendorf vs BF16 vs prime encoding | CANONICAL (pre-existing) |
| `.claude/knowledge/zeckendorf-spiral-proof.md` | φ-spiral proof with scope limitation header | CANONICAL (corrected) |
| `.claude/knowledge/bf16-hhtl-terrain.md` | 5 correction chain + 5 hard constraints + probe queue | CANONICAL (distilled from this session) |
| `.claude/knowledge/two-basin-routing.md` | Two-basin doctrine + pairwise rule + representation routing table | CANONICAL (new insight) |
| `.claude/agents/truth-architect.md` | Agent that guards measurement-before-synthesis | CANONICAL (process rail) |

### Session (this session, non-canonical, committed to branch)

| File | Role | Relationship to canonical |
|---|---|---|
| `.claude/INVARIANT_MATRIX_RESEARCH.md` | Invariant axes, encoding-as-lens framing | FRAMING, overlaps with `two-basin-routing.md § Representation Routing Table` |
| `.claude/ONE_FORTIETH_SIGMA_LENS.md` | 1/40 σ radial signature, Probes A/B/C | CONJECTURE-SESSION, not covered in canonical; deserves probe queue entry |
| `.claude/RING_PERTURBATION_PROPAGATION.md` | NARS+ONNX+SiLU+ReLU runtime pipeline | CONJECTURE-SESSION, not covered in canonical; depends on 1/40 σ |
| `.claude/BGZ17_ELEVEN_SEVENTEEN_RATIONALE.md` | Source-grounded 11/17 golden-step rationale | SOURCE-GROUNDED, duplicates `zeckendorf-spiral-proof.md` + hard constraint C5 |

## Where session notes overlap with or contradict canonical

### Session ✓ Canonical (consistent, session is a long-form derivation)

- **11/17 golden step** (BGZ17_ELEVEN_SEVENTEEN_RATIONALE.md) matches
  canonical C5 in `bf16-hhtl-terrain.md`: "gcd(11,17) = 1 → full coverage.
  |17/φ − 11| = 0.4934 → nearest integer to 17/φ. Fibonacci mod 17 misses
  {6,7,10,11} — only 13/17 residues visited."
- **γ+φ valid regime** (INVARIANT_MATRIX_RESEARCH.md § Constants) matches
  canonical C3: "Pre-rank discrete selector valid. Post-rank monotone
  transform dead."
- **Invariant axes framing** (INVARIANT_MATRIX_RESEARCH.md) is a coarser
  version of `two-basin-routing.md § Representation Routing Table`, which
  organizes by question/basin/encoding instead of by axis.

### Session ✗ Canonical (session got it wrong — read the correction)

- **Progressive prefix decoding of one BF16 word** (the BF16-HHTL-D proposal
  during iteration 3 of the session) is **wrong**. Canonical C1:
  *HHTL is branching, not prefix. Each cascade level reads a SEPARATE stored
  slot. Do NOT pack multiple cascade levels into one word. Do NOT sacrifice
  mantissa precision to fit a descriptor.*
- **Slot D as structural tag** (family × stride × phase, during iteration 4)
  is **wrong**. Canonical C2: *Bucketing > resolution. Slot D is the primary
  carrier (bucket address); Slot V is refinement. Slot D should be a CLAM
  tree path, not a structural descriptor.*

### Session missed (canonical adds this, session did not derive it)

- **The Pairwise Rule** from `two-basin-routing.md`:

  > **Pairwise cosine MUST stay pairwise.**
  > DO NOT: average pairwise distances into centroid distances; replace
  > pairwise with centroid-to-centroid; assume centroid proximity implies
  > pair proximity.
  > **The deep asymmetry: centroids survive compression, pairs do not.**

  This is a fifth hard constraint I did not derive. It directly limits the
  Slot D = CLAM path proposal: Slot D is a centroid address, which carries
  bucket identity but not pairwise ranking. Pairwise cosine must live in a
  **separate encoding**, kept non-aggregated.

- **Two-Basin Doctrine** from `two-basin-routing.md`:

  > **Basin 1 (Semantic)**: discrete, stable, addressable — CAM + COCA 4096 + DeepNSM.
  >   Answers "WHAT IS IT?"
  > **Basin 2 (Distribution)**: continuous, compressible, dynamic — BGZ family, centroids, wave models.
  >   Answers "HOW DOES IT BEHAVE?"
  > Mixing them is the root cause of most design confusion.

  My session kept blending basins (treating Slot D as both a semantic address
  and a distribution shape). Two-basin doctrine says these are **different
  encodings in different basins** — do not fuse.

## Integrated view: the current best architecture

This is the state of the proposal after all corrections, marked with each
claim's status.

### Storage layer (2× BF16 branching)

```
Per weight: 32 bits total
┌─────────────────┬─────────────────┐
│     Slot V      │     Slot D      │
│   BF16 value    │  CLAM tree path │
│   (Basin 2)     │   (Basin 2)     │
└─────────────────┴─────────────────┘
```

- **Slot V**: standard BF16 (1 sign + 8 exponent + 7 mantissa). Zero precision
  loss vs. plain BF16. Status: CANONICAL form factor.
- **Slot D**: 16 bits. 12-bit CLAM path + 4-bit flags. Status: CONJECTURE
  pending Probe M1.

### Slot D layout (CONJECTURED)

```
bits 15..12 = CLAM L0: 16 coarse clusters (HEEL scan target)
bits 11..8  = CLAM L1: 256 mid-clusters  (HIP, 1:1 Jina-v5 centroids)
bits  7..4  = CLAM L2: 4096 terminal     (TWIG, COCA alignment)
bits  3..0  = flags: 1 polarity + 1 γ-phase + 2 reserved
```

All four alignments (16 coarse, 256 Jina-centroid, 4096 COCA, 4 γ-phase) are
**CONJECTURES** until Probe M1 returns. Status in `bf16-hhtl-terrain.md`
probe queue: NOT RUN.

### What lives in which basin (from `two-basin-routing.md`)

- **Basin 1 (Semantic)**: CAM fingerprints (48-bit), COCA 4096 indices, Jina v5
  centroid tags. These are **stable identities**, do not change under
  compression. Do NOT put these in Slot D of a Basin-2 encoding.
- **Basin 2 (Distribution)**: Slot V (BF16 value), Slot D (CLAM tree path),
  pairwise cosine, signed residuals, BGZ wave shape. These are **behavioral
  descriptors**, continuous, compressible.

**The pairwise rule applies to Basin 2**: pairwise cosine must not be replaced
by centroid-to-centroid distance.

### What the session notes contribute that canonical does not

Two session conjectures that are **not covered** in any canonical file:

1. **1/40 σ radial signature** (ONE_FORTIETH_SIGMA_LENS.md):
   40-band distance quantization around a query, σ-normalized per query,
   with Hole variant (diagonal=0) to exclude self-reference. Claims coverage
   of manifold curvature (via band occupancy), magnitude (via σ
   normalization), and sparse structure (via empty bands).
   - **Status**: CONJECTURE-SESSION. Not in probe queue. Should be added as
     a new probe entry: "Probe R1: Does 1/40 σ profile distance correlate
     with semantic distance at ρ > 0.5?"

2. **Ring perturbation pipeline** (RING_PERTURBATION_PROPAGATION.md):
   NARS + ONNX + SiLU + ReLU stacked as a runtime mechanism for query-adaptive
   ring signatures with contradiction diagnostics. Depends on the 1/40 σ
   lens passing first.
   - **Status**: CONJECTURE-SESSION, second-order (only meaningful if R1 passes).
     Probe R2: "Does NARS adjacency agreement between ring bands converge
     over a held-out query sequence?"

These should be added to `bf16-hhtl-terrain.md § Probe Queue` with status
NOT RUN and priority P5/P6 (below M1-M4 which are higher priority).

## The combined probe: Probe M1 tests everything at once

The canonical `bf16-hhtl-terrain.md § Probe Queue` lists Probe M1 as P0:

> **M1 (P0)**: CLAM 3-level 16-way tree on 256 Jina centroids? Knees at L1/L2?
> **Pass**: Clean tree, 16-way natural
> **Fail**: Degenerate tree, wrong depth
> **Status**: NOT RUN

**Why M1 is the combined test**: M1 directly exercises the claim that the
entire Slot D layout makes sense. If CLAM naturally builds a 3-level 16-way
tree over the 256 Jina centroids, then:

- ✓ Slot D 4-bit-per-level allocation is justified (canonical Slot D layout).
- ✓ HIP level aligning with 256 centroids is justified (two-basin routing).
- ✓ TWIG level aligning with 4096 COCA terminal buckets is justified (if the
  tree naturally descends that deep).
- ✓ The bucketing > resolution principle has a foundation (because the
  CLAM tree gives us clean, hierarchical bucket identity).
- ✓ Contradiction detection via bitwise Slot D path comparison becomes
  meaningful (because the path is a valid CLAM address).
- ✓ The 1/40 σ lens conjecture can be tested NEXT (because it requires the
  same 256-centroid codebook that M1 operates on).

**If M1 fails**: the Slot D layout, the HIP/TWIG alignment claims, and the
contradiction detection approach all need to be rethought. The fallback is
to use a different bucketing scheme (e.g., direct K-means at 256, or the
existing 4096² sparse table from the L2 lane), and the 16/256/4096 alignment
is abandoned.

**One probe. One afternoon. Decides the entire stack.**

## The pending push situation (not a probe, but blocks this file from being pushed)

The rebase is done. Local is 3 commits ahead of remote. Remote has 2 pre-rebase
commits of mine (old hashes) that `git cherry` flagged as non-equivalent by
patch-id. **Investigation showed**: the +lines introduced by each pair of
commits (remote 5500235 vs local 8e74971, remote 98be5e9 vs local b0540a4)
are byte-for-byte identical. The patch-id difference is due to surrounding
context lines shifting during the rebase, not actual content divergence.

The only file deletion in the diff is `.claude/prompts/SESSION_BGZ_TENSOR_HYDRATE.md`,
which was a 0-byte empty placeholder, deleted by main's `9eee5a6` (not by any
of my commits). The diff is strictly additive in terms of meaningful content.

**Force-push-with-lease is safe and additive.** Awaiting user go-ahead.

## Action items, in order

1. **Force-push-with-lease** this branch (once authorized) so the session notes
   and this combined research note reach the remote. The rebase + commit state
   is clean; no content is lost.

2. **Run Probe M1** (CLAM 3-level 16-way tree on 256 Jina centroids).
   - Location: new example file `crates/thinking-engine/examples/clam_jina_256_probe.rs`.
   - Uses the existing ndarray CLAM tree build function
     (`lance-graph/src/graph/neighborhood/clam.rs`).
   - Loads the 256-centroid semantic codebook from disk
     (built earlier in session via `semantic_codebook.rs`).
   - Reports: tree depth, branching factors per level, knee-deltas, and
     whether the natural tree shape matches the 16/256/4096 target.
   - Cost: ~200 LOC, one afternoon.

3. **Record result in `bf16-hhtl-terrain.md § Probe Queue`**:
   - If PASS: promote the Slot D layout from CONJECTURE to FINDING.
     Add Probe R1 (1/40 σ radial signature) and Probe R2 (ring perturbation)
     as new queue entries at P5/P6.
   - If FAIL: update `bf16-hhtl-terrain.md § Current Architecture` with the
     correction. The Slot D layout needs rethinking.

4. **Continue in probe-driven order**, never synthesis-first. The canonical
   process rule in `CLAUDE.md § Knowledge Activation`:
   *If the relevant probe is NOT RUN, the next deliverable is the probe,
   not more synthesis.*

## Cross-reference table

For every claim in the 4 session notes, its canonical replacement:

| Session note section | Canonical replacement |
|---|---|
| `INVARIANT_MATRIX_RESEARCH.md § Invariant axes` | `two-basin-routing.md § Representation Routing Table` |
| `INVARIANT_MATRIX_RESEARCH.md § Constants discipline` | `bf16-hhtl-terrain.md § C3 (γ+φ regime)` |
| `INVARIANT_MATRIX_RESEARCH.md § Encoding candidates table` | `two-basin-routing.md § Encoding Comparison Matrix` |
| `ONE_FORTIETH_SIGMA_LENS.md § What 1/40 σ is` | NEW — not yet in canonical |
| `ONE_FORTIETH_SIGMA_LENS.md § Probes A/B/C` | Should become Probe R1 in `bf16-hhtl-terrain.md § Probe Queue` |
| `RING_PERTURBATION_PROPAGATION.md § Four-stage pipeline` | NEW — not yet in canonical |
| `RING_PERTURBATION_PROPAGATION.md § Probes D/E/F/G` | Should become Probe R2 in queue |
| `BGZ17_ELEVEN_SEVENTEEN_RATIONALE.md § 11/17 math` | `zeckendorf-spiral-proof.md § Fibonacci mod 17 (NOT covered)` + `bf16-hhtl-terrain.md § C5` |
| `BGZ17_ELEVEN_SEVENTEEN_RATIONALE.md § PCDVQ polar decomposition` | Not in canonical; should be added to `zeckendorf-spiral-proof.md` as a footnote |

## The one-sentence summary

**The session produced coherent long-form derivation that the canonical knowledge
files now supersede; the two pieces of session content not yet in canonical
(1/40 σ radial signature and the ring perturbation pipeline) should be added
to the probe queue as R1/R2 and run after Probe M1 passes or fails.**
