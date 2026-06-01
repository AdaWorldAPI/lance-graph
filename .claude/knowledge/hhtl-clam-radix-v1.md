# HHTL · CLAM · radix — one bucketed-cascade lookup, four names

> **Created 2026-06-01** (at jan's request). A **MAP** of existing, *probe-gated*
> architecture — it names how four vocabularies already in the tree describe **one
> structure**. It introduces **no new code** and is **not** a build directive. Every
> behavioural claim inherits the probe status of `bf16-hhtl-terrain.md` (M1 PARTIAL,
> M3/M4 NOT RUN) — treat as **CONJECTURE** until the relevant probe drains.

## READ BY: truth-architect, family-codec-smith, palette-engineer, container-architect, integration-lead, savant-research

---

## The unification (one paragraph)

One structure, four roles — the BF16-HHTL-D two-slot weight/value encoding:

- **CLAM** builds the **hierarchy** — a cluster tree (16 → 256 → 4096 buckets) over the
  vectors. `ndarray::hpc::clam::{ClamTree, Cluster}`; farthest-pair split
  (`build_hip_families`), **not** k-means, **not** Ward.
- **radix** is how the CLAM path is **encoded + navigated** — a nibble-prefix key
  (**Slot D**). 16-way branching = one nibble per level = ART `Node16`. The CLAM path *is*
  a radix key; `vart` (north-star WD-3 4-tree index, `Tree::clone()` snapshots) is the trie
  that stores it.
- **HHTL** is the **access discipline** — the cascade that reads progressively deeper
  prefixes of Slot D, rejecting most candidates early. **Branching, NOT prefix-decoding-of-
  one-word** (C1: each level reads a *separate* slot; early termination = skip deeper reads).
- **BF16** is the **value at the bottom** — **Slot V**, the LEAF, loaded only when
  within-bucket precision matters.

> **heel = lookup — the *coarse* one.** The HEEL reads the top nibble (CLAM L0, 16 coarse
> buckets): 3 integer comparisons on the address, **zero data access, no float, ~80 %
> rejected**. It is the cheap *front* of the radix descent, **not** the value fetch — the
> value is the LEAF (BF16 Slot V).

## The map

| HHTL level | Slot D bits (radix depth) | CLAM level | buckets | reads | ~cost |
|---|---|---|---|---|---|
| **HEEL** | `[15:12]` (top nibble) | L0 | 16 coarse | address only, integer reject | 0.5 B |
| **HIP**  | `[15:8]` (2 nibbles) | L1 | 256 (≈ Jina-v5 centroids, 1:1) | + 2nd nibble | 1 B |
| **TWIG** | `[15:4]` (3 nibbles) | L2 | 4096 (≈ COCA alignment) | + 3rd nibble | 1.5 B |
| **LEAF** | Slot D full + **Slot V (BF16)** | leaf | exact value | + BF16 load | 4 B |

Two slots, 32 bits: **Slot D** (radix/CLAM path, 16 b) is **PRIMARY** — the bucket address;
**Slot V** (BF16 `1+8+7`, 16 b) is **REFINEMENT** (C2: *which bucket you land in dominates how
precisely you encode*). Amortized ≈ **0.9 B/weight** *if* 60/25/10/5 termination holds (M4
unrun) vs **2 B** plain BF16.

## Code anchors (where each name lives — `file:line`)

- **CLAM tree:** `ndarray::hpc::clam::{ClamTree, Cluster}` — used `bgz-tensor/src/adaptive_codec.rs:16,154` (CHAODA precision-classify); 16-way at `holograph/src/nntree.rs:51`.
- **HHTL-D / Slot D:** `bgz-tensor/src/hhtl_d.rs` — `build_hip_families`, `HeelBasin`, `HhtlDEntry`, `HhtlDTensor`; shared at `bgz-tensor/src/shared_palette.rs:40,241`.
- **HEEL reject (integer, no float, ~80 %):** `highheelbgz/src/simd_hardened.rs:238` (`heel_filter` — bitmask of survivors), `highheelbgz/src/tensor_bridge.rs` (three-finger HEEL, zero data access), `highheelbgz/src/lib.rs` (spiral-address heel, high-stride walk).
- **Cascade strokes:** `lance-graph-planner/src/physical/cam_pq_scan.rs:23` (HEEL → BRANCH → full; `heel_threshold`), `lance-graph-planner/src/cache/kv_bundle.rs:8` (HEEL 8×8 = 64 routing entries).
- **Heel calibration:** `lance-graph-contract/src/high_heel.rs` — `LensProfile::build`, ICC-calibrated family heel vector (**DESIGNED, not yet called** per `CALIBRATION_STATUS_GROUND_TRUTH.md`).
- **Config knobs (no recompile):** `bgz-tensor/src/cascade.rs` — `heel_min_agreement`, `hip_max_distance`.
- **JIT scan kernel:** `lance-graph/src/cam_pq/jitson_kernel.rs` — `LOAD_HEEL → GATHER → FILTER → … → TOP_K` (Cranelift, AVX-512); contract `lance-graph-contract/src/jit.rs`.

## Firewall placement — why this is the *sound* side

The float/similarity work is in **BUILDING** the CLAM tree — offline, upstream:
*similarity PROPOSES*. The **query-time** path is pure **integer radix navigation** over
Slot D: *CAM ADDRESSES*. BF16 appears **only at the LEAF** refinement, off the reject path.
So the hot cascade has **no query-time float, no resonance** — the exact line drawn for
OGIT-amortized vs. texture-resonance (`a4-resolver-v1.md`) and `I-VSA-IDENTITIES` Test 0 (an
exact bucket key beats resonance). Contradiction detection stays **O(1) bitwise** on Slot D
(XOR + popcount; same bucket / opposite polarity flag — no float decode).

## Where the pattern recurs (axes)

- **Weights / centroids** — the origin (`bf16-hhtl-terrain.md`, `bgz-tensor`).
- **CAM-PQ scan** — `cam_pq_scan.rs` strokes (HEEL → BRANCH → full).
- **Thinking styles** *(deferred, NO driver — `a4-resolver-v1.md`)*: Slot D = class-id / style
  bucket path (radix; *ephemeral* = `vart` COW snapshot), Slot V = BF16 composed-style value
  at LEAF; HEEL = coarse class/style bucket reject; **amortize** = sediment the stable style
  to the OGIT class register (warm→cold). Same cascade, one level up.

## CONJECTURE / probe gate (inherited from `bf16-hhtl-terrain.md`)

Label CONJECTURE until the probe relevant to a change runs:
- **M1** (CLAM 3-level 16-way fit on 256 Jina centroids; knees at L1/L2) = **PARTIAL**.
- **M3** (bucket-only retrieval, no Slot V, ≥ 90 % of full-BF16 quality) = **NOT RUN**.
- **M4** (HHTL termination % per level — the 0.9 B/weight assumption) = **NOT RUN**.
- **Endgame gate** (FINDING): naive-u8 ULP floor = Pearson 0.99986 / Spearman 0.99975; the
  cascade only earns its overhead **above 0.9980**. γ+φ+CDF buys **+2.4 × 10⁻⁴** — the entire
  value of the transform. A thin margin; do not assume the cascade pays for itself unbenchmarked.

## Cross-refs

`bf16-hhtl-terrain.md` (parent — corrections + probe queue) · `a4-resolver-v1.md` (the styles
steer, deferred) · north-star WD-3 (`vart` 4-tree index, `Tree::clone()` snapshots) ·
`ephemeral-warm-cold-lifecycle.md` (hot/warm/cold) · `I-VSA-IDENTITIES` Test 0 (register/radix
beats resonance) · CLAUDE.md § The Codec Stack.
