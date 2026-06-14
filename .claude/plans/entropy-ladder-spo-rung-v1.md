# Entropy-Ladder SPO Rung Decomposition — v1

> **Status:** PLATEAU — R1 (foundation) SHIPPED; R2–R6 planned (probe-gated where marked).
> **Confidence:** R1 measured (ρ=−0.78 reliability-proxy); R2/R3 unblocked; R4/R5 probe-first; R6 gated on the Mailbox-SoA map.
> **Date:** 2026-06-14
> **Branch:** `claude/wonderful-hawking-lodtql` (ndarray + lance-graph)
> **Owner:** main thread (Opus)

## The Click — one coordinate unifies the stack

**A fact's position on the cognitive ladder IS an entropy level.** The same
scalar orders SPO triples, NARS truth, and the linguistic ladder:

```
  Staunen  ── high entropy ── raw stimulus, not yet crystallized ── SYNTAX
     │                                                                 │
  (semantics sits between)                                       (meaning)
     │                                                                 │
  Wisdom   ── low  entropy ── crystalline knowledge / settled fact ─ PRAGMATICS
```

`entropy = 1 − c·|2f − 1|` over a `CausalEdge64`'s NARS `(f, c)` — one minus the
decisiveness of the NARS expectation. Validated as a **reliability proxy**:
ρ(entropy, empirical prediction accuracy) = **−0.78** over a synthetic NARS
population (the more crystalline the edge, the more reliably its belief matches
fresh reality), grounded via `ndarray::hpc::reliability`.

## Architecture decisions (locked this session)

1. **No re-quantization.** The SPO rungs read the **3×palette-256 indices already
   inside `CausalEdge64`** (`s_idx`/`p_idx`/`o_idx`) — "exact enough" (operator).
   `cam_pq` is NOT used to re-encode SPO for the ladder.
2. **Pearl 2³ mask = the 8 SPO iterations.** The existing causal mask at
   `CausalEdge64` [42:40] enumerates the 8 `(S,P,O)` subsets; `decompose_spo`
   zeroes inactive components and emits an HHTL-routable `basin_key`.
3. **Flavors / classes are INTERPRETATION, not layout.** No `NODE_ROW_STRIDE`
   change, no `ENVELOPE_LAYOUT_VERSION` bump (canon "registry-resolved via
   `classid → ClassView`").
4. **deterministic↔residue.** Coarse = nearest-centroid palette index
   (recomputed via AMX `matmul_i8_to_i32`); residue = signed-4-bit correction.
5. **I-VSA-IDENTITIES correction.** COCA "codebook superposition for 2/3 pruning"
   must bundle cluster **identity fingerprints**, NOT the CAM-PQ codes
   (superposing PQ codes is register-loss). Pruning needs a cluster-identity
   layer that does not exist yet → design before build.
6. **Placement.** `ndarray` = hardware/math (codecs, reliability, entropy);
   `lance-graph-contract` = the selector + the edge type; `causal-edge` = the
   bit-storage; `lance-graph` = thinking (orchestration, basins, context).

## The Plateau (delivered) — R1

| Artifact | Repo / path | Commit |
|---|---|---|
| `hpc::reliability` (Pearson, Spearman, Cronbach α, ICC(2,1), `FidelityReport`) | ndarray `src/hpc/reliability.rs` | `d3b608f` |
| `hpc::edge_codec` (Codebook k-means, `CoarseResidueCodec`, `ProductQuantizer`, `reconstruct_coarse`) | ndarray `src/hpc/edge_codec.rs` | `d3b608f` |
| `examples/edge_codec_compare` (measure all flavors × regimes) | ndarray | `d3b608f` |
| `hpc::entropy_ladder` (`nars_entropy`, `EntropyRung`, `Quadrant`, `PEARL_SUBSETS`, `decompose_spo`, `entropy_class`) | ndarray `src/hpc/entropy_ladder.rs` | `83be7c3` |
| `examples/entropy_ladder_probe` (rung/quadrant partition + SPO decomposition) | ndarray | `83be7c3` |
| `EdgeCodecFlavor` + `ClassView::edge_codec_flavor` selector | lance-graph `lance-graph-contract` | `920671d` |
| bgz17 SIMD gather OOB guard (P1) | lance-graph `crates/bgz17/src/simd.rs` | `6d48ced` |

**Measured:**
- Edge-codec fidelity: CoarseResidue dominates agreement (ICC 0.97–0.99, ρ 0.98,
  α 0.99); Pq32x4 preserves rank (ρ 0.60–0.67) but not absolute distance
  (ICC 0.11–0.29); CoarseOnly collapses on continuous data (ICC 0.003). AMX
  assign 100% vs scalar, 24–28 GMAC/s.
- Entropy ladder: ρ(entropy, accuracy) = −0.78; balanced rung/quadrant partitions;
  crystalline SPO → H=0.107 (Pragmatics/Wisdom), ambiguous S-only → H=1.000
  (Syntax/Staunen).

Tests: ndarray 28 new unit + 14 doctests; contract +3 (609 lib green); bgz17 +1.
All clippy `-D warnings` clean.

## Roadmap — R2..R6 + COCA

| Rung | What | Plugs into | Dependency / gate |
|---|---|---|---|
| **R1** | NARS f/c → validity/reliability + entropy coordinate | `nars_entropy` + `reliability` | ✅ **SHIPPED** (ρ=−0.78) |
| **R2** | Store the entropy/reliability class where it belongs | `entropy_class` → `CausalEdge64` spare bits [63:61] | `causal-edge` (zero-dep bit ops), **version-gated** + field-isolation tests (I-LEGACY-API-FEATURE-GATED) |
| **R3** | CAM-PQ AMX centroid assignment (the "scanning" win) | `matmul_i8_to_i32`; 2×2/4×4 tiled centroid grid | independent of the ladder; `cam_pq.rs:precompute_distances` → AMX variant + bit-exact/GMAC-s probe |
| **R4** | HHTL + helix basin attraction | `decompose_spo.basin_key` + `helix` residue | 🔬 **probe**: residue-Δ from basin centroid predicts re-access > HHTL-tier alone (+15% recall gate) |
| **R5** | Markov SPO rung-ladder → episodic context/basins/supporting edges | `EntropyRung` over a `CausalEdge64` stream | 🔬 **probe**: ladder prunes without recall loss; builds on R1+R2 |
| **R6** | Energy axis / particle↔wave | real `MailboxSoA.energy` → `Quadrant`'s energy input | ⏳ gated on the Mailbox-SoA map (in flight) |
| **COCA** | superposition 2/3 pruning between 2⁸ SPO | cluster-identity FP layer (NEW) | 🛑 **I-VSA-IDENTITIES**: bundle identities not PQ codes — design first |

## Probe specs (the gates)

- **R4 helix-basin** — synthetic/real episodic workload; for nodes in the same
  HHTL basin, does a small Fisher-2z helix-residue Δ from the basin centroid
  predict re-access better than HHTL-tier membership alone? Promote to FINDING
  and wire only if ≥ +15% recall@k; else archive ("HHTL alone sufficient").
- **R5 Markov rung-ladder** — build word→clause→discourse centroid-SPO levels;
  measure whether checking the low-entropy (Pragmatics) rung first prunes ≥ X%
  of candidates with < Y% recall loss vs flat scan. Entropy band = the prune key.

## D-ids

- **D-EL-1** (R1) — entropy-ladder foundation + reliability + edge-codec — **Shipped**
- **D-EL-2** (R2) — entropy class in CausalEdge64 spare bits — **Queued** (next)
- **D-EL-3** (R3) — CAM-PQ AMX assignment — **Queued**
- **D-EL-4** (R4) — helix-basin attraction — **Probe queued**
- **D-EL-5** (R5) — Markov SPO rung-ladder context — **Probe queued**
- **D-EL-6** (R6) — energy axis / particle↔wave — **Blocked** (SoA map)
- **D-EL-COCA** — superposition pruning (cluster-identity layer) — **Design**

https://claude.ai/code/session_01D2WSmezQBNC3bUdHuGfGmo
