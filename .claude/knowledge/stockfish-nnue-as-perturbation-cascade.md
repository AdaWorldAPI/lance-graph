# Stockfish NNUE = the shipped, byte-exact reference for the perturbation-shader cascade

> **READ BY:** anyone touching the perturbation shader (§4/§4b of ndarray
> `guid-prefix-shape-routing.md`), the HHTL cascade, the V3 L4 palette256² tenant,
> `PrefixShapeTable`, or the E-CHESS incrementality ruling. Companion to the
> ndarray perturbation-pyramid doc and OGAR's perturbation-encoding pins.
>
> **Status: SYNTHESIS / mostly [H], one [G] anchor.** Written to the workspace's
> own anti-theater discipline (§5 "eigenvalue theater"; the `cross-domain-synthesizer`
> mechanism-vs-rhyme test). Every correspondence is graded; the rhymes are fenced
> off from the mechanisms. NOT canon until the epiphany-council + a probe run.

## Why this exists (the operator's framing, 2026-07-11)

> *"The idea why lance-graph, ndarray and OGAR is to learn from Stockfish how to
> use the Morton Cascade inverse-pyramid perturbation-shader cascade — not as a
> hand-rolled niche but as reusable — and to benefit from SoA and wire Stockfish
> into V3, even if the database is fixed 90 MB; we hopefully learn how to use the
> compute 64×64 = 4096 HHTL."*

The workspace's perturbation-shader / Morton-cascade / HHTL surface is, by its
own admission, **CONJECTURE as code** — "`PrefixShapeTable` … not yet coded"
(ndarray §3), the pyramid "CONJECTURE as code" (§4), the Walsh-Hadamard bipolar
pyramid "CONJECTURE" (§4b). We have the *algebra* (`vsa_bind`/`vsa_bundle`,
palette LUTs, HHTL escalate) and the *theory* (deterministic phase, magnitude-
only storage) but **no shipped end-to-end instance that proves the decomposition
carries a real, high-value workload byte-exactly.**

**Stockfish NNUE is exactly that instance** — and stockfish-rs now transcodes it
byte-exact (E-CHESS-TRANSCODE-COMPLETE-1). So it is not a chess niche; it is the
**oracle for the reusable primitive**: a 30-year-hardened, production, byte-exact
implementation of *base state + deterministic-address-indexed perturbations over
a hierarchically-addressed grid, stored as an SoA, updated incrementally.* That
is the perturbation-shader cascade, shipped.

## The correspondence — NNUE mechanism ↔ workspace primitive (graded)

| NNUE mechanism (proven byte-exact in stockfish-rs) | Workspace primitive | Grade | Why it holds / where it strains |
|---|---|---|---|
| **Accumulator** = FT biases + Σ active-feature weight columns, per perspective, as `[[i16;1024];2] + [[i32;8];2]` | **SoA magnitude envelope** — "Lance column ≡ Arrow buffer ≡ ndarray SoA, same bytes" (§4) | **[G]** | The accumulator *is* an SoA; it is the "magnitude M — the only stored bits" of §4's decomposition. Direct, not analogy. |
| **make_index** (HalfKA: `KingBuckets[ksq]·704 + PieceSquareIndex + (s^orient^flip)`) | **HHTL cascade addressing** — coarse tier (king bucket) → fine tier (piece-square); "escalate one HHTL tier" | **[H]** | Two-level hierarchical address (bucket × square) over a 64-indexed grid = a 2-tier cascade. It is *a* hierarchy, but NOT (yet) a Morton 2bit×2bit 4×4 tile — the tiers are king-bucket/piece, not spatial quadrants. Strong structural rhyme, needs the Morton re-projection probe to become mechanism. |
| **Incremental update** (E-CHESS/L4): a move = bounded add/remove of feature columns; king move → full refresh | **The perturbation shader**: `perturb(addr,L) = M[addr@coarse] · P(phase(addr,L))`; king move = **tier escalation** (`RouteAction::Escalate`) | **[G] for the delta, [H] for the escalation identity** | The delta *is* a perturbation on a base; the king-move-refresh *is* a coarse-address change forcing recompute = exactly §5's "quorum fail → escalate one HHTL tier". The i16-wrapping group identity (`refresh(after)=refresh(before)−rm+add`) is the proven kernel (L4 self-oracle, chained). |
| **phase = make_index (which feature index is active) is DERIVED from the position/move, never stored; only the weight column is stored** | **§4 decomposition**: "phase = deterministic recurrence from the address — 0 bits stored; magnitude = the only stored bits" | **[G]** | This is the sharpest correspondence. NNUE stores weights (magnitude); *which* weights fire is computed from the board (the address). Storage scales with the weight table, not the game tree — exactly "cost scales with magnitude smoothness, not perturbation bandwidth." |
| **Bundling** = Σ feature columns, `wrapping_add` (i16) | **`vsa_bundle`** (sum + threshold), the magnitude-side algebra | **[H]** | Same *shape* (order-independent accumulation; I-SUBSTRATE-MARKOV / Chapman-Kolmogorov holds — proven order-independent in L4). But NNUE bundles **i16 magnitudes**, not ±1 bipolar signs — it is the **magnitude side only** of the two-algebra rule. There is **no sign/phase XOR side** in the accumulator. |
| **transform** (pairwise `clamp(·,0,255)²/512 → u8`) + **PSQT** | **Palette / magnitude quantizer** (`RollingFloor`); the L4 `6×palette256²` tenant | **[S]→[H]** | The transform quantizes the accumulator to u8 for the affine — analogous to palette quantization, but it is a fixed nonlinearity, not a 256-centroid codebook lookup. Rhyme unless re-expressed as a palette LUT (a probe). |
| **int8 affine layers** (`fc_0/1/2`) via `matmul_i8_to_i32` | **ndarray `simd_runtime::matmul_i8_to_i32`** (the shared compute) | **[G]** | Literally the same function. Already wired (stockfish-rs L5). |
| **bucket = (pieces−1)/4 → LayerStack[bucket]** | **classid-prefix dispatch → ClassView** | **[H]** | A prefix (piece count) selecting a compute lens = a classid selecting a ClassView. Structural; the "8 buckets" ↔ "8 render lenses" is a rhyme until a ClassView actually dispatches the stack. |
| **64 × 64 = 4096** (king-square × piece-square-index; and the butterfly from-to lane) | **the 64×64 = 4096 gridlake / 1BRC lane J sweet spot** | **[G] on the number, [H] on the identity** | The 4096 is real and load-bearing in both. Whether NNUE's 64×64 is *the same* 4096 as the gridlake (vs. a coincident cardinality) is the open question the wiring answers. |

## Mechanism vs rhyme — the honest ledger (do not skip)

**Genuine mechanism (transfer these):**
- **Deterministic-address phase + stored-magnitude** ([G]). NNUE *proves* you can
  store only the envelope and regenerate "which cells fire" from the address. This
  de-risks PROBE-PERT-RHO: a shipped workload already lives at "magnitude-only".
- **Incremental perturbation with coarse-tier escalation** ([G]/[H]). NNUE proves
  the base+delta update is byte-exact and that a coarse-address change (king move)
  cleanly forces a tier refresh — the exact shape of `RouteAction::Escalate`.
- **SoA + int8 GEMM** ([G]). Already shared code.

**Rhyme (do NOT ship as if proven — fence them):**
- **Walsh-Hadamard / bipolar sign pyramid (§4b) has NO analog in the NNUE
  accumulator.** NNUE is magnitude-only bundling; there is no ±1 sign/phase side,
  no XOR algebra, no superposition-unbind. Claiming NNUE "is" the Walsh-Hadamard
  cascade would be eigenvalue theater. NNUE informs the **magnitude side**; §4b's
  sign side is a *separate* conjecture NNUE does not witness.
- **Morton 2bit×2bit 4×4 tile** is not NNUE's addressing — NNUE's hierarchy is
  king-bucket × piece, not spatial-quadrant Morton. The "cascade" is real; the
  "Morton" is aspirational until the re-projection probe (below).
- **palette256² L4 tenant** is not how NNUE stores weights (i16, not palette
  pairs). The similarity-as-one-table-read is a rhyme unless a probe shows NNUE's
  distance structure survives palette quantization at the ρ anchors.

## The reusable primitive (the thing that must not stay chess-niche)

**A `PerturbationAccumulator` — an SoA over a HHTL-addressed feature grid, updated
by bounded, deterministic-address-indexed perturbations, with coarse-tier
escalation.** Domain-agnostic. Chess is one consumer; the compute is the point.

- **ndarray = MECHANISM** (per `data-flow.md` + §2 "ndarray is mechanism, never
  policy"): the SoA accumulator kernel — `refresh` / `apply_delta` (i-wrapping
  group op) / `escalate` — parameterized by *(feature-count, lane-width, an
  address→active-index closure, a stored magnitude table)*. This is stockfish-rs's
  `Accumulator` + `HalfKaAccumulator` **generalized to drop the chess types**: the
  `make_index` closure and the weight table become inputs. That is the whole
  reusability move — lift L3/L4 from `Chess`-typed to closure-parameterized, land
  the kernel in ndarray, and stockfish-rs becomes a *consumer* that supplies the
  chess closure + the 90 MB table.
- **lance-graph / V3 = POLICY**: the accumulator's SoA *is* a V3 tenant lane; the
  address is a `NodeGuid` prefix; the perturbation is the shader over that lane;
  the escalation is `RouteAction::Escalate` on the HHTL cache.

## Wiring Stockfish into V3 (the plan — probe-gated, not asserted)

Stated as deliverables so it can be executed and falsified, never hand-waved:

- **D-SF-V3-1 — lift the accumulator to a closure-parameterized ndarray kernel.**
  Generalize `Accumulator::{refresh, apply-delta}` to
  `PerturbationAccumulator<const LANES, const BUCKETS>` taking an
  `active_indices: impl Fn(&Addr) -> SmallVec<Feature>` + a `&MagnitudeTable`.
  stockfish-rs re-expresses its L3/L4 as the chess instantiation. **Gate:** the
  existing L3/L4 byte-exact oracles STILL pass through the generic kernel (no
  regression — the reference stays green).
- **D-SF-V3-2 — the NNUE accumulator AS a V3 SoA tenant.** Map `[[i16;1024];2]`
  onto a V3 tenant lane (le-contract): is it an L4 palette pair carrier, or a new
  raw-i16 lane? **Probe D-PALETTE-NNUE:** does palette256²-quantizing the FT
  weight columns preserve the eval within the ρ anchor (Pflug-10 / Jirak floor)?
  If YES → NNUE weights are a genuine L4 tenant (huge: 90 MB → palette-compressed
  + one-table-read similarity). If NO → NNUE needs a raw-magnitude lane and the
  palette-tenant rhyme is fenced. *This is the single highest-value probe — it
  tells us whether the frozen 90 MB net is a palette tenant or not.*
- **D-SF-V3-3 — make_index → HHTL/Morton route.** Probe D-MORTON-KA: re-project
  HalfKA's king-bucket × piece-square addressing onto a Morton 2bit×2bit tile and
  measure whether nearest-in-Morton ⇒ nearest-in-feature (the quorum τ). Confirms
  or kills the "NNUE 4096 = gridlake 4096" identity ([H] → [G] or dropped).
- **D-SF-V3-4 — the escalation identity.** Assert (test) that NNUE's
  king-move-refresh == `RouteAction::Escalate` at a named tier — the first place
  the chess reference and the HHTL cache share a code path.

## What the frozen 90 MB net buys us (the operator's point, made precise)

The net is fixed — that is a **feature for this purpose**: a frozen, byte-exact,
externally-validated magnitude table lets us test the *compute* (addressing,
perturbation, escalation, palette-quantization, one-table-read similarity)
against a ground truth that cannot drift. We are not learning chess; we are using
chess as the **only workload in the building where "did the perturbation cascade
reproduce the answer byte-for-byte?" has a yes/no oracle.** Every probe above is
gradeable precisely because the net is frozen.

## Fences (no theater)

1. This doc is SYNTHESIS. The [G] anchors are the accumulator-as-SoA, the
   deterministic-phase/stored-magnitude decomposition, and the shared int8 GEMM.
   Everything else is [H]/[S] and **named as such** — do not cite the rhymes as
   proven.
2. NNUE witnesses the **magnitude side** of the two-algebra rule only. It says
   nothing about the bipolar Walsh-Hadamard sign side (§4b) — keep them separate.
3. No deliverable here ships without its probe green (the workspace's own §5
   quorum-or-escalate discipline turned on our own claims).

## Cross-refs

- ndarray `guid-prefix-shape-routing.md` §4 (perturbation pyramid), §4b
  (Walsh-Hadamard), §5 (φ-quorum / anti-theater).
- V3 `soa_layout/le-contract.md` (L1–L4 tenant ladder; L4 = 6×palette256²).
- OGAR `CLAUDE.md` — "Perturbation encoding — DETERMINISTIC PHASE",
  "Bipolar-phase pyramid", "256×256 CENTROID TILE", "256 = 4⁴ hierarchy".
- Board: `E-CHESS-TRANSCODE-COMPLETE-1`, `E-CHESS` #539.
- `AdaWorldAPI/stockfish-rs`: L3 `src/eval/accumulator.rs`, L4
  `src/eval/incremental.rs` (the reference kernel to lift), L5 `network_eval.rs`
  (the int8 GEMM consumer).
