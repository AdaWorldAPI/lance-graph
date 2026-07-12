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
- **D-SF-V3-2 — the NNUE accumulator AS a V3 SoA tenant. ✅ GREEN (2026-07-12,
  MEASURED with the CERTIFIED codec — supersedes the 2026-07-11 scalar-256 fence).**
  The question is whether the FT weights are a palette256 tenant — i.e. whether
  the workspace's **certified Fisher-z cosine-replacement** (`bgz-tensor::fisher_z::
  FisherZTable`, per-family 3σ gamma; certified ρ≥0.999 on 21 Qwen3-TTS roles /
  256 Jina-v5 centroids) preserves the **pairwise-cosine ranking** of the FT
  columns, read off the i8 table with **no materialization**. Measured on 256 FT
  columns sampled from `nn-1b6a82263149` (32 640 off-diagonal pairs — the Jina-v5
  cert setup):
  - ρ_all = **0.99971** (Pearson r 0.99993); cosine MAE 0.00655, 64 KB one-table-read.
  - ρ_mid = **0.99873** on the hard **near-orthogonal** band (|cos| ≤ 0.3, 19 126
    pairs — where Fisher-z stretch is smallest and discrimination is hardest).
  **Verdict: the FT columns ARE a palette256 tenant** — the certified cosine-
  replacement preserves one-table-read similarity ranking, clearing the ρ≥0.999
  anchor even on the hard cut. Fisher-z (`arctanh`) is monotone → rank-preserving
  by construction; only the i8 3σ quantization is lossy, and it is negligible here.
  Probe: `bgz-tensor examples/nnue_palette_cosine.rs` (+ `stockfish-rs
  examples/export_ft_columns.rs` for the FT-column fixture).

  **D-PALETTE-NNUE-VEC — the tenant-SHAPE ladder. ✅ GREEN at the Base17 anchor
  (2026-07-12, MEASURED, held-out).** Escalation from the distance codec (above) to
  the actual VECTOR tenant shapes: codebooks trained on 2048 FT columns, measured
  on 256 DISJOINT held-out columns (32 640 pairs — no memorization degeneracy; an
  earlier same-set run gave a degenerate ρ=1.0 and was discarded):

  | lane | bytes/col | ρ_all | near-orth ρ | MAE |
  |---|---|---|---|---|
  | cam_pq 48-bit (6×256, `CamCodebook`) | 6 B (341×) | 0.929 | 0.823 | 0.083 |
  | **V3-L4 96-bit (6×256², the le-contract `6×(u8:u8)` ²centroid pairs)** | 12 B (170×) | **0.966** | 0.881 | 0.053 |
  | 64×256 512-bit (64 subspaces × 16d — "one centroid per square", board-shaped) | 64 B (32×) | 0.977 | 0.918 | 0.042 |
  | + turbovec-style n×4bit edge residue (uniform FLOOR) | ~526 B (3.9×) | **0.998** | **0.994** | 0.009 |

  **Operator ruling confirmed by measurement: the full 96-bit `6×(256×256)` V3
  tenant is strictly better than 48-bit cam_pq** (it is the faithful/"perfect"
  tenant shape; cam_pq is the lossy approximation). The 96-bit tenant clears the
  **Base17 vector-compression anchor (ρ=0.965)** — the correct reference class for
  vector compression; the 0.999 Fisher-z lane is a DISTANCE codec (in-sample
  quantization of the pairwise answer, a strictly easier operation) and was the
  wrong pre-registered gate (corrected in-file). The near-orthogonal loss (0.881)
  concentrates where quantization hurts — and the **turbovec n×4bit edge residue
  recovers it to the 0.999 class even at the uniform floor** (real Lloyd-Max ≥
  uniform; the `lance-graph-turbovec` crate needs its sibling checkout to wire the
  real lane). Probe: `stockfish-rs examples/palette_nnue_vec.rs`.
  *Design-space alternatives (operator-named):* **64×256 "one centroid per
  piece-square"** — measured above (board-shaped: 64 squares ↔ 64 subspaces; beats
  the 96-bit tenant at 32×, but near-orth 0.918 still wants the residue lane for the
  0.99 class); **classid × wide-FieldMask in-memory filtering** — filter what is
  already resident by classid prefix + mask tenant lanes per focus with the contract
  `FieldMask` (ClassView-projection doctrine applied to resident rows: ZERO
  serialization, per-focus reasoning masks) — design note, queued; **holograph 7×7×7
  signed bits** — information-theoretically roomy but consumer-UNPROVEN, stays [H]
  until a consumer probe; **Pythagorean-comma Morton-cascade inverse-pyramid,
  NON-collapsing** — the D-MTS-5 comma result (N_eff 11/12 independent witnesses,
  replay bit-identical, pyramid never materialized) is the anti-mush mechanism for
  the perturbation-shader cascade, scaling upstream 64×64=4096 → 256k×256k
  replayability without materialization; operator-recalled precedent: the comma
  stride carried a gaussian-splat 3DGS top-k over 10000×10000 at ~440 ms
  unoptimized — [S] here (the `perturbation-sim` `splat.rs`/`morton2`/`ewa_coarsen`
  substrate is shipped, but that figure is not reproduced in this repo's source) —
  orthogonal to (and composable with) the tenant compression above.

  > **Correction (why the first cut was wrong).** The 2026-07-11 scalar-256 result
  > (Lloyd/k-means codebook over the raw i16 weights, then reconstruct + re-run
  > eval → ρ_quiet 0.7812, "FENCED") measured the WRONG thing with the WRONG tool:
  > (a) a hand-rolled scalar k-means codebook is NOT the workspace's palette256
  > (which is the Fisher-z cosine-replacement — `encoding-ecosystem.md` is the
  > MANDATORY map that was skipped); (b) it MATERIALIZED (reconstructed weights and
  > re-evaluated) when for RANKING you never materialize — you read off the Fisher-z
  > distance directly. Both errors inflated a real similarity codec into an eval-
  > reconstruction failure. The scalar-256 numbers stand only as "naive scalar
  > k-means reconstructs eval poorly" — a true statement about a tool nobody uses,
  > NOT about the palette tenant. Honest scope of the GREEN: proven for FT-column
  > pairwise-cosine SIMILARITY (the one-table-read tenant claim); it does NOT claim
  > byte-exact eval from palette codes (that remains the raw net's job — the palette
  > is a similarity/routing tenant, not an exact-eval substitute).
- **D-SF-V3-3 — make_index → HHTL/Morton route. ✅ GREEN (2026-07-12, MEASURED).**
  Probe D-MORTON-KA re-projected HalfKA's piece-square axis onto the CERTIFIED
  workspace Morton primitive (`lance_graph_contract::facet::FacetTier::morton`,
  2bit×2bit Z-order — reused, not hand-rolled) and measured nearest-in-Morton vs
  nearest-in-feature over 24 (king × piece) configs of the FT columns on
  `nn-1b6a82263149`:
  - feature locality IS spatial: **board recall@8 = 0.440 (3.5× chance)**, ρ 0.407;
  - Morton preserves a majority of it: **Morton recall@8 = 0.347 (2.7× chance)**,
    ρ 0.328 (recall@4: Morton 0.217 = 3.4×, board 0.258 = 4.1×).
  **Verdict: the NNUE HalfKA square addressing IS a Morton-preserved gridlake** —
  nearest-in-Morton ⇒ nearest-in-feature well above chance. "NNUE 4096 = gridlake
  4096" [H]→[G]. Honest scope: Morton ≤ board is expected and observed (board is
  exact 2-D adjacency; Morton is a 1-D locality map preserving ~79% of board's
  signal); the modest absolute ρ (~0.33) says the addressing is *locality-
  preserving*, not a perfect isometry. Probe: `stockfish-rs examples/morton_ka.rs`.
- **D-SF-V3-4 — the escalation identity. ✅ GREEN (structure) + FENCE (2026-07-12,
  MEASURED).** The structural half is exact: over 75 080 (move × node) pairs of
  deterministic playouts, a king move ⟺ coarsest-tier (king) change ⟺ EVERY
  feature of the mover's perspective re-indexed ⟺ full refresh — **4851/4851**
  king moves re-index the sentinel feature, **0/70 229** non-king moves do, 0
  cross-tab mismatches. A change at the coarsest HalfKA address tier forces full
  recompute — the escalation STRUCTURE the identity rests on, proven.
  **FENCE (the literal name-match is a rhyme):** the shipped bgz-tensor
  `RouteAction::Escalate` (`hhtl_cache.rs`) fires on *distance-percentile
  ambiguity* (a HIP-band pair drops to the finer TWIG tier), NOT on a
  coarsest-tier-change event — so "king-refresh == `RouteAction::Escalate`" is
  NOT claimed; the two escalations share structure (drop to finer/full work),
  not a code path. Probe: `stockfish-rs examples/escalation_identity.rs` (no net).

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

## The temporal/episodic layer — a chess game IS a Markov version-stream

The operator's extension: *"we could wire the AriGraph episodic and SurrealQL
AST time series and/or episodic basins = opening variants."* The chess workload
witnesses the **temporal** substrate as cleanly as it witnesses the spatial one —
and the two are the SAME operation seen from two axes.

**The key unification.** The NNUE incremental update
(`incremental.rs::apply_move`) is BOTH:
- a **spatial** perturbation — the L4 delta over the 64×64 board grid
  (add/remove piece features, king-refresh on escalation); AND
- one **temporal** Markov step — ply *n* → ply *n+1* along the game stream.

There is no separate "temporal engine." A game is the accumulator's own trajectory
through version-space, and each `apply_move` is exactly one `temporal.rs` stream
entry. The spatial cascade and the temporal stream are one accumulator, read on
two axes.

| Chess object | Temporal substrate | Grade |
|---|---|---|
| A game (ply sequence) | a `temporal.rs` sorted version-stream; ply *v* = one Lance version | **[G]** — moves already carry a total order; `apply_move` IS the step |
| "position after ply *v*" | `QueryReference::at(v, rung)` + deinterlace — a zero-copy projection, no replay | **[G]** (2026-07-11) — D-SF-EPISODIC-1 ran GREEN: the accumulator projected at ply *v* is byte-identical to the fresh-from-FEN computation (34/34), and to an out-of-order replay (11/11); see the deliverable below |
| Analysis horizon (present vs hindsight) | the **rung**: low rung = reason strictly at ply *v*; high rung = spoiler-read the game's outcome to label the move (training-target labeling) | **[H]** — D-SF-RUNG-1 ran INCONCLUSIVE (2026-07-12): the NNUE-eval-only oracle is not mate-aware (inverts on sacrificial mates) and random playouts never settle; needs a real outcome oracle. Stays [H] |
| Opening variants (e4/d4/…) | **episodic basins** = le-contract **L1 `part_of:is_a`** rails (a variant is-a line is-a opening) | **[G]** (2026-07-12) — D-SF-BASIN-1 GREEN: opening trees fit the L1 rail shape; the `part_of:is_a` `hi_distance` clusters lines by opening (5 same-opening < 6 cross) |
| Transpositions (same position, different move-order) | **L2 `memberof:members`** — one position node, member-of many variant basins | **[G]** (2026-07-12) — D-SF-BASIN-1 GREEN: a 1.d4/1.c4 transposition to the same QGD position collapses to ONE `NodeGuid` carried in a probe-local facet-L2-rail multimap over both basins. NB: this is the **facet L2 rails** (up to 6 basin pairs, not yet persisted), NOT the shipped graph `memberof` in `mailbox_scan.rs`, which is HHTL-tier **many-to-one** (`Option<BasinOf>`, a single parent) |
| Episodic recall ("games that reached this pawn structure") | AriGraph `EpisodicMemory` / `markov_soa` — retrieve-similar over position fingerprints | **[G]** (2026-07-12) for the REPRESENTATION — D-SF-ARIGRAPH-1 GREEN: positions cluster by opening at 3.7× chance under SimHash-Hamming (96% of raw cosine) in the store's own `[u64;8]`/Hamming shape. The STORE has a named GAP: string-only FNV `label_fp`, no fingerprint-in `retrieve_similar` |
| Time-series query over games | SurrealQL AST — a **query adapter** over the version-stream | **[G]** — adapter only; see the fence |

### Fence — SurrealQL AST is the query adapter, NOT the episodic spine

The episodic logic (basins, transpositions, rung-selection, the Markov step) lives
in the accumulator trajectory + the le-contract rails + AriGraph. **SurrealQL AST
is only the read/time-series *adapter* over that stream** — per the OGAR
SURREAL-AST-TRAP: behavior never lives in DDL, and a `DEFINE EVENT … WHEN … THEN`
carrying game-lifecycle logic is the negative-beauty hijack the doctrine rejects.
Games flow: `apply_move` → temporal-stream version → le-contract basin →
(optionally) a SurrealQL SELECT projects it. Never: SurrealQL DDL *drives* the game.

### Probe-gated deliverables (temporal)

- **D-SF-EPISODIC-1 — a game as a temporal version-stream. ✅ GREEN (2026-07-11,
  MEASURED).** Modelled the Opera Game (33 plies, captures + queenside castling)
  as a `temporal.rs`-shaped version-stream over `HalfKaAccumulator` (the
  incrementally-updated carrier — the temporal analog of the spatial L4 delta),
  and measured two gates on the pinned net `nn-1b6a82263149`:
  - **GATE A (projection):** fresh `refresh(pos@v)` == the incrementally-maintained
    version-*v* accumulator, byte-for-byte — **34/34** plies. `QueryReference::at(v)`
    reproduces from the version's own FEN identity, no recomputation of the stream.
  - **GATE B (order-independent replay):** an out-of-order / random-access read to
    ply *v* (independent incremental walk from the start) == the forward-walk
    version-*v* — **11/11** queries. The stream is replayable.
  Together these promote "position-at-version is a zero-copy projection" **[H]→[G]**.
  Reused the L4 byte-exact oracle as a temporal-replay oracle at zero new
  ground-truth cost, exactly as designed. Honest scope: proven for the HalfKA
  incremental carrier; the full eval's threats term is refresh-only in current
  code (does not affect the version-stream claim). Probe:
  `stockfish-rs examples/temporal_replay.rs` (PR #5); net-gated, CI-safe.
- **D-SF-BASIN-1 — opening variants as L1 basins. ✅ GREEN (2026-07-12, MEASURED).**
  Encoded a small opening forest on the SHIPPED contract facet types
  (`lance_graph_contract::{facet::FacetCascade, canonical_node::NodeGuid}` — reused,
  not reimplemented) over real chess rules (shakmaty). Three gates, all PASS:
  - **G1 transposition:** the QGD position reached via 1.d4 (d4 d5 c4 e6 Nc3 Nf6)
    and via 1.c4 (c4 e6 Nc3 d5 d4 Nf6) has a move-order-independent identity
    (board+turn+castling+ep) that collapses to ONE `NodeGuid`; a non-transposing
    English line maps to a distinct node.
  - **G2 many-basins-one-node:** that single node belongs to BOTH basins
    {Queen's-Pawn, English} in the probe-local facet-L2-rail multimap — the round-trip.
  - **G3 L1 taxonomy:** the `part_of:is_a` rails cluster lines by opening —
    `FacetCascade::hi_distance` = 5 (same-opening) < 6 (cross-opening).
  **Verdict: opening variants ARE le-contract episodic basins; a transposition is
  the many-basins-one-node round-trip.** Honest scope — two distinct `memberof`
  notions, do not conflate: (1) the **facet L2 `memberof:members` rails** carry up
  to 6 basins per node and are what this probe exercises, but the facet lane is not
  yet a persisted `ValueTenant` (Phase-2), so the probe holds the multi-basin set in
  an in-memory `HashMap<NodeGuid, HashSet<basin>>` alongside the shipped facet/NodeGuid
  types; (2) the **shipped graph `memberof`** (`crates/lance-graph/src/graph/mailbox_scan.rs`)
  is HHTL-tier **many-to-one** — `Option<BasinOf>`, a single parent basin (Local/Route/
  Top). The many-basins-one-node relation therefore rides the facet L2 rails, NOT the
  graph HHTL `memberof`; the *encoding* is proven, the *persistence* is future work.
  Probe: `stockfish-rs examples/basin_transposition.rs`.
- **D-SF-RUNG-1 — hindsight labeling via rung. ⚠ INCONCLUSIVE — stays [H]
  (2026-07-12, MEASURED; the ORACLE failed, not the thesis).** With an
  NNUE-eval-only hindsight oracle (`sign(eval@final)`) the rung split does NOT
  land. THREE identified oracle defects — the third found by codex on PR #8 and
  fixed: (0) **`evaluate()` is side-to-move POV**, so the first cut's cross-ply
  sign comparison was move-parity noise (flat 0.53 ≈ 0.54); with fixed-White-POV
  normalization a REAL convergence gradient appears — mean first-third 0.48 →
  last-third 0.74, 13/25 games — but still below the pre-registered 0.75/0.9
  gate; (1) **NNUE is not mate-aware** — a game ending in a
  SACRIFICIAL mate (the Opera Game: Qxb8+ Nxb8 Rd8#) reads as the mating side
  DOWN material at the final ply, inverting the hindsight label exactly on the
  most decisive games; (2) random playouts never settle to a legible verdict.
  Recorded as a measured negative on the ORACLE; the probe needs a REAL outcome
  oracle (played-out game-result labels or a mate-aware search) before the rung
  thesis can be graded. Probe: `stockfish-rs examples/rung_hindsight.rs`.
- **D-SF-ARIGRAPH-1 — positions as episodic keys. ✅ GREEN (representation) + GAP
  named (2026-07-12, MEASURED).** The gap first (grindwork finding, honest): the
  shipped `EpisodicMemory` has **no** `retrieve_similar` — its only fingerprint
  path is `label_fp(&str)`, an FNV avalanche hash of STRINGS with **no locality**,
  so position-similarity recall is structurally impossible with the store as-is.
  The probe supplies the missing primitive — a SimHash of the NNUE accumulator
  into the store's own `[u64;8]`/Hamming shape — and measures the representation:
  over 96 positions from 4 structurally-distinct openings (Ruy/QGD/KID/English),
  **precision@5 = 0.894 under SimHash-Hamming vs chance 0.242 (3.7× chance),
  retaining 96% of the raw-cosine clustering (0.935)**. Positions ARE good
  episodic keys, in exactly the store's own metric. **GAP (queued as the store's
  next leaf):** a locality-preserving vector→`Fingerprint` constructor + a
  fingerprint-keyed top-k — this probe is that primitive's reference impl.
  Probe: `stockfish-rs examples/episodic_recall.rs`.

### Why this closes the loop

Spatial (the board), temporal (the game), and episodic (the archive of games) are
**three reads of one accumulator trajectory** — the same `apply_move` delta,
projected on the square axis (perturbation cascade), the version axis (Markov
stream), and the similarity axis (episodic recall). The frozen 90 MB net keeps all
three honest: every temporal projection has the same byte-exact yes/no oracle the
spatial cascade has, because replay-to-ply-*v* and compute-from-ply-*v*-FEN must
agree to the bit. That is the whole reason chess is the learning vehicle — it is
the only workload in the building where *all three axes* answer to one frozen
ground truth.

## Cross-refs

- ndarray `guid-prefix-shape-routing.md` §4 (perturbation pyramid), §4b
  (Walsh-Hadamard), §5 (φ-quorum / anti-theater).
- V3 `soa_layout/le-contract.md` (L1–L4 tenant ladder; L1 `part_of:is_a` +
  L2 `memberof:members` = episodic basins/transpositions; L4 = 6×palette256²).
- `temporal.rs` — `QueryReference::at(v, rung)` + deinterlace (the game
  version-stream); `E-MARKOV-TEMPORAL-STREAM-1` (Markov moves off the VSA braid
  onto the sorted stream — the ruling the temporal layer here rides on).
- AriGraph — `EpisodicMemory` / `markov_soa` / `triplet_graph` (the episodic
  store; positions as episodic keys).
- OGAR `SURREAL-AST-TRAP-PREFLIGHT.md` — SurrealQL AST is a query adapter, not
  the episodic spine (behavior never in DDL).
- OGAR `CLAUDE.md` — "Perturbation encoding — DETERMINISTIC PHASE",
  "Bipolar-phase pyramid", "256×256 CENTROID TILE", "256 = 4⁴ hierarchy".
- Board: `E-CHESS-TRANSCODE-COMPLETE-1`, `E-CHESS` #539.
- `AdaWorldAPI/stockfish-rs`: L3 `src/eval/accumulator.rs`, L4
  `src/eval/incremental.rs` (the reference kernel to lift), L5 `network_eval.rs`
  (the int8 GEMM consumer).

## Stacked awareness / opponent-modeling arc (2026-07-12, operator design inputs + 5 probes)

**Operator design inputs (recorded verbatim in intent, graded honestly):** rung levels
as STACKED AWARENESS (cognitive Maslow pyramid; Piaget's stages; the Three-Mountains
perspective test); Go's Raumgewinn vs infight as CONCURRENTLY COMPLEMENTARY strategies;
sacrifices in the infight buying freed piece influence + development-time momentum;
pros waiting solid, storing momentum, then driving the wedge unexpectedly; L2 opponent
modeling as a DIVERSION VECTOR from one's own L1 (the "(w)edge residue") rather than a
coarse hand-picked feature basis; prehydrated opening/middlegame/endgame strategy
memory behind a classid × wide-FieldMask attention filter (zero-serialization in-memory
masking for per-focus reasoning); positions/strategies as AriGraph V3 substrate (V3
4+12 facet nodes, Lance-version time series per D-SF-EPISODIC-1, chess-specific
markers); and rung decomposition applied to chess causality trajectories. The analogies
are framing; every mechanism claim below carries a measured gate. All probes:
stockfish-rs examples, net-gated, deterministic, exit-0 measurement style.

**D-SF-OPPONENT-1 — the awareness ladder (PARTIAL, commit `7a8381e`).** Observer
predicts styled opponents' moves at three rungs. n=511 decision points, 4 synthetic
styles (Aggro/Solid/Greedy/Drifty):

| rung | model | top-1 | top-3 |
|---|---|---|---|
| L0 egocentric (predict what's best FOR the observer) | raw `evaluate(after)` un-negated | 0.006 | 0.035 |
| L1 perspective (Three Mountains passed) | mover-POV `-evaluate(after)` | 0.616 | 0.937 |
| L2 naive tendency (hand-picked capture/check/greed basis) | L1 + feature bias | 0.591 | 0.861 |

The L0→L1 jump is ~100× — perspective-taking is THE awareness step, and **L0 is
structurally identical to the codex POV defect fixed on stockfish-rs PR #8**: the
egocentric read of the opponent's position is the same failure mode, now measured at
0.006 predictive power. L2-naive FAILED its gates (worse than L1 on 3/4 styles; style
identification 0.375 vs chance 0.217, short of the 2× bar). Root cause measured, not
assumed: in a random-playout corpus, base-eval swings span thousands of cp and dwarf
the 90–140cp injected style biases — the behavioral signal is thin by construction.

**D-SF-OPPONENT-3 — diversion-vector L2 (NEGATIVE on synthetic, stays [H]; commit
`c263ac0`).** Operator correction under test: opponent model = own L1 + measured
residue of their choices against it, in accumulator space (mover-perspective HalfKA
slice, unit-normalized deltas, online, no peeking), instead of the hand-picked basis.
Same corpus/seeds as OPPONENT-1 (census + L0/L1 reproduced exactly). Result: L2-residue
0.589/0.534/0.491 at λ=1/3/6 (vs naive 0.591, L1 0.616) — monotonically worse with λ;
residue-direction identification 1.5× chance vs the feature basis's 1.7×. **Honest
localization, not a kill:** on a corpus whose deviations ARE three scalar biases, the
hand-picked basis is the generating basis and cannot lose, while the raw 1024-d
deviation direction is noise-dominated (style-centroid cosines: Solid–Aggro +0.02). The
diversion-vector claim is about REAL opponents with structured deviations — decisive
test = D-SF-LICHESS-1 (below). Mechanical note banked: raw accumulator deltas carry
material/placement mass; whiten/normalize before direction on real data.

**D-SF-OPPONENT-2 — Raumgewinn/infight, sacrifice economics, wedge, counterfactual
rung (PARTIAL 5/7, commit `6f7f7bc`).**

| clause | gate | result |
|---|---|---|
| 1a style separation (Territory/Fighter LOO classification ≥0.75) | ✗ 0.25 | synthetic-style SNR again |
| 1b axes complementary (cross-axis pearson &lt;0.5) | ✓ r=0.069 | orthogonal, not antipodal |
| 2a ≥1 sacrifice detected (Opera) | ✓ 5 detected | — |
| 2b all detected show influence/tempo compensation | ✗ 1/5 | the 1 passer is the REAL Qxb8+!! (ply 31); the 4 fails are false positives of the shallow 1-ply greedy-reply material oracle (cannot see recaptures) |
| 3 NNUE residual (eval − material) loads on space/influence/development (≥2/3 Spearman ρ ≥ +0.3) | ✓ 2/3 | **the net PRICES the compensation** — the "positional perturbation" layer has a measured interpretation |
| 4 wedge detection (quiet ≥6 plies → capture burst; stored momentum + surprise concentration) | ✓ 8/9 games wedge; Opera wedge = **Nxb5 ply 19, the pre-registered prediction, hit exactly**; 6/8 satisfy both sub-gates | waiting-then-conversion is detectable |
| 5 counterfactual rung (Pearl Rung 3, model-based rational-L1 replay of the material-preserving alternative, 8 plies) | ✓ 5/5 | every sacrifice's actual line beats its counterfactual on influence or development — "bought momentum" is a measured trajectory divergence |

**D-SF-PHASE-1 — classid × wide-mask phase filter (O3: MASK HURTS at retrieval;
commit `ab7d9f4`).** 271 positions, 4 families × 3 phases (80/120/71). Unmasked
family-precision@5 0.697; phase-masked 0.654 — the filter removed cross-phase
same-family neighbours that were correct. Cross-phase contamination 21.2% (Middle
36.8%, Opening 1.7%); phase is readable off the unmasked top-5 neighbourhood at 0.815
(Opening 0.988, End 0.887) — **phase is emergent in accumulator geometry; prehydrated
basins should be family-keyed with phase as a soft read, not a hard partition.** The
classid × wide-mask filter earns its keep as SCOPING (zero-serialization ~3× candidate
reduction at −0.04 precision), not as an accuracy device. Contract ground truth: the
real `lance_graph_contract::class_view::FieldMask` is a field-PRESENCE mask (has(n)) —
a different semantic axis than a classid phase-equality filter; the probe rendered the
pattern locally and says so rather than force-fitting.

**Meta-finding of the arc (first-class):** every mechanism clause anchored on REAL play
(the Opera Game) went green — sacrifice compensation, NNUE pricing, wedge at Nxb5,
counterfactual rung — while every clause requiring SYNTHETIC styled opponents to carry
behavioral signal failed (style classification, tendency prediction, residue
identification). Synthetic 90–140cp biases drown in playout eval noise three probes in
a row. The corpus, not the mechanisms, is the binding constraint.

**Measurement landscape (operator, recorded):** the stack can only be measured against
open/offline anchors — the lichess open database (CC0 PGN dumps WITH player ratings;
partial fishnet `[%eval]` annotations), the lichess bot API (the only live rating
anchor open to engines), and fishnet as the external analysis oracle. chess.com is
gated to players. Reachability from this environment verified (2013-01 dump: 17 MB,
~120k rated games, HTTP 200 + range requests). Search-strength/ELO claims remain a
non-goal per the stockfish-rs plan; the lichess data buys external falsification of
the SUBSTRATE claims on real opponents at known strength.

**Queued: D-SF-LICHESS-1** (`examples/lichess_ladder.rs`, in flight at writing): (a)
L1-agreement per rating band — if the awareness ladder is real, agreement with our
static-NNUE rational-best climbs with Elo (simultaneously calibrating our L1 against
the human strength ladder); (b) per-player tendency/residue identification on real
style structure (the decisive OPPONENT-3 test); (c) embedded `[%eval]` census +
wedge/surprise cross-check against the external oracle where coverage allows.
