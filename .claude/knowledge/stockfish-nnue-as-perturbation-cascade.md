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
| "position after ply *v*" | `QueryReference::at(v, rung)` + deinterlace — a zero-copy projection, no replay | **[H]** — the read is [G]; that it needs no recomputation is the D-SF-EPISODIC-1 gate |
| Analysis horizon (present vs hindsight) | the **rung**: low rung = reason strictly at ply *v*; high rung = spoiler-read the game's outcome to label the move (training-target labeling) | **[H]** — rung semantics are shipped; the chess *use* (hindsight = eval-vs-result delta) is the D-SF-RUNG-1 probe |
| Opening variants (e4/d4/…) | **episodic basins** = le-contract **L1 `part_of:is_a`** rails (a variant is-a line is-a opening) | **[H]** — rails are [G]; that opening-trees fit the L1 rail shape is the D-SF-BASIN-1 probe |
| Transpositions (same position, different move-order) | **L2 `memberof:members`** — one position node, member-of many variant basins | **[H]** — the many-basins-one-node shape is exactly L2; the probe confirms it round-trips |
| Episodic recall ("games that reached this pawn structure") | AriGraph `EpisodicMemory` / `markov_soa` — retrieve-similar over position fingerprints | **[G]** for the store; **[H]** that chess positions are good episodic keys (D-SF-ARIGRAPH-1) |
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

- **D-SF-EPISODIC-1 — a game as a temporal version-stream.** Model an N-ply game
  as N `temporal.rs` entries; read position-at-ply-*v* via `QueryReference::at`.
  **Gate:** the projected accumulator at ply *v* == the accumulator freshly
  computed from the ply-*v* FEN, byte-for-byte — *reusing the L4 chained oracle
  already green in `incremental.rs`*. This is the **strongest temporal probe**:
  it turns the existing incremental oracle into a temporal-replay oracle at zero
  new ground-truth cost. If it passes, "position-at-version is a zero-copy
  projection" is [H]→[G].
- **D-SF-BASIN-1 — opening variants as L1 basins.** Encode a small opening tree
  (e.g. 3 openings × 2 variants) as L1 `part_of:is_a` rails; assert a position
  resolves to its basin set. **Gate:** transposition (same FEN via two move-orders)
  lands on ONE position node that is `memberof` both variant basins (L2) — the
  many-basins-one-node round-trip.
- **D-SF-RUNG-1 — hindsight labeling via rung.** For a decided game, read each
  move at a **low rung** (eval as-of that ply) and at a **high rung** (spoiler-read
  the result); the low↔high delta = the "was this move objectively good given the
  outcome?" signal. **Gate:** the two reads differ exactly where eval and result
  disagree (blunders that won, brilliancies that lost) — proving rungs carry the
  present/hindsight horizon, not just a version index.
- **D-SF-ARIGRAPH-1 — positions as episodic keys.** Store game positions in
  AriGraph `EpisodicMemory`; `retrieve_similar(position_fp)` returns games that
  reached a near structure. **Gate:** retrieval precision above the Jirak noise
  floor on a labeled set (e.g. all games with an isolated queen's-pawn) — MEASURED,
  never asserted.

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
