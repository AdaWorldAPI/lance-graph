# dismech-causal-replay-v1 — causality replay over the DisMech palette

> **Status: PROPOSED — PLAN/BOARD ONLY.** No production implementation in
> this PR. Sibling and FIRST DOMAIN INSTANTIATION of
> `rubicon-loco-rung-cognitive-fabric-v1` (2026-08-29): that plan verified
> the fabric (loco carrier, callable atoms, masks, counterfactual,
> `temporal.rs` replay); this plan binds ONE domain to it — recorded
> disease-mechanism causal chains under the `ogar-dismech` predicate
> palette. **Every `F-RLR-*` STOP gate is inherited unchanged**; nothing
> here re-litigates a fabric verdict.
>
> **Operator direction (2026-08-31, verbatim anchors):** the mechanics are
> *"causality replay shaped"* (the keystone); candidates evaluate by
> *"Mengenlehre"*; *"Shannon proprioception + EWA Sandwich"* carry
> uncertainty; *"rung 0-9 inference frontier scheduling"*;
> *"ogar-loco Orchestration as config / ogar-r2il thinking as config"*;
> *"lance-graph p64 stockfish-rs ALU 64x64 4096 shape later,
> cognitive-shader-driver as ALU crunching"*.

## §0 State audit — what EXISTS (verified 2026-08-31, sibling repos checked per F-RLR-11)

| thing | state | where |
|---|---|---|
| causal predicate palette | **MINTED, loco-addressable** — 19 predicates (`causes`..`variant_of`), `FnIndex` `0x90..=0xA2`, concept `0x0333` | OGAR `ogar-dismech` (mirrors `ogar-ro`'s `RelationVocabulary` pattern) |
| mechanism modules | **ADDRESSED** — 124 cross-disease templates, 585 nodes, `conforms_to` address inheritance, append-only registry at `0x0333` | consumer-side transcode (`module_registry`) |
| corpus transcode + replay gate | **SHIPPED** — resolver reproduces the upstream builder field-for-field; parity diffs against the 1,870 pathographs the upstream resolver produced (replay-equivalence IS its gate) | consumer-side transcode (per OGAR's own doc: the public `dismech-rs` posture — zero OGAR dep, like `ogar-obo`) |
| baked slab | **NONE EXISTS.** The bake bin emits loose TSVs to a local dir; no release tag, no pin-table row anywhere | — (consumer leg, §4 W-C) |
| `lance-graph-ontology` cache | **UNRELATED** — OGIT-TTL tenant naming registry; zero dismech references | this repo |
| set algebra (∩ ∖ ⊆) | **EXISTS + CALLABLE** — `contract::revision::EvidenceMask` (#1075) | this repo |
| counterfactual / intervention | **EXISTS** — `contract::counterfactual`; R2IL V4 probe rows | this repo |
| replay substrate | **EXISTS** — `planner/src/temporal.rs` (version-range reads, per-reader rung, replayable) | this repo |
| EWA covariance | **EXISTS, not loco-addressable** — `jc::ewa_sandwich` (Pillar 9/9b certify `J·W·Σ·Wᵀ·Jᵀ` push-forward) | this repo |
| Shannon entropy | **EXISTS, SCATTERED** — ≥6 uncoordinated `entropy()` surfaces (rubicon §C) | this repo |
| loco call pattern to copy | `lance-graph-ogar/src/recipe_vocab.rs` — `op_of` / `recipe_of` / `ladder_program` | this repo |
| chain-step eval kernel | `CausalEdge64` + `NarsTables` ("precomputed NARS as lookup tables") | causal-edge / planner cache |

**The gap is exactly one thing: nothing REPLAYS a recorded causal chain
against an evidence set.** Every ingredient exists; the composition does not.

## §1 The object

**Replay, never generate.** The domain knowledge is trajectory-shaped
(curated mechanism chains); differential evaluation is *which recorded
trajectories are consistent with this evidence*, i.e.:

```text
candidates = { recorded chains }                       (sets of chain ids)
evidence arrives → EvidenceMask ∩ on support,
                   ∖ on refute (the skip-word semantics as ordinals)
frontier pick   → expected entropy reduction over candidates (§W4)
each expansion  → REPLAY the chain: loco calls under the dismech
                   vocabulary, one CausalEdge64 + NarsTruth revision per
                   step, trace on temporal.rs
counterfactual  → the SAME replay with one edge cut (Pearl rung 3)
grade           → replay equivalence: same inputs ⟹ byte-identical trace
```

The trace of an evaluation is itself a loco program — thinking as config
makes causality replay **self-hosting**: yesterday's evaluation replays
today byte-for-byte, and template-equivalence machinery grades it.

## §2 Design constraints (inherited + house)

1. **No new carrier** (`F-RLR-2`, automatic STOP): steps are loco
   `(FnIndex, value)` calls under the already-minted dismech vocabulary;
   traces ride `temporal.rs`; per-step state is `CausalEdge64` +
   `NarsTruth`. A `ReplayFrame` struct that owns what a lane already holds
   is the zero-copy-warden's MATERIALIZES verdict — methods on carriers,
   never a new layer.
2. **No second set algebra:** Mengenlehre = `EvidenceMask` ops. If a
   needed op (symmetric difference, popcount-weighted split) is missing,
   it is added THERE, with the field-isolation discipline.
3. **No 7th entropy.** W4 opens with a consolidation DECISION — which of
   the ≥6 existing `entropy()` surfaces is canonical for the candidate
   readout — recorded on the board before any call site is written.
4. **Constitutional (rubicon, carried verbatim):** ambiguity/entropy/
   parallax never terminate cognition — only the Rubicon boundary owns
   stop/commit/veto. **Shannon reduction ≠ evidential increase**; the
   frontier picks WHERE to look next, never WHAT is true. Prefetch ≠
   belief.
5. **Fixtures in this repo are SYNTHETIC.** The real corpus and its bake
   live consumer-side (§4 W-C pointer); this repo's gates run on
   generated chains whose ground truth is constructed, exactly the
   `dismech_parity` discipline at engine scale.
6. **Predicate ordinals, not strings**, on every hot path; the palette
   binds at the membrane (`op_of`-shaped). A `typed → stringify → parse`
   step anywhere inside is `F-RLR-7`'s STOP.
7. **Rung semantics:** `r` = the `RungLevel` discriminant 0..=9 (rubicon
   §I pin). Horizon reads via `EpistemicMode::for_rung`; nothing here
   privileges a rung.

## §3 Waves

### W0 — the two NNUE numbers + palette round-trip (measure, no code) — `D-DCR-0`

Stockfish-depth claims are arithmetic over two measured quantities this
workspace has never measured for THIS composition:

- **evals/ms**: one chain step = `NarsTables` lookup + `CausalEdge64`
  revision, benched over synthetic chains (existing kernels, bench only).
- **effective branching**: candidate-set shrink per evidence item under
  `EvidenceMask` ops on generated corpora of 10³/10⁴/10⁵ chains.
- **palette round-trip**: dismech `FnIndex` `0x90..=0xA2` compose/
  decompose through `ogar_loco` (cite the OGAR-side tests; add none here
  unless a gap is measured).

**Gate (two-sided):** numbers recorded with the harness committed;
the deferred ALU wave (§5) states its BUY threshold in terms of them.
KILL: if a single full scan of 10⁵ chains costs less than one frontier
decision, the scheduler is decoration at that scale — record it and
descope W5 accordingly.

### W1 — replay core — `D-DCR-1`

Replay ONE synthetic chain end-to-end: loco calls under the dismech
vocabulary → per-step `CausalEdge64` + `NarsTruth` revision → trace on
`temporal.rs` → typed receipt on the #879 sealed-cycle path.

**Gates:** (a) determinism — two replays of identical inputs produce
byte-identical traces; (b) can-fire — a one-edge perturbation of the
chain produces a trace that DIFFERS at exactly the perturbed step index;
(c) silence — a perturbation of a non-replayed sibling chain changes
nothing. Disable-verified per the falsifiability rule.

### W2 — Mengenlehre candidate evaluation — `D-DCR-2`

Candidate sets as masks; support evidence intersects, refuting evidence
subtracts — as ordinals, never strings.

> **⊘ CORRECTION (2026-08-31, preflight before the wave was spawned).** The
> line above originally named the refute class as *"the skip-word semantics —
> `biomarker/disputed/modifier/protective/refuted/unknown`"*. That set is
> real, but it is **the wrong axis for this wave**, and building W2 on it
> would have been a category error:
>
> | axis | what it decides | where it belongs |
> |---|---|---|
> | evidence **stance** — `dismech_evidence::Supports` (`Support` / `Partial` / `Refute` / `NoEvidence`), already shipped, measured ~89,800 occurrences | does this evidence item support or refute a candidate | **W2's ∩ / ∖** |
> | the **graph-construction skip filter** — a predicate over the source's own `relationship_type` / free-text `association` fields | does a source item infer a mechanism edge AT ALL | upstream of any candidate set; it runs while the graph is built, not while candidates are evaluated |
>
> The skip filter is not a refutation. An item it skips never becomes an edge,
> so there is nothing for `∖` to subtract — a candidate set built from the
> graph has already had them excluded. Wiring it as the refute class would
> subtract a second time, against a set that never contained them.
>
> Two further facts make the original line worse than merely mis-axised, and
> both are reasons to leave that vocabulary where it is:
>
> 1. **It is two lists, not one, and the asymmetry is load-bearing.** The
>    enum-valued arm and the free-text-token arm do not carry the same
>    members — `refuted` appears in one and not the other. That asymmetry is
>    upstream's, preserved deliberately by its transcode as *not a bug to
>    fix*. A single 6-ordinal enum in this crate would flatten it, and the
>    flattening would be invisible.
> 2. **Its authority is consumer-side**, in the crate that transcodes the
>    source graph builder. This crate would be holding a second copy of a
>    filter it does not own and cannot check — the exact mirror-without-a-fuse
>    shape `ogar_codebook` and (this PR) `DISMECH_PREDICATES` both avoid by
>    pairing every mirror with a drift gate. There is no gate available here.
>
> **So W2 uses `Supports`**, which is already in the contract, already
> measured, already exhaustively round-tripped, and is the axis the ∩/∖
> algebra is actually about. If a wave ever genuinely needs the skip filter,
> it belongs where the graph is built, with its two arms intact.

**Gates:** anti-vacuity (`kept * 3 < total` on the synthetic corpus);
two-sided discrimination (a discriminating evidence item must split the
set; a redundant one must NOT shrink it further); the refute path has
its own can-fire + stay-silent pair.

### W3 — counterfactual replay (Pearl rung 3) — `D-DCR-3`

The SAME W1 replay with one edge cut, through `contract::counterfactual`
— never a second replay path.

**Gates (two-sided):** cutting a load-bearing edge flips the chain's
consistency verdict; cutting a redundant (parallel-path) edge does not.
Both fixture halves constructed so the property is reachable — the
fixture's SHAPE is part of the coverage.

### W4 — Σ transport + the Shannon readout — `D-DCR-4`

Per-edge evidence counts → `NarsTruth` (`w/(w+1)`) → a small Σ; pushed
along the replayed path via `jc::ewa_sandwich`'s certified form (Pillar
9/9b are the standing gates — the mint rule from the math-atoms
discussion applies: no loco address for the atom while its pillar is
red). Candidate-set entropy = the proprioception readout; value of the
next observation = expected entropy drop.

**Gates:** Σ growth monotone under fan-in; a discriminating observation
must score strictly above a redundant one (two-sided); the entropy
CONSOLIDATION decision (constraint 3) recorded before the first call.

### W5 — frontier scheduling — `D-DCR-5` — **HELD**

Priority = expected-information-gain / rung-cost, horizons via
`EpistemicMode::for_rung`. **HELD behind the operator rung 5–9 table
ruling** (persona-vs-rung-ladder O-items) AND W0's KILL check. Interface
note only; the scheduler never decides truth, causality, band promotion
or revision acceptance (§K carried).

### W-C — consumer leg (pointer, not this repo's work)

The corpus bake (TSVs → a tagged, gated artifact) and the live-evidence
binding happen consumer-side under that repo's own pin doctrine. This
plan's engine consumes classid-addressed rows and synthetic fixtures
only. §0's "NONE EXISTS" row is the reason this pointer exists.

### Deferred — p64 64×64 ALU + shader-driver dispatch

64×64 = 4096 = one node's bit budget: the ALU tile is exactly one node.
NNUE's real trick is the incrementally-updated accumulator over a delta
— evidence landing IS the delta. **BUY only on W0's numbers**; until
then this row exists so nobody builds it early. Dispatch belongs to
`cognitive-shader-driver` (the loop that can't-not-think while F is
above floor) — never a bespoke driver.

## §4 What this plan deliberately does NOT do

No new carrier, mask type, entropy surface, replay substrate, or
scheduler-that-decides-truth. No renames. No consumer corpus data in
this repo. No R2IL transcode of anything (`ogar-r2il`'s own ruling:
executed, never pre-converted). No second Rubicon.

## §5 Model allocation

Opus: this plan, wave specs, review, W4's consolidation decision.
Sonnet: W0 harness, W1–W3 implementation against written specs, one
source in / one shape out. Haiku: only the guarded-executor card.

## §6 D-id table

| id | wave | status |
|---|---|---|
| D-DCR-0 | W0 measurements | Queued |
| D-DCR-1 | replay core | Queued (after D-DCR-0) |
| D-DCR-2 | Mengenlehre evaluation | Queued |
| D-DCR-3 | counterfactual replay | Queued |
| D-DCR-4 | Σ transport + Shannon readout | Queued (entropy decision first) |
| D-DCR-5 | frontier scheduling | **HELD** (operator rung table + W0 KILL) |
| D-DCR-6 | consumer-leg pointer honoured (no corpus data lands here) | standing gate |

---

## §3a — PRIOR-ART RECONCILIATION (added 2026-08-31, W0)

⊘ **§0's audit was under-cited.** It swept the sibling repos (F-RLR-11) but
not this repo's own contract module list or plans index. Two pieces of prior
art belong in the state table, and both STRENGTHEN the plan:

| prior art | what it gives this plan |
|---|---|
| `contract::dismech_evidence` (686 LOC, shipped) | the measured evidence vocabulary AND a ready-made two-sided falsifier population: `IndirectKnownIntermediates` = hidden-mediator oracle; `IndirectUnknownIntermediates` = epistemic-restraint control (recovering a mediator there IS the failure) |
| `dismech-causality-v3-v1.md` §11 + `D-CV3-0..6` | the held-out benchmark, already specified with measured arms: **2,449** oracle edges / **534** diseases, **4,076** restraint rows, **361** unknown rows; splits A–E reported separately; "two-sided by construction" |

**The join:** D-CV3's benchmark **is** this plan's W1–W3 falsifier; D-DCR is
the engine it grades. W1–W3 therefore consume those arms (once D-CV3-0..2
land the frozen TSVs consumer-side) and MUST NOT invent a parallel corpus.
Nothing in §0 is contradicted — "no baked slab exists" still holds, and the
corpus pin is D-CV3-0, Queued.

## §3b — W0 RESULT (D-DCR-0, measured 2026-08-31)

Harness: `crates/lance-graph-contract/examples/dcr_w0_replay_budget.rs`
(release; deterministic LCG, no clock seeding — a probe for a replay plan is
itself replayable). Corpus magnitudes are READ from §11, never re-derived.

**1. Step throughput** — one step = `NarsTruth::revision` + `EvidenceMask::intersection`:

| candidate set | chain len | steps/ms | ns/step |
|---|---|---|---|
| 64 | 16 | 29,184 | 34.3 |
| 1,024 | 16 | 23,628 | 42.3 |
| 4,096 | 16 | 14,285 | 70.0 |

**2. Branching shrink** — 1.53×–1.66× per evidence item, flat from 10³ to 10⁵
candidates. **Fixture-set densities** (2/3 support, 1/10 refute): this measures
the MECHANISM's cost and scaling, never the corpus's real discriminative power.
That number needs the frozen oracle/restraint TSVs (D-CV3-0..2).

**3. KILL check — did NOT fire.** Full scan of the oracle arm (2,449 chains,
len 4) = **0.906 ms**; one frontier decision over 64 observations = **0.008 ms**
(~100× cheaper), crossover at **~25 chains**. The corpus sits ~98× above
crossover ⇒ **W5 stays live on cost grounds** (it remains HELD on the operator
rung 5–9 table, which is a different gate).

**4. Kernel split — the ALU wave's actual question.** `NarsTruth::revision`
alone 41,596 ops/ms (24.0 ns); 4096-bit intersect+count 11,038 ops/ms
(90.6 ns) ⇒ **MASK dominates by 3.8×**. A 64×64 tile accelerates the half that
costs — the "4096 bits = one node" shape argument survives measurement.

**5. ALU BUY threshold (stated, as W0 owed).** The whole oracle arm replays in
**2.74 ms** at chain length 16. **BUY only when a workload sustains
>10× that in one budget** (≈143,000 steps/ms); below it the scalar path is not
the bottleneck. The tile is correctly aimed and correctly deferred.

**What W0 did NOT measure**, so nobody cites it as if it had: the real
per-evidence discriminative power (needs D-CV3-0..2), `CausalEdge64`'s packed
step (planner-side, one dependency layer out of this zero-dep crate), and any
loco dispatch cost (OGAR-side palette; the round-trip is covered by
`ogar-dismech`'s own tests, and duplicating them here would be a second
truth).

## §3c — W0 CORRECTED (codex review on #1118; three findings, all valid)

⊘ **§3b's numbers are SUPERSEDED.** They are kept above (append-only) because
the correction is the finding. Three defects, all pushing the same direction —
they inflated the mask half and measured a kernel this plan never named:

| # | defect | consequence |
|---|---|---|
| P1 | v1 timed `NarsTruth::revision` (f32, contract-side); §3 W0 defines the eval as **`NarsTables` lookup + `CausalEdge64` revision** | the headline throughput and the 2.74 ms corpus figure came from a substitute kernel. Disclosing the substitution did not make it the promised measurement |
| P1 | v1's mask fixture was `Bits(Vec<u64>)` → a heap alloc inside every timed `intersection`, while `impl<const N: usize> EvidenceMask for [u64; N]` **already ships** (`revision.rs:70`) and is the real p64 shape (`[u64; 64]` = 4096 bits) | `malloc` was charged to the arithmetic a tile would accelerate — the exact claim it supported |
| P2 | v1 ran the KILL gate on the 2,449-edge corpus; §3 names **10^5 chains** verbatim | a pre-registered gate was never evaluated as pre-registered |

**Corrected measurements** (probe moved to `lance-graph-planner/examples/`,
where `causal-edge` is reachable; `[u64; 64]` via the shipped impl; both KILL
scales at one candidate width):

| quantity | v1 (superseded) | corrected |
|---|---|---|
| chain step | 70.0 ns (f32 substitute + allocating mask) | **34.7 ns** (`NarsTables::revise` + `CausalEdge64::forward`) — 28,818 steps/ms |
| 4096-bit mask | 90.6 ns (allocating) | **61.5 ns** alloc-free `[u64; 64]`; the Vec shape costs 73.4 ns, so allocation was ~16% of it |
| kernel split | "MASK dominates 3.8×" | **MASK dominates 1.77×** |
| KILL @ 10^5 (pre-registered) | never run | scan **13.88 ms** vs decision **0.007 ms** ⇒ does not fire |
| KILL @ 2,449 (real corpus) | 0.906 ms vs 0.008 ms | scan **0.340 ms** vs 0.007 ms ⇒ does not fire |
| crossover | ~25 chains | **~53 chains** |
| oracle arm @ len 16 | 2.74 ms | **1.36 ms** |

**What survives and what does not.** The *direction* survives — the mask half
still dominates on the corrected, allocation-free numbers, so a 64×64 tile is
aimed at the half that costs. The *margin* does not: 1.77× is under half of
what v1 claimed, so the ALU case is materially weaker than the first pass
said, and the BUY threshold stands at >10× this corpus in one budget
(≈288,000 steps/ms). W5 stays live on cost at BOTH scales.

**The lesson (recorded because it repeated inside one probe):** every one of
the three defects was a *fixture* defect, not a code defect — a substitute
kernel, an allocating container, a moved goalpost. A measurement's fixture is
part of its claim. A fourth instance was caught in the same pass without
review help: `dense_mask(rng, 1)` sets **no** bits (`x % 1 == 0` always), so
the frontier decision was scored against an EMPTY live set; `all_ones()` is
now its own constructor and `dense_mask` asserts `one_in >= 2`.

## §3d — "Does the replay engine need to borrow masking from ndarray?" — MEASURED

Operator question, 2026-08-31. Answered with the W0 harness rather than from
architecture. Three different objects are called "masking" in this stack and
**they are not the same kind of thing**:

| object | width | shape | verdict |
|---|---|---|---|
| `ogar_r2il::CallMask` | `[u64; 3]` = 192 bits (≤180 calls/node) | lazy word-tests, no alloc, no `Vec<Call>` built | **needs nothing.** At three words, a slice-API call costs more than the work. It is already allocation-free and lazy — the properties SIMD would be bought for |
| replay candidate set | `[u64; 64]` = 4096 bits | the shipped `impl EvidenceMask for [u64; N]` | the real candidate — W0 measures it dominating the promised step by ~1.8× |
| `ndarray::hpc::jitson` | n/a | JSON config → Cranelift **native scan kernels** (`ScanParams`/`RecipeIR`/`ScanKernel`) | **not a masking library at all.** A kernel *compiler*. Its relevance is to W5's frontier decision (a scan-shaped workload), never to the mask half |

**Measured, at 4096 bits (probe §2):**

```
[u64; 64] scalar (EvidenceMask)        65.2 ns
ndarray simd_int_ops::mask_and (U64x8) 60.4 ns   => 1.08x — DEAD HEAT
decomposition:  SIMD and 11.1 ns  +  scalar popcount 56.7 ns
                => the POPCOUNT half is 5.1x the AND
```

**So the answer is neither yes nor no.** ndarray's SIMD mask kernel is fast —
11.1 ns for 64 words is the `U64x8` path working exactly as advertised. It
changes nothing end-to-end because **the AND was never the cost**: the
reduction is, at 84% of the mask half. `mask_and` writes 512 bytes to `dst`
and the popcount then re-reads them scalar.

**The primitive that would pay does not exist yet:** a fused
`mask_and_popcount(&[u64], &[u64]) -> u32` that keeps the AND result in
registers and reduces with `VPOPCNTDQ`. ndarray HAS `popcnt` on its AVX-512
typed wrapper (`simd_avx512.rs:2987`) but exposes no fused slice-level API, and
the workspace's SIMD invariant ("all SIMD from `ndarray::simd`") means that
primitive must be **added in ndarray**, never hand-rolled here.

**Consequence for the deferred p64 64×64 wave:** its target moves. A tile that
accelerates the *bitwise op* is aimed at 11.1 ns of a 65.2 ns half; a tile that
fuses op+reduction is aimed at all of it. The BUY threshold is unchanged
(>10× this corpus in one budget) — what changed is what to buy.

**Not scheduled, deliberately.** This is a measured direction, not a wave.
The cross-repo ask (an ndarray fused mask+popcount primitive) is the
operator's call to make, per commitment on upstream asks.
