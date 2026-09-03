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

> **⊘ OPERATOR RULING (2026-09-01) — there are THREE kinds of Mengenlehre,
> and W2 shipped only the third.** Verbatim shape of the correction: *"fox
> mammal wombat whale should not simply eliminate."*
>
> A whale disagreeing with the typical mammal features is **information about
> the field**, not grounds to remove the whale from the mammals. Set
> difference applied to that disagreement destroys exactly the structure that
> makes the map worth having.
>
> | # | kind | what it does | status |
> |---|---|---|---|
> | 1 | **propagation / the field map** | propagate precision about a knowledge stage over the WHOLE field. Global. Produces the map of mammal-agreement vs whale and wombat: agreement, disagreement, support chains, and MISSING LINKS, filled into the HHTL nodes. Maps evidence out of the boring rails (`is_a` / `part_of` taxonomy) into a non-boring causality graph with propagated node edges. **This is what explains Mengenlehre.** | **UNBUILT** — `D-DCR-2b` |
> | 2 | **threshold elimination** | elimination as a READING of that map at a measured threshold, on a mathematical scale — Shannon proprioception, EWA sandwich, Hambly, Lyons | belongs with W4's Σ / entropy machinery, not with the set algebra |
> | 3 | **question masking** | scope to ONE case / patient / question. **Logically distinct from any generalization** | **SHIPPED as W2 / `dismech_candidates`** |
>
> **W2 is not wrong; its CLAIM was too wide.** `∩`/`∖` over a candidate set is
> a correct kind-3 mask and nothing else. It was documented as "Mengenlehre
> candidate evaluation", which annexes kinds 1 and 2 by implication — and a
> later wave building on that label would have taken elimination for the
> substrate's primary operation. The module is re-scoped in place; the algebra
> is unchanged.
>
> **Kind 1 is the substrate's real product and is the bigger wave.** It does
> not narrow a set at all: it writes agreement/disagreement/support/missing-link
> structure onto the field, and only THEN can kind 2 read a threshold off it
> or kind 3 mask it to one question. Ordering the three that way is the ruling.
>
> **The question must be part of the reasoning.** *"Our logical reasoning
> should be aware of the question it is asking and the substrate should be
> precise about the answers"* — which is why the relation's flavour is carried
> rather than flattened: Tarski (what a relation ASSERTS, not merely that it
> holds), `CausalEdge64` bits **59-60** (the same two bits read as
> `TrustTexture` or as `CausalTopology`) and **61-63** (`ReasoningBand`), and
> the precision between **"explains"** and **"relates to"**. A verdict that
> collapses those into a boolean has thrown away the answer's precision to
> report its polarity.
>
> **One term left OPEN rather than guessed:** the operator's *"24×i4
> flavours"*. Two readings exist in-tree and this plan does not pick one
> without a ruling — `contract::atoms::I4x32` is **32** signed-i4 lanes in 16
> bytes (33 atoms allocated over it), while the V3 content-blind facet's
> 12-byte payload is exactly **24** nibbles. The second matches the count; the
> first matches the name. Recorded as a question, not resolved by inference.
>
> **⊘ ANSWERED (operator, 2026-09-01) — the facet reading, not `I4x32`.** The
> width is chosen so a position can collect enough dimensions from its children
> to speak for itself. It is the 12-byte payload read at NIBBLE granularity: 24 signed
> lanes summarising an HHTL position's children, in the width the node already
> owns. `I4x32` matched the name and nothing else. Full reasoning — including
> why the SIGN collapses agreement / disagreement / silence into one quantity —
> in §W2b below. Left in place rather than deleted: the wrong candidate is
> worth carrying so it is not re-proposed.

### W2b — the field map — `D-DCR-2b` — **PROPOSAL, awaiting operator scope**

Kind 1 of the ruling above: *propagate precision about a knowledge stage over
the WHOLE field*. **Not built.** What follows is a survey; every "exists" line
was read in the tree.

#### ⊘ CORRECTION (operator, 2026-09-01) — the first survey had the node backwards

The first version of this section carried the row *"where an HHTL position
lives → `episodic_basin` → in the node's own KEY, **never** a second copy in
the value slab"*. Read as a statement about the NODE that is exactly wrong, and
it forbids the thing this wave exists to build. The operator's correction: the
plan is for HHTL nodes to BE SoA, so a key-only reading of the node is exactly
backwards.

The reconciliation, and it is clean rather than a reversal:

- **The ADDRESS stays in the key.** `episodic_basin`'s ruling is about the
  cascade POSITION (`HEEL`/`HIP`/`TWIG`, key bytes `4..10`) — no second copy of
  an address the key already carries. That much survives untouched.
- **The node has a VALUE, and the value is the whole point.** An HHTL node is
  a node: `key(16) | edges(16) | value(480)`. Making HHTL nodes SoA means the
  value slab carries a **self-organizing summary of the position's children** —
  upstream/downstream inheritance, basin agreement, disagreement, missing
  links. That is what "hydration" has been naming all along.

The constraint that DECIDES the placement, rather than leaving it open: a
summary changes when its children change, and a key is an identity. A mutable
summary in the key would re-address the node on every sweep — so it is a value
lane, necessarily. The two halves do not compete.

#### Three readiness states, not two

`episodic_basin`'s doc names two corpora (ontologies hydrated; books must be
spawned). The operator names **three**, and the third is the one neither that
doc nor the first survey had:

| state | corpus | what exists |
|---|---|---|
| (a) **not yet created** | books, DeepNSM-v2 | nothing — the tree must be spawned first (the TOC minted as the skeleton, an SoA node per entry) |
| (b) **already present** | ontologies | the `part_of:is_a` rails are hydrated nodes; the anchor exists before the basin does |
| (c) **implicit in the rails, not hydrated** | — | a rail edge exists, so the position is *implied*; no node was ever hydrated at it |

**(c) has no carrier and no primitive.** It is distinct from both neighbours —
not absent (the rail names it) and not present (nothing holds a value there) —
and it is the state a `hydrate` step would consume. `lance-graph-hydrate` is
NOT that step: it is artifact-level (`Absent → Hydrated → Dirty|Flushed` over
an object store), and the shared vocabulary is a rhyme, not a reusable
mechanism. Node-level hydration does not exist in the tree.

#### Why 24×i4 — and it answers two of the three gaps below

The 12-byte content-blind payload is 24 nibbles. Every shipped `CascadeShape`
carves it at BYTE granularity — `G6D2` (6×2), `G4D3` (4×3), `G3D4` (3×4), all
12 units (`facet.rs`, `CASCADE_UNITS == 12`). A **nibble**-granular reading —
24 units — is not in the shipped set. **⊘ Stale: measured 2026-09-01, it IS —
`ValueTenant::CausalWitness` ships `G24N4`. See the census below; the sharper
statement is that `G24N4` is a lane shape name, never a `CascadeShape` variant.** Signed i4 in `[−8, 7]` IS shipped carrier
semantics (`atoms::I4x32::sext4`), just at other widths (`I4x32` = 16 B,
`I4x64` = 32 B); neither is 12 B.

On why 24, per the ruling: enough dimensions to summarise a position's children
so the node speaks for itself. Twenty-four signed dimensions summarise a position's
children in the width the node already owns — no new field, no widening.

And the SIGN is what makes it one quantity instead of three: **`+` agreement,
`−` disagreement, `0` silence.** That is directly the whale case — a whale
records as a negative lane against the mammal neighbourhood and stays a mammal,
because a lane is a value and not a removal.

This also resolves the standing "24×i4 flavours" question against the wrong
candidate: it is NOT `atoms::I4x32` (name matches, width does not); it is the
facet register read at nibble granularity.

#### It is NOT greenfield — what already ships

| piece | where | what it already does |
|---|---|---|
| one-hop propagation | `lance-graph-planner/src/adjacency/propagate.rs` | `adjacent_truth_propagate`: `truth_out[t] = semiring.multiply(truth_in[s], edge_truth)`, merging at fan-in with `semiring.add`. The propagation KERNEL exists |
| the position's address | `contract::episodic_basin` + `contract::hhtl` | key bytes `4..10`; `NiblePath` (16ⁿ nibble router, O(1) shift) |
| inheritance down the path | `class_view::FieldMask::inherit` | mask-inherits-as-delta: a child's presence mask is the parent's OR its own delta |
| the rails as nodes | `contract::episodic_basin` | state (b) above |
| the relation's flavour | `CausalEdge64` bits 59-60 / 61-63 | `CausalTopology` / `ReasoningBand`, already read by W3's `EdgeRole` |

#### The gaps that remain

1. ~~A disagreement quantity that is not an elimination~~ — **carrier named**:
   the sign of an i4 lane. What is still open is the *arithmetic* that fills it.
2. ~~A missing-link representation~~ — **carrier named**: lane `0` is silence,
   distinct from a negative. Still open: whether "the rail says these should
   relate" is derivable at the node or needs the rail's own reference.
3. **Convergence semantics for a GLOBAL sweep.** One hop is a function; a field
   map is a fixpoint. Bounded-iteration vs to-convergence vs
   single-sweep-per-version is a substrate decision with real cost. **Still
   fully open.**
4. **A node-level hydrate primitive for state (c).** Absent, and it gates (a)
   and (c) both. **⊘ RULED buildable (see below): it is MECHANICAL — structure
   from the hierarchy only, epistemic lanes zero.**

#### The value-lane census — measured, 2026-09-01 (`D-DCR-2b`)

Read off `canonical_node.rs`'s `VALUE_TENANTS` descriptors, not off prose.
Offsets are slab-relative (row offset − 32).

| # | tenant | kind×elems | bytes | slab `[from,to)` |
|---|---|---|---|---|
| 0 | `Meta` | U64×1 | 8 | `0..8` |
| 1 | `Qualia` | U64×1 | 8 | `8..16` |
| 2 | `MaterializedEdges` | U64×4 | 32 | `16..48` |
| 3 | `Fingerprint` | U8×32 | 32 | `48..80` |
| 4 | `HelixResidue` | U8×6 | 6 | `80..86` |
| 5 | `TurbovecResidue` | U8×16 | 16 | `86..102` |
| 6 | `Energy` | F32×1 | 4 | `102..106` |
| 7 | `Plasticity` | U32×1 | 4 | `106..110` |
| 8 | `EntityType` | U16×1 | 2 | `110..112` |
| 9 | `Kanban` | U64×1 | 8 | `112..120` |
| 10 | `FrozenStyle` | U8×12 | 12 | `120..132` |
| 11 | `LearnedStyle` | U8×12 | 12 | `132..144` |
| 12 | `ExploreStyle` | U8×12 | 12 | `144..156` |
| 13 | `Tekamolo` | U8×16 | 16 | `156..172` |
| 14 | `CausalWitness` | U8×16 | 16 | `172..188` |
| 15 | `EpisodicBasin` | U8×32 | 32 | `188..220` |
| | **occupied** | | **220** | of 480 |
| | **free** | | **260** | next offset `220` (row `252`) |

**Three findings, and the first corrects this very section.**

1. **`G24N4` DOES ship — the paragraph above is stale.** It says a
   nibble-granular 24-unit reading "is not in the shipped set". Measured, it is:
   `ValueTenant::CausalWitness` (14) is documented as *"read in the **`G24N4`**
   shape as 24 signed `i4` loci"*, with the codec (`CausalWitnessFacet::get` /
   `set`, sign-extension and `[−8,7]` clamp) shipped and tested. What remains
   true — and is the sharper statement — is that `G24N4` is a **lane shape
   name, never a `CascadeShape` variant**: sub-byte granularity's sanctioned
   home is a lane, and `CascadeShape` stays byte-axis-only. So W2b's carrier is
   **not greenfield**; only its semantics are.

2. **…and precisely because it ships, W2b cannot live in that lane.** Two
   independent blockers, both operator-locked in `CausalWitness`'s own doc:
   its value law is *"a **context pointer**, never a strength/magnitude (loci,
   not values)"* — and a W2b child-summary is exactly the magnitude case; and
   its slots `16..24` are RESERVE-DON'T-RECLAIM. Reusing it would break its
   value law and its reserve rule at once. That answers half of the first open
   decision below: **a carve inside `CausalWitness` is ruled out**; a new
   tenant vs a carve elsewhere is still the operator's call.

3. **Space is not the constraint.** 260 of 480 slab bytes are free and the next
   tenant appends cleanly at slab offset `220`. A 12-byte `G24N4` register costs
   ~4.6 % of the remaining slab. The doctrine that applies is *"clean / SoC over
   packed"* — with this much headroom, packing a second concern into an existing
   lane is the rare last resort, not the default.

**What the census does NOT settle:** versioned-vs-live, sweep granularity, and
whether kind 2 reads the map or the sweep applies it — all still open below.
The census is about bytes, not about semantics.

#### The one-node falsifier — shipped, `tests/w2b_one_node_field.rs`

Five falsifiers, each disable-verified red-then-green, at the scale that is
decidable **today**:

| falsifier | disable that makes it red |
|---|---|
| F1 three states distinguishable through the register | `summarise` returns `acc.abs()` |
| F2 **the whale case** — no value the lane can hold revokes membership | `record_summary` takes `&mut` rail and clears it when `summary < 0` |
| F3 saturation, never wrap (a sign flip is the silent catastrophe) | drop the clamp; mask instead |
| F4 all 24 lanes independently addressable | drop the neighbour mask in the nibble codec |
| F5 re-summarising the same children is stable | `summarise` ignores its input (fires the all-zero guard) |

It deliberately answers **none** of the open decisions: no lane is minted, no
byte reserved, and the summarising arithmetic is the test's own minimal choice,
labelled as such — the ruling fixes what the SIGN means, not the sum. Gap 3
(convergence for a global sweep) is untouched by construction: one hop is a
function, a field map is a fixpoint, and one node cannot speak to a fixpoint.

Worth recording that F2's first version was **vacuous** — it asserted a local
`bool` that nothing between the two lines could change, so its "disable" meant
editing the test rather than the code. It is now structural: sweep every value
the lane can hold, including the whole negative half, and require the membership
carrier to stay byte-identical while the disagreement stays legible.

#### The decisions this wave still cannot make for itself

#### ⊘ RULED (operator, 2026-09-01, second round) — the DN dissolution + the two-layer hydration split

**The siting questions were a category error**, caught with the sharpest
possible analogy: asking where the map is written is asking where the OUs are
in a distinguished name. An HHTL position is addressed by its path, and every
truncation of the path is itself an entry — `hhtl::NiblePath` already ships
`parent()`, `prefix(depth)`, `is_ancestor_of`. The basin-level node is the row
whose deeper tiers are zero, in the same store, same stride, same key layout.
Three of the four "decisions" dissolve:

| former decision | verdict |
|---|---|
| which value lane / tenant / column | **split, on reconciliation with the #1127 census**: the TREE-siting half is dissolved — the node at each prefix IS the row, no separate map store. The SLAB half (which `ValueTenant` inside the 480-byte value carries the magnitude register) is real and census-constrained: NOT `CausalWitness` (loci-never-magnitude law + reserved slots); append margin at slab 220, 260 B free; the mint is the operator's |
| versioned or live | **dissolved** — rows are Lance-versioned because every row is |
| sweep granularity | **dissolved** — a sweep is scoped by a prefix; field / classid / subtree is one mechanism at three depths |
| does the sweep eliminate | settled by the three-kinds ruling itself: the sweep RECORDS; elimination is kind 2's threshold READING, always a separate act |

**Every node is an SoA row**, and the three corpora differ only in how much of
the tree already has rows:

| corpus | what exists before the wave |
|---|---|
| books | nothing — the SoA rows are not yet minted by **DeepNSM-v2** (v2, not v1); the ontology is built from scratch |
| rails without nodes | at least the HIERARCHY — parent/child is nameable even where no row sits |
| OWL / any ontology | parent/child almost always preserved; rows largely present |

In most of the three, hydration is incomplete — and the ruling's load-bearing
split is what keeps that safe to fix:

**Mechanical hydration ≠ original causality predicates.**

- **Mechanical hydration** mints the SoA row at a nameable DN from the
  hierarchy alone. It is structural, runs in all three corpora (TOC skeleton,
  rail parent/child, OWL parent/child), and writes STRUCTURE ONLY: address,
  `is_a`/`part_of` edges, presence. Its epistemic output is exactly **silence**
  — every agreement lane `0` — which is the zero-fallback ladder holding one
  level up: an unwritten lane reads as absent, never as an assertion.
- **Original causality predicates** — the dismech palette (`causes`,
  `explains`, `relates_to`, …), Tarski-precise assertions, the signed
  agreement lanes — are epistemic knowledge. They arrive only from evidence
  and propagation, carry provenance, and are NEVER fabricated by the minting
  step. A mint that writes a nonzero lane is the wave's hardest defect class:
  structure impersonating knowledge.

#### ⊘ RULED (operator, 2026-09-01, third round) — one hop up, one hop down

The corpora list is wider than the three-state table suggested, and every
entry lands in one of the same states: **book** (skeleton to mint), **redmine
/ odoo / AD** (hierarchy present — AD literally IS the DN case), **OWL / RDF**
(parent/child almost always preserved). Same ladder, more members.

The register is a floor, not a cap: **the nibbles can be expanded if
necessary**, and a node further up may carry **multiple 24×i4 registers**
where that proves valuable (mammal: accumulated agree/disagree; whale family:
generic vs mammal-specific; whale: specific). The lanes are upstream/
downstream inheritance on the mathematical scale — Shannon proprioception /
EWA sandwich / Mengenlehre readouts over them.

**The hard constraint, load-bearing:** a parent's register expresses its
DIRECT children accumulated — agreement AND disagreement — never the
grandchildren. **One hop up and down.** Grandchild information reaches a
grandparent only through the child's own accumulated register; the global
field map is a composition of one-hop summaries. This is the tree-shaped
Chapman-Kolmogorov discipline (`I-SUBSTRATE-MARKOV`) and the same locality
`FieldMask::inherit` already has (parent ∪ own delta), and it settles the
sweep's convergence question structurally: bottom-up one-hop passes, each
node a pure function of its direct children.

Shipped against it (slice 2): `hhtl::direct_children` (the one-hop selector —
exactly depth+1, never grandchildren, so any accumulator it feeds is
structurally unable to reach past the children) and
`BasinLanes::accumulate_children` (per-lane saturating signed sum; empty →
SILENT). Locality is made observable across two levels: a grandchild move
absorbed by the child's own saturation leaves the grandparent byte-identical;
one that moves the child moves the grandparent.

#### ⊘ RULED (operator co-architect, 2026-09-01, fourth round) — the pair, the scope fix, the grounded lanes

- **The signed net is FALSIFIED, not limited**: balanced conflict must be
  distinguishable from silence. Carrier superseded by
  `epistemic_bassin::EpistemicBassin24` (`agree_u4[24] + disagree_u4[24]`;
  net/contest/entropy derived; `Contested` survives accumulation). The old
  pin stays green on the old type as the record.
- **#1127's law re-scoped**: loci-never-magnitude binds the A9 READING, not
  tenant 14's bytes — readings are classid-selected per row, so the bassin
  can read the same physical lane. **No tenant mint** until one real row
  needs ContextLoci and the bassin simultaneously (the built
  `EpistemicWitness = 16` mint was discarded uncommitted).
- **Nibbles are values on named axes, never addresses** — topology is the
  key + an indexed 16-bit child mask; episodic identity is `EpisodicBasin`
  references; exact proof/Σ stay in their exact carriers. The facet classid
  names the axis basis and its version.
- **Lanes grounded in shipped math, checked not invented**: Shannon ΔH via
  `info_gain_u4` over `dismech_candidates` counts; EWA tension via
  `sigma_tension_u4` in quarters of `sigma_propagation::pillar_5plus_bound`
  (7 = the 1.75× slack); Tarski pressure = the pair's own counts, exact
  depth stays in premise ancestry; **Hambly-Lyons gets NO lane** while
  sigker's classification is gated on jc Pillar 11 (the red-pillar rule).
- Associativity claim corrected: exact within one call; recursive
  composition of clamped registers is monotone, not associative —
  conflict-preserving either way, which is the property that matters.

#### ⊘ RULED (operator, 2026-09-01, fifth round) — 256:256 by classid; bytes as microcode

Palette space is NOT scarce and the epistemic vocabulary must not be
designed as if it were: **the classid swaps the whole 256-entry palette**
(256:256), exactly as blockly-rs runs the Scratch vocabulary and ogar-r2il
runs 82 machine ops over the SAME `0x90..` indices ogar-dismech uses — no
collision, different classids. And ogar-r2il's addressing is the macro
mechanism: **one byte can address an entire 360-byte script** — microcode.
Consequences, recorded before the catalogue is designed:

- The six universal calls stay CORE (`0x86..0x8B`) — core bytes are read
  from the core in EVERY vocabulary (`VocabularyTable::compose` cannot be
  forged), so TERNLOG/BELNAP_JOIN work inside every palette: "for literally
  everything" is a property of core placement.
- Everything domain-shaped — the 24 named axes (as a `ValueCodebook`),
  per-axis macros, revision verbs — belongs in an **epistemic vocabulary**
  with its own classid and a full 256 space, never crammed into reserved
  core slots.
- Compound reasoning macros follow r2il's shape: a byte in the epistemic
  palette addresses a whole loco body (360 bytes = 180/120/90 calls under
  the lane shape) — scripts all the way down, each still replayable.

**⊘ Extension (operator, same day): one classid per RUNG level — the brutal
form.** Each rung of the reasoning ladder gets its own 256-entry palette, so
a program's vocabulary IS its epistemic altitude: rung-2 bytes resolve
against the 144 verb atoms, rung-3 against the 34 NARS tactic recipes (THE
runbooks), rung-4 against StyleFamily macros — and with the microcode shape,
**one byte at rung N addresses a whole script at rung N−1**. Escalation and
de-escalation become ADDRESSING operations (a classid change), not control
flow; every level stays replayable because every level is still loco bodies.
This also mirrors, at the palette level, the tier-graded register ruling at
the node level (granularity down, aggregation up). Consequence for W5
(D-DCR-5, HELD): the operator's rung 5–9 table, when it lands, is now also
the CLASSID MINT LIST for the upper palettes — the hold therefore stays
exactly where it is; no rung classid is minted before the table.

#### ⊘ RECOVERY (operator, 2026-09-02) — semantic-family ruling; the four rounds above are regraded, not rewritten

The fourth-round "pair" and the fifth-round loco band were built on a
register that aliased three families (episodic loci, qualia magnitude,
population basin). Removed: `basin_lanes`, `epistemic_bassin`, the 24-axis
basis (`ogar-epistemic`), loco 0x87..0x8B. Kept: the DN dissolution, the
one-hop law, `hhtl::{missing_ancestors, direct_children}`, tenants 14/15.
Population-basin geometry is an accepted vacancy; W2b's kind-1 field map
reopens as a falsifier-first design step whose inputs are listed in
`E-SIX-SEMANTIC-FAMILIES-MUST-NOT-IMPERSONATE-EACH-OTHER-1` (not an axis set,
not a tenant, not a dimensionality).

#### Still open — probes, not rulings

- ~~The named 24-axis catalogue~~ — **SHIPPED as v3** (2026-09-01,
  operator: mach weiter): `ogar-epistemic` (0x0334) + the
  `epistemic_bassin::axes` mirror; derived from the ruled projection, every
  axis a grounded pressure, supersedable by a v4 classid mint. Armed
  catalogue parity follows the OGAR merge.
- **The child-mask index** — a u16 per node vs derive-by-scan; an index
  decision, not a semantics one.
- ~~The contested-collapse~~ — CLOSED by the pair; the reversed falsifier
  asserts the distinction.
- **The contested-collapse (historical).** In ONE register a balanced conflict (`+3` vs
  `−3` on a lane) sums to `0` — indistinguishable from silence. Pinned as a
  test (`a_balanced_conflict_collapses_to_silence_in_one_register`) so it
  stays loud; it is the concrete case for the ruled multi-register expansion,
  whose semantics (net + contested-mass? per-band?) are the operator's to
  shape, not inferred here.
- **The provenance marker**: where a row records mechanically-hydrated vs
  original, so state (c) stays distinguishable from state (b) after the mint.
  Not invented unilaterally; surfaced as the next placement question.

#### Falsifiers this wave would owe

- The whale case, two-sided: a whale that disagrees with the mammal
  neighbourhood must be RECORDED as disagreeing (a negative lane) and must
  still be a mammal afterwards. A sweep that removes it fails.
- A missing link must be distinguishable from a refuted one — `0` and a
  negative are different cells, and collapsing them is the same class of error
  as `NoEvidence` narrowing a set.
- Propagation must be idempotent at the fixpoint, or "the map" is not a value.
- A node hydrated from state (c) must be distinguishable from one that was
  always present — otherwise (c) silently becomes (b) and the readiness
  question stops being askable.
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

## W4 gate — entropy-surface consolidation decision, RECORDED (2026-09-03)

D-DCR-4's first clause is discharged. `examples/entropy_surface_census.rs`
(`e5e2520`) measured seven Shannon-entropy surfaces across four conventions.

**The decision:** the consolidation target is `contract::thought_atoms`
(operator ruling 2026-08-31, "universale denk atome"), **not `jc`** — `jc`
carries no entropy surface, and `E-JC-IS-THE-HOME-OF-ALL-CALIBRATED-MATH-1`
would otherwise capture entropy by analogy with cronbach/spearman. That
module currently has ZERO consumers anywhere in the tree.

**The constraint the measurement added:** routing the nearest caller
(`insight::confidence_entropy`) through the atom is NOT a drop-in. The two
agree to `0.0` exactly on non-degenerate input and carry OPPOSITE zero-mass
conventions (atom `1.0`, caller `0.0`), so the caller's empty-arena early
return must be preserved and pinned, or an empty arena's entropy inverts.

**Out of scope, named:** the shader-driver and the two cognitive forms (two
in excluded crates); the two thinking-engine forms, which are not entropies
of a distribution at all and whose correction is a lab behaviour change with
its own gate. The Σ-transport half of D-DCR-4 is untouched.
Full result: `E-THE-ENTROPY-HOME-WAS-RULED-AND-LEFT-EMPTY-1`.
