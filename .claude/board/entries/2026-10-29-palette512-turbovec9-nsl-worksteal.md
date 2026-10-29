# 2026-10-29 — Palette512 × TurboVec9 × NSL: work-steal / consolidation map

**Status: RESEARCH / FALSIFIER PLAN, not an implementation or an admission.**
**Single dated entry; do not copy this text into ndarray or TurboVec boards.**
Repo boundaries: ndarray owns numerical math and DR512, TurboVec owns the
TurboQuant index, lance-graph owns graph identity/edge evidence and NSL
integration. Prior reports in `crates/lance-graph-turbovec/KNOWLEDGE.md`
are historical measurements, **not current API compatibility evidence**.

## Thesis

**Two 9-bit spaces with deliberately different meaning:**

| Area | `ndarray::hpc::codec9` DR512 | `turbovec` 9-bit lab |
|---|---|---|
| Codes | 512 words = 511 signed numeric values + 1 NEVER_TOUCHED | 512 Lloyd-Max centroid IDs, all numeric |
| Word | P×M (one presence/polarity bit + u8 magnitude); touched zero is +0 | unsigned 0..511 index, packed into nine bit planes |
| Source truth | `Codec9::encode_touched`, `EpiLane9` | `codebook::lab_codebook9`, current upstream `Rotation` |
| Calibration | Compile-time `Calibration` + nearest rounding | Scalar lab Beta Lloyd-Max + norm correction; **no TQ+ calibration** in lab |
| Utility | Preserve negative/positive/zero/never-seen **epistemic state** | Retain the ordering of continuous embedding similarities |
| Learner role | Encode selected values and absence; preserve sparse update integrity | Generate/retrieve plausible near-neighbor / hard-negative candidate IDs |
| Forbidden inference | `P` cannot be reused as arbitrary numeric centroid-high-bit | 9-bit codes do **not** carry DR512 NaN, signedness or permissions |

Never reinterpret one codeword as the other, even though the widths match.
A mapping needs a decoded numeric interpretation and a selected encoding
policy. Never manufacture evidence by converting NEVER_TOUCHED into +0.

**Concrete LUT reuse boundary (source checked 2026-10-29):** ndarray
`src/simd_stride.rs::lut9` is an **odd-function** transform backed by
one `[u8; 256]` magnitude LUT, with `table[0] == 0` and a prescribed
presence/rounded-zero repair. It cannot implement an arbitrary TurboVec9
**512-centroid** decode table, despite the method's '9' name. Reuse the
underlying `lut_u8`/SIMD-byte operations or introduce a measured,
semantics-neutral 512-address gather **only if a consumer demonstrates
the need**; never silently turn `lut9` into a generic U9 lookup.

**Storage math (payload only)**: 9b/dim = 1.125 B/dim vs i8 1 B,
i4 = 0.5 B and f32 = 4 B. Therefore 9b costs **12.5% over i8**
and **2.25× i4**, and saves **71.875% vs f32** before metadata,
padding, blocked layouts and LUT/codebook caches. 511 signed values
in DR512 do not represent 512 distinct numerical centroid values.

## Work stealing from existing substrate (no third codec)

| Work unit | Existing canonical source | STEAL / COMPOSE | Do not reproduce |
|---|---|---|---|
| Presence / sparse lane | ndarray `src/hpc/codec9.rs`; LG `lane_codec.rs` | Codec9 encode & 512-codeword parity | Alternative `-0`/Asym/sentinel law |
| u8 / i8 distance lookup | ndarray `simd_stride::{lut_u8,lut9}` | Facade LUTs if matching exact operation | Second nibble-LUT implementation in LG |
| Coarse + signed i4 residue | ndarray `src/hpc/edge_codec.rs` | `CoarseResidueCodec`, `ProductQuantizer` | Claiming it is TurboQuant's Beta Lloyd-Max |
| Coarse search cascade | ndarray `src/hpc/cam_pq.rs` | PQ-ADC / six 256-entry per-query tables | Duplicate trained `CamCodebook` |
| Search index | AdaWorldAPI/turbovec `turbovec/src/search.rs`, `codebook.rs`, `rotation.rs` | Vector-major, grouped VNNI dot, TQ+ if available, index/ID/filter | New clone of upstream index |
| Learning input | LG `lance-graph-nsl/src/lib.rs` | `FeatureTable` + `NeighborBatch`, `SourceRef`/seal; Google `pack_nbrs` oracle | Copying features or silently reordering edges |
| Learning loss | LG `lance-graph-nsl/src/eval.rs` | Graph loss fixtures and paired DR512 loss experiments | Another TensorFlow-loss semantics |
| KV/cache precedent | LG `bgz-tensor/src/turboquant_kv.rs` | Gain/shape i4 and Hamming as an independent **control** | Calling this byte-compatible with TurboVec or certified top-k |
| AVX/VNNI matmul baseline | ndarray `simd::matmul_i8_to_i32` | Measure on GEMM-shaped rerank | Replace LUT-ADC scan with full GEMM by default |

The old `lance-graph-turbovec` bridge references `turbovec`'s removed
`ndarray-simd` feature and removed `search_polyfill_with_db` /
`reconstruct_db_i8_transposed` methods after the 2026-10-29 upstream
refresh. **Unverified and likely non-building with current sibling
main**. First run its standalone cargo gate and repair as a separate
version-aware compatibility PR; do not let this block the independent
NSL and TurboVec lab probes.

## DR64³ as the structured exact-residual learning substrate (2026-10-29 hypothesis)

**Operator hypothesis, not a proven new learner:** the intended DR64³
native relational/focus representation can provide the **addresses,
typed dependencies and rebindable context** for an *exactly recoverable*
residual learning format. That is stronger and more central than
treating DR512 as ancillary graph metadata or using TurboVec9 merely
to retrieve candidates.

### Structural law: relation first, values second

Think of a logical relationship `R_F(i,j,r)` where `i` and `j`
are globally typed identity operands admitted to the same witnessed
frame `F`, and `r` denotes a typed relation/property **only if**
the DR owner resolves NFR-OP-2 to that axis. No physical 64³ cube,
64³ allocated array, new ID codec or fixed third-axis ABI is asserted.
The current prototype remains a DR64² borrowed register with typed
frame/focus/scope guards; DR64³ is an intended logical native form
(D-NFR S9–S10, D-DR3-HOLO-ONNX, D-DR64C-A).

Within the admitted coordinate/relationship frame, a node feature or
learned state can be described by a shared, sealed anchor
`a_F(c,p)` and an exact node residual `e_F(i,p)`:

```text
F       = source/relation/coordinate/codebook generation witness
R_F     = typed relationship projection over admitted identities
x_F(i,p) = a_F(cluster(i),p) (+) e_F(i,p)
delta_F(i,j,p) = x_F(j,p) (-) x_F(i,p)
```

Here `(+)` and `(-)` are chosen **typed algebraic operations**, not
an automatic property of palette rounding or vector superposition.
For a common anchor, the anchor cancels and
`delta = e_F(j,p) - e_F(i,p)`; for different anchors include the anchor
difference. A pairwise difference is **computed on demand**, not
stored once per edge or cloned into NSL's neighbor samples. Relation
`R_F` filters/selects the eligible pairs and their typed roles.
Where the intended learning method needs an edge-specific correction,
key it by the admitted relationship and witness; do not mislabel it
as the node residual.

**"Exact" has three non-interchangeable meanings:**

1. **Exact routing/identity:** the read addresses the correct source,
   identities, typed relation, frame, presence, focus and scope, with
   stale/foreign/out-of-focus refusal. This is the DR claim.
2. **Losslessly recoverable payload:** represent the *original
   codewords* as anchor-codeword XOR delta, or use checked integers/
   fixed-point with an exactly defined scale and sufficient width.
   `anchor_bits XOR delta_bits == original_bits` holds bitwise, but
   XOR has **no numeric gradient meaning** until decoded. Checked
   integer `anchor + residual == original` holds only if no overflow,
   and a different calibration is a different type/witness.
3. **Exact training arithmetic:** the loss/gradient equal the declared
   f32/f64/TF oracle under the same operation order. Neither (1) nor
   (2) proves this; comparing gradients and convergence is a separate
   gate. Floating `anchor + (x-anchor)` is not generally bitwise
   reconstructive because of rounding/cancellation.

**Existing approximate VSA is NOT an exact multi-item residual:**
`bgz-tensor/src/holographic_residual.rs` uses XOR binding plus
majority-vote bundling and sign-only, BF16-scaled reconstruction.
Single-item XOR unbinds exactly, but multi-item bundling can collide;
its own test only admits average cosine >0.5. Work-steal its
binding ideas and its falsifier, **not** its exactness claim. A new
lossless path must either retain an invertible per-address payload,
use collision-free structured slots with accounted capacity, or
provide an explicit lossless exception/fallback. A frame is an
addressability contract, not an information-theoretic compressor.

**Palette512 (DR512)** is one optional **typed value reading** behind
this geometry. Its never-touched `P=0,M=0` must never become touched
`+0` by applying a delta, merge or quantizer; a negative nonzero
`P=0,M>0` remains a written value. The sparse Thought-Git probe
already skips never-touched and merges over previous observed state;
it is a distinct *thought overlay*, not DR³'s invented third axis.
The physical values may be dense, sparse, integer, float or encoded
codewords as admitted by the owner. DR does not acquire learning
semantics, and DT's ClassId method binding remains authoritative.

**TurboVec-U9** serves as an optional *approximate candidate generator*
for nearby embeddings or hard negatives. Its 512 numeric Lloyd-Max
centroid indices are NOT the DR512 words. The **exact** training path
must re-admit candidate IDs under `F`, follow typed `R_F` when an
edge is asserted, reconstruct the relevant values/residuals or refuse.
Approximate retrieval never confers graph membership, source
authorization or exact numerical evidence.

### Smallest decisive falsifier: native structured residual vs NSL

1. Use the existing `dr-math-binding-probe` and `lance-graph-nsl`
   fixtures. On a fixed, witnessed 64-slot focus, preserve the Google
   NSL `pack_nbrs` order/weight/dedup and select neighbor **ordinals**,
   never duplicate feature records.
2. On integer/fixed-point feature lanes, compare exact
   anchor+residual reconstruction with an **independent scalar
   oracle** on every admitted edge; test negative/positive,
   touched-zero, never-touched, overflow, changed scale, changed
   relation and stale frame. Require bitwise original recovery for
   the separate XOR-codeword control.
3. Include a **same-anchor**, **cross-anchor**, **shared neighbor used
   by many seeds**, and **out-of-focus** case. The shared neighbor
   must carry one node residual plus references, not N copies;
   escaped focus must rebind through existing admission or refuse.
4. Read residuals only at graph-demanded addresses through existing
   borrow/fold/typed projection. Compare decoded floats and NSL
   L1/L2/cosine/KL/JS loss and gradient against the frozen TF
   reference at near-zero gaps (down to 1e-3 or lower).
   Float parity is a **separate** criterion from exact storage.
5. Report end-to-end bytes including anchors, residuals, bitsets,
   tables, codebook, frame witnesses and rebuilds; count allocations,
   global→local lookups, address-space escapes, stale refusals, and
   ns/queried edge. Compare standalone per-edge residual encoding,
   shared-anchor per-node residuals and full f32 storage.
6. Prove which part of DR³ is actually necessary by disabling its
   typed relation/focus projection and comparing against a plain
   CSR + SoA + exact residual control. If only the *address binding*
   matters, admit the minimal reusable part, not a gratuitous 3D
   materialization.

This uses **existing DR and NSL labs** as oracles. No new codec,
gradient engine, physical DR³ cube or production contract follows
until the exactness, correctness, memory and learner-benefit gates
are measured. The third-axis meaning remains explicitly OPEN.

### Potential work package W7 (lance-graph owner)

**D-DR3-ERL-0 (PROPOSED):** a docs+falsifier-only probe for
`typed relation → shared anchor → exact node residual → on-demand
pair delta → NSL loss`, using the existing DR/Quack/NSL substrate.
Owned in lance-graph; ndarray owns only existing exact numerical
kernels and Codec9, TurboVec only approximate candidate search.
No authorization to introduce physical layouts or modify production
ABI. Accept if it beats the CSR+SoA exact control in at least one
measured dimension *without losing exactness*; otherwise keep the
simpler control.

## Predicate64 × Thought64: learning to recognize grounded epiphany candidates

**2026-10-29 operator hypothesis, PROPOSED research only.** Once
DR64³ provides witnessed addressability and exact structured residuals,
a *derived* `Predicate64 × Thought64` learning projection may recognize
which cross-thought compositions deserve investigation. This is a
**second-order learner about productive reasoning**, not proof of a
new fact. It reuses the C0–C3c epistemic-CoT substrate instead of
adding another truth engine, a new CE64 encoding or an automatic
epiphany publisher.

### Axis semantics and the boundary to existing DR³

- **Predicate64**: at most 64 *admitted, typed predicate slots* within
  a witnessed focus. Neither a new universal 64-predicate ontology
  nor a predicate-ID bitwidth contract.
- **Thought64**: the existing 64-slot thought budget **0..63**,
  with slot **0 = sealed givens** and **1..63 = speculative
  hypotheses**. There is **no slot 64**. A statement about "thought
  64" can name the 64-th occupied slot / a full 64-thought window,
  never silently address an out-of-range slot.
- `L_F[p,t]` is a *read-only, derived* projection of whether predicate
  `p` is present, derived or contested under thought `t` and
  sealed frame `F`, together with references to its source roots,
  proof chains, and residual values. The input is not 4,096
  independent observations. Labels from the same source root across
  63 replayed thoughts contribute evidence **once**.
- A 64×64 Boolean presence/support plane is **4,096 bits =
  512 bytes**, if materialized. Actual proof roots, numerical residuals
  and witness identities remain in their existing carriers and
  do **not** magically fit in that plane. The logical 64³
  comparison shape is 32 KiB at one bit per cell; no physical
  allocation or choice of DR³'s still-open third axis follows.
- DR³'s third-axis owner decision (NFR-OP-2) remains **OPEN**. This
  thought axis is a separately witnessed **learning projection**
  over DR relations, not a unilateral redefinition of the third
  DR axis. Frame/focus/scope/coordinate/admission checks still apply.
  The corrected `WORLD / ANGLE / THOUGHT` words are metaphors, not
  new mutable schema types.

### What could count as an epiphany?

A **candidate** is a derived conclusion `c` or mediator `m`
that was absent or not grounded in the sealed givens and becomes
groundedly derivable through an eligible *combination* of thought
paths. To learn that this was an illuminating composition, use
**externalized, replay-deduplicated evidence and ablation labels**:

1. Construct two independently identified premise chains `A`, `B`
   through the existing `DerivationLedger`, keeping the proof root
   sets and their exact original source identities. A repeated
   source in another thought is **not** independent.
2. Compare `c` with `A` alone, `B` alone and both, under a fixed,
   admitted frame; independently check inference-law validity and
   compatibility of mutually exclusive thought contexts. The model
   must not combine incompatible hypothetical worlds into a truth.
3. Read `DerivationLedger::load(c, ablate(A), ablate(B))`: its
   existing `Load::Synergistic` means either ablation alone destroys
   the conclusion; `Redundant` means two routes can support it.
   `Sensitivity` supplies the change in pooled frequency,
   confidence and proof chains. This is **derivation
   dependence**, *not* a causal intervention or a guaranteed Eureka.
4. Label a **verified useful discovery** only after a separately
   held-out test, grounded source or other admitted verification
   discriminates the proposed mediator from noise, at a controlled
   search budget. `scent_learn` currently learns from
   "hydrated/refuted" outcomes, whereas the C3c plan explicitly
   recommends **discriminating ablation sensitivity** as the next
   target. Keep candidate, investigated, corroborated and
   canonical epiphany **distinct statuses**.
5. The learner emits a ranking / resource-allocation hint, never
   NARS truth, CE64/Epi5 bits, certification, ledger roots or
   automatic `.claude/board/EPIPHANIES.md` entries. That canonical
   board is generated from a curated source: research predictions
   must not publish themselves as surviving mechanisms.

The strong hypothesis is **cross-thought composition novelty**, not
"high surprise". A good score might learn which predicate/relationship
configurations have historically produced independently verified,
load-bearing new bridges. This is appropriate for NSL
**structured learning**: adjacency consists of *typed,
source-attested proof/derivation relationships*, rather than
automatically equating neighboring embeddings.

### Exact-residual connection and possible implementation

```text
seated frame F + identity / relation map (DR) + ClassId meaning (DT)
   → selected predicate p / thought slot t (0 givens, 1..63 branches)
   → source-attested premise chains and sparse ThoughtDelta
   → exact anchor + residual values for demanded cells
   → grouped masks / folds / derivation-ledger ablation
   → epiphany candidate features (read-only)
   → NSL / scent_learn ranking update from independently checked outcome
   → candidate investigated in a NEW admitted thought (no truth write)
```

The shared-anchor exact residual is useful because it can measure a
small change in *one relation across multiple thoughts*, instead of
subtracting separately rounded absolute vectors. It does **not**
make the semantic claim exact by itself. Compare the same exact
codewords using the existing sparse XOR row delta control and a
separate checked-integer/fixed-point numeric control. Don't use
lossy majority-vote holographic bundling for the exactness gate.

### Falsifiers: learn **epiphany utility**, not correlation with replay

- **E1: 64-slot boundaries.** 0 givens, 1..63 valid branches; slot
  64 rejects. Empty / never-touched lanes contribute no update;
  negative DR512 lanes are still touched; rounded zero is +0.
- **E2: cross-thought shared sources.** The same proof replayed in
  63 thoughts must not increase evidence or learner reward.
  Two root-disjoint observations inside one thought must contribute
  twice. Neither number of thoughts nor number of paths alone is a
  learning label.
- **E3: actual synergy.** Independently typed and eligible chains
  A+B derive a novel conclusion that neither supports alone; ablate
  A, B, and both. Pinned `Load::Synergistic` and sensitivity receipts
  must agree with an independent scalar oracle; redundant, critical,
  contradictory and non-load-bearing controls must *not* be
  mislabeled as synergistic.
- **E4: incompatible hypotheses.** Branches about mutually exclusive
  counterfactual worlds cannot silently union their premises.
  Missing or stale source/frame, foreign identity, different
  codebook generation, invalid scope and out-of-focus reads refuse
  or rebind through existing laws.
- **E5: guard against hindsight.** Learn from chronologically
  earlier investigation outcomes; measure candidate ranking on held-out
  *worlds, roots and thought generations*, never on the same
  inference/verification labels used to train it. Baselines:
  unlearned C3b scent, C3c learned order, uncertainty-only, random,
  and a simple predicate-frequency heuristic.
- **E6: multiple-comparison discipline.** A 64×64 search surface
  and combinations across paths can manufacture rare-looking
  patterns from noise. Test an uninformative twin and shuffled
  root/label controls, then report candidate precision,
  false-discovery rate, verified discoveries/investigation,
  token/CPU budget and end-to-end latency.
- **E7: structural ablation.** Disable the Predicate×Thought
  interaction but preserve identical per-thought inputs, or replace
  DR³ with a plain witnessed CSR+SoA read. If it does not outperform
  these simpler controls on *verified useful discoveries*, do not
  attribute a benefit to the new logical axis.

**Explicit limitations:** The current `DerivationLedger` proves
logical derivation dependence, not intervention causality. C3c
`ScentLearner` ranks fixture mediators, not verified epiphanies.
No trained Predicate64×Thought64 epiphany recognizer exists. No
hardware performance claim or production register/bit assignment
is introduced here.

### Stronger W8 projection: Predicate64 × Thought64 × CausalEdge64

**Operator extension, 2026-10-29.** This is not merely a
Predicate×Thought score grid. Each admitted pair `(p,t)` can resolve
a complete **typed CausalEdge64 word**. The result can be viewed
as 64 *bit-slices* over 64 selected predicates × 64 thought slots:

```text
E_F[p,t]      = admitted CausalEdge64 value, IF valid/present
B_F[b,p,t]    = (E_F[p,t].raw >> b) & 1, b ∈ 0..63
L_F[p,t]      = derived evidence/ablation/read-only learning label
```

This is a **64×64×64-bit logical view, 32 KiB dense payload**
(`64×64×8 B`), or 64 separate 512-byte bitplanes. There is no
automatic dense allocation, and value-validity / proof sources /
witnesses / codebook / ClassView metadata are **additional**. A
zero CE64 word is *not* equivalent to an absent relation; require
an independently valid presence/read contract. This is **not**
a 64-element axis of independently stored causal-edge objects.
If the operator instead intends 64 separate edge-ordinal slots
each containing a full CausalEdge64, that is a **different**
64×64×64×64-bit payload (2 MiB dense), requiring edge-ordinal
authority and an explicit identity binding before investigation.

**Bitplanes are not independent semantic features.** A
`CausalEdge64` has coordinated fields: S/P/O codeword indices,
NARS frequency/confidence, Pearl causal mask, orientation,
signed i4 activation/inference reading, three bits for
plasticity/maturation by declared class reading, the six-bit W
thought slot, and upper five bits `EpistemicState5` when
provenance + ClassId × rail declaration admits that reading.
Within Epi5, topology and certification are **different**
coordinates. No raw per-bit CE64 regularizer may learn new truth
or causality from an incidental XOR/popcount distance.
`CausalEdgeV3` is a staged 96-bit successor, a distinct
typed control; do not silently reinterpret its 12-byte payload
as CE64.

**Thought breadcrumb with stage-specific validity, not double identity.**
The `t` coordinate is a *selected* thought slot (0 sealed
givens, 1..63 hypotheses, slot 64 invalid). The operator's 2026-10-17
O3 decision makes CE64 `W` bits 53..58 a **thought breadcrumb
stamped at merge/materialization**, not the corpus-root identifier
still described in stale `edge.rs`/`layout.rs` comments.
A read of an **unchanged givens** CE64 in thought t>0 can legitimately
carry `W=0`; **do not assert W==t for all merged reads**. On a
materialized write whose producer claims thought t, require the
producer's W to identify that thought and validate its frame/generation.
After a Lance seal the W value is not authoritative for identifying
the active thought (epistemic-frontier-capstone.md §4/§8).
Independent source roots stay in the DerivationLedger, never W.
`Predicate64` is a *focus of 64 predicate ordinals*: it does
not convert CE64's 8-bit palette P code into a 6-bit global
predicate ID. Check the binding (typed map/selection) and
source; never trust numeric ordinal equality across domains.

**Epiphany transition signal (read-only):** Group a fixed
predicate `p` across thought slices and compare
source-verified **typed** transitions, not raw bits. For
example, a causal claim may remain
`IndirectUnknown × Causes` (causality certified,
mediator unknown) in one thought and become
`IndirectKnown × Causes` when a grounded mediator is
hydrated in another. This is *topology resolution*, not
earned causality by thought agreement. Learning can predict
which combinations of earlier root-disjoint derivations
will make this *verified* transition possible, then schedule
the next investigation. A reward requires:
(a) independently verified, source-grounded conclusion;
(b) replay-deduplicated root sets;
(c) explicit ledger `Load::Synergistic` / sensitivity
or appropriately discriminating controls;
(d) held-out world / source / generation success;
(e) stable frame and no mutually exclusive
counterfactual-world union. An unexpected CE64 word,
changed bit, or repeated speculative replay is not itself
an epiphany.

**Work-steal the actual mechanisms, not a new engine:**
`causal-edge/src/edge.rs` typed field readers and `w_slot`;
`epistemic_state5::Epi5Declarations` for raw5 authority;
`nars::support::DerivationLedger::{load,sensitivity}`;
`nars::mediator::hydrate_from_ledger` and `pearl::hydrate`;
C3b `scent`, C3c `ScentLearner`;
`cognitive-shader-driver/examples/thought_layers_probe.rs`
for exact XOR deltas and no-NaN-write; mask-risc /
ndarray masked folds for bitplane projections;
`lance-graph-nsl` frozen Google graph-loss oracle.
The **learned output remains a priority/ranking**,
not mutation of CausalEdge64's Pearl/Epi5/NARS
truth, or new ledger evidence.

**Tightly scoped falsifier E8:** On one fixed frame,
enumerate `p=0..63`, `t=0..63` and compare
(1) a scalar source+proof oracle, (2) a borrowed
`[CausalEdge64; 64×64]` control, and
(3) bit-sliced projection/fold **without copying
the edge values**. Check byte-identical typed reads,
missing-vs-zero, **inherited W=0 vs materialized W=t**, a
stale W-reading corpus-root claim, a reserved Epi5
code, foreign class/rail and changed frame,
one true cross-thought synergy and one 63-replay
negative control. Measure query demand, prep and
bytes alongside verified-epiphany precision;
no correctness claim from speed. **DR64³'s proposed
third axis remains NFR-OP-2 OPEN:** this learning
view cannot silently settle that question.

### W8 research handoff (no new PR required yet)

**D-DR3-EPI-0 / candidate only**: work-steal the current
`DerivationLedger::{load,sensitivity,grounded_since}`,
`nars::scent::scent`, `nars::scent_learn::ScentLearner`,
`ThoughtTables::write_epi` and `lance-graph-nsl`
`FeatureTable` / `NeighborBatch` fixtures. Make a small,
independently labeled 64-thought × 64-predicate scalar
oracle and staged falsifiers E1–E7 *before* introducing new
storage or learned weights. It belongs in lance-graph, not ndarray
and not the upstream NSL implementation. A separate PR must
first check current ownership / open work and be approved.

## Why U9 may offer complementary learning benefits

Google NSL regularizes a model with labeled/unlabeled neighbor
relationships. Retrieval and loss are separate operations. TurboQuant's
random/structured rotation plus Beta Lloyd-Max quantization is
**data-oblivious in its codebook**, enabling candidate search without
per-corpus centroid training; online index maintenance is its potential
advantage. This does not confer privacy guarantees or automatically
preserve training quality.

1. **Construct better candidate graphs**: measure held-out *labeled
   graph-edge* precision/recall as well as f32-neighbor recall when
   replacing i4 with U9. False neighbors can produce incorrect regularizer
   targets. Existing controlled probe: `AdaWorldAPI/turbovec#7`.
2. **Hard-negative mining**: U9 might separate close negatives that i4
   aliases. Falsify with class/labeled graph identities, class-balanced
   precision@k, training loss/gradient comparison. Do not infer
   improvement from synthetic random-vector recall alone.
3. **Online / changing embeddings**: measure codebook stability and rebuild
   cost. The *fixed* Lloyd-Max codebook may avoid retraining that PQ needs,
   but TQ+ calibration and index caches still have cost.
4. **Progressive search**: coarse Palette256/HHTL or CAM-PQ candidates
   followed by U9 rerank only if pruning carries an admissible bound or a
   measured miss rate; otherwise no claim of exact top-k.
5. **QJL residual debias**: Google's TurboQuant paper adds a **separate
   1-bit residual stage** to correct inner-product bias. Compare
   i4 + 1-bit QJL against i9 and i8 under fixed memory and retrieval
   budget before concluding that more centroids always win. Our
   TurboVec9 scalar lab **does not** implement QJL.

DR512's complementary contribution is the **meaning of an update**:
presence/absence with one sentinel, touched-zero canonicalization,
signed value encoding and sparse read/write hygiene. It can transport
selected graph evidence, weights or states where the declared class
meaning fits, but **it does not by itself implement NSL learning,
gradient updates, or an unbiased dot-product estimator**.

## Measured receipts vs untested hypotheses

**Measured** on 2026-10-29 TurboVec9 scalar CI, deterministically
generated random unit vectors:

- n=512, d=64, q=32, k=10: scalar i4 Recall@10=0.78750,
  scalar i9=0.98438; payload 32 vs 72 B/vector; scan 149.8 vs
  268.8 us/query on AMD EPYC 7763 (Rust 1.99).
- n=2048, d=128, q=64, k=10: scalar i4=0.73750,
  scalar i9=0.97969; payload 64 vs 144 B/vector;
  scan 1.103 vs 2.102 ms/query.
- The shipped SIMD i4 is a **separate** baseline with calibration,
  so its speed cannot be interpreted as the overhead of adding bits.
- The first run says nothing about true graph-edge recall, gradients,
  convergence or production SIMD i9 performance.

**Historical** 2026-06-13 LG bridge: NativeLut 76,073 ns/query vs
PolyfillGemm 866,899 ns/query on n=20,000/d=512 (11.4×).
That compares *algorithm shapes*, but it predates current TurboVec
block-Hadamard, vector-major VNNI and current API; remeasure before
projecting ratios onto TurboVec9.

## Graph retention falsifier: initial three-noise result

2026-10-29 TurboVec PR #7 CI run
https://github.com/AdaWorldAPI/turbovec/actions/runs/40880844314,
`n=2048, dim=128, nq=64, k=10`, four ground-truth group neighbors
independent of quantized search; five tests passed.

| Noise | f32 graph-edge coverage | scalar i4 | scalar U9 | Finding |
|---|---:|---:|---:|---|
| 0.8 | 1.00000 | 1.00000 | 1.00000 | i4 already perfect on labels; U9 has no edge gain |
| 1.7 | 0.48438 | 0.42578 | 0.47656 | U9 +5.078 percentage points vs i4 |
| 2.5 | 0.06250 | 0.06641 | 0.07031 | neither encoder recovers a severely noisy graph |

f32 top-k **vector** recall for U9: 0.98125 / 0.97187 /
0.97031, vs scalar i4 0.77812 / 0.65469 / 0.67969.
Yet the **explicit graph-label** coverage can be identical or almost
equally poor. This explicitly refutes "higher ANN recall proves better
structured learning". f32 is not the label-truth upper bound: at
noise=2.5 approximate ranking even exceeds f32's chance of retrieving
a labeled edge by coincidence. No model was trained; next gate is
Google-NSL loss/gradient/held-out task evaluation.

## Ordered work packages and stop conditions

| ID / destination | Concrete deliverable | Admission gate |
|---|---|---|
| **W1 / TurboVec** | #7: i4/i9 graph labels independent of quantized retrieval; compare f32, scalar i4, scalar i9, shipped i4 | 4-edge-any/coverage and f32 top-k vs same seeds; CI output checked |
| **W2 / LG NSL** | Bind *existing* `FeatureTable` + `NeighborBatch` to selected neighbor IDs; preserve the Google weight/order/dedup oracle | 0 changed `pack_nbrs` fixtures; no feature cloning; stale `SourceRef` refusal |
| **W3 / LG NSL** | Compare f32/DR512/i4/U9 graph loss and *gradient direction* on a fixed labeled graph, with held-out examples | Gradient cosine/sign, loss error, label accuracy across epochs; f32/TF reference |
| **W4 / ndarray** | Import `dot4_u8_i8_add` + nibble ops ONLY if needed by a proven consumer; parity all backends and saturating-AVX2 counterexample | Same results vs scalar at boundary values; VNNI/GFNI tiers gated |
| **W5 / LG bridge** | Compatibility audit against current TurboVec API, then version-aware thin adapter | Standalone `cargo test --manifest-path crates/lance-graph-turbovec/Cargo.toml` green |
| **W6 / cross** | Compare fixed-budget i4, i4+QJL, i8, i9 and DR512 on **different** respective use-cases | No apples-to-oranges byte count, direct accuracy, p95 latency and L1 behavior |
| **W7 / LG DR³ + NSL** | **D-DR3-ERL-0**, source-witnessed typed relation → shared anchor → exact node residual → on-demand edge delta; compare CSR+SoA baseline | Bitwise/integer exactness independent oracle, guard refusal, no per-edge feature copy, loss/gradient parity, and end-to-end memory/time win vs simpler control |
| **W8 / LG planner + NSL** | **D-DR3-EPI-0**, derived Predicate64 × Thought64 × CausalEdge64 typed read, plus ledger ablation and C3c ranking: predict grounded epistemic/topology transitions across thought paths | E1–E8: 64-slot and stage-specific CE64-W materialization checks, CE64 field/Epi5 declaration, source dedup, synergistic ablation, hypothesis incompatibility, held-out precision, borrowed/bit-sliced vs scalar oracle; no CE64/Epi5 truth writes |

Do not open duplicate issues in ndarray/OGAR/Linear for the same
W-number without checking existing boards. **No existing production
semantics are relaxed by this research note.**

## Source references

- Google TurboQuant research: https://arxiv.org/abs/2504.19874 and
  https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/
- Google NSL:
  https://github.com/tensorflow/neural-structured-learning and
  https://www.tensorflow.org/neural_structured_learning/framework
- https://github.com/AdaWorldAPI/turbovec/pull/3 (scalar i9 codebook and baseline)
- https://github.com/AdaWorldAPI/turbovec/pull/4 (AVX2 facade route)
- https://github.com/AdaWorldAPI/turbovec/pull/7 (graph-edge falsifier)
- https://github.com/AdaWorldAPI/ndarray/pull/383 (canonical DR512)
- https://github.com/AdaWorldAPI/ndarray/pull/384 (synergy map)
- https://github.com/AdaWorldAPI/lance-graph/pull/1846 (NSL oracle)
- `crates/lance-graph-turbovec/KNOWLEDGE.md` (older measured counterexample)
- `crates/bgz-tensor/src/turboquant_kv.rs` (non-equivalent KV control)
