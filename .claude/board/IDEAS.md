# Ideas Log — Open + Implemented + Integration (triple-entry, append-only)

> **Append-only ledger** for every architectural idea, speculative
> design, "what if we tried X" moment. Ideas accumulate here
> whether or not they're ready to ship. When one gets implemented,
> it moves from Open → Implemented → Integration (a row linking
> the idea to the plan entry that scheduled it + the PR that
> shipped it).
>
> **Purpose:** a speculation has nowhere else to live until it's
> scoped into a plan. This file is the speculation surface. Ideas
> die or graduate here; nothing is lost.

---

## Triple-entry discipline

Every idea moves through three ledger sections in this file:

1. **Open Ideas** — speculative; captured when proposed.
2. **Implemented Ideas** — idea became real; row appended with PR
   anchor + integration-plan D-id reference.
3. **Integration Plan Update Log** — the paired "what the plan
   changed when this idea landed" row, citing the specific
   `INTEGRATION_PLANS.md` version bump or `STATUS_BOARD.md` row
   flip triggered by the idea.

The row in Open is NEVER moved; its Status flips. The Implemented
row is a NEW append that cites the Open anchor. The Integration
row is a THIRD append that cites both.

This is **triple-entry bookkeeping** — three sections, same idea,
cross-linked. The cost is a bit more writing; the benefit is that
every shipped idea has an audit trail from speculation → code →
plan consequence.

---

## Rejected / Deferred

Ideas that don't graduate go into a fourth section:

4. **Rejected / Deferred Ideas** — with `**Rationale:**` and cross-
   ref to the original Open entry. The Open row's Status flips to
   `Rejected YYYY-MM-DD` or `Deferred to <when>`.

Deferred ideas can later reactivate — append a new Open entry
citing the Deferred one; Deferred row's Status flips to `Reactivated
YYYY-MM-DD <new-entry>`.

---

## Governance

- **Append-only.** Never delete a row.
- **Mutable fields:** `**Status:**` line, `**Rationale:**` line (if
  added later with more context).
- **`permissions.ask` on Edit** (same rule as other bookkeeping
  files). Write for appends stays unprompted.

## Cross-references

- `EPIPHANIES.md` — if an idea came from an epiphany, both entries
  cross-reference each other.
- `INTEGRATION_PLANS.md` — the plan version that incorporated the
  idea.
- `STATUS_BOARD.md` — the D-id status row that reflects the idea's
  shipping status.
- `PR_ARC_INVENTORY.md` — the PR that landed the code.
- `ISSUES.md` — if implementing an idea surfaced a bug, both rows
  link.

---

## Kanban Format (priority + scope on every entry)

Every idea carries:
- **Priority** — `P0` must-ship-this-phase / `P1` next-phase / `P2`
  eventual / `P3` speculative.
- **Scope** — which agent / deliverable / domain: `@<agent-name>`,
  `D<N>` (plan D-id), `domain:<grammar|codec|arigraph|infra|...>`.

Ticket tag on each entry: `[P2 @family-codec-smith D7 domain:grammar]`.
Agents filter by `@`-mention or domain to see what's theirs.

## Open Ideas

(Prepend new ideas here with today's date. Format:)

```
## YYYY-MM-DD — <short title>
**Status:** Open
**Priority:** P0 | P1 | P2 | P3
**Scope:** @<agent> D<N> domain:<tag>

<one paragraph: what the idea is, rough scope, why it matters>

Cross-ref: <epiphany entry / plan D-id / related knowledge doc>
```

---

## Implemented Ideas

(When an Open idea ships, APPEND here with same title + PR anchor.)

```
## YYYY-MM-DD — <same title as Open entry> (from YYYY-MM-DD)
**Status:** Implemented YYYY-MM-DD via PR #NNN
**Shipped as:** D<N> in integration plan v<K>
**PR:** #NNN (commit SHA)

<verbatim original Open paragraph>

Cross-ref: <same + PR link + plan D-id>
```

The original Open entry's Status flips to `Implemented YYYY-MM-DD`.

---

## Integration Plan Update Log

(When an idea triggers a plan change — version bump, D-id status
move, new deliverable — APPEND here. This is the third-entry row.)

```
## YYYY-MM-DD — Plan consequence of <idea title> (from YYYY-MM-DD)
**Trigger idea:** <idea title> (YYYY-MM-DD)
**Plan change:** <version bump / D-id flip / deliverable added>
**Plan entry:** `INTEGRATION_PLANS.md` v<K> entry or new v<K+1> entry
**Status board update:** <D-id> → <new Status>

<one paragraph: what the plan documented differently after this idea>
```

---

## Rejected / Deferred Ideas

(Ideas that don't graduate go here.)

```
## YYYY-MM-DD — <same title as Open entry> (from YYYY-MM-DD)
**Status:** Rejected YYYY-MM-DD  |  Deferred to <when / trigger>
**Rationale:** <short explanation>

<original Open paragraph>

Cross-ref: <original + any related>
```

---

## How to use this file

**When a new architectural idea surfaces** — prepend to **Open
Ideas** with today's date. One paragraph. If it needs more, create
a knowledge doc and link.

**When an Open idea ships** — APPEND to **Implemented Ideas**; flip
Open Status to `Implemented YYYY-MM-DD`. Then APPEND to
**Integration Plan Update Log** with the plan consequence.

**When an Open idea is rejected** — APPEND to **Rejected /
Deferred Ideas** with Rationale; flip Open Status.

**When a deferred idea reactivates** — prepend a NEW Open entry
citing the deferred one; flip the deferred entry's Status to
`Reactivated YYYY-MM-DD <new-entry>`.

Nothing is lost. Every idea has a trail from speculation to
disposition.

## 2026-04-19 — FP_WORDS = 256 (supersede the 160 plan)
**Status:** Open
**Priority:** P1
**Scope:** @container-architect @truth-architect ndarray domain:vsa domain:codec

Current: `FP_WORDS = 157`  (10,048 bits, 5-word remainder on AVX-512)
Planned (H6 harvest): `FP_WORDS = 160`  (10,240 bits, SIMD-clean)
**Proposed: `FP_WORDS = 256`**  (16,384 bits, cache-line-perfect, matches `Container<[u64; 256]>`)

**Why 256 over 160:**

- LanceDB `FixedSizeList<UInt8, 2048>` = 2 KB per row = 16,384 bits already.
  Padding 157 → 256 in Container currently wastes 99 u64 per fingerprint (62%).
- Container primitive is already `[u64; 256]`; unifying `FP_WORDS` with it
  means zero padding, zero remainder loops at any SIMD level, cache-line
  alignment guaranteed (2 KB / 64 B = 32 cache lines, every level clean).
- VSA capacity: Plate's bound rises ~1.6× (bundled-items-per-fingerprint
  capacity ~1,500 → ~2,400 at error < 1%).
- No rebake of stored fingerprints needed — Container was already 256 wide.

**Cost:** ~30 LOC in `ndarray::hpc::vsa` constants + test updates;
docs shift "10k VSA" language → "16k VSA". Plate's capacity math re-tune.

**Supersedes:** TECH_DEBT entry "FP_WORDS = 157 (not 160); SIMD remainder
loops remain" — the 160 plan was the right direction, 256 is the correct
destination.

**Cross-ref:** `.claude/knowledge/cross-repo-harvest-2026-04-19.md` H6,
`.claude/board/TECH_DEBT.md` FP_WORDS entry. Container layout in
`lance-graph-contract::cam::Container`.

## 2026-04-19 — CORRECTION-OF 2026-04-19 FP_WORDS = 256 (supersede the 160 plan)
**Status:** Open
**Priority:** P1
**Scope:** @container-architect @truth-architect ndarray domain:vsa domain:codec

The prior entry conflated **two distinct substrates** and used
"10,000-D binary VSA" framing that must be eliminated from the workspace.

### Two substrates (never collapse them again)

1. **Hamming binary fingerprint** — `Container<[u64; 256]>` = 16,384
   BITS = **2 KB**. For popcount-Hamming queries. **Not VSA.** FP_WORDS
   going from 157 → 256 applies here.

2. **VSA superposition substrate** — 16,384 DIMENSIONS × float.
   For bind / bundle / permute / unbind. **Never binary.**

   | Encoding | Bytes / fingerprint | LanceDB column |
   |---|---|---|
   | `Vsa16kF32` (lossless baseline) | **64 KB** | `FixedSizeList<Float32, 16384>` |
   | `Vsa16kBF16` | **32 KB** | `FixedSizeList<BFloat16, 16384>` |
   | `Vsa16k` u8 × 5-lane | **80 KB** | struct of 5 × `FixedSizeList<UInt8, 16384>` |
   | `Vsa16k` BF16 × 5-lane | **160 KB** | struct of 5 × `FixedSizeList<BFloat16, 16384>` |

   Current `Vsa10kF32` = 10,000 × f32 = 40 KB is the legacy narrower
   size. Move to 16,384-D.

### Governance: ban "10,000 binary" framing

**There shall be zero occurrences of "10,000-D binary VSA" / "10,000-bit
VSA" in any `.claude/*`, knowledge doc, skill doc, or board file.**
Those phrases collapse two distinct objects. When writing about:

- Binary fingerprint: say "16,384-bit Hamming fingerprint" / "2 KB
  Container" — never "VSA".
- VSA substrate: say "16,384-D float VSA (64 KB lossless / 80 KB u8-5-lane
  / 160 KB BF16-5-lane)" — never "binary", never "10k".

### Tasks (follow-up PR, not this one)

1. Rename `CrystalFingerprint::Vsa10kF32` → `Vsa16kF32` and
   `Vsa10kI8` → `Vsa16kI8` in `lance-graph-contract::crystal`.
2. Re-address role-key slices from [0..10000) → [0..16384) in
   `lance-graph-contract::grammar::role_keys`. Maintain disjoint
   slices; scale each segment proportionally (e.g., SUBJECT 2000 → 3200).
3. Update storage contracts to `FixedSizeList<Float32, 16384>` and
   the 5-lane struct variant. LanceDB needs no patching — both are
   native.
4. Sweep 21 lance-graph + 7 ndarray files for "10,000" / "Vsa10k*"
   / "10 000-D" / "10K VSA" → rename or reclassify. Exclude
   legitimate uses (query limits, sample counts, dollar amounts,
   speedup ratios, scale factors).

**Supersedes:** 2026-04-19 IDEAS entry "FP_WORDS = 256 (supersede the
160 plan)" — that entry was correct for the binary Hamming substrate
but mislabeled the VSA as "16,384 bits". The VSA dimension is 16,384
FLOAT, not bits.

## 2026-04-19 — REFINEMENT-OF 2026-04-19 CORRECTION-OF FP_WORDS scope
**Status:** Open
**Priority:** P2
**Scope:** @integration-lead domain:vsa

Scope the "no 10000-D VSA" ban to the three contexts where it is
LEGITIMATELY in use and must be preserved:

1. **Grammar prototype** — `lance-graph-contract::grammar::{role_keys,
   context_chain}`. Role-key slices `[0..10000)` are shipped in PR #210.
   Rename to 16384-D is a follow-up that must re-scale all slice
   boundaries proportionally; until that PR lands, 10,000-D addressing
   stays in grammar docs.
2. **Quantum prototype** — `CrystalFingerprint::Vsa10kF32` holographic
   residual mode (`crystal-quantum-blueprints.md`). Quantum-mode docs
   keep 10,000-D naming until the rename PR.
3. **Ladybug-rs / bighorn fresh imports** — PRs #200-203 brought the
   cognitive stack + CognitiveShader + BindSpace at 10,000-D. Known
   memory cost (see TECH_DEBT "Ladybug 10000-D memory blowup"). Do not
   rewrite these imports; migrate as part of the ladybug → contract
   consolidation PR.

**Elsewhere** (epiphanies, session handovers, OSINT plans, calibration
docs, prompts not in the above scopes): strip 10,000-D / Vsa10k*
references — they propagate the legacy substrate into contexts where
only 16,384-D is relevant.

**Files in-scope (keep as-is):**
- `.claude/plans/elegant-herding-rocket-v1.md` (grammar + quantum)
- `.claude/knowledge/crystal-quantum-blueprints.md` (quantum)
- `.claude/knowledge/integration-plan-grammar-crystal-arigraph.md` (grammar)
- `.claude/knowledge/linguistic-epiphanies-2026-04-19.md` (grammar)
- `.claude/knowledge/endgame-holographic-agi.md` (quantum / holographic)
- `.claude/prompts/session_ndarray_migration_inventory.md` (i8 10000D
  transient accumulation layer is the ladybug-import artifact)
- `.claude/board/PR_ARC_INVENTORY.md` (historical record of #208-#210)

**Files out-of-scope (sweep-candidate for rename / restatement):**
- `.claude/board/LATEST_STATE.md` — snapshot says `Vsa10kI8/F32` in
  CrystalFingerprint; append correction row naming the target
  (`Vsa16kI8/F32`) when rename PR lands.
- `.claude/prompts/session_deepnsm_cam.md` — "10,000 bits each
  (= Base17 compatible)" is a binary-VSA confusion; correct.
- `.claude/board/EPIPHANIES.md` — "10,000-D f32 VSA is lossless under
  linear sum" entry from another session — keep as historical record;
  append correction that the target is 16,384-D.

**Acknowledges:** the prior CORRECTION-OF entry framed the ban as
workspace-wide; it is not. Three scopes preserve 10,000-D legitimately
until the coordinated rename PR lands.

## 2026-04-19 — REFINEMENT-2 HDC substrate is FP16 / BF16, not FP32
**Status:** Open
**Priority:** P1
**Scope:** @container-architect @truth-architect domain:vsa domain:codec domain:memory

The prior CORRECTION-OF + REFINEMENT-OF entries assumed f32 as the
HDC baseline. Correction: **HDC superposition substrate is FP16 (or
BF16), not FP32.** Bundle-accumulation magnitudes stay in half-precision
range; f32 only buys ceremony.

**Size table (corrected):**

| Substrate | Per-row bytes |
|---|---|
| 1,024-D Jina v3 (FP16) | 2 KB |
| 1,536-D OpenAI text-embedding-3-small (FP16) | 3 KB |
| 3,072-D Upstash Vector cap (FP16) | 6 KB |
| **10,000-D HDC (current, FP16)** | **20 KB** |
| 10,000-D HDC (legacy f32 naming) | 40 KB ← `Vsa10kF32` today |
| **16,384-D HDC target (FP16)** | **32 KB** |
| 16,384-D HDC × u8 5-lane | 80 KB |
| 16,384-D HDC × BF16 5-lane | 160 KB |

**Revised memory math for the ladybug 700-1100 MB blowup:**

- At f32 (40 KB/row): observed 700-1,100 MB = ~17-27 K live rows.
- **Had it been FP16 (20 KB/row): same population = 350-550 MB.**
- 16k × FP16 (32 KB/row): same population = 560-880 MB — **cheaper
  than the current f32 state**, not worse.

The 16k rename is memory-positive IF paired with f32 → FP16 migration.
Without the precision drop, 16k × f32 (64 KB/row) does inflate the
problem. The coupled change is the right design.

**Architectural constraint (why LanceDB, not a vector-db SaaS):**

Commercial managed vector DBs cap at ≤ 3072 dimensions (Upstash).
Pinecone, Weaviate, Qdrant — all optimize for 768-3072 dense
embeddings. HDC substrate at 16,384-D is an order-of-magnitude wider
and cannot live in those systems. LanceDB's `FixedSizeList<BFloat16,
16384>` is the only viable column type across OSS + managed
offerings. **This is why lance-graph is The Spine, not a plug-in.**

**Updated rename scope for the follow-up PR:**

1. Type rename: `CrystalFingerprint::Vsa10kF32` → `Vsa16kBF16`
   (not `Vsa16kF32`). f32 variant retires.
2. Role-key slices re-address `[0..10000)` → `[0..16384)`.
3. Storage contract: `FixedSizeList<BFloat16, 16384>` as the canonical
   HDC column; 5-lane struct for multi-representation workloads.
4. Compute: preserve f32 accumulation internally where numerical
   stability matters (unbundle / unbind hot path), round-trip via BF16
   for storage.

**Supersedes:** prior CORRECTION-OF entry's "Vsa16kF32 (lossless
baseline): 64 KB" line. The lossless baseline is BF16 at 32 KB.

## 2026-04-19 — lance-graph-cognitive refactor: dedup + merge + excise
**Status:** Open
**Priority:** P2
**Scope:** @integration-lead @container-architect domain:cognitive domain:refactor

26,240 LOC across 11 modules, yesterday's ladybug-rs harvest staged
here. Not wholesale duplicate of other crates — it's the complementary
cognitive layer sitting above `lance-graph::graph::spo` (store) and
`lance-graph-contract::{grammar,crystal}` (primitives). Needs targeted
cleanup, not deletion.

**Keep in place (canonical cognitive-layer impl):**

- `grammar/` — `GrammarTriangle` (NSM × Causality × Qualia); plan D3
  explicitly calls it (`.claude/plans/elegant-herding-rocket-v1.md`).
- `spo/` — Crystal layer: `sentence_crystal`, `context_crystal`,
  `gestalt`, `meta_resonance`, `cognitive_codebook`. Sits on top of
  the SPO store + contract's `CrystalFingerprint` enum.
- `spectroscopy/` — detector 511 + features 408 LOC, standalone
  unique cognitive-spectroscopy work.

**Merge into active crates (if feasible):**

- `search/temporal.rs` (187 LOC) → `lance-graph-planner::strategy`
  (temporal search as a strategy, not a separate module).
- `cypher_bridge.rs` → check overlap with `lance-graph::parser`;
  merge or retire.

**Inspect and decide DTO vs excise:**

- `fabric/` — protocol surface? If yes → move to contract. If no →
  keep or excise.
- `world/` — world model, likely DTO. If yes → move to contract
  (parallel to `state_classification_pillars` already there).
- `container_bs/` — BindSpace container. If DTO → move to contract
  OR let ada-rs consume through contract. If stub → excise.

**Excise:**

- `learning/` — empty stub inside lance-graph-cognitive (distinct
  from the standalone `crates/learning/` DTO crate which is a
  different thing). Delete.
- `wip` feature-flagged modules — finish or excise, not both.
- `core_full/` — catch-all; decompose into themed modules or
  migrate contents into the modules above.

**Cost:** ~1 week refactor PR. Zero functional change; contract
compliance improves, dependency graph tightens. Contract-adoption
rule from CLAUDE.md (§Current Status In-Progress) is the governing
principle: public surface through contract, implementations behind
traits.

**Cross-ref:** TECH_DEBT "ladybug-rs retired — ada-rs + lance-graph
exclusively" (2026-04-19). Active plan:
`.claude/plans/elegant-herding-rocket-v1.md` D3 depends on
lance-graph-cognitive's grammar module staying put.

## 2026-04-19 — CORRECTION-OF 2026-04-19 lance-graph-cognitive refactor
**Status:** Open

Remove all ada-rs mentions from the prior entry — ada-rs is documented
only in ada-rs, not here.

Correction: the contract surface for cognitive DTOs **already exists** —
it shipped in PR #206 (Pumpkin NPC framed: state classification pillars
+ shader-driver endpoints). The lance-graph-cognitive refactor is about
cleaning up yesterday's messy imports against that EXISTING contract, not
creating new traits.

Replace "let ada-rs consume through contract" → "exposed through
existing contract from PR #206". Replace "move to contract" in
fabric/world/container_bs bullets → "check if PR #206 contract already
covers it; if yes, delete the import; if no, extend contract via Pumpkin
framing".

## 2026-04-19 — Fractal round-trip codec: phase+magnitude preservation
**Status:** Open (research)
**Priority:** P3
**Scope:** @cascade-architect domain:codec domain:fractal

Follow-on to the fractal-leaf CORRECTION (EPIPHANIES 2026-04-19).
The unsolved codec problem:

**Encode both phase and magnitude in fractal form so that decode is
a usable round-trip (not just a statistical twin).**

Pure fractal parameters (D, w, H, σ) reconstruct a *statistical twin* —
same shape, different bits. That's argmax-usable for random queries
(Meyer cardiac-FD analogy), but loses exact inner products. Two rows
with same (D, w, H) produce indistinguishable argmax rankings, which
is a feature for compression but means per-row identity is gone.

Round-trip requires pinning enough reference points that fractal
interpolation fills between them faithfully. Candidate recipe:

1. Hadamard-rotate row → coefficients c[0..n).
2. Sample at 17 golden-step positions → Base17 anchors (34 bytes).
3. Compute fractal params of the full sequence → Descriptor (7 bytes).
4. Decode: generate fractal interpolation that matches (D, w, H) AND
   passes through the Base17 anchor points with correct signs +
   magnitudes. Fractal-interpolation-between-samples.
5. Inverse Hadamard → reconstructed row.

This binds the existing workspace primitives (Base17 golden-step,
Stacked samples, fractal descriptor) into a single round-trip codec
where:
- Base17 carries the PHASE ANCHORS (sign + coarse magnitude at 17
  golden positions).
- FractalDescriptor carries the SHAPE (D, w, σ, H) for interpolation.
- Combined: 34 + 7 = 41 bytes/row, self-similar reconstruction between
  anchors, exact at anchors.

Open research questions:
- Does fractal interpolation actually converge to something close to
  the original between anchor points? Iterated Function System theory
  says yes for self-similar sequences; empirical for Qwen3 unknown.
- Phase half (sign-sequence fractal) still needs its own probe.
- How to parameterize the sign flips between anchors without storing
  them bit-by-bit? Barnsley fern-style IFS over sign space?

All gated behind `lab` feature until the round-trip math works.
Not a production codec priority until the two unmeasured probes
(sign-sequence fractal CoV, fractal-interp-between-samples fidelity)
return positive.

Cross-ref: fractal-codec-argmax-regime.md, EPIPHANIES 2026-04-19
CORRECTION, PR #216 magnitude-only half.

## 2026-04-19 — Fractal codec validation path: use codec_rnd_bench + ICC_3_1
**Status:** Open (operational)
**Priority:** P2
**Scope:** @cascade-architect domain:codec domain:psychometry

Existing infrastructure (no new tooling needed):

**`crates/bgz-tensor/src/quality.rs`** (shipped):
- `spearman` · `pearson` · `kendall_tau` · `icc_3_1` · `cronbach_alpha`
- `mae` · `rmse` · `top_k_recall` · `bias_variance`

**`crates/thinking-engine/examples/codec_rnd_bench.rs`** (shipped):
- Loads 128 rows from safetensors
- Computes ground-truth pairwise cosines
- Runs each registered codec through the 10-metric suite
- Outputs markdown table (see `bench_qwen3_tts_62codecs.md` / `bench_gemma4_e2b_62codecs.md`)

**Correct fractal validation** (replaces the hand-rolled CoV probe):

1. Implement `FractalCodec::decode(anchors: Base17, desc: FractalDescriptor) -> Vec<f32>`
   - Fractal interpolation between 17 golden-step anchor points
   - Shape constrained by (D_mag, w, H_mag, D_phase, σ)
   - IFS / wavelet-interp / similar — this is the "genius" piece
2. Register as `FractalCodec(41 B)` candidate in codec_rnd_bench.rs
3. Run:
   ```
   cargo run --release --features lab \
     --manifest-path crates/thinking-engine/Cargo.toml \
     --example codec_rnd_bench -- /path/to/Qwen3-8B/shard.safetensors
   ```

Output: markdown row with ICC_3_1 + Cronbach's α + Spearman ρ + Pearson r
+ top-5 recall vs ground truth. Direct comparison against the existing
67-codec sweep (I8-Hadamard leader at 9 B, adaptive codec, etc.).

**Gates:**
- ICC_3_1 ≥ 0.95 on k_proj @ 41 B/row → fractal codec beats I8-Hadamard on
  argmax-rank reliability (real argmax-wall crack, measurable).
- ICC ∈ [0.85, 0.95] → useful hybrid layer, not standalone winner.
- ICC < 0.85 → fractal codec inferior; the unpublished negative.

All gated behind `lab` feature. Bench-only, never main. Endpoint already
has ICC / Cronbach / Spearman — no new dependencies. The only missing
code is the decode function.

Cross-ref: EPIPHANIES 2026-04-19 fractal-leaf CORRECTION.
`crates/bgz-tensor/src/quality.rs` lines 47/279/362. `codec_rnd_bench.rs`
for the bench structure + existing codec registration pattern.
