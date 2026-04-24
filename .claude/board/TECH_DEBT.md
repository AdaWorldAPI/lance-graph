# Technical Debt Log — Open + Paid (double-entry, append-only)

> **Append-only ledger** for knowingly-deferred work: TODOs, shortcuts,
> workarounds, unsafe assumptions, missing probes, hardcoded
> thresholds, stubs, and anything else we shipped with intentional
> debt. Debt moves Open → Paid by status-flip; rows are NEVER
> deleted.
>
> **Purpose:** separate from `ISSUES.md` (bugs) and `IDEAS.md`
> (speculation). Tech debt is **code that works but we know
> something better is owed**. This file is where we admit it.

---

## Double-entry discipline

Same pattern as `ISSUES.md`:

1. **Open Debt** — known shortcut or deferral, captured at shipping
   time.
2. **Paid Debt** — when the shortcut is replaced with the proper
   implementation, append here with PR anchor + Status flip on the
   original Open entry to `Paid YYYY-MM-DD`.

The Open entry stays in its section forever (chronology). The Paid
section accumulates retirements for audit.

---

## Governance

- **Append-only.** Never delete a row.
- **Mutable fields:** `**Status:**` and `**Payoff:**` lines only.
- **`permissions.ask` on Edit** (same rule as other bookkeeping
  files).

## Cross-references

- `ISSUES.md` — an issue may become tech debt when knowingly
  deferred rather than fixed.
- `IDEAS.md` — a rejected idea that was "the better way" often
  leaves behind a debt entry documenting the compromise shipped
  instead.
- `PR_ARC_INVENTORY.md` — which PR introduced the debt + which PR
  paid it.
- `STATUS_BOARD.md` — debt items that block deliverables are
  cross-referenced from the D-id row.
- `EPIPHANIES.md` — an epiphany often retroactively turns something
  into tech debt (the old approach is now known to be suboptimal).

---

## Kanban Format (priority + scope on every entry)

Every debt item carries:
- **Priority** — `P0` must-pay-before-next-phase / `P1` pay-soon /
  `P2` eventual / `P3` keep-tracked-but-low.
- **Scope** — which agent / deliverable / domain owes it:
  `@<agent-name>`, `D<N>`, `domain:<tag>`.

Ticket tag: `[P2 @truth-architect D10 domain:grammar]`. Same
filter discipline — agents pull their own debt by `@`-mention.

## Open Debt

(Seeded with known deferrals from recent PRs. New items PREPEND
with today's date.)

```
## YYYY-MM-DD — <short title>
**Status:** Open
**Priority:** P0 | P1 | P2 | P3
**Scope:** @<agent> D<N> domain:<tag>
**Introduced by:** PR #NNN
**Payoff estimate:** <rough LOC / time>

<one paragraph: what shortcut was taken, why, what the proper fix
looks like, any blocking dependencies>

Cross-ref: <file:line / deliverable D-id / epiphany entry>
```

### Seeded from PRs #204–#211

## 2026-04-21 — Vsa10k = [u64; 157] is a fifth representation format (Frankenstein effect)
**Status:** Open
**Priority:** P0
**Scope:** @truth-architect @container-architect D5 D7 domain:vsa domain:grammar
**Introduced by:** PR #242 / #243 (session 2026-04-21)
**Payoff estimate:** ~200 LOC refactor across role_keys.rs + content_fp.rs + markov_bundle.rs + trajectory.rs

`Vsa10k = [u64; 157]` introduced in `role_keys.rs` is a standalone
type alias that doesn't interoperate with the four existing vector
representations (`Binary16K [u64; 256]`, `Vsa10kI8 [i8; 10_000]`,
`Vsa10kF32 [f32; 10_000]`, DeepNSM `VsaVec [u64; 8]`).

**Blast radius (5 disconnection points):**
1. ContextChain operates on Binary16K (256 words). Trajectory uses
   Vsa10k (157 words). Step 8 KL-feedback blocked by type mismatch.
2. EpisodicMemory stores CrystalFingerprint. No Vsa10k conversion.
3. crystal::fingerprint has f32 VSA algebra. role_keys has bitpacked
   VSA algebra. Two parallel surfaces, incompatible types.
4. 157 words not SIMD-aligned (debt entry says 160).
5. Prior debt says rename to 16K + re-scale slices to [0..16384).
   This session went opposite (kept 10K).

**Resolution recommendation:** Option (A) — adopt Binary16K (256
words) as THE bitpacked carrier. Re-scale role-key slices to
[0..16384) proportionally. Eliminates the type mismatch, aligns
with existing debt entry, ContextChain/EpisodicMemory interop free.

Cross-ref: debt "FP_WORDS = 157 (not 160)", debt "VSA substrate
renaming: Vsa10k* → Vsa16k*", `role_keys.rs:41`.

---

## 2026-04-19 — Contract `ContextChain::coherence_at` returns 0 for non-Binary16K variants
**Status:** Open
**Priority:** P2
**Scope:** @resonance-cartographer @container-architect D4 domain:grammar
**Introduced by:** PR #210
**Payoff estimate:** ~80 LOC + tests

D4 shipped with Hamming-based coherence on the `Binary16K` variant
only; other `CrystalFingerprint` variants (`Structured5x5`,
`Vsa10kI8`, `Vsa10kF32`) return 0 as a zero-dep fallback. Cosine
coherence on the f32 variants would unlock richer disambiguation
but requires adding a minimal linear-algebra shim without breaking
the zero-dep invariant of the contract.

Cross-ref: `crates/lance-graph-contract/src/grammar/context_chain.rs`.

## 2026-04-19 — CausalityFlow has 3/9 TEKAMOLO slots; modal/local/instrument + beneficiary/goal/source deferred
**Status:** Open
**Priority:** P1
**Scope:** @integration-lead @truth-architect D1 domain:grammar
**Introduced by:** PR #208 + #210 (deliberate deferral)
**Payoff estimate:** 6 new `Option<String>` fields in
`lance-graph-cognitive/src/grammar/causality.rs` + tests

Full thematic-role inventory needs 6 more slots on `CausalityFlow`.
Deferred per user decision; D2 ticket emission and D3 triangle
bridge map only the 3 existing slots for now. Phase 2 work is
consistent with 3/9; Phase 3 may benefit from the extension.

Cross-ref: `grammar-landscape.md` §3, `STATUS_BOARD.md` D1 row.

## 2026-04-19 — Named-Entity pre-pass (NER) is the biggest OSINT blocker; stubbed out
**Status:** Open
**Introduced by:** architectural choice (all PRs)
**Payoff estimate:** dedicated PR, ~800 LOC new crate / subsystem

COCA 4096 has zero coverage of proper nouns (Altman, Anthropic,
Riyadh). Every unknown entity falls through to hash-bucket
collisions in the SPO graph. Grammar work proceeds without NER;
OSINT pipeline is blocked on this.

Cross-ref: `grammar-tiered-routing.md` §C8,
`STATUS_BOARD.md` Research threads section.

## 2026-04-19 — FP_WORDS = 157 (not 160); SIMD remainder loops remain
**Status:** Open
**Introduced by:** architectural choice (ndarray::hpc::vsa)
**Payoff estimate:** coordinated ndarray + lance-graph change,
~30 LOC in ndarray + 0 in lance-graph if field naming is stable

160 u64 = 10,240 bits (SIMD-clean for AVX-512 / AVX2 / NEON), zero
remainder loop in every SIMD pass. Current 157 u64 has 5-word
scalar tail. Performance delta is measurable but not critical yet.

Cross-ref: `cross-repo-harvest-2026-04-19.md` H6.

## 2026-04-19 — Abduction-threshold for unbundle-to-graph is hand-picked (0.88)
**Status:** Open
**Introduced by:** #208 (inherits PR design)
**Payoff estimate:** empirical calibration run on real corpus +
~30 LOC threshold parameterization

NARS Abduction confidence threshold for promoting facts into the
triplet graph is hand-picked at 0.88. Miscalibration on a specific
corpus (e.g. Animal Farm) would compound errors. Calibration is
pending D10 validation harness.

Cross-ref: `integration-plan-grammar-crystal-arigraph.md` E8,
`STATUS_BOARD.md` D10 row.

---

## Paid Debt

(No debt paid at initial commit. When an Open entry is retired,
APPEND here with same title + PR anchor.)

```
## YYYY-MM-DD — <same title as Open entry> (from YYYY-MM-DD)
**Status:** Paid YYYY-MM-DD
**Payoff:** PR #NNN (commit SHA) — <one-line description>

<verbatim original Open paragraph>

Cross-ref: <same + PR link>
```

---

## How to use this file

**When shipping with a known shortcut** — prepend to **Open Debt**
with `**Status:** Open` + `**Introduced by:** PR #NNN` +
`**Payoff estimate:**`. One paragraph describing what's owed.

**When paying debt** — append to **Paid Debt** with the same title
+ date anchor + `**Status:** Paid YYYY-MM-DD` + `**Payoff:** PR
#NNN`. Flip the Open entry's Status to `Paid YYYY-MM-DD`.

**When debt becomes irrelevant** (e.g. the feature it blocked got
abandoned) — flip Open Status to `Moot YYYY-MM-DD`. Keep the row.

Nothing is lost. Every shortcut has a trail from introduction to
payoff (or abandonment).

## 2026-04-19 — VSA substrate renaming: Vsa10k* → Vsa16k* + float framing
**Status:** Open
**Priority:** P1
**Scope:** @container-architect @truth-architect D6 D0 domain:vsa domain:codec
**Introduced by:** architectural choice (all PRs referring to "10K VSA")
**Payoff estimate:** 1 contract-crate PR + 1 audit-sweep PR across
~28 `.claude/*.md` files (21 lance-graph + 7 ndarray).

The VSA substrate is misnamed and mis-scaled throughout the workspace:

1. **Naming:** `Vsa10kF32`, `Vsa10kI8` in
   `lance-graph-contract::crystal::CrystalFingerprint` should be
   `Vsa16kF32`, `Vsa16kI8`.
2. **Scale:** 10,000-D is a legacy narrower width; move to 16,384-D
   (64 KB lossless f32, 32 KB BF16, 80 KB u8-5-lane, 160 KB BF16-5-lane).
3. **Role-key slices:** `grammar::role_keys` addresses `[0..10000)`;
   re-scale to `[0..16384)` proportionally, keeping slices disjoint.
4. **Framing:** eliminate every occurrence of "10,000 binary VSA" — it
   collapses the 2 KB Hamming fingerprint (Container) with the
   ≥64 KB float VSA substrate. They are different objects.

Cross-ref: `.claude/board/IDEAS.md` CORRECTION-OF entry (2026-04-19).
Audit list of 28 files affected: `grep -rn "10000\|10,000\|Vsa10k\|10 000-D" .claude/`.

## 2026-04-19 — Ladybug 10000-D VSA import caused 700-1100 MB memory blowup
**Status:** Open
**Priority:** P1
**Scope:** @container-architect @integration-lead domain:vsa domain:memory
**Introduced by:** PRs #200-#203 (ladybug-rs / bighorn imports —
cognitive crate + CognitiveShader + BindSpace + adaptive codecs)
**Payoff estimate:** LanceDB zero-copy mmap migration + sparse/lazy
materialization audit + VSA scale decision (10k → 16k adds ~60% per
row: 40 KB → 64 KB at f32, worse memory not better).

Observed (per user, 2026-04-19): importing ladybug-rs at 10,000-D VSA
pushed runtime memory to **700-1,100 MB**. Arithmetic: at 40 KB/row
(Vsa10kF32), this is ~17-27 K live fingerprints in RAM
simultaneously. Pathologies:

1. **Migrating to 16,384-D makes it worse** — 64 KB/row × same
   population = 1.1-1.8 GB. 5-lane encoding (80 KB u8, 160 KB BF16) is
   worse still. Wider substrate without a storage-layer fix inflates
   the problem.

2. **Storage-layer fix is mandatory** — LanceDB `FixedSizeList<Float32,
   N>` natively supports mmap zero-copy. Hot rows stay OS-cached; cold
   rows pay page-fault but not RAM. Target runtime memory: ≤ 100 MB
   working set regardless of row count.

3. **Population audit** — before the 16k rename, audit which consumers
   keep VSA fingerprints live in RAM vs fetch-on-demand. Working set
   should be bounded by cache policy, not row count. Candidates for
   fetch-on-demand: Markov ±5 trajectory (rarely revisited), AriGraph
   episodic (LRU-evictable), VSA table snapshots.

4. **Sparse representation gap** — most VSA role-bundles have a few
   active slices at a time. A sparse encoding (indices-of-nonzero or
   Structured5x5 middle cells only) could drop per-row to a few KB for
   the common case. Only the "full field" queries need the dense
   f32 representation.

**Gate before 16k rename PR:** measure peak RAM on a representative
workload (Animal Farm D10 corpus when it lands) with the current 10k
substrate. If the fix is "mmap only hot rows", the substrate switch
is safe. If the fix requires sparse representation, the rename needs
to redesign the storage contract.

**Cross-ref:** IDEAS CORRECTION-OF + REFINEMENT-OF entries
(2026-04-19). PRs #200-#203 introduced the memory footprint.

## 2026-04-19 — CORRECTION-OF 2026-04-19 Ladybug 700-1100 MB blowup — it's a 10k × 10k GLITCH MATRIX, not HDC population
**Status:** Open
**Priority:** P0
**Scope:** @container-architect @integration-lead domain:memory domain:cleanup
**Introduced by:** PRs #200-#203 (ladybug-rs / bighorn imports — carried
over code that allocates a dense 10,000 × 10,000 structure)
**Payoff estimate:** identify + delete the single glitch allocation.
No migration, no redesign, no substrate change.

The prior entry framed the 700-1,100 MB blowup as a per-row HDC
population cost (17-27 K live fingerprints at 40 KB/row). **This was
wrong.** Per user (2026-04-19):

**There is no 10,000 × 10,000 matrix we actually want.** The memory
blowup came from a dense 10k × 10k structure imported as a glitch
from outdated ladybug-rs / bighorn code. Arithmetic:

- 10,000 × 10,000 × f32 = **400 MB** (single allocation)
- Plus cognitive-stack state + other working memory → 700-1,100 MB.

**Revised fix:**

1. **Identify the glitch** — grep ladybug-imported modules (cognitive
   crate, CognitiveShader, BindSpace, CollapseGate, adaptive codecs)
   for any dense `[[T; 10000]; 10000]`, `Vec<Vec<T>>` of that shape,
   `FixedSizeList<T, 100000000>`, or 400 MB-scale buffer. High-probability
   candidates: a token-token distance matrix, co-occurrence matrix, dense
   attention matrix, or K=10000 CLAM centroid distance table that was
   imported intact without trimming to the workspace's actual scale.
2. **Delete it.** It has no consumer in lance-graph proper (grammar /
   crystal / quantum use 1-D 10k-width HDC vectors, never square
   matrices).
3. **Verify** with peak-RAM measurement on a minimal workload.

**Invalidates:**

- Prior "16k rename makes memory worse" analysis — the per-row HDC
  math was sound but irrelevant to this specific blowup.
- Mmap zero-copy requirement for HDC population — still good hygiene
  but not the fix for the 700-1,100 MB observation.
- Sparse encoding as a Structured5x5-alternative — still architecturally
  useful but unrelated to the glitch.

The 16k rename + f32 → BF16 migration proceed independently of this
debt item. This one is a one-shot deletion.

## 2026-04-19 — SUPERSEDES all "stoneage import" / ladybug-refactor entries — retire ladybug-rs entirely
**Status:** CLOSED (via architectural decision)
**Scope:** @integration-lead @workspace-primer domain:architecture
**Decision:** per user (2026-04-19), ladybug-rs is retired. Migration
target is **ada-rs + lance-graph exclusively**. Ladybug-rs becomes
read-only historical reference; no maintenance, no refactor, no
integration obligation. Harvest-only.

**Consequences (ledger cleanup):**

- "Ladybug 700-1100 MB memory blowup" (glitch matrix): the matrix
  lives in ladybug-import code that is itself scheduled for removal.
  Deletion still P0 **only if** it ends up compiled into the current
  ada-rs / lance-graph binaries; otherwise it goes away with the
  import archival. Downgrade to P2, gate on "does cargo tree show the
  ladybug dep?"
- "Vsa10k → Vsa16k rename sweep" (2026-04-19): scope tightens to
  **ada-rs + lance-graph only**. Ladybug's internal types don't get
  renamed — they get left alone in the archive.
- "Ladybug import refactor resistance" table: obsolete. Don't refactor
  ladybug code; if a pattern is useful (PhaseTag, BindSpace,
  CognitiveShader, adaptive codec), reimplement cleanly in the target
  repo against canonical `Fingerprint<256>`.
- `CLAUDE.md` architecture diagram: "ladybug-rs = The Brain" line is
  stale; ada-rs + lance-graph now carry both the Brain and the Spine.
  Update in a follow-up PR.

**Repos in the canonical stack (post-retirement):**

```
ndarray            = The Foundation  (SIMD, GEMM, HPC, Fingerprint<256>, CAM-PQ)
lance-graph        = The Spine       (query + codec + semantics + contracts)
ada-rs             = The Brain       (BindSpace, SPO server, 4096 surface, cognitive shader)
crewai-rust        = The Agents      (agent orchestration, thinking styles)
n8n-rs             = The Orchestrator (workflow DAG, step routing)
```

(Was 5 repos; still 5, with ada-rs taking ladybug-rs's Brain slot.)

**Cross-ref:** prior entries "Ladybug 700-1100 MB memory blowup",
"Vsa10k* → Vsa16k* rename sweep", "CORRECTION-OF ... 10k × 10k GLITCH
MATRIX" — all carry this decision as their closing context.

## 2026-04-21 — D5 deepnsm files built on wrong VSA substrate (Frankenstein revert needed)
**Status:** Open
**Priority:** P0
**Scope:** @truth-architect @container-architect D5 domain:vsa domain:grammar
**Introduced by:** PR #243 (this session)
**Payoff estimate:** ~200 LOC rewrite — revert four files + reimplement on Vsa16kF32

Three deepnsm files and one contract addition built on `Vsa10k = [u64; 157]`
bitpacked binary + XOR algebra. Correct substrate is `Vsa16kF32 = Box<[f32;
16_384]>` (pending Vsa10k→Vsa16k coordinated rescale) with element-wise
multiply/add (`vsa_bind`/`vsa_bundle`/`vsa_cosine` that already exist in
`crystal/fingerprint.rs`).

Files to rewrite:

1. `crates/deepnsm/src/content_fp.rs` — produce `Box<[f32; 16_384]>`
   bipolar fingerprints (sign bit from SplitMix64), not `[u64; 157]`.
2. `crates/deepnsm/src/markov_bundle.rs` — use `vsa_bundle` (add) not
   `vsa_xor`. Braiding via `vsa_permute` on f32 indices, not bit rotation.
3. `crates/deepnsm/src/trajectory.rs` — `free_energy` likelihood term uses
   `vsa_cosine(unbind, expected)` per role, not Hamming recovery margin.
4. `crates/lance-graph-contract/src/grammar/role_keys.rs` — DELETE
   `RoleKey::bind/unbind/recovery_margin` (algebra belongs on carrier).
   DELETE `vsa_xor`, `vsa_similarity`. DELETE `Vsa10k` type alias.
   Keep role key static definitions but retype as `Box<[f32; 16_384]>`
   bipolar values (sign bit from FNV-seeded SplitMix64, ±1 in slice,
   zero elsewhere).

Dependency: blocks on the Vsa10k→Vsa16k coordinated rescale PR (ndarray
+ contract). Until that lands, interim fix is to use existing `Vsa10kF32`
(10,000 dims) without renaming.

Cross-ref: `.claude/knowledge/vsa-switchboard-architecture.md` (full
three-layer architecture), `EPIPHANIES.md` 2026-04-21 CORRECTION-OF entry,
audit agent report 2026-04-21.

## 2026-04-21 — Vsa10k→Vsa16k coordinated rescale (ndarray + lance-graph-contract)
**Status:** Open
**Priority:** P1
**Scope:** @container-architect @truth-architect D6 D0 domain:vsa domain:codec cross-repo:ndarray
**Introduced by:** architectural choice (originally flagged 2026-04-19 P1 debt, now coordinated two-repo scope)
**Payoff estimate:** 2 coordinated PRs (ndarray + contract) — ~40 LOC ndarray,
~200 LOC contract (rename + rescale + sandwich constants + passthrough funcs)

Rescale the float VSA carrier from 10,000 dims to 16,384 dims. This:
- Makes Binary16K ↔ Vsa16kF32 passthrough 1:1 (currently 16K bits vs 10K dims)
- Gives +60% capacity for orthogonal role superposition (Johnson-Lindenstrauss)
- Allows power-of-2 role slice boundaries (SUBJECT[0..4096), etc.)
- Aligns float VSA dim with Binary16K bit width

NOT a SIMD alignment issue for floats — both 10,000 and 16,384 f32 are
AVX-512-clean. The 157→160 u64 alignment issue was a BINARY-format-only
concern.

ndarray side: `VSA_DIMS = 16_384`, `VSA_WORDS = 256`, rescale VsaVector,
vsa_permute, vsa_bundle, VsaAccumulator.

contract side: rename `Vsa10kF32/I8` → `Vsa16kF32/I8`, rescale Box array
sizes, update sandwich constants (SANDWICH_TAIL_END = 16_384), update
passthrough funcs (to_vsa10k_f32 → to_vsa16k_f32, etc.), update role-key
slice boundaries proportionally.

Cross-ref: `cross-repo-harvest-2026-04-19.md` H6, prior tech debt entry
from 2026-04-19.

## 2026-04-21 — Jirak-derived thresholds probe (replace hand-tuned σ constants)
**Status:** Open
**Priority:** P1
**Scope:** @truth-architect @probe-runner D7 D10 domain:vsa domain:stats
**Introduced by:** CLAUDE.md `I-NOISE-FLOOR-JIRAK` iron rule
**Payoff estimate:** ~100 LOC in contract crate (jirak_threshold function)
+ one-shot calibration probe on Animal Farm to measure ρ

Hand-picked constants:
- `UNBUNDLE_HARDNESS_THRESHOLD = 0.8` (PR #208)
- `UNBUNDLE_ABDUCTION_THRESHOLD = 0.88` (this session, D7)
- `HOMEOSTASIS_FLOOR = 0.2`, `EPIPHANY_MARGIN = 0.05`, `FAILURE_CEILING = 0.8`
  (this session, free_energy.rs)

All hand-tuned. Should become functions of measured bit-level weak
dependence ρ per Jirak 2016 (arxiv 1606.01617).

Probe:
1. Measure ρ on actual Binary16K fingerprint distribution (autocorrelation
   of bit i with bit j across a representative corpus).
2. Implement `jirak_threshold(n: usize, rho: f32, confidence: f32) -> f32`.
3. Replace hand-picked constants with calls to this function.
4. First-run calibration: Animal Farm corpus for the grammar/coref pipeline.

Ship this BEFORE claiming the stack is "grounded." Currently it's
"tuned," which is defensible but must be documented honestly.

Cross-ref: CLAUDE.md `I-NOISE-FLOOR-JIRAK`, `.claude/jc/examples/prove_it.rs`
(3/5 pillars already run in 5s).

## 2026-04-21 — ONNX 16kbit story-arc learning (D9 deferred)
**Status:** Open
**Priority:** P2
**Scope:** @truth-architect @onnx-bridge D9 domain:arc domain:learning
**Introduced by:** `categorical-algebraic-inference-v1.md` D9 architectural
placeholder
**Payoff estimate:** new crate or bridge module (~300 LOC) + ONNX model
training pipeline (out of scope for lance-graph; upstream in thinking-engine)

The 3x16kbit Plane accumulator (ndarray `plane.rs`, shipped in
CROSS_REPO_AUDIT_2026_04_01.md) is the write-path for AriGraph edge
learning. It is NOT a VSA carrier — it's a saturating i8 accumulator.

D9 ONNX arc export should consume Vsa16kF32 trajectory fingerprints
(identity layer) and predict state transitions. The model is trained
on (state, arc_pressure, arc_derivative) tuples emitted per cycle.

Deferred until:
1. D5 rewrite (correct VSA substrate)
2. D8 AriGraph commit bridge
3. D10 Animal Farm benchmark running

Then ONNX training has a corpus to consume.

Cross-ref: `categorical-algebraic-inference-v1.md` §4 Next deferred,
ndarray `plane.rs`.

## 2026-04-21 — Callcenter / persona / archetype catalogues (speculative intent preservation)
**Status:** Open
**Priority:** P3 (tracked but low)
**Scope:** @host-glove-designer @truth-architect domain:persona domain:callcenter
**Introduced by:** architectural discussion 2026-04-21
**Payoff estimate:** deferred — not yet greenlit as deliverables

The three-layer VSA architecture explicitly supports callcenter intent
classification, persona-agent routing, and archetype-based character
inference. None of these are shipped features; they are SPECULATIVE
applications of the switchboard pattern.

Intent preservation (for future planning sessions):
1. Callcenter agents as persona-registry consumers (each agent role =
   one persona identity fingerprint).
2. Intent classification via VSA resonance (caller turn → role bundle
   → cosine-rank against intent codebook → dispatch).
3. Supabase as content layer (NOT VSA layer) — stores persona YAML,
   intent definitions, call transcripts, cumulative awareness. Supabase
   never stores VSA bundles (those are ephemeral compute state).
4. "BBB" references in prior conversations = speculative benchmarks
   (Better Business Bureau complaint handling), not shipped features.
5. Archetype catalogue unification with existing palette archetypes
   (256 per plane in bgz17), VoiceArchetype (16 channels in ndarray::
   hpc::audio), Glyph5B (5-byte archetype addressing).

Do NOT ship without explicit green-light. The architecture supports it;
the current scope doesn't include it. Preserve the framing for when
these become tracked deliverables.

Cross-ref: `.claude/knowledge/vsa-switchboard-architecture.md` § The
Archetype ↔ AriGraph ↔ Persona ↔ ThinkingStyle Unification.


## 2026-04-21 — L3 naming collision: CPU cache L3 vs cognitive-shader Layer 3
**Status:** Open
**Priority:** P2
**Scope:** @truth-architect @integration-lead domain:docs domain:cognitive-shader
**Introduced by:** architectural naming (pre-existing, flagged 2026-04-21)
**Payoff estimate:** ~40 LOC docs cleanup + grep pass over `.claude/*.md`

The 7-layer cognitive stack uses L0..L6 labels (L0 ndarray SIMD, L1
BindSpace, L2 CognitiveShader, L3 CollapseGate, L4 Planner, L5 GPU
meta, L6 LanceDB). This NAMING COLLIDES with CPU cache levels (L1/L2/
L3 caches).

The confusion point: when memory analysis says "Vsa16kF32 (64 KB)
fits in L3 cache" — does L3 mean cache level 3 (typical 8-32 MB, plenty
of room) or cognitive-shader Layer 3 (CollapseGate, a logical dispatch
layer unrelated to physical memory)?

The poisoning risk: documentation or session handovers that casually
reference "L3" may be read by a future session as either (a) CPU
cache level or (b) cognitive-shader layer, and the wrong reading can
lead to wrong design decisions (e.g., sizing the trajectory bundle to
"fit in L3" when "L3" was actually referring to the collapse gate).

**Fix:** Disambiguate in all future writing:
- CPU cache → "CPU-L3" or "L3-cache"
- Cognitive stack layer → "cog-L3" or "Layer 3 (CollapseGate)"
- Grep pass over existing `.claude/*.md` for "L3" with implicit
  context; annotate each occurrence.

Cross-ref: `CLAUDE.md § What This Is` (7-layer stack diagram),
this session's memory analysis discussion of Vsa16kF32 cache fit.


## 2026-04-21 — Lazy-VSA principle (reclassification of queued VSA work)

**Status:** Open (architectural guidance, applies to all VSA items below)
**Priority:** P1 (guidance)
**Scope:** @integration-lead all domain:vsa entries
**Introduced by:** user direction 2026-04-21 post-cleanup

**Principle:** VSA substrate work is PULLED IN by downstream deliverables,
not PUSHED OUT speculatively. Specifically:

1. **Vsa10k→Vsa16k coordinated rescale** — POSTPONED. Deserves its own
   dedicated test session. Cross-repo (ndarray + contract), non-trivial
   calibration impact, merits focused planning. Do NOT bundle with D5
   rewrite or any other in-flight work.

2. **D5 rewrite on Vsa10kF32 (current 10K)** — pull in ONLY IF a
   downstream deliverable (steps 4–8 wiring, D8 AriGraph bridge, D10
   Animal Farm benchmark) concretely needs the VSA trajectory and the
   Vsa16k rescale hasn't landed yet. Until that trigger fires,
   `Vsa10kF32` stays unused by the grammar/persona/callcenter
   consumers. The cleanup has reverted to pre-D5 state — that's the
   terminal state for this session.

3. **Jirak-derived thresholds probe** — pull in ONLY IF a downstream
   deliverable uses the thresholds (currently only `UNBUNDLE_HARDNESS`
   and `ABDUCTION_THRESHOLD` in shipped code; both hand-tuned are
   defensible). First consumer to need calibrated thresholds triggers
   the probe.

   **HOWEVER:** Jirak's ACTIVE role right now (NOT deferred) is the
   scientific framework for **CAM-PQ vs Vsa10k format decisions**.
   `FormatBestPractices.md § 1` quantifies the weak-dependence ρ that
   makes CAM-PQ bitpacked codes POOR candidates for VSA bundling
   (ρ ≈ 0.3–0.5 after centroid quantization; effective capacity drops
   proportionally) vs near-IID role-key-generated bits (ρ ≈ 0.01)
   where `Vsa10kF32` bundling retains full capacity. This is Jirak
   as **DECISION FRAMEWORK**, not Jirak as CALIBRATION PROBE. The
   decision framework is already shipped in the cleanup docs; the
   probe is what measures specific ρ per corpus to replace the
   hand-tuned thresholds when a downstream consumer demands it.

**Why lazy:** every VSA substrate touch has calibration ripple effects.
Doing them in the order "substrate first, then consumers" means calibrating
substrate changes without a real consumer to test against — which is how
the D5 Frankenstein happened. Inverting to "consumer first, pull substrate
when needed" grounds every substrate change in a concrete benchmark.

**Action implication:** next session doesn't start with "rewrite D5 on
Vsa10kF32." Next session starts with "what's the FIRST downstream
deliverable that forces a VSA substrate decision, and what does that
decision need?" If the answer is D10 Animal Farm, then D5+D8+D10 plan
together. If the answer is persona bank for callcenter, D5+persona
catalogue. Etc.

Cross-ref: `FormatBestPractices.md` § 0 (the 5-question checklist
before picking a format), CHANGELOG.md 2026-04-21 entries.

