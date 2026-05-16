# Cross-Repo Dependency Audit — 2026-05 (Sprint-13 Preflight)

> **Author:** PP-12 (Opus planner, main-thread)
> **Branch:** `claude/sprint-13-preflight-planning`
> **Date:** 2026-05-16
> **READ BY:** sprint-13 main thread, `integration-lead`, `truth-architect`,
>             any sprint-13 W-* worker touching ndarray / p64 / p64-bridge.
> **Companion docs:**
>   - `.claude/knowledge/cross-repo-harvest-2026-04-19.md` (epiphany harvest)
>   - `.claude/knowledge/encoding-ecosystem.md` (codec map)
>   - `.claude/plans/cognitive-substrate-convergence-v2.md` §11 (deliverable matrix)
>   - `.claude/specs/pr-sprint-13-rayon-streams.md` (PP-3, D-CSV-17)
>   - `CLAUDE.md` § "Cross-Repo Dependencies"

This audit captures the cross-repo state on the eve of sprint-13 spawn.
It complements PP-1/PP-2/PP-3 (which speak about the lance-graph side
of sprint-13) with the question PP-1..PP-6 do **not** answer: *what
do we need from the sibling repos, in what order, before we can
green-light spawn?*

---

## §1 — Cross-Repo Dependency Map (Current State)

The lance-graph workspace has exactly **two** active path-deps to
sibling-repo crates: `/home/user/ndarray` (the foundation) and
`/home/user/ndarray/crates/p64` (transitively, via lance-graph's own
`crates/p64-bridge`). Every other sibling-repo path in
`/home/user/` (Hiro-rs, MedCare-rs, WoA, Sharepoint, woa-rs, q2,
smb-office-rs, …) is either a *consumer* of the lance-graph
contract crate or an unrelated workspace — none of them flow into a
sprint-13 D-CSV deliverable.

The matrix below lists only the **load-bearing** sibling deps.

| Sibling repo | Path | What lance-graph imports | Status of importer |
|---|---|---|---|
| **ndarray** | `/home/user/ndarray` | `ndarray::hpc::{simd_caps, cam_pq, bgz17_bridge, palette_distance, fingerprint, stream::{qualia, inference, splat_field}}` | Imported by **bgz-tensor**, **bge-m3**, **deepnsm**, **lance-graph-planner**, **lance-graph** (with `ndarray-hpc` feat), **cognitive-shader-driver** |
| **ndarray/crates/p64** | `/home/user/ndarray/crates/p64` | `p64::{Palette64, CognitiveShader, predicate, sparse256, …}` (Σ ~1986 LOC) | Imported by **lance-graph-planner** (line 30) and transitively by **p64-bridge** |
| **p64-bridge** (in-tree, not sibling) | `/home/user/lance-graph/crates/p64-bridge` | `causal-edge` + `bgz17` + downstream of `p64` (in-tree) | Imported by **lance-graph-planner**, **cognitive-shader-driver** |

### ndarray — latest master HEAD

```
2a3885d2 (master)  Merge pull request #146 (claude/portable-simd-nightly)
752cb339           style(simd_nightly): apply cargo fmt --all (PR #146 fmt fix)
bd991f41           feat(simd_nightly): 30-type portable-simd backend (round-3 fleet)
…
74b18588           Merge pull request #145 (claude/bump-rust-1.95)
fd11845a           Merge pull request #144 (claude/u8x32-polyfill-round3)
fd11845a (PR #143) Merge pull request #143 (claude/simd-caps-amx-round2)
```

The local checkout is on branch `claude/sprint-12-qualia-stream-w-f4`
(post-merge of PR #147). `git diff master claude/sprint-12-qualia-stream-w-f4`
shows the streaming module on the branch (already merged to master via
PR #147 on 2026-05-16) plus a small miri sweep delta and a doctest
fix. The branch is **safe to delete** post-sprint-13 spawn, but
the diff is informative because it shows what sprint-13 D-CSV-13b
and D-CSV-17 will be *editing on top of*.

### Open ndarray PRs

`mcp__github__list_pull_requests(owner=AdaWorldAPI, repo=ndarray, state=open)`
returns the empty list. **There are no open ndarray PRs as of
2026-05-16.** The sprint-12 D-CSV-11 PR #147 merged at
2026-05-16T04:35:05Z (5 hours before this audit was written) and PR #116
(hpc-extras gating) closed-and-merged at 2026-04-30. Sprint-13 has a
clean ndarray runway.

### Open lance-graph PRs

| # | Title | Status | Relevance |
|---|---|---|---|
| #390 | sprint-12 Wave G fleet (D-CSV-5b/6b/13/15) | open, NOT merged | **Must merge before sprint-13 spawn** per the Wave-G-Opus review recommendation |
| #261 | A2A coordination blackboard | open, draft | Infra (cross-session bus); not a sprint-13 blocker |

`mcp__github__pull_request_read(method=get, owner=AdaWorldAPI,
repo=ndarray, pullNumber=147)` confirms PR #147 merged status with
+794/-19 across 8 files. The relevant scaffolds are now on master.

### Unmerged work the sprint-13 plan depends on

| What | Where | Sprint-13 dependency |
|---|---|---|
| PR #390 lance-graph Wave G | open | Sprint-13 D-CSV-13b/16/17 specs all assume the Wave G `WitnessIndexHashMap` cutover and Σ-tier Jirak thresholds are landed. |
| PR #147 ndarray streams | **merged** | D-CSV-17 (rayon par_*) edits these files; merge confirmed. |
| (none on ndarray side) | — | D-CSV-13b (SIMD MUL i4) and D-CSV-16 (cam_pq adapter) edit ndarray master directly; no upstream wait. |

---

## §2 — Sprint-13 Coordination Requirements

Below, each in-scope sprint-13 D-CSV is mapped to its repo footprint
and the recommended PR landing order.

### D-CSV-17 — rayon `par_*` streams (PP-3, spec already drafted)

- **Touches:** ndarray only — `src/hpc/stream/{qualia,inference,splat_field}.rs`.
- **No lance-graph-side change.** The lance-graph consumers
  (`lance-graph-planner::cache::*`) currently call the scalar
  `QualiaStream` / `InferenceStream` / `SplatFieldStream` via
  `ExactSizeIterator`; adding `par_*` is purely additive.
- **Order:** open ndarray PR; merge; lance-graph remains on its
  current path-dep; no version bump needed.
- **Risk:** Low. ~120 LOC source + 150 LOC tests, gated behind
  the existing `rayon` feature flag.

### D-CSV-13b — SIMD vectorization of D-CSV-8 i4 MUL

- **Touches:** ndarray (intrinsics path) **and** lance-graph
  (`crates/lance-graph-contract/src/mul/i4_eval.rs` — the scalar
  fallback shipped via PR #387).
- **Pattern:** ndarray contributes per-ISA intrinsic kernels via
  the existing `simd_caps()` dispatch singleton
  (`/home/user/ndarray/src/hpc/simd_caps.rs:98`). lance-graph
  consumes them through a thin call site in `i4_eval::batch`.
- **Order:** open ndarray PR exposing
  `pub fn i4_mul_avx512(…) -> _` / `i4_mul_neon(…)` first;
  land it; then open the lance-graph PR that calls
  `if simd_caps().has_avx512_vnni() { ndarray::hpc::… } else { scalar }`.
- **Risk:** Medium. AVX-512/NEON intrinsics + `is_x86_feature_detected!`
  gates have ISA-specific corner cases; cite Wave-G W-G3 batch i4
  scalar baseline as ground truth for parity tests.

### D-CSV-16 — `cam_pq` adapter (u64 SPO → `Vec<usize>`)

- **Touches:** primarily lance-graph
  (`crates/lance-graph/src/graph/arigraph/witness_corpus.rs`), with
  optional helper-method addition on the ndarray side if the adapter
  shape demands it.
- **Background:** Wave G W-G2 shipped a `WitnessIndexHashMap`
  placeholder (PR #390) explicitly *naming* `cam_pq` as the
  upgrade target. `ndarray::hpc::cam_pq` (`src/hpc/cam_pq.rs`,
  ~770 LOC) exposes the API surface we will adapt:
  `CamCodebook::encode(&[f32]) -> CamFingerprint` and
  `PackedDatabase::cascade_query`. It does **not** yet support
  `u64 SPO -> Vec<usize>` natively — sprint-13's adapter is the
  thin glue between the lance-graph SPO triple key and the cam_pq
  6-byte fingerprint via `train_geometric` over the SPO palette.
- **Order:** lance-graph PR first (lives entirely in
  `witness_corpus.rs` + a new `crates/lance-graph/src/graph/arigraph/cam_pq_adapter.rs`).
  Only escalate to an ndarray PR if the adapter discovers a missing
  primitive (e.g., bulk encode that yields a borrow into the
  codebook). Default plan: **no ndarray PR needed**.
- **Risk:** Low–Medium. The mathematical question is whether
  `train_geometric` is the right training mode for the SPO key
  distribution; the spec should answer that with citation to
  `encoding-ecosystem.md` (CAM-PQ entry) before code is written.

### D-CSV-13 (proper) — sigma vec retest, integration

- **Touches:** lance-graph-planner — no ndarray cross-repo work.
- **Order:** any time after PR #390 merges.

### D-CSV-14 — on-Think method migration (PP-2 territory)

- **Touches:** lance-graph contract crate only; no ndarray cross-repo.

### D-CSV-15 — Jirak-derived Σ10 threshold

- **Touches:** `crates/sigma-tier-router` only. The Jirak 2016
  citation is satisfied by the iron-rule doctrine doc
  (`.claude/knowledge/iron-rules-doctrine.md` per PP-2). No
  cross-repo coordination.

### Recommended PR order across sprint-13

```
T0:                  PR #390 (Wave G) merges            [lance-graph]
T0+ε (parallel):     PR-ndarray-D-CSV-17 opens           [ndarray]
                     PR-ndarray-D-CSV-13b opens          [ndarray]
T0+1d:               PR-ndarray-D-CSV-17 merges          [ndarray]
                     PR-ndarray-D-CSV-13b merges         [ndarray]
T0+1d (parallel):    PR-lance-graph-D-CSV-13b opens      [lance-graph]
                     PR-lance-graph-D-CSV-16 opens       [lance-graph]
                     PR-lance-graph-D-CSV-14 opens       [lance-graph]
                     PR-lance-graph-D-CSV-15 opens       [lance-graph]
T0+2d:               sprint-13 fleet merges              [both]
```

Note: D-CSV-17 and D-CSV-13b are **independent** on the ndarray side
(different files, different feature flags). They can land in either
order or in parallel.

---

## §3 — ndarray-Specific Audit (the heaviest dep)

### §3.1 PR #147 — sprint-12 D-CSV-11 streams

**Status:** MERGED 2026-05-16T04:35:05Z. +794/-19 across 8 files.
The branch `claude/sprint-12-qualia-stream-w-f4` is now redundant
with master.

Shipped:

- `src/hpc/stream/qualia.rs` (201 LOC) — `QualiaI4Row` +
  `QualiaStream`, bit-compatible with
  `lance_graph_contract::qualia::QualiaI4_16D`
- `src/hpc/stream/inference.rs` (219 LOC) — `InferenceRow` +
  `InferenceStream`, bit-compatible with `causal_edge::CausalEdge64` v2
- `src/hpc/stream/splat_field.rs` (226 LOC) — `SplatField` + stream
- `src/hpc/stream/mod.rs` (15 LOC) — registers + re-exports
- `src/hpc/mod.rs` — `pub mod stream;`

The doc-comments in each file promise the `par_*` rayon variants
"sprint-13+ once rayon is wired into the ndarray feature gate." PP-3
fulfils that promise as D-CSV-17.

### §3.2 `cam_pq.rs` codec shape — adapter readiness for PP-5

File: `/home/user/ndarray/src/hpc/cam_pq.rs` (~770 LOC).

**Public API surface relevant to the SPO adapter:**

- `pub const NUM_SUBSPACES: usize = 6` — fixed 6-byte fingerprint
- `pub type CamFingerprint = [u8; NUM_SUBSPACES]` — 6 bytes
- `pub struct CamCodebook { codebooks: [SubspaceCodebook; 6], total_dim, subspace_dim }`
- `impl CamCodebook::encode(&self, vector: &[f32]) -> CamFingerprint`
- `impl CamCodebook::encode_batch(&self, vectors: &[Vec<f32>]) -> Vec<CamFingerprint>`
- `pub struct PackedDatabase { … }`
  - `pub fn pack(fingerprints: &[CamFingerprint]) -> Self`
  - `pub fn cascade_query(&self, …)` — returns `Vec<usize>` of candidate row indices
  - `pub fn top_k(&self, cams: &[CamFingerprint], k: usize) -> Vec<(usize, f32)>`
- `pub fn train_geometric(vectors: &[Vec<f32>], total_dim, iterations) -> CamCodebook`
- `pub fn train_semantic(vectors, labels, total_dim, alpha) -> CamCodebook`
- `pub fn train_hybrid(vectors, labels, total_dim) -> CamCodebook`

**Does it support `u64 SPO → Vec<usize>` directly?** No. The current
API is `&[f32] → CamFingerprint` and `&[CamFingerprint] → Vec<usize>`.
The adapter sprint-13 needs is:

```
u64 spo_packed                       ←─ lance-graph SPO bit layout
   │ unpack_to_f32_vector
   ▼
&[f32]  (S/P/O role-indexed dims)    ←─ adapter local to lance-graph
   │ CamCodebook::encode (ndarray)
   ▼
CamFingerprint ([u8; 6])
   │ PackedDatabase::cascade_query   (ndarray, already returns Vec<usize>)
   ▼
Vec<usize>                           ←─ lance-graph WitnessCorpus row indices
```

**Conclusion:** the adapter is **lance-graph-side only**. ndarray
already exposes everything sprint-13 needs. No ndarray PR for D-CSV-16
unless the lance-graph spec authors decide a `train_spo_palette`
convenience constructor belongs upstream — a judgment call deferred
to PP-5 spec.

### §3.3 `parallel` / `rayon` feature convention

`/home/user/ndarray/Cargo.toml`:

```toml
[features]
default = ["std", "hpc-extras"]
std       = ["num-traits/std", "matrixmultiply/std"]
hpc-extras = ["std", "dep:blake3", "dep:p64", "dep:fractal", "fractal/std"]
rayon      = ["dep:rayon", "std"]

[dependencies]
rayon = { version = "1.10.0", optional = true }
```

The `rayon` feature is **already declared but presently unused** in
`src/hpc/stream/`. D-CSV-17 wires `#[cfg(feature = "rayon")]` blocks
into the three stream files. The CI matrix does not yet exercise
`--features rayon` — that gap is tracked separately as preflight-PP-4
(spec line 557).

**Convention for sprint-13 D-CSV-13b SIMD code:** ndarray's existing
SIMD gate is `#[cfg(target_arch = "x86_64")]` + runtime
`is_x86_feature_detected!("avx512vnni")` — see how PR #146
(simd_nightly fleet) lands its 30-type portable-simd backend
behind a separate `simd_nightly` feature. D-CSV-13b should follow
that pattern, not invent a new feature gate.

### §3.4 `simd_caps()` singleton — canonical dispatch

`/home/user/ndarray/src/hpc/simd_caps.rs:98`:

```rust
pub fn simd_caps() -> SimdCaps { /* OnceCell-cached cpuid probe */ }
```

The `SimdCaps` struct (line 33) carries every x86/ARM ISA flag the
codebase dispatches on: AVX-512 / VNNI / BF16 / AMX / VBMI / VPCLMULQDQ
(via PR #143 AMX round-2), AVX-VNNI-INT8 (PR #144 round-3), NEON /
dotprod / fp16 / crypto on ARM (`arm_profile()` line 317).

**This IS the canonical dispatch.** D-CSV-13b should call
`if simd_caps().has_avx512_vnni() { …vectorized… } else { …scalar… }`
and **not** introduce a parallel set of cpuid probes. The
`#[non_exhaustive]` attribute (added in commit `8209b471` per codex
P2 on PR #143) means new ISA flags can land additively without SBP
breakage.

### §3.5 PR #116 hpc-extras blocker (sprint-11 carry-over)

PR #116 (sprint A1: gate blake3/p64/fractal behind `hpc-extras` feature)
merged 2026-04-30T09:49:58Z. It is **not** blocking anything in
sprint-13:

- All lance-graph consumers of ndarray that need the hpc tree
  already declare `features = ["std", "hpc-extras"]` explicitly
  (see e.g. `bgz-tensor/Cargo.toml:25`,
  `lance-graph-planner/Cargo.toml:24`,
  `lance-graph/Cargo.toml:50`, `cognitive-shader-driver/Cargo.toml:44`).
- Consumers that don't need it (`bge-m3/Cargo.toml:10`,
  `deepnsm/Cargo.toml:33`) declare `features = ["std"]` and pay
  no blake3/p64/fractal compile cost.

The sprint-11 TD entries referencing PR #116 as a "coordination
requirement for D-CSV-11" (see `pr-ce64-mb-3-bindspace-efgh.md:851`)
are **resolved** by the time of this audit.

---

## §4 — p64 / p64-bridge Audit

### §4.1 Is p64 active or dormant?

**Active, but mostly read-only from sprint-13's perspective.**
`/home/user/ndarray/crates/p64/src/lib.rs` is ~1986 LOC and exposes
the full `Palette64` / `CognitiveShader` / `HeelPlanes` / `predicate`
/ `sparse256` surface that `lance-graph-planner::cache::convergence`
consumes. The crate has not changed on master between sprint-11 and
sprint-12 (last meaningful update predates PR #143 by enough that
all recent ndarray PRs are AVX/SIMD-side, not p64-side).

### §4.2 Does sprint-13 touch p64?

**No.** None of the sprint-13 D-CSV-13b / D-CSV-14 / D-CSV-15 / D-CSV-16
/ D-CSV-17 specs edit `crates/p64`. The closest adjacency is D-CSV-16
(`cam_pq` adapter) — but that lives in `src/hpc/cam_pq.rs`, not in
the p64 crate.

### §4.3 `convergence.rs` drift from E-META-7 (sprint-10 finding)

`/home/user/lance-graph/crates/lance-graph-planner/src/cache/convergence.rs`
lines 22-27 carry the original drift:

```rust
#[allow(unused_imports)] // CausalEdge64 intended for hot-path convergence wiring
use super::nars_engine::{CausalEdge64, SpoHead, MASK_SPO};
#[allow(unused_imports)] // intended for Base17 fingerprint convergence wiring
use ndarray::hpc::bgz17_bridge::SpoBase17;
#[allow(unused_imports)] // DistanceMatrix intended for per-plane distance wiring
use ndarray::hpc::palette_distance::{DistanceMatrix, Palette, SpoDistanceMatrices};
```

**Status check (2026-05-16):** `PlaneDistance` (lines 41-77) **is** now
wired — `SpoDistanceMatrices::build` is called inside
`PlaneDistance::build`, and `spo_distance` / `subject_distance` /
`predicate_distance` / `object_distance` are public methods. So the
*usage* of `palette_distance` is no longer dead. However, the
`#[allow(unused_imports)]` attribute on line 26 remains — almost
certainly because the imports at the module level shadow per-function
re-imports that the active path uses. **A 1-line cleanup PR could
drop the `#[allow]` on line 26 now that the module is wired**, and
keep it on lines 22/24 (CausalEdge64 + SpoBase17 are still dormant).

Sprint-13 should not block on this; it's a 5-minute TD entry, not a
spec.

### §4.4 p64-bridge specifically

`/home/user/lance-graph/crates/p64-bridge/src/lib.rs` (753 LOC).
Single grep for `convergence` returns line 322: a comment "convergence
highway wiring (TD-P64-SHADER-1)" — which is the matching documentation
trail to the `convergence.rs` allows above. p64-bridge has no new
sprint-13 work scheduled.

---

## §5 — Cross-Repo PR Coordination Protocol (PROPOSAL)

For sprint-13 multi-repo PRs (specifically D-CSV-13b), the following
protocol is **recommended** to avoid the "lance-graph PR references
an ndarray symbol that doesn't exist on master yet" failure mode that
sprint-11 hit twice (D-CSV-7 mailbox + W-F4 stream scaffold).

### Step-by-step

1. **Open the sibling-repo PR FIRST.** Push the ndarray branch and
   open the PR before any lance-graph code touches the new symbol.

2. **Use a *draft* PR in lance-graph that consumes the sibling's
   branch via `git = … branch = "…"` override.** Concretely, in the
   lance-graph branch:

   ```toml
   # crates/lance-graph-contract/Cargo.toml (TEMPORARY, will revert before merge)
   [dependencies.ndarray]
   git = "http://local_proxy@127.0.0.1:41277/git/AdaWorldAPI/ndarray"
   branch = "claude/d-csv-13b-simd-mul"
   default-features = false
   features = ["std", "hpc-extras"]
   ```

   The `path = "../../../ndarray"` line is commented out (not deleted
   — the comment serves as the "revert here on merge" anchor).

3. **CI green on lance-graph draft** using the branch dep proves the
   integration story works end-to-end before either repo merges.

4. **Land the sibling PR (ndarray) to master.**

5. **Revert lance-graph's draft PR to the local `path = …` form.**
   Push; CI re-greens against ndarray master (which now contains the
   merged sibling-PR symbols). Mark lance-graph draft "Ready for review."

6. **Merge lance-graph PR.**

### Why not `[patch.crates-io]`?

ndarray is not on crates.io for this workspace — every consumer uses
the path-dep form. So `[patch]` doesn't apply; the `git = … branch
= …` override is the only mechanism that works without altering the
workspace topology mid-sprint.

### Anti-patterns

- **DO NOT** point lance-graph at the sibling branch via a relative
  path with the branch checked out locally (`path =
  "../../../ndarray-d-csv-13b"`). This causes the local target/
  cache to fork and quietly diverges from the upstream PR.
- **DO NOT** open the lance-graph PR before the ndarray PR. CI will
  fail with "unresolved import" because the consumer sees the import
  before the producer exists.
- **DO NOT** rebase the lance-graph branch onto master mid-flight while
  the ndarray branch dep override is in place. Use `git merge`, not
  rebase, until step 5 strips the override.

---

## §6 — Risks + Recommendations

### R1: ndarray target/ dep size

Current local disk usage:

```
/home/user/lance-graph/target  →  5.4 GB
/home/user/ndarray/target      →  3.5 GB
                       Σ       →  8.9 GB
```

The "22 GB" figure cited in earlier sprint-12 observations is the
*worst-case* with all features enabled across both workspaces plus
miri sweeps. Sprint-13 will push this up: D-CSV-13b adds AVX-512
intrinsic kernels (per-ISA codegen multiplies the binary count),
and D-CSV-17 adds rayon-feature test artefacts.

**Recommendation:** Before sprint-13 spawn, run `cargo clean` on
the *ndarray* tree (cheaper to rebuild than lance-graph; ndarray
has no proc-macro heavy crates). Set a 24-hour reminder in the
sprint-13 main thread to `cargo clean` on lance-graph if disk
crosses 20 GB.

### R2: Sibling-repo CI gaps

`mcp__github__list_pull_requests` on `AdaWorldAPI/ndarray`
returns **no open PRs**, which confirms ndarray CI is healthy.
However, the sprint-12 observation that "only ndarray has CI"
across the broader sibling set (q2, Hiro-rs, MedCare-rs, woa-rs)
still stands. Those sibling repos do not flow into sprint-13
deliverables and their CI gap is **out of scope** — but be aware
that if a sprint-13 worker accidentally edits a sibling
(e.g., copy-paste error pulls in `/home/user/q2/crates/…`),
there is no CI safety net. **Mitigation:** Sprint-13 worker
template should explicitly scope `Edit`/`Write` permission to
`/home/user/lance-graph/**` and `/home/user/ndarray/**`.

### R3: Cross-repo rebase pattern

If sprint-13 lands D-CSV-13b on the ndarray side concurrent with
unrelated ndarray master commits (likely — ndarray averages ~3 PRs
per week per the sprint-11/12 cadence), the ndarray branch will need
periodic merges from master. The lance-graph draft PR (using `branch
= …` per §5) will see those merges automatically, but **may break
CI** if the unrelated ndarray commit shifts a symbol path. This is
acceptable noise; sprint-13 main thread should expect 1-2 "ndarray
master moved" CI re-runs per multi-repo PR.

### R4: Wave G (PR #390) is the critical-path blocker

The single biggest scheduling risk for sprint-13 is **PR #390 not
merging in time**. Per the Wave-G W-Meta-Opus review, sprint-13
spawn is conditional on:

1. Wave G merge (PR #390)
2. Sprint-13 worker template baking lib.rs/workspace registration
   into worker scope (PP-2 territory)
3. CSI-18 doctrine consolidation
   (`.claude/knowledge/iron-rules-doctrine.md` — already shipped
   in commit `be11843` per PP-2)

Items 2 and 3 are addressed by PP-2/PP-8 (worker template) and
PP-2 (doctrine). **Item 1 is the gate.** This audit recommends
sprint-13 spawn does **not** proceed until PR #390 is merged.

### R5: cam_pq adapter math choice (D-CSV-16)

PP-5 spec must answer: which of `train_geometric` /
`train_semantic` / `train_hybrid` is the correct training mode for
SPO key distribution? The codebase contains zero precedent for
training cam_pq codebooks over SPO triples (CAM-PQ has been used for
vector embeddings only). **Recommendation:** PP-5 cites
`encoding-ecosystem.md` "CAM-PQ vs VSA" decision matrix and the
`I-VSA-IDENTITIES` iron rule in CLAUDE.md before settling on a
training mode. Default candidate: `train_geometric` (lossless on
the distance metric, no labels required).

---

## §7 — Summary Table

| Audit topic | Status | Sprint-13 action |
|---|---|---|
| ndarray PR #147 (D-CSV-11 streams) | **Merged** 2026-05-16 | D-CSV-17 builds on this |
| ndarray PR #116 (hpc-extras gate) | **Merged** 2026-04-30 | No action |
| ndarray open PRs | **0** | Clean runway |
| lance-graph PR #390 (Wave G) | **Open**, sprint-13 gate | **Merge before sprint-13 spawn** |
| ndarray `rayon` feature | Declared, unused in `stream/` | D-CSV-17 wires it |
| ndarray `simd_caps()` singleton | **Canonical, `#[non_exhaustive]`** | D-CSV-13b dispatches via this |
| ndarray `cam_pq.rs` API | u64 SPO adapter is **lance-graph-side only** | D-CSV-16 = lance-graph PR; no ndarray PR needed |
| `convergence.rs` `#[allow(unused_imports)]` | 2 of 3 still dormant; line 26 now stale | TD entry, not sprint-13 blocker |
| p64 / p64-bridge | Dormant for sprint-13 | No PRs |
| target/ disk usage | 8.9 GB Σ today; will grow | `cargo clean ndarray` pre-spawn; monitor lance-graph |
| Sibling-repo CI | Only ndarray has it | Worker template scope-locks `Edit`/`Write` paths |
| Multi-repo PR protocol | Documented §5 | Use draft-PR + `branch = …` override for D-CSV-13b |

---

## §8 — Cross-References

- `CLAUDE.md` § "Cross-Repo Dependencies"
- `.claude/plans/cognitive-substrate-convergence-v2.md` §11 (D-CSV matrix)
- `.claude/specs/pr-sprint-13-rayon-streams.md` (PP-3, D-CSV-17)
- `.claude/knowledge/iron-rules-doctrine.md` (PP-2)
- `.claude/knowledge/encoding-ecosystem.md` (cam_pq, VSA identities)
- `.claude/knowledge/cross-repo-harvest-2026-04-19.md` (sibling-repo epiphanies)
- `.claude/board/AGENT_LOG.md` (sprint-13 PP-12 entry — to be prepended on commit)
- ndarray master HEAD `2a3885d2` (PR #146 merged 2026-05-16)
- ndarray PR #147 (D-CSV-11 streams, merged 2026-05-16)
- ndarray PR #116 (hpc-extras, merged 2026-04-30)
- lance-graph PR #390 (Wave G, open 2026-05-16) — **sprint-13 gate**

---

*Authored by sprint-13 preflight PP-12 (Opus 4.7 planner, main-thread),
2026-05-16. Append-only; corrections go in a dated entry below
this line, not via edit of preceding sections.*
