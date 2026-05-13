# Agent W7 — Sprint-3 log

**Role:** Worker Agent W7 of Sprint-3 (12 + meta CCA2A).
**Branch:** `claude/tier-1-implementation-specs` of `AdaWorldAPI/lance-graph`.
**Tier:** Tier-3 specialization spec.
**Tech-debt anchor:** TD-INT4-32D-ATOMS-6.
**Pattern letter (post-PR #359):** Pattern J — INT4-32D Thinking Atoms.

---

## Deliverable

`.claude/specs/pr-j-1-int4-32d-atoms.md` — PR-ready spec for the
INT4-32D thinking-atom fingerprint + K-NN proximity search over the
12-entry `p64-bridge::STYLES` codebook. After this spec, an engineer
picks up the PR and starts coding.

## Status

**DONE — spec drafted and pushed to branch via pygithub.**

## Decisions logged

1. **Type lives in `lance-graph-contract`, not `thinking-engine`.**
   `ThinkingAtom32x4` is a cross-cutting type (consumed by p64-bridge,
   thinking-engine, planner, and eventually ContextBundle's
   `thinking_styles` slot). Putting it in the contract crate keeps
   downstream consumers free of a heavy thinking-engine dep — same
   policy W2 used for `SpoQuad` (PR-A-1). Contract crate stays
   `no_std`-clean.
2. **`cosine_int4` = nibble-min-sum, NOT popcount-AND.** INT4
   nibbles encode an ordinal axis (0..15). Popcount-AND treats them
   as 4 unrelated bits and throws away the ordinality. Worked test
   in open-question 2: `min-sum(0xF, 0xF) - min-sum(0xF, 0x0) = 15`
   versus `popcount(0xF & 0xF) - popcount(0xF & 0x0) = 4`. Min-sum
   gives a 4x richer signal at INT4 resolution and stays in `u32`
   arithmetic with no overflow risk (max `32 x 15 = 480 << 2^32`).
3. **32 dims hand-curated for v1; PCA deferred to v2.** Hand-curating
   from existing thinking-engine literature (NARS, ReAct, Six Hats)
   bootstraps in one engineer-day. PCA needs a `(situation,
   chosen-style)` corpus we do not yet have. v2 PCA is gated on ~10K
   dispatch decisions logged from deployed v1; flagged as follow-up
   `TD-INT4-32D-PCA-v2`.
4. **Per-style fingerprint cached as `pub const`, not lazy.** 12 x 16
   = 192 bytes static is negligible; lazy adds startup-order
   entanglement with no measurable benefit. Re-visit only if the
   codebook grows past ~10K entries.
5. **Dim ORDER is contract-stable.** `DIM_NAMES` index is the wire
   format of the fingerprint. Reordering or inserting (vs appending)
   is a breaking change to the codebook layout. Documented on the
   const definition.
6. **`invert_nibbles` exposed but unused by PR-J-1.** Reserved for the
   future Pattern G dispatcher to query "anti-style" atoms when
   negative evidence is present. Consumer wiring lands in PR-G-* once
   Pattern G is spec'd (deferred this sprint).
7. **Tie-breaking guarantee documented.** Rust's `sort_by_key` is
   stable; `knn_thinking_styles` resolves score-ties in codebook
   order. Downstream consumers MUST NOT assume any other rule;
   documented on the function.

## Dependency call-out

PR-J-1 is **independent** of PR-B-1 for compile / test purposes — the
K-NN function takes a generic `&[(ThinkingStyle, ThinkingAtom32x4)]`
codebook slice and does not touch `ContextBundle`. PR-J-1 can land in
parallel with PR-B-1.

The **integration point** is PR-B-1's `ContextBundle.thinking_styles`
slot: once PR-B-1 lands, that slot is populated by calling
`knn_thinking_styles(query_atom, &p64_bridge::STYLES, k=3)` and
storing the top-3 styles + scores. PR-J-1 does NOT modify
`ContextBundle` directly — that surface is W3's deliverable.

This independence is deliberate so Week-3 (PR-J-1) does not stall on
Week-1 (PR-B-1) slipping. Open-question 4 in the spec flags the
consumer wiring for the engineer.

## Cross-worker handover

- **W1 (master plan):** `sprint-3-execution-plan.md` — references
  PR-J-1 as the Week-3 specialization deliverable.
- **W2 (PR-A-1, SPO-G u32 slot):** sibling Tier-1 spec; same crate
  partitioning policy (contract crate hosts the cross-cutting type).
- **W3 (PR-B-1, ContextBundle):** consumer of `knn_thinking_styles`
  via the `thinking_styles` slot. Integration is W3's responsibility,
  not W7's. PR-J-1 stays a leaf API.
- **Future Pattern G PR:** consumer of `invert_nibbles` for negative-
  evidence queries. Deferred this sprint; flagged in open-question 4.

## Files written this session

- `.claude/specs/pr-j-1-int4-32d-atoms.md` (spec, ~14 KB — heavier
  than the ~8 KB target because the 32-name `DIM_NAMES` const
  inlines ~2.5 KB of cognitive-axis labels that are load-bearing
  for the engineer)
- `.claude/board/sprint-log-3/agents/agent-W7.md` (this log)

## Brutally-honest self-review

### What this deliverable does well

- **Crate-partitioning consistent with W2.** Type lives in the
  contract crate; downstream consumers depend on the type without
  pulling in thinking-engine. Same policy as `SpoQuad` in PR-A-1 —
  no surprise for engineers reading both specs back-to-back.
- **Independence from PR-B-1 stated up front.** PR-J-1 can land in
  parallel; the integration point is a one-line consumer call in
  PR-B-1's bundle resolver. Avoids the Week-1 bottleneck risk
  (Risk #1 in the W1 master plan).
- **Metric choice has a worked test.** Open-question 2 quantifies
  why nibble-min-sum beats popcount-AND (15 vs 4 on the F/F vs F/0
  delta), so the engineer is not picking on vibes.
- **DIM_NAMES is full**, not "// 27 more dims" hand-wave. 32 names
  with axis hints means an engineer can hand-code the 12 fingerprints
  without re-deriving the cognitive axes from scratch.

### What is weak / honest gaps

- **The 32 dims are an architectural call without a citation chain.**
  Names are drawn from the existing thinking-engine literature
  (NARS, ReAct, Six Hats) but the spec does not point to the source
  for each. An engineer wanting to push back on a specific axis name
  has no anchor doc to argue against. Mitigation: v2 PCA replaces
  hand-curation entirely once telemetry is available.
- **No benchmark for the 12-entry scan.** "Compiler auto-vectorises
  the 16-byte loop" is asserted; not measured. At 12 entries this
  is unlikely to matter (sub-microsecond regardless), but if the
  codebook grows past ~1K entries this assertion needs a real
  benchmark and possibly explicit SIMD intrinsics. Out of scope for
  PR-J-1; flag for the codebook-growth follow-up.
- **Pattern G consumer wiring is hand-waved.** `invert_nibbles` is
  exposed for the future dispatcher but no spec exists yet for how
  Pattern G consumes it. PR-J-1 ships the primitive without the
  consumer story. Acceptable separation of concerns but means
  PR-J-1 lands a slightly-orphaned API surface.
- **No fixture for the 12 hand-coded fingerprints.** The spec says
  "12 hand-coded literals, ~30 LOC" but does not provide one
  worked example fingerprint matrix. The engineer will spend the
  cognitive-curation budget alone. Mitigation: the inline comment
  in the spec sketches one (`STYLE_DEDUCTIVE_RIGOR`-shaped) but
  full curation is the engineer's first task.
- **Spec is ~14 KB vs ~8 KB target.** The DIM_NAMES const accounts
  for ~2.5 KB and is load-bearing (an engineer cannot hand-code 12
  fingerprints without seeing the axis catalogue), but the prose
  could be tightened further. Trade-off: shorter spec ships faster
  but forces the engineer to grep for the dim list elsewhere.

### Honest grade

B+. Type design is clean and consistent with W2's crate-partitioning
policy. Metric choice is justified with a worked test, not vibes.
Integration with PR-B-1 is correctly hands-off (parallel-landable).
Weak points: dim catalogue lacks per-axis citations, no benchmark
for the auto-vectorised scan, Pattern G consumer story deferred. Ship
it; flag dim-citation gap for the v2 PCA TD entry.

## Protocol notes

- Used pygithub `create_file` for both writes (both files were new on
  branch — verified via `repo.get_contents` returning
  `UnknownObjectException`).
- Token quotes stripped per W7 brief: `os.environ['GITHUB_TOKEN']
  .strip().strip('"').strip("'")`.
- One commit per file (2 commits total on this branch from W7).
- Branch HEAD on entry: `7a70f852017a0990e333ba1ec7a17bff84d11054`.
- No MCP, no local FS for the writes — clean pygithub-first path.

## Next handover

Engineer pickup. The spec's "Open questions for the engineer" section
has five items with recommendations; engineer should confirm or push
back before coding starts. First task on engineer pickup is
hand-curating the 12 `int4_32d_fingerprint` literals against the 32
named dims — this is the cognitive-curation step the spec cannot do
itself.
