# Issues Log — Open + Resolved (double-entry, append-only)

## ISS-GRADING-IGNORES-EACH-THREADS-TEMPORAL-HORIZON (2026-07-29) — OPEN ON `main`

**Severity: P1 correctness. Shipped in #868 (merged `1bbc890`), unfixed.** Filed
here because its only other record is a review thread on a closed PR.

**The defect (Codex, #868, verified against the code).** `WitnessLens::at(pos)`
always reads the **current snapshot**. `visible(pos)` can only *include or
exclude* an address — it structurally **cannot** select the historical row
revision that `QueryReference::at(v, rung)` + `deinterlace` produce. So when the
standing wave holds threads pinned to different Lance versions:

- a thread whose row was modified after its horizon is graded from the **newer**
  register — it reads a future it should not be able to see;
- one invocation shares **one** peer domain, so quorum, trajectory and basin
  membership describe a **global snapshot**, not each thread's corpus-as-of view.

**Why no better predicate fixes it.** Include/exclude is the wrong *arity* for
the question. The row bytes are fetched from the wrong version **before** the
predicate is consulted, so no `visible` implementation can recover the right
ones. The input must carry the focal thread's **temporal projection / address
set**, not a spatial filter over the current lens.

**Scope — this subsumes three items previously filed separately.** `GradedRow::pos`
being a lens index, the Θ(N·k) peer-domain scan, and `MetaBasin`/`MiniBasin::members`
holding rows already in the tail buffer are **not** three defects with a shared
cause; they are all downstream of the lens exposing one snapshot to every thread
at once. Fixing any one alone is incoherent.
Supersedes the address-list plan in `TD-LENS-QUORUM-SCANS-THE-WHOLE-LENS` — that
plan was still a single-version design, just a cheaper one.

**Why the suite is silent.** Every fixture in `meta_basin` builds one snapshot at
one version, so the entire temporal axis is **constant across every comparison**,
including the equivalence test that feeds the *same* fixture to both sides. Third
instance today of `E-THE-EQUALITY-PASSED-WHILE-AN-AXIS-WAS-CONSTANT-1` — and this
one was found by asking a reviewer "what else is constant across my fixtures?"
rather than by the suite. **Any fix must land with a multi-version fixture**, or
the same silence repeats.

**Operator framing that produced it:** ~64k reasoning threads, each temporally
situated, reconciled through `temporal.rs`. The merged code models rows as data
swept at one instant; the substrate is threads reading their own corpus-as-of.


## 2026-07-27 — ISS-841-856-NEVER-ANSWERED-REVIEW-COMMENTS — the forensic recovery's full ledger of GitHub review/issue comments across #849–#856 that never received a reply, sorted by whether the underlying finding was fixed anyway

> Filed by the arc-841-856-postmortem recovery session. Every item below was
> verified directly against the GitHub API (`pull_request_read` /
> `get_review_comments` / `get_comments`) on 2026-07-27, and every "fixed
> forward" claim was cross-checked against the actual diffs of the PRs named,
> not assumed from a later PR's description. See
> `.claude/board/PR_ARC_INVENTORY.md` (#851–#856 entries) for the full
> per-PR context and `.claude/handovers/2026-07-27-2114-arc-841-856-postmortem.md`
> for the process analysis of why these went unanswered.

### A. #852 — 20 review threads, ZERO replies on the thread itself — content mostly fixed forward in #853, process never closed out

Every one of #852's 20 review threads (`chatgpt-codex-connector` + `coderabbitai`)
is `is_resolved: false` with `total_count: 1` (bot comment only, no author
reply) as of the API read above. Cross-checked against #853's two follow-up
commits (`92742b1`, `8e71483`):

- **Fixed forward, verified in the #853 diff** (11 of 20): `meta_basin.rs`
  ×3 (recompute-on-perturbation over the complete window; `stability_around`
  windowing off a fixed cap; the "marks budget artifacts" test-coverage gap),
  `style_strategy.rs` ×2 (eligibility-before-stride; the `cross_family_dissent`
  monoculture), `witness_fabric.rs` ×1 (an assertion that couldn't fail),
  `cam_pq_scan.rs` ×1 (same), `insight_reason_wired.rs` ×2 (schema-corruption
  visibility), `build_alignment.py` ×1 (`houses`/`prizes` mis-stemming).
- **Fixed forward, verified in the #853 diff, Python generators** (6 of 20):
  `build_rosetta_probe.py` ×2 (dead row cap, report typo), `closed_class_transfer.py`
  ×1 (hardcoded scratch path — **note:** this is a *different* defect in the
  same file than the one #853's own review later found unfixed, see §C below),
  `fetch_greek_lane.py` ×2 (`--no-fetch` silent network + hardcoded scratch
  path), `build_wordnet_rail.py` ×1 (per-POS denominator), `tier_delta.py` ×2
  (WNDB world-writable fallback, `word_b` absence asymmetry — **one of two**
  `tier_delta.py` findings from #852; a *third, unrelated* `tier_delta.py`
  defect surfaced independently in #853's own review, see §C).
- **Acknowledged but deliberately not reverted** (1 of 20): the coderabbitai
  finding that `EPIPHANIES.md` had four historical entries redacted in place
  (a private-archive path reference) — #853's board work records a trace
  entry stating the redaction stands and why, rather than reverting it. This
  is a considered decision, not an oversight, but it was never posted back
  as a reply to the originating thread.
- **Fixed forward via board reorganization** (1 of 20): the `TECH_DEBT.md`
  append-vs-prepend violation — #853 reordered the file newest-first.
- **No reply ever posted to any of the 20 threads on GitHub**, regardless of
  whether the finding was acted on. A reviewer (human or bot) opening PR
  #852 today sees 20 apparently-ignored findings, indistinguishable from a
  stonewalled review without cross-referencing #853's diff by hand — which
  is what this recovery had to do.

**Status:** content CLOSED (verified fixed or deliberately kept, 19/20);
process gap OPEN and will recur unless a merge-time check requires either a
reply or a cross-referencing commit note before a PR with open review
threads can be treated as done. See the postmortem §5 for the proposed guard.

### B. #851 — 2 review threads, both P1, both never answered, both still open

Both threads (`chatgpt-codex-connector`, posted 2026-07-26T10:31:32Z, the PR
merged at 10:34:10Z — three minutes later) dispute the central claim of
`E-CODEBOOK-LICENSE-REGIMES-ONE-ASSET-EACH-1`:

1. The German codebook's PUBLIC/BY-SA verdict drops HDT's own stated
   NC restriction — `build_de_codebook.py:247`'s generated lexicon labels
   itself "CC BY-SA / CC BY-NC-SA," an apparent direct contradiction of the
   ruling's "commercial-OK BY-SA" classification for the combined codebook.
2. The ruling treats packaging location (Release vs. repo tree) as the
   legal ShareAlike/aggregation boundary test; the reviewer argues
   adaptation-vs-collection must be determined by the relationship between
   the works, and that "keep it in a separate Release" is a project policy,
   not itself a legal boundary — a distinction future publishers could miss.

Neither critique was rebutted, revised, or acknowledged anywhere in #852
through #856. **Status: OPEN.** One inconclusive lead: `EPIPHANIES.md` (as it
stands after later, unrelated edits) lists "UD German-HDT | CC BY-SA 4.0
(annotation)" as a table row distinct from GSD, which may gesture at an
annotation-vs-underlying-text licence distinction relevant to critique 1 —
but no commit or comment anywhere in the traced range explicitly connects
that wording to this thread. Treat as UNRECONSTRUCTED, not as a fix.

### C. #853 — 3 review threads, none answered, none fixed forward (verified absent from every #854/#855/#856 diff)

1. `closed_class_transfer.py:224` (codex P1) — the evaluation only iterates
   `predicted.items()`, excluding every unaligned German token from the
   transfer method's false-negative count, while the baseline dictionary
   uses the full codebook vocabulary; whenever alignment coverage is below
   100% (which the script itself measures) the two recall/F1 denominators
   differ and the headline "transfer beats/loses to baseline" claim can be
   wrong. **Distinct from** the hardcoded-scratch-path defect in the same
   file that #852's review found and #853 itself fixed — two independent
   bugs, one paid down, one not.
2. `meta_basin.rs:466` (codex P2) — `coarse_flags` constructs each basin
   from `tail_rows` but reclusters the *complete window* and compares
   against the tail-only member set; a non-tail row sharing the basin's
   shape falsely marks the basin unstable. **Distinct from** the
   `stable_under_perturbation`/`stability_around` findings #852 raised on
   the same file, which #853 itself fixed in the same commit pass.
3. `tier_delta.py:318` (codex P2) — when both inputs resolve to the same
   non-root synset, `lca_depth_from_root` is hardcoded to `0` instead of
   using the same `synset_root_depth` computation every other path uses,
   producing an inconsistent depth for identical-synset pairs in the
   tier-delta report.

Confirmed via `git log --oneline b0b6419..62658af -- <file>` for all three
files: **zero commits touch any of them again anywhere in #854–#856.**
**Status: OPEN**, genuine unpaid technical debt.

### D. #855 — 15 review threads, all eventually answered (4 pre-merge fixed same-day; 11 post-merge closed out one PR later in #856)

Not a gap by the time the range ends, but worth recording precisely because
it is the one case in this range where a PR closed with unanswered threads
and the *next* PR explicitly, itemizedly closed every one of them (posted as
an issue comment on #855 itself, `2026-07-27T16:51:39Z`, listing all 11 by
number with CONFIRMED/rejected verdicts). This is the pattern #852 and #853
should have followed and didn't. **Status: CLOSED**, cited here as the
counter-example.

### E. #856 — 3 review threads posted around merge time, none answered, one substantive and unresolved

1. **(codex P1, substantive, unresolved)** — disputes this PR's own headline
   finding ("the Base17 ceiling is dimensional, not the fold's grouping") by
   citing a **pre-existing measurement already in the same file's history**
   (`EPIPHANIES.md` former lines 3899–3901: naive PCA-17 scored Spearman
   0.72 on raw Jina vs. the golden fold's 0.32). Argues that scoring one
   random Gaussian JL draw with L1 cannot establish that *no* trained or
   data-dependent 17-dim projection escapes the ceiling, and that closing
   the "tighter projection" payable option on this basis risks prematurely
   killing a promising line of work. **This is a live, unresolved challenge
   to a finding this ledger and `TECH_DEBT.md`'s `TD-BASE17-FOLD-CEILING-
   SINGLE-WORD` entry currently treat as settled** (`E-BASE17-CEILING-IS-
   DIMENSIONAL-AND-THE-GOLDEN-STEP-IS-A-RELABEL-1`). Recommended next step:
   re-run the probe with a PCA-fitted (not random) 17-dim projection on the
   *same* sample and readout as the golden fold, and either reconcile the
   0.72-vs-0.32 figure or explain why it doesn't transfer to this probe's
   setup. **Status: OPEN, and should gate reliance on the dimensional-
   ceiling finding until answered.**
2. **(coderabbitai, minor)** — `F32-RETIREMENT-SCOPE.md` is missing a
   required `file:line` citation. **Status: OPEN, trivial.**
3. **(coderabbitai, major, self-referential)** — the very corrections this
   PR made to the trace files (rewriting "6" to "7" call sites, "276" to
   "308" bytes, etc.) were applied by editing historical text in place
   rather than by prepending a dated correction — exactly what CLAUDE.md's
   append-only board-hygiene rule (itself extended by #853 the same week)
   forbids. **Status: OPEN** — ironic given #856 itself closed out a
   similar finding against #852 in its own PR body.

**Confidence (2026-07-27):** all six sub-sections above are sourced directly
from the GitHub review-comment API, cross-checked against `git log` for the
"fixed forward" claims. Section A's 19/20 "fixed" count is the one synthetic
judgment call in this entry — it required matching each thread's file/line
against #853's commit diffs by hand, since no thread was ever marked
resolved with a linking comment; treat individual file attributions in §A as
high-confidence but not GitHub-verified in the way §C's "confirmed absent"
claim is (that one is a mechanical `git log` fact, not a judgment).

## 2026-07-27 — ISS-CONTRACT-DISTANCE-IS-THE-FORBIDDEN-UMBRELLA + ISS-COSINE-REPLACEMENT-SOURCES-CONTRADICT — **section B RESOLVED (dissolved, §E); section A STANDS and is STRENGTHENED (§E, §G measured zero consumers); section D OPEN (typed per-metric surface unbuilt)**

> Status line updated 2026-07-27 (was: `OPEN`) so the heading matches §E/§G
> below. Sections A–G are unedited — this file is append-only and only the
> status marker is revisable. §A is NOT superseded: §E says verbatim
> *"Section A stands and is STRENGTHENED"*.

Source: ndarray `.claude/knowledge/cognitive-distance-typing.md` (operator-cited),
the binding API-design authority for distance typing.

### A. `lance_graph_contract::distance::Distance` is the anti-pattern that doc forbids

> *"No `Box<dyn Distance>` / no `enum DistanceMetric { Palette, Hamming, Base17, … }`
> / no `fn distance<T: HasMetric>(a, b) -> f32` umbrella. The type system
> distinguishes the metrics for a reason."*

`contract::distance::Distance` is exactly that: one trait, `fn distance(&self,
&Self) -> u32`, impl'd for `[u64;256]` (Hamming), `[u8;6]` (PQ byte-L1) and
`[u8;3]` (palette byte-L1) — **three different metrics under one generic API
returning one untyped scalar.** The doc requires instead: one named fn per
metric, newtyped outputs (`PaletteDistance(f32)` / `HammingDistance(u16)` /
`Base17L1(i32)`) so cross-metric arithmetic does not compile, and REQUIRED
`buckets` + `EulerGammaOffset` on every palette-256 call.

This inverts the 2026-07-27 cosine census, which was organised around migrating
call sites TOWARD that umbrella. It also explains the measured ρ = −0.0030 for
`[u8;6]` structurally: the umbrella returns a `u32` the type system cannot
distinguish from a real distance.

### B. Two in-repo sources contradict on WHAT the cosine replacement is — REPORTED, NOT RESOLVED

| source | claim |
|---|---|
| ndarray `cognitive-distance-typing.md` | **HDR popcount early-exit** *"IS the cosine replacement on the cascade — NOT a derivative or approximation of cosine"* (Level 1, ~1M → ~20K). Fisher-z is *"**NOT a distance** — a normalization applied to palette 256 OUTPUT… Calling Fisher-z on a non-correlation value is a category error."* |
| bgz-tensor `nnue_palette_cosine.rs:172` + `EPIPHANIES E-FISHERZ-CANONICAL-COSINE-REPLACEMENT-1` | `FisherZTable` is *"the certified palette256 cosine-replacement"* |

`ISS-FISHERZ-COSINE-REPLACEMENT-IS-SHIPPED-BUT-UNWIRED` (filed earlier today) took
bgz-tensor's wording as settled. That was an assumption; it is **downgraded to
CONTESTED** pending an operator ruling. Both artifacts are in-tree and current.

### C. The three-level cascade — the probes used the wrong level

| Level | metric | scale |
|---|---|---|
| 1 | HDR popcount early-exit (`&Fingerprint256` ×2 + `u16` threshold → `Option<HammingDistance>`) | ~1M → ~20K |
| 2 | Base17 L1 on `[i16; 17]` — *"don't pad to 16 or 18"* | ~20K → ~200 |
| 3 | Palette-256 table lookup + **`buckets` + `EulerGammaOffset`** | ~200 finalists |

`probe_palette256_ndarray` ran a **Level-3** pair LUT across all 4096 candidates,
skipping L1 and L2 — and its "HDR gate" passed `hamming_distance_raw` ONE BYTE
instead of two 256-bit fingerprints, which is the root cause of the
already-reported can't-fire defect. Its LUT also carried neither `buckets` nor
`EulerGammaOffset`, which the doc says *"changes the answer"*.

Also: the direct in-palette fast path (`palette256_bf16_mantissa_transform`,
one typed hop, no cascade) was never considered by any probe this session.

### D. Full-read addendum (doc verified byte-identical to pinned commit 6ff231ad)

**The doc is a binding spec AHEAD of the code.** Its cross-refs and prescribed
types are unbuilt: `src/hpc/distance.rs` does not exist (`layered_distance.rs` /
`palette_distance.rs` do); the `CLAUDE.md § "Three-Level Cascade"` section is
absent; and 5 of 7 prescribed types exist in NEITHER repo (`PaletteIdx`,
`EulerGammaOffset`, `Fingerprint256`, `BF16MantissaCtx`, ndarray-side
`PaletteDistance`). So the answer to "what part of the cosine replacement didn't
you find" is complete: **the mechanisms are shipped in pieces; the typed API
surface the doc mandates is assembled nowhere.**

**Proposed reconciliation of §B (textual, NOT ratified):** the doc scopes
popcount's claim to the SEARCH TOPOLOGY (*"IS the cosine replacement on the
cascade… substitutes for FP cosine in the search topology"* — cosine as metric,
L1), while its Fisher-z row permits exactly what `FisherZTable` does
(*"variance-stabilizing transform of correlation"* — and `FisherZTable.entries`
ARE cosines of centroid representatives, genuine correlations, so no category
error). Two senses of "replacement": popcount replaces cosine-as-METRIC;
FisherZTable replaces cosine-as-STORED-VALUE (i8 value encoding in palette
space). Different axes, no conflict — pending operator confirmation.

**The doc's own audit item, retargeted:** "audit `src/hpc/distance.rs` for a
`fn distance<T>` umbrella" names a nonexistent file; the audit belongs on
`layered_distance.rs` + `palette_distance.rs`, and — if the doc's scope extends
beyond ndarray — on `lance-graph-contract::distance`, which remains the one
LIVE umbrella in code (§A).


### E. RESOLVED-BY-RULING (operator, 2026-07-27 -- "only palette256 and ONLY [a,b]; FisherZ COULD materialize but why, if palette256 has lower entropy: it IS normalized distance")

Section B's "contradiction" **dissolves** -- and not into a "two senses"
arbitration. There is nothing to arbitrate:

- **Only palette256.** One encoding. The census's rival-encoder framing was the
  error (same shape as ClassId-vs-classid: one carrier, many projections --
  transferred from identity to value and re-committed).
- **Only [a,b].** The pair-indexed table read is THE distance operation, and
  because the codes are normalized, that read IS normalized distance. Popcount /
  Base17-L1 / pair-table are this one operation at three cascade scales, not
  three metrics competing for a slot.
- **Fisher-z: could materialize, no reason to.** It decodes palette
  relationships into z-space -- more entropy, same ordering information.
  `FisherZTable` is a materialization ARTIFACT, not the canon. This ruling was
  ALREADY RECORDED in primer section 13 hours earlier ("palette256 could be
  materialized as FisherZ but doesn't need to -- lower entropy and higher value
  when normalized; a stored Fisher-Z column would be a forbidden materialized
  alternate representation") -- and this session then found FisherZTable and
  crowned it "the certified replacement" anyway. The failure was not missing
  information; it was not applying a ruling already on file.

Status: section B CLOSED (dissolved). ISS-FISHERZ-...-UNWIRED closes as INVALID
-- there is nothing to wire; wiring it would ship the unnecessary
materialization. Section A stands and is STRENGTHENED: contract::Distance is
the forbidden umbrella, and a generic distance() over uniform u8 codes is
exactly how a wrong-level read gets applied silently (measured: [u8;6] at rho
-0.0030). Section D's spec-ahead-of-code finding stands: the typed per-metric
surface over the ONE encoding remains unbuilt.


### F. THE CANON PREDATES THE SESSION BY TWO MONTHS (ndarray board, 2026-05-26)

`ndarray/.claude/board/EPIPHANIES.md` (verified at pinned commit 6ff231ad)
already carries, dated **2026-05-26**:

- **"Palette256 + Fisher-z IS the exact cosine replacement (integer, no float)"
  — Status: VALIDATED** (operator: 10 000×10 000 splat, theta ~ 1.45-1.6
  Fisher-z ~ cos 0.90-0.92). Ranking-exact Palette256 ADC integer lookup, gated
  by a Fisher-z aperture theta; no float MAC in the O(D) kernel.
- The section-B "contradiction" resolved in five words: *"popcount IS the cosine
  replacement **by topology, not value**; Fisher-z is a palette-output
  normalization, not a cosine reconstruction."* Sections D/E re-derived this.
- **Two lanes**: SELECT = integer cascade (L1 popcount -> L2 Base17-L1 -> L3
  Palette256 ADC); uncertainty = tiny per-edge float Sigma metadata. Co-certified
  siblings (Pflug-10 certifies the CAM-PQ quantization).
- **The cam.rs gap is a RECORDED MAY DEBT**: *"cam-pq-production-wiring (cam_pq
  shipped, unrouted through CamCodecContract)"* — today's "discovery" was on the
  ledger.
- **Correction to section A**: the validated entry GROUNDS the contract Distance
  trait as part of the design (theta lives at `lance-graph-contract::distance::
  similarity_z = atanh`). The typing rule binds ndarray's per-metric fn surface;
  "contract::Distance is the forbidden umbrella" OVERSTATED. The [u8;6]-is-noise
  measurement stands; the demolition verdict is withdrawn. What remains open is
  the narrower question: whether the [u8;6] byte-L1 fallback impl should be
  removed/renamed so no caller mistakes it for the ADC path.
- **Today's failure mode is a documented epiphany**: *"Grounding-discipline
  (meta — the expensive one)"* — a prior session built float code from ChatGPT
  inspiration without grounding against the integer/palette substrate in the
  same repo, net-zero-usable. Binding fix (2026-05-26): whole-file reads only;
  L0 source/tests/standards, L1 audit (spot-check never inherit), L2
  plans/perspective-docs are NOT evidence. Sibling epiphany: ledger-first,
  code-never-unless-necessary (10^7x cheaper). This session inverted both.

**Net status of this whole issue cluster:** B/D/E were re-derivations of the
2026-05-26 validated canon. A is narrowed as above. What was genuinely new
today: the [u8;6] noise measurement (rho -0.0030), the cost measurements
(276 vs 5-9 ns/cand; 0 B per-query state; table-build vs 550 ms SLA), and the
kmeans-vs-hand-rolled codebook delta (0.8494 -> 0.9725).

### G. MEASURED: the contract `Distance` trait has ZERO production consumers (2026-07-27)

Grep of the whole workspace for `use lance_graph_contract::distance::Distance`
returns exactly ONE hit: `crates/lance-graph-planner/examples/probe_palette256_ndarray.rs:28`
— this session's own probe. `similarity_z` (where the 2026-05-26 VALIDATED entry
places the Fisher-z theta aperture) appears only in `cam.rs:226,240,432`
doc-comments, never in a call. (`helix::distance::DistanceLut` hits are a
different module.)

Consequences:
1. **Removing or renaming the `[u8;6]` byte-L1 impl breaks nothing** — no caller
   to migrate. The section-A open item is therefore free to action, and its
   noise measurement (rho -0.0030 vs exact) is the justification.
2. **The cosine census's "REPLACE #1 = migrate cam.rs onto Distance" pointed at
   an unconsumed trait.** Recorded as a second-order error of that census.
3. **The trait is SPECIFICATION-SHAPED, not load-bearing** — declared,
   in-crate-tested, unwired. Consistent with the standing May debt
   `cam-pq-production-wiring` ("cam_pq shipped, unrouted through
   `CamCodecContract`"). Neither "forbidden umbrella" (section A, already
   withdrawn as overstated) nor "the canonical dispatch consumers use"
   (this session's framing) described it correctly: it is an unwired contract
   surface awaiting that debt's resolution.


## 2026-07-27 — ISS-FISHERZ-COSINE-REPLACEMENT-IS-SHIPPED-BUT-UNWIRED — the certified replacement exists; nothing in the spine reaches it — **CLOSED-INVALID** (operator palette256-ONLY ruling, §E: `FisherZTable` is a materialization artifact; there is nothing to wire, and wiring it would ship the unnecessary materialization. Was `CONTESTED`; history retained — ISS-COSINE-REPLACEMENT-SOURCES-CONTRADICT: ndarray `cognitive-distance-typing.md` says HDR popcount IS the cosine replacement and Fisher-z is NOT a distance; this entry took bgz-tensor's "certified cosine-replacement" wording as settled — an assumption, pending operator ruling; CLOSED-INVALID 2026-07-27 -- palette256-ONLY ruling: FisherZTable is a materialization artifact, nothing to wire, see section E)

**The cosine replacement is not missing. It is shipped, certified, and named** —
and the 2026-07-27 cosine census missed it by asking "is this cosine a violation?"
instead of "is this cosine the replacement?". `fisher_z.rs` was filed LAB /
table-build; `bgz-tensor` was written off as "zero spine imports".

- **`FisherZTable`** — `bgz-tensor/src/fisher_z.rs:100`: `entries: Vec<i8>` =
  *"k×k i8 encoded cosine values (row-major)"* + `gamma: FamilyGamma`, with
  `lookup_i8(a: u8, b: u8) -> i8` / `lookup_f32(a, b)`. **This is `distance is
  [a,b]`** — the pair indexes the table. *"k=256 → 64 KB + 8 bytes. 26 groups ×
  64 KB = 1.6 MB for the entire 1.7B model."*
  `nnue_palette_cosine.rs:172` calls it *"the certified palette256
  cosine-replacement"*.
- **`FamilyGamma`** — `fisher_z.rs:28`, `BYTE_SIZE = 8`, `from_cosines()`
  (atanh over the pairwise distribution → `z_min`/`z_range`), `encode`→i8,
  `decode`→cosine, **`to_le_bytes` / `from_le_bytes`**. The gamma travels WITH
  the table.
- **`CosineGamma`** — `gamma_calibration.rs:136`: *"γ_cosine: cosine replacement
  offset (4 bytes per codebook)"*, *"redistributes u8 levels so the crowded
  center (cosine ≈ 0, where most pairs land) gets more resolution"*; stores
  measured `gamma` / `center` / `spread`.
- Board: **`E-FISHERZ-CANONICAL-COSINE-REPLACEMENT-1`** — *"helix = the analytic
  2z rung"*. Siblings: `E-FREQ-IS-COSINE-REPLACEMENT-1` (rank distance
  `|Δrank|/16`), and deepnsm `fingerprint16k.rs:10` *"replaces cosine with
  popcount — same bucket resolution"*.

**What is actually missing is WIRING, not a mechanism.** Three pieces exist and
none are connected:
1. `FisherZTable::lookup_i8(a,b)` — certified, in a workspace-EXCLUDED crate with
   no spine import;
2. `contract::distance::Distance` — canonical dispatch, **no palette impl**;
3. `impl Distance for [u8;6]` — the nearest existing thing, and it **measures as
   noise** (Spearman −0.0030, recall@10 0.0125 vs exact, probe
   `probe_palette256_ndarray`), because byte-wise L1 over centroid INDICES is not
   a metric.

**Two corrections this supersedes:** (a) the claim that the γ offset is not
stored (it is — `FamilyGamma::to_le_bytes`, 8 B, plus `CosineGamma` 4 B/codebook;
the earlier check looked at `euler_fold::FoldedFamily`, the wrong struct);
(b) `probe_palette256_ndarray`'s hand-built `Vec<u16>` k×k LUT — it
re-implements `FisherZTable` in a worse encoding, so its ρ 0.9725 measures a
hand-roll competing with the certified table, not the certified table itself.

**Shape not proposed.** Where the impl lands, whether `FisherZTable` re-homes,
and how a zero-dep contract reaches a calibrated table are decisions.

## 2026-07-27 — ISS-PALETTE256-HAS-NO-DISTANCE-IMPL — the canonical value carrier and the canonical distance dispatch both live in lance-graph-contract and are NOT connected — OPEN

Verified:
- `pub type Palette256Pair = (u8, u8)` — `awareness_facet.rs:32`, *"a palette256²
  centroid — (basin, identity)"*. `AwarenessFacet::from_rails([Palette256Pair; 6])`
  is the L4 `6×(8:8)` layout; `legacy_outliers.rs:30` calls it *"the exit — the
  real destination"*.
- `Distance::distance(&self, &Self) -> u32` — `distance.rs:19-24`.
- **`impl Distance for` exists for exactly three types**: `[u64;256]` (Hamming),
  `[u8;6]` (CamPq — its own doc at `:95-96` calls it *"an L1 fallback"*,
  explicitly not the real ADC), `[u8;3]` (PaletteEdge — byte-wise L1 **computed
  inline, not a table lookup**).
- **`awareness_facet.rs` contains the string `distance` ZERO times.**

The calibrated LUTs exist and ARE reachable — `ndarray::hpc::palette_distance::
{Palette, DistanceMatrix}` (*"every subsequent distance lookup becomes a single
u16 array load"*), with `ndarray::hpc::cascade::calibrate` supplying the
`mu + 3σ` rolling-floor threshold and `expose()` the HDR bands. ndarray is a
declared path dep of `lance-graph-planner`. (An earlier note claiming these were
unreachable was FALSE and is withdrawn — see the primer §13 addendum.)

**This supersedes the cosine census's "REPLACE #1 = cam.rs".** Migrating `cam.rs`
off float without a palette256 `Distance` impl relocates the hand-roll rather
than removing it. The head of the migration is the missing impl.

**Shape not proposed** — whether the impl belongs on `Palette256Pair`, on
`AwarenessFacet`, or on the 12-byte lane, and how the calibrated table is reached
(the contract is zero-dep; ndarray is not a contract dependency) are decisions,
not inferences. Reported, not designed.

## 2026-07-27 — ISS-PEARL-VOCABULARY-WITHOUT-PEARL-MECHANICS — the substrate has the Pearl TAXONOMY comprehensively and the Pearl OPERATOR not at all; four different kinds of "cause" share one untyped edge — OPEN

**Status:** OPEN (audit result, measured on the main thread — operator asked
directly: *"I'm not convinced that we implemented MIT proposed causality
learning properly"*. The suspicion is correct.) **Priority:** P1 for any claim
of causal capability; P3 for the running system, which does not currently
depend on the missing half.

**What EXISTS (verified, not guessed):** `orchestration_mode::pearl_level()`
returning 1/2/3; the SPO 2³ mask taxonomy mapping S×O → `P(Y|X)` (SEE, L1),
P×O → `P(Y|do(X))` (DO, L2), full SPO → `P(Y|do(X'),X=x)` (IMAGINE, L3);
`InferenceOp::Counterfactual`; `RungLevel::Counterfactual`; recipe 31 (ICR)
carrying the counterfactual label; a `pearl_junction` module. The mask→level
mapping is principled and worth keeping — this is a good taxonomy.

**What does NOT exist:**

1. **No intervention operator.** Nothing in `lance-graph-contract` or
   `lance-graph-planner` severs a node's incoming mechanisms, invalidates
   evidence derived from its old parents, or recomputes descendants. Grep for
   `sever` / `disable_mechanism` / causal-ancestry invalidation returns
   nothing; the only `lineage` hits are property-version lineage
   (`upsert_with_lineage`), which is storage versioning, not evidence ancestry.
   **Overwriting a value while retaining evidence derived from its old parents
   is a contradictory mutation, not `do(X = x)`.**
2. **The one counterfactual kernel is a stub.** `recipe_kernels::Icr` (#31)
   XORs three hardcoded `u32` constants, counts bit divergence, and multiplies
   its confidence contribution by **`0.0`**. Its parameter is `_ctx` — it reads
   nothing, so it returns the same result for every input and **no test over it
   can fail**. It is now labelled as a stub in source (2026-07-27); before that,
   recipe 31's rung was the only "evidence" of counterfactual capability.
   (3 other kernels are also context-blind: `Are`#19, `Zcf`#24, `Hkf`#34.
   31/35 do read context — this is a targeted gap, not a blanket stub problem.)
3. **No `CausalDomain` distinction.** Four structurally different causal claims
   share one untyped edge: **World** (`betrayal → loss of trust`, a claim about
   the represented world), **Interpretive** (`accusative marking → object-role
   resolution` — the case marker did not cause the event, it caused the PARSER
   to choose), **Derivational** (`WordNet rail + recipe 17 → proposition P
   entered the belief field` — why the substrate believes something), and
   **Experiential** (`negative qualia residue → later ambiguity read as
   threatening` — a causal relation inside the reader). Each needs a DIFFERENT
   intervention (`do(event = absent)` / mask grammatical evidence / disable a
   recipe or evidence lineage / lesion one qualia channel). Untyped, an
   intervention proving a token causes a parser decision can be recorded as
   evidence that the token's concept caused the narrated event — a deterministic
   category error.
4. **`InferenceType` duplication is live.** The contract's `nars::InferenceType`
   is `{Deduction, Induction, Abduction, Revision, Synthesis}` — **no
   Intervention, no Counterfactual** — while `CLAUDE.md`'s
   I-LEGACY-API-FEATURE-GATED discusses `InferenceType::Counterfactual` with
   mantissa −6 (the `causal-edge` crate's copy). `TYPE_DUPLICATION_MAP` records
   3 copies; this audit confirms they have DIFFERENT VARIANT SETS, which makes
   "the contract is canonical" false for this type today.

**Why it matters more here than elsewhere:** the substrate genuinely does make
causal EXPERIMENTS cheap (versioned snapshots, deterministic replay, parallel
branches, observable recipes, read-as-of bounds). That is the expensive half in
the external literature. But **cheap experiments do not make causal
IDENTIFICATION free** — observational evidence identifies an equivalence class
of graphs, and orienting the remaining edges needs surgical interventions,
preserved equivalence classes, represented hidden common causes (shared
manuscript, translation genealogy, shared tokenizer, shared WordNet binding,
shared Jina model, shared earlier inferred edge — the false-witness probe
already measured that lanes are not independent), cross-environment invariance,
and negative controls. None of those six are implemented.

**Resolution path (tasks #47/#48, deliberately ordered audit-before-build):**
FIRST classify every existing edge labelled causal into World / Interpretive /
Derivational / Experiential / Unknown, and for each record whether it was
observed, inferred from temporal order, supported by intervention, or supported
only by simulation. THEN type the domain. Do NOT build more causal machinery
before that census — the likely finding is that `preceded` / `enabled` /
`provided evidence for` / `derived` / `predicted` / `caused` are already
travelling under one umbrella, and separating them is the actual unlock.

**Cross-ref:** `E-CONFIDENCE-SHOULD-COMPRESS-KNOWLEDGE-OUT-OF-AWARENESS-1`,
`E-MUTATION-WAVE1-…-1` (measured witness non-independence = a hidden common
cause, already in hand), CLAUDE.md § falsifiability rule ("a doc-comment claim
is not a behaviour" — recipe 31 is the exemplar), `TYPE_DUPLICATION_MAP.md`.

## 2026-07-26 — ISS-VERSIFICATION-SCRIPT-BLIND — the anchor-overlap versification detector cannot measure cross-script lane pairs and reports the tie as `offset=0` — the Greek lane's versification is UNVERIFIED — OPEN

**Status:** OPEN (P1 for any consumer of `versification_map.tsv` rows involving
tischendorf; the German/Czech rows are unaffected — same-script pairs carry real
anchor signal). Surfaced by the `Mutate_VersificationOffset` operator
(`E-MUTATION-WAVE1-VERSIFICATION-DETECTOR-IS-SCRIPT-BLIND-1`).

**Defect.** `build_versification_map.py`'s `fuzzy_present` prefix-matches
Latin-alphabet anchor tokens against Greek-script text — zero shared codepoints,
structurally can never match. All three candidate offsets tie at score 0.0 and
the tie-break silently picks `offset=0`, at confidence 0.0000, on clean and
corrupted data alike (0/258 recovery of an injected +1 shift on anchor-basis
chapters; the length-ratio fallback recovered 2/2). An absent measurement is
being read as a zero offset — the exact absent≠zero violation the substrate
bans, in shipped tooling.

**Fix (task #43).** (1) A script-compatibility guard: when the anchor token set
and the target lane share no script, the chapter's verdict is `CannotMeasure`,
never `offset=0`. (2) A tie at score 0.0 across all candidates is also
`CannotMeasure` regardless of script. (3) Cross-script pairs route to the
length-ratio basis (the only mechanism that worked) with its basis labelled, or
to a transliteration/alignment-based anchor set if one is ever built. The
generator lives on the bake branch (`claude/rosetta-codebook-bakes-z30uij`), so
the fix lands there; consumers of the published map must treat existing
tischendorf rows as unverified until regenerated.

**Cross-ref:** `E-MUTATION-WAVE1-VERSIFICATION-DETECTOR-IS-SCRIPT-BLIND-1`,
`exec-runs/mutate-verseoffset.txt` (tag-file with the full census), the
falsifiability rule (CLAUDE.md P0 — "a guard/channel needs a can-it-fire test":
this detector's first can-it-fire test was this mutation operator, and it
couldn't).

## 2026-07-21 — ISS-BUNDLE-RULING-SCOPE — does E-NO-BUNDLE-STANDING-WAVE-1's niche-closure retire the deepnsm MarkovBundler cluster? The ruling's LETTER says yes; its stated MECHANISM (single-owner SoA violation) does NOT describe this code — ruling-author decision

**Status:** RULED 2026-07-21 (operator, path **(b)**) — KEEP the MarkovBundler cluster as is (path (a) full-retire NOT taken; path (c) unwired-retire deferred). The standing-wave **resolution** complement is built where the parallel rebuild lives: **`deepnsm-v2::wave::WitnessStream`** — version-stamped single-owner loci events, resolved `Causal`/`Escalate` by `witness_fabric::standing_wave_grounded`/`resolve_chain` over `TemporalStream`'s version-range window (out-of-version target → Escalate; the ±8 horizon meets the version read). 10 tests green. NOT in old `deepnsm` (that would have been the redundant third artifact this entry warned against). **Priority:** P2. **Scope:** @truth-architect domain:deepnsm domain:substrate.

**Surfaced by** a full read of the six-file cluster (`markov_bundle`/`trajectory`/`arcs`/`arcuate`/`disambiguator_glue`/`trajectory_audit`) AFTER an earlier session this day deleted it on a pattern-match (never read) as "the bundle violation" and had to fully revert (`AGENT_LOG` 2026-07-21). The read reframes the claim; this entry replaces the reverted, ungrounded `TD-BUNDLE-RESIDUE-1`.

**What the cluster actually is (read, not grepped).** `MarkovBundler::bundle_current` (`markov_bundle.rs:118`) IS a ±radius-window superposition of multiple sentences into one `Trajectory`, kernel-weighted into disjoint role slices (SUBJECT[0..2000)…), Σ|w|-normalized — structurally the "±5 braid" the ruling names. BUT: (a) LOCAL owned computation — `push` returns an owned `Trajectory` by value; it never writes a shared multi-owner SoA (the mechanism the ruling's "single-ownership violation" describes). (b) FIREWALLED — only a sign-binarized `Binary16K` fingerprint crosses into the contract `ContextChain`; the contract takes no deepnsm dep (`arcs.rs:19-22`, `arcuate.rs:28-35`). (c) `disambiguator_glue.rs:29-35` EXPLICITLY frames it as the sanctioned `I-VSA-IDENTITIES` niche (identity superposition of role-bound fingerprints; the Vsa16kF32→Binary16K switchboard hop). (d) UNWIRED from the live pipeline — `arcuate.rs:21-26` states it is a separate, offline-testable seam, NOT in `pipeline.rs`'s live `ContextWindow`. It is the Broca / arcuate-fasciculus / Wernicke faculty (conduction-aphasia gap) — coherent design, not junk.

**The tension (why it needs the ruling author).** The ruling's LETTER closes the I-VSA-IDENTITIES "≤32 within one compartment" niche → this ±radius bundle is out. But the ruling's stated MECHANISM — "mixes multiple owners' contributions into one carrier, breaking single-owner SoA" — does NOT describe this code: one producer bundling its own input window into its own owned, firewalled, unwired output, with none of the shared-SoA aliasing the ruling cites. So: is the niche closed even for a local, owned, firewalled, unwired computation? Ruling-author's call, not a pattern-match.

**Resolution paths (operator decision).** (a) Ruling applies in full → the cluster is real (understood) debt to retire → then a replacement standing-wave faculty must be WRITTEN + tested-to-parity against the current behaviour BEFORE any deletion (never the reverse — the reverted-mistake lesson); moves to `TECH_DEBT.md`. (b) Niche stays open for local owned firewalled computations → the cluster stays; the ruling's niche-closure wording is scoped to shared-SoA carriers. (c) Retire as UNWIRED on its own ruling-independent ground once a consumer sweep confirms zero live callers — separate from the bundle question.

**Cross-ref:** E-NO-BUNDLE-STANDING-WAVE-1 (the ruling — currently unrecorded in EPIPHANIES per its own revert), `I-VSA-IDENTITIES` (CLAUDE.md), `AGENT_LOG` 2026-07-21 (the deletion + revert), `markov_bundle.rs:118` / `arcuate.rs:21-35` / `disambiguator_glue.rs:29-35`.

## 2026-07-21 — ISS-DCSW-REAL-CORPUS-BLOCKED — D-CSW-1 leg 2 + the real-corpus D-CSW-2 are blocked on data + a real `temporal.rs` binding, not code — OPEN

**Status:** OPEN (blocker, not debt — the code path is fine; the inputs don't exist here). Surfaced by the #789/#791 arc.

**What is blocked.** Both the full D-CSW-1 leg 2 (real `temporal.rs`/Lance version stream over a wild corpus) and the real-corpus D-CSW-2 (labeled causal-edge candidate set) need two things this sandbox does not have: (1) a **sourced labeled corpus** (real causal pairs / candidate labels — none is in this repo or session; fabricating one would repeat the numpy-stand-in mistake `E-CODEC-IS-PALETTE256-SQUARED-IMPLICIT-1` corrected); (2) a **real `temporal.rs`/Lance version binding** — the contract-side `temporal_pov.rs` is the zero-dep filter, but the canonical `QueryReference`/`deinterlace` over actual Lance datasets lives in `lance-graph-planner`, which builds here but has no wired real-version stream to read.

**What is NOT the blocker (corrected).** The earlier "#789 leg-2 infra-blocked on `protoc`" claim was WRONG (PR #791, `E-DCSW1-LEG2-BLOCK-CORRECTION-1`): `lance-graph-planner` builds in ~20s and needs no `protoc`; that was a mislabelled too-tight timeout. Do not re-file the block as a build/toolchain problem.

**What IS done (so this issue is scoped, not open-ended).** The contract-level *mechanism* is proven (`E-DCSW2-CONTRACT-MECHANISM-GREEN-1`, PR #789: joint basin+rung precision@25 = 1.000 vs 0.520/0.520 ablations on a synthetic AND-gate fixture using the real `PairPalette`/`witness_fabric` primitives). Leg 1's v5 core standing-wave result is VALIDATED-IN-SCOPE (`E-DCSW1-V5-SPLIT-VERDICT`). The **feasible narrower next step** is a real-`temporal.rs` probe (planner-side, small labeled slice) — operator-gated, flagged not launched. **Resolution path:** source a small labeled causal corpus + wire one real `lance-graph-planner` version-stream read; until then this stays OPEN and neither leg is claimed as more than its scoped result. Refs: plan `causal-rung-standing-wave-v1.md` §6.2/§6.3, `E-DCSW2-CONTRACT-MECHANISM-GREEN-1`, `E-DCSW1-LEG2-BLOCK-CORRECTION-1`, `E-DCSW1-V5-SPLIT-VERDICT`, `TD-CERTIFIED-DISTANCE-TABLE-UNCONSUMED` (the codebook a real D-CSW-2 also needs).

## 2026-07-08 — ISS-V1-TAIL-RESIDUE — woa-rs arm — RESOLVED (`make_account_guid_bytes` migrated to the V3 tail)

**Status:** RESOLVED (operator ruling 2026-07-08, landed in woa-rs — sibling repo,
not this branch). This is a THIRD residue arm under the `ISS-V1-TAIL-RESIDUE`
umbrella, alongside the two lance-graph-contract mint sites (`ocr.rs`/`aiwar.rs`,
resolved 2026-07-07 below) — woa-rs's ERP account-GUID minter carried its own
independent V1-tail residue (`family` hash stuffing the Personenkonten trailing
digits) that the lance-graph-side fix did not touch.

**Landed:** woa-rs `src/erp/canon.rs::make_account_guid_bytes` now produces
`leaf(u16)/family(u16)/identity(u16)` byte-identical to `NodeGuid::new_v2`/
`mint_for(TailVariant::V3, …)` — the V1→V3 semantic move: the Personenkonten
trailing digits (SKR03 70000-99999) that V1 stuffed into the `family` **hash**
now live in the **LEAF tier** as a real `(ten_bucket:final_digit)` `(part_of:is_a)`
rail (`skr_leaf()` decomposes `70123` → LEAF `(2,3)`), matching the same
canonical `HEEL·HIP·TWIG·LEAF·family·identity` cascade shape this branch's
`mint_for` dispatch uses. **Parallelbetrieb invariant pinned** (doc block
"READ THIS FIRST" in `canon.rs`): the MySQL ORM mapping stays authoritative —
`identity` = the MySQL `erp_accounts` row id, converted `u16`-by-signature with
a loud `try_from` (`.expect("Parallelbetrieb: erp_accounts row id must …")`) at
every call site, never a silent `as` alias/truncation. **11/11 tests green** in
woa-rs (`src/erp/canon.rs` — 4-digit accounts, Personenkonten LEAF decompose,
round-trip byte layout, MySQL-id round-trip).

**Scope note:** woa-rs is a sibling repo (not this lance-graph worktree) —
this entry is board-hygiene bookkeeping only, landed same-commit as any other
ISSUES.md bookkeeping in this session per the V3 plan's standing gate 3
("board hygiene same-commit"). No lance-graph-contract code changed for this
arm.

Cross-ref: woa-rs `src/erp/canon.rs` (`make_account_guid_bytes`, `skr_leaf`);
this file's `ISS-V1-TAIL-RESIDUE` 2026-07-07 / 2026-07-04 entries (the
lance-graph-contract arms of the same residue umbrella); `.claude/v3/INTEGRATION-PLAN.md`
standing gate 3.

## 2026-07-07 — ISS-V1-TAIL-RESIDUE — RESOLVED (un-gate + default-on + both mint sites V3-routed; aiwar mints real V3)

**Status:** RESOLVED. Landed in one PR (#663):
- **`mint_for` un-gated** (`canonical_node.rs`): moved to an unconditional `impl NodeGuid` — V1 arm always available, V2/V3 arms feature-gated with a dead V1 fallback so `--no-default-features` still compiles.
- **`guid-v3-tail` default-on** (`lance-graph-contract/Cargo.toml`), per the operator ruling.
- **Both mint sites route through `mint_for`** — `ocr.rs` (classid-param-driven) and `aiwar.rs` (`CLASSID_OSINT_V3` in a normal build; V1 fallback under `--no-default-features`). No hardcoded `NodeGuid::new` remains in either production path.
- **`OSINT_GOTHAM` flipped to the V3 classid** (`soa_graph.rs`), so the projector's exact-classid filter matches the V3-minted aiwar rows; the `node()` test helper now mints via `mint_for` too.
- **`/v3-audit` check #6** forbids `NodeGuid::new(` in non-test production code.

**Correction of the mid-work diagnosis (recorded so it doesn't mislead).** An interim note in this entry claimed the aiwar V3 flip was blocked because `soa_graph::project_snapshot` reads `family` via the V1 `family()` u24 accessor. **That was WRONG** — `soa_graph`'s read path is already tail-aware (`family_of`/`identity_of` route through `classid_read_mode(guid.classid()).tail_variant` → `family_v2`/`identity_v2` for V3; proven by `v3_rows_decode_family_and_identity_via_tail_variant`). The *actual* blocker was a one-line domain-spec pin: `OSINT_GOTHAM.classid = CLASSID_OSINT` (V1), and `project_snapshot` filters rows by **exact** classid, so V3-minted rows were dropped (empty snapshot). Flipping the domain spec to the V3 classid fixed it. Consumers were V3-ready; the straggler was the domain constant, not the read path.

**Verification:** default build **854** lib tests green (V3), `--no-default-features` **840** green (V1 fallback), downstream `lance-graph` + `lance-graph-planner` check clean, fmt + clippy clean.

**Follow-up (separate, non-blocking):** `CLASSID_OSINT` (V1, `0x0700_0000`) remains a registered legacy alias — its retirement is corpus-proof-gated (W6), not part of this issue.

## 2026-07-04 — ISS-V1-TAIL-RESIDUE — two pre-existing `NodeGuid::new` (V1 `u24+u24`) mint sites must migrate to V3 (`mint_for` / V3-marked classid)

**Status:** OPEN — **MIGRATION MANDATORY** (operator ruling 2026-07-04, `E-V1-TAIL-FORBIDDEN-V3-IS-CONTENT-BLIND-1`). Deferred in *timing*, not in *obligation*; NOT to be churned into unrelated PRs. Owner: whoever next moves each output path onto a V3-marked classid.

**The residue.** The flat V1 tail `family(u24) ++ identity(u24)` is forbidden for new units; V3 is the content-blind `classid(4)+12B` facet (`E-V3-FACET-4-PLUS-12`). A read-only conformance audit of the whole repo found the V1 tail *produced* at exactly two live sites, both hardcoding `NodeGuid::new(...)` instead of the canonical `mint_for(classid_read_mode(c).tail_variant, …)` dispatch:
- `crates/lance-graph-contract/src/ocr.rs:121` — the #496 OCR→`NodeRow` keystone.
- `crates/lance-graph-contract/src/aiwar.rs:104` — the aiwar `NodeRow` builder.

Both currently target V1-default/OSINT classids, so they are **behaviorally correct today** — the defect is that they bypass the `mint_for` dispatch that is supposed to make a class's V1→V3 flip a one-line registry change. Everything else in the repo is either a test (`#[cfg(test)]`) or a legitimate legacy-compat *read* (`family()`/`identity()` fallback arms in `soa_graph.rs`, `hhtl.rs` prefix routing) — reads stay, per `I-LEGACY-API-FEATURE-GATED`; only new *mints* are forbidden.

**Resolution (mandatory, when each output path is next touched).** Route each site through `mint_for(classid_read_mode(classid).tail_variant, …)` with a V3-marked classid; add a `/v3-audit` grep that forbids new `NodeGuid::new(` in non-test code so the guard is mechanical. Blocker to note: `mint_for`/`new_v2` sit behind `guid-v2-tail` (default-off) — un-gate `mint_for` (V1 arm unconditional, V2/V3 under the feature) before pointing production mints at it.

**Cross-ref:** `EPIPHANIES.md` `E-V1-TAIL-FORBIDDEN-V3-IS-CONTENT-BLIND-1`, `E-V3-FACET-4-PLUS-12`, `canonical_node.rs` (`TailVariant`, `mint_for`, `new`), OGAR PR (canon supersession) + `D-V1-TAIL-RETIRED`.

## 2026-07-01 — ISS-Q2-CASCADE3-NIBBLE-ANCESTRY — q2 `cascade3` FNV bytes are byte-hierarchical but NOT nibble-hierarchical; HHTL routing over bake mints is sound only at whole-tier granularity

**Status:** OPEN (falsifier specified, not yet run — q2 push gate WAIVED 2026-07-02, "temporary precaution … you can unarm that"; runnable now, D-VCW-5). Owner:
q2 `cpic/src/lib.rs::cascade3` (+ any bake reusing it) vs the OGAR canon's
256=4⁴ hierarchical-codebook condition. Plan: `v3-convergence-wiring-v1.md` §5.

**The claim tension.** OGAR canon: each tier's 256-entry codebook is a 4-level
4-ary centroid HIERARCHY so a byte's nibbles are the centroid's ancestry —
`is_ancestor_of` = containment, prefix routing rigorous at nibble depth. q2's
`cascade3` derives tier byte `i` as the FNV-1a low byte of the cumulative DN
prefix at depth `i`: siblings share leading BYTES (per-tier prefix routing
holds), but a hash byte's nibbles carry NO ancestry — below whole-byte
granularity the tree structure is noise.

**Falsifier (runnable in q2 when opened):** two DNs sharing a 3-deep prefix
must show `common_prefix_depth` at nibble granularity ≈ random beyond the
shared-byte boundary (vs the 4⁴ condition's prediction of structured nibble
sharing). Confirmed ⇒ either (a) HHTL routing over bake mints clamps to tier
granularity (document the boundary), or (b) the cascade generator moves to a
hierarchical codebook (bigger change, operator call). No routing code should
assume sub-byte ancestry on these mints until this runs.

---

## 2026-07-01 — ISS-Q2-CPIC-MIRROR-DIVERGES-FROM-CPIC-V3-REGISTRY — q2's local `cpic::NodeGuid` mirror is V1-layout-parity-true but diverges from the registered CPIC-V3 read-mode on BOTH domain and tail shape

**Status:** RESOLUTION RULED 2026-07-02 (execution queued as flip P3 / D-CCF-3) — the operator ruling ("Same for cpic also under q2, which has a different domain for separation") + the triggered canon:custom flip settle this as shape (a) below, with the target classid updated by the flip: q2's CPIC class = Genetics:q2 = `0x0E:01::…` (post-flip stored `0x0E01_1000`; the pre-flip root `0x1000_0E00` normalizes `:00`→`:01`). q2 push gate WAIVED. cpic re-mints by PULLING the contract (`mint_for(classid_read_mode(…).tail_variant, …)`), retiring the local mirror + the `0x0C` domain. See `classid-canon-custom-flip-v1.md` §2/§4. Owner: q2 `cpic/src/lib.rs` +
`lance-graph-contract::canonical_node` (CPIC-V3 registry). Surfaced 2026-07-01
by a verify-the-mirror read after the V3 tenant-carve certification.

**Ground truth (read, not grepped).** q2 `cpic/src/lib.rs`:
- `NodeGuid::mint(classid, part[3], isa[3], family, identity)` builds
  HEEL/HIP/TWIG as `(part<<8)|isa` — the V3 `(part_of:is_a)` 8:8 tile. ✓ canon.
- `key16()` packs `classid·heel·hip·twig·family(u24)·identity(u24)` LE —
  **byte-identical to the contract's V1 layout** (its parity doc-comment is
  TRUE at the byte-order level). But that is the **V1 tail**, not the V3
  (`leaf·family·identity` 3×u16) tail the registry's
  `ReadMode::CPIC_V3.tail_variant = V3` reads.
- classids are `0x000C_0001..0x000C_0006` (`CID_GENE..CID_REC`) — domain
  **`0x0C`**, NOT the operator-allocated Genetics `0x0E`
  (`CLASSID_CPIC_V3 = 0x1000_0E00`), and no `0x1000` V3 gen-marker.

**So:** V3 *tiles* on a V1 *tail* under an unregistered *domain* — three
divergences from the wired CPIC-V3 read-mode. A bake produced with this mirror
will not resolve to `ReadMode::CPIC_V3` (falls to `ReadMode::DEFAULT`) and its
tail bytes read differently under the registry's V3 lens.

**Stale-brief correction (same sweep):** `soa-value-tenant-migration-v1.md`
§2.5's blocker — "q2 `osint-bake/fma.rs` calls `NodeGuid::new_v2(...)`, a
7-group API that does **not** exist" — is stale on both halves: `new_v2` DOES
exist (7 groups, feature `guid-v2-tail`, shipped + matrix-tested), and no
`new_v2` call site exists in q2 today (grep: only `cpic::NodeGuid::mint`).

**CORRECTION (2026-07-01, same session — the previous paragraph's second half
is WRONG; truncated-grep error, head_limit cut before osint-bake):** q2
`osint-bake` DOES call `new_v2` — `crates/osint-bake/src/lib.rs:606` mints the
classid-`0x0700` OSINT rows via `NodeGuid::new_v2(NodeGuid::CLASSID_OSINT, …)`
(also `:745`), and it imports the REAL contract
(`use lance_graph_contract::canonical_node::{NodeGuid, classid_read_mode}`) —
no mirror in osint-bake; only `cpic` carries the local mirror. What stands:
the brief's "API does not exist" half is stale (`new_v2` exists and q2 links
it fine). NEW observation for the same operator decision: osint-bake's OSINT
rows mint a **V2 tail** directly (`new_v2`) for legacy `CLASSID_OSINT`, whose
registered read-mode is `tail_variant = V1` — the known per-classid-legacy-tail
pending noted in `ReadMode::DEFAULT`'s docs, while its FMA bins already use the
sanctioned `mint_for(classid_read_mode(c).tail_variant, …)` dispatch. The V3
class `CLASSID_OSINT_V3 = 0x1000_0700` exists precisely for that migration.

**Resolution paths (operator decision):** (a) q2 cpic re-mints via the
contract's `mint_for(classid_read_mode(CLASSID_CPIC_V3).tail_variant, …)`
pull (consumer-preflight shape — pull, never mirror); (b) the registry gains
the `0x0C` pharmacogenomics classids q2 actually minted; or (c) the q2 POC is
declared registry-exempt (bake-only) and its parity comment is scoped to
"V1 byte layout" explicitly. No action taken pending direction.

---

## 2026-07-01 — ISS-OSINT-SYSTEM-ROOT-SLOT-VIOLATION — OGAR shipped `osint_system` at the reserved `0x0700` root slot; the lance-graph mirror canon forbids it (`CC==0x00` = domain root, reserved) — the parallel-mirror is BLOCKED on a remap decision

**Status:** RESOLVED 2026-07-02 (operator ruling, executed in OGAR PR #146) — and the resolution is SHARPER than either recorded option: "OSINT Person was a hallucination"; within the OSINT domain the low byte is APPID space applied domain-wise (`00` = the domain itself, `01` = q2 the consumer), so OSINT contributes **zero vocabulary rows**. OGAR #146 removed BOTH #145 mints (`osint_system@0x0700` AND `osint_person@0x0701`); count 67 → 65 == the mirror's 65 — the COUNT_FUSE balances with ZERO mirror-side changes, and the zero-slot invariant is untouched. Options A and B below are preserved as history; neither was taken (B came closest — its addendum's two-id-spaces reading is confirmed, but even the `0x0701` "concept" was a mislabel on q2's appid slot). See `E-CLASSID-CANON-HIGH-TRIGGERED` + `.claude/plans/classid-canon-custom-flip-v1.md` §0 for the full ruling (which also triggers the canon:custom flip).

**The violation.** The shared codebook canon (documented in `ogar_codebook.rs` module header: *"`CC == 0x00` = the domain root, reserved"*) requires every concept id `0xDDCC` to have `CC ≥ 0x01`; `0x__00` is the domain-root/default, NOT a concrete concept. OGAR main ships **`("osint_system", 0x0700)`** — `CC == 0x00`, the reserved root. `("osint_person", 0x0701)` is valid (`CC==01`, operator-frozen). The lance-graph mirror enforces the canon via the workspace-member test `codebook_has_no_duplicate_ids_or_zero_concept_slot` (`assert_ne!(id & 0x00FF, 0x00)`), so **mirroring `0x0700` fails lance-graph's own default CI** (748 pass, 1 fail). The `COUNT_FUSE` (in the *excluded* `lance-graph-ogar`) is a separate, downstream break; this one is in-tree.

**Current blast radius.** lance-graph main's default CI is GREEN (mirror still 65, zero-slot test passes; the `COUNT_FUSE` lives in the excluded `lance-graph-ogar`). Consumers vendoring `lance-graph-ogar` against OGAR-main-67 vs mirror-65 will break on the count fuse. The parallel-mirror fix is **blocked** because the obvious "+2 rows" fix trips the zero-slot invariant.

**Decision needed (operator).** Two coherent reads of `osint_system @ 0x0700`:
- **Option A — it's a concrete concept → remap.** Move `osint_system` to `0x0702` in OGAR (fresh PR; `0x0701` frozen for `osint_person`); mirror `{0x0701, 0x0702}` (count 67); update q2 `OSINT_SYSTEM_CLASS 0x0700 → 0x0702`. Canon satisfied, but a merged id moves + q2 change.
- **Option B (recommended) — `0x0700` IS the OSINT domain root/default class, not a counted concept.** This is exactly what the canon reserves `0x__00` for ("zero = fall through to the broader default"). OGAR drops `osint_system` from the *concept* `CODEBOOK`/`class_ids::ALL` (keep an `OSINT_SYSTEM = 0x0700` const documented as the domain-root class if useful); `ALL` → 66; mirror carries only `("osint_person", 0x0701)` → 66; the fuse balances at 66; q2 keeps `0x0700` as the renderable domain-default class (canon-legal: the root IS a real default class, just not a codebook *concept* row). No id moves; aligns with the user's "0x0701 is the frozen concept" framing.

Both are OGAR-side follow-ups (OGAR #145 is merged) landed in parallel with the lance-graph mirror rows, per `E-OGAR-LANCEGRAPH-MOVE-IN-PARALLEL`.

**ADDENDUM (2026-07-01, later session — ground-truth strengthening of Option B;
still the operator's decision, no action taken):** the codebase ALREADY lives
Option B's distinction. There are two id spaces aliasing in the lo u16: the
**classid space** (what nodes mint under) and the **concept-vocabulary space**
(what the codebook counts). Evidence: `canonical_node.rs` ships
`CLASSID_OSINT = 0x0000_0700` as a LIVE registered class (`ReadMode::OSINT` in
`BUILTIN_READ_MODES`) and q2 `osint-bake/src/lib.rs:606` mints real `0x0700`
rows — while the mirror's zero-slot invariant only ever governed *vocabulary
rows*. So `0xDD00` = "the ONE class per domain" (valid classid, exactly the
operator's "OSINT is ONE class") and simultaneously "not a nameable concept"
(no codebook row) — no contradiction once the spaces are named. Under this
reading OGAR's `osint_system` mint was the same move lance-graph already made,
just landed in the wrong space (a vocabulary row instead of a classid const).
Option B resolves it without deleting the idea or moving any id.

## 2026-07-01 — ISS-OGAR-OSINT-MIRROR-PENDING — OGAR #145's OSINT mint (+2 to `class_ids::ALL`) breaks the contract-mirror `COUNT_FUSE` on merge; the paired lance-graph mirror rows must land in the same arc

**Status:** RESOLVED 2026-07-02 — dissolved by the same operator ruling as `ISS-OSINT-SYSTEM-ROOT-SLOT-VIOLATION`: OGAR PR #146 removes both #145 OSINT mints (67 → 65), so the fuse balances with NO mirror rows to land; the "2 mirror rows in parallel" path below is moot (preserved as history). The fuse itself stays, per the earlier ruling ("keep the fuse — it IS the dependency contract"). · Original: **Resolution path RULED by operator 2026-07-01: keep the fuse (it IS the dependency contract enforcing OGAR↔lance-graph parallel movement); do NOT pin to a rev — "option 1" is REJECTED. Land the 2 mirror rows + `domains_agree` arm in parallel with OGAR #145 (option 2 / coordinated merge; brief transient red is acceptable — "the fuse is okay for now"). See `E-OGAR-LANCEGRAPH-MOVE-IN-PARALLEL`.** · Owner: OGAR `ogar-vocab` (PR #145) + `lance-graph-contract::ogar_codebook` mirror + `lance-graph-ogar::parity::domains_agree`. Surfaced 2026-07-01 while self-reviewing PR #624 / #145. Same cross-repo-arc shape as `ISS-OGAR-AUTH-MIRROR-DRIFT` (which took medcare CI red) and `ISS-OGAR-GENETICS-MIRROR-PENDING`; cited by `E-CODEBOOK-MINT-IS-A-CROSS-REPO-ARC`.

**READY PATCH (apply to lance-graph the moment OGAR #145 is on OGAR main; NOT to #624 while OGAR main is still 65 — that breaks #624's own fuse):** in `crates/lance-graph-contract/src/ogar_codebook.rs` add the two rows `("osint_system", 0x0700), ("osint_person", 0x0701)` to `mirror::CODEBOOK` (65 → 67); add the `(O::Osint, C::Osint)` arm to `lance-graph-ogar::parity::domains_agree` (the `ConceptDomain::Osint` enum + `0x07 => Osint` route already exist). Then `mirror::CODEBOOK.len() == ogar_vocab::class_ids::ALL.len()` (67 == 67) restored.

**The break.** OGAR PR #145 mints `osint_system` (0x0700) + `osint_person` (0x0701) into `ogar_vocab::class_ids::ALL` (+2). `lance-graph-ogar` pins `ogar-vocab = { git = ".../OGAR", branch = "main" }` (tracks main, NOT a rev), and carries the compile-time `COUNT_FUSE`: `assert!(mirror::CODEBOOK.len() == ogar_vocab::class_ids::ALL.len())` (`lance-graph-ogar/src/lib.rs:119`). The contract mirror `lance-graph-contract::ogar_codebook::CODEBOOK` currently has **65 rows with NO osint entries** (it reserved `ConceptDomain::Osint` + the `0x07 => Osint` route + a domain-nibble test, but not the two concept rows). So **the instant #145 merges to OGAR main, `COUNT_FUSE` fires `error[E0080]` in every consumer vendoring `lance-graph-ogar`** — medcare, smb, woa, etc.

**Why the mirror rows can't just be added to PR #624 now.** #624's `lance-graph-ogar` compiles against OGAR **main**, which still has 65 (osint mint is unmerged on #145). Adding +2 to the mirror now → mirror 67 vs OGAR-main 65 → breaks #624's OWN CI. The two sides are chicken-and-egg across the `branch = "main"` tracking.

**Resolution (coordinated arc, per the auth precedent):** land in lock-step —
1. OGAR #145 merges to OGAR main (ALL → 67); **at this moment lance-graph main's `COUNT_FUSE` goes red** (known transient, as with the auth mint).
2. Immediately merge a lance-graph change adding the 2 osint rows to `ogar_codebook::CODEBOOK` (`("osint_system", 0x0700)`, `("osint_person", 0x0701)`) + the `(O::Osint, C::Osint)` arm to `lance-graph-ogar::parity::domains_agree` → 67 == 67 restored.
   - The `ConceptDomain::Osint` enum + `0x07 => Osint` route already exist in the mirror, so only the 2 CODEBOOK rows + the `domains_agree` arm are missing.

**Merge-ordering decision needed from operator:** whether to (a) merge #145 + the mirror follow-up back-to-back accepting the brief transient red, (b) hold #145 until the mirror PR is staged, or (c) pin `lance-graph-ogar` to a rev instead of `branch = "main"` to decouple the cadence. Flagged to the operator 2026-07-01.

## 2026-06-26 — ISS-OGAR-GENETICS-MIRROR-PENDING — contract mirror gained `ConceptDomain::Genetics` (0x0E) ahead of OGAR; the `domains_agree` arm + OGAR side follow

**Status:** OPEN (tracked) · Owner: OGAR `ogar-vocab` + `lance-graph-ogar` · Surfaced by: CodeRabbit on #618. The same cross-repo-arc shape as `ISS-OGAR-AUTH-MIRROR-DRIFT` / `E-CODEBOOK-MINT-IS-A-CROSS-REPO-ARC`, but **domain-only** so it does not break in isolation.

#618 added `ConceptDomain::Genetics` + `0x0E => Genetics` to the contract mirror (`ogar_codebook.rs`) so CPIC-V3 `0x1000_0E00` routes Genetics (operator-allocated 2026-06-26). OGAR's `ogar_vocab::ConceptDomain` has **no Genetics variant yet**, and `lance-graph-ogar::parity::domains_agree` (`lib.rs:128-148`) still stops at `HR`/`Unassigned`. **Why it's safe in isolation (not a build break like the Auth drift):** the addition is a *domain enum variant + route*, NOT a CODEBOOK **concept** — `mirror::CODEBOOK.len()` is unchanged, so the compile-time `COUNT_FUSE` still holds, and `assert_codebook_parity` iterates CODEBOOK concept-ids (none at `0x0E`), so `domains_agree(0x0E00)` is never called. `domains_agree` is a `matches!` (never exhaustiveness-checked), so adding `C::Genetics` does not break compile either; the `(O::Genetics, C::Genetics)` arm **cannot** be added today because `O::Genetics` does not exist.

**Resolution (the coordinated arc, when Genetics concepts are minted):** (1) OGAR `ogar-vocab` adds `ConceptDomain::Genetics` + `0x0E => Genetics` + any Genetics concept rows; (2) the contract mirror's `CODEBOOK` gains the matching concept rows (keeping `COUNT_FUSE` balanced); (3) `lance-graph-ogar::parity::domains_agree` gains the `(O::Genetics, C::Genetics)` arm. Per `E-CODEBOOK-MINT-IS-A-CROSS-REPO-ARC`, those three land together, never split. Until then the drift guard correctly reflects "contract ahead of OGAR on the Genetics domain."

## 2026-06-23 — ISS-OGAR-AUTH-MIRROR-DRIFT — `0x0B` AuthStore mint broke the contract mirror's COUNT_FUSE in every consumer

**Status:** RESOLVED 2026-06-23 (this commit). OGAR `ogar-vocab` PR #110 minted the `0x0B` AuthStore family (4 concepts: auth_store 0x0B01, auth_zitadel 0x0B02, auth_zanzibar 0x0B03, auth_ory_keto 0x0B04) and merged to OGAR `main`, taking `ogar_vocab::class_ids::ALL` from 39 → 43. The paired `lance-graph-contract::ogar_codebook::CODEBOOK` mirror was NOT updated in the same arc, so the compile-time `COUNT_FUSE` in `lance-graph-ogar` (`assert!(mirror::CODEBOOK.len() == ogar_vocab::class_ids::ALL.len())`) fired `error[E0080]` (`vendor/lance-graph/crates/lance-graph-ogar/src/lib.rs:113`) in **every** consumer vendoring the OGAR git dep — medcare CI went red on `cargo build`. **Resolution:** added the 4 auth rows + `ConceptDomain::Auth` + `0x0B => Auth` to the mirror, and the `(O::Auth, C::Auth)` arm to `lance-graph-ogar::parity::domains_agree` (else the runtime `assert_codebook_parity` test panics). 43 == 43 restored; `cargo test -p lance-graph-contract` green. **Process fix (see EPIPHANIES E-CODEBOOK-MINT-IS-A-CROSS-REPO-ARC):** an OGAR concept mint is a cross-repo arc — the OGAR entry + the contract mirror + the `domains_agree` arm land together, never split across sessions. **Merge note (2026-06-23):** main landed #595 (auth sync) + #597 (PRODUCT + ACCOUNTING_ACCOUNT, OGAR #111) first; on merge this branch took main's superset `ogar_codebook.rs` (45 concepts incl. the `AppPrefix` render layer), so the auth mirror rows here are subsumed — the `domains_agree` Auth arm + this finding stand.

## 2026-06-22 — ISS-CONTRACT-APP-PREFIX-MIRROR — `contract::ogar_codebook` lacks the OGAR#97 `APP_PREFIX` / `render_classid_for` mirror, so membrane consumers must hand-stamp the hi-u16 render prefix

**Status:** RESOLVED 2026-06-22 (`claude/contract-app-prefix-mirror`) · Owner: lance-graph-contract · Surfaced by: `.claude/knowledge/ogar-consumer-preflight.md` (the consumer spellbook).

**Resolution:** `contract::ogar_codebook` now mirrors the hi-u16 APP-prefix layer — `AppPrefix` (the OGAR#95 §2 allocation table as typed data: `0x0001` OpenProject / `0x0002` Odoo / `0x0003` WoA / `0x0004` SMB / `0x0005` Healthcare / `0x0007` Redmine), `render_classid` + `render_classid_for_concept` (compose), `classid_app_prefix` + `classid_concept` (decompose). A membrane consumer (BBB-safe) now pulls BOTH halves from one source — no hand-stamped `0x000N`. Wire-compat parity test `app_prefixes_match_ogar_allocation_table` pins the prefixes against OGAR `PortSpec::APP_PREFIX`; `render_classid_composes_decomposes_and_preserves_the_concept_half` pins the `0x0005_0901` MedCare-patient worked example. Mirrors OGAR#97 (`ogar_vocab::app`), following the OGAR#98 `canonical_concept_name` precedent.

`contract::ogar_codebook` mirrors `canonical_concept_id` / `canonical_concept_name` (the lo-u16 concept pull, BBB-safe for membrane consumers woa-rs / medcare-rs / smb-office-rs) but does NOT mirror OGAR#97's `PortSpec::APP_PREFIX` + `render_classid_for` (the hi-u16 render composition: `render_classid = APP << 16 | concept`, OGAR#95 §2). A membrane consumer (BBB-barrier: contract/ontology/callcenter only — `lance_graph_ogar` forbidden) can therefore pull the shared concept but must re-derive the app prefix from the OGAR#95 allocation table by hand. Per Core-First the consumer MUST NOT hard-code `0x000N`. **Fix:** mirror the app-prefix table + a `render_classid` helper into `contract::ogar_codebook` (the `canonical_concept_name` reverse-map mirror, OGAR#98, is the precedent) so the membrane stamps from one source. Interim: the spellbook's Q5 says "stamp from the allocation table." Cross-ref: `.claude/knowledge/ogar-consumer-preflight.md` § "A Core gap this spellbook surfaces"; OGAR#95/#97/#98.

---

## 2026-06-20 — F64-TENANT-VS-F32-ENERGY — perturbation f64 narrows to the F32 `Energy` tenant; a true-f64 tenant is a canon EXTENSION (operator decision)

**Status:** RESOLVED 2026-06-20 (operator) — **NOT F64.** F32 is the fast NaN-hunt tenant (half of f64; NaN test is one integer exponent mask). The compute tenant pivots to **BF16 + AMX** (operator: "use BF16 and add_mul where possible and use amx"); the perturbation/Spain workload is deprioritised in favour of a BF16 4×4-Morton-tile Domino POC. No F64 canon extension. Cross-ref: AGENT_LOG BF16/AMX pivot.

The D1 bridge (`crates/symbiont/src/bridge.rs`) stores each bus's f64 perturbation magnitude in `ValueTenant::Energy` (F32) — "one external f64 → one internal typed tenant," per the operator's architecture. The operator's phrasing was "F64 tenant," but the canon has **no F64 tenant**: `Energy` is F32 (`canonical_node.rs:410`, `VALUE_TENANTS:481`). The f64→f32 narrowing is exact at f32 but lossy vs f64. **Decision needed:** (a) accept F32 `Energy` (the substrate's deliberate accumulator precision; no change), OR (b) extend the canon with a NEW F64 tenant — a value-slab layout addition (RESERVE-DON'T-RECLAIM; bumps `ENVELOPE_LAYOUT_VERSION`; the canon is operator-locked). Not done autonomously. Cross-ref: EPIPHANIES `E-NODE-IS-SOA-IS-KANBAN-BOARD`.

---

## 2026-05-30 — OD-CANONICAL-SPEC-DISAGREEMENT-TIER-SET — `cognitive-risc-core.md` and `wikidata-hhtl-load.md` disagree on the ProvenanceTier value-set; SPEC-OWNER decision, not Claude-session

**Status:** Open · Owner: spec author (NOT a Claude session) · Blocks: D-ARM-1 (ProvenanceTier in `lance-graph-contract`), D-ARM-2 (`Proposer` trait + `CandidateRule`), D-ARM-SYN-1/2/3 (per PR #436 follow-ups).

The four canonical specs at `.claude/specs/` disagree among themselves:
- `cognitive-risc-core.md:58` → tier set `{Curated, Extracted, ArmDiscovered, Ratified}` marked `[stable]`.
- `wikidata-hhtl-load.md:25` → tier set `{Curated, Extracted, Derived}`.
- `faiss-homology-cam-pq.md:14` → "Reasoning layer = separate indexed store, **Derived tier**" — argues `Derived` is a *separate axis*, not a tier value.
- Code today (`crates/lance-graph-ontology/src/odoo_blueprint/mod.rs:450`) → `OdooConfidence::{Curated, Extracted, Conjecture}` — a third value-set, neither matching the core spec nor the wikidata spec.

4-of-4 council reviewers (2026-05-30, recorded in `AGENT_LOG.md` + `post-438-integration-options-v1.md` §4) verdict: do NOT ship `ProvenanceTier` into `lance-graph-contract` until the spec owner reconciles. Two of the four reviewers (R2 + R4) explicitly call this a SPEC FREEZE issue, not a Claude-session decision.

**Council's recommended default if the spec owner wants one to ratify or reject:** keep the core's stable-4 as the on-byte tier; treat `Derived` as a separate orthogonal "reasoning provenance" axis (per faiss-homology + wikidata "orthogonal=beside, not mixed in"); decide `Conjecture`'s fate by either dropping it from code (it's unused per `git grep`) or mapping it to a proposer-local discovery-time label that never crosses the wire.

Cross-ref: `.claude/knowledge/discovery-origin-provenance-reconciliation-v1.md` §2.1 (full conflict matrix), §6 (OD-1/2/3), §8 (specs-on-branch correction).

---

## 2026-05-30 — OD-PROPOSER-ID-WIDTH-CHOICE — 6-bit (64 slots, u8) vs `u16` for `discovery_origin` proposer-id field; SPEC-OWNER lean exists, decision is pending

**Status:** Open · Owner: spec author · Blocks: D-ARM-1, D-MBX-A6-P3 (if `discovery_origin` rides alongside `KanbanMove`).

`cognitive-risc-core.md:62` explicitly says "Widen proposer field (steal reserved → 6 bits/64, or go u16) before surrealkv WAL hardens the LE wire format" — names two alternatives, does not pick. `cognitive-risc-classes.md:64` restates the same problem as freeze-time move N2.

The current `streaming-arm-nars-discovery-v1.md` §7.2 (committed on PR #435 branch, NOT in code) allocates 2 bits = 4 slots and is already full (AstWalker/PairStats/Aerial/Other). #436's PR-note explicitly defers the contract carrier to D-ARM-1.

Council R1 (architectural-fit): u16 (because `class_id`/N1 must ship in the same freeze pass and u16 fits both decisions). Council R3 (integration-coordination): defer this choice until #439 lands (it's mid-flight on `lance-graph-contract`).

**This issue and OD-CANONICAL-SPEC-DISAGREEMENT-TIER-SET are paired** — both touch the same byte grammar; ship them in one council-ratified pass or wait until both are settled.

Cross-ref: `.claude/knowledge/discovery-origin-provenance-reconciliation-v1.md` §6 OD-1; `cognitive-risc-classes.md` §"NON-DEFERRABLE freeze-time moves" N1+N2; PR #439 (open, kanban Phase 2).

---

> **Append-only ledger.** Every issue (bug, regression, invariant
> violation, blocker) gets a dated entry here. Entries move from
> Open → Resolved by status-flip; they are NEVER deleted.
>
> **Format invariant:** every entry starts with `## YYYY-MM-DD — `
> followed by a short title. Body is short — one paragraph of
> problem + cross-references. Full repro / fix / test details go
> in the PR or in a dedicated doc and are LINKED, not duplicated.
>
> **Mutable field:** `**Status:**` line only (Open / Resolved /
> Wontfix / Superseded). Resolved entries keep a `**Resolution:**`
> line pointing at the PR + commit SHA that fixed them.

---

## Double-entry discipline

Every issue has TWO corresponding rows, both in this file:
1. **Open section** — issue captured when first seen.
2. **Resolved section** — same entry, appended when closed, with
   `**Resolution:**` line pointing at fix.

The resolved entry cites the open entry's date as anchor. Old
"Open" entry's **Status:** flips to `Resolved YYYY-MM-DD` — it
stays in the Open section (never moved) so chronology is
preserved. The Resolved section accumulates fixes for discovery.

This is **bookkeeping discipline**, not a storage optimization:
- Open section = what broke and when.
- Resolved section = how and when it was fixed.
- Both sections keep the same row forever; the view depends on
  which section you're reading.

---

## Governance

- **Append-only.** Never delete a row from either section.
- **Mutable:** `**Status:**` and `**Resolution:**` fields only.
- **`permissions.ask` on Edit** (same rule as PR_ARC_INVENTORY).
  Write for appends stays unprompted.
- **Supersedure:** if an issue turns out to be a duplicate of an
  older one, Status → `Superseded by YYYY-MM-DD <title>`; old entry
  stays.

## Cross-references

- `PR_ARC_INVENTORY.md` — which PR shipped the fix.
- `STATUS_BOARD.md` — deliverable-level view (an issue may block
  one or more D-ids).
- `EPIPHANIES.md` — if debugging surfaced an architectural
  insight, that lands in Epiphanies; this file tracks the concrete
  fix.
- `TECH_DEBT.md` — if an issue is knowingly deferred rather than
  fixed, it moves (via cross-ref) into technical debt.

---

## Kanban Format (priority + scope on every entry)

Every issue carries:
- **Priority** — `P0` blocker / `P1` high / `P2` medium / `P3` low.
- **Scope** — which agent / deliverable / domain owns it. One or
  more of: `@<agent-name>`, `D<N>` (plan D-id),
  `domain:<grammar|codec|infra|arigraph|...>`.

Together they form the ticket tag: `[P1 @truth-architect D5 domain:grammar]`.
Agents filter by their own `@`-mention or their domain; nothing
gets buried.

## Open Issues

## 2026-05-30 — [ARM-JIRAK-FLOOR] Aerial+ proposer (D-ARM-13) ships without the mandatory Jirak Stage-A floor

**Status: OPEN.** Surfaced by the 3-savant brutal review of D-ARM-13 (iron-rule-savant #1 finding, brutally-honest-tester P1). The transcoded Aerial+ proposer (`crates/lance-graph-arm-discovery`) gates rule emission only on classical `min_support`/`min_confidence` (`extract.rs` → `rule::CandidateRule::passes`). `I-NOISE-FLOOR-JIRAK` and `streaming-arm-nars-discovery-v1.md` §4 (line 395 "This is not optional") + §11.1 declare the Jirak weak-dependence significance floor **mandatory at Stage A** — but `jirak` exists nowhere in the crate and **D-ARM-7 (the Jirak module) is Queued**. Consequence: with `c = m/(m+k)` saturating as `m = support×n` grows, a thin-but-frequent spurious rule at a 200K window becomes a high-confidence candidate → "substrate calcifies on noise." **Hard prerequisite:** D-ARM-7 MUST land before this proposer is wired into D-ARM-5 (the first stage where `(f,c)` meets a live `SpoStore` + `TruthValue::revision`). Documented honestly in `rule.rs::passes` doc + the synergy doc §4. Resolve by: implement D-ARM-7 and route `extract_rules` emission through `jirak_significance_threshold` BEFORE the classical floor.

## 2026-04-20 — [E-MEMB-1] Python↔Rust slice layouts are incompatible at the 10 kD membrane

**Status:** Open
**Priority:** P1
**Scope:** @integration-lead @truth-architect domain:membrane

PR #210's `role_keys.rs` (Rust) defines disjoint slices of the 10K VSA: Subject [0..2000), Predicate [2000..4000), Object [4000..6000), Modifier [6000..7500), Context [7500..9000), TEKAMOLO [9000..9900), Finnish [9840..9910), tenses [9910..9970), NARS [9970..10000). Python `adarail_mcp/membrane.py::DIMENSION_MAP` uses a different layout entirely: [0..500) "Soul Space" (qualia_16 / stances_16 / verbs_32 / tau_macros / tsv), dim 285 = hot_level, [2000..2018) = qualia_pcs_18. Any vector round-tripped across the two stacks will be reinterpreted by the other side's slice geometry → semantic noise, silent mis-binding.

**Impact:** blocks cross-language reconciliation for the AGI-as-glove surface (Ada σ/τ/q ↔ Rust BindSpace SoA). Until resolved, the Membrane cannot use raw 10K transfer — only serialized σ/τ/q at the REST edge.

**Secondary blocker:** E-MEMB-7 (Ada has its own 3-space incoherence between `membrane.py` 10kD, `rosetta_v2.py` 1024D Jina, and Fingerprint<256> 16K-bit — reconcile internally before Python↔Rust).

**Substrate constraint (added 2026-04-20 per [FORMAL-SCAFFOLD] reclassification):** any bridge between Python-membrane and Rust-role_keys MUST respect E-SUBSTRATE-1. An identity-map between the two layouts would violate bundle associativity — the two layouts encode different algebraic structures over d=10000. The reconciliation doc must EITHER pick one layout as canonical (likely Rust's `role_keys` disjoint slices) and re-express Python's into it, OR define a projector that preserves commutativity of bundle under translation. **A naive bit-by-bit remap is not acceptable** — it would silently break the Markov guarantee that D7 and the rest of the NARS revision stack rely on (see I-SUBSTRATE-MARKOV in CLAUDE.md).

**Next action (when queued):** author a `slice-layout-reconciliation.md` knowledge doc mapping every Python DIMENSION_MAP region to either (a) a Rust role_keys slice, (b) a dropped region, or (c) a new Rust slice to add. The doc MUST include the substrate-respect analysis above. Not yet scheduled.

Cross-ref: `.claude/board/EPIPHANIES.md` 2026-04-20 E-MEMB-1; `.claude/board/EPIPHANIES.md` E-SUBSTRATE-1 + [FORMAL-SCAFFOLD]; Deposit log E-MEMB-7; PR #210 role_keys.rs; `adarail_mcp/membrane.py::DIMENSION_MAP`; CLAUDE.md I-SUBSTRATE-MARKOV.

---

## 2026-05-13 — ndarray:master missing `hpc-extras` feature (latent downstream build break)
**Status:** Open (upstream-blocked)
**Priority:** P2
**Scope:** domain:infra D-NDARRAY-MASTER-HPC-EXTRAS

The `hpc-extras` feature on `ndarray` lives on `AdaWorldAPI/ndarray` branch `claude/burn-A1-dep-gating` (PR #116, **never merged to master**). lance-graph PR #364 (`a3c753f`) declares `features = ["hpc-extras"]` on its `ndarray` path dep — this works for us because the local `/home/user/ndarray` checkout is on the integration branch that carries the feature. **Any consumer that points at `ndarray:master` (post-#142, pre-#116) will hit `feature hpc-extras not found`** — surfaced by MedCare-rs PR #118 (doc-only investigation, merged 2026-05-13). The fix is upstream: `ndarray PR #116 → master`. Outside this session's scope; tracked here so it doesn't get rediscovered.

Cross-ref: MedCare-rs#118, lance-graph PR #364 commit `a3c753f`, ndarray PR #116 (`claude/burn-A1-dep-gating`), ndarray PR #142 (VBMI+Inf clamp, merged but does NOT add hpc-extras to master).

---

## 2026-05-16 — [W-F9-X1] Subagent Edit/Write permission isolation gap — workers must use python3 heredoc fallback

**Status:** Open
**Priority:** P2
**Scope:** domain:infra domain:cca2a @adk-coordinator
**Filed by:** W-F9 (sprint-12 Wave F sweep); originally surfaced per E-META-8

The Claude Code SDK subagent context used in sprint-11 CCA2A workers had `Edit`, `Write`, and `MultiEdit` tools blocked by permission policy. Every worker that needed to write files was forced to use `python3 << 'PYEOF'` heredocs via the Bash tool as a fallback. This pattern works but is awkward, undiscoverable, and error-prone (heredoc quoting rules differ from Edit semantics). Workaround: explicitly instruct workers in their prompt ("Edit/Write blocked — use `python3` heredocs"). Resolution requires either an upstream SDK permission fix or acceptance of the heredoc pattern as the CCA2A standard for write operations in restricted subagent contexts.

Cross-ref: EPIPHANIES.md E-META-8; `.claude/agents/BOOT.md` subagent spawn policy; sprint-11 W-D2/W-F1..W-F9 agent logs.

---

## 2026-05-16 — [W-F9-X2] Stop-hook fires on uncommitted in-flight state during subagent handoff

**Status:** Open
**Priority:** P2
**Scope:** domain:infra domain:cca2a domain:hooks
**Filed by:** W-F9 (sprint-12 Wave F sweep)

When a CCA2A subagent stops mid-task with uncommitted files, the stop-hook fires and may trigger board-hygiene checks or branch guards against a dirty state. Subsequent workers or branch switches then require a stash dance (`git stash` / `git stash pop`) before they can proceed. The workaround is: commit incrementally and stash before any branch switch. A proper resolution would require the stop-hook to detect known-active-worker state (e.g., via a sentinel file or `STATUS_BOARD.md` marker) and tolerate mid-task uncommitted changes without erroring.

Cross-ref: `.claude/hooks/` (stop-hook scripts); `.claude/board/STATUS_BOARD.md`; sprint-11 Wave D multi-step stash dance notes.

---

## 2026-05-16 — [W-F9-X3] Workspace disk quota at 91%+ during cargo builds; ENOSPC risk recurring

**Status:** Open
**Priority:** P1
**Scope:** domain:infra domain:build
**Filed by:** W-F9 (sprint-12 Wave F sweep); first hit during PR #386 rebase cycle

During the sprint-11 PR #386 cycle the workspace hit ENOSPC mid-rebase; 21 GB was freed by running `cargo clean`. The `target/` directory accumulates incrementally built artifacts from multiple workers building different crates in parallel, and the quota ceiling (~91% at the time of the incident) leaves insufficient headroom for rebase + build operations. Risk is recurring: every sprint with heavy parallel cargo work will approach the ceiling. Resolution options: (a) periodic `cargo clean` as a sprint-start hygiene step, (b) smaller per-worker `CARGO_TARGET_DIR` so artifacts don't accumulate in one location, (c) larger disk quota.

Cross-ref: PR #386 (sprint-11); sprint-11 Wave D rebase log.

---

## 2026-05-16 — [W-F9-X4] `cargo check -p lance-graph` may fail locally due to missing `protoc` binary

**Status:** Open
**Priority:** P2
**Scope:** domain:infra domain:build crate:lance-graph
**Filed by:** W-F9 (sprint-12 Wave F sweep)

`lance-encoding` (a transitive dependency of `lance-graph`) requires the `protoc` system binary for its build script. In sprint-11 this binary was absent from the default environment; W-D2 installed it manually. As a result, `cargo check -p lance-graph` (and any other command that pulls `lance-encoding`) will fail with an opaque `protoc not found` error on any worker environment that has not had the binary pre-installed. **CI is the canonical validator**; workers should note that a local compile failure of `lance-graph` may be an environment issue, not a code issue. Resolution: automate `protoc` installation in workspace setup (see TECH_DEBT.md TD-PROTOC-ENV-SETUP-1).

Cross-ref: TECH_DEBT.md TD-PROTOC-ENV-SETUP-1; D-CSV-6a agent log (W-D2 manual install); sprint-11 Wave D build notes.

---

## 2026-05-16 — [W-F9-X5] Background-worker file collisions during main-thread rebase require multi-step stash dance

**Status:** Open
**Priority:** P2
**Scope:** domain:infra domain:cca2a
**Filed by:** W-F9 (sprint-12 Wave F sweep)

During sprint-11 Wave D, a background worker had modified workspace files while the main thread needed to rebase onto updated `main`. The conflict required a multi-step stash dance: stash local changes → rebase → pop stash → resolve conflicts → continue. The pattern works but is fragile: if the stash contains large or structurally complex diffs the pop may produce confusing three-way conflicts. Proper resolution would coordinate worker commits with main-thread rebase windows (e.g., all workers commit before any rebase is initiated), or use per-worker branches that are rebased independently.

Cross-ref: Sprint-11 Wave D / sprint-12 Wave D rebase log; TECH_DEBT.md TD-PROTOC-ENV-SETUP-1 (related infra gap); `.claude/agents/BOOT.md` handover protocol.

(No other tracked open issues. New issues PREPEND here
in reverse chronological order. Format below.)

```
## YYYY-MM-DD — <short title>
**Status:** Open
**Priority:** P0 | P1 | P2 | P3
**Scope:** @<agent> D<N> domain:<tag>

<one paragraph: what's broken, where it surfaces, rough impact>

Cross-ref: <file:line or PR # or knowledge doc>
```

---

## Resolved Issues

(No resolved issues at initial commit. When an Open issue is fixed,
APPEND a copy here with the same date anchor + `**Resolution:**`
line. Old Open entry's Status flips to `Resolved YYYY-MM-DD`. Old
entry stays in the Open section for chronology.)

```
## YYYY-MM-DD — <same title as Open entry>
**Status:** Resolved YYYY-MM-DD
**Resolution:** PR #NNN (commit SHA) — <one-line description>

<original problem paragraph, verbatim>

Cross-ref: <same as Open entry>
```

---

## How to use this file

**When an issue is found** — prepend to **Open Issues** section with
today's date + `**Status:** Open` + one-paragraph description.

**When an issue is fixed** — append to **Resolved Issues** section
with the same title and date anchor + `**Status:** Resolved
YYYY-MM-DD` + `**Resolution:** PR #NNN`. Don't edit the Open entry
body; just flip its Status to `Resolved YYYY-MM-DD`.

**When an issue is a duplicate** — append a new entry in Resolved
section noting `**Resolution:** duplicate of YYYY-MM-DD <title>`;
flip Open entry to Superseded.

**When an issue is deferred knowingly** — leave it Open here but
also append a row to `TECH_DEBT.md` with cross-ref back.

## ISS-CLASSID-OGAR-DRIFT — 2026-06-20 (cont.) — RESOLVING (operator signed off; landed)
**Status:** RESOLVING — operator greenlit the realign (`AskUserQuestion`: "Realign to 0xDDCC", "Wire-compat now", "FMA = Health 0x09XX"). Landed D-OVC-1/2/4 on the jirak branch: `CLASSID_OSINT 0x0007 → 0x0700` (OSINT domain root), `CLASSID_FMA 0x0008 → 0x0901` (anatomy concept in Health, `0x0900` = Health root); minted `CLASSID_PROJECT = 0x0100` + `CLASSID_ERP = 0x0200` with `ReadMode::{PROJECT,ERP}` registered; NEW `contract::ogar_codebook` (wire-compat mirror, zero-dep — `ConceptDomain` / `canonical_concept_domain` / `classid_concept_domain` / `source_domain_concept` / `CODEBOOK` / `canonical_concept_id` / `LabelDTO::from_canonical`); `soa_graph::{PROJECT,ERP}` DomainSpecs. Drift guard test pins the shared `0xDDCC` ids; contract 710 default / 716 v2 green, clippy clean. **Dependency direction = (b) wire-compat (no OGAR↔contract dep);** the `u16` LE wire is the only contract. D-OVC-3 (cutover/version-gate audit of the *value* realign per `I-LEGACY-API-FEATURE-GATED`) remains; the classids are layout-preserving (a const value change, not a bit-layout reclaim), so no `ENVELOPE_LAYOUT_VERSION` bump. Closes when the PR merges.

## ISS-CLASSID-OGAR-DRIFT — 2026-06-20 — OPEN (needs operator sign-off)
**What:** merged `lance-graph-contract` classids drifted from OGAR `ogar-vocab`'s domain-encoded codebook (`0xDDCC`, `crates/ogar-vocab/src/lib.rs:1073` CODEBOOK + `:1163` `canonical_concept_domain`). `CLASSID_OSINT=0x0007` → `0x00` = OGAR *Reserved* domain (OSINT is `0x07XX`); `CLASSID_FMA=0x0008` → OGAR *OCR* block (FMA/anatomy is clinical → Health `0x09XX`). OGAR's own note (`lib.rs:1204-1212`): codebook id == `NodeGuid.classid` low u16, and `LabelDTO` "long-term belongs in lance-graph-contract." So contract + OGAR currently disagree on what `0x07`/`0x08` mean.
**Impact:** the contract↔OGAR↔q2 triangle has an inconsistent classid space; `canonical_concept_domain(id>>8)` mis-routes contract's OSINT/FMA; project/ERP un-minted.
**Fix (proposed):** `.claude/plans/ogar-vocab-contract-codebook-migration-v1.md` D-OVC-1..4 — host the codebook/`ConceptDomain`/`LabelDTO` in contract, classids follow `0xDDCC` (mint project `0x01XX`+ERP `0x02XX`; realign OSINT→`0x0700`, FMA→Health `0x09XX`). **Realigning merged OSINT/FMA rewrites canon (#557/#560 + CLAUDE.md canon block) → operator sign-off required** (plan §5). Origin: `CLASSID_OSINT=0x0007` minted from the early "OSINT is 0x0007" guess before ogar-vocab's `0xDDCC` layout was consulted.
