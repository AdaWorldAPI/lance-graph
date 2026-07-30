# PROBE-WORDNET-44-ACTIVATION — execution record

Operator directive: *"Feel free to document your ideas and test it, besides
wordnet makes CLAM HHTL spacial activation via 4^4"* (2026-07-30). Probe-first
rule applied: the deliverable is the falsifier, not more synthesis.

## Files

- NEW `crates/lance-graph-contract/examples/probe_wordnet_44_activation.rs`
  (the probe; zero-dep crate, ~470 lines, 5 gates).
- NEW `crates/lance-graph-planner/examples/data/wordnet/build_isa_tree.py`
  (the corpus generator; lives beside its `build_wordnet_rail.py` sibling per
  this directory's convention — generator committed, data gitignored).
- Board: `EPIPHANIES.md` prepend (`E-WORDNET-MAKES-THE-4-ARY-ADDRESS-SEMANTIC-1`),
  this file, `bf16-hhtl-terrain.md` routing row. Same commit (board-hygiene rule).

## Corpus (LOCAL-ONLY, gitignored)

WordNet 3.1 `dict` from Princeton → `/tmp/wn31/`. Generator emits
`child_offset / parent_offset / depth / lemma`: **82,192 noun synsets, one root
(`entity`), 65,292 leaves, max depth 19.** First `@`/`@i` parent only (DAG→tree;
declared, and both comparison arms see the same tree).

## What the probe asks

NOT "can k-means find hierarchy" (that is `PROBE-CODEBOOK-44`, and its real-data
leg is capped by a Base17 fold ceiling ρ≈0.26 on single words). WordNet supplies
ancestry as ground truth, so the question sharpens to: **does folding real
taxonomic ancestry into a fixed 4-ary depth-4 address preserve semantic distance
well enough that address-adjacency is a usable search prior — and does 4-ary buy
anything the shipped 16-ary `NiblePath` cannot express?**

## Gates — 5/5 GREEN

| gate | result |
|---|---|
| W1 ancestry-by-construction (shuffle falsifier) | real **+0.4938**, shuffled **−0.0356** |
| W2 monotone ladder | 15.78 → 12.76 → 11.15 → 8.69 → 7.05; strictly decreasing, spread **8.73 hops** |
| W3 spatial activation (twin-tested) | **out-of-cell** band **0.754** vs random **0.052** = **14.43×**; cover guard calibrated at 0.998 (see § W3 correction below) |
| W4 sub-nibble structure | nibble sees ONE bucket (10.55); 4-ary splits 11.15 vs 8.69 = **2.47 hops** |
| W5 fold balance | 256/256 cells, occupancy 29 / 255 / 1270 (min/median/max) |

Worked example (nearest in-band by hops): `dog` @ cell `0x5a` (`01|01|10|10`) →
`foster-brother(3), macho(3), man(3), mother's_son(3), sirrah(3), Adam(4)`;
best distance OUTSIDE the band = 4 hops.

## The two first-run failures (kept — they are part of the finding)

1. **W5 max cell 15,769 vs median 20.** Expanding each level to exactly `ARITY`
   roots before balancing reproduces the le-contract's own warning — "lacking
   proper bucket rollover … saturates silently" — in this probe's code. LPT can
   only isolate a giant subtree and hope the next level splits it; at the
   terminal level there is none. Fixed by expanding to `ARITY·24` roots
   (`GRAIN`) so the balancer has fine-grained items. **Substrate consequence: a
   4-ary fold of a real taxonomy needs an explicit granularity knob; arity alone
   does not balance it.**
2. **W1's falsifier could not falsify.** A cell-label permutation is a
   bijection, so same-cell pairs stay same-cell and the shuffle arm scored
   **+0.645** on cell identity alone — while looking like validation. Both arms
   now exclude same-cell pairs (the claim is about levels 1–3); shuffled
   collapses to −0.036. Cross-ref `E-VACUOUS-ASSERTION-IS-THE-HOUSE-STYLE-1`:
   this is that pattern inside a falsifier.

Also corrected pre-commit: the worked example printed the first six band
residents in FILE ORDER, which reads as a semantic result while being
arbitrary. Now prints nearest-by-path plus the best out-of-band distance.

## Verification

- `cargo run --release -p lance-graph-contract --example probe_wordnet_44_activation`
  → ALL GATES GREEN (re-run after fmt: still green).
- `cargo clippy -p lance-graph-contract --example probe_wordnet_44_activation
  -- -D warnings` → exit 0.
- `cargo fmt -p lance-graph-contract` → applied.
- Scoped `-p` per repo practice; the workspace-wide sweep is
  `TD-WORKSPACE-FMT-DRIFT` (measured: 1,094 hunks across 64 files in 9 crates,
  none in these files).

## Honest boundaries

- STRUCTURE probe, not a codec probe — nothing here transfers automatically to
  an embedding-trained 4⁴ codebook (Phase B, still fold-ceiling-gated).
- First-parent-only means a genuinely polysemous concept gets ONE address. The
  `dog` example shows it: WordNet's informal-term-for-a-man sense, not the
  animal. This is a real limit of any fixed-arity address, reported not hidden.
- Path length ignores information content (Resnik/Lin would weight by corpus
  frequency). Used because it is what the address approximates.
- **4-ary measured BETTER, not CHEAPER.** No traversal-cost benchmark ran.
  `RouteAction`'s four variants (Skip/Attend/Compose/Escalate = 2 bits) matching
  a rung's width is an observed consonance, NOT a wired mechanism.

## W3 correction (pre-merge, from the #875 review) — two defects, and the fixed result is ~10× stronger

Review found the metric mislabelled and the gate arithmetically mis-specified.
Both are confirmed; neither was fixed by moving a threshold.

1. **Mislabelled.** The candidate pool was sampled WITH replacement and the
   nearest 32 taken without dedup, so one synset could fill several slots —
   a weighted sampled-entry recall, not "recall of the 32 nearest neighbours".
   Both arms shared the bias so the ratio survived, but the LABEL was false.
   Pool is now deduplicated.
2. **The twin gate left a 0.018-wide window.** Both arms credited the anchor's
   OWN cell, inflating the baseline to 0.621 — which caps the achievable ratio
   at `1/0.621 = 1.61` while the fire-half demanded `> 1.5` and the cover guard
   demanded `< 0.95`. Passing that the first time was luck. After dedup it read
   1.44× and correctly FAILED.
3. **Fix = measure the claim, not lower the bar.** Sparse adjacency is about
   references landing OUT OF CELL (the operator's *"if the sentence refers to
   out of bounds meaning"*); a neighbour already in the home cell needs no band
   to reach. Out-of-cell only: **band 0.754 vs random 0.052 = 14.43×**. The null
   validates itself — random-12 scores 5.2 %, against 12/256 = 4.7 %, the cells'
   exact share of the codebook. **Home-cell inflation was MASKING the effect.**
4. **The cover guard is now calibrated.** The same measurement against a
   deliberately coarse 2-level address (6+1 of 16 cells) yields **0.998**, which
   the guard rejects — so `0.95` demonstrably separates a prior from a cover on
   this data. Satisfies the inertness rule: a threshold that never bites is
   decoration.

All-neighbour figure retained as SECONDARY and labelled saturating (0.895 vs
0.621 = 1.44×). **Lesson for the next twin gate: check that the two halves are
mutually satisfiable by more than a hair BEFORE running — compute the maximum
achievable value of the fire statistic under the silent guard.**

## Division of labour — ADDRESS vs CALCULATOR vs ORACLE (operator, 2026-07-30)

Operator ruling, mid-run: *"and CLAM to calculate, that's established"* /
*"alternative is using HHTL+ helix residue"*. This fixes a boundary the probe
must not blur:

| role | who | status |
|---|---|---|
| **ADDRESS / activation** | the 4⁴ fold — which cell, which 12-cell band | measured here, 5/5 green |
| **CALCULATOR** | **CLAM** (ndarray: build + search + `rho_nn`, 46 tests) | **ESTABLISHED — do not re-derive** |
| *alternative calculator* | **HHTL + helix residue** (place deterministic, residue stored) | the named fork; unmeasured against CLAM |
| **ORACLE** | this probe's `path_distance` LCA walk | scoring only — NEVER a runtime path |

**The correction this makes:** nothing in this probe should be read as "the
substrate computes semantic distance by walking to an LCA." It does not. The
LCA walk exists here for the same reason the tesseract-rs oracles link
libtesseract — to produce ground truth a transcode can be scored against, never
to ship. The address says WHERE to look; CLAM (or HHTL+helix residue) computes.

**Why the fork is a real choice, not a preference.** CLAM calculates over a
built tree — its cost is tree traversal plus `rho_nn` at the leaf. HHTL+helix
splits differently: PLACE is regenerated deterministically from the address
(never stored) and only the RESIDUE is read, so the calculation rides the
address rather than a second structure. The 4⁴ result matters to that fork
because a finer address makes a larger share of the answer deterministic-place
and a smaller share stored-residue — W4's 2.47 hops is exactly the granularity
the residue would no longer have to carry. **Unmeasured; the head-to-head is
the next probe, not an assertion here.**

## Follow-ons (not done here)

- **PROBE-CLAM-VS-HELIX-RESIDUE** (the fork above): same 4⁴-addressed corpus,
  two calculators — CLAM tree search vs HHTL+helix place/residue — scored
  against this probe's oracle. Gates: agreement with the oracle (both must
  reproduce it, else one is simply wrong), and cost per resolved query. The
  interesting hypothesis is that a finer address shifts work from stored
  residue to deterministic place; the falsifier is that residue size does not
  move with address granularity, which would make 4⁴ irrelevant to the
  calculator and leave it an addressing-only win.
- Feed the three-rung reference taxonomy (in-cell / sparse-adjacent /
  beyond-adjacency) into `PROBE-FREE-ENERGY-DESCENT` Phase A as the escalation
  classifier — metonymy should resolve in-band, metaphor should not. CLAM is
  the calculator there too; the descent measures F per tier, it does not invent
  a distance function.
- Phase B: nested 4×4 codebook build (two 4-way splits per 16-way level) so
  sub-nibble ancestry exists for EMBEDDING-trained codebooks too. W4's 2.47-hop
  gap is the justification; the Base17 fold ceiling is the standing blocker.
- A traversal-cost benchmark, to convert "better" into "cheaper" or refute it.
