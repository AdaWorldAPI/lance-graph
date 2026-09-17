# three-carrier-blast-radius-v1 — how far do the three prefix-fold carriers reach?

**Status:** PROPOSAL (2026-09-17). Gates only; no code change is authorized by
this document.
**Doctrine:** `.claude/knowledge/three-prefix-fold-carriers.md`
**Board:** `E-THREE-CARRIERS-THREE-FOLDS-1` (EPIPHANIES entry 17)
**D-ids (STATUS_BOARD § three-carrier prefix folds):** the four census passes
below are **D-TCF-5**; the carrier-2 probe + rewrite they gate is **D-TCF-4**.
The shipped rows this plan builds on are D-TCF-1 (probe), D-TCF-2 (revert) and
D-TCF-3 (doctrine).

## Why a blast-radius pass at all

The defect that produced this plan was not a bad optimization — arm C is
competently written and correctly tested. It was a **scope error**: a result
true of one carrier was applied to another because the two share a vocabulary.
That failure mode is invisible to every gate the workspace already runs. Clippy
does not know what a carrier is; the differential test passed (both forms
compute the same answer); the board entry was well-formed. **The only thing
that catches it is knowing how far each carrier reaches, and where they touch.**

So the deliverable is not "a faster fold". It is a **carrier census**: for each
of the three, the exhaustive set of types, folds, call sites, tests, docs and
cross-repo consumers — and, critically, the set of places where one carrier's
output feeds another's input, because those seams are where a future transfer
will happen.

## Method, in four passes

Each pass is `grep FINDS, reading DECIDES` (CLAUDE.md P0): grep produces
candidates, every candidate is **opened** before it enters a census row, and a
negative result is never recorded without opening the place the thing would
live. Delegate the sweeps to Sonnet (grindwork: "find every site that X"),
keep the classification on the main thread (accumulation: "is this site
carrier 2 or carrier 3").

### Pass 1 — type census (per carrier, exhaustive) — D-TCF-5

For each carrier, the type(s) that ARE it, and every constructor / accessor /
`from_*_bytes` / `to_*_bytes` on them. Output: a table of
`type → file:line → carrier → is it a fold, a lens, or a materialization`.

Seed greps (candidates only):
- C1: `identity_plane_at`, `WORDS_PER_FP`, `IdentityPlane`, `DistanceMeans`
- C2: `NiblePath`, `common_prefix_depth`, `HhtlKey`, `prefix(`, `MAX_DEPTH`
- C3: `FacetCascade`, `FacetTier`, `hi_chain`, `lo_chain`, `shared_prefix_tiles`,
  `row_match_mask`, `CascadeShape`

Known-present trap: `perturbation-sim/examples/{outage_over_hhtl_hops,
basin_placement_learning}.rs` each define a LOCAL `common_prefix_depth` over
`HhtlKey`. Two more carrier-2-shaped folds, neither routed through `NiblePath`.
They must be classified, not skipped as examples.

### Pass 2 — fold census (the thing that actually moved)

Every site that computes a prefix, an LCP, a shared depth, a Hamming distance,
or a "how far do these agree" answer — regardless of what it is named. Grep
`trailing_zeros|leading_zeros|count_ones|popcnt|common_prefix|shared_|_distance|
is_ancestor|prefix_depth`, then open each and assign a carrier.

**Gate G-FOLD:** every row in this census names its carrier. A row that cannot
be assigned is a finding — it means a fourth carrier exists, or that a site
mixes two.

### Pass 3 — the seams (highest value; do not skip) — D-TCF-5

Where does one carrier's output become another's input? Three candidate seams
are already visible and each must be opened and characterized:

- **S1 — GUID → `NiblePath`.** `NiblePath::from_guid_prefix` / `_v3` fold a
  128-bit GUID (byte-addressed, carrier-3-shaped) into a packed `u64`
  (carrier 2). This is a *deliberate* carrier change and is the seam most
  likely to invite "the fold is the same on both sides".
- **S2 — `NiblePath` → `mailbox_scan`.** `mailbox_scan.rs:263` calls carrier
  2's fold while `mailbox_soa.rs` supplies carrier 1's planes to the same
  scan. **One file consuming two carriers is exactly the confusion surface**
  that produced this plan; `identity_plane_at`'s own doc comment points at
  `DistanceMeans::Hamming`, and that pointer was the thread that unravelled it.
- **S3 — facet ↔ SoA row.** The 12-byte facet register and the 480-byte value
  slab live in one 512-byte row. Any code that reads both in one sweep is a
  seam.

**Gate G-SEAM:** each seam gets one sentence stating which carrier owns the
input, which owns the output, and what the conversion costs. A seam with no
stated cost is unassessed, not free.

### Pass 4 — outward radius (docs, tests, cross-repo)

- Board + knowledge + plan mentions of each fold (the words travel further than
  the code; entry (16) is the proof).
- `ndarray::simd` masking-ops docs that cite the facet result.
- Cross-repo consumers of `lance-graph-contract`: in-tree (planner, callcenter,
  smb-bridge), `ladybug-rs`, `lance-graph-java` (`lgj-abi` imports
  `lance_graph_contract::facet` — the G11 fence names it). **`hi_distance` /
  `lo_distance` have zero in-tree callers; the Java side is where an
  out-of-tree caller would be**, and it is checkable — the G11 fence test
  enumerates exactly what crosses.

## Falsification gates (the plan is only real if it can fail)

| gate | passes when | fails when |
|---|---|---|
| **G-EXH** | the type census is exhaustive by construction — a directory walk, not a file list someone chose | any row was found by intuition rather than by the walk |
| **G-FOLD** | every fold site carries a carrier assignment | a site resists assignment (⇒ fourth carrier, or a mixed site) |
| **G-SEAM** | every seam states owner-in / owner-out / conversion cost | a seam is listed as "just a cast" |
| **G-PROBE** | each carrier that gets a *change* has a probe on **that** carrier, red-then-green | a change lands citing another carrier's number — the original defect, recurring |
| **G-ZERO** | a claimed "no callers" was verified by opening the place a caller would live, not by an empty grep | a negative grep is the only evidence |

## Sequencing, and what is explicitly NOT authorized

1. Passes 1–2 (census). Read-only.
2. Pass 3 (seams). Read-only. **Highest value — if only one pass runs, run this.**
3. Pass 4 (outward radius). Read-only.
4. *Then* the carrier-2 probe — **D-TCF-4** (four arms, same harness, `NiblePath` workloads
   incl. `EMPTY`, unequal depths, ancestor pairs, full-16 agreement).
5. *Then*, and only on a green probe, the carrier-2 rewrite.

**Not authorized by this plan:** any rewrite of carrier 2 before step 4; any
change to carrier 1 (it is correct and this plan produces no evidence about
it); any widening of `FacetTier`'s `u8:u8` into a `u16` to "make the mask
cheaper" — that is a canon violation (`E-V3-FACET-4-PLUS-12`), and it is
precisely the shape a carrier-3 masking argument tends toward. **The fold
adapts to the layout; the layout never adapts to the fold.**

## Cost

Passes 1–4 are one session with 3–4 Sonnet sweeps and main-thread
classification. The carrier-2 probe reuses `facet_axis_lcp_probe.rs`'s harness
(SplitMix64 seed `0x9E37_79B9_7F4A_7C15`, min-of-7, oracle-first, anti-vacuity
gates) — the arms change, the scaffolding does not. That reuse is itself a
finding worth keeping: **the harness is carrier-agnostic even though the
folds are not.**
