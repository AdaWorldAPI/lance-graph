## 2026-08-11 — E-THE-DOCTRINE-DOC-EXISTED-AND-I-NEVER-READ-IT-1

**Status:** FINDING `[G]` (operator-directed; authority is
`.claude/knowledge/helix-cartesian-vs-fisher2z.md`, in this repo, with a
`READ BY:` header naming this exact kind of session).

**Corrects E-A-CORRECTION-IS-A-CLAIM-AND-CARRIES-A-CLAIM-S-BURDEN-1 (below) —
the third link in that same chain, and the entry it corrects is its own rule
firing again.** The `Pair48` successor shape that entry recommends is
**WITHDRAWN, not deferred.** Full regrade:
`.claude/knowledge/weather-normalized-substrate.md` §12.12.

**The doctrine doc for orientation codecs exists in `.claude/knowledge/`, its
`READ BY:` header says *"ANY session that encodes/decodes/renders an
orientation, normal, or direction"*, and the entire helix arc never opened it** —
not before the design, not during the envelope audit, not through four
successive corrections. Reading the primary source (`crates/helix/src/*.rs`) was
mistaken for reading the doctrine. `residue.rs` states the byte layout
truthfully; it does **not** say which field is metric and which is render, nor
that the sign already completes the sphere. That is precisely what the knowledge
doc carries, and what a `READ BY:` header is *for*. `CLAUDE.md` § *Consult,
don't guess* already orders it: card → **knowledge doc** → board → *then* source.

**What the doc settles, against the audit's inferences (the FACTS all stood; the
INFERENCES were wrong):**

1. **`Signed360` alone is a COMPLETE full-sphere direction** — *"you do NOT need
   a second 'pos' helix to complete it"* (`helix-cartesian-vs-fisher2z.md:77-82`);
   the sign partition is what makes it whole. Hence no pair lane, hence the
   withdrawal.
2. **`ResidueEdge` / `rim` is the METRIC carrier, not a render input** — *"never
   run the rim's atanh/tanh to recover a direction"* (`:83-86`). Direction lives
   in `(polar, azimuth)`. The operator said this first: *"ResidueEdge is
   turbovec, nobody was asking you to use turbovec."*
3. **The crate's *"no free 2-DOF direction codec"* means no *helper*, not no
   *capacity*.** `azimuth` is *"`n·φ mod 2π` mapped to `[0, 65536)` over the
   full 360°"* (`residue.rs:85`). What is genuinely absent is `from_normal`;
   encoding is a **nearest spherical-Fibonacci search** with a worked reference
   pair (`:90-93`, q2 `scratch-fma/helixbake`).
4. **The metric hazard I "found" is crate-documented, not a discovery** —
   `distance_heuristic` names its own failure mode as *"the raw-azimuth 2π
   wrap"* and forbids it for CAKES bounds (`residue.rs:52-59`). The real shape
   is a **split, not a defect**: rim = L1-metric (`DistanceLut`, triangle
   inequality); azimuth = circular render carrier, deliberately not L1. My error
   was routing a bearing through the *metric* field, then reporting the
   consequence as a property of the codec.

**Also settled — the name.** `helix360` **does not exist and never did**
`[G-absence]`: pickaxe over full history (all refs + unreachable objects) returns
12 blobs, all authored by that session, **zero deletions, zero blast radius**;
ndarray and the `lance-graph2` backup carry none. The symbol is **`Signed360`**.
A hunt for a deleted artifact that was only ever a mis-remembered name.

**What still stands from the entry below:** the no-per-value-lane-reading-selector
finding (`[G-absence]`, 15 hits / zero writers / zero decoders), the
`Signed360::sign()` all-zero dormant-lane defect (still filed, still unfixed),
and — sharpened by this entry — its own **rule**.

**Rule (unchanged, now three links deep):** a correction is a claim and carries a
claim's burden. **Extension:** before correcting a domain claim, check whether
the domain already has a `READ BY:` doc. If it does, that doc is a **mandatory
read, not a suggestion** — four corrections written without it were four
corrections written blind.

