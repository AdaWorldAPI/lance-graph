# Parked designs — the #841–856 salvage (2026-07-27, forensic recovery session)

> READ BY: whoever picks up the CLAM/HHTL, WordNet, `source_registry`/evidence-
> identity, or Pearl-causality threads next. This is not new architecture — it
> is a rescue of ideas that were built, discussed, or measured during
> #841–856 and then either withdrawn (correctly, for a stated reason) or
> simply left mid-sentence when the arc's memory practice broke (see
> `.claude/handovers/2026-07-27-2114-arc-841-856-postmortem.md`). Every entry
> states what it was, why it stopped, and the precise condition that
> unblocks it — per the operator's standing rule, a falsified-for-now design
> is parked, not deleted.

## (a) What was reverted — `source_registry` + the evidence-event identity gap

**What it was.** A complete, working implementation, built across 7 commits
in #854 (`0d1e566` contract type, `f4e1576` settlement typing, `cbed5a4`
`VersionedSnapshot`, `89a80f9`/`114e58c` the `libet_offset_us` removal in
excluded crates, `4bf9fd7` "migrate BOTH `BeliefArena`s"): a stable-id-to-
dense-slot registry (`contract::source_registry`) replacing the ad hoc
64-bit `Stamp` bitset, with both the planner's and `deepnsm-v2`'s
`BeliefArena` migrated onto it so NARS revision's `disjoint()` check ran
against real registry slots instead of a folded bitset.

**Why it stopped.** Reverted whole (`b085048`, 27 files, +558/−1074) on four
converged findings, all measured or type-verified, not asserted:

1. The `disjoint()` guard exists to answer "has this evidence *event* already
   been counted" — but `SourceId` models source *membership*. One sensor
   observing the same fact twice is two events; keying disjointness on the
   source means the second observation can never raise confidence, which
   disables the most basic form of evidence accumulation, not a rare edge
   case.
2. **No canonical evidence-event identity exists in this codebase today**,
   verified against the actual types: `ClassId` is `u16`, the GUID's
   `classid` is a `u32` composite (two types sharing one word); `ClassView`
   is a late-bound projection trait, not an instance identity; `AppPrefix::
   Core` is documented "no render lens"; `LanceVersion`/`DatasetVersion` is a
   dataset snapshot, not an event. The closest existing candidate,
   `ClassId:AppId:ClassView + version`, fails 4 of 6 ingestion cases (two
   rows of one class in one commit collide; re-observing an unchanged value
   mutates nothing at all — so it can never mint an "event"; one observation
   can span many rows; an external statement mutates nothing). `NodeGuid`
   cannot substitute — its own uniqueness assertion is debug-only and its
   message admits values may be "reused."
3. A fixed-width digest (the obvious cheap fix) was **measured** safe but
   useless: 20,000 trials/cell, genuinely disjoint bases,
   `digest_a & digest_b == 0` — P(false overlap) ≈ n²/m at k=1 (6.1% at
   n=2/m=64, **63.4% at n=8/m=64**), and k>1 is catastrophic *even though it
   improves membership queries* (98.1% at n=8/m=64/k=2). Never
   false-disjoint (conservative, safe), but starves revision two times in
   three at realistic evidence-base sizes (useless).
4. A `bool disjoint()` cannot distinguish "not known to overlap" from "known
   disjoint" — the guard needs tri-state, and a Boolean silently converts
   ignorance into permission.

Codex and CodeRabbit hit the resulting 64-source ceiling from four
independent angles during review; CodeRabbit's own prescribed fix ("reuse a
bounded `SourceId`") was explicitly **rejected** because it would have made
every symptom green while calcifying the actual defect.

**The condition that unblocks it, precisely.** This design is *correct and
waiting*, not wrong. It becomes buildable the moment one thing exists:

> **An immutable receipt type that names a single evidence-admission event** —
> not a class, not a rendering, not a dataset version, but *this specific
> observation, once* — such that two calls to `observe()` on the same
> underlying fact from the same source produce **the same** receipt id, and
> two calls from different sources (or the same source at genuinely
> different moments) produce **different** ones. Candidate shapes not yet
> evaluated: a hash of `(source, subject, predicate, object, lance_version)`
> at admission time (idempotent on exact repeats, distinct on anything that
> differs); or a receipt row co-located with its evidencing `NodeRow` via the
> V3 facet register (per primer §17 — evidence lives where the SoA already
> has it, never in a sidecar).

Once that receipt type exists, the withdrawn design's remaining shape —
`EvidentialBase` as an exact set with `overflow → Unknown` and a
non-evicting ledger fallback (so no false disjointness), `OverlapKnowledge`
and `Independence` both tri-state — is ready to build largely as designed;
only the "what is a source" layer (`source_registry` itself, cleanly
separated as *attribution*, renamed `EvidenceSourceId` in the same PR) needs
no rework at all.

## (b) What was withdrawn in prose — primer §13 vs §15

**What it was.** Primer §13, "Capability confirmation — the standing wave
produces the same or better" (2026-07-27): a full two-column table scoring
every `BeliefArena` capability (belief-by-statement, revision, `Stamp`
disjointness, budget-capped ordering, premises, `close_transitive`,
adjacency propagation) against the intended V3 standing-wave substrate,
concluding "better" or "strictly better" in every row, and marking the
substrate column `[CODE-PROVEN]`.

**Why it stopped.** §14's substrate trace (four read-only lanes,
`exec-runs/trace-{A,B,C,D}-*.md`) then measured what §13 had asserted, and
found the substrate side of every comparison was **specified, not built**:
`MailboxSoA` never implements `SoaEnvelope`; no Lance-version-producing code
exists anywhere in `crates/`; `BatchWriter::cast()` has zero production
callers; `deinterlace` is test-only end-to-end with no HLC producer in the
workspace. §15 replaced §13's table with a three-column one (current LIVE
implementation / target CONTRACT status) and stated the correction plainly:
*"`BeliefArena` is de facto authoritative. It is NOT thereby architecturally
canonical."* §13 itself was **not deleted** — both sections remain in the
primer, in order, with §15 marked as the supersession, per the append-only
rule.

**Why this is a finding, not noise.** §13's error was not sloppiness; it was
scoring "a type exists" (`SoaEnvelope`, `MailboxSoA`, the trait definitions)
as "the system works" — exactly the failure this recovery session's own
handover names in §4 as "context loss," one layer up: a session asserting
a capability it had type-level evidence for but no call-graph evidence for.

**The condition that unblocks §13's actual claim.** §13's table becomes true
the moment §15's own "implementation gate" is satisfied — an uninterrupted
production path `resident belief tenant → owner-authorized mutation →
synchronous Kanban transition → ahead-firing descriptor cast → new Lance
standing-wave position → production temporal read` — with **P0 #1** (the
`MailboxSoA → SoaEnvelope → Lance write` seam) as the first missing link, per
`AUDIT-FIXLIST-2026-07-27.md`. Nothing in §13 needs to be re-derived once
that path exists; it needs to be re-verified, because it was reasoned
correctly about a system that did not yet exist.

## (c) What was deferred and never re-entered

| item | from | condition to unblock |
|---|---|---|
| **CLAM/HHTL ↔ WordNet hypernym-ancestry alignment probe (D-RCC-5)** — does common-prefix-length in the hierarchical-4⁴ centroid address track WordNet LCA depth (Spearman ρ vs a flat-256 null), and does adding the vertical lane shrink the D-RCC-6 unresolved residual? | Specified `rosetta-codebook-convergence-v1.md` D-RCC-5; named again in #851's deferred list; **#856 built its entire probe specifically to isolate this one's confound** (the Base17 fold ceiling, so a future null-result on the WordNet probe reads as "no alignment" and not as "hit the upstream 17-dim ceiling") | **The through-line — see below.** WordNet rail v2 data already exists and is verified (`build_wordnet_rail.py`, `exec-runs/rcc-wordnet-rebuild.txt` — 176,532 rows, both named anchor words pass `--verify` against live WNDB). What's missing is the 4⁴ hierarchical codebook itself: the one built earlier (`build_hierarchical`) was a 16×16 two-level structure, and #856's own body states it explicitly: *"withheld — rebuilding on the correct cascade shape before it measures anything."* Rebuild `HierarchicalPalette` as 4 levels of 4-ary, Morton-interleaved 2-bit×2-bit per level, then run the probe that already has its inputs, its null, and its confound isolated. |
| **Level-3 PROMOTION experiment** | #849, explicitly ratified as "the next PR" (induce a template from Confirmed witness cases → freeze → hide the witness → replay held-out KJV; the witness-derived rule must survive witness removal) | Never re-entered anywhere in #850–#856. Unblocked as-is — no prerequisite identified beyond scheduling it; it is the witness architecture's actual falsifier and the arc moved on to Rosetta/causality work instead. |
| **W5 gap 4 — keep-first polysemy** | #850/#851, flagged as a **BLOCKER for W11** (the hypernym walk caught `swallow → consumption` [verb sense bleeding into the noun walk] and `grape → shot` [grapeshot] — false hypernym attachments from picking the first WordNet sense rather than the contextually correct one) | Must be fixed before the Aesop-fable identity probe (W11) runs, or W11's results are silently poisoned by the same defect `build_wordnet_rail.py` v2 already fixed for the *rail* (12.76% wrong-sense rate) but which the *walk* logic doesn't yet inherit. |
| **`ISS-PEARL-VOCABULARY-WITHOUT-PEARL-MECHANICS`** | #853, the causality audit — "nothing severs" (no mechanism-disabling, no descendant recomputation), Recipe 31 (`Icr`) is a stub wearing a Pearl label, four kinds of cause share one untyped edge | #854 shipped the **typing** half (`causal_audit`: kind/locus/domain/scope, receipt-ledger support) but explicitly did NOT build severing/mechanism-disabling or fix `Icr`'s stub body — the audit-before-build ruling was honored for the typing, not yet for the mechanics. Unblocked: the typed edge now exists to classify what a severing operation would act on; building `do(X=x)` mechanics is the next PR, gated on nothing new. |
| **`nars::InferenceType` contract drift** | #853's causality audit — three recorded copies of this type have *different variant sets* (no `Intervention`/`Counterfactual` in the current enum, despite `CLAUDE.md`'s own I-LEGACY-API-FEATURE-GATED discussing `InferenceType::Counterfactual` at mantissa −6) | A contract-integrity hole, not a design question — needs a reconciliation pass across the three copies, unblocked, not started anywhere in #854–#856. |
| **Substrate write path — P0 #1/#2 (`AUDIT-FIXLIST` row 1/2)** | #855's §12 trace | `MailboxSoA` never implements `SoaEnvelope`; no HLC producer anywhere in `crates/`. This is the single dependency chain everything in (a) and (b) above is gated on. Sequencing per the fixlist: 4 (MetaWord width mismatch) → 3 (resident tenant for rung/contradiction/premises) → 1 (the write seam) → 2 (the read). |
| **L5 γ-fold validation** | #855's `probe_l5_fisherz_amortization` | Honestly reported as NOT validated — the probe folded random rows against the fold's own CLAM-family precondition (a rig defect, not a finding about the fold). Needs a rig that respects the precondition before the γ-fold can be scored either way. |
| **HEEL pruning power on a full-width tier** | #855's `probe_furnace_amortization` | Measured: same-HEEL pairs are 20% closer than random (locality proven, falsifier fires both directions) but pruning power is weak on the thin 17-dim/6-subspace rig (99.7% survivors at the t/4 band) — the "95%-skip HHTL" claim needs a full-width tier to actually test pruning, not just locality. |
| **`impl Distance for [u8;6]`** | #855, measured ρ −0.0030 (noise), zero production consumers by grep | One-PR deletion or rename (`ISS-CONTRACT-DISTANCE-IS-THE-FORBIDDEN-UMBRELLA` §G), unblocked, purely mechanical, not done anywhere in the traced range. |
| **Lemma-key vs. `tongue` anchor trade-off** | #852 | Lemma-key coverage lift (39.2→43.0% German) broke the `sprache`/`sprachen` anchor via the `-chen` suffix rule; shipped opt-in, default OFF. Needs a smarter suffix rule (or a documented accepted trade-off), not re-touched since #852. |
| **`heel_threshold: 50.0` is inert** | #852, measured (max sub-distance in the fixture is 25.5, so the threshold never binds) | Mechanical: either lower the threshold to something the fixture can exercise, or replace the fixture with one whose distances exceed it — either way, a one-line change not made anywhere in #853–#856. |

## The question this salvage exists to answer

> **Of everything parked in #849–#856, which single item was the lost
> session most excited about — and what was it about to do next?**

**The CLAM/HHTL ↔ WordNet centroid-ancestry alignment probe (D-RCC-5).**
The evidence is behavioral, not tonal — it is the one item the arc kept
building toward without ever running it, across four separate PRs:

1. **#849/#851** name it explicitly as the deferred item, twice, in nearly
   identical language ("does centroid ancestry track hypernym ancestry,
   against a flat-256 null").
2. **#852** spent an entire 44-commit arc building its *inputs* — the
   Rosetta lanes, the WordNet rail v2 rebuild (correcting a 12.76%/33.84%
   wrong-sense rate the v1 rail carried), the lane codebooks — none of which
   the probe strictly needs on its own, but all of which feed the same
   convergence plan (`rosetta-codebook-convergence-v1.md`) whose D-RCC-5
   section is this exact probe.
3. **#853**'s causality audit and #854's typed causal edge are a detour into
   a different (also real) gap, but neither touches WordNet or HHTL at all
   — the arc visibly changed subject.
4. **#856**, the very last PR before the session's memory failed
   completely, is not about WordNet on its surface — it measures the Base17
   fold ceiling. But its own body states the reason in one sentence: the
   WordNet falsifier "is written but withheld... rebuilding on the correct
   cascade shape before it measures anything," and the fold-ceiling
   measurement exists specifically so that when the WordNet probe *does*
   run, a null result will mean "no alignment" and not "hit the known 17-dim
   ceiling." **#856 is the last piece of scaffolding for a probe the session
   never got to run.**

Read as a single arc, #849→#856 is not six disconnected PRs — it is one
session building, piece by piece, every precondition a single falsifier
needs (a corrected WordNet rail, isolated confounds, a stated null
hypothesis, a named cascade shape to rebuild on) and then running out of
context one rebuild away from executing it. The next session's highest-
leverage move, if it does only one thing from this file, is: rebuild
`HierarchicalPalette` as 4⁴ (not 16×16), and run the probe that has been
waiting, fully specified, since #849.

This item is **scheduled, not silently re-parked a third time**: it is
named here with its exact blocking condition (the 4⁴ rebuild) and its
exact falsifier (Spearman ρ of prefix-length vs. LCA depth, against a
flat-256 null, plus the D-RCC-6 residual-shrinkage test). Whoever reads this
next should either run it or state explicitly why not.
