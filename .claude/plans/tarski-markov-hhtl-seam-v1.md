# TARSKI-MARKOV-HHTL-SEAM-1 — a stream-order reading bound to a tree address

> Status: DEFERRED INTEGRATION ITEM. Named, not built. Gated on
> `BELIEF-ABI-RESTORATION-1` Step 2 (operator ruling on the Step 1 residue)
> landing first — this plan is the eventual Step 3 probe's design surface,
> not a replacement for it.

## The gap this plan names

`BELIEF-ABI-RESTORATION-1`'s Step 1 audit (`.claude/plans/belief-abi-restoration-v1.md`,
merged in #1006) found that the operator-ruled delegation for
`Belief.rung`/`stamp` — *"rung = HHTL tree depth, stamp = accumulate from
children and siblings, inherit from parent"* — names a mechanism that does
not exist in code. A follow-up asked whether either of the two closest
shipped candidates already IS that mechanism. Grounding both
(`.claude/board/EPIPHANIES.md`,
`E-STREAM-ORDER-VS-PREFIX-TREE-NEITHER-ACCUMULATES-1`) found: no.

```
  Candidate A: deepnsm-v2::wave::WitnessStream (G24N4 / Markov)
      events: Vec<(u64, CausalWitnessFacet)>  — flat, append-ordered
      ground_at / resolve_at: signed-offset walk within a VERSION window
      explicit refusal to accumulate (E-NO-BUNDLE-STANDING-WAVE-1):
        "there is no accumulator and no shared register"
      → a TOTAL ORDER over time. No tree.

  Candidate B: AttentionFocusFacet (AriGraph / HHTL basins)
      covers / common_prefix — coarse→fine PREFIX containment
      answers "is A an ancestor of B" / "what do A, B share"
      never "what do B's children contribute to A"
      → a PARTIAL ORDER over address. No fold.
```

Neither candidate accumulates. The operator's ruling therefore describes a
mechanism that must be BUILT by composing the two, not selected from
what already exists.

## The seam (sketched, not designed)

```
  HHTL address (Candidate B)
        │  gives: which subtree, what "depth" means, ancestor/descendant
        ▼
  a per-address WitnessStream-shaped window (Candidate A's machinery,
  rebased from "version order" to "address-scoped stream order" —
  the events visible to a fold are the ones IN that subtree, ordered
  however the fold needs, not necessarily by version)
        │  gives: a bounded, single-owner sequence to walk/reduce
        ▼
  an accumulate operator over that sequence
  (children's registers → parent's register; NOT `MergeMode::Xor`,
  per I-SUBSTRATE-MARKOV — magnitude-side accumulation is `vsa_bundle`
  or an equivalent commutative fold, never raw XOR)
        │
        ▼
  Belief.rung / Belief.stamp AS A PROJECTION of the folded result,
  never a separately-stored field
```

Every arrow above is a genuine design decision, not a restatement:

1. **Is the per-address window literally `WitnessStream` re-scoped, or a
   new type?** `WitnessStream::window_range` already takes an arbitrary
   `VersionRange` — whether an "address range" (an HHTL subtree) can be
   expressed as a re-purposed version-like ordering, or needs its own
   window abstraction, is open.
2. **What is the fold?** `carried_awareness`, the Horner sum in
   `rail_geometry.rs:183`, and `causal_audit`'s append-only history were
   each checked in Step 1 and shown to be a DIFFERENT mechanism from
   "children+siblings accumulate, inherit from parent" — none is a
   template to copy. The fold itself is undesigned.
3. **Does `rung` become a read-time query over the fold, or does the fold
   get materialized per address?** The zero-copy-warden's law ("zero
   copy is a law without escape hatches... the array itself is a
   ClassView projection") argues for read-time; but a fold that walks an
   unbounded subtree on every read is a real cost question, not yet
   measured.
4. **Does this replace `BeliefArena.rung: u32` entirely, or does `rung`
   become `support_ceiling()` over a resident G24N4 register as
   `PROBE-TARSKI-SIGNED-WITNESS-1` already demonstrates (PR #1007), with
   the HHTL-address binding as a SEPARATE, later-composed axis?** These
   are not obviously the same migration. The Tarski-signed-witness probe
   proved the SIGN/MAGNITUDE reading works over a resident register with
   no address at all (a flat `Dock` array, no `FacetCascade` address
   minted per belief). Binding that register to a real HHTL address is
   an ADDITIONAL step this plan is scoping, not one already done.

## What is already proven and can be reused as-is

- `PROBE-TARSKI-SIGNED-WITNESS-1` (PR #1007): the G24N4 signed reading
  (`SupportedBy`/`Contradiction` as depth, not magnitude) reproduces
  `Belief::rung` exactly on the positive lane, and retains what the
  shipped `admit_derived` CHOICE law provably discards. This is the
  EVIDENCE-content half of the eventual fold's input — settled, reusable.
- `PROBE-FOUR-PLANE-CAUSAL-MEDIUM-1` (PR #1007): proves the four-plane
  separation (WHERE / WHAT / WHICH-LENS / WHY) composes over disjoint
  lanes of one resident row without any plane auto-deriving another. This
  is the discipline the eventual fold must respect: an accumulate
  operation over the WHY plane must not silently rewrite the WHAT/WHICH
  planes, exactly as `PROBE-FOUR-PLANE-CAUSAL-MEDIUM-1`'s FP3 already
  requires for a single row.
- The A9 "loci, not magnitudes" law and the DOCK/ROUTE separation
  (`E-TYPE-COMPLEXITY-EXPOSED-A-MEMORY-ABI-ESCAPE-1`) bound the shape any
  new reading must take: classid chooses the reading, the route chooses
  the traversal, the bytes never change shape.

## What this plan does NOT do

- Does not mint a tenant, a classid, or a `CascadeShape` variant.
- Does not implement the fold. This is a design surface, not a probe.
- Does not resolve open questions 1-4 above — they are the actual content
  of a future Step 3, and should be answered by a probe with pre-registered
  falsifiers (per this repo's own standing falsifiability rule: "a guard
  that fires on everything carries exactly as much information as one
  that never fires" — an accumulate-fold that always reproduces
  `max(premise rungs)+1` regardless of tree shape has proven nothing).

## Falsifiers a Step-3 probe against this seam must clear

- **SF1** — the fold, bound to a real per-belief HHTL address, reproduces
  `BeliefArena`'s CURRENT `rung = max(premise rungs)+1` on the positive-only
  corpus (parity oracle, same shape as `PROBE-TARSKI-SIGNED-WITNESS-1`'s
  A1/A2 gates).
- **SF2** — the fold survives a NON-trivial tree shape (siblings with
  different depths, a node with zero children) without collapsing to a
  constant — the anti-vacuity discipline this repo's `CLAUDE.md`
  ("the falsifiability rule") already mandates for any new guard/fold.
- **SF3** — accumulation uses a commutative operator (`vsa_bundle` or
  equivalent), never raw XOR on the magnitude side, per
  `I-SUBSTRATE-MARKOV`.
- **SF4** — binding an address to a belief does not retroactively change
  any OTHER plane of that belief's row (FP3's discipline, generalized).
- **SF5** — if SF1-SF4 cannot all be cleared, the probe reports the EXACT
  missing bit/field/route invariant (per the charter's own instruction:
  "report the exact falsifier if it cannot reproduce the arena's current
  rung/stamp results. Do not force it.") rather than silently degrading
  the parity requirement.

## Trigger to promote this from "named" to "active"

Do not start this until `BELIEF-ABI-RESTORATION-1` Step 2 (the operator
ruling on the Step 1 residue table — `premises`, `stmt`/`truth`, and this
tree-overlay mechanism, itemized) has landed. This plan is input to that
ruling, not a substitute for it.
