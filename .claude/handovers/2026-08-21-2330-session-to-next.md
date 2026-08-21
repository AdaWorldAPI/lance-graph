# Session handover — 2026-08-21 (D-ACR-0 / D-ACR-1 / D-ACR-7 council)

**Why this exists:** this session took over the alpha-channel arc from the
prior session's token wall (`2026-08-21-2200-session-to-next.md`), shipped two
deliverables, and left a 5+3 council mid-flight. Assume the next session has
none of this context.

## What shipped

| branch | deliverable | state |
|---|---|---|
| `claude/d-acr-0-attention-mask-audit` | **D-ACR-0** — the attention_mask audit | pushed, no PR yet |
| `claude/d-acr-1-row-focus-mask` | **D-ACR-1** — `contract::attention_facet` | pushed, no PR yet; 14 tests, 1194 contract tests green, clippy+fmt clean |

Both carry board hygiene in-commit (EPIPHANIES prepend, STATUS_BOARD flip,
LATEST_STATE contract inventory for D-ACR-1).

### D-ACR-0 — the result is load-bearing for everything after it

`attention_mask.rs` / `attention_mask_actor.rs` are **EXISTS-UNCALLED** (three
hits workspace-wide, all non-consumers; one is a doc comment saying *"NO
AttentionMask/LRU"*) **and are a different mechanism wearing the name**: a
finished **rename register file** (`causaledge64-mailbox-rename-soa-v1.md` §4)
— wide identity → scarce narrow slot, LRU because slots are scarce, keyed by
`MailboxId`, `Vec` + linear scan, no address, no mask algebra, no trajectory.

**Do not build on them.** Piece E regrades from *"shipped; unaudited for this
use"* to *"shipped for a DIFFERENT use; uncalled; not a basis for piece D"*.
Also measured: `plasticity_residual` is write-once-zero (2 grep hits total),
`BindReply` is a NoOp handler, and the originating §4's **singleton** actor
would rebuild what the V3 mailbox ruling removed — treat "sprint-12+ work" as
superseded, not pending.

### D-ACR-1 — the basis was a REUSE, not a new type

`contract::facet::FacetCascade` already **is** `6 × 2 × u8` under
`CascadeShape::G6D2`. Zero new bytes were added. What was genuinely missing was
narrower: **a composition that is not a bitset union**. Every `union`/
`intersect` in the crate is a bit op over *field positions* (`FieldMask` u64/64,
`WideFieldMask` u8/256, `StepMask`, `rbac`) — and a bit-OR of two addresses is a
third address neither side visited.

Shipped instead: **prefix containment**, reusing `NiblePath::is_ancestor_of`'s
rule and its **explicit depth** (inferring the wildcard from zero bytes would
collide with the zero-fallback ladder, where `0` is a dormant tier). `depth`
lives OUTSIDE the 12 bytes; the wire shape stays exactly `6 × 2 × u8`.

**`FocusAxis` is `Axis0..Axis5` — a position, not a meaning.** A first draft
named them `Heel/Hip/Twig/…` and the operator caught it while raising a second
candidate reading (six ontology scopes). Both readings now live only in a test,
over byte-identical input. **This would have been the fifth homonym collision
of the arc and the first we minted ourselves.**

## The council, mid-flight — READ THIS BEFORE TOUCHING D-ACR-7

`/5plus3` was convened for **D-ACR-7** (the 59..63 reading contract). State:

- **Phase 0** SPEC v1 — written, 6 sections, inventory measured
- **Phase 1** the 5 savants — **all returned**, 39 findings
- **Phase 2** draft v2 — **consolidated**
- **Phase 3** the 3 reviewers — **reviewer 2 returned (8 PASS / 1 FIX(P2) on
  L6); reviewers 1 and 3 were still running when this handover was written**
- **Phase 4/5** — NOT DONE. **v3 is not ratified.**

Artifacts (scratchpad, ephemeral — re-derive from this handover if gone):
`dacr7-spec-v1.md`, `dacr7-spec-v2.md`.

**The sequencing is the point.** Do not ratify v3 without reviewers 1 and 3.
Do not let them see the raw savant output — draft v2 only. If a `BLOCK(P0)`
lands, return to Phase 0 rather than arguing it away in a commit message.

### What the council found that matters most

1. **Two savants converged independently on the same defect.** The spec's
   `Result`-returning resolver is the **opposite** convention from every
   shipped `ClassView` lens selector (`rail_carving`, `edge_codec_flavor`,
   `value_schema` — all infallible with zero-fallback defaults), AND a hot-path
   batch scan needs a total function. **Resolved by splitting:** the
   *declaration lookup* is total and sibling-consistent; the *projection of raw
   bits* is fallible, because that is where mismatch and stale-v1 bits actually
   live. They were two operations, conflated in v1.
2. **Code truth returned 6/6 CONFIRMS.** The v1 inventory is measured, not
   assumed — which was the one thing this council most needed to verify.
3. **The scope was wrong, and no savant could have caught it** (my spec did not
   mention V3, so no question set asked). See below.

## The operator input that changed the spec's scope (L0)

*"causaledge64 is the muscle memory / causaledgev3 for granularity."* Verified:

- `edge_v3.rs:96-103` — `CausalEdgeV3 { payload: [u8; 12] }`, const-asserted;
  *"`classid(4) | payload(12)` = the canonical 16-byte facet, the payload half"*
- `edge_v3.rs:49-50` — the SAME two fields: `[8] w_slot(6) | truth/topology
  RAW(2)`, `[9] spare/ReasoningBand RAW(3) | reserved(5)`
- `edge_v3.rs:199,206` — `truth_raw()` / `spare_raw()`
- `edge_v3.rs:16-26` — V3 **rehydrates into CE64 to reason** (`syllogize` reads
  only SPO + freq/conf + causal_mask). The muscle-memory framing is literal.

**And the V3 module doc already states D-ACR-7's problem, unsolved**
(`edge_v3.rs:86-90`): *"Which lens the ordinal was written through is **the
producer's knowledge, not the conversion's**."* The reading contract is exactly
what supplies that. So v2 spans **both carriers, one contract** — with an
honest asymmetry: the v1-provenance trap (a v1 edge with `temporal >= 512`
reads a non-zero band, `layout.rs:74-76`) does **not** apply to V3, whose bytes
were never temporal.

## Deferred with its tension stated — do not silently adopt

The operator also raised *"causaledgev3 can even use 6×2×8bit ↑n as BNN
planning equivalent"*. **Not taken into v2**, because the shipped doc forbids
that reading today (`edge_v3.rs:29-36`): *"a packed EDGE REGISTER, **NOT** a
slot-pure §3 facet … Do not read this as a content-blind facet."* Adopting it
would also be a **sixth** homonym against `attention_facet`'s `6×2×8bit`
(landed today). It needs its own deliverable and its own resolution of the
typed-register-vs-content-blind-facet contradiction.

## `ogar-loco` resonance — measured, and the answer is DON'T wire

Asked whether the same `6×2×8bit` should be wired into `ogar-loco`. It is
**already the same format, independently**: `ogar-loco/src/lib.rs:267-276` —
`Pairs = 6 × (u8:u8)`, `Triples = 4 × (u8:u8:u8)`, `Quads = 3 × (u8:u8:u8:u8)`,
`PAYLOAD_BYTES_PER_SLOT = SLOT_STRIDE(16) − CLASSID_BYTES(4) = 12`, selected
**per classid**, with `const _` asserts pinning `calls_per_lane × N == 12`.

And `lib.rs:256-258` states the decision already: *"**Mirrors** the LE
contract's `CascadeShape` … **defined locally so this crate keeps its
plug-and-play posture and takes no substrate dependency**."* `ogar-loco`'s only
dep is optional `serde`.

**So: deliberate mirroring, not overlooked duplication** — the opposite of the
five homonym collisions (those shared a *name* with different meanings; this
shares an *algebra* with identical meaning and a documented reason to stay
separate). Do not "fix" it by importing.

The real finding: the 12-unit algebra has now appeared at **six independent
sites** (`FacetCascade`, `TekamoloFacet` G4D3, `AttentionFocusFacet` G6D2,
`CausalEdgeV3`'s 12-byte payload, le-contract §3, `ogar-loco::LaneShape`), two
of them carrying explicit "mirrors" notes. That is evidence `12 = 6·2 = 4·3 =
3·4` is **forced**, not chosen.

## New plan written this session

`.claude/plans/known-unknown-handover-network-v1.md` — the operator's
*"awareness to hand over missing links in a 6×2×8bit growing BNN network"*
framing, scraped onto existing homes. Headline results:

- **`↑n` is stacking, never widening.** Two independent measurements forbid
  widening: `_LAYOUT_COVERAGE` const-asserts all 64 bits covered exactly once,
  and D-CV3-3 says verbatim *"Not in CE64 — it has zero free bits."*
- The loop's links are **individually shipped or individually designed, except
  one**: **handover** (`D-ACR-16`, NOT DESIGNED, zero precedent). The plan's
  contribution is naming what handover IS — a Hole as a kanban card moving as
  an **owned row** between mailboxes, never a message, never a shared log.
- `HoleV3 = ValueTenant 16` is **hard-blocked** on `BoardAggregates = 15` being
  resolved (contiguous discriminants), not merely queued.
- Ground truth is **27–113 usable cases**, not thousands — the gap count says
  how much work exists, the match count says how much is checkable today.

## Next steps, in order

1. **Finish the council**: collect reviewers 1 and 3, apply Phase 4 fixes
   (reviewer 2's L6 FIX(P2) — the "accepted-as-stated" bullets need the same
   file:line rigor L0/L5 use; evidence for the weakest of them is already
   gathered: no glob imports of `lance_graph_contract::*` exist, so a new
   module is additive-only), ratify v3, implement, run gates G1–G10.
2. **Open PRs** for D-ACR-0 and D-ACR-1 (both pushed, neither has a PR).
3. **D-KUH-1** (handover design) is unblocked and is the only thing in the new
   plan that can start today.

## Standing discipline this session reinforced

Content-blindness had to be defended twice in one deliverable, and the operator
caught it both times. **Once a second plausible reading of a byte exists, any
name in the substrate is a premature commitment to the first.** The five
homonym collisions of this arc (witness / nibble / hydration / attention-mask /
TrustTexture ×4) were the expensive form; naming the focus axes would have been
the first self-inflicted one.

---

## ⊘ Nachtrag (später am selben Tag): das Handover-Framing oben ist superseded

The line above — *"a Hole as a kanban card moving as an **owned row between
mailboxes**"* — is corrected by the operator: **kein Owner-Wechsel.** Handover
is a focus-of-attention entry in the NEXT rung layer of the alpha overlay at
the same address; the Hole never moves and ownership stays static (its absence
of a transfer operation is the design, not a gap). Full correction:
`known-unknown-handover-network-v1.md` §9's ⊘ block. The remaining work is an
overlay operation (`RowFocusMask::insert` in the n+1 layer — shipped, D-ACR-1),
and D-ACR-16's cascade is the stack of rung layers, the card being the focus
entry, never the row.

**⊘⊘ Second refinement, same day:** handover is TWO-ARMED by substrate nature —
static ontology → alpha-layer entry (contamination boundary); **dynamic
substrate → in place with Lance versioning** (episodic = Lance versions;
rung n+1 reads "where rung n looked" via `QueryReference::at(v, rung)` — zero
copies, replayable). The dynamic arm needs nothing built. Full table:
`known-unknown-handover-network-v1.md` §9 ⊘⊘.
