# Known-unknown handover network — v1

> **Status:** PROPOSED. No code. Every "exists" claim below was verified by
> reading the file named, this session; every "absent" by a grep that returned
> nothing. Register-before-code, per the dialectic build order.
>
> **What this plan is:** the operator's 2026-08-21 framing — *"an awareness to
> hand over missing links in a 6×2×8bit growing BNN network as a
> self-organizing AGI-aspiring network … that includes expanding CE64 59..63 as
> a known-unknowns exploring brain plasticity"* — scraped onto homes that
> already exist, plus the short list of what genuinely does not.
>
> It mints no type. It is the **connective plan** between two arcs that were
> being built separately: `alpha-channel-rung-overlay-v1.md` (the attention
> overlay) and `dismech-causality-v3-v1.md` (the Hole). The claim of this plan
> is that they are one loop, and that exactly one fibre of it is missing.

## §0 — The fence, first, because it is the load-bearing constraint

**"Expanding 59..63" cannot mean widening.** Two independent measurements
forbid it:

1. `causal-edge/src/layout.rs:93-111` — `_LAYOUT_COVERAGE` const-asserts that
   all 64 bits are covered **exactly once**. There is no slack to take.
2. `.claude/plans/dismech-causality-v3-v1.md:503` (D-CV3-3), verbatim:
   *"`awareness_state` ⟂ `unknown_kind`. **Not in CE64 — it has zero free
   bits.**"*

So the operator's own notation is the answer: **`↑n` is stacking, not
widening** — the canon's *"scale is the next cascade level, never
field-widening"* applied to the awareness axis. Every deliverable below adds a
**level**, never a bit. Any future session reading "expand 59..63" as "take
some spare bits" is reading it wrong, and this section is the record.

## §1 — The three-stage expansion, and where each already lives

| stage | what it buys | carrier | state (verified) |
|---|---|---|---|
| **1. readable** | the 5 bits stop being ambiguous — which lens wrote them, and is the provenance trustworthy | D-ACR-7 reading contract, spanning `CausalEdge64::{truth, reasoning_band}` **and** `CausalEdgeV3::{truth_raw, spare_raw}` | **in council now** (spec v2 consolidated) |
| **2. discriminating** | a known-unknown is *distinguishable* from a weak known | F5: the band **grades**, the **witness reference discriminates** — `WitnessKind` points at a Hole | `WitnessKind` is in D-ACR-7 spec v2; its Hole target is stage 3 |
| **3. granular** | the unknown gets its own row, with lifecycle | `HoleV3` as `ValueTenant = 16`; `CausalEdgeV3`'s 12-byte register (bytes `[10..12]` still dormant) | **BLOCKED** — see §4 |

**Why stage 1 is not optional plumbing.** `causal-edge/src/edge_v3.rs:86-90`
already states the gap in shipped code and leaves it open:

> *"`w_slot` / truth / spare are preserved as **RAW ORDINALS** … **Which lens
> the ordinal was written through is the producer's knowledge, not the
> conversion's**."*

A network that cannot tell which lens wrote a bit cannot tell a known-unknown
from a low-trust known. Stage 1 supplies exactly the producer knowledge the
conversion structurally cannot carry.

## §2 — The loop, with every link's real status

```text
  pothole OPENS         QueryReference::at(v, rung) bounds what is derivable
       │                 → "not yet knowing" is manufactured honestly
       ▼
  STAMP                 with_reasoning_band() — explicit, never derived (F2)
       │
       ▼
  DISCRIMINATE          WitnessKind → the Hole (awareness × unknown_kind)
       │
       ▼
  LOCATE                RowFocusMask — where attention actually went
       │
       ▼
  HAND OVER             the Hole as an owned row in another mailbox
       │                 (one-writer-per-mailbox preserved)
       ▼
  EXPLORE               the 14 delta_conf-capable recipes
       │                 (the 20 mute ones are eigenvalue 1 by construction)
       ▼
  CLOSE                 Revision fires; the pothole-open SPAN is the measurement
       │
       ▼
  REWIRE                plasticity (CE64 bits 50-52 / V3 byte [3])
                         growth = minting new rows, never widening a field
```

| link | home | state |
|---|---|---|
| pothole opens | `temporal.rs` `QueryReference::at` | **SHIPPED**, unbuilt as a live horizon (D-ACR-15) |
| stamp | `edge.rs:1056` `with_reasoning_band` | **SHIPPED**; the ONLY writer (F2) |
| discriminate | `WitnessKind` (D-ACR-7 v2) | in council |
| locate | `contract::attention_facet::RowFocusMask` | **SHIPPED 2026-08-21** (D-ACR-1) |
| **hand over** | — | **THE MISSING FIBRE — see §3** |
| explore | `recipe_kernels` `delta_conf` (14/34) | **SHIPPED**; filter is D-ACR-7's acceptance condition |
| close | `RecipeInference::Revision` + span | design in `alpha-channel` §3p; probe is D-ACR-10 |
| rewire | `PLAST_SHIFT = 50` (`layout.rs:37`) | **SHIPPED as a field.** Measured: no external consumer reads `plasticity()` for rewiring — `high_heel.rs:236,589,845` reads it for basin state, which is a different use |

## §3 — The one missing fibre: handover

Everything else above exists or is designed. **Handover does not**, and the
plan that would own it says so plainly — `alpha-channel-rung-overlay-v1.md`
§4, `D-ACR-16`:

> *"Nested kanban cascade for awareness build-up — **NOT DESIGNED** — zero
> shipped precedent."*

and §3m's grep receipt: *"Grepped `lance-graph-supervisor/src/*.rs` for
`nested.*kanban|kanban.*cascade` — zero. No near-miss found this time,
foveated or otherwise."*

**This plan's contribution is to name what handover IS, so D-ACR-16 has a
shape to be designed against:** a Hole is a **kanban card for a missing link**.
It is not a message and not a shared log — it is an owned row that moves
between mailboxes, which is the only handover shape this substrate permits
(`E-CE64-MB-4` one-writer-per-mailbox; `E-AGENT-LOG-SHARED-SINK-ANTIPATTERN-1`
for why a shared append-log is pseudo-handover with a race).

The self-organizing property follows from that and nothing more exotic: a
mailbox that cannot close its own Hole hands it to one that might. No
scheduler, no central planner — `E-PROGRESSION-IS-EXISTENCE-NOT-COMMAND-1`
already deleted the per-owner `advance()` RPC shape, so handover must be
existence (the row is now yours), never command.

## §4 — Blockers, named with their real cause

| # | blocked | real blocker | not the blocker |
|---|---|---|---|
| B1 | `HoleV3 = ValueTenant 16` | `BoardAggregates = 15` is a **gated reservation only**; `ValueTenant`'s discriminant→`VALUE_TENANTS` index requires **contiguous** descriptors, so 16 has no valid slot until 15 resolves (`dismech-causality-v3-v1.md:503,509-510`) | not the benchmark, and not D-CV3-0..2 — CodeRabbit corrected exactly this on 2026-08-21 |
| B2 | `D-ACR-16` handover | no design, no precedent (§3) | not a mint decision — nothing to mint yet |
| B3 | 64k parallel exploration | `D-ACR-5` gates on dialectic V4: V0–V3 green at small scale first | not this plan's to unblock |
| B4 | live horizon (pothole opening) | `D-ACR-15` — `WorkflowDAG::plan()` is a registered stub whose body is comments | not absent; specified but unbuilt |

**B1 is the sharp one.** The Hole is stage 3's whole content, and it is blocked
on an unrelated mint's width being decided. That is worth stating loudly
because it looks like a queue position and is actually a hard prerequisite.

## §5 — First real corpus, and the honest size of its ground truth

The public DisMech transcode (`AdaWorldAPI/dismech-rs`) is the first corpus
where this loop has something to run on: its `causal_link_type` field marks
**INDIRECT_UNKNOWN_INTERMEDIATES** and **UNKNOWN** edges explicitly — a
knowledge base that publishes its own gaps, which is rare and is exactly what
a handover network needs as input.

**But the usable ground truth is far smaller than the gap count, and this plan
must not overstate it.** Measured on the upstream corpus by the parallel
session (`dismech-rs/bakes/stage3-current-truth-2026-08-20/mediator-feasibility.tsv`):

| | |
|---|---|
| edges labelled `INDIRECT_KNOWN` (pathophysiology) | 3,844 |
| …of which name **zero** intermediates | 1,466 (38 %) |
| named intermediate strings | 3,465 |
| …that **match an actual graph node** | **27** |
| …with a real 2-hop path in the graph | 113 |

So a supervised evaluation has **27–113 usable cases**, not thousands. Any
claim that this corpus validates the loop must cite that number, not the gap
count. The gap count says how much work there is; the match count says how much
of it is *checkable today*.

## §6 — Deliverables

| D-id | Scope | Falsifier |
|---|---|---|
| **D-KUH-1** | Name the Hole's handover shape: an owned row moving between mailboxes, with the lifecycle `Open → Proposed → {Resolved \| Refuted}`. Design only — feeds `D-ACR-16`. | a design that requires a scheduler, a broadcast, or a shared log has rebuilt what `E-PROGRESSION-IS-EXISTENCE-NOT-COMMAND-1` and `E-AGENT-LOG-SHARED-SINK-ANTIPATTERN-1` deleted |
| **D-KUH-2** | Wire `WitnessKind` → `HoleV3` once B1 clears. | a `KNOWN-UNKNOWN` band read must resolve to a Hole row, and a low-trust KNOWN must **not** — both asserted (fire + stay-silent) |
| **D-KUH-3** | Plasticity as the rewire signal: measure whether `plasticity()` moves when a Hole closes. | **BLOCKED on B1.** Two-sided: it must move on a real close AND must NOT move on a refuted one; a signal that fires on both carries no information |
| **D-KUH-4** | Growth = minting rows, never widening fields: a probe that adds N Holes and asserts `ENVELOPE_LAYOUT_VERSION` and every field width are unchanged. | any width change is the failure; this is §0's fence as a test |

**Sequencing:** D-KUH-1 (design, unblocked) → B1 clears → D-KUH-2 → D-KUH-3.
D-KUH-4 can run at any point after D-KUH-2 and is cheap.

## §7 — What this plan does NOT claim

It does not claim the network learns, that handover improves resolution, that
plasticity currently rewires anything, or that 27 checkable cases validate a
method. It claims one thing: **the loop's links are individually shipped or
individually designed, and exactly one — handover — is neither.** Every number
that would justify more has to be measured after D-KUH-1.

The grade discipline of the parent plan applies verbatim
(`alpha-channel-rung-overlay-v1.md` §0 piece 7): this is **a pruner, not a
proof**. "AGI-aspiring" names a direction; it is not a property any deliverable
here asserts.

## §8 — The arithmetic of ↑n (operator sharpening, 2026-08-21, appended while the council ran)

Operator: *"6×2×8bit ↑n is making n^n ⇒ n↑log(n) in any given thinking space —
we take universes of rabbitholes one epistemic pothole at a time."*

Translated into the measured property rather than left as poetry, this is the
radix arithmetic the canon already pins (*"the key prerenders nodes with zero
value decode"*), applied to the awareness axis:

- One 12-byte atom under `G6D2` addresses `256^12 ≈ 7.9×10^28` distinct points
  per class at full depth. A focus at depth `d` covers `256^(12−d)` of them —
  measured in D-ACR-1's own test suite (`one_shallow_focus_covers_an_unbounded_
  population`: depth 2, 65,536 addresses across the two units varied).
- **The space is exponential in depth; the path is linear in depth.** Reaching
  any specific address costs at most 12 refinement steps = `log₂₅₆(space)`.
  That is the `n^n ⇒ n↑log(n)` claim in checkable form: exploration cost grows
  with the *logarithm* of the space explored, because each step is one prefix
  level, never a scan.
- **Stacking (`↑n`) multiplies exponents while paths add.** A second register
  (a V3 stack level, another of the node's 32 facet slots) squares the
  addressable space and adds 12 to the worst-case path — exponent
  multiplication bought at additive path cost. This is why growth is minting
  rows/levels and never widening fields (§0): widening buys linear space at
  layout-break cost; stacking buys exponential space at logarithmic
  navigation cost.
- **"One epistemic pothole at a time" is the descent rule.** The pothole marks
  WHICH subtree to refine next; each handover/exploration step descends exactly
  one prefix level of one Hole. The rabbit-hole universe is never entered
  whole — it is entered one level of one hole at a time, which is what keeps
  the sweep O(holes × depth) instead of O(space).

Grade: the radix arithmetic is [G] (it is what a 256-ary prefix tree is); the
identification of "pothole" with "descent selector" is design intent carried
by D-KUH-1, not yet a measured behaviour. The `n↑log(n)` notation is the
operator's shorthand for exponential-space/logarithmic-path and is recorded as
such, not as a formal tetration claim.
