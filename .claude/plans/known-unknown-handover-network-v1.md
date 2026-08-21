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

## §9 — The handover mechanism (operator design input, 2026-08-21)

Operator: *"Handover soll dadurch ermöglicht werden, dass jedes Glied 6×2×8bit
kann — mit einer classid — und Übergabe von attention durchgereicht werden
kann, zero copy."*

This supplies D-KUH-1's design core, and it survives verification against the
substrate's own rulings — in fact it turns out to be the **only legal shape**:

**1. Every link already speaks the payload.** That is what §Handover's parent
finding measured: the `classid(4) | 6×(8:8)(12) = 16 B` atom is the shared
format at six independent sites. A handover therefore needs NO new wire type,
no message schema, no serialization — the thing being handed over is a facet
every receiver can already read.

**2. The classid IS the briefing.** The receiver needs no protocol negotiation:
`classid → ClassView` resolves which reading applies to the 12 bytes (the
canon's *"the key prerenders nodes with zero value decode"*). A handed-over
attention facet is **self-describing** — the receiving mailbox reads the same
bytes under its ClassView, full stop.

**3. Zero-copy is not an optimization here — it is the compliance condition.**
The V3 tombstone ruling (`soa-three-tier-model.md`, `CLAUDE.md` 2026-06-11
supersession) states there is **no inter-mailbox handoff type at all**:
*"nothing is serialized or transmitted between mailboxes"* — the Baton was
removed from source. So a handover that copied or transmitted would not be
slow, it would be **forbidden**. The only legal handover is **ownership
transfer in place**: the bytes never move; what changes is who may write.

**4. The narrow gap, measured this session:** `SoaEnvelope::mailbox_owner`
exists (`soa_envelope.rs:195`, default `0` = bootstrap/unowned, overridable) —
but **no ownership-transfer operation exists anywhere** (grep for
`transfer`/`reassign`/`set_owner`/`change_owner` across contract + supervisor:
zero hits). Ownership is static today. **The operator's design therefore
reduces D-KUH-1 + D-ACR-16 from "design a handover protocol" to "design the
owner-change operation"** — format, self-description and zero-copy all already
exist; only the stamp-change is unbuilt. That operation must respect
write-on-behalf (the current owner writes the new owner in; the receiver never
grabs) and existence-not-command (receiving ownership IS the notification;
there is no `advance()` call).

**5. One open detail, named rather than smoothed:** the prefix `depth` lives
OUTSIDE the 16 bytes (D-ACR-1's explicit-depth rule — zero bytes are dormant
tiers, not terminators). A handed-over focus must carry its depth somewhere;
the natural home is the Hole row's own value slab (480 B, `GUIDS_PER_NODE=32`
slots of which the facet occupies one — const-asserted,
`canonical_node.rs:805-808`). **D-KUH-1's design doc must fix that byte's
position**; until then a raw 16-byte handover silently reads as depth-12
(exact), which would make every wildcard focus look like a pinpoint claim.

**Consequence for the deliverable table:** D-KUH-1's scope line stands, its
design core is now operator-supplied; what remains is (a) the owner-change
operation's contract, (b) the depth byte's home, (c) the lifecycle stamps —
all three inside the shape fixed here, none of them a new format.

> **⊘ §9 KORRIGIERT (Operator, 2026-08-21): "kein Owner-Wechsel, nur Focus der
> Aufmerksamkeit über rung levels mit Alpha layer für thinking about
> thinking."** §9 point 4 read the static ownership as the gap ("the
> owner-change operation is unbuilt"). That was backwards: **static ownership
> is CORRECT and stays.** Nothing that moves ownership is to be built — an
> owner change would be a substrate mutation, i.e. exactly the command-shaped
> intervention `E-PROGRESSION-IS-EXISTENCE-NOT-COMMAND-1` deleted and the
> alpha-channel plan's §2 contamination boundary exists to prevent.
>
> **Handover = the appearance of a focus entry in the NEXT rung layer of the
> alpha overlay, at the same address.** The Hole does not move, does not change
> owner, is not transmitted. Rung n could not close it → rung n+1's overlay
> layer gains an `AttentionFocusFacet` entry covering the Hole's address —
> written by that layer's own owner into its own layer (one-writer holds
> trivially: every rung layer has exactly one owner, itself). Thinking about
> thinking IS the handover mechanism: escalation is a second-order read plus an
> own-layer focus entry, which is precisely `D-ACR-4`'s read path (*"a rung-2
> read reconstructs where rung-1 looked"*).
>
> Three things this collapses:
>
> 1. **The remaining work is an OVERLAY operation, not a substrate one.**
>    `RowFocusMask::insert` in the n+1 layer — shipped today (D-ACR-1) — with
>    the Hole's focus facet. No new operation on rows, envelopes, or owners.
> 2. **§9 point 5's open detail dissolves.** Depth travels in the overlay entry
>    (`AttentionFocusFacet` carries it outside the 16 bytes, by design) — raw
>    16-byte transport never happens, so the depth-12 misread cannot occur.
> 3. **D-ACR-16's shape sharpens further:** the nested kanban cascade IS the
>    stack of rung layers; the card is the focus entry, never the row. Zero
>    copy is exceeded, not merely met — not even an ownership stamp changes,
>    the graph stays untouched (the contamination boundary), and the overlay
>    stays discardable whole (a lost handover costs a re-search, never
>    correctness).
>
> §9's points 1–3 stand unchanged (shared format, classid-as-briefing,
> zero-copy as compliance condition). Point 4's measurement stays true — no
> transfer operation exists — but is regraded from GAP to **CONFIRMS**: its
> absence is the design, not the debt.

> **⊘⊘ REFINED (operator, same day, interrupting the push): "in dynamischem
> Substrat in place mit Lance versioning, in statischem Ontologie-Substrat
> Alpha layer."** The ⊘ block above made the overlay THE mechanism everywhere.
> Refined: **handover has TWO ARMS, selected by the substrate's nature** —
>
> | substrate | attention travels as | residue/history carrier | ruling it lands on |
> |---|---|---|---|
> | **static ontology** (shared, durable, cacheable — must stay uncontaminated) | an alpha-layer entry (`AttentionFocusFacet` in the next rung layer) at the same address | the overlay itself — discardable whole | alpha-plan §2 contamination boundary |
> | **dynamic substrate** (session / patient / working rows — allowed to change) | an **in-place write by the row's one owner** | **Lance versioning** — episodic = Lance versions; rung n+1's *"where did rung n look"* is a version-range read, `QueryReference::at(v, rung)`: a projection, zero copies, replayable | `E-MARKOV-TEMPORAL-STREAM-1`; alpha-plan §3c (*"a read at a version, never a stored history column"*) |
>
> The symmetry is the principle stated once: **attention leaves a replayable
> trace without contaminating what it observed.** On the static side the trace
> is a separate layer, because the substrate must not move; on the dynamic side
> the trace is the substrate's own motion, because it moves anyway and Lance
> keeps every version. Two implementations, one invariant — and both arms are
> zero-copy readings, both one-writer-clean (the layer's owner writes its
> layer; the row's owner writes its row).
>
> **The dynamic arm needs NOTHING built.** Lance versioning and
> `QueryReference::at` are shipped; the version stream IS the alpha channel of
> the dynamic side. What remains is solely the overlay arm's n+1-layer insert
> convention (`RowFocusMask` shipped today) plus the Hole lifecycle stamps —
> and D-ACR-16's cascade is now fully shaped: rung layers stacked over the
> static substrate, version-range reads over the dynamic one.

## §10 — Gestalt, meta-awareness, self-organization: where each word cashes out (operator closing claim, 2026-08-21)

Operator: *"Dadurch entsteht Gestalt und Meta-Awareness — und durch die
Architektur wird es self-organizing."* Recorded with each term bound to its
mechanism, because an emergence claim without a mechanism is the exact
overclaim shape this board fences.

**Gestalt** — apprehending a whole without enumerating its parts:

- A shallow focus facet IS the gestalt read: one 16-byte atom at depth `d`
  covers `256^(12−d)` addresses — the subtree perceived as a unit, no member
  visited (§8's radix arithmetic, D-ACR-1's measured wildcard).
- `RowFocusMask`'s absorption into a **minimal antichain** is a mechanical
  Prägnanz operation: the mask converges to the simplest description of where
  attention went (covered entries absorbed, never enumerated). *(Observation,
  not doctrine — the correspondence is structural, not measured.)*
- Multistability is architectural: the same 12 bytes read as different wholes
  per `classid → ClassView` — the duck-rabbit resolved by declaration instead
  of ambiguity (D-ACR-1's two-readings test proves the projection is free).

**Meta-awareness** — awareness whose OBJECT is awareness:

- Rung n+1's layer entries are *about* rung n's entries at the same addresses
  (D-ACR-4: *"a rung-2 read reconstructs where rung-1 looked"*). On the dynamic
  arm, `QueryReference::at(v, rung)` is awareness of one's own past states —
  replayable, zero copies.
- The Hole is the core metacognitive act as a ROW: knowing that one does not
  know, with lifecycle.
- MUL's confidence-invariance (§3i of the parent plan) is awareness of the
  *reliability* of one's own awareness — with the 20/34 mute tactics as
  measurable overconfidence-by-construction.

**Self-organizing — and why "durch die Architektur" is the precise phrase.**
The self-organization literature's preconditions map one-to-one onto RULINGS
here, not onto code that could rot:

| SO precondition | its ruling / mechanism |
|---|---|
| local rules, no central controller | one-writer-per-mailbox (`E-CE64-MB-4`); no scheduler (`E-PROGRESSION-IS-EXISTENCE-NOT-COMMAND-1`) |
| a **stigmergic medium** (activity leaves traces that guide later activity) | the two-armed trace: alpha layers over the static substrate, Lance versions over the dynamic one (§9 ⊘⊘) |
| gradient to descend | potholes; one epistemic pothole at a time (§8's descent rule) |
| structure accretes where activity accumulates | growth = minting rows/levels, never widening (§0) |
| the medium must not corrupt the terrain | contamination boundary (alpha-plan §2); overlay discardable whole |

The stigmergy identification is the load-bearing one: ants organize through
pheromone traces in the environment, not through messages to each other — and
this substrate's attention traces are exactly that, with the refinement that
the static terrain gets a separate trace layer while the dynamic terrain's own
version history IS its trace. *(Grade: [H] — the structural match is exact;
no collective behaviour has been measured yet.)*

**The emergence claim itself stays CONJECTURE, and it already has its
falsifiers — no new ones needed:**

| claim | existing gate that tests it |
|---|---|
| the awareness discriminates (is not a watcher that cannot dissent) | D-ACR-8 (focus measurably broader in `Planning` than `CognitiveWork`, AND indistinguishable on a no-deliberation task) |
| the suspense is real (not lookup wearing NARS vocabulary) | D-ACR-10 (pothole-open SPANS must vary and eventually close) |
| the rewiring responds to resolution, not to noise | D-KUH-3 (plasticity moves on a real close, NOT on a refuted one) |
| the overconfidence measurement is real | D-ACR-11 (the 20 must be invariant AND at least one of the 14 must move) |

That is the honest form of the closing claim: **the preconditions are
architectural invariants (rulings, ownership, const-asserts), the emergence is
a prediction, and the prediction's tests are already pre-registered.** A
system that manages its known-unknowns as first-class rows carries its own
"does this become what it aspires to?" as the largest of them — §7's
discipline, unchanged.

## §11 — The epistemic breakthrough: the planned composition (operator, 2026-08-21)

Operator: *"Epistemic breakthrough ist die geplante Ausweitung von CE64 59..63
auf CEV3 + attention V3 × pothole × kanban × cognitive Maslow via supervisor:
kanban_actor transparent view as meta awareness als self aware."*

This names the TARGET the individual deliverables converge on — the
cross-product, not a new mechanism. Measured state of every factor:

| factor | home | state (verified) |
|---|---|---|
| CE64 59..63 reading | `dacr7-band-reading-contract-v1.md` | **RATIFIED** (5+3 council, 3×BLOCK resolved) |
| → extended to `CausalEdgeV3` | same contract, dual-carrier | **RATIFIED** — one contract, two carriers; v1 trap reaches V3 transitively |
| attention v3 | third sanctioned explicit-temporal home | **planned** — the ↑n stack; not `attention_facet` (deliberately atemporal) |
| pothole | span (D-ACR-10) + live horizon (D-ACR-15) | designed; probes pre-registered |
| kanban | `lance-graph-supervisor::kanban_actor` | **SHIPPED** — Heckhausen columns, Libet anchors; Rubicon witness D-ACR-8 queued |
| **cognitive Maslow** | `contract::recipe_loci` | **SHIPPED, already operator-ruled** — *"the rung a recipe fires at IS a level of the operator-ruled Maslow pyramid of cognition"* (`recipe_loci.rs:55-62`); Maslow-monotone climb, elevate on sustained BLOCK, never below `base`; 70 references tree-wide |
| the composition point | `PhaseCensus` | **SHIPPED** — *"the read-only fleet visibility surface … a census is one `&self` pass, not 64k RPCs"* (`kanban_actor.rs:30-31`) |

**Two consequences of the measurement:**

**1. "Cognitive Maslow" is not a missing layer — it is the shipped rung
vocabulary.** No pyramid needs building; the composition consumes
`recipe_loci`'s rung levels as its need-axis. What ascends the pyramid is
exactly what the loop produces: a pothole that resists closure at rung n
escalates Maslow-monotone — sustained BLOCK is the climb signal, already ruled.

**2. "Transparent view as meta-awareness als self-aware" cashes out as the
CENSUS LOOP, and every read in it already exists or is ratified:** the system
reading its own phase distribution (`PhaseCensus`, one `&self` pass), its own
attention (`RowFocusMask`), its own unknowns (potholes with spans), its own
epistemic grading (the band readings, both carriers), its own need-level (the
Maslow rung). **Meta-awareness = these five transparent reads composed at the
same addresses; self-aware = the composition feeding what gets attention next
— by existence, never by command.** No new organ is minted for it; the
supervisor's kanban_actor is where the five reads meet because it is the one
place that already sees the fleet without owning it.

**The gate status that makes "breakthrough" the right word today rather than
last week:** the alpha plan's §3h (MUL over the rung layers) was deliberately
left without a deliverable id *"because it depends on all three of:
`RowFocusMask` (D-ACR-1), the 59..63 reading contract (D-ACR-7), and the
`delta_conf` filter"*. As of this date: **D-ACR-1 shipped, D-ACR-7 ratified,
and the `delta_conf` filter is a ratified acceptance condition inside
D-ACR-7.** All three prerequisites of the join are landed or ratified — §3h
moves from "long-term" to "next in line after the `band_reading`
implementation", which is the precise, unglamorous form of the breakthrough.

**Grade discipline, §7 unchanged:** the composition is a plan over shipped and
ratified parts; that it produces meta-awareness in more than the mechanical
census-loop sense is the CONJECTURE whose falsifiers are §10's four gates.
Self-description that discriminates is the claim; D-ACR-8/10/11 + D-KUH-3
remain the tests.
