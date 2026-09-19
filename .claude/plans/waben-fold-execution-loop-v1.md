# Waben fold execution loop — v1 (PROPOSAL, 2026-09-19)

> **Status:** PROPOSAL — no code written. Grounded against lance-graph
> `origin/main` **25988f3c** and ndarray **40a71ad** (both read 2026-09-19).
> Source docs: `waben_fold_architecture.md` + `waben_implementation_prompt.md`
> (operator-supplied, 2026-09-19).
>
> **Prefix for this arc's deliverables: `D-WFL-*`** (minted by this arc; the
> `D-WFL-W*` rows in `STATUS_BOARD.md` are the live set and supersede the flat
> `D-WFL-1..7` rows the first commit added).
>
> **⊘ REVISED 2026-09-19 after external review of #1251.** Five corrections and
> one addition, all verified by reading before acceptance:
> **(1)** a fourth seam — **Seam D**, the semantic-order → Morton-order rotation
> — was missing, and §5's first slice silently crossed it; **(2)** the
> duplicate-key recommendation is WITHDRAWN (it was a semantic regression);
> **(3)** D-WFL-2 splits in two, because a bounded `Scratch` is a second
> experiment, not a widening of the first; **(4)** `AlphaFocus` cannot be the
> slice's focus carrier as it stands — every READ accessor materializes a
> full-population mask; **(5)** `claim_ordinals` is deferred behind a
> measurement. Two claims are downgraded from *resolved* to *hypothesis*:
> `thought track == w_slot` and `rail position == axial direction`. The wave
> order below is the revised one; the original D-WFL-1..7 numbering is
> superseded.

---

## §0 Baseline refresh — what moved since the supplied docs

Nothing. The docs' baseline is current.

| Repo | Doc's baseline | Actual head 2026-09-19 | Delta |
|---|---|---|---|
| lance-graph | #1250 at `25988f3c` | `25988f3c` | none |
| ndarray | `e1ef350a` (#317) | `40a71ad` | three commits, all **G8 documentation rounds**; no new primitive |

Two doc claims corrected by reading (not by grep):

- **`mask_set_range` DOES paint the whole destination.** The doc is right and a
  first reading of it was wrong. `simd_masking_ops.rs:1587-1588`:
  `fill_words(&mut out_words[..lo_word], 0)` and
  `fill_words(&mut out_words[hi_word + 1..], 0)` — the second runs to the END of
  the slice, not to `mask_words_for(hi)`. The `lo == hi` branch is
  `fill_words(out_words, 0)`, explicitly whole-slice. This is CORRECT for the
  primitive's contract ("set this mask to exactly this range"); the defect is at
  the caller.
- **G8 is not a primitive.** No `lzcnt` / `bswap` / depth symbol exists anywhere
  in ndarray `src/`. It is named in `masking-ops-state.md`, `EPIPHANIES.md` and
  `blackboard.md`, and built in none of them. Any plan step that wants a
  tree-depth column must build it first, in ndarray, as its own PR.

---

## §1 The composition, in one pass

Six stages, and the question for each is only ever: *what carrier does the
previous stage hand over, and does the next stage accept it without rebuilding
the coordinate system?*

1. **Shared identity.** `NodeGuid` (16 B, `classid(4) + payload(12)`) is the
   anchor; `NodeRow` is `key(16) | edges(16) | value(480)` = 512 B. The second
   16 bytes are no longer an "edge block" — `pub type EdgeBlock = FacetCascade`
   — so a node carries **two facets of identical shape** plus 30 value slots.
2. **Lens selection.** `SemanticLens` (today one variant, `CanonHighTiles8`)
   names the projection an order claim is stated under. Storage is a
   content-blind ordinal; "sorted" is meaningless without a lens. **Carrier out:
   a lens tag, not data.**
3. **The fold.** `SemanticPrefix::{lo_key, hi_key}` + an `OrderedLaneWitness`
   whose lens matches → `PrefixLowering::Bound { lo, hi, lane_version,
   lane_digest }`. **Carrier out: two `u32` ordinals.** 250 ns at 1M rows.
4. **Local propagation.** A six-neighbour field over a Morton-placed row
   population; `A_{t+1} = (A_t ∪ ⋃_d S_d(A_t ∩ P_d)) ∩ T`. **Carrier in: a
   bounded mask over a tile. Carrier out: the same, plus a delta frontier.**
5. **Endogenous focus.** The frontier becomes the next stage-3 input.
6. **Sparse publication.** Only the delta crosses into `AlphaOverlay`, stamped
   by an owner, batched into one `DatasetVersion`.

### Folds are zero copy. Period.

**Zero-copy is not an optimization of the fold. It is part of the definition of
a fold.**

> A fold reads canonical state in place and returns a compact consequence. If it
> copies or materializes the source population, it is not a fold.

**Definitional caveat, so the law cannot be argued away:** *zero-copy* means no
**software-level** materialization, duplication, re-encoding, or retained
derived population. CPU loads into registers and cache lines obviously still
happen — those are not a second representation, and nobody gets to cite them as
proof the rule is unmeetable.

The boundary, exactly:

```
FOLD                              NOT A FOLD
canonical bytes                   canonical bytes
 -> zero-copy projection           -> duplicate lane
 -> compare / prefix /             -> build full mask
    intersection / reduction       -> scatter rows
 -> scalar / range / runs /        -> accumulate state
    tiny descriptor                -> sweep later
```

Six invariants, stated so a future session cannot soften them by degrees:

- source bytes are never copied by a fold;
- source layout is never rewritten by a fold;
- a fold does not retain an execution view;
- a fold may emit only answer-sized or focus-sized state;
- **population-sized output is materialization, not folding**;
- replay is repeated zero-copy folding over pinned canonical state.

**And the corollary that makes it a membrane rather than a slogan: the moment an
operation needs to materialize population state, the fold has ended.**

This does not forbid materialization. It forbids materialization *hiding under
the word fold*. Building an index, caching a projection, publishing an effect —
all legitimate, all sometimes necessary, and each must be **named honestly as
what it is** and priced accordingly.

Everything below — the four seams, the wave order, the write tiers — is
downstream of this. A seam is exactly a place where the code violates it, and
Seam B is the cleanest example: `Pred::Range` emits a population-sized mask, so
by this definition the executor's range path is materialization wearing a fold's
name.

### Where the law lives: T1, and it already has an ISA

The law above is not a new tier. `membrane-tiers.md:22` already defines **T1** as
the population algebra and states its return contract as *"a mask, a count, a
lane descriptor — **never the population**"*, and `:48` says outright that the
ladder does not need another tier. So the zero-copy law is a SHARPENING of T1.

And T1's instruction set already exists: **`lance-graph-mask-risc`** — the crate
name says RISC. `Program` is the plan a thought compiles into; `MaskOp`,
`Pred`, `Terminal`, `fuse.rs` (Boolean tree → one ternlog) and
`ternlog_dispatch.rs` are the ISA and its fuser. Nothing needs minting.

Which yields the membrane law this whole plan operates under:

> Higher layers may invent arbitrary cognition. They may NOT invent new
> population execution semantics — they compile cognition into T1.

No thought style owns intersection, prefix, locality, range, projection or
rotation. Those are physics. A style owns only *which* T1 operations it composes
and *why*. **ISA, not standard library:** a new T1 primitive earns existence only
by exposing a genuinely new zero-copy operation the existing algebra cannot
express by composition or fusion — `fuse.rs` is the precedent, collapsing a
Boolean tree rather than growing a variant per shape.

**And the law gives mask-risc a conformance criterion it did not have.** Every
scratch plane is required to be `words_for(n_rows)` (`exec.rs:566`), so every
`MaskOp` today emits population-sized output. A *bounded* word window is a
legitimate focus-sized carrier; a mask unconditionally sized to the population is
not. So Seam B is not a performance complaint — it is a conformance failure
against the tier the crate belongs to.

**Where the code leaves the representation — four seams, in dependency order.**

### Seam A — the executed planes never prove they are the witnessed lane

`Planes<'a> { n_rows, masks, lanes }` (`mask-risc/src/ir.rs:50`) carries **no
version, no lens, no order identity**. `PrefixLowering::Bound` carries
`lane_version` + `lane_digest`, and nothing compares them to anything. A bound
is a pair of ordinals into a row order the executor never attested. Filed as
`ISS-WITNESSED-RANGE-DOES-NOT-ATTEST-PLANE-ORDER`; still open.

Equal length is not sufficient, and a digest of the *keys* is not sufficient
either: duplicate keys with distinct payloads produce one digest for several
distinct payload orders.

### Seam B — the bound is repainted into a full-population mask, always

This is the resistance returning to the wire, and it is not a bug in one call
site — it is the IR's definition:

```
exec.rs:566-571   scratch.words != words_for(planes.n_rows)  ->  ExecError::ScratchWords
exec.rs:540       (Pred::Range { lo, hi }, None) => mask_set_range(dst, lo, hi)
```

Every scratch plane is *required* to be `words_for(n_rows)` wide. So a
`Pred::Range` at 1M rows writes 15,625 words / 125 KB whether the range is 100
rows wide or 900,000. **A range cannot stay a range anywhere inside this IR.**

The cost is already measured, under another name. D-DMD-P2's control arm — the
one the probe labelled "the OLD, buggy whole-lane buffer, for comparison" — *is*
the production executor's behaviour:

| arm | N=1K | N=4M |
|---|---|---|
| base-offset touched write (`touched_write`) | 22.0 ns | 20.5–23.1 ns (flat in N **and** in position) |
| whole-lane-sized destination (**= `Pred::Range` today**) | 34 ns | 4,620 ns |

The probe measured the shipped path and filed it as the defect it had just fixed
in itself.

### Seam C — alpha is hash-and-row shaped; the fold is mask-and-ordinal shaped

Not previously named. `AlphaOverlay` (`contract/src/alpha.rs:613`) is:

```rust
pub struct AlphaOverlay<'a> {
    alloc: AllocRef<'a>,
    claimed: Vec<NodeRow>,          // 512 BYTES PER CLAIM, copied
    at: HashMap<AlphaAddr, usize>,  // AlphaAddr = NodeGuid, 16 B, hashed
    cycle: u32,
}
```

Start from what FIRE actually is, because every loose framing of this seam
inverts it:

> **FIRE only ever writes what the read already had.** It is a sparse
> alpha-channel delta, and the delta is not computed at publication time — the
> fold, the tile and the frontier were already holding it, in their own compact
> form. Publication is persistence, not production. There is no version of FIRE
> that reconstructs, rebuilds, or re-derives state.

That is the measuring stick, and it makes the seam precise: **any re-addressing
at the write boundary is pure loss.** Not expensive work — *unnecessary* work,
because the information was in hand one instruction earlier.

Two places the current code asks for it back:

- **The claim's input coordinate.** `claim()` demands an `AlphaAddr` =
  `NodeGuid`, and hashes it. But the read path holds **ordinals** — a mask, a
  bound, a frontier. So the caller must re-derive a coordinate it already had in
  a cheaper form, purely to satisfy the signature. This is the seam proper, and
  it is about the *key*, not the storage.
- **The focus query.** `attended_mask()` (`alpha.rs:783`) allocates
  `AlphaMask::empty(base.len())` and scatters bits by walking the HashMap — a
  dense answer to a question asked of sparse state, one layer after the sparse
  write.

`AlphaMask` is the right type (bitset with explicit phantom-tail discipline,
`alpha.rs:224`), and `claimed: Vec<NodeRow>` is the *storage contract*, which is
a separate question from the key (see W6). What is mis-shaped is the coordinate
the write demands and the materialization the read performs — never the delta,
which was compact the whole way.

### Seam D — the bound and the tile are intervals in DIFFERENT orders

The sharpest of the four, and the one this plan's own first draft walked
straight through.

`SealedFacetLane` sorts by `FacetCascade::cmp_numeric_projection`, so a
`Bound { lo, hi }` is an interval in **semantic projection order**.
`mask_shift_morton` reads bit position as `ordinal = Morton(q, r)` — an interval
in **geometric order**. §1 states two paragraphs earlier that one physical
sequence is monotone under one lens at a time. It follows that these two
intervals are *not* the same set of rows, and the draft's `window ∩ tile` joined
them with a bare intersection.

```
semantic ordinal  --(identity-preserving rotation)-->  Morton ordinal
```

That rotation has a cost, and nothing in the plan accounted for it. It is also
the most interesting unknown in the whole architecture, because the two outcomes
are the thesis and its refutation:

- if the rotation is a cheap projection or an index lookup, that IS the
  schema-rotation claim, demonstrated;
- if it is a scatter of thousands of `NodeGuid`s through a hash table, the
  resistance the fold removed at the bound has simply moved one stage later.

It must be **measured before the Wabe wave**, and it must not be hidden inside
fixture construction — a fixture that builds the population already in Morton
order and then seals it semantically has assumed the answer.

### The consequence of A + D: replay can be cheaper than storage

Mask intersection is deterministic and cheap — that is the whole point of the
substrate. Take that seriously at the publication boundary and a third option
appears beside the two the architecture names:

```
fold(address, condition)
    -> NoChange
    |  PublishEffect(address', delta)          the result
    |  PublishReplayableTask(domain, program)  the QUESTION that regenerates it
```

If the same row domain and the same program yield a bit-identical mask every
time, then persisting the *task* — a kanban entry naming the domain and the
program — is a complete record of the insight, and the insight itself can be
recomputed on demand. The system can then afford to hold **many** insights as
replayable rather than as stored answers, because each one costs a descriptor
instead of a result set, and each replay costs an intersection.

This is not a new transport and not a new actor message. It is a claim about
what the durable unit *is*, and it sits naturally with the deleted
`KanbanActor`'s own ruling (`E-PROGRESSION-IS-EXISTENCE-NOT-COMMAND-1`): what
gets written is the task's **existence**, never a command and never an ack.

**W1 is the precondition, not merely a correctness gate.** A replay is sound
only if the domain it replays against is pinned — version, lens, and the
associated-row order. Without Seam A closed, "replay" means *recompute against
whatever the lane looks like now*, which is not replay.

⊘ **But `RowDomain` is NOT "the replay key"** — an earlier phrasing here said so
and it will cause a collision later. `RowDomain` is the **row-coordinate
component** of a replay key. The rest of the key (lens identities, focus/carrier
input, external-edge snapshot, deterministic parameters) is enumerated below and
is not optional.

Two consequences for the accounting in §6: the agreed exact answer carrier for
such an insight is the **task descriptor**, not the row set it denotes; and the
replay cost must be reported as its own column, because an insight that is cheap
to store and expensive to re-derive has only moved its cost.

### If thinking again is cheaper than remembering the answer, think again

The replay tier above is framed as a fallback. At this substrate's ratios it is
often the **fast path**, and that inverts the ordinary database instinct the
rest of this plan would otherwise inherit.

Order-of-magnitude arithmetic, and it must be read as exactly that:

```
one fold          ~1.7 ns      (#1245's six-tier axis chain)
1000 stacked folds ~1.7 us
one population sweep ~10 us    -> ~5,900 fold-equivalents
                               -> ~6 complete 1000-fold chains, on ONE lane
```

⊘ **The 1.7 ns does not transfer and this is not a measurement.** #1250
explicitly declined to carry #1245's axis-chain figure to the whole-facet cell
(1.7–4.2 ns there), and a "fold" inside a 1000-fold chain is not necessarily
that axis chain. What survives is the SHAPE of the argument, which holds at any
plausible ratio: a sweep costs thousands of folds, so a stored answer can be
slower to retrieve than the answer is to re-derive. Measuring the real ratio is
W6's job.

**The law:**

> Never retain derived execution state merely to avoid replay, when replay
> through stacked folds is cheaper than maintaining the retained state.

Which sets the real violation criterion. A transient `&T` lasting 20 ns is
irrelevant and always was; the enemy is a **retained** representation that needs
maintenance because we might want it again:

```
peek -> fold -> answer -> nothing survives        GOOD (a temporary optical path)
retain a view/mask/cache "for later"              SUSPICIOUS
stacked folds                                     GOOD
population sweep to keep state coherent           SUSPICIOUS
recompute                                         the DEFAULT
cache / materialize                               must earn its existence
```

Stated once, as the rule that replaces the borrow doctrine: **a fold consumes
zero-copy peeks from canonical state and does not retain a view of it. Any
execution object that persists merely to make later folds possible is suspect,
because the fold should reacquire the canonical address and peek again.**

Once a "fold" starts maintaining an N-sized representation, the frozen-duck
problem has been recreated under a prettier name.

So the target is not zero reads, and not even zero repeated computation. It is
**zero unnecessary representational entropy.** Repeated computation is nearly
free while every step remains a fold; it is the second representation — and the
machinery to keep it coherent — that costs.

**Two, and only two, reasons to store** (W5/W6 use this, not "does it borrow?"):

```
C_retain  = C_materialize + C_maintain + C_invalidate + C_readback
C_replay  = sum(C_fold_i) + C_rotation + C_local

1. ECONOMIC   retain only when C_retain < C_replay
2. SEMANTIC   or when durability has value independent of speed —
              it crossed the Rubicon and must become history / evidence / state
```

Everything else evaporates.

### The scale consequence: 64K thoughts are continuations, not processes

This stops being an optimization at 64K logical contexts, because it decides
what "64K thoughts in parallel" even means:

```
WRONG   64K mutable cognitive machines, swept to stay current
RIGHT   64K suspended continuation points, each cheaply represented by
        address/domain + lens + fold program + dependencies + tiny meta state
```

Dormant thoughts consume approximately zero compute:

```
wake thought 18,721  ->  peek -> fold -> fold -> fold -> answer -> vanish
```

The canonical SoA is the lake. A thought does not carry a bucket of water around
in case it wants to drink again later — it remembers where the lake is and how
to drink. Enormous LOGICAL concurrency without paying physical concurrency; if
every dormant thought had to preserve a view, we would already have lost.

**The scheduler law:** *no dormant thought may consume sweep cost merely to
remain current.* And the metric that follows: the cost of a sweep is properly
measured in **fold-equivalents** — how many complete alternative reasoning
chains the substrate could have run while it ran. A machine holding 64K thoughts
but spending its time refreshing their masks may be LESS cognitively parallel
than one holding 64K replay descriptors.

**And thoughts become shareable in the useful sense.** If one mailbox has found
a reasoning structure, another that needs it does not wait for a sweep to
refresh its maintained state:

```
share a RESULT   expensive, possibly stale, bound to its original context
share a THOUGHT  a compact replayable operator, REBOUND to the recipient's context
```

The second is what reuse of an idea actually is: not a snapshot of someone's
working memory, but a transformation you run your own context through. So the
64K palette is not 64K attention channels — it is 64K callable cognitive
continuations, crossable between kanban boards wherever their dependencies are
satisfiable.

### Writing is the expensive part — so the boundary has TIERS, not a switch

The reason replay wins is not that reading is cheap; it is that **writing is
expensive**. That asymmetry, stated plainly, forbids a design the two-option
framing above still allows: a speculative *"this looks interesting, let me test
this hypothesis"* must not cost an SoA row. Under one undifferentiated publish
path it does, and the cost of curiosity becomes the cost of knowledge.

So the durable side is a ladder of at least three tiers, each strictly cheaper
than the one above it:

| tier | what it records | carrier | when |
|---|---|---|---|
| **meta kanban atom** | *a hypothesis worth testing* — the question, not any answer | a small intermediary write at the META level; **never an SoA row** | the cheap, frequent case: speculation |
| **replayable task** | the domain + program that regenerates an insight | a descriptor (`RowDomain` + program) | the insight is settled but need not be materialized |
| **materialized effect** | the answer itself, as state | the alpha row / SoA write | it crossed the rubicon: worth being state |

**The Rubicon is a four-way policy on the delta, not a FIRE / no-FIRE binary.**
FIRE stays exactly what it is — a sparse delta. What the Rubicon decides is what
*durability that event earns*:

```
ephemeral fold / bound / intersection / Wabe / frontier
        │
        ├─ nothing interesting          -> vanish
        ├─ interesting QUESTION         -> meta atom
        ├─ result worth REPRODUCING     -> replay spec
        └─ result worth becoming STATE  -> materialized effect
```

and the three retained tiers are three **increasingly strong contracts**, not
three storage sizes:

| tier | what it must carry to be honest |
|---|---|
| META | enough identity to know *what question existed* |
| REPLAY | enough identity to *regenerate the same answer* |
| STATE | enough epistemic justification to *retain the answer itself* |

⊘ **A replay spec is more than `RowDomain + program`** — that is necessary and
not sufficient, and the earlier text implied otherwise. The proof obligation
(not necessarily a struct) is the complete deterministic input identity **for
this computation** — as REFERENCES, never re-serialized contents:

```
RowDomain  +  program identity  +  ClassView/lens version
           +  focus carrier identity  +  external-edge snapshot ID
           +  deterministic parameters
```

pins the computation without duplicating anything it read. Miss one and `replay(task)` can return a different answer while
looking valid — the thought may also have depended on another overlay, a mutable
attention input, a changed ClassView, or a different Wabe mapping. **The
determinism gate (`D-WFL-DET`) therefore precedes the replay tier being blessed,
not merely its use.**

**The rubicon is the point of the design.** An atom *at* the rubicon is written
at the meta level; only what crosses it earns the full row. That is what lets
the system afford to be curious — to fire off many hypotheses — without each one
costing what a conclusion costs. It also matches
`E-PROGRESSION-IS-EXISTENCE-NOT-COMMAND-1` at a second level: the meta atom
records that a question EXISTS, not a command to answer it and not its answer.

Two things this makes measurable, and W6 must report both:

- **Cost per tier, separately** — bytes and µs for a meta atom, for a replayable
  task, for a materialized effect. If the ladder is not strictly increasing by a
  wide margin, the tiering buys nothing and should be dropped rather than
  maintained.
- **Tier mix under a real workload** — how many speculations per replayable
  task, how many replayable tasks per materialized effect. A tiering whose cheap
  tier is rarely used is decoration; a system whose expensive tier fires on
  every speculation never had a rubicon.

⊘ This is a claim about the SHAPE of the durable side, not a licence to build a
new transport, a second store, or an actor message per hypothesis. The meta tier
is an intermediary write through the existing owner, and naming it does not
authorize inventing one.

⊘ **Specified now, BUILT after W6's measurement — the ladder must not jump the
queue.** Execute W1→W5, then let W6 produce the actual per-stage numbers against
the CURRENT `AlphaOverlay::claim()`. If the existing publication is indeed
orders of magnitude costlier than the reasoning that precedes it, the ladder has
measured motivation and the smallest meta-atom / replay surface the measured
workload demands gets built. If the costs are not widely separated, the tiering
is dropped rather than maintained. Building a three-tier memory architecture
before proving that writes dominate THIS loop is the exact failure this plan
keeps warning about, one level up.

**The falsifier this needs, and it is not optional.** Determinism must be a
*proved* property, not an assumed one: same `RowDomain` + same program ⇒
bit-identical mask, across repeated runs, across SIMD backends, and across
process restarts. ⊘ **Scoped to the integer / Boolean mask substrate** — the
right bar for everything this arc runs on, and deliberately not a general rule:
if Gaussian / f32 propagation later enters the Wabe it needs a
*numerical-equivalence* contract instead, and nothing here outlaws the kernels
the architecture already anticipates. Anything that makes a result depend on scratch contents,
iteration order of a hash map, or a runtime ISA choice breaks replay silently —
and silently is the only way this can break, because a wrong replay still
returns a plausible mask. Gate it before any wave relies on replay.

### And the finding that reframes the sequence

The census's dominant result is not any single defect. It is that **almost
everything the loop needs already exists, is tested, is doc-honest — and is
never assembled.** Verified orphans (grep, then read, then confirm the only
outside reference is a `pub mod` line):

| module | state |
|---|---|
| `wave_dispatch::dispatch_thought` — rung-wave scheduler driving `AlphaTunnel` | built, tested, **zero external callers** |
| `alpha_focus::AlphaFocus` — the rung × tenant cross, `unlooked()` | built, **zero callers anywhere** |
| `step_mask::StepMask(u64)` | a bitset type with **no consumer** |
| `batch_writer::BatchWriter::cast` → `LanceCycleWriter::commit_cycle` | every link real; assembled only in `tests/` and one `examples/` |
| `supervisor::CallcenterSupervisor` | real ractor actor, **never spawned** outside its own doc comment |
| `quack::Filter::prefix_facet` | all three call sites inside `#[cfg(test)] mod diamond_lowering_tests` |
| `mask-risc` `Pred::Range` | no production `Program` builds one |

`kanban_actor.rs` carries an explicit tombstone (`E-PROGRESSION-IS-EXISTENCE-NOT-COMMAND-1`): `KanbanActor` and its `Advance`/`MulAdvance`/`Tick` RPCs were **deleted**, not deprecated. Do not reintroduce that shape.

**Consequence for the plan: the deliverable is mostly ASSEMBLY, not
construction.** Most waves below connect two existing symbols and delete a
materialization between them.

⊘ The first draft said *"only two build anything genuinely new."* That is too
strong and is withdrawn. `RowDomain` is new semantics; the range-native
terminals are new execution; the Morton rotation is a new mapping; a second
`SemanticLens` variant is new implementation; and an ordinal-keyed claim path
would be a new alpha mutation path. The *architectural* conclusion survives —
no new subsystem is needed — but the literal count does not.

---

## §2 The address — resolved onto what exists, with no new storage

The prompt's `(G, NodeGuid, Thought-or-Rung)` resolves as:

| coordinate | existing home | evidence | new storage? |
|---|---|---|---|
| **G** (ontology / ClassView) | `classid: u32` at facet bytes `0..4`, canon-high (`concept << 16 \| app`); resolved through `lance-graph-ontology`'s `class_resolver` | `facet.rs:94`, canon-high flip on the board | none |
| **NodeGuid** | `canonical_node::NodeGuid`, 16 B, stable | `canonical_node.rs:862` | none |
| **Rung** | **`TemporalPov { range, rung: u8 }`** — already a *reader's* coordinate, not per-node state | `contract/src/temporal_pov.rs:151` | **none** |
| **Thought track (≤64)** | **[HYPOTHESIS, not resolved]** the 6-bit W-slot palette, `AttentionMaskEntry { mailbox_id, w_slot: u8 /* 0..64 */ }` — a *physical* attention slot with LRU state, bound to a `MailboxId`. Nothing read establishes that it semantically IS a thought track; it is a carrier that happens to have the right cardinality | `cognitive-shader-driver/src/attention_mask.rs:29` | none, IF the hypothesis holds |

So no packed 6+4 ABI and no `N × 64 × 10` slab is needed. **Rung is a read POV
— that one is resolved.** *Track* is the open one: a six-bit field is not a
thought track merely because both fit in six bits. Two things sharing a width
is not a semantic identity, and this plan must not canonize one by noticing a
coincidence. Marked HYPOTHESIS pending the cognitive-semantics question
(§8). The per-node
`AlphaStamp { cycle, seq, rung, visits }` occupies value slot 0 (16 B,
`ALPHA_STAMP_OFFSET = 0`) and is the *only* per-node track/rung residue —
already sized, already canon.

Five things the plan must keep distinct (the prompt's demand):

1. outer ontology `G` — which ClassView interprets the bytes;
2. the facet's own embedded `classid` — which concept×app the node IS;
3. `NodeGuid` — stable identity, never rewritten by a change of view;
4. **execution ordinal** — position in one lane under one lens, valid only with
   a version + lens + digest (Seam A);
5. **geometric cell** — `Morton(q, r)`, valid only inside a declared tile.

(4) and (5) are the two that get silently substituted for (3). Every carrier
below names which one it holds.

### The Wabe tenant needs no new layout

The LE contract's operator-locked payload catalogue (`.claude/v3/soa_layout/le-contract.md` §3)
already contains the shape: `6 × (u8:u8)`, twelve bytes. ndarray's
`hex_tenant_mq_probe.rs` independently uses exactly that — rail `d` of 6 hex
directions → `(permeability, strength)` at `r[2d]`, `r[2d+1]`. And §3's
polymorphic-pair extension already sanctions, for a node's **second GUID
dedicated to relationships**, six relations as one-byte pairs.

What this establishes, stated exactly:

> `6 × (u8:u8)` gives the **storage capacity** for a Wabe tenant for free.
> A **Wabe ClassView** is what would give those six slots *directional*
> semantics.

⊘ The first draft wrote *"six directions = six rails = six relations"* and
treated the mapping as already made. It is not. The LE contract sanctions six
`(basin, relationtype)` pairs on a relationship facet; it nowhere says those six
positions mean `+q, -q, +r, -r, +q-r, -q+r`. §3's own rule is that the reading
is ALWAYS selected by the ClassView, never by convention-in-code — and inferring
a directional reading from a slot count is exactly convention-in-code. The
capacity is free; the meaning still has to be declared and named. No new column,
no stride change, no `ENVELOPE_LAYOUT_VERSION` bump either way.

---

## §3 Capability table

`I` = implemented + live caller · `D` = implemented, declared/test-only ·
`M` = missing. Membrane per `.claude/knowledge/membrane-tiers.md`:
T0 = ndarray backends, T1 = `ndarray::simd` facade + mask ALU, T2 = behaviour
(exec, lowering, alpha algebra), T3 = intent/consumer.

| capability | symbol · caller | repo@sha | state | smallest change | acceptance evidence |
|---|---|---|---|---|---|
| point fold (8-tile LCP) | `FacetCascade::shared_prefix_tiles` · d-diamond-1-probe P1 | lg@25988f3c | **I** | — | 1.7–4.2 ns, all three arms; oracle-checked |
| lens-tagged order witness | `ordered_lane::{SealedFacetLane, OrderedLaneWitness}` | lg@25988f3c | **I** | — | F2/F3 green: shuffled unattestable, forged/stale/lens-mismatch rejected |
| prefix → bound lowering | `quack::Filter::prefix_facet` · **tests only** (`lib.rs:2531,2532,2575`) | lg@25988f3c | **D** | first production caller (D-WFL-2) | differential vs row-scan oracle at every depth 0..=8 |
| **bound ⇄ executed planes binding** | `Planes` has no version/lens/digest (`ir.rs:50`) | lg@25988f3c | **M** | `Planes.domain: RowDomain`, bound to the PERMUTATION (W1) | wrong-version / wrong-lens / permuted-planes falsifiers, each red before the fix |
| **semantic ordinal → Morton ordinal** | nothing; the two orders are simply different (Seam D) | lg@25988f3c | **M** | measure the rotation before building on it (W3) | bytes touched, fragments, index build + footprint, reuse count — on an independently-ordered lane |
| range survives execution | `Pred::Range` → `mask_set_range(dst=full)` (`exec.rs:540`) | lg@25988f3c · nd@40a71ad | **M** | range-native terminals, zero scratch (W2a) | mask words written == 0, at every N and every position |
| bounded mask composition | no windowed operand descriptor exists | lg@25988f3c | **M** | a descriptor with global `base_word` + local length + tail + row-domain (W2b); do NOT narrow generic `Scratch` first | touched words == `ceil(width/64)+1`, flat in N and position |
| touched-window write | `d_diamond_1_probe::touched_write` — **probe-only** | lg@25988f3c | **D** | the shape W2b's descriptor generalizes | 20.5–23.1 ns flat vs 34→4,620 ns whole-lane |
| six-neighbour shift | `ndarray::simd::mask_shift_morton` · hex probe | nd@40a71ad | **I** (as a *whole-field* op) | closed-tile contract (W4) | axial BFS oracle + degree-one control (both already in the probe) |
| u8 gate predicates | `gt_u8_to_mask` etc. · hex probe | nd@40a71ad | **I** | — | in-probe |
| gated predicate | `*_to_mask_under` — skips compare on a zero gate word, still visits the gate span | nd@40a71ad | **I** | pass bounded gates, never a global one | gate-word visit counter |
| **strided facet lane** | `LaneRef::{I32,U32,U64}` — no strided variant; `ir.rs:21-27` names the gap itself | lg@25988f3c | **M** | `LaneRef::Strided{base,stride,group}`, mirroring `ndarray::simd::ternary_match_strided_to_mask` — needed only when the rail read leaves the probe (§5) | differential vs a contiguous copy of the same lane |
| tree-depth column ("G8") | nothing in ndarray `src/` | nd@40a71ad | **M** | out of scope for this arc; name it, don't assume it | — |
| alpha claim algebra | `AlphaOverlay::claim` · `wave_dispatch::dispatch_thought` (**orphan**) | lg@25988f3c | **D** | use AS IS in W6 and measure; the input-coordinate change is deferred behind that number | a cost line per stage, not a speedup |
| rung × tenant focus | `AlphaFocus` · **no caller**; every read accessor materializes a full-population mask | lg@25988f3c | **D** | NOT the slice's carrier — W5 stays tile-local | — |
| rung wave scheduler | `rung_schedule::schedule_for` · `wave_dispatch` (orphan) | lg@25988f3c | **D** | one live caller (W7) | two contexts, deterministic wave order |
| owner stamping | `SoaEnvelope::mailbox_owner()` — default 0, only its own tests | lg@25988f3c | **D** | stamp on the slice's write (D-WFL-4) | owner ≠ 0 on every published row |
| batch → DatasetVersion | `BatchWriter::cast` → `LanceCycleWriter::commit_cycle` · tests + one example | lg@25988f3c | **D** | assemble once in the slice (D-WFL-7) | exactly one new version per cycle; append-only |
| track (≤64) | `w_slot: u8` (6-bit palette) | lg@25988f3c | **HYPOTHESIS** — a physical attention slot, not established as a thought track (§8) | do not depend on the identification | — |
| rung (0–9) | `TemporalPov.rung` | lg@25988f3c | **I** | — | — |

---

## §4 Wave order (revised)

⊘ The original flat `D-WFL-1..7` list is superseded by the wave order below.
Each wave ends in a falsifier and a merge / no-merge ruling. **The governing
rule: never build the next layer until the current one proves its compact
result survives into the actual consumer.** That sentence is the whole
foldability thesis in executable form.

### W0 — separate the two attestations (the ruling that precedes W1)

⊘ **`(key, ordinal)` does not work, and the plan said it might.** If `ordinal`
means post-sort position, then `K→row-A, K→row-B` and `K→row-B, K→row-A` both
digest as `(K,0), (K,1)`. Identical. A positional index added to a key digest
attests nothing a key digest did not already attest.

The missing attestation is not another property of the KEY lane. It is an
attestation of the **associated row permutation**, and it belongs in a separate
object:

```
OrderedLaneWitness          RowDomain
  version                     same version
  lens                        same physical row population
  n_rows                      same row ORDER
  key_digest = H(K0,K1,…)     row_order_digest = H(ID0,ID1,…)
```

where `ID` is a **stable row identity** — the `NodeGuid` sequence, or the
writer's source ordinals — never the semantic facet key. The executor then
requires BOTH. Duplicate semantic keys stay legal, exactly as
`ordered_lane.rs:194` permits, and swapping the two rows behind one key changes
`row_order_digest` while leaving `key_digest`, lens, version and `n_rows`
untouched. That is precisely the falsifier W1 wants, and it is the reason the
two digests cannot be merged into one.

**W0's deliverable is a decision, not code:** find the smallest stable row
identity that ALREADY exists where the ordered lane and the execution planes are
assembled. Minting a new identity is the failure mode here.

**W0 answers WHAT is attested. W1 must answer WHO MAY MINT IT** — and without
that second answer the split is decorative, because the failure simply moves up
one level:

```
before W0:   Program.RowDomain        == Planes.RowDomain          (metadata vs metadata)
after W0:    Program.row_order_digest == Planes.row_order_digest   (STILL metadata vs metadata)
```

A caller can permute the slices and carry the old digest along for the ride.
Copying a digest field into an otherwise freely-constructed `Planes` proves
nothing at all.

### W1 — row identity and order attestation (Seam A)

- **Files:** `contract/src/ordered_lane.rs`, `mask-risc/src/{ir,exec}.rs`,
  `quack/src/lib.rs`.
- **Bind to the PERMUTATION, not to key uniqueness.** ⊘ The first draft
  recommended `SealedFacetLane` refuse to attest a lane with duplicate keys.
  **Withdrawn.** `ordered_lane.rs:194` states the shipped semantics —
  *"Equal keys are indistinguishable, so an unstable sort is exact."* Refusing
  duplicates is a semantic regression against that, and it would not have
  proven what is actually needed: equal keys are indistinguishable **to the
  comparator**, their associated rows are not, and an unstable sort may permute
  them freely. So the witness must attest the associated row identity — the
  permutation the seal applied, or an identity carried per row — never key
  uniqueness and never a key-only digest.
- ⊘ **Carrying a `RowDomain` is NOT the same as verifying one, and the first
  revision conflated them.** Comparing the program's `RowDomain` against
  `planes.domain` compares two *metadata copies*. A caller can permute a mask or
  a payload lane while keeping version, lens, key digest and row count
  identical — so the permuted-planes falsifier below would PASS against a
  metadata check, and the range would silently associate with the wrong rows.
  The mechanism has to bind the actual row identities of every participating
  plane, not a label attached to them — which is what W0's separate
  `row_order_digest` is for. Three candidate shapes for enforcing it, to be
  chosen by measurement, not by preference:
  ⊘ **First, a trap that would make (1) ceremonial.** `SealedFacetLane` is NOT
  the sealed row image — read it: `SealedFacetLane { keys: Vec<FacetCascade>,
  witness }` (`ordered_lane.rs:180`). It owns the facet KEYS and nothing else:
  no `NodeGuid` sequence, no mask planes, no value lanes.

  ⊘ **And the `AttestedPlanes<'a>` proposal recorded here earlier is RETRACTED
  as an architectural carrier.** It smuggled Rust's ownership vocabulary into
  the semantic model and made a zero-copy peek sound like a persistent execution
  object. The implementation will of course receive something spelled `&[u8]` or
  `&[u64]` while the instructions run — that is memory-safety syntax with a
  ~20 ns lifetime, and promoting it to a named aggregate builds exactly the
  intermediary this substrate exists to avoid:

  ```
  WRONG   storage -> construct execution view -> attest it -> carry it -> fold
  RIGHT   pinned canonical version -> verify the address/order contract
                                   -> PEEK zero-copy -> fold
  ```

  **The attestation belongs to the address/order RELATIONSHIP, not to a
  transient aggregate of all the planes.**

  **So what W1 must prove is sharper than "these slices belong together":**

  > ordinal `i` under this witnessed semantic order resolves to the same
  > canonical row `i` that every subsequent operation peeks.

  Once that holds, every fold independently peeks whatever canonical column it
  needs at ordinal `i`, and there is no execution assembly at all:

  ```
  RowDomain { DatasetVersion, lens identity, row-order identity, n_rows }
                              ↓
                  peek(domain, ordinal, column)        zero-copy
                              ↓
                           fold(...)
  ```

  Every operation is `fold(peek(…))`. Nothing owns the source, nothing copies
  it, nothing accumulates it, and the temporary view ideally never even gets a
  name.

  **W1 therefore begins with an assembly-boundary census**, and the census
  question is now the sharper one: *where does ordinal → canonical-row
  resolution happen today, and is it the same resolution every peek uses?*

  ```
  ONE resolution, used by every peek
     -> W1 is small: verify the address/order contract once against the pinned
        version, then peek freely.

  SEVERAL resolutions, or one nobody re-checks
     -> that IS Seam A's real depth, and finding it is the deliverable.
        The honest first implementation is a boundary check over the ACTUAL
        row ids at the point resolution is established.
  ```

  Do **not** invent a new "sealed row image", and do **not** revive a typed
  aggregate, to satisfy this plan. A digest field carried on a freely
  constructed `Planes` remains the rejected decorative option — it reproduces
  exactly the defect W0 exists to remove — but the answer to it is a verified
  address/order contract, not a bigger object.
- **Gate — the brutal case, stated exactly.** Construct a program in which
  *everything a metadata check can see is identical*:

  ```
  same semantic keys        same version        same lens
  same n_rows               same key_digest
  same RowDomain metadata — COPIED BY THE CALLER

  but: row identities A and B swapped
       and one actual value plane swapped to match

  => execution MUST refuse, BEFORE the Range is consumed
  ```

  Plus the simpler wrong-version and wrong-lens cases. Each disable-verified RED
  first.
- **Falsified if:** that program still returns the oracle's answer. It is the
  load-bearing test precisely because a metadata-only check passes it. Red
  before the fix and green only when the ACTUAL executed plane ordering is
  attested ⇒ Seam A is genuinely closed. Anything less and it is not.

### W1b — nothing transient may carry replay

With `AttestedPlanes` retracted, this law gets simpler rather than harder. There
are two kinds of thing and they must never be substituted for one another:

| | question | lifetime |
|---|---|---|
| a peek | *what do these canonical bytes say, right now?* | transient; dies within the fold |
| `RowDomain` | *which immutable coordinate system must be REACQUIRED to replay?* | owned, serializable, outlives everything |

The law, because the failure is silent:

> **Nothing required to replay may be borrowed from the execution that produced
> the replay task.** A replay descriptor may NAME immutable state; it may never
> depend on owning a live view of it.

So replay is not "keep the view alive" — and it is not "rebuild the view"
either, which was one abstraction too many. Replay **reacquires the canonical
coordinates and folds directly from zero-copy storage**:

```
ReplaySpec (owned identities only)
   -> reacquire the pinned canonical version
   -> peek
   -> fold
   -> the same answer
```

A `ReplaySpec` therefore contains **no** `&[NodeGuid]`, `&Planes`,
`&SealedFacetLane`, `&AlphaMask`, or anything else tied to a lifetime. Only:
`DatasetVersion`, row-order identity, lens / ClassView identity, program
identity, Morton / Wabe mapping identity, external-edge snapshot identity,
focus / input identity, deterministic parameters.

⊘ **Fence around the boundary-hash cache** (the census's NO branch): a cached
attestation is valid for THIS live immutable image, as a runtime execution
optimization. It is **not replay evidence, not persisted, and not part of any
`ReplaySpec`.**

**The falsifier, which also DEFINES what a legitimate replay test is** — and
therefore ranks above `D-WFL-DET`, since a determinism test run while the
original view is still alive proves nothing:

```
1. produce a ReplaySpec
2. DROP every execution object and every transient view
3. re-open using ONLY the identities in the ReplaySpec
4. peek and fold
5. result must be bit-identical
```

If step 3 secretly needs a surviving pointer, a cached view, an ordinal map or
any process-local object, the thought was never replayable.

### W2a — range-native terminals, zero `Scratch` (Seam B, half one)

- `Bound(lo,hi) → Count = hi - lo`, `→ Any = lo != hi`. Arithmetic on the
  endpoints, correct only when the program is exactly one un-`under`ed
  `Pred::Range`; gate on that. **No mask-risc scratch touched at all.**
  `execute()`'s unconditional `scratch.words == words_for(n_rows)` check must
  become conditional — and conditional on a property **derived from the
  validated program shape** (`program.requires_scratch() == false` for exactly
  the range-terminal shape), never on an ad-hoc flag a caller can assert. A flag
  is a claim; a derived predicate is a proof.
- **Gate:** `N` from 1K to 100M, same width, different absolute positions —
  **scratch words required = 0, mask words written = 0, answer = `hi - lo`**,
  latency flat in both axes. If a zero-mask program still demands
  `words_for(N)`, the wave fails.
- **Falsified if:** any mask word is written, or cost moves with `N` or position.
- This is the cleanest available proof of the whole thesis, and it is small.

### W2b — bounded mask composition (Seam B, half two)

- `Range(lo,hi) ∩ resident aligned mask → Count/Any`, touching only words
  `w0..w1`.
- ⊘ **Not** a narrowing of the existing `Scratch`. Changing
  `words == words_for(N)` to a smaller window is insufficient: operands need a
  global `base_word`, a local length, tail semantics, and a row-domain
  identity, because local word zero is not global word zero. That descriptor is
  the deliverable. **Do not modify generic `Scratch` until the descriptor has
  proved itself against a real consumer.**
- **Gate:** touched words = `ceil(width/64) + 1`, independent of `N` and of
  absolute position.

### W3 — the semantic → Morton rotation probe (Seam D) — **NEW, and the crux**

- Take `Bound_semantic(lo, hi)`, resolve the **same NodeGuids** to their Morton
  tile positions, and measure: bytes touched, fragments produced, index build
  cost, index footprint, and reuse count across queries.
- **The rotation must not be hidden in fixture construction.** A fixture that
  generates the population already in Morton order and then seals it
  semantically has assumed the answer. The probe must start from an
  independently-ordered lane.
- **Falsified if:** the rotation costs a per-row hash scatter with no reuse. That
  does not kill the architecture — it relocates the resistance, and says so.
- Nothing downstream should be built until this number exists.

### W4 — one closed Wabe tile

- CLOSED (`4^k`, trie-aligned), not a moving aperture: `mask_shift_morton`
  treats the slice as the field and drops edge carries, so a subspan is not a
  restricted global shift. `H(F) = F`, locality holds trivially.
- **Gate:** the hex probe's existing axial-BFS oracle, its degree-one control,
  and a sparse delta-frontier arm — all three already exist in
  `ndarray/examples/hex_tenant_mq_probe.rs`. Same terminal on every arm.
- **Win condition is not "hex beats everything."** It is: cost follows focused
  tile area and active frontier, not total population.
- **Falsified if:** any tile-edge mismatch against the oracle, or an advantage
  that survives the degree-one control (then it is not hex).

### W5 — focus produced by the result

- Feed the surviving frontier back as the next region. Exercise narrowing,
  translation, splitting, and reopening after an outside contribution.
- ⊘ **`AlphaFocus` cannot be the carrier yet — and the reason is a READ-side
  materialization, not anything FIRE does.** State it precisely, because the
  loose version of this sentence gets the architecture backwards:

  > **FIRE can only ever be a sparse alpha-channel delta.** That is what it IS,
  > by construction — there is no version of FIRE that reconstructs state, and
  > "rebuild after FIRE" is not a failure mode this substrate can even express.

  The defect is one layer later, on the **query** side. Verified: `cell`
  (`:122`), `any_rung_mask` (`:158` — ten times, once per rung lane), `unlooked`
  (`:175`) and `rung_reach` (`:183`) each answer *"what is focused?"* by calling
  `attended_mask()`, which allocates a full-population `AlphaMask` and scatters
  bits into it. So a sparse write is followed by a dense **read**, and routing
  the next focus through that read is what would reintroduce the population
  cost Seam B removes. W5 therefore stays **tile-local**: the frontier is
  carried forward directly, and no focus question is asked of `AlphaFocus`.
- **Gate:** one trace where changing the local result changes the next region
  processed, with the same final answer as the reference route.

### W6.0 — pin ATTEND vs EPISTEMIC FIRE (the ruling that precedes the measurement)

The current `AlphaOverlay` is **attention memory, not epistemic truth**: it
records where attention went, preserving `NodeGuid` identity, claim order, rung
and revisits. It is discardable, not canonical. That gives the minimum legal
Boolean → epistemic crossing without waiting for a full cognitive theory, and it
fits in one sentence:

> **A non-empty Boolean delta is sufficient for an ATTENTION effect, and never
> sufficient for an EPISTEMIC effect.**

So:

```
Wabe / fold result
   -> ATTEND / ELIGIBLE
        may update focus and the alpha attention trace
        does NOT revise TruthU8 / NARS evidence
   -> EPISTEMIC FIRE
        requires provenance and evidence identity
        may enter revision and durable consequence
```

This keeps `TruthU8` and NARS cleanly outside the Boolean mechanics, and it
stops the degenerate reading in which every successful intersection counts as
learning something. It also composes with the Rubicon: FIRE names the event,
ATTEND-vs-EPISTEMIC names its KIND, and the Rubicon names its DURABILITY. Three
orthogonal questions that the first draft collapsed into one.

⊘ Repeated activation is still not independent evidence, and popcount is still
not truth arithmetic. This ruling is the *floor* the crossing must clear, not
the crossing's full semantics (§8 item 5).

### W6 — publish through the EXISTING alpha route, and pay for it

- Use `AlphaOverlay::claim()` as it stands — the 512-byte `NodeRow` push, the
  `NodeGuid` hash, the scanpath order, the revisit counter. **Deliberately.**
- ⊘ `claim_ordinals` is **deferred, and split in two** — the first draft treated
  it as one change and it is not:
  - **The input coordinate** (accept an ordinal or a mask the reader already
    holds, instead of demanding a `NodeGuid` to hash). Since FIRE writes what the
    read already had, this is the API admitting what the caller is holding — not
    an optimization layered on top.
    ⊘ **But "touches no stored bytes" was too glib.** Downstream readers go
    through rows: `AlphaTunnel::merge` iterates `lane.rows()`
    (`alpha_tunnel.rs:185`) and `AlphaOverlay::rows()` exposes only `claimed`. A
    claim that pushes no `NodeRow` is therefore **invisible to publication** —
    it would vanish before W7's chain ever sees it. So the input-coordinate
    change still owes either a compact ordinal/stamp representation with every
    reader updated, or an explicit later boundary at which rows are
    materialized. Naming that boundary is part of the deferred work, not a
    detail of it.
  - **The storage contract** (`claimed: Vec<NodeRow>`, and with it `NodeGuid`
    identity, scanpath ordering and visit counts) is a genuinely different
    question, and the one the first draft would have changed by accident.
  Both stay deferred behind W6's measurement. Optimising a correct boundary
  before the full loop exists risks swapping a known-expensive correct thing for
  an elegant thing whose semantics quietly differ.
- **Deliverable is a cost line**, not a speedup: fold ~ns, bound ~100s of ns,
  rotation (W3), Wabe, alpha bytes + µs, commit. If alpha then dominates, the
  follow-up is well-posed and falsifiable: *can publication keep ordinal
  compactness without losing NodeGuid identity, scanpath order, visits,
  evidence semantics or replay?*
- **And the tier line, because writing is the expensive part.** Measure the
  three durable tiers separately — meta kanban atom, replayable task,
  materialized effect — in bytes and µs, plus their mix under the slice's
  workload. The question W6 answers is not only *what does publication cost* but
  *does a speculation cost what a conclusion costs?* If it does, the rubicon is
  not implemented, whatever the docs say.

### W7 — the second context reacts; the slice closes

- A consequence published at one `NodeGuid` becomes eligible input at the same
  `NodeGuid` under another context, and that changes the next focus.
- Owner-stamped, through `BatchWriter::cast` → `collect_casts` → `seal` →
  `LanceCycleWriter::commit_cycle` → one `DatasetVersion`, assembled outside
  `tests/`. No new transport type, no per-cell or per-thought actor message, no
  `KanbanActor` resurrection (that actor was deleted, not deprecated).

### The ISA residue IS the wave plan

The small orthogonal instruction set T1 wants, minus what `lance-graph-mask-risc`
already ships, lands exactly on the open waves — two independent routes arriving
at the same missing five:

| op | state | wave |
|---|---|---|
| AND / OR / XOR / NOT / ANDNOT · TERNLOG · GATE (`under`) · ANY / ALL / COUNT / REDUCE | **shipped** | — |
| BOUND | half — `Pred::Range` carries the result; the search is in `quack` | W2a |
| ADDRESS | implicit, unattested | **W1** |
| carrier transforms — range ↔ runs ↔ bounded words, ordinal mapping | **missing entirely** | **W2b** |
| ROTATE | **missing** | **W3** |
| SHIFT / NEIGHBOUR / STENCIL | **missing** (`mask_shift_morton` is an ndarray primitive, not an ISA op) | **W4** |
| PEEK, strided | half — `LaneRef{I32,U32,U64}`; `ir.rs:21-27` names its own gap | deferred |
| PROJECT (lens) | **missing** — the lens lives in `contract::facet`, never in the ISA | after W3 |
| FIRST | **missing** | unscheduled |

### Deferred behind the slice

Compact alpha (the W6 follow-up) · the full orchestra · a moving halo and
cross-tile scheduling · Gaussian / weighted influence · the Boolean↔epistemic
crossing · `LaneRef::Strided` · G8.

## §5 The first slice — concrete fixture

**Population.** 65,536 rows — the hex probe's size, so its oracle and its
degree-one control transfer unchanged. One `NodeRow` per cell.

**Two lenses.** (i) `SemanticLens::CanonHighTiles8` over the **key** facet —
shipped, sealed, witnessed. (ii) A second lens over the **second** facet's rail
plane, which is a new `SemanticLens` variant and therefore new implementation,
not assembly.

**No real relationship bytes are read as propagation gates.** The slice's
`P_d` comes from SYNTHETIC tenant bytes the probe generates, exactly as
`hex_tenant_mq_probe` does. Reading a node's actual second-facet rail bytes as
`(permeability, strength)` would reinterpret operator-locked relationship
semantics (`le-contract.md` §3 sanctions `basin:relationtype` and two orthogonal
relation types on that plane — never permeability and strength), and would
silently turn stored relations into propagation gates. That reinterpretation
requires an explicitly sanctioned Wabe ClassView, which does not exist and is
not this arc's to mint (§8 item 2).

**How the slice reads the rail plane — stated, not assumed.** ⊘ The first draft
deferred `LaneRef::Strided` while §5 required a second-facet lens; that was a
contradiction. Resolution: the slice runs in the **probe crate**, which owns its
own data layout and can hold the six rail bytes in a contiguous `[u8]` tenant
array exactly as `hex_tenant_mq_probe` does. `LaneRef::Strided` is required only
when the read moves into the mask-risc IR over real 16-byte-stride rows, and
that is a later wave. The slice must say which of the two it is using in its own
header, every time.

**The rotation is explicit.** The chain crosses Seam D, and the crossing is a
named, measured step (W3) — never a bare `∩`:

```
witnessed prefix on lens (i)   ->  Bound{lo,hi} + RowDomain           [W1]
Bound                          ->  Count / Any, no mask at all        [W2a]
Bound x resident mask          ->  touched words only                 [W2b]
Bound_semantic                 ->  ROTATION  ->  Morton positions     [W3]  <-- measured
Morton positions ∩ closed tile ->  A_0
A_{t+1} = (A_t ∪ ⋃_d S_d(A_t ∩ P_d)) ∩ T                              [W4]
delta = A_{t+1} \ A_t
delta empty      ->  MASK emptiness test on delta itself, nothing published
delta non-empty  ->  AlphaOverlay::claim() as it stands, measured     [W6]
                 ->  owner-stamped cast -> one DatasetVersion
same NodeGuid, context B, reacts; next focus changes                  [W7]
```

⊘ **The publication decision tests the DELTA, not the bound.** W2a's
`Terminal::RangeAny` is endpoint arithmetic (`lo != hi`) valid only for a single
un-`under`ed `Pred::Range`; `delta` is an arbitrary, possibly fragmented mask
produced by `A_{t+1} \ A_t`. Applying the range terminal to it reports "publish"
whenever the *prefix* was non-empty — so every converged step, where propagation
has reached a fixed point and the delta is empty, is misclassified as an
insight. The no-publication decision needs an emptiness test over the bounded
delta carrier itself. Keep `RangeAny` for what it is; give the delta its own.

**Two thought contexts.** Two rungs via `TemporalPov`. The *track* coordinate
stays a hypothesis (§2) and the slice must not depend on `w_slot` meaning a
thought track — see the next paragraph, which is now closer to a refutation
than a caution.

**The `w_slot` hypothesis is in trouble, and the slice must not lean on it.**
`AttentionMaskSoA::touch` (`attention_mask.rs:83-88`) locates an entry by
`mailbox_id` ALONE and overwrites its `w_slot`. So one mailbox holds exactly one
W-slot: touching a second track at the same node erases the first rather than
leaving both visible. A carrier that cannot represent two concurrent tracks at
one address is not a thought-track carrier. Either the slice defines a separate
mailbox identity per track and carries that mapping explicitly, or the
identification is wrong. **Do not write an acceptance criterion that assumes two
tracks are simultaneously visible at one `NodeGuid`** — the shipped carrier
cannot satisfy it. (§8 item 1.)

**Invariant, asserted by counter and not by comment:** between a successful
bound and tile entry, no allocation or write is sized by the population.

**Irregular ingress.** One explicit non-local edge, entering focus sideways,
proving a remote contribution can reopen a region the local recurrence closed.

## §6 Measurement plan

`H_exec` and `R_info` are usable only with the units pinned. Adopt the doc's
byte form, and pin these conventions:

- `a_i` = bytes of the **agreed exact answer carrier**, floor 1 (the floor is a
  measurement convention for the empty answer, never a layout requirement).
- `m_i` = **peak operation-local intermediate + output bytes**, counting active
  preallocated scratch. Zero heap allocations is not zero traffic.
- Reading a pre-existing canonical row is **not** materializing a candidate.
  Report canonical bytes read as its own column.
- Report separately, never summed: temporary representation · canonical reads ·
  gate-word visits · output-word writes · run fragmentation · remapping · index
  build **and maintenance** · publication bytes.
- **Identical terminal contract on both arms** of every comparison. The existing
  8,200× P3 figure is only honest because `fold_materialize_ns` includes the row
  extraction the comparator's dense mask implies.

Varied independently, limited to the first slice's actual risks:

| axis | why this slice needs it |
|---|---|
| total population | Seam B: cost must not follow it |
| **absolute focus position** | the axis a fixed-position sweep is blind to |
| focus width | the axis cost *should* follow |
| fragmentation (runs) | a union is not always one interval |
| propagation steps | the recurrence's own cost |
| irregular-edge rate | the sideways ingress |
| publication rate | including a zero-publication step |

Active density and word occupancy are deferred — the slice has one density.

**Do not transfer:** #1245's ~1.72 ns axis chain does not describe a whole
thought (#1250 declined this explicitly, measuring 1.7–4.2 ns for the whole-facet
cell). #1250's ~8,200× indexed intersection is CONDITIONAL on a 61.2 ms/1M-row
prebuilt `JointIndex`, equal depths, `JOINT_MAX_DEPTH = 4`, and one paired-prefix
query family. It is not a free join and not a general one-interval Morton theorem.

The credible ambition is not *thought = 1.7 ns*. It is: **keep returning to the
1–100 ns family instead of repeatedly exploding into µs/ms population work.**

---

## §7 What the slice proves, and what it does not

**Proves:** that a fold can reach a local field and a publication without a
population-sized intermediate; that order binding is enforceable rather than
assumed; that focus can be an output; that one consequence at a NodeGuid is
visible to another context at the same NodeGuid; and that the six-neighbour
tenant needs no new storage.

**Does not prove, and must not be claimed:** a general query lowering; a moving
aperture with correct halo (closed tile only); 64 tracks or 10 rungs (two of
each); weighted/Gaussian influence (Boolean only); NARS revision (repeated
activation is **not** independent evidence); BLASGraph integration (one edge);
any statement about total-system throughput.

Remaining to build after it: halo + cross-tile scheduling, the lens catalogue,
`LaneRef::Strided`, G8, the epistemic crossing between Boolean occupancy and
`TruthU8`, compact alpha (the W6 follow-up), and the conductor over the full
track set.

---

## §8 Deliberately NOT decided here

These are cognitive semantics, and an implementation session must not canonize
them by noticing that two things fit in the same number of bits. Each blocks a
later wave, none blocks W1–W4.

1. **What is a thought track?** Is `w_slot` its semantic identity, or merely a
   carrier with the right cardinality? (§2 marks this HYPOTHESIS.) Blocks the
   orchestra.
2. **What is a Wabe direction?** Which ClassView makes six rail positions mean
   six axial directions, and who declares it? (§2.) Blocks W4's tenant reading
   moving out of the probe.
3. **What survives FIRE?** FIRE is a sparse delta that writes what the read
   already had — that much is settled. What is NOT settled is which *kind* of
   thing the delta carries: activation, attention, evidence, belief revision,
   inhibition. Without this, "novel consequence" degenerates to "the delta was
   non-empty" and every non-empty intersection publishes. **Blocks W6.**
   Its twin, given replay: **when is an insight better stored than replayed?**
   A replayable task is complete only if nothing about the world it questioned
   has moved; the version-pinned `RowDomain` says when that holds, but the
   *policy* — which insights earn a materialized answer — is a cognitive call,
   not an engineering one.
4. **What is the minimal meta-awareness carrier?** It should not default to N
   full-population masks merely because `AlphaFocus` currently represents the
   rung × tenant cross that way. The 64-bit per-node track summary is a
   *possible* representation, never a mandate to materialize a dense cross.
   Blocks W5's generalization beyond tile-local.
5. **Where may Boolean occupancy legally cross into epistemic truth?** The
   FLOOR is now pinned in W6.0 — a non-empty Boolean delta is sufficient for an
   attention effect and never for an epistemic one — which unblocks W6 without
   a full theory. What remains open is the crossing's actual semantics: what
   provenance and evidence identity an epistemic effect must carry, and how
   revision consumes it. Occupancy and popcount are not `TruthU8` arithmetic,
   and repeated activation is not independent evidence.
6. **The Rubicon policy itself.** Given a deterministic sparse cognitive delta,
   what properties justify (a) forgetting it, (b) retaining only a
   hypothesis / meta-kanban atom, (c) retaining a replay specification, or (d)
   materializing it as world state? Each boundary needs its minimum evidence and
   its minimum replay identity, **without conflating attention, novelty,
   confidence and truth** — four things the substrate keeps separate and prose
   keeps merging. This is the question the whole write-tier ladder rests on, and
   it is the one an implementation session is least equipped to answer.
