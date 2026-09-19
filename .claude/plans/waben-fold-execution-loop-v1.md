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

### Where the law lives: a membrane, not a tier — and T1 ≠ T2

The law above does not mint a tier. `membrane-tiers.md:48` says outright that
the ladder does not need another one. **"Layer 0" is the photolithographic
computational MEMBRANE**, and it spans two existing tiers that must not be
flattened into each other:

```
            METACOGNITION
                  │  compiles a thought
                  ▼
      mask-risc Program / MaskOp / Pred / Terminal      T2
      ──────────────────────────────────────────────
        the photolithographic RISC PLAN LANGUAGE
                  │  names primitive ops
                  ▼
      ndarray::simd — mask_*, ternlog, popcount, …      T1
      ──────────────────────────────────────────────
        the zero-copy primitive EXECUTION algebra
                  │
                  ▼
            canonical state                            T0
```

⊘ **An earlier revision of this section said "mask-risc is T1's ISA." That was
wrong, and it contradicted this document's own legend two sections up** (§3:
*"T1 = `ndarray::simd` facade + mask ALU, T2 = behaviour (exec, lowering, alpha
algebra)"*). The receipt is one line of the crate itself: `exec.rs:25` is
`use ndarray::simd::{…}` — mask-risc **consumes** T1, so it is T2 by the
doctrine's own definition. The DuckDB analogy makes the same point: a planner is
not the vectorized primitive it dispatches.

So the ruling is: **`mask-risc` is the existing T2 RISC plan language that
composes T1 primitives without exposing population mechanics upward.** Neither
needs minting; neither is the other.

⊘ **And T1 is NOT universally zero-copy — do not say it is.** T1 contains
primitives that write mask outputs (`mask_set_range`, every `*_to_mask` compare,
the import paths). The accurate statement: *the Layer-0 FOLD SUBSET executes
through T1 primitives zero-copy; T1 also contains explicitly materializing
primitives, and those are **reconstruction** operations when invoked that way.*
They remain legitimate substrate machinery. What the law changes is their
**classification inside a Layer-0 program**, never their right to exist.

Which yields the membrane law this whole plan operates under:

> Higher layers may invent arbitrary cognition. They may NOT invent new
> population execution semantics — they compile cognition into T1.

No thought style owns intersection, prefix, locality, range, projection or
rotation. Those are physics. A style owns only *which* T1 operations it composes
and *why*. **ISA, not standard library:** a new T1 primitive earns existence only
by exposing a genuinely new zero-copy operation the existing algebra cannot
express by composition or fusion — `fuse.rs` is the precedent, collapsing a
Boolean tree rather than growing a variant per shape.

**And the law gives the plan language a conformance criterion it did not have —
a NEW one.** Every scratch plane is required to be `words_for(n_rows)`
(`exec.rs:566`), so every `MaskOp` today emits population-sized output.

⊘ **Do not claim the old doctrine already forbade this; it did not, and an
earlier revision here said so.** `membrane-tiers.md:22` lets T1 return *"a mask,
a count, a lane descriptor — never the population"*, and **a full-length bitmap
is still a mask under that wording.** "Never the population" historically meant
*do not return rows or arrays of the represented population* — it never said a
derived mask sized to N is itself forbidden. Starting a constitutional law by
back-dating it leaves a crack anyone can quote the table back through.

The honest statement is two laws, one strictly stronger:

```
OLD T1 LAW    may return a Mask; must not return the Population
NEW FOLD LAW  a FOLD writes NO derived software representation at all —
              it returns a descriptor, a reduction, or a register consequence
```

⊘ **An earlier revision stated the new law size-conditionally** — *"may not
materialize an N-sized derived representation when its compact consequence can
stay a range / runs / bounded window / scalar"* — which contradicts *"folds are
zero copy, period"* two lines above it. **Zero-copy is not a size threshold.** A
smaller materialization is still a materialization, and a size-graded rule is
exactly the loophole through which, six months from now, someone argues that a
fourteen-word buffer is "basically zero-copy". The frozen ducks come back in
tiny hats.

**The line is whether Layer 0 wrote a second software-visible representation at
all:**

```
FOLD CARRIERS                        NOT FOLD RESULTS
Count                                a populated bounded-mask buffer
Any                                  [u64; 12] filled from an intersection
an ordinal                           Vec<Run>
[lo, hi)                             a full mask
base_word + length descriptor        any newly written derived buffer
(a run DESCRIPTOR, if it names
 rather than populates)
```

A descriptor **names** an answer or a region. A buffer **holds** one. Only the
first is a fold carrier, at any size.

`Pred::Range → words_for(N)` is a fold-conformance failure under the **new**
law. Recorded as a doctrine sharpening, not as something already entailed. So,
now, is a bounded-mask write — and that is the point of stating the law without
a size clause.

**And the generous corollary that keeps the law usable: not every full mask is
illegal.** If the consumer genuinely demands a population mask as its answer,
producing one is legitimate — it simply **is not a fold**. `mask_set_range` over
a full destination is not forbidden code; it is **misclassified execution** when
it happens inside a fold. Which is exactly the A/B split:

```
A) COMBINE      PEEK → PROJECT → BOUND → ROTATE → AND → TERNLOG → …
                zero-copy, compact carriers, no population materialization

B) RECONSTRUCT  a Layer-0 result
                   ↓  an EXPLICITLY NAMED materialization boundary
                full mask / rows / SoA state / publication
```

Folds are zero-copy, period. **Reconstruction is not a fold** — and it does not
get to hide under the word.

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

### Three convergences, and they must never be conflated

The termination question this plan kept deferring ("when is thinking finished?")
has three separate answers, and collapsing any two of them is how a budget
becomes a superstition:

```
TARSKI   have I exhausted the consequences?      X_{n+1} = X_n
SHANNON  am I still learning anything?            ΔH ≈ 0
JC       is the apparent information bigger than substrate noise,
         dependence and representation drift?
```

**TARSKI gives `delta == empty` a reason.** On a finite lattice
`L = (P(E), ⊆)` with monotone operators (`X ⊆ Y ⟹ F_H(X) ⊆ F_H(Y)`), the
inflationary chain `X_{n+1} = X_n ∪ ⋃_H F_H(X_n)` must stop, and its stopping
point is the least fixed point of the admitted operators. So W5's empty-delta
test is not a heuristic — it is lattice termination. No mystical done-thinking
detector.

**But ANDNOT, retraction, confidence decay, inhibition and counterfactual
replacement DESTROY monotonicity.** They are not illegal; they belong to a
different phase. A two-stroke engine:

```
MONOTONE CLOSURE    accumulate admissible consequences -> Tarski fixed point
NON-MONOTONE REVISION   retract / counterfact / revise / decay
                        -> a NEW starting state
then MONOTONE CLOSURE again
```

⊘ This plan's W4/W5 recurrence `A_{t+1} = (A_t ∪ ⋃_d S_d(A_t ∩ P_d)) ∩ T` is the
monotone stroke only. The `ANDNOT explained` step in §5's chain is already the
other one — so the slice crosses the phase boundary and must say so.

### The scheduler: eligibility is a cheap gate, not a score

```
X_{t+1} = X_t ∪ ⋃_H  1[ G(f_H, c_H, T_H, B_H, C_t) ] · F_H(X_t)
U(H)    = IG(H) · R_JC(H) / C_fold(H)
stop when X_{t+1} = X_t, or earlier when max_H U(H) < ε
```

`G` is composed from currencies that **already exist**, read-verified in
`crates/causal-edge/src/layout.rs`:

| currency | bits | verified |
|---|---|---|
| `f`, `c` | evidential | the CE64 truth pair |
| `CausalTopology` | **59–60** | `{Direct, IndirectKnownIntermediates, IndirectUnknownIntermediates, Unknown}` — documented as an *"additive factual view over `TrustTexture`"*: ONE field, two lenses, not two variables |
| `ReasoningBand` | **61–63** | `{Surface, Association, Relation, Causal, Counterfactual, Perspective, Meta, Transcendent}`, with explicit orthogonality notes to `CausalMask`, the inference mantissa's −6 slot, and `direction` |

**The band is a semantic sandbox, and this is the load-bearing part.** A donor
thought at `band = Relation` with `f = .81, c = .94` and spectacular information
gain **must not be silently promoted to causal knowledge.** High IG makes it a
*candidate causal investigation*, which then requires an explicit Causal-band
operation before any new CE64 is written. Likewise `Counterfactual` can explore
freely without mutating factual causal state. **Usefulness is not causal
licensing** — recorded here on its own merits. ⊘ The first cut said a citation to
"#1224" *does not resolve*; that was a search of `.claude/` and `docs/` only.
**Corrected 2026-09-19 by reading the PR itself, and corrected AGAIN the same
day:** #1224 exists and was **withdrawn and closed without merging because it
was fundamentally wrong** — it converted a criticism into an invented
maturity-ladder specification, and it placed causal-licensing semantics inside
a DisMech module against the ruling that DisMech is not a thinking atom. ⊘ The
first correction here said the helpers were "deleted after a measured zero
production consumers" as if that were the reason; it was not. The zero-consumer
count only meant the deleted helpers needed no re-homing. Making a side
measurement into the verdict is itself the error #1224 warns about. It is cited
here as a *negative* receipt, never as architecture: a helper existing is not
enforcement, and association or usefulness does not grant causal status.

And `ReasoningBand::Meta` earns its keep without any meta-opcode: bind the
fold/mask machinery to the thought programs and transfer histories themselves —
*which donors transfer? which repeatedly fail? which masks are worth caching?
which motifs collapse uncertainty fastest?* Reasoning about reasoning, exactly
as the variant's own doc comment says.

### JC is the brake, and the Jirak trap is already an iron rule

Read-verified in `crates/jc/src/`: `jirak.rs`, `cartan.rs`, `weyl.rs`,
`drift.rs`, `ewa_sandwich{,_3d}.rs`, `reliability.rs`, `quorum.rs`, `pearl.rs`,
and `stats.rs` (`cohen_kappa`, `omega_total`, `phi`, `binary_association`,
`kr20`, `multiple_r_squared`, `eta_squared`, the t-test family).

⊘ **`jc` contains NO Shannon/entropy implementation** — zero hits for
`shannon`/`entropy` across its source. So the information-gain half of the
scheduler **does not exist and would have to be built.** Naming it is not having
it.

**And the trap it must not fall into is already `I-NOISE-FLOOR-JIRAK`**, one
level up. 500 candidate thoughts sharing prefixes, vocabularies, R2IL motifs and
masks are **not independent lottery tickets**. Raw `−Σ p log p` is fine as a
descriptive quantity of a distribution; an *information-gain claim about this
substrate* needs dependence calibration, because the iron rule already
establishes that classical IID Berry-Esseen is wrong for these fingerprints.
`R_JC` is therefore not one scalar — it means *passes the relevant dependence,
noise and reliability gates*.

**Keep the three convergences apart in every report.** Tarski convergence
(`X_{n+1} = X_n`) is semantic exhaustion. JC convergence is numerical
stabilisation within a calibrated error regime. `ΔH ≈ 0` is diminishing returns.
Conflating them produces a thinking budget that stops for the wrong reason and
cannot say which.

### A VARNODE IS NOT A BUFFER — and the architecture is already in the tree

Same category error as the section below, one level over. Cross-repo reads
(OGAR `5055b06`, r2sleigh `99d2553`, lance-graph this branch):

> **A varnode is not a buffer.** R2IL varnodes carry behavioral DEPENDENCIES,
> not an obligation to materialize their values. `v2 = ternlog(v0,v1,…)` may
> never exist as bytes.

Compiler instinct reads `v0 = …; v1 = AND(v0,…)` as *allocate a representation
per name*. The intended semantics is SSA in a **fused dataflow engine**: each
varnode names a logical value / address / membership expression, and only a
terminal that asks for membership as a carrier forces bytes.

**The fragments are already built, and nobody had joined them:**

- `ogar-r2il/src/lib.rs` module doc — *"proxy glue: r2sleigh's R2IL opcode set
  as an `ogar_loco::Vocabulary`, plus the **masked lane projection** that
  re-reads one already-written body under any `LaneShape` **without rebuilding
  it**."* That last clause IS the non-materializing re-projection this plan
  spent a day deriving — shipped, in another repo, under another name.
  `project()`, `project_r2il()`, `r2il_mask()`, `CallMask{shape: LaneShape}`.
- `ogar-loco/src/basin.rs:94-98` — *"Orchestration is agnostic; the thinking IR
  is a caller… `ogar-r2il` plugs its vocabulary and codebooks in."* The wrapper
  this needs is **`Vocabulary`**, already the seam.
- `ogar-r2il` carries **no `r2sleigh` dependency** by design: 82 arities as a
  table, pinned to the source enum by a drift test. The opcode set travels as an
  *arity table*, not an object graph.
- `LOCO-ORCHESTRATION-GAP.md` — R2IL occupies one **classid** vocabulary while
  query / NARS-tactic / Blockly occupy their own through the same registry.

**⊘ Two corrections to the premise, both read-verified:**

1. **r2il is STRONGLY TYPED, not "nontyped".** `r2sleigh/doc/r2il.md`: *"r2il is
   a strongly-typed intermediate language based on Ghidra's P-code operations.
   Every operation has explicit input and output varnodes with known sizes and
   address spaces."* It exists precisely to fix ESIL's untypedness. **And that
   strengthens the design rather than weakening it:** it types the MACHINE
   (sizes, address spaces) and never the MEANING — which is exactly the
   "keep the opcode table embarrassingly mechanical" property worth protecting.
   ABI-friendliness comes from sized varnodes + explicit spaces + serde + the
   arity table, not from absent types.
2. **Name collision, and it is the shape that bit this arc twice today.**
   `membrane-tiers.md:24-25` places an **"R2IL" at T3** — *"emits T3 artifacts…
   its ceiling IS T3's; door-knocker test"* — beside the Java facade and
   low-code. The R2IL here is behavioral microcode far below that. Two things
   named R2IL at opposite ends of the ladder is exactly how T1 and T2 got
   flattened this morning. **Whichever survives, the other needs renaming
   before either is built against.**

**The sentence that joins the fragments:**

> R2IL is **not** the harvested representation of machine code. R2IL is the
> **vocabulary-neutral behavioral microcode for masked thinking.** Harvesting
> via r2sleigh/ruff is one PRODUCER of R2IL programs. `ogar-loco` orchestrates
> them, `classid` selects their vocabulary, and the substrate executes their
> mask/fold expressions **without materializing intermediate populations unless
> a terminal explicitly requests one.**

```
Odoo / SPOG / UI / LLM / harvested code / cognition
        ↓  ogar-loco        branch · loop · call · suspend; vocabulary by classid
        ↓  ogar-r2il        tiny behavioral program: varnodes + operators
        ↓  bindings         ogar-ro · quack · NARS/… — named domain operators
        ↓  MaskExpr/FoldExpr
        ↓  zero-materialization execution
   ANY / COUNT / FIRST / BOUND / NEXT_FOCUS / KEEP / PUBLISH
```

**The whole doctrine, shorter than this plan's Layer-0 prose:** *SPOG says what
exists. ClassView says how to see it. `ogar-loco` says what to do next. R2IL says
how the thought behaves. Mask/fold algebra executes it without constructing what
it can merely observe.*

**And it re-reads the harvest.** Not *"assembly instructions we collected"* but
*millions of tiny programs humans already wrote to transform, compare, gate,
branch, select, normalize, search and decide.* BPE/macro mining over R2IL
streams then discovers **reusable behavioral motifs**, promotable to callable
operators and bindable to entirely different classid-selected vocabularies —
harvest → R2IL → motif discovery → promote → bind → execute as masked thought.
⊘ CONJECTURE: no motif mining has been run; this names the endgame, not a result.

### THE GLOBAL PRIMITIVE: a mask expression does not imply a bitmap

Everything above this line circles a symptom. **The primitive is one level up
from folds**, and it is the thing the whole arc was missing:

> **Mask algebra is globally non-materializing by default. A mask EXPRESSION
> denotes membership; it does not imply a bitmap exists. Materialization occurs
> only at an explicit TERMINAL, when the membership set itself is requested as a
> carrier.**

That is stronger and more accurate than *"folds are zero-copy"*, because it lets
folding, masking, ternlog, gating, projection and reduction all participate in
**one** zero-materialization algebra. The fold was never the whole trick. The
trick is that **the expression can remain unevaluated as population state all
the way to a low-entropy terminal** — DuckDB's pipeline insight, in a semantic
substrate.

**Three concepts, and this plan collapsed 1 and 3 for a full day:**

```
1. MASKING            a Boolean / ternlog / gating OPERATION.
                      May be completely zero-materialization.
2. MASK EXPRESSION    a composition of predicates, folds, fields, populations.
                      Still need not exist as a bitmap.
3. MATERIALIZED MASK  an actual membership bitmap, chosen because its BITS are
                      useful downstream.
```

Collapsing 1 into 3 is why the discussion oscillated between *masks are
wonderful* and *masks violate folds*. Both were true of different referents.

**And `ClassView` × `WideFieldMask` were designed for exactly this.**
`WideFieldMask` is a field-PARTICIPATION currency, not a tiny bitmap: it says
*only these semantic facets take part*. Stack the apertures optically —
ClassView × WideFieldMask × semantic bound × focus × permissions × temporal POV
× another dataset — and measure what survives. **You do not manufacture a new
transparency after every aperture.** Cognition and rendering are then the same
photolithographic machine with different lenses and terminals:

```
rendering   ClassView(render) × WideFieldMask(visible) × viewport × tenant
              -> render descriptors
cognition   ClassView(read)   × WideFieldMask(relevant) × bound × focus × set B
              -> Any / Count / next focus
```

Neither inherently needs a population mask. If rendering wants the same survivor
set for five passes, or cognition wants to share a hot one across 10,000
continuations — **then** materialize, once, deliberately.

### The election is ALREADY IN THE ISA, and the ops annihilate it

This is the read-verified part, and it relocates the defect:

```
Terminal::Keep { mask }   "The final mask itself stays in `mask` (a scratch slot
                           the caller reads back); nothing is reduced."   ir.rs:184
Terminal::{Count, Any, All, MaskedSum/Min/Max}   reduce; the membership set is
                                                 never requested as a carrier
```

**`Keep` IS the materialization election.** The distinction was designed in. But
one level below, `MaskOp::And { a, b, dst }` is documented `dst = a & b` — and
every `MaskOp` is likewise an **assignment to a destination slot**. So the ops
destroy at level N−1 exactly the choice the terminals encode at level N, and
`exec.rs:566` then forces every one of those slots to `words_for(n_rows)`.

> **`MaskOp` must not semantically mean "produce a Scratch mask". It must mean
> CONTRIBUTE TO A MASK EXPRESSION. Scratch is one possible physical LOWERING,
> never the semantics.**

**Which reframes `Pred::Range → Scratch` as a symptom, not the disease.** Fixing
`Range` alone would leave `And`, `Or`, `Xor`, `AndNot` and `Ternlog` all still
writing full planes. The correction belongs to the execution MODEL:

```
MaskExpr:  A AND B ANDNOT C TERNLOG D,E,F RANGE lo,hi GATE focus
     │
     ├─ Terminal::{Any, Count, First, Reduce}
     │     the fuser does whatever register/SIMD work is needed and
     │     NEVER emits intermediate membership bits -> returns a u64
     │
     └─ Terminal::{Keep, Cache, Publish}
           permission to create the bitmap
```

**And half the "missing primitives" then dissolve into lowering rules.** The
fused `popcount(a & b)` of `D-WFL-T1-FUSED` is not a bespoke instruction to add
— it is what a fuser emits for `MaskExpr → Terminal::Count`. `fuse.rs` already
exists and already collapses a Boolean tree into one ternlog; what it does not
do is fuse **across the op → terminal boundary**, which is precisely the
boundary this ruling moves.

**⊘ Consequence for the wave plan, and it should be settled before W0/W1:** the
deepest correction is to `MaskOp`'s semantics, not to `Pred::Range`. If that is
pinned first, several waves shrink from "add a primitive" to "add a fusion
rule".

### Masking is an OPERATION. A mask is a CARRIER. Never confuse the two.

⊘ **The section below frames the choice as "FOLD path vs MASK path", and that
axis is wrong** — it implies electing to mask means giving up zero-copy. It does
not. **You can fold two datasets and mask them against each other with no
materialization at all**, and that is not a compromise; it is probably the ideal
Layer-0 operation:

```
dataset A ──fold──┐
                  ├─ AND / TERNLOG / gate ──> tiny answer
dataset B ──fold──┘

    no mask population ever exists
```

The membership relation lives **logically, in registers**. The answer is a
`Count`, an `Any`, a `[lo,hi)`, a `First`, a next focus. This is the
photolithography metaphor landing exactly: **shine two patterns through each
other and measure where the light survives — you do not manufacture a
transparency showing every surviving pixel.**

So the taxonomy has three independent axes, not one binary:

```
OPERATORS            CARRIERS                MATERIALIZATION CHOICE
fold                 canonical lane          fused / zero-copy
mask · ternlog       range                   materialized bitmap
project              descriptor
rotate               resident mask
neighbour            cached mask
reduce               …
```

**The entropy principle that falls out, and it is the sharpest statement of the
whole arc:**

> **Representation entropy should follow ANSWER entropy.**

*"Do these two million-row semantic regions intersect?"* carries about one bit.
Constructing 125 KB of mask to discover that bit is the obscenity — and 125 KB
is not rhetorical, it is Seam B's measured number at N = 1M. *"How many
overlap?"* is 32 or 64 bits. Both belong in the fused arm.

Whereas *"give me the overlap, because six later thoughts will manipulate it
spatially"* justifies materializing: the bitmap is now the **low-entropy working
representation relative to its future workload**, even though it dwarfs the
immediate scalar.

**So the BBB's precise question is not "fold or mask?" but:**

> **Is this membership relation transient algebra, or has it been PROMOTED to a
> mask carrier?**

That promotion is the deliberate boundary. Everything the section below says
about elections and visibility is right; it is the *axis* that needed fixing.

**Shortest form of the whole doctrine: fold the datasets, mask the folds,
materialize only when the mask itself is worth keeping.**

### FOLD and MASK as physical plans — read through the correction above

The shortest form of everything below: **masks are allowed; accidental masks
aren't.**

⊘ Four sections of this plan read as *fold good, mask bad*. That was never the
claim and it would sabotage the machine. **Both are first-class execution
strategies**, and *"folds are zero copy, period"* stays exactly true — it
defines what a FOLD is, not what the machine is allowed to do.

```
FOLD                              MASK
canonical bytes                   canonical bytes / folds / other masks
   ↓ peek / bound / compose          ↓ materialize or REUSE a bitmap
   ↓ reduce                          ↓ mask algebra / cache / fan-out
compact answer                    a resident plane
```

**When each is attractive:**

| choose FOLD when the output entropy is low | choose MASK when the mask itself has computational value |
|---|---|
| `Count`, `Any`, `Bound`, `First`, a descriptor | reused many times · shared by many thoughts |
| the answer is consumed once | AND / OR / TERNLOG fan-out |
| | ~11 ns lookup · a resident attention/focus plane |
| | an expensive derivation worth caching |

At the point a mask is reused twelve thousand times, insisting on recomputation
*because folds are pure* is self-sabotage. The decision is economic and
semantic: **choose MASK if `C_build + C_reuse < C_repeated_fold`, or if a later
operation genuinely wants mask algebra.** Both of these are correct for 64K
thoughts at once:

```
thought A  -> fold, because it is asked once
thought B  -> fold once -> mask -> reused by 12,000 thoughts at ~11 ns
```

**So the rule is about the TRANSITION, not the bytes:**

> The crossing from fold-native to mask-native execution must be **deliberate
> and visible at the T2 planning membrane.**

```
T2 planner
   ├── FOLD path   preserve the compact carrier; reduce directly
   └── MASK path   explicitly ELECT the bitmap; materialize / cache / reuse,
                   and then behave like a mask engine without shame
```

What is forbidden is only this:

```
the planner believes it is executing a fold
        ↓
a helper silently allocates words_for(N)
        ↓
everything downstream is mask-native — and NOBODY MADE THE DECISION
```

**Which restates Seam B more precisely than anywhere above.** The defect in
`Pred::Range` is not that it writes a mask. It is that the planner has no way to
elect that and no way to decline it — **there is only one path, so the choice
does not exist.** Seam B is an *absent decision*, not a present mask.

**And the BBB question becomes answerable:** *who decided this computation should
become a mask, and on what basis?* It need not be runtime cost estimation at
first; static plan knowledge is enough to start:

```
terminal Count           -> stay FOLD
one AND then Count       -> probably stay FOLD
reuse_count > 1          -> consider MASK
shared cached result     -> MASK
Wabe frontier reused     -> maybe MASK
a ~11 ns cached mask     -> almost certainly MASK
```

DuckDB-style dynamic costing can come later. This is pipeline-vs-materialize,
and it is a solved shape.

### Frozen is fine. Marching is the disaster.

⊘ **The previous section's rule slid from a definition into an anti-cache
position, and that is wrong.** *A fold is zero-copy* remains exactly true, and
writing a bounded mask remains not-a-fold. What does NOT follow is that writing
one is a sin. It is a **cache decision**, with its own economics.

**The real boundary was never copy vs no-copy. It is
recompute-or-freeze vs continuously maintain.**

```
FOLD         canonical state -> zero-copy computation -> consequence
CACHE MISS   fold consequence -> materialize it ONCE, deliberately
CACHE HIT    cached mask -> zero-copy peek -> ~11 ns
```

If a cached result is callable in ~11 ns, discarding it because it once crossed
a materialization boundary would be absurd. So 64K cached masks are entirely
welcome:

```
64K cached masks
   no sweeping · no incremental refresh · no coherence work
   no CPU while dormant
          -> frozen ducks, and FROZEN IS THE POINT
```

Frozen is cheap. **Marching 64K ducks around every cycle is the disaster**, and
that — not storage — was always the enemy.

> **Materialization is allowed when its amortized retrieval value earns it.
> What is forbidden is entropy accumulation solely to keep derived state
> current.**

**So COMBINE vs RECONSTRUCT is an EXECUTION DECISION, not a permanent type
distinction:**

```
thought P over domain D, first use:  replay folds -> result R
   cheap / unlikely reuse    -> discard R
   expensive / likely reuse  -> cache R
   semantic commitment       -> persist R
```

**Invalidation is the load-bearing part, and it must be by key mismatch, never
by update.** A world change must NOT walk 64K entries:

```
CacheKey = DatasetVersion + RowDomain + lens/ClassView + program
         + focus/input identity + external-edge snapshot

new DatasetVersion -> old entries stay FROZEN (no work at all)
                   -> a request arrives -> MISS -> replay -> optionally re-cache
```

**And notice what that key IS.** It is the `ReplaySpec` from W1b, field for
field. The replay descriptor and the cache key are **the same artifact used two
ways**: as a recipe it regenerates the answer; as a key it memoizes it. That is
not a coincidence to be admired — it is the reason invalidation can be free,
because a descriptor made of owned identities either matches the world or does
not, and nothing has to be swept to find out.

**The economics are memoization economics**, not a prohibition:

```
C_cache         = C_lookup + p_miss · C_replay + amortized C_materialize
C_always_replay = C_replay
```

⊘ **The ~11 ns is a premise, not a measurement** — the same discipline applied
to the 1.7 ns figure. A cached-mask lookup cost must itself be measured before
any policy leans on it. But if lookup is tens of ns and replay is hundreds, a
heavily reused mask earns caching almost immediately; a thought called once
never did.

Which gives the metacognitive policy directly:

```
novel thought      -> replay
frequent thought    -> cache
historical truth    -> persist
stale thought       -> ignore; do NOT maintain
```

**And it makes reuse across kanban boards richer than either half alone.** A
heavily reused operator can carry both a `ReplaySpec` (the reasoning recipe) and
a hot cache entry (the precomputed result for one exact domain and version).
Same domain and version ⇒ the cached answer. Different context ⇒ replay the
operator against it. Memory and thought-reuse at once, with no maintenance
treadmill.

### Writing is the expensive part — so the boundary has TIERS, not a switch

The reason replay wins is not that reading is cheap; it is that **writing is
expensive**. That asymmetry, stated plainly, forbids a design the two-option
framing above still allows: a speculative *"this looks interesting, let me test
this hypothesis"* must not cost an SoA row. Under one undifferentiated publish
path it does, and the cost of curiosity becomes the cost of knowledge.

So the durable side is a ladder of at least three tiers, each strictly MORE
EXPENSIVE than the one above it:

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
| bounded mask composition | no windowed operand descriptor exists | lg@25988f3c | **M** | a descriptor with global `base_word` + local length + tail + row-domain (W2b); do NOT narrow generic `Scratch` first | touched words == the offset-aware span `floor((hi-1)/64) - floor(lo/64) + 1` (equivalently `<= ceil(width/64)+1` — a span straddling a word boundary touches one more word than an aligned one of the same width), tested at BOTH aligned and unaligned `lo`, and flat in N and in position |
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

⊘ **And the gate must range over the COMPLETE identity, not two of its
components.** An earlier phrasing tested only `RowDomain + program`, which is
the very insufficiency this section establishes: a replay can return a
plausible-but-wrong mask when an omitted input moves. Every component gets a
perturbation arm, each one disable-verified:

| perturb, holding all else fixed | replay must |
|---|---|
| row domain / snapshot | differ, or be refused |
| program identity | differ |
| **lens / ClassView identity** | differ |
| **focus / carrier input** | differ |
| **external-edge snapshot** | differ |
| **deterministic parameters** | differ |
| **Morton / Wabe mapping identity** | differ |
| nothing | be **bit-identical** |

A component whose perturbation leaves the result unchanged is either not an
input or not in the spec — and either way the finding is the point of the
matrix. This is the can-it-fire twin applied to a determinism gate: a gate that
passes under every perturbation carries no information.

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

### W2b — BOTH physical plans for the same semantics (Seam B, half two)

⊘ **Not "never write the bounded mask."** W2b must demonstrate **both legal
paths over identical semantics**, because that is the whole point — the defect
was the missing choice, not the mask:

```
W2b-A   Range × resident mask -> FUSED masking -> Count / Any
        the masking happens; no result mask is ever written

W2b-B   Range × resident mask -> masking -> a MATERIALIZED bounded mask
        and the arm must PROVE downstream reuse makes the carrier worthwhile
```

**Same masking semantics. Different result carrier.** Not "fold arm vs mask
arm" — both arms mask; they differ only in whether the membership relation is
promoted to a carrier.

- The two arms are differentially checked against each other and the oracle:
  identical row sets, identical `Count`/`Any`.
- **W2b-B carries a burden W2b-A does not:** it must name and measure the
  downstream reuse that justifies the carrier. A materialization with no
  demonstrated consumer fails the arm — that is the whole point of making the
  promotion deliberate.
- **This is where the anti-zoo rule licenses a new T1 primitive** — and the
  framing is stronger than "an optimization". A fused `popcount(a[i] & b[i])`
  over a word span cannot be expressed by the existing algebra without an
  intermediate buffer, so **without it the fold-native arm does not exist at
  all.** The primitive is what CREATES the choice; before it there is only the
  mask path, which is precisely why Seam B had no decision in it. The descriptor
  (`base_word + length`) crosses; the intersection never exists as bytes.
- ⊘ **Not** a narrowing of the existing `Scratch`. Changing
  `words == words_for(N)` to a smaller window is insufficient: operands need a
  global `base_word`, a local length, tail semantics, and a row-domain
  identity, because local word zero is not global word zero. That descriptor is
  the deliverable. **Do not modify generic `Scratch` until the descriptor has
  proved itself against a real consumer.**
- **Gate:** ⊘ NOT a constant — the touched-word count depends on the starting
  bit offset. Assert the offset-aware span
  `floor((hi-1)/64) - floor(lo/64) + 1` (equivalently `<= ceil(width/64)+1`: a
  span straddling a word boundary touches one more word than an aligned one of
  the same width), and test BOTH aligned and unaligned `lo`. Independent of `N`
  and of absolute position.

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
- **The question W6 answers, restated now that the law is in place.** Not *are
  writes expensive?* but: **where should the zero-copy program terminate and
  reconstruction become economically or semantically justified?** That is the
  Rubicon in computational terms, and it is the same two reasons as ever —
  economic (`C_retain < C_replay`) or semantic (it must become history).
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

### The residue IS the wave plan — and it is an EXPOSURE gap, not a compute gap

Split by tier, the inventory says something better than "we rediscovered five
operations":

**T1 — the photolithographic algebra (`ndarray::simd`), SHIPPED:** Boolean
composition, ternlog, the compare families, the `_under` gate, reductions,
popcount, **`mask_shift_morton`**, and the **strided** matchers
(`eq_u32_strided_to_mask`, `ternary_match_strided_to_mask`).

**T2 — the mask-risc RISC exposure, SHIPPED:** `Pred`, `And`/`Or`/`Xor`/
`AndNot`/`Not`, `Ternlog` (+ `fuse.rs`, `ternlog_dispatch.rs`), gates,
`Any`/`All`/`Count`, the masked reductions.

**BROKEN REPRESENTATION:** `Range` → an N-sized scratch mask.

**MISSING at T2 (the carriers and the exposures):**

| gap | wave |
|---|---|
| ADDRESS integrity — ordinal → canonical row, attested | **W1** |
| compact addressing / window DESCRIPTORS (`base_word + length`, `[lo,hi)`) | **W2a / W2b** |
| direct bounded REDUCTIONS — fused `popcount(a & b)` over a span, no written intersection | **W2b** |
| *optional* named reconstruction of runs / windows, when a consumer demands the buffer | after W6 |
| semantic → geometric ROTATE | **W3** |
| local NEIGHBOUR / STENCIL exposure (the T1 primitive exists) | **W4** |
| strided PEEK exposure (`ir.rs:21-27` names its own gap; the T1 primitive exists) | deferred |
| PROJECT — the lens lives in `contract::facet`, never in the plan language | after W3 |
| FIRST | unscheduled |

**That asymmetry is the finding.** The substrate is further along than the
language that exposes it: Morton shift and strided matching already exist one
tier down. So most of the remaining work is **not inventing computation** — it
is making existing computation speak the fold algebra **without forcing an
N-sized carrier between instructions.** Which is a much cheaper programme than
the op list first suggested.

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

## §9 Imports from other domains — held as PROBES, with one fence in front

Added 2026-09-19 after a lateral pass. None of these is doctrine; each is a
question with a falsifier, and none reorders W0–W7. The fence comes first
because the pass produced one attractive unification that must not be written
anywhere as a rule.

### The fence: one observable is not three instruments

A frontier XOR-popcount is a perfectly good **observable** — it can be *fed to*
a Tarski check, a Shannon estimate and a JC noise test. It **is not** those three
things, and no scalar is:

```
TARSKI    Δset = ∅  /  F(X) = X          a semantic fixed point of a monotone operator
SHANNON   H(P) = −Σ p log p              uncertainty; information gain is a DIFFERENCE of it
JC        is the observed signal distinguishable from the calibrated
          weak-dependence noise regime?  (I-NOISE-FLOOR-JIRAK, not a threshold on a count)
```

`popcount → 0` is neither a fixed point (a non-monotone step can raise it again)
nor zero entropy (a stable non-zero disagreement has *low* entropy and non-zero
popcount). Any sentence of the form "this one dial is all three" is the same
failure shape as `MaskOp`-means-bitmap: a carrier promoted to a semantics. The
board already carries the sibling fence — entropy says WHERE closure is, never
WHAT kind of hole (CE64 59–60) nor HOW it may be asserted (61–63).

### Roaring belongs UNDER `KEEP`, not instead of the mask expression

Density-adaptive containers (run list when sparse, bitmap when dense, chosen per
chunk) answer *"given that I want a represented set, which representation?"* The
arc's primitive answers the **prior** question — *"do I need a represented set at
all?"*

```
Fold(A) × Fold(B) × AND × TERNLOG(C) → COUNT     no run list, no bitmap, ever
```

Where a set IS elected (`Terminal::Keep`, a cached mask), the container is a
free choice and Roaring's rule is a good one. It is not a reason to make ops
eager. Recorded so the next session does not re-import it one level too high.

### The probes (each names what would falsify it)

| import | mechanism | where it sits in the two-stroke engine | probe / falsifier |
|---|---|---|---|
| **Retina** — read as contrast | the read itself is `current field × six-neighbour field → local residual`; the substrate reads *difference*, not state, so a sparse FIRE is a consequence rather than a discipline | inside the monotone stroke (a lens, not a write) | on the W5 fixture, residual-read vs field-read-then-diff: same delta set, count words touched; falsified if the residual read touches ≥ the field read |
| **Inhibition of return** | a decaying mask over recent foci that suppresses re-fixation | **outside** Tarski closure, by construction — it is intentionally non-monotone: `closure → choose focus → apply inhibition → new episode` | W5+: with vs without, count revisits of the same focus in N episodes; falsified if revisit rate is unchanged or coverage drops |
| **Erosion** (Go, Bouzy dilation-then-erosion) | pure dilation `A_{t+1} = A_t ∪ N(A_t)` paints everything reachable; a structural-support requirement lets the unsupported fringe evaporate | **not** "add erosion to Tarski" — erosion can break monotonicity; the shape is `monotone expansion → closure → structural pruning → new closure` | W4+: dilation-only vs dilation+support on the same tile; falsified if the supported set is not a strict subset with lower entropy |
| **Pre-shaped geometry** (lithography) | NOT one global order that makes every neighbourhood a prefix — that optimizes one lens by damaging another. Instead: a ClassView/ThoughtView **selects a codebook whose geometry was trained to make its own common predicates cheap** (a donor thought may bring its preferred lens with it) | mint time, per view | W3 sibling: same predicate family under a view-trained 4⁴ codebook vs the default; falsified if range/prefix mask cost does not drop for that view or rises for another |
| **Aperture** (camera obscura) | reasoning depth expressed as **address resolution** — coarse prefix = broad aperture, deep prefix = pinhole; "try at depth 3, if informative depth 5, then 8" | orthogonal to `ReasoningBand` 61–63: the band says what KIND of reasoning, the prefix says at what RESOLUTION — never collapse them | **evidence status:** the prefix machinery exists (`is_ancestor_of`, prefix routing); `facet.rs:645/683/796` `i >> 2` is `G3D4::group_of` — tier-of-tile-index, a carving shift. It is NOT a rung ladder. "The 0–9 rung is prefix depth" is CONJECTURE until a probe shows depth-stepping changes IG monotonically on a real lane |

The recurring rhythm across every row is the engine already ruled in W6.0 and
`E-THREE-CONVERGENCES-…-1`: a monotone stroke to a fixed point, then a
deliberately non-monotone stroke (inhibit, prune, revise, change resolution,
change dictionary, change band), then closure again. **"Thinking harder" need
not mean more state** — it can mean another dictionary, another resolution,
another band, another program, and a measurement of whether entropy actually
fell. That is the frame these probes are judged in.

## §10 The center this plan serves (added 2026-09-19)

Read `E-THE-CENTER-IS-SPOG-PLUS-FC-EVERYTHING-ELSE-IS-CAST-1` first. The
Waben loop, the four seams, the fold law and every wave above are EXECUTION
and PLUMBING for one center: **SPOG as the semantic address space, `f,c` as
the evidence currency, CE64 59–60 / 61–63 as the causal and reasoning context.**
An SPOG conjunction under G lowers to `Mask(P1,O1,G) AND Mask(P2,O2,G) →
Count / Any / NextFocus` — that is the whole reason Seam B and `D-WFL-MASKOP`
matter. Nothing in W0–W7 may redefine the center; a wave that needs a new
knowledge type is a wave that has drifted.

