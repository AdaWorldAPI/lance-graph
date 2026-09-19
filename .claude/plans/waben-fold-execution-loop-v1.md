# Waben fold execution loop — v1 (PROPOSAL, 2026-09-19)

> **Status:** PROPOSAL — no code written. Grounded against lance-graph
> `origin/main` **25988f3c** and ndarray **40a71ad** (both read 2026-09-19).
> Source docs: `waben_fold_architecture.md` + `waben_implementation_prompt.md`
> (operator-supplied, 2026-09-19).
>
> **Prefix for this arc's deliverables: `D-WFL-*`** (unused in `STATUS_BOARD.md`).

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

**Where the code leaves the representation — three seams, in dependency order.**

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

`attended_mask()` (`alpha.rs:783`) allocates `AlphaMask::empty(base.len())` and
scatters set-bits by iterating the HashMap. So the publication boundary
round-trips **ordinal → NodeGuid → hash → ordinal → full mask** exactly where
the architecture wants ordinals to stay ordinals. `AlphaMask` itself is the
right type (bitset with explicit phantom-tail discipline, `alpha.rs:224`); the
claim path is what is row-shaped.

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

**Consequence for the plan: the deliverable is an ASSEMBLY, not a construction.**
Most PRs below connect two existing symbols and delete a materialization between
them. Only two build something genuinely new (D-WFL-3's tile contract and
D-WFL-6's strided facet lane).

---

## §2 The address — resolved onto what exists, with no new storage

The prompt's `(G, NodeGuid, Thought-or-Rung)` resolves as:

| coordinate | existing home | evidence | new storage? |
|---|---|---|---|
| **G** (ontology / ClassView) | `classid: u32` at facet bytes `0..4`, canon-high (`concept << 16 \| app`); resolved through `lance-graph-ontology`'s `class_resolver` | `facet.rs:94`, canon-high flip on the board | none |
| **NodeGuid** | `canonical_node::NodeGuid`, 16 B, stable | `canonical_node.rs:862` | none |
| **Rung** | **`TemporalPov { range, rung: u8 }`** — already a *reader's* coordinate, not per-node state | `contract/src/temporal_pov.rs:151` | **none** |
| **Thought track (≤64)** | the **6-bit W-slot palette**, `AttentionMaskEntry { mailbox_id: MailboxId, w_slot: u8 /* 0..64 */ }` | `cognitive-shader-driver/src/attention_mask.rs:29` | **none** |

So neither a packed 6+4 ABI nor an `N × 64 × 10` slab is needed: **rung is a
read POV, track is a W-slot claim.** Both are already typed. The per-node
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

**The six-neighbour Wabe tenant is the second facet's rail plane, read under a
ClassView.** Six directions = six rails = six relations. No new column, no new
stride, no `ENVELOPE_LAYOUT_VERSION` bump. This is the single largest piece of
free ground in the whole proposal.

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
| **bound ⇄ executed planes binding** | `Planes` has no version/lens/digest (`ir.rs:50`) | lg@25988f3c | **M** | `Planes.domain: RowDomain` + check in `validate` (D-WFL-1) | wrong-version / wrong-lens / duplicate-key falsifiers, each red before the fix |
| range survives execution | `Pred::Range` → `mask_set_range(dst=full)` (`exec.rs:540`) | lg@25988f3c · nd@40a71ad | **M** | `Terminal::{RangeAny, RangeCount}` + `BoundedMask` scratch (D-WFL-2) | touched-word counter; cost flat in N and in absolute position |
| touched-window write | `d_diamond_1_probe::touched_write` — **probe-only** | lg@25988f3c | **D** | promote to a `BoundedMask` view in mask-risc (D-WFL-2) | 20.5–23.1 ns flat vs 34→4,620 ns whole-lane |
| six-neighbour shift | `ndarray::simd::mask_shift_morton` · hex probe | nd@40a71ad | **I** (as a *whole-field* op) | tile contract, D-WFL-3 | axial BFS oracle + degree-one control (both already in the probe) |
| u8 gate predicates | `gt_u8_to_mask` etc. · hex probe | nd@40a71ad | **I** | — | in-probe |
| gated predicate | `*_to_mask_under` — skips compare on a zero gate word, still visits the gate span | nd@40a71ad | **I** | pass bounded gates, never a global one | gate-word visit counter |
| **strided facet lane** | `LaneRef::{I32,U32,U64}` — no strided variant; `ir.rs:21-27` names the gap itself | lg@25988f3c | **M** | `LaneRef::Strided{base,stride,group}`, mirroring `ndarray::simd::ternary_match_strided_to_mask` (D-WFL-6) | differential vs a contiguous copy of the same lane |
| tree-depth column ("G8") | nothing in ndarray `src/` | nd@40a71ad | **M** | out of scope for this arc; name it, don't assume it | — |
| alpha claim algebra | `AlphaOverlay::claim` · `wave_dispatch::dispatch_thought` (**orphan**) | lg@25988f3c | **D** | ordinal-keyed claim path (D-WFL-4) | claimed bytes; no `Vec<NodeRow>` growth per claim |
| rung × tenant focus | `AlphaFocus` · **no caller** | lg@25988f3c | **D** | make it the slice's focus carrier (D-WFL-5) | `unlooked()` non-empty and non-total |
| rung wave scheduler | `rung_schedule::schedule_for` · `wave_dispatch` (orphan) | lg@25988f3c | **D** | one live caller (D-WFL-5) | two contexts, deterministic wave order |
| owner stamping | `SoaEnvelope::mailbox_owner()` — default 0, only its own tests | lg@25988f3c | **D** | stamp on the slice's write (D-WFL-4) | owner ≠ 0 on every published row |
| batch → DatasetVersion | `BatchWriter::cast` → `LanceCycleWriter::commit_cycle` · tests + one example | lg@25988f3c | **D** | assemble once in the slice (D-WFL-7) | exactly one new version per cycle; append-only |
| track (≤64) | `w_slot: u8` (6-bit palette) | lg@25988f3c | **I** as a claim; **M** as a dispatch | two active tracks only (D-WFL-5) | both tracks visible at one NodeGuid |
| rung (0–9) | `TemporalPov.rung` | lg@25988f3c | **I** | — | — |

---

## §4 Dependency-ordered PRs

Each names: files · invariant · the real consumer · the gate · **the result that
falsifies the step's claim**.

### D-WFL-1 — bind the bound to the planes it executes on (closes Seam A)

- **Files:** `contract/src/ordered_lane.rs` (new `RowDomain`), `mask-risc/src/ir.rs`
  (`Planes.domain`), `mask-risc/src/exec.rs` (`validate`), `quack/src/lib.rs`
  (carry it on `Bound`).
- **Shape.** No new registry, no second identity system: `RowDomain { version:
  LanceVersion, lens: SemanticLens, digest: u64, n_rows: u32 }` — the four
  fields `OrderedLaneWitness` already holds, lifted so `Planes` can hold one
  too. `OrderedLaneWitness::domain()` produces it; `validate()` rejects a
  `Pred::Range` whose program-carried domain ≠ `planes.domain`.
- **Duplicate keys.** The key digest cannot attest payload order. Either the
  digest must cover the associated-row identity (digest over `(key, ordinal)`
  pairs), or `SealedFacetLane` must refuse to attest a lane with duplicate keys.
  **Recommend the second** — it is one `if` and it is honest; widening the
  digest silently changes what every existing witness means.
- **Consumer:** `quack`'s lowering; the executor.
- **Gate:** three falsifiers, each disable-verified RED first — wrong version,
  wrong lens, duplicate key. Plus a permuted-planes case: same rows, same
  length, same key digest, different order ⇒ must be rejected.
- **Falsifies the step:** a permuted-planes program that still returns the
  oracle's answer. That would mean the binding is decorative.

### D-WFL-2 — let the range stay a range (closes Seam B)

- **Files:** `mask-risc/src/ir.rs` (`Terminal::{RangeAny, RangeCount}`,
  `Scratch` gains a bounded variant), `mask-risc/src/exec.rs`, `quack/src/lib.rs`.
- **Shape.** Two independent halves, and the first is nearly free:
  - **(a) terminals that never build a mask.** `Any` over a bound is `hi > lo`;
    `Count` is `hi - lo`. Both are arithmetic on the endpoints and both are
    correct *only* when every row in the range satisfies the query — i.e. when
    the program is exactly one `Pred::Range` with no `under`. Gate it on that.
  - **(b) `BoundedMask` scratch.** Lift `touched_write`'s `(base_word, words)`
    shape out of the probe: a scratch slot declares a word window, and
    `mask_set_range` is called on the window, not the population. `Scratch`'s
    current `words == words_for(n_rows)` check becomes `>=` on the window.
- **Consumer:** the slice's prefix→terminal path (D-WFL-5).
- **Gate:** a touched-word counter asserted against `(hi - lo)/64 + 2`, with
  equal-width windows at the near and far end of the population — **position
  and width varied independently**, because a fixed-position sweep cannot see an
  O(end-position) cost. This is exactly the falsifier that caught the probe's
  own first fix.
- **Falsifies the step:** touched words growing with `n_rows` at fixed width, or
  with absolute position at fixed width. Either means the carrier did not survive.
- **Hard rule this PR enforces.** Between a successful bound and Wabe entry, no
  allocation or write sized by the population may occur unless the requested
  terminal contract genuinely demands a dense full-domain mask. The rule has one
  mechanical violation site — `Scratch`'s full-population definition — which is
  exactly what (b) removes, so it is enforceable by a counter rather than by
  review discipline.

### D-WFL-3 — one exactly-specified Wabe tile

- **Files:** new module in the probe crate first, **not** in a shipped crate.
- **Decision to state up front — closed tile, not moving window.**
  `mask_shift_morton` requires `4^k` words, OR-accumulates, treats *the slice as
  the field*, and drops edge carries. A trie-aligned closed tile is therefore
  exactly supported today; a moving aperture over a larger interacting field is
  **not** the restriction of a global shift and needs a halo. Start closed. The
  locality condition `P_F S_d P_{H(F)} A = P_F S_d A` is what a later halo step
  must prove; for a closed tile `H(F) = F` and it holds trivially.
- **Identity mapping:** `ordinal = Morton(q, r)`, declared per tile, asserted
  against the tile's own base ordinal — never assumed globally.
- **Gate:** the hex probe's axial BFS oracle, its degree-one control, and a
  sparse delta-frontier arm (all three already exist in
  `ndarray/examples/hex_tenant_mq_probe.rs`; reuse, don't rebuild).
- **Falsifies the step:** any advantage that survives the degree-one control is
  not hex; any mismatch at a tile edge means the closed-tile claim is false.

### D-WFL-4 — an ordinal-keyed claim path (closes Seam C)

- **Files:** `contract/src/alpha.rs`.
- **Shape:** add `AlphaOverlay::claim_ordinals(&mut self, mask: &AlphaMask, rung:
  u8)` beside the existing `claim`. It writes stamps without hashing a
  `NodeGuid` and without pushing a `NodeRow`. The existing `claim` stays,
  unchanged, for callers that genuinely start from an address.
- **Consumer:** the slice's publication step.
- **Gate:** published bytes and claimed-row growth counted per claim; a no-change
  step publishes zero; an inhibitory change publishes non-zero.
- **Falsifies the step:** claimed bytes still scaling at 512 B/claim.

### D-WFL-5 — the assembly (the first slice; see §5)

### D-WFL-6 — strided facet lane in the IR

Deferred until the slice needs it. `ir.rs:21-27` already names the gap and
already names `ndarray::simd::ternary_match_strided_to_mask` as the shape. Adding
`LaneRef::Strided` before a consumer exists would be building a facade word with
no backend behind it.

### D-WFL-7 — one real publication through the existing owner

`BatchWriter::cast` → `collect_casts` → `seal` → `LanceCycleWriter::commit_cycle`
→ one `DatasetVersion`. Every link exists; assemble it once, outside `tests/`.
**No** new transport type, **no** per-cell or per-thought actor message, **no**
resurrection of `KanbanActor`. `FIRE` names the effect, not an object.

---

## §5 The first slice — concrete fixture

**Population.** 65,536 rows — the hex probe's size, so its oracle and its
degree-one control transfer unchanged. One `NodeRow` per cell;
`ordinal = Morton(q, r)` over a 256×256 axial field.

**Two lenses.** (i) `SemanticLens::CanonHighTiles8` over the **key** facet —
the shipped lens, sealed, witnessed. (ii) A second lens over the **second**
facet's rail plane. It gets its own `SemanticLens` variant, and the slice must
demonstrate that the lane is **not** attestable under both at once — one physical
sequence is monotone under one lens at a time. D-DIAMOND-1's P3 already showed
exactly this (tenant lane unattestable over the ontology ordinal, inversion at
row 1); the slice reproduces it as a *designed* property rather than a finding.

**Two thought contexts.** Two W-slots, two rungs via `TemporalPov`. Not 64, not 10.

**The chain.**

```
witnessed prefix on lens (i)      ->  Bound{lo,hi} + RowDomain        [D-WFL-1]
Bound  ->  BoundedMask window (no full-population mask)               [D-WFL-2]
window ∩ tile  ->  A_0
A_{t+1} = (A_t ∪ ⋃_d S_d(A_t ∩ P_d)) ∩ T   on a closed 4^k tile       [D-WFL-3]
   P_d = permeability byte of rail d, second facet, lens (ii)
delta = A_{t+1} \ A_t
delta empty      ->  Terminal::RangeAny == false, nothing published
delta non-empty  ->  claim_ordinals(delta, rung)                      [D-WFL-4]
                 ->  owner-stamped cast -> one DatasetVersion         [D-WFL-7]
publication at the same NodeGuid, other context, changes next eligibility
   ->  AlphaFocus(rung x tenant) is the next focus                    [D-WFL-5]
```

**What must be true throughout:** no allocation sized by `n_rows` occurs between
the bound and the tile entry. Assert it with a counter, not a comment.

**Irregular ingress.** One explicit non-local edge, entering focus sideways (the
architecture's `G → F` arrow), proving a remote contribution can reopen a region
the local recurrence had closed. One edge is enough to prove the seam; BLASGraph
integration is not in this slice.

---

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
`TruthU8`, and the conductor over the full track set.
