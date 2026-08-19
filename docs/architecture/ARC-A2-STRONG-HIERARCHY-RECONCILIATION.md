# ARC A′ — Reconciliation under the STRONG hierarchy model

> Deliverable for the operator's correction of 2026-08-19 ("the hierarchy node
> is NOT necessarily a computed view"), §15 items 1-5. Produced by five
> parallel read-only source sweeps (HHTL shared-trie capacity · is_a/part_of
> rails · AriGraph basin promotion seam · higher-order awareness
> materialization · WideFieldMask basis semantics). **NO CODE.** Supersedes
> `ARC-A-SOURCE-ARCHAEOLOGY.md` §3/§7/§9/§10 where they conflict; its factual
> inventory stands.

## THE HEADLINE

**The strong model is not hypothetical, and it is not absent. It exists as
five disconnected islands.** Every major mechanism the operator described is
already implemented somewhere in this workspace — correctly, with tests — and
none of them are connected to each other. The gap is not invention. It is that
no *shared shape* exists, so each island grew its own vocabulary.

| The operator's mechanism | Already implemented at | Connected? |
|---|---|---|
| Two independent rails, same leaf, no identity duplication | `rail_geometry::{RailAxis::Taxonomy, ::Mereology, RailPath}` (shipped 2026-08-13), with its OWN disable-test `the_pair_axes_are_two_separate_bytes` | **No** — sole caller is a `ClassView::rail_carving` default; `rail_geometry.rs` never mentions `NiblePath`, and `hhtl.rs` never mentions `RailPath` |
| Shared ancestry chain walked once, not copied per leaf (the DN-tree) | `deepnsm-v2::ancestry::FamilyTrie` — real parent-pointer forest, `dn()` literally the distinguished-name walk | **No** — private `u16` id space, zero tie to `NodeGuid`/`NiblePath`, no crate depends on it |
| `is_a` free as address-prefix containment; `part_of` explicit edges | `soa_bake/mod.rs` — the design is written down **unprompted**: *"a trie whose interior nodes ARE the reused n-grams… every ancestry node is shared across all descendants"* | **No** — self-marked "⚠ TYPE SCAFFOLDING, not a working bake" |
| Second-order: a record referencing another record of its own kind | `witness_fabric::resolve_chain` — real multi-hop chase, horizon + budget escalation, **no MetaAgent** | **Partially** — bounded ±8 window, one locus dimension, facet flagged experimental |
| Discovered grouping → stable addressable coordinate | `wikidata_hhtl` DOLCE basin assignment — shipped, tested, survives subclassing | **N/A** — Type-A only (assigned from pre-existing parents); no Type-B analogue exists |

## 1. CONCLUSIONS THAT STAND

**No freeze / no batch wall.** Unchanged and independently re-verified. No
barrier method on `BatchWriter`; the named ruling
`E-D-MBX-SPINE-IS-STRAIGHT-TRACK-VERSION-IS-NOT-A-FLEET-STEP-SIGNAL-1`
(`cycle_driver.rs:17-24`); the explicit falsifier test at `:1990-2026`
("A unfinished, never a barrier for B"). **"Dense/barrier mode" is not a
runtime mode** — it names a test's write pattern, and the surrounding prose
blesses ΔV > 1 for a hot wavefront as *"a measurement, not an error."*
`DatasetVersion` remains durable history, never a compute barrier.
**BatchWriter is not to be redesigned** absent a source-proven defect; none was
found.

**`temporal.rs` is a tool, not an orchestrator.** Zero real call sites in
`cycle_driver.rs`, `persist_sink.rs`, or `batch_writer.rs` — the boundary is
already true in shipped code, not aspirational.

**The stopped seal plan is a live documentation trap.**
`.claude/plans/cascade-seal-register-grid-v1.md` is marked RATIFIED v3 and says
implementation is gated on the #968 ready-flip — which has now *happened*. A
future session opening that file sees a ratified spec whose only stated
precondition is satisfied. Must be defused at the document a session actually
opens. Hygiene, not redesign; delete nothing.

**The lgj-abi contract-import fence is real.** `native/lgj-abi` imports exactly
`lance_graph_contract::{class_view, canonical_node, ontology}` — verified in
source. It constrains WHERE ABI-facing mechanics may live. It does **not**
decide what the hierarchy representation is (see §2).

**The mask basis-collision risk is real** — but not where I said. See §2.

**Zero production-code blast radius to date.** All work this arc is docs/board.

**New, and it stands: there is no reasoner in this repo, correctly.** The OWL
"hydrators" are name-interning passes that discard the triples they parse
(`hydrators/owl.rs:120-260`, `edge_types: Vec::new()`); `owl:subClassOf` et al.
exist only as human-curated predicate-string whitelists. `ontology_warrant.rs:6`
names the external factfinder (OGAR's `ogar-elk`) explicitly and is
*structurally prevented* from producing an entailment — *"deliberately no method
that turns a `NarsTruth` back into an entailment."* Grep for `ogar-elk`
workspace-wide: two hits, both prose; zero Cargo dep, zero call, zero FFI. This
matches the operator's §3 exactly (HHTL exposes structure; RO/ELK decides
transfer) — the boundary is correctly drawn and simply not yet wired.

**And no live conflation of structure with semantics exists.** Every
`part_of:is_a` consumer outside the ontology crate treats it as an address/basin
coordinate for routing, never as "parent fact ⇒ child fact." `ontology_warrant.rs`
is evidence of the *opposite* discipline, built after a measured incident where
treating ontology **silence** as **dissent** inverted a finding (~50% apparent
disagreement vs. a true 99.8% agreement).

## 2. RETRACTED OR DOWNGRADED

**RETRACTED — "the hierarchy node is a computed view over existing bits."**
Not the general definition. A computed projection is *one* case. The general
case includes explicit, shared, addressable structural topology — repeated
ancestry encoded ONCE as reference topology, leaves locating through the shared
chain rather than carrying private copies.

*Why I got it wrong, precisely:* the earlier correction said the four fields
must never materialize a per-node crosswalk, Valhalla/Panama-cheap. I
generalized "don't duplicate ancestry per leaf" into "don't store hierarchy at
all." Those are opposites — a shared trie encoded once is *de*-duplication; it
is the mechanism that stops per-leaf duplication. I collapsed the
**representation** question into the **evaluation** question and answered only
the second.

*And the audit shows the cost of that error concretely:* today every leaf
re-derives its full 16-nibble ancestry from its own key on every query
(`clam_contained`/`members`/`memberof`/`nearest_anchor`, all O(n) per-row
scans); `WikidataClass.subclass_path` stores each class's own literal array with
`human` = `&[0x1,0x2]` hand-copying `person`'s `&[0x1]` prefix; interior nodes
are explicitly *"virtual — no ownership, no SoA restructure."* The weak model
would have ratified exactly that duplication.

**RETRACTED — "zero new types, one free function in class_view.rs."** Derived
from the retracted premise, and answering the wrong axis besides. Per the
operator's follow-up, the axis is not *how few types* but **is the shape
reusable and agnostic, or hand-rolled per case?** One agnostic delegation shape
beats both zero-types-plus-ad-hoc-composition and N bespoke types. The
workspace already demonstrates the failure mode: **five distinct "basin"
vocabularies** sharing a word and no mechanism (AriGraph discovered cluster ·
`Locus::BasinAnchor` pointer slot · `EpisodicEdges64` class-family ·
`NodeGuid` `family` tier · Wikidata/DOLCE category), four Type-A and one Type-B.

**RETRACTED — "a newtype re-materializes what your correction removed."**
Factually wrong. A `#[repr(transparent)]` newtype is a compile-time distinction
that materializes nothing at runtime. Same category error as above, one level
down. Note the workspace has *already solved this exact problem once and wrote
down why*: Java's `FacetId.java` — *"Not a lane id… Mixing the two up is exactly
the bug the end-to-end test pins… A distinct type is what stops that confusion
at the call site instead of relying on a comment."* This is precedent, not a
recommendation; the operator has ruled the solution not yet chosen.

**RETRACTED — "meta-awareness is a future consumer."** It is present-but-
unmaterialized, which is a different and more actionable condition.
`nars/meta_basin.rs` performs real higher-order structural analysis today —
basin clustering over causal trajectories, outlier suggestion with evidence —
and its own doc states *"Nothing here prunes, commits, or scores."* Computed,
then discarded, every cycle. `witness_fabric::resolve_chain` already performs
second-order referencing. The capability is running; the materialization is
missing.

**DOWNGRADED — the mask-collision instance.** My hypothesis (class-field
position vs child/reference position) is **not** a shipped second basis: the
rail walk keeps `WideFieldMask` in class-field basis throughout; a rail-bearing
bit is still a field position whose `rail_target` is non-`None`. But the risk is
real in two places I did not predict: **live** — `causal_witness.rs`'s `Locus`
register (24 slots) read through the same naked `WideFieldMask`, where
`CausalWitnessFacet::project(mask)` and `ClassView::project(class, mask)` would
each accept the other's mask silently; **latent** — the lgj-abi's fixed
32-facet-lane basis, whose own `fixture.rs:5-6` admits the canonical per-class
basis is "a later slice." The single active enforcement anywhere is
`selection.rs:349-353`'s `view.class != class` check, scoped to the rail walk.

**DOWNGRADED — the lgj-abi fence as a design determinant.** The fence is a
fact; using it to conclude "therefore the helper lives in `class_view.rs`"
presumed a helper was the answer. Fence constrains placement, not
representation.

**CORRECTED — known-unknown is not the definition of epistemic causality.**
ARC A framed it as the centerpiece. It is one case; the broader job is
explicit-vs-implicit causal knowledge, what support exists, what is missing,
what may be inferred, and the horizon at which each was known.

## 3. SOURCE MAP — THREE DISTINCT CASES

### Case A — existing ontology hierarchy (parent already exists)

| Element | Where | State |
|---|---|---|
| Two independent rails | `rail_geometry::{RailAxis, RailCarving, RailPath}`; `RailPath::is_ancestor_of` :178-182; disable-test :265-286 | **REAL, shipped 2026-08-13.** Sole caller = `ClassView::rail_carving` default (`class_view.rs:1127-1132`) |
| …but fused at the routing layer | `NiblePath::from_guid_prefix_v3` (`hhtl.rs:386-410`) packs BOTH bytes into one 64-bit path; all ancestor arithmetic runs on the fused value; no per-axis constructor exists | **CONFLICT** — one leaf gets one route, not two loci. The two modules never cross-reference |
| Byte-level carrier | `FacetCascade` (`facet.rs:99-134`), L1 reading hi=`part_of` lo=`is_a`; separate 8:8 tile in `NodeGuid::TailVariant::V3` | Real; two physically distinct slots |
| Ancestry sharing | `WikidataClass.subclass_path` — each class its own literal array, `human` hand-copies `person`'s prefix | **DUPLICATED**, no interning |
| Transitive closure | none materialized anywhere; `is_ancestor_of` is cheap because the *address* encodes ancestry as prefix | By design |
| Reasoner | none in-repo; `ontology_warrant.rs` is the consumer-side grading contract for OGAR `ogar-elk`, unwired | Correctly external |
| Type-A grouping → addressable coordinate | `wikidata_hhtl` DOLCE basin, survives subclassing (:359-363) | **REAL, shipped** |
| The intended end-state | `soa_bake/mod.rs` — `is_a` = prefix containment (zero storage), `part_of` = explicit `EdgePair` rows | **SCAFFOLDING ONLY** |

### Case B — promoted episodic basin (no parent exists)

| Element | Where | State |
|---|---|---|
| Basin discovery | `EpisodicMemory::basins()` (`arigraph/episodic.rs:243-313`), union-find over held episodes | Real function, **ZERO callers workspace-wide** |
| Basin value | `EpisodicBasins { entities: Vec<String>, labels: Vec<u32>, num_basins }` | Returned by value; never written to any lane/column/row/dataset |
| Identity | dense `u32` re-densified per call; entities keyed by raw `String` | **None stable** |
| The pointer slot meant to reach it | `Locus::BasinAnchor` (`causal_witness.rs:134`) | Exists; **confirmed unwritten** — *"needs an AriGraph part_of:is_a rail; none is wired"* |
| Promotion seam | — | **DOES NOT EXIST.** Not built, not stubbed. The adjacent fact-level pipeline `witness_tombstone.rs` is 100% `todo!()` **and absent from `graph/mod.rs`** — orphaned, unreachable |
| The queued design | `soa_view.rs:257-275` — *"EpisodicWitness64… AriGraph promoted to the hot path as a per-row SoA column… NOT YET a code symbol"* | Comment only |
| Exact attach point if built | return of `episodic.rs:243-313`; needs a reserved `ValueTenant` variant (none exists), a `NodeGuid::mint_for` per basin, a write to `Locus::BasinAnchor` | Named, unbuilt |
| Participation (masks / HHTL / versioning / causal refs) | all **No**; grep of `DatasetVersion\|NodeGuid\|ClassView\|WideFieldMask` across `arigraph/*.rs` = zero hits | `E-ARIGRAPH-IS-AN-ISLAND`, reconfirmed |
| Reusable Type-B pattern to copy | — | **None exists anywhere** |

### Case C — higher-order / meta-awareness SoA

| Element | Where | State |
|---|---|---|
| `AwarenessRef` type | — | **Does not exist** (exhaustive grep) |
| Addressing scheme 1 | `Locus` = signed i4 offset ∈ [-8,+7] from the row's own stream position | Bounded; cannot address an arbitrary row/mailbox/classid |
| Addressing scheme 2 | `BasinOf`/`NiblePath` (`mailbox_scan.rs:319`) — unbounded, cross-shard capable | Targets ordinary nodes, not awareness records |
| Reconciliation between them | — | **None.** Structurally disjoint; nothing feeds one from the other |
| Second-order referencing | `witness_fabric::resolve_chain` (:421-495) — row→row through the same locus, multi-hop, horizon+budget escalation, **no MetaAgent** | **REAL**, but ±8 window, single locus dimension, facet flagged experimental |
| Higher-order analysis | `nars/meta_basin.rs` — clustering, outlier suggestion with evidence | **REAL, and discarded** — *"Nothing here prunes, commits, or scores"* |
| Materialized awareness state | `SpoFacet` (A1); `CausalWitnessFacet` (A9, byte offset 176..188, const-asserted) | Real; A9 experimental, not in the locked catalogue |
| `MetaWord` | 6+4+8+8+6 = 32 bits, **zero spare**; ~40 hand-assembling call sites; **zero `ClassView` references** | Hard-coded global meaning — the opposite of the class-projected 12-byte facets |
| `awareness.revise(key, outcome)` | `counterfactual.rs:379` | **`todo!()` stub**, header: *"BLOCKED… Do NOT infer from CLAUDE.md pseudo-code alone"* |
| Nudge / feedback record | `sensorium.rs` is a live call/trait loop; `HeldIntent` explicitly *"not a durable ledger"* | **No versioned, replayable record exists** |
| The four BindSpace columns | all four real types, but `QualiaColumn` is `#[deprecated]` and **`BindSpace` itself is scheduled for deletion** (`W7 deletes BindSpace`); live carrier is `MailboxSoA<N>` | CLAUDE.md's framing is stale |

## 4. DIAGRAM

```
 CASE A — ONTOLOGY (parent exists)                CASE B — BOOK (no parent exists)
 ════════════════════════════════                 ════════════════════════════════
                                                   concept─event─statement─concept…
  is_a TRIE (shared)   part_of TRIE (shared)                   │
  ┌──────────────┐     ┌──────────────┐            local episodic witness (±8)
  │   Animal     │     │  Organism    │            ✔ resolve_chain EXISTS
  │      │       │     │      │       │                        │
  │  Vertebrate  │     │    Body      │                        ▼
  │      │       │     │      │       │              recurring coherent basin
  │   Mammal ◄───┼──┐  │   Organ ◄────┼──┐          ✔ EpisodicBasins EXISTS
  │    /    \    │  │  │    /   \     │  │          ✘ zero callers, no identity
  │  Fox    Wolf │  │  │ Heart  Lung  │  │                     │
  └──────┬───────┘  │  └──────┬───────┘  │            ✘✘ PROMOTE ── THE GAP
         │          │         │          │              (no seam, no stub,
         └────┬─────┴─────────┘          │               no ValueTenant slot)
              │                          │                      │
       ONE stable leaf ref               │                      ▼
       two independent loci              │            stable higher-order ref
   ✔ rail_geometry EXISTS (disable-      │            ✘ MISSING entirely
     tested: "Taxonomy darf Mereology    │                      │
     nicht sehen")                       │                      │ becomes an
   ✘ NiblePath::from_guid_prefix_v3      │                      ▼ ordinary
     FUSES both into ONE route —         │         ┌────────────────────────┐
     the two modules never meet          └────────►│ ordinary participant:  │
   ✘ ancestry DUPLICATED per leaf                  │ masks · hierarchy ·    │
     (no interning; FamilyTrie does it              │ versioning · refs      │
      right, in a disconnected u16 space)          └───────────┬────────────┘
                                                               │
        CASE C — HIGHER-ORDER AWARENESS                        │
        ═══════════════════════════════                        │
          awareness ──targets──► "Mammal" locality ref  ◄──────┘   (case A)
          awareness ──targets──► promoted basin ref               (case B)
          awareness ──targets──► ANOTHER awareness ref            (second order)
          ✔ resolve_chain does 2nd-order for real, no homunculus
          ✘ but ±8 window, ONE locus dimension, experimental
          ✘ no AwarenessRef; TWO disjoint addressing schemes, unreconciled
          ✘ meta_basin computes real higher-order structure → DISCARDS it
```

## 5. THE MINIMAL EXPERIMENT

> **⊘ RETRACTED (operator canonical-substrate correction, 2026-08-19).** The
> "one agnostic delegation shape … generic over the reference space" this
> section proposes is exactly the **generic "structural algebra"** the operator
> forbade: *"do not invent another representation. The representation is
> already canonical and shared: 32×(4+12), 6×2×8-bit, 256:256 … Do not
> introduce: HierarchyPlane types / separate higher-order structs / promotion
> DTOs / a generic 'structural algebra' / another SoA."* Higher-order,
> ontology, episodic and meta-awareness are READINGS inside the one canonical
> substrate, never a new parametric shape beside it. The falsifier table below
> survives as a *question bank* (each row re-reads as "can the canonical
> substrate + LE addressing contract express this?"), and F-ORDER-GENERICITY
> is retired with the shape it tested. Successor: the ARC-B plan
> `.claude/plans/hhtl-thinking-tables-le-contract-v1.md` (readings + a
> little-endian addressing contract, zero new types). Original text retained
> below per append-only discipline — read it as the record of a corrected
> proposal, not as a pending experiment.

**Zero production change. An excluded probe crate, as with `rp-seal-t0-probe`.**
Its purpose is NOT to show mask AND/XOR is cheap (that would re-prove the weak
model). It is to test whether **one agnostic delegation shape** can carry
structurally unrelated cases over different reference spaces — order-genericity,
the sibling of domain-genericity.

**The one experiment worth running first**, because it is the cheapest thing
that can falsify the whole program: take `FamilyTrie` (which already implements
shared-ancestry correctly over `u16`) and `rail_geometry::RailPath` (which
already implements two independent rails correctly over facet bytes), and ask
whether a single shape — generic over the reference space, blind to what the
hierarchy means — can express **both**, plus a promoted-basin parent and an
awareness-targeting-awareness, without any case needing its own vocabulary.

Falsifiers, per the operator's §13 plus the order-genericity sibling:

| id | claim under attack |
|---|---|
| F-SHARED-RAIL | two leaves with common ancestry reference a shared path/locality without reproducing the chain |
| F-MULTI-RAIL | one stable ref occupies independent `is_a` and `part_of` structures without identity duplication |
| F-TRIE-VS-SEMANTICS | HHTL exposes locality while RO/ontology semantics stay outside the structural layer |
| F-LOCAL-MASK | a bounded child/locality surface projects without global Cartesian enumeration |
| F-PROMOTION | a discovered basin can *structurally* become a promoted stable ref, not only a sidecar |
| F-HIGHER-ORDER | a higher-order ref can itself be the subject of another reference/awareness projection |
| F-NO-PAYLOAD-COPY | shared parents/higher-order nodes require no child-payload copying |
| F-MASK-BASIS | a mask minted for basis A cannot silently cross into basis B |
| F-HIERARCHY-NOT-AUTHORITY | structural locality does not itself decide ontology/causal semantics |
| **F-ORDER-GENERICITY** *(new)* | base hierarchy, promoted higher-order, and second-order awareness are the SAME shape instantiated — not three hand-rolled vocabularies |

**Pre-registered kill condition:** if expressing Case A and Case B through one
shape requires either case to carry a field the other cannot use, the shape is
not agnostic and the experiment has failed — report that, do not widen the
shape to rescue it.

## OPEN QUESTIONS (source leaves a genuine fork; none decided here)

1. **`rail_geometry` vs `NiblePath` reconciliation.** Two un-reconciled readings
   of overlapping bytes: one gives two independent loci, the other fuses them
   into a single route with no per-axis constructor. Which is canonical, or do
   both stand at different layers?
2. **Where the shared-ancestry mechanism lives.** `FamilyTrie` works but is
   `u16`-keyed and disconnected; `soa_bake` describes the intended end-state but
   is scaffolding. Port, generalize, or leave both and build the shape beside
   them?
3. **Mask basis discrimination.** Audited, not solved, per instruction. The live
   collision is `Locus` vs class-field, not the one I hypothesized.
4. **`ValueTenant` slot for a promoted basin.** None reserved. Reserving one is
   a layout decision with `ENVELOPE_LAYOUT_VERSION` implications.
