# ARC A — Source Archaeology (dumb-storage architecture reset)

> Deliverable per `docs/architecture/DUMB-STORAGE-RESET-CHARTER.md` §20.
> Produced by 5 parallel Sonnet grindwork sweeps (lance-graph ×4,
> lance-graph-java ×1), synthesized here. **NO CODE was written to
> produce this document** — per the charter's explicit gate. Raw
> per-agent findings are banked in the session scratchpad
> (`ARCA-1` through `ARCA-5`).

---

## 1. Current source map (file:line anchors)

### HHTL — trie (door A)
- `crates/lance-graph-contract/src/hhtl.rs:56-59` — `NiblePath { path: u64,
  depth: u8 }`, `FAN_OUT=16`, `MAX_DEPTH=16`.
- `:72-129` root/child/try_child/is_full; `:134-239` parent/prefix/basin/leaf
  (pure bit-shift, no stored pointer); `:176-183,427-449`
  is_ancestor_of/is_descendant_of/is_sibling_of; `:251-268,404-417,459-503`
  common_prefix_depth/common_ancestor/family_hop_count (LCA + CLAM hop
  distance); `:302-402` from_guid_prefix/_v2/_v3 (GUID→trie bijection, v3
  routes full 8:8 part_of:is_a).

### HHTL — explicit hierarchy node (door B): **DOES NOT EXIST**
- No type anywhere carries `{parent_ref, child_or_ref_set,
  projection_mask, version_coordinate}` together (repo-wide grep for
  `HierarchyNode`/`RefNode`/`TreeNode`: zero hits).
- Closest partials, both insufficient alone:
  - `crates/lance-graph-ontology/src/hydrators/owl.rs:73-80`
    `ContextBundle { g, version, domain_name, inherits_from: Option<u32>,
    ontology, edge_types }` — has parent ref + version, **no child set, no
    projection mask**.
  - `crates/lance-graph-ontology/src/wikidata_hhtl.rs:47-60` `WikidataClass
    { dolce_id, subclass_path: &[u8], properties }` — has a
    `presence_mask()` via `FieldMask` and a derivable parent, but is a
    curated fixture with **no stored child set, no version**.
- `canonical_node.rs:645-650` `EdgeBlock` and `:706-735` `NodeRow` are
  payload containers addressed *by* a NiblePath-derivable key — explicitly
  NOT hierarchy nodes (the file's own doc says so for `EdgeBlock`).

### ClassView / FieldMask / WideFieldMask
- `crates/lance-graph-contract/src/class_view.rs:903` `ClassView` trait
  (fields/template/dolce_category_id/field_label/project/render_rows/
  facet_rows/is_a_parent/resolve_render_class/edge_codec_flavor).
  Impls: `WikidataClassView` (`wikidata_hhtl.rs:239`), `RegistryClassView`
  (`class_resolver.rs:104`).
- `class_view.rs:70` `FieldMask(u64)`, `MAX_FIELDS=64`.
- `class_view.rs:221` `WideFieldMask`, `WideRepr::{Small(u64),
  Wide(Box<[u64]>)}` (`:224-229`) — unbounded chunked, **field-presence
  only, no foveal/256-bound semantics attached anywhere**.
- `crates/lance-graph-contract/src/selection.rs:93-102` `NamedView
  { class, mask, template }`; `:123-125` `ViewRegistry`; `:205-221`
  `RailGraph` trait (`class_of`/`present_mask`/`rail_target`) —
  **deliberately separate from `ClassView`** (module doc `:25-59`
  explicitly analyzes and rejects routing rail-knowledge through
  `ClassView`); `:232-242` `FieldVisit<K>` the actual rail walker.
- `crates/lance-graph-contract/src/standing_mask.rs:1-120` — pub/sub
  interest-mask mechanism, real and separate, not a selection primitive.
- `crates/lance-graph-contract/src/facet.rs:1-105`,
  `canonical_node.rs:669-704` `EdgeCodecFlavor` selected via
  `class_view.rs:1099-1111` `ClassView::edge_codec_flavor`,
  `tekamolo_facet.rs:14-38` — the **polymorphic 8:8/256:256 per-ClassView
  reading is real and verified**, broader than edges alone; architecturally
  **separate** from `FieldMask`'s field-presence vocabulary (never unified).

### Zero-copy / BatchWriter / no-freeze
- `canonical_node.rs:1511-1568` `NodeRowPacket<'a>{ rows: &'a [NodeRow],
  cycle: u32 }`, not Clone/Copy (operator ruling 2026-07-29); `:1553-1570`
  `as_le_bytes()` zero-copy raw-pointer cast, no re-encode.
- `crates/lance-graph-planner/src/batch_writer.rs:95-182` `BatchWriter<P>`
  — cast/casts/intent_moves/on_behalf_of/resolve_owner/
  drain_pending_payloads. **No resolve/flush_and_wait/freeze/close method
  exists on the struct.** Zero grep hits for freeze/barrier/wait_for/
  close_the_world/authorize in this file. Doc: "melden macht frei — never
  refused" (operator 2026-07-17, `E-ACK-ELIMINATED-1`).
- `crates/lance-graph-planner/src/persist_sink.rs` `DetachedCycleBatch::
  freeze` (~`:377`) — **per-cycle** canonicalization (stable-sort own casts
  before one WAL append), not a cross-cycle lock. **OPEN FLAG:** a "dense/
  barrier mode" is named at `:1204,1269` pinning ΔV=1 — its opt-in-vs-
  default status was not fully confirmed by the archaeology pass (needs
  one follow-up read of `:1190-1270`, listed in §6 gap G1 below).
- `crates/lance-graph-supervisor/src/cycle_driver.rs:17-24` — **named
  ruling** `E-D-MBX-SPINE-IS-STRAIGHT-TRACK-VERSION-IS-NOT-A-FLEET-STEP-
  SIGNAL-1`: *"A DatasetVersion is global knowledge, NOT permission to
  advance every mailbox... the version tick never fans a step across the
  fleet."* — this is charter §7's exact distinction, already ruled and
  named.
- `cycle_driver.rs:1990-2026` — **explicit named falsifier test**, "B cast
  without waiting for A" / "A unfinished, never a barrier for B": Cycle 1
  leaves mailbox A unfinished; Cycle 2 still lets mailbox B advance.
- `cycle_driver.rs:26-60` — pre-commit failure = deterministic regen from
  `Vn`; post-commit interruption = idempotent replay via `recover_fleet`;
  *"the two mechanisms share no state."*

### temporal.rs / TemporalPov
- `crates/lance-graph-planner/src/temporal.rs:134-157` `QueryReference
  { server_id, ref_version, hlc_tick, mode: EpistemicMode, rung }`.
- `:184-197` `::at(v, rung)`; `:159-182` `Default` — `u64::MAX` sentinel at
  `:176`, **self-documented** (`:162-172`) as "NOT a knowledge horizon."
- `:77-97` `EpistemicMode::{Strict,Aware,Retro}` + `for_rung` (0-4/5-8/9+).
- `:117-126` `TemporalStatus::{Contemporary,Anachronistic,Spoiler,
  Unknowable}`; `:206-220` `classify()`; `:346` `knowable_from`.
- `:1,24-28` module doc, verbatim: *"Epistemology is a query-level
  annotation, not storage... none of these types belong in `ogar-vocab`."*
- **Zero real call sites** in `cycle_driver.rs`, `persist_sink.rs`,
  `batch_writer.rs` — the boundary is already true in shipped code, not
  aspirational.
- `crates/lance-graph-contract/src/temporal_pov.rs:1-196` — zero-dep mirror
  `TemporalPov`/`VersionRange`; `admits()` explicitly documented as ONLY
  range-membership. Real consumer: `crates/deepnsm-v2/src/wave.rs:86,130,
  134` (a read-window projection, not a scheduler).
- `u64::MAX`-as-default is isolated to exactly 2 spots (`temporal.rs:176`,
  `temporal_pov.rs:95`), both doc-flagged non-load-bearing.

### lance-graph-java (current main)
- `java/src/main/java/com/adaworldapi/lancegraph/RowStore.java:163,176`
  `hop(int edgeClassid, WideFieldMask, Mask)->Mask` and
  `hop(int, Mask)->Mask` — **already exist**, mask-in/mask-out.
- `consumers/graph/src/main/java/com/adaworldapi/graph/Graph.java:96-199`
  — full immutable fluent `.hop(...)`-chaining consumer: open/from/
  hop(int)/hop(int,WideFieldMask)/minus/count/materializeRows/close.
- `java/.../WideFieldMask.java:1-112` — mirrors the Rust type as a Java
  record. `ClassView` is **NOT** mirrored as a Java type (native-only via
  `class_view_provider`, by design).
- `native/lgj-abi/src/exports.rs` — 20 exported C symbols: lifecycle
  (`manifest`/`open`×3/`close`/`resource_info`), lane/mask describe, mask
  algebra (`and`/`or`/`andnot`), predicate ops incl. fused `plan_eval`,
  `reduce_sum_i32`, `row_facet_match`, `hop` (`:1223`). No temporal op.
- FFM containment verified clean: `Arena`/`MemorySegment` only private
  fields/locals; no public signature leaks.
- **Real gaps:** no `.children(mask)`/parent-child-named vocabulary
  (`.hop` is the functional equivalent already); `.at(version)`/
  `LanceVersion`/`TemporalPov` — **zero references anywhere** in `java/`
  or `consumers/`.

## 2. Already-shipped / gap matrix

| Capability | Status | Evidence |
|---|---|---|
| Stable exact reference | **SHIPPED** | `NodeGuid` (16B key) |
| HHTL trie routing | **SHIPPED** | `NiblePath`, full arithmetic |
| Explicit hierarchy reference node | **GAP (real)** | no type exists; nearest partials lack ≥2 of the 4 required fields |
| ClassView field-position interpretation | **SHIPPED** | `class_view.rs:903` |
| WideFieldMask ergonomics (field presence) | **SHIPPED** | `class_view.rs:221` |
| WideFieldMask AS A FOVEA (256-ish local ref surface) | **GAP (real)** | no foveal/spatial semantics attached to WideFieldMask anywhere; the "256" idiom that exists is a separate palette-coordinate mechanism |
| Rail-walk exposed THROUGH ClassView | **GAP (deliberate)** | `selection.rs` module doc explicitly routes it around ClassView via a separate `RailGraph` trait |
| 8:8/256:256 polymorphic per-class reading | **SHIPPED, verified** | `EdgeCodecFlavor`, `facet.rs`, `tekamolo_facet.rs` |
| DatasetVersion type | **SHIPPED** | `scheduler.rs:33-36` |
| DatasetVersion ≠ permission-to-advance doctrine | **SHIPPED, named, tested** | `cycle_driver.rs:17-24` ruling + `:1990-2026` falsifier |
| Zero-copy payload ownership | **SHIPPED** | `NodeRowPacket::as_le_bytes` |
| No freeze / no batch wall (cross-cycle) | **SHIPPED, tested** | no barrier method on `BatchWriter`; explicit falsifier test |
| No freeze (per-cycle canonicalization) | **SHIPPED, scoped correctly** | `DetachedCycleBatch::freeze` — one cycle, one WAL append, not a lock |
| "Dense/barrier mode" opt-in status | **UNCONFIRMED** | flagged, needs one follow-up read |
| temporal.rs as query tool, not scheduler | **SHIPPED, verified zero call sites** | already true in code, not aspirational |
| Java `.hop(...)` traversal | **SHIPPED, end-to-end** | `RowStore.hop` + `Graph.hop` |
| Java `.children(mask)` naming | **GAP (cosmetic)** | `.hop` is the functional equivalent |
| Java `.at(version)` temporal binding | **GAP (real, unbuilt)** | zero references anywhere in `java/`, `consumers/` |
| Java FFM containment | **SHIPPED, verified clean** | no public leak found |
| BatchWriter production wiring | **GAP (known, pre-existing)** | `cast()`/`deinterlace` have zero production call sites per the module's own doc — orthogonal to this reset, not caused by it |

## 3. Minimal storage PR sequence (ARC B, NOT started — awaiting your go)

1. **B1 — the explicit hierarchy reference node** (the one real, load-bearing
   gap in §2). Minimal type: `{ref: NodeGuid, parent: Option<NodeGuid>,
   children_mask_ref: ViewId /* or similar */, presence: WideFieldMask,
   version: DatasetVersion}` — composed from EXISTING pieces
   (`NodeGuid`, `WideFieldMask`, `DatasetVersion`, and `selection.rs`'s
   `ViewId`/`NamedView` machinery for the child-ref-set), not a new
   subsystem. Zero-copy: a borrowed view over existing SoA columns, not an
   owned struct with copied children (per charter §1's "reference-set
   connective tissue, not payload container").
2. **B2 — expose child/reference positions through ClassView** (charter
   §2's attention-economy pattern) — extends `ClassView` (or a sibling
   trait it composes) so a hierarchy node's children are selectable via
   WideFieldMask without a Cartesian pairwise scan. This directly resolves
   the `RailGraph`-routed-around-ClassView finding in §1 — needs a
   RATIFICATION QUESTION (§10 below) since `selection.rs`'s module doc
   argued deliberately AGAINST this exact routing.
3. **B3 — the F-HIERARCHY-NOT-AUTHORITY falsifier** — a test proving
   changing B1's hierarchy geometry (e.g. re-keying which node is "parent")
   does not change the exact reference-query result set. Written against
   B1+B2 once they exist.
4. **B4 — F-TRIE-VS-NODE falsifier** — a test proving `NiblePath` routing
   and B1's explicit nodes can diverge in shape (different fanout) without
   breaking either.

No implementation starts before you ratify this sequence — this is the
PROPOSED order, not a commit log.

## 4. Java integration PR sequence (ARC C, NOT started)

1. **C1 — `.at(version)` temporal binding.** The one real, unbuilt gap.
   Mirror `TemporalPov`/`QueryReference::at` as a Java record + one native
   ABI symbol (`lgj_temporal_at` or similar — naming deferred to
   implementation), following the exact pattern `WideFieldMask.java`
   already establishes for mirroring a contract type.
2. **C2 — `.children(mask)` as a named overload of the existing
   `.hop(...)`** (not a new mechanism — `Graph.hop(int edgeClassid,
   WideFieldMask)` already does this; C2 is purely a hierarchy-flavored
   convenience name/overload, IF you want the vocabulary distinct from
   generic edge-hops. Optional — flagged as a RATIFICATION QUESTION.
3. **C3 — ontology genericity proof**: an ontology read-cache consumer
   using B1 hierarchy nodes + C1 temporal + existing `.hop`.
4. **C4 — OSM genericity proof**: a geographic-locality consumer using the
   SAME B1/C1/hop primitives, zero new storage vocabulary. C3 and C4
   together are charter §10's required falsifier
   (`F-DOMAIN-GENERICITY`).

## 5. Types/APIs to reuse UNCHANGED

`NodeGuid`, `NodeRow`, `EdgeBlock`, `NodeRowPacket`, `NiblePath` (full API),
`ClassView` trait (as-is — extension in §6 is additive, not a rewrite),
`FieldMask`, `WideFieldMask` (as a field-presence mechanism — its foveal
USE is new, the type itself does not change shape), `DatasetVersion`,
`QueryReference`/`EpistemicMode`/`TemporalStatus`/`classify()`,
`TemporalPov`/`VersionRange`, `BatchWriter<P>` (as-is — the charter's §7
"stays AS-IS unless a source-proven defect requires a change" applies
literally; none was found), `RailGraph`/`FieldVisit`/`NamedView`/
`ViewRegistry`, `RowStore.hop`, `Graph` (Java consumer class), all 20
`native/lgj-abi` exports, `EdgeCodecFlavor`/`ClassView::edge_codec_flavor`.

## 6. Types/APIs that need EXTENSION (not replacement)

- **`ClassView`** — needs the child/reference-position exposure from B2
  (additive method, not a signature change to existing methods).
- **`selection.rs`'s `RailGraph`** — B2 may fold into it rather than into
  `ClassView` directly, pending §10's ratification question.
- **`WideFieldMask`'s USE** (not its type) — needs to be exercised as a
  local bounded selector over B1's child-ref surface; no field/method
  changes to `WideFieldMask` itself are evidenced as necessary.
- **Java `RowStore`/`Graph`** — needs the C1 temporal binding as new
  methods; existing `.hop` methods stay untouched.
- **`native/lgj-abi/src/exports.rs`** — needs one new symbol for C1.
- **`FrameMeta`/reconcile machinery in `persist_sink.rs`** — orthogonal to
  this reset (it belonged to the STOPPED seal arc); no charter item touches
  it. Left as-is, historical.

## 7. Proposed NEW types, each with proof an existing type cannot express it

- **The explicit hierarchy reference node (B1).** Proof: §1 confirms no
  existing type carries `{parent_ref, child_or_ref_set, projection_mask,
  version_coordinate}` together — `ContextBundle` has 2 of 4,
  `WikidataClass` has a different 2 of 4, `EdgeBlock`/`NodeRow` are payload
  containers by explicit design. This is the ONE new type the charter's own
  gap analysis licenses — everything else in this document is composition
  of existing pieces.

No other new type is proposed. (The charter's §12 rung axis, §13-17
meta-awareness, and §11 episodic/epistemic layer are explicitly deferred
to ARC D/E — not proposed here.)

## 8. How #968 / the seal work changes under no-freeze + hierarchy refs + versioned frontier

- **#968 itself is unaffected as a merged historical record** — merge
  commit `66fec27`, head `88210f7`. Nothing in this reset un-merges it or
  edits its content.
- **What changes: task #25 (the seal implementation) does not launch.**
  Its 5+3-ratified design (`.claude/plans/cascade-seal-register-grid-v1.md`
  v3) assumed a freeze-then-seal-then-publish shape
  (`DetachedCycleBatch::freeze` → `content_hash`/root → one WAL append)
  that is architecturally consistent with the *existing, per-cycle-scoped*
  `freeze` this archaeology confirms is NOT a cross-cycle barrier — so the
  seal design was never in conflict with charter §7's no-freeze rule at
  the CROSS-cycle level. What it WOULD conflict with, if implemented as
  specified, is charter §0's broader reframing: the seal's `ContentRoot`/
  `ControlRoot` identity is a PER-CYCLE artifact, and this reset's
  "versioned frontier, not batch barrier" model treats `DatasetVersion` as
  a durable coordinate through an EVOLVING reference structure — the
  seal's batch-identity concept doesn't obviously compose with a hierarchy
  node whose children can be added *between* two DatasetVersions without
  a bounding cycle. This is a genuine open question, not resolved here
  (see §10 ratification question 3).
- **The research and falsifiers survive intact as prior art**: the X-C2-1
  injection harness (`crates/rp-seal-t0-probe/`), the register-grid
  correction, the fail-closed cross-version identity finding
  (`E-CROSS-VERSION-IDENTITY-MIGRATES-BLIND-SO-IT-FAILS-CLOSED-1`), and the
  council's own hardening method (the 5+3 sequencing itself) are reusable
  IF a future arc revisits sealed-batch identity for the new hierarchy
  substrate — none of that work is wasted, it's scoped-out, not wrong.
- **Board state**: PR_ARC + LATEST_STATE + EPIPHANIES already record this
  supersession (commit `8d8fba8`) — no further board action needed for
  #968 itself.

## 9. Architecture diagram

```
                    PRESENT — the substrate this arc builds
   ┌─────────────────────────────────────────────────────────────┐
   │  ontology inputs        OSM / geo inputs      mechanical Java │
   │       │                       │                     │        │
   │       └───────────┬───────────┴─────────────────────┘        │
   │                   ▼                                          │
   │     refs (NodeGuid)  +  ClassView  +  WideFieldMask           │
   │                   │           (field presence, extended       │
   │                   │            with the B2 fovea use)         │
   │                   ▼                                          │
   │   HHTL trie (NiblePath, door A)   +   explicit hierarchy      │
   │   — routing, ancestry, LCA            reference node (door B, │
   │                                        NEW — the one gap: B1) │
   │                   │                                          │
   │                   ▼                                          │
   │        versioned dumb substrate                              │
   │   (NodeRow/EdgeBlock/NodeRowPacket zero-copy, BatchWriter     │
   │    no-freeze, DatasetVersion = durable frontier not           │
   │    permission-to-think, temporal.rs pure query annotation)    │
   │                   │                                          │
   │                   ▼                                          │
   │       Java mechanical facade (RowStore.hop / Graph.hop,       │
   │       + NEW: .at(version) via C1, WideFieldMask mirror)       │
   └─────────────────────────────────────────────────────────────┘

                    FUTURE — deferred to ARC D/E, seams only
   ┌─────────────────────────────────────────────────────────────┐
   │            episodic witness  (grounded historical evidence)   │
   │                       │                                      │
   │                       ▼                                      │
   │       epistemic causality / known-unknown                    │
   │       (unresolved expected sibling = a hole WITH an address,  │
   │        never stored as "unknown" in the substrate above)      │
   │                       │                                      │
   │                       ▼                                      │
   │                     rung  (single shared axis — WHAT KIND/    │
   │                     DEPTH of epistemic operation)             │
   │                       │                                      │
   │                       ▼                                      │
   │           orchestration meta-awareness                       │
   │           (top-down HHTL observation, starts abstract,        │
   │            descends only where interesting)                   │
   │                       │                                      │
   │                       ▼                                      │
   │       top-down versioned nudges (never storage backpressure,  │
   │       never permission-to-think — epistemic PARTICIPATION,    │
   │       not execution PERMISSION, per charter §15)              │
   └─────────────────────────────────────────────────────────────┘
```

## 10. Ratification questions (genuine forks only)

1. **B2's home**: does the child/reference-position exposure land ON
   `ClassView` directly (charter §2's literal phrasing: "expose... THROUGH
   its ClassView"), or on the existing, deliberately-separate `RailGraph`
   trait (which `selection.rs`'s own module doc argues for, with reasons
   already on record)? Source leaves a genuine, already-argued fork here —
   I have a lean (extend `RailGraph`, since its separation from `ClassView`
   was a considered decision with a written rationale, not an oversight)
   but this is your call to make, not mine to override.
2. **Java naming (C2)**: keep `.children(mask)` as charter-literal
   vocabulary, or treat `.hop(edgeClassid, mask)` as already sufficient and
   skip C2 entirely? The charter itself says "DO NOT freeze those method
   names before auditing current APIs" — the audit is done; `.hop` already
   exists and works. Your call on whether hierarchy-specific naming still
   earns its keep.
3. **Seal/hierarchy composability** (§8): does the STOPPED seal design's
   per-cycle `ContentRoot`/`ControlRoot` identity concept get revisited
   for the hierarchy-node substrate in a LATER arc, or is sealed-batch
   identity itself out of scope for as long as this reset's architecture
   stands? No source or prior ruling resolves this — it's a forward
   design choice, not an archaeology finding.
4. **"Dense/barrier mode" flag** (`persist_sink.rs:1204,1269`): should I
   spend one more grindwork pass confirming it's genuinely opt-in before
   ARC B starts, given charter §7's absoluteness? (Low cost, closes an
   open flag — recommend yes, but noting it as a question since it's
   discretionary sequencing, not a fork in direction.)
