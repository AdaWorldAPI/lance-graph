# Backend::MailboxSoa — Cypher over the canonical 512B GUID-node + turboquant-edge substrate

> **Status: CONJECTURE — design spec for review. NOT YET BUILT. No code in this doc.**
> **Author:** session 2026-06-18 (Opus). **Scope:** `AdaWorldAPI/lance-graph` only.
> **Type:** purely-additive read path. No `ENVELOPE_LAYOUT_VERSION` bump (proven in §5).

---

## 0. One-paragraph summary

Add a fourth `Backend` variant — `Backend::MailboxSoa` — to
`crates/lance-graph/src/graph/graph_router.rs` so a parsed Cypher
`NodePattern` / `RelationshipPattern` can scan over a borrowed
`&dyn MailboxSoaView` (the canonical SoA), filter node rows by `classid`
prefix, walk the 16-byte `EdgeBlock` (12 in-family + 4 out-of-family
slots), and decode each edge through the row's `classid → ClassView →
EdgeCodecFlavor` (`Pq32x4` / `CoarseOnly` / `CoarseResidue`). Today the
three existing backends (DataFusion, Blasgraph, Palette) run over Arrow /
SPO / blasgraph adjacency; **none of them reads the canonical
`NodeGuid` / `EdgeBlock` / `NodeRow` substrate** — those types are
compile-locked in `lance-graph-contract` and never appear in the core
query engine (verified: `grep -r 'NodeRow|canonical_node|NodeGuid'
crates/lance-graph/src/` returns empty). This is the read path that
closes that gap.

---

## 1. Verified source facts (file:line, confirmed this session)

### 1.1 The canonical substrate lives ONLY in lance-graph-contract

`crates/lance-graph-contract/src/canonical_node.rs`:

- `NodeGuid([u8; 16])` — `#[repr(C, align(16))]`, line 33-35. Accessors:
  `classid()` L88-90, `family()` L93-95, `identity()` L98-100,
  `heel/hip/twig()` L104-118, `local_key()` L153-157, `decode()` L127-136.
- Zero-fallback ladder guards: `is_default_class()` L162-164
  (`classid == CLASSID_DEFAULT == 0x0000_0000`, L39), `is_unbasined()`
  L167-169 (`family == 0`), `is_bootstrap_address()` L172-174.
- `EdgeBlock { in_family: [u8; 12], out_family: [u8; 4] }` —
  `#[repr(C, align(16))]`, L247-254.
- `EdgeCodecFlavor` enum, L273-287: `CoarseOnly = 0` (default, each edge
  byte is a palette index — the EdgeBlock read literally), `CoarseResidue
  = 1` (coarse + signed-4-bit residue in the value slab), `Pq32x4 = 2`
  (the 16 bytes read as 32 × 4-bit PQ codes — turbovec/turboquant).
  `is_layout_preserving(self) -> bool { true }` for ALL flavors, L305-307.
- `NodeRow { key: NodeGuid /*0..16*/, edges: EdgeBlock /*16..32*/,
  value: [u8; 480] /*32..512*/ }` — `#[repr(C, align(64))]`, L315-321.
- Compile-locked sizes: `const _: () = assert!(size_of::<NodeGuid>()==16)`
  / `EdgeBlock==16` / `NodeRow==512`, L324-326. `NODE_ROW_STRIDE = 512`
  L376.
- `ReadMode { value_schema, edge_codec }` L639-645 + `classid_read_mode(classid)
  -> ReadMode` L691-696 (registry resolver, falls through to
  `ReadMode::DEFAULT` for any unconfigured classid — the same zero-fallback
  ladder as the key). `NodeGuid::read_mode()` carrier method L145-147.
- `NodeRowPacket<'a> { rows: &'a [NodeRow], cycle: u32 }` L711-714 — a
  zero-copy `SoaEnvelope` over `&[NodeRow]` (`as_le_bytes` is
  `from_raw_parts(rows.as_ptr().cast::<u8>(), len*512)`, L753-771).

### 1.2 ClassView is the class→layout resolver

`crates/lance-graph-contract/src/class_view.rs`:

- `pub type ClassId = u16;` L53 — the per-row discriminator (reuses the
  width of `MailboxSoaView::class_id`).
- `trait ClassView` L276-391. Edge/value resolution lives here:
  - `fn edge_codec_flavor(&self, class: ClassId) -> EdgeCodecFlavor` L342-344,
    default `CoarseOnly`.
  - `fn value_schema(&self, class: ClassId) -> ValueSchema` L366-372,
    **TEMPORARY POC default `Full`** (so consumers transcode against a
    fully-populated NodeRow; revert to `Bootstrap` before merge).
  - `fn compute_dag(&self, class) -> &[ComputeEdge]` L388-390 (PR #539
    recompute manifest; default `&[]`).
  - `fn fields(&self, class) -> &[FieldRef]` L280 + `field_label()` L292-294
    — **the late-bound property/label resolver. This is the keystone for
    `RETURN n.name` (see §8 open question).**
- `FieldMask(u64)` presence bits, L69; `from_positions` L83-93, `has(n)`
  L110-112. The instance's delta from its class.

### 1.3 The MailboxSoaView / MailboxSoaOwner read surface

`crates/lance-graph-contract/src/soa_view.rs`:

- `trait MailboxSoaView` L28-105. Relevant accessors:
  - `fn n_rows(&self) -> usize` L31 — "Number of POPULATED rows" (NOT
    capacity; see `MailboxSoA::n_rows` returning `self.populated`, mailbox_soa.rs L711-719).
  - `fn edges_raw(&self) -> &[u64]` L46 — per-row packed `CausalEdge64` as
    raw `u64` (kept raw so the contract stays zero-dep).
  - `fn entity_type(&self) -> &[u16]` L50, aliased by
    `fn class_id(&self) -> &[u16]` L61-63 and `class_id_at(row) -> u16`
    L67-69. **THIS is the classid the scan filters on.**
  - `fn meta_raw() -> &[u32]` L48, `fn energy() -> &[f32]` L43,
    `current_cycle()` L36, `phase()` L38, `w_slot()` L34.
- `trait MailboxSoaOwner: MailboxSoaView` L112-139 — adds `advance_phase` /
  `try_advance_phase` (the mutation airgap). The scan needs only the
  **read** trait.

### 1.4 The owner impl — column-array SoA, not `&[NodeRow]`

`crates/cognitive-shader-driver/src/mailbox_soa.rs`:

- `pub struct MailboxSoA<const N: usize>` L58 — **parallel column arrays**
  (`edges: [CausalEdge64; N]`, `entity_type: [u16; N]`, `content`,
  `qualia`, `meta`, `temporal`, `expert`, `sigma`, …), NOT an array of
  512B `NodeRow`.
- `pub fn edge(&self, row) -> CausalEdge64` L550-552 (single-row read);
  `entity_type_at(row) -> u16` L589-591; `content_row(row) -> &[u64]`
  L645-647 (zero-copy fingerprint slice).
- `impl MailboxSoaView for MailboxSoA<N>` L705-767: `edges_raw()` casts
  `[CausalEdge64; N]` → `&[u64]` via `#[repr(transparent)]` (L737-752);
  `entity_type()` returns `&self.entity_type` L764-766.

> **CRITICAL NUANCE (drives §8 design fork):** there are TWO concrete
> shapes that satisfy `MailboxSoaView`:
> - **`MailboxSoA<N>`** (cognitive-shader-driver) — column arrays;
>   `edges_raw()` returns **decoded `CausalEdge64`** as `u64`, NOT the raw
>   16-byte `EdgeBlock`. There is no `EdgeBlock` / `NodeRow` here.
> - **`NodeRowPacket<'a>`** (canonical_node.rs) — `&[NodeRow]`; each row
>   carries the raw 16B `EdgeBlock` + 480B value slab, but `NodeRowPacket`
>   does **not** currently implement `MailboxSoaView` (it implements
>   `SoaEnvelope`).
> The `EdgeBlock` (12+4 slot, palette-index) walk the operator describes is
> a property of `NodeRow`, while `MailboxSoaView::edges_raw()` exposes
> `CausalEdge64`. **These are two different edge representations.** See §8.

### 1.5 The Cypher parser + AST already exist

`crates/lance-graph/src/`:

- `parser.rs:23` — `pub fn parse_cypher_query(input: &str) -> Result<CypherQuery>`.
- `ast.rs`: `NodePattern { variable: Option<String>, labels: Vec<String>,
  properties: HashMap<String, PropertyValue> }` L137-144;
  `RelationshipPattern { variable, types: Vec<String>, direction:
  RelationshipDirection, properties, length }` L166-177;
  `RelationshipDirection { Outgoing, Incoming, Undirected }` L181-188;
  `PathPattern { start_node, segments: Vec<PathSegment> }` L148-153;
  `PropertyRef { variable, property }` L220-225 (the `n.name` in `RETURN n.name`).

### 1.6 The existing router + its three backends

`crates/lance-graph/src/graph/graph_router.rs`:

- `enum Backend { DataFusion, Blasgraph, Palette }` L46-53.
- `struct GraphRouter { typed_graph: TypedGraph, spo_store: SpoStore,
  truth_values, node_names: Vec<String>, node_labels: HashMap<String,
  Vec<usize>> }` L76-87.
- Dispatch: `classify_query(query_text) -> QueryClass` L223-242 (keys on
  `MATCH` / `HAMMING` / `FINGERPRINT`), `query_routed()` L215-220 selects a
  backend. Today only Blasgraph is wired; the comment at L214 says "palette
  routing will be added in Phase 4." `GraphHit { source, target, distance,
  truth, backend }` L31-42 is the per-hit result.
- Registered in the mod tree: `crates/lance-graph/src/graph/mod.rs:14`
  `pub mod graph_router;` (alongside `arigraph` L10, `scheduler` L19).

### 1.7 Prior consumers of MailboxSoaView (reuse their iteration pattern)

- `crates/lance-graph/src/graph/scheduler.rs:116`
  `pub async fn drive_once<V: MailboxSoaView>(...)` and `:137`
  `drive_at_latest<V: MailboxSoaView>` — already generic over the view.
- `crates/lance-graph/src/graph/arigraph/markov_soa.rs:170`
  `SoaWavePrimer::project<V, F>(&self, soa: &V, focal_row, row_triple)
  where V: MailboxSoaView` — the canonical **row-window iteration**:
  reads `soa.n_rows()` (L177), `soa.class_id()` (L179), loops
  `for d in -r..=r { let row = focal+d; if out_of_range continue; … }`
  (L180-193). **`Backend::MailboxSoa`'s scan reuses this exact loop shape.**

---

## 2. Goal

```
Cypher text
  → parse_cypher_query()                      (exists, parser.rs:23)
  → CypherQuery { MATCH (n:Label)-[r:REL]->(m:Label2) RETURN n.name }
  → Backend::MailboxSoa { view: &dyn MailboxSoaView }
       │
       ├─ MATCH (n:Label):  scan rows 0..view.n_rows()
       │     filter by  classid_of(row)  ∈  classids_for_label("Label")
       │     (zero-fallback: classid==0 ⇒ default class ⇒ matches the
       │      label-less / bootstrap pattern)
       │
       ├─ (a)-[r:REL]->(b): for each matched source row, walk its 16B
       │     EdgeBlock — 12 in-family slots then 4 out-of-family slots —
       │     decode each non-zero slot via
       │       classid_of(row) → ClassView::edge_codec_flavor → {Pq32x4,
       │       CoarseOnly, CoarseResidue}
       │     resolve the slot byte to a target row, gate by :REL type
       │
       └─ RETURN n.name: resolve property "name" via
             classid_of(row) → ClassView::fields/field_label  (§8 OPEN)
       ↓
   Vec<GraphHit>  (extended; see §6)
```

The scan is **read-only** (`MailboxSoaView`, never `MailboxSoaOwner`) and
honors the data-flow rule (no `&mut self` during computation): it reads
borrowed columns, computes on owned `Copy` microcopies (`NodeGuid`,
`EdgeBlock`, `ClassId`), returns results.

---

## 3. The new `Backend::MailboxSoa` variant

### 3.1 Enum slot

```text
enum Backend {
    DataFusion,   // cold path — SQL over RecordBatch  (existing)
    Blasgraph,    // hot path  — BitVec semiring        (existing)
    Palette,      // hot path  — bgz17 compressed        (existing)
    MailboxSoa,   // hot path  — canonical 512B NodeRow / MailboxSoaView  (NEW)
}
```

`Backend` is `#[derive(Debug, Clone, Copy, PartialEq, Eq)]` (L45) — adding
a unit variant is trivially compatible. `GraphHit.backend` already carries
it (L41); a `MailboxSoa` hit stamps `backend: Backend::MailboxSoa`.

### 3.2 State it holds

The scan does NOT own the SoA. The owner is `cognitive-shader-driver`'s
`MailboxSoA<N>` (ractor-driven hot path) or the canonical `NodeRowPacket`.
Two carrier options, ranked:

- **PREFERRED — borrow the view (matches the existing generic pattern):**
  a scan function generic over `V: MailboxSoaView`, exactly like
  `scheduler::drive_once<V>` and `SoaWavePrimer::project<V>`. No new field
  on `GraphRouter`; the view is passed at call time:
  ```text
  GraphRouter::scan_mailbox_soa<V: MailboxSoaView>(
      &self, view: &V, pattern: &NodePattern, classes: &dyn ClassView,
  ) -> Vec<GraphHit>
  ```
  This keeps `GraphRouter`'s struct unchanged and avoids a lifetime
  parameter on the router itself. The `&dyn ClassView` is the label/edge
  resolver (impl'd in `lance-graph-ontology`).

- **ALTERNATIVE — store `&dyn MailboxSoaView` on a dedicated scan struct:**
  a separate `MailboxSoaScanner<'a> { view: &'a dyn MailboxSoaView, classes:
  &'a dyn ClassView }` if a long-lived bound view is needed. Heavier
  (lifetime on the struct); only adopt if a caller must hold the view
  across many queries.

> **Recommendation:** start with the generic-function form (option 1) — it
> is the smallest additive change, reuses the proven `<V: MailboxSoaView>`
> bound, and keeps `Backend::MailboxSoa` a pure routing tag rather than a
> stateful field. The router stays a classifier; the view arrives per call.

### 3.3 How the router selects it

`classify_query` (L223) is extended with a routing predicate. Route to
`Backend::MailboxSoa` when **all** of:

1. The query is a structural `MATCH` over **node labels that resolve to a
   `ClassId`** (i.e. the ontology/`ClassView` knows the label) — a
   GUID-node workload, not a free-form Arrow/SQL workload.
2. A `MailboxSoaView` is supplied to the call (the hot SoA is live).
3. The pattern is reachable by EdgeBlock adjacency (in-/out-family slot
   walk), i.e. it is not a vector-similarity (`HAMMING`/`FINGERPRINT`)
   query — those still route to Blasgraph, and not a multi-table SQL join —
   those still route to DataFusion.

Decision table (additive — existing rows unchanged):

| Query shape | Has live MailboxSoaView? | Label∈ClassView? | Route |
|---|---|---|---|
| `MATCH (n:Label)…` structural | yes | yes | **MailboxSoa** (NEW) |
| `MATCH … HAMMING/FINGERPRINT` | — | — | Blasgraph (existing) |
| SQL / multi-table join | — | — | DataFusion (existing) |
| compressed palette traversal | — | — | Palette (existing) |
| `MATCH (n:Label)…` structural | no (no SoA) | — | Blasgraph fallback |

The fallback row matters: if no view is supplied, the canonical path is
simply unavailable and the router degrades to the existing Blasgraph
behavior — no regression for current callers.

---

## 4. Scan semantics

### 4.1 `MATCH (n:Label)` — classid-prefix node filter

The `NodePattern.labels: Vec<String>` (ast.rs L141) is mapped to a set of
`ClassId`s by the ontology (`lance-graph-ontology`'s `ClassView` /
`OntologyRegistry`: `label → entity_type_id`). The scan then:

```text
let want: SmallSet<ClassId> = classes.classids_for_labels(&pattern.labels);
let class_ids: &[u16] = view.class_id();        // soa_view.rs L61
for row in 0..view.n_rows() {                   // populated rows only, L31
    let cid = class_ids[row];
    if pattern.labels.is_empty() || want.contains(&cid) {
        emit(row);
    }
}
```

**Zero-fallback ladder integration** (canonical_node.rs L11-14, L162-174):

- A `NodePattern` with **no labels** (`(n)`) matches **every** row —
  including `classid == 0x0000_0000` (the default class). This is the
  natural reading of "no prefix routing": label-less = match-all.
- `classid == 0` rows are the **bootstrap address** (`is_bootstrap_address`,
  only `identity` discriminates). They match a label-less pattern and match
  a `:Label` pattern only if the ontology maps that label to classid 0
  (it generally does not — classid 0 is the unconfigured default).
- A `:Label` whose ontology classid is non-zero filters by **prefix
  equality on the 4-byte classid** (`NodeGuid::classid()` folds bytes
  0..4 LE). This is the "prefix-routable" property (canon L26). A true
  HHTL radix walk over HEEL/HIP/TWIG (bytes 4..10) is a **later
  optimization** — v1 does a linear classid-equality scan, which is correct
  and matches the `SoaWavePrimer` linear-window pattern.

> Note: the `MailboxSoaView` surface exposes the classid via the
> `entity_type`/`class_id` `u16` column (L61), **not** the full 16-byte
> `NodeGuid`. So v1's "classid filter" is really a `u16 ClassId` filter —
> the `ClassId` (u16) IS the per-row discriminator the SoA carries (canon
> says the 4-byte `NodeGuid.classid` is the prefix-routable key; the
> hot-SoA column is the u16 `ClassId` projection of it). For
> `NodeRowPacket`/`&[NodeRow]` the full `NodeGuid::classid() -> u32` is
> available and the filter can be the full 32-bit classid. **This dual
> width is a real seam — see §8 / Open Question 2.**

### 4.2 `(a)-[r:REL]->(b)` — EdgeBlock walk + codec decode

For a `&[NodeRow]` source row (the canonical shape), the 16-byte
`EdgeBlock` is walked slot-by-slot:

```text
let row: &NodeRow = &rows[src];
let flavor = classes.edge_codec_flavor(row.key.classid() as ClassId);
// 12 in-family slots (basin-local adjacency)
for (slot_idx, &b) in row.edges.in_family.iter().enumerate() {
    if b == 0 { continue; }                 // 0 = unused (reserved, zeroed)
    let target = decode_edge_slot(b, flavor, row, slot_idx, /*in_family*/ true);
    if rel_type_matches(&pattern.types, target) { emit_edge(src, target, flavor); }
}
// 4 out-of-family slots (inherited adapter interfaces)
for (slot_idx, &b) in row.edges.out_family.iter().enumerate() {
    if b == 0 { continue; }
    let target = decode_edge_slot(b, flavor, row, slot_idx, /*in_family*/ false);
    if rel_type_matches(&pattern.types, target) { emit_edge(src, target, flavor); }
}
```

Direction (`RelationshipPattern.direction`, ast.rs L172):
- `Outgoing` (`->`): walk src's EdgeBlock as above (src → slot-target).
- `Incoming` (`<-`): the EdgeBlock is the *outgoing* adjacency, so an
  incoming walk requires either a reverse index or a full scan testing
  every row's EdgeBlock for a slot pointing at the focal node. v1: full
  scan (correct, O(n·16)); a reverse-adjacency index is a later optimization.
- `Undirected` (`-`): union of both.

**Edge-codec decode by flavor** (`EdgeCodecFlavor`, canonical_node.rs
L273-287; encode/reconstruct kernels live in `ndarray::hpc::edge_codec`):

- `CoarseOnly` (default): each edge byte IS a palette/centroid index read
  literally. `decode_edge_slot` interprets the byte as a coarse index; the
  target node is resolved by the palette/adjacency convention (the slot
  byte → neighbor id). This is the zero-fallback reading and the only one
  needed for a bootstrap (`classid == 0`) graph.
- `CoarseResidue`: coarse index + a signed-4-bit per-dimension residue
  carried in the value slab (`ValueTenant`/`TurbovecResidue` region). The
  decode reads the coarse byte AND the residue bytes from `row.value`.
- `Pq32x4` (turbovec / turboquant): the **16 EdgeBlock bytes** are read as
  **32 × 4-bit PQ codes**, not as 16 separate neighbor indices. So for a
  `Pq32x4` row the "12+4 slot" framing is the *byte* layout, but the
  *semantic* decode treats all 16 bytes as one PQ-coded vector. The
  neighbor resolution for `Pq32x4` therefore differs from `CoarseOnly`
  (PQ reconstruct → nearest, vs literal index). **This asymmetry must be
  explicit in `decode_edge_slot`: the flavor decides whether a slot is an
  independent neighbor index (`CoarseOnly`/`CoarseResidue`) or part of a
  32-code PQ block (`Pq32x4`).** (Open Question 3.)

For the `MailboxSoA<N>` (column-array) source, there is **no `EdgeBlock`** —
`edges_raw()` returns decoded `CausalEdge64` per row. The edge walk over
that shape reads the `CausalEdge64` target/W-slot fields instead of the
16-byte palette block. v1 should pick ONE source shape (recommend
`&[NodeRow]` / `NodeRowPacket`, since that is where the operator's
EdgeBlock/EdgeCodecFlavor semantics actually live) and document that the
column-array `MailboxSoA<N>` edge walk is a separate follow-on.

### 4.3 Property filter `{name: 'X'}` and `RETURN n.name`

`NodePattern.properties` / `RETURN n.name` resolve through
`ClassView::fields` + `field_label` (class_view.rs L280, L292). This is the
single largest open question — see §8.

---

## 5. Layout-preserving guarantee (no version bump)

**Claim:** `Backend::MailboxSoa` is a pure *read* path over existing 512B
rows; it changes no on-disk/in-RAM layout and forces no
`ENVELOPE_LAYOUT_VERSION` bump.

**Proof:**

1. The scan reads `NodeGuid` (16B), `EdgeBlock` (16B), `value` (480B) from
   rows that already exist at `NODE_ROW_STRIDE == 512` (asserted at compile
   time, canonical_node.rs L324-326). It writes nothing to a row.
2. Edge-codec selection is `EdgeCodecFlavor::is_layout_preserving() == true`
   for every flavor (L305-307, regression-tested at L832-844
   `every_flavor_preserves_node_layout`). Choosing `Pq32x4` vs `CoarseOnly`
   re-*interprets* the same 16 bytes; it never re-strides the row.
3. Value-schema selection is `ValueSchema::is_layout_preserving() == true`
   for every preset (L613-615) — the property read (§8) carves *within* the
   reserved 480B slab, never beyond it.
4. `classid → ClassView` resolution stores nothing on the row (the
   meta-DTO "resolves; it does not store", class_view.rs L22) — labels,
   edge-codec, value-schema all resolve ABOVE the SoA.
5. No new SoA column is added (the AGI-as-glove rule: capability lands as a
   read over existing columns, not a new layer). `Backend::MailboxSoa` adds
   a router enum variant + a scan function — both in
   `crates/lance-graph/src/`, neither touches the contract layout.

Therefore the change is additive: a new reader, zero layout delta. (If
§8's property read ever needs a NEW `ValueTenant`, THAT would be a
contract-layer change gated by the canon "reserve, don't reclaim" rule and
filed as a separate upstream issue — but the v1 scan does not require it.)

---

## 6. Result shape

Extend `GraphHit` or add a sibling. `GraphHit` today (L31-42) is
`{ source: usize, target: usize, distance: u32, truth: TruthValue, backend:
Backend }`. For node-only `MATCH (n:Label) RETURN n` there is no edge, so
`target`/`distance`/`truth` are not meaningful. Two options:

- **Minimal:** reuse `GraphHit` with `target == source`, `distance == 0`,
  `truth == TruthValue::unknown()` for node hits, `backend: MailboxSoa`.
  Keeps one result type; cheap; the caller distinguishes node vs edge hits
  by the pattern it asked for.
- **Cleaner:** add `MailboxSoaHit { row: usize, class_id: ClassId, target:
  Option<usize>, flavor: EdgeCodecFlavor, properties: Vec<(String,
  PropertyValue)> }` and a `From<MailboxSoaHit> for GraphHit`. Carries the
  resolved properties for `RETURN n.name`. Recommended once §8 lands.

---

## 7. What already exists vs what to build

| Piece | Status | Where |
|---|---|---|
| `NodeGuid` / `EdgeBlock` / `NodeRow` / `EdgeCodecFlavor` | **EXISTS** (locked) | contract `canonical_node.rs` |
| `ClassView::edge_codec_flavor` / `value_schema` / `compute_dag` | **EXISTS** (#539) | contract `class_view.rs` |
| `MailboxSoaView` (`class_id`, `edges_raw`, `n_rows`) | **EXISTS** | contract `soa_view.rs` |
| `MailboxSoA<N>` owner + view impl | **EXISTS** | cognitive-shader-driver `mailbox_soa.rs` |
| `NodeRowPacket` zero-copy `SoaEnvelope` over `&[NodeRow]` | **EXISTS** | contract `canonical_node.rs` L711 |
| Row-window iteration over `MailboxSoaView` | **EXISTS — REUSE** | `arigraph/markov_soa.rs:170` (`SoaWavePrimer::project`), `scheduler.rs:116` (`drive_once`) |
| Cypher parser + `NodePattern`/`RelationshipPattern` AST | **EXISTS** | `parser.rs:23`, `ast.rs:137/166` |
| `Backend` enum + `GraphRouter` + `classify_query` | **EXISTS** | `graph_router.rs:46/76/223` |
| `Backend::MailboxSoa` variant | **BUILD** | `graph_router.rs` enum |
| `scan_mailbox_soa<V: MailboxSoaView>` node filter | **BUILD** | `graph_router.rs` (new fn) |
| EdgeBlock walk + `decode_edge_slot(flavor)` | **BUILD** | `graph_router.rs` (new fn) |
| label → `ClassId` resolution (`classids_for_labels`) | **BUILD / wire** | needs `&dyn ClassView` or an ontology helper; confirm `lance-graph-ontology` exposes label→classid |
| property read for `RETURN n.name` | **BUILD — but OPEN** | §8 |
| `NodeRowPacket: MailboxSoaView` impl (if scan reads NodeRow) | **BUILD (small)** | so the canonical `&[NodeRow]` shape flows through the same generic scan |

**Reuse the iteration pattern verbatim** from `SoaWavePrimer::project`
(markov_soa.rs L170-202): bound the loop with `view.n_rows()`, read
`view.class_id()` once, index by row. Do not invent a new traversal.

---

## 8. Open questions / risks (HONEST — read before building)

### OQ-1 (BIGGEST) — Where do node *properties* live for `RETURN n.name`?

The 480-byte `value` slab is carved into typed `ValueTenant`s (Meta,
Qualia, MaterializedEdges, Fingerprint, HelixResidue, TurbovecResidue,
Energy, Plasticity, EntityType — canonical_node.rs L394-415). **None of
these is a string `name`.** The value slab is cognitive/numeric state, not
arbitrary key-value properties. `ClassView::fields` returns `FieldRef`s
(predicate IRI + label) but `FieldRef` is the *schema* (which fields a
class has + their labels), resolved late from the OGIT cache — it is the
column basis, NOT the per-instance string values.

So `RETURN n.name` has **no obvious byte source on the row today.** Three
candidate resolutions, none yet chosen:

- **(a) Properties live OUTSIDE the SoA, keyed by node identity.** The SoA
  row carries `classid + identity` (the `NodeGuid`) and a `FieldMask`
  (which fields are populated); the *values* live in a side store
  (TripletGraph / a Lance column / the OGIT cache) retrieved by
  `(classid, identity)`. `RETURN n.name` = `field_label`-resolve "name" to
  its predicate, then look up the value in that side store. This matches
  the canon "the SoA stays agnostic; semantics resolve above it"
  (class_view.rs L20-32) and `I-VSA-IDENTITIES` ("bundle identities, not
  content; content lives in YAML/TripletGraph/EpisodicMemory"). **Most
  architecturally aligned.** Cost: the scan needs a handle to that side
  store, not just the `MailboxSoaView`.
- **(b) A `ValueTenant` carries a fingerprint, and `name` is recovered by
  cleanup-codebook lookup.** The `Fingerprint` tenant (32B, L461-465)
  points to content; a name codebook resolves fp → string. Lossy/indirect;
  only works for codebook-resident names. Probably wrong for arbitrary
  user strings.
- **(c) v1 punts on string properties entirely.** `MATCH (n:Label) RETURN
  n` returns row indices + classid + resolved `FieldMask` presence (which
  fields exist), NOT their string values. `RETURN n.name` is deferred until
  the side-store wiring (a) is specified. **Recommended for v1** — it keeps
  the scan honest (it returns exactly what the substrate holds) and defers
  the property-store decision to a focused follow-on rather than guessing.

> **This is the single design decision to resolve before the scan is
> useful for `RETURN n.<prop>`. Recommend (a) as the target and (c) as the
> v1 shipping shape.** Flag to the user.

### OQ-2 — classid width seam: `u16 ClassId` (SoA column) vs `u32 NodeGuid.classid` (key)

`MailboxSoaView::class_id()` is `&[u16]` (the `ClassId`/`entity_type`
column). `NodeGuid::classid()` is `u32` (the prefix-routable key). The
ontology must define how a `:Label` maps to which width, and how the u16
column relates to the u32 key prefix (is the u16 a dense interning of the
u32 classid? a separate discriminator?). v1 filtering on the u16 column is
correct for the `MailboxSoA<N>` shape; filtering on the u32 key is correct
for `&[NodeRow]`. **Pick the source shape first (recommend `&[NodeRow]`),
then this seam resolves.**

### OQ-3 — `Pq32x4` slot semantics vs `CoarseOnly` slot semantics

For `CoarseOnly`/`CoarseResidue`, each EdgeBlock byte is an independent
neighbor index → up to 16 edges per node. For `Pq32x4`, the 16 bytes are
ONE 32-code PQ vector → the "edges" are reconstructed differently (PQ
decode → nearest neighbors, not 16 literal slots). `decode_edge_slot` must
branch on flavor. The exact `Pq32x4` neighbor-resolution convention
(reconstruct-then-search? a fixed adjacency the PQ codes index into?) is
defined by `ndarray::hpc::edge_codec` and must be read before implementing
the `Pq32x4` arm. v1 may ship `CoarseOnly` only (the bootstrap default) and
defer `Pq32x4`/`CoarseResidue` edge walks.

### OQ-4 — Which concrete view does the scan bind?

`MailboxSoA<N>` (column arrays, decoded `CausalEdge64` edges, no EdgeBlock)
vs `&[NodeRow]` (raw EdgeBlock + value slab). The operator's spec
(EdgeBlock 12+4, `EdgeCodecFlavor` decode) only makes sense over
`&[NodeRow]`. **Recommend: v1 scans `&[NodeRow]` via a new
`NodeRowPacket: MailboxSoaView` impl (or a direct `&[NodeRow]` scan), and
the column-array `MailboxSoA<N>` edge walk is explicitly out of scope for
v1.** State this in the build ticket so a worker does not wire the wrong
edge representation.

### OQ-5 — label → ClassId resolver availability

`classids_for_labels` is assumed on `&dyn ClassView` or an ontology helper.
Confirm `lance-graph-ontology` exposes a `label → entity_type_id/ClassId`
map (the `ClassView` trait has `fields`/`template` but not an explicit
reverse `label → classid`; the `OntologyRegistry` hashtable likely has it).
If absent, that resolver is a small build item (or a new `ClassView`
method) — file upstream rather than hand-rolling a label table in the
router (per the "no parallel object model" doctrine).

### OQ-6 — `lance-graph` core does not build offline

`markov_soa.rs`'s own header (L48-55) warns the core crate's
lance/datafusion/arrow deps fetch from crates.io and it does NOT build in
the offline sandbox. Any implementation must compile-verify on a full
checkout. The scan itself depends only on the contract surface
(`MailboxSoaView`, `ClassView`, `canonical_node`) — consider whether the
scan can live behind a feature so it compiles without the full
lance/datafusion tree (the contract types are zero-dep).

---

## 9. Minimal test plan

All tests build a **hand-rolled** `MailboxSoaView` (or `&[NodeRow]`) with
2-3 rows — exactly the `FakeSoa` pattern already in `soa_view.rs` L147 and
`markov_soa.rs` L210. No Lance, no network.

1. **`match_node_by_label_filters_classid`** — build 3 rows:
   row0 classid→`:System`, row1 classid→`:System`, row2 classid→`:Other`.
   `MATCH (n:System) RETURN n` ⇒ exactly rows {0,1}; each hit
   `backend == Backend::MailboxSoa`.
2. **`match_labelless_matches_all_including_default`** — `MATCH (n)` over
   rows including a `classid == 0` (bootstrap) row ⇒ all rows returned
   (zero-fallback: label-less = match-all, default class included).
3. **`edge_walk_over_pq32x4_row`** — build a `&[NodeRow]` source row whose
   `classid` resolves (via a `FakeClasses: ClassView`) to
   `EdgeCodecFlavor::Pq32x4`, with a known `EdgeBlock` (e.g. in_family[0]
   pointing at row1). `(a)-[r]->(b)` ⇒ an edge hit src→1, with the hit
   recording `flavor == Pq32x4`. (If §8/OQ-3 defers `Pq32x4`, this test
   uses `CoarseOnly` and asserts the literal-index decode instead.)
4. **`edge_walk_in_family_then_out_family_order`** — a row with one
   in_family and one out_family non-zero slot ⇒ both targets emitted, in
   the canonical (12-then-4) order; zeroed slots produce no edge.
5. **`no_view_falls_back_no_panic`** — `classify_query` for a `:Label`
   `MATCH` with no view supplied routes to Blasgraph (existing behavior),
   no panic, no regression.
6. **`layout_preserving_smoke`** — assert `NODE_ROW_STRIDE == 512` and
   every `EdgeCodecFlavor::is_layout_preserving()` after a scan (guards the
   §5 claim at the call site).
7. **(deferred, gated on OQ-1)** `return_property_name` — only once the
   property-store decision lands.

---

## 10. Build order (when promoted from CONJECTURE)

1. Resolve OQ-4 (source shape = `&[NodeRow]`) and OQ-5 (label→ClassId
   resolver) — both are prerequisites, both small.
2. Add `Backend::MailboxSoa` variant + extend `classify_query` routing
   table (§3.3).
3. Add `NodeRowPacket: MailboxSoaView` impl (or direct `&[NodeRow]` scan
   entry) so the canonical shape flows through one generic scan.
4. Implement `scan_mailbox_soa` node filter (§4.1) reusing the
   `SoaWavePrimer::project` loop shape — tests 1, 2, 5.
5. Implement EdgeBlock walk with `decode_edge_slot` for `CoarseOnly` first
   — tests 3 (CoarseOnly form), 4, 6.
6. Defer `Pq32x4`/`CoarseResidue` edge arms (OQ-3) and property read (OQ-1)
   to focused follow-ons, each its own ticket.
7. Per board-hygiene: this plan → `INTEGRATION_PLANS.md` (PREPEND); on
   implementation, a `STATUS_BOARD.md` D-id row + `AGENT_LOG.md` entry.

---

## Appendix — agreement / disagreement with the operator's framing

- **Confirmed exactly as stated:** the GAP (canonical types only in
  contract, never in core router — `grep` empty); the `Backend` enum's 3
  current variants + dispatch; `ClassView::edge_codec_flavor`/`compute_dag`
  as the class→layout resolver; the EdgeBlock 12+4 layout;
  `is_layout_preserving == true` for all flavors (no version bump);
  `markov_soa.rs` + `scheduler.rs` already consume `MailboxSoaView`.
- **Refined / corrected:** the operator said the variant holds "a
  `&dyn MailboxSoaView` or owned `MailboxSoA`." Per the existing generic
  pattern (`drive_once<V>`, `project<V>`), the cleaner shape is a
  **generic scan function** bound `V: MailboxSoaView`, with `Backend::MailboxSoa`
  as a pure routing TAG — not a stateful field on `GraphRouter`. (Both work;
  the generic form is the smaller additive change.)
- **New risk surfaced (not in the operator's framing):** the live
  `MailboxSoA<N>` is a **column-array** SoA whose `edges_raw()` returns
  **decoded `CausalEdge64`**, NOT the raw 16B `EdgeBlock`. The operator's
  EdgeBlock/`EdgeCodecFlavor` walk is a property of `&[NodeRow]`. v1 must
  pick `&[NodeRow]` as the edge source; the two edge representations are a
  real seam (OQ-4).
- **Honest gap (the biggest):** node *string* properties for `RETURN n.name`
  have no byte home on the 512B row — the value slab is typed cognitive
  tenants, not arbitrary KV. Recommend properties resolve from a side store
  by `(classid, identity)` (OQ-1 option a), with v1 shipping classid +
  `FieldMask` presence only (option c) and deferring string values.
