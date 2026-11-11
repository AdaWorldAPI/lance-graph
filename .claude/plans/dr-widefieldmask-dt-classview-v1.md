# D-DRDT-SPLIT-1 — DR stores addressing as WideFieldMask; DT stores schema and behaviour as classes

**Status:** PROPOSED, docs only. Read at `main` `95168f9e3` (2026-11-11).
Exploration map: `.claude/board/entries/2026-11-11-coresearch-drdt-split.md`.
Parent: `dr-dt64-class-governance-v3.md` (PR 2014), decisions D14 and D15,
and the master brief §4. Coordination: EXO-253; decisions: EXO-254.

## 1. The rule

**`WideFieldMask` is agnostic algebra. A class is a computational concept:
it holds schema and methods.** DR uses the first to address; DT uses the
second to map. Everything below is consequence.

As stated by the operator (2026-11-11):

- **DR is the binding store of masks.** It stores the versioned, immutable
  DR × DT selections (bindings), prefetched and reusable. A mask computed
  for one request is not versioned and is not stored (§1b).
- **What is versioned is the mapping, not the content (operator,
  2026-11-11; supersedes the content-replay reading added in PR 2015).** A
  contextual view (`PSCustomObject64` / XVERWEIS) is built from context. Its
  mapping joins identities, properties and source positions; that mapping is
  versioned. The addressed contents stay with their source owner and are not
  versioned by this plan. The same mapping reproduces the same assignment;
  identical result values follow only when the contents read are also
  identical. Mapping replay and historical content replay are separate
  requirements; this plan asks only for the first.
- **The mapping is DR-adjacent.** It binds the context to DR addresses
  without being stored as a DR mask, and it lives where the existing
  contract puts it: no new store instance for this distinction alone, and
  no extra mask representation derived from it. Identity order may be
  implicit (codebook, trie, Cartesian coordinates); the contextual property
  selection uses that assignment. Methods stay addressable entries of an
  immutable register.
- **DT is immutable methods.** The algebra they name is handed over bit-exact
  from `ndarray`, mask-RISC, relalg, XLOG and the other registered
  technologies.
- **DT64 controls behaviour through `ClassView`.**
- **`F###` is the methods of that behaviour**, expressed as ON × WHEN × DO.
- **The Codegraph of PR 2000 is built on this**: it reads the registered
  methods' ON × WHEN × DO and the algebra words they use (CG-BEH-1 and
  CG-BEH-2, PRs 2003 and 2005).

| | concern | what it stores | stored as |
|---|---|---|---|
| **DR** | addressing: *where*, and at which version | selections over any axis (fields, identities, rows of a slab); the versioned bindings that seal them | selections as `WideFieldMask`; bindings sealed by frame and version witnesses (`FrameStore`, `FrameRevisions`, `MapTable`) |
| **DT** | mapping: *what* | schema (ordered field basis, revision) and behaviour (ON × WHEN × DO) | classes (`ClassView`, keyed by classid) |

The only value that crosses from DT to DR is a position. DR never
interprets; DT never addresses. Crate ownership of the mask type does not
move (brief §4: the logical concern and the Rust module boundary are
separate questions).

## 1a. One reusable selection, never untyped (2026-11-11)

`WideFieldMask` is DR's reusable selection: the role DuckDB gives its
selection vector, in bit form. It is built once per version and shared by
every consumer (storage read, fold, display, route, population). It is not
confined to one axis.

It is also never untyped. A selection always travels with what binds it:
the class and basis (DT), and the coordinate space and frame (DR
witnesses). That is how this reads together with the texts already on main:

- **Master brief §7** names three concepts — column projection, row
  selection, relation index — and forbids mixing them "into an untyped
  carrier". They stay three: they differ in what they are bound to (a class
  basis; a population under a snapshot; a directed relation with its own
  revision), not in needing three mask types. Which one an operand
  is gets fixed when the binding is built, so a binding cannot hold the
  wrong kind in a slot: a type check at bind time, never a refusal at query
  time (plan v3 W4's falsifier is that bind-time check). Missing values are
  presence bits, already in the fold.
- **D-LANESOCK-0** ("`WideFieldMask` … never a row-survivor bit") describes
  the property request of one lane binding. It stands for that binding; it
  is not a limit on the carrier.
- **Locality contract, boundary 2** lists `WideFieldMask` with DT64. Read:
  DT64 says what a selection selects; DR carries it.

**256 × 256 × 256 is a shape, not a meaning: universal algebra.** The same shape is read in
whatever way the binding names:

- a flat row space: 256 × 256 rows per slab, × 256 slabs, up to 16.7 M rows
  (1,000,000 rows is 16 slabs; the PR 2010 lab);
- a three-axis cube: identity × property × predicate (`Extent`,
  `lance-graph-dt64/src/dr.rs:91-125`);
- 64 K × WHERE: one 256 × 256 plane of 65,536 positions under up to 256
  selections (criteria, methods, perspectives).

A mask holds 256 positions and that cap stays; every reading is built from
256-position masks. Because every reading is masks, every reading can use the same
three-operand fold: one ternlog (any Boolean function of three masks, one
instruction per word) plus a popcount, through `ndarray::simd`
(`mask_ternlog`, `mask_ternlog_popcount`, `mask_ternlog_any`;
`ndarray/src/simd_masking_ops.rs:577,673,734`). The PR 2010 lab already does
it for the flat reading (presence × scope × selection, 188 µs at 1,000,000
rows). The cube and 64 K × WHERE readings are untested. The one condition:
the three operands share a coordinate space, which the binding checks before
the fold, never the fold itself. Whether the ternlog fold is the best path depends on
context: over a whole selection it is bandwidth-bound (188 µs at 1,000,000
rows), while a key range over a prepared order index answers in 1.76 µs. The
choice is made once per binding, from the context the binding names, and
measured; it is never a fixed rule. For a one-dimensional context (one key over
1,000,000 rows) the index reorder is the cheapest thing to store. The measured
`OrderIndex` holds a `u32` permutation (4 MB at 1,000,000 rows) and the
sorted `i32` keys (another 4 MB); `PreparedOrder` adds visibility planes and
the lane bindings. Built once per frame (41,300 µs), it answers any range on
that key in µs, where a mask per range would cost 125 KB each and grow with
every new range. A persisted permutation without its keys needs an explicit
contract for where the keys come from, and does not inherit the 1.76 µs
until measured. Lance stores no usable order: its BTree keeps sorted values
with row addresses but every search returns a row set
(`SearchResult::{Exact, AtMost, AtLeast}(NullableRowAddrSet)`, checked in
Lance 13.0.0 and 14.0.2), so the permutation is our own sidecar (EXO-114). The permutation is a DR-adjacent
mapping of positions, valid for the source revision its keys were read
from (its witness); the sorted keys it carries are a derived cache of
source content, not a versioned copy of it. Which reading applies is never in the bits: the binding
names it (class and basis from DT, coordinate space and frame from DR).

## 1c. Reference picture: DR / DT64 virtual address contract (operator, 2026-11-11)

The operator's architecture picture, read against this plan:

| picture | this plan |
|---|---|
| NodeGuid → DR (ordered address contract) → DT64 (behaviour and method binding) → Quack (algebraic lowering) → mask-RISC / `ndarray` SIMD → typed result; "DT64 binds, then Quack executes" | §1, §1b (D-QSP: ClassView resolves class and method, the Quack binding resolves snapshot and lanes) |
| DR = where / coordinate / version; DT64 = what / method / behaviour | §1 table |
| 256 × 256 ordered identity plane, 65,536 canonical positions; WHO → WHAT → WHERE → WHY, "resolve what is needed before loading values" | §1a shape readings |
| one canonical index over several source-owned SoA property lanes (A[i]…E[i]); "virtual transpose / stride / block transpose possible; logical order stays fixed" | see below |
| versioned SoA projection: immutable per sealed version, source-owned values | §1 mapping vs content |
| virtual relations X × Y as a callable relation, no mandatory intermediate table | not yet a work item (G8) |
| two execution modes: direct ISA / ALU, and Abakus (lab: algebraic diffusion × atomization) | not in this plan |
| key idea: a computation need not become data before it may become another computation | the zero-copy law and §4 (prepared once, no tuple materialization) |

Two consequences for this plan:

- **The identity order is the one canonical index.** Every property lane is
  read through it (`ndarray::simd` stride words, `src/simd_stride.rs`). A
  per-key order index (S3) is a physical relayout made as a preparation
  step; it must never change the logical identity order, and its answers are
  stated in canonical positions.
- **G8, virtual relations.** A relation X × Y is called, not built as an
  intermediate table. Plan v3 W4 names the relation index as the third
  typed concept; how a callable relation is bound (DT64) and executed
  (Quack, mask-RISC) without materializing is open and has no work item.

## 1b. The binding store is already planned: the Quack Binding Store

"DR is the binding store of versioned immutable masks" is the store the
existing plans call the **Quack Binding Store**. It is planned and not built.
Read on main:

- **DR256 WHERE capstone** (`dr256-where-capstone-v1.md`):
  - §B.1: "the generalized Quack Binding Store: PLANNED / UNBUILT", defined by
    D-QSP and D-NFR S5, smallest seam D-NFR-1. Its role is "a DTO /
    ClassView-to-native-algebra binding authority over folding algebra. It
    owns reusable binding descriptors, their witnesses and lifetimes."
  - §B.7 finding 5: "A witness on a kept R is the blocking correctness gap.
    Every reuse control ran silently stale through Quack." The binding store
    is named as the owner.
  - §B.8: the store must keep "reusable, version-witnessed semantic
    bindings" as compiler input, reusable without materializing
    intermediates.
  - §C.3: four lifetimes with separate owners: bound expression (binding
    store); prepared computation (mask-RISC); computed intermediate ("the
    caller's buffer, under the descriptor's witness"); materialized result
    (consumer).
  - §D.3: the smallest steps, in order: a frame on a bound Quack `Query`
    (D-NFR-1); a dependency witness for a kept plane (owner D-QSP); the
    codebook generation inside the frame (D-NFR-2).
- **D-NFR** (`d-native-repr-contract-v1.md`) S5: "binding store … MISSING
  in Quack"; D-NFR-1 is the lab probe: bind a Quack `Draft`, seal a frame,
  execute, advance one revision, the second execution is refused.
- **D-QSP** (`d-qsp-classview-quack-postgresql-parity-v1.md`): ClassView
  resolves class, method ordinal and operation law; Quack Binding resolves
  snapshot, typed domains, codebook, focus, validity and lanes; "do not put a
  behavior choice into the binding store".
- **D-DYNBAG** (`d-dynbag-holographic-bags-v1.md`): X4 / MISSING-2 "no
  Quack `Binder` backed by `BindStore` / `ClassView` (8 impls, none)";
  MISSING-5 "a reuse cache for prepared `LocalMap`s keyed by frame".
- **D-DRDT amendment**: Quack's `Binder` "stays the operational
  query-binding membrane; its prepared numeric bindings are not DT's
  semantic bindings".
- Linear: EXO-121 (ClassView / WideFieldMask "adjacent to Quack Binder and
  DTO/store"), EXO-207 (CodeTable + Quack admit, PR 1982), EXO-251 (medcare
  WP4, DT64 bind store for view classes).

**What is versioned and what is not (operator, 2026-11-11).** The mapping
that binds DR to DT (which identities, which fields of a class, which
method, which source positions) is versioned: it is the binding the store
keeps. The contents it addresses are not; they stay with the source
owner. A mask
computed for one request (a user clicks a menu) is not versioned: it is
computed under the binding's witness and lives in the caller's buffer.
This is the capstone's split (§C.3: lifetime 1 kept by the binding store,
lifetime 3 in the caller's buffer under the witness), so there is no
conflict with "not a population bitmap store".

**The kept, witnessed selection already exists in the lab.**
`dr-math-binding-probe/src/where_dr3.rs` (D-DRCA-2) takes Quack's WHERE
output as one borrowed 8 KiB mask, projects it into a frame
(`LocalSelection`), and re-checks on every read the source seal (owner,
source revision, codebook, address space), the local frame, the focus, the
scope, the identity axis, the DR method and the criteria
(`LocalSelection::admit`); "nothing is silently reused". Its own header says
what is missing: "It stores nothing … There is no cache, map of selections
or persistent mask store." That missing piece is the DR store of the
versioned DR × DT selections; per-request masks stay out of it. Quack's `Binder` (`bind.rs:92`, ten
implementations, none backed by `BindStore` / `ClassView`) and `Query`
(`lib.rs:1301`, no frame recorded) are the other end of the seam.

**What this plan's work maps onto.** S4 (`requested ∩ visible` at bind
time) is lifetime 1 of §C.3; the PR 2010 lab's prepared planes are lifetime
3. Build order: D-NFR-1 (a frame on a bound Quack `Query`) first, then the
store of versioned selections (S7) over it. The store is not built before
the frame exists to seal it.

## 2. Why it fits what is on main

Read in this session, with locations:

- `WideFieldMask` holds up to 256 positions, `u8`-indexed, as `u64` words
  (`lance-graph-contract/src/class_view.rs:243`, `with` `:281`, `words`
  `:511`, `intersect` `:385`; cap 256 at `from_universe_present` `:693`).
- The 256 × 256 × 256 shape is read three ways on main today: `Extent` in
  `dr.rs` reads it as identity × property × predicate; the PR 2010 lab reads
  it as a flat row space (slabs of 65,536 rows, `Plane` with a runtime
  `CoordSpace`); the D-DRCA-2 lab reads one 65,536-position plane as a WHERE
  over a global address space (`where_dr3.rs`). One shape; the binding says
  which reading.
- Schema and behaviour are already keyed by classid: `BindStore::register`,
  `declare_algebra`, `resolve`, `admit` (`lance-graph-dt64/src/lib.rs`);
  a behaviour lens is a classid (`class_view.rs:1053`); EXO-199 keeps the
  classid as the class and attests the binding.
- The versioning store is DR's: `FrameStore` / `FrameRevisions`, `MapTable`,
  and one Lance version per map advance (EXO-225).

## 3. What does not fit yet

| # | gap | where | kind |
|---|---|---|---|
| G1 | The lab carries its planes as `Plane { space, words }`, not as `WideFieldMask` rows. DR's storage therefore has two spellings. | `dr256-ternlog-lab/src/lib.rs:361` | lab code |
| G2 | ANSWERED (exploration map). The order index stays a permutation: DR's relation index, sibling of `LocalMap`, sealed per frame. A key range as a mask is already `ndarray::simd` compare-to-mask in place; a bit-sliced copy would be a second copy. Open only: its seal gaps (ordered-lane identity, presence version; fold law 2). | `dr256-ternlog-lab/src/lib.rs:825` | seal gaps |
| G3 | The locality contract lists `WideFieldMask` under DT64 (boundary 2). A dated note is added (PR 2015); the EXO-220 description still says it. | `.claude/knowledge/dr-dt64-locality-execution-contract.md`; EXO-220 | stale text (Linear) |
| G4 | Four production paths execute without a class binding (governance v2, F4). | BYPASS-MATRIX | enforcement |
| G5 | Method bindings are held in `BindStore` and are not persisted; a `MethodRef` stamp is process-local (governance v2, F7). "Behaviour stored as classes" does not survive a restart. | `lance-graph-dt64/src/lib.rs` | replay |
| G7 | A same-count field reorder passes the hot binding check under a positional source (pinned red-first, `lance-graph-dt64/tests/lane_socket.rs:416`). | `lane.rs:296-349` | correctness |
| G8 | Virtual relations: a relation X × Y is called, never built as an intermediate table (§1c). How it is bound by DT64 and executed by Quack / mask-RISC without materializing has no work item. | plan v3 W4 (relation index) | design |
| G6 | ANSWERED (exploration map). `on` / `reads` stay in `MethodBinding`, covered by its digest (agnostic algebra held by an immutable method). A reverse index (`methods_touching`) may be derived, never a second store. | `lance-graph-dt64/src/lib.rs` (`MethodBinding`) | none |

## 4. Enforcement, and why it is affordable

Checks happen when a binding is made, once per frame: DT resolves class,
basis and method; DR seals mask, planes and index against the frame. A
query carries the sealed binding and pays only the frame check. Measured in
PR 2010 at 1,000,000 rows (16 slabs), including that check: key range over
an order index 1.76 µs, hop over prepared planes 4.86 µs, filtered hop
2.17 µs; preparation once per frame 3,600–41,300 µs
(`.claude/board/entries/2026-11-11-cypher-df-vs-quack-bench.md`). A source
advance does not invalidate a mapping: the mapping is versioned and
immutable, so mapping N reproduces the same assignment of identities,
properties and source positions whenever it is replayed. The result values
equal the earlier ones only if the contents read are equal; replaying
historical contents is a separate requirement owned by the source, not by
this plan. A mismatch still refuses: a mapping whose source positions or
identities do not match the source it is applied to, a foreign source, or
the wrong codebook generation fails at the frame check, never per row. Kept Lance versions do not by themselves
restore historical frames, methods or typed interpretation (map-table
persistence EXO-225; historical typed attestation still blocked, EXO-202),
and authorization stays independent at the read and execution boundary
(EXO-220).

So the executor tier stays free of DT64 (governance v2, F5) and the
per-query cost of enforcement is the frame check, already inside the
figures above. Enforcement itself is not built: four paths still bypass it
(G4), and the reorder gap (G7) is open.

## 5. Work items, mapped into plan v3 and the existing Linear issues

No new issue is opened; each item lands under the issue that already owns
the question.

| item | content | plan v3 | Linear (existing) | gate | kind |
|---|---|---|---|---|---|
| S1 | Dated notes on the locality contract and D-LANESOCK-0 (DONE in PR 2015); the EXO-220 description still to note | D15 | EXO-220, EXO-254 | docs review | docs |
| S2 | Lab: carry slab planes as the reusable selection (256 masks of 256 with their `CoordSpace` witness), no behaviour change | W4 (row selection) | EXO-253, EXO-121 | the 15 lab tests unchanged and green; the angles bench within noise, in ns/µs | lab code |
| S3 | Seal the order index per frame: ordered-lane identity and presence version (G2's remaining gaps) | W4 | EXO-121, EXO-114 | a small fixture with duplicate keys, an explicit tie-break and frame-relative positions: persist, reopen, apply the permutation without sorting, compare the full ordered row sequence (not counts) against an independent oracle; the same index over the same source reproduces the same sequence; an index whose witnessed source revision differs from the source it is applied to refuses | lab code |
| S4 | Field half of reader scope: `requested ∩ visible` as one `WideFieldMask` intersection at bind time; row half as a prepared DR plane; frame ownership for an external lane source | W12 | EXO-243, EXO-244, EXO-228, EXO-206 | the EXO-243 test: two roles, 12 fields, 10^3 rows; refused across roles; non-Allow binds nothing | test-only first |
| S5 | Bind-time admission for the four direct paths, with the executor census test | v2 cut 4; W2, W13 | EXO-253, EXO-245, EXO-246 | red on a planted call site | needs council |
| S6 | Behaviour as classes that outlives the process: binding digest recorded with the execution and the DR version; typed lane interpretation from the class | v2 cut 7; W5; G5 | EXO-199, EXO-225, EXO-221, EXO-202, EXO-247 | replay falsifier (row 17) red first | needs council |
| S7 | The versioned DR × DT mappings over D-NFR-1 (identities, properties, source positions, class binding, criteria), DR-adjacent, held where the existing contract puts them (no new store instance), handed out by reference; contents stay with the source owner; per-request masks stay outside | §1b; W5 | EXO-253, EXO-220 | D-NFR-1 first: bind, seal, execute, advance one revision, gates: mapping N replayed reproduces the same assignment (identities, properties, source positions) bit for bit; over unchanged contents the result is bit-equal too; over changed contents with unchanged positions the assignment is unchanged and the values follow the source; a mapping whose positions or identities do not match the source refuses; then the same for a mapping kept in the store; authorization checked independently | lab code |
| S8 | Close the reorder gap on the hot check (D-DT64-7 L5d), keyed on revision, not a digest | W2 | EXO-199 | the pinned test turns red-to-green; an unchanged view still admits; at most a few ns per check | dt64 |

Order: S1 (finish) and S2 now. S4 next (a consumer is waiting on EXO-243
and EXO-244). Then D-NFR-1 and S7, S3 and S8 in parallel. S5 and S6 need
the council.

## 6. Not verified

Nothing was built or run for this plan. D-NFR-1's own gate reads "the
second execution is refused" (§1b quote, `d-native-repr-contract-v1.md`);
with versioned mappings it is read as: the mapping replays and reproduces
the same assignment; it is refused only where its positions or identities
no longer match the source. Historical content replay is the source's
requirement, not this plan's. D-NFR-2 (codebook generation) and step 8 of
the D-NFR §6 fixture read the same way. The board's own D-NFR-1 line ("no
invented FrameStore") agrees with "no new store instance for this
distinction". Not yet written into that plan. G2 and G6 were answered by the
exploration run, from code and literature, not by a measurement. Whether `WideFieldMask`'s `Small`/`Wide` representation is the
right in-memory form for 256 rows of a slab (allocation per row) is
unmeasured; S2's gate measures it.
