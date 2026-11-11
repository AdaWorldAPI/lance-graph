# D-DRDT-SPLIT-1 — DR stores addressing as WideFieldMask; DT stores schema and behaviour as classes

**Status:** PROPOSED, docs only. Read at `main` `9f2137c4d` (2026-11-11).
Parent: `dr-dt64-class-governance-v3.md` (PR 2014), decisions D14 and D15,
and the master brief §4. Coordination: EXO-253; decisions: EXO-254.

## 1. The rule

| | concern | what it stores | stored as |
|---|---|---|---|
| **DR** | addressing: *where*, and at which version | field positions, row positions, slabs, frame and version witnesses | `WideFieldMask` |
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
  revision), not in needing three mask types. Passing one where another is
  expected is refused by the witness (`LaneRefusal::CoordinateSpace` today).
- **D-LANESOCK-0** ("`WideFieldMask` … never a row-survivor bit") describes
  the property request of one lane binding. It stands for that binding; it
  is not a limit on the carrier.
- **Locality contract, boundary 2** lists `WideFieldMask` with DT64. Read:
  DT64 says what a selection selects; DR carries it.

Size: a mask holds 256 positions and that cap stays. A 256 × 256 slab is 256
masks, which is the 1,024-word plane already in the lab; ×256 slabs under an
outer byte when needed.

## 2. Why it fits what is on main

Read in this session, with locations:

- `WideFieldMask` holds up to 256 positions, `u8`-indexed, as `u64` words
  (`lance-graph-contract/src/class_view.rs:243`, `with` `:281`, `words`
  `:511`, `intersect` `:385`; cap 256 at `from_universe_present` `:693`).
- A DR256 slab is 256 × 256 positions. One row of it is one 256-bit mask,
  so a slab plane is 256 masks = 1,024 words = 8 KiB. The lab's `Plane`
  (`dr256-ternlog-lab/src/lib.rs:361`) is exactly that many words but is its
  own type.
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
| G2 | The order index is a permutation plus keys (`Vec<u32>`, `Vec<i32>`), not a mask. It is addressing, so it is DR's, but it is not expressible as a `WideFieldMask`. OPEN: keep it as the one non-mask DR artifact, or find its mask form. | `dr256-ternlog-lab/src/lib.rs:825` | open question |
| G3 | The locality contract lists `WideFieldMask` under DT64 (boundary 2). EXO-220 repeats it. | `.claude/knowledge/dr-dt64-locality-execution-contract.md`; EXO-220 | stale text |
| G4 | Four production paths execute without a class binding (governance v2, F4). | BYPASS-MATRIX | enforcement |
| G5 | Method bindings are held in `BindStore` and are not persisted; a `MethodRef` stamp is process-local (governance v2, F7). "Behaviour stored as classes" does not survive a restart. | `lance-graph-dt64/src/lib.rs` | replay |
| G6 | `MethodBinding::on` and `reads` are `WideFieldMask`s inside a DT structure. Under the rule they are DR values that a class refers to. OPEN: is that a field of the class, or a DR address the class names? | `lance-graph-dt64/src/lib.rs` (`MethodBinding`) | open question |

## 4. Enforcement, and why it is affordable

Checks happen when a binding is made, once per frame: DT resolves class,
basis and method; DR seals mask, planes and index against the frame. A
query carries the sealed binding and pays only the frame check. Measured in
PR 2010 at 1,000,000 rows (16 slabs), including that check: key range over
an order index 1.76 µs, hop over prepared planes 4.86 µs, filtered hop
2.17 µs; preparation once per frame 3,600–41,300 µs
(`.claude/board/entries/2026-11-11-cypher-df-vs-quack-bench.md`). A source
advance makes every stale binding refuse by name.

So the executor tier stays free of DT64 (governance v2, F5) and central
enforcement costs nothing per row.

## 5. Work items, mapped into plan v3 and the existing Linear issues

No new issue is opened; each item lands under the issue that already owns
the question.

| item | content | plan v3 | Linear (existing) | gate | kind |
|---|---|---|---|---|---|
| S1 | Dated supersession notes: locality contract boundary 2 moves `WideFieldMask` to DR; old text stays visible | D15 | EXO-220 (the contract's decision), EXO-254 | docs review | docs |
| S2 | Lab: carry slab planes as the reusable selection (256 masks of 256 with their `CoordSpace` witness), no behaviour change | W4 (row selection) | EXO-253, EXO-121 | the 15 lab tests unchanged and green; the angles bench within noise, in ns/µs | lab code |
| S3 | Decide G2 (order index) and G6 (`on` / `reads`) | W3, W4 | EXO-254, EXO-121 | council | decision |
| S4 | Field half of reader scope: `requested ∩ visible` as one `WideFieldMask` intersection at bind time; row half as a prepared DR plane; frame ownership for an external lane source | W12 | EXO-243, EXO-244, EXO-228, EXO-206 | the EXO-243 test: two roles, 12 fields, 10^3 rows; refused across roles; non-Allow binds nothing | test-only first |
| S5 | Bind-time admission for the four direct paths, with the executor census test | v2 cut 4; W2, W13 | EXO-253, EXO-245, EXO-246 | red on a planted call site | needs council |
| S6 | Behaviour as classes that outlives the process: binding digest recorded with the execution and the DR version; typed lane interpretation from the class | v2 cut 7; W5; G5 | EXO-199, EXO-225, EXO-221, EXO-202, EXO-247 | replay falsifier (row 17) red first | needs council |

Order: S1 and S2 now. S4 next (a consumer is waiting on EXO-243 and
EXO-244). S3 before S5 and S6.

## 6. Not verified

Nothing was built or run for this plan. G2 and G6 are open questions, not
findings. Whether `WideFieldMask`'s `Small`/`Wide` representation is the
right in-memory form for 256 rows of a slab (allocation per row) is
unmeasured; S2's gate measures it.
