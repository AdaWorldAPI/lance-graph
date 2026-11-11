# 2026-11-11 — coresearch: D-DRDT-SPLIT-1 (DR addresses with WideFieldMask, DT maps with ClassView)

**Kind:** exploration map (co-research council). Ratifies nothing.
**Plan explored:** `.claude/plans/dr-widefieldmask-dt-classview-v1.md`.
**Linear (existing):** EXO-253 (coordination), EXO-254, EXO-220, EXO-121,
EXO-243, EXO-244, EXO-199, EXO-225, EXO-202.
**Run:** premise gate, 5 scouts (code, internal prior art, literature,
systems, concepts), crosswalk of 14 ideas, 3 co-architects.

## 1. Anchor as stated by the operator during the run

`WideFieldMask` is not to be boxed. It is DR's content-blind addressing
carrier: a set of positions with no meaning of its own. What a position
stands for (a field, an identity, a row of a slab) is DT's concern, through
`ClassView` and the coordinate space. This restates D15 and the master brief
§4 ("a set bit must not by itself mean a business semantic").

**The council worked from the boxed reading.** The premise gate, the
bridge architect and the firewall critic all treated `WideFieldMask` as
"class field positions only" and proposed keeping row populations in a
separate type (ideas I1, I2). Those verdicts are recorded below as they were
returned and are marked where the anchor overrides them.

**Texts on main that state the box** (found by the scouts, read first-hand):

- D-LANESOCK-0, ownership table: `WideFieldMask` owns "which properties are
  requested", never "a row-survivor bit"
  (`.claude/plans/d-lanesock-dr-dt64-lane-socket-v1.md:15`).
- Plan v3 §3: "one selection type with several carriers" recorded as
  rejected; three distinct typed concepts required
  (`dr-dt64-class-governance-v3.md`, corrections; W4).
- Locality contract boundary 2 lists `WideFieldMask` under DT64.
- The type's own docs describe field positions of a class
  (`lance-graph-contract/src/class_view.rs:203-251`).

Whether these are superseded is the operator's call; this entry changes none
of them.

### 1a. The wider frame (operator, same run)

"Everything masking positions, routing and population selection is similar to
what DuckDB does via a reusable selection vector."

Read that way, `WideFieldMask` plays the ROLE DuckDB's selection vector
plays: one reusable selection object, built once and shared by every
consumer of the same data — the storage read, the fold, the display, the
route, the population — instead of each consumer deriving its own. (DuckDB:
one selection is applied to all columns of a chunk by slicing, without
copying the columns, and is reused across operators; SECONDHAND here, the
systems scout read only the vector page.) The difference is the form: a bit
mask instead of an index list.

The council missed this. It filed "selection vectors" as one idea among
fourteen (I14) and skipped it on fold law 5, which rejected the INDEX-LIST
form on one workload. The role was never the thing rejected: the mask is
this workspace's selection vector, and the amortization measured in PR 2010
(1.76 µs per range at 1,000,000 rows) is what reuse of one prepared
selection buys.

So the questions in §3 are sub-questions of one statement: **one reusable
selection, bound once per version, addressed by DR, given meaning by DT,
shared by every consumer.** Enforcement then reads: a consumer takes its
selection from that carrier or it does not run.

## 2. What the code holds (VERIFIED-IN-CODE unless marked)

- A DR tile's axes are identity × property × predicate
  (`lance-graph-dt64/src/dr.rs:91-125`). `DR256_SQ` is 256 identities × 256
  properties. One identity's row of that tile is 256 bits: one
  `WideFieldMask`.
- `WideFieldMask`: positions are `u8`, at most 256; `Small(u64)` up to 64,
  then boxed words (`class_view.rs:243,281-324`).
- The lab (`dr256-ternlog-lab`) reads 256×256 as rows × rows: a `Plane` is
  65,536 row bits = 1,024 words, tagged at runtime with a `CoordSpace`
  (`src/lib.rs:359-376`). Same bits as 256 masks of 256; not typed as such.
- Row populations elsewhere are plain `u64` words (mask-risc `Planes.masks`,
  Quack `Mask(u16)`); `WideFieldMask` is used today for field, method
  footprint and lens-word selections only.
- Use-time checks are already two-sided: data side `FrameStore::check`
  (names `SourceChanged` / `MapChanged` / `CoordinatesChanged`), lane
  `CoordSpace` and `Scope`; class side route position, schema revision and
  field count, `BasisDigest`, lane mapping (`lane.rs:167-263,296-349`).
  **Gap:** a same-count field reorder passes the hot check.
- Behaviour is immutable per class; `Stamp` is process-local; `BindStore` is
  not persisted; only the DR map table is durable
  (`lib.rs:370-468,893-911`; `lance-graph-dt64-lance/src/lib.rs:1-41`).
- `LocalMap` is already a permutation-shaped DR artifact (`dr.rs:568-655`).
- No bit-sliced, range-encoded, rank/select or wavelet index of a sort order
  exists in `crates/**/*.rs` (closed pattern set). `RankMap` has rank, no
  select.

## 3. Verdicts

| # | idea | outside evidence (grade) | council | verdict | note |
|---|---|---|---|---|---|
| I3 | A DR256 tile row is a `WideFieldMask`; the lab slab is the same 8 KiB read on another axis | inside only | bridge OPPORTUNITY; firewall PASS | **ADOPT-NOW (docs)** | Say in the plan and the lab header which axis each reading addresses, and that the coordinate space names it. |
| I8 | Flat 1,024 words is the standard dense form at 2^16 | Roaring (SECTION-READ) | OPPORTUNITY; dense PASS, sparse array CONFLICT (fold law 5) | **ADOPT-NOW (docs)** dense; **PARK** sparse/run | One dense carrier; 256 bands of 256 is a schedule over it. |
| I10 | Two refusal axes, two mechanisms (schema/behaviour vs data version) | PostgreSQL plan cache vs snapshots; Lance manifest vs index coverage (SECTION-READ) | OPPORTUNITY; PASS | **ADOPT-NOW (test)** | Probe P9. Already the shape on main. |
| I11 | Pin the resolution context, not only the touched objects | PostgreSQL PREPARE failure mode (SECTION-READ); capabilities (SECTION-READ) | OPPORTUNITY; PASS | **PROBE** P4 | Red-first: the same-count reorder passes today. Fix keyed on revision, not on a digest. |
| I6 | Keep the order index as a named DR relation index, sealed per frame | DuckDB sort, ClickHouse sparse index (READ-IN-FULL) | OPPORTUNITY; permutation PASS, index-ordered visibility plane CONFLICT (second copy of a row mask) | **PROBE** P1 arm C + seal test | Open: is the index-ordered visibility plane a second copy (firewall) or the only copy in that order? |
| I4 | Bit-sliced key planes answer a range as a mask | O'Neil & Quass 1997, BitWeaving (SECTION-READ) | WORTH-EXPLORING; CONFLICT as a sidecar (same information as the key column) | **PROBE** P1 | Admissible only as the key lane's one layout, never beside it. Under the unboxed anchor the slices are ordinary DR masks. |
| I13 | Behaviour that outlives the process | PostgreSQL catalog rows; Erlang code server (READ-IN-FULL); Smalltalk image | WORTH-EXPLORING; CONFLICT | **PROBE** P5 | Key = classid + method + issuing store + durable version; digest only detects change. "Rebuild from compiled tables" passes outright. Image snapshot and digest-keyed rows are traps. |
| I12 | Method footprint as a derived relation | SQL `routine_column_usage` (READ-IN-FULL, one system) | WORTH-EXPLORING; CONFLICT if a second store | **PARK** | `on` / `reads` stay in the binding (covered by its digest); a reverse index may be derived, never authoritative. D-DT64-7 O4 stays deferred. |
| I7 | Rank/select directory; wavelet tree | Kurpicz 2022, Gagie et al. 2009 (READ-IN-FULL) | rank/select WORTH-EXPLORING, PASS; wavelet DROP | **PARK** (select); **SKIP** (wavelet) | Add select when a consumer needs the k-th survivor (P7). |
| I9 | Per-block value summary | Column imprints (SECTION-READ) | DROP (conditional); CONFLICT | **PARK** | Admissible only if measured (P8); never written into Lance. |
| I5 | Range-encoded bitmaps | Chan & Ioannidis 1998 (SECTION-READ) | DROP; TRAP | **SKIP** | About 2 MiB per 65,536 rows for an 8-bit key. |
| I14 | Selection vectors | DuckDB (SECTION-READ) | DROP; TRAP | **ALREADY-HAVE as a role (§1a)**; index-list form **SKIP** | The council judged the form and missed the role. `WideFieldMask` is the reusable selection; only the index-list form is rejected by fold law 5. |
| I1 | Row population and property selection as two separate TYPES | FCA, C-Store, DuckDB, PostgreSQL, Lance (SECTION-READ); no source reports a failure from one type | OPPORTUNITY; PASS | **SKIP as returned (conflicts with §1)** | What survives: one carrier, and the axis it addresses is named by the coordinate space / `ClassView`, never guessed from the bits. FCA's own form agrees: extents and intents are both bit vectors, told apart by their index set. |
| I2 | Compile-time domain tag on the mask | Rust `PhantomData` (READ-IN-FULL) | WORTH-EXPLORING; CONFLICT (A9) | **SKIP** | A tag boxes the type; the runtime `CoordSpace` already names the axis. |

## 4. Named probes (designed, not run)

All timings in ns or µs; a per-query millisecond is a failed gate.

- **P1** bit-slices vs permutation on one 65,536-row slab, 8-bit and 32-bit
  keys, ranges at 1 / 10 / 50 %: (A) unmasked, (B) under a random row mask,
  (C) under a mask prepared in index order. Kill "replace the index": slices
  more than 2× slower in A or C. Kill "stand beside": slices not 2× faster
  in B. Oracle: row-at-a-time count. Controls: empty and full range; keys
  shuffled after build must refuse.
- **P3** state what each axis of `DR256_SQ` addresses and round-trip one
  identity row through `WideFieldMask` bit for bit.
- **P4** same-count field reorder on the hot check: passes today (red
  first); the fix must refuse it, cost at most 5 ns per check, and leave an
  unchanged view admitted.
- **P5** restart replay: a binding recorded in one process is checked in
  another. Kill: a different binding carrying the same number is admitted.
  Oracle: field-by-field equality, never the digest.
- **P6** flat 1,024 words vs a sorted u16 array below 4,096 set bits, AND +
  popcount. Kill the sparse form unless 2× faster somewhere.
- **P9** the two refusal axes stay independent: a frame advance reports only
  a frame refusal; a class change reports only a class refusal.
- P7 (select), P8 (per-block summary), P10 (footprint reverse index): only
  if the parked idea is taken up.
- P2 (a compile-fail wall between a row population and a property mask) is
  **withdrawn**: it tests the box. Its replacement: a mask bound in one
  coordinate space is refused in another (`LaneRefusal::CoordinateSpace`
  already does this; keep it as the test).

## 5. Open points for the operator

1. Do D-LANESOCK-0's "never a row-survivor bit" and plan v3's "three
   distinct typed concepts" stand as written, or are they superseded by §1?
   The split plan's S2 depends on the answer.
2. `WideFieldMask` holds at most 256 positions (`u8`). A slab of 65,536 is
   256 of them. Is the slab carried as 256 masks, or does the mask itself
   grow past 256? (The cap is stated as intended at `class_view.rs:642-646`.)
3. The lab's index-ordered visibility plane: second copy, or the one copy
   in that order?

## 6. Not searched

OWL / DOLCE / schema.org (no term offered). Chan & Ioannidis 1999 (interval
encoding; scanned PDF, unread). FeatureBase bit-sliced fields (primary pages
unreachable; secondhand only). DuckDB validity masks. CLOS MOP. Object
models other than Smalltalk. Wavelet-tree range counting (deferred by its
paper). Nothing was built or measured in this run.

## Closeout

`PR pending | STATUS: open | OUTCOME: exploration map for D-DRDT-SPLIT-1; 3 adopt-now (docs/test), 4 probes, 3 parked, 4 skipped | OPEN: §5 points 1–3`

MIRROR
- BLIND SPOT: every role, the premise gate included, took the mask type's
  doc comment as its definition and did not weigh the operator's D15 wording
  against it.
- STILL OPEN: whether one carrier for every axis costs anything at the 256
  cap.
