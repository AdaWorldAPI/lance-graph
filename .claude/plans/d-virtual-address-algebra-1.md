# Virtual address algebra: one address law shared by order and geometry (lab plan)

**Status:** PLAN (clean recovery). Pass conditions fixed 2026-11-13, before any candidate of THIS branch ran. Lab only.
**D-ids:** D-VIRTUAL-ADDRESS-ALGEBRA-1 (`STATUS_BOARD.md`).
**Crate:** `crates/virtual-address-lab` (new workspace member). No production crate changes.
**Follows:** D-DR-SELECT-K (#2033, #2035, #2039, #2040), D-QUACK-BOUNDARY-LAB (#2045), D-WX-DR256-TERNLOG (#2055), D-QUACK-DERIVED-PLANE-PROBE (#2058).

> **Correction of the first attempt.** Branch `ccr-8062bcc7-l7lkz9` (kept as
> evidence, not merged) carried a plan that said no indexed reorder exists.
> That was a name search, and it is wrong: the path is
> `OrderIndex` / `SealedOrder` → `SelectionSource` → `LaneBinding::selected`
> → `fold_bound_selection` → `execute_sel` → `ndarray::simd::sum_i32_at`.
> That attempt also built its own `Permutation`, `MaskView` and a lane
> `Vec`-holding table beside that path. None of the three is ported. Its
> recipe algebra, materialising oracle and 256^3 module are, after review.

## 1. Two transformations, never exchanged

| | A. ordered row selection | B. geometric address recipe |
|---|---|---|
| answers | WHICH source-local rows are visited, in what order | HOW a requested view cell `(row, property)` resolves to `(field, position)` |
| carrier | a version-bound position list issued by the source | three bits; nothing is stored |
| existing code | see the seam map | none; this lab adds the recipe only |
| can change the field axis | no | yes (`swap`) |

`Rotate90(r, p) = (p, N-1-r)`: view cell `(r, p)` reads lane `p` at position
`N-1-r`. For lanes `[1 2 3] [4 5 6] [7 8 9]` the view is
`3 6 9 / 2 5 8 / 1 4 7`. A reversal of the row index alone is not this map.

## 2. Seam map (read at `main` `630ad6a34`)

| edge | state | where |
|---|---|---|
| Lance produces an order or an exact selection | OPEN | `.claude/probes/2026-11-12-dr-select-k-1/REPORT.md` "Scope limits"; `lance-graph-dt64-lance` persists the `MapTable` only |
| order as a version-bound map, lanes not moved | LAB | `OrderIndex` (`crates/dr256-ternlog-lab/src/lib.rs`), `SealedOrder` (`src/order_store.rs`); borrowed positions through `SealedOrder::rows` |
| frame, revisions, global to local | SHIPPED | `FrameStore`, `Frame`, `MapTable` in `crates/lance-graph-dt64/src/dr.rs` |
| `NodeGuid` to row | OPEN | `dr-select-k-v1.md` §2 |
| positions and lane under one check | SHIPPED | `SelectionSource`, `SelAddr`, `LaneBinding::selected` in `crates/lance-graph-dt64/src/lane.rs` |
| a production `SelectionSource` | OPEN | the only implementors are fixtures under `crates/lance-graph-report/{tests,examples}` |
| the boundary before the fold | SHIPPED | `fold_bound_selection` in `crates/lance-graph-report/src/bound_sel.rs` |
| Quack | SHIPPED, two `i32` fold shapes | `lower_sel` in `crates/lance-graph-quack/src/lib.rs` |
| mask-RISC | SHIPPED | `execute_sel`, `SelFold` in `crates/lance-graph-mask-risc/src/sel.rs` |
| ndarray | SHIPPED, a plain loop | `sum_i32_at`, `count_gt_i32_at` in `ndarray/src/simd_masking_ops.rs` |
| `Scope` as a grant | BLOCKED | `LaneBinding::selected` doc: a consistency check |
| an order witness outside the lab | BLOCKED | `dr-select-k-v1.md` §15, item 2 |

`fold_bound_selection` builds its `SelFold` itself; it does not call
`lower_sel`. Both edges exist and no call path joins them.

## 3. What the lab adds, and what it does not

Adds: `Recipe` (three bits, composition, inverse, normalisation), a
materialising oracle, the 256^3 module, and a fixture source that implements
the EXISTING `LaneSource` and `SelectionSource`. The fixture's ordered
selection is a `SealedOrder` built by `OrderIndex::by_i32` and lent through
`SealedOrder::rows`; nothing is copied out of it.

Does not add: a position or permutation owner, a mask view, a map table, a
frame store, an executor, an operand form, a ClassId, an opcode.

The smallest integration: a recipe chain is normalised to one recipe BEFORE
`LaneBinding::selected` is called, and only chooses the `field` argument of
that call. If a view property is not one field, the lab reports the missing
operand and runs nothing.

## 4. Pass conditions (fixed before running)

| id | condition | FAILS IF |
|---|---|---|
| P0 | the eight recipes compose as their addresses do | a cell of the 8 x 8 table differs from pointwise composition for N = 1..8; an inverse is missing; `Rotate90 then Rotate270`, `Rotate180` twice, either mirror twice is not `Identity` |
| P1 | a chain that normalises to `Identity` reaches the fold with the same `field` and `sel` as no chain, allocates nothing and moves nothing | another field, another selection, an allocation, or a different answer. The materialising oracle must show the same chain moving data |
| P2 | a row object holds a recipe and a row number; a demanded property is one resolution and one value read | a value read at construction; more reads than demands; another lane's value |
| P3 | a source-issued selection (an order from `OrderIndex` / `SealedOrder`, a filter, a bag) reaches `execute_sel` through `LaneBinding::selected` and `fold_bound_selection`, and equals a scalar oracle | the positions are copied on the way (pointer inequality), or the answer differs |
| P4 | a row-order change cannot stand in for a field-axis rotation | some permutation of the rows of the lane matrix equals its `Rotate90` view on a fixture with distinct cells; or a position list over one field yields a `Rotate90` view row |
| P5 | the coupled relation `x + y + z ≡ 0 (mod 256)` is counted exactly, unrestricted (65,536) and under axis masks, with no allocation | unequal to the enumeration of 16,777,216 addresses, or any allocation |

Negative controls, each a test that must be able to fail:

1. `Rotate90` differs from row reversal alone.
2. Reordering one of two aligned lanes changes a two-lane answer.
3. A selection with a duplicate is a bag: its sum differs from the set's.
4. A selection in another coordinate space is refused (`SelectionSpace`).
5. A frame moved after binding is refused.
6. RECORDED, not required to refuse: a lane rewritten under an unmoved frame
   is read under an unchanged recipe digest. The digest names a recipe, not
   the producer of bytes.
7. RECORDED: `Scope` equality is equality of a plain value; nothing in the
   path observes a grant.

Kill condition: if P3 needs a new owner of positions, or P1 needs a new
executor or operand, stop and report the missing seam.

## 5. Ledger (measured per route)

Allocated address descriptors; copied value bytes; constructed position-list
bytes; constructed mask bytes; value reads; gathers (positions folded);
physical transpositions; warm allocations; admission latency; fold latency.
Cold terms are reported in µs and are not gates.

## 6. Scope

No Lance path is claimed: the source is a fixture. `lower_sel` admits two
fold shapes and is not called a virtual-property executor.

## 7. Architecture correction, 2026-11-13: the 512-byte unit (implementation STOPPED)

The lab committed so far (`82adffdc6`, `d60d89d22`) models 32 separately
allocated `i32` lanes. That is superseded: the unit is one source-owned
512-byte record. Nothing below was run; it is what the code says.

Read in full: `ndarray/src/simd_stride.rs`; `ndarray/src/simd_masking_ops.rs`
lines 185-274, 995-1125, 5040-5340; `lance-graph-contract/src/canonical_node.rs`
lines 730-1454 and 1820-1951; `lance-graph-mask-risc/src/ir.rs` lines 1-125 and
170-329; `exec.rs` lines 505-554 and 660-752; `sel.rs`;
`lance-graph-report/src/bound_sel.rs`. Not read: PR #283 and #382 bodies, the
per-backend ternlog realization, `facet.rs`, DT64 `lib.rs` (searched for
"presence" only: no per-instance presence type found, which is not proof of
absence).

### 7.1 Physical geometry, as coded

| claim | state | where |
|---|---|---|
| A record is 512 bytes, contiguous, `repr(C, align(64))` | VERIFIED-IN-CODE | `NodeRow`, const asserts, `canonical_node.rs:837-848` |
| `&[NodeRow]` and `&[u8]` are the same bytes both ways | VERIFIED-IN-CODE | `NodeRowPacket::as_le_bytes`, `node_rows_from_le_bytes` (needs a multiple of 512 and 64-byte alignment) |
| A Lance `FixedSizeBinary(512)` buffer satisfies that | STATED in a doc comment, not traced | `canonical_node.rs:1914-1923` |
| The record is 32 facets of 16 bytes | **NOT the coded layout** | `GUIDS_PER_NODE = 32` is a capacity ("up to 32 GUID-sized entries") |
| `addr = base + 512 * row + 16 * slot` | holds for `key` (0) and `edges` (16) only | below |

The coded carve is `key(16) | edges(16) | value(480)`, and the value slab is 21
tenants of mixed width (`VALUE_TENANTS`, contiguous, compile-asserted):
8, 8, 32, 32, 6, 16, 4, 4, 2, 8, 12, 12, 12, 16, 16, 32, 16, 16, 16, 16, 16
bytes at row offsets 32 to 332; 332 to 512 is reserved. Eight tenants are 16
bytes wide, at row offsets 118, 188, 204, 252, 268, 284, 300 and 316. None of
those is a multiple of 16. The address law the code supports is therefore

```
addr(row, tenant) = base + 512 * row + VALUE_TENANTS[tenant].row_offset
```

with the offset read from the table, never computed as `16 * slot`. The strided
kernels load with `from_le_bytes` over byte slices, so an unaligned 16-byte
field is legal.

Presence, as coded, is per CLASS: `ValueSchema` is a `FieldMask` over tenants,
chosen through `ClassView::value_schema`. An all-zero tenant reads as
unasserted (zero-fallback). A per-instance presence plane was not found.

### 7.2 Execution seams for a 16-byte field

| step | state |
|---|---|
| A field view over unchanged record bytes: `LaneRef::Strided(StridedRef { bytes, first_offset, stride, records })` | SHIPPED; this is the address law above, with `stride = 512` |
| 16-byte care-masked match to a mask: `Pred::MatchFacet16Strided`, `ternary_match_strided16_to_mask` | SHIPPED; field loads are scalar by design, the compare is `U64x8` ternlog |
| Sum over a strided register: `Terminal::MaskedStridedGroupSum` | SHIPPED; unsigned groups of 1 to 4 bytes, scalar, takes a MASK |
| A strided view over a subset of rows | NOT EXPRESSIBLE: `records` must equal `n_rows`, and the stride is regular |
| A position-list fold over a strided field: `execute_sel` | **MISSING**: `SelFold` is `SumI32` and `CountGtI32` over an `i32` lane |
| A source returning `LaneRef::Strided` through `LaneBinding::selected` | type-admissible (`Lane<'s> = LaneRef<'s>`); the fold then refuses the lane kind |
| A general 512-byte, 32-facet fold in `simd_stride.rs` | ABSENT: the module holds `lut_u8` and `lut9` |
| "1 to 4 cycles" for the register-local operation | UNVERIFIED: no measurement read; both strided kernels document themselves as memory-bound at this stride |

So a source-issued selection reaches a 16-byte facet today only as MEMBERSHIP
(a mask over all rows of the record store), which drops order and
multiplicity. The exact missing operand seam is a strided lane kind in
`execute_sel`. It is recorded here and not built.

### 7.3 Decision, 2026-11-13

DECISION: the experiment measures the coded `NodeRow` layout (A). The uniform
32 x 16 unit is at most a later synthetic counter-experiment.
SCOPE: two separate proofs, never merged by repacking. A1 is the ABI512
address law over `NodeRow` / `NodeRowPacket`. A2 is ownership: the consumer
works on lawfully borrowed values of the real owner, `MailboxSoA<N>`.
BASIS: section 7.1. No alignment follows from the field width: the eight
16-byte value tenants start at offsets that are not multiples of 16.
REVISIT WHEN: a source owns a uniform 16-byte-slot record.

## 8. ABI512 x MailboxSoA ownership x ISA seam map (nothing run)

Read in full for this section: `soa_view.rs`; `soa_envelope.rs` lines 1-290;
`mailbox_soa.rs` lines 1-985 (everything but its tests);
`canonical_node.rs` lines 1-456, 476-2133 (everything but its tests);
`lance-graph-report/src/lane_source.rs`; `symbiont/src/bridge.rs` lines
60-169; the `[dependencies]` of `cognitive-shader-driver/Cargo.toml`.
Not read: the test modules of those files, `surreal_container/src/view.rs`,
`facet.rs`, DT64 `lib.rs`. Rows marked "searched" rest on a search over
`crates/**/*.rs` and name candidates only.

### 8.1 The four things, kept apart

| | `NodeRow` | `NodeRowPacket` | `MailboxSoA<N>` | `MailboxSoaView` |
|---|---|---|---|---|
| what it is | 512-byte record ABI | borrowed packet over an existing `&[NodeRow]` | columnar hot store | read-only trait |
| layout | `key(16) edges(16) value(480)`, `repr(C, align(64))` (`canonical_node.rs:837-848`) | the same bytes, stride 512 (`:1860-1899`) | one array per field (`mailbox_soa.rs:58-230`) | borrows `&[f32]`, `&[u64]`, `&[u32]`, `&[u16]` (`soa_view.rs:82-89`) |
| owner of the bytes | whoever holds the `[NodeRow]` | none; not `Clone`, not `Copy` (`:1812-1831`) | the mailbox, by move | the implementor |
| who mutates | the holder (`set_kanban`, `set_style_lane`: "owner-only by convention", `:2025`, `:2092`) | nobody | `&mut self` methods only | nobody; `MailboxSoaOwner` adds `advance_phase` only (`soa_view.rs:295-301`) |
| version stamp | none in the type | `cycle`, whatever the caller passed (`:1849`) | `current_cycle`, `last_write_cycle[row]` (`:86`, `:179`) | `current_cycle()` |

### 8.2 The ten questions

| # | question | ABI512 (`NodeRow` slab) | `MailboxSoA<N>` |
|---|---|---|---|
| 1 | layout, alignment | 512 B, align 64. SOURCE VERIFIED | struct of arrays; three `Box<[u64]>` planes of `N * 256` words. SOURCE VERIFIED |
| 2 | bytes per element | 21 tenants, 2 to 32 B; eight are 16 B at row offsets 118, 188, 204, 252, 268, 284, 300, 316 (`:1129-1310`) | 4, 1, 4, 4, 8, 8, 4, 2, 8, 2, 1 B per row; 2048 B per identity plane; 12 B per style lane. **No 16-byte field. No `NodeGuid` key column. No edge-block column.** SOURCE VERIFIED |
| 3 | owner | no long-lived owner found: `symbiont/bridge.rs:122-129` and `soa-view-probe` build a local `Vec<NodeRow>` (searched). NOT PROVEN that a production owner exists | the mailbox. SOURCE VERIFIED |
| 4 | who may mutate | the holder of `&mut [NodeRow]`; convention, not a type boundary | `&mut self`: `apply_edges`, `consume_firing`, `tick`, `write_row`, `set_*`, `set_populated`, `reset_row`, `set_style_*`, `promote_family`, `advance_phase`. SOURCE VERIFIED |
| 5 | borrow or copy | `as_le_bytes` and `node_rows_from_le_bytes` borrow; `NodeRow::kanban`, `qualia`, `style_lane` copy 8 or 12 bytes out | `energy`, `edges_raw`, `meta_raw`, `entity_type`, `identity_plane_at` borrow; `style_lane_at` returns `[u8; 12]` by value. SOURCE VERIFIED |
| 6 | identity to row | `NodeGuid` is bytes 0..16 of the row; no index from key to row in this file. NOT PROVEN | `row_for_local_key` is not overridden, so it returns `None` (`soa_view.rs:126`). BLOCKED |
| 7 | what invalidates an old read | nothing in the type | `write_row` refuses a write for another cycle (`:424-470`); reads are not gated. No `FrameStore`: the crate has no `lance-graph-dt64` dependency. SOURCE VERIFIED |
| 8 | property to location | `ValueTenant` -> `VALUE_TENANTS[t].row_offset`; presence per class via `ValueSchema` (`:1354-1437`) | fixed struct fields; `class_id()` aliases `entity_type` (`soa_view.rs:100`); no ClassView mapping in the store |
| 9 | ISA kernels in place | `LaneRef::Strided` at stride 512: `EqU32Strided`, `MatchFacetStrided`, `MatchFacet16Strided`, `MaskedStridedGroupSum`. SOURCE VERIFIED | `meta_raw` is a `&[u32]` and `edges_raw` a `&[u64]`, the shapes of `LaneRef::U32` / `U64`. `f32` energy and `u16` entity type have no `LaneRef` kind. NOT PROVEN that any kernel is wired to a mailbox |
| 10 | needs staging, copy or gather | a fold over a position list (`execute_sel` is `i32` only); a strided view of a row subset | a 16-byte facet (none exists); the `[u8; 12]` lanes are returned by value |

### 8.3 Bridges that do not exist in the code read

- `MailboxSoA<N>` does not implement `SoaEnvelope`; the only non-test
  implementor is `NodeRowPacket` (searched). Its module doc says
  "`SoaEnvelope` describes the geometry" (`mailbox_soa.rs:22-23`); no code in
  the file does.
- No conversion between `MailboxSoA<N>` and `NodeRow` in either file. The
  field sets differ: the mailbox has no key, no edge block and none of the
  16-byte tenants, and its `meta` is 4 B where the `Meta` tenant is 8 B.
- No mailbox mutation advances a DR frame. `LaneSource`, `SelectionSource`
  and `FrameStore` are not reachable from `cognitive-shader-driver`.
- The only `LaneSource` outside labs, tests and examples is
  `lance-graph-report`'s `AbiLanes` (searched). Its sole constructor is
  `#[cfg(test)]`, and it serves `I32` and `U32` lanes of an `AbiBatch`,
  neither a mailbox nor a `NodeRow` slab.

### 8.4 Stop

A2 as specified ("one selected identity and one ClassView property resolve to
one 16-byte physical facet of the mailbox") **cannot be demonstrated**: the
mailbox holds no 16-byte field, resolves no identity to a row, and has no
frame. Per the stop condition nothing is built.

What the code does support, as two separate and smaller statements:

- A1: the address law of section 7.1 over a `[NodeRow]` held by a lab owner,
  through `NodeRowPacket` and `LaneRef::Strided`. The owner would be the
  lab's; no production owner of a `NodeRow` slab was found.
- A2': a typed view the mailbox really lends (`meta_raw`, `edges_raw`) read in
  place by position. Putting it behind `LaneBinding` needs a `LaneSource` and
  a `FrameStore` for the mailbox, which do not exist and are not to be
  invented here.

### 7.4 Superseded text
### 7.3 Open decision before any rebuild

The brief's unit (32 uniform 16-byte facets, `16 * slot`) and the coded
`NodeRow` (mixed tenants, table offsets, ten 16-byte fields including key and
edges) are different layouts. The fixture is not rebuilt until it is decided
which one the experiment measures.

## 9. ABI512 correction, 2026-11-13: physical facets are not semantic formats

Sections 7.1 and 8.2 read the 16-byte-wide entries of `VALUE_TENANTS` as if
they were the physical unit and concluded "not 32 facets" and "no alignment".
That mixed two things. Corrected model, with what the code does and does not
say about it:

- **Physical:** one ABI512 unit is 32 facets of 16 bytes; facet `k` is bytes
  `16k..16k+16` of the record. Facet 0 is the `NodeGuid`. Coded:
  `GUIDS_PER_NODE = 32`, const-asserted (`canonical_node.rs:918-923`).
- **Semantic:** a field is a byte-aligned run that may be part of a facet or
  cross facets. Sub-byte values are packed inside bytes: 24 x i4 = 96 bits =
  12 bytes, two nibbles per byte, low nibble first
  (`causal_witness.rs:270-282`).
- **Recipe:** which bytes and nibbles a class reads is the ClassView's.

Additionally read in full for this section: `facet.rs` lines 1-1004,
`register128.rs` lines 1-435, `causal_witness.rs` lines 1-456 (each up to its
test module). Not read: their tests, `atoms.rs`, `hotplug.rs`,
`tekamolo_facet.rs`, `moore_tenant.rs`, `.claude/v3/soa_layout/*.md`.

### 9.1 Difference between the coded carve and the facet model (reported, not resolved)

The coded `VALUE_TENANTS` carve is contiguous in tenant order and was not laid
out on 16-byte facet boundaries. Per tenant, the physical facets it covers
(`facet@byte`):

| tenant | row bytes | first .. last | in one facet |
|---|---|---|---|
| key (`NodeGuid`) | 0..16 | 0@0 .. 0@15 | yes |
| edges | 16..32 | 1@0 .. 1@15 | yes |
| Meta | 32..40 | 2@0 .. 2@7 | yes |
| Qualia | 40..48 | 2@8 .. 2@15 | yes |
| MaterializedEdges | 48..80 | 3@0 .. 4@15 | two whole facets |
| Fingerprint | 80..112 | 5@0 .. 6@15 | two whole facets |
| HelixResidue | 112..118 | 7@0 .. 7@5 | yes |
| TurbovecResidue | 118..134 | 7@6 .. 8@5 | no |
| Energy, Plasticity, EntityType | 134..144 | 8@6 .. 8@15 | yes |
| Kanban | 144..152 | 9@0 .. 9@7 | yes |
| FrozenStyle | 152..164 | 9@8 .. 10@3 | no |
| LearnedStyle | 164..176 | 10@4 .. 10@15 | yes |
| ExploreStyle | 176..188 | 11@0 .. 11@11 | yes |
| Tekamolo (4 + 12) | 188..204 | 11@12 .. 12@11 | no; its 12-byte payload is 12@0..12@11 |
| CausalWitness (4 + 12) | 204..220 | 12@12 .. 13@11 | no; its 12-byte payload is 13@0..13@11 |
| EpisodicBasin | 220..252 | 13@12 .. 15@11 | no |
| Register0 | 252..268 | 15@12 .. 16@11 | no |
| Register1 | 268..284 | 16@12 .. 17@11 | no |
| Nars16x8 | 284..300 | 17@12 .. 18@11 | no |
| MoorePalettePairs | 300..316 | 18@12 .. 19@11 | no |
| MooreNars16 | 316..332 | 19@12 .. 20@11 | no |

So the code as it stands does NOT implement "one 16-byte tenant = one physical
facet": every tenant from `Tekamolo` on starts at byte 12 of a facet, and a
classid-prefixed 4 + 12 lane has its classid in one facet and its payload at
the start of the next. Nothing is misaligned at the byte level, and no
storage boundary falls inside a byte. Whether the carve is to be re-based
onto facet boundaries is a layout decision (`ENVELOPE_LAYOUT_VERSION`), not
taken here.

### 9.2 Facet -> interpretation -> owner -> ISA operation

| physical bytes | class-bound reading (where coded) | owner today | ISA operation that reads it in place |
|---|---|---|---|
| facet 0 | `NodeGuid`: `classid(4)` + 6 x `(u8:u8)`; `FacetCascade` lens, `CascadeShape` 6x2 / 4x3 / 3x4 (`facet.rs:94-100`, `:801-815`) | holder of the `[NodeRow]`; `MailboxSoA` has no key column | `EqU32Strided` (classid), `MatchFacetStrided` (12-byte payload at +4), `MatchFacet16Strided` |
| facet 1 | second content-blind facet, read through `EdgeCodecFlavor` (`canonical_node.rs:646-708`) | holder of the `[NodeRow]`; none in `MailboxSoA` | same three |
| 12 bytes at row 208 (facet 13, bytes 0..12) | `CausalWitnessFacet`: 24 signed `i4`, slot `s` = nibble `s & 1` of byte `s / 2` (`causal_witness.rs:270-282`). EXPERIMENTAL reading | holder of the `[NodeRow]` | `MatchFacetStrided` with a nibble care (`0x0F` / `0xF0`) -> mask -> `Count`; no fold SUMS nibbles |
| 16 bytes at row 252 / 268 | `Register128`: four `u32`, or `I4x32` / `I8x16` under a `RegisterLaw` (`register128.rs:54-92`, `:188-236`) | holder of the `[NodeRow]`; the contract accessors COPY 16 bytes out (`:159-164`, `:358-368`) | `MatchFacet16Strided`; `MaskedStridedGroupSum` as 4 x `u32` (unsigned only) |
| 12-byte style lanes | 12 x `u8` palette atoms or 6 x `(u8:u8)` rails (`soa_view.rs:181-251`); the 24 x i4 reading is "the third ... interpretation of the identical storage" (`causal_witness.rs:36-44`) | **`MailboxSoA<N>`**: `frozen_style`, `learned_style`, `explore_style` are `[[u8; 12]; N]`, a contiguous 12-byte stride the mailbox owns (`mailbox_soa.rs:170-176`) | `MatchFacetStrided` at stride 12 over the lane's own bytes |
| `meta`, `edges` columns | `MetaWord` (`u32`), `CausalEdge64` (`u64`) | `MailboxSoA<N>`, lent as `&[u32]` / `&[u64]` (`mailbox_soa.rs:925-950`) | the contiguous `U32` / `U64` predicates |

Not available for any of them: a fold over a position list (`execute_sel` is
`i32` only), a signed nibble sum, and a DR frame on the mailbox.

### 9.3 Test, written before it is run

`crates/virtual-address-lab/tests/i4x24.rs`, two owners, one reading:

1. **ABI512.** 300 `NodeRow`s held by the test; each row's 24 loci are packed
   by a packer written in the test (not the contract's) into the 12 payload
   bytes of the `CausalWitness` tenant. Checked against a plain `[i8; 24]`
   table kept beside the rows:
   - the payload starts at row byte 208 = facet 13, byte 0, and the 4 + 12
     unit starts at facet 12, byte 12 (fails if the carve moves);
   - `CausalWitnessFacet::from_register_ref` over the packet's own bytes reads
     every nibble, and its address is inside the row slab (a borrow);
   - `MatchFacetStrided` at stride 512 with a nibble care counts the rows
     where one locus has one value, for a low-nibble and a high-nibble slot
     and a negative value; the count equals the table's.
2. **Mailbox owner.** A `MailboxSoA<64>` owns its `frozen_style` lane; the
   same 24 loci are written through the owner's `set_style_lane`; the lane's
   own bytes (`as_flattened`, no copy, address checked) are read at stride 12
   by the same predicate; counts equal the table's. The 24 x i4 reading of a
   style lane is a class election this test does not make; it shows the
   mechanics only.

Controls: a care of `0xFF` on the byte (reading the byte as one value) gives a
different count than the nibble care; a row count of 0 matches is rejected as
uninformative.

### 9.4 Result of the 24 x i4 test (2026-11-13, release build)

`cargo test --release -p virtual-address-lab --test i4x24`: 2 passed on the
first run. Four disable runs on the committed test, each restored with
`git checkout`, each red:

| disable | outcome |
|---|---|
| nibble halves swapped in the care | both tests fail |
| strided view one byte past the payload | the `NodeRow` test fails |
| mailbox lane read at stride 11 | the mailbox test fails |
| sign dropped when building the pattern | both tests fail |

What this shows: 24 signed nibbles in 12 bytes are read where they lie, from
a `[NodeRow]` slab (stride 512, offset 208) and from a lane the mailbox owns
(stride 12), by one existing predicate with a nibble care; nothing is
widened, copied or transposed. What it does not show: a DR frame, a DT64
binding, a source-issued selection or a fold of the nibble VALUES (only
equality to a mask and a count). No latency was measured.

## 10. The owner-backed proof (directive of 2026-11-13), written before any run

Question: can one version-bound address recipe let a consumer read exactly the
demanded property from the existing `MailboxSoA` owner, through the existing
DR/DT64 selection contract, without constructing the object or moving data?

Additionally read: `class_view.rs` lines 1-700 and 1030-1210 (the masks and
the `ClassView` trait head; the rest of the file is not read), `sel.rs` in
full, `order_store.rs` lines 60-289.

### 10.1 What exists and what the lab must supply

- The owner lends two columns a `LaneRef` can carry: `meta_raw() -> &[u32]`
  and `edges_raw() -> &[u64]` (`mailbox_soa.rs:925-950`). `energy` (`f32`) and
  `entity_type` (`u16`) have no `LaneRef` kind. No column is `i32`.
- The owner implements neither `LaneSource` nor `SelectionSource`, holds no
  `FrameStore`, and resolves no identity to a row.

So the DR/DT64 contract cannot reach the mailbox without a socket. The lab
supplies the smallest one, modelled on `lance-graph-report`'s `AbiLanes`:
`MailboxLanes<'m, V: MailboxSoaView>` holds **a shared borrow of the owner**
and no value byte. Its frame revisions are read from the owner
(`mailbox_id`, `current_cycle`); it owns one `FrameStore` (the existing type)
and the source-issued selections. Because it borrows the owner, the owner
cannot be mutated while it lives: a mutation ends the socket, and a socket
made afterwards is another frame store.

This socket is the missing production seam, built in the lab only. It is
recorded as such in 10.4 whatever the tests show.

### 10.2 Tests (`tests/mailbox.rs`, `tests/mailbox_alloc.rs`)

Owner: `MailboxSoA<64>`. Class: two fields, `meta` and `edge`. The view is the
entity view (`TRANSPOSE` of 2 lanes x n rows). Oracle: plain arrays kept by
the test.

1. One identity, one property: a source-issued selection of one position;
   `meta` read through `LaneBinding::selected` equals the oracle; the lane's
   address is the mailbox's own `meta` array; the row object is 16 bytes.
2. 64 identities in key order (`OrderIndex::by_key` over `meta`, sealed):
   the values arrive in (key, row) order; the positions are lent, not copied;
   exactly 64 values are read; the `edge` column is not borrowed.
3. Two properties, and one property spanning two regions (`meta` and `edge`
   of the same identity): both lanes read at the same source-issued positions.
4. A bag with duplicates: multiplicity and order preserved.
5. `[T, R90, R270]` resolves like `[T]`; a chain that normalises to Identity
   is reported (`PropertyCrossesFields`), nothing read; `ANTI_TRANSPOSE`
   resolves the other field.
6. Falsifiers: field mapping swapped under the same frame -> refused; owner
   advanced (`tick`) -> the old binding is refused by the new socket; a
   selection in another row order -> refused; a property the class does not
   have -> refused; a position at or past `populated` -> refused, never the
   zero padding of the capacity rows.
7. RECORDED LIMITATION: `set_meta` without `tick` leaves the revisions a new
   socket derives unchanged while the value differs.
8. RECORDED GAP: `fold_bound_selection` over `meta` is refused with
   `ExecError::LaneKind` (the fold admits `i32` only). Expected to hold; if it
   does not, the gap closed and 10.4 is corrected.
9. Warm path allocates nothing (separate binary).

### 10.3 Ledger to report

Owner bytes (`byte_footprint`), descriptor bytes, values copied into
intermediate storage, values read, lane borrows per column, physical
transposes, allocations, cold bind and warm read latency (ns). The 1-4 cycle
hypothesis is NOT measured here: no register-local kernel runs on this path,
the read is a position-indexed scalar load.

## 11. Binding directive, 2026-11-13: prove the address law on the real owner

ABI512, `NodeRow`, `VALUE_TENANTS` and `MailboxSoA<N>` stay exactly as coded.

⊘ Section 10 is superseded on one point. Its `MailboxLanes` socket gave the
mailbox a `FrameStore`, a `LaneSource` and a `SelectionSource` of the lab's
making. The directive forbids exactly that substitute, so the socket and its
tests are removed from the tree (they remain in commit `584f880e1`; there 10
tests passed and 6 of 7 disable runs were red, the seventh showing that the
"stale frame" refusal came from the socket's identity, not from the owner).
Nothing of section 10 is evidence about the mailbox's own guarantees.

### 11.1 What is tested now, written before the run

- **P0** `tests/owner_read.rs`, `tests/owner_read_alloc.rs`: one property of
  one source-local row of a real `MailboxSoA<64>`, through `MailboxSoaView`
  only (`meta_raw`, `edges_raw`). The identity is a row POSITION supplied by
  the caller. Checks: value against a scalar oracle; the lent slice is the
  owner's own array inside the owner's allocation; two separate columns, not
  one record; one value loaded per request and the other column not borrowed;
  the chain normalised before the read; absent property and absent row give no
  value; `style_lane_at` is a copy; no allocation.
- **P2** `tests/algebra.rs`: `Rotate90 ∘ Rotate270` and `Rotate180 ∘
  Rotate180` are Identity, by the product and cell by cell.
- **P1 and P3 are not run.** See 11.2.

### 11.2 Outcome A: owner-backed virtual-property read — LAB VERIFIED

Release build, 2026-11-13. `cargo test --release -p virtual-address-lab`:
29 tests in 8 binaries pass (P0: 5 + 1; algebra: 4). Six disable runs on the
committed code, each restored with `git checkout`, each red, none a compile
error:

| disable | tests that fail |
|---|---|
| the row is ignored (always row 0) | value, chain, absent-row, alloc |
| the populated-row guard is removed | absent-row |
| the chain is not normalised (first recipe only) | chain |
| the predicate maps to the wrong column | value, chain, absent-row, alloc |
| both columns are borrowed on every read | value (borrow count) |
| the composition law is broken | three algebra tests, chain, alloc |

What holds for a real `MailboxSoA<64>` through `MailboxSoaView`:

- the value equals the scalar oracle;
- `meta_raw()` and `edges_raw()` have the address of the owner's own `meta`
  and `edges` arrays, inside the owner's allocation; they are two columns of
  4-byte and 8-byte stride, not one record;
- one request loads one value and borrows one column;
- a chain ending in `Rotate90, Rotate270` or `Rotate180, Rotate180` reads what
  the direct `meta_raw()[row]` reads; a chain that is Identity overall is
  reported and reads nothing;
- an absent property and a row at or past `populated` give no value (the
  owner's array spans capacity and holds zero there);
- no allocation; no `NodeRow`, row object or second buffer exists on the path.

The lifetime of the borrow is the compiler's: the slice borrows the owner, so
the owner cannot be mutated while it is held. That is not a test.

### 11.3 Outcome B: DR/DT64 version-bound admission — BLOCKED

First missing source-to-consumer boundary, in order:

1. **Identity.** `MailboxSoA` does not override `row_for_local_key`; it
   answers `None` for every key (`soa_view.rs:126`; tested). P0 therefore uses
   a caller-supplied row position, which is not an identity.
2. **Lane socket.** `MailboxSoA` does not implement `LaneSource`, and
   `cognitive-shader-driver` does not depend on `lance-graph-dt64`.
3. **Frame.** The owner holds no `FrameStore`. Its stamps are `current_cycle`
   and `last_write_cycle`; `set_meta` moves neither (tested: the row's
   `last_write_cycle` stays `u32::MAX`).
4. **Selection.** The owner implements no `SelectionSource` and has no home
   for an order index.
5. **Fold.** Were 2 to 4 present, `execute_sel` admits `i32` lanes only and
   the owner lends `u32`, `u64`, `f32`, `u16` and `[u8; 12]`.

P3 (64 selected identities) was not run, and no substitute was built.

Negative controls:

| control | state |
|---|---|
| changed property-to-physical mapping | NOT TESTABLE on the owner: no binding exists that could go stale. The lab's own mapping is disable-tested (row 4 above) |
| stale source frame | NOT TESTABLE: the owner has no frame |
| changed source row order | NOT TESTABLE: the owner declares no coordinate space |
| duplicated selection positions | NOT TESTABLE as source-issued: the owner issues no selection |
| absent property / row | TESTED |
| inverse-recipe cancellation | TESTED |
| owner mutation without frame advancement | TESTED AS A LIMITATION: the write changes the value and no stamp |

### 11.4 Ledger (`examples/owner_read_cost.rs`, two runs, one VM)

| metric | value |
|---|---|
| source-owned bytes | 398,544 (`byte_footprint`, `MailboxSoA<64>`) |
| virtual descriptor | 32 bytes: row, property index, borrowed recipe chain |
| values copied into intermediate storage | 0 |
| demanded reads | 1 value, 1 column borrow per request |
| facet / register loads | none: one scalar indexed load of 4 or 8 bytes |
| physical rearrangements | 0 |
| cold cost | none: no binding is made |
| warm cost | 7.47 and 7.84 ns per read with a 3-recipe chain (median of 21); the direct `meta_raw()[row]` is 0.44 and 0.45 ns |
| warm allocations | 0 |
| admission | BLOCKED (11.3) |

The 7 ns are the chain normalised on every read. Normalising once per
binding is the obvious remedy and was not measured. The 1-4 cycle figure is
not measured and no ISA kernel runs on this path.

## 12. Audit correction, 2026-11-13 (starting point `435ef10b6`)

### 12.1 The defect

`VirtualProperty::read` resolved only the FIELD of the normalised recipe and
then read `self.row` unchanged. For `ANTI_TRANSPOSE` over 2 fields x 64 rows,
`resolve(2, 64, 12, 0) = (1, 51)`, and the code read `edges_raw()[12]`. The
test expected `edge_of(12)`, so it confirmed the wrong coordinate. Section
11.2's sentence "the reversed property axis is the other column" was true of
the column and false of the row. The same shortcut is in `view::field_in`'s
other caller, the superseded `Grid` fold, where it is harmless only because
the folds there are order-free sums over a selection the fixture issues.

### 12.2 Correction, written before the run

- A recipe that changes source-local row coordinates (`flip_pos` after
  normalisation: `ROTATE90`, `ANTI_TRANSPOSE`) is REFUSED in P0
  (`Refused::RowCoordinatesChanged`). The owner has issued no coordinate or
  selection witness, so nothing here may decide that position `r` now means
  position `len - 1 - r`. `self.row` is never rewritten.
- Property-axis-only recipes stay admitted: `TRANSPOSE` and `ROTATE270`
  (field order reversed, positions unchanged).
- The refusal is by recipe, not by instance: with an odd row count the middle
  row maps to itself under a flip and is refused all the same.
- Test: an independent coordinate oracle. A matrix whose cell `(lane, pos)`
  holds the number `lane * 1000 + pos` is rearranged by the data-moving
  reference (`physical::apply`, which does not share the recipe bits); the
  cell it shows at `(row, property)` IS the coordinate the recipe denotes.
  For all eight recipes, every row and property: an admitted read must return
  the owner's value at exactly that column and position, and that position
  must equal the requested row. Acceptance: no admitted read changes which
  source-local position is read.

### 12.3 Evidence classes, corrected

| claim | class |
|---|---|
| `NodeGuid` is the canonical 16-byte identity (`NodeRow.key`) | SOURCE VERIFIED |
| a source-local row ordinal | an address, not an identity |
| `NodeGuid` -> source-local row, for the mailbox | BLOCKED (`row_for_local_key` answers `None`; a `local_key` is basin-local only after its prefix is resolved, and is not treated as unique here) |
| a scalar read of a demanded column value from the mailbox's own memory | LAB VERIFIED |
| class-bound interpretation | NOT PROVEN: `owner::BASIS` is a lab-local predicate table, not a `ClassView` binding |
| version-bound | NOT PROVEN: `VirtualProperty` carries no source version |
| inverse-recipe cancellation before the read | LAB VERIFIED |
| a recipe that reorders rows, on the owner | REFUSED in P0; needs an owner-issued coordinate witness, which does not exist |
| the complete virtual-address law | PARTIALLY VERIFIED |
| DR/DT64 admission | BLOCKED (11.3) |

Selection semantics kept apart: an `OrderIndex` is a bijective ordering; a
`SelectionSource` may issue a bag with repeated positions. Neither is
exercised against the mailbox.

### 12.4 Results (commit `7de838fc7`, tests committed before the run)

`cargo test --release -p virtual-address-lab --no-fail-fast`: exit 0; 30
tests, 0 failed (`owner_read` 6, `algebra` 4, `alloc` 1, `cube` 4, `i4x24` 2,
`owner_read_alloc` 1, `row_object` 3, `selection` 9). `cargo clippy
--all-targets --no-deps -- -D warnings`: exit 0.

`no_admitted_read_changes_which_source_position_is_read`, over all eight
recipes, every row and both properties, at 64 and at 41 populated rows:

| recipes | outcome | cells at 64 rows | at 41 rows |
|---|---|---|---|
| `TRANSPOSE`, `ROTATE270` | admitted; value equals the owner's column at the oracle's coordinate, and that position is the requested row | 256 | 164 |
| `ROTATE90`, `ANTI_TRANSPOSE` | `RowCoordinatesChanged`; nothing read | 256 | 164 |
| the four that do not exchange the axes | `PropertyCrossesFields`; nothing read | 512 | 328 |

Disable runs, each patch asserted to apply exactly once, each restored with
`git checkout`:

| disable | exit | tests that failed |
|---|---|---|
| the refusal in `VirtualProperty::column` never taken | 101 | the oracle test: "Rotate90 admitted a moved row", position 63 for row 0 |
| `Recipe::reverses_positions` always `false` | 101 | the oracle test, same assertion |
| `reverses_positions` reads `flip_lane` instead of `flip_pos` | 101 | the oracle test and `the_chain_is_normalised_before_the_read` (`ROTATE270` refused) |
| `read` takes `meta_raw()[n_rows - 1 - row]` | 101 | four of six, the oracle test among them |

Acceptance, as tested: no admitted P0 property read changes which
source-local position is read. What this does NOT show: that a source-local
position denotes a particular canonical entity. That needs `NodeGuid` ->
row, which is BLOCKED (12.3), so the statement is about positions only.

## 13. Address function and admission, kept apart (written before the run)

`Read(S, x, f) = S[f(x)]`. The address function `f` is mathematics over
coordinates; whether a read is ALLOWED is a separate decision and is not part
of `f`. Section 12 put the refusal inside `VirtualProperty::column`, which
mixed the two. Behaviour is unchanged; the shape is corrected:

- `VirtualProperty::address(rows) -> Address { column, position }`: pure,
  total over the view, reads nothing, refuses nothing but a cell outside the
  view or a property that is not one field. It answers for position-reversing
  recipes too (`ANTI_TRANSPOSE`, row 12 of 64 -> position 51).
- `VirtualProperty::admit(rows) -> Address`: the P0 gate. With no owner-issued
  coordinate witness it admits only a recipe that leaves positions in place.
- `read` = the owner's column at `admit(..).position`. It reads the position
  the address function computed, not the caller's row; admission is what
  makes the two equal in P0.

Test added: for the four axis-exchanging recipes, every row and property, the
address equals the data-moving oracle's coordinate, including the two recipes
admission refuses, and computing it performs no read. Two inverse recipes
give the address of `TRANSPOSE` alone.

### 13.1 Results (commit `666a4faa4`, test committed before the run)

`cargo test --release -p virtual-address-lab --no-fail-fast`: exit 0; 31
tests, 0 failed (`owner_read` 7). clippy `-D warnings`: exit 0.

`the_address_function_is_the_oracles_coordinate_and_reads_nothing`: 420
cells (four axis-exchanging recipes, 64 and 41 rows, two properties); every
address equals the data-moving oracle's coordinate; 416 of them are for
position-reversing recipes whose position differs from the requested row
(256 + 160); zero values loaded.

Disable runs, each patch asserted to apply once, each restored:

| disable | exit | failed |
|---|---|---|
| the address keeps the caller's row | 101 | the address test |
| admission never refuses | 101 | the acceptance test and the address test |
| the address names the other column | 101 | five of seven |

Stated limits, unchanged: the grid is 2 fields x N rows, so one axis is the
owner's row axis and admission must refuse what moves it. A grid whose both
axes are properties of ONE entity would admit all eight recipes; that is not
built here, because no owner-lent per-row property block was verified.
