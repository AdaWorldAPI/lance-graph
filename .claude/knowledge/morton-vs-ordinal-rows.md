# Morton cells and ordinal rows: two coordinate kinds the register must not mix

> **READ BY:**
> - any session that lays out rows, tiles or planes for a DR projection
>   (`dr::address`, `Morton8x8`, `Width`, `Axis::tile_of`);
> - any session that reads lanes through the register contract
>   (`borrow_all*`, `lance-graph-pg-row`);
> - any session that writes a range or prefix predicate, or a selection;
> - `fold-carrier-scientist`, `v3-envelope-auditor`, `layer-boundary-warden`.
>
> **Status:** VERIFIED-IN-CODE and MEASURED where cited. §5 is PROPOSED and
> waits on a decision.
>
> **Written:** 2026-11-21, after the register-contract work
> (`register-contract-reconciliation.md`, `lance-graph-pg-row`). That work
> changed no DR descriptor. What changed is how a reader reads, and reading
> forced the question of which coordinate a "row" is.

## 1. Two coordinates

| | Ordinal row position | Morton cell code |
|---|---|---|
| What it indexes | a lane element and a plane bit, in one row order (`CoordSpace { population, coordinates }`) | a cell `(x, y)` of one DR tile, the identity × property projection surface |
| Type | `u32` position; `u16` local position in a tile | `Morton8x8` (`u16`: `x` on even bits, `y` on odd) |
| Where in code | `LaneSource`, `SelectionSource`, `PresenceSource`, mask-risc `Planes`, `lance-graph-pg-row` | `dr::address`, `dr::coordinates`, `dr::rotate_yz` (`lance-graph-dt64/src/dr.rs:1309-1334`) |
| Bijective | global ↔ local per frame: `MapTable` / `LocalMap` | `(x, y)` ↔ code within `width`; `rotate_yz` is its own inverse (`tests/dr.rs:131-148`) |
| Read by the register today | every read | none |

VERIFIED-IN-CODE at `87d7b41`:
- `dr::address`, `dr::coordinates` and `dr::rotate_yz` have no caller outside
  `lance-graph-dt64/tests/dr.rs`.
- On the read path (dt64, mask-risc, quack, xlog, report, mailbox-lanes,
  r2il, pg-row), `Morton8x8` is used only by DT64's `dr` module. In
  mask-risc, "morton" appears only in the D-HOLO examples. Crates off the
  register path use Morton codes of their own: perturbation-sim, helix,
  onebrc-probe, symbiont.

## 2. When rows are Morton codes: an address restriction is a row interval

D-HOLO H6 (`plans/d-holo-address-indexed-algebra-v1.md` §H6,
`mask-risc/examples/holo_interval_probe.rs`) laid a dense 256 × 256 tile out
so that **the row is the Morton code**. On that layout:
- a prefix is one row interval;
- an aligned box `[x0, x1) × [y0, y1)` is a union of aligned quadtree
  intervals, so it runs as `Pred::Range`.

MEASURED, avx512f, 2026-10-14, HOLDS:
- With 4 intervals, symbolic evaluation stays flat at 45–84 ns while the box
  area grows 256×.
- Thin boxes multiply the intervals: 321 for 256×1, 641 for 1×256. In 8 of
  the 12 many-interval cells, a value sweep (x and y as lanes, `GeI32` /
  `LtI32`) beats the `Range` unions.
- H6b: Morton intervals are cheapest only on a prefix (0.04–0.05× the dense
  mask per query). They lose 120× on a checkerboard and 400–474× on random
  50 %.

**Lesson:** a Morton row order makes geometric restrictions cheap when they
are prefixes or aligned boxes, and expensive when they are thin or
fragmented. It is a layout choice per projection, not a default.

## 3. When rows are ordinal: an order is attested, never assumed

The rows of a table (zammad-rs, the pg-row gate) are records in storage
order. A key range is not a row interval:
- `Cmp::Range` is minted only by `Filter::prefix_facet` from a validated
  `OrderedLaneWitness` (`lance-graph-quack/src/lib.rs`, D-DIAMOND-1 R2);
- without the witness, the read is a value sweep.

Three consequences for readers:
- A selection is a bag in list order. The socket never sorts or deduplicates
  it (`SelectionSource` doc).
- A kept mask is read ascending in position (`Reader::rows_in`).
- Neither is in Morton order unless the source laid its rows out that way.

## 4. Row order is part of the coordinate, so mixing is refused, not converted

- `CoordSpace.coordinates` is the revision of a population's row order. A
  lane in one row order and a selection in another are refused
  (`LaneRefusal::CoordinateSpace` / `SelectionSpace`).
  - MEASURED: zammad-rs#95 `disable-vrow-swap`.
  - MEASURED: the lab gate's "swapped coordinate"
    (`lance-graph-r2il/examples/vrow_ticket_states_pg_parity.rs`).
- A refold or a logged edge moves rows without changing the frame (#2136,
  #2146). So "same frame" is not "same rows": the row order has to be
  checked too.
- Every plane is a bit per position, LSB-first `u64` words, in the lanes' own
  row order: presence, live, kept and validity.
  - A Morton-laid tile needs its planes laid out in Morton order as well.
  - No read converts between the two orders. A conversion would be a gather,
    which ADMISSION B forbids during prepare.
- `Axis::tile_of(position) = position / width` is ordinal. The local index
  `position % width` is the tile's row coordinate `x`, not a Morton code.
  MEASURED: zammad-rs#96 `tests/cube_bijectivity.rs`, 51 feeds over 5–6
  tiles of 256 in both layouts. The round trip and injectivity both hold.

## 5. Proposed, waiting on a decision: name the row order in the descriptor

Today a row order is only a revision number (`coordinates: u64`). Nothing
says whether the rows are ordinal records or Morton cells of a tile, so a
reader cannot tell a prefix predicate's meaning from the descriptor.

**Proposal.** The DR projection descriptor (or the class's `ClassView`
reading) names its row order: `Ordinal`, or `Morton { width }`.
- A predicate whose meaning depends on the order is refused against the
  other order. Examples: a Morton prefix, an `OrderedLaneWitness` range.
- `ORDER BY` in `lance-graph-pg-row` would read the declared order instead
  of assuming storage order.

**Falsifier:** the same prefix predicate on the same lanes, declared
`Ordinal` and `Morton { 256 }`. It is admitted as a `Range` only under
Morton, and both answers equal a value-sweep oracle.

**Decides:** the DT64 / DR owner. Until then, no reader infers the order.

## 6. Cross-references

- `register-contract-reconciliation.md`: C5 (bijection), C8 (strided rows
  over 512-byte rows, the third physical form), §4 (the pg-row reader).
- `folding-doctrine.md`: fold coordinates rather than materializing them.
