# weather-soa-bake-v1 — Zarr → NodeRow, the missing bake, and the re-homing of the open weather work onto the substrate

> **Status:** PLAN (2026-08-13). Doc-only. No code shipped, no board file written by this
> plan (the orchestrator is the sole writer of `.claude/board/*`; proposed rows are in §11).
> **Does not supersede** `weather-substrate-poc-v2.md` — §9 states part-by-part what survives,
> what is superseded by having the data on the substrate, and what was never built.
>
> **The one-sentence reason this plan exists:** an extended weather R&D arc ran entirely as
> Python over a Zarr file because **the Zarr→`NodeRow` path does not exist and no plan ever
> specified one** — `weather-substrate-poc-v2.md` names `crates/weather-poc`, declares "New
> repositories required: ZERO", and then describes encoder arms with **no bake step at all**.

---

## §0 Corrections to the framing this plan was commissioned under

Stated first, per the workspace rule that a plan which designs around a wrong premise is worse
than one that names it.

**§0.1 — `D-WXA-5` is NOT on `STATUS_BOARD.md`. Neither is any `D-WXA-*` / `D-WXB-*` /
`D-WXC-*` row.** `grep -c "WXA" .claude/board/STATUS_BOARD.md` = **0** (verified this session,
and re-verified by the orchestrator before landing). The D-ids exist only inside
`.claude/plans/weather-substrate-poc-v2.md` §2–§4 and in the `INTEGRATION_PLANS.md` prose entry
dated 2026-08-10. The board-hygiene rule (`CLAUDE.md` § Mandatory Board-Hygiene Rule, row "A
new D-id / deliverable") was **not discharged** for that plan. Consequence: the weather POC's
deliverable ladder has never been visible on the deliverable dashboard, which is a plausible
mechanism for how a whole arc ran past it in Python without anyone tripping over an unmet gate.

**§0.2 — the label "the C2 gate" does not match the artifacts.** In
`weather-substrate-poc-v2.md` §2, `D-WXA-5` is *"the gate"* (PASS: ρ ≥ 0.98 for ≥1 arm **AND**
the shuffled-codebook control FAILS). `C2` is a bar in a *different* plan
(`substrate-comfort-zones-v1.md` §3 — the degenerate-row verification). They are unrelated
bars. This plan uses `D-WXA-5` and never the C2 label.

**§0.3 — `D-WXA-5` as pre-registered is at serious risk of being VACUOUS, and that is the
single most load-bearing thing this plan has to say about it.** `substrate-comfort-zones-v1.md`
§6.4 **measured** that Spearman ρ *saturates* on smooth pressure fields at 256 levels: the
spread between the two real arms was **3×10⁻⁶ … 4.7×10⁻⁵**, and the plan's own C4 bar
*"could not have fired."* `D-WXA-5`'s bar is ρ ≥ 0.98 between code-distance and
field-distance — four orders of magnitude below where real arms actually land. **Every real
arm will pass it.** A bake-off in which every arm passes has measured nothing, which is what
`D-WXA-5`'s own anti-vacuity paragraph says. §5 below re-specifies the bar. This is not a
licence to weaken the gate; it is the `E-VACUOUS-ASSERTION-IS-THE-HOUSE-STYLE-1` rule applied
to the gate itself, using a number already committed to an artifact.

**§0.4 — "32 × (4+12) facets per cell" and "33.2 M facets per timestep" are arithmetically
right and semantically misleading.** `GUIDS_PER_NODE = 32` is const-asserted
(`canonical_node.rs:805-808`, `512 / 16`), and 1,038,240 × 32 = 33,223,680. But: **2 slots are
`key` and `edges`**, and the 480-byte value slab is **not free space** — `VALUE_TENANTS` carves
it *contiguously, gap-free, compile-asserted*. The committed assertion is exact
(`canonical_node.rs:2292-2307`, `value_tenants_contiguous_within_slab`): **the current Full
carve uses 188 of 480 B** (kanban×Rubicon 8 + autopoiesis triangle 3×12 = 36 + TEKAMOLO facet
16 + CausalWitness facet 16), ending at row offset **220**. Discriminant **15 is reserved for
`BoardAggregates`** (`canonical_node.rs:1027-1033`); a new tenant takes 16.

**Free budget for a weather lane today = 480 − 188 = 512 − 220 = 292 bytes** = 18 whole
16-byte facets (+4 spare bytes) = **216 bytes of facet payload**. §2 does the capacity
arithmetic against that number, not against 384.

**§0.5 — the ~8.1 s/timestep figure is a serial extrapolation from ONE measured cycle, and it
is also a category error.** The measurement (`AGENT_LOG.md` 2026-08-05, PROBE-IGNITION-64K)
is: **65,536 real `MailboxSoA<4>` owners**, build 8.4 s, c1 cast 225 ms, **seal+apply 514 ms**,
c2 rest 73 ms. 1,038,240 / 65,536 = 15.84 ⇒ 16 × 514 ms = **8.22 s** — arithmetic `[G]`,
*conclusion* **CONJECTURE**, for three independent reasons:

1. It assumes seal+apply is **linear in owner count**. `D-KIA-A2` — the pre-registered
   parallelism falsifier (median-of-5, ≥2× at ≥4k owners, ±10 % stay-silent twin) — is
   **`Queued` and unbuilt**. Nothing has been measured about scaling.
2. The measured run bought scale on the **owners** axis only (`MailboxSoA<4>`, **one populated
   row per owner**). Weather's shape is the *rows-per-owner* axis, which that probe explicitly
   did not exercise.
3. **A bake is not a mailbox cycle.** Writing 1,038,240 rows into a Lance version is a
   columnar write; seal+apply is the cognition loop *over* rows that already exist. Quoting
   514 ms/cycle as a bake cost conflates two different operations. §4's `D-WXS-11` replaces
   the extrapolation with a measurement rather than refining it.

Everywhere below, the 8.1 s figure appears only as **CONJECTURE (serial extrapolation from one
measured cycle; `D-KIA-A2` unbuilt)**.

**§0.6 — variable-census inputs used by this plan** (from a live `.zmetadata` read of
`1959-2022-6h-1440x721.zarr`, and **required to be committed as a re-runnable probe** —
`D-WXS-1a` — because a census asserted in prose is exactly the chat-only-figure defect this
arc has already catalogued three times): 17 time-varying surface + 7 pressure-level × 13
levels = 91 + 14 static = **122 fields/cell**; time axis 92,044 six-hourly steps. Per
timestep: 1,038,240 × 122 = **126,665,280 values** landing in 1,038,240 × 512 B =
**531,578,880 B = 0.495 GiB** (arithmetic `[G]`).

---

## §1 The key — does HEEL/HIP/TWIG address 721 × 1440 directly?

**Verdict: YES, directly, with three named wrinkles. This is the literal-x/y case OGAR's
cascade doctrine explicitly sanctions, and it is the cleanest instance of prefix-routing in
the workspace.**

### §1.1 The arithmetic

`/home/user/OGAR/CLAUDE.md` § *Tier interpretation — 256×256 CENTROID TILE*: each tier's 16
bits reads as a **256×256 tile** = two axes × one byte, nibble-interleaved; and the rigour
condition is **256 = 4⁴ — each codebook is a 4-level 4-ary centroid hierarchy**, so a byte's
nibbles are ancestry and `is_ancestor_of` = containment. Three tiers × 4 quaternary levels =
**12 quaternary levels per axis** = 4¹² = 16,777,216 addressable positions per axis.

The grid needs 721 lats and 1440 lons. 4⁵ = 1024 ≥ 721; 4⁶ = 4096 ≥ 1440. **The 0.25° global
grid consumes 5–6 of the 12 available quaternary levels per axis** — leaving room for a 0.05°
grid (3600 × 7200) inside the same three tiers without touching the stride.

**And the hierarchy condition holds *exactly*, not approximately.** OGAR's warning is
*"flat k-means-256 breaks this; hierarchical 4⁴ preserves it."* For a literal x/y grid the
"centroid codebook" is the **identity mipmap**: a coarse tile *contains* its fine cells by
construction. There is no codebook to train and nothing to certify — `is_ancestor_of` is exact
containment. That makes the weather grid the **strongest available test case** for the whole
prefix-routing claim, and §4's `D-WXS-2` turns it into a test rather than a remark.

### §1.2 The proposed assignment

| field | content | width used |
|---|---|---|
| `classid` (bytes 0..4) | the weather-cell class (§1.4 — **not minted by this plan**) | u32 |
| `HEEL` (4..6) | `(lat_idx >> 6, lon_idx >> 6)` — the **16° × 16° tile**: lat 0..11, lon 0..22 | 2 bytes, one per axis |
| `HIP` (6..8) | `(lat_idx & 63, lon_idx & 63)` — position within the tile | 2 bytes, 6 bits each |
| `TWIG` (8..10) | **0 — dormant, reserved** (zero-fallback: "not consulted", never "compacted") | — |
| bytes 10..16 | **untouched** — V1-legacy u24 tail; new units mint the V3 facet, never `NodeGuid::new`. The bake calls `mint_for(classid_read_mode(c).tail_variant, …)`. | — |

**The payoff, and it is the headline of this section:** the four hand-picked **16° boxes** the
entire Python arc ran on **are HEEL values**. `substrate_comfort_d_cz_2_7.py`'s `box_index()`
becomes a **key predicate**, and a box read becomes a **HEEL-prefix range scan** — the
"prerender from keys alone, zero value decode" property applied to a physical field. The
65 × 65 = 4225-cell box of the Python is the 64 × 64 HIP space of one HEEL tile (plus the
inclusive edge row/column the probe took).

`TWIG` stays dormant deliberately: it is the reserved home for a **pressure level** if a later
wave decides a (cell, level) is its own node, or for a sub-0.25° refinement. RESERVE, DON'T
RECLAIM — a non-zero mint later wakes it with **zero `ENVELOPE_LAYOUT_VERSION` change**.

### §1.3 The three wrinkles — stated, not designed around

1. **Neither axis tiles evenly at a power-of-two tile size.** 721 = 11 × 64 + 17 (the grid is
   pole-inclusive: 720 + 1), and 1440 / 64 = **22.5**. So the last tile on each axis is
   **ragged** (17 rows; 32 columns). Decision: **accept ragged tiles**, address by index with
   half-open tile ranges, and *never* pad — padding would invent cells that do not exist and
   would silently enter any global statistic. `D-WXS-2`'s bar asserts the ragged tiles
   round-trip exactly.
2. **Longitude is cyclic; Morton prefix ranges are not.** A box crossing the 0°/360° seam is
   **two prefix ranges, not one**. The Python already does `% len(LONS)`; the Rust key API must
   return a **range-set**, and the type must make a single-range answer impossible to write by
   accident. Latitude has no wrap (it has poles — a different, non-cyclic degeneracy: the pole
   *row* is a single physical point repeated 1440 times, which will show as an exact-tie
   plateau in any rank statistic; the Python's tie-averaging `spearman` already handles it,
   and the Rust side must too).
3. **`MailboxId ≠ NiblePath` in code** (`le-contract.md` §5, discrepancy 3: `MailboxId = u32`,
   `NiblePath{path: u64, depth: u8}`, no conversion, no shared trait). The three-tier doc's
   *"MailboxId IS the NiblePath"* is doc-only and awaiting a ruling. **This plan must not lean
   on it.** A weather cell's key is a `NodeGuid`; whether a HEEL tile is also a mailbox id is
   out of scope and is not assumed anywhere below.

### §1.3a ⚠ UNDECLARED DEVIATION from the OGAR cascade doctrine — whole-byte axes, not nibble-interleaved (found 2026-08-13 by the D-WXS-2 worker, during implementation)

§1.2 above assigns **one whole byte per axis** (`HEEL` = byte 4 lat, byte 5 lon;
`HIP` = byte 6 lat, byte 7 lon). OGAR's cascade doctrine — the same passage §1.1
leans on for the 256×256 tile reading — specifies the two axis bytes
**nibble-interleaved**, and names the consequence explicitly: *"the x/y
nibble-interleave = alternating-axis refinement (Morton in centroid space)."*
§1.2 therefore **deviates from the canon it cites, and did not say so.** Recorded
here rather than silently corrected.

**What survives unchanged.** The plan's actual load-bearing claim —
*a 16° box becomes a HEEL-prefix range scan* — holds under **both** layouts:
fixing both HEEL bytes fixes a prefix either way, so a tile-aligned box is one
contiguous byte range regardless. The `is_ancestor_of` = containment argument in
§1.1 also survives: interleaving governs how two axes **share** a tier's 16 bits,
not the per-tier ancestry the identity mipmap supplies.

**What actually differs.** Byte-lexicographic order over whole-byte axes is
**row-major** (lat-major, lon-minor), not locality-preserving. Consequence for
`D-WXS-9` (the ζ stencil): a `lon ± 1` neighbour is adjacent in key order, while
a `lat ± 1` neighbour is 1440 cells away. Under Morton both are near. An
arbitrary (non-tile-aligned) box is also not one contiguous range under either
layout, but Morton bounds it in far fewer ranges.

**Not changed now, deliberately.** Three reasons: the prefix-scan claim holds as
written; which layout is better *for this workload* is a measurable question and
this arc's discipline is measure-don't-assume; and rewriting a spec the moment a
worker surfaces a consequence is how a correction starts building
(`E-ACTOR-IS-NOT-THE-PHASE-PATH-1`'s recorded lesson: *when correcting an
over-built design, the correction must not itself build*). The shipped
`key.rs` implements §1.2 as specified and its five falsifiers are
disable-verified green.

**The decision this owes.** A pre-registered comparison — row-major vs Morton
over the ζ-stencil neighbour read (`D-WXS-9`) and over a non-tile-aligned box
scan — with the metric stated before the run. Until it runs, §1.2's layout is
**a stated deviation, not a ruling**, and anything downstream that assumes Morton
locality is unfounded. Tracked as `D-WXS-2a` on `STATUS_BOARD.md`.

**A second, smaller item from the same worker, resolved by convention and not by
test:** `box_ranges` treats `lon_lo == lon_hi` as *wrap the whole circle*, not as
*empty box*. Neither reading is forced by the spec. It needs a test pinning the
chosen one, or an API that makes the ambiguity unrepresentable.

### §1.3b `D-WXS-2a` — the row-major-vs-Morton bar, PRE-REGISTERED 2026-08-13 (written and committed BEFORE the run)

§1.3a leaves the layout a *stated deviation, not a ruling*. This section is the
bar that resolves it. **Split in two, because only one half is runnable today:**

- **Half A — pure key-space. Runnable now**, needs no ERA5 data, no classid,
  no bake. This is what is pre-registered here.
- **Half B — the ζ stencil under each layout.** Gated on `D-WXS-9`, itself gated
  on the bake and therefore on `D-WXS-0`. Not pre-registered here; it inherits
  this section's metrics when it runs.

**First, a correction to §1.3a's own wording.** §1.3a called the shipped layout
"row-major". That is imprecise. The key orders bytes
`[lat_tile, lon_tile, lat_hip, lon_hip]`, so lexicographic order is
**tile-row-major, then row-major *within* a tile** — a two-level blocked order,
which already has better locality than a flat row-major over 721×1440 would.
Recording this before measuring, so the comparison is against what is actually
shipped rather than against the looser word.

#### The two metrics (both computed over key-order index, not raw bytes)

1. **Range count** — the number of maximal contiguous runs in key order needed
   to cover **exactly** a given box, with **no false positives** (a scan that
   over-reads and filters is a different, weaker thing and does not count).
2. **Neighbour locality** — the median `|key_index(a) − key_index(b)|` over the
   4-neighbourhood (`lat ± 1`, `lon ± 1`, longitude wrapping), across a
   deterministic sample of cells.

#### Arms

| arm | layout |
|---|---|
| **SHIPPED** | `[lat_tile, lon_tile, lat_hip, lon_hip]` — what `key.rs` emits today |
| **MORTON** | the OGAR-canon reading: the two axis bytes of a tier nibble-interleaved |
| **CONTROL-BAD** | a deliberately locality-destroying order (axis bytes byte-reversed, i.e. `lon_hip` most significant) |

#### The bar, with both halves and a kill

- **Primary:** MORTON beats SHIPPED on **both** metrics, over a box set that
  includes tile-aligned, non-tile-aligned, seam-crossing and pole-adjacent
  boxes. "Beats" is stated before the run as: strictly fewer ranges on the
  **median** non-tile-aligned box, **and** strictly smaller median neighbour
  distance.
- **Control that can lose:** **CONTROL-BAD must be worse than both** on both
  metrics. If a deliberately bad order scores like the good ones, the metric is
  not measuring locality and no verdict may be read off it.
- **Stay-silent twin (non-trivial):** on a **tile-aligned** box, SHIPPED and
  MORTON must produce **exactly one range each** — identical. This is §1.2's
  actual load-bearing claim, and it must show **no difference** where the plan
  claims none. A comparison that reports MORTON better *everywhere*, including
  here, is measuring something other than what it says.
- **KILL:** if MORTON does **not** win on both metrics, the deviation is
  **harmless for this workload**, §1.3a downgrades from "stated deviation owing a
  decision" to a recorded note, and `D-WXS-2a` closes without a code change.
  A negative result here is a real result and is the cheaper outcome — it retires
  an open question rather than opening a migration.

**Discipline note.** Half A cannot settle the *whole* question, because the
stencil (half B) is where locality is actually spent. Half A can only show
whether a difference exists **in key space at all**. If half A kills, half B is
moot; if half A confirms, half B still has to run before any migration. Stated
now so a green half A is not later read as a mandate.

### §1.3c `D-WXS-2a` half A — RUN 2026-08-13. **The KILL fires. Closed, no code change.**

Probe: `crates/weather-poc/examples/layout_probe.rs` (+ `--json` output committed
beside it). Bar: §1.3b, committed **before** this run (`d83b4d3e`). Selftest run
first, green — including the check the brief demanded be *worked out rather than
assumed*: a full longitude row under SHIPPED is **23 ranges** (one per longitude
tile), not 1.

#### Measured

| arm | rc median (non-tile-aligned) | neighbour-locality median |
|---|---|---|
| **SHIPPED** | **140.00** | 32.00 |
| **MORTON** | 212.50 | **16.00** |
| CONTROL-BAD | 3100.00 | 15862.00 |

Box set: 20 boxes — 4 tile-aligned, 8 non-tile-aligned, 4 seam-crossing,
4 pole-adjacent. Neighbour sample: deterministic stride 677, n = 1534.

#### The three verdicts, as pre-registered

- **primary — FAIL.** The bar required MORTON to beat SHIPPED on **both**
  metrics. It won one and lost one: **2× better** on neighbour locality
  (16 vs 32) and **~1.5× worse** on range count (212.50 vs 140.00).
- **control — PASS.** CONTROL-BAD is far worse on both (3100 ranges;
  neighbour distance 15862). The metrics therefore measure locality rather
  than nothing — without this the primary result would be unreadable.
- **stay-silent twin — PASS.** Every tile-aligned box is **exactly one range
  under both** arms (`[1,1,1,1]` each). §1.2's load-bearing claim — *a 16° box
  is a HEEL-prefix scan* — holds, and shows **no difference where the plan
  claims none**.

#### Consequence (the pre-registered one, unchanged)

**`D-WXS-2a` closes. No code change. §1.3a's layout stands as shipped.** The
migration is not opened.

#### ⚠ A correction to §1.3b's own wording, not a softening of the verdict

§1.3b phrased the KILL as *"the deviation is **harmless for this workload**"*.
The data shows that phrasing was imprecise, and it is corrected here rather
than left to read as more than was measured:

**MORTON wins, 2×, on exactly the metric half B would care about.** Neighbour
locality is what a ζ stencil spends; range count is what a box scan spends. So
the honest statement is **"no unambiguous win in key space"**, not "harmless".

The verdict is untouched: the bar said *both*, it got *one*, and one is a FAIL.
Re-reading a split as a win is the failure this arc has a rule against
(*"re-normalising a metric until it stops showing the confound is how a
confound becomes a finding"*). What changes is only my own summary sentence,
which claimed more than the measurement.

**The prior this leaves for half B**, if the ζ stencil is ever measured under
both layouts (gated on `D-WXS-9` → `D-WXS-0`): MORTON is expected to win there,
by roughly 2×, because that is what the locality metric already shows. That is
a **prior, not a result** — and a 2× locality win still would not by itself
justify a migration, because the same measurement shows it costs ~1.5× on
range count.

#### Why SHIPPED's numbers look the way they do (arithmetic, checked)

SHIPPED orders `[lat_tile, lon_tile, lat_hip, lon_hip]`, so within one tile the
order is `lat_hip`-major / `lon_hip`-minor. A `lon ± 1` neighbour is therefore
**1** apart and a `lat ± 1` neighbour is **64** apart; the median over the
4-neighbourhood is the median of `[1, 1, 64, 64]` = **32** — exactly the
measured value. The two-level blocked order §1.3a identified (rather than the
flat "row-major" I first wrote) is what keeps its range count competitive.

### §1.4 classid — a mint decision, NOT taken here

`0x0F = Geo` already exists in the OGAR domain table; free domains are `0x03–0x06`
(`weather-substrate-poc-v1` §, carried into v2 §5). The composed classid is canon-high:
`domain ++ appid` in the high u16, **ClassView selector** in the low u16. Minting is an
OGAR-side, operator-gated act — **this plan does not mint it and does not invent a placeholder
that could reach a committed dataset.** `D-WXS-0` (§4, W0) is the blocked row. Until it is
resolved the bake **must refuse to write** rather than write under `0x0000_0000` (the
zero-fallback ladder owns that value and a weather dataset carrying it is unroutable garbage
with no way to tell it apart from a bootstrap row).

---

## §2 The value — which variables ride which facets, in which layout

### §2.1 Capacity, against the real budget

Free slab after existing tenants: **292 bytes** (§0.4). As 16-byte facets: **18 facets =
216 payload bytes**.

| scheme | payload bytes needed | fits as 4+12 facets? | fits as a raw byte lane? |
|---|---|---|---|
| all 122 fields @ 1 B | 122 | **yes** (11 facets = 176 B of slab) | yes |
| 108 time-varying fields @ 1 B (statics split out, §2.4) | 108 | **yes** (9 facets = 144 B) | yes |
| all 122 fields @ 2 B | 244 | **NO** (21 facets = 336 B > 292) | yes (244 ≤ 292) |

**The headline survives, with the budget corrected: the entire ERA5 state for a cell — all 17
surface, all 91 upper-air across every one of the 13 pressure levels, all 14 statics — fits
inside one canonical 512-byte `NodeRow` at 1 byte/field, with room to spare. One cell is ONE
node.** No spillover table, no second row, no level-as-node scheme.

But the 2-byte column matters and must not be skipped: **at 2 bytes/field the state does *not*
fit as proper 4+12 facets** — only as an unprefixed raw byte lane (the shape `FrozenStyle` /
`LearnedStyle` / `ExploreStyle` already use: 12 raw palette bytes, no classid prefix). If
`D-WXS-7` (§5) shows 1 byte/field is insufficient for some variable, the fallback is
**per-variable 2-byte stacked floors on a subset**, not a global 2-byte scheme.

### §2.2 The layout choice, argued against the §3 catalogue

Rejected, with reasons:

| # | layout | why not |
|---|---|---|
| L1/L2/L3 | rails `part_of:is_a` / `memberof:members` / mereology:taxonomy | Relational planes. A cell's temperature is not a relation to anything. The classview is a *focus lens*: relational data focuses rails, and this data is not relational. |
| L5 | `4×(8:8:8)` SPO triplets | The SPO reading is wrong for a physical field. **Kept as the named alternative** for a 3-D wind lane `(u,v,w)` per level if `vertical_velocity` is ever baked alongside u/v — four levels per facet. Not W1. |
| L6 | `3×(8:8:8:8)` SPOG quads | The Odoo factoring. Rejected outright. |
| L7 | `2×48` `hhtl ++ helix` absolute location | This IS location — and for a weather cell **location is the key, not the payload**. Storing it again in the value slab is the duplication §2's "never waste a slot" rule exists to prevent. (The `le-contract` §3 open item — whether the CANON key tail is literally an L7 facet — is untouched here and must not be unified silently.) |
| L8 | `helix ++ CAM_PQ` analog | Retired for this data by measurement, not taste: `weather-normalized-substrate.md` §12.12 **RETIRED the `Pair48` mint**, and §12.12's own regrade fixes the split — **rim = L1-metric carrier; azimuth = render carrier, deliberately not L1-metric**. Routing a weather scalar through the analog lane re-opens the exact "invented round-trip API" the helix crate names twice, unprompted. |
| G1/G2/G3 | wide carvings | **"New classes MUST NOT be born into G1–G3; the waiting room is not a destination."** The weather class is new. |

**Chosen: L4 — `6 × (8:8)` palette256², one byte per ERA5 field, the pair being a physically
paired couple.**

Justification in the catalogue's own terms: the classview is the **focus lens the data shape
wants**, and this data is *similarity data over commensurable scalars* — which is precisely
what L4 is for, and precisely what the arc has already measured. The pairing is not
decoration: `(u : v)` in one 8:8 pair means **wind-vector similarity is one 256×256 table
read**; `(msl : sp)`, `(T : q)` at a level, `(T_2m : T_sst)` are the natural couples. Slot
purity is preserved because *which* couple a pair holds is a ClassView fact (§3), never
inspectable from the bytes.

### §2.3 ⚠ A stated deviation from L4's default reading — and the measurement that licenses it

`le-contract.md` §3 pins L4's canonical distance/rank read to the **analytic Fisher-z codec**
(`bgz-tensor::fisher_z::{FamilyGamma, FisherZTable}`, certified ρ ≥ 0.999 on cosines).
**For weather scalars that is measured to be WRONG**, on this repo's own committed artifacts:

- `weather-normalized-substrate.md` §12.1, real ERA5 `2m_temperature`, 1,038,240 gridpoints,
  same 256 buckets, same percentile window: **linear MAE 0.0684 K / 0 empty buckets / 115.7
  effective buckets**, vs **Fisher-z-then-linear MAE 0.2168 K / 76 empty buckets / 28.1
  effective**. Fisher-z is **3.2× worse and burns 228 of 256 buckets**. Mechanism measured, not
  assumed: `arctanh` moves resolution toward the bounds; ERA5 anomaly has **77 % of mass inside
  |s| < 0.25** — mass in the middle. (§12.10 later narrowed this: scored against a noise floor
  rather than round-trip, the two are indistinguishable; **what survives is bucket economy, not
  a validity failure**. Bucket economy is exactly what a 1-byte-per-field scheme is spending.)
- `le-contract.md` §3 supplies the demarcation itself: *for a **monotone BOUNDED** continuous
  field the exit is **analytic, not materialized** —* helix `RollingFloor`
  (`quantize` → `bucket_center`), with elevation as the canonical case at **1 byte / RMSE ≈
  0.11 % of range**. A weather scalar over a robust percentile window is exactly that shape.

**Decision: the weather ClassView declares an L4 lane whose analytic codebook is the linear
`helix::quantize::RollingFloor`, not the Fisher-z gamma.** Distance is
`helix::distance::DistanceLut::from_floor` — 256 × 256 × u16 = **128 KB, L1/L2-resident**, with
the triangle-inequality regression already shipped (`distance.rs:87-105`, `:118-135`). The
deviation is recorded here in the plan, not smuggled into code, and the `le-contract` §3 note
is the authority for it — not an exception to it.

**Two hard consequences, both falsifiable:**

- **Circular variables must NEVER ride an L1 palette byte.** `end_idx` is monotone, so a
  wrapped bearing puts 359° and 1° at *maximum* L1 distance (§12.2 point 5 — **stands**, and is
  crate-documented as the "raw-azimuth 2π wrap"). Wind therefore rides as **(u, v) components,
  two linear bytes**; a bearing, if ever wanted, is derived at read time from the pair. This is
  also why `wind_speed` must be baked as its **own byte** and not reconstructed: §6.5 measured
  stored speed ≥ `hypot(ū, v̄)` at **100 %** of samples, mean ratio **1.115**, max gap
  **14.37 m/s** — the Jensen gap **is gustiness**, real signal.
- **The layout choice must not presuppose `D-WXA-5`'s outcome.** One byte per field is exactly
  the quantisation whose fidelity the gate has never tested. Mitigation is structural, not
  verbal: **W1 bakes a minimal justified field set, not all 122** (§2.5), the lane is
  additive/reserve-don't-reclaim so it grows without a stride change, and `D-WXS-7`'s
  graded-resolution ladder (16/64/256 levels) measures the marginal value of a byte *before*
  296 byte-positions are committed.

### §2.4 The 14 statics — a separate class, a separate dataset, ONE version

Time-invariant fields (orography family, `geopotential_at_surface`, `land_sea_mask`,
`lake_cover`/`lake_depth`, vegetation ×4, `soil_type`) baked into the versioned row would be
rewritten **92,044 times**: 1,038,240 × 14 B × 92,044 ≈ 1.3 PB of identical bytes.

**Decision: statics get their own classid and their own dataset, baked once, ONE Lance
version, joined by the identical HEEL/HIP key.** The join is free precisely because the key is
the address. `D-WXS-5`'s bar is a version-count assertion — a bug that rewrites statics per
timestep is caught by `version_count == 1`, not by review.

### §2.5 The W1 field set — minimal and justified, not all 122

Baking 122 fields in W1 would commit 122 byte-positions under an untested quantisation and
would make every downstream bar slower to run for no measured gain. W1 bakes exactly the
fields the open work needs:

| facet | pair 0 | pair 1 | pair 2 | pair 3 | pair 4 | pair 5 |
|---|---|---|---|---|---|---|
| F0 (surface) | `msl` : `sp` | `u10` : `v10` | `wspd10` : `t2m` | `sst` : `tcwv` | `tcc` : `sic` | reserved-zero |
| F1 (850 hPa) | `u850` : `v850` | `t850` : `q850` | `z850` : `w850` | reserved | reserved | reserved |
| F2 (500 hPa) | `u500` : `v500` | `t500` : `q500` | `z500` : `w500` | reserved | reserved | reserved |

`msl` + `u10/v10` are what `D-CZ-8` needs (ζ from the 10 m wind). `z500` is the classic
verification target and the field `weather-substrate-poc-v1`'s bake-off named.
Everything else is **reserved-zero**, dormant, expandable per class without a version bump
(`NodeRow` doctrine 1: *clean ⇒ expansion is classid-inherited*).

### §2.6 Slot purity (§2) — how it is enforced, and where the temptation is

**Nothing in the payload says what a byte means.** No variable names, no level numbers, no
column ordinals, no display labels. The mapping *"F1 pair 0 hi-byte = `v_component_of_wind` at
850 hPa, unit m/s, floor id 7"* lives in the **ClassView-side field manifest** selected by the
classview u16 — the register-file model: the SoA is dumb bytes, the class makes them
meaningful.

Three places the temptation is real, and the ruling for each:

1. **lat/lon.** It is a position, and §2 forbids positions in payload slots. It is **in the
   key**, which is the key's job (`HEEL`/`HIP` are the cascade address; the OGAR pin is
   explicitly *"the key prerenders nodes with zero value decode"*). Writing lat/lon into the
   value slab as well would be both a §2 break and a duplication of the address. Forbidden.
2. **The codebook edges** `(lo, hi, floor_version)` per variable. Per-row they would cost
   ~1 KB/row and would be a slot-purity break. They are **dataset (table) metadata**, aligned
   to the Lance version boundary — which is exactly `weather-normalized-substrate.md` §4's
   recommended operating rule: *calibrate once per epoch, freeze, align any re-roll with a
   version boundary, record `(lo, hi, version)` in dataset metadata, so "same code ⇒ same
   value" holds exactly within a Lance version and `QueryReference::at` rehydrates with the
   floor that produced it.* The floor stamp and the Lance version become the same kind of
   object: a calibration epoch.
3. **The timestamp.** It is the **version**, not a payload slot (§3).

**Honest open item, carried not resolved:** `le-contract.md` §5 discrepancy 6 (via
`weather-normalized-substrate.md` §12.2 point 6, `[G-absence]`) — **there is no per-value-lane
*reading* selector in the contract at all.** `ReadMode` has three axes (tail / value_schema /
edge_codec); `ValueSchema` selects **presence**, not reading. So "the classview selects the L4
reading" is doctrinally correct and **structurally unimplemented**. This plan does **not**
invent one. `D-WXS-1` ships the manifest as a **committed data artifact** the bake reads, with
its identity stamped into dataset metadata, and flags the missing selector as an
architecture-entropy item for the orchestrator. Anything else would be a contract-surface mint
smuggled in through a weather plan.

---

## §3 The time axis — a timestep is a Lance version

**Ruling in force:** `E-MARKOV-TEMPORAL-STREAM-1` — *episodic = Lance versions* (the OGAR
D-DELTA mapping, now primary); the ±5 window generalizes to a **version-range read**
(`QueryReference::at(v, rung)` + `deinterlace`) at any width, per-reader rung, replayable, and
**still a projection — zero copies**.

| need | shipped surface (consumed, never re-implemented) |
|---|---|
| one timestep = one version (writer) | `graph::cycle_sink::LanceCycleWriter` (`open`/`bootstrap`/`head()`/`scan_image`/`reconcile_scans`), `VersionedGraph::commit_encounter_round` |
| time-travel read | `VersionedGraph::at_version(v)` (`versioned.rs:428`) · `current_version()` (`:435`) |
| version-range read | `planner::temporal::{QueryReference::at(v, rung), deinterlace, LanceVersion = u64, EpistemicMode, TemporalStatus, classify}` |
| per-key trajectory | `temporal::{local_trajectories, local_trajectory_of}` |
| diff two timesteps | `GraphDiff` (`versioned.rs:70`), `GraphSealStatus` (`:54`) |

Two shipped tests already pin the invariant
(`a_whole_cycle_of_casts_is_one_wal_write_one_version`,
`p4a_drains_casts_and_seals_one_wal_write_one_version`). **Any weather deliverable that
re-implements a version writer is the defect, not the feature.**

**What this replaces, concretely.** `substrate_comfort_d_cz_2_7.py:50` reads
`TIMESTEPS = [54358, 54358 + 4*182, 54358 + 4*365]` — three hand-typed integers with an
`EPOCH`-arithmetic anchor guard. On the substrate that is a **version range**, and the
19-storm set (each at its own `t0`) is a **version set**. The seasonal spacing that made those
three integers a real anti-cherry-pick control becomes a *stated version-selection policy* that
the harness can vary — which is what makes `C1`'s "≥3 independent timesteps" bar cheap to raise
to 30 instead of an argument about fetch budget.

**Two honest risks, both pre-registered as bars (§4 W2):**

- **92,044 versions on one dataset is unverified.** CONJECTURE. `D-WXS-6` measures manifest
  growth and `at_version(1)` latency at N = 1 / 10 / 100 / 1000 versions and states whether the
  growth is sub-linear. **KILL:** superlinear growth ⇒ "one version per timestep" does not
  survive 65 years and the time axis needs an explicit chunking policy — a pre-registered
  alternative, not a patch discovered mid-run.
- **The forecast/analysis two-lane model** (poc-v2 §1's *"one dataset, versions are cycles"*)
  is untouched by this plan and stays as ruled.

---

## §4 Waves, D-ids, and pre-registered bars

**The rule for every bar below** (`CLAUDE.md` § The falsifiability rule): the bar is written
and **committed before the run**; it has a **control that can lose**; it has a **stay-silent
twin** using a **non-trivial** input (an empty-input silence case proves nothing); and it names
its **KILL** condition. An assertion implied by the code it tests is not a test.

### W0 — decisions and pre-registration (no executable code)

| D-id | deliverable |
|---|---|
| **D-WXS-0** | **classid mint** for the weather-cell class + the statics class. **BLOCKED** — OGAR-side, operator-gated. `0x0F = Geo` exists; the appid/classview half is open. **KILL:** if no domain:appid can be assigned, the bake must **refuse to write** — a dataset under `0x0000_0000` is unroutable and indistinguishable from a bootstrap row. Nothing downstream runs against a placeholder. |
| **D-WXS-1** | **the field manifest v1** — the ClassView-side table mapping (facet, pair, byte) → (ERA5 variable, level, unit, floor id). A committed data artifact, not literals in Rust. |
| **D-WXS-1a** | **the variable census, as a committed re-runnable probe** — the `.zmetadata` read behind §0.6. A census asserted in prose is the chat-only-figure defect, third instance in this arc. |

**Bar B0 (manifest is load-bearing).** *Can-it-fire:* mutating one manifest entry must change
the bytes the bake writes for at least one cell. *Stay-silent twin (non-trivial):* editing a
**reserved-zero** slot's description, or reordering two entries that describe the same
(facet, pair, byte), must change **not one written byte**. **KILL:** if the bake's output is
insensitive to the manifest, the mapping is hard-coded somewhere and slot purity is theatre.

### W1 — the bake (the missing Zarr → `NodeRow` path)

| D-id | deliverable | file |
|---|---|---|
| **D-WXS-2** | key codec: `(lat_idx, lon_idx) ↔ NodeGuid` (HEEL/HIP, ragged tiles, lon-wrap range-**set**) | `crates/weather-poc/src/key.rs` |
| **D-WXS-3** | shared canonical floor calibration (per-variable `[lo,hi]` at the 0.4–99.6 percentile over a **global** sample, frozen per epoch, stamped into dataset metadata) | `.../src/floor.rs` |
| **D-WXS-4** | the bake: one timestep slab → 1,038,240 `NodeRow`s → **one Lance version** | `.../src/bake.rs` |
| **D-WXS-5** | statics bake — separate classid, separate dataset, one version | `.../src/statics.rs` |

**Bar B1 (key).** *Primary:* all 1,038,240 `(lat,lon)` → key → `(lat,lon)` round-trip exactly,
**and no two distinct cells collide on one key**. *Control that can lose:* an off-by-one on
the **ragged last tile** (17 lat rows / 32 lon cols) must make the round-trip fail — if the
suite passes with that injected, it never tested the ragged case. *Stay-silent twin:* a box
entirely inside `[0°, 360°)` must return a **single** range and must **not** report a wrap;
a box crossing the seam must return **exactly two**. **KILL:** any collision ⇒ HEEL/HIP alone
does not address the grid and `TWIG` must be recruited — a design change, reported, not
patched.

**Bar B2 (floor).** *Primary:* on the global calibration sample, **zero empty buckets** and
saturation below a pre-registered ε per variable (`RollingFloor::occupancy()` **is** the
histogram — no separate instrument). *Control that can lose:* a floor calibrated on **one 16°
box** applied globally must show high saturation — this is `GEO-DEGENERATE` re-homed, and
D-CZ-1 measured that construction saturating **72–97 %** per regime, so it is known-losable.
*Stay-silent twin:* the **global** floor must **not** show high saturation on any single box —
including the storm boxes, where the field is widest. **KILL:** if one global floor cannot
cover a variable within ε, that variable needs its own floor (§4 policy (c) hybrid) and the
"one comparable substrate" claim narrows *for that variable*, stated per-variable rather than
globally.

**Bar B3 (bake round-trip).** *Primary:* read back version *v*, dequantize via
`bucket_center`, compare against the retained f32 truth — per-variable MAE within the
pre-registered ±½-bucket bound (≈ 0.195 % of span, `quantize.rs:11-18`). Plus the **512 B
verbatim** check (PR #907's compression-detection falsifier reused: the stride clears Lance's
256 B mini-block cutoff, takes the full-zip path, `1.00×`). *Control that can lose:* a bake
whose `floor_version` stamp is omitted must be **detected as unreadable**, never silently
mis-dequantized — this is the `quantize.rs:20-26` contract made into a test. *Stay-silent
twin:* a bake and re-read at the **same** floor version must produce **byte-identical** rows
across two runs (determinism). **KILL:** if the round-trip error exceeds the bucket bound, the
quantisation path has a bug and no fidelity number computed on it means anything.

**Bar B4 (statics).** *Primary:* after committing N timestep versions to the dynamic dataset,
the statics dataset has **exactly one** version. *Control:* a deliberately re-writing statics
driver must trip it. *Stay-silent:* a legitimate one-off statics re-bake (a new epoch) must
produce exactly **two** and be labelled, not silently tolerated.

### W2 — the time axis

| D-id | deliverable |
|---|---|
| **D-WXS-6** | version-range read for a cell's time series: `QueryReference::at(v, rung)` + `deinterlace`; plus the version-scaling measurement (§3) |

**Bar B5.** *Primary:* a 3-step series read by version range **byte-equals** the same 3
versions read individually. *Control that can lose:* a `v_ref` outside the committed range must
return **empty**; a rung that excludes a row must **actually** exclude it (both halves — a
filter that never excludes and one that excludes everything carry identical information).
*Stay-silent twin:* a rung that should exclude **nothing** must return the full set — proving
the filter discriminates rather than always firing. **KILL:** as §3.

### W3 — re-homing `D-WXA-5`, the real gate

| D-id | deliverable |
|---|---|
| **D-WXS-7** | representation fidelity on the substrate: ρ(code-distance, field-distance) over pairs sampled from the **whole grid**, computed with `jc::reliability::spearman` |
| **D-WXS-8** | cross-variable comparability at grid scale — `p2_probe.py` re-run as a substrate read, ≥4 variables incl. ≥2 distinct units, ≥3 seasons |

**Bar B6 — `D-WXA-5`, re-specified in three parts.** The poc-v2 bar (ρ ≥ 0.98 + shuffle
control fails) is **kept as the floor and demoted from the verdict**, per §0.3:

- **(a) primary — ρ ≥ 0.9996**, the bar `p2_probe.py` actually pre-registers and where a real
  pair **genuinely failed** (K×K = **0.999556**, below bar; the cross-unit pairs passed at
  0.999736 / 0.999722). A bar with a *demonstrated* failure on real data is a bar that can
  fire.
- **(b) control that can lose — the shuffled decode table must score below 0.98.** Known
  losable: D-CZ-1 measured `CAL-SHUFFLE` at ρ = **0.003 … 0.159** and RMSE **1498.6 Pa**
  against real arms at 0.99999. This is the poc-v2 anti-vacuity control, retained verbatim.
- **(c) can-it-DIFFER — a graded-resolution ladder at 16 / 64 / 256 levels must produce a
  strictly monotone ρ ordering, run BEFORE any verdict.** This is the half D-CZ-1 discovered
  the hard way (§6.4: real-arm ρ spread 3×10⁻⁶…4.7×10⁻⁵ ⇒ *"C4 could not have fired"*).
  **KILL for the metric:** if 16 levels is indistinguishable from 256, ρ is **decorative** on
  this data and the gate moves to a **physical-unit** metric (per-variable MAE in K / Pa /
  m·s⁻¹, range-normalised) *before* a single fidelity claim is made — exactly the amendment
  D-CZ-1 made legitimately in preflight, and which would be illegitimate the moment a verdict
  cell had been scored.
- **KILL for the substrate (the thesis-level one, unchanged from poc-v2):** every arm below
  ρ ≈ 0.9 ⇒ the hierarchical-prefix-is-ancestry assumption does not survive contact with a
  real physical field, and a large part of the substrate's `[H]`/`[S]` map downgrades from
  "faithful code" to "useful router." That is a thesis result, not a weather result.

**Bar B7 (`D-WXS-8`).** *Primary:* cross-unit ρ ≥ 0.9996 on the shared canonical floor.
*Control that can lose:* **per-variable floors must lose on cross-unit pairs** — measured
0.9997 shared vs **0.857–0.875** per-variable, a four-orders-of-magnitude discrimination that
appeared without being tuned for. *Stay-silent twin:* within-variable, shared must **not**
cost resolution — measured ≤ 0.0001 difference, zero empty buckets. **KILL:** if the shared
floor loses to per-variable floors on any cross-unit pair at grid scale, §4 policy (a) is
refuted and the substrate is a per-variable store — which would retract the arc's product
claim, and is the reason this runs early.

#### ⚠ CORRECTION 2026-08-13 — `D-WXS-7` is NOT blocked on the bake, and I said it was, four times

The session repeatedly reported `D-WXS-7` as gated behind `D-WXS-4` (the bake)
and therefore behind `D-WXS-0` (the classid mint). **That is wrong, and it was
never checked before being repeated.**

Read bar B6 above against what it actually needs:

| bar B6 needs | where it lives | blocked? |
|---|---|---|
| real ERA5 field values, whole-grid scale | the store, over HTTP | **no** — `era5_variable_census.py` proved the fetch path works |
| a 256-level quantiser + its `[lo,hi]` window | `floor.rs`, **shipped** (`D-WXS-3`) | **no** |
| a shuffled decode table (the control) | a permutation of that codebook | **no** |
| a 16/64/256 resolution ladder | three `calibrate` calls | **no** |
| Spearman ρ | `jc::reliability::spearman`, shipped | **no** |
| a Lance dataset, a `NodeRow`, a classid | — | **not required by any part of B6** |

**The mis-read, named.** The deliverable line says *"representation fidelity on
the substrate … over pairs sampled from the whole grid"*. "On the substrate"
and "at grid scale" describe the **scale and the source** — the whole real
field, through the shipped codec, rather than four hand-picked boxes through a
numpy re-derivation. They do **not** say "persisted to Lance". I read the phrase
as implying the bake and then repeated the conclusion without going back to the
bar.

**This is the session's own recurring defect, committed by me, about my own
plan** — a claim that was plausible, load-bearing, never verified against the
artifact, and propagated by repetition. It is the same shape as the stale
saturation figure, the vacuous disable probes, and the "no external review"
statement, all of which this session also caught. The countermeasure is the one
already on the board: **an audit must terminate at an artifact**, and a blocker
is an artifact-checkable claim like any other.

**What IS blocked, precisely.** `D-WXS-4` (the bake) and `D-WXS-5` (statics) need
the mint, because they write rows and a row needs a routable classid.
`D-WXS-6` (the version-range read) needs datasets to read. `D-WXS-9`/`D-WXS-10`
(ζ) need the neighbour read the bake provides. **`D-WXS-7` and `D-WXS-8` need
none of that** — they measure the *codec*, and the codec is shipped.

**Consequence:** the gate everything else hangs on has been runnable since
`D-WXS-3` landed. It is re-scoped here from *blocked* to *ready*, and the
sequencing note in §4 ("`W0 → W1 → W2 → W3`") is a dependency ordering for the
*bake* deliverables, not a licence to treat W3 as unreachable while W1 is
incomplete.

#### `D-WXS-7` / `D-WXS-8` — RUN 2026-08-13. Mixed: `D-WXS-7` PASSES cleanly at grid
scale (a prior smaller-scale near-miss does **not** replicate); `D-WXS-8`'s
control (the thesis-level claim) PASSES at every pair; its strict primary and
its stay-silent twin each have real, honest FAILs that qualify but do not
retract the claim.

**Method, real not simulated.** `probes/weather-p1/fidelity_probe_fetch.py`
live-fetched real ARCO-ERA5 grid chunks (public HTTPS, the `fetch.py`-proven
404=all-NaN=valid-missing-chunk path) at **3 real calendar seasons** found by a
live HEAD-sweep of 24 candidate timesteps across the whole 1959–2023 archive —
**not assumed**: a first attempt at 4 fixed 2021 calendar-season anchors found
only the SAME 3 variables (`2m_temperature`, `2m_dewpoint_temperature`,
`10m_u_component_of_wind`) present at **every one** of them;
`10m_v_component_of_wind` / `mean_sea_level_pressure` / `surface_pressure` /
`total_column_water_vapour` / `total_cloud_cover` / `sea_surface_temperature`
were absent at all 4 — confirming `probes/weather-p1/README.md` §1's
"sparse by design" finding is not a one-timestep artifact. `10m_wind_speed`
does not exist in this store at all, confirming the pre-existing
`weather-normalized-substrate.md` §2 finding. A follow-up HEAD sweep found
**winter** (Dec 1969, 5 vars, 4 units: K/m·s⁻¹/Pa/¹), **spring** (Mar 1970,
4 vars, 3 units: K/m·s⁻¹/kg·m⁻²) and **summer** (Aug 1970, 4 vars, 3 units:
K/m·s⁻¹/¹), each independently meeting bar B7's ≥4-variable/≥2-unit floor with
genuinely present, fully-finite fields (`sea_surface_temperature` came back
**physically masked** — 686,364/1,038,240 finite, land is NaN by definition,
not a missing-chunk case — and was correctly excluded as not-fully-usable
rather than force-included).

`fidelity_probe_prep.py` quantises via the exact `floor.rs` formula
(re-expressed in Python for the same reason `floor.rs` itself is zero-dep —
this stage fetches over HTTP, which the Rust crate deliberately cannot do) and
writes raw `(truth_distance, code_distance)` f64 pair arrays — **200,000 pairs
per comparison**, matching `p2_probe.py`'s own `N`. `crates/weather-poc/
examples/fidelity_probe.rs` (jc added as a **dev-dependency only** — see
`Cargo.toml`'s comment on why that is safe, unlike `helix`) reads them and
computes every ρ with **`jc::reliability::spearman`**, per bar B6's own
wording, and applies every verdict exactly as pre-registered above — no
softened bar, no filtered output. Sanity-checked before trusting any number:
the shuffled-decode arm is a genuinely different array from the unshuffled one
(range `[0, 4.92]` vs `[0, 202]`, not a copy or a stub), and its ρ collapses to
`0.02–0.024` — the pipeline demonstrably discriminates.

**`D-WXS-7` (bar B6) — 12/12 PASS, all three seasons.** The K×K pair (winter,
spring) or its within-variable degenerate twin (summer, where only one K
variable was available):

| season | ρ(L16) | ρ(L64) | ρ(L256) | ρ(shuffled) | (a) primary ≥0.9996 | (b) shuffle <0.98 | (c) monotone |
|---|---|---|---|---|---|---|---|
| winter | 0.986011 | 0.998926 | **0.999909** | 0.023855 | PASS | PASS | PASS |
| spring | 0.984137 | 0.998792 | **0.999895** | 0.020747 | PASS | PASS | PASS |
| summer | 0.969079 | 0.997344 | **0.999684** | 0.020245 | PASS | PASS | PASS |

**This directly updates §0.3's own cited evidence, honestly, not silently.**
§0.3 (and this section's own bar text above) cites `p2_probe.py`'s earlier,
smaller-scale finding — *"K×K = 0.999556, below bar"* — as the reason bar
B6(a) "can fire" at all. At real grid scale (200,000 pairs from the full
1,038,240-cell field, 3 independent real seasons, computed with `jc` not
`scipy`), **the K×K pair does not replicate that near-miss** — it clears
0.9996 with margin at all three. Two things can both be true without
contradiction: the bar was correctly falsifiable when written (a real failure
existed to justify it), and the failure does not hold at the scale/instrument
bar B6 actually specifies. The `p2_probe.py` number is not retracted — it
was measured on a real, smaller fixture and is left as-is — but it is no
longer read as "the substrate's fidelity is marginal"; at grid scale, on this
data, it is not.

> **⊘ FIGURE CORRECTION 2026-08-13, same day, found by re-counting the
> committed JSON while writing the product-lead update.** This block first
> read *"control 16/16 PASS; primary 10/16 PASS, 6 FAIL"*. **Both denominators
> were wrong and the primary pass-rate was overstated.** Counted from
> `fixture/fidelity_probe_results.json` rather than by eye from the runner's
> stdout: there are **19** cross-unit pairs, not 16 (winter 9, spring 5,
> summer 5 — the same-unit pairs K×K and m/s×m/s are informational and
> correctly excluded from the bar). Corrected below.
>
> **The direction of the error matters:** primary was written as 10/16 (63 %)
> and is actually **9/19 (47 %)** — the strict bar fails on a **majority** of
> cross-unit pairs, not a minority. The control is 19/19, not 16/16 — better
> in absolute terms, and its KILL still does not fire.
>
> **Diagnosis, precise:** every figure the program *computed and printed*
> (56 verdicts, 42 pass / 14 fail, every ρ) was carried correctly. The two
> wrong numbers are exactly the ones I derived by **counting rows in terminal
> output by eye** instead of counting the artifact. Same session, same rule
> the arc keeps restating — *an audit must terminate at an artifact* — and
> this is its narrowest form yet: **a figure you tallied yourself is a derived
> figure, and derived figures need the artifact too.** The error reached
> `main` via PR #950 (plan, `STATUS_BOARD`, `EPIPHANIES`, PR body); the first
> three are corrected here, the merged PR body cannot be.

**`D-WXS-8` (bar B7) — control 19/19 PASS; primary 9/19 PASS; twin 2/6 PASS.**
Reported in full, nothing filtered:

- **Control (per-variable floor must LOSE) — 19/19 PASS, at every cross-unit
  pair, every season.** Per-variable ρ ranges **0.245–0.939**; shared-floor ρ
  is **0.9987–0.9999** on the identical pairs. This is the KILL-gated claim —
  *"if the shared floor loses to per-variable on ANY cross-unit pair, §4
  policy (a) is refuted"* — and it does not lose once. **The KILL does not
  fire.**
- **Primary (cross-unit ρ_shared ≥ 0.9996) — 9/19 PASS, 10 FAIL** (winter
  **2/9**, spring **4/5**, summer **3/5**). Every failure is a close miss
  rather than a collapse — `0.998681`–`0.999591`, all still ≥ 0.9986 and all
  still dramatically ahead of per-variable — but **the strict bar fails on
  more cross-unit pairs than it passes**, which the earlier 10/16 phrasing
  obscured. The failures concentrate in **winter**, the only season carrying
  `mean_sea_level_pressure`, and skew toward wind and pressure rather than
  temperature — a pattern, not reported as a proven cause.
- **Stay-silent twin — the two halves diverge, and only one holds.**
  `|ρ_shared − ρ_pervar| ≤ 0.0001`: PASS at spring (0.000044) and summer
  (0.000025), **FAIL at winter** (0.000174 — 1.7× the tolerance). **Zero
  empty buckets: FAIL at all three seasons** (38, 39, 45 of 256 buckets
  empty — 15–18%). This is the one finding this run adds that is not merely
  "known-losable-control held" — the *literal* "zero empty buckets" half of
  the original small-fixture claim (`p2_probe.py`, 1 timestep, 3 variables)
  does **not** hold once the shared floor is calibrated across a wider,
  real, multi-unit pooled window at grid scale. Read plainly: a
  percentile-trimmed shared window necessarily has *some* slack for a single
  variable's own narrower distribution — the direction is unsurprising; the
  bar's literal "zero" was not re-verified before this run and is now known
  to be a real, small, repeated gap rather than a re-confirmed exact zero.

**What this does and does not license.** `D-WXS-8`'s own pre-registered KILL
(the control losing) is the ONLY clause that would retract the arc's core
product claim, and it did not fire — the shared canonical floor beats
per-variable floors on cross-unit comparability **unambiguously and by a wide
margin**, confirmed at real grid scale across 3 real seasons and 16 real
variable pairs spanning 4 physical units, none of it assumed. What the primary
and twin FAILs correctly block is treating **0.9996 exactly** and **literally
zero empty buckets** as *proven at grid scale for every pair* — they are not,
and no verdict above pretends otherwise. Downstream deliverables that need the
directional claim (shared floor is the right design) may proceed; anything
that would need the *exact* numeric thresholds met on every pair must treat
this as open.

**Not run this session, named rather than silently deferred:** L64/L16 ladder
for the cross-unit pairs (bar B6's ladder was only computed for the K×K pair);
a fourth+ season; the wind/pressure-skew pattern in the primary FAILs was
observed, not tested as a hypothesis.

Board: `STATUS_BOARD.md` `D-WXS-7`/`D-WXS-8` rows updated from *READY* to their
real outcomes; `EPIPHANIES.md` prepend recording the K×K non-replication and
the empty-bucket gap.

### W4 — re-homing `D-CZ-8` (ζ + a range-normalised transfer metric)

| D-id | deliverable |
|---|---|
| **D-WXS-9** | ζ = ∂v/∂x − ∂u/∂y as a **substrate read** — a 5-point stencil over neighbour keys; plus the quantisation falsifier below |
| **D-WXS-10** | the **vorticity regime ladder over the whole grid** (regimes = ζ-percentile bands, not four hand-picked boxes) + coverage-matched donor selection + a range-normalised `L` |

**Why this is a substrate read.** A neighbour in HEEL/HIP address space is `lon_idx ± 1` /
`lat_idx ± 1` — **address arithmetic**, and within a HEEL tile the cells are contiguous. The
stencil that `grad_flat()` currently expresses as `np.gradient` over a Zarr chunk becomes a
key-range read. That is the whole point of §1.

**Bar B8 — the falsifier this design *needs* and which does not exist anywhere in the arc:
differencing amplifies quantisation error.** ζ is a *difference of differences* of two
1-byte-quantised fields. A ±½-bucket error (≈0.195 % of span) is negligible on a value and can
be **O(1) relative** on a small gradient. *Primary:* ρ(ζ from quantised u,v ; ζ from retained
f32 truth) ≥ a pre-registered bar, reported **per ζ-magnitude decile** (the error will be
worst where ζ is smallest, and a global ρ would hide exactly that). *Control that can lose:*
ζ derived from a **16-level** palette must be visibly worse. *Stay-silent twin:* on a laminar,
near-zero-ζ region the discriminator must **not** flag turbulence — the `closed_class_guess`
150/150 defect in its weather clothing. **KILL:** if quantised ζ does not track true ζ,
**ζ must be baked as its own derived lane** rather than derived at read time — a real design
fork, pre-registered here so it is a decision and not a mid-run rescue.

**Bar B9 — the range confound, and exactly what the full grid does and does not buy.**

*What the full grid does NOT buy:* it **does not dissolve §7.9's confound.** `L` was measured
to be essentially a function of **coverage** — `L` vs cell `saturation` at Pearson **+0.917** /
Spearman **+0.833**, and `L̄[T]` vs the regime's own value range at **Spearman +1.000,
perfectly monotone** (R4's implied range ≈ 18× R1's: ~7075 Pa vs ~386 Pa). A foreign fixed
codebook applied to a wider target will saturate more, on 1.04 M cells exactly as on 4225.
Moving to the full grid changes the sample, not the arithmetic. **This plan asserts no
dissolution.**

*What the full grid DOES buy — three specific, non-rhetorical things:*

1. **Coverage becomes controllable.** With 1,038,240 cells you can **select** donor and target
   populations **matched on value range and offset** by construction. With four hand-picked
   boxes you could only observe the confound; with the grid you can hold it constant — which
   is the methodological frame (hold variables constant to test the others) applied to the
   confound instead of to the regime.
2. **"Donor" can finally mean a *regime* rather than a *range*.** The shared canonical z-floor
   (`D-WXS-8`, measured cross-unit ρ 0.9997 vs 0.857–0.875 per-variable) puts every variable on
   one commensurable scale, so a donor codebook can be defined by a **ζ-band** rather than by
   whatever range four boxes happened to have.
3. **The regime axis stops being n = 4.** ζ-percentile bands over the globe give a regime
   sample of millions of cells per band and let `C1`'s "≥3 timesteps" bar become ≥30 without a
   fetch-budget argument. It also removes `C5`'s structural blocker's *cause* — though **not**
   `C5` itself: the golden index floor needs N ≥ F(17)² = **2,550,409** and the global grid has
   **1,038,240** cells, so `GEO-GOLDEN-HI` is **still not constructible**, now short by ~2.5×
   rather than by three orders of magnitude. Stated so nobody re-discovers it as a surprise.

*The bar itself.* **Primary:** the ζ-regime effect on a **range-normalised** `L` must survive
**coverage matching** — report `L` both raw and residualised on `saturation`, and report the
matched-population version alongside. **Control that can lose:** a regime ladder built on a
**shuffled ζ field** (same marginal distribution, destroyed spatial structure) must show **no**
monotone `L̄` trend. **Stay-silent twin:** two donor/target sets drawn from the **same** ζ-band
must show `L̄ ≈ 0` — if band-internal transfer is as costly as cross-band transfer, the bands
are not a regime axis at all. **KILL:** if `L̄` still rank-correlates ρ ≈ 1.0 with range *after*
coverage matching, then the transfer metric **is** a coverage metric, `D-CZ-8`'s premise fails,
and the result is reported as such — **no further normalisation is attempted until it agrees.**
Re-normalising a metric until it stops showing the confound is how a confound becomes a finding.

### W5 — gated on W3 green: the honest tail

| D-id | deliverable |
|---|---|
| **D-WXS-11** | **measure** a full-grid bake wall time. The artifact must state the ~8.1 s serial extrapolation as the **prior** and either confirm or correct it in the same document (§0.5). No claim about parallelism — `D-KIA-A2` owns that. |
| **D-WXS-12** | `jc` ↔ `ndarray` reliability agreement — poc-v2 `D-WXB-4` carried over unchanged: identical non-degenerate inputs must agree; the degenerate case must be **reported** (`jc` → `None`), never folded as `0.0`. **`jc` is the authority; `ndarray::hpc` is the SIMD-side mirror.** |

**Sequencing.** `W0 → W1 → W2 → W3 → (W4, W5)`. **Nothing in W4 or W5 runs before W3's gate
reports.** A negative W3 is the most informative result available and is thesis-relevant rather
than weather-relevant.

---

## §5 The Python probes, classified: substrate read vs numpy function

Per the brief, every computation in `substrate_comfort_d_cz_0_1.py` and
`substrate_comfort_d_cz_2_7.py` is classified. **"Substrate"** = a shipped primitive exists and
the Python is re-deriving it; **"numpy"** = no substrate equivalent exists and the Python is
correct to be Python; **"probe-side over a substrate read"** = the statistic stays a statistic
but its *input* becomes a key-range scan instead of a Zarr chunk.

| Python | classification | substrate surface / note |
|---|---|---|
| `encode_decode(values, lo, hi)` — 256 uniform levels, clip-saturate, decode to level centre | **SUBSTRATE** | `helix::quantize::RollingFloor::{uniform, quantize, bucket_center}` — the *identical* linear formula (`quantize.rs:99-108`, `:248-250`). `CAL-ABS` **is** a `RollingFloor` with a donor's bounds. This is the single largest re-derivation in the arc. |
| `cell_metrics` → `occupancy`, `saturation` | **SUBSTRATE** | `RollingFloor::occupancy()` **is** the histogram — *"no separate histogram"* (`quantize.rs:5-9`); saturation = the two rim buckets' share; `drift_score()` is the regime-change telltale, free. |
| `spearman(a,b)` (tie-averaged, `nan` on constant input) | **SUBSTRATE** | `jc::reliability::spearman(&[f64], &[f64]) -> Option<f64>` — same tie semantics, and `None` where Python returns `nan`. `jc` is the operator-named authority. |
| `grad_flat` (`np.gradient`, Pa/cell, **no cos(lat)**) | **SUBSTRATE-ADJACENT** — becomes a neighbour-key stencil | ⚠ Carry **both** definitions, labelled with units: the committed ladder used the **flat, no-cos** definition (D-CZ-0 §6.2 identified it from the data at max\|ratio−1\| = 0.069), and it **understates the zonal gradient by 1/cos(lat)** — R3 at 60 N is ~40 % low. Silently switching to the metric-correct definition would break comparability with every committed figure. |
| `box_index(clat, clon)` + `% len(LONS)` | **SUBSTRATE** | HEEL-prefix **range-set** (§1.2/§1.3). The `% 1440` is the seam split. |
| `rank_codec` (256 **quantile** levels in-window) | **NUMPY — no substrate carrier exists** | `RollingFloor` is *linear over a window*, not quantile-spaced. `CAL-RANK` has no shipped Rust equivalent. Either it stays the Python reference arm, or it is a **new codec** owing its own probe. Do not paper over this by calling `RollingFloor` "the rank codec". |
| `fisherz_codec` | **SUBSTRATE** (`helix::fisher_z` / `bgz-tensor::fisher_z`) — but **control-arm only** | §12.1 measured it 3.2× worse and 228/256 buckets burnt on these shapes; §12.10 narrowed that to *bucket economy, not validity*. Keep it as the shape-mismatch control, never as the incumbent. |
| `decay_length_cells`, `gini`, `tail_ratio` (C1c instruments) | **NUMPY** — no substrate equivalent (`jc::stats` has none) | Stay probe-side statistics, but computed **over a substrate read** (a key-range scan), not over a Zarr chunk. |
| `fetch(var, key)` + numcodecs blosc decode | **PYTHON, correctly** | poc-v2 §1's Stage-A **disposable** ingest. Do not gold-plate the throwaway; do not put GRIB2 or a Rust Zarr reader on the critical path. |
| `t_index()` + the `EPOCH` anchor assert | **SUPERSEDED** | The time axis is Lance versioning (§3). The anchor guard's *spirit* survives as a manifest/metadata check that fails loudly if the store is re-chunked. |
| `TIMESTEPS = [...]`, per-storm `t0` | **SUPERSEDED** | A version range and a version set (§3). |

---

## §6 Work split — design/judgment vs mechanical grindwork

### §6.1 Design / judgment (Opus or the main thread — accumulation, never delegated)

- The **facet-layout choice** and its stated deviation from L4's default Fisher-z reading
  (§2.2/§2.3) — it holds two measured findings and a doctrine note in mind at once.
- The **key assignment** and the three wrinkles (§1) — a wrong tiling is a silent
  data-corruption class, not a bug.
- **Every bar in §4** — a bar written by the code's author is the
  `E-ZERO-FOR-ELEVEN-THE-AUTHOR-CANNOT-AUDIT-HIS-OWN-FALSIFIERS-1` rule waiting to fire. Bars
  are pre-registered by design, committed before the run, and (for any verdict-tier claim)
  independently spec-audited.
- The **`D-WXA-5` re-specification** (§0.3 / bar B6) — a judgment call about an instrument's
  dynamic range, grounded in a committed measurement.
- The **§7.9 confound treatment** (bar B9) — deciding what the grid buys and what it does not.
- **`D-WXS-0`** (the classid mint) — operator/OGAR-gated, not a worker task.

### §6.2 Mechanical grindwork (Sonnet workers, one file each, disjoint)

Every worker brief pastes `.claude/v3/knowledge/sonnet-worker-guardrails.md` §1 **verbatim**,
plus §2 (vocabulary disambiguation) and §5 (STOP+report triggers). No grindwork spawn without
it. Every brief also carries: *read `.claude/board/AGENT_LOG.md` before starting; do NOT write
it — leave your record in your own tag-file under `probes/weather-p1/exec-runs/`; the
orchestrator consolidates.*

| worker | file (sole owner) | task shape |
|---|---|---|
| W-key | `crates/weather-poc/src/key.rs` | encode/decode + range-set; bar B1's tests |
| W-floor | `crates/weather-poc/src/floor.rs` | drive `RollingFloor` calibration from a sample; emit `(lo,hi,version)`; bar B2 |
| W-lane | `crates/weather-poc/src/lane.rs` | L4 pack/unpack **given** the manifest; zero domain knowledge |
| W-bake | `crates/weather-poc/src/bake.rs` | slab → `NodeRow`s → `LanceCycleWriter`; bar B3 |
| W-statics | `crates/weather-poc/src/statics.rs` | the one-version statics path; bar B4 |
| W-stencil | `crates/weather-poc/src/stencil.rs` | neighbour-key arithmetic + ζ; bar B8 |
| W-metrics | `crates/weather-poc/src/metrics.rs` | `jc` battery wiring; bars B6/B7 |
| W-tests | `crates/weather-poc/tests/<one file per bar>.rs` | one file per bar, never shared |

**Orchestrator-only files (shared — a worker touching one is the defect):**
`crates/weather-poc/src/lib.rs` (the `mod` lines), `crates/weather-poc/Cargo.toml`, the root
`Cargo.toml` (**not touched at all** — `weather-poc` is workspace-**EXCLUDED**, on the same
pattern as `jc`, `helix`, `sigker`, `perturbation-sim`),
`crates/lance-graph-contract/src/canonical_node.rs` (the `ValueTenant` enum + `VALUE_TENANTS` —
a canon act, Opus/orchestrator only, and note discriminant **15 is reserved for
`BoardAggregates`**; a weather tenant takes **16**, with its offset **derived** via
`value_offset()` and never written as a literal), and every `.claude/board/*` file.

### §6.3 Crate scaffold constraints (non-negotiable)

- **No new repositories.** `crates/weather-poc`, workspace-**EXCLUDED**, on the
  `perturbation-sim` template. Verified via
  `cargo test --manifest-path crates/weather-poc/Cargo.toml`.
- **Zero-dep default build.** Any `ndarray` dependency is **optional, behind an off-by-default
  feature, and sourced by GIT URL — never `path`** (an optional *path* dep is read at manifest
  resolution and breaks a clean checkout even with the feature off: codex P2 on #504, the
  helix-#460 lesson, recorded verbatim in `perturbation-sim/Cargo.toml`).
- **SIMD only via `ndarray::simd::*`** (`ndarray/src/lib.rs:241`, `src/simd.rs`). **Never
  `ndarray::hpc::*` from a consumer**, and never raw intrinsics.
- **The lance family is upstream-authoritative and moves in exact lockstep.** No version bump
  is proposed by this plan: `lance = "=9.0.0"` family, `lancedb = "=0.33.0"`, `arrow 58`,
  `datafusion 54` **and** `53` (both required — `deltalake-core 0.32.4` pins 53 upstream;
  collapsing the lock to one breaks the `delta` feature), rust `1.97.1`.
- **Credentials by name only** — `dev_s3_env::s3_options()` reads the set and returns `None`
  when any is missing; on a remote path that must be a **hard error**, never a silent local
  fallback. **S3 hydrates; a local mmap-capable directory stores** (#901); a network mount that
  looks local is the named trap.

---

## §7 Honest grading

| claim | grade | provenance |
|---|---|---|
| 1,038,240 cells × 512 B = 0.495 GiB/timestep | `[G]` | arithmetic on the const-asserted stride |
| 122 fields fit one 512 B row at 1 B/field | `[G]` | arithmetic against the **292 B** free budget (§0.4), not 384 |
| 122 fields do **not** fit as 4+12 facets at 2 B/field | `[G]` | 21 × 16 = 336 > 292 |
| HEEL/HIP addresses 721×1440 with 5–6 of 12 quaternary levels/axis | `[G]` | 4⁵ ≥ 721, 4⁶ ≥ 1440 |
| the axes tile raggedly (721 = 11×64+17; 1440/64 = 22.5) | `[G]` | arithmetic |
| shared canonical z-floor beats per-variable floors cross-unit (0.9997 vs 0.857–0.875) | `[H]` | `p2_probe.py`, one timestep, 3 variables, §12.8 — graded `[H]` there pending seasons + a 4th unit |
| the K×K pair (0.999556) is **below** the 0.9996 bar | `[G]` | measured, corrected under review on #920 |
| Fisher-z burns 228/256 buckets on ERA5 temperature | `[G]` on the buckets; `[H]` on the verdict | §12.1 measured; §12.10 narrowed to bucket economy, not validity |
| ρ saturates on the diagonal (real-arm spread 3e-6…4.7e-5) | `[G]` | D-CZ-1, `substrate_comfort_d_cz_0_1.json` |
| `L` vs `saturation` Pearson +0.917; `L̄` vs range Spearman +1.000 | `[G]` | §7.9, committed JSON |
| **`D-WXA-5`'s ρ ≥ 0.98 bar is at risk of being vacuous** | **`[H]` — inference from the two rows above; the reason W3 part (c) exists** | §0.3 |
| ~8.1 s per full-grid timestep | **CONJECTURE** — serial extrapolation from ONE measured cycle (514 ms, 65,536 owners); `D-KIA-A2` Queued/unbuilt; and a bake is not a mailbox cycle | §0.5 |
| 92,044 Lance versions on one dataset is viable | **CONJECTURE** — unmeasured; `D-WXS-6` is the falsifier | §3 |
| L4 + linear `RollingFloor` is the right lane for weather scalars | `[H]` — composed from measured pieces, unrun **as a lane** | §2.2/§2.3 |
| ζ derived from 1-byte-quantised u,v is usable | **CONJECTURE — and the plan's most likely failure point** | bar B8 |
| the grid dissolves the §7.9 confound | **FALSE, asserted nowhere** — it buys *control*, not dissolution | bar B9 |
| `GEO-GOLDEN-HI` becomes constructible at grid scale | **FALSE** — needs 2,550,409; grid has 1,038,240 | §4 W4 |

---

## §8 What contradicts the commissioning framing (consolidated)

1. **`D-WXA-5` is not on `STATUS_BOARD.md`; no `D-WXA-*` row is** (§0.1). It lives only in the
   plan file and the `INTEGRATION_PLANS` prose.
2. **The "C2 gate" label belongs to a different plan** (§0.2).
3. **`D-WXA-5` as written is probably unable to fail** on this data (§0.3) — the arc's own
   committed measurement says so.
4. **"32 facets / 33.2 M facets per timestep" overstates the usable budget by ~78 %** — the
   real free budget is 292 bytes / 18 facets (§0.4).
5. **The ~8.1 s figure is both an unverified extrapolation and a category error** — a bake is a
   columnar write, not 16 mailbox seal cycles (§0.5).
6. **`crates/weather-poc` does not exist** — confirmed against the crate listing; and
   `weather-substrate-poc-v2.md` indeed specifies **no bake step**, which is the direct cause
   of the arc staying in Python.
7. **`helix`/`Signed360` is not a wind carrier and the `Pair48` mint is RETIRED** (§12.12) —
   any design that reaches for `HelixResidue` for weather is re-opening a four-corrections-deep
   dead end. This plan does not.
8. **There is no per-value-lane *reading* selector in the contract** (`[G-absence]`, §2.6) — so
   "the classview selects the L4 reading" is doctrine without a mechanism. Flagged, not
   invented around.

---

## §9 `weather-substrate-poc-v2.md`, part by part

| part | verdict |
|---|---|
| §1 disposable Stage-A vs permanent Stage-C ingest split | **SURVIVES** — adopted verbatim |
| §1 "versioning is CONSUMED, not built" | **SURVIVES** — §3 consumes exactly that surface |
| §1 one dataset, versions are cycles | **SURVIVES** |
| §1 S3 hydrates / local mmap stores (#901) | **SURVIVES** |
| §2 `D-WXA-1` Stage-A ingest | **SURVIVES**, re-scoped: emit the **shared `soa:*` metadata block** and the field manifest, not just a flat slab |
| §2 `D-WXA-2` `crates/weather-poc` scaffold, workspace-EXCLUDED | **SURVIVES — and is `D-WXS-4`'s prerequisite. NEVER BUILT.** |
| §2 `D-WXA-3` encoder arms A–E | **SUPERSEDED in part.** Arms A/B (helix48, helix+residue) are retired for scalars by §12.1/§12.12. Arm E (`cascade_key`) is **not an encoder arm at all** — it is the **key design** (§1). What remains a live bake-off is **arm D (bgz17 hierarchical palette) vs the measured incumbent (linear `RollingFloor` + `DistanceLut`)**, which is also `F-1`. |
| §2 `D-WXA-4` the `jc` battery | **SURVIVES** — `D-WXS-7`/`D-WXS-8` |
| §2 `D-WXA-5` the gate | **SURVIVES as the floor, re-specified as the verdict** (§0.3, bar B6). **NEVER RUN.** |
| §3 `D-WXB-1..4` ndarray parity / no-silent-scalar / throughput / jc↔ndarray divergence | **SURVIVES**, deferred to W5 (`D-WXS-11`, `D-WXS-12`). **NEVER BUILT.** |
| §4 `D-WXC-1..5` permanent ingest, two-stage retrieval, external score, three lanes | **SURVIVES as future work**, unchanged and still gated behind the representation gate. **NEVER BUILT.** |
| §5 repos / zero new repositories | **SURVIVES** |
| §6 pins (incl. the ⊘ double correction on datafusion 54 **and** 53) | **SURVIVES** |
| §7 credentials | **SURVIVES** |
| §0 C1/C2/⊘C3 (GRIB2 gone; `ecmwf-opendata` is Phase C; the invented Zarr object name) | **SURVIVES**, and ⊘C3's apparatus lesson — *a title match is not an existence check; list the bucket* — is the reason `D-WXS-1a` commits the census as a probe |

**Net:** poc-v2's *structure* survives almost entirely. What it was missing is the thing this
plan adds — **the bake** — and what it got wrong is one bar's threshold, which the arc's own
later measurement exposed.

---

## §10 Cross-references

`.claude/v3/soa_layout/le-contract.md` §2/§3/§3a/§5 · `CLAUDE.md` § CANON — Minimal SoA node,
§ The falsifiability rule, § Mandatory Board-Hygiene Rule, § Key Dependencies ·
`/home/user/OGAR/CLAUDE.md` § Tier interpretation — 256×256 CENTROID TILE, § the 3×4 path ·
`.claude/plans/weather-substrate-poc-v2.md` · `.claude/plans/substrate-comfort-zones-v1.md`
§7.8/§7.9 · `.claude/knowledge/weather-normalized-substrate.md` §3/§4/§5/§8/§9/§12.1/§12.2/
§12.6/§12.8/§12.10/§12.12 · `.claude/knowledge/encoding-ecosystem.md` (mandatory, read) ·
`.claude/knowledge/helix-cartesian-vs-fisher2z.md` (the doctrine §12.12 was retired for not
reading) · `.claude/knowledge/s3-hydration-lifecycle.md` ·
`.claude/v3/knowledge/sonnet-worker-guardrails.md` · `.claude/board/EPIPHANIES.md`
`E-MARKOV-TEMPORAL-STREAM-1`, `E-V3-FACET-4-PLUS-12`, `E-VACUOUS-ASSERTION-IS-THE-HOUSE-STYLE-1`,
`E-A-CONTROL-THAT-CANNOT-LOSE-IS-NO-CONTROL-1`,
`E-ZERO-FOR-ELEVEN-THE-AUTHOR-CANNOT-AUDIT-HIS-OWN-FALSIFIERS-1` ·
`.claude/board/AGENT_LOG.md` 2026-08-05 (PROBE-IGNITION-64K) ·
iron rules `I-VSA-IDENTITIES`, `I-NOISE-FLOOR-JIRAK`, `I-LEGACY-API-FEATURE-GATED`.

---

## §11 Board rows landed with this plan

> Written by the orchestrator (sole writer of `.claude/board/*`) in the same commit as this
> plan, per the Mandatory Board-Hygiene Rule. §11.1 is the `STATUS_BOARD.md` block; §11.2 is
> the `INTEGRATION_PLANS.md` entry.

### §11.1 `STATUS_BOARD.md` — the D-WXS block

Landed as its own section. Every row `Queued` except `D-WXS-0` (**Blocked**, operator/OGAR).
Carries two standing notes: the poc-v2 ladder's board-hygiene gap (`grep -c WXA` = 0, nothing
ever built), and that the global grid does **not** unblock `GEO-GOLDEN-HI` (needs 2,550,409;
grid has 1,038,240).

### §11.2 `INTEGRATION_PLANS.md` — PREPEND entry

Landed with the plan, recording: the bake as the closed gap, the HEEL/HIP key design and its
three wrinkles, the L4 + linear-`RollingFloor` lane with its stated deviation, the corrected
292-byte capacity budget, statics-as-own-class, timestep-as-version, the `D-WXA-5`
re-specification, and the §7.9 position (the grid buys control, not dissolution).
