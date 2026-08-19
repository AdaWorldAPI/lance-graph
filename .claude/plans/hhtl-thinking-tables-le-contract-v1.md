# HHTL thinking tables × the little-endian structural contract — v1

> **⊘ SUPERSEDED IN PART — READ `docs/architecture/ARC-B-OWNERSHIP-AND-ADDRESSING-REASSESSMENT.md`
> FIRST (operator rulings, 2026-08-19).** Four things in this document are
> now WRONG and are corrected in place below; the corrections are marked ⊘
> at each site and the original text is retained per append-only discipline.
>
> 1. **F1's `32 × (4+12)` canonical substrate is FALSE.** The canonical row
>    is `16 B HHTL key | 16 B edges | 480 B value slab`
>    (`canonical_node.rs:706-730`). The 32-homogeneous-facet shape F1 cited
>    is a **Java-side FIXTURE that says so itself**
>    (`lance-graph-java/native/lgj-abi/src/rowstore.rs:5-8,33-39`) and must
>    conform to canon, not define it. See ⊘F1 below.
> 2. **`AddressingMode::{Rails, Cartesian}` is deleted as a concept.** There
>    is ONE HHTL address fabric; Cartesian is an additional *read/projection*
>    where useful — never a second carrier, hierarchy, storage mode, or
>    parenthood model. See ⊘§2.2 and ⊘D-HTT-2.
> 3. **Morton is NOT canonical and NOT the basis of anything here.** Morton
>    is the separate `4⁴` construct formerly called "nibble"; HHTL ancestry
>    is never derived from it. **D-HTT-6 is WITHDRAWN** (⊘D-HTT-6).
> 4. **HHTL is never "V1".** Ordinary HHTL is `6 × 2 × 8 bit`, read as rails
>    or centroid per the class's reading. The retired V1 shape is the flat
>    u24 *tail* — a different thing. HHTL+ Helix (`6×2×8` **plus** `4×24`)
>    is DEFERRED and must not be canonized in this wave.
>
> **And the gap itself is re-stated.** This plan was written as though the
> question were *"how do we make parent-of-x a bit-op."* Measured: **HHTL is
> already canonical AND zero on every baked row in both production bakes**
> (`ogar-obo/src/lib.rs:344-353`; MedCare `join-map.md:103` — *"heel/hip/twig/
> leaf all 0 across 68797 rows"*). The work is therefore **mint + read**, not
> design: populate the key from what `compute_cascade` already derives in RAM
> at load, then convert consumers. Every bespoke route now carries the burden
> of proof — *why can I not begin from the HHTL already in the first 16
> bytes?*
>
> **Status: RATIFIED (operator review, 2026-08-19).** The four corrections
> above are the operator's own review verdict and are applied in place; the
> reassessment's §3 migration order is the ratified order. Implement from
> **this corrected document plus the reassessment**, in that pairing — never
> from an uncorrected read of the struck-through sections (⊘F1, ⊘§2.2,
> ⊘D-HTT-2, ⊘D-HTT-6, ⊘X2). No further architecture round is required for
> this wave; the next open decision is the canonical-depth ruling
> (spanning-tree depth vs DAG longest path — measured 58.2% agreement),
> which gates D-HTT-9 and is called out at its own site.

> **Status:** PROPOSED — ratification vehicle, **zero code**. ARC B of
> `docs/architecture/DUMB-STORAGE-RESET-CHARTER.md` §19. The charter's §20
> gate ("NO CODE until this map is returned") is still closed and this
> document does not open it; it is the *shape* that must be ratified before
> the gate moves.
> **Ratification path:** operator review; optionally a 5+3 council pass
> (`.claude/agents/5plus3-council.md` — 5 streamline FIRST, then 3 attack v2).
> **Predecessors (read first):** the charter; `ARC-A-SOURCE-ARCHAEOLOGY.md`;
> `ARC-A2-STRONG-HIERARCHY-RECONCILIATION.md` (ARC A′ — supersedes ARC A §3/
> §7/§9/§10 where they conflict).
> **D-ids:** D-HTT-1 … D-HTT-11 (STATUS_BOARD rows land with the same commit
> as this file; the orchestrator is the sole board writer).

---

## §0 FROZEN DECISIONS (operator, restated — not re-litigated here)

| # | Ruling | Source |
|---|---|---|
| ~~F1~~ ⊘ | ~~**ONE canonical substrate: the 32×(4+12) facet register.** `n_rows × 512` bytes, 32 facet lanes of *(4-byte LE classid + 12-byte payload)*.~~ **CORRECTED 2026-08-19 (operator).** The canonical row is **`16 B HHTL key \| 16 B edges \| 480 B value slab`** — `canonical_node.rs:706-730`, independently restated by `OGAR/crates/ogar-obo/src/lib.rs:22-35`. The cited 32-lane shape is a **fixture**, self-declared: *"The Java side may lay its view out differently… The 64-byte-aligned guarantee arrives with the real `NodeRow` (`#[repr(C, align(64))]`) wiring, not here"* (`lance-graph-java/native/lgj-abi/src/rowstore.rs:5-8,33-39`); grep confirms **zero import** of `NodeRow`/`EdgeBlock`/`NodeGuid` in that path. Java conforms to canon; canon is not read off Java. **What SURVIVES from F1:** higher-order / ontology / episodic / meta-awareness are **READINGS**, never parallel structures — that half is unaffected and still frozen. | ⊘ operator 2026-08-19; `canonical_node.rs:706-730`; fixture self-declaration `rowstore.rs:5-8,33-39`; `docs/abi.md:433-438` already names `16\|16\|480` as the target |
| F1′ | **HHTL is the FIRST canonical tenant and it is EMPTY.** The address fabric needs no promotion, no rename, no new carrier, and no layout migration — it needs *minting* (write the bytes) and *reading* (consume them instead of walking edges). | ⊘ operator 2026-08-19; `ogar-obo/src/lib.rs:344-353` (*"dormant cascade"*, HEEL/HIP/TWIG zero on all 68,797 rows); MedCare `join-map.md:103` |
| F2 | **The 12 payload bytes are `6 × (u8:u8)` rails.** `u8:u8` = two separate bytes, never widened to u16/u24. The `256:256` pair is rails-vs-centroid polymorphic per class. | `.claude/v3/soa_layout/le-contract.md:50-68` (L1–L8 catalogue); `CLAUDE.md` CANON `E-V3-FACET-4-PLUS-12`; charter §3 "keep 256:256 polymorphic" |
| F3 | **FORBIDDEN:** `HierarchyPlane` types · separate higher-order structs · promotion DTOs · a generic "structural algebra" crate · another SoA. | operator, this session; consistent with ARC A §7 ("**ZERO new types are proposed**") |
| F4 | **The missing piece is a little-endian structural/addressing CONTRACT** that makes `parent-of-x` a projection/bit-op — **no label lookup**. `classid + WideFieldMask` resolve the reading. The Java side's where-as-masking is the precedent. | operator, this session; `lance-graph-java/CLAUDE.md:24-25` — *"WHERE MAY LOOK LIKE WHERE. IT MUST EXECUTE LIKE MASK. / HOP MAY LOOK LIKE HOP. IT MUST EXECUTE AS MASK × CLASSVIEW/WIDEFIELDMASK → MASK."* |
| F5 | **Higher-order thinking table(s) for HHTL** in location / NARS reasoning / rung ladder / traversal — *"as in cesium (Cartesian addressing) vs rails"*. | operator, this session |
| F6 | **No code until the shape is ratified.** | charter §20 (`DUMB-STORAGE-RESET-CHARTER.md:840-893`) |
| F7 | Inherited and unchanged: hierarchy geometry is an accelerator, never semantic authority (charter §1); WideFieldMask is the fovea, not the address (§2); rung is a SINGLE shared axis (§12); no freeze / no batch wall (§7). | charter |

**Consequence of F3 + F4 taken together:** the deliverable of ARC B is a
*contract* — names, laws, and pre-registered falsifiers over bytes that
already exist — not a module of new carriers. Where an operation already has
a shipped home, the contract **names that home**; it does not re-implement it.

---

## §1 INPUT INVENTORY — every line verified by reading it

### 1.1 The rail reading (contract crate)

| Element | Anchor | State |
|---|---|---|
| `RailAxis::{Taxonomy, Mereology}` | `crates/lance-graph-contract/src/rail_geometry.rs:43-48` | shipped |
| `RailCarving::{InterleavedPairs{reg,axis_byte}, AxisSlab{reg,cont}}` | `rail_geometry.rs:65-72` | shipped |
| `RailCarving::zero_fallback` — key facet `reg: 4`, Taxonomy=byte 0, Mereology=byte 1 | `rail_geometry.rs:80-88` | shipped; **the canon default** |
| `RailCarving::level(row, i)` — the byte read | `rail_geometry.rs:103-129` | shipped but **PRIVATE** (`fn`, not `pub fn`) |
| `read_path` + **the hole rule** (`[1,0,7]` is depth 1, never 2) | `rail_geometry.rs:131-149`; test `the_hole_rule_ends_the_chain` `:253-267` | shipped, disable-tested |
| `RailPath::{depth, slots, is_ancestor_of, arc, placement}` | `rail_geometry.rs:166-212`; `is_ancestor_of` = prefix containment `:178-180` | shipped |
| axis independence, disable-tested (*"Taxonomy darf Mereology nicht sehen"*) | `rail_geometry.rs:269-285` | shipped |
| `ClassView::rail_carving(class, axis)` — registry resolution, defaults to `zero_fallback` | `crates/lance-graph-contract/src/class_view.rs:1127-1133` | shipped; **zero consumers** outside its own default and `rail_geometry.rs` (grep verified) |

### 1.2 The facet register (contract crate)

| Element | Anchor | State |
|---|---|---|
| `FacetTier { lo, hi }` — always exactly two bytes; content-blind | `crates/lance-graph-contract/src/facet.rs:36-43` | shipped |
| `FacetTier::morton()` — `lo` even bits, `hi` odd; *"every nibble of the result is a 2 bit × 2 bit Morton tile, so a nibble prefix is a quad-tree quadrant in BOTH bytes at once (`256 = 4⁴` hierarchical ancestry)"* | `facet.rs:54-64` | **shipped, ZERO consumers** (only its own test `:642-643`) |
| `FacetCascade { facet_classid: u32, tiers: [FacetTier; 6] }`, `size == 16` const-asserted | `facet.rs:92-106` | shipped |
| `shared_prefix_tiles` — whole-facet LCP, `(xor).trailing_zeros()/16`, one `vpxor`+`tzcnt` over `u128` | `facet.rs:255-262` | shipped |
| `hi_distance` / `lo_distance` — per-axis `6 − shared prefix` | `facet.rs:238-248` | shipped |
| `cascade_group_shared(other, shape, group)` — per-group LCP redout | `facet.rs:326-335` | shipped |
| `CASCADE_UNITS = 12`; `CascadeShape` G·D=12 (`6×2` / `4×3` / `3×4`) | `facet.rs:342`, `:344-546` | shipped |

### 1.3 The nibble trie (contract crate)

| Element | Anchor | Note |
|---|---|---|
| `NiblePath { path: u64, depth: u8 }`, `FAN_OUT=16`, `MAX_DEPTH=16` | `crates/lance-graph-contract/src/hhtl.rs:40,45,56,63` | shipped |
| `parent` / `prefix` / `is_ancestor_of` / `common_prefix_depth` / `common_ancestor` / `family_hop_count` | `hhtl.rs:155,223,176,251,459,414` | shipped, pure bit ops |
| `from_guid_prefix_v3` — **FUSES both axes**: `heel = b[4] \| b[5]<<8`, … `path = (heel<<48)\|(hip<<32)\|(twig<<16)\|leaf` | `hhtl.rs:386-402` | **the ARC A′ conflict**: one leaf gets ONE fused route, not two loci |

### 1.4 Compute-side, already shipped in ndarray

`/home/user/ndarray/src/hpc/clam_v3.rs` — the module doc is the closest
existing statement of this plan's thesis and is quoted verbatim in §2:

| Element | Anchor |
|---|---|
| module doc: *"which bytes are live is a field-mask question, not a parsing question"* | `clam_v3.rs:38-40` |
| module doc: `d(a,b) = depth(a) + depth(b) - 2·depth(lca(a,b))` | `clam_v3.rs:35` |
| `RailSpec` (byte-range + axis handle a caller derives from its ClassView / WideFieldMask **and passes in**) | `clam_v3.rs:114`, `v3_facet` `:133`, `slab` `:146`, `stacked` `:157` |
| `RailSpec::depth` / `lca_depth` / `geodesic` | `clam_v3.rs:196`, `:210-221`, `:225-229` |
| honest limits: bounded depth ⇒ pseudometric; a DAG is not a tree | `clam_v3.rs:57-70` |

### 1.5 The candidate table rows

| Row | Table / binding | Anchor | State |
|---|---|---|---|
| **Location** | `GEO_V3_FACET` — *"Rails 0–3 are the four HHTL cascade tiers (heel/hip/twig/leaf), each a `256×256` tile with x and y bound literally. Rails 4–5 are the identity tail"* | OGAR `crates/ogar-osm/src/lib.rs:204-212` (`FACET_RAILS = FACET_PAYLOAD_BYTES / 2`, `:200-201`) | **MINTED** (OGAR #249) |
| **Taxonomy / Mereology** | `RailCarving::zero_fallback` — rail pair, Taxonomy on byte 0, Mereology on byte 1 | `rail_geometry.rs:80-88`; catalogue rows L1–L3 in `le-contract.md:56-58` | **MINTED** as a carving; **unbound** to any thinking table |
| **NARS reasoning** | `NarsTables { revision: Vec<[PackedTruth; 256*256]>, deduction: [PackedTruth; 256*256], c_levels }`; `deduction[f1*256 + f2]`; revision table selected by **c-quantile** | `crates/causal-edge/src/tables.rs:41-49`, `:124-126`, `:115-119` | table **SHIPPED**; the rail **binding is MISSING** |
| **Rung ladder** | — | see §3 | **the one genuinely new design decision** |
| **Traversal** | not a row — the *reading column* (§2.3) | `lance-graph-java/CLAUDE.md:25` | ruled |

### 1.6 The 256:256 companion tables (why the pair byte is the right unit)

`le-contract.md:59` L4 — *"each byte pair indexes the 256×256 palette
distance/compose tables (bgz17 lineage); similarity = ONE table read"* — is
structurally the same shape as `NarsTables.deduction[f1*256 + f2]`
(`tables.rs:124-126`). **One `(u8:u8)` rail ⇒ one 256×256 companion table ⇒
one indexed read.** That correspondence, not an analogy, is what makes the
thinking table a *table* rather than a metaphor.

---

## §2 THE PROPOSED RESOLUTION

### 2.0 The contract, in one sentence

> **Coarse-to-fine runs in ascending little-endian order** — across rails as
> tiers, across bytes as levels, within a byte as nibbles — **so ancestry is
> always a prefix and every hierarchy question is AND / XOR / shift / tzcnt on
> the 16-byte key. The only lookup is `classid → ClassView`, once per class.**

Three clauses, each load-bearing:

- **"ascending little-endian order"** — the ordering is the contract. Today
  it is *true but unstated*: `read_path` walks `i = 0..max_depth` and stops at
  the first zero (`rail_geometry.rs:140-147`); `shared_prefix_tiles` reads
  `trailing_zeros()/16`, i.e. the LOW tile first (`facet.rs:255-262`);
  `morton()` puts `lo` on even bits (`facet.rs:62-64`). Nothing *states* that
  these must agree, so nothing catches it when a new reading disagrees.
- **"ancestry is always a prefix"** — the hole rule (`[1,0,7]` is depth 1)
  is what makes prefix-containment sound. It is tested once
  (`rail_geometry.rs:253-267`) for one carving.
- **"the only lookup is `classid → ClassView`"** — this is the anti-
  label-lookup clause (F4). No ancestor query may consult a name table, a
  string, an interned id, or a materialized closure.

### 2.1 Seven primitive ops — and their existing homes

| # | Op | Meaning | Existing home | Gap |
|---|---|---|---|---|
| 1 | `level(key, rail, i)` | byte read at level `i` of one rail | `RailCarving::level` `rail_geometry.rs:103-129`; `RailSpec::level` (ndarray) | **private in both** — no public contract name |
| 2 | `parent_rails(key, rail)` | zero the deepest live byte | prose + law only: hole rule `:131-149`, `is_ancestor_of` `:178-180` | **no `parent()` on `RailPath`** — this is the "parent-of-x is a bit op" the operator names |
| 3 | `parent_cartesian(key, rail, k)` | per-axis shift by `2k` (drop `k` Morton nibbles) | `FacetTier::morton()` `facet.rs:54-64` | **no home** for the *shift*; `morton()` itself has **zero consumers** |
| 4 | `is_ancestor` | per-rail mask+compare; whole-facet prefix | per-rail `RailPath::is_ancestor_of` `:178-180`; whole-facet `shared_prefix_tiles` `facet.rs:255-262` | the two are **unrelated in source** |
| 5 | `lca_depth` | leading-agreement count | `RailSpec::lca_depth` `clam_v3.rs:210-221`; `NiblePath::common_prefix_depth` `hhtl.rs:251`; `cascade_group_shared` `facet.rs:326-335` | **three implementations, no shared law** |
| 6 | `geodesic` | `depth(a)+depth(b)−2·lca` | `RailSpec::geodesic` `clam_v3.rs:225-229`; `NiblePath::family_hop_count` `hhtl.rs:414-418` | contract crate has no rail-side geodesic |
| 7 | `project(key, wfm)` | AND with `WideFieldMask` | `WideFieldMask` `class_view.rs:221`; `ClassView::project` `:933` | **not yet the selector for ops 1–6** |

**Reading of this table:** five of seven ops are shipped somewhere; the work
is *naming the law they share*, not writing them. Ops 2 and 3 are the two
genuine additions, and both are single expressions.

### 2.2 ~~Two addressing modes over the SAME bytes~~ ⊘ WITHDRAWN

> **⊘ THE TWO-MODE TAXONOMY IS DELETED AS A CONCEPT (operator, 2026-08-19).**
> There is **ONE HHTL address fabric**. Different consumers may READ it
> differently, and Cartesian is one such read/projection where useful — but
> it is **not** another address carrier, another hierarchy, another storage
> mode, another HHTL variant, or a competing parenthood model. **Morton is
> unrelated to Cartesian here** and is not canonical: it is the separate
> `4⁴` construct formerly called "nibble". Do not derive HHTL ancestry from
> Morton, do not define `parent_cartesian` from it, and do not use it to
> justify Cartesian hierarchy semantics.
>
> The table below is retained **only** as the record of a withdrawn model.
> Its one surviving observation is operational and worth keeping: a reader
> that mis-reads a rail's *zero* — data vs terminator — gets a well-formed,
> plausible, wrong answer (`clam_v3.rs:16-19`). That hazard is real and
> belongs to the ClassView's reading, not to a mode enum.



| | **Cartesian / cesium** | **Rails** |
|---|---|---|
| A byte pair is | `(x, y)` in a `256×256` tile | byte `i` = the level-`i` occupant |
| Ancestry within a byte | nibble = 4-ary centroid step (`256 = 4⁴`, OGAR canon) | n/a — the byte is atomic |
| Parent | **shift** (drop the finest Morton nibble) | **truncate** (zero the deepest live byte) |
| Zero means | a real coordinate (`(0,0)` is a tile) | **a hole** — the chain ends |
| Depth source | fixed by rail index (zoom tier) | counted (`read_path`) |
| Shipped anchor | `morton()` `facet.rs:54-64`; `GEO_V3_FACET` rails 0–3 = HHTL tiers ⇒ **rail *is* the zoom choice** | `RailCarving` + `RailPath`, `rail_geometry.rs` |

**These two do not compose and must never be silently mixed.** In Cartesian
mode `0` is data; in Rails mode `0` terminates the walk. A reader that picks
the wrong mode gets a well-formed, plausible, wrong answer — the exact defect
class `clam_v3.rs:16-19` already names (*"it looks plausible because every
number is well-formed"*). **The mode is a per-rail property resolved through
`ClassView`, exactly as `edge_codec_flavor` and `rail_carving` already are.**

### 2.3 The thinking table

**It is ClassView-resolved DATA rows, not a type.** One row per
`(classid, rail)`; the columns are: *rails used · reading · axis semantics ·
companion 256×256 table*. Traversal is not a fifth row — it is the
**reading column**, and `hop = Mask × ClassView/WideFieldMask → Mask` (the
ratified lance-graph-java invariant, `lance-graph-java/CLAUDE.md:25`) is what
a traversal *is* once the class's reading is resolved. (*`reading`, not
`mode` — renamed 2026-08-19 with §2.2's withdrawal: one fabric, several
ClassView-resolved reads, never alternative addressing systems.*)

> **⊘ COLUMN RENAMED 2026-08-19.** The third column was `Mode` and named
> `Cartesian`/`Rails` as if they were alternative address carriers. They are
> not: there is ONE fabric, and these are **ClassView-resolved READINGS** of
> the same bytes. The column is now `Reading`; the values name what a
> consumer does with the pair, never a second addressing system.

| Row | Rails | Reading (ClassView-resolved) | Axis semantics | Companion table | Mint state |
|---|---|---|---|---|---|
| **Location** | 0–3 (tiers), 4–5 tail | coordinate-pair read (a byte pair as `x : y`) | `x : y`, literal | tier centroid tables | **MINTED** — OGAR `ogar-osm/src/lib.rs:204-212` |
| **Taxonomy / Mereology** | 4 (canon default) | level-occupant read (byte `i` = level `i`, `0` = hole) | `part_of : is_a` | — (prefix only) | **MINTED as a carving** (`rail_geometry.rs:80-88`), unbound |
| **NARS reasoning** | *(unassigned)* | coordinate-pair read | `f₁ : f₂` (frequency pair) | `NarsTables.deduction[f1*256+f2]` `tables.rs:124-126`; `revision` selected by c-quantile `:115-119` ⇒ **c IS the zoom axis** | table shipped, **rail unminted** |
| **Rung ladder** | *(unassigned)* | see §3 — **two axes** | see §3 | see §3 | **unminted, undesigned** |

The NARS row is the cleanest evidence that the shape is right: a 256×256
table indexed by a byte pair, with a *third* quantile axis selecting which
table — which is precisely "rail = zoom" (`GEO_V3_FACET`'s rails 0–3) arriving
independently in a different subsystem.

---

## §3 THE RUNG CARVE — two axes wearing one name (session-measured)

Charter §12 says *one canonical rung vocabulary*. Measured this session, the
shipped code carries **two different quantities** under that one name. A rung
row that does not say **which byte carries which** re-creates the conflation
one layer down, in bytes, where it is far more expensive.

| Axis | What it is | Anchor | Shape |
|---|---|---|---|
| **(a) Admissibility floor** | the earliest rung at which a tactic may fire, derived from `Bucket` (cost/role) and `Tier` | `crates/lance-graph-contract/src/recipes.rs:497-523` (`Recipe::min_rung`), `:525-531` (`admissible_at`) | **monotone**, total order, **no SPO involvement** — *"Monotone: once admissible, a deeper rung never withdraws it"* (`:525-526`) |
| **(b) Plane projection** | which SPO planes the rung's Pearl level consults | `crates/lance-graph-contract/src/cognitive_shader.rs:244-250` (`RungLevel::causal_mask_bits`): L1→`0b001`, L2→`0b011`, L3→`0b111` | a **3-bit set**, not an ordinal |

(a) is an ordinal you compare; (b) is a mask you AND. Putting them in the
same byte means a comparison and an intersection read the same bits.

**Two pre-existing hazards the rung row must carry, not inherit silently:**

1. **rung-2 → 144 verbs is BLOCKED.** `recipes.rs:489-493`, verbatim:
   *"this wires rung → tactic admissibility. It does NOT wire rung 2 → the
   144 verb atoms, which stays blocked on O7 (`sigma_rosetta` and
   `verb_table` carry divergent 144 vocabularies with skewed ordinals —
   `TD-RUNG2-144-VOCAB-SPLIT`). Nothing below reads either vocabulary."*
   A rung rail that addresses verb atoms would consume the split.

2. **The L1 mask is a hand-chosen convention that CONTRADICTS `pearl.rs`.**
   `cognitive_shader.rs:239-242` says so itself — *"Level 1 → `O = 0b001` …
   CONVENTION, hand-chosen pending its own probe"*. But
   `crates/causal-edge/src/pearl.rs:40-42,75` defines **Level 1 Association =
   `SO = 0b101`** (Subject + Object). Under `pearl.rs`'s reading the ascent
   is **not superset-monotone**: `0b101 ⊄ 0b011`. Under
   `cognitive_shader.rs`'s it is: `0b001 ⊂ 0b011 ⊂ 0b111`. Monotone superset
   ascent is exactly the property a *prefix* contract would want, so this is
   not cosmetic — **D-HTT-9 is the probe, and it gates the rung row.**

---

## §4 DELIVERABLES

Every deliverable names, before it starts, what would falsify it.
D-HTT-1 … D-HTT-4 are documents; 5 … 8 are contract text; 9 … 11 are probes.
**Nothing here is code until §0 F6 is lifted.**

| id | Deliverable | Pre-registered gate / falsifier |
|---|---|---|
| **D-HTT-1** | **The LE ordering law**, written as contract prose: ascending rail → ascending byte → ascending nibble, one sentence, with the three shipped sites it already describes (`rail_geometry.rs:140-147`, `facet.rs:255-262`, `facet.rs:62-64`). | **FALSIFIED IF** any shipped reading walks the opposite direction and is nonetheless correct — i.e. if the "law" is a description of two of three sites and an accident at the third. Requires an explicit read of all three orderings and a statement of agreement or disagreement, not an assertion of agreement. |
| ~~**D-HTT-2**~~ ⊘ | ~~**Mode taxonomy** — Cartesian vs Rails as a per-rail, ClassView-resolved property.~~ **WITHDRAWN 2026-08-19** with §2.2: one fabric, many reads; a "mode" enum would be the second addressing abstraction the rulings forbid. **REPLACED BY D-HTT-2′** — state the *zero-semantics* hazard as a ClassView reading obligation (a rail's `0` is data or terminator per the class's reading, never per a global mode), with the `clam_v3.rs:16-19` plausible-but-wrong defect as its rationale. | **D-HTT-2′ FALSIFIED IF** any shipped reading resolves a rail's zero-meaning from something *other* than the class's own reading — i.e. if a global switch is load-bearing anywhere. |
| **D-HTT-3** | **The thinking-table row schema** (rails · **reading** · axis semantics · companion table — `reading` is the ClassView-resolved interpretation of a byte pair, **not** an addressing mode; renamed 2026-08-19 with §2.2's withdrawal), plus the four candidate rows of §2.3 with their mint states. | **FALSIFIED IF** the Location row (already minted, `ogar-osm/src/lib.rs:204-212`) cannot be expressed in the schema without an extra column no other row uses — order-genericity's local form (ARC A′ `F-ORDER-GENERICITY`, and its kill condition: *do not widen the shape to rescue it*). |
| **D-HTT-4** | **Op catalogue** — the seven ops of §2.1 with, for each, either its shipped home or an explicit "no home" mark. | **FALSIFIED IF** any op marked "no home" turns out to have one (repo-wide grep required, both repos), or any op marked "shipped" is private/unreachable without a signature change that the catalogue does not admit. Note ops 1's home is **private in both crates** — the catalogue must say so. |
| **D-HTT-5** | **`parent_rails`** named in the contract as *zero the deepest live byte* — one expression, sitting beside `RailPath::is_ancestor_of`. | **FALSIFIED IF** `parent_rails(x)` is not always an ancestor of `x` under `is_ancestor_of` (`rail_geometry.rs:178-180`), or if applying it `depth` times does not reach the empty path (*"leerer Pfad = dominante Wurzel"*, `:264-266`). |
| ~~**D-HTT-6**~~ ⊘ | ~~**`parent_cartesian`** named as a per-axis shift, with `morton()` as its stated basis.~~ **WITHDRAWN 2026-08-19 (operator ruling C).** Morton is not canonical and HHTL ancestry is never derived from it; a `parent_cartesian` defined *from* Morton is precisely the derivation the ruling forbids. The `morton()` primitive stays shipped, unconsumed, and **non-canonical research** (X2 below is regraded accordingly) — it is not repaired into the HHTL contract. Nothing replaces this deliverable: the one parent operation the contract needs is the rails truncate (D-HTT-5). | n/a — withdrawn, not re-gated. |
| **D-HTT-7** | **Ancestry-is-prefix law** stated once, covering both the per-rail form and the whole-facet form (`shared_prefix_tiles`), with the hole rule as its precondition. | **FALSIFIED IF** the two forms disagree on any pair: a pair that is a per-rail ancestor but not a whole-facet prefix (or vice versa) means the "one law" is two laws. This is a *real* possibility — `shared_prefix_tiles` includes the classid tiles (`facet.rs:250-252`); the rail walk does not. **Expect this to fire; the deliverable is the honest scope statement, not a forced agreement.** |
| **D-HTT-8** | **The rung row, carved** — an explicit statement of which byte carries the (a) admissibility ordinal and which carries the (b) plane mask, per §3, or a ruling that the rung gets no rail at all. | **FALSIFIED IF** any single-byte proposal survives the question *"does comparing this byte with `<` mean the same thing as ANDing it?"* — if both readings are live on one byte, the carve failed. Also **BLOCKED-BY** D-HTT-9. |
| **D-HTT-9** | **PROBE-RUNG-L1-MASK** — resolve `cognitive_shader.rs:244-250` (L1 = `0b001`) against `pearl.rs:40-42,75` (L1 Association = `SO = 0b101`). Decide which is canon and whether superset-monotone ascent is a required property of the rung ladder. | **PASS** = one reading is chosen, the other is regraded in place (append-only), and the monotonicity property is either asserted with a test or explicitly disclaimed. **FAIL** = the probe cannot discriminate — then D-HTT-8's rung row must NOT be minted, and the plan says so rather than picking. |
| **D-HTT-10** | **PROBE-NARS-RAIL** — does binding the NARS row to a rail pair `(f₁:f₂)` with c-quantile as the zoom axis (`tables.rs:115-119`) actually reproduce `NarsTables::deduce`/`revise` results, or does the byte pair lose information the table needs? | **PASS** = a rail-addressed read is bit-identical to the direct table call across the full `f₁ × f₂` grid at fixed c. **FAIL** = any divergence — record it; a NARS rail that is *nearly* right is worse than none, because both sides are well-formed `u8`s. |
| **D-HTT-11** | **PROBE-ONE-SHAPE** (the ARC A′ §5 minimal experiment, localized) — can one row schema carry Location (coordinate-pair reading, minted) and Taxonomy/Mereology (level-occupant reading, minted) with **no case needing a column the other cannot use**? | **PASS** = both express fully in the D-HTT-3 schema. **FAIL** = report it and do NOT widen the schema (ARC A′'s pre-registered kill condition, verbatim: *"if expressing Case A and Case B through one shape requires either case to carry a field the other cannot use, the shape is not agnostic and the experiment has failed — report that, do not widen the shape to rescue it"*). |

**Ordering** (updated 2026-08-19 — D-HTT-6 withdrawn, D-HTT-2 → 2′).
D-HTT-1 → 2′ → 3/4 (documents, independent) → 9 (unblocks 8) → **5/7** → 8 →
10/11. D-HTT-11 is the cheapest thing that can kill the whole
program and should not be sequenced last out of politeness.

---

## §5 ARC C — the Java mirror (GATED on ARC B ratification)

Not started; listed so the seam is visible, per charter §19's "prepare the
seams now, do not implement."

- **C-HTT-1 — ONE new mask source.** `parentOf` / `ancestorsOf` as **mask
  transforms**, not row-id returns: `Mask × ClassView/WideFieldMask → Mask`,
  the shipped `RowStore.hop(int, WideFieldMask, Mask)` shape
  (`lance-graph-java/java/src/main/java/com/adaworldapi/lancegraph/RowStore.java:163-171`).
  A `long[]` frontier is forbidden as normal execution state; the only named
  exceptions are `importRows` (`RowStore.java:202`) and `materializeRows`
  (`Mask.java:101`) — both already classified in the Java repo.
- **The missing-capability STOP rule applies.** If the Java facade needs a
  primitive the substrate lacks, it does **not** hand-roll it one layer up —
  the capability lands as a substrate-tier change first
  (`lance-graph-java/CLAUDE.md` § "Missing-capability STOP rule";
  `docs/abi.md` §6). Concretely: `parentOf` needs D-HTT-5 *in the contract*
  before it can exist in Java.
- **No new ABI symbol is proposed here.** Whether ops 2/3 need one is a
  question for ARC C, after ARC B says what they are.

---

## §6 NON-GOALS

1. **No new carrier type.** No `HierarchyPlane`, no higher-order struct, no
   promotion DTO, no "structural algebra" crate, no second SoA (F3). ARC A §7
   already concluded *"ZERO new types are proposed"*; nothing found this
   session reopens that.
2. **No stride change.** `NODE_ROW_STRIDE` and the 16-byte facet are canon
   and untouched (`facet.rs:103-106` const asserts; `le-contract.md:65-67`).
3. **No `ENVELOPE_LAYOUT_VERSION` bump.** Every reading here reads bytes that
   already exist, selected per class — the same discipline
   `EdgeCodecFlavor`/`rail_carving` already follow.
4. **No reasoner.** ARC A′ §1 confirmed there is none in-repo, correctly;
   `ontology_warrant.rs` is the consumer-side grading contract for an
   external factfinder. A prefix is not an entailment.
5. **No semantics in storage.** The rung/NARS/location *meanings* live above;
   the substrate carries bytes and a per-class reading (charter §0, §17).
6. **No mint.** This plan proposes no classid, no `ValueTenant`, no rail
   assignment. Mint is gated (§8 Q3).
7. **No freeze, no barrier, no scheduler.** Untouched (charter §7, §8).
8. **No V1-tail revival.** The u24 tail stays read-only; no widening of any
   `u8:u8` (F2).

---

## §7 DEFERRED — missing integration

Each item is a *known* disconnect. None is closed by this plan; all are
listed so a future session does not rediscover them.

| # | Item | Evidence | Why deferred |
|---|---|---|---|
| **X1** | **`NiblePath` fused-axis vs per-axis contract layering — UNRULED.** `from_guid_prefix_v3` packs BOTH the `part_of` (hi) and `is_a` (lo) byte of each tier into one `u64` (`hhtl.rs:386-402`), so all ancestor arithmetic runs on the fused value and no per-axis constructor exists. `rail_geometry.rs` never mentions `NiblePath`; `hhtl.rs` never mentions `RailPath`. | verified this session; ARC A′ §3 Case A "**CONFLICT**"; ARC A′ open question 1 | This is the single largest fork and it is an operator ruling, not a finding. §8 Q1. |
| **X2** ⊘ | **`morton()` has no consumer, and under the 2026-08-19 ruling it acquires none here.** Shipped `facet.rs:54-64`, referenced only by its own test `:642-643`. ~~It is the *entire* basis of Cartesian-mode parenthood (D-HTT-6).~~ **CORRECTED:** Morton is the separate `4⁴` construct, **non-canonical**, and HHTL ancestry is never derived from it; D-HTT-6 (which would have consumed it) is WITHDRAWN. It stays shipped and unconsumed as **non-canonical research**. | grep verified, both repos | Deferred as research, not as a blocked deliverable. Nothing in the HHTL contract depends on it. |
| **X3** | **The basin-promotion seam (Type-B: discovered → promoted) DOES NOT EXIST.** `EpisodicMemory::basins()` (`crates/lance-graph/src/graph/arigraph/episodic.rs:243`) returns `EpisodicBasins` (`:79`) by value; verified callers are **only its own tests** (`:742,753,761,768`). No `ValueTenant` slot is reserved. | verified this session; ARC A′ §3 Case B | A promoted basin is exactly a *new* thinking-table row with no minted rail. Blocked on mint (§8 Q3) and on ARC D. |
| **X4** | **The latent third mask basis at the Java ABI.** `native/lgj-abi/src/abi.rs:173-175` fixes 32 facet lanes; `fixture.rs` admits the canonical per-class basis is "a later slice". A `WideFieldMask` minted in lane basis and one minted in class-field basis are the same type. | ARC A′ §2 "DOWNGRADED — the mask-collision instance" (live: `Locus` vs class-field; latent: the lane basis) | Op 7 (`project`) is the op that would cross the bases. The discrimination question is audited, not solved — deliberately, per ARC A′. |
| **X5** | **NARS and rung rail bindings await mint gating.** Both rows in §2.3 are "unassigned". `rail_carving` has **zero consumers** outside its own default (grep verified), so no bake currently overrides the canon `reg: 4` pair. | `class_view.rs:1127-1133`; `rail_geometry.rs:80-88` | Assigning a rail is a mint. §8 Q3. |
| **X6** | **`FamilyTrie` is a disconnected `u16` island.** `crates/deepnsm-v2/src/ancestry.rs:34` — a real parent-pointer forest with `dn()` (`:153`) and `is_ancestor_of` (`:172`), whose own doc states the same law as this plan (*"ancestry lives in the KEY … `is_ancestor_of(A, Z)` = A's DN is a strict prefix of Z's DN — radix-trie containment, the same law as the 4⁴ centroid-hierarchy canon"*, `:1-9`). Verified consumers: only inside `deepnsm-v2` (`lib.rs:54`, `shape.rs:38,224`, two examples). **No crate depends on `deepnsm-v2`.** | verified this session; ARC A′ §5 names it as one half of the minimal experiment | It already implements shared ancestry correctly, over the wrong key space (`u16`, not `NodeGuid`). Porting vs generalizing vs leaving it is ARC A′ open question 2 — unresolved, §8 Q4. |
| **X7** | **`RailCarving::level` is private in both crates.** Op 1 has an implementation and no public name. | `rail_geometry.rs:103` (`fn`, not `pub fn`) | A signature change, however small, is code. F6. |

---

## §8 OPEN OPERATOR DECISIONS

Only genuine forks; nothing here is decidable from source.

**Q1 — `NiblePath` fused vs per-axis (X1).** Is the fused v3 route canon,
with rails a *second, separate* reading at a different layer? Or must the
routing prefix become per-axis so one leaf carries two loci? Both are
currently shipped and neither references the other. *This blocks the whole
rail-truncate ancestry story at the trie layer, though not at the rail layer
("Rails-mode" renamed 2026-08-19 — there is no mode, only the class's
reading).*

**Q2 — does the rung get a rail at all?** §3 shows the rung is two
quantities. Three options: (i) two bytes, one rail, explicitly carved;
(ii) one byte for (a) only, with (b) staying a computed projection from the
ordinal (`causal_mask_bits` is already a pure function of the rung —
`cognitive_shader.rs:244-250`); (iii) no rail — the rung stays a dispatch
parameter and never enters the register. **(ii) is the smallest and is
consistent with charter §12's "one canonical rung vocabulary"** — but it is
the operator's call, and it is gated on D-HTT-9 either way.

**Q3 — mint gating.** NARS and rung rows need rail assignments; a promoted
basin (X3) needs a `ValueTenant`. Does ARC B *propose* assignments (to be
minted later in one batch), or does it stop at the schema and leave every
row unassigned? The Location row is already minted upstream, so the schema
can be exercised on one real row either way.

**Q4 — `FamilyTrie` (X6).** Port to `NodeGuid`, generalize over the key
space, or leave it and build the contract beside it? ARC A′ left this open;
this plan does not narrow it.

**Q5 — D-HTT-7's expected disagreement.** If the per-rail and whole-facet
ancestry forms genuinely disagree (classid tiles included in one, not the
other — `facet.rs:250-252`), is that a defect to fix or a scope boundary to
document? *Recommendation: document. Forcing agreement would mean either
dropping the classid from the whole-facet prefix or adding it to the rail
walk, and both change a shipped, tested behaviour to satisfy a document.*

---

## §9 What this plan does NOT claim

- It does not claim the seven ops compose into a working traversal —
  D-HTT-11 is the probe that would show it, and its kill condition is real.
- It does not claim the LE ordering is currently consistent across the three
  shipped sites — D-HTT-1's falsifier exists precisely because that was
  *observed at three sites*, not *proven across all readings*.
- It does not claim the rung belongs in the register at all (Q2 option iii).
- It does not claim `morton()`'s quad-tree property is measured — it is
  documented (`facet.rs:54-64`) and has zero consumers (X2). **The
  deliverable that would have measured it (D-HTT-6) is WITHDRAWN**, so this
  stays unmeasured *by design*: Morton is non-canonical and the HHTL contract
  does not rest on it.
