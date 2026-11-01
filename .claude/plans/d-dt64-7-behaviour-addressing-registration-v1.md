# D-DT64-7 — Behaviour addressing and registration: Frozen, Learned, Explore over one classid

**Status:** PROPOSED, docs only. No code, crate, ABI or test lands with this plan. Each claim is labelled:
- **DECISION**, with who decided and when;
- **VERIFIED-IN-CODE**, with its location on `main` `b38d88e`;
- **WORKING-MODEL** (the design this plan proposes);
- **OPEN** (undecided; carries an owner).

**Date:** 2026-10-31. **Parent work:** D-DT64-6 (`d-dt64-6-routing-bindstore-behavior-v1.md`, #1869, #1871, #1883), D-LITHO (`entries/2026-10-30-d-litho-photolithography-contract.md`), D-DYNBAG (`d-dynbag-holographic-bags-v1.md`), the DR256 WHERE capstone §C (`dr256-where-capstone-v1.md`), the V3 mandatory hot-plug reading (`v3-mandatory-hotplug-reading-v1.md`), #1892/#1897 (R1 binding), #1896 (XLOG × egglog).

**What this plan settles:** where agnostic and registered behaviour lives, how a classid addresses it, who may register and withdraw it, and what that means for the current dependency of DT64 execution on `ogar-r2il`. **What it does not settle:** selection and tiling (D-DYNBAG), routing and frames (D-DT64-6, DR), the traceroute (DR256 capstone §C.5).

---

## 1. Decisions carried (operator, 2026-10-31, in session)

| # | decision |
|---|---|
| D1 | Behaviour for fixed relalg has three categories: **Frozen**, **Learned**, **Explore**. |
| D2 | **Frozen = compile time.** **Learned = registered in the `BindStore`.** **Explore = not yet registered.** There is no lifecycle state besides versioned, replayable immutability. |
| D3 | ONNX, SQL and Gremlin belong to Frozen: their semantics are declared by an external specification. The R2IL codegraph harvest (and analyzer output) belongs to Learned / Explore. |
| D4 | XLOG is Learned during the lab (`xlog-relcomp-probe`) and becomes Frozen once migrated to production by minting `lance-graph-xlog`, where it is permanently registered in lance-graph. This is consistent with #1888's "no `lance-graph-xlog` crate until a measured source gap exists": that rule governs the lab phase; the mint is the migration. |
| D5' | Registration needs common-sense constraints: anything that runs code through the `BindStore` (a code analyzer, an R2IL macro) can register **and unregister**. |
| D6' | **Left side empty (`0x0000`) = hardcoded** behaviour. **Left side = a `Domain::AppId` concept = registered by that domain/app.** Registered behaviour may also take a left side in the `0xF###` range. |
| D7' | The **right side** (low 16 bits) is ClassView / `WideFieldMask` selection and routing within the concept, not render only: Quack addresses a mask from the DR versioning store for the current bag, comparable to DuckDB reusing a bag selection. The right side's `0xF###` range addresses behaviour selections. |
| D8' | `0x1000` (the V3 marker) is deprecated, from when V3 was opt-in. Recorded earlier: `v3-mandatory-hotplug-reading-v1.md` item 5 (2026-10-10, "deprecated, not evicted"). |
| D9' | Graph work is mask selection, not traversal: a selection mask before DR/DT and a selection mask after DR/DT. |
| D10' | The DR/DT global → local selection must be a versioned lance-graph table, accessible through `ClassView × WideFieldMask` if nothing better exists. Recorded earlier as D-LITHO Q5 ("which table" open for the DR/DT owner). |

The primes mark decisions numbered in this plan, not D-DT64-6's D4–D6.

### 1a. Revision: hardcoded behaviour is `0xFFFF::0xF###`, and DT64 owns its registration (operator, 2026-10-31, later in session)

| # | decision |
|---|---|
| D11'' | **Hardcoded behaviour is `0xFFFF :: 0xF###`**: left side `0xFFFF` (domain `0xFF`, app `0xFF`), right side `0xF###` naming the behaviour. **Supersedes D6'** "left side empty (`0x0000`) = hardcoded". |
| D12'' | The hardcoded families include the relalg polyfill, **XLOG** (after the `lance-graph-xlog` mint, D4), **ONNX** and **TFLite**; D3 already places SQL and Gremlin in Frozen. |
| D13'' | **DT64 is the delegate for registering behaviour classids for `lance-graph-ogar`, without asking `ogar-vocab`.** Separation of concerns is hardcoded by address: the address range says which concern a behaviour belongs to. |
| D14'' | **Consumers still register behaviour through `0xF###` on the right side, under their own `Domain::AppId` on the left side.** |
| D15'' | Assigning behaviour classids to DT64 means assigning the concern with them: DT64 holds the registration authority for that range, not only the storage of its bindings. |

WORKING-MODEL consequences, each checked against `main` `b38d88e`:

- **One classid per behaviour family, never per word.** `0xFFFF_F###` names one lens; its words stay field positions (`AlgebraRef { lens, word: u8 }`, `lance-graph-dt64/src/lib.rs` "Agnostic algebra" module doc: "No word gets a classid of its own"). This also answers the objection in `entries/2026-10-31-cross-frontend-projection-crosswalk.md` that a per-method slot would be a second spelling of `MethodRef`: `F###` addresses a lens, not a method. A lens holds at most `MAX_FIELDS = 256` words (`lib.rs`); a family larger than that takes more than one classid inside its range.
- **The compat misread no longer touches hardcoded behaviour.** `classid_canon_compat` returns the high half when it is `>= 0x0100` and not `0x1000` (VERIFIED-IN-CODE), so `0xFFFF_F000` reads concept `0xFFFF`. F5 remains true of legacy `0x0000_XXXX` ids only, which I-LEGACY keeps on purpose; D-DT64-7-1 and O3 no longer gate this plan.
- **The delegate needs no OGAR crate in DT64.** `CapabilityAuthority` lives in the zero-dependency contract (`hotplug.rs`), `lance-graph-dt64` depends only on the contract, and `lance-graph-ogar` does not depend on `lance-graph-dt64` (VERIFIED-IN-CODE, both `Cargo.toml`s). DT64 can implement `CapabilityAuthority` for concept `0xFFFF`; whatever composes the two authorities routes `0xFFFF` to DT64 and every other concept to `OgarAuthority` / `ogar-vocab`. Where that composition lives is O8.
- **Concern ranges inside `0xF###` (proposal, O7).** The second hex digit names the concern; the low byte gives up to 256 classids per concern, so a new revision of a frozen specification (an ONNX opset, a TFLite schema version) takes a new classid in its range, which is D5's "a changed hypothesis takes a new classid" applied to Frozen:

  | right side | concern | Frozen from |
  |---|---|---|
  | `0xF0__` | relalg / mask-risc polyfill (the lens `declare_algebra` declares today) | now (D-DT64-7-3) |
  | `0xF1__` | XLOG | the `lance-graph-xlog` mint (D4); in the lab XLOG registers under the probe's own `Domain::AppId :: 0xF###` (D14'') |
  | `0xF2__` | ONNX | when an ONNX surface lands |
  | `0xF3__` | TFLite | when a TFLite surface lands |
  | `0xF4__`, `0xF5__` | SQL, Gremlin (D3) | when each surface lands |
  | `0xF6__ ..= 0xFF__` | reserved | — |

### 1b. Polyglot aliasing over one algebra (operator, 2026-10-31, later in session)

| # | decision |
|---|---|
| D16'' | **Behaviour is normalized across languages** (C, Rust, Java, SQL and the other frontends): each language's operator names are **aliases over one algebra**, not separate behaviour. |
| D17'' | **A classid is cleaner than a hardcoded string.** Identity is the Frozen classid plus the word position; a name is never the key. |

VERIFIED-IN-CODE (`lance-graph-r2il/src/binding.rs` module doc, step 3, and `fn opcode`): today the reverse holds. `prepare` reads the word's label from the lens's `ClassView` and scans `R2ILVocabulary` for an opcode with that name; the module doc states it: "the position a lens gives a word is free and only the name binds". The string join is in `prepare` (cold). `run` (hot) re-resolves nothing.

WORKING-MODEL:

- **Position binds, labels alias.** The Frozen table at `0xFFFF :: 0xF0__` fixes each word's position and its semantics. Each language supplies a `ClassView` over the same positions whose labels are that language's names (SQL `SUM`, a Java method name, a Rust method name, a C symbol). This is the mechanism DT64 already documents ("labels are resolved late through that view"); only the direction of binding changes.
- **Normalization is a parity test, not a convention.** For every alias view, the label at a position must resolve to the same `AlgebraRef`; a name that maps to two positions, or a position with no name in a view that claims full coverage, refuses.
- **What this does to the concern table (O7).** A language that only renames the shared algebra gets an alias view, not a concern range. A concern range under `0xF###` is for behaviour with no equivalent in the shared algebra (tensor operators for ONNX and TFLite; whatever XLOG adds beyond the relational core). SQL's three-valued logic is a known case: `prepare` refuses `Truth::ThreeValued` today (`binding.rs`), so SQL's NULL behaviour either gets its own words or stays refused. The `F4__` / `F5__` rows for SQL and Gremlin are therefore provisional.
- **Capacity: 256 is per lens, not per namespace.** A word is `AlgebraRef { lens, word: u8 }` and a lens holds at most `MAX_FIELDS = 256` words (VERIFIED-IN-CODE, `lance-graph-dt64`). A concern range `0xF#__` holds up to 256 lens classids, so up to 65,536 words per concern. ⊘ **Corrected in §1c (H3):** DT64 keys a lens's label basis by its LEFT half only, so today every lens under `0xFFFF` shares one basis of at most 256 words, and the 65,536 figure needs O9. A large operator set therefore spans several lenses in its range; it does not need a wider word. Operator figure: TensorFlow (not Lite) has about 1,500 ops, which is about six lenses. Figures relayed from an outside summary, SECONDHAND and not recounted here: GGML graph opcodes 102, Burn IR 212 shared operation kinds (328 counting float/int/bool variants separately), ONNX about 180.
- **Datatype belongs to the lens, not the word.** The Burn IR count above is lower than its variant count because one operation kind serves several tensor types. The same split holds here: an `Effect` is a word plus ordered operands (`lance-graph-dt64`), and the operand types are the `ClassView`'s fields. A dtype-specific variant is not a new word.
- **The semantic union is the open question, not the ceiling.** After aliasing (above), how many distinct words do relalg, R2IL, XLOG, ONNX, TF/TFLite, GGML and Burn need between them, and which belong in the shared lens versus a concern range? That count decides the layout of `0xF0__` and the tensor ranges (D-DT64-7-7).
- **Calling overhead (HYPOTHESIS, unmeasured).** Binding by position removes the label scan from `prepare`. Nothing in the code shows a string lookup on the hot path, so any per-call cost on that path comes from somewhere not yet identified. The probe (D-DT64-7-6) reports the cost of `prepare` and of one `run`, each in nanoseconds, before and after; per the workspace rule, a millisecond reading is a failed gate, not a result.

### 1c. Label hydration: from a consumer codebook to DT64 (2026-10-31, later in session)

Question (operator): does the `hotplug.rs` → `ogar-vocab` → `lance-graph-ogar` delegation need planning so that labels from consumer codebooks reach DT64? **Yes.** Five findings, VERIFIED-IN-CODE on `main` `b38d88e` and OGAR `2a080a9`:

- **H1. Activation carries no labels.** `hotplug::Activation` holds `(concept name, id)` rows, capability names, read modes and register laws. A field basis (the per-position labels a `ClassView` returns) does not travel through activation. The `ClassView` DT64 reads is whatever the caller passes, and nothing ties it to the codebook that activated the concept.
- **H2. `OgarClassView` hydrates only OGAR's promoted concepts.** `ogar-class-view` builds its map from `ogar_vocab`'s `all_canonical_classes()`; any other class gets an empty basis. A consumer codebook (medcare-rs, woa-rs, …) has no route into it. The other `impl ClassView`s in the tree serve their own sources: `RegistryClassView` (OGIT cache, `lance-graph-ontology`), `WikidataClassView`, `CatsSchema` (`lance-graph-sap`).
- **H3. DT64 keys the basis by the left half.** `lance_graph_dt64::view_class` is `classid_concept`, and `check_basis` reads `view.fields(view_class(class))`. Consequences:
  - every lens under `0xFFFF` (`F000`, `F100`, `F200`, …) shares one basis of at most 256 words, so the right side can only select within it, which is D7' (right side = `WideFieldMask` selection);
  - a consumer's behaviour lens `Domain::AppId :: 0xF###` shares the basis of that consumer's data class `Domain::AppId`, so data fields and algebra words would sit in one basis.
- **H4. No view composes the sources.** A binary that runs DT64 passes one `ClassView`, which must answer canon concepts (OGAR), consumer concepts (the consumer's codebook) and `0xFFFF` (the polyfill aliases). Nothing in the tree routes by concept across those sources.
- **H5. An unknown class fails closed.** An empty basis makes DT64 refuse with `FieldBasisChanged` (count mismatch): absence is a refusal, not an empty answer. This holds and needs no change.

WORKING-MODEL items (not started; each names the decision it needs):

| item | what | needs |
|---|---|---|
| L1 | Key behaviour-lens bases by the full classid: a `ClassView` lookup by `u32` (a defaulted contract method that falls back to the concept half), used by DT64 for lenses whose right side is `0xF###`. Without it, a concern range cannot exceed 256 words, and a consumer's behaviour words share its data basis. | O9 |
| L2 | A composed `ClassView` routing by left side: `0xFFFF` → the polyfill alias view for the caller's language (L4); canon concepts → `OgarClassView`; a consumer's `Domain::AppId` → the view built from that consumer's codebook. It lives where the composing authority lives. | O8 |
| L3 | Tie basis to activation: DT64 refuses to resolve a class whose concept the caller's `Activation` does not cover (C2, extended from registration to reading). | O8 |
| L4 | Ship the polyfill lens's language alias views (R2IL, SQL, Quack) as a library surface derived from the `POLYFILL` table (#1899), instead of test-only views. | — |
| L5 | How a consumer codebook hands its labels over: the consumer implements `ClassView` itself (as `CatsSchema` does), or the authority returns label rows with the activation. The second changes `Activation`; the first needs no contract change. | O10 |

### 1d. Decisions O7–O10 (operator, 2026-10-31, after #1898–#1900 merged)

| # | decision |
|---|---|
| O7 | **Concern ranges.** `F0__` shared relational algebra (relalg / mask-risc polyfill); `F1__` XLOG-exclusive semantics, only after promotion; `F2__` ONNX; `F3__` the TensorFlow family including TFLite (shared concern, not necessarily shared lenses: equal names do not imply equal semantics); `F4__` genuine SQL-specific semantics (for example `NULL`); `F5__` genuine Gremlin-specific semantics; `F6__ ..= FF__` reserved. SQL and Gremlin use the shared algebra wherever semantics are exactly equal; `F4__`/`F5__` gain words only for a measured gap. Aliases never mint duplicate operators. No implementation commitment for unmeasured operator families. |
| O8 | **Composition in the host.** DT64 owns the `0xFFFF` authority. It is composed with OGAR's authority in the host / application assembly layer, never by a dependency from `lance-graph-ogar` to `lance-graph-dt64`. The routing policy is one tested piece, not reimplemented per consumer. |
| O9 | **Full-classid basis for behaviour lenses.** A behaviour lens's field basis resolves by the full `u32` classid, through a backward-compatible, lens-specific `ClassView` accessor. Data-class lookup and the `0xC401` R2IL reading keep their current resolution. An unknown behaviour lens fails closed; it never inherits the concept's data-field basis. `AlgebraRef.word`, `WideFieldMask` and the 256-position limit are not widened. 256 is the limit per lens, not per concern. |
| O10 | **The consumer supplies its own `ClassView`** from its codebook. No label arrays are added to `Activation`. Activation alone does not establish label provenance: registration associates the authorized owner and activation with a versioned schema/basis witness, and refuses a missing activation, a mismatching basis, a stale revision and unauthorized ownership. An arbitrary caller-supplied view is not trusted silently. |

**Execution order** (each a separate PR; each step test-first, with a disable run and a backward-compatibility twin):

1. **L4:** production alias views (R2IL, SQL, Quack) derived from the existing `POLYFILL` table. No parallel opcode table, no change to the `0xC401` path.
2. **L1:** two behaviour lenses with the same left half, different right halves and different bases resolve independently; a consumer data class and a behaviour lens sharing a left half do not cross-read. The red-first case: `0xFFFF_F000` with 17 fields and `0xFFFF_F200` with a different count must not both read `fields(0xFFFF)`.
3. **L5:** the smallest consumer-`ClassView` and immutable basis-witness seam; `Activation` unchanged.
4. **L2/L3:** host composition and explicit activation checks; DT64 and OGAR stay independent.

Preserved throughout: the `0xC401` name-bound reading, the `FnIndex` opcode contract, positional polyfill semantics, refusal on wrong aliases, and the existing integration tests. Not introduced: a new runtime, a second authoritative registry, premature removal of the OGAR dependency, automatic XLOG promotion, a shared data-field/algebra-field namespace.

---

### 1e. O5 design: the versioned global → local table (2026-11-01)

**Status:** core shipped (`dr::MapTable`, T1–T4 green, both disable runs red). Stored selections wait on O4.

O5 is a BindStore plan item (D-DT64-6 Amendment A). Litho is the mental model it came from, not a component.

**What exists (VERIFIED-IN-CODE, `lance-graph-dt64/src/dr.rs`):**

| Piece | What it does |
|---|---|
| `LocalMap::prepare(width, globals)` | One tile's global → local table, at most 64 or 256 entries. Duplicates and overflow are refused. |
| Row shape | One row per local position: `globals[local]`. `local()` reads global → local (counted), `global()` reads local → global. Ids only, never values. |
| `LocalMap::digest` | FNV-1a over the local-order table. Its doc says a frame carries this value as its `coordinates` revision. |
| `FrameStore` | `seal` / `advance` / `check`. A stale frame is refused and names the revision that moved (`CoordinatesChanged` for a new map). |

So "versions = frame revisions" already works as a mechanism. **The gap:** nothing ties a `LocalMap` to the frame it was sealed under.
- A caller can keep a map after `advance` and read through it with no refusal.
- `coordinates` is whatever number the caller passes, not the digest of the maps in use.

**Proposal (WORKING-MODEL), in `dr`, additive. No existing signature changes:**

1. **`MapTable`.** It owns one `FrameStore` and the `LocalMap` of each tile, keyed by `TileId` (a `u32` the caller names). Insert and replace go through `MapTable::advance(changes)`. That sets `coordinates` to a digest over every tile's `(TileId, LocalMap::digest)` in tile order, and leaves `source` and `relation_map` as passed. Tiles are inserted only through `advance`, so the revision always matches the maps actually held.
2. **Reads take the frame.** `MapTable::map(&Frame, tile) -> Result<&LocalMap, MapRead>`. `MapRead` is either `Stale(FrameRefusal)` or `NoSuchTile`. Old frames are refused, so a map cannot outlive its frame. This is `Pop`'s `Stale` rule (F7) applied to coordinates.
3. **Where it sits.** DR owns the frame, so the table lives in `dr`, beside `FrameStore`. The BindStore keeps reading frames through `Guards`, exactly as `prepare` does now. A BindStore field holding the table is the alternative; it would make the store own DR state, which §3 assigns to DR.
4. **Not in this item: stored selections (D7').** Storing a `WideFieldMask` over a tile's local positions, keyed to the frame, depends on O4: whether one `0xF###` value names one stored selection or one position in a wider basis. It is added once O4 is decided, as a sibling read with the same `Stale` refusal.

**Falsifiers (test-first):**

| Id | Falsifier |
|---|---|
| T1 | A map read under a frame sealed before `advance` is refused with `Stale(CoordinatesChanged)`, and read under the new seal it succeeds. Twin: an `advance` that changes nothing in a tile still re-seals, so the old frame gets `Stale(FrameResealed)`. |
| T2 | `coordinates` is a function of the held maps. The same maps in the same tiles give the same revision. Reordering one tile's globals changes it. Swapping two tiles' maps changes it. Renaming a tile with the same map in the same position changes it; this case alone tests that the `TileId` is in the digest (a disable run showed the swap case passes without it). |
| T3 | A frame from another `MapTable` is refused with `Stale(ForeignFrame)`, even at equal revisions. |
| T4 | Backward compatibility: existing `LocalMap` / `FrameStore` callers (`tests/dr.rs`, `xlog-relcomp-probe` `litho.rs` and `gate_b.rs`) are unchanged and green. |

Each falsifier gets a disable run: drop the frame check in `map`, and drop the `TileId` from the digest.

## 2. The classid, both sides

```text
    left (high 16): concept                 right (low 16): selection within it
    [ Domain 8 | AppId 8 ]          ::      [ ClassView / WideFieldMask selection ]

    0xFFFF                 hardcoded behaviour (Frozen), registered by DT64   (D11'', D13'')
    Domain::AppId          that domain/app's registered classes (Learned)
                                            0xF000 ..= 0xFFFF   behaviour (by concern under 0xFFFF, §1a)

    0x0000                 ⊘ was "hardcoded" (D6'); superseded by 0xFFFF (D11'')
    0xF000 ..= 0xFFFE      ⊘ was "registered behaviour seats" (D6', §6); unallocated (O1)
```

- **VERIFIED-IN-CODE.** Under the current order a classid is `[canon u16][custom u16]` (`ogar_codebook.rs` `compose_classid`, `:484`). `classid_concept` (`:573`) is the left side; `classid_app_prefix` (`:561`) still names the right side as the app/render prefix, which is the fossil reading D8' deprecates.
- **VERIFIED-IN-CODE.** `0x00` is `ConceptDomain::Reserved` and `0xF0..=0xFF` is `Unassigned` (`ogar_codebook.rs:179-210`); `ogar-vocab` at `2a080a9` mints nothing in `0xF0..=0xFF`.
- **WORKING-MODEL.** The two `0xF###` ranges are different roles. On the left, `0xF###` names a behaviour concept. On the right, it is a selection under whatever concept is on the left. No reader infers one from the other.

---

## 3. Where each category lives

| category | home | changes by | VERIFIED-IN-CODE today |
|---|---|---|---|
| **Frozen** | compiled code and `const` tables; `0xFFFF :: 0xF###`, registered by DT64 (§1a) | a new build only | relalg, mask-risc, `ndarray::simd`; the contract's `const` mirrors (`ogar_codebook`) |
| **Learned** | `BindStore` entries; `Domain::AppId :: 0xF###` (D14'') | `register`; withdrawal (§5); revisions as new classids + re-route | `xlog-relcomp-probe/tests/dto_binding.rs:145-154` registers an XLOG method and admits it |
| **Explore** | outside the store | becomes Learned by `register` | Kythe/rust-analyzer facts (#1894), the R2IL codegraph harvest |

**Migration is not a store operation.** Explore → Learned is a `register`. Learned → Frozen is moving the behaviour into compiled code (for XLOG: minting `lance-graph-xlog`).

---

## 4. Findings against the current code

**F1 — Frozen polyfill words are runtime-registered, through the Learned vocabulary.** VERIFIED-IN-CODE:
- `BindStore::declare_algebra(lens, schema_rev, extent, arity)` (`lance-graph-dt64/src/lib.rs:628`) declares the polyfill lens at runtime. `AlgebraLens` holds a schema and one arity per word, nothing else (`:119-122`); "the store never resolves or runs the word" (`:105-109`).
- Executing a bound effect goes through `lance-graph-r2il::binding`, which:
  - refuses any lens whose concept is not `ogar_r2il::CONCEPT_R2IL_FOLD` (`binding.rs:278`; `0xC401`, marked PROVISIONAL, not minted in `ogar-vocab`);
  - resolves each word by label through `ogar_r2il::R2ILVocabulary` and checks `stack_arity` / `pushes_result` (`binding.rs:496-525`);
  - takes `FnIndex` from `ogar-loco` and the fold opcodes from `ogar-r2il`.
- So a Frozen word (D2) is named and typed by the R2IL vocabulary, which is harvest-side (D3). WORKING-MODEL: the Frozen polyfill is a compile-time lance-graph table at `0xFFFF :: 0xF0__` (§1a); R2IL maps onto it with a parity test where OGAR is compiled; `ogar-loco` / `ogar-r2il` leave the execution path.
- Dependency weight is not the issue: `cargo tree -p lance-graph-r2il` shows OGAR contributing exactly `ogar-r2il` and `ogar-loco`, and `ogar-r2il`'s only normal dependency is `ogar-loco` (no r2sleigh). The issue is the category, and the git `main` coordinate (`ISS-SIX-DECISIONS-NOT-EXECUTED-FLOAT-REFS`).

**F2 — Nothing can be withdrawn but a route.** VERIFIED-IN-CODE:
- `BindStore::retract(position)` writes a new `RouteVersion` and keeps history (`lib.rs:747`; `dr.rs` `RouteLog::retract`/`push`); its doc says the classid keeps its behaviour.
- There is no withdrawal for `register`, `register_derived` or `declare_algebra`.
- `admit` (`lib.rs:796`) checks method, signature revision and guard only; a class whose routes are all retracted still admits.

**F3 — Registration moves no version.** VERIFIED-IN-CODE: only routes are versioned (`RouteLog`). Recorded as open for the frame owner (D-DT64-6 plan `:293`, D8). With withdrawal (§5) a frame-visible version for registration becomes necessary.

**F4 — No registrant is recorded and no activation is consulted.** VERIFIED-IN-CODE: `register` takes any `u32` (`lib.rs:556`); `lance-graph-dt64` contains no `hotplug`, `Activation` or `CapabilityAuthority` reference. D-DT64-6 lists "wiring OGAR's `CapabilityAuthority` (D6)" as not done (`:261`).

**F5 — The compat reader misreads a left-empty classid.** VERIFIED-IN-CODE: `classid_canon_compat` (`ogar_codebook.rs:530-540`) takes the high half only when it is `>= 0x0100` and otherwise, with a non-zero right side, re-reads the id in the legacy order. `0x0000_F123` therefore yields concept `0xF123`, not `0x0000`. The strict `classid_concept` is correct. Left side `0xF###` is unaffected (`>= 0x0100`).

**F6 — The global → local selection is in memory, not a versioned table.** VERIFIED-IN-CODE: `dt64::dr::LocalMap` (`dr.rs:424-505`) is "the prepared global → local pointer table for one tile under one frame"; it is versioned only by `digest()`, which a frame carries as its `coordinates` revision. No lance-graph table holds it (D-LITHO Q5 open) and it is not addressable as `ClassView × WideFieldMask`.

**F7 — A witnessed, reusable population exists in the lab.** VERIFIED-IN-CODE: `dir-rmap-probe/src/pop.rs` `Pop { witness, tiles, kept }` refuses `Stale` when populations from different snapshots combine (`:35-98`, #1666). DR256 capstone §C.3 assigns named reusable binding descriptors (witness, lifetime, dependencies) to the planned binding store, with the computed mask in the caller's buffer under that witness. A right-side behaviour selection (D7') is that precedent, not a new design.

**F8 — The traceroute is a chain of stage identities, not a path.** DR256 capstone §C.5: derivation, proof, binding, execution and replay, each referenced by an immutable identity; "it stores no intermediate population, symbolic graph or duplicated SoA payload". `recursive::trace` in #1896 rebuilds a node path, which is not that traceroute (D9'); a per-round delta-mask store would violate §C.5.

---

## 5. Registration and withdrawal constraints (WORKING-MODEL)

Each constraint is a test that fires plus a twin that stays silent, on non-trivial input.

| # | constraint | fires | stays silent |
|---|---|---|---|
| C1 | **Left side `0xFFFF` is registered only by DT64, from compiled tables.** A plug cannot activate concept `0xFFFF`, so `register`, `register_derived`, `declare_algebra` and withdrawal by a plug refuse it. *(Was: left side `0x0000`; superseded by D11''.)* | a plug's `register(0xFFFF_F001, ..)` refused | the same plug's `register(<its Domain::AppId>_F001, ..)` accepted |
| C2 | **Registration needs an activation for its left side.** A `Domain::AppId` concept must be among the concepts an `Activation` covers (`hotplug.rs` `CapabilityAuthority::activate`); concept `0xFFFF` is activated only by DT64's own authority (§6). The activated plug is the registrant. | registering a concept the plug did not activate | registering the activated concept |
| C3 | **Withdrawal never deletes.** At versions before the withdrawal the class still resolves and admits (replay holds); from the withdrawal on, `admit` and `route` refuse with a named reason (`Withdrawn { class, at }`). | `admit` after withdrawal | `admit` at a version before withdrawal |
| C4 | **Registration and withdrawal move a frame-visible version** (closes F3). | a frame sealed before a withdrawal fails `check` | a frame sealed after it passes |
| C5 | **No dangling routes.** A class that is still routed cannot be withdrawn; the refusal names the positions. | withdraw while routed | withdraw after `retract` of every position |
| C6 | **No orphaned dependents.** Withdrawal refuses while a derived class names it as base, or registered methods use the lens's words (`users_of` exists, `lib.rs:700`). | withdraw a base with a live derived class | withdraw a class nothing depends on |
| C7 | **Only the registrant withdraws.** | another plug's withdrawal | the registrant's withdrawal |
| C8 | **D5 holds after withdrawal.** A different binding under a withdrawn classid is refused (`BehaviourImmutable`); the identical binding reinstates it at a new version; a changed hypothesis takes a new classid. | different binding, same classid | identical binding |

---

## 6. Left-side behaviour seats `0xF###` (WORKING-MODEL)

> ⊘ **Superseded by §1a (D13'').** Concept `0xFFFF` is not a seat that plugs claim; DT64 is its authority and registers it from compiled tables. The palette-seat precedent below still describes the mechanism a composing authority uses to route a reserved concept before the capability join. Left sides `0xF000 ..= 0xFFFE` are unallocated (O1).

- **Precedent, VERIFIED-IN-CODE:** `OgarAuthority::activate` (`lance-graph-ogar/src/lib.rs:565-660`) routes a reserved range outside the minted codebook through its own arm before the capability join: palette seats `0x17XX` (`is_palette_seat` `:501`, `palette_seat_owner` `:541`, `impersonated_seat` `:550`). A claimed seat is owner-checked (`UnexpectedConsumer`); an unclaimed seat is plug-and-play; readings come from `plug_readings` (`:452`).
- **Proposal:** behaviour seats `0xF000..=0xFFFF` take the same kind of arm: activation without minting each behaviour as a domain concept, an owner for C2/C7, and readings from the plug.

---

## 7. Open decisions

| # | question | owner |
|---|---|---|
| O1 | ~~Seat ownership for left `0xF###`~~ — answered for `0xFFFF` by D13'' (lance-graph owns it, DT64 registers). Open remainder: do left sides `0xF000 ..= 0xFFFE` ever hold anything? | operator |
| O2 | OGAR reserves concept `0xFFFF` (at least; domain `0xFF` if preferred) for lance-graph behaviour, never minted in `ogar-vocab`, with a mirror test that fails if a codebook row lands there. | DONE: domain `0xFF` reserved (OGAR #365); contract mirror `HARDCODED_BEHAVIOUR` + guard test |
| O3 | F5 for legacy `0x0000_XXXX` ids: exclude them, or retire the heuristic. No longer gates this plan (§1a). | operator |
| O4 | Right-side `0xF###` selection: does one value name one stored selection over the existing 256-position basis, or one position in a wider basis (which widens `WideFieldMask` and `AlgebraRef.word` past `u8`)? D7' points to the first. | operator |
| O5 | (Litho is a mental model; this is a BindStore plan item, D-DT64-6 Amendment A) Which lance-graph table versions the global → local selection, and its row shape (WORKING-MODEL: versions = frame revisions, one row per local position, read through a ClassView, selected by a `WideFieldMask`; it also holds the stored selections of D7'). | DR/DT owner |
| O6 | Whether `recursive::trace` (#1896) stays as a semi-naive test oracle or is removed; either way it is documented as not the traceroute. | operator |
| O7 | The concern ranges inside `0xF###` (§1a table): confirm the digit-per-concern split and the order. Per §1b, languages that only rename the shared algebra become alias views; which of SQL, Gremlin and XLOG keep a range? Does full TensorFlow share TFLite's range, or take its own? | **DECIDED §1d** |
| O9 | Behaviour-lens basis key (L1): keep the left-half key (256 words per left side; a large operator set takes several left sides), or key behaviour lenses by the full classid (a contract `ClassView` change). | **DECIDED §1d** |
| O10 | Label handover from a consumer codebook (L5): the consumer's own `impl ClassView`, or label rows returned by the authority with the activation. | **DECIDED §1d** |
| O8 | Where the composing authority (and the composed `ClassView`, L2) lives: in `lance-graph-ogar` (which then depends on `lance-graph-dt64`), or in the binary that holds both authorities. | **DECIDED §1d** |

---

## 8. Plan items (test-first; none started)

| id | item | gate (red first, disable-verified) |
|---|---|---|
| D-DT64-7-0 | this document and the board rows | docs review |
| D-DT64-7-1 | F5 falsifier for legacy ids: `classid_canon_compat(0x0000_F123) == 0x0000`; plus the silent twin `classid_canon_compat(0xFFFF_F000) == 0xFFFF` (green today) | first red on `main` today; no longer gating (§1a) |
| D-DT64-7-2 | C1–C8 as tests over a `BindStore` withdrawal API, behind the activation of C2 (C1 as revised: a plug cannot register under `0xFFFF`) | each constraint's fire and silent twin (§5) |
| D-DT64-7-3 | Frozen polyfill table: a compile-time lance-graph table at `0xFFFF :: 0xF000` (opcode, label, arity, pushes-result), registered by DT64, with a parity test against `ogar_r2il::R2ILVocabulary` where OGAR is compiled | parity test red on one deliberately changed row; `lance-graph-r2il` builds with `ogar-loco` / `ogar-r2il` optional |
| D-DT64-7-4 | DT64 `CapabilityAuthority` for concept `0xFFFF`, plus the composing authority (O8) | a plug claiming `0xFFFF` is refused; DT64 activates it; any other concept still reaches `OgarAuthority`; an `ogar-vocab` row at `0xFFFF` fails the mirror test (O2) |
| D-DT64-7-6 | polyglot alias views over `0xFFFF :: 0xF000` (§1b): position-bound words, one `ClassView` per language, binding by position instead of `R2ILVocabulary` label scan | every alias resolves to the same `AlgebraRef` (red on one deliberately duplicated name); `prepare` and `run` timed in ns before and after |
| D-DT64-7-7 | semantic union census (§1b): published operator enums of ONNX, TF, TFLite, GGML and Burn IR, plus R2IL and XLOG, each recounted from source, mapped to shared words or to a concern range | every count carries the command or file it came from; the mapping is checked by the D-DT64-7-6 parity test |
| D-DT64-7-5 | O5 table design | D-LITHO Q5 answered; a selection read under a stale version refuses (`Pop`'s `Stale`, F7) |

D-DT64-7-6 follows D-DT64-7-3; D-DT64-7-1 needs none; D-DT64-7-3 needs O7 (the `0xF0__` range only); D-DT64-7-2 needs O8; D-DT64-7-4 needs O2 and O8; D-DT64-7-5 needs O5.

---

## 9. Not covered

- How a selection is evaluated, relabelled and tiled, and its semiring readings: D-DYNBAG.
- Routing, frames, focus and scope: D-DT64-6 and DR.
- Proof checking: DR256 capstone §C.2 lists the independent checker as MISSING.
- Promotion evidence for XLOG (D4): its lab oracles exist (BFS oracle and 200-case differential for the closure, two oracles for bag counts, Scallop), but which must be green for the mint is not decided here.
