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

---

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
| O2 | OGAR reserves concept `0xFFFF` (at least; domain `0xFF` if preferred) for lance-graph behaviour, never minted in `ogar-vocab`, with a mirror test that fails if a codebook row lands there. | OGAR owner |
| O3 | F5 for legacy `0x0000_XXXX` ids: exclude them, or retire the heuristic. No longer gates this plan (§1a). | operator |
| O4 | Right-side `0xF###` selection: does one value name one stored selection over the existing 256-position basis, or one position in a wider basis (which widens `WideFieldMask` and `AlgebraRef.word` past `u8`)? D7' points to the first. | operator |
| O5 | D-LITHO Q5: which lance-graph table versions the global → local selection, and its row shape (WORKING-MODEL: versions = frame revisions, one row per local position, read through a ClassView, selected by a `WideFieldMask`; it also holds the stored selections of D7'). | DR/DT owner |
| O6 | Whether `recursive::trace` (#1896) stays as a semi-naive test oracle or is removed; either way it is documented as not the traceroute. | operator |
| O7 | The concern ranges inside `0xF###` (§1a table): confirm the digit-per-concern split and the order (relalg, XLOG, ONNX, TFLite, SQL, Gremlin). | operator |
| O8 | Where the composing authority lives: in `lance-graph-ogar` (which then depends on `lance-graph-dt64`), or in the binary that holds both authorities. | operator |

---

## 8. Plan items (test-first; none started)

| id | item | gate (red first, disable-verified) |
|---|---|---|
| D-DT64-7-0 | this document and the board rows | docs review |
| D-DT64-7-1 | F5 falsifier for legacy ids: `classid_canon_compat(0x0000_F123) == 0x0000`; plus the silent twin `classid_canon_compat(0xFFFF_F000) == 0xFFFF` (green today) | first red on `main` today; no longer gating (§1a) |
| D-DT64-7-2 | C1–C8 as tests over a `BindStore` withdrawal API, behind the activation of C2 (C1 as revised: a plug cannot register under `0xFFFF`) | each constraint's fire and silent twin (§5) |
| D-DT64-7-3 | Frozen polyfill table: a compile-time lance-graph table at `0xFFFF :: 0xF000` (opcode, label, arity, pushes-result), registered by DT64, with a parity test against `ogar_r2il::R2ILVocabulary` where OGAR is compiled | parity test red on one deliberately changed row; `lance-graph-r2il` builds with `ogar-loco` / `ogar-r2il` optional |
| D-DT64-7-4 | DT64 `CapabilityAuthority` for concept `0xFFFF`, plus the composing authority (O8) | a plug claiming `0xFFFF` is refused; DT64 activates it; any other concept still reaches `OgarAuthority`; an `ogar-vocab` row at `0xFFFF` fails the mirror test (O2) |
| D-DT64-7-5 | O5 table design | D-LITHO Q5 answered; a selection read under a stale version refuses (`Pop`'s `Stale`, F7) |

D-DT64-7-1 needs none; D-DT64-7-3 needs O7 (the `0xF0__` range only); D-DT64-7-2 needs O8; D-DT64-7-4 needs O2 and O8; D-DT64-7-5 needs O5.

---

## 9. Not covered

- How a selection is evaluated, relabelled and tiled, and its semiring readings: D-DYNBAG.
- Routing, frames, focus and scope: D-DT64-6 and DR.
- Proof checking: DR256 capstone §C.2 lists the independent checker as MISSING.
- Promotion evidence for XLOG (D4): its lab oracles exist (BFS oracle and 200-case differential for the closure, two oracles for bag counts, Scallop), but which must be green for the mint is not decided here.
