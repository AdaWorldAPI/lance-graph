# MORK / PathMap Schnellverschluss — v1

**Status:** D-MPS-0, D-MPS-1 and D-MPS-8 MEASURED (probe `crates/mork-pathmap-probe`, results only; D-MPS-8 is the native arm over a versioned relationship table, §7b, not admitted). D-MPS-2..7 PROPOSAL / BLOCKED as listed in §8. No production executor, no CE64 change, no Lance storage, no upstream file edited by this work.

**Ownership rule:** our types stay ours, upstream types stay upstream's, and the one replaceable boundary sits where they meet: `crates/mork-pathmap-probe/src/boundary.rs` plus the `[dependencies]` / `[patch]` block of the probe's `Cargo.toml`.

Read with: `d-rmap-separate-tables-thought-palettes-v1.md` (D-RMAP-1A, ownership), `d-holo-h14c-thought-first-v1.md` and `d-holo-h14d-logic-mask-read-v1.md` (no-gather ruling), `metta-datalog-assimilation-v1-council-v3.md` and entry `2026-10-16-asm-accepted-proposals.md` (D-ASM-10: MORK is an oracle outside the workspace).

## 1. Source inventory (verified 2026-10-19)

| Repository | Local path | Revision used | Relation to upstream | Licence |
|---|---|---|---|---|
| AdaWorldAPI/MORK | `/home/user/mork` | `de9312e` (branch `ccr-9f3b8075-74u47l-stable`) | upstream `trueagi-io/MORK` `main` = `ed57c67`; fork `main` = `ed57c67` (0 files differ); the stable branch adds one commit, **18 files, +421 / −408** | **none declared**: no LICENSE file, no `license` field. Upstream PR #148 (`a6281fa`, adds MIT LICENSE + manifest metadata) is **not merged** into upstream `main` |
| AdaWorldAPI/PathMap | `/home/user/pathmap` | `ef02591` | upstream `Adam-Vandervorst/PathMap` `master` = `ef02591`; **0 files differ** | MIT (`Cargo.toml` + LICENSE) |
| AdaWorldAPI/PeTTa | `/home/user/petta` | `1d19a1d` | not compared (Prolog/Python, no Rust crate) | MIT (LICENSE, © Patrick Hammer 2025) |
| AdaWorldAPI/mettaclaw | `/home/user/mettaclaw` | `7b30527` | not compared (MeTTa/Python, no Rust crate) | MIT (LICENSE, © Patrick Hammer 2026) |

All four clones are shallow. PeTTa and MeTTaClaw have no Rust surface to link; they stay oracle/reference material and are out of scope for the dependency switch.

## 2. Type inventory (read in source, not inferred)

**MORK** (`expr/src/lib.rs`, `interning/src/lib.rs`, `kernel/src/leapfrog.rs`, `kernel/src/space.rs` at `de9312e`):

| Item | Definition | Consequence at the boundary |
|---|---|---|
| `Expr` | `#[derive(Clone, Copy)] #[repr(transparent)] pub struct Expr { pub ptr: *mut u8 }`, `unsafe impl Send/Sync` (lib.rs:197, 1802–1803) | raw pointer, no lifetime: nothing stops it outliving its buffer. It must never cross the boundary or be stored. |
| `ExprEnv` | `{ pub n: u8, pub v: u8, pub offset: u32, ground_skip: u16, pub base: Expr }` (lib.rs:1881) | a view into an `Expr`; same hazard |
| `Bindings` | direct-indexed slab `{ slots: Vec<Option<ExprEnv>>, touched: Vec<u16> }`, key `ExprVar = (u8, u8)` = (namespace, introduction index) (lib.rs:1808, 1820) | variables are positional, ≤ 63 per namespace; not Lance variables |
| `Tag` | `NewVar`, `VarRef(u8)`, `SymbolSize(u8)`, `Arity(u8)` (lib.rs:90) | symbols are inline byte strings < 64 bytes, or 8-byte interned handles |
| interned symbol | `type Symbol = [u8; SYM_LEN]`, `SYM_LEN = 8` (interning/lib.rs:49–51), table `SharedMapping` behind a `NonNull` handle | an 8-byte table-relative handle. **Not** a `NodeGuid` (16 bytes, globally canonical); never map one onto the other |
| `SubtermCursor<Z>` | generic: `impl<Z: Zipper + ZipperMoving + ZipperPath + ZipperIteration>` (leapfrog.rs:88, 168) | generic over PathMap's public traits |
| `query_multi_leapfrog` | `pub fn query_multi_leapfrog<F: FnMut(Result<&[u32], &Bindings>, Expr) -> bool>(map: &PathMap<()>, pat_expr: Expr, effect: F) -> usize` (leapfrog.rs:1319), behind feature `leapfrog` | **concrete** `&PathMap<()>`; its `UnifyJoin` holds `SubtermCursor<ReadZipperUntracked<'a, 'static, ()>>`, and `build_reindex` materializes a new `PathMap<()>` for an inverted (cyclic) factor |
| `Space` | `{ pub btm: PathMap<()>, pub sm: SharedMappingHandle, … }` (space.rs:59) | MORK's store is a set of byte paths |

**PathMap** (`src/zipper.rs`, `src/trie_map.rs` at `ef02591`): `PathMap<V, A>` is a byte-keyed trie; `insert<K: AsRef<[u8]>>(&mut self, k, v)`. The traits `SubtermCursor` needs have these required methods: `Zipper { path_exists, is_val, child_count, child_mask }`, `ZipperMoving { depth, focus_byte, val_count, descend_to, ascend, ascend_until, ascend_until_branch }`, `ZipperPath { path }`; every `ZipperIteration` method has a default. None of the four is sealed. Read-only and witness types exist (`ZipperReadOnlyValues`, `ZipperReadOnlyConditionalValues`, `ReadZipperWitness`, `ProductZipperWitness`, `TrieRefWitness`); `OverlayZipper` and `ProductZipper` are public.

**Lance-graph:** `NodeGuid` and the OGAR/ClassView/envelope contracts exist (`canonical_node.rs`, `lance-graph-contract`). `RelationshipMapStore`, `RelationOrder`, `MapPalette`, `ThoughtMapPaletteBinding` and the frame/index witnesses exist **only in plans** (D-RMAP, PROPOSED); no Rust type has those names. `lance-graph-relalg` describes single-table selections and lowers them through R2IL to mask-risc; its own doc says "recursion (μ) and the hop (preimage) are not built".

## 3. Ownership and mapping design

```
Lance side (owner of meaning)            boundary.rs (the Schnellverschluss)     upstream (owner of its types)
RelationView<'a>                          bind(backend, input, semantics)          mork::space::Space
  rows: &'a [[u32; 2]]   ── encode ──▶       -> BoundIndex { generations, … }  ──▶   pathmap::PathMap<()>
  generation: u64                         answers(&mut self, current)
Answers { [u32; 3] -> count }  ◀── decode ──  -> Result<Answers, Refusal>
```

- **One canonical owner of meaning.** The rows and their generations belong to the Lance side. In production that is a sealed `RelationOrder` view under D-RMAP; in the probe it is a borrowed slice. MORK and PathMap hold only a disposable index built from that view.
- **Nothing upstream-typed crosses.** Rows go in as `[u32; 2]` and answers come out as `[u32; 3]`. `Expr`, `Bindings`, `Space` and `PathMap` stay private to `boundary.rs`.
- **The upstream index is a copy and is labelled one.** `bind` encodes the rows (directly as 9-byte trie keys, or as S-expression text through MORK's parser) and the index owns those bytes. That is the translation cost in §6. A stale copy is caught by the generation witness (`Refusal::Stale`), not by a lifetime, because the copy does not borrow.
- **Typed refusals instead of silent degradation:**
  - `Refusal::SetOnly(backend)`: PathMap and MORK store sets, so a request for bag multiplicities is refused rather than answered with every count 1.
  - `Refusal::NoAdmittedNativeJoin`: there is no admitted lance-graph join for this input shape (§7).
  - `Refusal::Stale`: the generations do not match.
  - `Refusal::Upstream`: the upstream index rejected the encoded input.
- **Removal test:** deleting a backend means deleting its `Index` variant, its two helper functions, and its line in `Cargo.toml`. Nothing outside `boundary.rs` names `mork` or `pathmap`. This was checked by construction: `lib.rs` and `reference.rs` import neither.

## 4. Source switch (fork ↔ upstream)

**Mechanism:** Cargo's own, with no custom machinery.

1. Git dependencies pinned by `rev` in the probe's `Cargo.toml` (`mork`, `mork-expr` → `AdaWorldAPI/MORK@de9312e3`; `pathmap` → `AdaWorldAPI/PathMap@ef02591`).
2. One `[patch."https://github.com/AdaWorldAPI/MORK"] pathmap = { git = …PathMap, rev = "ef02591" }`.
   - **Why it is needed (MEASURED):** MORK's workspace declares `pathmap = { path = "../PathMap/" }`, which lies outside the MORK repository. A plain git dependency on MORK fails with `no matching package named pathmap found … Git repository https://github.com/AdaWorldAPI/MORK`.
   - **What the patch does (MEASURED):** it resolves that requirement to the same PathMap revision, so the graph has exactly **one** `pathmap` package (`cargo tree -i pathmap`, and the lockfile has 1 `name = "pathmap"`).
3. Switching to upstream = changing the URL and `rev` of three dependency lines plus the patch key. **No consumer source changes.**

**Measured constraints on the switch:**

| Constraint | Measured | Who owns the fix |
|---|---|---|
| Upstream MORK `main` (`ed57c67`) on Rust 1.99 stable | fails: `error[E0554]: #![feature] may not be used on the stable release channel` (fast-slice-utils, via `nightly` defaults) | upstream: D-MPS-3 |
| Fork stable branch `de9312e` on 1.99 | builds (`mork-expr` + `mork`, 1 min 02 s debug) | carried fork commit until D-MPS-3 lands |
| MORK's gxhash fork | `compile_error!` without `+aes,+sse2`; the probe sets them in its own `.cargo/config.toml`, never via `RUSTFLAGS` | upstream: optional |
| PathMap `ef02591` alone on 1.99 | builds (20 s debug), no flags needed | — |
| MORK enables PathMap feature `jemalloc` | PathMap then installs a `#[global_allocator]`; any binary linking MORK has its allocator replaced and cannot add its own (the probe's allocation counter failed to link: "conflicts with global allocator in: pathmap") | upstream: D-MPS-4 |

## 5. Entropy ledger

| Kind | Item | Cost | Disposition |
|---|---|---|---|
| Semantic | MORK/PathMap answer sets; Lance relations are bags | duplicate derivations collapse (2 × 2 → 1) | refused by type (`SetOnly`); pinned by `duplicates_are_refused_under_bag_semantics` |
| Semantic | 8-byte MORK symbol vs 16-byte `NodeGuid` | none, because no identity is shared: ids cross as `u32` and are re-encoded per index | kept separate by design |
| Semantic | second fact store | none: both indexes are disposable copies keyed by generation | gate "storage ownership" |
| Semantic | generation witness compares numbers only | a reused generation number is not detected (ABA), pinned by `a_reused_generation_number_is_not_detected` | the producer must never reuse a generation; D-RMAP seal digests would close this |
| Maintenance | upstream files modified by us | MORK: 18 files on one carried branch (nightly → opt-in); PathMap: 0 | D-MPS-3 upstreams the 18 |
| Maintenance | out-of-repo `../PathMap` path in MORK | one `[patch]` line per consumer | D-MPS-2 upstream |
| Maintenance | library-installed global allocator | blocks any MORK linkage into a lance-graph binary | D-MPS-4 upstream |
| Maintenance | AES/SSE2 target-feature requirement | a per-crate `.cargo/config.toml`; `cargo --manifest-path` from the repo root does not read it | documented; run from the crate directory |
| Translation | encode + index build | see §6: PathMap 177–296 ns/row, MORK (text + parse) 386–938 ns/row | paid once per bound generation |
| Translation | MORK answer decode | `exec` writes `Q` facts into the space; the probe dumps text, parses it and removes the facts again | oracle-only; dominates the MORK query time in §6 |

## 6. Measured cost (`cargo run --release --example cost`, one run per cell)

Rows `R(i, y)` with `y` drawn from `n / 4` values, `S(y, z)` one row per `y`, answers ≈ n. Shared = one index built once and queried by k thoughts; rebuilt = an index built per thought. Both are given per thought and as a total over 64 thoughts, so the two columns compare like with like.

| join | backend | n | build ms | ns/row | query ms | shared per thought, k=64 | rebuilt per thought | shared total, k=64 | rebuilt total, k=64 |
|---|---|---|---|---|---|---|---|---|---|
| product-zipper | PathMapZipper | 1 000 | 0.259 | 207 | 0.205 | 0.210 | 0.465 | 13.4 | 29.7 |
| | Mork | 1 000 | 0.515 | 412 | 3.006 | 3.014 | 3.521 | 192.9 | 225.4 |
| | PathMapZipper | 10 000 | 2.069 | 165 | 2.846 | 2.879 | 4.915 | 184.2 | 314.6 |
| | Mork | 10 000 | 4.510 | 361 | 28.29 | 28.36 | 32.80 | 1 815 | 2 099 |
| | PathMapZipper | 50 000 | 14.77 | 236 | 24.09 | 24.32 | 38.86 | 1 556 | 2 487 |
| | Mork | 50 000 | 25.98 | 416 | 161.7 | 162.1 | 187.7 | 10 377 | 12 013 |
| leapfrog | PathMapZipper | 50 000 | 14.75 | 236 | 21.93 | 22.16 | 36.68 | 1 418 | 2 348 |
| | Mork | 50 000 | 23.71 | 379 | 150.7 | 151.1 | 174.4 | 9 670 | 11 164 |

Re-run 2026-10-19 after the PR #1620 review: the first version compared a per-thought shared cost with a 64-thought rebuilt total and reported about 110×, which overstated the saving.

What it shows, and what it does not:

- **Sharing saves the build, not the query.** Over 64 thoughts a shared PathMap index costs 1.6× less than rebuilding it per thought (n = 50 000: 1 556 ms vs 2 487 ms total; 24.3 vs 38.9 ms per thought). The query is most of the cost, so the saving is bounded by build / (build + query): 38% for PathMap at n = 50 000, 14% for MORK.
- **The MORK query column measures the probe's text decode too, not MORK's join alone.** It covers `exec`, dump, parse and removal of the `Q` facts. It is not evidence about MORK's join speed. A binding-level decode through `Space::query_multi` would be needed for that, and that path hands back `Bindings` or `&[u32]` references whose decoding is not done here.
- The leapfrog build is verified to compile MORK's `leapfrog` module (a type alias on `mork::leapfrog::SubtermCursor`). That each query took the leapfrog route is MORK's `cfg` dispatch (space.rs:1058–1064), read in source but not observed at runtime.
- No allocation counts: see the global-allocator constraint in §4.

## 7. RelationshipMap integration and the gather ruling

- **Can PathMap's zippers navigate a sealed `RelationOrder` without copying?** Possibly, but this is not proved. `SubtermCursor` needs only the public traits listed in §2: about 12 required methods, all byte-path navigation. A Lance-owned type implementing them over a borrowed, byte-ordered sealed column would let MORK's cursor walk Lance data in place. Nothing like that exists yet. `RelationOrder` itself is unbuilt, and its layout (fixed-width sorted keys vs a trie) decides whether `child_mask` / `descend_to` can be answered without materializing. **Status: HYPOTHESIS, D-MPS-6.**
- **Can MORK's join consume such a provider?** Not with the current API. `query_multi_leapfrog` and `UnifyJoin` are concrete over `&PathMap<()>` / `ReadZipperUntracked`, and `build_reindex` creates a new `PathMap` for cyclic factors. The generic piece already exists (`SubtermCursor<Z>`). The upstream change would be to make `UnifyJoin` generic over a cursor provider with a reindex hook. That is useful to MORK on its own, for joins over ACT arenas and mmaps (D-MPS-5).
- **Lance-side execution of this join is blocked by the 2026-10-16 ruling, not by an API.** A shared-variable join reads `S` at positions chosen by `R`'s values, and leapfrog's `seek` is that read. Inside lance-graph's admitted path the pair must already sit at one coordinate (`Thought(t)[T_a[i] : T_b[i]]`). So:
  - `Backend::Native` refuses (`NoAdmittedNativeJoin`).
  - The PathMap arm's seek into `S ‖ y` is oracle code and is labelled as such.
  - Nothing here is wired into Quack, R2IL or mask-risc.
  - The quarantined #1616 `OutgoingHop` is not used and is not a precedent.

  The native arm becomes testable only when a co-addressed producer for this input shape exists. **BLOCKED.**

  > **2026-10-19, §7b:** a co-addressed producer for this input shape now exists as a probe (`src/join.rs`), and `Backend::Native` answers through it. The arm is measured, not admitted.

## 7a. Versioning: what `RelationView.generation` stands for (2026-10-19)

`RelationView.generation: u64` is a probe-local stand-in. It is not a design. Inside the probe it is enough: the `Stale { bound, current }` refusal fires on any change. Outside the probe a bare `u64` fails in two ways:

- **Across datasets.** Two datasets can both be at version 7. The code already has this ambiguity: `temporal.rs`, `temporal_pov.rs` and `scheduler.rs` all use a bare `u64`, and `versioned.rs` returns only the nodes version of a write that spans three datasets.
- **Across epistemic modes.** A version alone does not say which reader horizon applies.

D-RMAP-1A §7 (#1619) replaces the stand-in with a binding witness built from three parts:

- `DurableRef{dataset, branch, version}`, defined in #1605 §1;
- an index seal, which carries the row-address map unless RelationFacts uses stable row ids;
- the read coordinate `QueryReference::at(v, rung)` from `temporal.rs`. Replay uses `Strict` and never `QueryReference::default()`.

The Schnellverschluss does not touch the witness. Switching the backend between MORK, PathMap and native changes how the join runs; the witness describes which data was bound. When the witness lands, it replaces `generation` in this crate, and the backend switch stays as it is. Today no `DurableRef` type exists in `crates/`, and CPG-00 has not run.

## 7b. Native arm: a versioned relationship table puts R and S at one coordinate (2026-10-19)

> **Status: MEASURED, probe only; not admitted.** D-RMAP-1 (§ on BIND) says a pre-fold global-to-local binding is a candidate membrane that needs its own review. Code: `src/join.rs`; tests `tests/join.rs` (8) and the native arms added to `tests/differential.rs`; cost `examples/join_cost.rs`. Nothing is wired into Quack or R2IL.

**Layout.** `JoinTable::prepare` builds one domain, the sorted distinct values of the shared variable `y` over both datasets, and lays each relation out by `y` slot. Each side gets a presence plane, offsets, and the other column's values. That is the one global-to-local translation (binary search into the domain), and it is timed inside `prepare`. Each side carries its own dataset's generation, so R and S can come from different sources and be resealed independently: `replace_side` re-prepares one side against the same domain, and refuses `OutsideDomain` if the new rows bring a `y` the domain lacks.

**Evaluation.** Leapfrog triejoin with `y` first intersects R's `y` level with S's `y` level, then enumerates `x` and `z` independently, since they share no variable. Here the intersection is one mask-risc `And` over the two presence planes (`execute_into`, `Terminal::Keep`). Enumeration then reads both sides at each surviving slot. Every read is at the coordinate the mask names, in every operand; the `x` and `z` values are emitted and never used as addresses. That is the `Thought(t)[T_a[i] : T_b[i]]` shape of H14c/H14d, not a gather. For this query (one shared variable) the leapfrog join reduces to `And` plus enumeration. A star of k relations on `y` is an `And` chain.

**What the plane decides.** Disable runs showed that the presence-plane intersection is an *answer* only when the joint plane itself is read, as the existence mask (which `y` every relation has). For enumeration and counts it only prunes, because a slot where one side is empty emits nothing. Seven guards were each disabled, and each turned its test red: per-side generation, domain refusal, filter shape, filter in the program, `And` (vs `Or`), presence only for non-empty slots, and every side's plane in a star.

**Measured** (release, medians of 5, `cargo run --release --example join_cost`; MORK product-zipper build, leapfrog build within about 10%):

| Fixture | prepare/build: native · PathMap · MORK (ms) | set answers into a map: native · PathMap · MORK (ms) | native: stream answers · bag count · joint plane (µs) |
| --- | --- | --- | --- |
| cost.rs, n = 10 000 | 0.61 · 1.86 · 4.26 | 1.55 · 2.44 · 27.8 | 175 · 11.9 · 0.35 |
| cost.rs, n = 50 000 | 2.75 · 14.65 · 22.87 | 7.39 · 26.06 · 144.4 | 332 · 35.5 · 0.15 |
| heterogeneous (ids over `u32`, 50 % `y` overlap, S fan-out 3), \|R\| 50 000, \|S\| 75 000 | 5.78 · 24.27 · 60.46 | 22.80 · 44.72 · 461.0 | 251 · 16.2 · 0.31 |

- **The output dominates, not the join.** Native set answers into a `BTreeMap` cost 7.4 ms at n = 50 000; streaming the same answers costs 0.33 ms. The PathMap and MORK columns include the same kind of map insertion (MORK also its text decode, §6), so the fair comparison is the middle column.
- **A join amortizes into a fold when the consumer does not need tuples.** Over 1 024 queries at n = 50 000 (arithmetic, prepare + N × query): bag count 39 ms on native, against 7.6 s for native set answers and 26.7 s for PathMap answers. These answer different questions (count vs tuples), so this is a statement about demand, not a like-for-like speed-up.
- **Bags.** The native table keeps multiplicities and answers `Semantics::Bag` exactly. Both upstream indexes are sets and refuse (`SetOnly`).
- **Heterogeneous reseal.** Re-preparing one side costs 0.21 ms vs 2.47 ms for the whole table (n = 50 000); 1.17 ms vs 4.10 ms on the heterogeneous fixture.
- **Thoughts.** 64 thoughts over one shared table (403 KB at n = 50 000), each restricting `y` to its own eighth with a filter plane (1 568 B per thought), take 5.2 µs per thought for a bag count. The table is not copied per thought.

**What this does not cover.** Cyclic bodies: the triangle `R(a,b), S(b,c), T(a,c)` needs two variables co-addressed at once, which a one-variable domain cannot give. The 256 × 256 tile of D-RMAP-1A §8 does it for a small focus only, and nothing here covers large domains. Chain joins: stepping from `y` results to `z` slots over a large domain would need a scatter, which stays forbidden. Enumeration and the count's `Σ_y |R_y| · |S_y|` are plain loops; only the intersection runs on mask-risc, which delegates to `ndarray::simd`.

## 8. Deliverables

| D-id | Item | Status | Gate / evidence |
|---|---|---|---|
| D-MPS-0 | Source and type inventory; source switch through Cargo `[patch]`; upstream-diff count | MEASURED | §1, §2, §4; `cargo tree -i pathmap` = one package |
| D-MPS-1 | Differential probe: MORK (product-zipper and leapfrog builds) and PathMap zipper vs a nested-loop oracle; typed refusals; generation witness; cost | MEASURED | 8 integration tests + 1 unit test pass in both builds; 4 disable runs, each turning its guard's test red; clippy `-D warnings` clean both ways |
| D-MPS-2 | Upstream (MORK): name PathMap as a git/version dependency instead of `path = "../PathMap/"` | PROPOSAL | MORK as a git dependency resolves without `[patch]` |
| D-MPS-3 | Upstream (MORK): `nightly` as an opt-in feature (the fork's `de9312e`, 18 files) | PROPOSAL; the board entry of 2026-10-16 says the maintainer opens it | upstream `main` builds on stable; the fork branch can be dropped |
| D-MPS-4 | Upstream (MORK): do not enable PathMap `jemalloc` in a library dependency (leave the allocator to binaries) | PROPOSAL | a consumer binary can set its own allocator |
| D-MPS-5 | Upstream (MORK): `UnifyJoin` / `query_multi_leapfrog` generic over a cursor provider + reindex hook | PROPOSAL | MORK's own differential corpus unchanged on `PathMap<()>` |
| D-MPS-6 | Lance-owned PathMap-trait cursor over a sealed `RelationOrder` view | BLOCKED: `RelationOrder` is not built (D-RMAP PROPOSED) | must show no copy, or state the copy it needs |
| D-MPS-7 | Licence: no MORK source moved into or redistributed from lance-graph until MORK declares a licence (PR #148 unmerged) | ~~OPEN~~ DECIDED: MIT (operator, 2026-10-20: "Mork hat die MIT Entscheidung bereits auf dem Board bestätigt"). Source: Adam-Vandervorst, trueagi-io/MORK issue #2 "License", 2025-05-21: "Yes! It's now under MIT." (https://github.com/trueagi-io/MORK/issues/2; PR #148 follows up on it). The upstream LICENSE file (PR #148) is still unmerged as of the last check | the probe fetches MORK and vendors nothing; vendoring now needs only the MIT notice carried with the source |
| — | Native lance-graph arm for this join | ~~BLOCKED by the no-gather ruling~~ MEASURED as a probe (§7b); not admitted until the D-RMAP-1 BIND membrane review | a co-addressed producer for the input shape (`src/join.rs`); 8 + 2 tests vs the oracle; 7 guards disable-verified |
| D-MPS-8 | Versioned relationship table: one shared-variable domain, per-side generations, side reseal, filter planes per thought, star on one variable | MEASURED, probe | §7b; cyclic bodies and chain joins over large domains open |

## 9. Upstream contribution proposal (short)

Each item is useful to MORK without lance-graph, and none carries a Lance type:

1. **Dependency path** (D-MPS-2): `pathmap = { git = "https://github.com/Adam-Vandervorst/PathMap", … }` or a crates.io version in `[workspace.dependencies]`, keeping `path` only via a local `[patch]` for developers.
2. **Stable build** (D-MPS-3): move `nightly` out of `default` in `mork-expr`, `mork` and the workspace `pathmap` features, as `de9312e` does.
3. **Allocator** (D-MPS-4): drop `jemalloc` from the library's PathMap features and enable it from MORK's own binary (`kernel/src/main.rs`) instead.
4. **Generic join input** (D-MPS-5): `UnifyJoin<'a, Z>` with `Z: Zipper + ZipperMoving + ZipperPath + ZipperIteration`, the same bound `SubtermCursor` already uses, plus a hook for the reindex of inverted factors.

No upstream PR is opened by this work.
