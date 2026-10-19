# MORK / PathMap Schnellverschluss — v1

**Status:** D-MPS-0 and D-MPS-1 MEASURED (probe `crates/mork-pathmap-probe`, results only). D-MPS-2..7 PROPOSAL / BLOCKED as listed in §8. No production executor, no CE64 change, no Lance storage, no upstream file edited by this work.

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

Rows `R(i, y)` with `y` drawn from `n / 4` values, `S(y, z)` one row per `y`, answers ≈ n. `amort_k` = build/k + query: one shared index serving k thoughts. `rebuild_k64` = 64 × (build + query), an index rebuilt per thought.

| join | backend | n | build ms | ns/row | query ms | amort k=1 | k=8 | k=64 | rebuild k=64 |
|---|---|---|---|---|---|---|---|---|---|
| product-zipper | PathMapZipper | 1 000 | 0.265 | 212 | 0.203 | 0.468 | 0.236 | 0.207 | 29.95 |
| | Mork | 1 000 | 0.518 | 415 | 3.032 | 3.551 | 3.097 | 3.041 | 227.3 |
| | PathMapZipper | 10 000 | 2.216 | 177 | 3.846 | 6.062 | 4.123 | 3.880 | 388.0 |
| | Mork | 10 000 | 4.822 | 386 | 28.744 | 33.57 | 29.35 | 28.82 | 2 148 |
| | PathMapZipper | 50 000 | 17.75 | 284 | 21.98 | 39.73 | 24.20 | 22.26 | 2 543 |
| | Mork | 50 000 | 27.25 | 436 | 170.2 | 197.5 | 173.6 | 170.7 | 12 639 |
| leapfrog | PathMapZipper | 50 000 | 14.89 | 238 | 20.50 | 35.39 | 22.36 | 20.73 | 2 265 |
| | Mork | 50 000 | 31.39 | 502 | 142.2 | 173.6 | 146.1 | 142.7 | 11 107 |

What it shows, and what it does not:

- **Amortization is real at the boundary.** Sharing one bound index across 64 thoughts instead of rebuilding it per thought cuts total cost about 110× (PathMap, n = 50 000: 22.3 ms vs 2 543 ms).
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
| D-MPS-7 | Licence: no MORK source moved into or redistributed from lance-graph until MORK declares a licence (PR #148 unmerged) | OPEN | the probe fetches MORK and vendors nothing |
| — | Native lance-graph arm for this join | BLOCKED by the no-gather ruling | a co-addressed producer for the input shape |

## 9. Upstream contribution proposal (short)

Each item is useful to MORK without lance-graph, and none carries a Lance type:

1. **Dependency path** (D-MPS-2): `pathmap = { git = "https://github.com/Adam-Vandervorst/PathMap", … }` or a crates.io version in `[workspace.dependencies]`, keeping `path` only via a local `[patch]` for developers.
2. **Stable build** (D-MPS-3): move `nightly` out of `default` in `mork-expr`, `mork` and the workspace `pathmap` features, as `de9312e` does.
3. **Allocator** (D-MPS-4): drop `jemalloc` from the library's PathMap features and enable it from MORK's own binary (`kernel/src/main.rs`) instead.
4. **Generic join input** (D-MPS-5): `UnifyJoin<'a, Z>` with `Z: Zipper + ZipperMoving + ZipperPath + ZipperIteration`, the same bound `SubtermCursor` already uses, plus a hook for the reindex of inverted factors.

No upstream PR is opened by this work.
