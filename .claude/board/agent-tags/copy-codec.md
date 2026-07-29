# `copy-codec` — `derive(Clone, Copy)` verdicts across the four codec crates

**Run:** 2026-07-29 · branch `claude/x265-x266-plans-review-h9osnl`
**Scope:** every `Copy` derive in `crates/holograph`, `crates/highheelbgz`,
`crates/bgz17`, `crates/bgz-tensor`.
**Mode:** EDIT ONLY. No cargo run (orchestrator compiles centrally).
**Census consumed:** `.claude/board/exec-runs/copy-derive-blast-radius.txt`.
**Mandatory reads honoured:** `zero-copy-lens-law.md`,
`ndarray/.claude/rules/data-flow.md` (the file is in the ndarray repo, not
lance-graph — noted for the next brief), `encoding-ecosystem.md` (P0 before
codec work), `AGENT_LOG.md` (read, not written).

---

## Headline

**Zero edits made — and that is the correct outcome, not an abstention.**

- **1 VIOLATION in scope** (`holograph/src/bitpack.rs` `VectorSlice<'a>`) — it
  was **already fixed by the concurrent `copy-tierA` sibling** between my first
  and second read of the file. I verified the fix rather than duplicating it.
  Re-writing it would have been a lost-write race on a shared checkout.
- **93 other Copy derives: all LEGITIMATE** owned-value microcopies
  (data-flow.md §2). None carries a lifetime, a generic, or a reference field.
- **3 graded ELEVATED** (higher awareness stage than every input; one of them
  is legitimately stored today).
- **2 findings the derive census structurally cannot see** — reported below,
  not acted on.

## Census correction (mechanical)

The census's grep matched the literal `derive(Clone, Copy)`. An
order-insensitive scan of the same four crates finds **95 sites** (94 live +
`VectorSlice`'s, already stripped), where the census lists **73**. The **22
missed** all spell it `derive(Debug, Clone, Copy)`:

- `holograph` (10): `query/parser.rs:28,88,155,170,204` · `hamming.rs:37,209`
  · `graphblas/mod.rs:55` · `hdr_cascade.rs:82,612`
- `highheelbgz` (6): `lib.rs:86,151,176` · `simd_hardened.rs:62,73,86`
- `bgz17` (2): `clam_bridge.rs:37,197`
- `bgz-tensor` (4): `fractal_descriptor.rs:30,275` · `zipper.rs:50,161`

All 22 are value types, so no *verdict* changes — but the count does, and a
future sweep should match `Copy` as a word, not `Clone, Copy` as a phrase.
Extrapolated repo-wide, the true total is meaningfully above 369.

## The one VIOLATION — verified, not re-fixed

`crates/holograph/src/bitpack.rs:552` `VectorSlice<'a> { words: &'a [u64] }`.

Borrow-carrying, and worse than the general case: `as_words(&self) -> &'a [u64]`
re-exports the borrow at the **full** `'a`, so a `Copy` duplicate outlives every
scope a reviewer can see. The owner is an Arrow/mmap buffer some other mailbox
holds. Exactly the `WitnessLens` shape stripped in `b3515ba`.

The sibling's replacement comment is sound and cites the same ruling. I
independently verified the removal does not break anything, since that is the
half a report cannot establish for itself:

- Every consumer takes the slice **by reference**: `Belichtung::meter_ref(q,
  &slice)`, `StackedPopcount::compute_with_threshold_ref(q, &slice, r)`,
  `xor_ref(&slice, key)` — `storage.rs:541,590,641,797`, `navigator.rs:1098`,
  `bitpack.rs:946,964,993`.
- The only by-value use is a terminal **move**: `ZeroCopyCursor::next` returns
  `Some((id, slice, total))` after its borrows end (`navigator.rs:1098-1119`).
  `collect_all` drops it with `_`.
- **The sibling's own open worry (their tag line 108) is closed:** an exhaustive
  match for `VectorSlice` + `.clone()` / `: VectorSlice` field / `Vec<…>` /
  `[…]` over all of `crates/` returns **only** `get_slice -> Option<VectorSlice<'_>>`
  and two `use` lines. There is no `.clone()` in the `datafusion-storage`-gated
  block or anywhere else, and the type is never stored in a field.

## SIMD exception — checked, and NOT tripped (the P0 the other way)

`VectorSlice` *is* the type that exists to hand SIMD a slice — it is the whole
`Arrow buffer → &[u64] → cascade` path in `bitpack.rs:520-548`. Removing `Copy`
does **not** convert it to an owned copy: the `&[u64]` into the backing store is
untouched, and every SIMD consumer already takes `&dyn VectorRef`. The
data-flow.md §1 / borrow-strategy.md invariant survives the fix intact. This
needed saying out loud, because "the lens is the SIMD path" is precisely the
argument that would have been used to keep the derive.

## ELEVATED (higher rung than every input)

Honest framing first: none of these sit on the `persona-vs-rung-ladder` content
rungs (0–1 observation · 2 verb atoms · 3 NARS recipes · 4 StyleFamily). They
clear the law's *structural* bar — a value of a different KIND, computed across
multiple reads, not reproducible by a cast of any single lane.

| site | why it is an elevation |
|---|---|
| `bgz-tensor/src/hhtl_cache.rs:35` `RouteAction` | **Stored** (`HhtlCache.routes: Vec<RouteAction>`, k×k). `build_route_table` (`:402`) computes each cell from the pair's distance **plus the population-relative p25/p75 of the whole k×k distribution, per-entry perceptual weights, and a triangle-shortcut search over every intermediate `c`**. A fact about the SET; no cast reproduces it. Inputs = measured L1 distances (observation); output = a cascade verdict. The doc-comment already says it: *"NOT just distance — it's the routing decision."* |
| `bgz17/src/clam_bridge.rs:197` `Lfd` + `generative.rs:51` `LfdProfile` | Local fractal dimension over a neighbourhood + `lfd_median` across the scope + CHAODA `anomaly_score` — all population-relative. Transient today, so no lane is claimed; **tenant-eligible** if ever persisted. |
| `bgz17/src/clam_bridge.rs:37` `LayerStats` | Aggregate over the whole run (per-layer resolution counts). Population fact, not a member. |

## LEGITIMATE — the bulk (90 sites)

All owned, no borrow, all small. Representative groups:

- **Addresses** (the thing the law says you *should* copy instead of a view):
  `highheelbgz` `SpiralAddress` / `SpiralAddr` / `NeuronPrint` ×2. `simd_hardened.rs:14`
  states it outright: *"Not a projection — a READ INSTRUCTION into source data."*
  `holograph` `PackedDn(u64)` / `EdgeDescriptor(u64)`, both `repr(transparent)`.
- **Palette bytes + codebook indices** (VALUES, per the brief): `PaletteEdge`
  (3 B), `CodebookIndex(u16)`, `ScentByte(u8)`, `CrystalTriple`, `InlineEdge`,
  `HhtlDEntry` (4 B), `HhtlF32Entry` (1 B), `SlotL` ([i8;8]).
- **Bands / caps / modes** — named verbatim in data-flow.md §2:
  `bgz-tensor::belichtungsmesser::Band`, `QuarterSigmaBand` (its doc: *"NOT
  stored — computed on the fly"*), `bgz17::simd::SimdLevel` (a `CpuCaps`),
  `Precision`, `PaletteResolution`, `SigmaBand`, `CoarseBand`, all the
  `GrB*` op enums, `Dimension`, `BlockMask`.
- **Wire/storage headers** — `holograph/storage_transport.rs` `StorageHeader`
  (32 B, `repr(C, packed)`), `TransportHeader` (8 B), `StorageFlags`,
  `MetaBlock128`, `VersionFlags` + the `repr(u8)` enums. These are the
  *definition* of the bytes (round-tripped by `transmute_copy`), not a second
  reading of bytes that already have a projection. See the finding below.
- **Inline schema markers** — all 12 in `holograph/width_16k/schema.rs`. Each
  is a `pack`/`unpack` of bits held *inside* u64 words; no byte-aligned `&Self`
  cast exists to prefer, so there is no lens being passed over.
- **Cascade results** — `StackedPopcount`, `Belichtung`, `MexicanHat`,
  `TransformSpectrum`, `FractalDescriptor`, `ZipperDescriptor`, `PhaseDescriptor`.

## Findings the derive census cannot see (REPORTED, not acted on)

**F1 — `bgz-tensor/src/morton_cascade/mod.rs:34` `L4Tenant`: the missing lens
twin.** `from_bytes(&[u8; 12]) -> Self` materializes the V3 L4 palette tenant —
the same 12-byte-register-in-a-512-byte-stride geometry as `CausalWitnessFacet`,
which has the free `from_register_ref` cast. `L4Tenant` **cannot** have that
twin as written: it is not `repr(C)`/`repr(transparent)`, and its
`[(u8, u8); 3]` fields are Rust tuples, which carry **no layout guarantee** — a
`&[u8;12] → &Self` cast today would be unsound, not merely absent. The repair is
`repr(C)` (or a `[u8;12]` newtype) + a `ref_from_bytes`, which is a design change
well past "remove a derive," so I left it. Removing `Copy` here would be the
worst of both: it forbids the microcopy without creating the lens.
Also note `L4Tenant` has **zero consumers outside its own module tree** —
adjacent to the law's *"a type whose only constructors are `#[cfg(test)]` is a
shadow of storage"* clause. Worth a producer check before it grows.

**F2 — `bgz17/src/scope.rs:31-45` `Bgz17Scope` stores THREE per-edge lanes over
the same edges** (`scent: Vec<u8>`, `palette_indices: Vec<PaletteEdge>`,
`base_patterns: Vec<SpoBase17>`). By the letter of *"a projection is never
stored"* this is three stored readings of one content. **I did not rule on it,
because ruling would decide an architecture question above my brief:** this is
the certified HHTL precision ladder (`Precision::{Scent, Palette, Base, Exact}`,
`bgz17/src/lib.rs:78`; the atlas chain in `encoding-ecosystem.md` with measured
ρ = 0.937 / 0.965 / 0.992). Each rung is *lossy* relative to the one above, so
no cast reproduces a coarse lane from a fine one, and the coarse lane exists
precisely to avoid reading the fine one. Two readings are available and they
disagree: **(a)** a lossy coarsening is the same awareness stage at lower
resolution → "never stored"; **(b)** `scent` here is computed against the
**scope centroid** (`scope.rs:79-88`), making it population-relative and thus
elevation-shaped, not a pure coarsening. This is a live tension between the
zero-copy law and a FINDING-graded certified cascade, and it deserves an
operator ruling, not a subagent's edit. No `Copy` derive causes it — the `Vec`
fields do.

## Coordination note

`copy-tierA` and I overlapped on exactly one file. The shared checkout has no
lease, so the only thing that prevented a lost write was re-reading the file
before editing. Recommend the orchestrator partition future sweeps by **file**,
not by tier — a tier cuts across every crate boundary, so tier-scoped and
crate-scoped workers are guaranteed to collide.
