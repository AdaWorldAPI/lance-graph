# D-BLW-5 API inventory — Sonnet grindwork lane

> Edit-only. No cargo run of any kind. All signatures below were read from
> source in the same pass that wrote this file. Compliance with
> `.claude/v3/knowledge/sonnet-worker-guardrails.md` §1: full-file reads,
> no invented types, no board-file writes other than this one file. Branch
> `claude/x265-x266-plans-review-h9osnl` was NOT switched.

---

## A. `BeliefArena` + NARS revision

**Location:** `crates/lance-graph-planner/src/nars/belief.rs` (the type
`super::belief::BeliefArena` imported by `crates/lance-graph-planner/src/nars/stance.rs:24`).
Module path: `lance_graph_planner::nars::BeliefArena` — re-exported from
`crates/lance-graph-planner/src/nars/mod.rs` (confirmed via the
`use lance_graph_planner::nars::{BeliefArena, CStmt};` import in
`crates/lance-graph-supervisor/tests/d_ign_b_lenses.rs:110`).

### Stamp (`belief.rs:31`)

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Stamp(pub u64);

impl Stamp {
    pub fn source(id: u32) -> Self;          // Stamp(1u64 << (id % 64))
    pub fn disjoint(self, other: Self) -> bool;
    pub fn union(self, other: Self) -> Self;
}
```

### Copula / CStmt (`belief.rs:54,77`)

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Copula {
    Inh,
    Sim,
    Impl,
    Rel(u16),
}
impl Copula {
    pub fn transits(self) -> bool; // matches!(self, Inh | Sim)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CStmt {
    pub s: u16,
    pub cop: Copula,
    pub p: u16,
}
```

### Belief entry type — exact fields (`belief.rs:88-104`)

```rust
#[derive(Debug, Clone)]
pub struct Belief {
    pub stmt: CStmt,
    pub truth: TruthValue,     // frequency/confidence — see below
    pub stamp: Stamp,          // evidential base (S4)
    pub rung: u32,             // Tarski rung; 0 = observed
    pub premises: Vec<u32>,    // arena indices, derived beliefs only
    pub contradiction: f32,    // preserved max |f1-f2| across revisions
}
```

There is **no** separate "provenance" or "verse" field on `Belief` itself
— that lives in `stance::Provenance` (a SEPARATE struct the caller
maintains alongside the arena; `stance.rs:90-99`: `{ verse: String, stmt:
CStmt, negated: bool }`). The arena itself carries no rung-1/verse
metadata.

### ReviseOutcome (`belief.rs:107-122`)

```rust
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReviseOutcome {
    Admitted { id: u32 },
    Revised { id: u32, synthesis_c: f32, depth: f32 },
    Chosen { id: u32, kept_existing: bool },
}
```

### BeliefArena — full public surface (`belief.rs:129-337`)

```rust
#[derive(Debug, Default)]
pub struct BeliefArena {
    // entries: Vec<Belief>, index: HashMap<CStmt, u32> — both private
    pub passes: u32,
    pub reached_fixed_point: bool,
}

impl BeliefArena {
    pub fn new() -> Self;                                   // #[must_use]
    pub fn entries(&self) -> &[Belief];                      // #[must_use]
    pub fn get(&self, stmt: CStmt) -> Option<&Belief>;        // #[must_use]

    /// Admission path #1 — observation. Absent -> Admitted; present ->
    /// routes through revise_at (disjoint stamp -> Revised, else Chosen).
    pub fn observe(&mut self, stmt: CStmt, truth: TruthValue, stamp: Stamp)
        -> ReviseOutcome;

    /// The S4 revision guard on an EXISTING belief id.
    pub fn revise_at(&mut self, id: u32, new: TruthValue, stamp: Stamp)
        -> ReviseOutcome;

    /// Admission path #2 — derived candidate (no observation source of its
    /// own). Ground (non-empty-stamp) beliefs are NEVER overwritten; a
    /// pure-derived belief updates only when the candidate's
    /// `expectation()` strictly exceeds the stored one (+1e-6 epsilon).
    /// Returns whether the arena changed.
    pub fn admit_derived(&mut self, stmt: CStmt, truth: TruthValue,
        premises: &[u32], rung: u32) -> bool;

    /// Copula-gated transitive closure (Inh/Sim only), NARS deduction
    /// truth per pair, CHOICE on expectation(), true fixed point or
    /// `max_passes` backstop. Sets `self.passes` / `self.reached_fixed_point`.
    pub fn close_transitive(&mut self, max_passes: u32);
}
```

**Can an externally-constructed belief be inserted directly (bypassing
`stream`'s text-parsing path)?** YES, on both admission paths:

- `arena.observe(CStmt { s, cop, p }, TruthValue::new(f, c), Stamp::source(id))`
  — hand-built `CStmt`/`TruthValue`/`Stamp`, no text parsing involved
  (this is exactly what `belief.rs`'s own `#[cfg(test)]` module does, e.g.
  `revision_disjoint_moves_truth_and_terminates`, `belief.rs:355-385`).
- `arena.admit_derived(stmt, truth, premises, rung)` — the derived-candidate
  path; also fully hand-constructible, no text.

`stream` (in `stance.rs`, see §B) is ONE caller of `observe`/`admit_derived`
via its own tokenizer, not the only way to populate an arena. A D-BLW-5
build wanting programmatic beliefs (not KJV text) can call `observe`/
`admit_derived` directly against a fresh `BeliefArena::new()`.

### TruthValue — exact fields (`crates/lance-graph-planner/src/nars/truth.rs:8-15`)

```rust
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TruthValue {
    pub frequency: f32,   // 0..1, proportion of positive evidence
    pub confidence: f32,  // 0..1, total evidence weight
}
impl TruthValue {
    pub fn new(frequency: f32, confidence: f32) -> Self;   // clamps both to 0..1
    pub fn expectation(&self) -> f32;                       // confidence*(freq-0.5)+0.5
    pub fn surprise(&self, prior: f32) -> f32;
    pub fn evidence_weight(&self) -> f32;                   // c/(1-c), f32::MAX at c>=1
    pub fn revise(&self, other: &TruthValue) -> TruthValue;  // NARS revision
    pub fn deduction(&self, other: &TruthValue) -> TruthValue;
    pub fn induction(&self, other: &TruthValue) -> TruthValue;
    pub fn abduction(&self, other: &TruthValue) -> TruthValue;
    pub fn analogy(&self, sim: &TruthValue) -> TruthValue;
}
impl Default for TruthValue { /* frequency: 0.5, confidence: 0.0 */ }
```

No separate `Fingerprint`/hashed-distance truth path exists in this
module — truth moves ONLY through the `TruthValue` methods above (per
`belief.rs`'s own module doc: "the arena … moves truth ONLY by the one
engine's truth functions").

**Query/read surface:** `arena.entries()` (whole slice, admission order),
`arena.get(stmt)` (point lookup by exact `CStmt`). There is no filtered/
indexed query beyond these two — any subject/copula-scoped view (e.g.
`stance_panel`'s Wittgenstein games map, `stance.rs:512-532`) is built by
the CALLER iterating `entries()`, not by an arena method.

---

## B. The stance/readout surface as consumed by `d_ign_b_lenses.rs`

File: `crates/lance-graph-supervisor/tests/d_ign_b_lenses.rs`
(feature-gated `#[cfg(feature = "cycle-driver")]`, module
`d_ign_b_lenses`).

### Imports of the stance surface (`d_ign_b_lenses.rs:109-110`)

```rust
use lance_graph_planner::nars::stance::{stance_panel, stream, FlipKind, Interner, ReadOut};
use lance_graph_planner::nars::{BeliefArena, CStmt};
```

### Construction per owner — verbatim call site (`run_lens`, `d_ign_b_lenses.rs:690-706`)

```rust
fn run_lens(z: u8, verses: &[(String, String)]) -> LensReadout {
    let mut arena = BeliefArena::new();
    let mut intern = Interner::new();
    let mut out = ReadOut::default();
    stream(verses, &mut arena, &mut intern, &mut out, false);
    let (hegel, nietzsche, kant, wittgenstein) = stance_panel(&arena, &intern, &out);
    match z {
        1 => LensReadout::Hegel(hegel),
        2 => LensReadout::Nietzsche(nietzsche),
        3 => LensReadout::Kant(kant),
        4 => LensReadout::Wittgenstein(wittgenstein),
        other => panic!("run_lens: z={other} is outside the armed range 1..=4 …"),
    }
}
```

**One arena is built fresh per call** — there is no shared/cached arena
across owners in this file. `stream`'s exact signature
(`stance.rs:161-167`):

```rust
pub fn stream(
    verses: &[(String, String)],   // (label, text) pairs
    arena: &mut BeliefArena,
    intern: &mut Interner,
    out: &mut ReadOut,
    pass2: bool,
);
```

`(String, String)` pairs are built by `labelled_verses` (`d_ign_b_lenses.rs:395-402`):
label format `"kjv:{global_index:05}"`, matching `blw_fusion.rs:913`'s
subject format (`format!("kjv:{row:05}")`) — the SAME subject-string
convention both files use, confirmed by direct read of both sites.

### Readouts derived — `stance_panel` (`stance.rs:469-478`, quoted verbatim)

```rust
#[allow(clippy::type_complexity)]
pub fn stance_panel(
    arena: &BeliefArena,
    intern: &Interner,
    out: &ReadOut,
) -> (
    Vec<(CStmt, f32)>,        // Hegel: Aufhebung ranking
    Vec<(CStmt, FlipKind)>,   // Nietzsche: genealogy partition
    Vec<(String, f32, f32)>,  // Kant: (lift label, graded quale, ablated quale)
    Vec<(u16, usize)>,        // Wittgenstein: (concept, distinct games)
)
```

This is ONE call returning all four stances as one 4-tuple — there is no
per-stance dispatch function. `d_ign_b_lenses.rs` never calls
`contradiction_ranking` directly for its lens selection (it goes through
`stance_panel`, which itself calls `contradiction_ranking` internally for
the Hegel element, `stance.rs:480`).

### `LensReadout` — probe-local type, NOT a shipped contract type (`d_ign_b_lenses.rs:614-685`)

```rust
#[derive(Debug)]
enum LensReadout {
    Hegel(Vec<(CStmt, f32)>),
    Nietzsche(Vec<(CStmt, FlipKind)>),
    Kant(Vec<(String, f32, f32)>),
    Wittgenstein(Vec<(u16, usize)>),
}
impl LensReadout {
    fn is_empty(&self) -> bool;
    /// Stable fold over the variant's own contents (floats via .to_bits()).
    /// Deliberately NO variant-discriminant tag (falsifiability rule —
    /// see the doc comment at d_ign_b_lenses.rs:632-646).
    fn digest(&self) -> u64;
}
```

`digest()` folds ONLY the variant's payload, never a type tag — two
EMPTY readouts of different lenses hash equal by design (this is called
out explicitly as load-bearing for the L3/L4 non-vacuity checks).

### `run_all_lenses` — single-owner cross-lens helper (`d_ign_b_lenses.rs:710-722`)

```rust
fn run_all_lenses(verses: &[(String, String)]) -> [LensReadout; 4]
```
Same construction as `run_lens` but keeps all four tuple elements instead
of selecting one.

### Selection ordinal source — `owner.meta_at(0).thinking()`

`d_ign_b_lenses.rs:575,858`: the arming ordinal `z` is read from
`owner.meta_at(0).thinking()` — a `MetaWord`'s packed 6-bit `thinking`
field (see §F for `MetaWord`). This is the SAME field
`plan_context_for`/`thinking_style_for` (§F) consume for the
`StyleStrategy` dispatch input — one field, two consumers (lens
selection vs `ThinkingStyle` mapping), as the module doc's "deviation 2"
states explicitly.

---

## C. `jc` oracle — `crates/jc/src/stats.rs`

### `BinaryAssociation` — full struct (`stats.rs:612-634`)

```rust
/// A 2x2 contingency table with both marginals, agreement decomposition,
/// and the two association coefficients that read off it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BinaryAssociation {
    pub n00: u64,                    // count of (false, false)
    pub n01: u64,                    // count of (false, true)
    pub n10: u64,                    // count of (true, false)
    pub n11: u64,                    // count of (true, true)
    pub positive_rate_a: f64,        // rate of true in rater A
    pub positive_rate_b: f64,        // rate of true in rater B
    pub observed_agreement: f64,     // p_o
    pub expected_agreement: f64,     // p_e (chance agreement from marginals)
    pub kappa: Option<f64>,          // Cohen's kappa; None iff p_e == 1
    pub phi: Option<f64>,            // phi; None iff either variable constant
}
```

### Constructor path from two `&[bool]` (`stats.rs:653-693`)

```rust
pub fn binary_association(a: &[bool], b: &[bool]) -> Option<BinaryAssociation>
```

Returns `None` ONLY on structurally unusable input (length mismatch or
empty). A degenerate table (all-same-category) still returns
`Some(BinaryAssociation { .. })` with `kappa`/`phi` individually `None`
— the counts/marginals/agreement fields remain populated even when the
coefficients are undefined. `phi` is delegated to
`crate::reliability::pearson` on the two vectors cast to `0.0/1.0` f64
(`stats.rs:596-600`); `kappa` is computed inline from the 2x2 counts
(NOT delegated — it duplicates the arithmetic of the standalone
`cohen_kappa` function for the binary case, since `cohen_kappa` takes
`&[usize]` category labels rather than a pre-tabulated 2x2).

### Degeneracy contracts, precisely

- **`kappa == None`** iff `expected_agreement == 1.0` (or non-finite) —
  i.e. `p_e == 1`, which happens when the marginals make chance agreement
  certain (both raters use one identical category throughout, or the
  complementary boundary case). Doc comment (`stats.rs:630`): "or `None`
  when `p_e == 1` (undefined, `0/0`)."
- **`phi == None`** iff either input vector is constant (all-true or
  all-false — zero variance), per `pearson`'s own contract (`phi`'s doc
  comment, `stats.rs:588`: "Returns `None` under [`pearson`]'s
  conditions: lengths differ, `n < 2`, or either vector is constant.").
- Neither degeneracy voids the OTHER field: a run can get `kappa: None,
  phi: Some(x)` or vice versa, since they degenerate under DIFFERENT
  conditions (kappa on `p_e==1`, phi on constancy of either input alone).

### `blw_fusion.rs`'s call site — verbatim (`examples/blw_fusion.rs:91,1115-1118`)

```rust
use jc::stats::{binary_association, BinaryAssociation};
// …
let assoc_zz = binary_association(&z_strict, &z_aware);
if let Some(zz) = assoc_zz {
    print_association_table("G2 kappa(Z,Z)", &zz);
}
```

`print_association_table` (`blw_fusion.rs:662-682`) takes `&BinaryAssociation`
and formats every field (never a bare kappa/phi scalar) — the C8
correction cited in that file's module doc ("every kappa ships the FULL
`BinaryAssociation` table").

### `jc` dependency wiring — confirmed by direct Cargo.toml reads

- **`lance-graph-planner/Cargo.toml`** (`[dev-dependencies]`, read in
  full): `jc = { path = "../jc" }` — **dev-dependency only**, with an
  explicit comment: "**dev-only, never a production dependency of the
  planner**." This is what makes `examples/blw_fusion.rs` (an example,
  which compiles under dev-deps) able to `use jc::stats::...`.
- **`lance-graph-supervisor/Cargo.toml`** (full file read): dependencies
  are `lance-graph-callcenter`, `lance-graph-contract`,
  `lance-graph-planner` (optional, `cycle-driver` feature),
  `thiserror`, `tracing`, `ractor` (optional), `static_assertions`
  (optional), `tokio` (optional); `[dev-dependencies]` are `tokio`,
  `static_assertions`, `cognitive-shader-driver`. **`jc` appears
  NOWHERE in this manifest**, direct or dev.

### The load-bearing placement consequence (verified, stated precisely)

- **`crates/lance-graph-supervisor/tests/*.rs`** (e.g.
  `d_ign_b_lenses.rs`, `probe_ignition.rs`) compile against
  `lance-graph-supervisor`'s own dependency graph. That graph has NO
  `jc` edge (direct or transitive-usable — Rust does not let a crate
  `use` a dependency's OWN dependency unless it is re-exported, and `jc`
  is not re-exported by `lance-graph-planner`). **A supervisor test
  file cannot `use jc::stats::*` without a manifest change to
  `lance-graph-supervisor/Cargo.toml`.** This is exactly why
  `d_ign_b_lenses.rs`'s own module doc says (verbatim, lines 42-43):
  "`jc` is also not a dependency of this crate (`lance-graph-supervisor/
  Cargo.toml` has no `jc` edge — a manifest change, not a worker's call)."
- **`crates/lance-graph-planner/examples/*.rs`** (e.g. `blw_fusion.rs`)
  compile against `lance-graph-planner`'s dev-dependency graph, which
  DOES include `jc`. But planner **examples** cannot reach
  `lance_graph_supervisor::cycle_driver::run_cycle` /
  `run_cognitive_work_gated_over` — `lance-graph-planner`'s own
  Cargo.toml (read in full above) has NO dependency, dev or otherwise,
  on `lance-graph-supervisor` (confirmed: the dependency edge is
  ONE-WAY, supervisor -> planner, per `lance-graph-supervisor`'s own
  module doc at `cycle_driver.rs:41-45`: "This driver depends **one-way**
  on the planner … planner never deps supervisor — no cycle").

**Precise statement for D-BLW-5 placement:** today, NEITHER crate sees
BOTH `jc::stats::binary_association` AND
`lance_graph_supervisor::cycle_driver::run_cycle` from the same
compilation unit. A planner example sees `jc` but not `run_cycle`; a
supervisor test sees `run_cycle` (via `cycle-driver` feature) but not
`jc`. Any D-BLW-5 test that needs BOTH the fusion oracle AND the real
cycle-driven cast/scan/seal machinery in one file requires adding `jc`
as a dev-dependency of `lance-graph-supervisor/Cargo.toml` — a manifest
change outside a Sonnet grindwork lane's scope (guardrails §5.2: "A
needed type/lane/mask does not exist" / dependency wiring — STOP+report,
needs orchestrator/operator sign-off, not silently assumed here).

---

## D. Percentile / bucketing — `ndarray::simd::cascade` and `ndarray::hpc::statistics::percentile` reachability

Sibling repo: `/home/user/ndarray` (read directly; NOT part of this
lance-graph checkout).

### `ndarray::simd::cascade` — REACHABLE, confirmed re-export

`/home/user/ndarray/src/simd.rs:626-633` (verbatim):

```rust
// The Belichtungsmesser — banded multi-resolution cascade search
// (`Cascade::expose(distance) → Band`, `recalibrate(ShiftAlert)`,
// `PackedDatabase`, `adaptive_resolution`). Trampolined as a whole module so
// consumers under the "all SIMD from `ndarray::simd`" invariant reach the
// exposure-meter surface as `ndarray::simd::cascade::*` without dipping into
// `crate::hpc` directly. Module alias, not an item list — new cascade items
// arrive here without a re-export edit. Same `std` gate as this module.
pub use crate::hpc::cascade;
```

The underlying module is `/home/user/ndarray/src/hpc/cascade.rs`. Feature
gating: `pub mod simd;` in `ndarray/src/lib.rs:241` is `#[cfg(feature =
"std")]` only (NOT gated on `hpc-extras`); `pub mod hpc;` at
`ndarray/src/lib.rs:500` is likewise `#[cfg(feature = "std")]` only.
`std` is in ndarray's `default` feature set (`Cargo.toml:276`:
`default = ["std", "hpc-extras"]`), so `ndarray::simd::cascade::*` is
reachable under plain default features — `hpc-extras` is not required
for this specific path (though it happens to be enabled too wherever
ndarray is pulled with defaults).

`Cascade`'s public surface (`ndarray/src/hpc/cascade.rs`, grepped
signatures):

```rust
pub struct RankedHit { /* … */ }
pub enum Band { /* Foveal / … / Reject, per test at cascade.rs:472-473 */ }
pub struct ShiftAlert { /* … */ }
pub enum PreciseMode { /* … */ }
pub struct Cascade { /* … */ }
impl Cascade {
    pub fn mu(&self) -> f64;
    pub fn sigma(&self) -> f64;
    pub fn observations(&self) -> usize;
    pub fn from_threshold(threshold: u64, vec_bytes: usize) -> Self;
    pub fn calibrate(distances: &[u32], vec_bytes: usize) -> Self;
    pub fn expose(&self, distance: u32) -> Band;
    pub fn test(&self, a: &[u8], b: &[u8]) -> bool;
    pub fn observe(&mut self, distance: u32) -> Option<ShiftAlert>;
    pub fn recalibrate(&mut self, alert: &ShiftAlert);
    pub fn query(&self, query: &[u8], database: &[u8], vec_bytes: usize,
        num_vectors: usize) -> Vec<RankedHit>;
    pub fn query_candidates(/* … */) -> /* … */;
    pub fn query_precise(/* … */) -> /* … */;
}
pub fn adaptive_resolution(query_entropy: f32, corpus_cv: f32) -> Band;
pub struct PackedDatabase { /* … */ }
impl PackedDatabase {
    pub fn pack(database: &[u8], vec_bytes: usize) -> Self;
    pub fn cascade_query(&self, query: &[u8], cascade: &Cascade, top_k: usize)
        -> Vec<RankedHit>;
}
```

(Field-level detail of `RankedHit`/`Band`/`ShiftAlert`/`PreciseMode` NOT
individually verified beyond the grep of struct/impl headers above — see
Not Verified section.)

### `ndarray::hpc::statistics::percentile` — REACHABLE, as a trait method

`percentile` is NOT a free function — it is a method on the
`Statistics<A>` trait (`ndarray/src/hpc/statistics.rs:22-41`, verbatim):

```rust
pub trait Statistics<A> {
    fn median(&self) -> A;
    fn variance(&self) -> A;
    fn var_axis(&self, axis: Axis) -> Array<A, IxDyn>;
    fn std_dev(&self) -> A;
    fn std_axis(&self, axis: Axis) -> Array<A, IxDyn>;
    /// Percentile (0-100). Uses linear interpolation between nearest ranks.
    fn percentile(&self, p: A) -> A;
    fn sorted(&self) -> Array<A, Ix1>;
    fn argmin(&self) -> usize;
    fn argmax(&self) -> usize;
    fn top_k(&self, k: usize) -> (Vec<usize>, Vec<A>);
    fn cumsum(&self) -> Array<A, Ix1>;
    fn cosine_similarity(&self, other: &Self) -> A;
    // (module continues past the grepped window — not all methods listed)
}
```

Usage requires `use ndarray::hpc::statistics::Statistics;` in scope (the
trait method, called as `x.percentile(50.0)`), per the module's own
doctest (`statistics.rs:14-21`). Module path `ndarray::hpc::statistics`
is public (`pub mod statistics;` in `ndarray/src/hpc/mod.rs:27`), gated
by the same `#[cfg(feature = "std")]` on `pub mod hpc;` noted above.

### Reachability from THIS workspace's crates — verified by manifest read

- **`lance-graph-planner/Cargo.toml`** `[dependencies]` (full read):
  ```toml
  ndarray = { path = "../../../ndarray", default-features = false,
              features = ["std", "hpc-extras"] }
  ```
  `std` is explicitly enabled, so `ndarray::simd::cascade` and
  `ndarray::hpc::statistics::Statistics::percentile` are BOTH reachable
  from `lance-graph-planner` (its lib code and its examples, including
  `blw_fusion.rs`) — direct dependency, non-optional, always compiled.
- **`lance-graph-supervisor/Cargo.toml`** (full read, quoted above under
  §C): **no `ndarray` dependency at all**, direct or dev. A
  `lance-graph-supervisor` test file (`tests/d_ign_b_lenses.rs`,
  `tests/probe_ignition.rs`, or a new D-BLW-5 test) **cannot** `use
  ndarray::...` of any kind without adding `ndarray` to
  `lance-graph-supervisor/Cargo.toml` — the SAME class of gap as the
  `jc` gap in §C (a manifest change, not something the existing crate
  graph already grants).

**Plain statement for D-BLW-5:** if percentile/bucketing work is wanted
INSIDE `lance-graph-supervisor`'s test tree (alongside the real
`run_cycle`/`MailboxSoA` machinery `d_ign_b_lenses.rs` and
`probe_ignition.rs` already use), that is currently impossible without a
manifest edit. It IS possible today from `lance-graph-planner` (lib code
or examples), where `ndarray` is already a live dependency with `std`
enabled.

---

## E. Version stamping — `blw_fusion.rs` and `persist_sink.rs`

### How sealed versions are obtained

- **`sink.head()`** (a caller-defined helper on the in-process `MemWal`
  fake, NOT a `WalSink` trait method — `blw_fusion.rs:417-424`,
  identical shape in `d_ign_b_lenses.rs:316-323`):
  ```rust
  fn head(&self) -> DatasetVersion {
      self.sealed.lock().expect("MemWal poisoned")
          .last().map_or(DatasetVersion(0), |s| s.version)
  }
  ```
  This reads the LAST sealed `DatasetVersion` from the fake's own
  internal `Vec<SealedCycle>` — it is test/example-harness scaffolding,
  not part of the shipped `persist_sink`/`cycle_driver` API.

- **The shipped version source is `persist_cycle`'s return value**
  (`persist_sink.rs:335-362`, verbatim signature):
  ```rust
  pub async fn persist_cycle<S: WalSink>(
      sink: &S,
      frame: CycleFrame,
      casts: Vec<SweepSlot>,
  ) -> Result<DatasetVersion, PersistError>
  ```
  `blw_fusion.rs:872` calls it directly:
  ```rust
  let version = persist_cycle(&sink, CycleFrame::new(spec.id, base), slots).await?;
  ```
  and stamps it: `let vc: LanceVersion = version.0; sealed_versions.insert(c, vc);`
  (`blw_fusion.rs:895-896`) — `DatasetVersion` is a `pub struct
  DatasetVersion(pub u64)` (`lance-graph-contract/src/scheduler.rs:36`),
  so `.0` is the raw `u64` and `LanceVersion` (from
  `lance_graph_planner::temporal`) is a type alias/newtype over the same
  representation used as the `deinterlace`/`QueryReference::at` horizon.

- **In `d_ign_b_lenses.rs`**, the higher-level `run_cycle` wraps both the
  seal and the apply in one call (`cycle_driver.rs:446-471`, quoted in
  full in §F) and returns `CycleOutcome { sealed: SealedCycle, applied:
  AppliedCycle, held: Vec<HeldIntent> }`; the version is
  `outcome.sealed.version` (field on `SealedCycle`,
  `cycle_driver.rs:103-114`):
  ```rust
  pub struct SealedCycle {
      pub version: DatasetVersion,
      pub transitions: Vec<SealedTransition>,
      pub next_position_base: u64,
  }
  ```
  `d_ign_b_lenses.rs` itself never reads `outcome.sealed.version`
  directly (it reads `outcome.sealed.next_position_base` and
  `outcome.sealed.transitions`, `d_ign_b_lenses.rs:918,923-949`) — but
  the field is there and is the version-stamp equivalent of
  `blw_fusion.rs`'s `persist_cycle` return.

### What a version-stamped one-shot record would key on

A `SweepSlot` (`persist_sink.rs:127-151`, the durable-write unit both
files use) carries NO version field itself — the version is assigned
AFTER sealing, one per whole cycle, not per slot:

```rust
pub struct SweepSlot {
    pub cycle: CycleId,
    pub stream_position: u64,      // cross-cycle monotonic order key
    pub owner: MailboxId,
    pub row: u64,
    pub paired_move: Option<KanbanMove>,
    pub payload: Vec<u8>,
}
```

The version-stamp KEY, as read back, is `LandedSlot`
(`persist_sink.rs:157-161`):

```rust
pub struct LandedSlot {
    pub version: DatasetVersion,   // the version its CYCLE sealed into
    pub slot: SweepSlot,
}
```

So a version-stamped one-shot record's natural key is
`(version: DatasetVersion, slot.owner: MailboxId, slot.row: u64)` or
`(version, slot.stream_position)` — `blw_fusion.rs`'s own
`VerdictRow::lance_version()` (its `DeinterlaceRow` impl,
`blw_fusion.rs:231-247`) keys on exactly this: `horizon: u64` set from
`vc` (the `persist_cycle`-returned version's `.0`), paired with
`subject: String` (the row's stable text key, `"kjv:NNNNN"`).

`CycleFrame` (`persist_sink.rs:104-120`) is the storage-identity input
side (`{ cycle: CycleId, base_version: DatasetVersion }`, constructed via
`CycleFrame::new(cycle, base_version)`) — it carries the SEALED
PREDECESSOR a cycle reads, not the version it produces; the produced
version only exists after `commit_cycle`/`persist_cycle` returns.

---

## F. MetaWord + cohort scaffolding — reusable helper signatures

All of these are copied (with provenance comments) between
`probe_ignition.rs` and `d_ign_b_lenses.rs`; signatures below are
identical in both files unless noted.

### `MetaWord` (`lance-graph-contract/src/cognitive_shader.rs:44-76`)

```rust
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(transparent)]
pub struct MetaWord(pub u32);   // thinking(6) + awareness(4) + nars_f(8) + nars_c(8) + free_e(6)

impl MetaWord {
    pub const fn new(thinking: u8, awareness: u8, nars_f: u8, nars_c: u8, free_e: u8) -> Self;
    pub fn thinking(&self) -> u8;    // low 6 bits
    pub fn awareness(&self) -> u8;   // next 4 bits
    pub fn nars_f(&self) -> u8;
    pub fn nars_c(&self) -> u8;
    pub fn free_e(&self) -> u8;
}
```

Both files construct arming via `MetaWord::new(armed, 0, 0, 0, 0)` and
read it back via `owner.meta_at(0).thinking()`.

### `QualiaI4_16D` (`lance-graph-contract/src/qualia.rs:175-208`)

```rust
#[repr(C, align(8))]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct QualiaI4_16D(pub u64);
impl QualiaI4_16D {
    pub const ZERO: Self;
    pub fn get(self, dim: usize) -> i8;
    pub fn set(&mut self, dim: usize, value: i8);
    pub fn with(self, dim: usize, value: i8) -> Self;   // builder-shape, clamps -8..7
}
```

`flow_qualia()` helper (`d_ign_b_lenses.rs:193-195`, provenance-noted as
re-derived from `cycle_driver.rs:1669`'s test fixture):
```rust
fn flow_qualia() -> QualiaI4_16D {
    QualiaI4_16D(0).with(3, 4).with(14, 3).with(9, 4).with(1, 2)
}
```

### `gate_decision_i4` (`lance-graph-contract/src/mul.rs:575`, inside `pub mod i4_eval`)

```rust
pub fn gate_decision_i4(qualia: &QualiaI4_16D, signed_mantissa: i8) -> GateDecision;
```
Reachable as `lance_graph_contract::mul::i4_eval::gate_decision_i4`.

### `mantissa_of` (`d_ign_b_lenses.rs:187-189`, identical shape in `probe_ignition.rs`)

```rust
fn mantissa_of(owner: &Tenant) -> i8 {
    owner.pending_count().min(7) as i8
}
```
`Tenant = MailboxSoA<ROWS_PER_OWNER>` (a type alias local to the test file).

### Fleet construction — bloom-plane seeding helpers (identical across both files)

```rust
const BLOOM_K: usize = 4;
fn fnv1a(bytes: &[u8], seed: u64) -> u64;
fn bloom_add(plane: &mut [u64], token: &str, salt: u64);
fn tokens(text: &str) -> impl Iterator<Item = String> + '_;
fn encode_plane(text: &str, salt: u64) -> Vec<u64>;   // Vec<u64> of WORDS_PER_FP words
```

`WORDS_PER_FP` is `cognitive_shader_driver::mailbox_soa::WORDS_PER_FP =
256` (256 u64 words = 16,384-bit identity plane;
`mailbox_soa.rs:36-39`).

### `build_owner` (`d_ign_b_lenses.rs:404-438`, verbatim signature)

```rust
fn build_owner(
    id: MailboxId,
    verses: &[String],
    content_salt: u64,
    armed: u8,
    qualia: QualiaI4_16D,
    firing_rows: usize,
) -> Tenant
```
Body: `MailboxSoA::new(id, TENANT_W_SLOT, TENANT_THRESHOLD)`, per-row
`WriteCell` write via `owner.write_row(row, cycle, &cell)` (asserting
`WriteOutcome::Accepted`), `owner.set_populated(verses.len())`,
`owner.tick()`, then `owner.energy[r] = FIRE_ENERGY` for `r in
0..firing_rows` (direct field write — `energy: [f32; N]` is `pub`, see
§ below).

### `MailboxSoA<const N: usize>` — full public method list (`cognitive-shader-driver/src/mailbox_soa.rs`, grepped)

```rust
impl<const N: usize> MailboxSoA<N> {
    pub fn new(mailbox_id: MailboxId, w_slot: u8, threshold: f32) -> Self;   // panics if w_slot >= 64
    pub fn apply_edges(&mut self, deliveries: &[(u16, CausalEdge64)]) -> usize;
    pub fn consume_firing(&mut self, row: usize) -> bool;
    pub fn tick(&mut self);
    pub fn write_row(&mut self, row: usize, cycle: u32, cell: &WriteCell<'_>) -> WriteOutcome;
    pub fn last_write_cycle_at(&self, row: usize) -> u32;
    pub fn stale_write_count(&self) -> u64;
    pub fn populated(&self) -> usize;
    pub fn set_populated(&mut self, n: usize);
    pub fn reset_row(&mut self, row: usize);
    pub fn energy_at(&self, row: usize) -> f32;
    pub fn plasticity_at(&self, row: usize) -> u8;
    pub fn cycle(&self) -> u32;
    pub fn w_slot(&self) -> u8;
    pub fn pending_count(&self) -> usize;
    pub fn edge(&self, row: usize) -> CausalEdge64;
    pub fn set_edge(&mut self, row: usize, e: CausalEdge64);
    pub fn qualia_at(&self, row: usize) -> QualiaI4_16D;
    pub fn set_qualia(&mut self, row: usize, q: QualiaI4_16D);
    pub fn meta_at(&self, row: usize) -> MetaWord;
    pub fn set_meta(&mut self, row: usize, m: MetaWord);
    pub fn entity_type_at(&self, row: usize) -> u16;
    pub fn set_entity_type(&mut self, row: usize, t: u16);
    pub fn temporal_at(&self, row: usize) -> u64;
    pub fn set_temporal(&mut self, row: usize, t: u64);
    pub fn expert_at(&self, row: usize) -> u16;
    pub fn set_expert(&mut self, row: usize, e: u16);
    pub fn sigma_at(&self, row: usize) -> u8;
    pub fn set_sigma(&mut self, row: usize, s: u8);
    pub fn content_row(&self, row: usize) -> &[u64];
    pub fn set_content(&mut self, row: usize, words: &[u64]);
    pub fn topic_row(&self, row: usize) -> &[u64];
    pub fn set_topic(&mut self, row: usize, words: &[u64]);
    pub fn angle_row(&self, row: usize) -> &[u64];
    pub fn set_angle(&mut self, row: usize, words: &[u64]);
    pub fn cast_on_behalf<P>(/* … */);
    pub fn set_style_lane(&mut self, row: usize, lane: StyleLane, atoms: [u8; 12]);
    pub fn set_style_atom(&mut self, row: usize, lane: StyleLane, family: u8, atom: u8);
    pub fn promote_family(&mut self, row: usize, family: u8) -> bool;
}
// Also implements MailboxSoaView + MailboxSoaOwner (contract traits;
// gives .mailbox_id(), .phase(), .current_cycle(), .n_rows(), .try_advance_phase(), etc.)
```

**Direct public field access used by both test files (not accessor
methods):** `owner.energy[r] = FIRE_ENERGY` (`d_ign_b_lenses.rs:436`,
`blw_fusion.rs:519`) — `pub energy: [f32; N]` is a genuinely public
struct field (`mailbox_soa.rs:66`), so this is legal direct indexing,
not a method call. Likewise `owner.mailbox_id` is `pub` (`mailbox_soa.rs:61`),
though both test files use the `.mailbox_id()` trait accessor instead.

`WriteCell<'a>` (`mailbox_soa.rs:262-283`, all fields `Option<...>`,
`#[derive(Debug, Clone, Default)]` so `..WriteCell::default()` works):
```rust
pub struct WriteCell<'a> {
    pub content: Option<&'a [u64]>,
    pub topic: Option<&'a [u64]>,
    pub angle: Option<&'a [u64]>,
    pub edge: Option<CausalEdge64>,
    pub qualia: Option<QualiaI4_16D>,
    pub meta: Option<MetaWord>,
    pub entity_type: Option<u16>,
    pub temporal: Option<u64>,
    pub expert: Option<u16>,
    pub sigma: Option<u8>,
}
```

`WriteOutcome` (`mailbox_soa.rs:241-254`): `enum { Accepted, Stale, Future }`.

### Scan / column-pass helpers (`d_ign_b_lenses.rs:514-571`, identical shape to `probe_ignition.rs`)

```rust
struct ScanResult {
    planning: Vec<MailboxId>,
    cognitive: Vec<MailboxId>,
    evaluation: Vec<MailboxId>,
    absorbed: Vec<MailboxId>,
    missing: usize,
}
fn scan_board(fleet: &Fleet, ids: impl IntoIterator<Item = MailboxId>) -> ScanResult;

struct ColumnPassOutcome { cast: usize }
fn column_pass(
    fleet: &Fleet,
    ids: &[MailboxId],
    writer: &mut BatchWriter<Vec<u8>>,
    think: impl FnMut(&Tenant) -> Option<(StrategyOutcome, Vec<u8>)>,
) -> ColumnPassOutcome;
```

### The cycle-driver seam — shipped signatures (`lance-graph-supervisor/src/cycle_driver.rs`)

```rust
pub struct CycleOutcome {
    pub sealed: SealedCycle,
    pub applied: AppliedCycle,
    pub held: Vec<HeldIntent>,
}
pub enum CycleError {
    Seal(Box<SealFailure>),
    Apply { partial: AppliedCycle, cause: PersistError },
}
pub async fn run_cycle<S, F>(
    sink: &S,
    fleet: &mut F,
    writer: &mut BatchWriter<Vec<u8>>,
    frame: CycleFrame,
    position_base: u64,
    watermarks: &mut HashMap<MailboxId, Option<u64>>,
    row_of: impl FnMut(MailboxId) -> u64,
) -> Result<CycleOutcome, CycleError>
where S: WalSink, F: MailboxFleet;

pub struct CognitiveWorkOutcome { pub cast: usize, pub held_owners: Vec<MailboxId> }

pub fn run_cognitive_work_gated_over<F>(
    fleet: &F,
    owners: &[MailboxId],
    writer: &mut BatchWriter<Vec<u8>>,
    read_gate: impl FnMut(&F::Owner) -> Option<(QualiaI4_16D, i8, f32, Vec<u8>)>,
) -> CognitiveWorkOutcome
where F: MailboxFleet;

#[must_use]
pub fn shade_owner<O: MailboxSoaOwner>(
    owner: &O,
    qualia: &QualiaI4_16D,
    mantissa: i8,
    reliability: f32,
) -> Option<StrategyOutcome>;
```

`d_ign_b_lenses.rs`'s exact `run_cognitive_work_gated_over` call
(`d_ign_b_lenses.rs:853-882`) embeds the lens capture (§B) INSIDE the
`read_gate` closure — this is "design §1's chosen seam" per the file's
own module doc.

### `emit_bootstrap_intent` (`lance-graph-planner/src/owner_adapter.rs:92-101`)

```rust
pub fn emit_bootstrap_intent<P>(
    outcome: &StrategyOutcome,
    owner: MailboxId,
    owner_cycle: u32,
    writer: &mut BatchWriter<P>,
    payload: P,
) -> Option<CastId>;
```

### `BatchWriter<P>` (`lance-graph-planner/src/batch_writer.rs`)

```rust
pub struct CastId(pub u64);
pub struct BatchWriter<P> { /* private fields */ }
impl<P> BatchWriter<P> {
    pub fn new() -> Self;
    pub fn cast(&mut self, on_behalf: MailboxId, moves: Vec<KanbanMove>, payload: P) -> CastId;
    pub fn casts(&self) -> Vec<CastId>;
    pub fn intent_moves(&self, cast: CastId) -> Option<&[KanbanMove]>;
    pub fn on_behalf_of(&self, cast: CastId) -> Option<MailboxId>;
    pub fn resolve_owner(&mut self, on_behalf: MailboxId,
        resolver: impl FnOnce(MailboxId) -> MailboxId) -> (MailboxId, bool);
    pub fn drain_pending_payloads(&mut self) -> impl Iterator<Item = (CastId, P)> + '_;
}
```

### `StyleStrategy` / `PlanContext` dispatch surface (`lance-graph-planner/src/strategy/style_strategy.rs`)

```rust
pub struct StyleStrategy;
impl StyleStrategy {
    pub fn reliability_for(style: ThinkingStyle, ctx: &PlanContext) -> f32;
}
impl PlanStrategy for StyleStrategy { /* .plan(PlanInput, &mut Arena) -> … */ }
```
`d_ign_b_lenses.rs`'s local helpers around this (`d_ign_b_lenses.rs:158-185`):
```rust
fn thinking_style_for(z: u8) -> ThinkingStyle;   // 1=>Analytical, 2=>Creative, _=>Reflective
fn style_vector_for(z: u8) -> Vec<f64>;          // 23-length one-hot vector
fn plan_context_for(z: u8) -> PlanContext;
```
`ThinkingStyle` (`lance-graph-contract/src/thinking.rs:23-25` onward) —
confirmed variants include `Logical = 0`, `Analytical = 1`, and (per the
existing d_ign_b_lenses.rs comment, not independently re-verified here
beyond a grep count) 36 total variants across 6 clusters
(τ 0x40-0x4F etc. per doc comments) — `Creative` and `Reflective` were
NOT individually grepped for their discriminant values in this pass (see
Not Verified).

### `DatasetVersion` (`lance-graph-contract/src/scheduler.rs:36`)

```rust
pub struct DatasetVersion(pub u64);
```

---

## NOT VERIFIED (explicit — do not guess from this list)

1. **Field-level layout of `RankedHit` / `Band` / `ShiftAlert` /
   `PreciseMode`** in `ndarray::hpc::cascade` — only struct/impl/fn
   HEADERS were grepped (§D); bodies and exact field names were not read.
   `Band`'s variant list beyond `Foveal`/`Reject` (seen in a cascade.rs
   test at line 472-473) was not enumerated.
2. **The full `Statistics<A>` trait member list** past `cosine_similarity`
   (`statistics.rs` line ~60 onward) — the grepped window
   (`statistics.rs:22-60`) may not be the complete trait; only the
   members through `cosine_similarity` were read.
3. **`ThinkingStyle`'s complete 36-variant list and discriminant values**
   for `Creative`/`Reflective` specifically — only `Logical`/`Analytical`
   discriminants (0/1) were directly read; the file has ~10 lines
   matching the enum-variant grep pattern used, which undercounts a
   36-variant enum (multi-variant or comment lines likely interfere with
   the pattern) — this count is NOT reliable and was not corrected by a
   full read of `thinking.rs`.
4. **`QueryReference::at`, `deinterlace`, `DeinterlaceRow`, `NoDeps`,
   `LanceVersion`** (`lance_graph_planner::temporal`) — cited by
   `blw_fusion.rs`'s import and call sites (quoted verbatim where seen)
   but `temporal.rs` itself was NOT opened in this pass; only what
   `blw_fusion.rs`'s own call sites and doc comments state about it is
   reported above (§E's `VerdictRow::lance_version()` mapping).
5. **`MailboxSoaOwner` / `MailboxSoaView` trait method lists** — only the
   methods actually called by the two test files (`phase()`,
   `mailbox_id()`, `current_cycle()`, `n_rows()`, `try_advance_phase()`)
   were confirmed by call-site read; the full trait definitions in
   `lance-graph-contract/src/soa_view.rs` were not opened.
6. **`GateDecision` enum's complete variant list** — only the
   `Block`/`Hold`/`Flow` arms visible in `gate_decision_i4`'s match
   (`mul.rs:579-594`, partially read) were seen; the full enum
   definition was not located/read.
7. **`Interner` full API** beyond `new()`, `id()`, `name()` — these three
   were read in full from `stance.rs:50-87`; no further methods exist in
   that file (this one IS complete, listed for clarity, not a gap).
8. **`probe_ignition.rs` in full** — this file was NOT read end-to-end;
   only `d_ign_b_lenses.rs`'s own citations of it (line-numbered
   provenance comments, e.g. "provenance: `probe_ignition.rs:604-638`")
   were relied on for cross-file claims. Any helper unique to
   `probe_ignition.rs` and NOT copied into `d_ign_b_lenses.rs` is not
   inventoried here.
9. **Whether `jc::stats` exports anything else useful to D-BLW-5**
   beyond `BinaryAssociation`/`binary_association` (e.g. `cohen_kappa`,
   `phi`, `omega_total` — all read in §C's source pass and quoted in
   their doc comments, but their exact call sites in THIS workspace
   beyond `blw_fusion.rs`'s single `binary_association` use were not
   searched for).

---

## Summary of load-bearing findings for the orchestrator

- **A/B are fully live and reusable as-is**: `BeliefArena::observe` /
  `admit_derived` accept hand-built `CStmt`/`TruthValue`/`Stamp` with no
  text-parsing dependency; `stance_panel` is one call returning all four
  stances; `run_lens`'s exact shape in `d_ign_b_lenses.rs` is the
  established pattern.
- **C and D share the SAME structural gap**: `jc` (fusion oracle) and
  `ndarray` (percentile/cascade) are BOTH reachable only from
  `lance-graph-planner` (lib + examples), and BOTH absent from
  `lance-graph-supervisor`'s manifest (direct or dev). Neither gap can
  be closed by a Sonnet edit-only lane without a `Cargo.toml` change —
  this is a STOP+report item per guardrails §5.2, flagged here rather
  than silently worked around.
- **E**: version stamping is per-CYCLE (`DatasetVersion` from
  `persist_cycle`/`seal_cycle`/`run_cycle`), never per-row; a row-level
  record keys on `(version, owner, row)` or `(version, subject_string)`,
  read back as `LandedSlot { version, slot: SweepSlot }`.
- **F**: the fleet/scan/cast/cycle scaffolding in `d_ign_b_lenses.rs` is
  a direct, unmodified copy of `probe_ignition.rs`'s pattern (each site
  provenance-commented); a D-BLW-5 build can copy the same helpers
  verbatim from either file.
