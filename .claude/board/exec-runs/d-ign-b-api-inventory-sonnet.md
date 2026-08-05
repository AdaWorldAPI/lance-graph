# D-IGN-B — API inventory (Sonnet, edit-only, no cargo run)

Scope: four-stance machinery (`stance.rs`), the consumer precedent
(`probe_eyes_opened.rs`), the Fusion arm (`blw_fusion.rs`), the seam
(`cycle_driver.rs`), `MetaWord`, and the GREEN probe's reusable
helper signatures (`probe_ignition.rs`). Everything below is copied
verbatim from source; nothing was compiled or run by this lane.

---

## A. Four-stance machinery — `crates/lance-graph-planner/src/nars/stance.rs`
(read in full) + `crates/lance-graph-planner/src/nars/mod.rs`

### Module visibility

`crates/lance-graph-planner/src/nars/mod.rs:20` declares `pub mod stance;`
and does **NOT** re-export any `stance::*` symbol via the crate's `pub use`
list (lines 25-41 re-export `basin_resonance`, `belief`, `dissolution`,
`elevation`, `epiphany`, `facet_fold`, `inference`, `insight`, `insights`,
`reach_out`, `regulate`, `tactic_select`, `tactics`, `truth` — `stance` is
absent from that list). Consequence: a consumer must import via the full
path `lance_graph_planner::nars::stance::{...}` (exactly what
`probe_eyes_opened.rs:87-89` and `blw_fusion.rs` do NOT do — `blw_fusion.rs`
does not touch `stance` at all; only `probe_eyes_opened.rs` imports it).

**Nothing in `stance.rs` is `#[cfg(test)]`.** The whole file (lines 1-536) is
plain `pub`/private items with no `#[cfg(test)]` gate anywhere — every `pub`
item below is importable from another crate via the full path.

### Every `pub` type, fn, and method — exact signatures (file:line)

```rust
// line 50-53
#[derive(Default)]
pub struct Interner {
    map: HashMap<String, u16>,   // private field
    names: Vec<String>,          // private field
}

// line 56 — impl Interner
pub fn new() -> Self

// line 63 — impl Interner
pub fn id(&mut self, w: &str) -> u16
// (panics via `assert!` past u16::MAX distinct strings — line 73-77)

// line 84 — impl Interner
pub fn name(&self, id: u16) -> &str

// line 90-99
#[derive(Debug, Clone)]
pub struct Provenance {
    pub verse: String,
    pub stmt: CStmt,      // from super::belief
    pub negated: bool,
}

// line 104-127
#[derive(Debug, Clone)]
pub struct RungLift {
    pub verse: String,
    pub knower: u16,
    pub verb: u16,
    pub object: u16,
    pub modal: f32,
    pub cell: u8,
    pub staunen_at: f32,
    pub quale: f32,
    pub self_referential: bool,
}

// line 132-145
#[derive(Default)]
pub struct ReadOut {
    pub provenance: Vec<Provenance>,
    pub lifts: Vec<RungLift>,
    pub impls: Vec<(String, u16, u16)>,
    pub pass2_admitted: usize,
    pub pass2_revised: usize,
}

// line 161-167 — the FREE FUNCTION (not a method on a carrier)
pub fn stream(
    verses: &[(String, String)],
    arena: &mut BeliefArena,
    intern: &mut Interner,
    out: &mut ReadOut,
    pass2: bool,
)
// Return type: () (mutates `arena`, `intern`, `out` in place via &mut).

// line 418
pub fn contradiction_ranking(arena: &BeliefArena) -> Vec<(CStmt, f32)>

// line 430-437
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlipKind {
    Transvaluation,
    Devaluation,
}

// line 468-478 — the B6 aspect panel
#[allow(clippy::type_complexity)]
pub fn stance_panel(
    arena: &BeliefArena,
    intern: &Interner,
    out: &ReadOut,
) -> (
    Vec<(CStmt, f32)>,       // Hegel: Aufhebung ranking
    Vec<(CStmt, FlipKind)>,  // Nietzsche: genealogy partition
    Vec<(String, f32, f32)>, // Kant: (lift label, graded quale, ablated quale)
    Vec<(u16, usize)>,       // Wittgenstein: (concept, distinct games)
)
```

### Module-private (not `pub`, listed for completeness — cannot be
imported from another crate)

- `const STOP: &[&str]` (line 33) — private.
- `const AUX: &[&str]` (line 43) — private.
- `Interner::map` / `Interner::names` fields — private (only via the pub
  methods above).

### What a stance run CONSUMES and RETURNS

- **`stream(...)`** consumes: `verses: &[(String, String)]` (borrowed,
  shared ref), `arena: &mut BeliefArena` (borrowed, exclusive — the arena is
  MUTATED, not read-only: `stream` calls `arena.observe(...)` and
  `arena.admit_derived(...)` internally, lines 281, 338), `intern: &mut
  Interner` (borrowed, exclusive — new strings get interned during the
  pass), `out: &mut ReadOut` (borrowed, exclusive — accumulates), `pass2:
  bool` (owned `Copy`). Returns `()`.
- **`stance_panel(...)`** consumes: `arena: &BeliefArena` (borrowed,
  SHARED-only — signature has no `&mut`, so mutation is impossible by
  signature per the doc comment lines 447-448), `intern: &Interner`
  (shared), `out: &ReadOut` (shared). Returns the 4-tuple above (all owned
  `Vec`s — the readout is fully owned, no lifetime tied to `arena`).
- **Readout type fields and their derive status** (this is the
  bit-identical-comparison-relevant fact):
  - `CStmt` (from `belief.rs:77`, re-exported at `nars/mod.rs:26`):
    `#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]` — **Eq + Hash
    present**, fields `s: u16`, `cop: Copula`, `p: u16`.
  - `Copula` (`belief.rs:54`): `#[derive(Debug, Clone, Copy, PartialEq, Eq,
    Hash)]` — **Eq + Hash present**.
  - `FlipKind` (`stance.rs:430`): `#[derive(Debug, Clone, Copy, PartialEq,
    Eq)]` — **Eq present, Hash absent**.
  - The `f32` fields inside the panel tuples (`Vec<(CStmt, f32)>`,
    `Vec<(String, f32, f32)>`) mean **the panel's own tuple types do NOT
    derive `Eq`/`Hash`** (f32 has no `Eq`); only `PartialEq`/`PartialOrd`
    apply to those tuples as a whole. A bit-identical comparison gate on
    the panel output must compare `f32` fields by bit pattern
    (`.to_bits()`) or exact `==`, not derive-based `Eq`/`Hash` on the whole
    tuple.
  - `TruthValue` (`truth.rs:9-10`): `#[derive(Debug, Clone, Copy,
    PartialEq)]` — **no `Eq`** (holds `f32` fields per `truth.rs:1-20`
    read; confirmed no `Eq`/`Hash` derive on the struct line).

### Stance selection — exact discriminants

There is **no enum of "the four stances."** The four stances (Hegel /
Nietzsche / Kant / Wittgenstein) are **not** named by any enum discriminant
or const — they are four **hard-coded computation blocks inside the single
function `stance_panel`** (lines 479-534), each producing one element of
the returned 4-tuple in FIXED POSITIONAL ORDER:
- index 0 = Hegel (`hegel` local, line 480, delegates to
  `contradiction_ranking`),
- index 1 = Nietzsche (`nietzsche` local, lines 483-496, inline loop),
- index 2 = Kant (`kant` local, lines 499-510, inline `.map()` with the
  hard-coded `const UNIFORM_MODAL: f32 = 0.5` ablation),
- index 3 = Wittgenstein (`wittgenstein` local, lines 513-532, inline
  `HashMap`-based game-counting).

There is no per-stance function a caller can invoke individually — calling
`stance_panel` always computes all four. Selecting "just one stance" is not
a thing the API exposes.

---

## B. The consumer precedent — `crates/lance-graph-planner/examples/probe_eyes_opened.rs`

Exact call sites (verbatim):

```rust
// import (lines 87-89)
use lance_graph_planner::nars::stance::{
    contradiction_ranking, stance_panel, stream, FlipKind, Interner, Provenance, ReadOut, RungLift,
};

// construction, in order (report(), lines 219-226)
fn report(label: &str, verses: &[(String, String)]) -> (BeliefArena, Interner, ReadOut) {
    let mut arena = BeliefArena::new();
    let mut intern = Interner::new();
    let mut out = ReadOut::default();

    // Pass 1 — the reading.
    stream(verses, &mut arena, &mut intern, &mut out, false);
    arena.close_transitive(64);
    ...
```

Construction order, exactly: `BeliefArena::new()` → `Interner::new()` →
`ReadOut::default()` → `stream(verses, &mut arena, &mut intern, &mut out,
false)` (pass 1, `pass2=false`) → `arena.close_transitive(64)` (not part of
`stance.rs`; a `BeliefArena` method).

Elsewhere in `main()` a second `stream` call re-presents the SAME verses
with `pass2=true` (line 301-302 area):
```rust
let mut pass2 = ReadOut::default();
stream(verses, &mut arena, &mut intern, &mut pass2, true);
```
(fresh `ReadOut`, SAME `arena`/`intern` reused — this is how the
hermeneutic-circle termination check is driven, per `stream`'s own doc
comment lines 159-160.)

Stance-panel call sites (two, both after the corpus has been streamed +
closed):
```rust
// line 177 — inside print_stance_panel(arena, intern, out)
let (hegel, nietzsche, kant, wittgenstein) = stance_panel(arena, intern, out);
```
```rust
// line 569 — inside main()
let (hegel, nietzsche, kant, wittgenstein) = stance_panel(&scene_arena, &intern, &out);
```
Both call `stance_panel` on a `&BeliefArena` immediately after (not
interleaved with) the `stream`/`close_transitive` calls that built it —
`print_stance_panel` (lines 175-217) additionally snapshots
`arena.entries().len()` BEFORE calling `stance_panel` and re-asserts it
AFTER (lines 176, 212-216) to runtime-prove the "arena unchanged" claim in
the doc comment.

`contradiction_ranking` is also called standalone, independent of the
panel, e.g. line 254 `let ranking = contradiction_ranking(&arena);` and
line 396, 421 — always on a `&BeliefArena` reference.

---

## C. The Fusion arm — `crates/lance-graph-planner/examples/blw_fusion.rs`

### Imports (lines 103-105)

```rust
use lance_graph_planner::temporal::{
    deinterlace, DeinterlaceRow, LanceVersion, NoDeps, QueryReference,
};
```

### `deinterlace` call sites (verbatim, all from the `#[tokio::test]` body
around lines 981-1400 — variable names as in source)

```rust
// G1 — line 981-986
let qref_strict_pin = QueryReference::at(v_pin, RUNG_STRICT);
let qref_aware_pin = QueryReference::at(v_pin, RUNG_AWARE);
let strict_v4 = deinterlace(&all_rows, &qref_strict_pin, &NoDeps);
let aware_v4 = deinterlace(&all_rows, &qref_aware_pin, &NoDeps);
```
```rust
// line 1001-1004
let qref_strict_v8 = QueryReference::at(v8, RUNG_STRICT);
let qref_aware_v8 = QueryReference::at(v8, RUNG_AWARE);
let strict_v8_rows = deinterlace(&all_rows, &qref_strict_v8, &NoDeps);
let aware_v8_rows = deinterlace(&all_rows, &qref_aware_v8, &NoDeps);
```
```rust
// line 1025-1026 (G1c retro-only check)
let qref_retro_pin = QueryReference::at(v_pin, RUNG_RETRO);
let retro_v4 = deinterlace(&all_rows, &qref_retro_pin, &NoDeps);
```
```rust
// line 1164
let aware_v4_desc = deinterlace(&rows_desc, &qref_aware_pin, &NoDeps);
```
```rust
// lines 1379-1382 (loop over some `vk` set)
let qref_s = QueryReference::at(vk, RUNG_STRICT);
let qref_a = QueryReference::at(vk, RUNG_AWARE);
let s_rows = deinterlace(&all_rows, &qref_s, &NoDeps);
let a_rows = deinterlace(&all_rows, &qref_a, &NoDeps);
```

`QueryReference::at(...)` constructions used: `(v_pin, RUNG_STRICT)`,
`(v_pin, RUNG_AWARE)`, `(v8, RUNG_STRICT)`, `(v8, RUNG_AWARE)`, `(v_pin,
RUNG_RETRO)`, `(vk, RUNG_STRICT)`, `(vk, RUNG_AWARE)` — always the two-arg
form `at(ref_version: LanceVersion, rung: u8)` (see `temporal.rs:167`
below); `RUNG_STRICT = 0`, `RUNG_AWARE = 5`, `RUNG_RETRO = 9` (consts at
`blw_fusion.rs:134,136,138`).

**In every call site `deinterlace` is invoked against `&all_rows` (or
`&rows_desc`) — the WHOLE sealed corpus row set, not a single owner's
rows.** `all_rows` is built once (outside these excerpts, from the whole
tenant's emitted `VerdictRow`s across all cycles/projections) and reused
across every `deinterlace` call in the file; there is no call site anywhere
in `blw_fusion.rs` that passes a per-owner-filtered slice into
`deinterlace`. Consequence per the design comment at
`temporal.rs:332-345`: `deinterlace` needs the WHOLE sealed row set
(offline projection over everything emitted so far), not one owner's rows
— the fusion arm's own usage matches that shape exactly.

### `DeinterlaceRow for VerdictRow` — required methods, exact signatures
(lines 219-247)

```rust
#[derive(Clone, Debug)]
struct VerdictRow {
    subject: String,   // e.g. "kjv:00417"
    horizon: u64,      // the SEALED VERSION this verdict was computed from
    projection: Proj,  // A | B | Z
    verdict: bool,
}

impl DeinterlaceRow for VerdictRow {
    fn subject(&self) -> &str {
        &self.subject
    }
    fn lance_version(&self) -> LanceVersion {
        self.horizon
    }
    /// CONSTANT `0` — a class-registration clock, NOT a per-row warrant time.
    fn knowable_from(&self) -> LanceVersion {
        0
    }
    // hlc_tick() DEFAULTED (not overridden) — trait default returns `None`.
}
```

The trait itself (`temporal.rs:318-330`, quoted for cross-check — the
example's impl provides exactly the 3 required methods; `hlc_tick` is
optional/defaulted):
```rust
pub trait DeinterlaceRow {
    fn subject(&self) -> &str;
    fn lance_version(&self) -> LanceVersion;
    fn knowable_from(&self) -> LanceVersion;
    fn hlc_tick(&self) -> Option<u64> { None }   // default
}
```

`VerdictRow` also derives `Clone` (line 218: `#[derive(Clone, Debug)]`) —
required because `deinterlace<R, D>` bounds `R: DeinterlaceRow + Clone`
(`temporal.rs:348`).

### The fold/rank criterion

```rust
// line 346 (exact signature)
fn rank_verdicts(owner: &Tenant, pool_size: usize, seed: &[u64]) -> Vec<bool>
```
Consumes: `owner: &Tenant` (`Tenant = MailboxSoA<N_CAP>`, borrowed
shared — reads `identity_plane_at` via `score_row`), `pool_size: usize`
(owned), `seed: &[u64]` (borrowed bloom bits, owned-by-caller). Returns
`Vec<bool>` in ROW-INDEX order (not sorted-score order — doc comment lines
343-345).

```rust
// line 583 (exact signature)
fn fold_last_by_subject(rows: &[VerdictRow], proj: Proj) -> Vec<(String, bool)>
```
Consumes: `rows: &[VerdictRow]` — a slice of the (already-deinterlaced)
`VerdictRow`s, borrowed. Internally filters to ONE projection first, then
folds "last row wins" per subject (relies on `deinterlace`'s own ascending
sort — comment lines 577-582, 598-600). This function operates on whatever
slice it's given; it does not itself require the WHOLE sealed set — it is
`deinterlace`'s own output/input contract (Section C's opening finding)
that requires the whole set. `rank_verdicts`, by contrast, is a
PRE-deinterlace scoring function over ONE tenant's populated rows
(`owner: &Tenant`, not `&[VerdictRow]`) — it needs one owner's SoA state,
not a row-array.

---

## D. The seam — `crates/lance-graph-supervisor/src/cycle_driver.rs`

### `run_cognitive_work_gated_over` — EXACT signature (lines 662-676)

```rust
pub fn run_cognitive_work_gated_over<F>(
    fleet: &F,
    owners: &[MailboxId],
    writer: &mut BatchWriter<Vec<u8>>,
    mut read_gate: impl FnMut(&F::Owner) -> Option<(QualiaI4_16D, i8, f32, Vec<u8>)>,
) -> CognitiveWorkOutcome
where
    F: MailboxFleet,
```

The closure type it takes, written out character for character:
```
impl FnMut(&F::Owner) -> Option<(QualiaI4_16D, i8, f32, Vec<u8>)>
```
i.e. a `FnMut` taking `&F::Owner` (a shared borrow of the fleet's owner
type — never `&mut`), returning `Option<(QualiaI4_16D, i8, f32, Vec<u8>)>`
— a 4-tuple of `(qualia, signed_mantissa, reliability, payload)`, `None`
meaning "declined / no gate result this pass" (doc comment lines 640-643).

Internally (lines 671-676) it forwards to `run_cognitive_work_over` by
wrapping the caller's `read_gate` in a second closure that additionally
calls `shade_owner(owner, &qualia, mantissa, reliability)`:
```rust
run_cognitive_work_over(fleet, owners, writer, |owner| {
    let (qualia, mantissa, reliability, payload) = read_gate(owner)?;
    let outcome = shade_owner(owner, &qualia, mantissa, reliability)?;
    Some((outcome, payload))
})
```

### `shade_owner` — EXACT signature (lines 614-620)

```rust
pub fn shade_owner<O: MailboxSoaOwner>(
    owner: &O,
    qualia: &QualiaI4_16D,
    mantissa: i8,
    reliability: f32,
) -> Option<StrategyOutcome>
```
Body (lines 621-634): reads `owner.phase()`, computes
`gate_decision_i4(qualia, mantissa)`, calls
`phase.advance_on_gate(&gate)?` (returns `None` on Hold / no legal
successor), and on success returns a **bootstrap sentinel**
`StrategyOutcome` whose `intended_move` is
`KanbanMove { mailbox: 0, from: phase, to, witness_chain_position: 0, exec:
ExecTarget::Native }` — `mailbox: 0` is a fixed sentinel value that
`owner_adapter::emit_bootstrap_intent` later rebinds to the real owner
(never the live `MailboxId`).

### `MailboxFleet` trait — EXACT (lines 179-186)

```rust
pub trait MailboxFleet {
    type Owner: MailboxSoaOwner;
    fn owner(&self, id: MailboxId) -> Option<&Self::Owner>;
    fn owner_mut(&mut self, id: MailboxId) -> Option<&mut Self::Owner>;
}
```
Blanket impl for `HashMap<MailboxId, O>` where `O: MailboxSoaOwner` (lines
190-198) — `owner`/`owner_mut` are plain `HashMap::get`/`get_mut`.

### What a caller CAN and CANNOT pass through the `read_gate` closure

- The closure receives ONLY `&F::Owner` — a **shared, read-only borrow**
  of the owner. It cannot mutate the owner (`&F::Owner`, never `&mut`).
  Whatever it reads (qualia, mantissa, payload bytes) must come from
  methods callable on `&Owner` — i.e. whatever `MailboxSoaOwner` /
  `MailboxSoaView` / the concrete `MailboxSoA<N>` expose as read accessors
  (`phase()`, `meta_at(row)`, `qualia_at(row)`, `pending_count()`, etc., per
  Section F's usage in `probe_ignition.rs`).
- **This DOES decide whether a lens ordinal can reach a thought body**: any
  value the closure hands onward (into `shade_owner`'s `qualia`/`mantissa`
  args, or into the `Vec<u8>` payload) must be DERIVABLE from a read of the
  owner alone — there is no channel in this signature for external
  context, no `&mut` state, and no async/await inside the closure (it is a
  plain synchronous `FnMut`, not `async fn`). A lens ordinal reaches the
  thought body only if it is first materialized as owner-readable state
  (e.g. written into `MetaColumn`/`meta_at` before the pass runs) — the
  closure itself is a pure projection function, not a side-channel.
- The closure is `FnMut`, so it MAY carry captured mutable state across
  calls within one `run_cognitive_work_gated_over` invocation (e.g. an
  external counter), but that captured state is local to the closure's
  environment, not derived from or written back to the fleet.

---

## E. MetaWord — `crates/lance-graph-contract/src/cognitive_shader.rs`

Exact signatures (lines 42-76):

```rust
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(transparent)]
pub struct MetaWord(pub u32);

impl MetaWord {
    #[inline]
    pub const fn new(thinking: u8, awareness: u8, nars_f: u8, nars_c: u8, free_e: u8) -> Self {
        let w = (thinking as u32 & 0x3F)
            | (((awareness as u32) & 0x0F) << 6)
            | ((nars_f as u32) << 10)
            | ((nars_c as u32) << 18)
            | (((free_e as u32) & 0x3F) << 26);
        Self(w)
    }
    #[inline]
    pub fn thinking(&self) -> u8 {
        (self.0 & 0x3F) as u8
    }
    #[inline]
    pub fn awareness(&self) -> u8 {
        ((self.0 >> 6) & 0x0F) as u8
    }
    #[inline]
    pub fn nars_f(&self) -> u8 {
        ((self.0 >> 10) & 0xFF) as u8
    }
    #[inline]
    pub fn nars_c(&self) -> u8 {
        ((self.0 >> 18) & 0xFF) as u8
    }
    #[inline]
    pub fn free_e(&self) -> u8 {
        ((self.0 >> 26) & 0x3F) as u8
    }
}
```

Packing layout (doc comment line 38, verified against the bit-shift
arithmetic above): `thinking(6 bits, mask 0x3F) + awareness(4 bits, mask
0x0F) + nars_f(8 bits) + nars_c(8 bits) + free_e(6 bits, mask 0x3F)` = 6 +
4 + 8 + 8 + 6 = 32 bits, one `u32` per row.

**The `thinking` field is 6 bits wide (mask `0x3F` = 0..63).** `thinking()`
masks with `0x3F` on read; `new()` masks the input `thinking` arg with
`0x3F` on write — an input `>= 64` is silently truncated to its low 6 bits,
not rejected.

No setter (`set_thinking`) exists on `MetaWord` itself — `MetaWord` is
constructed fresh via `new()` and OVERWRITES the whole packed word; there
is no `with_thinking`/mutator method in this file. (`owner.set_meta(row,
MetaWord::new(...))` — seen in Section F — is a method on the OWNER, not on
`MetaWord`.)

### How `probe_ignition.rs` (the GREEN probe) writes and reads `MetaWord` —
verbatim call sites

**Write, inside `build_owner` (test-file line ~443):**
```rust
let meta = MetaWord::new(armed, 0, 0, 0, 0);
// ... later, per row:
let cell = WriteCell {
    content: Some(content.as_slice()),
    qualia: Some(qualia),
    meta: Some(meta),
    entity_type: Some((row % 251) as u16),
    temporal: Some(row as u64),
    ..WriteCell::default()
};
let outcome = owner.write_row(row, cycle, &cell);
```
Only the `thinking` field is ever non-zero in the probe (`armed: u8` — 0/1/2/3
per cohort); `awareness`/`nars_f`/`nars_c`/`free_e` are always literal `0`.

**A second, standalone write site (line ~1128):**
```rust
owner_mut.set_meta(0, MetaWord::new(1, 0, 0, 0, 0));
```
(`owner_mut` — arms row 0's thinking bit to `1`/Analytical directly, outside
`build_owner`'s per-row loop — used in a later, separate fixture in the
same test file.)

**Read, inside `plan_or_evaluate_think` (line ~605) and the
`run_cognitive_work_gated_over` closure (line ~809):**
```rust
let armed = owner.meta_at(0).thinking();
```
Both read sites call `owner.meta_at(0)` (row 0 only — every owner in this
probe carries its arming bit at row 0) then `.thinking()` on the returned
`MetaWord`. `armed == 0` is checked explicitly as the UNARMED sentinel
(`plan_or_evaluate_think` line ~606: `if armed == 0 { return None; }`).

---

## F. Reusable helper signatures from `probe_ignition.rs` (quoted, not
narrated — bodies omitted per the brief)

```rust
// corpus loader
fn load_verses(path: &str, limit: usize) -> Option<Vec<String>>
fn synthetic_corpus(n: usize) -> Vec<String>
fn load_or_synthesize_corpus() -> (Vec<String>, &'static str)

// fleet construction
type Tenant = MailboxSoA<ROWS_PER_OWNER>;   // ROWS_PER_OWNER = 64
type Fleet = HashMap<MailboxId, Tenant>;
fn owner_verses(all: &[String], owner_idx: MailboxId) -> &[String]
fn build_owner(
    id: MailboxId,
    verses: &[String],
    armed: u8,
    qualia: QualiaI4_16D,
    firing_rows: usize,
) -> Tenant
fn build_fleet(corpus: &[String]) -> Fleet

// MemWal (in-process WalSink; NOT durability — struct + fields, no method
// bodies quoted here, all are private to the test module)
struct MemWal {
    sealed: Mutex<Vec<SealedCycle>>,
    next_version: AtomicU64,
    wal_writes: AtomicU64,
    reads: AtomicU64,   // MUST stay 0 across the main loop (P4b reads no dataset)
}
impl WalSink for MemWal { /* ... */ }
fn MemWal::new() -> Self
fn MemWal::wal_writes(&self) -> u64
fn MemWal::reads(&self) -> u64
fn MemWal::head(&self) -> DatasetVersion

// the scan function
#[derive(Default)]
struct ScanResult {
    planning: Vec<MailboxId>,
    cognitive: Vec<MailboxId>,
    evaluation: Vec<MailboxId>,
    absorbed: Vec<MailboxId>,
    missing: usize,
}
fn scan_board(fleet: &Fleet, ids: impl IntoIterator<Item = MailboxId>) -> ScanResult

// mantissa derivation (fed into shade_owner via the gated closure)
fn mantissa_of(owner: &Tenant) -> i8   // owner.pending_count().min(7) as i8

// qualia fixtures
fn flow_qualia() -> QualiaI4_16D
fn block_qualia() -> QualiaI4_16D

// the probe-local Planning/Evaluation pass (NOT the shipped seam —
// shipped `run_cognitive_work_gated_over` only drives CognitiveWork)
struct ColumnPassOutcome {
    cast: usize,
    held: Vec<MailboxId>,
    missing: usize,
}
fn column_pass(
    fleet: &Fleet,
    ids: &[MailboxId],
    writer: &mut BatchWriter<Vec<u8>>,
    mut think: impl FnMut(&Tenant) -> Option<(StrategyOutcome, Vec<u8>)>,
) -> ColumnPassOutcome
fn plan_or_evaluate_think(owner: &Tenant) -> Option<(StrategyOutcome, Vec<u8>)>
```

### The cycle loop's call sequence (per cycle `c`, quoted structurally —
exact ordering, from the `for c in 1..=CYCLES` body)

1. (if `c == WAKE_CYCLE`) re-energize one row per REST-cohort owner —
   `owner.energy[row] = FIRE_ENERGY` (direct field write, no method).
2. `scan_board(&fleet, SCOPE_LO..SCOPE_HI)` → `scan: ScanResult`.
3. Bookkeeping: `sink.wal_writes()` snapshot, `phase_cycle_snapshot(&fleet)`
   snapshot ("before cast").
4. `column_pass(&fleet, &scan.planning, &mut writer, plan_or_evaluate_think)`
   → `planning_outcome`.
5. `run_cognitive_work_gated_over(&fleet, &scan.cognitive, &mut writer,
   |owner| { ... })` → `cognitive_outcome` (the closure reads
   `owner.meta_at(0).thinking()`, `owner.qualia_at(0)`, `mantissa_of(owner)`,
   `StyleStrategy::reliability_for(style, &ctx)`, and returns
   `Some((qualia, mantissa, reliability, row_span_payload(owner)))`).
6. `column_pass(&fleet, &scan.evaluation, &mut writer,
   plan_or_evaluate_think)` → `evaluation_outcome`.
7. `phase_cycle_snapshot(&fleet)` snapshot ("after cast") — asserted equal
   to step 3's snapshot (G3a: staging casts must not mutate any owner's
   phase/cycle).
8. `total_casts = planning_outcome.cast + cognitive_outcome.cast +
   evaluation_outcome.cast`.
9. If `total_casts == 0`: REST branch — no `run_cycle` call, no seal,
   `wal_writes` unchanged, `continue` to the next cycle (with an
   end-of-c5-vs-end-of-c6 fingerprint comparison hook on the last two
   cycles).
10. Else: `run_cycle(&sink, &mut fleet, &mut writer,
    CycleFrame::new(CycleId(u64::from(c)), base_version), position_base,
    &mut watermarks, u64::from).await` → `Result<CycleOutcome, CycleError>`
    (panics on any `Err` — the probe treats a seal/apply failure in the
    main run as impossible given `MemWal` never injects one there).
11. `position_base = position_base.max(outcome.sealed.next_position_base)`.
12. Post-apply snapshot + assertions comparing `changed` owners to
    `outcome.sealed.transitions`' owners, and `sink.reads() == 0`.

`run_cycle`'s own internal sequence (from `cycle_driver.rs:446-471`, for
cross-reference — this is INSIDE step 10 above, not called separately by
the probe): `collect_casts(writer, frame.cycle, position_base, row_of)` →
`seal_cycle(sink, frame, collected.slots).await` →
`apply_sealed_transitions(fleet, &sealed, watermarks)` → assembles
`CycleOutcome { sealed, applied, held: collected.held }`.

---

## NOT VERIFIED (explicit — nothing below was confirmed; do not treat as fact)

- `MailboxSoaOwner`, `MailboxSoaView`, `Owner::phase()`,
  `Owner::mailbox_id()`, `Owner::current_cycle()`, `Owner::meta_at()`,
  `Owner::qualia_at()`, `Owner::pending_count()`, `Owner::set_meta()`,
  `Owner::write_row()`, `Owner::set_populated()`, `Owner::tick()` — none of
  these trait/method definitions were opened in this pass; only their CALL
  SITES were read (in `probe_ignition.rs` / `blw_fusion.rs` /
  `cycle_driver.rs`). Their exact signatures (esp. return types, whether
  `meta_at`/`qualia_at` take `&self` or borrow-return) are asserted here
  only as inferable from call-site usage (`owner.meta_at(0).thinking()`
  implies `meta_at(&self, usize) -> MetaWord` or `-> &MetaWord`, but the
  precise return type — owned vs `&MetaWord` — was NOT confirmed by
  opening the trait/impl).
- `BeliefArena::close_transitive` — called by `probe_eyes_opened.rs` but
  its signature was not opened in this pass (only its call site).
- `owner_adapter::emit_bootstrap_intent` — signature not opened; only its
  call sites (`cycle_driver.rs` internals, `probe_ignition.rs`'s
  `column_pass`) were read.
- `gate_decision_i4`, `trust_texture_i4`, `KanbanColumn::advance_on_gate`,
  `StyleStrategy::plan` / `StyleStrategy::reliability_for` — signatures not
  opened; only call sites read.
- `WalSink` trait's full method set — only `scan_sealed` was named in a
  grep hit; the trait definition itself was not opened.
- Nothing in this file was compiled, type-checked, or run. All signatures
  above are transcribed from source text as read; any transcription error
  is possible and would only be caught by the orchestrator's central
  `cargo` gate.
