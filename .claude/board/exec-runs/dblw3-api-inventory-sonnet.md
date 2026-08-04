# D-BLW-3 API Inventory (Sonnet grindwork lane)

Mechanical inventory only. No cargo run. No `.rs` edits. Every claim below is
`file:line`-anchored to a full read of the named file (temporal.rs and
blw_tenant.rs read in full via one `Read` call each; stats.rs read lines
1–1161, which covers every item this task asked for — `omega_total`'s tail
past line 1161 was NOT read since nothing in scope needed it).

All paths below are relative to `/home/user/lance-graph` unless stated.

---

## A. `crates/lance-graph-planner/src/temporal.rs`

File read in full (871 lines, one `Read` call, no offset/limit needed).

### A.1 `trait DeinterlaceRow` — `temporal.rs:318-330`

```rust
pub trait DeinterlaceRow {
    /// The subject's canonical identity (for the [`DependsClosure`] lookup).
    fn subject(&self) -> &str;
    /// The storage-frame clock (this row's Lance version).
    fn lance_version(&self) -> LanceVersion;
    /// The schema-frame clock (when this row's class became knowable). Sourced
    /// by `ogar-adapter-surrealql`'s `DEFINE TABLE` registration.
    fn knowable_from(&self) -> LanceVersion;
    /// The cross-server causal tick; `None` single-server.
    fn hlc_tick(&self) -> Option<u64> {
        None
    }
}
```

- `subject(&self) -> &str` — **required** (`temporal.rs:320`). Doc: "The
  subject's canonical identity (for the `DependsClosure` lookup)."
- `lance_version(&self) -> LanceVersion` — **required** (`temporal.rs:322`).
  Doc, quoted verbatim: **"The storage-frame clock (this row's Lance
  version)."** (`temporal.rs:321`)
- `knowable_from(&self) -> LanceVersion` — **required** (`temporal.rs:325`).
  Doc, quoted verbatim: **"The schema-frame clock (when this row's class
  became knowable). Sourced by `ogar-adapter-surrealql`'s `DEFINE TABLE`
  registration."** (`temporal.rs:324-325`)
- `hlc_tick(&self) -> Option<u64>` — **provided** (default body `None`,
  `temporal.rs:327-329`). Doc: "The cross-server causal tick; `None`
  single-server."

**The load-bearing distinction (per the task brief):** `lance_version` is the
row's *storage* clock (which Lance version wrote this row). `knowable_from`
is the *schema* clock (which Lance version is when the row's **class** first
became definable at all, sourced from a `DEFINE TABLE`-style registration
event upstream). `classify` (§A.5) reads both, and reads them for **different
purposes**: `knowable_from` gates `Unknowable` outright (checked first, before
`row_version` is even compared); `lance_version` (via `row_version`) gates
`Contemporary` vs `Anachronistic`/`Spoiler`. They are not interchangeable and
not the same axis.

### A.2 `fn deinterlace` — `temporal.rs:345-376`

```rust
#[must_use]
pub fn deinterlace<R, D>(rows: &[R], v_ref: &QueryReference, deps: &D) -> Vec<R>
where
    R: DeinterlaceRow + Clone,
    D: DependsClosure,
```

- **Returns:** `Vec<R>` — the causally-coherent, dispatchable projection at
  `v_ref`, cloned out of `rows`.
- **What it filters out** (`temporal.rs:351-364`): every row `r` for which
  `classify_ready(r.subject(), r.lance_version(), r.knowable_from(), v_ref,
  deps).dispatchable(v_ref.mode)` is `false` — i.e. rows failing EITHER the
  TIME-causal admission (`EpistemicMode::admits`, §A.6) OR the DATA-causal
  readiness (`DependsClosure::closure_at(...).satisfied`). Both must hold.
- **Exact sort key** (`temporal.rs:369-374`):
  ```rust
  out.sort_by_key(|r| {
      (
          r.hlc_tick().unwrap_or_else(|| r.lance_version()),
          r.lance_version(),
      )
  });
  ```
  Primary key: `hlc_tick()`, falling back to the row's OWN `lance_version()`
  when `hlc_tick()` is `None` (explicitly NOT falling back to `0` — a Codex
  P2 fix documented in the comment directly above, `temporal.rs:365-368`).
  Secondary key: `lance_version()` (tie-break / stable ordering within one
  HLC tick).

### A.3 `struct QueryReference` — `temporal.rs:134-148`

```rust
pub struct QueryReference {
    pub server_id: u16,
    pub ref_version: LanceVersion,
    pub hlc_tick: Option<u64>,
    pub mode: EpistemicMode,
    pub rung: u8,
}
```

Field docs, each quoted:
- `server_id: u16` — "The reader's frame of reference (which writer's version
  line). `0` = single-server."
- `ref_version: LanceVersion` — "The `KnowledgeHorizon` — the Lance version the
  reader is pinned at. `u64::MAX` = \"latest\" (the single-server default)."
- `hlc_tick: Option<u64>` — "Cross-server causal tick; `None` single-server.
  Wakes up under the peer-Raft / cluster-bus policy (deferred)."
- `mode: EpistemicMode` — "What the reader is allowed to know."
- `rung: u8` — "The reader's rung (drives `EpistemicMode::for_rung`)."

**Constructors — exact field values, load-bearing:**

`impl Default for QueryReference` — `temporal.rs:150-161`:
```rust
fn default() -> Self {
    // The single-server reading: latest version, strict, rung 0, no HLC.
    Self {
        server_id: 0,
        ref_version: u64::MAX,
        hlc_tick: None,
        mode: EpistemicMode::Strict,
        rung: 0,
    }
}
```
`server_id = 0`, `ref_version = u64::MAX`, `hlc_tick = None`, `mode =
EpistemicMode::Strict`, `rung = 0`.

`QueryReference::at(ref_version: LanceVersion, rung: u8) -> Self` —
`temporal.rs:163-176`:
```rust
#[must_use]
pub fn at(ref_version: LanceVersion, rung: u8) -> Self {
    Self {
        server_id: 0,
        ref_version,
        hlc_tick: None,
        mode: EpistemicMode::for_rung(rung),
        rung,
    }
}
```
`server_id = 0` (always, unconditionally), `ref_version` = the caller's
argument (verbatim), `hlc_tick = None` (always — `::at` never sets an HLC
tick; only direct struct-literal construction, e.g. the test at
`temporal.rs:689-692`, can set one), `mode = EpistemicMode::for_rung(rung)`
(derived, NOT independently settable), `rung` = the caller's argument
(verbatim, stored alongside the derived mode).

**No other named constructor exists.** Direct struct-literal construction
(with `..QueryReference::default()`) is used in three tests
(`temporal.rs:689-692`, `733-736`) to set `hlc_tick`/`ref_version` outside
`::at`'s policy — there is no `::with_hlc` or similar helper.

### A.4 `enum EpistemicMode` — `temporal.rs:76-85,87-113`

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EpistemicMode {
    /// Only `CONTEMPORARY` rows (`row_version ≤ ref_version`). The default.
    Strict,
    /// May also use `ANACHRONISTIC` rows — hindsight from a future frame.
    Aware,
    /// May also take a `SPOILER` — an intentional `V_now` read past the
    /// horizon (rung 9+).
    Retro,
}
```

- `Strict` — "Only `CONTEMPORARY` rows (`row_version ≤ ref_version`). The
  default."
- `Aware` — "May also use `ANACHRONISTIC` rows — hindsight from a future
  frame."
- `Retro` — "May also take a `SPOILER` — an intentional `V_now` read past the
  horizon (rung 9+)."

Also on this type: `for_rung(rung: u8) -> Self` (`temporal.rs:90-97`,
`0..=4 → Strict`, `5..=8 → Aware`, `_ → Retro`) and `admits` (§A.6).

### A.5 `enum TemporalStatus` — `temporal.rs:117-126`

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TemporalStatus {
    /// In-phase: `row_version ≤ ref_version` and the class was already knowable.
    Contemporary,
    /// A future frame's row (`row_version > ref_version`) — hindsight.
    Anachronistic,
    /// An intentional read past the horizon under `Retro` mode.
    Spoiler,
    /// The class's `knowable_from` is past the horizon — not yet knowable.
    Unknowable,
}
```

- `Contemporary` — "In-phase: `row_version ≤ ref_version` and the class was
  already knowable."
- `Anachronistic` — "A future frame's row (`row_version > ref_version`) —
  hindsight."
- `Spoiler` — "An intentional read past the horizon under `Retro` mode."
- `Unknowable` — "The class's `knowable_from` is past the horizon — not yet
  knowable."

### A.6 `fn classify` — `temporal.rs:184-199`

```rust
#[must_use]
pub fn classify(
    row_version: LanceVersion,
    knowable_from: LanceVersion,
    v_ref: &QueryReference,
) -> TemporalStatus {
    if knowable_from > v_ref.ref_version {
        TemporalStatus::Unknowable
    } else if row_version <= v_ref.ref_version {
        TemporalStatus::Contemporary
    } else if matches!(v_ref.mode, EpistemicMode::Retro) {
        TemporalStatus::Spoiler
    } else {
        TemporalStatus::Anachronistic
    }
}
```

Parameters:
- `row_version: LanceVersion` — the specific row's own Lance/storage version
  (the caller supplies `DeinterlaceRow::lance_version()` here in practice,
  via `classify_ready`, §A.7).
- `knowable_from: LanceVersion` — the row's *class's* knowable-from version
  (caller supplies `DeinterlaceRow::knowable_from()`).
- `v_ref: &QueryReference` — the reader's reference (horizon + mode).

Decision order (checked exactly in this order, first match wins):
1. `knowable_from > v_ref.ref_version` → `Unknowable` (checked BEFORE
   anything else — a class not yet knowable is unknowable regardless of the
   individual row's own version).
2. else `row_version <= v_ref.ref_version` → `Contemporary`.
3. else (row is in the future) `v_ref.mode == Retro` → `Spoiler`.
4. else → `Anachronistic`.

### A.7 `dispatchable` — method on `Classification`, `temporal.rs:281-297`

Not a free function — a method on `struct Classification`:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Classification {
    /// The TIME-causal status (HLC axis).
    pub temporal: TemporalStatus,
    /// The DATA-causal readiness (depends-closure axis); `true` under `NoDeps`.
    pub data_ready: bool,
}

impl Classification {
    #[must_use]
    pub fn dispatchable(&self, mode: EpistemicMode) -> bool {
        mode.admits(self.temporal) && self.data_ready
    }
}
```

`dispatchable(&self, mode: EpistemicMode) -> bool` = `mode.admits(self.temporal)
&& self.data_ready` — a conjunction of the TIME axis (via `EpistemicMode::admits`,
below) and the DATA axis (`self.data_ready`, populated by `DependsClosure`,
trivially always `true` under `NoDeps`, §A.8).

**`EpistemicMode::admits` — `temporal.rs:99-113`:**
```rust
#[must_use]
pub fn admits(self, status: TemporalStatus) -> bool {
    match status {
        TemporalStatus::Contemporary => true,
        TemporalStatus::Anachronistic => {
            matches!(self, EpistemicMode::Aware | EpistemicMode::Retro)
        }
        TemporalStatus::Spoiler => matches!(self, EpistemicMode::Retro),
        TemporalStatus::Unknowable => false,
    }
}
```

**MODE × STATUS pass/block table** (this is `EpistemicMode::admits` alone —
the TIME axis only; `Classification::dispatchable` ANDs this with the
separate, orthogonal `data_ready` bool from `DependsClosure`, which is NOT
part of this table):

| Mode \\ Status | `Contemporary` | `Anachronistic` | `Spoiler` | `Unknowable` |
|---|---|---|---|---|
| `Strict` | **PASS** | BLOCK | BLOCK | BLOCK |
| `Aware`  | **PASS** | **PASS** | BLOCK | BLOCK |
| `Retro`  | **PASS** | **PASS** | **PASS** | BLOCK |

`Unknowable` is BLOCKED under every mode, unconditionally — confirmed by the
`admits_per_mode` test (`temporal.rs:638-653`) which loops all three modes
asserting `!m.admits(TemporalStatus::Unknowable)`.

### A.8 `NoDeps` — `temporal.rs:270-277`

```rust
/// The trivial DATA-causal impl — no dependencies, always ready. The
/// single-server default until the SPO frontends emit real edges.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoDeps;

impl DependsClosure for NoDeps {
    fn closure_at(&self, _subject: &str, _v_ref: &QueryReference) -> DepClosure {
        DepClosure::ready()
    }
}
```

- A unit struct (`pub struct NoDeps;`) implementing `trait DependsClosure`
  (`temporal.rs:263-266`: `fn closure_at(&self, subject: &str, v_ref:
  &QueryReference) -> DepClosure;`).
- Its `closure_at` ignores both arguments and always returns
  `DepClosure::ready()` — `{ edges: Vec::new(), satisfied: true }`
  (`temporal.rs:238-246`).
- Constructed either as the unit-struct literal `NoDeps` (used directly at
  every call site in the test module, e.g. `temporal.rs:665`,
  `&NoDeps`) or via its derived `Default` (`NoDeps::default()` — same value,
  `#[derive(Default)]` on a unit struct).

Supporting context (not separately asked for, but load-bearing for the
table above): `trait DependsClosure` (`temporal.rs:263-266`), `struct
DepClosure` (`temporal.rs:229-236`, fields `edges: Vec<DepEdge>` +
`satisfied: bool`), `DepClosure::ready()` (`temporal.rs:240-246`) and its
`impl Default for DepClosure` which is EXPLICITLY made to match `ready()`
(`temporal.rs:249-257`, a Codex P2 fix so `..Default::default()` does not
silently produce `satisfied: false`), and `fn classify_ready` (which composes
`classify` + `DependsClosure` into `Classification`, `temporal.rs:302-314`).

### A.9 Full `#[cfg(test)] mod tests` roster — `temporal.rs:450-870`

In file order:

1. **`layer1_deinterlaces_interleaved_global_log_into_owner_local_chain`**
   (`temporal.rs:525`) — asserts the layer-1 `local_trajectory_of` /
   `local_trajectories` split the interleaved global log
   `A@s0,C@s0,B@s0,A@s1` into per-owner chains, with A's chain strictly
   shorter than the global log (anti-vacuity: other owners' rows are actually
   *removed*, not merely reordered).
2. **`layer1_orders_one_owners_chain_by_cast_seq_not_log_order`**
   (`temporal.rs:586`) — asserts one owner's chain replays in `cast_seq`
   order even when the durable log stored the casts out of order
   (seq-descending input).
3. **`for_rung_policy`** (`temporal.rs:614`) — asserts `EpistemicMode::for_rung`
   boundaries: 0/4→Strict, 5/8→Aware, 9/255→Retro.
4. **`classify_time_axis`** (`temporal.rs:623`) — asserts `classify`'s four
   outcomes directly: in-phase→Contemporary, boundary-equal→Contemporary,
   `knowable_from` past horizon→Unknowable, future row under Strict→
   Anachronistic, the SAME future row under Retro→Spoiler.
5. **`admits_per_mode`** (`temporal.rs:638`) — asserts each cell of the
   MODE×STATUS table in §A.7 individually, plus the "Unknowable is never
   admitted" loop over all three modes.
6. **`deinterlace_filters_and_orders_single_server`** (`temporal.rs:655`) —
   asserts `deinterlace` on a strict single-server reference keeps only the
   contemporary rows and orders them by version.
7. **`data_causal_axis_can_drop_time_contemporary_rows`** (`temporal.rs:670`)
   — asserts a time-contemporary row is still excluded under `BlockDeps`
   (a `DependsClosure` that reports `satisfied: false`), proving the AND
   conjunction in `dispatchable` is real (not short-circuited to the time
   axis alone).
8. **`deinterlace_hlc_orders_across_frames`** (`temporal.rs:685`) — asserts
   rows are ordered by `hlc_tick`, not by per-frame `lance_version`, when an
   HLC tick is present.
9. **`query_reference_default_is_single_server_latest`** (`temporal.rs:706`)
   — asserts `QueryReference::default()`'s exact field values (mirrors §A.3).
10. **`dep_closure_default_is_ready_not_blocking`** (`temporal.rs:718`) —
    Codex P2 regression: asserts `DepClosure::default().satisfied == true`
    and `edges.is_empty()`.
11. **`deinterlace_mixed_hlc_falls_back_to_lance_version`** (`temporal.rs:731`)
    — Codex P2 regression: asserts a legacy row with no HLC tick sorts by its
    OWN `lance_version` (not by `0`), interleaved correctly among HLC-bearing
    rows.
12. **`no_hindsight_streamed_known_game`** (`temporal.rs:791`) — flagged below
    in detail.

**`no_hindsight_streamed_known_game` in detail** (`temporal.rs:791-869`):

Models a "known game" as a 10-ply move sequence, one `Row` per ply, with
`lance_version == ply index` and `knowable_from == 0` for every row (the
*class* "a game is being observed" was always knowable; only individual ply
rows arrive over time). For three representative "present" readers
(`v ∈ {2, 5, 8}`, each `QueryReference::at(v, 0)` → rung 0 → `Strict`):

- Loops every ply and asserts: `ply <= v` classifies `Contemporary`;
  `ply > v` classifies **`Anachronistic`** (NOT `Spoiler` — the test's own
  doc comment calls this out as the surprising part, since `Spoiler` only
  ever appears for a reader whose OWN mode is already `Retro`) and is refused
  by `strict.mode.admits(status)` (asserts `false`).
- Then calls `deinterlace(&rows, &strict, &NoDeps)` and asserts the visible
  version set is EXACTLY `0..=v` — the whole future is excluded from the
  projection outright, not merely flagged.
- Contrasts with a `Retro` reader (`QueryReference::at(v, 9)`) at the SAME
  `v_ref`: the immediate next future ply classifies `Spoiler` (not
  `Anachronistic`) and IS admitted (`retro.mode.admits(retro_status)` is
  `true`), and appears in that reader's `deinterlace` output — demonstrating
  Strict's blindness is structural/default and Retro's visibility is an
  explicit opt-in, never an accidental leak.
- Doc comment cross-references `AdaWorldAPI/stockfish-rs`
  `examples/hindsight_stream.rs` (`D-SF-HINDSIGHT-1`) as a downstream
  consumer of a zero-dep mirror of this exact machinery.

---

## B. `crates/jc/src/stats.rs`

File read lines 1–1161 (of 2157 total) via one `Read` call — this fully
covers every item requested (`BinaryAssociation` ends `line 634`,
`binary_association` ends `line 693`, `cohen_kappa` ends `line 302`, `phi`
ends `line 600`, `kr20` ends `line 723`; all comfortably inside the read
range). The unread tail (1162–2157) covers `multiple_r`/`eta_squared`/t-tests/
ANOVA machinery — none of it was in scope and none of it was read.

### B.1 `struct BinaryAssociation` — `stats.rs:612-634`

```rust
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BinaryAssociation {
    /// Count of `(false, false)`.
    pub n00: u64,
    /// Count of `(false, true)`.
    pub n01: u64,
    /// Count of `(true, false)`.
    pub n10: u64,
    /// Count of `(true, true)`.
    pub n11: u64,
    /// Rate of `true` in the first rater — the marginal φ's ceiling depends on.
    pub positive_rate_a: f64,
    /// Rate of `true` in the second rater.
    pub positive_rate_b: f64,
    /// `p_o` — proportion of cells where the two agree.
    pub observed_agreement: f64,
    /// `p_e` — agreement expected from the marginals alone.
    pub expected_agreement: f64,
    /// Cohen's κ, or `None` when `p_e == 1` (undefined, `0/0`).
    pub kappa: Option<f64>,
    /// φ, or `None` when either variable is constant (zero variance).
    pub phi: Option<f64>,
}
```

Every field, name/type/meaning, exactly as commented above:
- `n00: u64` — count of `(false, false)`.
- `n01: u64` — count of `(false, true)`.
- `n10: u64` — count of `(true, false)`.
- `n11: u64` — count of `(true, true)`.
- `positive_rate_a: f64` — rate of `true` in the first rater (the marginal
  φ's ceiling depends on).
- `positive_rate_b: f64` — rate of `true` in the second rater.
- `observed_agreement: f64` — `p_o`, proportion of cells where the two agree.
- `expected_agreement: f64` — `p_e`, agreement expected from the marginals
  alone.
- `kappa: Option<f64>` — Cohen's κ, or `None` when `p_e == 1` (undefined,
  `0/0`).
- `phi: Option<f64>` — φ, or `None` when either variable is constant (zero
  variance).

### B.2 `fn binary_association` — `stats.rs:653-693`

```rust
pub fn binary_association(a: &[bool], b: &[bool]) -> Option<BinaryAssociation> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }
    let (mut n00, mut n01, mut n10, mut n11) = (0u64, 0u64, 0u64, 0u64);
    for (&x, &y) in a.iter().zip(b.iter()) {
        match (x, y) {
            (false, false) => n00 += 1,
            (false, true) => n01 += 1,
            (true, false) => n10 += 1,
            (true, true) => n11 += 1,
        }
    }
    let n = a.len() as f64;
    let pa = (n10 + n11) as f64 / n;
    let pb = (n01 + n11) as f64 / n;
    let p_o = (n00 + n11) as f64 / n;
    let p_e = pa * pb + (1.0 - pa) * (1.0 - pb);
    let kappa = {
        let denom = 1.0 - p_e;
        if denom == 0.0 || !denom.is_finite() {
            None
        } else {
            let k = (p_o - p_e) / denom;
            k.is_finite().then_some(k)
        }
    };
    Some(BinaryAssociation {
        n00, n01, n10, n11,
        positive_rate_a: pa,
        positive_rate_b: pb,
        observed_agreement: p_o,
        expected_agreement: p_e,
        kappa,
        phi: phi(a, b),
    })
}
```

- **Returns:** `Option<BinaryAssociation>`.
- **The ONLY conditions under which the function itself returns `None`**
  (`stats.rs:654-656`): `a.len() != b.len()` OR `a.is_empty()`. Doc
  (`stats.rs:640-643`) states this explicitly: *"Returns `None` only on
  structurally unusable input (length mismatch or empty); a degenerate
  *table* still returns the counts, with `kappa` / `phi` individually `None`,
  because the counts remain informative even where the coefficients are
  undefined."* — i.e. for any structurally-valid non-empty equal-length
  input, `binary_association` returns `Some(...)` always; the DEGENERACY
  (undefined κ/φ) is expressed as `None` on the individual `kappa: Option<f64>`
  / `phi: Option<f64>` FIELDS inside that `Some`, not as the function
  returning `None`.
  - `kappa` field is `None` when `1.0 - p_e == 0.0` or is non-finite, or the
    resulting `k` is non-finite.
  - `phi` field is `None` under whatever conditions `phi(a, b)` itself
    returns `None` (see §B.4).

### B.3 `fn cohen_kappa` — `stats.rs:279-302`

```rust
pub fn cohen_kappa(a: &[usize], b: &[usize]) -> Option<f64> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }
    let n = a.len() as f64;
    let cats: BTreeSet<usize> = a.iter().chain(b.iter()).copied().collect();

    let agree = a.iter().zip(b.iter()).filter(|(x, y)| x == y).count() as f64;
    let p_o = agree / n;

    let mut p_e = 0.0;
    for c in &cats {
        let ma = a.iter().filter(|&v| v == c).count() as f64 / n;
        let mb = b.iter().filter(|&v| v == c).count() as f64 / n;
        p_e += ma * mb;
    }

    let denom = 1.0 - p_e;
    if denom == 0.0 || !denom.is_finite() {
        return None;
    }
    let k = (p_o - p_e) / denom;
    k.is_finite().then_some(k)
}
```

- **Returns:** `Option<f64>` — `κ = (p_o − p_e) / (1 − p_e)`.
- **Full `None` condition list**, in the order checked:
  1. `a.len() != b.len()` (length mismatch), OR
  2. `a.is_empty()` (empty input) — checked together at `stats.rs:280-282`.
  3. `denom == 0.0 || !denom.is_finite()` where `denom = 1.0 - p_e`
     (`stats.rs:296-299`) — i.e. `p_e == 1` (both raters used a single
     identical category throughout — doc calls this "chance-corrected
     agreement is undefined, `0/0`") or `p_e` is non-finite.
  4. The final `k.is_finite().then_some(k)` (`stats.rs:301`) — returns `None`
     if the computed `k` is non-finite (NaN/inf), even if `denom` passed the
     check above.
- **Category type is `usize`, exact-equality partition** — not `f64` (doc,
  `stats.rs:259-261`: "float equality on measured values is a defect waiting
  to happen").

### B.4 `fn phi` — `stats.rs:596-600`

```rust
pub fn phi(x: &[bool], y: &[bool]) -> Option<f64> {
    let xf: Vec<f64> = x.iter().map(|&v| if v { 1.0 } else { 0.0 }).collect();
    let yf: Vec<f64> = y.iter().map(|&v| if v { 1.0 } else { 0.0 }).collect();
    pearson(&xf, &yf)
}
```

- **Argument type: `&[bool]` for BOTH parameters, `x: &[bool]` and
  `y: &[bool]`** — the plan's claim is **CONFIRMED**, exact as written.
- Converts each to `Vec<f64>` (`true → 1.0`, `false → 0.0`) and delegates
  entirely to `crate::reliability::pearson(&xf, &yf)` — φ IS Pearson r on the
  0/1-coded binary vectors, per the module doc (`stats.rs:26-27`: "φ IS
  Pearson r computed on two binary variables").
- **`None` conditions are exactly `pearson`'s** (not re-derived here; per doc
  `stats.rs:587-588`): "Returns `None` under `pearson`'s conditions: lengths
  differ, `n < 2`, or either vector is constant (all-true or all-false → zero
  variance)." `pearson`'s own body is in `crate::reliability`, NOT read as
  part of this task (out of file scope) — see NOT VERIFIED below.

### B.5 `fn kr20` — `stats.rs:717-723`

```rust
pub fn kr20(items: &[Vec<bool>]) -> Option<f64> {
    let numeric: Vec<Vec<f64>> = items
        .iter()
        .map(|it| it.iter().map(|&v| if v { 1.0 } else { 0.0 }).collect())
        .collect();
    crate::reliability::cronbach_alpha(&numeric)
}
```

- **Signature:** `pub fn kr20(items: &[Vec<bool>]) -> Option<f64>`.
- Converts every item (`true → 1.0`, `false → 0.0`) then delegates entirely
  to `crate::reliability::cronbach_alpha(&numeric)` — KR-20 IS Cronbach's α
  computed on dichotomous items (doc, `stats.rs:695-696`); "a naming surface,
  not a second implementation" (`stats.rs:701`).
- **`None` conditions are exactly `cronbach_alpha`'s** (doc, `stats.rs:704-706`):
  "Same shape and degeneracy conditions as `cronbach_alpha`: `k ≥ 2` items,
  equal-length non-empty rows, and non-zero variance of the per-subject
  totals (all-identical totals → `None`)." `cronbach_alpha`'s own body is in
  `crate::reliability`, NOT read as part of this task — see NOT VERIFIED
  below.

---

## C. `crates/lance-graph-planner/examples/blw_tenant.rs`

File read in full (1057 lines, one `Read` call).

### C.1 Exact `use` list — `blw_tenant.rs:85-100`

```rust
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use cognitive_shader_driver::mailbox_soa::{MailboxSoA, WriteCell, WriteOutcome, WORDS_PER_FP};
use lance_graph_contract::cognitive_shader::MetaWord;
use lance_graph_contract::collapse_gate::MailboxId;
use lance_graph_contract::kanban::{ExecTarget, KanbanColumn, KanbanMove};
use lance_graph_contract::scheduler::{DatasetVersion, NextPhaseScheduler, VersionScheduler};
use lance_graph_contract::soa_view::{IdentityPlane, MailboxSoaOwner, MailboxSoaView};
use lance_graph_planner::batch_writer::BatchWriter;
use lance_graph_planner::owner_adapter::emit_bootstrap_intent;
use lance_graph_planner::persist_sink::{
    persist_cycle, recover_and_apply, CycleFrame, CycleId, DetachedCycleBatch, LandedSlot,
    SweepSlot, WalSink, WriteFailed,
};
use lance_graph_planner::traits::StrategyOutcome;
```

**Load-bearing negative finding:** `lance_graph_planner::temporal` (the
`deinterlace` / `DeinterlaceRow` / `QueryReference` module inventoried in
§A) is **NOT imported anywhere in this file**, and `jc` (§B) is **not
imported either**. This is not an oversight — the module doc says so
explicitly (`blw_tenant.rs:46-49`): *"No `deinterlace` / `DeinterlaceRow`
read. There is still no production `DeinterlaceRow` implementor and no
production caller of `deinterlace` (`batch_writer.rs` module doc). The
durability *observation* seam is **not** wired here — see the report's
'seam I stopped at'."* Whatever D-BLW-3 is building, it is **new** wiring on
top of `blw_tenant.rs`'s pattern, not something already present in this file.

### C.2 How a version is sealed — exact call sequence, in order

Inside the per-cycle loop (`for spec in &plan { ... }`, `blw_tenant.rs:758-974`),
sealing happens at step "④ the seal":

1. **Build the landing slots** — one `SweepSlot` per fired row
   (`blw_tenant.rs:853-875`, each with `paired_move: None`), plus exactly ONE
   more `SweepSlot` carrying the tenant-level kanban move
   (`blw_tenant.rs:876-884`, `paired_move: Some(cast_move)`). Both are pushed
   into a single `Vec<SweepSlot>` called `slots`.
2. `let appends_before = sink.wal_writes();` — `blw_tenant.rs:887`
   (baseline read of the append counter, for the post-check).
3. `let base = sink.head();` — `blw_tenant.rs:888` (read the sink's current
   sealed head version, to pass as the optimistic-concurrency fence).
4. **The seal call itself:**
   ```rust
   let version = persist_cycle(&sink, CycleFrame::new(spec.id, base), slots).await?;
   ```
   — `blw_tenant.rs:889`. `persist_cycle` is imported from
   `lance_graph_planner::persist_sink` (§C.1); its own body was NOT read
   (out of file scope — see NOT VERIFIED). Within THIS file's own `MemWal`
   (the local `WalSink` impl, `blw_tenant.rs:411-501`), the actual append
   happens inside `MemWal::commit_cycle` (`blw_tenant.rs:446-470`): it
   locks `self.sealed`, checks `base != head` (fences a stale base,
   returning `WriteFailed` if so), THEN does
   `self.wal_writes.fetch_add(1, Ordering::SeqCst)` (the ONE physical append
   counter increment) and computes
   `let version = DatasetVersion(self.next_version.fetch_add(1, Ordering::SeqCst));`
   (`blw_tenant.rs:461-462`), pushing a `SealedCycle{ cycle, version,
   image_rows: batch.image.len(), landings: batch.landings }` into `sealed`.
5. **Post-check** — assert exactly one physical append happened:
   ```rust
   assert_eq!(sink.wal_writes() - appends_before, 1, "{n_landings} landings → exactly ONE WAL append");
   ```
   — `blw_tenant.rs:890-894`.

So: N landings (rows) + 1 tenant-level landing → ONE `persist_cycle` call →
ONE `MemWal::commit_cycle` → ONE `wal_writes` increment → ONE
`DatasetVersion` returned.

**Separately, immediately after** (step "⑤ THE TRAP", `blw_tenant.rs:896-899`),
the sealed `version` is used to ask what the scheduler WOULD have proposed
(`NextPhaseScheduler.on_version(&owner, version, ExecTarget::Native)`) — this
is a read-only comparison, NOT part of sealing.

**Then applying the seal** (step "⑥ the post-seal apply",
`blw_tenant.rs:901-923`): `sink.scan_sealed(Some(base)).await?` reads back
the sealed landings, and `recover_and_apply(&mut owner, &sealed,
watermark)?` (from `lance_graph_planner::persist_sink`) applies them to the
live tenant, advancing `watermark` and the tenant's kanban phase. This is a
SEPARATE step from sealing — sealing produces a durable (in this harness,
in-process) version; applying is what makes the live `owner` catch up to it.

### C.3 What identifies a version

**Type:** `DatasetVersion` — imported from
`lance_graph_contract::scheduler::{DatasetVersion, NextPhaseScheduler,
VersionScheduler}` (`blw_tenant.rs:92`). Its own definition was NOT read (out
of file scope — lives in `lance-graph-contract`, not one of the four files
this task names; see NOT VERIFIED).

**Where the number comes from:** the harness's own `MemWal.next_version:
AtomicU64` field (`blw_tenant.rs:413`), initialised to `AtomicU64::new(1)`
(`blw_tenant.rs:422`), and incremented via `self.next_version.fetch_add(1,
Ordering::SeqCst)` inside `commit_cycle` (`blw_tenant.rs:462`) each time a
cycle is sealed. So version numbers are `1, 2, 3, ...` in strict seal order,
scoped to this `MemWal` instance only — this is **explicitly NOT a Lance
version** (module doc, `blw_tenant.rs:44-45`: *"The version numbers here are
sequence numbers, not Lance versions."*).

`MemWal.head()` (`blw_tenant.rs:429-435`) reads the LAST sealed cycle's
version (or `DatasetVersion(0)` if none sealed yet) — this is the value read
as `base` before each seal (§C.2 step 3) and is the optimistic-concurrency
fence `commit_cycle` checks against.

### C.4 How rows/verses are addressed within the tenant

Each verse is assigned a plain `usize` row index at seed time
(`seed_tenant`, `blw_tenant.rs:528-550`): `for (row, text) in
verses.iter().enumerate().take(N_CAP)` — i.e. row index = the verse's
position in the (TSV-order, length-bounded) corpus vector, `0..N_CAP` where
`N_CAP = 2048` (`blw_tenant.rs:107`). `owner.write_row(row, cycle, &cell)`
writes that verse's encoded content/topic planes into the tenant's row
`row`.

Landing slots (`SweepSlot`, from `lance_graph_planner::persist_sink`) carry
the row as `row: u64` (e.g. `blw_tenant.rs:861`, `payload: RowSpanDescriptor
{ row_lo: row, row_hi: row + 1, cycle: owner_cycle }.to_le_bytes().to_vec()`
per fired row, `blw_tenant.rs:866-873`) — so within one sealed cycle, an
individual verse/row is addressed by that same `usize`/`u64` index used at
seed time, carried through the `SweepSlot.row` field and the
`RowSpanDescriptor` payload.

There is no separate "verse id" type — the corpus-order index IS the row
address, used consistently from `seed_tenant` through `sweep_rows` (returns
`Vec<u32>` of fired row indices, `blw_tenant.rs:317-358`) through the landing
slots.

### C.5 Where the per-verse stance verdict is produced

**Nowhere in this file.** This harness deliberately does NOT produce a
per-verse stance verdict. Module doc, quoted verbatim (`blw_tenant.rs:50-54`):

> "**No stance instrument.** The row body is a deterministic bloom-containment
> read over the content identity plane, deliberately NOT a
> Hegel/Nietzsche/Kant/Wittgenstein projection: §12.3c retired κ and §12.7
> recorded the texture rewrite as a KILL. **D-BLW-1 is the substrate
> deliverable**; the instrument is D-BLW-2's problem and is not smuggled in
> here."

The only per-row computation this file performs is `sweep_rows`
(`blw_tenant.rs:311-358`): a bloom-filter overlap check (`fn bloom_add` /
`fn encode_plane` / `fn probe_plane`, `blw_tenant.rs:248-294`) against a
single probe term (e.g. `"god"`, `blw_tenant.rs:688`), producing a `Sweep`
aggregate (`scanned`/`fired`/`mean_similarity`/`mean_energy`,
`blw_tenant.rs:296-309`) and a `Vec<u32>` of fired row indices — this is
explicitly a substrate falsifier instrument, not a stance/semantic
verdict. No `jc` statistic (§B) and no `temporal.rs` classification (§A) is
invoked anywhere in this file.

---

## D. Reachability check — `crates/lance-graph-planner/Cargo.toml`

File read in full (89 lines).

**`jc` is currently NEITHER a `[dependencies]` NOR a `[dev-dependencies]`
entry.** Confirmed by scanning both sections (`Cargo.toml:8-44` for
`[dependencies]`, `Cargo.toml:46-77` for `[dev-dependencies]`) — no `jc = {
path = ... }` line exists in either.

**The comment block discussing re-adding it**, quoted verbatim
(`Cargo.toml:67-77`, immediately below the `cognitive-shader-driver`
dev-dependency entry):

```
# NOTE — `jc` dev-dep REMOVED with `examples/blw_lens_twin.rs` (the retired κ
# instrument; §12.3c). It was added solely for that harness and is now unused,
# so `crates/jc` is back to ZERO consumers in the workspace.
#
# If the D-BLW-2 rebuild needs it, re-add as `jc = { path = "../jc" }` under
# `[dev-dependencies]` and keep the constraint that made it safe: **dev-only,
# never a production dependency of the planner.** `jc` is the INDEPENDENT
# reference frame a discrimination measure is graded against (§12.5 — "the `jc`
# additive constraint continues to hold"), and a measure cannot be its own
# oracle. Do NOT invert this edge, and do NOT modify `crates/jc` while using it
# as the oracle.
```

If D-BLW-3 needs `jc::stats` (§B), the exact re-add line per this comment
would be `jc = { path = "../jc" }` under `[dev-dependencies]` — this file was
NOT edited (per the hard rules) and no such line was added.

---

## NOT VERIFIED

- **`crate::reliability::pearson`, `crate::reliability::cronbach_alpha`,
  `crate::reliability::{all_finite, mean}`** — these are consumed by `phi`
  and `kr20` (§B.4, B.5) and by internal helpers in `stats.rs`, but their own
  bodies live in `jc/src/reliability.rs`, a file NOT named in this task's
  scope and NOT read. `phi`'s and `kr20`'s exact `None` conditions are
  therefore stated as "whatever `pearson`/`cronbach_alpha` return `None`
  for" (per their own doc comments, quoted), not independently confirmed
  against `reliability.rs` source.
- **`persist_cycle`, `recover_and_apply`, `CycleFrame`, `CycleId`,
  `DetachedCycleBatch`, `LandedSlot`, `SweepSlot`, `WalSink`, `WriteFailed`**
  (from `lance_graph_planner::persist_sink`) and `BatchWriter`,
  `emit_bootstrap_intent`, `StrategyOutcome` (from `lance_graph_planner::
  batch_writer`/`owner_adapter`/`traits`) — all consumed by `blw_tenant.rs`
  (§C) but their own module source (`crates/lance-graph-planner/src/
  persist_sink.rs`, `batch_writer.rs`, `owner_adapter.rs`, `traits.rs`) was
  NOT read; this task named only `temporal.rs`, `stats.rs`,
  `examples/blw_tenant.rs`, and the planner `Cargo.toml`. Their signatures
  above are stated only insofar as `blw_tenant.rs`'s own call sites and
  `use` list constrain them (e.g. `persist_cycle`'s call shape at
  `blw_tenant.rs:889`), not independently confirmed against their
  definitions.
- **`DatasetVersion`, `NextPhaseScheduler`, `VersionScheduler`,
  `MailboxSoaOwner`, `MailboxSoaView`, `IdentityPlane`, `MetaWord`,
  `MailboxId`, `KanbanColumn`, `KanbanMove`, `ExecTarget`** (all from
  `lance_graph_contract::*`) and `MailboxSoA`, `WriteCell`, `WriteOutcome`,
  `WORDS_PER_FP` (from `cognitive_shader_driver::mailbox_soa`) — consumed
  throughout `blw_tenant.rs` but defined in `lance-graph-contract` /
  `cognitive-shader-driver`, neither of which was in this task's file list;
  not read.
- **`omega_total`, `multiple_r`, `multiple_r_squared`, `eta_squared`,
  `t_test_one_sample`, `t_test_paired`, `t_test_welch`, `t_test_student`,
  `anova_one_way`, and every `stats.rs` line past 1161** — the file exceeds
  the tool's 256 KB single-read cap (2157 lines total; only 1–1161 were
  read). None of these were asked for by the task brief and none were read;
  flagging explicitly so no downstream reader assumes `stats.rs` was read in
  full.
- **`.claude/board/AGENT_LOG.md`** — file is 534 KB, too large for a single
  `Read`; only the first 150 lines (most recent entries, the file is
  prepend-ordered) were read, per the task's "read before starting"
  instruction. Not read in full; no write was made to it (per the hard
  rules).
