# Trace B — writer-key `(server_id, lance_version, hlc_tick)` dormancy map

Scope note: the brief names `crates/lance-graph-contract/src/temporal.rs` — that
path does not exist. The canonical epistemology module is
`crates/lance-graph-planner/src/temporal.rs` (confirmed via grep for
`QueryReference`/`DeinterlaceRow`). The contract crate's *mirror* is
`crates/lance-graph-contract/src/temporal_pov.rs` (a range-only, HLC-blind
shape — see its module doc, which explicitly defers `EpistemicMode`/
`TemporalStatus`/HLC to the planner side). Both read in full below.
`crates/lance-graph-contract/src/scheduler.rs` also read in full: it is the
kanban `VersionScheduler` IN-direction (Lance `DatasetVersion` → `KanbanMove`)
and does **not** touch `QueryReference`, `server_id`, or `hlc_tick` at all —
irrelevant to this trace except as a negative data point (another temporal
seam in the same crate that also never wires a real clock).

## QueryReference sites

All confirmed via `grep -n "QueryReference"` across every `.rs` file in
`crates/`, then read in context.

| Site | file:line | Classification | `server_id` | `hlc_tick` |
|---|---|---|---|---|
| `Default for QueryReference` | `lance-graph-planner/src/temporal.rs:126-136` | production (canonical ctor) | hardcoded `0` | hardcoded `None` |
| `QueryReference::at(ref_version, rung)` | `lance-graph-planner/src/temporal.rs:139-151` | production (canonical ctor) | hardcoded `0` | hardcoded `None` |
| `QueryReference::at(10, 2)` / `(11, 2)` etc. | `lance-graph-planner/src/temporal.rs:418, 427, 451, 467, 598, 640` | **test** (`#[cfg(test)] mod tests`) | via `at()` → `0` | via `at()` → `None` |
| `QueryReference { ref_version: 1000, ..QueryReference::default() }` | `lance-graph-planner/src/temporal.rs:482-485, 526-529` | **test** | `0` (from `default()`) | `None` (from `default()`) |
| `VersionedSnapshot::new(QueryReference::at(10, 2), 7, snap)` and 4 siblings | `lance-graph-planner/src/nars/insight.rs:284, 285, 296, 299, 303` | **test** (`#[cfg(test)] mod tests`) | `0` | `None` |
| `let mut other_line = QueryReference::at(11, 2); other_line.server_id = 3;` | `lance-graph-planner/src/nars/insight.rs:307-308` | **test** | **non-zero (`3`)** — the ONLY site in the entire Rust workspace that ever sets a non-zero `server_id` | `None` |
| `QueryReference::at(...)` (doc prose only, no actual call) | `lance-graph/examples/reasoning_loop.rs:13, 33, 51-52, 251` | **example, doc/println text only** — the string `"temporal.rs QueryReference::at(t, rung=0)"` is printed/commented, never imported or called; `main()` never imports `crate::temporal` | n/a (no call site) | n/a |
| `` `QueryReference` `` mentioned in doc comment (mirror module doc) | `lance-graph-contract/src/temporal_pov.rs:7, 20, 23, 33-34, 39, 145, 169, 173, 186-187` | doc-only cross-reference; this module deliberately does **not** implement the type | n/a | n/a |
| `` `QueryReference`/`deinterlace` `` mentioned | `lance-graph-contract/src/lib.rs:164-165` (module doc for `temporal_pov`) | doc-only | n/a | n/a |
| `` richer `QueryReference` once that type is reachable from the contract `` | `lance-graph-contract/src/causal_audit.rs:260` | doc-only — explicitly documents that `SupportReceipt::at` uses the *weaker* `DatasetVersion` **because** `QueryReference` is not reachable from the zero-dep contract crate | n/a | n/a |
| `` a `temporal.rs` version-range read (`QueryReference::at`) is required `` | `lance-graph-contract/src/witness_fabric.rs:134` | doc-only — see "Touch points" below | n/a | n/a |
| `` durability evidence ... read through `crate::temporal` (`QueryReference::at` + deinterlace) `` | `lance-graph-planner/src/batch_writer.rs:9-10` | doc-only — module-doc claim; **no actual call anywhere in `batch_writer.rs`** (grepped, zero hits of `QueryReference`/`deinterlace`/`temporal` as code, only in the doc comment) | n/a | n/a |

**Zero production (non-test, non-doc, non-example-string) construction sites
exist anywhere in the workspace.** Every literal `QueryReference::at(...)` /
`QueryReference::default()` / struct-literal call that actually executes is
inside a `#[cfg(test)]` module.

## DeinterlaceRow impls

Exhaustive grep for `impl DeinterlaceRow` across `crates/` returns exactly
**one** implementor:

- `struct Row` — `lance-graph-planner/src/temporal.rs:358-388`, declared
  **inside** `#[cfg(test)] mod tests` (the `mod tests` block opens at line 355).
  - `lance_version()` (line 379-381) returns the struct's own `v: LanceVersion`
    field, which test call sites set directly to small integers (`Row::new(30,
    10, None)` etc.) — a real counter in shape, but fed only synthetic test
    data, never a real Lance `Dataset::version()` read.
  - `knowable_from()` (382-384) likewise returns the plain stored field.
  - `hlc_tick()` (385-387) returns `self.hlc: Option<u64>` — and test call
    sites DO exercise `Some(n)` values (`Row::new(900, 1, Some(3))` at line
    487, `Row::new(100, 1, Some(500))` at line 534, etc.) — but this is the
    **row's** HLC tick (`DeinterlaceRow::hlc_tick`), a different field from
    `QueryReference::hlc_tick` (the **reader's** tick). The row-level field is
    exercised with real `Some` values in tests; the reader-level
    `QueryReference::hlc_tick` field is never `Some` anywhere, test or
    production (see table above).

No production type anywhere (no Lance row wrapper, no SoA row struct, no
odoo/callcenter/witness type) implements `DeinterlaceRow`. There is no
`#[derive(DeinterlaceRow)]` macro, no blanket impl, no adapter.

## deinterlace callers

Exhaustive grep for `deinterlace(` returns 6 call sites, **all inside
`lance-graph-planner/src/temporal.rs`'s own `#[cfg(test)] mod tests`**:
lines 458, 469, 470, 491, 538, 627, 653. Zero calls outside that test module,
in this or any other crate.

**Verdict: the mechanism is test-only end to end.** No production Lance read
path, no route handler, no kanban step, no SPO ingestion path, nowhere in the
workspace actually calls `deinterlace()` against real rows. The doc-comment
claims that reference it in production prose (`batch_writer.rs:9-10`,
`reasoning_loop.rs:51-52`, `witness_fabric.rs:134`) are aspirational —
none of them is backed by an actual call.

## Touch points hardcoding 0/None

Exact functions/signatures that currently hardcode the zero/`None` writer-key
components, i.e. what would need to change for the key to go live:

1. `QueryReference::at(ref_version: LanceVersion, rung: u8) -> Self` —
   `lance-graph-planner/src/temporal.rs:143-151`. Body hardcodes
   `server_id: 0` (line 145) and `hlc_tick: None` (line 147). This is the
   **only non-test constructor** callers anywhere would realistically reach
   for (its sibling `Default` also hardcodes both, lines 129-133). Any
   caller wanting a non-zero `server_id` or `Some(hlc_tick)` today must
   hand-build the struct literal directly (as the one test at
   `insight.rs:307-308` does) — there is no `QueryReference::at_server(...)`
   or `QueryReference::with_hlc(...)` constructor.
2. `DeinterlaceRow::hlc_tick(&self) -> Option<u64>` —
   `lance-graph-planner/src/temporal.rs:303-305`. The trait's **default
   method body is `None`** — any production type that implements
   `DeinterlaceRow` without overriding this method silently gets no HLC
   participation. (Moot today since there are zero production implementors,
   but this is the exact place a future SoA-row impl would need to override.)
3. `deinterlace<R, D>(rows: &[R], v_ref: &QueryReference, deps: &D)` —
   `lance-graph-planner/src/temporal.rs:322-352`. The sort key at line
   345-350 (`r.hlc_tick().unwrap_or_else(|| r.lance_version())`) is already
   HLC-aware in shape — it does not need to change; it simply has never been
   exercised with a non-`None` `hlc_tick()` outside the two ordering tests
   (`deinterlace_hlc_orders_across_frames`,
   `deinterlace_mixed_hlc_falls_back_to_lance_version`).
4. `VersionedSnapshot::new` / `VersionedSnapshot::of` —
   `lance-graph-planner/src/nars/insight.rs:83-89, 93-100`. Both simply
   accept `at: QueryReference` as a caller-supplied parameter (no
   hardcoding inside these two functions themselves), but **every actual
   call site supplying that parameter is test-only** (see table above) —
   so the hardcoding is one hop upstream, at whichever `QueryReference`
   constructor the (currently nonexistent) production caller would use.
5. `ActionInvocation::emitted_at_millis: Option<u64>` —
   `lance-graph-contract/src/action.rs:207-208`. Named "HLC emit stamp" in
   its doc comment, but it is a **plain wall-clock millis stamp**, not a
   `(server_id, lance_version, hlc_tick)` triple — no `server_id` field, no
   comparison/merge logic. `temporal.rs`'s own module doc (line 45) calls
   this exact field out by name as "the `emitted_at_millis: u64` (decision
   #4) non-`Option` trap" it is designed to avoid repeating — i.e. this
   field is a documented **near-miss/anti-pattern reference**, not a
   wired HLC source, and is not on the `QueryReference` path at all.

None of the above proposes a wiring design (per the brief) — they are the
touch points as they stand today.

## HLC source

**MISSING.** Grepped the entire workspace (`crates/`) for `hlc`, `HLC`,
`logical_clock`, `lamport`, and `tick` in HLC-adjacent contexts:

- No type, struct, or function named `HybridLogicalClock`, `Hlc`, `LamportClock`,
  or similar exists anywhere in `crates/`.
- Every `hlc`/`HLC` occurrence in `.rs` files is either (a) the
  `QueryReference.hlc_tick: Option<u64>` / `DeinterlaceRow::hlc_tick()` field
  names themselves (already covered above), or (b) prose in doc comments
  describing the *concept* (`temporal.rs:16-17,42,45,106`; `lib.rs:164-165`).
- `action.rs:207`'s `emitted_at_millis` (see touch point 5 above) is a plain
  millis stamp, not an HLC tick generator — no `server_id`/logical-counter
  pairing, no comparison operator, no merge-on-receive logic (the three
  things that make a timestamp an HLC rather than a wall clock).
- No `tick()`-style monotonic counter generator was found feeding
  `hlc_tick` anywhere; every non-`None` `hlc_tick` value in the codebase is a
  literal integer written directly in a test (`Some(3)`, `Some(500)`, etc.).

**Conclusion:** there is no HLC generator anywhere in the workspace today.
The `hlc_tick: Option<u64>` field is a pure type-level placeholder — exactly
as `temporal.rs`'s own module doc states it should be ("Cross-server is
type-visible, policy-deferred... single-server body can ignore them").

## UNDETERMINED

- Whether any **non-Rust** component (e.g. a Python harness, a SurrealQL
  adapter script, or infra config outside `crates/`) generates or would feed
  an HLC tick was not checked — out of scope for a `crates/`-only Rust trace
  per the brief's read list; flagging as unexamined rather than asserting
  absence.
- Whether `WaveGrounding::Escalate` (the caller of `witness_fabric.rs`'s
  `out_of_horizon` flag, `witness_fabric.rs:325/425/502`) is itself consumed
  by any code path that would eventually reach a `QueryReference::at` call
  was not traced past its immediate return value — the flag's own module
  never calls `temporal::deinterlace` or constructs a `QueryReference`, so
  the "signal" described in the `witness_fabric.rs:134` doc comment is,
  at minimum, not resolved within `lance-graph-contract` itself; whether a
  downstream consumer (outside `crates/`, e.g. a private consumer repo) acts
  on it is unexamined.
