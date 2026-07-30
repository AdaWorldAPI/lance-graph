# T5 CLOSURE — schema.has() gate on the two fixed-offset sweepers

## What T5 actually was (located, not assumed)

Grepped for "two fixed-offset sweepers" and found no literal string match anywhere
in the tree — the task label was a description, not a quote. Located by finding
every file with multiple `value_offset()` call sites and reading each candidate:
`crates/lance-graph-contract/src/nan_projection.rs`'s `project_energy_nonfinite`
and `energy_all_finite` are the two functions that (a) literally sweep
`rows: &[NodeRow]`, (b) read a fixed offset (`ValueTenant::Energy.value_offset()`)
unconditionally, with (c) zero schema gate — unlike the already-correct sibling
pattern in `ocr.rs`'s `to_node_row` (`if schema.has(ValueTenant::EntityType) { ... }`).

## The real risk (not hypothetical)

`value_offset()` is a FIXED reserved byte position per tenant, identical across
every `ValueSchema` (RESERVE, DON'T RECLAIM) — so an ungated read is never memory-
unsafe. The risk is semantic: a row whose resolved schema does NOT materialise
`Energy` (e.g. `ValueSchema::Compressed`, used by `NodeGuid::CLASSID_FMA` — no
writer obligated to keep that byte range meaningful) could have foreign/garbage/
uninitialized bytes at the Energy offset misread as a real accumulator — a false
non-finite flag on data that was never Energy at all.

## Why this wasn't already biting anyone (and why it still needed fixing)

`ReadMode::DEFAULT` (what an unconfigured/classid-0 row resolves to) is currently
pinned to `ValueSchema::Full` as a **documented TEMPORARY 2026-06-15 POC** setting
— the doc comment on `ReadMode::DEFAULT` says explicitly: "When the POC ends, flip
`value_schema` back to `ValueSchema::Bootstrap` HERE and in `ClassView` together."
`Full` includes every tenant, so EVERY row in the tree resolves to a schema that
has `Energy` right now — the gate is currently a no-op in practice. It stops being
a no-op the moment that POC pin reverts (or any classid is minted to a narrower
schema), which is exactly why `ocr.rs`'s own test comment reads: "No classid
resolves to Bootstrap today — when one is minted, the same `schema.has()` gate
leaves its slab empty." T5 is the same principle applied to the sweep surface,
landed BEFORE the flip rather than as a fire drill after.

## The one real caller — checked, not assumed

`crates/symbiont/src/domino.rs` calls `project_energy_nonfinite(&rows)` on rows
built via `NodeGuid::local(idx)` (classid 0 → currently `Full` → has `Energy`).
Confirmed by direct read that its only consumption of `NanReport` is
`.is_clean()`, `.count()`, `.nonfinite` (field/method access, never an exhaustive
struct-destructure) — so adding the `skipped` field is compile-safe, and since
domino.rs's rows all currently resolve to a schema that includes `Energy`, the
gate changes nothing about its runtime behaviour today (`skipped` will be 0 for
every row it constructs). Not a vacuous no-op risk: verified `symbiont` doesn't
construct or destructure `NanReport` anywhere else (`grep -rn "NanReport"` outside
`nan_projection.rs` returns nothing).

## The fix

- `row_has_energy(row) -> bool` — reads `row.key.read_mode().value_schema.has(ValueTenant::Energy)`.
  One branch, on schema presence, never on the float value.
- `project_energy_nonfinite`: skip (don't read) rows failing the gate; added
  `NanReport::skipped: usize` so the gate's effect is observable rather than a
  silent no-op (the workspace's can-it-fire testing rule).
- `energy_all_finite`: filters to Energy-bearing rows before the finiteness `.all()`.
- Module doc corrected: the old "no branch on the value" framing is now precise —
  the finiteness test itself stays branchless; the NEW branch is on schema
  presence, documented as such rather than left to silently contradict the code.

## Tests (both halves of the falsifiability rule)

Fixtures switched from `NodeGuid::local(0)` (classid 0 / `DEFAULT`, temporarily
`Full`) to `NodeGuid::CLASSID_OSINT` (permanently `Cognitive`, no sunset) — a
fixture pinned to the temporary POC default would have silently gone vacuous the
moment that default reverts to `Bootstrap`.

- **Can-it-fire (new):** `schema_gate_excludes_boards_whose_schema_omits_energy`
  — real registered classids, not synthetic overrides: one `CLASSID_OSINT` row
  (Cognitive, has Energy) + two `CLASSID_FMA` rows (Compressed, no Energy) with
  their Energy-offset bytes poisoned to `NAN`/`INFINITY`. Asserts the poison IS
  real (`f32_bits_nonfinite(energy_bits(...))` true on the raw read — proves the
  test isn't vacuously passing because nothing was actually adversarial), then
  asserts the gated sweep: `total == 1`, `skipped == 2`, `nonfinite.is_empty()`,
  `is_clean()`, and `energy_all_finite` agrees.
- **Can-it-stay-silent (existing 2 tests, extended):** `finite_batch_is_clean`
  and `nan_and_inf_are_flagged_neg_inf_too` now also assert `skipped == 0` — an
  all-Energy-bearing batch is swept in full, proving the gate doesn't degrade
  the pre-existing behaviour it's meant to leave alone.

## Verification

- `cargo test -p lance-graph-contract nan_projection --lib` → 4/4 green (the 3
  pre-existing + the 1 new adversarial test).
- `cargo test -p lance-graph-contract --lib` (full crate) → 1135/1135 green, no
  collateral breakage from the added `NanReport` field.
- `cargo clippy -p lance-graph-contract --all-targets -- -D warnings` → clean.
- `cargo fmt -p lance-graph-contract` → ran, re-verified tests green after.
- `symbiont` (the one external caller, bin-only crate, heavy SurrealDB/OGAR/AMX
  dependency tree, excluded from the default workspace): verified compile-safety
  by direct code read (field/method access only, no destructure — see above)
  rather than a full build, given the crate's build cost relative to the
  verification value already established analytically. A `cargo check
  --manifest-path crates/symbiont/Cargo.toml --bin symbiont` was also launched
  to confirm end-to-end; see this file's companion commit / follow-up note for
  its result if it completed within budget.

## What I did NOT do

- Did not touch `domino.rs`'s classid choice. Its rows are Bootstrap/classid-0
  today, which currently resolves to `Full` (has Energy) only because of the
  TEMPORARY POC default pin — once that pin reverts to `Bootstrap`, domino.rs's
  own `set_energy`/`energy_of` writes would themselves become a schema-contract
  violation (writing into a tenant the row's own schema declares absent). That
  is a SEPARATE, larger design decision (does the POC need a proper classid
  mint into the shared `classid_read_mode` registry, or does it stay a private
  raw-byte-region borrow outside the schema system entirely) that I did not make
  unilaterally — flagging it as a follow-on rather than folding it into T5's
  scope, which was specifically the sweepers.
- Did not change `ReadMode::DEFAULT`'s POC pin — out of scope, already tracked
  by its own doc comment and the paired `ClassView` revert it names.
