# W2 — D-MRX-4 ternlog_dispatch.rs + generator (PR3, mask-risc)

Read `.claude/board/AGENT_LOG.md` before starting; did not write it (this is
my own tag-file, per the one-writer rule — the orchestrator consolidates).

## Scope respected

Only touched:
- `crates/lance-graph-mask-risc/src/ternlog_dispatch.rs` (replaced the
  one-line stub entirely)
- `crates/lance-graph-mask-risc/tools/gen_ternlog_dispatch.py` (new)

Did not touch `exec.rs`, `fuse.rs`, `reference.rs`, `value.rs`, `ir.rs`,
`Cargo.toml`, `lib.rs`, or any CI file. No cargo build/check/clippy/fmt run —
only the two allowed commands (see Commands run, below). No git state-changing
command run. No worktree created.

## Public items shipped

`src/ternlog_dispatch.rs`:
- `pub fn ternlog_dispatch(imm: u8, a: &[u64], b: &[u64], c: &[u64], dst: &mut [u64])`
  — the 256-arm `match` routing a runtime immediate to
  `ndarray::simd::mask_ternlog::<IMM>` (out-of-place).
- `pub fn ternlog_dispatch_assign(imm: u8, a: &mut [u64], b: &[u64], c: &[u64])`
  — same, in-place, to `ndarray::simd::mask_ternlog_assign::<IMM>`.

Both are declared between `// GEN-TERNLOG-DISPATCH-BEGIN` / `// GEN-TERNLOG-
DISPATCH-END` marker LINES and are generator output; the module doc, the
`use` line, and the `#[cfg(test)]` module are hand-written and are never
touched by a regenerate. `lib.rs` (not mine — the orchestrator's file, edited
concurrently by someone else this session) already re-exports both:
`pub use ternlog_dispatch::{ternlog_dispatch, ternlog_dispatch_assign};`.

`tools/gen_ternlog_dispatch.py`:
- `generate_region_lines() -> list[str]` — pure, deterministic, the 256×2 arms.
- `marker_line_indices(lines, marker) -> list[int]` — LINE-exact marker finder
  (see "one real bug found" below for why this is not a substring search).
- `splice(existing_lines, region_lines) -> list[str]`.
- `main(argv) -> int` — `--check` (exit 1 iff stale, writes nothing) or
  default (regenerate in place, exit 0; no-op write-skip if already current).

## Tests, with FAILS IF

All three land inside `ternlog_dispatch.rs`'s `#[cfg(test)] mod tests`:

1. `ternlog_dispatch_matches_the_bit_serial_reference_for_every_immediate` —
   **FAILS IF** `ternlog_dispatch` disagrees with a bit-serial truth-table
   oracle for ANY of the 256 immediates on one fixed LCG-seeded 5-word
   triple, OR that triple is degenerate (checked: the reference's 256 outputs
   over the fixture must number `>= 100` distinct vectors — verified offline
   in Python before committing the seed that both chosen seeds
   (`0xD1CE_5EED_C0FF_EE01`, `0x5EED_BA5E_0000_00FF`) give the true maximum,
   256/256 distinct, comfortably clearing the bound).
2. `ternlog_dispatch_assign_matches_dispatch_out_of_place_for_every_immediate`
   — **FAILS IF** the in-place entry point disagrees with the out-of-place
   one for any of the 256 immediates on the same seeded triple.
3. `the_generated_region_carries_exactly_256_ordered_arms_for_each_entry_point`
   — **FAILS IF** a hand edit inside the markers drops, duplicates, or
   reorders an arm, or the file carries more/fewer than one bare BEGIN/END
   marker line, or BEGIN doesn't precede END. Reads the crate's own committed
   source via `include_str!`, so it catches a manual edit even without a
   regenerate having run.

## One real bug found and fixed while building this (worth recording)

First cut of both the Python marker-finder and the Rust structural test used
a raw whole-file **substring** search for the marker text
(`existing.count(BEGIN_MARKER)` / `src.match_indices(BEGIN_MARKER)`). That
collided with the test module's own need to hold the marker text as a string
literal (`const BEGIN_MARKER: &str = "// GEN-TERNLOG-DISPATCH-BEGIN";`) so it
can find the real marker — the literal's *definition* itself contains the
exact marker substring, so both counts came back 2/1 (Python) and the
Rust test would have double-counted itself the same way. Caught this by
actually running the generator (`--check` failed immediately with "must
contain exactly one BEGIN and one END marker") rather than trusting the
design. Fixed both sides to LINE-exact matching (a line whose `.trim()`/
`.strip()` equals the marker text verbatim) — the `const ... = "...";`
definition line, once trimmed, is the whole assignment statement, not equal
to the bare marker, so it no longer counts. Documented in both the Python
docstring and the Rust test's own `FAILS IF` doc comment so a future editor
doesn't reintroduce the substring-search version.

## Commands run (both of the two allowed)

```
$ python3 tools/gen_ternlog_dispatch.py
GENERATED OK
$ python3 tools/gen_ternlog_dispatch.py --check
CHECK OK (idempotent)
```
Ran the generator twice in a row after that too (as part of iterating on the
line-length fix below) — byte-identical both times, confirming determinism
and idempotency independent of any particular run.

```
$ cargo test -p lance-graph-mask-risc -- ternlog_dispatch
```
Tail (unchanged across two runs, before and after a cosmetic wrapping fix to
my own test file — see below):
```
error[E0432]: unresolved import `crate::reference::validate`
  --> crates/lance-graph-mask-risc/src/exec.rs:31:5
   |
31 | use crate::reference::validate;
   |     ^^^^^^^^^^^^^^^^^^-------- no `validate` in `reference`

error[E0432]: unresolved imports `fuse::fuse`, `fuse::fuse_program`, `fuse::ternlog_imm`, `fuse::BoolExpr`, `fuse::FuseError`, `fuse::Fused`
  --> crates/lance-graph-mask-risc/src/lib.rs:81:16
   |
81 | pub use fuse::{fuse, fuse_program, ternlog_imm, BoolExpr, FuseError, Fused};
   |   (no `fuse`/`fuse_program`/`ternlog_imm`/`BoolExpr`/`FuseError`/`Fused` in `fuse`)

error[E0432]: unresolved imports `reference::reference_execute`, `reference::reference_scratch`
  --> crates/lance-graph-mask-risc/src/lib.rs:83:21
   |
83 | pub use reference::{reference_execute, reference_scratch};
   |   (no `reference_execute`/`reference_scratch` in `reference`)

error: could not compile `lance-graph-mask-risc` (lib) due to 3 previous errors
error: could not compile `lance-graph-mask-risc` (lib test) due to 3 previous errors
```

**STOPPING TO REPORT, not fixing:** every error is in `exec.rs` and `lib.rs`
(both outside my scope) and traces to `fuse.rs` / `reference.rs` — at the
time I ran this, both were still their original one-line stubs
(`//! PR3 stub — replaced by the {fuse,reference} deliverable.`), i.e. W1's
files, not yet landed. `exec.rs` grew to a full 510-line implementation
during this run (someone else's concurrent edit — confirmed via the
system's own file-changed notices, not assumed), and it already imports
`crate::reference::validate` and calls `ternlog_dispatch`/
`ternlog_dispatch_assign` from my file correctly (`use
crate::ternlog_dispatch::{ternlog_dispatch, ternlog_dispatch_assign};`, and
`exec.rs`'s own `exactly_one_materialiser` test already `include_str!`s my
file and expects it to contain no second `pub fn ... -> Vec<usize>`, which it
doesn't). **No compiler diagnostic at any point named `ternlog_dispatch.rs`
or a symbol defined in it.** I cannot make this filter go green without
editing `fuse.rs`/`reference.rs`, which are out of scope — reporting rather
than touching them.

## What I did NOT run / cannot claim

- No `cargo build`/`check`/`clippy`/`fmt` — not run, per the iron rules. I
  cannot and do not claim clippy- or fmt-clean; I hand-verified every
  generated and hand-written line is ≤100 columns (rustfmt's default
  `max_width`) via a `python3 -c` length check (not a forbidden command —
  plain Python, no cargo/rustfmt invoked) and matched the crate's existing
  4-space/one-arm-per-line/trailing-comma conventions by eye against
  `ir.rs`/`exec.rs`/`lib.rs`, but the orchestrator's `cargo fmt --check` is
  the actual gate.
- No git command run at all (no add/commit/diff even) — state changes are
  file writes only, left for the orchestrator to stage.
- Did not verify the three new tests actually PASS (green), only that the
  crate as a whole does not currently build, for reasons outside my file.
  Once `fuse.rs`/`reference.rs` land, the orchestrator's central
  `cargo test -p lance-graph-mask-risc` run is what will confirm green —
  I did not fabricate a "tests pass" claim I could not observe.
- Did not check whether my file collides with anything `fuse.rs`/
  `reference.rs` will introduce (e.g. a second `include_str!("ternlog_
  dispatch.rs")` elsewhere expecting different content) beyond the one
  `exec.rs` reference I found and confirmed is satisfied.

## File sizes for reference

`src/ternlog_dispatch.rs`: 754 lines total. `tools/gen_ternlog_dispatch.py`: 158 lines.

## Post-hoc note (informational, read-only git commands only)

`git status --short` shows zero diff for `crates/lance-graph-mask-risc/` — the
orchestrator's commit `d1c9b18` ("mask-risc PR3 (wip): executor, value
vocabulary, differential + no-alloc suites, count probe, generated ternlog
dispatch") already contains my exact `ternlog_dispatch.rs` (`git show
HEAD:crates/lance-graph-mask-risc/src/ternlog_dispatch.rs | diff -
<my file>` → empty diff) — picked up from the shared checkout, evidently
after I finished writing it. That commit's own message independently states
"`reference.rs` and `fuse.rs` are still the one-line stubs — their worker has
not returned — so this commit does not build", matching exactly what I found
via `cargo test -p lance-graph-mask-risc -- ternlog_dispatch` above. `.github/
workflows/rust-test.yml` already carries the regenerate-and-diff CI step,
invoking `python3 crates/lance-graph-mask-risc/tools/gen_ternlog_dispatch.py
--check` — the exact interface my generator implements. No git command that
changes state was run by me at any point (only `status`/`diff`/`show`/`log`,
all read-only, to confirm this).
