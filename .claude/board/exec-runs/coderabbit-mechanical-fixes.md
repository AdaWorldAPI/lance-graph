# CodeRabbit mechanical fixes — execution record (2026-07-29)

Branch `claude/x265-x266-plans-review-h9osnl`. Ten findings (F1-F10), all
verified real before this run. Files touched: exactly the six named in the
brief — `probe_sudoku_teacher.rs`, `probe_antecedent_binder.rs`,
`witness_fabric.rs`, `tenants.md`, and the two exec-run records (append-only).

## Per-finding outcome

- **F1 — FIXED.** `quadrant_census` in `probe_sudoku_teacher.rs`: a
  contradicted (empty, ZERO-candidate) cell now scores entropy `1.0`
  (maximal), not `0.0/9.0 = 0.0`. Added a comment explaining why. **Gate
  numbers did NOT change** — verified by `git stash push` on just this file
  (restoring the pre-fix version), re-running the probe, and `diff`-ing full
  stdout against the post-fix run: byte-identical output, all gates PASS
  both before and after. This fixture's runs never produce a genuine
  zero-candidate cell during census computation (the G4 hard/refuse and
  hard/bifurcate runs plateau with candidate sets > 0 everywhere, not full
  contradictions), so the bug is real (confirmed by code inspection and
  would flip a cell from Wisdom to Staunen/Confusion on a fixture that DOES
  reach cand_len==0 with energy>=0.5) but does not manifest as a numeric
  change on this specific probe today.
- **F2 — FIXED.** Doc-comment above `build_ambiguity_fixture` corrected from
  "k=0, a DIFFERENT row band from the shared row" to "k=3, the box's MIDDLE
  row — a different row band from the shared row", matching the code's
  `cib == 7 || cib == 3` and its own inline comment.
- **F3 — FIXED.** `candidates_from_box_lane`: replaced the unchecked
  `(pos as isize + off as isize) as usize` cast with
  `usize::try_from(peer_signed)` + `continue` on `Err`, and switched to
  `grid.get(peer_pos).and_then(read_cell)` so an out-of-range (not just
  negative) result is also handled without panicking.
- **F4 — FIXED.** `hidden_singles`: precomputed
  `cands: Vec<Vec<u8>> = (0..81).map(|p| candidates_from_full_sweep(grid, p)).collect()`
  once at function entry (grid is immutable for the whole function),
  replacing the ~27×-per-cell redundant O(81) sweep with a single pass.
  Behaviour is identical (verified: probe output unchanged).
- **F5 — FIXED.** Annotated `g4_hard_refuse: [NodeRow; 81]` to match the
  sibling `g4_hard_bifurcate` binding's existing annotation.
- **F6 — FIXED.** `tenants.md` §byte-math: corrected `field_mask()`'s cited
  line range from `1132-1157` to `1132-1162` (verified against
  `canonical_node.rs` — the `Full` match arm actually spans to line 1162,
  closing paren for `CausalWitness`). Corrected "totalling 220 B of 480 —
  260 B headroom" to "spanning row range [32,220) — 188 B of the 480-byte
  value slab consumed (220 − 32) — 292 B headroom", matching
  `VALUE_SLAB_ROW_OFFSET = 32` and `tenant_bytes() <= VALUE_SLAB_LEN`
  (canonical_node.rs:1196-1197).
- **F7 — FIXED (append-only correction).** Both exec-run files corrected via
  `tee -a`, never Edit/Write:
  - `lens-migration-zc2.md`: appended a dated `⊘ Correction` explaining the
    14-vs-20 arithmetic (table sums to 20; minus 4 WitnessStream rows
    (separately adjudicated) = 16; minus 1 BLOCKED + 1 example call site =
    14 live migratable *parameters* — unit made explicit).
  - `w6-antecedent-binder.md`: appended a dated correction noting
    `write_register` takes `&CausalWitnessFacet` (a reference), not by
    value, per the real signature at `witness_fabric.rs:167`.
- **F8 — FIXED.** `probe_antecedent_binder.rs` comment corrected: "a full 20
  tokens back" → "a full 21 tokens back" (real displacement `3 - 24 = -21`,
  magnitude 21), and "Filler tokens 13..32" → "Filler tokens 13..23" (the
  actual filler span; the stream has only 26 tokens total, positions 0..25).
- **F9 — FIXED.** Added `#[cfg(test)] mod decide_tests` in
  `probe_antecedent_binder.rs` with 5 boundary tests: self-reference (d==0)
  Err, d==+8 Err, d==-9 Err, d==-8 → `Ok(-8)`, d==+7 → `Ok(7)`. All 5 pass.
- **F10 — FIXED.** `elect_peers_lens` in `witness_fabric.rs`: bounded the
  peer scan to `focal_pos.saturating_sub(8) ..= (focal_pos+7).min(lens.len()-1)`
  instead of `0..lens.len()` — semantics-preserving (every position outside
  that range was already rejected by the `delta` check, so the bound changes
  no election, only the work done to reach it). Did NOT touch
  `quorum_mantissa_lens` (its ceiling counts every visible peer, so bounding
  it would change the answer — left untouched per the brief). The existing
  equivalence test `lens_peer_fabric_matches_gathered_across_visibility`
  still passes (verified below), which is the safety argument.

## Gates run (all green)

- `cargo fmt -p lance-graph-planner && cargo fmt -p lance-graph-contract` →
  ran clean (reformatted the two touched Rust files; re-read and confirmed
  fixes survived formatting).
- `cargo clippy -p lance-graph-planner --all-targets -- -D warnings` → clean
  (only the pre-existing unrelated `cognitive-shader-driver` duplicate
  bin-target warning, not from this crate).
- `cargo clippy -p lance-graph-contract --all-targets -- -D warnings` → clean
  (same pre-existing unrelated warning only).
- `cargo test -p lance-graph-contract` → 1123 + 7 + 8 + 7 + 4 + 21 pass, 0
  fail (unit tests unchanged in count from the prior exec-run record,
  including `lens_peer_fabric_matches_gathered_across_visibility` and
  `peer_fabric_is_non_trivial`, both still green after the F10 bound).
- `cargo run -p lance-graph-planner --example probe_sudoku_teacher` → ALL
  GATES GREEN (G1-G6), byte-identical stdout to the pre-F1-fix run (see F1
  above for the stash-diff methodology).
- `cargo run -p lance-graph-planner --example probe_antecedent_binder` → ALL
  GATES GREEN (A1-A5), unchanged from the shipped exec-run record.
- `cargo test -p lance-graph-planner --example probe_antecedent_binder` → the
  new `decide_tests` module: 5/5 pass.

## What I did NOT do

- Did not touch any file outside the six named in the brief.
- Did not modify `AGENT_LOG.md` or any other shared board file.
- Did not run a full `cargo build`/`cargo check` — only the explicitly
  granted `cargo fmt`, `cargo clippy --all-targets`, `cargo test`, and
  `cargo run --example` commands, scoped with `-p`.
- Did not force any fix that turned out not to be real — all ten were
  confirmed against the actual file contents before editing.
