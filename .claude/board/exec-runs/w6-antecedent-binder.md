# W6 — Antecedent lane binder — execution record

## Files touched
- NEW: `crates/lance-graph-planner/examples/probe_antecedent_binder.rs`
  (only file created/modified; no other file touched).

## Searches run (confirming the gap before writing)
- Grep `pattern="WitnessLens|NodeRow|value_offset|CausalWitnessFacet|ValueTenant"`
  over `crates/jc/examples/l9_loci_real_text.rs` → 0 hits (re-confirmed the
  orchestrator's exhaustive check; the resolver never touches the lane).
- Read (full) `crates/lance-graph-contract/src/causal_witness.rs` (787 lines).
- Read (offset-chunked, full coverage of relevant sections) `witness_fabric.rs`
  lines 1-220 (module doc, `WitnessLens`, `write_register`, `at`, tenant-offset
  const-asserts).
- Grep for `NodeRow`/`ValueTenant::CausalWitness`/`impl NodeRow`/`NodeGuid::local`/
  `EdgeBlock::default` in `canonical_node.rs` to confirm constructors and the
  `byte_len()`/`value_offset()` derivation path (no bare literals used).
- Read `probe_sudoku_teacher.rs` head + tail (gate-table + `ALL GATES GREEN`
  idiom) to match house style.

## What the example does
- Builds a 26-token fixture stream as `Vec<NodeRow>` (zero-init value slabs,
  then 0xEE-canaried outside the CausalWitness tenant span for a falsifiable
  write-isolation baseline).
- For each pronoun token with a given `(pos, antecedent_pos)` pair, computes
  the signed displacement and either binds it via
  `WitnessLens::write_register(&mut row, CausalWitnessFacet::ZERO.with(Locus::Antecedent, offset))`
  or escalates (no write at all) when `d == 0` or `d` falls outside `-8..=7`
  — escalate, never clamp/saturate.
- Reads bound values back via `WitnessLens::at(pos)` — zero-copy, no gather
  of `Vec<CausalWitnessFacet>` or `Vec<(usize, ...)>` anywhere in the file.
- The A5 write-isolation gate locates the touched byte offset by DIFFING a
  zero row against a `write_register`-populated scratch row (never a bare
  literal offset — `WITNESS_REGISTER_START`/`WITNESS_FACET_CLASSID_BYTES`
  are private to `witness_fabric`, so this is the only offset-derivation path
  available to an external crate; it goes through the same producer path the
  probe itself exercises, not a re-derivation of the private constants).

## Gate-by-gate result (falsifying input named per gate)
- **A1 round-trip** — PASS. 4 bound pronouns (offsets -3, -3, -2, -1) all
  read back identical via the lens. Anti-vacuity: 3 distinct nonzero offsets
  observed (`{-3, -2, -1}`), so the round-trip isn't just "reads back one
  repeated value". Falsifier: any bound offset that mismatched on read-back,
  or fewer than 3 distinct offsets.
- **A2 escalation fires** — PASS. Token `they`@24 → antecedent@3 (d = -21,
  outside `-8..=7`) escalates AND its `Locus::Antecedent` nibble reads `0`
  (not a saturated ±8/±7). Falsifier: escalation not recorded, or the nibble
  reading nonzero (evidence of silent clamping).
- **A3 escalation stays silent** — PASS over the in-range set `{3, 9, 11, 12}`
  (offsets -3, -3, -2, -1 respectively) — none escalated, all bound.
  Falsifier: any of these 4 positions appearing in the escalated set.
- **A4 zero is unbound** — PASS. Token `itself`@25 with `antecedent_pos =
  Some(25)` (d == 0) escalates with the self-referential reason and its
  nibble reads `0`. Falsifier: no escalation recorded, wrong reason string,
  or nonzero nibble.
- **A5 write isolation** — PASS over all 4 bound rows. Full 480-byte value
  slab diffed byte-by-byte before/after; only the one byte holding the
  `Locus::Antecedent` nibble (the high nibble of the register's 4th byte —
  slot 7 is odd, so it lands in the high nibble) changed; every other byte,
  including the 0xEE canary region and the other 11 bytes of the
  CausalWitness register, matched the pre-write snapshot exactly. Falsifier:
  any byte outside that single offset changing, or the target byte NOT
  changing.

## Verification run
- `cargo run -p lance-graph-planner --example probe_antecedent_binder` →
  `ALL GATES GREEN`.
- `cargo clippy -p lance-graph-planner --all-targets -- -D warnings` → clean
  (0 warnings from this crate; only a pre-existing unrelated workspace warning
  about `cognitive-shader-driver`'s duplicate bin target, not from this file).
- `cargo fmt -p lance-graph-planner` → ran; re-ran the example afterward to
  confirm still `ALL GATES GREEN` after formatting.

## What I did NOT do
- Did not modify `causal_witness.rs`, `witness_fabric.rs`, `canonical_node.rs`,
  or either existing resolver example (`spo_anaphora_nibble.rs`,
  `l9_loci_real_text.rs`) — brief forbids it, none needed.
- Did not implement or copy any coreference-resolution linguistics — the
  fixture supplies `(pronoun_pos, antecedent_pos)` pairs directly, per brief.
- Did not touch any board file other than this one (no `AGENT_LOG.md` write).
- Did not run a full `cargo build`/`cargo check`; only the explicitly-granted
  `cargo run --example`, `cargo clippy`, and `cargo fmt` for this crate.

## ⊘ Correction (2026-07-29, CodeRabbit mechanical-fixes pass)

Line 27's recorded snippet shows
`WitnessLens::write_register(&mut row, CausalWitnessFacet::ZERO.with(Locus::Antecedent, offset))`
passing the facet BY VALUE. The real signature
(`crates/lance-graph-contract/src/witness_fabric.rs:167`) is
`write_register(row: &mut NodeRow, facet: &CausalWitnessFacet)` — it takes a
REFERENCE, and the probe's actual call site passes `&facet` accordingly.
