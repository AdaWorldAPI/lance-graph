# D-BLW-3 build (Sonnet grindwork lane)

**Scope:** created `crates/lance-graph-planner/examples/blw_fusion.rs` (~1145
lines); edited `crates/lance-graph-planner/Cargo.toml` (one dev-dep line +
comment, `jc = { path = "../jc" }`). No other file touched. `temporal.rs`,
`crates/jc`, `persist_sink.rs`, `blw_tenant.rs` — read only, not modified.
`AGENT_LOG.md` — read (first 150 lines), not written.

**NOT COMPILED, NOT RUN.** Edit-only per the hard rules — no cargo invoked,
no worktree created. Everything below is a source-level self-review (I
re-read the whole file after writing it and fixed two real issues found
that way — see "Self-caught bugs").

## B5 note

B5 in the design note says "RESOLVED by the coordinating lane" (i.e. the
`jc` dev-dep addition). **This build IS that coordinating lane** — the
`jc = { path = "../jc" }` line under `[dev-dependencies]` was added in this
change, not by a separate prior commit. Recorded explicitly per the
coordinator's mid-build instruction.

## The eight corrections (C1-C8), one line each on where they landed

- **C1** (three-way extensional identity) — gate G1c: asserts
  `at(V4,5)`/`at(V4,9)`/`at(V8,0)` (Aware/Retro/Strict) return the identical
  `(subject,horizon,proj)` sequence, plus the "one function, three names"
  line printed.
- **C2** (knowable_from precondition) — the `G3` block: iterates every
  emitted `VerdictRow`, asserts `knowable_from()` is constant AND `<= v_pin`.
- **C3** (no substrate-exercise claim) — printed verbatim in the `§6`
  not-claimed block at the end of `main`.
- **C4** (fold-order bug) — `fold_last_by_subject`: filters to ONE
  projection first, then a `HashSet`-backed anti-duplicate assert on
  `(subject, horizon)` before folding.
- **C5** (signed churn beside Hamming) — `churn()` helper (gained/lost
  separately), called at V_pin and printed beside (never folded into)
  `delta_kappa`.
- **C6** (pre-registered null) — printed immediately after the main loop,
  before any gate or result.
- **C7** (8-horizon drop test) — the `== C7: drop test over ALL EIGHT
  horizons ==` loop; prints all 8 `(kappa_strict, kappa_aware, delta,
  hamming_A, hamming_B)` rows, sanity-asserts Hamming=0 at k=8, and the DROP
  verdict requires `max|delta_kappa| < 0.01` across all 8 AND the k=8 sanity.
- **C8** (full table + degeneracy-before-pairing) — `print_association_table`
  always prints the full `BinaryAssociation` (n00/n01/n10/n11, both
  marginals, p_o, p_e, kappa, phi); the G4 DEGENERATE-can-stay-silent block
  runs BEFORE the `binary_association(a_strict, b_strict)` call, per §3.2a
  ordering.

## The ninth correction (coordinator's mid-build message on G6)

Re-derived the arithmetic myself before coding it (see the doc comment on
`seating_slice`): a fixed-prefix subject seated in slice `s` (s=1..4) has
`9 - s` Aware rows and `5 - s` Strict rows, not a uniform 8/4. Implemented
as a per-subject `BTreeMap<row, count>` built from the raw (pre-fold)
deinterlaced row sets, asserted against `seating_slice(row)`-derived
expected counts for every one of the 1000 fixed-prefix subjects, plus the
`== K_FIXED_PREFIX` cardinality check on both count maps.

## Deviations from the design note (with reasons)

1. **G4 COLLAPSED/DEGENERATE-can-stay-silent for the REAL (A,B) pair is a
   SOFT check (print + flag), not a hard `assert!`/panic**, unlike the
   can-fire halves (which ARE hard asserts on a manufactured case). The
   design's own prose calls a real collapse "a real possible outcome,
   pre-accepted" — panicking the whole harness on an honest corpus finding
   would contradict that framing. The DEGENERATE-can-stay-silent check for
   A/B IS a hard assert for the Strict read (guaranteed exactly `0.25` by
   construction, since Strict's pool == the fixed prefix) but a soft
   printed flag for the Aware read (a genuinely data-dependent marginal).
   This asymmetry is documented in-line where it happens.
2. **UNSTABLE guard (`expected_agreement > 0.95`) added beyond the
   corrections list** — design §3.5 lists it as a guard; implemented as a
   soft print+flag (not a hard gate), folded into the `fusion_permitted`
   check alongside COLLAPSED.
3. **§3.6's "movement test is a two-point contrast (V4 vs V8)" line was NOT
   implemented literally** — it reads ambiguously against §3.3's explicit
   "the two kappa are the Aware and Strict reads at the SAME pin," and C7
   supersedes it anyway with an explicit 8-horizon instruction. Implemented
   §3.3's movement test at V_pin only (as VERBATIM as stated), and C7's
   8-horizon drop test as the trajectory-wide extension. Flagged in the
   file's own module doc comment for the orchestrator to re-examine if the
   V4-vs-V8 reading was intended literally.
4. **No PROBE-TRAP / `NextPhaseScheduler` check** — not part of D-BLW-3's
   gates (G1-G7), and blw_tenant's own §4 trap is orthogonal to the fusion
   measurement. Dropped to keep the seal-sequence copy minimal; the DAG
   legality (`Plan => &[Planning]`) was independently re-verified via a
   direct grep of `kanban.rs` before writing the `plan` array.
5. **Dropped `SealedCycle.image_rows` and `topic`/`angle` plane writes**
   from the copied `blw_tenant.rs` pieces — `image_rows` was write-only in
   my usage (I don't call an `image_rows_of`-style reader), which would
   have been a `dead_code` field; `topic`/`angle` are never read by this
   harness's projections (§2.1 explicitly rejects `topic` as a criterion),
   so writing them would be dead weight, not fidelity.

## Self-caught bugs (found on my own re-read, fixed before finishing)

1. A garbled leftover expression in the first G7 draft
   (`fold_a_strict_pin.iter().cloned().collect::<Vec<_>>().is_empty().then(Vec::new).unwrap_or(...)`)
   — replaced with the intended direct comparison
   `fold_a_aware_desc == fold_a_aware_pin`.
2. Possible `clippy::needless_range_loop` on the emission loop (three
   parallel `Vec<bool>` indexed by a bare `for row in 0..seated_total`) —
   rewritten as a zipped `.enumerate()` over the three vectors.
3. Unused `Q_QUANTILE` const (documentation-only, never read in code) —
   now also printed in the startup banner so it isn't dead code.

## Open questions for the orchestrator

1. **§3.6's "V4 vs V8" phrasing** (deviation #3 above) — please confirm
   the interpretation (movement test = mode-contrast at V_pin only, C7 =
   the trajectory-wide extension) matches intent, or point at which
   reading is correct if I misjudged the ambiguity.
2. **G4 soft-vs-hard asymmetry** (deviation #1) — confirm the design intends
   a real corpus COLLAPSED/UNSTABLE finding to be reportable rather than
   fatal. If the design actually wants a hard gate there too (i.e. the run
   should panic if the real corpus collapses), that is a one-line change
   (swap the `println!` flags for `assert!`).
3. Not independently verified (no cargo run): whether
   `emit_bootstrap_intent`'s returned `cast` handle is `Copy` for the
   triple-use pattern (`on_behalf_of`, `intent_moves`, and the earlier
   `.expect()`) — mirrored verbatim from `blw_tenant.rs`'s own usage, which
   is presumed to compile since it is recorded as shipped in `AGENT_LOG.md`.
4. Wall-clock/memory of an 8-cycle x up-to-2000-row run with the added
   rank-sort work (2 sorts of up to 2000 elements per cycle) — not
   measured, expected trivial per the design's own B7 disclosure, but
   genuinely unmeasured by this lane.
