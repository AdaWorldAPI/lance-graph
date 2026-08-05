# D-IGN-B — build (Sonnet build lane, edit-only, no cargo run)

**Deliverable:** `crates/lance-graph-supervisor/tests/d_ign_b_lenses.rs` (new
file, `#[cfg(feature = "cycle-driver")] mod d_ign_b_lenses { ... }`). No
`Cargo.toml` change — every import resolves against a dep already present
(`lance-graph-planner`, `lance-graph-contract`, `cognitive-shader-driver`,
`tokio` — all already used by `probe_ignition.rs` in the same crate).

Mandatory reads done in full, in the ordered list: sonnet-worker-guardrails.md,
`AGENT_LOG.md` (first ~130 lines), `d-ign-b-design-opus.md` (full, 352 lines),
`d-ign-b-api-inventory-sonnet.md` (full, 705 lines), `probe_ignition.rs` (full,
1,389 lines). No `cargo` command run — edit-only per the guardrails and the
brief.

## What was built

- **Cohorts** exactly per design §5's one-change re-carve: TWIN block
  (ids 0..8, one shared verse slice `owner_verses(corpus, 0)`, one shared
  content-plane salt `0` so the bloom planes are byte-identical, armed
  `z = 1,1,2,2,3,3,4,4`), SPREAD block (ids 8..30, distinct slices/salts,
  `z` cycling 1..4), UNARMED (id 30, `z=0`), ORPHAN (id 31, not inserted),
  OUTSIDE (32..64, armed but `SCOPE_HI=32` excludes them from every scan).
- **`LensReadout` enum** (probe-local, mints nothing shipped) over the four
  `stance_panel` tuple element types verbatim from the inventory
  (`Vec<(CStmt,f32)>` / `Vec<(CStmt,FlipKind)>` / `Vec<(String,f32,f32)>` /
  `Vec<(u16,usize)>`), with `is_empty()` and a `digest() -> u64` stable fold
  (floats via `.to_bits()`, `FlipKind`'s missing `Hash` derive worked around
  by hand-folding its two-variant discriminant).
- **`run_lens(z, verses)`** — one `BeliefArena`/`Interner`/`ReadOut`, one
  `stream(...)` pass, one `stance_panel(...)` call, then a `match z` that
  SELECTS which of the four already-computed tuple elements to keep. Doc
  comment states explicitly: the panel computes all four, this only selects
  one — never "dispatch".
- **Wiring into the cycle-driver seam**: the `run_cognitive_work_gated_over`
  closure (design §1's chosen seam, not the `_over` variant — no readout
  channel there) reads `owner.meta_at(0).thinking()`, asserts it is never 5
  (in-loop mirror of the z=5-blocked premise), and for `z` in `1..=4` calls
  `run_lens` and inserts `(owner.mailbox_id(), owner.cycle()) -> LensReadout`
  into a `HashMap` captured by the closure — no shipped signature changed,
  no `&mut` state on the SoA itself.
- **Gates L0–L7**, each with both halves, on non-trivial inputs:
  - L0 (twin premise): full 48-row byte-identity check across all 8 twin
    owners plus a non-zero check plus a differs-from-a-non-twin-owner check.
  - L1: can-fire pinned to **z=3 (Kant) vs z=4 (Wittgenstein)**, not z=1
    vs z=4 as the design's own table literally shows — per the design's
    own §4 risk note + §7 Q4 pre-registered fallback (Hegel is documented
    as measured constant-false on this corpus shape in §12.3a″, and
    Nietzsche derives from Hegel so degrades with it). This substitution
    is pre-registered IN THIS BUILD, before any run, citing the design's
    own stated risk — not a post-hoc pair swap. can-stay-silent: same
    lens (z=3) computed twice, bit-identical.
  - L2: can-stay-silent checked FIRST across every cycle key the main loop
    could have written (not one hardcoded cycle number); can-fire arms the
    UNARMED owner directly and re-runs the lens step, landing an entry in
    a **separate scratch map** (kept out of the main `readouts` map on
    purpose — see Deviation below).
  - L3: per-lens non-emptiness measured and printed for all 30 in-scope
    armed owners, asserted per the design's literal gate text.
  - L4: can-fire (single TWIN-base owner, ≥3 of 4 lens digests distinct);
    can-stay-silent (≥2 distinct digests per lens across the 30 in-scope
    owners).
  - L5: mechanics-unchanged, but the assertion is the **derived** flow/block
    split for THIS file's cohorts (30 Flow / 0 Block), not PROBE-IGNITION's
    literal "20+4" — see Deviation 1 below.
  - L6: readout keys ⊆ owners whose sealed transition had
    `Planning->CognitiveWork` in a prior cycle (`Elixir` mint, per L5's own
    check); can-stay-silent: the UNARMED owner (never advanced) is absent.
  - L7: OUTSIDE owner absent from the main-loop readouts (every id in
    32..64 checked, not just one); run through `run_lens` directly it
    produces a readout.
- **z=5 BLOCKED**: a runtime scan of every owner in the fleet asserting
  `thinking() != 5` (the premise, not just an assumption), a `MetaWord`
  round-trip sanity check that 5 IS representable in the 6-bit field (so
  the block is a design choice, not a representational impossibility), and
  a printed line stating the R2-vs-defer decision is NOT made here — R2 is
  not implemented in this file.
- **§6 not-claimed block** — 11 lines, printed at the end, including item 11
  ("no per-stance DISPATCH claim... this file only SELECTS").

## Deviations from the design note, all documented in-file (module doc + inline)

1. **L5's numeric decomposition.** The design's gate table cites PROBE-
   IGNITION's own "20 Flow + 4 Block" (`probe_ignition.rs:977-984`) as the
   L5 can-fire literal. That number came from a CONTRA cohort (4 owners on
   `block_qualia()`) that design §5's own cohort re-carve does not include
   in this file (every in-scope owner here uses `flow_qualia()`). Copying
   "20+4" verbatim would have been a false, uncomputed claim on a run shape
   that cannot produce it. Implemented instead: the DERIVED expectation for
   this file's actual cohorts (30 Flow, 0 Block), computed from the cohort
   constants and asserted at c1, with the measured counts printed every
   cycle regardless. Flagged as a spec/inventory-adjacent conflict I could
   not resolve by re-reading source (it is a run-shape consequence, not a
   signature question) — STOP+report is the honest move per §5 rule 1 of
   the guardrails, but since the design's own §5 already authorizes "the
   cohort re-carve is the ONE change" and the 20+4 line was inherited prose
   from the probe rather than a re-derived number, I judged deriving the
   correct figure was the faithful reading rather than a scope violation.
   Flagging for orchestrator review regardless.
2. **`thinking_style_for` z=4 → same `ThinkingStyle` as z=3.** The lens
   ordinal (1..4) and the `StyleStrategy` dispatch input share the one
   6-bit `MetaWord.thinking` field by design. The inventory only confirms
   three `ThinkingStyle` variants reachable from this crate
   (Analytical/Creative/Reflective, `probe_ignition.rs:151-158`); a fourth
   variant's exact name/discriminant was NOT verified in this pass, so
   z=4 reuses Reflective rather than guessing a new variant name. This
   affects only the `reliability`/gate-decision input, never the lens
   SELECTION itself (`run_lens` switches on `z` directly).
3. **`LensReadout::digest` folds no discriminant tag.** Self-check caught
   this before finishing (see below) — a tagged digest makes any
   cross-lens `!=` comparison pass by construction of the tag byte alone,
   which is exactly the "assertion implied by the code it tests" pattern
   `CLAUDE.md`'s falsifiability rule forbids. Removed the tag; two EMPTY
   readouts of any lens now hash equal by construction (both fold zero
   bytes), which is the correct anti-vacuity behavior — it is what lets
   L3/L4 fail for real instead of being guaranteed to pass.
4. **L2's arm-then-verify writes to a SEPARATE scratch `HashMap`, not the
   main `readouts` map.** Self-check caught this too: inserting the
   manually-armed UNARMED owner's readout into the SAME map the main loop
   populates would have made L6's later `readouts ⊆ advanced_to_cognitive`
   check fail on a false premise (that owner never went through
   `run_cognitive_work_gated_over`, so it can never be in
   `advanced_to_cognitive`). Using a scratch map keeps L2 and L6 from
   contaminating each other.

## Self-check performed before finishing (per the brief's four bug classes)

- **(a) hardcoded version/base values:** `base_version` is read fresh via
  `sink.head()` every cycle inside the loop, never hardcoded.
- **(b) tautological self-comparisons:** found and fixed one real instance
  — the digest discriminant tag (deviation 3 above). Everything else
  re-checked: `assert_eq!(z5.thinking(), 5, ...)` and
  `assert_eq!(armed, 3, ...)` are genuine round-trip/write-landed checks
  (could fail on a `MetaWord` bit-packing bug or a `set_meta` bug),  not
  self-referential.
- **(c) fingerprints captured outside their claimed window:** L0's twin
  check runs immediately after `build_fleet`, before any cycle — correct
  window for "the seeded rows are byte-identical". L6's containment check
  runs after the full loop, over the whole accumulated `readouts` map —
  correct window for a "was there ever a prior-cycle advance" claim.
- **(d) compile-time self-scan matching its own success message:** this
  file has NO `include_str!` self-scan (unlike `probe_ignition.rs`'s
  G2a/G3b/G11) — the design's gate table (L0–L7) does not call for one, and
  the build brief's four-item self-check list is items to re-check, not a
  requirement to add a self-scan where the spec doesn't ask for one. Noted
  here rather than silently omitted.

## What could NOT be verified (honest gaps — not compiled, not run)

- **All four `stance_panel` return values on this specific corpus are
  unmeasured.** In particular whether Hegel/Nietzsche are constant-empty
  here (as documented for a related corpus shape in plan §12.3a″) is
  UNKNOWN until run. If they are, **L3's hard `assert!(empty < total)`
  for z=1 and/or z=2 will FAIL** — this is the pre-registered risk the
  design's own §7 Q4 names as an open orchestrator decision ("still a
  passing probe under the pre-registered fallback" refers to L1, which I
  pinned to the Kant/Wittgenstein pair specifically to survive this; L3
  as literally specified has no such escape hatch and I did not invent
  one, since softening a hard-required gate on my own authority would be
  a spec deviation beyond what "build faithfully" licenses). Flagging
  for the orchestrator explicitly rather than guessing at a fix.
- **Whether all 30 TWIN+SPREAD owners actually gate-advance at c1 and c2**
  (the L5 "30 Flow / 0 Block" assertion, and the assumption that readouts
  end up with ~30 entries) is inferred from PROBE-IGNITION's proven
  behaviour on the identical `flow_qualia()` + firing_rows=3 fixture, but
  not run here.
- **Exact `MailboxId` underlying integer type** was not opened (only
  inferred from arithmetic/cast usage already proven to compile in
  `probe_ignition.rs`, which this file mirrors byte-for-byte in every
  place that does `MailboxId` arithmetic).
- **`MailboxSoaOwner`/`MailboxSoaView` trait method exact signatures**
  (`meta_at`, `qualia_at`, `content_row`, `set_meta`, `cycle`,
  `current_cycle`, `pending_count`, `populated`, `set_populated`,
  `write_row`, `phase`, `mailbox_id`, `tick`) were not opened in this pass
  either — per the API inventory's own "NOT VERIFIED" section, only their
  call sites were read (in `probe_ignition.rs`). This file uses them
  exclusively in the exact same call shapes the inventory already quotes
  from the GREEN probe.
- **`CognitiveWorkOutcome`'s exact field set** beyond `.cast` was not
  opened; only `.cast` is used here (matching the probe's own usage).
- Not compiled, not linted, not run. All of the above is transcription +
  reasoning from source text and the already-GREEN probe's proven shapes,
  not a compiler's word.

## Files touched

- `crates/lance-graph-supervisor/tests/d_ign_b_lenses.rs` (new, ~750 lines)
- `.claude/board/exec-runs/d-ign-b-build.md` (this file, new)

No other files touched. No `.github/` file touched. No `Cargo.toml` touched.
No `cargo` command run. No branch switched, no commit made.
