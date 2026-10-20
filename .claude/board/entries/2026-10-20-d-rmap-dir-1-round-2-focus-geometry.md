# D-RMAP-DIR-1-R2: round 2, when 256-slot focus geometry beats compiled closure (2026-10-20)

**Status:** MEASURED, lab only; nothing promoted. **Basis:** `crates/dir-rmap-probe` (`tests/equivalence.rs` 11/11; `examples/bench.rs`, 4 runs at 8,000 users plus 1 at 65,000). Plan `d-rmap-dir-1-membership-closure-focus-v1.md` §9.

- **Completeness:**
  - F binds one chain and projects the full B+ closure through down rows, so paths through unfocused groups are represented (tested on two-hop cases for every chain);
  - it refuses >256 groups, duplicates, a stale B+ or focus, and unfocused groups;
  - the witness now compares the `Snapshot` by pointer, so two stores both starting at `VersionId(0)` refuse each other;
  - 8 guards disable-verified red.
- **Preparation:**
  - down-row projection costs 1.6–2.7 µs at 64 groups, against 72–82 µs for round 1's per-slot test, and that moves F's break-even from ~1,000 reuses to 2–100;
  - eager per-user rows cost 84–95 µs and 32 B per user; lazy rows fill at ~16 ns per user;
  - every arm's hot path makes 0 allocations.
- **Working set:**
  - B+'s closure passing L2 cost it only 5–17 % (warm vs hot); it slows with row width (218 → 436 ns from 1.5k to 11k groups);
  - F stays at 74–79 ns, a 2.9× → 5.5× advantage;
  - cold: F ~2×, F+users ~4×.
- **Question shape decides:**
  - composite membership (one bit): F 1.7–2.9× at every focus size;
  - conjunctions: 2.4–5.4× up to ~650 questions, but at 256 groups (~4,300 conjunctions, 135 KiB of masks, more than L1d) F's mask form is 1.5–1.7× *slower* than B+.
- **Rebuilding a focus per question never pays;** question-aware B+ (needed words only) never helps.

OPEN: real directories' question-to-group ratio; other CPUs (small L3); the gather admission (`ISS-ADMISSION-PREP-BINDING`); D-RMAP-DIR-2 not authorized.
